# Databricks notebook source
# Stage 30B-3 controlled taxi-out parity validation.

from pathlib import Path
import sys

from pyspark.sql import functions as F

print("=" * 100)
print("Stage 30B-3 read-only parity validation. This notebook must not mutate tables or workspace resources.")
print("=" * 100)

# COMMAND ----------

RUN_PARITY = False

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"
SILVER_CATALOG = "panda_silver_dev"
SILVER_SCHEMA = "ml_ops"

SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"
CLEANED_TABLE = "cleaned_flight_data_full_table"
CURRENT_MV_TABLE = "ft_airport_daily_taxi_out"

HISTORY_START = "2023-07-01"
DATA_CUTOFF_DATE = "2027-01-01"

LATEST_UPDATE_KEY_BATCHES = 1
MAX_DIRTY_LEGS = 25
MAX_AFFECTED_ENTITIES = 1
ENTITY_FILTER = ""
LAST_SEEN_UPDATE_KEY = None
REQUIRE_FULL_AFFECTED_WINDOW = True
TOLERANCE = 1e-9


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


def silver_table(table_name: str) -> str:
    return f"{SILVER_CATALOG}.{SILVER_SCHEMA}.{table_name}"


CONFIG_ROWS = [
    ("RUN_PARITY", str(RUN_PARITY)),
    ("SOURCE_LEG", source_table(SOURCE_LEG_TABLE)),
    ("SOURCE_LEG_TIMES", source_table(SOURCE_LEG_TIMES_TABLE)),
    ("CLEANED", silver_table(CLEANED_TABLE)),
    ("CURRENT_MV", silver_table(CURRENT_MV_TABLE)),
    ("HISTORY_START", HISTORY_START),
    ("DATA_CUTOFF_DATE", DATA_CUTOFF_DATE),
    ("LATEST_UPDATE_KEY_BATCHES", str(LATEST_UPDATE_KEY_BATCHES)),
    ("MAX_DIRTY_LEGS", str(MAX_DIRTY_LEGS)),
    ("MAX_AFFECTED_ENTITIES", str(MAX_AFFECTED_ENTITIES)),
    ("ENTITY_FILTER", ENTITY_FILTER or "<none>"),
    ("LAST_SEEN_UPDATE_KEY", str(LAST_SEEN_UPDATE_KEY)),
    ("REQUIRE_FULL_AFFECTED_WINDOW", str(REQUIRE_FULL_AFFECTED_WINDOW)),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_PARITY:
    print("RUN_PARITY is False. Review the configuration above, set RUN_PARITY = True, then rerun manually.")
    dbutils.notebook.exit("RUN_PARITY_FALSE")

# COMMAND ----------


def _get_notebook_path() -> str:
    try:
        path = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
    except Exception:
        return ""
    return f"/Workspace{path}" if path and not path.startswith("/Workspace") else path


def _resolve_project_root() -> Path:
    candidates = []
    notebook_path = _get_notebook_path()
    if notebook_path:
        notebook_file = Path(notebook_path)
        candidates.extend([notebook_file.parent, *notebook_file.parent.parents])
    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "src" / "ml_project" / "stage30b_dirty_keys.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root containing src/ml_project/stage30b_dirty_keys.py")


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_project.stage30b_dirty_keys import (  # noqa: E402
    extract_dirty_leg_keys,
    map_dirty_legs_to_taxi_out_events,
    select_current_latest,
)
from ml_project.stage30b_taxi_out_candidate import (  # noqa: E402
    DATE_COL,
    ENTITY_COL,
    AFFECTED_OUTPUT_DATE_COL,
    NON_EMA_PARITY_COLUMNS,
    build_taxi_out_candidate_for_affected_outputs,
    compare_taxi_out_candidate_to_current_mv,
    expand_dirty_taxi_out_events_to_affected_outputs,
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded local helper modules from: {SRC_PATH}")

# COMMAND ----------

leg_src = spark.table(source_table(SOURCE_LEG_TABLE))
leg_times_src = spark.table(source_table(SOURCE_LEG_TIMES_TABLE))
cleaned = spark.table(silver_table(CLEANED_TABLE))
current_mv = spark.table(silver_table(CURRENT_MV_TABLE))

max_current_mv_event_date_row = current_mv.agg(F.max(DATE_COL).alias("max_current_mv_event_date")).first()
MAX_CURRENT_MV_EVENT_DATE = max_current_mv_event_date_row["max_current_mv_event_date"]

print(f"MAX_CURRENT_MV_EVENT_DATE: {MAX_CURRENT_MV_EVENT_DATE}")
print(
    "Full affected-window eligibility condition: "
    "date_add(dirty_event_date, 30) <= MAX_CURRENT_MV_EVENT_DATE"
)
if MAX_CURRENT_MV_EVENT_DATE is None:
    print("Current MV has no max event_date. Parity sample is inconclusive; stop without failure.")
    dbutils.notebook.exit("NO_CURRENT_MV_EVENT_DATE")


def latest_update_key_threshold(source_dfs, batches: int):
    if LAST_SEEN_UPDATE_KEY is not None:
        return LAST_SEEN_UPDATE_KEY, ["manual"]

    keys_df = None
    for df in source_dfs:
        next_keys = df.select("update_key").where(F.col("update_key").isNotNull()).distinct()
        keys_df = next_keys if keys_df is None else keys_df.unionByName(next_keys)

    ordered = keys_df.distinct().orderBy(F.col("update_key").desc()).limit(batches + 1)
    keys = [row["update_key"] for row in ordered.collect()]
    if len(keys) <= batches:
        raise ValueError(
            "Not enough distinct update_key batches to derive a safe threshold. "
            "Set LAST_SEEN_UPDATE_KEY manually."
        )
    return keys[batches], keys[:batches]


last_seen_update_key, selected_update_keys = latest_update_key_threshold(
    [leg_src, leg_times_src],
    LATEST_UPDATE_KEY_BATCHES,
)

print(f"last_seen_update_key threshold: {last_seen_update_key}")
print(f"selected latest update_key batches: {selected_update_keys}")

# COMMAND ----------

dirty_leg = extract_dirty_leg_keys(leg_src, last_seen_update_key, "leg")
dirty_leg_times = extract_dirty_leg_keys(leg_times_src, last_seen_update_key, "leg_times")

dirty_legs = dirty_leg.unionByName(dirty_leg_times).dropDuplicates(["leg_no", "dirty_source_alias"])
dirty_legs_limited = dirty_legs.orderBy(F.desc("max_update_key"), "leg_no", "dirty_source_alias").limit(MAX_DIRTY_LEGS)

dirty_leg_count = dirty_legs_limited.count()
print(f"dirty leg/source candidates after cap: {dirty_leg_count}")
display(dirty_legs_limited)

# COMMAND ----------

current_leg = select_current_latest(leg_src, partition_cols=("leg_no",))
dirty_events = map_dirty_legs_to_taxi_out_events(
    dirty_legs_limited,
    current_leg,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)

if REQUIRE_FULL_AFFECTED_WINDOW:
    dirty_events = dirty_events.where(
        F.date_add(F.col("dirty_event_date"), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE)
    )
    eligible_count = dirty_events.count()
    print(f"dirty taxi-out events eligible for full affected-window comparison: {eligible_count}")
    if eligible_count == 0:
        print("No eligible dirty taxi-out events remain after full affected-window filtering.")
        print("The latest update_key sample maps only to too-recent events.")
        print("Increase LATEST_UPDATE_KEY_BATCHES or provide an older LAST_SEEN_UPDATE_KEY.")
        dbutils.notebook.exit("NO_FULL_WINDOW_ELIGIBLE_DIRTY_EVENTS")

if ENTITY_FILTER:
    dirty_events = dirty_events.where(F.col(ENTITY_COL) == F.lit(ENTITY_FILTER))

selected_entities = dirty_events.select(ENTITY_COL).distinct().orderBy(ENTITY_COL).limit(MAX_AFFECTED_ENTITIES)
dirty_events = dirty_events.join(selected_entities, on=ENTITY_COL, how="inner")

dirty_event_count = dirty_events.count()
print(f"dirty taxi-out events after entity cap: {dirty_event_count}")
if dirty_event_count == 0:
    print("No dirty taxi-out events remain after optional entity filter and entity cap.")
    dbutils.notebook.exit("NO_DIRTY_EVENTS_AFTER_ENTITY_FILTER")
display(dirty_events.orderBy(ENTITY_COL, "dirty_event_date", "leg_no"))

# COMMAND ----------

affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
affected_output_count = affected_outputs.count()
print(f"affected entity/output-date pairs: {affected_output_count}")
display(affected_outputs.orderBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL))

# COMMAND ----------

candidate_scoped = build_taxi_out_candidate_for_affected_outputs(
    cleaned,
    affected_outputs,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)
candidate_count = candidate_scoped.count()
print(f"candidate scoped non-EMA rows: {candidate_count}")
display(candidate_scoped.orderBy(ENTITY_COL, DATE_COL).limit(50))

affected_pairs = affected_outputs.select(
    ENTITY_COL,
    F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
).dropDuplicates()

current_scoped = (
    current_mv.select(*NON_EMA_PARITY_COLUMNS)
    .join(affected_pairs, on=[ENTITY_COL, DATE_COL], how="inner")
)
current_count = current_scoped.count()
print(f"current MV scoped non-EMA rows: {current_count}")
display(current_scoped.orderBy(ENTITY_COL, DATE_COL).limit(50))

# COMMAND ----------

parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped, tolerance=TOLERANCE)
status_counts = parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))

print("parity status counts")
display(status_counts)

print("top parity mismatches")
display(
    parity.where(F.col("parity_status") != F.lit("matched"))
    .orderBy(ENTITY_COL, DATE_COL)
    .limit(50)
)

print("Stage 30B-3 read-only parity validation completed.")
