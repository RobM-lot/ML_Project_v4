# Databricks notebook source
# Stage 30B-5 read-only end-to-end batch CDF parity POC.

from pathlib import Path
import sys

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30B-5 read-only end-to-end batch CDF parity POC. "
    "This notebook must not mutate production tables or workspace resources."
)
print("=" * 100)

# COMMAND ----------

RUN_PARITY = False

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"

SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"

CLEANED_FLIGHT_TABLE = "panda_silver_dev.ml_ops.cleaned_flight_data_full_table"
CURRENT_TAXI_OUT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"

LEG_CDF_STARTING_VERSION = None
LEG_CDF_ENDING_VERSION = None

LEG_TIMES_CDF_STARTING_VERSION = None
LEG_TIMES_CDF_ENDING_VERSION = None

HISTORY_START = "2023-07-01"
DATA_CUTOFF_DATE = "2027-01-01"

REQUIRE_FULL_AFFECTED_WINDOW = True
ENTITY_FILTER = ""
MAX_DIRTY_EVENTS = 1000
MAX_AFFECTED_ENTITIES = 1
MAX_SAMPLE_ROWS = 100
TOLERANCE = 1e-9


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


CONFIG_ROWS = [
    ("RUN_PARITY", str(RUN_PARITY)),
    ("SOURCE_LEG", source_table(SOURCE_LEG_TABLE)),
    ("SOURCE_LEG_TIMES", source_table(SOURCE_LEG_TIMES_TABLE)),
    ("CLEANED_FLIGHT_TABLE", CLEANED_FLIGHT_TABLE),
    ("CURRENT_TAXI_OUT_MV", CURRENT_TAXI_OUT_MV),
    ("LEG_CDF_STARTING_VERSION", str(LEG_CDF_STARTING_VERSION)),
    ("LEG_CDF_ENDING_VERSION", str(LEG_CDF_ENDING_VERSION)),
    ("LEG_TIMES_CDF_STARTING_VERSION", str(LEG_TIMES_CDF_STARTING_VERSION)),
    ("LEG_TIMES_CDF_ENDING_VERSION", str(LEG_TIMES_CDF_ENDING_VERSION)),
    ("HISTORY_START", HISTORY_START),
    ("DATA_CUTOFF_DATE", DATA_CUTOFF_DATE),
    ("REQUIRE_FULL_AFFECTED_WINDOW", str(REQUIRE_FULL_AFFECTED_WINDOW)),
    ("ENTITY_FILTER", ENTITY_FILTER or "<none>"),
    ("MAX_DIRTY_EVENTS", str(MAX_DIRTY_EVENTS)),
    ("MAX_AFFECTED_ENTITIES", str(MAX_AFFECTED_ENTITIES)),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
    ("TOLERANCE", str(TOLERANCE)),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_PARITY:
    print("RUN_PARITY is False. Review the configuration above, set explicit CDF versions, then rerun manually.")
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

CDF_COLUMNS = (
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
)

SOURCE_SPECS = [
    {
        "source_alias": "leg",
        "table_name": SOURCE_LEG_TABLE,
        "starting_version": LEG_CDF_STARTING_VERSION,
        "ending_version": LEG_CDF_ENDING_VERSION,
        "required_columns": (
            "leg_no",
            "update_key",
            "__START_AT",
            "__END_AT",
            "dep_ap_sched",
            "dep_sched_dt",
            "leg_state",
            "leg_type",
            "counter",
            *CDF_COLUMNS,
        ),
    },
    {
        "source_alias": "leg_times",
        "table_name": SOURCE_LEG_TIMES_TABLE,
        "starting_version": LEG_TIMES_CDF_STARTING_VERSION,
        "ending_version": LEG_TIMES_CDF_ENDING_VERSION,
        "required_columns": (
            "leg_no",
            "update_key",
            "__START_AT",
            "__END_AT",
            "offblock_dt",
            "airborne_dt",
            *CDF_COLUMNS,
        ),
    },
]


def _validate_cdf_versions() -> None:
    configured_sources = [
        spec["source_alias"]
        for spec in SOURCE_SPECS
        if spec["starting_version"] is not None
    ]
    if not configured_sources:
        raise ValueError("At least one source-specific CDF starting version must be set.")

    for spec in SOURCE_SPECS:
        source_alias = spec["source_alias"]
        starting_version = spec["starting_version"]
        ending_version = spec["ending_version"]
        if ending_version is not None and starting_version is None:
            raise ValueError(f"{source_alias} ending version requires a starting version.")
        if (
            starting_version is not None
            and ending_version is not None
            and int(ending_version) < int(starting_version)
        ):
            raise ValueError(f"{source_alias} ending version must be >= starting version.")


def _read_batch_cdf(table_name: str, starting_version, ending_version):
    reader = spark.read.option("readChangeFeed", "true").option("startingVersion", str(starting_version))
    if ending_version is not None:
        reader = reader.option("endingVersion", str(ending_version))
    return reader.table(source_table(table_name))


def _missing_columns(df, required_columns: tuple[str, ...]) -> list[str]:
    available = set(df.columns)
    return [column for column in required_columns if column not in available]


def _extract_dirty_leg_keys_from_cdf(cdf_df, source_alias: str):
    return (
        cdf_df.where(F.col("leg_no").isNotNull())
        .select(
            F.col("leg_no"),
            F.lit(source_alias).alias("dirty_source_alias"),
            F.col("update_key").alias("_stage30b_update_key"),
            F.col("_change_type"),
            F.col("_commit_version"),
            F.col("_commit_timestamp"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(
            F.max("_stage30b_update_key").alias("max_update_key"),
            F.min("_commit_version").alias("min_commit_version"),
            F.max("_commit_version").alias("max_commit_version"),
            F.max("_commit_timestamp").alias("latest_commit_timestamp"),
            F.collect_set("_change_type").alias("cdf_change_types"),
        )
    )


def _summarize_cdf_rows(cdf_df, source_alias: str) -> None:
    print(f"{source_alias} CDF sample")
    display(cdf_df.limit(MAX_SAMPLE_ROWS))

    summary = cdf_df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("leg_no").alias("unique_leg_no_count"),
        F.min("update_key").alias("min_update_key"),
        F.max("update_key").alias("max_update_key"),
        F.min("_commit_version").alias("min_commit_version"),
        F.max("_commit_version").alias("max_commit_version"),
    )
    display(summary)

    print(f"{source_alias} _change_type distribution")
    display(cdf_df.groupBy("_change_type").count().orderBy(F.desc("count"), "_change_type"))

    if source_alias == "leg":
        print("leg_state distribution")
        display(cdf_df.groupBy("leg_state").count().orderBy(F.desc("count"), "leg_state"))
        arr_count = cdf_df.where(F.col("leg_state") == F.lit("ARR")).count()
        taxi_out_candidate_count = cdf_df.where(
            (F.col("counter") == F.lit(0))
            & F.col("leg_type").isin("J", "C", "G")
            & (F.col("leg_state") == F.lit("ARR"))
        ).count()
        print(f"leg ARR rows: {arr_count}")
        print(f"leg taxi-out production-filter candidate rows: {taxi_out_candidate_count}")

    if source_alias == "leg_times":
        offblock_count = cdf_df.where(F.col("offblock_dt").isNotNull()).count()
        airborne_count = cdf_df.where(F.col("airborne_dt").isNotNull()).count()
        print(f"leg_times rows with non-null offblock_dt: {offblock_count}")
        print(f"leg_times rows with non-null airborne_dt: {airborne_count}")


_validate_cdf_versions()

# COMMAND ----------

dirty_key_dfs = []

for spec in SOURCE_SPECS:
    source_alias = spec["source_alias"]
    starting_version = spec["starting_version"]
    if starting_version is None:
        print(f"Skipping {source_alias}: no CDF starting version configured.")
        continue

    print("-" * 100)
    print(
        f"Reading batch CDF for {source_alias}: "
        f"startingVersion={starting_version}, endingVersion={spec['ending_version']}"
    )
    cdf_df = _read_batch_cdf(spec["table_name"], starting_version, spec["ending_version"])
    cdf_df.printSchema()

    missing = _missing_columns(cdf_df, spec["required_columns"])
    if missing:
        raise ValueError(f"Missing required CDF columns for {source_alias}: {missing}")

    _summarize_cdf_rows(cdf_df, source_alias)
    dirty_key_dfs.append(_extract_dirty_leg_keys_from_cdf(cdf_df, source_alias))

if not dirty_key_dfs:
    raise ValueError("No CDF sources were configured for dirty-key extraction.")

dirty_legs = dirty_key_dfs[0]
for next_dirty_keys in dirty_key_dfs[1:]:
    dirty_legs = dirty_legs.unionByName(next_dirty_keys)

dirty_legs = dirty_legs.dropDuplicates(["leg_no", "dirty_source_alias"])
dirty_leg_source_count = dirty_legs.count()
print(f"batch CDF dirty leg/source candidates: {dirty_leg_source_count}")
display(dirty_legs.orderBy(F.desc("max_commit_version"), F.desc("max_update_key"), "leg_no").limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

leg_src_current = spark.table(source_table(SOURCE_LEG_TABLE))
cleaned = spark.table(CLEANED_FLIGHT_TABLE)
current_mv = spark.table(CURRENT_TAXI_OUT_MV)

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

# COMMAND ----------

current_leg = select_current_latest(leg_src_current, partition_cols=("leg_no",))

mapped_dirty_events = map_dirty_legs_to_taxi_out_events(
    dirty_legs,
    current_leg,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)
mapped_before_eligibility_count = mapped_dirty_events.count()
print(f"mapped dirty taxi-out events before eligibility: {mapped_before_eligibility_count}")

dirty_events = mapped_dirty_events
if ENTITY_FILTER:
    dirty_events = dirty_events.where(F.col(ENTITY_COL) == F.lit(ENTITY_FILTER))
entity_filtered_count = dirty_events.count()
print(f"mapped dirty taxi-out events after ENTITY_FILTER: {entity_filtered_count}")

if REQUIRE_FULL_AFFECTED_WINDOW:
    dirty_events = dirty_events.where(
        F.date_add(F.col("dirty_event_date"), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE)
    )

full_window_eligible_count = dirty_events.count()
print(f"mapped dirty taxi-out events after full-window eligibility: {full_window_eligible_count}")
if REQUIRE_FULL_AFFECTED_WINDOW and full_window_eligible_count == 0:
    print("No eligible dirty taxi-out events exist for the selected CDF versions/entity/window.")
    dbutils.notebook.exit("NO_FULL_WINDOW_ELIGIBLE_DIRTY_EVENTS")

dirty_leg_markers = dirty_legs.groupBy("leg_no").agg(
    F.max("max_update_key").alias("latest_dirty_update_key"),
    F.max("max_commit_version").alias("latest_commit_version"),
    F.array_distinct(F.flatten(F.collect_set("cdf_change_types"))).alias("cdf_change_types"),
)
dirty_events_with_metadata = dirty_events.join(dirty_leg_markers, on="leg_no", how="left")
dirty_events_limited = dirty_events_with_metadata.orderBy(
    F.desc("latest_commit_version"),
    F.desc("latest_dirty_update_key"),
    ENTITY_COL,
    "dirty_event_date",
    "leg_no",
).limit(MAX_DIRTY_EVENTS)

selected_entities = (
    dirty_events_limited.select(ENTITY_COL).distinct().orderBy(ENTITY_COL).limit(MAX_AFFECTED_ENTITIES)
)
dirty_events = dirty_events_limited.join(selected_entities, on=ENTITY_COL, how="inner")

dirty_event_count = dirty_events.count()
print(f"mapped dirty taxi-out events after caps: {dirty_event_count}")
if dirty_event_count == 0:
    print("No dirty taxi-out events remain after final dirty-event and entity caps.")
    dbutils.notebook.exit("NO_DIRTY_EVENTS_AFTER_CAP")
display(dirty_events.orderBy(ENTITY_COL, "dirty_event_date", "leg_no").limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
affected_output_count = affected_outputs.count()
print(f"affected entity/output-date pairs: {affected_output_count}")
display(affected_outputs.orderBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL).limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

candidate_scoped = build_taxi_out_candidate_for_affected_outputs(
    cleaned,
    affected_outputs,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)
candidate_count = candidate_scoped.count()
print(f"candidate scoped non-EMA rows: {candidate_count}")
display(candidate_scoped.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

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
display(current_scoped.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped, tolerance=TOLERANCE)
status_counts = parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))

print("parity status counts")
display(status_counts)

print("top parity mismatches")
display(
    parity.where(F.col("parity_status") != F.lit("matched"))
    .orderBy(ENTITY_COL, DATE_COL)
    .limit(MAX_SAMPLE_ROWS)
)

print("Stage 30B-5 read-only end-to-end batch CDF parity POC completed.")
