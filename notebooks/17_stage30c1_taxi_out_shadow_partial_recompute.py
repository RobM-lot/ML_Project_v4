# Databricks notebook source
# ruff: noqa: F821
# Stage 30C-1 shadow partial recompute path for ft_airport_daily_taxi_out.

from pathlib import Path
import sys
from uuid import uuid4

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30C-1 taxi-out shadow partial recompute. "
    "Default mode is config-only; dev-shadow writes are gated and default off."
)
print("=" * 100)

# COMMAND ----------

RUN_SHADOW_PIPELINE = False

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"

SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"

CLEANED_FLIGHT_TABLE = "panda_silver_dev.ml_ops.cleaned_flight_data_full_table"
CURRENT_TAXI_OUT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"

SHADOW_TAXI_OUT_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

LEG_CDF_STARTING_VERSION = None
LEG_CDF_ENDING_VERSION = None

LEG_TIMES_CDF_STARTING_VERSION = None
LEG_TIMES_CDF_ENDING_VERSION = None

HISTORY_START = "2023-07-01"
DATA_CUTOFF_DATE = "2027-01-01"

ENTITY_FILTER = ""
MAX_DIRTY_EVENTS = 1000
MAX_AFFECTED_ENTITIES = 1
MAX_SAMPLE_ROWS = 100
TOLERANCE = 1e-6

REQUIRE_FULL_AFFECTED_WINDOW = False

DRY_RUN_ONLY = True

ALLOW_CREATE_SHADOW_TABLE = False
ALLOW_CREATE_WATERMARK_TABLE = False
ALLOW_SHADOW_MERGE = False
ALLOW_WATERMARK_ADVANCE = False

WRITE_CONFIRMATION = ""
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


WRITE_FLAGS = {
    "ALLOW_CREATE_SHADOW_TABLE": ALLOW_CREATE_SHADOW_TABLE,
    "ALLOW_CREATE_WATERMARK_TABLE": ALLOW_CREATE_WATERMARK_TABLE,
    "ALLOW_SHADOW_MERGE": ALLOW_SHADOW_MERGE,
    "ALLOW_WATERMARK_ADVANCE": ALLOW_WATERMARK_ADVANCE,
}

CONFIG_ROWS = [
    ("RUN_SHADOW_PIPELINE", str(RUN_SHADOW_PIPELINE)),
    ("SOURCE_LEG", source_table(SOURCE_LEG_TABLE)),
    ("SOURCE_LEG_TIMES", source_table(SOURCE_LEG_TIMES_TABLE)),
    ("CLEANED_FLIGHT_TABLE", CLEANED_FLIGHT_TABLE),
    ("CURRENT_TAXI_OUT_MV", CURRENT_TAXI_OUT_MV),
    ("SHADOW_TAXI_OUT_TABLE", SHADOW_TAXI_OUT_TABLE),
    ("WATERMARK_TABLE", WATERMARK_TABLE),
    ("LEG_CDF_STARTING_VERSION", str(LEG_CDF_STARTING_VERSION)),
    ("LEG_CDF_ENDING_VERSION", str(LEG_CDF_ENDING_VERSION)),
    ("LEG_TIMES_CDF_STARTING_VERSION", str(LEG_TIMES_CDF_STARTING_VERSION)),
    ("LEG_TIMES_CDF_ENDING_VERSION", str(LEG_TIMES_CDF_ENDING_VERSION)),
    ("HISTORY_START", HISTORY_START),
    ("DATA_CUTOFF_DATE", DATA_CUTOFF_DATE),
    ("ENTITY_FILTER", ENTITY_FILTER or "<none>"),
    ("MAX_DIRTY_EVENTS", str(MAX_DIRTY_EVENTS)),
    ("MAX_AFFECTED_ENTITIES", str(MAX_AFFECTED_ENTITIES)),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
    ("TOLERANCE", str(TOLERANCE)),
    ("REQUIRE_FULL_AFFECTED_WINDOW", str(REQUIRE_FULL_AFFECTED_WINDOW)),
    ("DRY_RUN_ONLY", str(DRY_RUN_ONLY)),
    ("ALLOW_CREATE_SHADOW_TABLE", str(ALLOW_CREATE_SHADOW_TABLE)),
    ("ALLOW_CREATE_WATERMARK_TABLE", str(ALLOW_CREATE_WATERMARK_TABLE)),
    ("ALLOW_SHADOW_MERGE", str(ALLOW_SHADOW_MERGE)),
    ("ALLOW_WATERMARK_ADVANCE", str(ALLOW_WATERMARK_ADVANCE)),
    ("WRITE_CONFIRMATION_PRESENT", str(bool(WRITE_CONFIRMATION))),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_SHADOW_PIPELINE:
    print(
        "RUN_SHADOW_PIPELINE is False. Review the configuration above. "
        "No source, current MV, shadow, or control tables were read."
    )
    dbutils.notebook.exit("RUN_SHADOW_PIPELINE_FALSE")

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
        if (candidate / "src" / "ml_project" / "stage30c_taxi_out_shadow.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root containing src/ml_project/stage30c_taxi_out_shadow.py")


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_project.stage30b_dirty_keys import (  # noqa: E402
    map_dirty_legs_to_taxi_out_events,
    select_current_latest,
)
from ml_project.stage30b_taxi_out_candidate import (  # noqa: E402
    AFFECTED_OUTPUT_DATE_COL,
    DATE_COL,
    ENTITY_COL,
    NON_EMA_PARITY_COLUMNS,
    build_taxi_out_candidate_for_affected_outputs,
    compare_taxi_out_candidate_to_current_mv,
    expand_dirty_taxi_out_events_to_affected_outputs,
)
from ml_project.stage30c_taxi_out_shadow import (  # noqa: E402
    SHADOW_CANDIDATE_FLAG_COL,
    TARGET_KEY_COLS,
    build_advance_watermark_sqls,
    build_create_shadow_table_sql,
    build_create_watermark_table_sql,
    build_shadow_merge_sql,
    build_shadow_replace_source,
    describe_shadow_write_plan,
    ensure_expected_columns,
    require_shadow_write_confirmation,
    validate_dev_shadow_table_name,
    validate_source_cdf_windows,
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded local helper modules from: {SRC_PATH}")

# COMMAND ----------

validate_dev_shadow_table_name(SHADOW_TAXI_OUT_TABLE, expected_suffix_or_token="shadow")
validate_dev_shadow_table_name(WATERMARK_TABLE, expected_suffix_or_token="watermark")

write_mode_requested = not DRY_RUN_ONLY or any(WRITE_FLAGS.values())
if write_mode_requested:
    require_shadow_write_confirmation(
        dry_run_only=DRY_RUN_ONLY,
        write_confirmation=WRITE_CONFIRMATION,
        required_write_confirmation=REQUIRED_WRITE_CONFIRMATION,
        write_flags=WRITE_FLAGS,
    )

cdf_windows = validate_source_cdf_windows(
    leg_starting_version=LEG_CDF_STARTING_VERSION,
    leg_ending_version=LEG_CDF_ENDING_VERSION,
    leg_times_starting_version=LEG_TIMES_CDF_STARTING_VERSION,
    leg_times_ending_version=LEG_TIMES_CDF_ENDING_VERSION,
    require_configured=True,
)

write_plan = describe_shadow_write_plan(
    shadow_table=SHADOW_TAXI_OUT_TABLE,
    watermark_table=WATERMARK_TABLE,
    write_flags=WRITE_FLAGS,
    dry_run_only=DRY_RUN_ONLY,
)
print("Intended Stage 30C-1 write plan. Disabled operations are not executed.")
display(spark.createDataFrame(write_plan))

# COMMAND ----------

STAGE_NAME = "stage30c1_taxi_out_shadow_partial_recompute"
RUN_ID = str(uuid4())
SHADOW_MERGE_SOURCE_VIEW = "stage30c1_taxi_out_shadow_merge_source"

CDF_COLUMNS = (
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
)

SUPPORTED_CDF_CHANGE_TYPES = (
    "insert",
    "update_preimage",
    "update_postimage",
    "delete",
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

summary_flags = {
    "read_cdf_leg_ok": False,
    "read_cdf_leg_times_ok": False,
    "dirty_events_available": False,
    "affected_pairs_available": False,
    "candidate_rows_available": False,
    "candidate_key_unique": False,
    "candidate_key_non_null": False,
    "current_overlap_compare_ok": False,
    "shadow_table_name_safe": True,
    "watermark_table_name_safe": True,
    "dry_run_only": DRY_RUN_ONLY,
    "shadow_table_created": False,
    "shadow_merge_executed": False,
    "shadow_post_merge_validation_ok": False,
    "watermarks_advanced": False,
    "read_only_or_dev_shadow_only": True,
}


def _read_batch_cdf(table_name: str, starting_version, ending_version):
    reader = spark.read.option("readChangeFeed", "true").option("startingVersion", str(starting_version))
    if ending_version is not None:
        reader = reader.option("endingVersion", str(ending_version))
    return reader.table(source_table(table_name))


def _missing_columns(df, required_columns: tuple[str, ...]) -> list[str]:
    return list(ensure_expected_columns(df.columns, required_columns))


def _extract_dirty_leg_keys_from_cdf(cdf_df, source_alias: str):
    return (
        cdf_df.where(F.col("leg_no").isNotNull())
        .select(
            F.col("leg_no"),
            F.lit(source_alias).alias("dirty_source_alias"),
            F.col("update_key").alias("_stage30c_update_key"),
            F.col("_change_type"),
            F.col("_commit_version"),
            F.col("_commit_timestamp"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(
            F.max("_stage30c_update_key").alias("max_update_key"),
            F.min("_commit_version").alias("min_commit_version"),
            F.max("_commit_version").alias("max_commit_version"),
            F.max("_commit_timestamp").alias("latest_commit_timestamp"),
            F.collect_set("_change_type").alias("cdf_change_types"),
        )
    )


def _table_exists(table_name: str) -> bool:
    try:
        return bool(spark.catalog.tableExists(table_name))
    except Exception as exc:
        print(f"Unable to check table existence for {table_name}: {exc}")
        return False


def _count_duplicate_keys(df, key_cols: tuple[str, ...]) -> int:
    return df.groupBy(*key_cols).count().where(F.col("count") > F.lit(1)).count()


def _count_null_keys(df, key_cols: tuple[str, ...]) -> int:
    null_expr = None
    for key_col in key_cols:
        expr = F.col(key_col).isNull()
        null_expr = expr if null_expr is None else null_expr | expr
    return df.where(null_expr).count()


def _display_summary() -> None:
    print("Stage 30C-1 final boolean summary")
    display(spark.createDataFrame([(name, bool(value)) for name, value in summary_flags.items()], ["check_name", "passed"]))


print("Supported CDF change types for dirty detection:")
display(spark.createDataFrame([(change_type,) for change_type in SUPPORTED_CDF_CHANGE_TYPES], ["_change_type"]))
print("update_preimage/update_postimage/insert rows are dirty-key signals; removal edge cases remain guarded by validation.")

# COMMAND ----------

dirty_key_dfs = []
source_commit_versions = {}
source_commit_timestamps = {}
read_cdf_ok_by_alias = {}

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

    print(f"{source_alias} CDF sample")
    display(cdf_df.limit(MAX_SAMPLE_ROWS))
    print(f"{source_alias} _change_type distribution")
    display(cdf_df.groupBy("_change_type").count().orderBy(F.desc("count"), "_change_type"))

    cdf_summary = cdf_df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("leg_no").alias("unique_leg_no_count"),
        F.min("update_key").alias("min_update_key"),
        F.max("update_key").alias("max_update_key"),
        F.min("_commit_version").alias("min_commit_version"),
        F.max("_commit_version").alias("max_commit_version"),
        F.max("_commit_timestamp").alias("latest_commit_timestamp"),
    ).first()
    print(
        f"{source_alias} CDF rows={cdf_summary['row_count']}, "
        f"unique_leg_no={cdf_summary['unique_leg_no_count']}, "
        f"commit_range={cdf_summary['min_commit_version']}..{cdf_summary['max_commit_version']}, "
        f"update_key_range={cdf_summary['min_update_key']}..{cdf_summary['max_update_key']}"
    )
    source_commit_versions[source_alias] = cdf_summary["max_commit_version"]
    source_commit_timestamps[source_alias] = (
        None if cdf_summary["latest_commit_timestamp"] is None else str(cdf_summary["latest_commit_timestamp"])
    )
    read_cdf_ok_by_alias[source_alias] = True
    summary_flags[f"read_cdf_{source_alias}_ok"] = True
    dirty_key_dfs.append(_extract_dirty_leg_keys_from_cdf(cdf_df, source_alias))

if not dirty_key_dfs:
    raise ValueError("No CDF sources were configured for dirty-key extraction.")

dirty_legs = dirty_key_dfs[0]
for next_dirty_keys in dirty_key_dfs[1:]:
    dirty_legs = dirty_legs.unionByName(next_dirty_keys)

dirty_legs = dirty_legs.dropDuplicates(["leg_no", "dirty_source_alias"])
dirty_leg_source_count = dirty_legs.count()
print(f"dirty leg/source candidates before mapping: {dirty_leg_source_count}")
display(dirty_legs.orderBy(F.desc("max_commit_version"), F.desc("max_update_key"), "leg_no").limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

leg_src_current = spark.table(source_table(SOURCE_LEG_TABLE))
cleaned = spark.table(CLEANED_FLIGHT_TABLE)
current_mv = spark.table(CURRENT_TAXI_OUT_MV)

max_current_mv_event_date_row = current_mv.agg(F.max(DATE_COL).alias("max_current_mv_event_date")).first()
MAX_CURRENT_MV_EVENT_DATE = max_current_mv_event_date_row["max_current_mv_event_date"]
print(f"MAX_CURRENT_MV_EVENT_DATE: {MAX_CURRENT_MV_EVENT_DATE}")

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
    print("Applying full affected-window eligibility: date_add(dirty_event_date, 30) <= MAX_CURRENT_MV_EVENT_DATE")
    if MAX_CURRENT_MV_EVENT_DATE is None:
        raise ValueError("Cannot require full affected-window eligibility when the current MV has no max event_date.")
    dirty_events = dirty_events.where(F.date_add(F.col("dirty_event_date"), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE))

full_window_eligible_count = dirty_events.count()
print(f"mapped dirty taxi-out events after full-window eligibility: {full_window_eligible_count}")

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

selected_entities = dirty_events_limited.select(ENTITY_COL).distinct().orderBy(ENTITY_COL).limit(MAX_AFFECTED_ENTITIES)
dirty_events = dirty_events_limited.join(selected_entities, on=ENTITY_COL, how="inner")

dirty_event_count = dirty_events.count()
summary_flags["dirty_events_available"] = dirty_event_count > 0
print(f"mapped dirty taxi-out events after caps: {dirty_event_count}")
display(dirty_events.orderBy(ENTITY_COL, "dirty_event_date", "leg_no").limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

print("Dirty event date D expands to affected output dates D+1...D+30. EMA remains deferred.")
affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
affected_output_count = affected_outputs.count()
summary_flags["affected_pairs_available"] = affected_output_count > 0
print(f"affected entity/output-date pairs: {affected_output_count}")
display(affected_outputs.orderBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL).limit(MAX_SAMPLE_ROWS))

affected_pairs = affected_outputs.select(
    ENTITY_COL,
    F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
).dropDuplicates()
affected_pair_duplicate_count = _count_duplicate_keys(affected_pairs, TARGET_KEY_COLS)
print(f"affected dep_ap_sched/event_date duplicate keys: {affected_pair_duplicate_count}")

# COMMAND ----------

candidate_scoped = build_taxi_out_candidate_for_affected_outputs(
    cleaned,
    affected_outputs,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)
candidate_count = candidate_scoped.count()
summary_flags["candidate_rows_available"] = candidate_count > 0
print(f"candidate scoped non-EMA rows: {candidate_count}")
display(candidate_scoped.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

missing_candidate_columns = _missing_columns(candidate_scoped, NON_EMA_PARITY_COLUMNS)
if missing_candidate_columns:
    raise ValueError(f"Candidate is missing expected non-EMA columns: {missing_candidate_columns}")

candidate_duplicate_key_count = _count_duplicate_keys(candidate_scoped, TARGET_KEY_COLS)
candidate_null_key_count = _count_null_keys(candidate_scoped, TARGET_KEY_COLS)
summary_flags["candidate_key_unique"] = candidate_duplicate_key_count == 0
summary_flags["candidate_key_non_null"] = candidate_null_key_count == 0
print(f"candidate duplicate dep_ap_sched/event_date keys: {candidate_duplicate_key_count}")
print(f"candidate null dep_ap_sched/event_date keys: {candidate_null_key_count}")

current_scoped = current_mv.select(*NON_EMA_PARITY_COLUMNS).join(affected_pairs, on=[ENTITY_COL, DATE_COL], how="inner")
current_count = current_scoped.count()
print(f"current MV scoped non-EMA rows: {current_count}")
display(current_scoped.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

current_parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped, tolerance=TOLERANCE)
current_status_counts = current_parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
print("candidate/current scoped compare status counts")
display(current_status_counts)
display(
    current_parity.where(F.col("parity_status") != F.lit("matched"))
    .orderBy(ENTITY_COL, DATE_COL)
    .limit(MAX_SAMPLE_ROWS)
)

current_status_rows = {row["parity_status"]: row["rows"] for row in current_status_counts.collect()}
if current_count > 0:
    summary_flags["current_overlap_compare_ok"] = all(
        status == "matched" for status, rows in current_status_rows.items() if rows > 0
    )
else:
    summary_flags["current_overlap_compare_ok"] = True

# COMMAND ----------

shadow_merge_source = build_shadow_replace_source(affected_pairs, candidate_scoped)
print("shadow merge source sample")
display(shadow_merge_source.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

shadow_merge_source_count = shadow_merge_source.count()
shadow_delete_candidate_count = shadow_merge_source.where(~F.col(SHADOW_CANDIDATE_FLAG_COL)).count()
print(f"shadow merge source rows: {shadow_merge_source_count}")
print(f"affected keys that would delete from shadow because no candidate exists: {shadow_delete_candidate_count}")

shadow_table_exists = _table_exists(SHADOW_TAXI_OUT_TABLE)
print(f"shadow table exists: {shadow_table_exists}")
if shadow_table_exists:
    shadow_table = spark.table(SHADOW_TAXI_OUT_TABLE)
    shadow_duplicate_key_count = _count_duplicate_keys(shadow_table, TARGET_KEY_COLS)
    print(f"shadow target duplicate dep_ap_sched/event_date keys before merge: {shadow_duplicate_key_count}")
    if shadow_duplicate_key_count:
        raise ValueError("Shadow target has duplicate dep_ap_sched/event_date keys before merge.")
else:
    print("Shadow target key uniqueness check skipped because the shadow table does not exist yet.")

print("Dry-run write plan. No SQL below is executed while DRY_RUN_ONLY is True.")
print(build_create_shadow_table_sql(shadow_table=SHADOW_TAXI_OUT_TABLE, current_mv_table=CURRENT_TAXI_OUT_MV))
print(build_create_watermark_table_sql(watermark_table=WATERMARK_TABLE))
print(
    build_shadow_merge_sql(
        shadow_table=SHADOW_TAXI_OUT_TABLE,
        merge_source_view=SHADOW_MERGE_SOURCE_VIEW,
        candidate_columns=NON_EMA_PARITY_COLUMNS,
    )
)

# COMMAND ----------

shadow_table_created = False
shadow_merge_succeeded = False
shadow_post_merge_validation_ok = False
watermarks_advanced = False

if ALLOW_CREATE_SHADOW_TABLE:
    sql_text = build_create_shadow_table_sql(shadow_table=SHADOW_TAXI_OUT_TABLE, current_mv_table=CURRENT_TAXI_OUT_MV)
    print(f"Creating shadow table if needed: {SHADOW_TAXI_OUT_TABLE}")
    spark.sql(sql_text)
    shadow_table_created = True
    summary_flags["shadow_table_created"] = True

if ALLOW_CREATE_WATERMARK_TABLE:
    sql_text = build_create_watermark_table_sql(watermark_table=WATERMARK_TABLE)
    print(f"Creating watermark table if needed: {WATERMARK_TABLE}")
    spark.sql(sql_text)

if ALLOW_SHADOW_MERGE:
    if affected_output_count == 0:
        raise ValueError("Shadow merge requested but no affected output pairs are available.")
    if candidate_duplicate_key_count != 0 or candidate_null_key_count != 0:
        raise ValueError("Shadow merge requested but candidate key validation failed.")

    shadow_merge_source.createOrReplaceTempView(SHADOW_MERGE_SOURCE_VIEW)
    sql_text = build_shadow_merge_sql(
        shadow_table=SHADOW_TAXI_OUT_TABLE,
        merge_source_view=SHADOW_MERGE_SOURCE_VIEW,
        candidate_columns=NON_EMA_PARITY_COLUMNS,
    )
    print(f"Merging affected keys into dev shadow table: {SHADOW_TAXI_OUT_TABLE}")
    spark.sql(sql_text)
    shadow_merge_succeeded = True
    summary_flags["shadow_merge_executed"] = True

    shadow_scoped = spark.table(SHADOW_TAXI_OUT_TABLE).select(*NON_EMA_PARITY_COLUMNS).join(
        affected_pairs,
        on=[ENTITY_COL, DATE_COL],
        how="inner",
    )
    shadow_duplicate_key_count = _count_duplicate_keys(shadow_scoped, TARGET_KEY_COLS)
    shadow_null_key_count = _count_null_keys(shadow_scoped, TARGET_KEY_COLS)
    print(f"shadow scoped duplicate dep_ap_sched/event_date keys after merge: {shadow_duplicate_key_count}")
    print(f"shadow scoped null dep_ap_sched/event_date keys after merge: {shadow_null_key_count}")

    shadow_parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, shadow_scoped, tolerance=TOLERANCE)
    shadow_status_counts = shadow_parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
    print("candidate/shadow scoped compare status counts after merge")
    display(shadow_status_counts)
    display(
        shadow_parity.where(F.col("parity_status") != F.lit("matched"))
        .orderBy(ENTITY_COL, DATE_COL)
        .limit(MAX_SAMPLE_ROWS)
    )
    shadow_status_rows = {row["parity_status"]: row["rows"] for row in shadow_status_counts.collect()}
    shadow_post_merge_validation_ok = (
        shadow_duplicate_key_count == 0
        and shadow_null_key_count == 0
        and all(status == "matched" for status, rows in shadow_status_rows.items() if rows > 0)
    )
    summary_flags["shadow_post_merge_validation_ok"] = shadow_post_merge_validation_ok

if ALLOW_WATERMARK_ADVANCE:
    if not (shadow_merge_succeeded and shadow_post_merge_validation_ok):
        raise ValueError("Watermark advancement requires a successful shadow merge and post-merge validation.")

    processed_versions = {
        source_alias: int(version)
        for source_alias, version in source_commit_versions.items()
        if version is not None and read_cdf_ok_by_alias.get(source_alias)
    }
    source_tables_by_alias = {
        "leg": source_table(SOURCE_LEG_TABLE),
        "leg_times": source_table(SOURCE_LEG_TIMES_TABLE),
    }
    watermark_sqls = build_advance_watermark_sqls(
        watermark_table=WATERMARK_TABLE,
        stage_name=STAGE_NAME,
        source_tables_by_alias=source_tables_by_alias,
        processed_versions_by_alias=processed_versions,
        processed_timestamps_by_alias=source_commit_timestamps,
        last_successful_run_id=RUN_ID,
    )
    for sql_text in watermark_sqls:
        spark.sql(sql_text)
    watermarks_advanced = bool(watermark_sqls)
    summary_flags["watermarks_advanced"] = watermarks_advanced

summary_flags["shadow_table_created"] = shadow_table_created
summary_flags["shadow_merge_executed"] = shadow_merge_succeeded
summary_flags["shadow_post_merge_validation_ok"] = shadow_post_merge_validation_ok
summary_flags["watermarks_advanced"] = watermarks_advanced
summary_flags["read_only_or_dev_shadow_only"] = True

_display_summary()
print("Stage 30C-1 taxi-out shadow partial recompute completed.")
