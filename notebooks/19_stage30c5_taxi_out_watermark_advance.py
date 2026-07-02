# Databricks notebook source
# ruff: noqa: F821
# Stage 30C-5 controlled taxi-out watermark advance preflight.

import json
from pathlib import Path
import sys
from uuid import uuid4

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30C-5 taxi-out watermark advance preflight. "
    "Defaults are safe: config-only, dry-run, no watermark advancement."
)
print("=" * 100)

# COMMAND ----------

RUN_WATERMARK_ADVANCE = False

DRY_RUN_ONLY = True

ALLOW_SHADOW_MERGE = False
ALLOW_WATERMARK_ADVANCE = False
ALLOW_WATERMARK_SCHEMA_MIGRATION = False

WRITE_CONFIRMATION = ""
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"

SOURCE_VERSION_MODE = "watermark"  # allowed: watermark, explicit
REQUIRE_CONTIGUOUS_SOURCE_WINDOWS = True
REQUIRE_EXISTING_WATERMARK_ROWS = True
REQUIRE_SHADOW_MERGE_BEFORE_WATERMARK = True
REQUIRE_POST_MERGE_VALIDATION = True
REQUIRE_NO_DUPLICATE_OR_NULL_KEYS = True
REQUIRE_NO_CANDIDATE_CURRENT_MISMATCHES = True
REQUIRE_NO_CANDIDATE_SHADOW_MISMATCHES = True

ALLOW_NON_WATERMARK_EXPLICIT_WINDOW = False
ALLOW_WATERMARK_BOOTSTRAP = False
BOOTSTRAP_LEG_VERSION = None
BOOTSTRAP_LEG_TIMES_VERSION = None

EXPLICIT_LEG_STARTING_VERSION = None
EXPLICIT_LEG_ENDING_VERSION = None
EXPLICIT_LEG_TIMES_STARTING_VERSION = None
EXPLICIT_LEG_TIMES_ENDING_VERSION = None

MAX_CDF_VERSION_SPAN_PER_SOURCE = 50
MAX_DIRTY_EVENTS = 5000
MAX_AFFECTED_ENTITIES = 20
MAX_SAMPLE_ROWS = 100
TOLERANCE = 1e-6
REQUIRE_FULL_AFFECTED_WINDOW = False

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"
SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"

SOURCE_LEG = "panda_silver_prod.occ_ops.netline___schedops__leg"
SOURCE_LEG_TIMES = "panda_silver_prod.occ_ops.netline___schedops__leg_times"
CLEANED_FLIGHT_TABLE = "panda_silver_dev.ml_ops.cleaned_flight_data_full_table"
CURRENT_TAXI_OUT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
SHADOW_TAXI_OUT_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

HISTORY_START = "2023-07-01"
DATA_CUTOFF_DATE = "2027-01-01"


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


def _display_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _display_metric_rows(mapping):
    return [(str(key), _display_value(value)) for key, value in mapping.items()]


CONFIG_ROWS = [
    ("RUN_WATERMARK_ADVANCE", str(RUN_WATERMARK_ADVANCE)),
    ("DRY_RUN_ONLY", str(DRY_RUN_ONLY)),
    ("ALLOW_SHADOW_MERGE", str(ALLOW_SHADOW_MERGE)),
    ("ALLOW_WATERMARK_ADVANCE", str(ALLOW_WATERMARK_ADVANCE)),
    ("ALLOW_WATERMARK_SCHEMA_MIGRATION", str(ALLOW_WATERMARK_SCHEMA_MIGRATION)),
    ("WRITE_CONFIRMATION_PRESENT", str(bool(WRITE_CONFIRMATION))),
    ("SOURCE_VERSION_MODE", SOURCE_VERSION_MODE),
    ("REQUIRE_CONTIGUOUS_SOURCE_WINDOWS", str(REQUIRE_CONTIGUOUS_SOURCE_WINDOWS)),
    ("REQUIRE_EXISTING_WATERMARK_ROWS", str(REQUIRE_EXISTING_WATERMARK_ROWS)),
    ("REQUIRE_SHADOW_MERGE_BEFORE_WATERMARK", str(REQUIRE_SHADOW_MERGE_BEFORE_WATERMARK)),
    ("REQUIRE_POST_MERGE_VALIDATION", str(REQUIRE_POST_MERGE_VALIDATION)),
    ("REQUIRE_NO_DUPLICATE_OR_NULL_KEYS", str(REQUIRE_NO_DUPLICATE_OR_NULL_KEYS)),
    ("REQUIRE_NO_CANDIDATE_CURRENT_MISMATCHES", str(REQUIRE_NO_CANDIDATE_CURRENT_MISMATCHES)),
    ("REQUIRE_NO_CANDIDATE_SHADOW_MISMATCHES", str(REQUIRE_NO_CANDIDATE_SHADOW_MISMATCHES)),
    ("ALLOW_NON_WATERMARK_EXPLICIT_WINDOW", str(ALLOW_NON_WATERMARK_EXPLICIT_WINDOW)),
    ("ALLOW_WATERMARK_BOOTSTRAP", str(ALLOW_WATERMARK_BOOTSTRAP)),
    ("BOOTSTRAP_LEG_VERSION", str(BOOTSTRAP_LEG_VERSION)),
    ("BOOTSTRAP_LEG_TIMES_VERSION", str(BOOTSTRAP_LEG_TIMES_VERSION)),
    ("EXPLICIT_LEG_STARTING_VERSION", str(EXPLICIT_LEG_STARTING_VERSION)),
    ("EXPLICIT_LEG_ENDING_VERSION", str(EXPLICIT_LEG_ENDING_VERSION)),
    ("EXPLICIT_LEG_TIMES_STARTING_VERSION", str(EXPLICIT_LEG_TIMES_STARTING_VERSION)),
    ("EXPLICIT_LEG_TIMES_ENDING_VERSION", str(EXPLICIT_LEG_TIMES_ENDING_VERSION)),
    ("MAX_CDF_VERSION_SPAN_PER_SOURCE", str(MAX_CDF_VERSION_SPAN_PER_SOURCE)),
    ("MAX_DIRTY_EVENTS", str(MAX_DIRTY_EVENTS)),
    ("MAX_AFFECTED_ENTITIES", str(MAX_AFFECTED_ENTITIES)),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
    ("TOLERANCE", str(TOLERANCE)),
    ("REQUIRE_FULL_AFFECTED_WINDOW", str(REQUIRE_FULL_AFFECTED_WINDOW)),
    ("SOURCE_LEG", SOURCE_LEG),
    ("SOURCE_LEG_TIMES", SOURCE_LEG_TIMES),
    ("CLEANED_FLIGHT_TABLE", CLEANED_FLIGHT_TABLE),
    ("CURRENT_TAXI_OUT_MV", CURRENT_TAXI_OUT_MV),
    ("SHADOW_TAXI_OUT_TABLE", SHADOW_TAXI_OUT_TABLE),
    ("WATERMARK_TABLE", WATERMARK_TABLE),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_WATERMARK_ADVANCE:
    print("RUN_WATERMARK_ADVANCE is False. Configuration displayed only; exiting before reads or writes.")
    dbutils.notebook.exit("RUN_WATERMARK_ADVANCE_FALSE")

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
        if (candidate / "src" / "ml_project" / "stage30c_taxi_out_watermark.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root containing src/ml_project/stage30c_taxi_out_watermark.py")


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
    TARGET_KEY_COLS,
    build_shadow_merge_sql,
    build_shadow_replace_source,
    validate_dev_shadow_table_name,
)
from ml_project.stage30c_taxi_out_watermark import (  # noqa: E402
    DEFAULT_STAGE_NAME,
    SOURCE_ALIASES,
    SourceWindow,
    build_next_source_windows_from_watermarks,
    build_watermark_schema_migration_sql,
    build_watermark_advance_merge_sql,
    build_watermark_advance_rows,
    classify_watermark_run_status,
    detect_missing_watermark_columns,
    summarize_watermark_run,
    validate_contiguous_source_window,
    validate_explicit_window_against_watermark,
    validate_watermark_advance_gates,
    validate_watermark_rows,
    validate_watermark_schema,
    validate_watermark_schema_migration_gates,
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded local helper modules from: {SRC_PATH}")

# COMMAND ----------

validate_dev_shadow_table_name(SHADOW_TAXI_OUT_TABLE, expected_suffix_or_token="shadow")
validate_dev_shadow_table_name(WATERMARK_TABLE, expected_suffix_or_token="watermark")

if SOURCE_VERSION_MODE not in {"watermark", "explicit"}:
    raise ValueError("SOURCE_VERSION_MODE must be 'watermark' or 'explicit'.")
if (ALLOW_SHADOW_MERGE or ALLOW_WATERMARK_ADVANCE) and DRY_RUN_ONLY:
    raise ValueError("Write flags require DRY_RUN_ONLY=False.")
if (
    ALLOW_SHADOW_MERGE
    or ALLOW_WATERMARK_ADVANCE
    or ALLOW_WATERMARK_BOOTSTRAP
    or ALLOW_WATERMARK_SCHEMA_MIGRATION
) and (
    WRITE_CONFIRMATION != REQUIRED_WRITE_CONFIRMATION
):
    raise ValueError("Write-capable Stage 30C-5 modes require the exact dev-shadow write confirmation string.")
if ALLOW_WATERMARK_ADVANCE and not ALLOW_SHADOW_MERGE and REQUIRE_SHADOW_MERGE_BEFORE_WATERMARK:
    raise ValueError("Watermark advancement requires shadow merge in this controlled runner.")

RUN_ID = str(uuid4())
STAGE_NAME = DEFAULT_STAGE_NAME
SHADOW_MERGE_SOURCE_VIEW = "stage30c5_taxi_out_shadow_merge_source"

# COMMAND ----------

watermark_table_exists = spark.catalog.tableExists(WATERMARK_TABLE)
if not watermark_table_exists:
    print("watermark_bootstrap_required: watermark table does not exist.")
    print("Identify baseline source versions for the existing shadow table before bootstrapping.")
    print("Do not infer baseline versions from Stage 30C-4 validation windows.")
    if not ALLOW_WATERMARK_BOOTSTRAP:
        dbutils.notebook.exit("watermark_bootstrap_required")

watermark_df = spark.table(WATERMARK_TABLE)
missing_watermark_columns = detect_missing_watermark_columns(watermark_df.columns)
if missing_watermark_columns:
    validate_watermark_schema_migration_gates(
        table_name=WATERMARK_TABLE,
        missing_columns=missing_watermark_columns,
        allow_schema_migration=ALLOW_WATERMARK_SCHEMA_MIGRATION,
        dry_run_only=DRY_RUN_ONLY,
        write_confirmation=WRITE_CONFIRMATION,
        required_write_confirmation=REQUIRED_WRITE_CONFIRMATION,
    )
    migration_sql = build_watermark_schema_migration_sql(
        table_name=WATERMARK_TABLE,
        missing_columns=missing_watermark_columns,
    )
    print("Executing additive dev watermark schema migration.")
    print(migration_sql)
    spark.sql(migration_sql)
    watermark_df = spark.table(WATERMARK_TABLE)

# Schema validation is intentionally re-run after any migration before source windows,
# shadow merge, or watermark advancement can continue.
validate_watermark_schema(watermark_df.columns)
watermark_rows = [row.asDict() for row in watermark_df.where(F.col("source_alias").isin(*SOURCE_ALIASES)).collect()]

try:
    watermark_rows_by_alias = validate_watermark_rows(
        watermark_rows,
        require_all_sources=REQUIRE_EXISTING_WATERMARK_ROWS,
    )
except ValueError as exc:
    print(f"watermark_bootstrap_required: {exc}")
    print("Next step: identify baseline source versions corresponding to shadow table initialization.")
    print("Insert initial source-specific watermark rows only after confirming shadow baseline equivalence.")
    print("Do not invent baseline versions in code and do not infer them from validation windows.")
    if not ALLOW_WATERMARK_BOOTSTRAP:
        dbutils.notebook.exit("watermark_bootstrap_required")
    raise

# COMMAND ----------


def _latest_delta_version(table_name: str) -> int:
    history_df = spark.sql(f"DESCRIBE HISTORY {table_name}")
    row = history_df.agg(F.max("version").alias("latest_version")).first()
    return int(row["latest_version"])


latest_available_versions = {
    "leg": _latest_delta_version(SOURCE_LEG),
    "leg_times": _latest_delta_version(SOURCE_LEG_TIMES),
}

if SOURCE_VERSION_MODE == "watermark":
    source_windows = build_next_source_windows_from_watermarks(
        watermark_rows_by_alias,
        latest_available_versions,
        max_cdf_version_span_per_source=MAX_CDF_VERSION_SPAN_PER_SOURCE,
    )
else:
    source_windows = {
        "leg": SourceWindow(
            source_alias="leg",
            starting_version=EXPLICIT_LEG_STARTING_VERSION,
            ending_version=EXPLICIT_LEG_ENDING_VERSION,
            previous_watermark_version=int(watermark_rows_by_alias["leg"]["last_processed_version"]),
        ),
        "leg_times": SourceWindow(
            source_alias="leg_times",
            starting_version=EXPLICIT_LEG_TIMES_STARTING_VERSION,
            ending_version=EXPLICIT_LEG_TIMES_ENDING_VERSION,
            previous_watermark_version=int(watermark_rows_by_alias["leg_times"]["last_processed_version"]),
        ),
    }
    for source_alias, window in source_windows.items():
        validate_explicit_window_against_watermark(
            source_alias=source_alias,
            previous_watermark_version=window.previous_watermark_version,
            starting_version=window.starting_version,
            ending_version=window.ending_version,
            allow_non_watermark_explicit_window=ALLOW_NON_WATERMARK_EXPLICIT_WINDOW,
        )

for source_alias, window in source_windows.items():
    validate_contiguous_source_window(
        source_alias=source_alias,
        previous_watermark_version=window.previous_watermark_version,
        starting_version=window.starting_version,
        ending_version=window.ending_version,
        require_contiguous=REQUIRE_CONTIGUOUS_SOURCE_WINDOWS,
    )

any_new_versions = any(window.has_new_versions for window in source_windows.values())
if not any_new_versions:
    status = classify_watermark_run_status(
        watermark_rows_present=True,
        any_new_versions=False,
        all_windows_contiguous=True,
    )
    print(status)
    display(spark.createDataFrame(_display_metric_rows(summarize_watermark_run(
        status=status,
        source_windows=source_windows,
        shadow_merge_executed=False,
        post_merge_validation_ok=False,
        watermarks_advanced=False,
    )), ["metric", "value"]))
    dbutils.notebook.exit(status)

print("Stage 30C-5 source windows")
display(
    spark.createDataFrame(
        [
            (
                source_alias,
                window.previous_watermark_version,
                window.starting_version,
                window.ending_version,
            )
            for source_alias, window in source_windows.items()
        ],
        ["source_alias", "previous_watermark_version", "starting_version", "ending_version"],
    )
)

# COMMAND ----------

CDF_COLUMNS = (
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
)


def _read_batch_cdf(table_name: str, starting_version, ending_version):
    reader = spark.read.option("readChangeFeed", "true").option("startingVersion", str(starting_version))
    if ending_version is not None:
        reader = reader.option("endingVersion", str(ending_version))
    return reader.table(source_table(table_name))


def _extract_dirty_leg_keys_from_cdf(cdf_df, source_alias: str):
    return (
        cdf_df.where(F.col("leg_no").isNotNull())
        .select(
            F.col("leg_no"),
            F.lit(source_alias).alias("dirty_source_alias"),
            F.col("update_key").alias("_stage30c5_update_key"),
            F.col("_change_type"),
            F.col("_commit_version"),
            F.col("_commit_timestamp"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(
            F.max("_stage30c5_update_key").alias("max_update_key"),
            F.min("_commit_version").alias("min_commit_version"),
            F.max("_commit_version").alias("max_commit_version"),
            F.max("_commit_timestamp").alias("latest_commit_timestamp"),
            F.collect_set("_change_type").alias("cdf_change_types"),
        )
    )


def _count_duplicate_keys(df, key_cols: tuple[str, ...]) -> int:
    return df.groupBy(*key_cols).count().where(F.col("count") > F.lit(1)).count()


def _count_null_keys(df, key_cols: tuple[str, ...]) -> int:
    null_expr = None
    for key_col in key_cols:
        expr = F.col(key_col).isNull()
        null_expr = expr if null_expr is None else null_expr | expr
    return df.where(null_expr).count()


def _status_counts_to_dict(status_counts_df) -> dict[str, int]:
    return {row["parity_status"]: row["rows"] for row in status_counts_df.collect()}


dirty_key_dfs = []
processed_timestamps_by_alias = {}
cdf_counts = {"leg": 0, "leg_times": 0}

for source_alias, window in source_windows.items():
    if not window.has_new_versions:
        print(f"{source_alias}: no new versions to process.")
        continue
    table_name = SOURCE_LEG_TABLE if source_alias == "leg" else SOURCE_LEG_TIMES_TABLE
    cdf_df = _read_batch_cdf(table_name, window.starting_version, window.ending_version)
    cdf_counts[source_alias] = cdf_df.count()
    cdf_summary = cdf_df.agg(
        F.max("_commit_version").alias("max_commit_version"),
        F.max("_commit_timestamp").alias("latest_commit_timestamp"),
    ).first()
    processed_timestamps_by_alias[source_alias] = (
        None if cdf_summary["latest_commit_timestamp"] is None else str(cdf_summary["latest_commit_timestamp"])
    )
    dirty_key_dfs.append(_extract_dirty_leg_keys_from_cdf(cdf_df, source_alias))

if dirty_key_dfs:
    dirty_legs = dirty_key_dfs[0]
    for next_dirty_keys in dirty_key_dfs[1:]:
        dirty_legs = dirty_legs.unionByName(next_dirty_keys)
    dirty_legs = dirty_legs.dropDuplicates(["leg_no", "dirty_source_alias"])
else:
    dirty_legs = spark.createDataFrame([], "leg_no LONG, dirty_source_alias STRING, max_update_key LONG")

dirty_leg_count = dirty_legs.count()
print(f"dirty leg/source candidates={dirty_leg_count}")

# COMMAND ----------

leg_src_current = spark.table(SOURCE_LEG)
cleaned = spark.table(CLEANED_FLIGHT_TABLE)
current_mv = spark.table(CURRENT_TAXI_OUT_MV)
current_leg = select_current_latest(leg_src_current, partition_cols=("leg_no",))

dirty_events = map_dirty_legs_to_taxi_out_events(
    dirty_legs,
    current_leg,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
).limit(MAX_DIRTY_EVENTS)

if REQUIRE_FULL_AFFECTED_WINDOW:
    max_current_event_date = current_mv.agg(F.max(DATE_COL).alias("max_event_date")).first()["max_event_date"]
    dirty_events = dirty_events.where(F.date_add(F.col("dirty_event_date"), 30) <= F.lit(max_current_event_date))

dirty_event_count = dirty_events.count()
affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
affected_entities = affected_outputs.select(ENTITY_COL).distinct().orderBy(ENTITY_COL).limit(MAX_AFFECTED_ENTITIES)
affected_outputs = affected_outputs.join(affected_entities, on=ENTITY_COL, how="inner")
affected_pairs = affected_outputs.select(
    ENTITY_COL,
    F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
).dropDuplicates()
affected_pair_count = affected_pairs.count()

candidate_scoped = build_taxi_out_candidate_for_affected_outputs(
    cleaned,
    affected_outputs,
    history_start=HISTORY_START,
    data_cutoff_date=DATA_CUTOFF_DATE,
)
candidate_count = candidate_scoped.count()
candidate_duplicate_key_count = _count_duplicate_keys(candidate_scoped, TARGET_KEY_COLS)
candidate_null_key_count = _count_null_keys(candidate_scoped, TARGET_KEY_COLS)

current_scoped = current_mv.select(*NON_EMA_PARITY_COLUMNS).join(affected_pairs, on=[ENTITY_COL, DATE_COL], how="inner")
current_parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped, tolerance=TOLERANCE)
current_status_counts_df = current_parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
current_status_counts = _status_counts_to_dict(current_status_counts_df)
candidate_current_mismatch_count = sum(
    rows for status_name, rows in current_status_counts.items() if status_name != "matched"
)

shadow_merge_source = build_shadow_replace_source(affected_pairs, candidate_scoped)
shadow_merge_executed = False
if ALLOW_SHADOW_MERGE:
    shadow_merge_source.createOrReplaceTempView(SHADOW_MERGE_SOURCE_VIEW)
    merge_sql = build_shadow_merge_sql(
        shadow_table=SHADOW_TAXI_OUT_TABLE,
        merge_source_view=SHADOW_MERGE_SOURCE_VIEW,
        candidate_columns=NON_EMA_PARITY_COLUMNS,
    )
    spark.sql(merge_sql)
    shadow_merge_executed = True

shadow_scoped = spark.table(SHADOW_TAXI_OUT_TABLE).select(*NON_EMA_PARITY_COLUMNS).join(
    affected_pairs,
    on=[ENTITY_COL, DATE_COL],
    how="inner",
)
shadow_duplicate_key_count = _count_duplicate_keys(shadow_scoped, TARGET_KEY_COLS)
shadow_null_key_count = _count_null_keys(shadow_scoped, TARGET_KEY_COLS)
shadow_parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, shadow_scoped, tolerance=TOLERANCE)
shadow_status_counts_df = shadow_parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
shadow_status_counts = _status_counts_to_dict(shadow_status_counts_df)
candidate_shadow_mismatch_count = sum(
    rows for status_name, rows in shadow_status_counts.items() if status_name != "matched"
)
post_merge_validation_ok = candidate_shadow_mismatch_count == 0 and shadow_duplicate_key_count == 0 and shadow_null_key_count == 0

# COMMAND ----------

watermarks_advanced = False
advance_rows = build_watermark_advance_rows(
    source_windows=source_windows,
    processed_timestamps_by_alias=processed_timestamps_by_alias,
    run_id=RUN_ID,
    updated_by_stage=STAGE_NAME,
)

if ALLOW_WATERMARK_ADVANCE:
    validate_watermark_advance_gates(
        shadow_merge_executed=shadow_merge_executed,
        post_merge_validation_ok=post_merge_validation_ok,
        candidate_duplicate_key_count=candidate_duplicate_key_count if REQUIRE_NO_DUPLICATE_OR_NULL_KEYS else 0,
        candidate_null_key_count=candidate_null_key_count if REQUIRE_NO_DUPLICATE_OR_NULL_KEYS else 0,
        shadow_duplicate_key_count=shadow_duplicate_key_count if REQUIRE_NO_DUPLICATE_OR_NULL_KEYS else 0,
        shadow_null_key_count=shadow_null_key_count if REQUIRE_NO_DUPLICATE_OR_NULL_KEYS else 0,
        candidate_current_mismatch_count=(
            candidate_current_mismatch_count if REQUIRE_NO_CANDIDATE_CURRENT_MISMATCHES else 0
        ),
        candidate_shadow_mismatch_count=(
            candidate_shadow_mismatch_count if REQUIRE_NO_CANDIDATE_SHADOW_MISMATCHES else 0
        ),
        write_confirmation=WRITE_CONFIRMATION,
        required_write_confirmation=REQUIRED_WRITE_CONFIRMATION,
        windows_contiguous=REQUIRE_CONTIGUOUS_SOURCE_WINDOWS,
        watermark_rows_valid=True,
        source_versions_present=bool(advance_rows),
    )
    watermark_sql = build_watermark_advance_merge_sql(
        watermark_table=WATERMARK_TABLE,
        advance_rows=advance_rows,
    )
    spark.sql(watermark_sql)
    watermarks_advanced = True

status = classify_watermark_run_status(
    watermark_rows_present=True,
    any_new_versions=any_new_versions,
    all_windows_contiguous=True,
    validation_ok=post_merge_validation_ok and candidate_current_mismatch_count == 0,
    advanced=watermarks_advanced,
)
summary = summarize_watermark_run(
    status=status,
    source_windows=source_windows,
    shadow_merge_executed=shadow_merge_executed,
    post_merge_validation_ok=post_merge_validation_ok,
    watermarks_advanced=watermarks_advanced,
    duplicate_or_null_key_count=(
        candidate_duplicate_key_count
        + candidate_null_key_count
        + shadow_duplicate_key_count
        + shadow_null_key_count
    ),
    mismatch_count=candidate_current_mismatch_count + candidate_shadow_mismatch_count,
)

print("Stage 30C-5 watermark run summary")
display(spark.createDataFrame(_display_metric_rows(summary), ["metric", "value"]))
print("candidate/current status counts")
display(current_status_counts_df)
print("candidate/shadow status counts")
display(shadow_status_counts_df)

if ALLOW_WATERMARK_ADVANCE and not watermarks_advanced:
    raise ValueError("Watermark advancement was requested but did not complete.")

print("Stage 30C-5 taxi-out watermark advance preflight completed.")
