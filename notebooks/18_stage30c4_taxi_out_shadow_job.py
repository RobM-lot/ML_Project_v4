# Databricks notebook source
# ruff: noqa: F821
# Stage 30C-4 job-style taxi-out shadow validation and overlap-aware execution.

import json
from pathlib import Path
import sys
from uuid import uuid4

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30C-4 taxi-out shadow job runner. "
    "Defaults are safe: config-only, dry-run, no watermark advancement."
)
print("=" * 100)

# COMMAND ----------

RUN_JOB = False

JOB_MODE = "validation"  # allowed: validation, shadow_merge, watermark_advance
SOURCE_VERSION_MODE = "explicit"  # allowed: explicit, watermark

DRY_RUN_ONLY = True

ALLOW_SHADOW_MERGE = False
ALLOW_WATERMARK_ADVANCE = False

WRITE_CONFIRMATION = ""
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"

REQUIRE_VALIDATION_OVERLAP = True
REQUIRE_AT_LEAST_ONE_PARITY_ROW_PER_WINDOW = True
REQUIRE_AT_LEAST_TWO_PARITY_WINDOWS = True
REQUIRE_AT_LEAST_TWO_ENTITIES_OVERALL = True

MAX_DIRTY_EVENTS_PER_WINDOW = 1000
MAX_AFFECTED_ENTITIES_PER_WINDOW = 3
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

VALIDATION_WINDOWS = [
    {
        "window_id": "A_KRK",
        "leg_start": 34600,
        "leg_end": 34620,
        "leg_times_start": 34519,
        "leg_times_end": 34538,
        "entity_filter": "KRK",
        "max_affected_entities": 1,
    },
    {
        "window_id": "B_WAW",
        "leg_start": 34680,
        "leg_end": 34700,
        "leg_times_start": 34598,
        "leg_times_end": 34618,
        "entity_filter": "WAW",
        "max_affected_entities": 1,
    },
    {
        "window_id": "C_MULTI_OVERLAP_AWARE",
        "leg_start": 34680,
        "leg_end": 34700,
        "leg_times_start": 34598,
        "leg_times_end": 34618,
        "entity_filter": "",
        "max_affected_entities": 3,
        "overlap_aware": True,
    },
]
VALIDATION_WINDOWS_JSON = ""


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
    ("RUN_JOB", str(RUN_JOB)),
    ("JOB_MODE", JOB_MODE),
    ("SOURCE_VERSION_MODE", SOURCE_VERSION_MODE),
    ("DRY_RUN_ONLY", str(DRY_RUN_ONLY)),
    ("ALLOW_SHADOW_MERGE", str(ALLOW_SHADOW_MERGE)),
    ("ALLOW_WATERMARK_ADVANCE", str(ALLOW_WATERMARK_ADVANCE)),
    ("WRITE_CONFIRMATION_PRESENT", str(bool(WRITE_CONFIRMATION))),
    ("REQUIRE_VALIDATION_OVERLAP", str(REQUIRE_VALIDATION_OVERLAP)),
    ("REQUIRE_AT_LEAST_ONE_PARITY_ROW_PER_WINDOW", str(REQUIRE_AT_LEAST_ONE_PARITY_ROW_PER_WINDOW)),
    ("REQUIRE_AT_LEAST_TWO_PARITY_WINDOWS", str(REQUIRE_AT_LEAST_TWO_PARITY_WINDOWS)),
    ("REQUIRE_AT_LEAST_TWO_ENTITIES_OVERALL", str(REQUIRE_AT_LEAST_TWO_ENTITIES_OVERALL)),
    ("MAX_DIRTY_EVENTS_PER_WINDOW", str(MAX_DIRTY_EVENTS_PER_WINDOW)),
    ("MAX_AFFECTED_ENTITIES_PER_WINDOW", str(MAX_AFFECTED_ENTITIES_PER_WINDOW)),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
    ("TOLERANCE", str(TOLERANCE)),
    ("REQUIRE_FULL_AFFECTED_WINDOW", str(REQUIRE_FULL_AFFECTED_WINDOW)),
    ("SOURCE_LEG", SOURCE_LEG),
    ("SOURCE_LEG_TIMES", SOURCE_LEG_TIMES),
    ("CLEANED_FLIGHT_TABLE", CLEANED_FLIGHT_TABLE),
    ("CURRENT_TAXI_OUT_MV", CURRENT_TAXI_OUT_MV),
    ("SHADOW_TAXI_OUT_TABLE", SHADOW_TAXI_OUT_TABLE),
    ("WATERMARK_TABLE", WATERMARK_TABLE),
    ("VALIDATION_WINDOWS_JSON_PRESENT", str(bool(VALIDATION_WINDOWS_JSON))),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_JOB:
    print("RUN_JOB is False. Configuration displayed only; exiting before reads or writes.")
    dbutils.notebook.exit("RUN_JOB_FALSE")

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
        if (candidate / "src" / "ml_project" / "stage30c_taxi_out_job.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root containing src/ml_project/stage30c_taxi_out_job.py")


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
from ml_project.stage30c_taxi_out_job import (  # noqa: E402
    parse_validation_windows,
    summarize_job_result,
    summarize_window_result,
    validate_job_success_preconditions,
    validate_job_write_gates,
)
from ml_project.stage30c_taxi_out_shadow import (  # noqa: E402
    TARGET_KEY_COLS,
    build_advance_watermark_sqls,
    build_shadow_merge_sql,
    build_shadow_replace_source,
    validate_dev_shadow_table_name,
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded local helper modules from: {SRC_PATH}")

# COMMAND ----------

validate_dev_shadow_table_name(SHADOW_TAXI_OUT_TABLE, expected_suffix_or_token="shadow")
validate_dev_shadow_table_name(WATERMARK_TABLE, expected_suffix_or_token="watermark")
validate_job_write_gates(
    job_mode=JOB_MODE,
    dry_run_only=DRY_RUN_ONLY,
    allow_shadow_merge=ALLOW_SHADOW_MERGE,
    allow_watermark_advance=ALLOW_WATERMARK_ADVANCE,
    write_confirmation=WRITE_CONFIRMATION,
    required_write_confirmation=REQUIRED_WRITE_CONFIRMATION,
)

validation_windows = parse_validation_windows(VALIDATION_WINDOWS, VALIDATION_WINDOWS_JSON)
display(spark.createDataFrame([window.__dict__ for window in validation_windows]))

if SOURCE_VERSION_MODE == "watermark":
    print(
        "SOURCE_VERSION_MODE='watermark' scaffolding: read source-specific versions from the watermark table, "
        "derive independent next windows for leg and leg_times, and do not advance watermarks unless explicitly gated."
    )
    watermark_state = spark.table(WATERMARK_TABLE)
    display(watermark_state.orderBy("source_alias").limit(MAX_SAMPLE_ROWS))
elif SOURCE_VERSION_MODE != "explicit":
    raise ValueError("SOURCE_VERSION_MODE must be 'explicit' or 'watermark'.")

# COMMAND ----------

STAGE_NAME = "stage30c4_taxi_out_shadow_job"
RUN_ID = str(uuid4())
SHADOW_MERGE_SOURCE_VIEW_PREFIX = "stage30c4_taxi_out_shadow_merge_source"

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
            F.col("update_key").alias("_stage30c4_update_key"),
            F.col("_change_type"),
            F.col("_commit_version"),
            F.col("_commit_timestamp"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(
            F.max("_stage30c4_update_key").alias("max_update_key"),
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


leg_src_current = spark.table(SOURCE_LEG)
cleaned = spark.table(CLEANED_FLIGHT_TABLE)
current_mv = spark.table(CURRENT_TAXI_OUT_MV)
shadow_table_available = spark.catalog.tableExists(SHADOW_TAXI_OUT_TABLE)

current_mv_horizon = current_mv.agg(
    F.min(DATE_COL).alias("min_current_mv_event_date"),
    F.max(DATE_COL).alias("max_current_mv_event_date"),
).first()
MIN_CURRENT_MV_EVENT_DATE = current_mv_horizon["min_current_mv_event_date"]
MAX_CURRENT_MV_EVENT_DATE = current_mv_horizon["max_current_mv_event_date"]
print(f"MIN_CURRENT_MV_EVENT_DATE={MIN_CURRENT_MV_EVENT_DATE}")
print(f"MAX_CURRENT_MV_EVENT_DATE={MAX_CURRENT_MV_EVENT_DATE}")

# COMMAND ----------

window_results = []
processed_versions_by_alias = {}
processed_timestamps_by_alias = {}

for window in validation_windows:
    print("=" * 100)
    print(f"Processing validation window: {window.window_id}")
    cdf_dfs = []
    cdf_counts = {"leg": 0, "leg_times": 0}

    if window.leg_start is not None:
        leg_cdf = _read_batch_cdf(SOURCE_LEG_TABLE, window.leg_start, window.leg_end)
        cdf_counts["leg"] = leg_cdf.count()
        cdf_dfs.append(_extract_dirty_leg_keys_from_cdf(leg_cdf, "leg"))
        leg_summary = leg_cdf.agg(
            F.max("_commit_version").alias("max_commit_version"),
            F.max("_commit_timestamp").alias("latest_commit_timestamp"),
        ).first()
        processed_versions_by_alias["leg"] = leg_summary["max_commit_version"]
        processed_timestamps_by_alias["leg"] = str(leg_summary["latest_commit_timestamp"])

    if window.leg_times_start is not None:
        leg_times_cdf = _read_batch_cdf(SOURCE_LEG_TIMES_TABLE, window.leg_times_start, window.leg_times_end)
        cdf_counts["leg_times"] = leg_times_cdf.count()
        cdf_dfs.append(_extract_dirty_leg_keys_from_cdf(leg_times_cdf, "leg_times"))
        leg_times_summary = leg_times_cdf.agg(
            F.max("_commit_version").alias("max_commit_version"),
            F.max("_commit_timestamp").alias("latest_commit_timestamp"),
        ).first()
        processed_versions_by_alias["leg_times"] = leg_times_summary["max_commit_version"]
        processed_timestamps_by_alias["leg_times"] = str(leg_times_summary["latest_commit_timestamp"])

    if not cdf_dfs:
        raise ValueError(f"Window {window.window_id} has no configured CDF sources.")

    dirty_legs = cdf_dfs[0]
    for next_dirty_keys in cdf_dfs[1:]:
        dirty_legs = dirty_legs.unionByName(next_dirty_keys)
    dirty_legs = dirty_legs.dropDuplicates(["leg_no", "dirty_source_alias"])
    dirty_leg_count = dirty_legs.count()
    print(f"{window.window_id}: dirty leg/source candidates={dirty_leg_count}")

    current_leg = select_current_latest(leg_src_current, partition_cols=("leg_no",))
    dirty_events = map_dirty_legs_to_taxi_out_events(
        dirty_legs,
        current_leg,
        history_start=HISTORY_START,
        data_cutoff_date=DATA_CUTOFF_DATE,
    )
    if window.entity_filter:
        dirty_events = dirty_events.where(F.col(ENTITY_COL) == F.lit(window.entity_filter))
    if REQUIRE_FULL_AFFECTED_WINDOW:
        dirty_events = dirty_events.where(F.date_add(F.col("dirty_event_date"), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE))

    dirty_events = dirty_events.orderBy(ENTITY_COL, "dirty_event_date", "leg_no").limit(MAX_DIRTY_EVENTS_PER_WINDOW)
    dirty_event_count = dirty_events.count()
    print(f"{window.window_id}: dirty taxi-out events={dirty_event_count}")

    affected_outputs_all = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
    affected_pairs_all = affected_outputs_all.select(
        ENTITY_COL,
        F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
    ).dropDuplicates()

    candidate_all = build_taxi_out_candidate_for_affected_outputs(
        cleaned,
        affected_outputs_all,
        history_start=HISTORY_START,
        data_cutoff_date=DATA_CUTOFF_DATE,
    )
    current_all = current_mv.select(*NON_EMA_PARITY_COLUMNS).join(
        affected_pairs_all,
        on=[ENTITY_COL, DATE_COL],
        how="inner",
    )

    candidate_keys = candidate_all.select(ENTITY_COL, DATE_COL).dropDuplicates().withColumn("has_candidate", F.lit(True))
    current_keys = current_all.select(ENTITY_COL, DATE_COL).dropDuplicates().withColumn("has_current_mv_key", F.lit(True))
    tagged_pairs_df = (
        affected_pairs_all.join(candidate_keys, on=[ENTITY_COL, DATE_COL], how="left")
        .join(current_keys, on=[ENTITY_COL, DATE_COL], how="left")
        .withColumn("has_candidate", F.coalesce(F.col("has_candidate"), F.lit(False)))
        .withColumn("has_current_mv_key", F.coalesce(F.col("has_current_mv_key"), F.lit(False)))
        .withColumn("has_validation_overlap", F.col("has_candidate") & F.col("has_current_mv_key"))
    )
    print(f"{window.window_id}: validation overlap tags")
    display(tagged_pairs_df.groupBy("has_candidate", "has_current_mv_key", "has_validation_overlap").count())

    if JOB_MODE == "validation" and (REQUIRE_VALIDATION_OVERLAP or window.overlap_aware):
        selected_entities_rows = (
            tagged_pairs_df.where(F.col("has_validation_overlap"))
            .groupBy(ENTITY_COL)
            .agg(F.count("*").alias("overlap_pair_count"))
            .orderBy(F.desc("overlap_pair_count"), ENTITY_COL)
            .limit(window.max_affected_entities)
            .collect()
        )
        selected_entities = tuple(row[ENTITY_COL] for row in selected_entities_rows)
    else:
        selected_entities_rows = (
            affected_pairs_all.select(ENTITY_COL)
            .distinct()
            .orderBy(ENTITY_COL)
            .limit(window.max_affected_entities or MAX_AFFECTED_ENTITIES_PER_WINDOW)
            .collect()
        )
        selected_entities = tuple(row[ENTITY_COL] for row in selected_entities_rows)

    if not selected_entities:
        selected_entities = ()

    if selected_entities:
        selected_entities_df = spark.createDataFrame([(entity,) for entity in selected_entities], [ENTITY_COL])
    else:
        selected_entities_df = current_mv.select(ENTITY_COL).where(F.lit(False))
    affected_outputs_selected = affected_outputs_all.join(selected_entities_df, on=ENTITY_COL, how="inner")
    affected_pairs_selected = affected_outputs_selected.select(
        ENTITY_COL,
        F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
    ).dropDuplicates()
    candidate_scoped = candidate_all.join(affected_pairs_selected, on=[ENTITY_COL, DATE_COL], how="inner")
    current_scoped = current_all.join(affected_pairs_selected, on=[ENTITY_COL, DATE_COL], how="inner")

    affected_pairs_count = affected_pairs_selected.count()
    candidate_count = candidate_scoped.count()
    current_count = current_scoped.count()
    candidate_duplicate_key_count = _count_duplicate_keys(candidate_scoped, TARGET_KEY_COLS)
    candidate_null_key_count = _count_null_keys(candidate_scoped, TARGET_KEY_COLS)
    print(
        f"{window.window_id}: affected_pairs={affected_pairs_count}, candidate_rows={candidate_count}, "
        f"current_rows={current_count}, selected_entities={selected_entities}"
    )

    parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped, tolerance=TOLERANCE)
    current_status_counts_df = parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
    current_status_counts = _status_counts_to_dict(current_status_counts_df)
    print(f"{window.window_id}: candidate/current status counts")
    display(current_status_counts_df)

    shadow_merge_executed = False
    shadow_status_counts = {}
    shadow_duplicate_key_count = 0
    shadow_null_key_count = 0

    shadow_merge_source = build_shadow_replace_source(affected_pairs_selected, candidate_scoped)
    print(f"{window.window_id}: shadow merge source sample")
    display(shadow_merge_source.orderBy(ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

    if ALLOW_SHADOW_MERGE and JOB_MODE in {"shadow_merge", "watermark_advance"}:
        if DRY_RUN_ONLY:
            raise ValueError("ALLOW_SHADOW_MERGE requires DRY_RUN_ONLY=False.")
        temp_view = f"{SHADOW_MERGE_SOURCE_VIEW_PREFIX}_{window.window_id.lower()}"
        shadow_merge_source.createOrReplaceTempView(temp_view)
        merge_sql = build_shadow_merge_sql(
            shadow_table=SHADOW_TAXI_OUT_TABLE,
            merge_source_view=temp_view,
            candidate_columns=NON_EMA_PARITY_COLUMNS,
        )
        spark.sql(merge_sql)
        shadow_merge_executed = True

    if shadow_table_available or shadow_merge_executed:
        shadow_scoped = spark.table(SHADOW_TAXI_OUT_TABLE).select(*NON_EMA_PARITY_COLUMNS).join(
            affected_pairs_selected,
            on=[ENTITY_COL, DATE_COL],
            how="inner",
        )
        shadow_duplicate_key_count = _count_duplicate_keys(shadow_scoped, TARGET_KEY_COLS)
        shadow_null_key_count = _count_null_keys(shadow_scoped, TARGET_KEY_COLS)
        shadow_parity = compare_taxi_out_candidate_to_current_mv(candidate_scoped, shadow_scoped, tolerance=TOLERANCE)
        shadow_status_counts_df = (
            shadow_parity.groupBy("parity_status").agg(F.count("*").alias("rows")).orderBy(F.desc("rows"))
        )
        shadow_status_counts = _status_counts_to_dict(shadow_status_counts_df)
        print(f"{window.window_id}: candidate/shadow status counts")
        display(shadow_status_counts_df)

    window_result = summarize_window_result(
        window_id=window.window_id,
        entity_filter=window.entity_filter,
        selected_entities=selected_entities,
        cdf_leg_count=cdf_counts["leg"],
        cdf_leg_times_count=cdf_counts["leg_times"],
        dirty_events_count=dirty_event_count,
        affected_pairs_count=affected_pairs_count,
        candidate_rows_count=candidate_count,
        current_scoped_rows_count=current_count,
        candidate_current_status_counts=current_status_counts,
        shadow_merge_executed=shadow_merge_executed,
        candidate_shadow_status_counts=shadow_status_counts,
        candidate_duplicate_key_count=candidate_duplicate_key_count,
        candidate_null_key_count=candidate_null_key_count,
        shadow_duplicate_key_count=shadow_duplicate_key_count,
        shadow_null_key_count=shadow_null_key_count,
        require_validation_overlap=REQUIRE_VALIDATION_OVERLAP,
    )
    window_results.append(window_result)
    display(
        spark.createDataFrame(
            _display_metric_rows(window_result),
            ["metric", "value"],
        )
    )

# COMMAND ----------

job_summary = summarize_job_result(
    window_results,
    require_at_least_two_parity_windows=REQUIRE_AT_LEAST_TWO_PARITY_WINDOWS,
    require_at_least_two_entities_overall=REQUIRE_AT_LEAST_TWO_ENTITIES_OVERALL,
    shadow_table_name_safe=True,
    watermark_table_name_safe=True,
    watermarks_advanced=False,
)
print("Stage 30C-4 job window results")
display(
    spark.createDataFrame(
        [
            (result["window_id"], key, _display_value(value))
            for result in window_results
            for key, value in result.items()
        ],
        ["window_id", "metric", "value"],
    )
)

print("Stage 30C-4 overall output")
display(spark.createDataFrame(_display_metric_rows(job_summary), ["metric", "value"]))

watermarks_advanced = False
if JOB_MODE == "watermark_advance" and ALLOW_WATERMARK_ADVANCE:
    validate_job_success_preconditions(
        job_summary=job_summary,
        job_mode=JOB_MODE,
        allow_watermark_advance=ALLOW_WATERMARK_ADVANCE,
        processed_versions_by_alias=processed_versions_by_alias,
        write_confirmation=WRITE_CONFIRMATION,
        required_write_confirmation=REQUIRED_WRITE_CONFIRMATION,
        configured_source_aliases=("leg", "leg_times"),
    )
    watermark_sqls = build_advance_watermark_sqls(
        watermark_table=WATERMARK_TABLE,
        stage_name=STAGE_NAME,
        source_tables_by_alias={
            "leg": SOURCE_LEG,
            "leg_times": SOURCE_LEG_TIMES,
        },
        processed_versions_by_alias=processed_versions_by_alias,
        processed_timestamps_by_alias=processed_timestamps_by_alias,
        last_successful_run_id=RUN_ID,
    )
    if DRY_RUN_ONLY:
        raise ValueError("Watermark advancement requires DRY_RUN_ONLY=False.")
    for sql_text in watermark_sqls:
        spark.sql(sql_text)
    watermarks_advanced = bool(watermark_sqls)

final_boolean_summary = {
    "run_job": RUN_JOB,
    "job_mode": JOB_MODE,
    "source_version_mode_explicit": SOURCE_VERSION_MODE == "explicit",
    "read_cdf_all_windows_ok": job_summary["read_cdf_all_windows_ok"],
    "dirty_events_any_window": job_summary["dirty_events_any_window"],
    "affected_pairs_any_window": job_summary["affected_pairs_any_window"],
    "overlap_aware_selection_enabled": REQUIRE_VALIDATION_OVERLAP,
    "parity_windows_required_met": job_summary["parity_windows_required_met"],
    "entities_required_met": job_summary["entities_required_met"],
    "candidate_keys_unique_all": job_summary["candidate_keys_unique_all"],
    "candidate_keys_non_null_all": job_summary["candidate_keys_non_null_all"],
    "current_overlap_compare_ok_all": job_summary["current_overlap_compare_ok_all"],
    "shadow_table_name_safe": job_summary["shadow_table_name_safe"],
    "watermark_table_name_safe": job_summary["watermark_table_name_safe"],
    "shadow_merge_executed_any": job_summary["shadow_merge_executed_any"],
    "shadow_post_merge_validation_ok_all": job_summary["shadow_post_merge_validation_ok_all"],
    "watermarks_advanced": watermarks_advanced,
    "read_only_or_dev_shadow_only": job_summary["read_only_or_dev_shadow_only"],
    "overall_pass": job_summary["overall_pass"] and (not REQUIRE_AT_LEAST_ONE_PARITY_ROW_PER_WINDOW or job_summary["total_current_matched_rows"] > 0),
}

print("Stage 30C-4 final boolean summary")
display(spark.createDataFrame(_display_metric_rows(final_boolean_summary), ["check_name", "passed"]))

if not final_boolean_summary["overall_pass"]:
    raise ValueError("Stage 30C-4 job validation did not meet the configured pass criteria.")

print("Stage 30C-4 taxi-out shadow job completed.")
