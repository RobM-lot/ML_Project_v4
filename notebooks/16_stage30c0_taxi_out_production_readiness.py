# Databricks notebook source
# ruff: noqa: F821
# Stage 30C-0 taxi-out partial recompute production readiness diagnostics.

from pathlib import Path
import sys

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30C-0 taxi-out production readiness diagnostics. "
    "This notebook is read-only and must not mutate production tables or workspace resources."
)
print("=" * 100)

# COMMAND ----------

RUN_READINESS = False

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
MAX_SAMPLE_ROWS = 100


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


CONFIG_ROWS = [
    ("RUN_READINESS", str(RUN_READINESS)),
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
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_READINESS:
    print("RUN_READINESS is False. Review the configuration above, then set RUN_READINESS = True for diagnostics.")
    dbutils.notebook.exit("RUN_READINESS_FALSE")

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

LEG_REQUIRED_COLUMNS = (
    "leg_no",
    "update_key",
    "__START_AT",
    "__END_AT",
    "dep_ap_sched",
    "dep_sched_dt",
    "leg_state",
    "leg_type",
    "counter",
)

LEG_TIMES_REQUIRED_COLUMNS = (
    "leg_no",
    "update_key",
    "__START_AT",
    "__END_AT",
    "offblock_dt",
    "airborne_dt",
)

CLEANED_REQUIRED_COLUMNS = (
    DATE_COL,
    ENTITY_COL,
    "taxi_out_sec",
    "scheduled_block_time_sec",
    "actual_block_time_sec",
)

SOURCE_SPECS = [
    {
        "source_alias": "leg",
        "table_name": SOURCE_LEG_TABLE,
        "starting_version": LEG_CDF_STARTING_VERSION,
        "ending_version": LEG_CDF_ENDING_VERSION,
        "required_columns": LEG_REQUIRED_COLUMNS,
    },
    {
        "source_alias": "leg_times",
        "table_name": SOURCE_LEG_TIMES_TABLE,
        "starting_version": LEG_TIMES_CDF_STARTING_VERSION,
        "ending_version": LEG_TIMES_CDF_ENDING_VERSION,
        "required_columns": LEG_TIMES_REQUIRED_COLUMNS,
    },
]

readiness_rows = []
readiness_flags = {
    "source_leg_schema_ok": False,
    "source_leg_times_schema_ok": False,
    "cleaned_flight_schema_ok": False,
    "current_mv_schema_ok": False,
    "current_mv_key_unique": False,
    "current_mv_key_non_null": False,
    "cdf_versions_configured": False,
    "cdf_leg_read_ok": False,
    "cdf_leg_times_read_ok": False,
    "dirty_preview_available": False,
    "read_only_no_writes": True,
}


def _record(area: str, check_name: str, passed: bool, detail: str) -> None:
    readiness_rows.append(
        {
            "area": area,
            "check_name": check_name,
            "passed": bool(passed),
            "detail": detail,
        }
    )
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {area}: {check_name} - {detail}")


def _missing_columns(df, required_columns: tuple[str, ...]) -> list[str]:
    available = set(df.columns)
    return [column for column in required_columns if column not in available]


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
            F.col("update_key").alias("_stage30c0_update_key"),
            F.col("_change_type"),
            F.col("_commit_version"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(
            F.max("_stage30c0_update_key").alias("max_update_key"),
            F.min("_commit_version").alias("min_commit_version"),
            F.max("_commit_version").alias("max_commit_version"),
            F.collect_set("_change_type").alias("cdf_change_types"),
        )
    )

# COMMAND ----------

leg_src = spark.table(source_table(SOURCE_LEG_TABLE))
leg_times_src = spark.table(source_table(SOURCE_LEG_TIMES_TABLE))
cleaned = spark.table(CLEANED_FLIGHT_TABLE)
current_mv = spark.table(CURRENT_TAXI_OUT_MV)

for label, df, required in (
    ("source_leg", leg_src, LEG_REQUIRED_COLUMNS),
    ("source_leg_times", leg_times_src, LEG_TIMES_REQUIRED_COLUMNS),
    ("cleaned_flight", cleaned, CLEANED_REQUIRED_COLUMNS),
    ("current_taxi_out_mv", current_mv, NON_EMA_PARITY_COLUMNS),
):
    print(f"{label} schema")
    df.printSchema()
    missing = _missing_columns(df, required)
    _record(label, "required columns", not missing, f"missing={missing}")
    flag_name = f"{label}_schema_ok" if label != "current_taxi_out_mv" else "current_mv_schema_ok"
    readiness_flags[flag_name] = not missing

print("Expected non-EMA parity columns from helper")
display(spark.createDataFrame([(column_name,) for column_name in NON_EMA_PARITY_COLUMNS], ["column_name"]))

# COMMAND ----------

current_mv_stats = current_mv.agg(
    F.count("*").alias("row_count"),
    F.min(DATE_COL).alias("min_event_date"),
    F.max(DATE_COL).alias("max_event_date"),
    F.countDistinct(ENTITY_COL).alias("distinct_dep_ap_sched"),
    F.sum(F.when(F.col(ENTITY_COL).isNull() | F.col(DATE_COL).isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
        "key_null_count"
    ),
).first()
_record("current_taxi_out_mv", "row count", current_mv_stats["row_count"] >= 0, f"rows={current_mv_stats['row_count']}")
_record(
    "current_taxi_out_mv",
    "event_date min/max",
    current_mv_stats["min_event_date"] is not None and current_mv_stats["max_event_date"] is not None,
    f"min={current_mv_stats['min_event_date']}, max={current_mv_stats['max_event_date']}",
)
_record(
    "current_taxi_out_mv",
    "distinct dep_ap_sched",
    current_mv_stats["distinct_dep_ap_sched"] >= 0,
    f"distinct_dep_ap_sched={current_mv_stats['distinct_dep_ap_sched']}",
)
_record(
    "current_taxi_out_mv",
    "key columns non-null",
    current_mv_stats["key_null_count"] == 0,
    f"null dep_ap_sched/event_date keys={current_mv_stats['key_null_count']}",
)
readiness_flags["current_mv_key_non_null"] = current_mv_stats["key_null_count"] == 0

print("current taxi_out MV sample rows")
display(current_mv.limit(MAX_SAMPLE_ROWS))

current_mv_key_duplicates = (
    current_mv.groupBy(ENTITY_COL, DATE_COL)
    .count()
    .where(F.col("count") > F.lit(1))
)
current_mv_duplicate_count = current_mv_key_duplicates.count()
_record(
    "current_taxi_out_mv",
    "unique output key",
    current_mv_duplicate_count == 0,
    f"duplicate dep_ap_sched/event_date keys={current_mv_duplicate_count}",
)
readiness_flags["current_mv_key_unique"] = current_mv_duplicate_count == 0
if current_mv_duplicate_count:
    display(current_mv_key_duplicates.orderBy(F.desc("count"), ENTITY_COL, DATE_COL).limit(MAX_SAMPLE_ROWS))

max_current_mv_event_date_row = current_mv.agg(F.max(DATE_COL).alias("max_current_mv_event_date")).first()
MAX_CURRENT_MV_EVENT_DATE = max_current_mv_event_date_row["max_current_mv_event_date"]
_record(
    "current_taxi_out_mv",
    "event_date horizon",
    MAX_CURRENT_MV_EVENT_DATE is not None,
    f"MAX_CURRENT_MV_EVENT_DATE={MAX_CURRENT_MV_EVENT_DATE}",
)

print("Recommended future target key columns: dep_ap_sched, event_date")
target_schema_compatible = (
    readiness_flags["current_mv_schema_ok"]
    and readiness_flags["current_mv_key_unique"]
    and readiness_flags["current_mv_key_non_null"]
)
_record(
    "future_target",
    "deterministic keyed replacement compatibility",
    target_schema_compatible,
    "current MV schema/key shape is compatible with future deterministic keyed replacement",
)

cleaned_horizon = cleaned.agg(
    F.count("*").alias("row_count"),
    F.min(DATE_COL).alias("min_cleaned_event_date"),
    F.max(DATE_COL).alias("max_cleaned_event_date"),
    F.sum(F.when(F.col(ENTITY_COL).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_dep_ap_sched"),
    F.sum(F.when(F.col(DATE_COL).isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_event_date"),
    F.sum(F.when(F.col("taxi_out_sec").isNull(), F.lit(1)).otherwise(F.lit(0))).alias("null_taxi_out_sec"),
    F.sum(F.when(F.col("actual_block_time_sec").isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
        "null_actual_block_time_sec"
    ),
    F.sum(F.when(F.col("scheduled_block_time_sec").isNull(), F.lit(1)).otherwise(F.lit(0))).alias(
        "null_scheduled_block_time_sec"
    ),
).first()
_record("cleaned_flight", "row count", cleaned_horizon["row_count"] >= 0, f"rows={cleaned_horizon['row_count']}")
_record(
    "cleaned_flight",
    "history availability",
    cleaned_horizon["min_cleaned_event_date"] is not None and cleaned_horizon["max_cleaned_event_date"] is not None,
    f"min={cleaned_horizon['min_cleaned_event_date']}, max={cleaned_horizon['max_cleaned_event_date']}",
)
_record(
    "cleaned_flight",
    "taxi_out recompute null counts",
    True,
    (
        f"dep_ap_sched={cleaned_horizon['null_dep_ap_sched']}, event_date={cleaned_horizon['null_event_date']}, "
        f"taxi_out_sec={cleaned_horizon['null_taxi_out_sec']}, "
        f"actual_block_time_sec={cleaned_horizon['null_actual_block_time_sec']}, "
        f"scheduled_block_time_sec={cleaned_horizon['null_scheduled_block_time_sec']}"
    ),
)

if ENTITY_FILTER:
    print(f"cleaned flight sample rows for ENTITY_FILTER={ENTITY_FILTER}")
    display(cleaned.where(F.col(ENTITY_COL) == F.lit(ENTITY_FILTER)).limit(MAX_SAMPLE_ROWS))

# COMMAND ----------

dirty_key_dfs = []
cdf_versions_configured = any(spec["starting_version"] is not None for spec in SOURCE_SPECS)
readiness_flags["cdf_versions_configured"] = cdf_versions_configured

if cdf_versions_configured:
    for spec in SOURCE_SPECS:
        source_alias = spec["source_alias"]
        starting_version = spec["starting_version"]
        ending_version = spec["ending_version"]

        if starting_version is None:
            _record("cdf_probe", f"{source_alias} skipped", True, "starting version is not set")
            continue
        if ending_version is not None and int(ending_version) < int(starting_version):
            _record("cdf_probe", f"{source_alias} version order", False, "ending version is before starting version")
            continue

        cdf_df = _read_batch_cdf(spec["table_name"], starting_version, ending_version)
        missing = _missing_columns(cdf_df, (*spec["required_columns"], *CDF_COLUMNS))
        _record("cdf_probe", f"{source_alias} CDF columns", not missing, f"missing={missing}")
        if missing:
            continue

        print(f"{source_alias} CDF sample rows")
        display(cdf_df.limit(MAX_SAMPLE_ROWS))

        cdf_summary = cdf_df.agg(
            F.count("*").alias("row_count"),
            F.countDistinct("leg_no").alias("unique_leg_no_count"),
            F.min("update_key").alias("min_update_key"),
            F.max("update_key").alias("max_update_key"),
            F.min("_commit_version").alias("min_commit_version"),
            F.max("_commit_version").alias("max_commit_version"),
        ).first()
        _record(
            "cdf_probe",
            f"{source_alias} CDF rows",
            cdf_summary["row_count"] >= 0,
            (
                f"rows={cdf_summary['row_count']}, unique_leg_no={cdf_summary['unique_leg_no_count']}, "
                f"commit_range={cdf_summary['min_commit_version']}..{cdf_summary['max_commit_version']}, "
                f"update_key_range={cdf_summary['min_update_key']}..{cdf_summary['max_update_key']}"
            ),
        )
        readiness_flags[f"cdf_{source_alias}_read_ok"] = True
        print(f"{source_alias} _change_type distribution")
        display(cdf_df.groupBy("_change_type").count().orderBy(F.desc("count"), "_change_type"))
        dirty_key_dfs.append(_extract_dirty_leg_keys_from_cdf(cdf_df, source_alias))
else:
    _record("cdf_probe", "batch CDF probe skipped", True, "no source-specific CDF versions configured")

# COMMAND ----------

if dirty_key_dfs:
    print("Dirty event date D affects output dates D+1...D+30; D itself is excluded. EMA remains deferred.")
    dirty_legs = dirty_key_dfs[0]
    for next_dirty_keys in dirty_key_dfs[1:]:
        dirty_legs = dirty_legs.unionByName(next_dirty_keys)
    dirty_legs = dirty_legs.dropDuplicates(["leg_no", "dirty_source_alias"])

    dirty_leg_source_count = dirty_legs.count()
    _record("dirty_keys", "dirty leg/source candidates", dirty_leg_source_count >= 0, f"rows={dirty_leg_source_count}")
    display(dirty_legs.orderBy(F.desc("max_commit_version"), F.desc("max_update_key"), "leg_no").limit(MAX_SAMPLE_ROWS))

    current_leg = select_current_latest(leg_src, partition_cols=("leg_no",))
    dirty_events = map_dirty_legs_to_taxi_out_events(
        dirty_legs,
        current_leg,
        history_start=HISTORY_START,
        data_cutoff_date=DATA_CUTOFF_DATE,
    )
    if ENTITY_FILTER:
        dirty_events = dirty_events.where(F.col(ENTITY_COL) == F.lit(ENTITY_FILTER))
    if REQUIRE_FULL_AFFECTED_WINDOW and MAX_CURRENT_MV_EVENT_DATE is not None:
        dirty_events = dirty_events.where(
            F.date_add(F.col("dirty_event_date"), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE)
        )

    dirty_event_count = dirty_events.count()
    _record("dirty_events", "eligible taxi-out dirty events", dirty_event_count >= 0, f"rows={dirty_event_count}")
    display(dirty_events.orderBy(ENTITY_COL, "dirty_event_date", "leg_no").limit(MAX_SAMPLE_ROWS))

    affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs(dirty_events)
    affected_output_count = affected_outputs.count()
    _record("affected_outputs", "affected output pairs", affected_output_count >= 0, f"rows={affected_output_count}")
    readiness_flags["dirty_preview_available"] = True
    display(affected_outputs.orderBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL).limit(MAX_SAMPLE_ROWS))

    affected_output_duplicates = (
        affected_outputs.groupBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL)
        .count()
        .where(F.col("count") > F.lit(1))
        .count()
    )
    _record(
        "affected_outputs",
        "unique affected output key",
        affected_output_duplicates == 0,
        f"duplicate dep_ap_sched/affected_output_date keys={affected_output_duplicates}",
    )
else:
    _record("dirty_keys", "dirty event mapping skipped", True, "no optional CDF probe rows configured")

# COMMAND ----------

readiness_df = spark.createDataFrame(readiness_rows)
print("Stage 30C-0 production readiness summary")
display(readiness_df)

final_boolean_summary = spark.createDataFrame(
    [(check_name, bool(passed)) for check_name, passed in readiness_flags.items()],
    ["check_name", "passed"],
)
print("Stage 30C-0 final boolean summary")
display(final_boolean_summary)

failed_count = readiness_df.where(~F.col("passed")).count()
print(f"readiness failed checks: {failed_count}")
if failed_count:
    print("Readiness result: NOT READY for production implementation. Review failed diagnostics above.")
else:
    print("Readiness result: no blocking read-only diagnostics failed. Production write design remains future work.")

print("Stage 30C-0 taxi-out production readiness diagnostics completed.")
