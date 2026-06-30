# Databricks notebook source
# Stage 30B-4b CDF readStream dirty-key detection POC.

from uuid import uuid4

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30B-4 readStream dirty-key detection POC. "
    "This notebook must not mutate production tables or workspace resources."
)
print("=" * 100)

# COMMAND ----------

RUN_STREAM = False
RUN_BATCH_CDF = False

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"

SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"

USE_READ_CHANGE_FEED = True
STREAM_TRIGGER_AVAILABLE_NOW = True

STARTING_VERSION = None
ENDING_VERSION = None

CHECKPOINT_BASE_PATH = ""
CLEANUP_CHECKPOINT = True
MEMORY_QUERY_PREFIX = "stage30b_dirty_key_poc"
MAX_SAMPLE_ROWS = 100


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


CONFIG_ROWS = [
    ("RUN_STREAM", str(RUN_STREAM)),
    ("RUN_BATCH_CDF", str(RUN_BATCH_CDF)),
    ("SOURCE_LEG", source_table(SOURCE_LEG_TABLE)),
    ("SOURCE_LEG_TIMES", source_table(SOURCE_LEG_TIMES_TABLE)),
    ("USE_READ_CHANGE_FEED", str(USE_READ_CHANGE_FEED)),
    ("STREAM_TRIGGER_AVAILABLE_NOW", str(STREAM_TRIGGER_AVAILABLE_NOW)),
    ("STARTING_VERSION", str(STARTING_VERSION)),
    ("ENDING_VERSION", str(ENDING_VERSION)),
    ("CHECKPOINT_BASE_PATH", CHECKPOINT_BASE_PATH or "<required for stream mode>"),
    ("CLEANUP_CHECKPOINT", str(CLEANUP_CHECKPOINT)),
    ("MEMORY_QUERY_PREFIX", MEMORY_QUERY_PREFIX),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_STREAM and not RUN_BATCH_CDF:
    print("RUN_STREAM and RUN_BATCH_CDF are False. This notebook is safe-gated.")
    print("Review the configuration above before starting CDF stream or batch diagnostics.")
    dbutils.notebook.exit("RUN_CDF_DIAGNOSTICS_FALSE")

if not USE_READ_CHANGE_FEED:
    raise ValueError("USE_READ_CHANGE_FEED must remain True for the Stage 30B-4b diagnostic path.")

if ENDING_VERSION is not None and RUN_STREAM:
    print("ENDING_VERSION is used only by batch CDF mode in this notebook.")

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
        "selected_columns": (
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
        "required_columns": (
            "leg_no",
            "update_key",
            "__START_AT",
            "__END_AT",
            "offblock_dt",
            "airborne_dt",
            *CDF_COLUMNS,
        ),
        "selected_columns": (
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


def _with_cdf_option(reader):
    return reader.option("readChangeFeed", "true")


def _read_cdf_stream_table(table_name: str):
    reader = _with_cdf_option(spark.readStream)
    if STARTING_VERSION is not None:
        reader = reader.option("startingVersion", str(STARTING_VERSION))
    return reader.table(source_table(table_name))


def _read_cdf_batch_table(table_name: str):
    reader = _with_cdf_option(spark.read)
    if STARTING_VERSION is not None:
        reader = reader.option("startingVersion", str(STARTING_VERSION))
    if ENDING_VERSION is not None:
        reader = reader.option("endingVersion", str(ENDING_VERSION))
    return reader.table(source_table(table_name))


def _missing_columns(df, required_columns: tuple[str, ...]) -> list[str]:
    available = set(df.columns)
    return [column for column in required_columns if column not in available]


def _select_raw_dirty_candidates(cdf_df, selected_columns: tuple[str, ...], source_alias: str):
    return cdf_df.select(
        *[F.col(column) for column in selected_columns],
        F.lit(source_alias).alias("source_alias"),
    )


def _validate_checkpoint_base_path() -> str:
    checkpoint_base_path = CHECKPOINT_BASE_PATH.strip().rstrip("/")
    if not checkpoint_base_path:
        raise ValueError("CHECKPOINT_BASE_PATH must be set for RUN_STREAM mode.")
    if not checkpoint_base_path.startswith("/Volumes/"):
        raise ValueError("CHECKPOINT_BASE_PATH must be a UC Volume path that starts with /Volumes/.")

    lowered = checkpoint_base_path.lower()
    if not any(marker in lowered for marker in ("tmp", "temp", "debug", "stage30b")):
        raise ValueError("CHECKPOINT_BASE_PATH must clearly be a temporary/debug Stage 30B path.")
    if any(marker in lowered for marker in ("/resources/", "/pipelines/", "/production/")):
        raise ValueError("CHECKPOINT_BASE_PATH must not point at a production or bundle-managed path.")
    return checkpoint_base_path


def _checkpoint_path_for(source_alias: str) -> str:
    checkpoint_base_path = _validate_checkpoint_base_path()
    return f"{checkpoint_base_path}/{MEMORY_QUERY_PREFIX}_{source_alias}_{uuid4().hex[:8]}"


def _cleanup_checkpoint(checkpoint_path: str) -> None:
    if not CLEANUP_CHECKPOINT:
        print(f"Checkpoint cleanup disabled for {checkpoint_path}")
        return
    try:
        dbutils.fs.rm(checkpoint_path, recurse=True)
        print(f"Cleaned diagnostic checkpoint: {checkpoint_path}")
    except Exception as exc:
        print(f"Could not clean diagnostic checkpoint {checkpoint_path}: {exc}")


def _start_memory_query(raw_stream_df, source_alias: str):
    query_name = f"{MEMORY_QUERY_PREFIX}_{source_alias}_{uuid4().hex[:8]}"
    checkpoint_path = _checkpoint_path_for(source_alias)
    query = None
    try:
        writer = (
            raw_stream_df.writeStream.format("memory")
            .queryName(query_name)
            .outputMode("append")
            .option("checkpointLocation", checkpoint_path)
        )
        if STREAM_TRIGGER_AVAILABLE_NOW:
            writer = writer.trigger(availableNow=True)
        query = writer.start()
        query.awaitTermination()
        if query.exception() is not None:
            raise query.exception()
        return query_name, checkpoint_path
    finally:
        if query is not None and query.isActive:
            query.stop()
        _cleanup_checkpoint(checkpoint_path)


def _summarize_static_cdf_rows(cdf_rows_df, source_alias: str, mode: str) -> dict[str, object]:
    print(f"Sample rows for {mode} {source_alias}")
    display(cdf_rows_df.limit(MAX_SAMPLE_ROWS))

    counts = cdf_rows_df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("leg_no").alias("unique_leg_no_count"),
        F.min("update_key").alias("min_update_key"),
        F.max("update_key").alias("max_update_key"),
        F.min("_commit_version").alias("min_commit_version"),
        F.max("_commit_version").alias("max_commit_version"),
    ).first()

    print(f"{mode} {source_alias} _change_type distribution")
    display(cdf_rows_df.groupBy("_change_type").count().orderBy(F.desc("count"), "_change_type"))

    summary = {
        "mode": mode,
        "source_alias": source_alias,
        "row_count": counts["row_count"],
        "unique_leg_no_count": counts["unique_leg_no_count"],
        "min_update_key": counts["min_update_key"],
        "max_update_key": counts["max_update_key"],
        "min_commit_version": counts["min_commit_version"],
        "max_commit_version": counts["max_commit_version"],
        "update_key_present": counts["min_update_key"] is not None or counts["max_update_key"] is not None,
        "arr_row_count": None,
        "taxi_out_candidate_row_count": None,
        "offblock_non_null_count": None,
        "airborne_non_null_count": None,
    }

    if source_alias == "leg":
        print(f"{mode} leg_state distribution")
        display(cdf_rows_df.groupBy("leg_state").count().orderBy(F.desc("count"), "leg_state"))
        summary["arr_row_count"] = cdf_rows_df.where(F.col("leg_state") == F.lit("ARR")).count()
        summary["taxi_out_candidate_row_count"] = cdf_rows_df.where(
            (F.col("counter") == F.lit(0))
            & F.col("leg_type").isin("J", "C", "G")
            & (F.col("leg_state") == F.lit("ARR"))
        ).count()

    if source_alias == "leg_times":
        summary["offblock_non_null_count"] = cdf_rows_df.where(F.col("offblock_dt").isNotNull()).count()
        summary["airborne_non_null_count"] = cdf_rows_df.where(F.col("airborne_dt").isNotNull()).count()

    return summary


def _schema_failure_summary(source_alias: str, mode: str, missing: list[str]) -> dict[str, object]:
    return {
        "mode": mode,
        "source_alias": source_alias,
        "started_or_read": False,
        "required_schema_ok": False,
        "missing_columns": ", ".join(missing),
        "rows_observed": False,
        "row_count": 0,
        "unique_leg_no_count": 0,
        "update_key_present": False,
        "arr_rows_observed": None,
        "taxi_out_candidate_rows_observed": None,
        "oooi_fields_observed": None,
        "min_commit_version": None,
        "max_commit_version": None,
    }


def _success_summary(source_summary: dict[str, object], started_or_read: bool, missing: list[str]) -> dict[str, object]:
    row_count = source_summary["row_count"]
    source_alias = source_summary["source_alias"]
    rows_observed = row_count > 0
    return {
        "mode": source_summary["mode"],
        "source_alias": source_alias,
        "started_or_read": started_or_read,
        "required_schema_ok": not missing,
        "missing_columns": "",
        "rows_observed": rows_observed,
        "row_count": row_count,
        "unique_leg_no_count": source_summary["unique_leg_no_count"],
        "update_key_present": source_summary["update_key_present"],
        "arr_rows_observed": (
            source_summary["arr_row_count"] > 0
            if source_summary["arr_row_count"] is not None
            else None
        ),
        "taxi_out_candidate_rows_observed": (
            source_summary["taxi_out_candidate_row_count"] > 0
            if source_summary["taxi_out_candidate_row_count"] is not None
            else None
        ),
        "oooi_fields_observed": (
            (
                source_summary["offblock_non_null_count"] > 0
                or source_summary["airborne_non_null_count"] > 0
            )
            if source_alias == "leg_times"
            else None
        ),
        "min_commit_version": source_summary["min_commit_version"],
        "max_commit_version": source_summary["max_commit_version"],
    }

# COMMAND ----------

summary_rows = []

if RUN_STREAM:
    _validate_checkpoint_base_path()
    for spec in SOURCE_SPECS:
        source_alias = spec["source_alias"]
        table_name = spec["table_name"]
        print("-" * 100)
        print(f"CDF stream diagnostic for {source_alias} ({source_table(table_name)})")

        stream_df = _read_cdf_stream_table(table_name)
        print("CDF streaming schema")
        stream_df.printSchema()

        missing = _missing_columns(stream_df, spec["required_columns"])
        if missing:
            print(f"Missing required columns for {source_alias}: {missing}")
            summary_rows.append(_schema_failure_summary(source_alias, "stream", missing))
            continue

        raw_dirty_candidates = _select_raw_dirty_candidates(
            stream_df,
            spec["selected_columns"],
            source_alias,
        )
        query_name, checkpoint_path = _start_memory_query(raw_dirty_candidates, source_alias)
        print(f"Memory query {query_name} completed for {source_alias}; checkpoint: {checkpoint_path}")

        memory_df = spark.table(query_name)
        source_summary = _summarize_static_cdf_rows(memory_df, source_alias, "stream")
        summary_rows.append(_success_summary(source_summary, started_or_read=True, missing=missing))

# COMMAND ----------

if RUN_BATCH_CDF:
    for spec in SOURCE_SPECS:
        source_alias = spec["source_alias"]
        table_name = spec["table_name"]
        print("-" * 100)
        print(f"Batch CDF diagnostic for {source_alias} ({source_table(table_name)})")

        batch_df = _read_cdf_batch_table(table_name)
        print("CDF batch schema")
        batch_df.printSchema()

        missing = _missing_columns(batch_df, spec["required_columns"])
        if missing:
            print(f"Missing required columns for {source_alias}: {missing}")
            summary_rows.append(_schema_failure_summary(source_alias, "batch_cdf", missing))
            continue

        raw_dirty_candidates = _select_raw_dirty_candidates(
            batch_df,
            spec["selected_columns"],
            source_alias,
        )
        source_summary = _summarize_static_cdf_rows(raw_dirty_candidates, source_alias, "batch_cdf")
        summary_rows.append(_success_summary(source_summary, started_or_read=True, missing=missing))

# COMMAND ----------

summary_df = spark.createDataFrame(summary_rows)
print("Stage 30B-4b CDF readStream dirty-key POC summary")
display(summary_df)

summary = [row.asDict() for row in summary_df.collect()]

stream_rows = [row for row in summary if row["mode"] == "stream"]
batch_rows = [row for row in summary if row["mode"] == "batch_cdf"]
leg_rows = [row for row in summary if row["source_alias"] == "leg"]
leg_times_rows = [row for row in summary if row["source_alias"] == "leg_times"]

stream_started_for_sources = all(row["started_or_read"] for row in stream_rows) if stream_rows else None
schema_ok_for_sources = all(row["required_schema_ok"] for row in summary)
rows_observed_for_any_source = any(row["rows_observed"] for row in summary)
update_key_observed_for_any_source = any(row["update_key_present"] for row in summary)
leg_arr_rows_observed = any(row["arr_rows_observed"] for row in leg_rows)
leg_taxi_out_rows_observed = any(row["taxi_out_candidate_rows_observed"] for row in leg_rows)
leg_times_oooi_fields_observed = any(row["oooi_fields_observed"] for row in leg_times_rows)

print("Interpretation")
print(f"stream_started_for_sources: {stream_started_for_sources}")
print(f"batch_cdf_sources_read: {len(batch_rows) if batch_rows else None}")
print(f"schema_ok_for_sources: {schema_ok_for_sources}")
print(f"rows_observed_for_any_source: {rows_observed_for_any_source}")
print(f"leg_arr_rows_observed: {leg_arr_rows_observed}")
print(f"leg_taxi_out_rows_observed: {leg_taxi_out_rows_observed}")
print(f"leg_times_oooi_fields_observed: {leg_times_oooi_fields_observed}")
print(f"update_key_observed_for_any_source: {update_key_observed_for_any_source}")

if schema_ok_for_sources and rows_observed_for_any_source and update_key_observed_for_any_source:
    print("POC signal: CDF can emit raw dirty-key candidate rows for at least one tested source.")
else:
    print("POC signal: inconclusive. A bounded CDF sample with no rows is not a failure by itself.")

print("Limitations")
print("- This depends on upstream source CDF behavior that already exists.")
print("- A durable checkpoint strategy is still needed for any later production design.")
print("- Change types must be handled explicitly before dirty ranges are trusted.")
print("- Entity/date moves and ARR state transitions need later dirty-range design.")
print("- No production writes are performed.")
print("- EMA partial recompute remains deferred.")

print("Stage 30B-4b CDF readStream dirty-key detection POC completed.")
