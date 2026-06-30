# Databricks notebook source
# Stage 30B-4 readStream dirty-key detection POC.

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

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"

SOURCE_LEG_TABLE = "netline___schedops__leg"
SOURCE_LEG_TIMES_TABLE = "netline___schedops__leg_times"

STREAM_DURATION_SECONDS = 30
MEMORY_QUERY_PREFIX = "stage30b_dirty_key_poc"
MAX_SAMPLE_ROWS = 100

USE_SKIP_CHANGE_COMMITS = True


def source_table(table_name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{table_name}"


CONFIG_ROWS = [
    ("RUN_STREAM", str(RUN_STREAM)),
    ("SOURCE_LEG", source_table(SOURCE_LEG_TABLE)),
    ("SOURCE_LEG_TIMES", source_table(SOURCE_LEG_TIMES_TABLE)),
    ("STREAM_DURATION_SECONDS", str(STREAM_DURATION_SECONDS)),
    ("MEMORY_QUERY_PREFIX", MEMORY_QUERY_PREFIX),
    ("MAX_SAMPLE_ROWS", str(MAX_SAMPLE_ROWS)),
    ("USE_SKIP_CHANGE_COMMITS", str(USE_SKIP_CHANGE_COMMITS)),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_STREAM:
    print("RUN_STREAM is False. This notebook is safe-gated and no streams were started.")
    print("Review the configuration above, set RUN_STREAM = True, then run once manually.")
    dbutils.notebook.exit("RUN_STREAM_FALSE")

# COMMAND ----------

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
        ),
        "selected_columns": (
            "leg_no",
            "update_key",
            "__START_AT",
            "__END_AT",
            "offblock_dt",
            "airborne_dt",
        ),
    },
]


def _read_stream_table(table_name: str):
    reader = spark.readStream
    if USE_SKIP_CHANGE_COMMITS:
        reader = reader.option("skipChangeCommits", "true")
    return reader.table(source_table(table_name))


def _missing_columns(df, required_columns: tuple[str, ...]) -> list[str]:
    available = set(df.columns)
    return [column for column in required_columns if column not in available]


def _select_raw_dirty_candidates(stream_df, selected_columns: tuple[str, ...], source_alias: str):
    return stream_df.select(
        *[F.col(column) for column in selected_columns],
        F.lit(source_alias).alias("source_alias"),
    )


def _start_memory_query(raw_stream_df, source_alias: str):
    query_name = f"{MEMORY_QUERY_PREFIX}_{source_alias}_{uuid4().hex[:8]}"
    query = None
    try:
        query = (
            raw_stream_df.writeStream.format("memory")
            .queryName(query_name)
            .outputMode("append")
            .start()
        )
        terminated = query.awaitTermination(STREAM_DURATION_SECONDS)
        if query.exception() is not None:
            raise query.exception()
        return query_name, terminated
    finally:
        if query is not None and query.isActive:
            query.stop()


def _summarize_memory_table(query_name: str, source_alias: str) -> dict[str, object]:
    memory_df = spark.table(query_name)

    print(f"Sample rows for {source_alias}")
    display(memory_df.limit(MAX_SAMPLE_ROWS))

    counts = memory_df.agg(
        F.count("*").alias("row_count"),
        F.countDistinct("leg_no").alias("unique_leg_no_count"),
        F.min("update_key").alias("min_update_key"),
        F.max("update_key").alias("max_update_key"),
    ).first()

    summary = {
        "source_alias": source_alias,
        "row_count": counts["row_count"],
        "unique_leg_no_count": counts["unique_leg_no_count"],
        "min_update_key": counts["min_update_key"],
        "max_update_key": counts["max_update_key"],
        "update_key_present": counts["min_update_key"] is not None or counts["max_update_key"] is not None,
        "arr_row_count": None,
        "offblock_non_null_count": None,
        "airborne_non_null_count": None,
    }

    if source_alias == "leg":
        print("leg_state distribution")
        display(memory_df.groupBy("leg_state").count().orderBy(F.desc("count"), "leg_state"))
        summary["arr_row_count"] = memory_df.where(F.col("leg_state") == F.lit("ARR")).count()

    if source_alias == "leg_times":
        summary["offblock_non_null_count"] = memory_df.where(F.col("offblock_dt").isNotNull()).count()
        summary["airborne_non_null_count"] = memory_df.where(F.col("airborne_dt").isNotNull()).count()

    return summary

# COMMAND ----------

summary_rows = []

for spec in SOURCE_SPECS:
    source_alias = spec["source_alias"]
    table_name = spec["table_name"]
    print("-" * 100)
    print(f"Testing source: {source_alias} ({source_table(table_name)})")

    stream_df = _read_stream_table(table_name)
    print("Streaming schema")
    stream_df.printSchema()

    missing = _missing_columns(stream_df, spec["required_columns"])
    schema_ok = not missing
    if not schema_ok:
        print(f"Missing required columns for {source_alias}: {missing}")
        summary_rows.append(
            {
                "source_alias": source_alias,
                "stream_started": False,
                "required_schema_ok": False,
                "missing_columns": ", ".join(missing),
                "rows_observed": False,
                "row_count": 0,
                "unique_leg_no_count": 0,
                "update_key_present": False,
                "arr_rows_observed": None,
                "oooi_fields_observed": None,
            }
        )
        continue

    raw_dirty_candidates = _select_raw_dirty_candidates(
        stream_df,
        spec["selected_columns"],
        source_alias,
    )
    query_name, terminated = _start_memory_query(raw_dirty_candidates, source_alias)
    print(f"Memory query {query_name} started for {source_alias}; terminated during timeout: {terminated}")

    source_summary = _summarize_memory_table(query_name, source_alias)
    row_count = source_summary["row_count"]
    rows_observed = row_count > 0
    arr_rows_observed = (
        source_summary["arr_row_count"] > 0
        if source_summary["arr_row_count"] is not None
        else None
    )
    oooi_fields_observed = (
        (
            source_summary["offblock_non_null_count"] > 0
            or source_summary["airborne_non_null_count"] > 0
        )
        if source_alias == "leg_times"
        else None
    )

    summary_rows.append(
        {
            "source_alias": source_alias,
            "stream_started": True,
            "required_schema_ok": schema_ok,
            "missing_columns": "",
            "rows_observed": rows_observed,
            "row_count": row_count,
            "unique_leg_no_count": source_summary["unique_leg_no_count"],
            "update_key_present": source_summary["update_key_present"],
            "arr_rows_observed": arr_rows_observed,
            "oooi_fields_observed": oooi_fields_observed,
        }
    )

# COMMAND ----------

summary_df = spark.createDataFrame(summary_rows)
print("Stage 30B-4 POC summary")
display(summary_df)

summary = {row["source_alias"]: row.asDict() for row in summary_df.collect()}
leg_summary = summary.get("leg", {})
leg_times_summary = summary.get("leg_times", {})

stream_started_for_sources = all(summary.get(alias, {}).get("stream_started") for alias in ("leg", "leg_times"))
schema_ok_for_sources = all(summary.get(alias, {}).get("required_schema_ok") for alias in ("leg", "leg_times"))
rows_observed_for_any_source = any(summary.get(alias, {}).get("rows_observed") for alias in ("leg", "leg_times"))
update_key_observed_for_any_source = any(
    summary.get(alias, {}).get("update_key_present") for alias in ("leg", "leg_times")
)

print("Interpretation")
print(f"stream_started_for_sources: {stream_started_for_sources}")
print(f"schema_ok_for_sources: {schema_ok_for_sources}")
print(f"rows_observed_for_any_source: {rows_observed_for_any_source}")
print(f"leg_arr_rows_observed: {leg_summary.get('arr_rows_observed')}")
print(f"leg_times_oooi_fields_observed: {leg_times_summary.get('oooi_fields_observed')}")
print(f"update_key_observed_for_any_source: {update_key_observed_for_any_source}")

if stream_started_for_sources and schema_ok_for_sources and rows_observed_for_any_source and update_key_observed_for_any_source:
    print("POC signal: readStream can emit raw dirty-key candidate rows for at least one tested source.")
else:
    print("POC signal: inconclusive. A short sampling window with no rows is not a failure by itself.")

print("Limitations")
print("- skipChangeCommits semantics must be considered for source correction visibility.")
print("- Whether updates appear as SCD2 append rows must be empirically confirmed from observed rows.")
print("- No production checkpoint is configured.")
print("- No production writes are performed.")
print("- EMA partial recompute remains deferred.")

print("Stage 30B-4 readStream dirty-key detection POC completed.")
