# Databricks notebook source
# Stage 30B-0 dirty-key source discovery.
#
# READ-ONLY DIAGNOSTIC NOTEBOOK.
# This notebook must not mutate tables or workspace resources. It only inspects
# schemas, counts, duplicates, recency buckets, and source-vs-current coverage.

from pyspark.sql import functions as F
from pyspark.sql import Window

print("=" * 90)
print("Stage 30B-0 dirty-key source discovery")
print("READ-ONLY diagnostics only. Do not add mutation logic to this notebook.")
print("=" * 90)

# COMMAND ----------

SOURCE_CATALOG = "panda_silver_prod"
SOURCE_SCHEMA = "occ_ops"
SILVER_CATALOG = "panda_silver_dev"
SILVER_SCHEMA = "ml_ops"

SOURCE_TABLES = {
    "leg": "netline___schedops__leg",
    "leg_times": "netline___schedops__leg_times",
    "leg_misc": "netline___schedops__leg_misc",
}

CURRENT_STREAM_TABLES = {
    "leg": "ft_leg_status",
    "leg_times": "ft_leg_times",
    "leg_misc": "ft_leg_misc",
}

FINAL_POC_TABLE = "ft_airport_daily_taxi_out"

SOURCE_CHANGE_CANDIDATES = [
    "update_key",
    "entry_dt",
    "__START_AT",
    "__END_AT",
    "_commit_timestamp",
    "commit_timestamp",
    "last_update_dt",
    "updated_at",
    "update_dt",
    "modified_at",
    "modification_dt",
    "created_at",
    "insert_dt",
]


def source_table(name: str) -> str:
    return f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}.{name}"


def silver_table(name: str) -> str:
    return f"{SILVER_CATALOG}.{SILVER_SCHEMA}.{name}"


def table_exists(full_name: str) -> bool:
    try:
        spark.table(full_name).limit(1).count()
        return True
    except Exception as exc:
        print(f"[WARN] Cannot read {full_name}: {exc}")
        return False


def columns_for(full_name: str) -> list[str]:
    try:
        return spark.table(full_name).columns
    except Exception as exc:
        print(f"[WARN] Cannot inspect columns for {full_name}: {exc}")
        return []


def safe_count(df, label: str) -> None:
    try:
        print(f"{label}: {df.count():,}")
    except Exception as exc:
        print(f"[WARN] Count failed for {label}: {exc}")


def existing_columns(full_name: str, candidates: list[str]) -> list[str]:
    cols = set(columns_for(full_name))
    return [col for col in candidates if col in cols]


def print_schema_summary(full_name: str) -> None:
    print(f"\n--- Schema summary: {full_name}")
    cols = columns_for(full_name)
    if not cols:
        return
    print(f"Column count: {len(cols)}")
    print("Columns:")
    print(", ".join(cols))
    change_cols = existing_columns(full_name, SOURCE_CHANGE_CANDIDATES)
    print(f"Candidate change/version columns: {change_cols or 'none found'}")


def inspect_duplicate_versions(full_name: str, key_col: str = "leg_no") -> None:
    cols = columns_for(full_name)
    if key_col not in cols:
        print(f"[INFO] {full_name}: no {key_col}; duplicate/version check skipped")
        return

    print(f"\n--- Multiple rows per {key_col}: {full_name}")
    df = spark.table(full_name)
    dup = (
        df.groupBy(key_col)
        .agg(F.count("*").alias("rows_per_key"))
        .filter(F.col("rows_per_key") > 1)
        .orderBy(F.desc("rows_per_key"))
    )
    safe_count(dup, f"{full_name} keys with multiple rows")
    display(dup.limit(20))


def inspect_recent_buckets(full_name: str) -> None:
    change_cols = existing_columns(full_name, SOURCE_CHANGE_CANDIDATES)
    if not change_cols:
        print(f"\n--- Recent change buckets: {full_name}")
        print("[INFO] No candidate timestamp/version columns found.")
        return

    df = spark.table(full_name)
    print(f"\n--- Recent change buckets: {full_name}")
    for col_name in change_cols:
        try:
            col_type = dict(df.dtypes).get(col_name, "")
            print(f"\nCandidate column: {col_name} ({col_type})")
            if col_type in {"timestamp", "date"} or "timestamp" in col_type or "date" in col_type:
                buckets = (
                    df.withColumn("_change_day", F.to_date(F.col(col_name)))
                    .groupBy("_change_day")
                    .agg(F.count("*").alias("rows"))
                    .orderBy(F.desc("_change_day"))
                )
                display(buckets.limit(14))
            else:
                buckets = (
                    df.groupBy(col_name)
                    .agg(F.count("*").alias("rows"))
                    .orderBy(F.desc(col_name))
                )
                display(buckets.limit(20))
        except Exception as exc:
            print(f"[WARN] Bucket check failed for {full_name}.{col_name}: {exc}")


def latest_by_candidate_column(full_name: str, key_col: str = "leg_no"):
    cols = columns_for(full_name)
    if key_col not in cols:
        return None

    change_cols = existing_columns(full_name, SOURCE_CHANGE_CANDIDATES)
    df = spark.table(full_name)
    order_exprs = []
    for col_name in change_cols:
        order_exprs.append(F.col(col_name).desc_nulls_last())

    if not order_exprs:
        order_exprs.append(F.monotonically_increasing_id().desc())

    w = Window.partitionBy(key_col).orderBy(*order_exprs)
    return df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")


def compare_latest_source_to_current(source_full_name: str, current_full_name: str, label: str) -> None:
    print(f"\n--- Latest source vs current stream table: {label}")
    if not table_exists(source_full_name) or not table_exists(current_full_name):
        return

    source_latest = latest_by_candidate_column(source_full_name)
    if source_latest is None:
        print(f"[INFO] {source_full_name}: no leg_no; comparison skipped")
        return

    current = spark.table(current_full_name)
    if "leg_no" not in current.columns:
        print(f"[INFO] {current_full_name}: no leg_no; comparison skipped")
        return

    missing_current = source_latest.select("leg_no").join(
        current.select("leg_no").distinct(),
        on="leg_no",
        how="left_anti",
    )
    safe_count(missing_current, f"{label}: latest source leg_no values not present downstream")
    display(missing_current.limit(20))


def inspect_arr_transitions() -> None:
    source_full_name = source_table(SOURCE_TABLES["leg"])
    current_full_name = silver_table(CURRENT_STREAM_TABLES["leg"])
    print("\n--- ARR visibility and possible state transitions")

    if not table_exists(source_full_name) or not table_exists(current_full_name):
        return

    source_cols = columns_for(source_full_name)
    current_cols = columns_for(current_full_name)
    if "leg_no" not in source_cols or "leg_state" not in source_cols:
        print("[INFO] Source leg table lacks leg_no or leg_state; ARR check skipped")
        return
    if "leg_no" not in current_cols:
        print("[INFO] Current ft_leg_status lacks leg_no; ARR check skipped")
        return

    source_arr = spark.table(source_full_name).filter(F.col("leg_state") == "ARR")
    current_leg = spark.table(current_full_name)

    source_current_arr_missing = source_arr.select("leg_no").distinct().join(
        current_leg.select("leg_no").distinct(),
        on="leg_no",
        how="left_anti",
    )
    safe_count(source_current_arr_missing, "Source ARR leg_no values not represented in ft_leg_status")
    display(source_current_arr_missing.limit(20))

    version_cols = existing_columns(source_full_name, SOURCE_CHANGE_CANDIDATES)
    if version_cols:
        print(f"[INFO] Candidate columns for inspecting NOT-ARR -> ARR transitions: {version_cols}")
        display(
            spark.table(source_full_name)
            .select([col for col in ["leg_no", "leg_state", *version_cols] if col in source_cols])
            .orderBy(F.desc(version_cols[0]))
            .limit(50)
        )
    else:
        print("[INFO] No candidate version/update columns found for transition ordering.")


def inspect_final_poc_table() -> None:
    full_name = silver_table(FINAL_POC_TABLE)
    print(f"\n--- POC target: {full_name}")
    if not table_exists(full_name):
        return

    cols = columns_for(full_name)
    print(f"Columns in {FINAL_POC_TABLE}:")
    print(", ".join(cols))

    needed = {
        "entity_key": "dep_ap_sched",
        "date_key": "event_date",
        "source_event_date": "event_date from cleaned flight data",
        "source_measures": "taxi_out_sec and duration_ratio",
    }
    print("Expected source/entity/date inputs for dirty-key POC:")
    for key, value in needed.items():
        print(f"- {key}: {value}")

    for required_col in ["dep_ap_sched", "event_date"]:
        if required_col not in cols:
            print(f"[WARN] Expected final table column missing: {required_col}")

    display(spark.table(full_name).orderBy(F.desc("event_date")).limit(20))


# COMMAND ----------

print("\nCandidate source tables")
for table_name in SOURCE_TABLES.values():
    full_name = source_table(table_name)
    print_schema_summary(full_name)
    inspect_duplicate_versions(full_name)
    inspect_recent_buckets(full_name)

print("\nCurrent candidate feature-store stream tables")
for table_name in CURRENT_STREAM_TABLES.values():
    full_name = silver_table(table_name)
    print_schema_summary(full_name)
    inspect_duplicate_versions(full_name)
    inspect_recent_buckets(full_name)

# COMMAND ----------

compare_latest_source_to_current(
    source_table(SOURCE_TABLES["leg"]),
    silver_table(CURRENT_STREAM_TABLES["leg"]),
    "leg status",
)
compare_latest_source_to_current(
    source_table(SOURCE_TABLES["leg_times"]),
    silver_table(CURRENT_STREAM_TABLES["leg_times"]),
    "OOOI times",
)
compare_latest_source_to_current(
    source_table(SOURCE_TABLES["leg_misc"]),
    silver_table(CURRENT_STREAM_TABLES["leg_misc"]),
    "stand assignment",
)

inspect_arr_transitions()
inspect_final_poc_table()

# COMMAND ----------

print("\n" + "=" * 90)
print("30B-0 conclusions to copy back")
print("=" * 90)
print("1. Which candidate source table has reliable change/version columns?")
print("2. Do source tables keep multiple versions per leg_no or only current state?")
print("3. Are NOT-ARR -> ARR transitions visible with ordering information?")
print("4. Are OOOI time corrections visible in source and represented in ft_leg_times?")
print("5. Are stand corrections visible in source and represented in ft_leg_misc?")
print("6. Is ft_airport_daily_taxi_out still the best 30B-1 POC target?")
print("7. Do not proceed to write strategy until candidate output matches the current MV.")
