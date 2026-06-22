# Databricks notebook source
from pathlib import Path
import importlib
import sys


def _get_notebook_path(dbutils_obj) -> str:
    try:
        notebook_path = (
            dbutils_obj.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
    except Exception:
        return ""

    if notebook_path and not notebook_path.startswith("/Workspace"):
        notebook_path = f"/Workspace{notebook_path}"
    return notebook_path


def _resolve_project_root(dbutils_obj) -> Path:
    notebook_path = _get_notebook_path(dbutils_obj)

    candidates = []
    if notebook_path:
        notebook_file = Path(notebook_path)
        candidates.extend([notebook_file.parent, *notebook_file.parent.parents])

    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])

    checked = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate_str)
        if (candidate / "src" / "ml_project" / "settings.py").exists():
            return candidate

    raise FileNotFoundError(
        "Nie udało się odnaleźć root projektu zawierającego src/ml_project/settings.py. "
        f"Sprawdzone lokalizacje: {checked}"
    )


PROJECT_ROOT = _resolve_project_root(dbutils)
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import ml_project as mp

mp = importlib.reload(mp)

ENV = mp.ensure_env_widget(dbutils, default="dev")

mp.ensure_text_widget(dbutils, "SOURCE_CATALOG", "panda_silver_prod", "Source catalog")
mp.ensure_text_widget(dbutils, "SOURCE_SCHEMA", "occ_ops", "Source schema")
mp.ensure_text_widget(dbutils, "SILVER_CATALOG", "", "Silver catalog")
mp.ensure_text_widget(dbutils, "SILVER_SCHEMA", "", "Silver schema")
mp.ensure_text_widget(dbutils, "GOLD_CATALOG", "", "Gold catalog")
mp.ensure_text_widget(dbutils, "GOLD_SCHEMA", "", "Gold schema")

SOURCE_CATALOG_WIDGET = mp.get_widget_value(dbutils, "SOURCE_CATALOG", "panda_silver_prod").strip()
SOURCE_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "SOURCE_SCHEMA", "occ_ops").strip()
SILVER_CATALOG_WIDGET = mp.get_widget_value(dbutils, "SILVER_CATALOG", "").strip()
SILVER_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "SILVER_SCHEMA", "").strip()
GOLD_CATALOG_WIDGET = mp.get_widget_value(dbutils, "GOLD_CATALOG", "").strip()
GOLD_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "GOLD_SCHEMA", "").strip()

SETTINGS = mp.load_settings(
    ENV,
    project_root=str(PROJECT_ROOT),
    source_catalog_override=SOURCE_CATALOG_WIDGET or "panda_silver_prod",
    source_schema_override=SOURCE_SCHEMA_WIDGET or "occ_ops",
    silver_catalog_override=SILVER_CATALOG_WIDGET or None,
    silver_schema_override=SILVER_SCHEMA_WIDGET or None,
    gold_catalog_override=GOLD_CATALOG_WIDGET or None,
    gold_schema_override=GOLD_SCHEMA_WIDGET or None,
)

globals().update(mp.settings_to_globals(SETTINGS))
mp.configure_runtime(SETTINGS, spark=spark)

print(f"ENV={ENV}")
print(f"SINK_TABLE={SINK_TABLE}")
print(f"EVENTS_SINK_TABLE={EVENTS_SINK_TABLE}")

# COMMAND ----------

from pyspark.sql import functions as F

hard_failures = []
warnings = []

if not spark.catalog.tableExists(SINK_TABLE):
    hard_failures.append(f"Brak SINK_TABLE: {SINK_TABLE}")

if not spark.catalog.tableExists(EVENTS_SINK_TABLE):
    hard_failures.append(f"Brak EVENTS_SINK_TABLE: {EVENTS_SINK_TABLE}")

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

sink_df = spark.table(SINK_TABLE)
events_df = spark.table(EVENTS_SINK_TABLE)

print("=== BASIC COUNTS ===")
print(f"sink_rows={sink_df.count()}")
print(f"events_rows={events_df.count()}")

events_24h = events_df.filter(F.col("logged_at") >= F.expr("current_timestamp() - INTERVAL 24 HOURS"))
events_7d = events_df.filter(F.col("logged_at") >= F.expr("current_timestamp() - INTERVAL 7 DAYS"))

print(f"events_last_24h={events_24h.count()}")
print(f"events_last_7d={events_7d.count()}")

print("\n=== PREDICTION STATUS ===")
display(
    sink_df.groupBy("prediction_status")
    .count()
    .orderBy(F.desc("count"))
)

print("\n=== OPERATIONAL ACTIVITY ===")
display(
    sink_df.groupBy("is_operationally_active")
    .count()
    .orderBy(F.desc("count"))
)

print("\n=== MISSING FEATURE COUNT ===")
display(
    sink_df.groupBy("missing_feature_count")
    .count()
    .orderBy(F.desc("count"))
)

print("\n=== LATEST MODEL URI ===")
display(
    sink_df.groupBy("model_uri")
    .agg(
        F.count("*").alias("rows"),
        F.max("scored_at").alias("latest_scored_at"),
        F.max("source_commit_version").alias("max_source_commit_version")
    )
    .orderBy(F.desc("latest_scored_at"))
)

print("\n=== LAST 24H STATUS ===")
display(
    events_24h.groupBy("prediction_status")
    .count()
    .orderBy(F.desc("count"))
)

# COMMAND ----------

summary = (
    sink_df.agg(
        F.count("*").alias("sink_rows"),
        F.avg(F.when(F.col("prediction_status") == "COLD_START_FALLBACK", 1.0).otherwise(0.0)).alias("cold_start_share"),
        F.avg(F.when(F.col("prediction_status") == "TOO_MANY_MISSING_FEATURES_FALLBACK", 1.0).otherwise(0.0)).alias("too_many_missing_share"),
        F.avg(F.when(F.col("is_operationally_active") == True, 1.0).otherwise(0.0)).alias("operationally_active_share"),
        F.max("scored_at").alias("latest_scored_at"),
        F.max("source_commit_version").alias("latest_source_commit_version"),
    )
    .first()
)

cold_start_share = float(summary["cold_start_share"] or 0.0)
too_many_missing_share = float(summary["too_many_missing_share"] or 0.0)
operationally_active_share = float(summary["operationally_active_share"] or 0.0)

print("=== SUMMARY ===")
print(f"sink_rows={summary['sink_rows']}")
print(f"cold_start_share={cold_start_share:.2%}")
print(f"too_many_missing_share={too_many_missing_share:.2%}")
print(f"operationally_active_share={operationally_active_share:.2%}")
print(f"latest_scored_at={summary['latest_scored_at']}")
print(f"latest_source_commit_version={summary['latest_source_commit_version']}")

if cold_start_share > 0.50:
    warnings.append(f"Wysoki udział COLD_START_FALLBACK: {cold_start_share:.2%}")

if too_many_missing_share > 0.20:
    warnings.append(f"Wysoki udział TOO_MANY_MISSING_FEATURES_FALLBACK: {too_many_missing_share:.2%}")

if operationally_active_share < 0.30:
    warnings.append(f"Niski udział is_operationally_active=True: {operationally_active_share:.2%}")

if warnings:
    print("\n=== WARNINGS ===")
    for w in warnings:
        print(f"- {w}")

print("\n[OK] Prediction audit / monitoring executed.")

# COMMAND ----------

from delta.tables import DeltaTable
from pyspark.sql import functions as F

AUDIT_TABLE = f"{SILVER_CATALOG}.{SILVER_SCHEMA}.block_time_prediction_quality_audit_v1"
MONITORING_TABLE = f"{SILVER_CATALOG}.{SILVER_SCHEMA}.block_time_prediction_monitoring_summary_v1"

print("AUDIT_TABLE =", AUDIT_TABLE)
print("MONITORING_TABLE =", MONITORING_TABLE)

# COMMAND ----------

def _ensure_columns(df, cols_with_types):
    for c, t in cols_with_types:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast(t))
    return df


def _ensure_table_columns(table_name: str, cols_with_types):
    existing_cols = {c.name.lower() for c in spark.catalog.listColumns(table_name)}
    missing_cols = [(c, t) for c, t in cols_with_types if c.lower() not in existing_cols]

    if missing_cols:
        add_stmt = ", ".join([f"{c} {t}" for c, t in missing_cols])
        spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({add_stmt})")
        print(f"[OK] Dodano brakujące kolumny do {table_name}: {[c for c, _ in missing_cols]}")


expected_event_cols = [
    ("logged_at", "timestamp"),
    ("leg_no", "long"),
    ("scored_at", "timestamp"),
    ("source_commit_version", "long"),
    ("source_commit_timestamp", "timestamp"),
    ("last_change_type", "string"),
    ("model_uri", "string"),
    ("prediction_status", "string"),
    ("is_operationally_active", "boolean"),
    ("dep_sched_dt", "timestamp"),
    ("arr_sched_dt", "timestamp"),
    ("dep_ap_sched", "string"),
    ("arr_ap_sched", "string"),
    ("ac_subtype", "string"),
    ("ac_registration", "string"),
    ("scheduled_block_time_sec", "double"),
    ("pred_actual_block_time_sec", "double"),
    ("pred_actual_block_time_p90_sec", "double"),
    ("pred_block_delay_sec", "double"),
    ("effective_actual_block_time_sec", "double"),
    ("effective_block_delay_sec", "double"),
    ("missing_feature_count", "int"),
    ("model_pred_actual_block_time_sec_raw", "double"),
    ("model_pred_block_delay_sec_raw", "double"),
    ("hours_to_departure_at_prediction", "double"),
]

events_df = spark.table(EVENTS_SINK_TABLE)
events_df = _ensure_columns(events_df, expected_event_cols)

labels_df = spark.table(LABELS_TABLE)
if "__END_AT" in labels_df.columns:
    labels_df = labels_df.filter(F.col("__END_AT").isNull())

labels_arr_df = (
    labels_df
    .filter(F.col("leg_state") == "ARR")
    .filter(F.col("dep_dt").isNotNull())
    .filter(F.col("arr_dt").isNotNull())
    .select(
        "leg_no",
        F.col("dep_dt").alias("actual_dep_dt"),
        F.col("arr_dt").alias("actual_arr_dt"),
        (
            F.col("arr_dt").cast("long") - F.col("dep_dt").cast("long")
        ).cast("double").alias("actual_block_time_sec"),
        (
            (
                F.col("arr_dt").cast("long") - F.col("dep_dt").cast("long")
            ) - (
                F.col("arr_sched_dt").cast("long") - F.col("dep_sched_dt").cast("long")
            )
        ).cast("double").alias("actual_block_delay_sec"),
    )
)

print("events rows =", events_df.count())
print("arrived rows =", labels_arr_df.count())

# COMMAND ----------

audit_source_full = (
    events_df.alias("e")
    .join(labels_arr_df.alias("a"), on="leg_no", how="left")
    .withColumn(
        "hours_to_departure_at_prediction",
        F.coalesce(
            F.col("hours_to_departure_at_prediction"),
            (
                F.col("dep_sched_dt").cast("long") - F.col("scored_at").cast("long")
            ) / F.lit(3600.0),
        )
    )
    .withColumn(
        "horizon_bucket",
        F.when(F.col("hours_to_departure_at_prediction") < 6, "<6h")
         .when(F.col("hours_to_departure_at_prediction") < 24, "6-24h")
         .when(F.col("hours_to_departure_at_prediction") < 72, "24-72h")
         .when(F.col("hours_to_departure_at_prediction") < 168, "3-7d")
         .when(F.col("hours_to_departure_at_prediction") < 336, "7-14d")
         .when(F.col("hours_to_departure_at_prediction") < 720, "14-30d")
         .when(F.col("hours_to_departure_at_prediction") < 1440, "30-60d")
         .when(F.col("hours_to_departure_at_prediction") < 2160, "60-90d")
         .otherwise(">90d")
    )
    .withColumn(
        "scheduled_block_bucket",
        F.when(F.col("scheduled_block_time_sec") < 90 * 60, "Krótkie (<1.5h)")
         .when(F.col("scheduled_block_time_sec") < 180 * 60, "Średnie (1.5h - 3h)")
         .when(F.col("scheduled_block_time_sec") < 360 * 60, "Długie (3h - 6h)")
         .otherwise("Ultra-Długie (>6h)")
    )
    .withColumn(
        "actual_available",
        F.when(F.col("actual_block_time_sec").isNotNull(), F.lit(True)).otherwise(F.lit(False))
    )
    .withColumn(
        "error_pred_block_time_sec",
        F.when(
            F.col("actual_block_time_sec").isNotNull() & F.col("pred_actual_block_time_sec").isNotNull(),
            F.col("pred_actual_block_time_sec") - F.col("actual_block_time_sec")
        ).otherwise(F.lit(None).cast("double"))
    )
    .withColumn(
        "abs_error_pred_block_time_sec",
        F.when(
            F.col("actual_block_time_sec").isNotNull() & F.col("pred_actual_block_time_sec").isNotNull(),
            F.abs(F.col("pred_actual_block_time_sec") - F.col("actual_block_time_sec"))
        ).otherwise(F.lit(None).cast("double"))
    )
    .withColumn(
        "error_effective_block_time_sec",
        F.when(
            F.col("actual_block_time_sec").isNotNull() & F.col("effective_actual_block_time_sec").isNotNull(),
            F.col("effective_actual_block_time_sec") - F.col("actual_block_time_sec")
        ).otherwise(F.lit(None).cast("double"))
    )
    .withColumn(
        "abs_error_effective_block_time_sec",
        F.when(
            F.col("actual_block_time_sec").isNotNull() & F.col("effective_actual_block_time_sec").isNotNull(),
            F.abs(F.col("effective_actual_block_time_sec") - F.col("actual_block_time_sec"))
        ).otherwise(F.lit(None).cast("double"))
    )
    .withColumn(
        "covered_p90_pred_block_time",
        F.when(
            F.col("actual_block_time_sec").isNotNull() & F.col("pred_actual_block_time_p90_sec").isNotNull(),
            F.col("pred_actual_block_time_p90_sec") >= F.col("actual_block_time_sec")
        ).otherwise(F.lit(None).cast("boolean"))
    )
    .withColumn(
        "audit_key",
        F.concat_ws(
            "||",
            F.col("leg_no").cast("string"),
            F.coalesce(F.col("source_commit_version").cast("string"), F.lit("NULL")),
            F.coalesce(F.col("last_change_type"), F.lit("NULL")),
            F.coalesce(F.col("logged_at").cast("string"), F.lit("NULL")),
        )
    )
    .withColumn("audit_loaded_at", F.current_timestamp())
)

audit_source = audit_source_full.filter(F.col("actual_available") == True)

display(
    audit_source_full.select(
        "audit_key",
        "leg_no",
        "logged_at",
        "prediction_status",
        "actual_available",
        "horizon_bucket",
        "scheduled_block_bucket",
        "model_uri",
        "pred_actual_block_time_sec",
        "pred_actual_block_time_p90_sec",
        "effective_actual_block_time_sec",
        "actual_block_time_sec",
        "abs_error_pred_block_time_sec",
        "abs_error_effective_block_time_sec",
        "covered_p90_pred_block_time",
    )
    .orderBy(F.desc("logged_at"))
    .limit(100)
)

print("rows with actuals for audit =", audit_source.count())

# COMMAND ----------

audit_schema = """
audit_key STRING,
leg_no BIGINT,
logged_at TIMESTAMP,
scored_at TIMESTAMP,
source_commit_version BIGINT,
source_commit_timestamp TIMESTAMP,
last_change_type STRING,
model_uri STRING,
prediction_status STRING,
is_operationally_active BOOLEAN,
dep_sched_dt TIMESTAMP,
arr_sched_dt TIMESTAMP,
dep_ap_sched STRING,
arr_ap_sched STRING,
ac_subtype STRING,
ac_registration STRING,
scheduled_block_time_sec DOUBLE,
scheduled_block_bucket STRING,
pred_actual_block_time_sec DOUBLE,
pred_actual_block_time_p90_sec DOUBLE,
pred_block_delay_sec DOUBLE,
effective_actual_block_time_sec DOUBLE,
effective_block_delay_sec DOUBLE,
model_pred_actual_block_time_sec_raw DOUBLE,
model_pred_block_delay_sec_raw DOUBLE,
missing_feature_count INT,
hours_to_departure_at_prediction DOUBLE,
horizon_bucket STRING,
actual_dep_dt TIMESTAMP,
actual_arr_dt TIMESTAMP,
actual_block_time_sec DOUBLE,
actual_block_delay_sec DOUBLE,
actual_available BOOLEAN,
error_pred_block_time_sec DOUBLE,
abs_error_pred_block_time_sec DOUBLE,
error_effective_block_time_sec DOUBLE,
abs_error_effective_block_time_sec DOUBLE,
covered_p90_pred_block_time BOOLEAN,
audit_loaded_at TIMESTAMP
"""

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {AUDIT_TABLE} (
  {audit_schema}
)
USING DELTA
""")

audit_cols_with_types = [
    ("audit_key", "STRING"),
    ("leg_no", "BIGINT"),
    ("logged_at", "TIMESTAMP"),
    ("scored_at", "TIMESTAMP"),
    ("source_commit_version", "BIGINT"),
    ("source_commit_timestamp", "TIMESTAMP"),
    ("last_change_type", "STRING"),
    ("model_uri", "STRING"),
    ("prediction_status", "STRING"),
    ("is_operationally_active", "BOOLEAN"),
    ("dep_sched_dt", "TIMESTAMP"),
    ("arr_sched_dt", "TIMESTAMP"),
    ("dep_ap_sched", "STRING"),
    ("arr_ap_sched", "STRING"),
    ("ac_subtype", "STRING"),
    ("ac_registration", "STRING"),
    ("scheduled_block_time_sec", "DOUBLE"),
    ("scheduled_block_bucket", "STRING"),
    ("pred_actual_block_time_sec", "DOUBLE"),
    ("pred_actual_block_time_p90_sec", "DOUBLE"),
    ("pred_block_delay_sec", "DOUBLE"),
    ("effective_actual_block_time_sec", "DOUBLE"),
    ("effective_block_delay_sec", "DOUBLE"),
    ("model_pred_actual_block_time_sec_raw", "DOUBLE"),
    ("model_pred_block_delay_sec_raw", "DOUBLE"),
    ("missing_feature_count", "INT"),
    ("hours_to_departure_at_prediction", "DOUBLE"),
    ("horizon_bucket", "STRING"),
    ("actual_dep_dt", "TIMESTAMP"),
    ("actual_arr_dt", "TIMESTAMP"),
    ("actual_block_time_sec", "DOUBLE"),
    ("actual_block_delay_sec", "DOUBLE"),
    ("actual_available", "BOOLEAN"),
    ("error_pred_block_time_sec", "DOUBLE"),
    ("abs_error_pred_block_time_sec", "DOUBLE"),
    ("error_effective_block_time_sec", "DOUBLE"),
    ("abs_error_effective_block_time_sec", "DOUBLE"),
    ("covered_p90_pred_block_time", "BOOLEAN"),
    ("audit_loaded_at", "TIMESTAMP"),
]

_ensure_table_columns(AUDIT_TABLE, audit_cols_with_types)

target = DeltaTable.forName(spark, AUDIT_TABLE)

audit_insert = audit_source.select(
    "audit_key",
    "leg_no",
    "logged_at",
    "scored_at",
    "source_commit_version",
    "source_commit_timestamp",
    "last_change_type",
    "model_uri",
    "prediction_status",
    "is_operationally_active",
    "dep_sched_dt",
    "arr_sched_dt",
    "dep_ap_sched",
    "arr_ap_sched",
    "ac_subtype",
    "ac_registration",
    "scheduled_block_time_sec",
    "scheduled_block_bucket",
    "pred_actual_block_time_sec",
    "pred_actual_block_time_p90_sec",
    "pred_block_delay_sec",
    "effective_actual_block_time_sec",
    "effective_block_delay_sec",
    "model_pred_actual_block_time_sec_raw",
    "model_pred_block_delay_sec_raw",
    "missing_feature_count",
    "hours_to_departure_at_prediction",
    "horizon_bucket",
    "actual_dep_dt",
    "actual_arr_dt",
    "actual_block_time_sec",
    "actual_block_delay_sec",
    "actual_available",
    "error_pred_block_time_sec",
    "abs_error_pred_block_time_sec",
    "error_effective_block_time_sec",
    "abs_error_effective_block_time_sec",
    "covered_p90_pred_block_time",
    "audit_loaded_at",
)

(
    target.alias("t")
    .merge(
        audit_insert.alias("s"),
        "t.audit_key = s.audit_key"
    )
    .whenNotMatchedInsertAll()
    .execute()
)

print("[OK] Audit table updated:", AUDIT_TABLE)

# COMMAND ----------

audit_df = spark.table(AUDIT_TABLE)

monitoring_schema = """
monitoring_key STRING,
monitoring_run_date DATE,
model_uri STRING,
prediction_status STRING,
horizon_bucket STRING,
scheduled_block_bucket STRING,
rows_cnt BIGINT,
mae_pred_block_time_sec DOUBLE,
mae_effective_block_time_sec DOUBLE,
p90_abs_error_effective_block_time_sec DOUBLE,
bias_effective_block_time_sec DOUBLE,
p90_coverage_pred_block_time_pct DOUBLE,
latest_scored_at TIMESTAMP,
latest_logged_at TIMESTAMP,
monitoring_loaded_at TIMESTAMP
"""

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {MONITORING_TABLE} (
  {monitoring_schema}
)
USING DELTA
""")

monitoring_cols_with_types = [
    ("monitoring_key", "STRING"),
    ("monitoring_run_date", "DATE"),
    ("model_uri", "STRING"),
    ("prediction_status", "STRING"),
    ("horizon_bucket", "STRING"),
    ("scheduled_block_bucket", "STRING"),
    ("rows_cnt", "BIGINT"),
    ("mae_pred_block_time_sec", "DOUBLE"),
    ("mae_effective_block_time_sec", "DOUBLE"),
    ("p90_abs_error_effective_block_time_sec", "DOUBLE"),
    ("bias_effective_block_time_sec", "DOUBLE"),
    ("p90_coverage_pred_block_time_pct", "DOUBLE"),
    ("latest_scored_at", "TIMESTAMP"),
    ("latest_logged_at", "TIMESTAMP"),
    ("monitoring_loaded_at", "TIMESTAMP"),
]

_ensure_table_columns(MONITORING_TABLE, monitoring_cols_with_types)

monitoring_snapshot = (
    audit_df
    .filter(F.col("actual_available") == True)
    .groupBy("model_uri", "prediction_status", "horizon_bucket", "scheduled_block_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("abs_error_pred_block_time_sec").alias("mae_pred_block_time_sec"),
        F.avg("abs_error_effective_block_time_sec").alias("mae_effective_block_time_sec"),
        F.expr("percentile_approx(abs_error_effective_block_time_sec, 0.9)").alias("p90_abs_error_effective_block_time_sec"),
        F.avg("error_effective_block_time_sec").alias("bias_effective_block_time_sec"),
        (
            F.avg(
                F.when(F.col("covered_p90_pred_block_time") == True, 1.0)
                 .when(F.col("covered_p90_pred_block_time") == False, 0.0)
            ) * 100.0
        ).alias("p90_coverage_pred_block_time_pct"),
        F.max("scored_at").alias("latest_scored_at"),
        F.max("logged_at").alias("latest_logged_at"),
    )
    .withColumn("monitoring_run_date", F.current_date())
    .withColumn("monitoring_loaded_at", F.current_timestamp())
    .withColumn(
        "monitoring_key",
        F.concat_ws(
            "||",
            F.date_format(F.current_date(), "yyyy-MM-dd"),
            F.coalesce(F.col("model_uri"), F.lit("NULL")),
            F.coalesce(F.col("prediction_status"), F.lit("NULL")),
            F.coalesce(F.col("horizon_bucket"), F.lit("NULL")),
            F.coalesce(F.col("scheduled_block_bucket"), F.lit("NULL")),
        )
    )
)

monitoring_target = DeltaTable.forName(spark, MONITORING_TABLE)

(
    monitoring_target.alias("t")
    .merge(
        monitoring_snapshot.alias("s"),
        "t.monitoring_key = s.monitoring_key"
    )
    .whenMatchedUpdateAll()
    .whenNotMatchedInsertAll()
    .execute()
)

print("[OK] Monitoring table updated:", MONITORING_TABLE)

print("\n=== STATUS x HORIZON ===")
display(
    audit_df.filter(F.col("actual_available") == True)
    .groupBy("prediction_status", "horizon_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("abs_error_pred_block_time_sec").alias("mae_pred_block_time_sec"),
        F.avg("abs_error_effective_block_time_sec").alias("mae_effective_block_time_sec"),
        F.expr("percentile_approx(abs_error_effective_block_time_sec, 0.9)").alias("p90_abs_error_effective_block_time_sec"),
        F.avg("error_effective_block_time_sec").alias("bias_effective_block_time_sec"),
        (
            F.avg(
                F.when(F.col("covered_p90_pred_block_time") == True, 1.0)
                 .when(F.col("covered_p90_pred_block_time") == False, 0.0)
            ) * 100.0
        ).alias("p90_coverage_pred_block_time_pct"),
    )
    .orderBy("prediction_status", "horizon_bucket")
)

print("\n=== MODEL URI ===")
display(
    audit_df.filter(F.col("actual_available") == True)
    .groupBy("model_uri")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("abs_error_effective_block_time_sec").alias("mae_effective_block_time_sec"),
        F.expr("percentile_approx(abs_error_effective_block_time_sec, 0.9)").alias("p90_abs_error_effective_block_time_sec"),
        F.avg("error_effective_block_time_sec").alias("bias_effective_block_time_sec"),
        (
            F.avg(
                F.when(F.col("covered_p90_pred_block_time") == True, 1.0)
                 .when(F.col("covered_p90_pred_block_time") == False, 0.0)
            ) * 100.0
        ).alias("p90_coverage_pred_block_time_pct"),
        F.max("logged_at").alias("latest_logged_at"),
    )
    .orderBy(F.desc("latest_logged_at"))
)

print("\n=== SCHEDULED BLOCK BUCKET ===")
display(
    audit_df.filter(F.col("actual_available") == True)
    .groupBy("scheduled_block_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("abs_error_effective_block_time_sec").alias("mae_effective_block_time_sec"),
        F.expr("percentile_approx(abs_error_effective_block_time_sec, 0.9)").alias("p90_abs_error_effective_block_time_sec"),
        F.avg("error_effective_block_time_sec").alias("bias_effective_block_time_sec"),
        (
            F.avg(
                F.when(F.col("covered_p90_pred_block_time") == True, 1.0)
                 .when(F.col("covered_p90_pred_block_time") == False, 0.0)
            ) * 100.0
        ).alias("p90_coverage_pred_block_time_pct"),
    )
    .orderBy("scheduled_block_bucket")
)

# COMMAND ----------

print("=== ROZKŁAD MISSING FEATURE COUNT DLA FALLBACKÓW ===")
display(
    sink_df.filter(F.col("prediction_status") == "TOO_MANY_MISSING_FEATURES_FALLBACK")
    .groupBy("missing_feature_count")
    .count()
    .orderBy("missing_feature_count")
)

print("\n=== MISSING FEATURE COUNT vs PREDICTION STATUS ===")
display(
    sink_df.groupBy(
        "prediction_status",
        F.when(F.col("missing_feature_count") <= 2, "0-2")
        .when(F.col("missing_feature_count") <= 10, "3-10")
        .when(F.col("missing_feature_count") <= 30, "11-30")
        .when(F.col("missing_feature_count") <= 60, "31-60")
        .otherwise("60+").alias("missing_bin")
    )
    .count()
    .orderBy("prediction_status", "missing_bin")
)

# COMMAND ----------

sink_with_actual = (
    sink_df
    .filter(F.col("is_operationally_active") == True)
    .join(labels_arr_df, on="leg_no", how="inner")
)

display(
    sink_with_actual.groupBy(
        F.when(F.col("missing_feature_count") <= 10, "0-10")
        .when(F.col("missing_feature_count") <= 20, "11-20")
        .when(F.col("missing_feature_count") <= 30, "21-30")
        .when(F.col("missing_feature_count") <= 40, "31-40")
        .otherwise("40+").alias("missing_bin")
    ).agg(
        F.count("*").alias("n"),
        F.avg(F.abs(
            F.coalesce(
                F.col("model_pred_actual_block_time_sec_raw"),
                F.col("effective_actual_block_time_sec")
            ) - F.col("actual_block_time_sec")
        )).alias("mae_sec"),
        F.avg(F.abs(
            F.col("scheduled_block_time_sec") - F.col("actual_block_time_sec")
        )).alias("schedule_mae_sec"),
    )
    .orderBy("missing_bin")
)