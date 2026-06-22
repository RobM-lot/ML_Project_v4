# Databricks notebook source
from pathlib import Path
import importlib
import sys
import mlflow 

from pyspark.sql import functions as F
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


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

mp.configure_runtime(SETTINGS, spark=spark)

print("ENV:", SETTINGS.ENV)
print("MODEL_URI:", SETTINGS.MODEL_URI)
print("EVAL_ALL_DATASET_TABLE:", SETTINGS.EVAL_ALL_DATASET_TABLE)
print("MAX_MISSING_FEATURES:", SETTINGS.MAX_MISSING_FEATURES)

# COMMAND ----------

def dtype_name(t) -> str:
    return str(t).split(".")[-1].lower()

def mlflow_type_to_spark_dtype(type_name: str):
    t = (type_name or "double").lower()
    if "string" in t:
        return StringType()
    if "bool" in t:
        return BooleanType()
    if "int" in t and "long" not in t:
        return IntegerType()
    if "long" in t:
        return LongType()
    if "date" in t:
        return DateType()
    if "timestamp" in t or "datetime" in t:
        return TimestampType()
    return DoubleType()

def _schema_specs(schema_obj):
    if schema_obj is None:
        return []
    return getattr(schema_obj, "inputs", schema_obj)

def get_model_signature_io(model_uri: str):
    info = mlflow.models.get_model_info(model_uri)
    input_specs = _schema_specs(info.signature.inputs)
    output_specs = _schema_specs(info.signature.outputs)

    input_cols = [s.name for s in input_specs]
    input_types = {s.name: dtype_name(s.type) for s in input_specs}
    output_cols = [s.name for s in output_specs]
    output_types = {s.name: dtype_name(s.type) for s in output_specs}
    return input_cols, input_types, output_cols, output_types

INPUT_COLS, INPUT_TYPES, OUTPUT_COLS, OUTPUT_TYPES = get_model_signature_io(SETTINGS.MODEL_URI)

REQ_INPUT_COLS = [
    c for c in INPUT_COLS
    if not c.startswith("marker_") and "stand_" not in c
]

CORE_REQ_INPUT_COLS = [
    c for c in REQ_INPUT_COLS
    if (
        not (
            c.startswith("std_")
            or c.startswith("p90_")
            or c.startswith("min_")
            or c.startswith("max_")
            or c.startswith("trend_")
            or c.startswith("delta_ema_")
        )
    )
]

print("All input cols:", len(INPUT_COLS))
print("Current required input cols:", len(REQ_INPUT_COLS))
print("Core required input cols:", len(CORE_REQ_INPUT_COLS))

display(
    spark.createDataFrame(
        [(c,) for c in CORE_REQ_INPUT_COLS],
        ["core_required_input_col"]
    )
)

# COMMAND ----------

out_schema = StructType(
    [
        StructField(col_name, mlflow_type_to_spark_dtype(OUTPUT_TYPES.get(col_name, "double")), True)
        for col_name in OUTPUT_COLS
    ]
)

pred_udf = mlflow.pyfunc.spark_udf(spark, SETTINGS.MODEL_URI, result_type=out_schema)

def ensure_signature_columns_for_simulation(df):
    out = df

    for c in INPUT_COLS:
        t = INPUT_TYPES.get(c, "string").lower()
        if "string" in t:
            cast_type = "string"
        elif "bool" in t:
            cast_type = "boolean"
        elif "int" in t and "long" not in t:
            cast_type = "int"
        elif "long" in t:
            cast_type = "long"
        elif "float" in t or "double" in t:
            cast_type = "double"
        elif "date" in t:
            cast_type = "date"
        elif "timestamp" in t or "datetime" in t:
            cast_type = "timestamp"
        else:
            cast_type = "string"

        if c not in out.columns:
            out = out.withColumn(c, F.lit(None).cast(cast_type))

    for c in INPUT_COLS:
        t = INPUT_TYPES.get(c, "string").lower()
        if "string" in t:
            default_val, cast_type = F.lit("UNKNOWN"), "string"
        elif "bool" in t:
            default_val, cast_type = F.lit(False), "boolean"
        elif "int" in t and "long" not in t:
            default_val, cast_type = F.lit(0), "int"
        elif "long" in t:
            default_val, cast_type = F.lit(0), "long"
        elif "float" in t or "double" in t:
            default_val, cast_type = F.lit(None), "double"
        elif "date" in t:
            default_val, cast_type = F.to_date(F.lit("1970-01-01")), "date"
        elif "timestamp" in t or "datetime" in t:
            default_val, cast_type = F.to_timestamp(F.lit("1970-01-01 00:00:00")), "timestamp"
        else:
            default_val, cast_type = F.lit(None), "string"

        if c.startswith("marker_"):
            default_val = F.lit(None)
            cast_type = "double"

        out = out.withColumn(c, F.coalesce(F.col(c).cast(cast_type), default_val.cast(cast_type)))

    return out

def add_missing_count(df, cols, target_col):
    if cols:
        null_checks = [F.when(F.col(c).isNull(), 1).otherwise(0) for c in cols]
        expr = null_checks[0]
        for x in null_checks[1:]:
            expr = expr + x
        return df.withColumn(target_col, expr.cast("int"))
    return df.withColumn(target_col, F.lit(0).cast("int"))

# COMMAND ----------

eval_df = spark.table(SETTINGS.EVAL_ALL_DATASET_TABLE)

required_eval_cols = [
    "scheduled_block_time_sec",
    "actual_block_time_sec",
]

missing_required_eval_cols = [c for c in required_eval_cols if c not in eval_df.columns]
if missing_required_eval_cols:
    raise RuntimeError(f"Brak wymaganych kolumn w eval dataset: {missing_required_eval_cols}")

eval_df = add_missing_count(eval_df, REQ_INPUT_COLS, "missing_feature_count_current_all_req")
eval_df = add_missing_count(eval_df, CORE_REQ_INPUT_COLS, "missing_feature_count_core_req")

scoring_input_df = ensure_signature_columns_for_simulation(eval_df)

preds_struct = pred_udf(*[F.col(c) for c in INPUT_COLS])

scored_df = (
    scoring_input_df
    .withColumn("preds", preds_struct)
    .withColumn(
        "raw_pred_actual_block_time_sec",
        F.col("preds.pred_actual_block_time_sec").cast("double")
    )
    .withColumn(
        "schedule_abs_error_sec",
        F.abs(F.col("scheduled_block_time_sec") - F.col("actual_block_time_sec"))
    )
    .withColumn(
        "raw_model_abs_error_sec",
        F.abs(F.col("raw_pred_actual_block_time_sec") - F.col("actual_block_time_sec"))
    )
)

print("Rows in simulation dataset =", scored_df.count())

# COMMAND ----------

policy_specs = [
    ("current_all_req_t5", "missing_feature_count_current_all_req", 5),
    ("all_req_t10", "missing_feature_count_current_all_req", 10),
    ("all_req_t15", "missing_feature_count_current_all_req", 15),
    ("core_req_t5", "missing_feature_count_core_req", 5),
    ("core_req_t10", "missing_feature_count_core_req", 10),
]

policy_rows = []

for policy_name, missing_col, threshold in policy_specs:
    policy_df = (
        scored_df
        .withColumn("would_fallback", F.col(missing_col) > F.lit(threshold))
        .withColumn(
            "effective_pred_block_time_sec",
            F.when(F.col("would_fallback"), F.col("scheduled_block_time_sec"))
             .otherwise(F.col("raw_pred_actual_block_time_sec"))
        )
        .withColumn(
            "abs_error_effective_sec",
            F.abs(F.col("effective_pred_block_time_sec") - F.col("actual_block_time_sec"))
        )
        .withColumn(
            "error_effective_sec",
            F.col("effective_pred_block_time_sec") - F.col("actual_block_time_sec")
        )
        .withColumn(
            "beats_schedule",
            F.when(
                F.col("abs_error_effective_sec") < F.col("schedule_abs_error_sec"),
                F.lit(1.0)
            ).otherwise(F.lit(0.0))
        )
    )

    row = policy_df.agg(
        F.count("*").alias("rows_cnt"),
        F.avg(F.col("would_fallback").cast("double")).alias("fallback_share"),
        F.avg("abs_error_effective_sec").alias("mae_effective_sec"),
        F.avg("error_effective_sec").alias("bias_effective_sec"),
        F.expr("percentile_approx(abs_error_effective_sec, 0.9)").alias("p90_abs_error_effective_sec"),
        F.avg("beats_schedule").alias("win_rate_vs_schedule"),
    ).first()

    policy_rows.append(
        (
            policy_name,
            missing_col,
            threshold,
            int(row["rows_cnt"]),
            float(row["fallback_share"] or 0.0),
            float(row["mae_effective_sec"] or 0.0),
            float(row["bias_effective_sec"] or 0.0),
            float(row["p90_abs_error_effective_sec"] or 0.0),
            float(row["win_rate_vs_schedule"] or 0.0),
        )
    )

policy_results_df = spark.createDataFrame(
    policy_rows,
    [
        "policy_name",
        "missing_count_column",
        "threshold",
        "rows_cnt",
        "fallback_share",
        "mae_effective_sec",
        "bias_effective_sec",
        "p90_abs_error_effective_sec",
        "win_rate_vs_schedule",
    ],
)

display(policy_results_df.orderBy("policy_name"))

# COMMAND ----------

baseline_row = scored_df.agg(
    F.count("*").alias("rows_cnt"),
    F.avg("raw_model_abs_error_sec").alias("mae_raw_model_sec"),
    F.expr("percentile_approx(raw_model_abs_error_sec, 0.9)").alias("p90_raw_model_abs_error_sec"),
    F.avg(
        F.when(F.col("raw_model_abs_error_sec") < F.col("schedule_abs_error_sec"), 1.0).otherwise(0.0)
    ).alias("raw_model_win_rate_vs_schedule"),
    F.avg(
        F.col("raw_pred_actual_block_time_sec") - F.col("actual_block_time_sec")
    ).alias("raw_model_bias_sec"),
    F.avg("schedule_abs_error_sec").alias("mae_schedule_sec"),
).first()

baseline_df = spark.createDataFrame(
    [(
        int(baseline_row["rows_cnt"]),
        float(baseline_row["mae_raw_model_sec"] or 0.0),
        float(baseline_row["p90_raw_model_abs_error_sec"] or 0.0),
        float(baseline_row["raw_model_win_rate_vs_schedule"] or 0.0),
        float(baseline_row["raw_model_bias_sec"] or 0.0),
        float(baseline_row["mae_schedule_sec"] or 0.0),
    )],
    [
        "rows_cnt",
        "mae_raw_model_sec",
        "p90_raw_model_abs_error_sec",
        "raw_model_win_rate_vs_schedule",
        "raw_model_bias_sec",
        "mae_schedule_sec",
    ],
)

display(baseline_df)

# COMMAND ----------

bucketed_df = (
    scored_df
    .withColumn(
        "missing_bucket",
        F.when(F.col("missing_feature_count_current_all_req") <= 5, "<=5")
         .when(F.col("missing_feature_count_current_all_req") <= 10, "6-10")
         .when(F.col("missing_feature_count_current_all_req") <= 15, "11-15")
         .when(F.col("missing_feature_count_current_all_req") <= 25, "16-25")
         .when(F.col("missing_feature_count_current_all_req") <= 50, "26-50")
         .otherwise(">50")
    )
)

display(
    bucketed_df.groupBy("missing_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("raw_model_abs_error_sec").alias("mae_raw_model_sec"),
        F.avg("schedule_abs_error_sec").alias("mae_schedule_sec"),
        F.avg(
            F.when(F.col("raw_model_abs_error_sec") < F.col("schedule_abs_error_sec"), 1.0).otherwise(0.0)
        ).alias("raw_model_win_rate_vs_schedule"),
    )
    .orderBy("missing_bucket")
)

# COMMAND ----------

print("[OK] Offline simulation finished.")
print("Interpretacja:")
print("- szukamy polityki, która wyraźnie obniża fallback_share")
print("- ale nie pogarsza mae_effective / p90 / bias")
print("- i najlepiej utrzymuje albo poprawia win_rate_vs_schedule")