# Databricks notebook source
from pathlib import Path
import importlib
import sys
import mlflow
from pyspark.sql import functions as F, Window as W


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
print("SINK_TABLE:", SETTINGS.SINK_TABLE)
print("FT_AIRPORT_DAILY_TAXI_OUT_TABLE:", SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE)
print("FT_ROUTE_DAILY_STATS_TABLE:", SETTINGS.FT_ROUTE_DAILY_STATS_TABLE)
print("FT_AIRPORT_DAILY_TAXI_IN_TABLE:", SETTINGS.FT_AIRPORT_DAILY_TAXI_IN_TABLE)

# COMMAND ----------

def _schema_specs(schema_obj):
    if schema_obj is None:
        return []
    return getattr(schema_obj, "inputs", schema_obj)

info = mlflow.models.get_model_info(SETTINGS.MODEL_URI)
input_specs = _schema_specs(info.signature.inputs)

INPUT_COLS = [s.name for s in input_specs]
INPUT_TYPES = {s.name: str(s.type).split(".")[-1].lower() for s in input_specs}

REQ_INPUT_COLS = [
    c for c in INPUT_COLS
    if not c.startswith("marker_") and "stand_" not in c
]

ROUTE_GROUP_PATTERNS = [
    ("avg_", "avg"),
    ("std_", "std"),
    ("p90_", "p90"),
    ("min_", "min"),
    ("max_", "max"),
    ("trend_", "trend"),
    ("ema_", "ema"),
    ("delta_ema_", "delta_ema"),
    ("count_", "count"),
    ("has_hist_", "has_hist"),
]

ROUTE_FEATURE_COLS = [
    c for c in REQ_INPUT_COLS
    if any(c.startswith(prefix) for prefix, _ in ROUTE_GROUP_PATTERNS)
]

BASE_NON_ROUTE_REQ_COLS = [c for c in REQ_INPUT_COLS if c not in ROUTE_FEATURE_COLS]

print("All input cols:", len(INPUT_COLS))
print("Required input cols used in missing_feature_count:", len(REQ_INPUT_COLS))
print("Route/history feature cols inside required inputs:", len(ROUTE_FEATURE_COLS))
print("Non-route required cols:", len(BASE_NON_ROUTE_REQ_COLS))
print("MAX_MISSING_FEATURES:", SETTINGS.MAX_MISSING_FEATURES)
print("Fallback threshold: missing_feature_count >", SETTINGS.MAX_MISSING_FEATURES)

display(
    spark.createDataFrame(
        [(c, INPUT_TYPES.get(c, "unknown")) for c in ROUTE_FEATURE_COLS],
        ["route_feature_col", "input_type"]
    )
)

# COMMAND ----------

LOOKBACK_DAYS = int(getattr(SETTINGS, "SHADOW_SYNC_LOOKBACK_DAYS", 2))
LOOKAHEAD_MONTHS = int(getattr(SETTINGS, "SHADOW_SYNC_LOOKAHEAD_MONTHS", 3))
MODEL_AIRCRAFT_FEATURE_COL = getattr(SETTINGS, "MODEL_AIRCRAFT_FEATURE_COL", "ac_registration")
AC_REGISTRATION_PREFIX_LEN = int(getattr(SETTINGS, "AC_REGISTRATION_PREFIX_LEN", 4) or 0)

sink_df = spark.table(SETTINGS.SINK_TABLE).select(
    "leg_no",
    "prediction_status",
    "missing_feature_count",
    "dep_sched_dt",
    "arr_sched_dt",
    "scored_at",
    "source_commit_version"
)

source_df = spark.table(SETTINGS.LABELS_TABLE)
if "__END_AT" in source_df.columns:
    source_df = source_df.filter(F.col("__END_AT").isNull())

source_df = (
    source_df
    .filter(F.col("counter") == 0)
    .filter(F.col("dep_sched_dt") >= F.expr(f"current_timestamp() - INTERVAL {LOOKBACK_DAYS} DAYS"))
    .filter(F.col("dep_sched_dt") <= F.expr(f"current_timestamp() + INTERVAL {LOOKAHEAD_MONTHS} MONTHS"))
)

base = (
    source_df
    .join(sink_df.select("leg_no").distinct(), on="leg_no", how="inner")
)

if MODEL_AIRCRAFT_FEATURE_COL == "ac_registration" and "ac_registration" in base.columns and AC_REGISTRATION_PREFIX_LEN > 0:
    base = base.withColumn(
        "ac_registration",
        F.when(F.col("ac_registration").isNull(), F.lit(None).cast("string"))
         .otherwise(F.substring(F.col("ac_registration"), 1, AC_REGISTRATION_PREFIX_LEN))
    )

base = mp.enrich_with_local_context(base, spark)

base = (
    base.withColumn("event_ts", F.col("dep_sched_dt").cast("timestamp"))
        .withColumn("event_date", F.to_date("dep_sched_dt"))
        .withColumn(
            "scheduled_block_time_sec",
            (F.col("arr_sched_dt").cast("long") - F.col("dep_sched_dt").cast("long")).cast("double"),
        )
        .withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
)

for i in range(1, SETTINGS.MAX_MARKER_LENGTH + 1):
    base = base.withColumn(
        f"marker_{i}",
        F.when(
            F.length(F.col("marker")) >= i,
            F.when(F.substring(F.col("marker"), i, 1) == "Y", 0)
             .when(F.substring(F.col("marker"), i, 1) == "N", 1)
             .otherwise(F.lit(None).cast("double")),
        ).otherwise(F.lit(None).cast("double")),
    )

leg_misc_raw = spark.table(SETTINGS.LEG_MISC_TABLE)
if "__END_AT" in leg_misc_raw.columns:
    leg_misc_raw = leg_misc_raw.filter(F.col("__END_AT").isNull())

if "update_key" in leg_misc_raw.columns:
    w_misc = W.partitionBy("leg_no").orderBy(F.col("update_key").desc())
else:
    w_misc = W.partitionBy("leg_no").orderBy(F.col("entry_dt").desc())

leg_misc_latest = (
    leg_misc_raw.withColumn("rn", F.row_number().over(w_misc))
    .filter(F.col("rn") == 1)
    .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
    .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
    .select("leg_no", "dep_stand", "arr_stand")
)

base = (
    base.join(leg_misc_latest, on="leg_no", how="left")
        .withColumn("fs_dep_ap_sched", F.col("dep_ap_sched"))
        .withColumn("fs_arr_ap_sched", F.col("arr_ap_sched"))
        .withColumn("fs_event_date", F.col("event_date"))
        .withColumn("_row_id", F.monotonically_increasing_id())
)

max_fs_row = spark.table(SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE).select(F.max("event_date").alias("max_event_date")).first()
max_fs_date = max_fs_row["max_event_date"] if max_fs_row and max_fs_row["max_event_date"] else None

if max_fs_date is not None:
    base = base.withColumn("fs_lookup_date", F.least(F.col("fs_event_date"), F.lit(max_fs_date)))
else:
    base = base.withColumn("fs_lookup_date", F.col("fs_event_date"))

print("Rows in diagnostic base:", base.count())
print("Max route FS event_date:", max_fs_date)

# COMMAND ----------

fs_out = spark.table(SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE)
fs_air = spark.table(SETTINGS.FT_ROUTE_DAILY_STATS_TABLE)
fs_in = spark.table(SETTINGS.FT_AIRPORT_DAILY_TAXI_IN_TABLE)

def _route_fs_value_cols(fs_df, join_cols, time_key):
    base_exclusions = set(join_cols) | {time_key}
    return [c for c in fs_df.columns if c not in base_exclusions]

def join_fs_exact_with_diagnostics(base_df, fs_df, join_cols, time_key, prefix):
    fs_value_cols = _route_fs_value_cols(fs_df, join_cols, time_key)

    fs_renamed = fs_df.withColumnRenamed(time_key, "_join_fs_date")
    for c in join_cols:
        fs_renamed = fs_renamed.withColumnRenamed(c, f"{c}_right")

    base_cols = set(base_df.columns)
    safe_fs_cols = [
        c for c in fs_renamed.columns
        if c == "_join_fs_date" or c.endswith("_right") or c not in base_cols
    ]
    fs_renamed = fs_renamed.select(*safe_fs_cols)

    conds = [F.col(c) == F.col(f"{c}_right") for c in join_cols]
    conds.append(F.col("_join_fs_date") == F.col("fs_lookup_date"))

    join_expr = conds[0]
    for cond in conds[1:]:
        join_expr = join_expr & cond

    joined = base_df.join(fs_renamed, on=join_expr, how="left")

    row_exists_col = f"{prefix}_row_exists"
    nonnull_hit_col = f"{prefix}_nonnull_feature_hit"
    nonnull_count_col = f"{prefix}_nonnull_feature_count"

    joined = joined.withColumn(
        row_exists_col,
        F.when(F.col("_join_fs_date").isNotNull(), F.lit(1)).otherwise(F.lit(0))
    )

    fs_value_cols_present = [c for c in fs_value_cols if c in joined.columns]
    if fs_value_cols_present:
        nonnull_expr = None
        for c in fs_value_cols_present:
            expr = F.when(F.col(c).isNotNull(), 1).otherwise(0)
            nonnull_expr = expr if nonnull_expr is None else (nonnull_expr + expr)

        joined = joined.withColumn(nonnull_count_col, nonnull_expr.cast("int"))
        joined = joined.withColumn(
            nonnull_hit_col,
            F.when(F.col(nonnull_count_col) > 0, F.lit(1)).otherwise(F.lit(0))
        )
    else:
        joined = joined.withColumn(nonnull_count_col, F.lit(0).cast("int"))
        joined = joined.withColumn(nonnull_hit_col, F.lit(0))

    return joined.drop("_join_fs_date", *[f"{c}_right" for c in join_cols])

joined = join_fs_exact_with_diagnostics(base, fs_out, list(SETTINGS.PK_TAXI_OUT), "event_date", "route_fs_out")
joined = join_fs_exact_with_diagnostics(joined, fs_air, list(SETTINGS.PK_AIRBORNE), "event_date", "route_fs_air")
joined = join_fs_exact_with_diagnostics(joined, fs_in, list(SETTINGS.PK_TAXI_IN), "event_date", "route_fs_in")

analysis_df = joined.join(
    sink_df,
    on=["leg_no", "dep_sched_dt", "arr_sched_dt"],
    how="inner"
)

analysis_df = analysis_df.withColumn(
    "hours_to_departure",
    (F.col("dep_sched_dt").cast("long") - F.col("scored_at").cast("long")) / F.lit(3600.0)
).withColumn(
    "horizon_bucket",
    F.when(F.col("hours_to_departure") < 6, "<6h")
     .when(F.col("hours_to_departure") < 24, "6-24h")
     .when(F.col("hours_to_departure") < 72, "24-72h")
     .when(F.col("hours_to_departure") < 168, "3-7d")
     .otherwise(">7d")
)

print("Rows for analysis:", analysis_df.count())

# COMMAND ----------

display(
    analysis_df.groupBy("prediction_status")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("missing_feature_count").alias("avg_missing_feature_count"),
        F.avg("route_fs_out_row_exists").alias("share_route_fs_out_row_exists"),
        F.avg("route_fs_air_row_exists").alias("share_route_fs_air_row_exists"),
        F.avg("route_fs_in_row_exists").alias("share_route_fs_in_row_exists"),
        F.avg("route_fs_out_nonnull_feature_hit").alias("share_route_fs_out_nonnull_feature_hit"),
        F.avg("route_fs_air_nonnull_feature_hit").alias("share_route_fs_air_nonnull_feature_hit"),
        F.avg("route_fs_in_nonnull_feature_hit").alias("share_route_fs_in_nonnull_feature_hit"),
        F.avg("route_fs_out_nonnull_feature_count").alias("avg_route_fs_out_nonnull_feature_count"),
        F.avg("route_fs_air_nonnull_feature_count").alias("avg_route_fs_air_nonnull_feature_count"),
        F.avg("route_fs_in_nonnull_feature_count").alias("avg_route_fs_in_nonnull_feature_count"),
    )
    .orderBy(F.desc("rows_cnt"))
)

# COMMAND ----------

display(
    analysis_df.groupBy("prediction_status", "horizon_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("missing_feature_count").alias("avg_missing_feature_count"),
        F.avg("route_fs_out_row_exists").alias("share_route_fs_out_row_exists"),
        F.avg("route_fs_air_row_exists").alias("share_route_fs_air_row_exists"),
        F.avg("route_fs_in_row_exists").alias("share_route_fs_in_row_exists"),
        F.avg("route_fs_out_nonnull_feature_hit").alias("share_route_fs_out_nonnull_feature_hit"),
        F.avg("route_fs_air_nonnull_feature_hit").alias("share_route_fs_air_nonnull_feature_hit"),
        F.avg("route_fs_in_nonnull_feature_hit").alias("share_route_fs_in_nonnull_feature_hit"),
    )
    .orderBy("prediction_status", "horizon_bucket")
)

# COMMAND ----------

fallback_df = analysis_df.filter(F.col("prediction_status") == "TOO_MANY_MISSING_FEATURES_FALLBACK")
model_ok_df = analysis_df.filter(F.col("prediction_status") == "MODEL_OK")

def compute_null_share_df(df, cols, label):
    exprs = []
    present_cols = [c for c in cols if c in df.columns]
    for c in present_cols:
        exprs.append(F.avg(F.when(F.col(c).isNull(), 1.0).otherwise(0.0)).alias(c))

    if not exprs:
        return spark.createDataFrame([], "feature_name string, null_share double, sample string")

    row = df.select(*exprs).first().asDict()
    rows = [(k, float(v or 0.0), label) for k, v in row.items()]
    return spark.createDataFrame(rows, ["feature_name", "null_share", "sample"])

fallback_nulls_df = compute_null_share_df(fallback_df, ROUTE_FEATURE_COLS, "fallback")
model_ok_nulls_df = compute_null_share_df(model_ok_df, ROUTE_FEATURE_COLS, "model_ok")

route_nulls_compare_df = (
    fallback_nulls_df.alias("f")
    .join(model_ok_nulls_df.alias("m"), on="feature_name", how="outer")
    .select(
        F.col("feature_name"),
        F.col("f.null_share").alias("fallback_null_share"),
        F.col("m.null_share").alias("model_ok_null_share"),
        (
            F.coalesce(F.col("f.null_share"), F.lit(0.0)) -
            F.coalesce(F.col("m.null_share"), F.lit(0.0))
        ).alias("fallback_minus_model_ok")
    )
)

display(route_nulls_compare_df.orderBy(F.desc("fallback_minus_model_ok"), F.desc("fallback_null_share")))

# COMMAND ----------

group_rows = []

for prefix, label in ROUTE_GROUP_PATTERNS:
    cols = [c for c in ROUTE_FEATURE_COLS if c.startswith(prefix) and c in analysis_df.columns]
    if not cols:
        continue

    fb_exprs = [F.avg(F.when(F.col(c).isNull(), 1.0).otherwise(0.0)) for c in cols]
    mo_exprs = [F.avg(F.when(F.col(c).isNull(), 1.0).otherwise(0.0)) for c in cols]

    fb_row = fallback_df.select(*fb_exprs).first()
    mo_row = model_ok_df.select(*mo_exprs).first()

    fb_vals = [float(x or 0.0) for x in fb_row]
    mo_vals = [float(x or 0.0) for x in mo_row]

    group_rows.append(
        (
            label,
            len(cols),
            sum(fb_vals) / len(fb_vals),
            sum(mo_vals) / len(mo_vals),
            (sum(fb_vals) / len(fb_vals)) - (sum(mo_vals) / len(mo_vals)),
        )
    )

group_sdf = spark.createDataFrame(
    group_rows,
    ["feature_group", "feature_count", "fallback_avg_null_share", "model_ok_avg_null_share", "gap_vs_model_ok"]
)

display(group_sdf.orderBy(F.desc("gap_vs_model_ok"), F.desc("fallback_avg_null_share")))

# COMMAND ----------

display(
    fallback_df.groupBy("horizon_bucket")
    .agg(
        F.count("*").alias("rows_cnt"),
        F.avg("missing_feature_count").alias("avg_missing_feature_count"),
        F.avg("route_fs_out_row_exists").alias("share_route_fs_out_row_exists"),
        F.avg("route_fs_air_row_exists").alias("share_route_fs_air_row_exists"),
        F.avg("route_fs_in_row_exists").alias("share_route_fs_in_row_exists"),
        F.avg("route_fs_out_nonnull_feature_hit").alias("share_route_fs_out_nonnull_feature_hit"),
        F.avg("route_fs_air_nonnull_feature_hit").alias("share_route_fs_air_nonnull_feature_hit"), 
        F.avg("route_fs_in_nonnull_feature_hit").alias("share_route_fs_in_nonnull_feature_hit"),
        F.avg("route_fs_out_nonnull_feature_count").alias("avg_route_fs_out_nonnull_feature_count"),
        F.avg("route_fs_air_nonnull_feature_count").alias("avg_route_fs_air_nonnull_feature_count"),
        F.avg("route_fs_in_nonnull_feature_count").alias("avg_route_fs_in_nonnull_feature_count"),
    )
    .orderBy("horizon_bucket")
)

# COMMAND ----------

top_gap_features = [
    r["feature_name"]
    for r in route_nulls_compare_df.orderBy(F.desc("fallback_minus_model_ok"), F.desc("fallback_null_share")).limit(30).collect()
]

print("Top route features with biggest null-share gap in fallback vs MODEL_OK:")
for c in top_gap_features:
    print("-", c)

# COMMAND ----------

summary = {
    "analysis_rows": analysis_df.count(),
    "fallback_rows": fallback_df.count(),
    "model_ok_rows": model_ok_df.count(),
    "max_missing_features_threshold": int(SETTINGS.MAX_MISSING_FEATURES),
    "route_feature_cols_in_required_inputs": len([c for c in ROUTE_FEATURE_COLS if c in analysis_df.columns]),
}

print(summary)
