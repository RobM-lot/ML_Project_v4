# Databricks notebook source
import sys, glob
_hits = glob.glob("/Workspace/Users/30002818@lot.pl/.bundle/**/src/ml_project/settings.py", recursive=True)
SRC_PATH = [h for h in _hits if "/dev/" in h][0][:-len("/ml_project/settings.py")]
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
import mlflow, ml_project.settings as st, ml_project.common as cm
from mlflow import MlflowClient
mlflow.set_registry_uri("databricks-uc")
SETTINGS = st.load_settings("dev", project_root=SRC_PATH[:-len("/src")],
    source_catalog_override="panda_silver_prod", source_schema_override="occ_ops")
cm.configure_runtime(SETTINGS, spark=spark)
client = MlflowClient()
results = {}

# COMMAND ----------

ft_all_tables = [
    SETTINGS.FT_LEG_STATUS_TABLE, SETTINGS.FT_LEG_TIMES_TABLE, SETTINGS.FT_LEG_MISC_TABLE,
    SETTINGS.FT_AIRPORT_TIMEZONE_TABLE, SETTINGS.FT_ROUTE_DAILY_STATS_TABLE,
    SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE, SETTINGS.FT_AIRPORT_DAILY_TAXI_IN_TABLE,
    SETTINGS.FT_STAND_DAILY_OUT_TABLE, SETTINGS.FT_STAND_DAILY_IN_TABLE,
]
expected_double_cols = {
    SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE: ["count_dep_7d", "count_dep_30d", "has_hist_dep_7d", "has_hist_dep_30d", "days_since_last_event"],
    SETTINGS.FT_ROUTE_DAILY_STATS_TABLE:      ["count_route_7d", "count_route_30d", "has_hist_route_7d", "has_hist_route_30d", "days_since_last_event"],
    SETTINGS.FT_AIRPORT_DAILY_TAXI_IN_TABLE:  ["count_arr_7d", "count_arr_30d", "has_hist_arr_7d", "has_hist_arr_30d", "days_since_last_event"],
    SETTINGS.FT_STAND_DAILY_OUT_TABLE:        ["stand_count_out_7d", "stand_count_out_30d", "days_since_last_event"],
    SETTINGS.FT_STAND_DAILY_IN_TABLE:         ["stand_count_in_7d", "stand_count_in_30d", "days_since_last_event"],
}
ok = True
for full in ft_all_tables:
    try:
        spark.table(full).schema
    except Exception as e:
        print(f"❌ brak tabeli {full}: {str(e)[:120]}")
        ok = False
for full, cols in expected_double_cols.items():
    schema = {f.name: str(f.dataType) for f in spark.table(full).schema.fields}
    for c in cols:
        if schema.get(c) != "DoubleType()":
            print(f"❌ {full}.{c}: {schema.get(c)} (expected DoubleType)")
            ok = False
results["ft_tables_double"] = ok
print("✅ ft_* tables + DOUBLE + days_since_last_event OK" if ok else "❌ ft_* tables check FAILED")

# COMMAND ----------

try:
    mv = client.get_model_version_by_alias(SETTINGS.UC_MODEL_NAME, "champion")
    print(f"✅ champion → v{mv.version}")
    results["champion_alias"] = True
    CHAMPION_VERSION = mv.version
except Exception as e:
    print(f"❌ champion alias missing: {e}")
    results["champion_alias"] = False
    CHAMPION_VERSION = None

# COMMAND ----------

import json
if CHAMPION_VERSION:
    info = mlflow.models.get_model_info(f"models:/{SETTINGS.UC_MODEL_NAME}/{CHAMPION_VERSION}")
    in_list = json.loads(info.signature.to_dict()["inputs"])
    names = {c["name"] for c in in_list}
    time_cols = [
        "local_hour_dep", "local_dow_dep", "local_hour_arr", "local_dow_arr", "month",
        "sin_local_hour_dep", "cos_local_hour_dep", "sin_local_hour_arr", "cos_local_hour_arr",
        "sin_local_dow_dep", "cos_local_dow_dep", "sin_local_dow_arr", "cos_local_dow_arr",
        "sin_month", "cos_month",
    ]
    missing = [c for c in time_cols if c not in names]
    ok = len(missing) == 0
    results["signature_time_cols"] = ok
    print(f"✅ signature ma 15/15 cech czasowych z FF" if ok else f"❌ brak {len(missing)} w signature: {missing}")
else:
    results["signature_time_cols"] = False
    print("⏭️  pominięto (brak champion)")

# COMMAND ----------

from datetime import date, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, DoubleType
from databricks.feature_engineering import FeatureEngineeringClient
from ml_project.common import get_cleaned_flight_data
from ml_project.training import _add_fs_lookup_keys

if CHAMPION_VERSION:
    fe = FeatureEngineeringClient()
    recent = (date.today() - timedelta(days=7)).isoformat()
    batch = _add_fs_lookup_keys(
        get_cleaned_flight_data(spark, recent, active_only=True)
    ).limit(50)
    info = mlflow.models.get_model_info(SETTINGS.MODEL_URI)
    _out = getattr(info.signature.outputs, "inputs", info.signature.outputs)
    out_schema = StructType([StructField(s.name, DoubleType(), True) for s in _out])
    try:
        import json as _json
        _sig_types = {c["name"]: c["type"]
                      for c in _json.loads(info.signature.to_dict()["inputs"])}
        for _c, _t in _sig_types.items():
            if _c in batch.columns:
                if _t == "long":
                    batch = batch.withColumn(_c, F.coalesce(F.col(_c).cast("long"), F.lit(0).cast("long")))
                elif _t == "integer":
                    batch = batch.withColumn(_c, F.coalesce(F.col(_c).cast("int"), F.lit(0).cast("int")))
                elif _t == "double":
                    batch = batch.withColumn(_c, F.col(_c).cast("double"))
        pred = fe.score_batch(model_uri=SETTINGS.MODEL_URI, df=batch,
                              result_type=out_schema, env_manager="local")
        rows = pred.select("leg_no",
                           F.col("prediction.pred_actual_block_time_sec").alias("pred"),
                           F.col("prediction.pred_actual_block_time_p90_sec").alias("pred_p90")
                          ).limit(5).collect()
        non_null_preds = sum(1 for r in rows if r["pred"] is not None)
        ok = non_null_preds >= 3
        results["score_batch_works"] = ok
        print(f"✅ score_batch — {non_null_preds}/5 wierszy ma predykcje" if ok
              else f"❌ score_batch — tylko {non_null_preds}/5 z predykcjami")
    except Exception as e:
        results["score_batch_works"] = False
        print(f"❌ score_batch FAIL: {str(e)[:300]}")
else:
    results["score_batch_works"] = False

# COMMAND ----------

print("\n" + "="*60)
print("SMOKE TEST SUMMARY")
print("="*60)
passed = sum(results.values())
total  = len(results)
for k, v in results.items():
    print(f"  {'✅' if v else '❌'} {k}")
print(f"\n{passed}/{total} checks passed")
if passed == total:
    print("\n>>> ALL SMOKE TESTS PASS — Plan A ready for production ✅")
else:
    print("\n>>> SMOKE TESTS FAILED — investigate before merge ❌")
