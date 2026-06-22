# Databricks notebook source
# ── KOMÓRKA 1: Bootstrap ──────────────────────────────────────────────────
import sys, glob, dataclasses
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

# Wróć champion → v7 (all-double signature, kompatybilny z pre-cast)
client.set_registered_model_alias(SETTINGS.UC_MODEL_NAME, "champion", "7")
print("champion → v7 ✅")
print(f"MODEL_URI: {SETTINGS.MODEL_URI}")

# COMMAND ----------

# ── KOMÓRKA 2: score_batch z pre-castem ──────────────────────────────────
from datetime import date, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import (LongType, IntegerType, ShortType, ByteType,
                                StructType, StructField, DoubleType)
from databricks.feature_engineering import FeatureEngineeringClient
from ml_project.common import get_cleaned_flight_data
from ml_project.training import _add_fs_lookup_keys

fe     = FeatureEngineeringClient()
recent = (date.today() - timedelta(days=7)).isoformat()
batch  = _add_fs_lookup_keys(get_cleaned_flight_data(spark, recent, active_only=True)).limit(50)

# PRE-CAST: wszystkie INT/LONG → DOUBLE przed score_batch
# (MLflow nie potrafi bezpiecznie konwertować int64→float64 = utrata precyzji dla dużych int)
_int_types = (LongType, IntegerType, ShortType, ByteType)
for f in batch.schema.fields:
    if isinstance(f.dataType, _int_types):
        batch = batch.withColumn(f.name, F.col(f.name).cast("double"))

print("batch rows:", batch.count())
print("sample dtypes after pre-cast:")
print([(f.name, str(f.dataType)) for f in batch.schema.fields if "Long" in str(f.dataType) or "Int" in str(f.dataType)][:5] or "✅ żadnych INT/LONG w batchu")

_info      = mlflow.models.get_model_info(SETTINGS.MODEL_URI)
_out       = getattr(_info.signature.outputs, "inputs", _info.signature.outputs)
out_schema = StructType([StructField(s.name, DoubleType(), True) for s in _out])
print("model resolved:", _info.model_uri)

try:
    pred = fe.score_batch(model_uri=SETTINGS.MODEL_URI, df=batch,
                          result_type=out_schema, env_manager="local")
    pred.select(
        "leg_no",
        F.col("prediction.pred_actual_block_time_sec"),
        F.col("prediction.pred_actual_block_time_p90_sec"),
    ).show(5, truncate=False)
    print(">>> PLAN A RDZEŃ DZIAŁA ✅")
except Exception as e:
    import traceback
    print(f"❌ {str(e)[:800]}")
    traceback.print_exc()