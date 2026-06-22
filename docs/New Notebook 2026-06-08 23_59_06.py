# Databricks notebook source
# ── KOMÓRKA 1: Bootstrap ──────────────────────────────────────────────────
import sys, glob, json, dataclasses
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
print("✅ Bootstrap OK")

# COMMAND ----------

# ── KOMÓRKA 2: Weryfikacja typów FS po refresh ────────────────────────────
# Upewnij się że count/has_hist/stand_count są teraz DOUBLE przed rejestracją v8
for tbl in [
    "panda_silver_dev.ml_ops.fs_taxi_out_features",
    "panda_silver_dev.ml_ops.fs_airborne_features",
    "panda_silver_dev.ml_ops.fs_taxi_in_features",
    "panda_silver_dev.ml_ops.fs_stand_out_features",
    "panda_silver_dev.ml_ops.fs_stand_in_features",
]:
    bad = [(f.name, str(f.dataType)) for f in spark.table(tbl).schema.fields
           if "Long" in str(f.dataType) or "Integer" in str(f.dataType)]
    if bad:
        print(f"❌ {tbl.split('.')[-1]}: still INT/LONG → {bad}")
    else:
        print(f"✅ {tbl.split('.')[-1]}: wszystkie numeryczne = DOUBLE")

# COMMAND ----------

# ── KOMÓRKA 3: Rejestruj v8 z poprawioną signature ───────────────────────
from mlflow.models import ModelSignature

# Base kolumny gwarantowanie non-null → wracają do long/integer
REVERT_LONG = {"leg_no","leg_update_no","update_key","ac_logical_no","__START_AT",
               "scheduled_block_time_sec","taxi_out_sec","airborne_sec","taxi_in_sec",
               "actual_block_time_sec","arrival_delay_sec","block_delay_sec"}
REVERT_INT  = {"counter","fn_number","flight_tm","cycles","eet","ldo_offset","isLO",
               "netline_eet_duration_min","month","is_eastbound",
               "dep_utc_offset_min","arr_utc_offset_min",
               "local_hour_dep","local_dow_dep","local_hour_arr","local_dow_arr",
               "dq_missing_keys","dq_invalid_sched","dq_same_airport","dq_airport_mismatch",
               "dq_sequence_invalid","dq_invalid_actuals","dq_outlier_segments",
               "dq_segment_gap","dq_any_flag"}

logged_uri = client.get_model_version(SETTINGS.UC_MODEL_NAME, "7").source
sig_dict   = mlflow.models.get_model_info(logged_uri).signature.to_dict()
in_list    = json.loads(sig_dict["inputs"])
for col in in_list:
    n = col["name"]
    if n in REVERT_LONG: col["type"] = "long"
    elif n in REVERT_INT: col["type"] = "integer"
sig_dict["inputs"] = json.dumps(in_list)
mlflow.models.set_signature(logged_uri, ModelSignature.from_dict(sig_dict))

mv8 = mlflow.register_model(logged_uri, SETTINGS.UC_MODEL_NAME)
client.set_registered_model_alias(SETTINGS.UC_MODEL_NAME, "champion", mv8.version)
print(f"✅ v{mv8.version} zarejestrowany + champion → v{mv8.version}")
SETTINGS = dataclasses.replace(SETTINGS, MODEL_URI=f"models:/{SETTINGS.UC_MODEL_NAME}/{mv8.version}")

# COMMAND ----------

# ── KOMÓRKA 4: score_batch test ───────────────────────────────────────────
from datetime import date, timedelta
from pyspark.sql import functions as F
from pyspark.sql.types import ByteType, ShortType, IntegerType, StructType, StructField, DoubleType
from databricks.feature_engineering import FeatureEngineeringClient
from ml_project.common import get_cleaned_flight_data
from ml_project.training import _add_fs_lookup_keys

fe      = FeatureEngineeringClient()
recent  = (date.today() - timedelta(days=7)).isoformat()
batch   = _add_fs_lookup_keys(get_cleaned_flight_data(spark, recent, active_only=True)).limit(50)
for f in batch.schema.fields:
    if isinstance(f.dataType, (ByteType, ShortType)):
        batch = batch.withColumn(f.name, F.col(f.name).cast(IntegerType()))

_info      = mlflow.models.get_model_info(SETTINGS.MODEL_URI)
_out       = getattr(_info.signature.outputs, "inputs", _info.signature.outputs)
out_schema = StructType([StructField(s.name, DoubleType(), True) for s in _out])
print("batch rows:", batch.count())

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
    print(f"❌ {str(e)[:500]}")
    traceback.print_exc()