# Databricks notebook source
# KOMÓRKA 1 — Bootstrap
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

# Znajdź najnowszą wersję
versions = [int(mv.version) for mv in client.search_model_versions(f"name='{SETTINGS.UC_MODEL_NAME}'")]
latest = max(versions)
print(f"Najnowsza wersja: v{latest}")

# COMMAND ----------

# KOMÓRKA 2 — Ustaw champion
client.set_registered_model_alias(SETTINGS.UC_MODEL_NAME, "champion", str(latest))
print(f"champion → v{latest} ✅")
print(f"MODEL_URI: {SETTINGS.MODEL_URI}")

# COMMAND ----------

# MAGIC
# MAGIC %run "/Workspace/Users/30002818@lot.pl/.bundle/ML_Project_v3/dev/files/notebooks/08_smoke_test_plan_a"

# COMMAND ----------

# MAGIC
# MAGIC %run "/Workspace/Users/30002818@lot.pl/.bundle/ML_Project_v3/dev/files/notebooks/09_register_feature_tables"