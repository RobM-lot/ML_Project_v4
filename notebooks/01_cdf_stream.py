# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

from ml_project.scoring import ensure_cdf_stream_widgets, run_cdf_scoring

ensure_cdf_stream_widgets(dbutils)
print(" Widgets scoringowe gotowe: RUN_BOOTSTRAP, RESET_CHECKPOINT, STARTING_VERSION")


# COMMAND ----------

result = run_cdf_scoring(
    spark=spark,
    dbutils=dbutils,
    settings=SETTINGS,
)

print("[OK] Wynik run_cdf_scoring:")
print(result)