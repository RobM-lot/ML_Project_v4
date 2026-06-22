# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

from ml_project.training import run_train_compare_models

result = run_train_compare_models(spark=spark, settings=SETTINGS)
 
print("[OK] Wynik run_train_compare_models:")
print(result)

try:
    dbutils.jobs.taskValues.set(key="run_id", value=str(result["run_id"]))
    print(f"[OK] taskValue run_id ustawiony: {result['run_id']}")
except Exception as e:
    print(f"[INFO] Nie udało się ustawić taskValue run_id (np. uruchomienie poza Job): {e}")