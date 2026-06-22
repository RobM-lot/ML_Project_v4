# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

from ml_project.registry import run_register_best

current_run_id = dbutils.widgets.get("RUN_ID").strip()

if not current_run_id:
    print("[WARN] RUN_ID jest pusty.")
    print("[WARN] Register wybierze najlepszy zgodny finished run z eksperymentu, a niekoniecznie świeżo wytrenowany run.")
    print("[WARN] Jeśli chcesz promować konkretny kandydat, podaj RUN_ID jawnie.")
else:
    print(f"[OK] RUN_ID ustawiony jawnie: {current_run_id}")

result = run_register_best(spark=spark, dbutils=dbutils, settings=SETTINGS)

print("[OK] Register decision:")
print(result)

# COMMAND ----------

display(result)