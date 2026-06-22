# Databricks notebook source
# MAGIC %run notebooks/10_register_on_demand_functions

# COMMAND ----------

!databricks pipelines start-update 0107e154-3133-4ec5-a843-fdd499de7400 --full-refresh -p panda-dev

# COMMAND ----------

# MAGIC %run notebooks/09_register_feature_tables

# COMMAND ----------

!databricks bundle run weekly_training_manual -t dev -p panda-dev
