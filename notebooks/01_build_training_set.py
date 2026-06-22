# Databricks notebook source
# MAGIC %run ./config

# COMMAND ----------

from ml_project.training import (
    build_training_datasets,
    materialize_training_datasets,
    expose_legacy_training_globals,
)

datasets = build_training_datasets(spark=spark, settings=SETTINGS)
written_tables = materialize_training_datasets(spark=spark, settings=SETTINGS, datasets=datasets, mode='overwrite')

expose_legacy_training_globals(globals(), datasets)

print('[OK] Zbiory gotowe i zapisane do tabel pośrednich:')
for name, table_name in written_tables.items():
    print(f' - {name}: {table_name}')

print(' Podsumowanie builda:')
print(datasets['summary'])

# COMMAND ----------

display(datasets['dq_summary'])
display(training_df_model.orderBy(F.desc('event_date')).limit(10))