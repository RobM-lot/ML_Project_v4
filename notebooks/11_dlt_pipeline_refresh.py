# Databricks notebook source
# MAGIC %md
# MAGIC # 11 — DLT refresh + retrening v10 (runbook na dzień deployu)
# MAGIC
# MAGIC **Komórki to INSTRUKCJE do wykonania PO `bundle deploy`, w kolejności.** Nie odpalaj całego
# MAGIC notebooka naraz — każdy krok ma własną bramkę (DLT refresh ~30-60 min, trening ~1.5h).
# MAGIC
# MAGIC Kolejność: deploy → (1) rejestracja UDF → (2) pełny DLT refresh → (3) rejestracja ft_* w UC FS →
# MAGIC (4) trening v10 → (5) champion→v10 → (6) smoke test 08.
# MAGIC
# MAGIC Pipeline ID: `0107e154-3133-4ec5-a843-fdd499de7400` (dev).

# COMMAND ----------

# Krok 1 — Po deploy: zarejestruj on-demand UDF w UC (15 funkcji: 6 geo/sin_cos + 9 local time)
%run notebooks/10_register_on_demand_functions

# COMMAND ----------

# Krok 2 — Pełny DLT refresh (~30-60 min). Buduje 9 ft_* tabel od zera.
!databricks pipelines start-update 0107e154-3133-4ec5-a843-fdd499de7400 --full-refresh -p panda-dev

# COMMAND ----------

# Krok 3 — Po DLT refresh: zarejestruj nowe ft_* w UC Feature Store
%run notebooks/09_register_feature_tables

# COMMAND ----------

# Krok 4 — Trening v10 na ft_* + FeatureFunction (~1.5h)
!databricks bundle run weekly_training_manual -t dev -p panda-dev

# COMMAND ----------

# MAGIC %md
# MAGIC ## Kroki 5-6 (manualnie, po sukcesie treningu)
# MAGIC
# MAGIC 5. **Champion → v10**: ustaw alias `champion` na wersję v10 w Unity Catalog
# MAGIC    (`panda_gold_dev.ml_ops.flight_delay_model`).
# MAGIC 6. **Smoke test**: `%run notebooks/08_smoke_test_plan_a` → oczekiwane `ALL SMOKE TESTS PASS`
# MAGIC    (Check 1: 9 ft_* + DOUBLE + days_since; Check 3: 15 cech czasowych w signature;
# MAGIC    Check 4: score_batch z predykcjami — bez `_add_local_time_cosine_features`, parytet FF).
# MAGIC 7. Po zielonym smoke: `cdf_scoring` confirm (zapis do sink) → Faza 7 (parytet predykcji v9 vs v10).