# TODO — Deferred from Tier 1

## I — SQL Alerts on monitoring metrics
- **Wymaga:** SQL Warehouse (Serverless lub Pro)
- **Co zrobić:**
  1. Utwórz 3 SQL Alerts w workspace:
     - `block_time_bias_alert`: `SELECT AVG(effective_block_delay_sec) as avg_bias FROM panda_gold_dev.ml_ops.block_time_predictions_v3 WHERE scored_at > current_timestamp() - INTERVAL 24 HOURS` — alert gdy |avg_bias| > 300 sec (5 min)
     - `cold_start_share_alert`: `SELECT COUNT_IF(prediction_status = 'COLD_START_FALLBACK') / COUNT(*) as cold_share FROM panda_gold_dev.ml_ops.block_time_predictions_v3 WHERE scored_at > current_timestamp() - INTERVAL 24 HOURS` — alert gdy > 0.15
     - `missing_features_alert`: `SELECT COUNT_IF(prediction_status = 'TOO_MANY_MISSING_FEATURES_FALLBACK') / COUNT(*) as missing_share FROM panda_gold_dev.ml_ops.block_time_predictions_v3 WHERE scored_at > current_timestamp() - INTERVAL 24 HOURS` — alert gdy > 0.05
  2. Schedule: co 15 minut
  3. Destination: email 30002818@lot.pl

## H2 — Fill prod schema placeholders in databricks.yml
- **Wymaga:** Wartości produkcyjnych schematów od DBA/infra team
- **Co zrobić:**
  1. Otwórz `databricks.yml`
  2. Zastąp `__TODO_PROD_SILVER_SCHEMA__` rzeczywistą nazwą (np. `panda_silver_prod.ml_ops`)
  3. Zastąp `__TODO_PROD_GOLD_SCHEMA__` rzeczywistą nazwą (np. `panda_gold_prod.ml_ops`)
  4. Uruchom `databricks bundle validate -t prod -p panda-prod` — RuntimeError powinien zniknąć
  5. Deploy: `databricks bundle deploy -t prod -p panda-prod`
