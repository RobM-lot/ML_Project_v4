# Runbook — training / scoring / rollback / recovery

## 1. Normalna kolejność pracy

### Ocena aktualnego modelu produkcyjnego
- uruchom `evaluate_manual`
- domyślnie notebook ocenia `@champion`

### Ocena świeżego kandydata przed rejestracją
- uruchom `weekly_training_manual`
- weź `run_id` z `01_train_compare_models`
- uruchom `evaluate_manual` z:
  - `EVAL_MODEL_SOURCE=run_id`
  - `EVAL_RUN_ID=<run_id_kandydata>`

### Rejestracja nowego modelu
- jeśli kandydat wygląda dobrze:
  - uruchom `register_manual`
  - podaj `RUN_ID=<run_id_kandydata>`

### Codzienny scoring
- uruchamiaj `daily_scoring_manual`

---

## 2. Co robi każdy job

### `daily_scoring_manual`
Kolejność:
1. `04_environment_check`
2. `01_cdf_stream` (CDF scoring stream z SHADOW_TABLE)
3. `06_prediction_audit_monitoring`

Użycie:
- standardowy scoring dzienny
- bez resetu checkpointu

### `weekly_training_manual`
Kolejność:
1. `04_environment_check`
2. `01_build_training_set` (buduje training dataset z Feature Store)
3. `01_train_compare_models` (trening 4 segmentów × p50/p90)
4. `05_quality_gate_check` (walidacja metryk, NIE promuje automatycznie)

Użycie:
- trening i quality gate
- bez automatycznej promocji modelu

### `evaluate_manual`
Użycie:
- audit championa
- albo audit konkretnego `run_id`

### `register_manual`
Użycie:
- świadoma rejestracja i promocja konkretnego runa

### `daily_scoring_reset_manual`
Użycie:
- tylko awaryjnie
- reset checkpointu + czysty replay scoringu dev

### `feature_coverage_diagnostics_manual`
Użycie:
- diagnostyka pokrycia Feature Store (route, stand)

---

## 3. Zasady operacyjne

### Zasada 1
Nie kasujemy ręcznie:
- checkpointu
- sinków
- events
- shadow
- runów MLflow
- modeli w registry

### Zasada 2
Reset scoringu robimy tylko przez:
- `daily_scoring_reset_manual`

### Zasada 3
Nowy model nie staje się championem sam z siebie.
Do tego służy tylko:
- `register_manual`

### Zasada 4
`evaluate_manual` domyślnie ocenia championa.
Kandydata oceniamy tylko wtedy, gdy jawnie podamy:
- `EVAL_MODEL_SOURCE=run_id`
- `EVAL_RUN_ID=<run_id>`

---

## 4. Kiedy użyć którego joba

### Chcę sprawdzić, jak działa obecny model
- `evaluate_manual`

### Chcę wytrenować nowego kandydata
- `weekly_training_manual`

### Chcę porównać nowego kandydata przed rejestracją
- `evaluate_manual` z `run_id`

### Chcę promować nowy model
- `register_manual`

### Chcę zrobić zwykły scoring
- `daily_scoring_manual`

### Stream / checkpoint się wysypał albo chcę czysty replay dev
- `daily_scoring_reset_manual`

---

## 5. Recovery / rollback

### Problem: trening przeszedł, ale kandydat jest słaby
- nie uruchamiaj `register_manual`
- zostaje obecny champion

### Problem: po rejestracji nowy model jest słaby
- cofnij alias `champion` na poprzednią wersję modelu

### Problem: scoring działa, ale wyniki wyglądają źle
- sprawdź `06_prediction_audit_monitoring`
- sprawdź statusy i fallbacki w sinku
- jeśli problem wynika z nowej wersji modelu:
  - wróć aliasem `champion` do poprzedniej wersji
  - uruchom ponownie zwykły `daily_scoring_manual`

### Problem: stream / checkpoint jest niespójny
- użyj `daily_scoring_reset_manual`

---

## 6. Minimalna checklista po ważnych runach

### Po `weekly_training_manual`
- zapisz `run_id`
- sprawdź metryki training runa
- sprawdź `05_quality_gate_check`

### Po `evaluate_manual`
- potwierdź, czy oceniany był:
  - `@champion`
  - czy konkretny `run_id`
- sprawdź MAE / bias / coverage / win rate

### Po `daily_scoring_manual`
- sprawdź `06_prediction_audit_monitoring`
- sprawdź status mix
- sprawdź fallback share
- sprawdź, czy sink i events się aktualizują

---

## 7. Architektura (stan po Iter2)

### Feature Store
- **DLT pipeline** (`ml_feature_store`, serverless): materializuje 9 tabel `ft_*` + 5 legacy `fs_*`
- **9 nowych tabel** ft_*: streaming (ft_leg_*) + materialized views (ft_*_daily_*, ft_airport_timezone)
- **On-demand features**: 15 UC Python UDF zarejestrowanych w `panda_silver_dev.ml_ops` (local_hour, sin/cos, haversine_km, is_eastbound)
- **FeatureLookup** (7): PIT lookup do daily stats + timezone po `event_date` / `dep_sched_dt`
- **FeatureFunction** (17): on-demand computation w `fe.score_batch` / `fe.create_training_set`

### Scoring (Plan A — fe.score_batch)
1. `sync_source_to_shadow()` — MERGE aktywnych lotów z Netline do SHADOW_TABLE
2. CDF stream (`availableNow=True`) → microbatch z lookup keys
3. `ensure_signature_columns(batch)` — koalesce nulls, type alignment
4. `fe.score_batch(model_uri, df=batch, result_type=out_schema)` — FS lookup + predict
5. Ekstrakcja predykcji z `prediction.pred_*` struct kolumny
6. Fallback/inactivation logic (cold start, too many missing, ARR state)
7. MERGE do SINK_TABLE + append do EVENTS_TABLE

### Model
- **Registry**: `panda_gold_dev.ml_ops.flight_delay_model`
- **Aktualny champion**: v9 (Iter1), v10 pending (Iter2)
- **Architektura**: `UltimateSegmentedModel` — 4 segmenty (TAXI-OUT, AIRBORNE, TAXI-IN, TOTAL-BLOCK) × p50/p90
- **Algorithm**: `HistGradientBoostingRegressor` (quantile loss, categorical_features="from_dtype")
- **CQR**: Conformal Quantile Regression shift per sched_block_time bucket

### Znane ograniczenia (do naprawy)
- `ft_leg_status/times/misc` mają event_ts = 1970 (fix w Iter2.5)
- Legacy `fs_*` tabele zachowane jako rollback path
- `common.py get_cleaned_flight_data` nadal joinuje 4 tabele (docelowo: fe.create_training_set)
