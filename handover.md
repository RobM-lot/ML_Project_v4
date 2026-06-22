# LOT ML Pipeline — Feature Store Migration: Kompletny status projektu

*Wygenerowany: 2026-06-10 wieczór. Przeznaczony do przeniesienia do nowego chatu.*
*Ten dokument zastępuje wszystkie poprzednie statusy i jest kompletnym źródłem prawdy.*

---

## SEKCJA 1: Środowisko i infrastruktura

### Dane środowiska

| Parametr                | Wartość                                                                             |
| ----------------------- | ----------------------------------------------------------------------------------- |
| Platforma               | Databricks / Azure (West EU)                                                        |
| Workspace (dev)         | https://adb-5711108594958773.13.azuredatabricks.net                                 |
| Workspace (prod)        | https://adb-4923213703602775.15.azuredatabricks.net                                 |
| Klaster                 | Robert's Cluster — `0204-144038-zpix5qya`                                           |
| Klaster spec            | Driver: Standard_F4s (4GB RAM), Workers: Standard_F4s 1-2, DBR 17.3.13 ML           |
| Bundle name             | ML_Project_v3, profil `panda-dev`, target `dev`                                     |
| DLT Pipeline ID         | `0107e154-3133-4ec5-a843-fdd499de7400`                                              |
| UC Model registry       | `panda_gold_dev.ml_ops.flight_delay_model`                                          |
| Model alias             | `champion` → v9 (v10 czeka na trening)                                              |
| Silver catalog          | `panda_silver_dev.ml_ops`                                                           |
| Gold catalog            | `panda_gold_dev.ml_ops`                                                             |
| Source catalog          | `panda_silver_prod.occ_ops`                                                         |
| Checkpoint (CDF stream) | `dbfs:/pipelines/flight_delay_cdf_dev/checkpoints`                                  |
| MLflow experiment       | `/Users/30002818@lot.pl/ML_Project_v3/Experiments/block_time_auto_segmented_dev_v4` |
| GCP project             | `jobswipe-production` (niezmienione, inne narzędzie)                                |

### Tabele w bazach danych

```
panda_silver_dev.ml_ops:
  -- Legacy Iter1 (aktywne, rollback path):
  fs_taxi_out_features          PK: dep_ap_sched, event_date TIMESERIES
  fs_airborne_features          PK: route_id, event_date TIMESERIES
  fs_taxi_in_features           PK: arr_ap_sched, event_date TIMESERIES
  fs_stand_out_features         PK: stand_id, event_date TIMESERIES
  fs_stand_in_features          PK: stand_id, event_date TIMESERIES

  -- Nowe Iter2 (aktywne, event_ts częściowo złe):
  ft_leg_status                 PK: leg_no, event_ts TIMESERIES  ⚠️ event_ts = 1970
  ft_leg_times                  PK: leg_no, event_ts TIMESERIES  ⚠️ event_ts = 1970
  ft_leg_misc                   PK: leg_no, event_ts TIMESERIES  ⚠️ event_ts = 1970
  ft_airport_timezone           PK: iata_ap_code, valid_ts TIMESERIES  ✅ poprawne
  ft_route_daily_stats          PK: route_id, event_date TIMESERIES  ✅
  ft_airport_daily_taxi_out     PK: dep_ap_sched, event_date TIMESERIES  ✅
  ft_airport_daily_taxi_in      PK: arr_ap_sched, event_date TIMESERIES  ✅
  ft_stand_daily_out            PK: stand_id, event_date TIMESERIES  ✅
  ft_stand_daily_in             PK: stand_id, event_date TIMESERIES  ✅

  -- Tabele operacyjne:
  netline_leg_shadow_cdf_v3     SHADOW_TABLE — aktywne loty do scoringu
  block_time_training_dataset_v1 — cache training set

panda_gold_dev.ml_ops:
  block_time_predictions_v3     SINK_TABLE — wyniki predykcji
  block_time_predictions_events_v3 — audit log zdarzeń scoringu
```

### Struktura SHADOW_TABLE (tabela aktywnych lotów dla scoringu)

SHADOW_TABLE `netline_leg_shadow_cdf_v3` to **wąska tabela** synchronizowana z `LABELS_TABLE`.
Zawiera tylko kolumny niezbędne do scoringu:

```sql
leg_no          BIGINT   -- PK
leg_state       STRING
leg_type        STRING
dep_sched_dt    TIMESTAMP
arr_sched_dt    TIMESTAMP
dep_ap_sched    STRING
arr_ap_sched    STRING
dep_ap_actual   STRING
arr_ap_actual   STRING
ac_subtype      STRING
ac_registration STRING
ac_owner        STRING
marker          STRING
row_hash        BIGINT
__END_AT        TIMESTAMP
```

Filtr przy sync: `counter == 0`, window -LOOKBACK_DAYS do +LOOKAHEAD_MONTHS.
**Kluczowe:** SHADOW_TABLE NIE ma wielu kolumn które ma training DataFrame (brak np. `taxi_out_sec`, `airborne_sec`, `leg_update_no`, `fn_number`, etc.) — to jest źródło problemów z `ensure_signature_columns`.

### Struktura SINK_TABLE (predykcje wychodzące)

`block_time_predictions_v3` — kolumny wyjściowe:

```sql
leg_no                          BIGINT
pred_taxi_out_sec               DOUBLE  -- p50 taxi-out
pred_airborne_sec               DOUBLE  -- p50 airborne
pred_taxi_in_sec                DOUBLE  -- p50 taxi-in
pred_actual_block_time_sec      DOUBLE  -- p50 całkowity block time
pred_taxi_out_p90_sec           DOUBLE  -- p90 taxi-out
pred_airborne_p90_sec           DOUBLE  -- p90 airborne
pred_taxi_in_p90_sec            DOUBLE  -- p90 taxi-in
pred_actual_block_time_p90_sec  DOUBLE  -- p90 całkowity block time
pred_block_delay_sec            DOUBLE  -- opóźnienie = pred_actual - scheduled
model_pred_actual_block_time_sec_raw DOUBLE  -- surowa predykcja przed CQR
model_pred_block_delay_sec_raw  DOUBLE
source_commit_version           BIGINT
source_commit_timestamp         TIMESTAMP
is_active                       BOOLEAN -- czy lot nadal aktywny
inactive_reason                 STRING  -- dlaczego nieaktywny (ARR_STATE, DELETED, etc.)
is_operationally_active         BOOLEAN
```

### Jobs w Databricks

**`weekly_training_manual`** — 4 taski:

1. `environment_contract_check` (~3 min) — sprawdzenie wersji bibliotek, UC dostępność
2. `build_training_set` (~10 min) — budowanie training dataset z FS
3. `train_compare_models` (~1h 20min) — właściwy trening 4 segmentów ×2 CV params ×3 folds
4. `quality_gate_check` (~25 sek) — walidacja metryk (bias, MAE, top-20 capture)

**`daily_scoring_manual`** — 3 taski:

1. `environment_contract_check` (~2 min)
2. `cdf_scoring` (~9-10 min) — streaming scoring z SHADOW_TABLE
3. `prediction_audit_monitoring` (~1.5 min) — sprawdzenie jakości predykcji

---

## SEKCJA 2: Architektura modelu

### UltimateSegmentedModel — 4 niezależne segmenty

Model to `mlflow.pyfunc.PythonModel` z 8 sub-modelami (p50 + p90 dla każdego segmentu):

| Segment     | Target                  | Features                                                                                                                           |
| ----------- | ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| TAXI-OUT    | `taxi_out_sec`          | dep_ap_sched, dep_stand, FS stats taxi_out 7d/30d, EMA, stand stats, isLO, local_hour_dep, sin/cos                                 |
| AIRBORNE    | `airborne_sec`          | dep/arr_ap_sched, FS stats airborne + arrival_delay + dur_ratio 7d/30d, distance_km, is_eastbound, local_hour/dow dep/arr, sin/cos |
| TAXI-IN     | `taxi_in_sec`           | arr_ap_sched, arr_stand, FS stats taxi_in 7d/30d, EMA, stand stats, isLO, local_hour_arr, sin/cos                                  |
| TOTAL-BLOCK | `actual_block_time_sec` | kombinacja wszystkich segmentów                                                                                                    |

Każdy sub-model to `HistGradientBoostingRegressor` (sklearn) — obsługuje brakujące wartości (NaN) natywnie przez specjalne split'y → kluczowe dla lotów bez historii (cold start).

**p50 (median prediction):** prosta predykcja.
**p90 (90th percentile):** p50 + CQR shift (Conformal Quantile Regression) kalibrowany na zbiorze walidacyjnym podzielonym po `scheduled_block_time_sec` bucket'ach.

### Finalna predykcja block time

```python
pred_actual_block_time_sec = w_out * pred_taxi_out + w_air * pred_airborne + w_in * pred_taxi_in
# domyślnie: w_out=0.33, w_air=0.33, w_in=0.34
# model TOTAL-BLOCK dostaje własną predykcję i może ją nadpisać
```

### Cechy modelu — pełna lista

**Taxi-out features (~65 cech):**

```
dep_ap_sched, dep_stand, [ac_registration or ac_subtype], leg_type
avg_taxi_out_7d/30d, std_taxi_out_7d/30d, p90_taxi_out_7d/30d
min_taxi_out_7d/30d, max_taxi_out_7d/30d, trend_taxi_out_7d
ema_taxi_out_7d/30d, ema_confidence_dep_7d/30d
delta_ema_avg_taxi_out_7d/30d, delta_ema_avg_dur_ratio_dep_7d/30d
avg_dur_ratio_dep_7d/30d, std_dur_ratio_dep_7d/30d
p90_dur_ratio_dep_7d/30d, min/max_dur_ratio_dep_7d/30d
ema_dur_ratio_dep_7d/30d, trend_dur_ratio_dep_7d
count_dep_7d/30d, has_hist_dep_7d/30d
stand_count_out_7d/30d, stand_avg_taxi_out_7d/30d
stand_p10/p50/p90_taxi_out_30d, stand_std_taxi_out_30d, stand_trend_taxi_out_7d
isLO, local_hour_dep, local_dow_dep
sin_local_hour_dep, cos_local_hour_dep, sin_local_dow_dep, cos_local_dow_dep
sin_month, cos_month
marker_1..marker_17
```

**Airborne features (~70 cech):**

```
dep_ap_sched, arr_ap_sched, [aircraft], leg_type
avg/std/p90/min/max_airborne_7d/30d, trend_airborne_7d
ema_airborne_7d/30d, ema_confidence_route_7d/30d, delta_ema_avg_airborne_7d/30d
avg/std_arrival_delay_7d/30d, p90/min/max_arrival_delay_7d/30d
ema_arrival_delay_7d/30d, delta_ema_avg_arrival_delay_7d/30d
avg/std/p90/min/max_dur_ratio_route_7d/30d
ema_dur_ratio_route_7d/30d, delta_ema_avg_dur_ratio_route_7d/30d
trend_arrival_delay_7d, trend_dur_ratio_route_7d
count_route_7d/30d, has_hist_route_7d/30d
isLO, distance_km, is_eastbound
local_hour_dep/arr, local_dow_dep/arr
sin/cos_local_hour_dep/arr, sin/cos_local_dow_dep/arr
sin_month, cos_month
marker_1..marker_17
```

**Taxi-in features (~55 cech):** analogicznie do taxi-out, dla arr.

### Inactivation logic (kiedy lot przestaje być aktywny)

```python
is_arr = F.col("leg_state").isin("A")           # lot już wylądował
is_div = F.col("leg_state").isin("D")           # divert
is_excl = ~F.col("leg_type").isin(["J","C","G"]) # nieobsługiwany leg type
is_del = F.col("_change_type") == "delete"       # usunięty z LABELS
is_too_far_future = dep_sched_dt > now + LOOKAHEAD_MONTHS

should_inactivate = is_arr | is_div | is_excl | is_del | is_too_far_future
```

### Cold start fallback

Gdy `has_hist_dep_30d == 0` OR `has_hist_route_30d == 0` OR `has_hist_arr_30d == 0`:

* `is_cold_start = True`
* Predykcje ustawiane na NULL
* `inactive_reason = "COLD_START_FALLBACK"`

Gdy `missing_feature_count > MAX_MISSING_FEATURES (10)`:

* Podobny fallback

---

## SEKCJA 3: Historia prac — Iter1 (zakończony ✅)

### Cel Iter1

Zbudowanie Feature Store z 5 tabelami statystyk historycznych, wytrenowanie modelu v9 który korzysta z FS lookup zamiast ręcznych joinów, wdrożenie scoringu przez `fe.score_batch`.

### Problem 1: Typy danych (BIGINT/INT vs DOUBLE)

MLflow ma restrykcyjną walidację typów. Kolumny `count_dep_7d`, `has_hist_dep_7d`, `stand_count_out_7d` były zadeklarowane jako `BIGINT`/`INT` w DDL tabel FS. Gdy te kolumny mają wartość NULL (brak historii), pandas konwertuje je na `float64`. Gdy mają wartość non-null, pandas daje `int64`. Signature modelu mówił `double` (float64). MLflow blokuje konwersję `int64 → float64` bo NumPy uznaje to za niebezpieczne (float64 ma mniej bitów mantysy niż int64).

**Fix:** zmiana DDL wszystkich `count_*`, `has_hist_*`, `stand_count_*` na `DOUBLE` + DLT full refresh.

### Problem 2: OOM podczas treningu (4 razy!)

Standard_F4s ma 4GB RAM. Model trenuje na 247k lotach. CV grid 4 params × 5 folds × 4 segmenty = 80 iteracji. Każda iteracja trzyma w pamięci: df_valid (~500MB), X (~300MB), fold data (~600MB peak). Python RSS nie spada do OS nawet po `del` + `gc.collect()`.

**Fixes w kolejności (każdy niewystarczający bez następnych):**

1. `gc.collect()` + `del model/X_train/X_test/X_train_clean/X_test_clean/y_train/y_test/preds` po każdym foldzie
2. `del X, y` po CV loop (przed final model fitting)
3. `del df_valid/df_fit/df_calib` przed `return` (zapisanie fit_rows/calib_rows do lokalnych zmiennych)
4. Usunięcie `.copy()` z `df_valid = df.dropna(...).copy()` i `X = df_valid[actual_features].copy()`
5. **Kluczowy fix:** `CV_PARAM_GRID` 4→2 słowniki + `CV_N_SPLITS` 5→3 → z 80 do 24 iteracji
6. `gc.collect()` między 4 segmentami (TAXI-OUT, AIRBORNE, TAXI-IN, TOTAL-BLOCK)

Trening v9 przeszedł przy wszystkich 6 fixach razem: **1h 20min**, bez retry.

### Problem 3: score_batch — trzy oddzielne błędy

Po sukcesie treningu `daily_scoring_manual` padało trzy razy z różnych przyczyn:

**FAIL 1: Missing required columns**
`fe.score_batch` wymaga że DataFrame ma WSZYSTKIE kolumny które były w training DataFrame. SHADOW_TABLE jest węższa od training df. Brakuje m.in. `leg_update_no`, `fn_number`, `taxi_out_sec`, etc.

Fix: drugi blok w `ensure_signature_columns` dodaje brakujące kolumny:

```python
for c, type_str in INPUT_TYPES.items():
    if c in out.columns:
        continue
    t = type_str.lower()
    if "long" in t:
        out = out.withColumn(c, F.lit(0).cast("long"))
    elif "int" in t and "long" not in t:
        out = out.withColumn(c, F.lit(0).cast("int"))
    elif "double" in t or "float" in t:
        out = out.withColumn(c, F.lit(None).cast("double"))
    # ... etc
```

Uwaga: `LONG/INT` dostają `0` zamiast `None` — bo `None LONG → float64` w pandas co powoduje FAIL 3.

**FAIL 2: event_date DATE vs TIMESTAMP**
Feature Engineering Client przy PIT lookup wymaga że `event_date` w DataFrame ma typ `DATE`, nie `TIMESTAMP`. Tabele FS mają `event_date TIMESERIES` jako `DATE`.

Fix: `F.to_date(F.col("event_date"))` na końcu `ensure_signature_columns`, po obu pętlach.

**FAIL 3: NULL LONG → float64 — MLflow "cannot safely convert"**
NumPy's `can_cast(int64, float64, casting='safe')` zwraca `False` bo float64 ma tylko 53 bity mantysy vs int64 64 bity. MLflow używa tej funkcji. Kolumna `leg_update_no` (LONG) dodana jako `NULL` → pandas daje `float64` → signature mówi `long` → FAIL.

Fix: j.w. w FAIL 1 — LONG/INT brakujące kolumny dostają `lit(0)` nie `lit(None)`.

### Metryki modelu v9

Z historii projektu (run `train_compare_models` który zarejestrował v9):

* Bias: ~2.6%
* Weekly bias: ~7.1%
* Top-20 capture: ~64.2%
* Quality gate: PASS

---

## SEKCJA 4: Historia prac — Iter2 (kod gotowy, wdrożony, DLT zielony)

### Cel Iter2

Przeprojektowanie Feature Store zgodnie z zaleceniami Tomka:

* Streaming tables zamiast full refresh materialized views
* Nowy układ 9 tabel ft_* (4 streaming + 5 daily stats)
* Brak crossJoin(calendar) — statystyki tylko dla dni z eventami
* On-demand features (sin/cos, distance, is_eastbound) jako UC Python UDF
* EMA bez densyfikacji kalendarza + days_since_last_event (B2)
* ft_airport_timezone jako źródło UTC offset dla local time computation
* Pełny parytet cech modelu z v9 przez FeatureFunction

### Sesja 1 — Nowy feature_store.py (Opus, think hard)

**Zmiany:** Całkowity redesign `src/pipeline/feature_store.py`.

Stara architektura:

```
df_labels + df_leg_times + df_leg_remark + df_leg_misc
  → v_cleaned_flight_data_full_table (mega-join)
  → enriched (+ airport features, sin/cos)
  → data_quality
  → cleaned_flight_data_full_table
  → 5x fs_* (przez crossJoin(calendar) + applyInPandas)
```

Nowa architektura:

```
df_labels → ft_leg_status (streaming table, skipChangeCommits=true)
df_leg_times → ft_leg_times (streaming table)
df_leg_misc → ft_leg_misc (streaming table)
df_ap_basics + df_time_zone → ft_airport_timezone (stream-static join, valid_ts = valid_since)

stary enriched/data_quality nadal istnieje (legacy):
cleaned_flight_data_full_table → 5x ft_*_daily_* (bez crossJoin, z days_since_last_event)
                               → 5x fs_* (LEGACY, zachowane jako rollback)
```

**EMA B2 — kluczowa zmiana:**

```python
# STARY kod — crossJoin(calendar):
entities = df.select(*entity_cols).distinct()
calendar = date_bounds.select(F.explode(F.sequence(...)).alias("event_date"))
daily_agg = entities.crossJoin(calendar).join(daily_agg_flights, ...)
ema_df = daily_agg.groupBy(*entity_cols).applyInPandas(ema_func, schema=...)

# NOWY kod — tylko dni z eventami:
daily_agg_flights = df.groupBy("event_date", *entity_cols).agg(*daily_cols)
# + days_since_last_event przez lag():
w = Window.partitionBy(*entity_cols).orderBy("event_date")
daily_agg_flights = daily_agg_flights.withColumn(
    "days_since_last_event",
    F.when(F.lag("event_date").over(w).isNull(), F.lit(0.0))
     .otherwise(F.datediff("event_date", F.lag("event_date").over(w)).cast("double"))
)
ema_df = daily_agg_flights.groupBy(*entity_cols).applyInPandas(ema_func, schema=...)
# applyInPandas zostaje (Iter3 — przepisanie na native Spark)
```

**ft_airport_timezone** — stream-static join (wyjątek: `valid_ts = valid_since`, NIE `__START_AT`):

```python
# Joini df_ap_basics (lat/lon w stopniach!) z df_time_zone
# PK: iata_ap_code, valid_ts TIMESERIES
# Eksponuje: lat_deg, lon_deg, utc_offset_min
```

Uwaga: lat/lon w STOPNIACH (nie radianach). UDF `haversine_km` robi `math.radians()` w środku.

**Streaming helpers:**

```python
_SCD2_VERSION_TS = "__START_AT"  # kolumna DLT APPLY CHANGES

def _stream_source(source_name, ...):
    return (
        spark.readStream.table(f"{source_catalog}.{source_schema}.{source_name}")
        .options(skipChangeCommits="true")  # DLT SCD2 update/delete handling
    )
```

⚠️ **Problem event_ts:** `__START_AT` = `update_key` = sequential counter (~174M), NIE timestamp. Szczegóły w Sekcji 6.

**Problemy naprawione w trakcie Sesji 1:**

* `leg_remark` i `leg_misc` nie mają `__END_AT` — defensive filter `if "__END_AT" in df.columns` zastąpiony bezwarunkowym, ale dla tych tabel całkowicie usunięty
* `leg_no` DDL: `STRING` → `LONG` (source ma `LONG`)
* `event_ts` z `__START_AT`: `/1e6` → `/1e3` → `/ 1000000` (wszystko złe — patrz Sekcja 6)

### Sesja 2 — on_demand_functions.py (Opus, think hard)

**Nowy plik:** `src/pipeline/on_demand_functions.py`

15 funkcji (6 obecnych + 9 nowych dla local time):

```python
# Geo (2):
haversine_km(lat1, lon1, lat2, lon2)  → distance_km  (lat/lon w stopniach)
is_eastbound(lon1, lon2)              → 0/1

# Local time raw (3):
local_hour(scheduled_dt, utc_offset_min)  → hour 0-23
local_dow(scheduled_dt, utc_offset_min)   → weekday 0-6 (Mon=0, Sun=6)
month_of(scheduled_dt)                    → month 1-12

# Sin/cos (9 × 2 = ale 6 UDF, każda wołana 2x z różnymi args):
sin_local_hour, cos_local_hour  (dep + arr przez różne input_bindings)
sin_local_dow, cos_local_dow    (dep + arr)
sin_month_of, cos_month_of      (tylko dep_sched_dt)

# Orphany — do cleanup Iter2.5:
sin_cos_hour, sin_cos_dow, sin_cos_month  (dict return, nieużywane)
duration_ratio                            (LEAKAGE — patrz Sesja 4)
```

Notebook `10_register_on_demand_functions.ipynb` — rejestracja przez SQL:

```sql
CREATE OR REPLACE FUNCTION panda_silver_dev.ml_ops.local_hour(
    scheduled_dt TIMESTAMP, utc_offset_min INT)
RETURNS INT LANGUAGE PYTHON AS $$
from datetime import timedelta
if scheduled_dt is None or utc_offset_min is None: return 0
local_dt = scheduled_dt + timedelta(minutes=utc_offset_min)
return local_dt.hour
$$;
```

Uwaga: ciała sin/cos są INLINE (UC UDF nie może wołać innych UC UDF).

**Test parytetu:** Spark `(dayofweek+5)%7` == Python `weekday()` dla wszystkich 7 dni — potwierdzone empirycznie:

| Dzień           | Spark     | Python | Match |
| --------------- | --------- | ------ | ----- |
| Sun             | (1+5)%7=6 | 6      | ✅     |
| Mon             | (2+5)%7=0 | 0      | ✅     |
| ... wszystkie 7 |           |        | ✅     |

### Sesja 3 — settings.py (Sonnet)

Dodano do `FlightDelaySettings` dataclass:

```python
# 9 nowych FT_* TABLE:
FT_LEG_STATUS_TABLE, FT_LEG_TIMES_TABLE, FT_LEG_MISC_TABLE
FT_AIRPORT_TIMEZONE_TABLE
FT_ROUTE_DAILY_STATS_TABLE
FT_AIRPORT_DAILY_TAXI_OUT_TABLE, FT_AIRPORT_DAILY_TAXI_IN_TABLE
FT_STAND_DAILY_OUT_TABLE, FT_STAND_DAILY_IN_TABLE

# 5 PK_FT_* (kluczowe — dict-form binding dla stand):
PK_FT_AIRPORT_TAXI_OUT = ["dep_ap_sched"]
PK_FT_AIRPORT_TAXI_IN  = ["arr_ap_sched"]
PK_FT_ROUTE            = ["route_id"]
PK_FT_STAND_OUT        = ["stand_id_out"]   # strona base df
PK_FT_STAND_IN         = ["stand_id_in"]    # strona base df

# 15 UC_FN_* (na format: f"{silver_full}.{function_name}"):
UC_FN_HAVERSINE_KM, UC_FN_IS_EASTBOUND
UC_FN_LOCAL_HOUR, UC_FN_LOCAL_DOW, UC_FN_MONTH_OF
UC_FN_SIN_LOCAL_HOUR, UC_FN_COS_LOCAL_HOUR
UC_FN_SIN_LOCAL_DOW, UC_FN_COS_LOCAL_DOW
UC_FN_SIN_MONTH_OF, UC_FN_COS_MONTH_OF
# Orphany:
UC_FN_DURATION_RATIO (usunięte z FF, stała zostaje)
UC_FN_SIN_COS_HOUR/DOW/MONTH (stare dict UDF)
```

### Sesja 4 — training.py _build_feature_lookups (Opus, think hard)

Przepisanie `_build_feature_lookups()` na nowe tabele. Zachowana `_build_feature_lookups_legacy_fs` (rollback).

**Kluczowy pattern: dict-form lookup_key:**

```python
# Tabele stand mają PK: stand_id, ale base df ma stand_id_out/stand_id_in
FeatureLookup(
    table_name=settings.FT_STAND_DAILY_OUT_TABLE,
    lookup_key={"stand_id": "stand_id_out"},  # {tabela.kolumna: base_df.kolumna}
    timestamp_lookup_key="event_date",
)

# Tabele timezone mają PK: iata_ap_code, ale base df ma dep_ap_sched/arr_ap_sched
FeatureLookup(
    table_name=settings.FT_AIRPORT_TIMEZONE_TABLE,
    lookup_key={"iata_ap_code": "dep_ap_sched"},
    feature_names=["lat_deg", "lon_deg", "dep_utc_offset_min"],
    rename_outputs={"dep_utc_offset_min": "dep_utc_offset_min"},  # rename z tabeli
    timestamp_lookup_key="dep_sched_dt",
)
```

**Wykryty data leakage w duration_ratio:**
`duration_ratio = actual_block_time_sec / scheduled_block_time_sec`
`actual_block_time_sec` to **label** (wartość którą model ma przewidzieć). Na treningu to powoduje że model "widzi odpowiedź". Na scoringu wartość jest NULL bo lot jeszcze się nie odbył. Usunięto z FeatureFunction.

### Sesja 5 — scoring.py (Opus, think hard)

Aktualizacja mechanizmu scoringowego.

**Wykryty train/serve skew:**
Problem: `_add_local_time_cosine_features` była wywoływana w training.py po `training_set.load_df()` (linia 548), ale signature modelu był budowany z innego `training_set_meta.load_df()` (linia ~1153) — bez sin/cos. Signature nie zawierał tych cech → score_batch nie dostarczałby ich → model dostaje NaN zamiast właściwych wartości.

Naprawione przez przeniesienie sin/cos do FeatureFunction (Sesja 5.5).

**`_add_local_time_cosine_features`** — funkcja nadal istnieje w scoring.py ale NIE jest wywoływana (orphan do cleanup).

### Sesja 5.5 — Opcja A: sin/cos jako FeatureFunction (Opus, think hard)

Rozwiązanie train/serve skew przez 15 FeatureFunction w `_build_feature_lookups`.

Ta sama UDF może być użyta z różnymi `input_bindings` → różne output_names:

```python
FeatureFunction(
    udf_name=settings.UC_FN_LOCAL_HOUR,
    input_bindings={"scheduled_dt": "dep_sched_dt", "utc_offset_min": "dep_utc_offset_min"},
    output_name="local_hour_dep",
),
FeatureFunction(
    udf_name=settings.UC_FN_LOCAL_HOUR,
    input_bindings={"scheduled_dt": "arr_sched_dt", "utc_offset_min": "arr_utc_offset_min"},
    output_name="local_hour_arr",
),
# ... analogicznie dla local_dow, sin/cos
```

**Finalna struktura `_build_feature_lookups` — 24 obiekty:**

```
7 FeatureLookup:
  ft_airport_daily_taxi_out   lookup: dep_ap_sched
  ft_route_daily_stats         lookup: route_id
  ft_airport_daily_taxi_in    lookup: arr_ap_sched
  ft_stand_daily_out          lookup: {stand_id: stand_id_out}
  ft_stand_daily_in           lookup: {stand_id: stand_id_in}
  ft_airport_timezone (dep)   lookup: {iata_ap_code: dep_ap_sched}, ts: dep_sched_dt
  ft_airport_timezone (arr)   lookup: {iata_ap_code: arr_ap_sched}, ts: arr_sched_dt

17 FeatureFunction:
  haversine_km → distance_km
  is_eastbound → is_eastbound
  local_hour (dep) → local_hour_dep
  local_hour (arr) → local_hour_arr
  local_dow (dep) → local_dow_dep
  local_dow (arr) → local_dow_arr
  month_of → month
  sin_local_hour (dep) → sin_local_hour_dep
  cos_local_hour (dep) → cos_local_hour_dep
  sin_local_hour (arr) → sin_local_hour_arr
  cos_local_hour (arr) → cos_local_hour_arr
  sin_local_dow (dep) → sin_local_dow_dep
  cos_local_dow (dep) → cos_local_dow_dep
  sin_local_dow (arr) → sin_local_dow_arr
  cos_local_dow (arr) → cos_local_dow_arr
  sin_month_of → sin_month
  cos_month_of → cos_month
```

**Dlaczego Opcja A (pełne sin/cos) a nie Opcja F (raw godziny)?**
Opcja A zachowuje pełny parytet z v9 — te same 15 cech czasowych. Opcja F dawałaby raw `local_hour` zamiast sin/cos. Dla HGBT (drzewa) różnica jest mała (~±1% MAE), ale zdecydowano na parytet z v9 żeby v10 był bezpośrednio porównywalny.

### Sesja 6 — Notebooki i testy (Sonnet)

* `notebooks/08_smoke_test_plan_a.ipynb` — zaktualizowany: Check 1 sprawdza 9 ft_* + `days_since_last_event DOUBLE`, Check 3 sprawdza 15 cech FF w signature v10
* `notebooks/09_register_feature_tables.ipynb` — zaktualizowany: rejestruje 9 ft_* zamiast 5 fs_*; fs_* w `legacy_tables_to_deregister` ale NIE wyrejestrowywane (rollback)
* `notebooks/11_dlt_pipeline_refresh.ipynb` — nowy runbook dla deployu
* `tests/test_feature_store_helpers.py` — +6 testów DDL builderów
* `pytest` — 22 passed (10 istniejących + 6 nowych DDL + 6 on_demand functions)

### Testowanie po deploy

Przebieg deployu:

```
Krok 1: git add -A && git commit && git push
Krok 2: bundle deploy → Deployment complete!
Krok 3: notebook 10 (rejestracja 15 UDF) — ✅
Krok 4: DLT full refresh → FAIL: leg_remark __END_AT not found → fix → FAIL: leg_no STRING vs LONG → fix
         FAIL: event_ts = 1970 (/ 1e6) → fix /1e3 → FAIL: event_ts = 1970 (/1e3 = 3 dni)
         → FAIL: /1000000 cast timestamp = 1970 → PROBLEM NIEROZWIĄZANY (patrz Sekcja 6)
         DLT zielony, tabele istnieją, ale ft_leg_* mają event_ts = 1970
Krok 5: notebook 09 → fs_* już zarejestrowane (z Iter1), ft_* czeka
Krok 6: trening v10 → NIE URUCHOMIONY (ft_leg_* nie blokują, ale można ruszyć teraz)
```

---

## SEKCJA 5: Kompletny status 26 punktów

| #  | Co                                     | Status       | Szczegóły                                                                               |
| -- | -------------------------------------- | ------------ | --------------------------------------------------------------------------------------- |
| 1  | 5 tabel FS zbudowanych                 | ✅ DONE       | `fs_*` w panda_silver_dev.ml_ops, zarejestrowane w UC FS                                |
| 2  | Pipeline FS działa                     | ✅ DONE       | DLT pipeline aktywny, auto-refresh                                                      |
| 3  | Trening z nowym FS                     | ✅ DONE       | v9 wytrenowany, 247k lotów, 1h 20min                                                    |
| 4  | Model w UC                             | ✅ DONE       | `panda_gold_dev.ml_ops.flight_delay_model` v9                                           |
| 5  | Checki decyzyjne                       | ✅ DONE       | Stand data null dla aktywnych lotów; HGBT obsługuje NaN split'ami                       |
| 6  | Decyzja: Plan A                        | ✅ DONE       | fe.score_batch; alternatywa Plan B (ręczne joiny) odrzucona                             |
| 7  | score_batch na atrapie                 | ✅ DONE       |                                                                                         |
| 8  | score_batch na prawdziwym modelu       | ✅ DONE       | 4/4 smoke test passes; 3 bugfixy w ensure_signature_columns                             |
| 9  | Implementacja scoringu                 | ✅ DONE       | scoring.py z Plan A + ensure_signature_columns (BLOK 1+2)                               |
| 10 | Weryfikacja EMA                        | ✅ DONE       | EMA matematycznie identyczna z poprzednią (commit f1aafc4); parytet predykcji do Fazy 7 |
| 11 | Smoke test                             | ✅ DONE       | 4/4 checks: fs_DOUBLE, champion_alias, signature, score_batch                           |
| 12 | Champion → v9                          | ✅ DONE       | alias ustawiony, v10 pending                                                            |
| 13 | Merge do main                          | ✅ DONE       | Iter1 i Iter2 na main                                                                   |
| 14 | Prod schematy databricks.yml           | ⬜ TODO       | `__TODO_PROD_SILVER_SCHEMA__`, `__TODO_PROD_GOLD_SCHEMA__`                              |
| 15 | Streaming tables                       | 🔵 DEPLOYED  | ft_leg_* streaming, event_ts fixu na Iter2.5; ft_**daily** jako MV                      |
| 16 | readStream                             | 🔵 DEPLOYED  | `_stream_source()` z `skipChangeCommits=true`                                           |
| 17 | Usunięcie joinów                       | 🟡 CZĘŚCIOWE | ft_leg_* niezależne; daily stats nadal z legacy enriched/cleaned_flight                 |
| 18 | Agregacje bez crossJoin                | 🔵 DEPLOYED  | `_build_daily_stats` bez calendar crossJoin + days_since_last_event                     |
| 19 | EMA bez applyInPandas                  | ⬜ TODO       | Iter3; applyInPandas zostaje (B2 = bez densyfikacji, ale nie native Spark)              |
| 20 | On-demand features                     | 🔵 DEPLOYED  | 15 UDF zarejestrowanych; 17 FF w _build_feature_lookups; notebook 10 ✅                  |
| 21 | Nowy układ tabel ft_*                  | 🔵 DEPLOYED  | DLT zielony; event_ts w ft_leg_* złe (patrz Sekcja 6)                                   |
| 22 | Jawne nazwy f-string                   | ✅ DONE       | `_source_table()`, `_fs_table()`, FT_* stałe w settings                                 |
| 23 | Nowsze API DLT                         | ✅ DONE       | `@dp.temporary_view()`, `@dp.materialized_view()`                                       |
| 24 | Usunięcie `__END_AT IS NULL` defensive | ✅ DONE       | Bezwarunkowe filtry + TODO Iter2 comments                                               |
| 25 | AS OF semantics                        | ⬜ TODO       | Iter3 — data leakage przy historycznym treningu                                         |
| 26 | create_feature_table UC                | ✅ DONE       | fs_* zarejestrowane; notebook 09 gotowy na ft_*                                         |

---

## SEKCJA 6: OTWARTY PROBLEM — ft_leg_* event_ts = 1970 (szczegółowa analiza)

### Diagnoza (potwierdzona 2026-06-10)

```python
# Wyniki diagnostyczne z klastra:
# __START_AT sample: 174601483 → event_ts = 1970-01-01 00:02:54 (po /1e6)
# __START_AT == update_key: 100.0%
# update_key ~174M-183M = SEQUENTIAL COUNTER z systemu Netline, NIE Unix timestamp
# entry_dt range: 2023-01-17 09:34:00 → 2023-11-08 14:09:27  ← PRAWDZIWY timestamp

# Dlaczego żadne dzielenie nie działa:
174_601_483 / 1_000_000 = 174.6 sek = 1970-01-01 00:02:54  ← widzimy
174_601_483 / 1_000     = 174_601 sek = 1972 rok
174_601_483 / 1         = 174_601_483 sek = 1975 rok
# Dla 2024 potrzebujemy ~1_700_000_000 sek — update_key jest ~10x za mały
```

### Rozwiązanie per tabela

**ft_leg_status** (źródło: `netline___schedops__leg`):
Źródło MA `entry_dt` — prawdziwy timestamp kiedy rekord był aktualizowany. Problem: ~27 duplikatów na `(leg_no, entry_dt)` z różnymi `update_key`.

Patch dla `feature_store.py`:

```python
@dp.table(
    name=_fs_table("ft_leg_status"),
    schema=leg_status_schema_ddl(),
    ...
)
def ft_leg_status():
    raw = _stream_source("netline___schedops__leg").filter(
        F.col("counter").isin(0, 1) &
        F.col("leg_type").isin("H", "S") &
        F.col("leg_state").isin("H", "C")
    )
    # Deduplikacja: dla tego samego (leg_no, entry_dt) bierz MAX(update_key)
    w_dedup = Window.partitionBy("leg_no", "entry_dt").orderBy(F.col("update_key").desc())
    deduped = raw.withColumn("_rn", F.row_number().over(w_dedup)).filter("_rn == 1").drop("_rn")
    return (
        deduped
        .withColumn("event_ts", F.col("entry_dt"))  # ← ZMIANA: entry_dt, nie __START_AT
        .select(
            F.col("leg_no").cast("long"),
            F.col("event_ts").cast("timestamp"),
            ...
        )
    )
```

⚠️ Uwaga: `row_number()` w streaming może powodować problemy ze stanem. Alternatywa: zmień PK na `(leg_no, update_key)` zamiast `(leg_no, entry_dt)` i usuń deduplikację.

**ft_leg_times** (źródło: `netline___schedops__leg_times`):
BRAK `entry_dt`. Dostępne timestamps: offblock_dt, airborne_dt, landing_dt, onblock_dt — to czasy zdarzeń operacyjnych, NIE czasy dostępności danych. `_creation_dt` to techniczna data backfillu (stała).

Rekomendacja Tomka: użyć `(leg_no, leg_update_no)` jako PK.

Patch:

```python
# DDL: zmień z (leg_no, event_ts TIMESERIES) na (leg_no, leg_update_no)
# NIE jest TIMESERIES feature table — brak PIT lookup
def leg_times_schema_ddl():
    return """
        leg_no LONG NOT NULL,
        leg_update_no LONG NOT NULL,
        -- ... reszta kolumn ...
        CONSTRAINT ft_leg_times_pk PRIMARY KEY (leg_no, leg_update_no)
    """
    # Brak TIMESERIES — nie rejestrujemy jako UC Feature Table z timestamp_keys

@dp.table(name=_fs_table("ft_leg_times"), ...)
def ft_leg_times():
    return (
        _stream_source("netline___schedops__leg_times")
        .filter(F.col("usage") == "F")  # tylko F (actual) usage
        .select(
            "leg_no", "leg_update_no",
            "offblock_dt", "airborne_dt", "landing_dt", "onblock_dt",
            # ... etc. BEZ event_ts
        )
    )
```

**ft_leg_misc** (źródło: `netline___schedops__leg_misc`):
Identyczna sytuacja jak leg_times.

```python
def leg_misc_schema_ddl():
    return """
        leg_no LONG NOT NULL,
        leg_update_no LONG NOT NULL,
        dep_stand STRING,
        arr_stand STRING,
        CONSTRAINT ft_leg_misc_pk PRIMARY KEY (leg_no, leg_update_no)
    """
    # Brak TIMESERIES

@dp.table(name=_fs_table("ft_leg_misc"), ...)
def ft_leg_misc():
    return (
        _stream_source("netline___schedops__leg_misc")
        .select("leg_no", "leg_update_no", "dep_stand", "arr_stand")
    )
```

### Dlaczego to NIE blokuje teraz

1. `ft_leg_*` NIE są w `_build_feature_lookups()` — żaden FeatureLookup ani FeatureFunction nie sięga do tych tabel
2. Scoring (`daily_scoring_manual`) nie używa tych tabel
3. Training (`weekly_training_manual`) nie używa tych tabel
4. DLT pipeline działa, tabele istnieją z danymi (tylko event_ts jest 1970)
5. Blokuje dopiero Iter2.5 gdy daily stats będą czytać z ft_leg_* zamiast legacy

---

## SEKCJA 7: Natychmiastowe następne kroki

### KROK 1 — Uruchom trening v10 (teraz, ~1.5-2h)

```bash
databricks bundle run weekly_training_manual -t dev -p panda-dev
```

**Co się stanie:** 4 taski (environment check → build training set → train → quality gate). Build training set używa `_build_feature_lookups()` z nowymi ft_* daily stats tabelami. Trening zobaczy nowe cechy: `days_since_last_event` dla każdej tabeli daily stats + 15 local time FF.

**Jeśli OOM:** Ten sam klaster 4GB. CV grid jest już zredukowany (2 params × 3 folds = 24 iteracji). Wszystkie .copy() usunięte. Jeśli nadal OOM:

* Redukcja CV do 1 param i 2 folds (tymczasowo)
* Lub: zmiana klastra na Standard_D4s_v5 (16GB)

**Jak sprawdzić po sukcesie:**

```bash
databricks jobs list -p panda-dev --output json | python3 -c "
import sys,json
[print(j['job_id'], j['settings']['name']) for j in json.load(sys.stdin)['jobs']]
"
```

### KROK 2 — Po sukcesie treningu: ustaw champion → v10

```python
from mlflow import MlflowClient
import mlflow
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
# Znajdź najnowszą wersję:
versions = [int(mv.version) for mv in client.search_model_versions(
    "name='panda_gold_dev.ml_ops.flight_delay_model'")]
latest = max(versions)
client.set_registered_model_alias(
    "panda_gold_dev.ml_ops.flight_delay_model", "champion", str(latest))
print(f"champion → v{latest}")
```

### KROK 3 — Smoke test

Uruchom `notebooks/08_smoke_test_plan_a.ipynb`.

Sprawdza:

* Check 1: 9 ft_* istnieją + count/has_hist/stand_count DOUBLE + `days_since_last_event DOUBLE`
* Check 2: alias `champion` istnieje i wskazuje na nową wersję
* Check 3: signature v10 ma 15 cech czasowych z FF (local_hour_dep, cos_local_hour_dep, etc.)
* Check 4: `fe.score_batch` zwraca predykcje dla próbki 50 aktywnych lotów

**Możliwy problem Check 3:** Jeśli FF outputs nie trafiają do signature (`training_set_meta.load_df()` nie materializuje FF), sprawdź:

```python
info = mlflow.models.get_model_info(f"models:/panda_gold_dev.ml_ops.flight_delay_model/{latest}")
import json
cols = [c["name"] for c in json.loads(info.signature.to_dict()["inputs"])]
ff_cols = ["local_hour_dep","local_hour_arr","sin_month","cos_month","distance_km","is_eastbound"]
missing = [c for c in ff_cols if c not in cols]
print("Missing FF outputs in signature:", missing)
```

### KROK 4 — Daily scoring confirm

```bash
databricks bundle run daily_scoring_manual -t dev -p panda-dev
```

Bramka: 3 taski zielone (environment_check, cdf_scoring, prediction_audit_monitoring), wiersze w `block_time_predictions_v3`.

### KROK 5 — Rejestracja 9 ft_* w UC Feature Store

Uruchom `notebooks/09_register_feature_tables.ipynb`.

* ft_leg_* zostaną zarejestrowane (ze złym event_ts, ale to nie blokuje)
* ft_**daily** i ft_airport_timezone — zarejestrowane poprawnie
* fs_* w `legacy_tables_to_deregister` — NIE wyrejestrowywać

### KROK 6 — Faza 7: parytet v9 vs v10

Sprawdź czy model v10 jest jakościowo podobny do v9. Nowe cechy: `days_since_last_event` w każdej daily stats tabeli.

```python
# Wybierz 100-500 lotów z ostatnich 7 dni które mają actual_block_time_sec
# Uruchom predict z v9 i v10
# Porównaj:
import numpy as np
mae_v9  = np.mean(np.abs(y_true - pred_v9))
mae_v10 = np.mean(np.abs(y_true - pred_v10))
bias_v9  = np.mean(pred_v9 - y_true)
bias_v10 = np.mean(pred_v10 - y_true)
print(f"MAE v9={mae_v9:.0f}s, v10={mae_v10:.0f}s, diff={abs(mae_v9-mae_v10)/mae_v9:.1%}")
```

Kryterium akceptacji: MAE diff < 5%. Jeśli więcej — debug FF outputs (sprawdź czy FeatureFunction rzeczywiście zwraca wartości, nie NaN).

---

## SEKCJA 8: Iter2.5 — pełny plan (po pomyślnym v10)

### Zadanie A: Fix ft_leg_* event_ts (1 sesja CC Opus)

Jak opisano w Sekcji 6:

* `ft_leg_status`: zmień `event_ts = __START_AT` → `event_ts = entry_dt`, dodaj deduplikację
* `ft_leg_times`: usuń event_ts, zmień PK na `(leg_no, leg_update_no)`, usuń TIMESERIES
* `ft_leg_misc`: analogicznie

Po zmianie kodu: DLT full refresh (~30-60 min) → sprawdź `event_ts` range (powinno 2023-2026).

### Zadanie B: Cleanup orphans (1 sesja CC Sonnet)

**`src/pipeline/on_demand_functions.py`** — usuń:

* `sin_cos_hour(hour)` → dict return, nieużywane (zastąpione przez sin_local_hour + cos_local_hour)
* `sin_cos_dow(dow)` → j.w.
* `sin_cos_month(month)` → j.w.
* `duration_ratio(actual_sec, scheduled_sec)` → LEAKAGE

**`src/ml_project/settings.py`** — usuń/wyczyść:

* `UC_FN_DURATION_RATIO: str = ""`
* `UC_FN_SIN_COS_HOUR: str = ""`
* `UC_FN_SIN_COS_DOW: str = ""`
* `UC_FN_SIN_COS_MONTH: str = ""`

**`src/pipeline/feature_store.py`** — usuń:

* Wszystkie `@dp.materialized_view()` dekorowane z prefixem `fs_*` (linie ~700-900)
* Funkcja `_build_feature_lookups_legacy_fs` (jeśli przeniesiona do training.py backup)
* Baner `# LEGACY fs_* (Iter1). TODO Iter2 cleanup.`
* Funkcja `_add_local_time_cosine_features` (przenoszone, ale orphan w scoring.py)

**`src/ml_project/training.py`** — usuń:

* Funkcja `_build_feature_lookups_legacy_fs` (rollback path po pomyślnym v10)

**`src/ml_project/scoring.py`** — usuń:

* Funkcja `_add_local_time_cosine_features(df)` (zdefiniowana, niewywoływana)

**`notebooks/10_register_on_demand_functions.ipynb`** — usuń bloki rejestracji:

* `sin_cos_hour`, `sin_cos_dow`, `sin_cos_month`, `duration_ratio`

### Zadanie C: Pełne odcięcie od legacy cleaned_flight_data (1-2 sesje CC Opus)

Daily stats (`ft_*_daily_*`) nadal czytają z `cleaned_flight_data_full_table` (legacy mega-join). Powinny czytać z `ft_leg_status` + `ft_leg_times`.

To jest trudne z kilku powodów:

1. `ft_leg_*` są streaming tables — stream-stream join wymaga watermarków
2. `ft_leg_status` ma dane o statusie/typie/marker, `ft_leg_times` ma OOOI timestamps
3. Potrzebne są: `scheduled_block_time_sec`, `taxi_out_sec`, `airborne_sec`, `taxi_in_sec` (z leg_times)

Schemat rozwiązania:

```python
# W _build_daily_stats zamiast:
df = spark.read.table(SETTINGS.CLEANED_FLIGHT_DATA_TABLE)

# Będzie:
leg_status = spark.readStream.table(FT_LEG_STATUS_TABLE)
leg_times = spark.readStream.table(FT_LEG_TIMES_TABLE)
# Stream-batch join z leg_status × leg_times przez (leg_no, leg_update_no)
# Watermark: leg_status.event_ts z opóźnieniem np. 7 dni
df = leg_status.join(
    leg_times.withWatermark("event_ts_proxy", "7 days"),
    on=["leg_no", "leg_update_no"],
    how="inner"
)
```

### Zadanie D: Update notebook 09 po ft_leg_* fix

Po naprawieniu event_ts w ft_leg_*:

* Zaktualizuj `timestamp_keys` dla ft_leg_status w notebook 09
* Usuń lub zmień rejestrację ft_leg_times/ft_leg_misc (nie będą TIMESERIES)

---

## SEKCJA 9: Iter3 — plan długoterminowy

### Punkt 19 — EMA bez applyInPandas

**Problem:** `applyInPandas` w `_build_daily_stats` jest single-node per group.

```python
ema_df = daily_agg.groupBy(*entity_cols).applyInPandas(ema_func, schema=ema_schema)
```

Dla lotniska WAW (~200 lotów/dzień, 3 lata historii = ~200k rekordów per group) → jedna ogromna pandas operacja na driverze = wąskie gardło.

**Opcja 1 — Native Spark EMA (przybliżenie):**
EMA z `lag()` przez window functions:

```python
# EMA = α × current + (1-α) × prev_ema
# Nie da się wyrazić dokładnej rekurencji przez Spark SQL window
# Ale można iteracyjne przybliżenie przez N etapów lag()
```

Ograniczenie: nie jest to dokładna EMA, tylko przybliżenie.

**Opcja 2 — Spark Structured Streaming flatMapGroupsWithState:**
Dokładna EMA z persistent state per group. Skomplikowane w implementacji, ale scalable.

**Opcja 3 — Zmiana algorytmu:**
Zamiast EMA użyć EWMA (Exponentially Weighted Moving Average) która da się wyrazić przez Spark:

```python
# Aggregation przez native Spark: weighted sum z wykładniczymi wagami
# Wymaga zdefiniowania predefiniowanego okna czasowego
```

**Rekomendacja:** Opcja 2 (flatMapGroupsWithState) dla dokładności, Opcja 3 dla prostoty.

### Punkt 25 — AS OF semantics (PIT correctness)

**Problem:** Przy treningu historycznym, dla lotu z 2023-06-01, `fe.create_training_set` robi lookup do `ft_route_daily_stats` z `timestamp_lookup_key="event_date"`. Infrastruktura PIT jest dostępna (TIMESERIES key). Ale czy `training_set_meta.load_df()` faktycznie daje PIT-poprawne dane?

Sprawdzenie: dla lotów historycznych, czy cechy z daily stats odpowiadają stanowi z DNIA LOTU czy OBECNEMU stanowi?

Jeśli model widzi dane "z przyszłości" względem lotu → data leakage → model wydaje się lepszy na walidacji niż jest na produkcji.

**Fix:** Upewnić się że `timestamp_lookup_key` w każdym `FeatureLookup` wskazuje na `event_date` z DNIA LOTU, nie na obecną datę. Powinno być poprawne jeśli `event_date` w base df = `dep_sched_dt.cast("date")`.

### Punkt 14 — Prod schematy

W `databricks.yml` dla profilu `prod`:

```yaml
silver_schema: "__TODO_PROD_SILVER_SCHEMA__"
gold_schema:   "__TODO_PROD_GOLD_SCHEMA__"
```

Trzeba podać rzeczywiste nazwy schematów w prod workspace przed deploy na produkcję.

---

## SEKCJA 10: Kluczowe wzorce kodu

### ensure_signature_columns — pełna logika

```python
def ensure_signature_columns(df):
    """Przygotowuje DataFrame PRZED fe.score_batch.

    Powód istnienia: SHADOW_TABLE jest węższa od training DataFrame.
    fe.score_batch wymaga WSZYSTKICH kolumn z training df.

    BLOK 1: Dla kolumn OBECNYCH w batchu — cast do typów z INPUT_TYPES.
            Ważne: LONG fillna 0 (nie None!) bo None → float64 → MLflow FAIL
    BLOK 2: Dla kolumn BRAKUJĄCYCH — dodaje z wartościami domyślnymi.
            LONG/INT → lit(0)      ← kluczowe, nie lit(None)
            DOUBLE   → lit(None)   ← NaN, HGBT obsługuje
            STRING   → lit(None) lub "UNKNOWN" dla kluczy lotnisk
    KONIEC:  F.to_date(event_date) — DATE dla PIT lookup w FS
    """
```

### _add_fs_lookup_keys — klucze lookupów

```python
def _add_fs_lookup_keys(df):
    """Dodaje klucze FK do FeatureLookup.
    Musi być wywołane PRZED fe.score_batch / fe.create_training_set.
    """
    return (
        df
        .withColumn("route_id",     F.concat_ws("_", "dep_ap_sched", "arr_ap_sched"))
        .withColumn("stand_id_out", F.concat_ws("_", "dep_ap_sched", "dep_stand"))
        .withColumn("stand_id_in",  F.concat_ws("_", "arr_ap_sched", "arr_stand"))
    )
```

### _pred_expr — wyciąganie predykcji ze struct

```python
def _pred_expr(field_name: str):
    # score_batch zwraca kolumnę "prediction" jako STRUCT
    # Dostęp: F.col("prediction.pred_actual_block_time_sec")
    return (
        F.col(f"prediction.{field_name}")
        if field_name in OUTPUT_COLS
        else F.lit(None).cast("double")
    )
```

### Trening — OOM-safe konfiguracja

```python
# settings.py — kluczowe wartości po optymalizacji:
CV_N_SPLITS = 3      # było 5 — redukcja foldów
CV_PARAM_GRID = [    # 2 konfiguracje zamiast 4
    {"max_leaf_nodes": 63, "learning_rate": 0.05, "min_samples_leaf": 20, "l2_regularization": 0.0},
    {"max_leaf_nodes": 63, "learning_rate": 0.03, "min_samples_leaf": 30, "l2_regularization": 0.10},
]

# training.py — po każdym foldzie CV:
del model, X_train, X_test, X_train_clean, X_test_clean, y_train, y_test, preds
gc.collect()  # na końcu _evaluate_param_set

# po CV loop — przed final model fitting:
final_mappings = _build_top_k_mappings(X, settings)
del X, y
gc.collect()

# przed return segmentu:
fit_rows = int(len(df_fit))
calib_rows = int(len(df_calib))
del df_valid, df_fit, df_calib
gc.collect()

# między segmentami w run_train_compare_models:
best_out = train_and_evaluate_segment(..., "TAXI-OUT", ...)
gc.collect()  # po każdym z pierwszych 3 segmentów
best_air = train_and_evaluate_segment(..., "AIRBORNE", ...)
gc.collect()
# etc.
```

---

## SEKCJA 11: Tests coverage

```
tests/
  test_feature_store_helpers.py  # 6 testów DDL builderów (leg_*, timezone, daily_stats, stand_daily)
  test_settings.py               # 4 testy (format catalog.schema.name dla wszystkich stałych)
  test_on_demand_functions.py    # 6 testów (local_hour=16 dla UTC+2, local_dow parytet Spark, sin²+cos²=1)
  conftest.py                    # sys.path setup

Łącznie: 16 testów (byli 22 w ostatnim raporcie CC — możliwe że dodano więcej w Sesji 6)
Uruchomienie: pytest tests/ -v
```

---

## SEKCJA 12: Legacy i rollback

### Gdzie są stare implementacje

* `feature_store.py`: blok `# LEGACY fs_* (Iter1)` z definicjami 5 fs_* tabel
* `training.py`: `_build_feature_lookups_legacy_fs()` — stara wersja z FeatureLookup do fs_*
* `scoring.py`: `_add_local_time_cosine_features(df)` — orphan, nie wywoływana
* `settings.py`: `FS_TAXI_OUT_TABLE`, `FS_AIRBORNE_TABLE` etc. — stare stałe zachowane

### Jak zrobić rollback do v9/Iter1

```python
# Ustaw champion → v9
MlflowClient().set_registered_model_alias(
    "panda_gold_dev.ml_ops.flight_delay_model", "champion", "9")

# Jeśli training.py zostało zaktualizowane na ft_*,
# zamień w _build_feature_lookups wywołanie na _build_feature_lookups_legacy_fs
# (stara implementacja zachowana w pliku)
```

---

## SEKCJA 13: Numeracja wersji modeli

| Wersja | Opis                                                                                | Status      |
| ------ | ----------------------------------------------------------------------------------- | ----------- |
| v1-v6  | Legacy modele, różne eksperymenty                                                   | Historyczne |
| v7     | Pierwszy model z FS lookup, signature all-double (za szeroka)                       | Historyczne |
| v8     | Signature z cofniętymi typami long/int (nie działało z nullable cols)               | Historyczne |
| v9     | FS-based, CV 2params×3folds, gc.collect fixes, AKTUALNY CHAMPION                    | ✅ Prod      |
| v10    | Iter2 model: ft_*_daily_stats + ft_airport_timezone + 17 FF + days_since_last_event | ⬜ Pending   |

Model v10 będzie pierwszym modelem który ma:

* `days_since_last_event` jako cechę (nowe per tabela daily stats)
* `distance_km` i `is_eastbound` przez FeatureFunction (było przez Spark UDF w enriched)
* local time features przez FeatureFunction (było przez enriched)
* UTC offset z ft_airport_timezone (był przez airport_features view)

---

## SEKCJA 14: Komendy startowe dla nowego chatu

```bash
# Sprawdź aktualny stan
cd ~/path/to/ML_Project_v3-main
git log --oneline -5
databricks bundle validate -t dev -p panda-dev

# Uruchom trening v10 (pierwszy krok po otworzeniu nowego chatu)
databricks bundle run weekly_training_manual -t dev -p panda-dev

# Sprawdź status jobów
databricks jobs list -p panda-dev --output json | python3 -c "
import sys, json
data = json.load(sys.stdin)
jobs = data.get('jobs', [])
for j in jobs:
    print(j['job_id'], j['settings']['name'])
"

# Sprawdź bieżące uruchomienie (po starcie treningu)
databricks runs list --job-id <JOB_ID> -p panda-dev --output json | python3 -c "
import sys,json
data = json.load(sys.stdin)
runs = data.get('runs', [])
for r in runs[:3]:
    print(r.get('run_id'), r.get('state',{}).get('life_cycle_state'), r.get('start_time'))
"
```

### Diagnostyka event_ts (jeśli potrzebna weryfikacja)

```python
# W notebooku Databricks:
from pyspark.sql import functions as F

# 1. Weryfikacja __START_AT
spark.table("panda_silver_dev.occ_ops.netline___schedops__leg").select(
    "__START_AT", "update_key", "entry_dt"
).limit(5).show()

# 2. Aktualny event_ts w ft_leg_status
spark.table("panda_silver_dev.ml_ops.ft_leg_status").select(
    "leg_no", "event_ts"
).limit(5).show()

# 3. Tabele ft_*_daily_* — czy days_since_last_event jest poprawny
spark.table("panda_silver_dev.ml_ops.ft_airport_daily_taxi_out").select(
    "dep_ap_sched", "event_date", "days_since_last_event", "avg_taxi_out_7d"
).filter("dep_ap_sched = 'WAW'").orderBy("event_date", ascending=False).limit(10).show()
```

### Przekazanie kontekstu do nowego chatu

```
Projekt: LOT Polish Airlines ML Pipeline — Feature Store Migration
Stack: Databricks, PySpark, DLT, MLflow UC, HistGradientBoostingRegressor
Repo: ML_Project_v3-main (main branch)
Bundle: panda-dev, workspace adb-5711108594958773.13.azuredatabricks.net
DLT Pipeline: 0107e154-3133-4ec5-a843-fdd499de7400

Stan: 
- Iter1 DONE (v9 champion, daily_scoring działa)
- Iter2 kod DONE (ft_* tabele, streaming, 17 FeatureFunction, days_since_last_event)
- DLT refresh: DONE (wszystkie 9 ft_* zielone)
- ft_leg_* event_ts: PROBLEM (1970, __START_AT = sequential counter nie timestamp)
  → NIE blokuje v10 trainingu (ft_leg_* nie są w _build_feature_lookups)
  → Fix w Iter2.5 po v10

NAJBLIŻSZE AKCJE:
1. Uruchom weekly_training_manual → v10
2. Champion → v10
3. Smoke test 08_smoke_test_plan_a (4 checks)
4. Daily scoring confirm
5. Faza 7: parytet v9 vs v10 (MAE diff < 5%?)
6. Iter2.5: fix ft_leg_* event_ts + cleanup orphans

Ref. dokument: PROJECT_STATUS_HANDOVER.md
```

---

*Koniec dokumentu. Wygenerowany: 2026-06-10.*

---

## SEKCJA 15: Uzupełnienia — rzeczy pominięte w poprzednich sekcjach

### KRYTYCZNY SZCZEGÓŁ: dep_stand/arr_stand w scoring.py

**W bieżącym scoring.py (po Iter2) nadal istnieje ręczny join do tabeli leg_misc.**

Powód: SHADOW_TABLE nie ma kolumn `dep_stand` i `arr_stand` (są w `netline___schedops__leg_misc`, nie w `netline___schedops__leg`). Bez stanowisk nie można zbudować `stand_id_out`/`stand_id_in` → klucze FS lookup dla `ft_stand_daily_out/in` → brak predykcji stand-level.

Przepływ w `microbatch_upsert`:

```python
# 1. Wczytaj aktualne stanowiska z leg_misc (źródło: LABELS_TABLE → panda_silver_prod.occ_ops.netline___schedops__leg_misc)
leg_misc_raw = spark.read.table(settings.LEG_MISC_TABLE)
if "__END_AT" in leg_misc_raw.columns:
    leg_misc_raw = leg_misc_raw.filter(F.col("__END_AT").isNull())
leg_misc_current = (
    leg_misc_raw
    .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
    .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
    .select("leg_no", "dep_stand", "arr_stand")
)

# 2. Dołącz stanowiska do batcha
batch_prep = batch_prep.join(leg_misc_current, on="leg_no", how="left")

# 3. Oblicz klucze stand_id dla FeatureLookup
batch_prep = batch_prep \
    .withColumn("stand_id_out", F.concat_ws("_", "dep_ap_sched", "dep_stand")) \
    .withColumn("stand_id_in",  F.concat_ws("_", "arr_ap_sched", "arr_stand"))
```

`settings.LEG_MISC_TABLE = f"{source_catalog}.{source_schema}.netline___schedops__leg_misc"` = `panda_silver_prod.occ_ops.netline___schedops__leg_misc`

**Dlaczego nie używamy ft_leg_misc (nowej streaming table)?**
Plan Sesji 5 zakładał usunięcie tego joinu. CC (KROK 2) nie wykonał tego bo SHADOW_TABLE nie ma stanowisk i join by się zepsuł. `ft_leg_misc` istnieje ale NIE jest używana w scoringu. Stanowiska wciąż czytane z oryginalnej tabeli Netline.

**Implikacja dla Iter2.5:** Dopiero gdy SHADOW_TABLE zostanie rozszerzona o `dep_stand`/`arr_stand` (np. przez zmianę sync logic) można usunąć ten manualny join i przestać używać `LEG_MISC_TABLE` bezpośrednio.

---

### isLO — definicja

`isLO: int` — binarny flag czy lot jest operowany przez LOT Polish Airlines.

```python
# Obliczany w scoring.py (microbatch) i training (enriched):
.withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
```

Wartość `ac_owner` pochodzi z tabeli `netline___schedops__leg`. `"LO"` = ICAO code LOT Polish Airlines.

Ważne dla modelu bo loty własne LOT vs code-share różnią się dynamicznie bloku czasowego.

---

### marker_1..17 — definicja

`marker_1` do `marker_17` — binarny rozkład kolumny `marker` z Netline.

`marker` to string o długości do 17 znaków. Każda pozycja (1-17) to flaga operacyjna/handlowa:

* `'Y'` → 0 (tak/aktywna)
* `'N'` → 1 (nie/nieaktywna)
* pusty/null → NaN (brakująca wartość — HGBT obsługuje przez specjalny split)

```python
# feature_store.py _add_marker_columns():
for i in range(1, MAX_MARKER_LENGTH + 1):  # MAX_MARKER_LENGTH = 17
    marker_i = f"marker_{i}"
    df = df.withColumn(
        marker_i,
        F.when(F.length(F.col("marker")) >= i,
            F.when(F.substring(F.col("marker"), i, 1) == "Y", 0)
             .when(F.substring(F.col("marker"), i, 1) == "N", 1)
             .otherwise(None)
        ).otherwise(None)
    )
```

Znaczenie konkretnych pozycji jest wewnętrzną dokumentacją Netline (LOT). Dla modelu są traktowane jako czarne skrzynki — HGBT sam odkryje które pozycje mają wartość predykcyjną.

---

### FEATURES_BLOCK — lista cech dla segmentu TOTAL-BLOCK

Segment TOTAL-BLOCK (`actual_block_time_sec`) używa kombinacji wszystkich cech z pozostałych segmentów — jest zbudowany w `load_settings()` jako suma taxi_out + airborne + taxi_in features:

```python
features_block = list(dict.fromkeys(
    features_taxi_out + features_airborne + features_taxi_in
))
# Deduplikowane przez dict.fromkeys() dla zachowania kolejności
# Zawiera: cechy taxi-out + cechy airborne + cechy taxi-in (~160 unikalnych cech)
```

Jeśli w kodzie nie znaleziono `features_block` — jest budowany dynamicznie w `load_settings()` jako `list(dict.fromkeys(features_out + features_air + features_in))`.

---

### _apply_top_k — obsługa rzadkich wartości kategorycznych

```python
def _apply_top_k(df: pd.DataFrame, mappings: dict, settings) -> pd.DataFrame:
    """Redukuje cardinality zmiennych kategorycznych.

    Dla każdej kolumny kategorycznej: wartości spoza top-K → "OTHER".
    Zapobiega overfitting'owi na rzadkich lotniskach, stanowiskach, typach samolotów.
    """
    for col, top_cats in mappings.items():
        out[col] = np.where(out[col].isin(top_cats), out[col], "OTHER")
        out[col] = out[col].astype("category")
    for col in settings.CATEGORICAL_FEATURES:
        if col in out and col not in mappings:
            out[col] = out[col].astype("category")
```

`CATEGORICAL_FEATURES = ["leg_type", "ac_registration", "ac_subtype", "commercial_carrier", "dep_ap_sched", "arr_ap_sched", "dep_stand", "arr_stand"]`

Top-K mappings są budowane podczas treningu na danych treningowych i zapisywane w artefaktach modelu (`model.p["top_deps"]`, `top_arrs`, `top_model_aircraft`, etc.).

---

### model_aircraft_feature_col — jakie kolumny lotnicze

```python
# Domyślnie: ac_registration (rejestracja konkretnego samolotu)
# Top-K aircraft to lista najczęstszych rejestracji
# Rzadkie rejestracje → "OTHER"
# To daje modelowi informację o historii konkretnego samolotu (np. starszy vs nowy)
model_aircraft_feature_col = "ac_registration"
top_model_aircraft = settings.TOP_K_MODEL_AIRCRAFT  # zazwyczaj 100-200 top rejestracji
```

---

### Training data split

```
Pełny dataset (training + walidacja):
  │
  ├── train_pdf: od TRAINING_START_DATE do max_ts - EVAL_DAYS
  │   │
  │   ├── df_fit: od początku do max_ts - CALIBRATION_DAYS (30 dni)
  │   │   └── Trenowanie głównych modeli (HistGradientBoostingRegressor p50/p90)
  │   │
  │   └── df_calib: ostatnie CALIBRATION_DAYS (30 dni) z train_pdf
  │       └── CQR calibration — wyznaczanie shift'u dla p90
  │
  └── valid_pdf: ostatnie EVAL_DAYS (30 dni)
      └── Ewaluacja jakości, quality gate check
```

`EVAL_DAYS = 30` — ostatnie 30 dni CAŁEGO datasetu to holdout evaluation set.
`CALIBRATION_DAYS = 30` — ostatnie 30 dni z DANYCH TRENINGOWYCH (przed eval) to CQR calibration.

**CV split:** W `_evaluate_param_set` używa `TimeSeriesSplit(n_splits=CV_N_SPLITS)` — nie random, ale chronologiczny (fold n zawsze ma dane późniejsze niż fold n-1).

---

### Scoring stream — mechanika CDF

**Trigger:** `availableNow=True` — przetwarza wszystkie dostępne zmiany od ostatniego checkpointu i kończy. Nie jest continuous stream.

**Źródło:** SHADOW_TABLE z `delta.enableChangeDataFeed = true`.

**_change_type filtering:**

```python
batch_df = batch_df.filter(F.col("_change_type") != "update_preimage")
# update_preimage = stara wersja rekordu przed update → odrzucamy
# update_postimage = nowa wersja po update → przetwarzamy
# insert = nowy lot → przetwarzamy
# delete = usunięty lot → przetwarzamy (inactivation logic)
```

**`_commit_version` i `_commit_timestamp`:** metadata CDF — numer commita Delta i jego timestamp.

**Idempotentność:**

```python
update_condition = """
    (t.source_commit_version IS NULL)
    OR (s.source_commit_version > t.source_commit_version)
    OR (
        s.source_commit_version = t.source_commit_version
        AND COALESCE(t.batch_id, -1) < COALESCE(s.batch_id, -1)
    )
"""
```

Oznacza: aktualizuj tylko jeśli nowy commit jest nowszy LUB ten sam commit z późniejszym batch_id. Zapobiega nadpisywaniu predykcji starszymi danymi przy retry.

---

### SHADOW_TABLE sync mechanics

```python
# sync_source_to_shadow() — wywoływana NA POCZĄTKU każdego run_cdf_scoring:
# Źródło: settings.LABELS_TABLE (netline___schedops__leg) — aktywne loty
# Okno: dep_sched_dt z -2 dni do +3 miesięcy od teraz
# Filtr: counter == 0 (tylko podstawowe rekordy), __END_AT IS NULL (aktualna wersja SCD2)

# Merge do SHADOW_TABLE:
delta_shadow.merge(source_df, on=KEY_COL)
    .whenMatchedUpdateAll(condition="row_hash IS NULL OR row_hash != s.row_hash")  # tylko zmiany
    .whenNotMatchedBySourceDelete(condition=window_cond)   # usuń loty z okna które zniknęły
    .whenNotMatchedInsertAll()                              # dodaj nowe loty
```

`row_hash = xxhash64(wszystkie kolumny poza leg_no i __END_AT)` — dedup, nie aktualizuj jeśli dane się nie zmieniły.

---

### Quality gate — jak działa

Quality gate (`05_quality_gate_check.ipynb`) porównuje kandydata z aktualnym championem.

Wywołuje `run_register_best(spark, dbutils, SETTINGS)` który:

1. Pobiera metryki z MLflow dla run_id przekazanego przez task value
2. Pobiera metryki aktualnego championa
3. Sprawdza zestaw kryteriów (`checks`)
4. Zwraca `gates_passed: bool` i `decision: str`

**WAŻNE:** `PROMOTE_IF_PASS` jest zablokowane na `False` — quality gate NIGDY nie promuje automatycznie do championa. To zawsze wymaga ręcznego `set_registered_model_alias`. Taksówka jest: gate PASS = rejestracja wersji w UC, ale alias `champion` ustawiamy ręcznie.

Jeśli quality gate FAIL → job kończy się `RuntimeError` i wersja nie jest rejestrowana.

---

### merge_update_cols w SINK_TABLE

Przy merge do `block_time_predictions_v3`, update obejmuje wszystkie kolumny predykcji PLUS:

* `is_active`, `inactive_reason`, `is_operationally_active`
* `source_commit_version`, `source_commit_timestamp`, `last_change_type`
* `batch_id` (numer mikrobatcha)

Do `EVENTS_SINK_TABLE` (`block_time_predictions_events_v3`) trafia insert-only (append) dla każdej zmiany — pełna historia zmian predykcji per lot.

---

### Zmienne settings — kompletna lista kluczowych wartości

```python
SHADOW_SYNC_LOOKBACK_DAYS  = 2      # jak daleko wstecz sync loty do SHADOW_TABLE
SHADOW_SYNC_LOOKAHEAD_MONTHS = 3    # jak daleko w przód sync loty
EVAL_DAYS                  = 30     # holdout na ewaluację modelu
CALIBRATION_DAYS           = 30     # split na CQR calibration w train danych
CV_N_SPLITS                = 3      # (zmniejszone z 5 z powodu OOM)
MAX_MISSING_FEATURES       = 10     # maks. brakujące cechy przed cold start fallback
MAX_MARKER_LENGTH          = 17     # długość stringa marker → marker_1..17
FS_TIMESTAMP_KEY           = "event_date"  # klucz TIMESERIES dla legacy fs_*
```

---

### asof_date vs event_date w PIT lookup

W obecnym scoring.py (post-Iter2 z `fe.score_batch`):

**`event_date`** (z `ensure_signature_columns` → `F.to_date(event_date)`) — data LOTU (dep_sched_dt jako DATE). Używana przez `fe.score_batch` jako `timestamp_lookup_key` dla daily stats tabel (`ft_*_daily_*`). PIT lookup: "daj mi statystyki dla lotniska WAW sprzed dnia lotu".

**`asof_date`** (= `F.to_date(_commit_timestamp)`) — data KIEDY zmiana trafiła do CDF stream. Może być różna od `event_date` (np. zmiana statusu lotu przyszła 2 dni przed odlotem → `_commit_timestamp` = dziś, `event_date` = za 2 dni).

Dla Iter2 (fe.score_batch) kluczowy jest `event_date`. `asof_date` może być pozostałością legacy Plan B kodu.

---

### Podsumowanie rzeczy do sprawdzenia w nowym chacie

Po otworzeniu nowego chatu, zanim cokolwiek zrobisz:

1. **Czy trening v10 się uruchomił?** Sprawdź UI Workflows → weekly_training_manual
2. **Czy `feature_store.py` ma `dep_stand`/`arr_stand` join z leg_misc?** Szukaj `LEG_MISC_TABLE` w scoring.py — powinien być, jest potrzebny
3. **Czy `ft_leg_*` event_ts jest nadal 1970?** `spark.table("panda_silver_dev.ml_ops.ft_leg_status").select("event_ts").limit(1).show()` — oczekiwane: 1970 (fix w Iter2.5)
4. **Czy DLT pipeline jest zielony?** UI → Delta Live Tables → 0107e154...
5. **Czy notebooki `08`, `09`, `10`, `11` są na workspace?** `/Workspace/Users/30002818@lot.pl/.bundle/ML_Project_v3/dev/files/notebooks/`

---

## SEKCJA 16: Uzupełnienia krytyczne po głębokiej analizie

### KRYTYCZNY: Prawdziwa struktura jobów (resources/job.yml)

W dokumencie w Sekcji 1 podałem błędne liczby tasków. Oto rzeczywista struktura:

**`daily_scoring_manual`** — 5 tasków (nie 3!):

```
environment_contract_check  (notebook 04, ~2 min, max_retries=1)
    ↓
route_fs  ──────────────────── (Feature_Store_Code_v5.ipynb, timeout=7200s, RUN_BOOTSTRAP=True)
stand_fs  ──────────────────── (Feature_Store_Stands.ipynb,  timeout=7200s, RUN_BOOTSTRAP=True)
    ↓ (oba muszą skończyć)
cdf_scoring                    (01_cdf_stream.ipynb, timeout=14400s, RESET_CHECKPOINT=False)
    ↓
prediction_audit_monitoring    (06_prediction_audit_monitoring.ipynb, timeout=3600s)
```

⚠️ **`route_fs` i `stand_fs` to legacy taski** — uruchamiają stare notebooki `Feature_Store_Code_v5.ipynb` i `Feature_Store_Stands.ipynb` które PRZEBUDOWUJĄ tabele `fs_*` (stary Feature Store) przy każdym uruchomieniu scoringu. To jest starszy mechanizm z przed Iter1. DLT pipeline (0107e154...) jest osobnym procesem.

To oznacza: przy każdym `daily_scoring_manual` FS tabele `fs_*` są przebudowywane przez notebook (legacy), a NIE przez DLT pipeline. DLT pipeline istnieje równolegle jako nowy mechanizm ale job.yml jeszcze nie został zaktualizowany.

**To jest potencjalne zadanie do zrobienia w nowym chacie: zaktualizować job.yml żeby `route_fs`/`stand_fs` taski wskazywały na DLT pipeline trigger lub zostały usunięte.**

**`weekly_training_manual`** — 6 tasków (nie 4!):

```
environment_contract_check
    ↓
route_fs ─────────────────── (Feature_Store_Code_v5.ipynb, RUN_BOOTSTRAP=True)
stand_fs ─────────────────── (Feature_Store_Stands.ipynb,  RUN_BOOTSTRAP=True)
    ↓
build_training_set           (01_build_training_set.ipynb, timeout=10800s)
    ↓
train_compare_models         (01_train_compare_models.ipynb, timeout=21600s, 6h limit)
    ↓
quality_gate_check           (05_quality_gate_check.ipynb, PROMOTE_IF_PASS=False)
```

**Dodatkowe joby w resources/job.yml:**

* `register_manual` — 1 task: `register_best` (02_register_best.ipynb, PROMOTE_IF_PASS=True) — ręczna rejestracja modelu
* `evaluate_manual` — 1 task: `evaluate_model` (03_evaluate_model.ipynb, EVAL_MODEL_SOURCE=champion) — ewaluacja aktualnego championa
* `feature_coverage_diagnostics_manual` — 1 task: `feature_coverage_diagnostics` (07_route_fs_coverage_diagnostics.ipynb)
* `daily_scoring_reset_manual` — jak `daily_scoring_manual` ale `RESET_CHECKPOINT=True` — reset całego CDF streamu od zera

**`daily_scoring_reset_manual`** — używać gdy CDF checkpoint jest uszkodzony lub chcemy przeprocesować od nowa.

---

### DLT Pipeline — konfiguracja (resources/pipeline.yml)

```yaml
name: ml_feature_store
development: true
serverless: true
channel: PREVIEW
catalog: ${var.silver_catalog}  # panda_silver_dev
schema:  ${var.silver_schema}   # ml_ops
libraries:
  - glob:
      include: ../src/pipeline/**   # wszystko z src/pipeline/ (feature_store.py, on_demand_functions.py)
configuration:
  ml.source_catalog: panda_silver_prod
  ml.source_schema:  occ_ops
  ml.silver_catalog: panda_silver_dev
  ml.silver_schema:  ml_ops
  ml.history_start:  "2023-07-01"   # ← data startu historii danych
  ml.data_cutoff_date: "2027-01-01" # ← data graniczna danych
  spark.sql.session.timeZone: UTC
notifications:
  - email: r.matysiak@lot.pl
    alerts: on-update-failure, on-update-fatal-failure, on-flow-failure
```

Kluczowe: `ml.history_start = "2023-07-01"` — dane historyczne zaczynają się od 1 lipca 2023. `ml.data_cutoff_date = "2027-01-01"` — filtr końcowy na dane.

---

### HistGradientBoostingRegressor — pełne parametry

```python
# Podczas CV (wybór hiperparametrów):
model = HistGradientBoostingRegressor(
    loss="quantile",
    quantile=0.5,
    random_state=42,
    categorical_features="from_dtype",  # ← używa dtype category z _apply_top_k
    **params,   # max_leaf_nodes, learning_rate, min_samples_leaf, l2_regularization z CV_PARAM_GRID
)

# Finalne modele p50 i p90 (po wyborze best_params):
ml_p50 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.5, random_state=42,
    categorical_features="from_dtype",
    **best_params,
)
ml_p90 = HistGradientBoostingRegressor(
    loss="quantile", quantile=0.9, random_state=42,
    categorical_features="from_dtype",
    **best_params,
)
```

`categorical_features="from_dtype"` — HGBT automatycznie rozpoznaje kolumny z dtype `category` (ustawione przez `_apply_top_k`). Obsługuje je przez natywne split'y kategoryczne zamiast one-hot encoding.

**CV objective (funkcja celu do minimalizacji):**

```python
objective = mae_sec.mean() + BIAS_PENALTY_WEIGHT * bias_sec.abs().mean()
# BIAS_PENALTY_WEIGHT — waga kary za bias (defaultowo np. 0.5)
```

---

### event_ts i event_date w scoring — pełna mechanika

```python
DEP_TS_COL = "dep_sched_dt"   # planowany czas odlotu
ARR_TS_COL = "arr_sched_dt"   # planowany czas przylotu

# W preprocessing mikrobatcha:
df = df.withColumn("event_ts", F.col(DEP_TS_COL).cast("timestamp"))
df = df.withColumn("event_date", F.to_date(F.col(DEP_TS_COL)))
# event_date = data planowanego odlotu → klucz TIMESERIES dla PIT lookup do FS tabel

# scheduled_block_time_sec (obliczany jeśli nie ma go w SHADOW_TABLE):
df = df.withColumn(
    "scheduled_block_time_sec",
    (F.col(ARR_TS_COL).cast("long") - F.col(DEP_TS_COL).cast("long")).cast("double")
)
```

Ważne: `event_date` jest podawany do `ensure_signature_columns` → `F.to_date()` → `fe.score_batch` używa go jako PIT lookup key.

---

### BIAS_PENALTY_WEIGHT i REGISTER_REQUIRED_METRICS

```python
REGISTER_REQUIRED_METRICS = [
    "TOTAL_MAE_actual_block_time",
    "TOTAL_ABS_BIAS_actual_block_time",
]
```

Te metryki muszą być zalogowane w MLflow żeby quality gate mógł je porównać z championem.

---

### ALL_FS_FEATURES

```python
all_fs_features = list(dict.fromkeys(
    features_taxi_out + features_airborne + features_taxi_in
))
```

To pełna deduplikowana lista wszystkich cech modelu (union taxi_out + airborne + taxi_in). Używana do walidacji że training dataset zawiera wszystkie potrzebne kolumny.

---

### Kompletne stałe settings.py — brakujące w poprzednich sekcjach

| Stała                        | Wartość                                                                                             | Opis                                                  |
| ---------------------------- | --------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `SINK_PRIMARY_KEY`           | `"leg_no"`                                                                                          | PK tabeli wynikowej                                   |
| `MODEL_AIRCRAFT_FEATURE_COL` | `"ac_registration"`                                                                                 | Kolumna identyfikująca samolot                        |
| `TOP_K_MODEL_AIRCRAFT`       | `250`                                                                                               | Ile top rejestracji trzymamy (reszta → "OTHER")       |
| `INCLUDED_LEG_TYPES`         | `["J", "C", "G"]`                                                                                   | Typy lotów do scoringu (charter, commercial, general) |
| `MAX_MARKER_LENGTH`          | `17`                                                                                                | Długość stringa marker → marker_1..17                 |
| `CATEGORICAL_FEATURES`       | `[ac_registration, leg_type, commercial_carrier, dep_ap_sched, arr_ap_sched, dep_stand, arr_stand]` | Kolumny traktowane jako kategoryczne przez HGBT       |
| `TRAINING_START_DATE`        | Budowane dynamicznie z `ml.history_start`                                                           | Data początku danych treningowych (2023-07-01)        |

---

### Krytyczna uwaga: job.yml vs DLT pipeline — potencjalny konflikt

**Sytuacja po Iter2:**

* DLT pipeline `0107e154...` zarządza tabelami `fs_*` (legacy) i `ft_*` (nowe) przez `feature_store.py`
* `resources/job.yml` (stary, z Iter0) nadal uruchamia `Feature_Store_Code_v5.ipynb` i `Feature_Store_Stands.ipynb` jako taski `route_fs`/`stand_fs` PRZED każdym scoringiem i treningiem
* Te stare notebooki prawdopodobnie też piszą do `fs_*` tabel — potencjalna kolizja z DLT

**Zalecenie dla nowego chatu:** Sprawdzić zawartość `Feature_Store_Code_v5.ipynb` i `Feature_Store_Stands.ipynb` — czy nadal są potrzebne po migracji do DLT? Prawdopodobnie `job.yml` wymaga aktualizacji aby:

1. Usunąć `route_fs` i `stand_fs` taski (DLT robi to automatycznie)
2. LUB podmienić je na trigger DLT pipeline i czekanie na zakończenie

To jest **Iter2.5 zadanie dodatkowe** które nie było w oryginalnym planie.

---

## SEKCJA 17: Ostatnie uzupełnienia po wyczerpującej analizie

### KRYTYCZNY: scoring.py ma dwie ścieżki — Plan A i stary Plan B!

W pliku `scoring.py` (w wersji z **ML_latest**, czyli PRZED pełnym Iter2) nadal istnieje stary kod Plan B (ręczne joiny do FS tabel):

```python
# Linie ~453-510 — STARA ścieżka Plan B:
fs_out = spark.table(settings.FS_TAXI_OUT_TABLE)
fs_air = spark.table(settings.FS_AIRBORNE_TABLE)
fs_in  = spark.table(settings.FS_TAXI_IN_TABLE)

def join_fs_exact(base, fs_table, join_cols, time_key): ...

batch_prep = join_fs_exact(batch_prep, fs_out, settings.PK_TAXI_OUT, "event_date")
batch_prep = join_fs_exact(batch_prep, fs_air, settings.PK_AIRBORNE, "event_date")
batch_prep = join_fs_exact(batch_prep, fs_in, settings.PK_TAXI_IN, "event_date")

# Komunikat błędu (linia ~552):
print(f"❌ KRYTYCZNY BŁĄD PODCZAS RĘCZNEGO SCORINGU w paczce {batch_id}:")
```

**Po Iter2 (CC Sesja 5+9):** Podstawowa ścieżka powinna być zamieniona na `fe.score_batch` (Plan A). Ale w pliku ZIP który mamy (stary) jest stary kod. W nowym chacie sprawdź czy w aktualnym `scoring.py` jest `fe.score_batch` jako główna ścieżka czy nadal `join_fs_exact`.

Komunikat "RĘCZNEGO SCORINGU" jest historycznym artefaktem nazewnictwa Plan B = "ręczny scoring" vs Plan A = "automatyczny scoring przez fe.score_batch".

---

### MAX_FS_DATE — obsługa lotów odległych w przyszłości

FS tabele daily stats mają dane tylko do teraz (np. do 2026-06-10). Dla lotu który ma odlecieć za 2 miesiące (2026-08-10), `event_date = 2026-08-10` → PIT lookup nie znajdzie statystyk dla tej daty.

Rozwiązanie w scoring.py:

```python
# Pobierz ostatnią dostępną datę w FS:
max_fs_row = spark.table(settings.FS_TAXI_OUT_TABLE)\
    .select(F.max("event_date").alias("max_event_date")).collect()[0]
max_fs_date = max_fs_row["max_event_date"] if max_fs_row else None

# Dla przyszłych lotów użyj ostatniej dostępnej daty FS:
if max_fs_date is not None:
    batch_prep = batch_prep.withColumn(
        "fs_lookup_date",
        F.least(F.col("event_date"), F.lit(max_fs_date))  # min(data lotu, ostatnia FS data)
    )
```

Efekt: loty za 2 miesiące dostają statystyki z dziś (najświeższe dostępne). Nie jest to "prawdziwy" PIT dla przyszłości — ale to jedyne możliwe podejście.

---

### hours_to_departure_at_prediction — kolumna w SINK

```python
"hours_to_departure_at_prediction" = (dep_sched_dt - _commit_timestamp) / 3600.0
```

Ile godzin przed planowanym odlotem była zrobiona predykcja. Kluczowe dla analizy jakości modelu — predykcje zrobione 2h przed odlotem są trafniejsze niż te zrobione 72h wcześniej.

Przechowywana w `SINK_TABLE` i używana w `prediction_audit_monitoring`.

---

### scored_at i prediction_status — kolumny SINK

```python
"scored_at"         # TIMESTAMP — kiedy predykcja była generowana
"prediction_status" # STRING — stan predykcji:
                    # NULL          → normalna predykcja
                    # "COLD_START_FALLBACK"               → brak historii trasy/lotniska
                    # "TOO_MANY_MISSING_FEATURES_FALLBACK" → > MAX_MISSING_FEATURES brakujących
                    # "INACTIVE_OPERATIONAL"               → lot już wylądował/anulowany
                    # "ARR_STATE" / "DIVERSION" / etc.     → powód inactivation
```

---

### prediction_audit_monitoring — co tworzy

Notebook `06_prediction_audit_monitoring.ipynb` (task 5/5 w daily_scoring_manual) tworzy/aktualizuje:

**`panda_silver_dev.ml_ops.block_time_prediction_quality_audit_v1`** — szczegółowy audit porównujący predykcje z rzeczywistością dla lotów które już się odbyły:

* `audit_key`: klucz unikalny
* `leg_no`, `logged_at`
* `pred_*` vs `actual_*` porównania
* `hours_to_departure_at_prediction`
* błędy (MAE, bias per lot)

**`panda_silver_dev.ml_ops.block_time_prediction_monitoring_summary_v1`** — agregaty jakości:

* `sink_rows`: liczba aktywnych predykcji
* `cold_start_share`: % lotów bez historii
* `too_many_missing_share`: % z za dużą liczbą brakujących cech
* `is_operationally_active` share

---

### payload.json — pełna zawartość

`payload.json` to JSON zapisywany jako MLflow artifact, ładowany przez `UltimateSegmentedModel.load_context()`:

```json
{
    "features_out":   ["dep_ap_sched", "dep_stand", ...],
    "features_air":   ["dep_ap_sched", "arr_ap_sched", ...],
    "features_in":    ["arr_ap_sched", "arr_stand", ...],
    "features_block": [...deduplicated union...],
    "model_aircraft_feature_col": "ac_registration",
    "top_model_aircraft": ["SP-LRA", "SP-LRB", ...],  // top 250 rejestracji
    "top_deps":   ["WAW", "KRK", "GDN", ...],         // top lotniska dep
    "top_arrs":   ["WAW", "KRK", "GDN", ...],         // top lotniska arr
    "known_dep_stands": [...],
    "known_arr_stands": [...],
    "weight_out":   0.33,   // = mae_out / (mae_out + mae_air + mae_in)
    "weight_air":   0.33,   // segment z WYŻSZYM MAE = WIĘKSZA waga
    "weight_in":    0.34,
    "cqr_out":   120.5,     // CQR shift dla taxi-out p90 (w sekundach)
    "cqr_air":   180.0,
    "cqr_in":    90.0,
    "cqr_block": 200.0,
    "block_cqr_by_sched_bucket": {"3600": 150.0, "7200": 220.0, ...},
    "block_cqr_buckets": [3600, 7200, ...]  // buckets scheduled_block_time_sec
}
```

**Uwaga o wagach:** `weight_out = mae_out / total_mae`. Segment z WYŻSZYM MAE dostaje WIĘCEJ wagi. To jest celowe — segmenty trudniejsze do przewidzenia mają większy wpływ na finalny block time error.

**BIAS_PENALTY_WEIGHT = 0.10** — CV objective = `mean(MAE) + 0.10 * mean(|bias|)`. Mała kara za bias, głównie minimalizujemy MAE.

---

### ALLOW_CHECKPOINT_RESET — safeguard na produkcji

```python
ALLOW_CHECKPOINT_RESET = (env == "dev")
# True dla dev, False dla prod
```

`daily_scoring_reset_manual` uruchamia `RESET_CHECKPOINT=True`. Jeśli `ALLOW_CHECKPOINT_RESET=False` (prod) → `PermissionError("CRITICAL SECURITY ERROR")`. Zabezpieczenie przed przypadkowym wyczyszczeniem danych produkcyjnych.

**Co robi reset:**

```python
dbutils.fs.rm(settings.CHECKPOINT_PATH, True)          # usuwa checkpoint CDF stream
spark.sql(f"TRUNCATE TABLE {settings.SINK_TABLE}")     # zeruje block_time_predictions_v3
spark.sql(f"TRUNCATE TABLE {settings.EVENTS_SINK_TABLE}")  # zeruje events table
```

Po resecie: następny scoring przetworzy wszystkie dostępne zmiany od początku CDF historii.

---

### Dodatkowe notebooki w projekcie

Pełna lista notebooków (`notebooks/`):

| Notebook                                     | Rola                          | Job                                    |
| -------------------------------------------- | ----------------------------- | -------------------------------------- |
| `01_cdf_stream.ipynb`                        | CDF scoring stream            | daily_scoring_manual (cdf_scoring)     |
| `01_build_training_set.ipynb`                | Buduje training dataset       | weekly_training (build_training_set)   |
| `01_train_compare_models.ipynb`              | Trening modelu                | weekly_training (train_compare_models) |
| `02_register_best.ipynb`                     | Rejestracja modelu            | register_manual                        |
| `03_evaluate_model.ipynb`                    | Ewaluacja modelu              | evaluate_manual                        |
| `04_environment_check.ipynb`                 | Check środowiska              | wszystkie joby (task 1)                |
| `05_quality_gate_check.ipynb`                | Quality gate                  | weekly_training (quality_gate_check)   |
| `06_prediction_audit_monitoring.ipynb`       | Monitoring predykcji          | daily_scoring (task 5)                 |
| `07_route_fs_coverage_diagnostics.ipynb`     | Diagnostyka pokrycia FS       | feature_coverage_diagnostics_manual    |
| `08_data_schema_contract.ipynb`              | Walidacja schematu tabel      | (standalone diagnostic)                |
| `08_smoke_test_plan_a.ipynb`                 | Smoke test Iter2 ✨NEW         | (standalone, nowy z Iter2)             |
| `09_missing_feature_policy_simulation.ipynb` | Symulacja brakujących cech    | (standalone diagnostic)                |
| `09_register_feature_tables.ipynb`           | Rejestracja ft_* w UC FS ✨NEW | (standalone, nowy z Iter2)             |
| `10_register_on_demand_functions.ipynb`      | Rejestracja UDF ✨NEW          | (standalone, nowy z Iter2)             |
| `11_dlt_pipeline_refresh.ipynb`              | Runbook DLT refresh ✨NEW      | (runbook, nowy z Iter2)                |
| `config.ipynb`                               | Współdzielona konfiguracja    | importowany przez inne                 |

**`08_data_schema_contract.ipynb`** — sprawdza że:

* wszystkie wymagane tabele istnieją (LABELS_TABLE, SHADOW_TABLE, SINK_TABLE, FS tabele)
* tabele mają wymagane kolumny
* brak duplikatów kluczy w kluczowych tabelach
* aktualność danych (max dep_sched_dt vs DATA_CUTOFF_DATE)

**`09_missing_feature_policy_simulation.ipynb`** — symuluje scoring z celowo usuniętymi cechami, żeby sprawdzić jak model zachowuje się przy różnych poziomach missing features i walidować politykę `MAX_MISSING_FEATURES=10`.

---

### commercial_carrier — tajemnicza kolumna w CATEGORICAL_FEATURES

`commercial_carrier` jest w `CATEGORICAL_FEATURES = [..., "commercial_carrier", ...]` ale NIE pojawia się w żadnej liście cech (features_taxi_out, features_airborne, features_taxi_in). Efekt: `_apply_top_k` sprawdza `if col in out.columns` — jeśli `commercial_carrier` nie jest w DataFrame, jest cicho pomijana. Nie jest to bug, ale może być historycznym artefaktem (kiedyś była planowana jako cecha, potem usunięta z list feature'ów ale zapomniano ją usunąć z CATEGORICAL_FEATURES). Harmless.

---

### Kompletna lista unikatowych tabel w projekcie

```
panda_silver_prod.occ_ops (READ-ONLY źródła):
  netline___schedops__leg                    # główna tabela lotów
  netline___schedops__leg_times              # czasy OOOI
  netline___schedops__leg_remark             # marker string
  netline___schedops__leg_misc               # stanowiska
  netline___schedops__ap_basics              # lotniska lat/lon
  netline___schedops__time_zone              # strefy czasowe

panda_silver_dev.ml_ops (WRITE — Feature Store + operacyjne):
  fs_taxi_out_features                       # Legacy FS (Iter1, rollback)
  fs_airborne_features
  fs_taxi_in_features
  fs_stand_out_features
  fs_stand_in_features
  ft_leg_status                              # Iter2 streaming ⚠️ event_ts=1970
  ft_leg_times                               # Iter2 streaming ⚠️ event_ts=1970
  ft_leg_misc                                # Iter2 streaming ⚠️ event_ts=1970
  ft_airport_timezone                        # Iter2 ✅
  ft_route_daily_stats                       # Iter2 ✅
  ft_airport_daily_taxi_out                  # Iter2 ✅
  ft_airport_daily_taxi_in                   # Iter2 ✅
  ft_stand_daily_out                         # Iter2 ✅
  ft_stand_daily_in                          # Iter2 ✅
  netline_leg_shadow_cdf_v3                  # SHADOW_TABLE (aktywne loty)
  block_time_training_dataset_v1             # cache training set
  block_time_prediction_quality_audit_v1     # audit predykcji
  block_time_prediction_monitoring_summary_v1 # monitoring summary

panda_gold_dev.ml_ops (WRITE — wyniki):
  block_time_predictions_v3                  # SINK_TABLE — predykcje
  block_time_predictions_events_v3           # EVENTS_SINK — audit log
```

---

### Python i runtime wersje

* Python: **3.12** (DBR 17.3.13 ML)
* Spark: z DBR 17.3.13 ML (Spark 3.5.x)
* sklearn: z DBR ML (HistGradientBoostingRegressor dostępny)
* DLT: `channel: PREVIEW`, `serverless: true` w pipeline.yml
* MLflow: wersja z DBR (UC-backed)

---

### Ostatnia lista rzeczy do sprawdzenia przy wznowieniu

Zanim zaczniesz pracę w nowym chacie, sprawdź:

```python
# 1. Sprawdź status treningu v10
import mlflow
mlflow.set_registry_uri("databricks-uc")
from mlflow import MlflowClient
client = MlflowClient()
versions = client.search_model_versions("name='panda_gold_dev.ml_ops.flight_delay_model'")
for v in sorted(versions, key=lambda x: int(x.version)):
    print(f"v{v.version}: {v.current_stage}, tags={v.tags}, aliases={v.aliases}")

# 2. Sprawdź aktualny champion
alias = client.get_model_version_by_alias("panda_gold_dev.ml_ops.flight_delay_model", "champion")
print(f"Champion: v{alias.version}")

# 3. Sprawdź event_ts w ft_leg_status
spark.table("panda_silver_dev.ml_ops.ft_leg_status").select("event_ts").limit(3).show()
# Oczekiwane: 1970-01-01 (problem znany, fix w Iter2.5)

# 4. Sprawdź rows w ft_*_daily_* (czy są dane)
for t in ["ft_route_daily_stats", "ft_airport_daily_taxi_out", "ft_stand_daily_out"]:
    cnt = spark.table(f"panda_silver_dev.ml_ops.{t}").count()
    print(f"{t}: {cnt:,} rows")

# 5. Sprawdź czy scoring działa
spark.table("panda_gold_dev.ml_ops.block_time_predictions_v3") \
    .filter("is_active = true") \
    .agg({"pred_actual_block_time_sec": "count", "pred_actual_block_time_sec": "mean"}) \
    .show()
```
