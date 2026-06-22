# Iter2 — postęp (KOMPLET: Sesje 1-6 + 5.5)

Wykonane: **1-2 (fundament)** + **3 (settings)** + **4 (training)** + **5 (scoring)** +
**5.5 (Opcja A: local time jako FeatureFunction)** + **6 (notebooki + testy)**.
Kod gotowy do recenzji/ZIP → deploy + DLT refresh + retrening v10 jutro.
Brak `bundle deploy`, brak uruchamiania notebooków, nic nie commitowane.

Recenzja Sesji 1-2: OK. Rozstrzygnięcia: __START_AT zostaje (weryfikacja w UI przed refresh);
skipChangeCommits=true OK; daily stats z legacy cleaned_flight_data_full_table jako transition
(pełne odcięcie -> Iter2.5); stand_id binding przez dict lookup_key={"stand_id": "stand_id_out"}
w Sesji 4.

## Sesja 1 — Nowy layout ft_* w feature_store.py: DONE
- Nowe DDL builders: `leg_status_schema_ddl`, `leg_times_schema_ddl`, `leg_misc_schema_ddl`,
  `airport_timezone_schema_ddl`, `daily_stats_schema_ddl`, `stand_daily_schema_ddl`
  (czyste stringi, zsmoke-testowane: PK + TIMESERIES + DOUBLE + `days_since_last_event`).
- Streaming tables (`@dp.table` + `spark.readStream`): `ft_leg_status`, `ft_leg_times`,
  `ft_leg_misc`, `ft_airport_timezone` (stream-static join ap_basics × time_zone).
- Materialized views (statystyki dzienne): `ft_route_daily_stats`, `ft_airport_daily_taxi_out`,
  `ft_airport_daily_taxi_in`, `ft_stand_daily_out`, `ft_stand_daily_in`.
- **B2 — EMA bez densyfikacji**: `_build_daily_stats` usuwa `entities.crossJoin(calendar)`;
  markery tylko na dniach z eventami; EMA liczona po dniach z eventami.
- **`days_since_last_event DOUBLE`** w każdej `ft_*_daily_*` (lag po event_date per encja).
- Legacy `fs_*` zostawione w pliku z bannerem `# LEGACY fs_* (Iter1). TODO Iter2 cleanup`.
- AST OK, `bundle validate -t dev`: OK.

## Sesja 2 — On-demand features (UC Python UDF): DONE
- `src/pipeline/on_demand_functions.py`: `sin_cos_hour/dow/month`, `haversine_km`,
  `is_eastbound`, `duration_ratio` — czyste Python (stdlib math), behavioralnie zsmoke-testowane
  (WAW→KRK ≈ 250 km, sin_cos_hour(6)=(1,0), duration_ratio(0,x)=None, itd.).
- `notebooks/10_register_on_demand_functions.ipynb`: rejestruje 6 UDF przez
  `CREATE OR REPLACE FUNCTION ... LANGUAGE PYTHON`, + komórka weryfikacji (test SELECT).
  Nazwy budowane z `SETTINGS.SILVER_CATALOG/SCHEMA` (niezależne od pól `UC_FN_*`, których
  Sesja 3 jeszcze nie dodała) — notebook działa niezależnie od kolejności sesji.
- AST OK, notebook JSON valid (4 komórki code), `bundle validate`: OK.

## Sesja 3 — settings.py nowe FT_*/UC_FN_*/PK_FT_*: DONE
- Dataclass: dodane pola `FT_LEG_STATUS_TABLE`...`FT_STAND_DAILY_IN_TABLE` (9), `UC_FN_*` (6),
  `PK_FT_AIRPORT_TAXI_OUT/IN`, `PK_FT_ROUTE`, `PK_FT_STAND_OUT/IN`. Stare `FS_*`/`PK_*` nietknięte.
- `PK_FT_STAND_OUT=["stand_id_out"]`, `PK_FT_STAND_IN=["stand_id_in"]` — zgodnie z dict-binding
  ustalonym do Sesji 4 (`lookup_key={"stand_id": "stand_id_out"}`).
- `load_settings()` wypełnia nazwy z `{silver_catalog}.{silver_schema}.<name>`; zweryfikowane
  (np. `panda_silver_dev.ml_ops.ft_route_daily_stats`, `...haversine_km`).
- AST OK, `tests/test_settings.py`: 4 passed, `bundle validate`: OK.

## Sesja 4 — training.py FeatureLookup ft_* + FeatureFunction: DONE
- `_build_feature_lookups` przepisane na ft_*; legacy zachowane jako
  `_build_feature_lookups_legacy_fs` (rollback). Import `FeatureFunction` dodany.
- Zwraca **10 obiektów**: 5 daily-stats lookups (taxi_out/route/taxi_in/stand_out/stand_in)
  + 2 timezone lookups (dep/arr) + 3 FeatureFunction (haversine→distance_km, is_eastbound,
  duration_ratio). Zweryfikowane statycznie (AST): FeatureLookup=7, FeatureFunction=3.
- **stand_id binding = DICT form** `lookup_key={"stand_id": "stand_id_out"}` / `{"stand_id":
  "stand_id_in"}` (PK tabeli `stand_id` ↔ base `stand_id_out/in`). Timezone też dict:
  `{"iata_ap_code": "dep_ap_sched"}` / `arr`.
- **Units fix (poprawka Sesji 1)**: `ft_airport_timezone` zapisuje teraz `lat_deg`/`lon_deg`
  (STOPNIE) zamiast radianów — bo UDF `haversine_km`/`is_eastbound` robią `math.radians` w środku.
  DDL + view zaktualizowane. Legacy `enriched`/`airport_features` (radiany, własny haversine) bez zmian.
- AST OK (training + feature_store), `bundle validate`: OK.

### ⚠️ duration_ratio — LEAKAGE (do rozstrzygnięcia)
`FeatureFunction(duration_ratio, actual_sec=actual_block_time_sec, ...)` bazuje na
`actual_block_time_sec` = LABEL → **przeciek na treningu, null na scoringu**. Włączone per plan,
ale z ostrzeżeniem w kodzie. **NIE dodawać "duration_ratio" do list cech modelu** (settings.features_*;
obecnie ich tam nie ma — model używa rolling-stats `avg_dur_ratio_*`, nie surowego). Decyzja:
wykluczyć z feature setu albo usunąć FeatureFunction. Rekomendacja: usunąć w Sesji 5/6.

### Deferred (NIE w tej sesji — świadomie)
- **sin/cos local_hour/dow/month**: plan odracza FeatureFunction sin/cos do iter3; lokalny czas
  wymaga `dep_utc_offset_min` z FS lookup, więc liczenie po `create_training_set`. NIE wpięte —
  zależy od kontraktu cech retrenowanego modelu v10. Do ustalenia w Sesji 5 (scoring parytet).
- Post-lookup compute w `run_train_compare_models` (plan Sesja 4 krok 3) — NIE dodane z tego
  samego powodu (spekulatywny pandas kod bez ustalonego feature contractu v10).

## Sesja 5 — scoring.py + post-lookup compute: DONE

### Zmiany
- training.py: usunięto FeatureFunction `duration_ratio` (LEAKAGE) → `_build_feature_lookups`
  zwraca teraz **9 obiektów** (7 FeatureLookup + 2 FeatureFunction). Statycznie zweryfikowane.
- scoring.py: nowa funkcja modułowa `_add_local_time_cosine_features(df)` (sin/cos local time +
  dep/arr_local_ts) — WSPÓLNE źródło dla training i scoring (parytet semantyki cech).
- scoring.py: wywołanie `_add_local_time_cosine_features(scored_df)` PO `fe.score_batch` (po
  ekstrakcji predykcji, przed fallback).
- training.py: `from .scoring import _add_local_time_cosine_features` + wywołanie po
  `training_set.load_df()` w `build_training_datasets` (brak cyklu: scoring nie importuje training).
- `ensure_signature_columns` — bez zmian funkcjonalnych (generyczne wg INPUT_TYPES; KROK 5).

### Orphans (do iter2.5 cleanup)
- `on_demand_functions.duration_ratio`, `settings.UC_FN_DURATION_RATIO`, rejestracja duration_ratio
  w notebooks/10 — zostawione (martwy kod nie szkodzi).

### Weryfikacja
- AST OK (scoring, training) · `bundle validate` OK · `pytest` 10 passed
- `_build_feature_lookups` statycznie: FeatureLookup=7, FeatureFunction=2 ✅

### ⚠️ ODSTĘPSTWO od planu — KROK 2 NIE wykonany (premisa fałszywa)
Plan: usunąć manualny join `dep_stand`/`arr_stand` z `leg_misc`, bo "SHADOW_TABLE już ma stand".
**SHADOW_TABLE NIE ma `dep_stand`/`arr_stand`** (schemat: leg_no, leg_state, leg_type, dep/arr_sched_dt,
dep/arr_ap_sched, dep/arr_ap_actual, ac_*, marker, row_hash, __END_AT — bez standów). Stany żyją w
`leg_misc` (≠ `leg`=LABELS), a `sync_source_to_shadow` kopiuje tylko kolumny obecne w schemacie SHADOW.
Usunięcie joinu zerwałoby liczenie `stand_id_out/stand_id_in` (i `fillna`). **Join ZOSTAWIONY.**
Do rozwiązania jeśli chcemy go usunąć: dodać `dep_stand`/`arr_stand` do schematu SHADOW_TABLE + sync
(zmiana dotyka CDF stream — osobny krok).

### ⚠️ RYZYKO PARYTETU — kolejność sin/cos vs score_batch (do weryfikacji w Fazie 7)
`_add_local_time_cosine_features` liczy sin/cos PO `score_batch`, bo wymaga `dep/arr_utc_offset_min`
z `ft_airport_timezone` (lookup dzieje się WEWNĄTRZ score_batch). Skutek: model uruchamiany w
score_batch widzi sin/cos jako brakujące (NaN), a realne wartości dochodzą dopiero do outputu.
Jeśli sin/cos są cechami INPUT modelu → **train/serve skew** (trening: realne sin/cos z
`build_training_datasets`; serving: NaN w predykcji). Analogiczne do zaakceptowanego stand_count NaN.
Pełny fix = sin/cos jako FeatureFunction (plan odracza do iter3) ALBO timezone lookup przed
score_batch. Dodatkowo: signature v10 budowany jest z `training_set_meta.load_df()` w
`run_train_compare_models` (BEZ `_add_local_time_cosine_features`), więc sin/cos mogą nie znaleźć się
w INPUT signature v10 — do potwierdzenia po retreningu. **Zweryfikować w Fazie 7 (parytet predykcji).**

## Sesja 5.5 — Opcja A: local time features jako FeatureFunction: DONE

Rozwiązuje train/serve skew z Sesji 5: local_hour/dow/month + sin/cos liczone teraz przez
FeatureFunction (w `create_training_set` i `score_batch`), więc trening, signature i scoring mają
DOKŁADNIE te same wartości. v10 = parytet z v9 (`enriched()`), ale przez FS+FF zamiast mega-joinu.

### Co zrobione
- on_demand_functions.py: +9 UDF (`local_hour`, `local_dow`, `month_of`, `sin/cos_local_hour`,
  `sin/cos_local_dow`, `sin/cos_month_of`). Helpery `local_hour/dow/month_of` używane wewn.;
  w rejestracji UC ciała sin/cos są INLINE (UC UDF nie woła innych UDF).
- settings.py: +9 `UC_FN_*` (local_hour..cos_month_of), wypełniane w load_settings.
- notebook 10: FUNCTIONS rozszerzone do 15 wpisów (6 + 9), smoke SELECT-y dla nowych
  (local_hour=16, local_dow=1/wt, month_of=6).
- training.py `_build_feature_lookups`: +15 FeatureFunction (5 raw + 10 sin/cos) → **24 obiekty**
  (7 FeatureLookup + 17 FeatureFunction). Docstring zaktualizowany.
- training.py + scoring.py: USUNIĘTO wywołania `_add_local_time_cosine_features` (redundancja
  powodująca skew). Funkcja zostaje jako orphan w scoring.py (import w training.py też usunięty).
- `ensure_signature_columns`: komentarz (bez zmian funkcjonalnych).

### Parytet z v9 (potwierdzony)
- Te same 15 cech czasowych co enriched(); ten sam wzór (`local_ts = sched_dt + utc_offset_min*60`).
- **local_dow: Spark `(dayofweek+5)%7` == Python `weekday()`** — potwierdzone testem
  (2026-06-09 = wtorek → 1; +12h → środa → 2). Test `test_cyclical_matches_raw` potwierdza
  sin/cos == wzór na raw.

### Orphans (iter2.5 cleanup)
- `on_demand_functions.sin_cos_hour/dow/month` (dict return, nieużywane przez FF) + `UC_FN_SIN_COS_*`
- `on_demand_functions.duration_ratio` + `UC_FN_DURATION_RATIO` (z Sesji 5) + rejestracje w nb10
- `scoring._add_local_time_cosine_features` (zdefiniowana, niewywoływana)

### Weryfikacja
- AST OK (4 pliki) · `bundle validate` OK · `pytest` **16 passed** (10 + 6 nowych)
- `_build_feature_lookups`: FeatureLookup=7, FeatureFunction=17 ✅
- `_add_local_time_cosine_features` statycznie: tylko definicja, brak wywołań ✅

### Uwaga (do potwierdzenia w Fazie 7)
- Signature v10 budowany jest z `training_set_meta.load_df()` w `run_train_compare_models`. Ponieważ
  FF są częścią `_build_feature_lookups` (a `_create_fs_training_set` używa tej samej funkcji), to
  load_df() training_set_meta TEŻ powinno zawierać te kolumny → signature je obejmie. Potwierdzić,
  że FeatureFunction działają na próbce limit(256) tak samo jak na pełnym zbiorze (powinny —
  są bezstanowe per-wiersz).

## Sesja 6 — notebooki + cleanup + testy: DONE
- **notebooks/09**: rejestruje **9 ft_*** (leg_status/times/misc + airport_timezone + 5 daily stats)
  zamiast 5 fs_*. `legacy_tables_to_deregister` = 5 fs_* z TODO Iter2.5 (NIE wyrejestrowywane —
  rollback path). Bootstrap print → ft_*.
- **notebooks/08**: Check 1 → 9 ft_* istnieją + count/has_hist/stand_count + `days_since_last_event`
  DOUBLE w daily stats; Check 3 → v10 signature ma **15 cech czasowych z FF** (5 raw + 10 sin/cos);
  Check 4 (score_batch) bez zmian (parytet FF: brak `_add_local_time_cosine_features`).
- **notebooks/11_dlt_pipeline_refresh.ipynb** (NOWY): runbook na dzień deployu — (1) %run 10
  rejestr UDF, (2) `pipelines start-update <id> --full-refresh`, (3) %run 09 rejestr ft_*,
  (4) `bundle run weekly_training_manual`, (5-6 markdown) champion→v10 + smoke 08. Komórki kodu to
  magics (%run / !databricks) — instrukcje, NIE do AST/auto-run.
- **tests/test_feature_store_helpers.py**: +6 testów dla nowych ft_* DDL builderów
  (leg_status/times/misc, airport_timezone w STOPNIACH, daily_stats + route extra_pk, stand_daily —
  wszystkie z `days_since_last_event`/TIMESERIES).
- `feature_store.py` fs_* defs i `_build_feature_lookups_legacy_fs` — NIETKNIĘTE (rollback do Iter2.5).

### Weryfikacja
- Notebooki 08/09/10: JSON OK + code cells AST OK. nb11: JSON OK (4 komendy obecne; magics).
- `pytest` **22 passed** (16 + 6 ft DDL). `bundle validate` OK.

---

## ⚠️ DO RECENZJI — decyzje/ryzyka, których `bundle validate` NIE sprawdza

`bundle validate` waliduje tylko config bundla, **nie wykonuje** grafu DLT. Poniższe potwierdzi
dopiero pełny DLT refresh jutro. Świadome wybory szkieletu:

1. **`event_ts` dla streaming leg_\* = `__START_AT`** (`_SCD2_VERSION_TS`). Założenie: źródła to
   tabele SCD2 (APPLY CHANGES) z parą `__START_AT`/`__END_AT`; `__START_AT` = początek ważności
   wersji → naturalny TIMESERIES dla PIT. **Jeśli kolumny brak** w źródle — zmienić na właściwy
   znacznik wersji. (`__END_AT` jest potwierdzone w obecnym kodzie, `__START_AT` zakładam.)

2. **`readStream` ze źródeł SCD2 z `skipChangeCommits=true`** — bo źródła dostają update/delete
   (stream domyślnie tego nie znosi). Do potwierdzenia, czy nie gubimy istotnych wersji vs.
   `ignoreChanges`.

3. **Daily stats czytają z legacy `cleaned_flight_data_full_table`** (proven DQ + segment times:
   taxi_out_sec/airborne_sec/itd.). Pełne odcięcie od `enriched`/`data_quality` (tabele w 100%
   niezależne, liczone z `ft_leg_*`) to osobny krok iter2 — NIE zrobione, żeby nie duplikować
   sprawdzonej logiki DQ na ślepo. `ft_leg_*` istnieją jako warstwa surowa, ale daily stats jeszcze
   z nich nie czytają.

4. **`ft_airport_timezone` jako streaming table** ze stream-static join (ap_basics stream ×
   time_zone broadcast). Stream-stream join wymagałby watermarków; static po prawej jest OK.

5. **Sesja 4 (FeatureLookup) jeszcze nie powstała** — lookup keys: `ft_route_daily_stats` po
   `route_id`, stand po `stand_id` (concat ap_stand), airport po `dep/arr_ap_sched`. Te klucze
   produkują obecne `_add_fs_lookup_keys` (route_id, stand_id_out, stand_id_in) — ale stand PK w
   nowych tabelach to `stand_id` (jedna kolumna), a base df ma `stand_id_out`/`stand_id_in`.
   **Do rozstrzygnięcia w Sesji 4**: rename/binding kluczy stand (lookup_key=["stand_id_out"] ->
   tabela ma "stand_id"). To samo dotyczy parytetu nazw kolumn po stronie modelu.

---

## Pliki zmienione/dodane w tej rundzie
- `src/pipeline/feature_store.py` — dodane ft_* (DDL + build + streaming/MV), legacy fs_* nietknięte
- `src/pipeline/on_demand_functions.py` — NOWY
- `notebooks/10_register_on_demand_functions.ipynb` — NOWY
- `docs/iter2_progress.md` — NOWY (ten plik)

## Następne sesje (po recenzji)
3. settings.py — stałe FT_*_TABLE / PK_FT_* / UC_FN_* (bez usuwania starych)
4. training.py — `_build_feature_lookups` na ft_* + FeatureFunction (rozstrzygnąć klucze stand)
5. scoring.py — dopasowanie do ft_* + delegacja on-demand do FeatureFunction
6. notebooki (08/09 update, 11 dlt refresh) + cleanup + testy
