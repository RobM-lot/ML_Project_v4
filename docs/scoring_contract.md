# Scoring contract — 01_cdf_stream

## Cel
Notebook `01_cdf_stream` utrzymuje:
- bieżący stan scoringu w `SINK_TABLE`
- historię zdarzeń scoringowych w `EVENTS_SINK_TABLE`

## Główne przypadki biznesowe

### 1. Normalny scoring
Warunek:
- rekord nie jest ARR
- nie jest diversion
- nie jest delete
- nie jest excluded leg
- nie jest too far future
- nie wpada w fallback scoringowy

Obecne zachowanie:
- model liczy `pred_*`
- rekord jest aktywny
- wynik trafia do `SINK_TABLE`
- event trafia do `EVENTS_SINK_TABLE`

### 2. Cold start fallback
Warunek:
- brak historii 30d w kluczowych sygnałach

Obecne zachowanie:
- fallback do schedule
- `pred_actual_block_time_sec = scheduled_block_time_sec`
- `pred_block_delay_sec = 0`
- segmentowe predykcje = NULL
- `inactive_reason = COLD_START_FALLBACK`
- rekord finalnie schodzi do logiki `is_active = False`

Ryzyko:
- `pred_*` nie oznacza już wyłącznie predykcji modelu
- `is_active` miesza status operacyjny i jakość scoringu

### 3. Too many missing features fallback
Warunek:
- `missing_feature_count > MAX_MISSING_FEATURES`

Obecne zachowanie:
- fallback do schedule
- `pred_actual_block_time_sec = scheduled_block_time_sec`
- `pred_block_delay_sec = 0`
- segmentowe predykcje = NULL
- `inactive_reason = TOO_MANY_MISSING_FEATURES`
- rekord finalnie schodzi do logiki `is_active = False`

Ryzyko:
- `pred_*` nie oznacza już wyłącznie predykcji modelu
- `is_active` miesza status operacyjny i jakość scoringu

### 4. Inaktywacja operacyjna
Przypadki:
- ARR
- diversion
- delete
- excluded leg
- too far future

Obecne zachowanie:
- `is_active = False`
- `pred_* = NULL`
- `pred_block_delay_sec = NULL`
- `scheduled_block_time_sec = NULL`
- ustawiony `inactive_reason`

To zachowanie jest biznesowo bardziej spójne niż fallback scoringowy.

## Aktualne ryzyka

### Ryzyko 1 — `is_active`
Dziś miesza:
- aktywność operacyjną
- jakość scoringu / fallback

### Ryzyko 2 — `pred_actual_block_time_sec`
Dziś czasem oznacza:
- wynik modelu
a czasem:
- plan jako fallback

### Ryzyko 3 — stand features
Stand join został poprawiony na as-of join, co zmniejsza ryzyko sztucznych missingów.

### Ryzyko 4 — merge do sinku
`COALESCE` może zostawiać stare predykcje, gdy logicznie rekord powinien mieć je wyczyszczone.

## Docelowy kierunek

### Docelowo chcemy:
- `is_active` = tylko status operacyjny
- `prediction_status` = status scoringu / fallbacku
- `pred_*` = tylko output modelu
- `effective_*` = wartość użytkowa / fallback
- logicznie odseparować fallback od predykcji modelowej

## Contract migration: legacy vs current fields

The scoring sink currently exposes both legacy compatibility fields and the new contract fields.

### Legacy compatibility fields
The following fields remain in place for backward compatibility with downstream consumers:
- `is_active`
- `inactive_reason`
- `pred_actual_block_time_sec`
- `pred_block_delay_sec`

Their current semantics are preserved temporarily during the migration period.

### Current contract fields
New consumers and monitoring should rely on:
- `is_operationally_active`
- `prediction_status`
- `effective_actual_block_time_sec`
- `effective_block_delay_sec`

### Field semantics

#### `is_operationally_active`
Indicates whether the flight is still an operationally active scoring candidate.
It is independent from fallback behavior.

#### `prediction_status`
Allowed values:
- `MODEL_OK`
- `COLD_START_FALLBACK`
- `TOO_MANY_MISSING_FEATURES_FALLBACK`
- `INACTIVE_OPERATIONAL`

#### `effective_actual_block_time_sec`
Operationally effective predicted block time used by the system:
- `NULL` for operationally inactive flights
- `scheduled_block_time_sec` for scoring fallback
- model prediction for normal scoring

#### `effective_block_delay_sec`
Operationally effective predicted delay used by the system:
- `NULL` for operationally inactive flights
- `0` for scoring fallback
- model predicted delay for normal scoring

### Merge semantics
`SINK_TABLE` is treated as a latest-state sink.
The latest action fully overwrites the current sink state, including explicit `NULL` values.