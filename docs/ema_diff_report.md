# EMA diff report — `features.py` (usunięty) vs `feature_store.py`

**Cel:** porównać starą logikę EMA z usuniętego `src/ml_project/features.py` z nową w `src/pipeline/feature_store.py`.

## Archeologia gitowa

- `f1aafc4` (2026-04-21) — **"EMA calculation changed"** — commit, który zmienił sposób liczenia EMA *wewnątrz* `features.py`.
- `0ab738d` (2026-06-08) — **"Tomek zmiany"** — commit, który **usunął** `src/ml_project/features.py` (kod EMA został przeniesiony do `src/pipeline/feature_store.py`).
- Stary plik odzyskany: `git show 0ab738d~1:src/ml_project/features.py > /tmp/features_old.py` (574 linie). Wersja ta jest **po** `f1aafc4`, więc zawiera już zmianę EMA.

Funkcje EMA:
- Stara: `create_ema_schema` + `get_ema_compute_function` (`compute_ema_dynamic`) + wiring w `build_route_feature_store`.
- Nowa: `_create_ema_schema` + `_get_ema_compute_function` (`compute_ema_dynamic`) + wiring w `_build_route_feature_store` ([feature_store.py:516-655](../src/pipeline/feature_store.py#L516-L655)).

---

## Sekcja 1 — Stara implementacja (`/tmp/features_old.py`, z `0ab738d~1`)

Rdzeń EMA (`get_ema_compute_function`):

```python
def get_ema_compute_function(entity_cols, target_cols_dict, count_prefix, ema_schema, half_life_days):
    def compute_ema_dynamic(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("event_date").reset_index(drop=True)
        lambdas = {k: math.log(2) / v for k, v in half_life_days.items()}

        states = {prefix: {k: np.nan for k in lambdas.keys()} for prefix in target_cols_dict.values()}
        conf = {k: 0.0 for k in lambdas.keys()}
        last_day = None
        out = {f.name: [] for f in ema_schema.fields}

        for _, row in pdf.iterrows():
            day = int(row["day_num"])
            cnt = float(row["daily_cnt"])
            dt = max(0, day - last_day) if last_day is not None else 0

            current_morning_conf = {}
            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                current_morning_conf[k] = conf[k] * decay if last_day is not None else 0.0

            out["event_date"].append(row["event_date"])
            for c in entity_cols:
                out[c].append(row[c])
            for _, prefix in target_cols_dict.items():
                for k in lambdas.keys():
                    out[f"ema_{prefix}_{k}"].append(states[prefix][k])
            for k in lambdas.keys():
                out[f"ema_confidence_{count_prefix}_{k}"].append(current_morning_conf[k])

            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                conf[k] = current_morning_conf[k] + cnt
                for _, prefix in target_cols_dict.items():
                    y = row[f"daily_avg_{prefix}"]
                    if np.isnan(states[prefix][k]):
                        states[prefix][k] = float(y) if pd.notnull(y) else np.nan
                    else:
                        if pd.notnull(y):
                            states[prefix][k] = (states[prefix][k] * decay) + (1.0 - decay) * float(y)
                        else:
                            states[prefix][k] = states[prefix][k] * decay
            last_day = day
        return pd.DataFrame(out)
    return compute_ema_dynamic
```

Densyfikacja kalendarza zasilająca EMA (po `f1aafc4`):

```python
daily_agg_flights = df.groupBy("event_date", *entity_cols).agg(*daily_cols, F.count("*").alias("daily_cnt"))
daily_agg = (
    entities.crossJoin(calendar)
    .join(daily_agg_flights, on=["event_date", *entity_cols], how="left")
    .fillna({"daily_cnt": 0})
    .withColumn("day_num", F.datediff(F.col("event_date"), F.lit("1970-01-01")))
)
```

---

## Sekcja 2 — Nowa implementacja (`feature_store.py`)

Rdzeń EMA (`_get_ema_compute_function`, [feature_store.py:516-565](../src/pipeline/feature_store.py#L516-L565)):

```python
def _get_ema_compute_function(entity_cols, target_cols_dict, count_prefix, ema_schema, half_life_days):
    def compute_ema_dynamic(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("event_date").reset_index(drop=True)
        lambdas = {k: math.log(2) / v for k, v in half_life_days.items()}

        states = {prefix: {k: np.nan for k in lambdas.keys()} for prefix in target_cols_dict.values()}
        conf = {k: 0.0 for k in lambdas.keys()}
        last_day = None
        out = {field.name: [] for field in ema_schema.fields}

        for _, row in pdf.iterrows():
            day = int(row["day_num"])
            cnt = float(row["daily_cnt"])
            dt = max(0, day - last_day) if last_day is not None else 0

            current_morning_conf = {}
            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                current_morning_conf[k] = conf[k] * decay if last_day is not None else 0.0

            out["event_date"].append(row["event_date"])
            for col_name in entity_cols:
                out[col_name].append(row[col_name])
            for _, prefix in target_cols_dict.items():
                for k in lambdas.keys():
                    out[f"ema_{prefix}_{k}"].append(states[prefix][k])
            for k in lambdas.keys():
                out[f"ema_confidence_{count_prefix}_{k}"].append(current_morning_conf[k])

            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                conf[k] = current_morning_conf[k] + cnt
                for _, prefix in target_cols_dict.items():
                    y = row[f"daily_avg_{prefix}"]
                    if np.isnan(states[prefix][k]):
                        states[prefix][k] = float(y) if pd.notnull(y) else np.nan
                    elif pd.notnull(y):
                        states[prefix][k] = (states[prefix][k] * decay) + (1.0 - decay) * float(y)
                    else:
                        states[prefix][k] = states[prefix][k] * decay
            last_day = day
        return pd.DataFrame(out)
    return compute_ema_dynamic
```

Densyfikacja kalendarza ([feature_store.py:634-641](../src/pipeline/feature_store.py#L634-L641)): **identyczna** (crossJoin entities×calendar, left join daily flights, `fillna({"daily_cnt": 0})`, `day_num = datediff(..., '1970-01-01')`).

---

## Sekcja 3 — Lista różnic

| Aspekt | Stara (`features.py`) | Nowa (`feature_store.py`) | Istotność |
|---|---|---|---|
| Half-life → lambda | `λ = ln(2)/half_life` | identyczne | brak |
| Decay między dniami | `decay = exp(-λ·dt)`, `dt = max(0, day - last_day)` | identyczne | brak |
| Aktualizacja EMA | `state·decay + (1-decay)·y` | identyczne | brak |
| Confidence (waga) | `conf = conf·decay + cnt`, raportowany "morning" (przed dzisiejszym cnt) | identyczne | brak |
| Inicjalizacja stanu | NaN do pierwszego `y`; potem `float(y)` | identyczne | brak |
| Braki danych (`y` null) | gdy stan ustawiony: `state·decay` (sam zanik, bez nowej obserwacji) | identyczne | brak |
| Densyfikacja kalendarza | crossJoin + fillna `daily_cnt=0` (EMA zanika także w dni bez lotów) | identyczne | brak |
| Nazwa funkcji | `get_ema_compute_function` / `create_ema_schema` | `_get_ema_compute_function` / `_create_ema_schema` (prefix `_`) | kosmetyczne |
| Zmienne pętli | `for c in entity_cols`, `out = {f.name...}` | `col_name`, `field.name` | kosmetyczne |
| Struktura warunku braku | `else: if pd.notnull` (zagnieżdżony) | `elif pd.notnull` (spłaszczony) | kosmetyczne, semantycznie identyczne |
| Parametr half_life | `half_life_days` (argument) | `HALF_LIFE_DAYS` (stała modułu) | kosmetyczne (wartości z `settings.HALF_LIFE_DAYS`) |

**Co naprawdę zmienił commit `f1aafc4` ("EMA calculation changed"):** wprowadził densyfikację kalendarza. PRZED: `daily_agg` agregował tylko dni, w których były loty → EMA zanikała wyłącznie skokowo z dnia-z-lotem na kolejny dzień-z-lotem. PO: kalendarz jest pełny (crossJoin entities×calendar, `daily_cnt=0` w dni bez lotów) → EMA i confidence zanikają na **każdy** dzień kalendarzowy, niezależnie od luk. To **materialna** zmiana semantyki zaniku w czasie. **Jest ona zachowana w nowym `feature_store.py`** (linie 634-641 identyczne).

---

## Sekcja 4 — Wniosek

**Stan końcowy `features.py` (pre-deletion) ↔ aktualny `feature_store.py`: matematycznie ekwiwalentne — algorytm EMA jest identyczny.** Jedyne różnice to renaming (prefix `_`, nazwy zmiennych) i spłaszczenie `else/if` → `elif`, bez wpływu na wynik. Densyfikacja kalendarza (kluczowy element liczenia EMA) jest taka sama w obu plikach.

Historyczna zmiana algorytmu zaszła w `f1aafc4` *wewnątrz* `features.py` (przejście na pełny kalendarz / zanik w dni bez lotów) i została w całości przeniesiona do `feature_store.py`. **Przeniesienie `features.py` → `feature_store.py` (commit `0ab738d`) NIE zmieniło logiki EMA** — to refaktor lokalizacji, nie matematyki.

> Parytet predykcji od strony EMA: bez ryzyka regresji wynikającej z migracji do Feature Store. Ewentualne różnice w predykcjach iter1 należy szukać gdzie indziej (np. traktowanie `stand_count_*` NaN, typy DOUBLE), nie w EMA.
