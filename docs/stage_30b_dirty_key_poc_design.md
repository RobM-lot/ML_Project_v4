# Stage 30B-0: Dirty-key POC Design

Stage 30B is the design path toward reducing full refresh pressure without
turning rolling feature aggregation into stream aggregation. The intended shape
is:

source changes -> detect dirty leg/date/entity keys -> expand affected output
dates -> recompute candidate output in batch -> compare with the current final
materialized view -> only later decide whether and how to write partial updates.

This document covers 30B-0 only: source discovery and POC design. It does not
change production pipelines, feature logic, training, scoring, or table names.

## Phase Split

- 30B-0: source discovery and POC design. Identify whether source or current
  feature-store stream tables can reliably expose dirty keys.
- 30B-1: implement a candidate dirty-key detector after the source is confirmed.
- 30B-2: build a batch recompute candidate for `ft_airport_daily_taxi_out`.
- 30B-3: compare candidate output against the current materialized view.
- 30B-4: decide write strategy, possibly `foreachBatch` or another controlled
  partial-update pattern.

## Why Dirty Keys, Not Streaming Aggregates

Streaming should be used as a trigger for dirty-key discovery, not as the engine
that computes final daily rolling features. The current feature logic performs
daily pre-aggregation, rolling windows, percentile approximations, trend
features, and pandas EMA calculations. Re-implementing that directly on streams
would introduce event-time, state, late-data, watermark, and correction behavior
that is not equivalent to the current materialized view.

The safer design is to use change signals only to identify affected business
keys and dates, then recompute those outputs with bounded batch logic that can
be compared against the existing final table.

CDF or another change feed may be useful later, but only for detecting changed
keys and affected dates. It should not be used to compute final feature
aggregates directly in 30B.

## First POC Target

The first POC target should be `ft_airport_daily_taxi_out` because it has a
single business key, `dep_ap_sched`, and depends on the taxi-out side of the
cleaned flight data. It is simpler than route features, which need a composite
route key, and simpler than stand features, which depend on stand assignment
quality.

The existing `ft_airport_daily_taxi_out` materialized view remains the source of
truth until parity is proven. A 30B candidate should first produce comparable
rows for selected dirty keys and dates, then compare values and null behavior
against the current MV.

## Current Dirty-key Source Risk

The existing `ft_leg_status`, `ft_leg_times`, and `ft_leg_misc` tables may be
insufficient as dirty-key sources. In `feature_store.py`, the stream source
helper reads source tables with `skipChangeCommits=true`. That can make these
tables miss source corrections, deletes, or state transitions such as
NOT-ARR -> ARR.

30B-0 therefore starts with source discovery. Before 30B-1 chooses an input, a
read-only diagnostic notebook should confirm whether source tables expose
usable version or update columns, and whether the current `ft_leg_*` tables
represent recent source changes well enough for dirty-key detection.

## 30B-1 Verified Decisions

Runtime diagnostics confirmed that `ft_leg_*` tables are not primary dirty-key
sources for the first POC. Dirty-key candidates should come from source
`netline___schedops__*` tables.

The first POC target remains `ft_airport_daily_taxi_out`.

The taxi-out POC uses:

- `netline___schedops__leg`
- `netline___schedops__leg_times`

`netline___schedops__leg_misc` is deferred for stand feature POCs.

The primary dirty-key strategy is `UPDATE_KEY`:

- `update_key == __START_AT` in the verified source tables.
- `update_key` is monotonic and suitable for checkpointing.
- `update_key` is batch-level, not row-level. One value can cover many rows.
- Conceptual checkpoint logic is: changed source rows are those where
  `update_key > last_seen_update_key`.
- After filtering, dirty rows must be deduplicated to unique dirty `leg_no` /
  entity / event_date candidates.

Latest/current source row logic is:

- first filter `__END_AT IS NULL`;
- then select latest by `update_key DESC` if multiple rows per key remain.

For taxi-out:

- `event_date = to_date(dep_sched_dt)`;
- entity key is `dep_ap_sched`;
- current-day source data is excluded from production rolling windows;
- affected output dates are `D+1 ... D+30` for a dirty event date `D`.

EMA remains deferred. 30B-1 does not implement EMA partial recompute because EMA
can propagate beyond `D+30`.

No production write or partial recompute strategy is implemented in 30B-1.
30B-2 should build a candidate batch recompute output for
`ft_airport_daily_taxi_out` and compare it with the current MV.

## 30B-2 Candidate Non-EMA Parity Helpers

30B-2 adds local candidate recompute and parity helpers for the first POC target,
`ft_airport_daily_taxi_out`. These helpers use `cleaned_flight_data_full_table`-
shaped input because that is the production source of the current final MV.

The candidate recompute is intentionally entity-scoped for parity safety. It
identifies affected `dep_ap_sched` entities and affected output dates, recomputes
non-EMA taxi-out features for those entities using available cleaned-flight
history, then filters output to the affected entity/date pairs. This is heavier
than the eventual optimized partial strategy, but it avoids prematurely choosing
a minimal bounded input window before parity is proven.

Affected outputs still follow the dirty event date rule:

- dirty source event date `D`;
- affected output dates `D+1 ... D+30`;
- output date `D` is excluded because current rolling windows exclude same-day
  source data.

The comparable 30B-2 feature set is non-EMA only:

- rolling avg/std/p90/min/max;
- rolling counts;
- trend columns;
- `has_hist` flags;
- `days_since_last_event`.

EMA columns and `delta_ema_avg_*` columns are explicitly excluded and deferred.
EMA can propagate beyond `D+30`, so it needs a separate design before any
partial update strategy can be considered.

30B-2 introduces no writes, MERGE, `foreachBatch`, production dirty-key tables,
or pipeline changes. 30B-3 should run a controlled read-only parity check in
Databricks and decide whether the entity-scoped approach is viable. 30B-4 and
any write strategy remain deferred.

## 30B-3 Read-only Taxi-out Parity Notebook

30B-3 adds exactly one controlled read-only parity notebook for
`ft_airport_daily_taxi_out`. The notebook is manually run only; it is not
deployed as a job, pipeline, or production workflow.

The notebook validates a small sample of recent dirty `update_key` batches. It:

- extracts dirty leg candidates from the source `leg` and `leg_times` tables;
- maps dirty legs to taxi-out dirty events;
- expands dirty event date `D` to affected output dates `D+1 ... D+30`;
- builds entity-scoped non-EMA candidate rows from
  `cleaned_flight_data_full_table`-shaped input;
- filters the current `ft_airport_daily_taxi_out` MV to the exact affected
  `dep_ap_sched` / `event_date` pairs;
- compares candidate rows to current MV rows using only non-EMA comparable
  columns.

Filtering the current MV to affected pairs before comparison is critical. The
POC should not compare a small candidate sample against the entire current MV,
because that would create artificial `missing_in_candidate` rows.

30B-3 introduces no writes, MERGE, `foreachBatch`, production tables, or
pipeline changes. EMA and `delta_ema_avg_*` columns remain excluded and
deferred.

Possible outcomes:

- Parity is clean enough: the 30B dirty-key POC is technically viable, but write
  strategy remains deferred.
- Parity mismatches appear: inspect whether differences come from the candidate
  helper, production feature math, data filtering, null behavior, tolerance, or
  EMA-excluded columns.
- Performance is too heavy: partial strategy needs further design before any
  production path.

Recommended project status after 30B-3:

- 30A runtime continuous refresh: complete.
- 30B dirty-key POC through read-only parity: complete.
- 30B/30C production write strategy: deferred.

## Relationship To 30A

30A made the current feature-store pipeline able to run continuously and gave
the heavy final materialized views a controlled one-hour trigger interval. 30B
is a later optimization path that may reduce the need for full recompute of
final feature tables.

The current 30A runtime state was enabled through a manual full-spec pipeline
update workaround. In this environment, Databricks Asset Bundles resolved config
has been observed to drop `continuous`, so 30B-0 does not require bundle deploy
or pipeline changes.

## Affected Date Semantics

Current rolling features exclude same-day data. In `feature_store.py`,
`_build_daily_stats()` and `_build_stand_daily()` both use Spark windows with:

```python
rangeBetween(-7 * SECONDS_IN_DAY, -1)
rangeBetween(-30 * SECONDS_IN_DAY, -1)
```

That means output date `T` uses source daily aggregates before `T`, not same-day
events from `T`.

For a changed source event date `D`:

- 7-day rolling features are likely affected for output dates `D+1 ... D+7`.
- 30-day rolling features are likely affected for output dates `D+1 ... D+30`.
- Output date `D` itself is not affected by rolling windows because same-day
  data is excluded.

For recomputing output date `T`, the rolling batch input range should be at
least `T-30 ... T-1`.

## EMA Semantics

EMA is also shifted by one day in `_get_ema_compute_function()`, so same-day
daily averages are excluded from the output for that same date. However, EMA can
be affected beyond 30 days because the exponentially weighted history continues
past the rolling-window horizon.

Possible EMA strategies for later phases:

- Full entity recompute from the first dirty date onward.
- Bounded approximation with an explicit tolerance and warm-up window.
- Keep EMA under the current MV until rolling-window parity is proven.
- Build separate EMA-specific recompute logic.

30B-0 does not implement EMA partial recompute.

## Open Decisions For 30B-1

- Which source should drive dirty keys: raw source tables, CDF/change feed, or
  current `ft_leg_*` tables if diagnostics prove they are sufficient?
- Which timestamp or version column should define "recent change" per source
  table?
- How should deletes, corrections, and NOT-ARR -> ARR transitions be detected?
- What is the minimum safe recompute window for taxi-out rolling features?
- How should EMA parity be handled before any write strategy is considered?
