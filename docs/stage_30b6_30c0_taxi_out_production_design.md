# Stage 30B-6 / 30C-0 Taxi-out Partial Recompute Production Design

## Summary

Stage 30B-5 proved the read-only end-to-end flow for
`ft_airport_daily_taxi_out` using batch CDF polling:

- source changes came from `netline___schedops__leg` and
  `netline___schedops__leg_times`;
- dirty `leg_no` values were mapped to taxi-out `dep_ap_sched` / event-date
  candidates;
- dirty event date `D` expanded to affected output dates `D+1 ... D+30`;
- non-EMA candidate rows were recomputed from
  `cleaned_flight_data_full_table`;
- candidate rows were scoped to the affected entity/date pairs and compared to
  the current `ft_airport_daily_taxi_out` materialized view;
- the successful partial-window runtime sample matched all scoped rows.

The next production target is only `ft_airport_daily_taxi_out`. Stage 30B-6 /
30C-0 does not implement production writes. It prepares the production design
and read-only diagnostics needed before a future write stage.

## Proposed Production Flow

```text
read source-specific CDF windows
-> extract dirty leg_no/source
-> map to dirty taxi_out entity/date
-> expand D+1...D+30 affected output dates
-> batch recompute candidate rows from cleaned_flight_data_full_table
-> validate candidate vs target/scoped rows
-> write/replace affected rows in production target in future stage
-> advance watermarks only after successful write + validation
```

This design intentionally does not aggregate on streams. CDF is used only as a
dirty-key signal. Final feature rows are recomputed with batch logic that can be
validated against the existing target semantics.

## Production Scope

In scope for the first production design:

- target table: `panda_silver_dev.ml_ops.ft_airport_daily_taxi_out`;
- CDF sources:
  - `panda_silver_prod.occ_ops.netline___schedops__leg`;
  - `panda_silver_prod.occ_ops.netline___schedops__leg_times`;
- source key: `leg_no`;
- output entity: `dep_ap_sched`;
- event timestamp: `dep_sched_dt`;
- affected output range: `D+1 ... D+30`;
- parity scope: non-EMA columns only.

Out of scope for this stage:

- production write implementation;
- production control tables or checkpoint creation;
- stream/readStream implementation;
- `foreachBatch`;
- source-table changes or CDF enablement;
- production handling for all move/removal edge cases;
- EMA or `delta_ema_avg_*` recompute.

## Source CDF Polling

Each source needs an independent watermark because source commit versions are
table-specific. Never use one global CDF version for both sources. The control
state should be source-specific, for example:

```json
{
  "leg": "<last_processed_leg_commit_version>",
  "leg_times": "<last_processed_leg_times_commit_version>"
}
```

If a run fails, rerun the same source-specific version range. Watermarks must
advance only after the full transaction succeeds: dirty-key extraction, affected
row recompute, validation, future target update, and audit recording.

A future watermark/control table should track at least:

- pipeline or stage name;
- source alias;
- source table full name;
- last processed CDF commit version;
- last processed CDF commit timestamp;
- last attempted CDF commit version;
- last successful run identifier;
- updated timestamp;
- run status;
- error information when useful;
- dirty row count;
- affected output row count;
- validation result;
- commit timestamp for audit.

The future production job should read each source with:

```python
spark.read.option("readChangeFeed", "true")
```

and source-specific `startingVersion` / `endingVersion` values. These are
source-specific commit versions and must not be shared across `leg` and
`leg_times`. Watermarks must advance only after the affected output rows have
been recomputed, validated, and successfully written by the future write stage.

If a CDF window is unavailable because retention has expired, the partial path
must stop and require a full or broader recovery strategy. It must not silently
skip versions.

## Dirty-key Extraction

Batch CDF rows from `leg` and `leg_times` should be normalized to:

- `leg_no`;
- `dirty_source_alias`;
- source `update_key`;
- `_change_type`;
- `_commit_version`;
- `_commit_timestamp`.

Dirty source rows should be deduplicated by `leg_no` and source alias before
mapping to taxi-out events. Source aliases should remain visible as metadata so
diagnostics can explain whether an affected output came from `leg`, `leg_times`,
or both.

CDF change types need explicit production handling:

- `insert` and `update_postimage` can map through the current source row;
- `update_preimage` may be required to detect previous entity/date values;
- deletes or source-state removals need a separate dirty-range strategy;
- ARR -> non-ARR and entity/date moves must mark both previous and current
  affected ranges before production writes are allowed.
- cancellation and other state changes must be treated as potential removals
  from the eligible taxi-out population.

The first production implementation should not claim full correction coverage
until these edge cases are designed and tested.

For `leg_times`, CDF gives dirty `leg_no` values. The current POC maps those
dirty legs to the current eligible `leg` rows to get `dep_ap_sched` and
`dirty_event_date = to_date(dep_sched_dt)`. This is acceptable for the current
POC, but it cannot fully recover an old entity/date if the current `leg` mapping
changed. Production edge-case design must account for that limitation.

## Mapping To Affected Outputs

The current proved mapping uses current/latest source `leg` rows:

- filter to current/open source rows with `__END_AT IS NULL`;
- choose latest per `leg_no` by descending `update_key`;
- require production taxi-out filters:
  - `counter = 0`;
  - `leg_type IN ('J', 'C', 'G')`;
  - `leg_state = 'ARR'`;
  - `dep_ap_sched IS NOT NULL`;
  - `dep_sched_dt IS NOT NULL`;
- derive `dirty_event_date = to_date(dep_sched_dt)`;
- derive affected output dates `dirty_event_date + 1 ... + 30`.

For production readiness, affected outputs must be unique by:

- `dep_ap_sched`;
- `affected_output_date`.

The affected-output metadata should preserve:

- dirty leg count;
- dirty event dates;
- source aliases;
- CDF change types;
- source commit-version range.

## Batch Recompute Candidate

The candidate should be computed from
`panda_silver_dev.ml_ops.cleaned_flight_data_full_table`, not from streams.

For `ft_airport_daily_taxi_out`, the first production candidate remains
entity-scoped:

- identify affected `dep_ap_sched` values;
- recompute non-EMA taxi-out features for those entities over available history;
- filter the output to affected entity/date pairs only.

This is heavier than a minimal rolling input window, but it is safer for the
first production path because it matches the read-only parity approach and keeps
the validation surface small.

## Validation Gates

Before any future production write, the job should validate:

- candidate row count is greater than zero unless there are no dirty keys;
- affected pairs count is recorded;
- source CDF read succeeded for each configured source window;
- required source and CDF columns exist;
- dirty keys were extracted or the run is explicitly empty;
- current source row mapping succeeded for mapped dirty keys;
- affected output pairs are unique;
- candidate rows are unique by `dep_ap_sched` / `event_date`;
- target scoped rows are unique by `dep_ap_sched` / `event_date`;
- candidate/current schemas match for the non-EMA comparable columns;
- numeric differences are within configured tolerance;
- missing candidate/current rows are understood before write;
- null entity/date issues are detected before write;
- a status summary is emitted for audit;
- full affected-window eligibility is satisfied when parity requires comparing
  against the current materialized view horizon.

Watermarks should advance only after these validation gates and the future write
stage both succeed.

## Future Write Strategy

This stage does not implement the production write mechanism. The main options
for a future 30C implementation are:

- Option A: create a separate partial target table for taxi-out and switch
  downstream readers later. This is the safest next implementation because it
  allows side-by-side validation against the current materialized view and
  avoids mutating the existing target during initial production hardening.
- Option B: write into a managed Delta target keyed by `dep_ap_sched` and
  `event_date`. This can be efficient, but it requires stronger transaction,
  concurrency, and rollback design.
- Option C: delete and insert affected keys. This is simple conceptually, but
  it is more fragile around partial failures and concurrent readers.

Recommended next implementation: Option A, a separate partial target table for
`ft_airport_daily_taxi_out`-equivalent rows, with the future target key:

- `dep_ap_sched`;
- `event_date`.

Only after side-by-side validation is stable should the project consider a
direct keyed update strategy against a managed Delta target.

A later stage must still define:

- whether the final target remains a materialized view or gets a separate
  partial-update target table;
- how affected rows are atomically replaced;
- how concurrent full refreshes and partial refreshes are coordinated;
- where production watermarks live;
- how failed runs are retried without skipping source CDF versions;
- how validation evidence is stored.

The first production write stage should be small and reversible. It should not
include EMA until EMA parity has a separate design.

## Idempotency And Retry

The partial recompute path should be deterministic for a given set of
source-specific CDF windows:

- the same `leg` and `leg_times` CDF commit ranges should produce the same dirty
  leg/source candidates;
- dirty event expansion should produce the same `dep_ap_sched` / `event_date`
  target keys;
- candidate recompute should produce the same non-EMA feature rows for the same
  cleaned-flight input snapshot;
- the future target update should be safe to retry for the same affected keys;
- source-specific watermarks should advance only after successful write and
  validation.

If any step fails, rerun the same CDF version ranges. Do not skip ahead to newer
versions and do not advance one source watermark independently after a failed
multi-source run unless the production control design explicitly supports that
state.

## Readiness Diagnostics

`notebooks/16_stage30c0_taxi_out_production_readiness.py` is the read-only
diagnostic for this design. It should answer whether the environment is ready
for a future implementation by checking:

- source table schemas;
- source CDF batch readability for optional configured versions;
- required CDF metadata columns;
- cleaned-flight input schema for non-EMA candidate recompute;
- current target schema and uniqueness on the output key;
- current target event-date horizon;
- affected-window feasibility for configured CDF samples;
- non-EMA parity helper compatibility;
- absence of stream/write requirements in this readiness stage.

No readiness diagnostic creates or modifies tables, jobs, pipelines, source
properties, checkpoints, or production outputs.

## Risks And Open Decisions

- CDF retention may be shorter than the desired recompute safety window.
- Full-window parity is not always possible with recent-only CDF retention.
- `leg` and `leg_times` versions are independent and need separate watermarks.
- Source-specific commit versions need durable control state in a future stage.
- `update_preimage` / `update_postimage` pairs need explicit production
  semantics.
- `leg_times`-only changes have an old-mapping edge case if current `leg`
  entity/date changed.
- Entity/date moves can require recomputing both old and new affected ranges.
- ARR -> non-ARR transitions can require removing previously valid affected
  outputs from partial rows.
- Future target update design must be validated carefully before any direct
  production mutation.
- Performance gain depends on Lakeflow overhead and affected key count.
- EMA remains deferred because it can propagate beyond `D+30`.
- Current parity has covered non-EMA partial-window behavior, not every
  production correction class.

## Acceptance Criteria For Future 30C Production Implementation

- source-specific watermarks for `leg` and `leg_times`;
- deterministic dirty extraction from batch CDF windows;
- `D+1 ... D+30` affected-date expansion;
- batch recompute only;
- idempotent target update for affected `dep_ap_sched` / `event_date` keys;
- validation before watermark advancement;
- no stream aggregations;
- rollback and retry behavior documented;
- EMA explicitly deferred until separately designed.
