# Stage 30C-1 Taxi-out Shadow Partial Recompute

## Summary

Stage 30C-1 implements the first shadow-first partial recompute path for
`ft_airport_daily_taxi_out`. It does not mutate the current materialized view.
The current MV remains a read-only reference, and any optional writes are
limited to dev-only shadow/control tables:

- `panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow`;
- `panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks`.

The notebook defaults to `RUN_SHADOW_PIPELINE = False`, `DRY_RUN_ONLY = True`,
and every write flag defaulting to `False`. The first runtime phase should be a
dry run with explicit source-specific CDF versions.

## Shadow-first Rationale

The shadow path keeps the existing `panda_silver_dev.ml_ops.ft_airport_daily_taxi_out`
materialized view read-only while validating a production-shaped partial update
flow side by side. This avoids changing the current feature serving target
during initial hardening and gives a concrete table where affected-key replace
semantics, validation, retries, and source-specific watermarks can be tested.

The shadow table is initialized as a full copy of the current MV only when
`ALLOW_CREATE_SHADOW_TABLE = True`, `DRY_RUN_ONLY = False`, and the exact write
confirmation string is provided:

```text
I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY
```

## Flow

```text
read source-specific batch CDF windows
-> extract dirty leg/source candidates
-> map to taxi_out dirty entity/date candidates
-> expand dirty event date D to affected output dates D+1...D+30
-> batch recompute non-EMA candidate rows from cleaned_flight_data_full_table
-> validate candidate/current/shadow assumptions
-> build dry-run write plan
-> optional gated shadow table initialization
-> optional gated shadow MERGE replace of affected keys
-> optional gated source-specific watermark advancement after validation
```

CDF is used only as a dirty-key signal. The feature rows are recomputed in
batch from `cleaned_flight_data_full_table`. There is no stream aggregation in
this stage.

## Target Key

The shadow target key is:

- `dep_ap_sched`;
- `event_date`.

Affected outputs are derived from dirty taxi-out event date `D` as `D+1...D+30`.
The same key shape is used for validation, merge source construction, and
shadow replacement.

## Single MERGE Replace Semantics

`build_shadow_replace_source(affected_pairs_df, candidate_df)` produces one
merge source by left-joining affected keys to candidate rows. It adds:

```text
_stage30c_has_candidate
```

That flag enables one idempotent single MERGE into the shadow table:

- matched and `_stage30c_has_candidate = true`: update the feature columns;
- not matched and `_stage30c_has_candidate = true`: insert the candidate row;
- matched and `_stage30c_has_candidate = false`: delete the affected key from
  the shadow table because it no longer has a candidate row;
- not matched and `_stage30c_has_candidate = false`: no-op.

This is safer than a separate delete and insert sequence because retries for
the same affected keys converge to the same shadow state.

## Write Gates

All writes are off by default; in other words, all writes are off by default.
Write mode requires:

- `RUN_SHADOW_PIPELINE = True`;
- `DRY_RUN_ONLY = False`;
- `WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"`;
- the specific operation flag set to `True`.

Operation flags:

- `ALLOW_CREATE_SHADOW_TABLE`;
- `ALLOW_CREATE_WATERMARK_TABLE`;
- `ALLOW_SHADOW_MERGE`;
- `ALLOW_WATERMARK_ADVANCE`.

The helper validation allows write targets only under
`panda_silver_dev.ml_ops.*` and requires the shadow target to contain `shadow`
and the control target to contain `watermark`. Source tables and the current MV
are rejected as write targets.

## Watermarks

Watermarks are source-specific. `leg` and `leg_times` have independent Delta
commit histories, so there must not be a single shared CDF version.

The Stage 30C-1 watermark table schema includes:

- `stage_name`;
- `source_alias`;
- `source_table`;
- `last_processed_version`;
- `last_processed_timestamp`;
- `last_successful_run_id`;
- `updated_at`;
- `status`.

Watermarks advance only after:

- CDF reads succeeded;
- dirty keys were mapped;
- affected outputs were recomputed;
- candidate key validation passed;
- the shadow merge succeeded;
- post-merge shadow/candidate validation passed.

## Idempotency And Retry

For the same source-specific CDF windows and the same cleaned-flight input
snapshot, dirty-key extraction, affected-output expansion, candidate recompute,
and shadow MERGE source construction are deterministic. Re-running the same
window should update, insert, or delete the same affected keys in the dev shadow
table.

If a run fails before watermark advancement, rerun the same `leg` and
`leg_times` CDF version ranges. Do not skip ahead. If only one source was
configured, only that source watermark can be advanced, and only after the
shadow merge and validation for that run succeed.

## Validation

Before any shadow merge, the notebook checks:

- affected pair count;
- candidate row count;
- candidate duplicate `dep_ap_sched` / `event_date` key count;
- candidate null key count;
- expected non-EMA feature columns;
- scoped candidate/current comparison when current MV overlap exists;
- shadow target key uniqueness when the shadow table already exists.

After a shadow merge, it checks:

- scoped shadow duplicate key count;
- scoped shadow null key count;
- scoped candidate/shadow comparison status counts.

Watermark advancement remains blocked unless post-merge validation succeeds.

## Deferred Scope

EMA remains deferred. The existing non-EMA parity path handles rolling
`D+1...D+30` windows, but EMA can propagate farther than 30 days and needs a
separate state or full-entity recompute design.

CDF retention is a limitation. If a requested source-specific version window
has aged out, the run must stop and use a recovery strategy instead of silently
skipping versions. Full-window parity can also be unavailable when CDF retention
only covers recent changes.

## Manual Run Phases

1. Safe config run:
   keep `RUN_SHADOW_PIPELINE = False` and confirm the notebook exits before
   table reads.
2. Dry run with explicit CDF versions:
   set `RUN_SHADOW_PIPELINE = True`, keep `DRY_RUN_ONLY = True`, set
   source-specific CDF windows, and review dirty events, affected pairs,
   candidate rows, validation, and intended write plan.
3. Optional create/init shadow table:
   set `DRY_RUN_ONLY = False`, provide the exact confirmation string, and set
   only `ALLOW_CREATE_SHADOW_TABLE = True`.
4. Optional shadow merge without watermark advancement:
   run with `ALLOW_SHADOW_MERGE = True` and keep
   `ALLOW_WATERMARK_ADVANCE = False`; review post-merge validation.
5. Optional watermark advancement:
   only after successful shadow merge and validation, run with
   `ALLOW_WATERMARK_ADVANCE = True`.

No phase writes to source tables, the current MV, pipeline configuration, or
production notebooks.

## Stage 30C-2 Handoff

Stage 30C-2 adds production guardrails around this shadow path: dirty/delete
semantics for `leg` CDF preimage/postimage rows, leg_times-only mapping
limitations, deterministic merge-source validation, source-specific watermark
preconditions, and a separate Databricks Job runbook. It keeps the current MV read-only and leaves EMA deferred.
