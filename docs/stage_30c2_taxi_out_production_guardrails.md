# Stage 30C-2 Taxi-out Production Guardrails

## Summary

Stage 30C-2 hardens the shadow partial recompute path for
`ft_airport_daily_taxi_out` after the Stage 30C-1 runtime pass. It does not add
a new exploratory notebook and does not change Lakeflow pipeline definitions.
The current materialized view remains untouched.

Runtime status from Stage 30C-1:

- shadow table exists:
  `panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow`;
- watermark table exists:
  `panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks`;
- shadow MERGE passed with `shadow_merge_executed = true`;
- post-merge validation passed with `shadow_post_merge_validation_ok = true`;
- watermark advancement stayed off with `watermarks_advanced = false`;
- candidate/shadow compare matched 5 rows with no mismatches.

Genie recommendation is a GO for the shadow-to-target pattern and a NO-GO for direct production target mutation now. The
future runtime should be a separate Databricks Job, not Lakeflow pipeline code.
Batch CDF polling remains the
direction; readStream investigation stays stopped.

## Dirty And Delete Semantics

Stage 30C-2 makes dirty-side semantics explicit in
`src/ml_project/stage30c_taxi_out_shadow.py`.

For `leg` CDF:

- `insert` and `update_postimage` can create or refresh new affected pairs;
- `update_preimage` and `delete` can create old affected pairs for removal and
  move cases;
- extraction must not rely only on the current postimage;
- `ARR -> non-ARR` marks the old eligible `(dep_ap_sched, event_date)`;
- `non-ARR -> ARR` marks the new eligible `(dep_ap_sched, event_date)`;
- airport moves mark both the old and new entities;
- date moves mark both the old and new event dates.

For `leg_times` CDF:

- leg_times-only dirty rows identify dirty `leg_no`;
- they map through the current eligible `leg` row when available;
- limitation: old `dep_ap_sched` / event-date mapping cannot be recovered from
  leg_times alone unless `leg` CDF also has old mapping data or a historical
  `leg` snapshot is available.

The shadow merge source includes `_stage30c_has_candidate`. A false value is
only valid for an explicitly affected key, and it means the single MERGE may
delete that affected key from the shadow table because no candidate row remains.
The guardrail forbids delete rows outside affected pairs.

## Idempotency Guardrails

For a fixed source-specific CDF window and input snapshot:

- dirty event requirements are deterministic;
- affected `D+1...D+30` output keys are deterministic;
- the shadow merge source is deterministic;
- merge source keys must be unique by `dep_ap_sched`, `event_date`;
- rerunning the same CDF window must not create duplicate keys.

The helper `build_shadow_replace_source_rows(...)` mirrors the Spark merge
source in pure Python for deterministic unit tests. The Spark path remains
`build_shadow_replace_source(...)`.

## Watermark Safety Guardrails

`validate_watermark_advance_preconditions(...)` blocks watermark advancement
unless all required conditions are true:

- shadow merge executed;
- post-merge validation passed;
- source-specific latest versions are available for each configured source;
- candidate keys have no duplicates or nulls;
- shadow keys have no duplicates or nulls;
- compare status has not failed;
- exact dev-shadow write confirmation is present.

There is no global watermark. `leg` and `leg_times` use independent source
versions because their Delta commit histories are independent.

## Deferred Scope

Stage 30C-2 still does not implement EMA. EMA remains deferred because it can
propagate beyond `D+30` and requires a separate design.

Stage 30C-2 also does not switch readers, mutate the current MV, update source
tables, deploy a bundle, or modify the Lakeflow pipeline.

## Multi-window Validation Plan

Before any production switch:

- keep the WAW tested window as the first baseline;
- run at least two more entities or an unfiltered controlled sample;
- run at least two separate source-specific CDF windows;
- compare shadow/candidate/current rows where current MV overlap exists;
- verify dirty leg count, affected pairs, candidate rows, duplicate/null keys,
  compare status counts, and source watermark versions;
- keep watermark advancement disabled until these controlled windows pass.

The purpose is controlled job-style validation, not more manual exploratory POC
work.
