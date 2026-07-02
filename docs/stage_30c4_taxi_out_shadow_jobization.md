# Stage 30C-4 Taxi-out Shadow Jobization

## Summary

Stage 30C-4 converts the taxi-out shadow partial recompute path into a
job-style validation runner. It is intended for a controlled Databricks Job / Workflow runtime, not for the Lakeflow pipeline definition. The current
`panda_silver_dev.ml_ops.ft_airport_daily_taxi_out` materialized view remains untouched and read-only.

The implementation keeps batch CDF polling as the source-change mechanism and
does not resume readStream investigation. It keeps the shadow-first execution
model and does not implement EMA.

## Why Stage 30C-4 Exists

Stage 30C-3 manual validation showed:

- KRK window passed candidate/current and candidate/shadow parity;
- WAW window passed candidate/current and candidate/shadow parity;
- a no-filter multi-entity run was technically safe, but candidate rows and
  current scoped rows were both zero.

The Run C lesson is that caps without overlap can create a false validation pass. A validation sample must prefer affected entity/date pairs with actual
candidate/current overlap before it can count as parity evidence.

## Notebook 18

`notebooks/18_stage30c4_taxi_out_shadow_job.py` is the job-style runner. Safe
defaults:

- `RUN_JOB = False`;
- `JOB_MODE = "validation"`;
- `SOURCE_VERSION_MODE = "explicit"`;
- `DRY_RUN_ONLY = True`;
- `ALLOW_SHADOW_MERGE = False`;
- `ALLOW_WATERMARK_ADVANCE = False`;
- empty `WRITE_CONFIRMATION`.

With `RUN_JOB = False`, the notebook displays config and exits before reads or
writes.

Supported job modes:

- `validation`: overlap-aware validation only;
- `shadow_merge`: optional dev-shadow merge after validation gates;
- `watermark_advance`: watermark advancement only after full success and
  explicit gates.

## Multi-window Execution

The runner supports a Python default `VALIDATION_WINDOWS` list and a
`VALIDATION_WINDOWS_JSON` override for Databricks Job parameters. Each window
uses source-specific versions:

- `leg_start` / `leg_end`;
- `leg_times_start` / `leg_times_end`.

There is no global CDF version because `leg` and `leg_times` have independent
Delta commit histories.

The default windows are:

- `A_KRK`;
- `B_WAW`;
- `C_MULTI_OVERLAP_AWARE`.

## Overlap-aware Validation

Validation mode computes affected pairs, candidate rows, and current scoped
rows before final entity capping. It then tags affected pairs with:

- `has_candidate`;
- `has_current_mv_key`;
- `has_validation_overlap`.

`has_validation_overlap` means the affected `(dep_ap_sched, event_date)` has
both a candidate row and a current MV key. In validation mode, entities with
overlap are selected first, then `MAX_AFFECTED_ENTITIES_PER_WINDOW` is applied.

If `REQUIRE_VALIDATION_OVERLAP = True` and a window has no candidate/current
overlap, its status is `validation_no_overlap`. It must not count as
`parity_pass`.

This overlap-aware filtering is for validation mode only. A future production
merge must not drop affected pairs just because they lack current MV overlap;
the production merge source still needs no-candidate affected keys for delete
semantics.

## Output

Each window reports:

- `window_id`;
- selected entities;
- CDF counts for `leg` and `leg_times`;
- dirty event count;
- affected pair count;
- candidate row count;
- current scoped row count;
- candidate/current status counts;
- shadow merge execution flag;
- candidate/shadow status counts when shadow is available;
- duplicate/null key counts;
- validation status.

The overall summary reports:

- number of windows;
- number of parity-pass windows;
- number of validated entities;
- total candidate rows;
- total current matched rows;
- total mismatches;
- duplicate/null key presence;
- shadow merge count;
- watermark advancement status;
- final boolean summary.

Default pass criteria require at least two parity windows and at least two
validated entities.

## Write And Watermark Safety

All writes are default off. Shadow merge requires:

- `JOB_MODE = "shadow_merge"` or `JOB_MODE = "watermark_advance"`;
- `DRY_RUN_ONLY = False`;
- `ALLOW_SHADOW_MERGE = True`;
- exact write confirmation string.

Watermark advancement remains disabled by default. It requires:

- `JOB_MODE = "watermark_advance"`;
- `ALLOW_WATERMARK_ADVANCE = True`;
- source-specific latest versions;
- successful shadow merge;
- post-merge validation success;
- no duplicate/null keys;
- no failed compare;
- exact write confirmation string.

Watermarks advance only after full success. No stage in 30C-4 mutates source
tables or the current MV.

## Next Runtime

The next intended runtime is:

```python
RUN_JOB = True
JOB_MODE = "shadow_merge"
SOURCE_VERSION_MODE = "explicit"
DRY_RUN_ONLY = False
ALLOW_SHADOW_MERGE = True
ALLOW_WATERMARK_ADVANCE = False
WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"
```

That run should execute the multi-window shadow validation, compare
candidate/current/shadow where overlap exists, validate duplicate/null keys,
and keep watermarks unadvanced.
