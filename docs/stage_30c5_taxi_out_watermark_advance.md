# Stage 30C-5 Taxi-out Watermark Advance

## Summary

Stage 30C-5 adds a controlled watermark-advance preflight for the
`ft_airport_daily_taxi_out` shadow partial recompute path. It does not mutate
the current materialized view and does not change the Lakeflow pipeline. The
runner is `notebooks/19_stage30c5_taxi_out_watermark_advance.py`.

Defaults are safe:

- `RUN_WATERMARK_ADVANCE = False`;
- `DRY_RUN_ONLY = True`;
- `ALLOW_SHADOW_MERGE = False`;
- `ALLOW_WATERMARK_ADVANCE = False`;
- `ALLOW_WATERMARK_BOOTSTRAP = False`;
- empty `WRITE_CONFIRMATION`.

With `RUN_WATERMARK_ADVANCE = False`, the notebook displays config and exits
before reads or writes.

## Watermark Meaning

A source watermark means all CDF changes for that source up to
`last_processed_version` are reflected in the shadow table. Therefore, advancing
to version `X` is allowed only when every CDF version from the previous
watermark plus one through `X` has been processed, merged to shadow, and
validated.

Watermarks are source-specific:

- `source_alias = leg`;
- `source_alias = leg_times`.

There is no single global watermark because the two Delta sources have
independent commit histories.

## Why Stage 30C-4 Windows Must Not Advance Watermarks

The Stage 30C-4 validation windows proved parity and shadow merge safety, but
they are non-contiguous samples:

- `A_KRK`: `leg 34600-34620`, `leg_times 34519-34538`;
- `B_WAW` / `C_MULTI`: `leg 34680-34700`, `leg_times 34598-34618`.

The gap between those source versions means they do not prove every CDF version
up to `34700` was processed. Stage 30C-5 blocks non-contiguous windows from
watermark advancement by default.

## Bootstrap Requirement

If the watermark table is missing valid rows for both `leg` and `leg_times`,
the runner returns `watermark_bootstrap_required`.

Bootstrap is blocked by default. A bootstrap baseline must be identified
outside the code:

- identify baseline source versions corresponding to the shadow table
  initialization;
- confirm the shadow baseline is equivalent to data through those source
  versions;
- insert initial source-specific watermark rows only after that confirmation;
- do not infer baseline versions from validation windows.

## Source Window Modes

`SOURCE_VERSION_MODE = "watermark"` reads current source-specific watermark rows
and computes the next contiguous window:

```text
start = last_processed_version + 1
end = min(latest_available_version, start + MAX_CDF_VERSION_SPAN_PER_SOURCE - 1)
```

`SOURCE_VERSION_MODE = "explicit"` uses explicit source-specific start/end
parameters. By default, explicit starts must equal the current watermark plus
one. `ALLOW_NON_WATERMARK_EXPLICIT_WINDOW = False` keeps non-contiguous explicit
windows blocked.

If both sources have no new versions, the run is a `watermark_noop`. If one
source has new versions and the other does not, the non-empty source can be
processed, but source-specific version handling remains explicit.

## Processing

Notebook 19 follows the production-like path:

```text
read source-specific CDF windows
-> extract dirty keys
-> map dirty taxi_out events
-> expand D+1...D+30 affected pairs
-> recompute candidate rows
-> compare candidate/current where overlap exists
-> build shadow merge source
-> optional gated shadow merge
-> validate candidate/shadow
-> optional gated watermark advance
```

Unlike validation-only mode, it does not drop affected pairs because they lack
current MV overlap. No-candidate affected keys remain in the merge source for
delete semantics.

## Watermark Advance Gates

Watermark advancement is blocked if any of these are true:

- shadow merge did not run;
- post-merge validation failed;
- candidate duplicate keys exist;
- candidate null keys exist;
- shadow duplicate keys exist;
- shadow null keys exist;
- candidate/current overlap compare has mismatches;
- candidate/shadow compare has mismatches;
- write confirmation is missing;
- watermark table schema or rows are invalid;
- source windows are non-contiguous;
- source-specific versions are missing.

Watermark advancement SQL is a source-specific MERGE into the dev control table
`panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks`. It includes:

- `source_alias`;
- `last_processed_version`;
- `last_processed_timestamp`;
- `updated_at`;
- `updated_by_stage = stage30c5_taxi_out_watermark_advance`;
- `run_id`.

## Rollback And Rerun

Do not advance watermarks on failure. Rerun the same source-specific window
after fixing the cause. If a shadow merge succeeded but watermark advancement
did not, rerun the same window; the shadow merge is keyed and should converge.

## Deferred Scope

Stage 30C-5 remains dev-shadow only. It does not switch readers, mutate the
current MV, change source tables, deploy a bundle, or modify pipeline config.
EMA remains deferred.
