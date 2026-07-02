# Stage 30C-2 Taxi-out Job Runbook Skeleton

## Runtime Shape

Run the taxi-out shadow partial recompute as a separate batch Databricks Job,
not inside the Lakeflow pipeline definition. The job should call the same
batch-CDF, affected-key, candidate recompute, validation, shadow merge, and
watermark guardrail code used by the Stage 30C shadow path.

Do not resume readStream investigation for this path. Use batch CDF polling.

## Parameters

Required or expected job parameters:

- `LEG_CDF_STARTING_VERSION`;
- `LEG_CDF_ENDING_VERSION`;
- `LEG_TIMES_CDF_STARTING_VERSION`;
- `LEG_TIMES_CDF_ENDING_VERSION`;
- optional watermark-driven mode flag once control-state selection is wired;
- `DRY_RUN_ONLY`;
- `ALLOW_SHADOW_MERGE`;
- `ALLOW_WATERMARK_ADVANCE`;
- `WRITE_CONFIRMATION`;
- optional `ENTITY_FILTER`;
- optional caps for dirty events and affected entities during controlled
  validation.

## Operational Sequence

1. Read source-specific watermarks.
2. Determine independent `leg` and `leg_times` CDF windows.
3. Read batch CDF for each configured source.
4. Build dirty event requirements, including old-side `update_preimage` and
   delete semantics.
5. Expand dirty event date `D` to affected output dates `D+1...D+30`.
6. Recompute candidate rows from `cleaned_flight_data_full_table`.
7. Validate candidate keys, affected pairs, scoped compares, and shadow target
   assumptions.
8. Merge affected keys into the dev shadow target, or future approved target,
   only when write gates allow it.
9. Post-merge validate shadow rows against candidate rows.
10. Advance source-specific watermarks only after full success.

## Rollback And Retry

- Do not advance watermarks on failure.
- Rerun the same source-specific CDF window after fixing the cause.
- Do not skip to a newer CDF version window after a failed run.
- Use Delta history on the shadow target if a shadow-only rollback is needed.
- Keep current `ft_airport_daily_taxi_out` read-only until a later approved
  switch stage.

## Monitoring

Capture and alert on:

- dirty leg/source count;
- dirty event requirements count;
- affected output pair count;
- candidate row count;
- merge counts when available;
- candidate duplicate/null key counts;
- shadow duplicate/null key counts;
- candidate/current and candidate/shadow compare status counts;
- source-specific processed versions;
- watermark advancement status.

## Guardrails

- Separate Databricks Job, not Lakeflow pipeline.
- Batch CDF polling only.
- No source table mutation.
- No current MV mutation.
- No global watermark.
- No EMA implementation in this stage.
- Watermark advancement only after successful merge and post-merge validation.

## Stage 30C-4 Handoff

Stage 30C-4 adds `notebooks/18_stage30c4_taxi_out_shadow_job.py` as the
job-style runner for this runbook. It supports multiple explicit CDF windows,
`VALIDATION_WINDOWS_JSON` for Job parameters, and overlap-aware validation so
caps cannot select only affected keys with zero candidate/current overlap. The
runner remains separate from the Lakeflow pipeline definition and keeps
watermark advancement disabled by default.

## Stage 30C-5 Handoff

Stage 30C-5 adds `notebooks/19_stage30c5_taxi_out_watermark_advance.py` as a
dedicated watermark-advance preflight. It must not use the non-contiguous
Stage 30C-4 A/B/C validation windows to advance watermarks. Source-specific
watermarks can advance only after a complete contiguous source window has been
processed, shadow-merged, and post-merge validated.
