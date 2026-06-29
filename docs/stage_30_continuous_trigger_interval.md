# Stage 30A: Continuous Pipeline Trigger Interval

Stage 30 is split into three smaller changes so the operational behavior can be
measured before changing feature computation:

- 30A: enable continuous pipeline operation and add a controlled trigger interval
  for the heavy final daily feature materialized views.
- 30B: later proof of concept for a dirty-key pattern on one table, likely
  `ft_airport_daily_taxi_out`.
- 30C: later rollout of the dirty-key pattern to route, taxi-in, and stand
  feature tables if the proof of concept is worthwhile.

This document covers only 30A.

## Event Log Finding

The Lakeflow/DLT event log showed that the final daily feature materialized view
flows were planned as `COMPLETE_RECOMPUTE`. That means the heavy final daily
feature tables may still recompute fully when refreshed.

The affected final daily feature materialized views are:

- `ft_airport_daily_taxi_out`
- `ft_route_daily_stats`
- `ft_airport_daily_taxi_in`
- `ft_stand_daily_out`
- `ft_stand_daily_in`

## What Changes In 30A

30A changes pipeline scheduling behavior only:

- `resources/pipeline.yml` sets `continuous: true` on
  `pipeline_ml_feature_store`.
- The five final daily feature materialized views use per-flow decorator Spark
  config `spark_conf={"pipelines.trigger.interval": "1 hour"}`.

The interval is set on the final flow decorators rather than globally, so source
and intermediate flows keep their existing table properties and do not receive
the final daily trigger config. This keeps the control focused on the heavy final
daily materialized views.

## What Does Not Change

30A intentionally does not change feature logic, helper math, schemas, table
names, expectations, model training, scoring, or model registry behavior.

The final daily feature materialized views are not converted to streaming tables
in this stage. Their current aggregation logic uses daily pre-aggregation,
rolling windows, and pandas EMA computation. Converting those queries to stream
aggregations would be a separate semantic and operational change.

Aggregation on streams is intentionally avoided in 30A because it would require
additional event-time and watermark design. That belongs in a later stage after
the baseline continuous behavior is measured.

Change Data Feed is also not used in 30A. CDF would be part of a dirty-key or
partial recompute design, which is explicitly deferred to 30B or later.

## Expected Behavior

When the pipeline is deployed and run in continuous mode, the five final daily
feature materialized views should refresh on the controlled one-hour interval.

They may still be planned as `COMPLETE_RECOMPUTE`. 30A controls refresh
frequency; it does not make these flows incremental.

The semantic output of the feature tables should remain unchanged. Row contents,
feature formulas, table names, and training/scoring consumers should stay the
same.

## Risks

- Hourly complete recompute can increase compute cost.
- Actual event log behavior must be measured after deployment.
- Runtime and cost must be measured after several hourly refreshes.
- Continuous mode changes pipeline operation even when Python feature logic is
  unchanged.

## Manual Databricks Validation

Run these checks manually after deploying from local changes:

- Verify the pipeline is configured as continuous.
- Inspect event log planning information for the five final daily materialized
  views.
- Confirm whether those flows still show `COMPLETE_RECOMPUTE`.
- Measure runtime and compute cost after several hourly refreshes.
- Compare row counts before and after deployment.
- Compare max `event_date` before and after deployment.
- Confirm training and scoring table names are unchanged.

No Databricks validation is performed as part of this local-only change.

## Rollback

To roll back Stage 30A locally:

- Remove `continuous: true` from `resources/pipeline.yml`.
- Remove `spark_conf=FINAL_DAILY_FEATURE_SPARK_CONF` from the five final daily
  feature materialized views.
- Change their table properties back to `DLT_TABLE_PROPERTIES` if the separate
  final daily table-property constant is no longer useful.
- Redeploy manually if the deployed Databricks pipeline needs to be reverted.
