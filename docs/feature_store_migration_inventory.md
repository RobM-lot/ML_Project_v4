# Feature Store migration inventory

## Executive summary

The current project works as a batch ML pipeline with manually assembled
features, but it is not yet a native Unity Catalog Feature Engineering flow.

The highest-value migration is not changing decorators first. The first
decision is the feature contract:

- which features are source-state features,
- which features are historical aggregate features,
- which features are deterministic on-demand transforms,
- which timestamp represents feature availability for point-in-time correctness.

No Databricks commands or SQL were run for this analysis.

## Additional findings

1. `src/ml_project/features.py` already has a separate batch path that attempts
   to use `FeatureEngineeringClient` / `FeatureStoreClient`.

   This means the statement "there is no Feature Store registration at all" is
   too broad. The more precise statement is:

   - the Lakeflow pipeline in `src/pipeline/feature_store.py` still creates
     `fs_*` outputs without declared `PRIMARY KEY` / `TIMESERIES` constraints,
   - training and scoring still use manual joins instead of `FeatureLookup`,
   - the model is logged through `mlflow.pyfunc.log_model`, not
     `FeatureEngineeringClient.log_model`,
   - scoring uses `mlflow.pyfunc.spark_udf`, not `FeatureEngineeringClient.score_batch`.

2. The helper path in `src/ml_project/features.py` should not be copied as the
   final pattern without cleanup.

   It passes `timestamp_keys` into `create_table`, which is legacy Workspace
   Feature Store terminology. Current UC Feature Engineering docs use a
   timeseries column / `TIMESERIES` primary key concept. The same module later
   adds primary keys with `ALTER TABLE`, but the helper does not add
   `TIMESERIES` in `_ensure_primary_key_constraint`.

3. `active_only=False` in training does not solve SCD2 correctness.

   `build_training_datasets()` calls `get_cleaned_flight_data(...,
   active_only=False)`, but `get_cleaned_flight_data()` still filters
   `LABELS_TABLE`, `LEG_TIMES_TABLE`, `LEG_REMARK_TABLE`, and `LEG_MISC_TABLE`
   to `__END_AT IS NULL`. This means training can still see current state,
   not historical state as of the observation timestamp.

4. `event_ts = dep_sched_dt` is not necessarily the right timestamp for every
   feature table.

   For flight facts it is a useful flight-time key. For point-in-time feature
   correctness, each table also needs a clear "available at" or "valid at"
   timestamp. These may not always be the same as scheduled departure time.

5. `ft_leg_times` is high-risk for online prediction.

   OOOI fields such as `offblock_dt`, `airborne_dt`, `landing_dt`, and
   `onblock_dt` are targets or post-event observations depending on scoring
   stage. They should not be general online features for pre-departure scoring
   unless the model contract becomes stage-aware.

6. `commercial_carrier` is listed as categorical metadata, but it is not in the
   current segment feature lists in `settings.py`.

   It is kept in raw/evaluation columns, but not explicitly used by the current
   model feature sets unless added elsewhere later.

7. Historical aggregate features are built from completed ARR rows.

   Both `src/ml_project/common.py` and `src/pipeline/feature_store.py` filter
   the base training/feature-store source to `leg_state == "ARR"` before
   deriving OOOI-based labels and aggregates. This is correct for historical
   aggregate generation, but those aggregate rows must be joined strictly
   point-in-time for online scoring.

## Current model feature families

The current segment feature lists are defined in
`src/ml_project/settings.py`.

## Repository validation result

Validated against `src/ml_project/settings.py`, `src/ml_project/common.py`,
`src/ml_project/training.py`, `src/ml_project/scoring.py`, and
`src/pipeline/feature_store.py`.

Feature counts from the repository:

- `FEATURES_TAXI_OUT`: 75 features
- `FEATURES_AIRBORNE`: 89 features
- `FEATURES_TAXI_IN`: 75 features
- `ALL_FS_FEATURES`: 181 unique features, built as the ordered union of the
  three segment lists

Coverage status: all concrete feature names from the segment lists are present
below. Because `ALL_FS_FEATURES` is computed as the union of these three lists,
that also covers every feature in `settings.ALL_FS_FEATURES`.

## Complete segment feature checklist

### `FEATURES_TAXI_OUT`

`dep_ap_sched`, `dep_stand`, `ac_registration`, `leg_type`,
`avg_taxi_out_7d`, `avg_taxi_out_30d`, `std_taxi_out_7d`,
`std_taxi_out_30d`, `trend_taxi_out_7d`,
`delta_ema_avg_taxi_out_7d`, `delta_ema_avg_taxi_out_30d`,
`p90_taxi_out_7d`, `p90_taxi_out_30d`, `avg_dur_ratio_dep_7d`,
`avg_dur_ratio_dep_30d`, `count_dep_30d`, `has_hist_dep_30d`,
`count_dep_7d`, `delta_ema_avg_dur_ratio_dep_30d`,
`delta_ema_avg_dur_ratio_dep_7d`, `ema_confidence_dep_30d`,
`ema_confidence_dep_7d`, `ema_dur_ratio_dep_30d`,
`ema_dur_ratio_dep_7d`, `ema_taxi_out_30d`, `ema_taxi_out_7d`,
`has_hist_dep_7d`, `max_dur_ratio_dep_30d`, `max_dur_ratio_dep_7d`,
`max_taxi_out_30d`, `max_taxi_out_7d`, `min_dur_ratio_dep_30d`,
`min_dur_ratio_dep_7d`, `min_taxi_out_30d`, `min_taxi_out_7d`,
`p90_dur_ratio_dep_30d`, `p90_dur_ratio_dep_7d`,
`std_dur_ratio_dep_30d`, `std_dur_ratio_dep_7d`,
`trend_dur_ratio_dep_7d`, `stand_count_out_7d`,
`stand_avg_taxi_out_7d`, `stand_trend_taxi_out_7d`,
`stand_count_out_30d`, `stand_avg_taxi_out_30d`,
`stand_p10_taxi_out_30d`, `stand_p50_taxi_out_30d`,
`stand_p90_taxi_out_30d`, `stand_std_taxi_out_30d`, `isLO`,
`local_hour_dep`, `local_dow_dep`, `sin_local_hour_dep`,
`cos_local_hour_dep`, `sin_local_dow_dep`, `cos_local_dow_dep`,
`sin_month`, `cos_month`, `marker_1`, `marker_2`, `marker_3`,
`marker_4`, `marker_5`, `marker_6`, `marker_7`, `marker_8`,
`marker_9`, `marker_10`, `marker_11`, `marker_12`, `marker_13`,
`marker_14`, `marker_15`, `marker_16`, `marker_17`.

### `FEATURES_AIRBORNE`

`dep_ap_sched`, `arr_ap_sched`, `ac_registration`, `leg_type`,
`avg_airborne_7d`, `avg_airborne_30d`, `std_airborne_7d`,
`std_airborne_30d`, `trend_airborne_7d`,
`delta_ema_avg_airborne_7d`, `delta_ema_avg_airborne_30d`,
`avg_dur_ratio_route_7d`, `avg_dur_ratio_route_30d`,
`avg_arrival_delay_7d`, `avg_arrival_delay_30d`,
`std_arrival_delay_7d`, `std_arrival_delay_30d`, `count_route_30d`,
`has_hist_route_30d`, `count_route_7d`,
`delta_ema_avg_arrival_delay_30d`,
`delta_ema_avg_arrival_delay_7d`,
`delta_ema_avg_dur_ratio_route_30d`,
`delta_ema_avg_dur_ratio_route_7d`, `ema_airborne_30d`,
`ema_airborne_7d`, `ema_arrival_delay_30d`,
`ema_arrival_delay_7d`, `ema_confidence_route_30d`,
`ema_confidence_route_7d`, `ema_dur_ratio_route_30d`,
`ema_dur_ratio_route_7d`, `has_hist_route_7d`,
`max_airborne_30d`, `max_airborne_7d`, `max_arrival_delay_30d`,
`max_arrival_delay_7d`, `max_dur_ratio_route_30d`,
`max_dur_ratio_route_7d`, `min_airborne_30d`, `min_airborne_7d`,
`min_arrival_delay_30d`, `min_arrival_delay_7d`,
`min_dur_ratio_route_30d`, `min_dur_ratio_route_7d`,
`p90_airborne_30d`, `p90_airborne_7d`, `p90_arrival_delay_30d`,
`p90_arrival_delay_7d`, `p90_dur_ratio_route_30d`,
`p90_dur_ratio_route_7d`, `std_dur_ratio_route_30d`,
`std_dur_ratio_route_7d`, `trend_arrival_delay_7d`,
`trend_dur_ratio_route_7d`, `isLO`, `distance_km`,
`local_hour_dep`, `local_dow_dep`, `local_hour_arr`, `local_dow_arr`,
`is_eastbound`, `sin_local_hour_dep`, `cos_local_hour_dep`,
`sin_local_hour_arr`, `cos_local_hour_arr`, `sin_local_dow_dep`,
`cos_local_dow_dep`, `sin_local_dow_arr`, `cos_local_dow_arr`,
`sin_month`, `cos_month`, `marker_1`, `marker_2`, `marker_3`,
`marker_4`, `marker_5`, `marker_6`, `marker_7`, `marker_8`,
`marker_9`, `marker_10`, `marker_11`, `marker_12`, `marker_13`,
`marker_14`, `marker_15`, `marker_16`, `marker_17`.

### `FEATURES_TAXI_IN`

`arr_ap_sched`, `arr_stand`, `ac_registration`, `leg_type`,
`avg_taxi_in_7d`, `avg_taxi_in_30d`, `std_taxi_in_7d`,
`std_taxi_in_30d`, `trend_taxi_in_7d`,
`delta_ema_avg_taxi_in_7d`, `delta_ema_avg_taxi_in_30d`,
`p90_taxi_in_7d`, `p90_taxi_in_30d`, `avg_dur_ratio_arr_7d`,
`avg_dur_ratio_arr_30d`, `count_arr_30d`, `has_hist_arr_30d`,
`count_arr_7d`, `delta_ema_avg_dur_ratio_arr_30d`,
`delta_ema_avg_dur_ratio_arr_7d`, `ema_confidence_arr_30d`,
`ema_confidence_arr_7d`, `ema_dur_ratio_arr_30d`,
`ema_dur_ratio_arr_7d`, `ema_taxi_in_30d`, `ema_taxi_in_7d`,
`has_hist_arr_7d`, `max_dur_ratio_arr_30d`, `max_dur_ratio_arr_7d`,
`max_taxi_in_30d`, `max_taxi_in_7d`, `min_dur_ratio_arr_30d`,
`min_dur_ratio_arr_7d`, `min_taxi_in_30d`, `min_taxi_in_7d`,
`p90_dur_ratio_arr_30d`, `p90_dur_ratio_arr_7d`,
`std_dur_ratio_arr_30d`, `std_dur_ratio_arr_7d`,
`trend_dur_ratio_arr_7d`, `stand_count_in_7d`,
`stand_avg_taxi_in_7d`, `stand_trend_taxi_in_7d`,
`stand_count_in_30d`, `stand_avg_taxi_in_30d`,
`stand_p10_taxi_in_30d`, `stand_p50_taxi_in_30d`,
`stand_p90_taxi_in_30d`, `stand_std_taxi_in_30d`, `isLO`,
`local_hour_arr`, `local_dow_arr`, `sin_local_hour_arr`,
`cos_local_hour_arr`, `sin_local_dow_arr`, `cos_local_dow_arr`,
`sin_month`, `cos_month`, `marker_1`, `marker_2`, `marker_3`,
`marker_4`, `marker_5`, `marker_6`, `marker_7`, `marker_8`,
`marker_9`, `marker_10`, `marker_11`, `marker_12`, `marker_13`,
`marker_14`, `marker_15`, `marker_16`, `marker_17`.

## Online scoring availability and risk classification

| Feature family | Concrete features covered | Available before ARR for online scoring? | Current risk marker |
| --- | --- | --- | --- |
| Scheduled route/request keys | `dep_ap_sched`, `arr_ap_sched` | Yes, if present in the CDF/request row before scoring | Current implementation reads current source rows with `__END_AT IS NULL`; training needs PIT alignment |
| Stand request keys | `dep_stand`, `arr_stand` | Conditional; stands can be operationally updated and may be missing or revised before ARR | Current `leg_misc` join uses `__END_AT IS NULL`; needs source update timestamp/PIT |
| Aircraft/type request features | `ac_registration`, `leg_type`, `isLO` | Yes, if `ac_registration`, `leg_type`, and `ac_owner` are available in request/source before scoring | `ac_registration` is truncated to prefix; `isLO` is derived from `ac_owner`; current-state reads can leak in historical training |
| Marker features | `marker_1` ... `marker_17` | Conditional; safe only for marker state known at prediction time | Current training derives markers from current `leg.marker` after `__END_AT IS NULL` filtering |
| Calendar/time transforms | `local_hour_dep`, `local_dow_dep`, `local_hour_arr`, `local_dow_arr`, `sin_local_hour_dep`, `cos_local_hour_dep`, `sin_local_hour_arr`, `cos_local_hour_arr`, `sin_local_dow_dep`, `cos_local_dow_dep`, `sin_local_dow_arr`, `cos_local_dow_arr`, `sin_month`, `cos_month` | Yes, if schedule timestamps and airport offset reference are available | Should be on-demand; airport reference validity must be PIT/valid-time correct |
| Geography transforms | `distance_km`, `is_eastbound` | Yes, if airport coordinates are available | Should be on-demand after airport reference lookup |
| Departure airport historical stats | all `*_taxi_out_*`, `*_dur_ratio_dep_*`, `count_dep_*`, `has_hist_dep_*`, `ema_confidence_dep_*` features | Yes only as historical PIT features, never from the same flight | Depend on actual OOOI/post-event labels from prior flights |
| Route historical stats | all `*_airborne_*`, `*_arrival_delay_*`, `*_dur_ratio_route_*`, `count_route_*`, `has_hist_route_*`, `ema_confidence_route_*` features | Yes only as historical PIT features, never from the same flight | Depend on actual OOOI, ARR delay, and post-event labels from prior flights |
| Arrival airport historical stats | all `*_taxi_in_*`, `*_dur_ratio_arr_*`, `count_arr_*`, `has_hist_arr_*`, `ema_confidence_arr_*` features | Yes only as historical PIT features, never from the same flight | Depend on actual OOOI/post-event labels from prior flights |
| Departure stand historical stats | all `stand_*_out_*` features | Yes only as historical PIT features, once the departure stand key is known | Depend on actual taxi-out OOOI and current-state stand joins unless fixed |
| Arrival stand historical stats | all `stand_*_in_*` features | Conditional; only if arrival stand is known before the scoring stage | Depend on actual taxi-in OOOI and current-state stand joins unless fixed |

### Source-state / request-level features

These are currently used directly by the model or needed to compute direct
features.

| Feature family | Current columns | Current source | Proposed treatment |
| --- | --- | --- | --- |
| Flight identity / route | `dep_ap_sched`, `arr_ap_sched` | `leg` | Keep in base training/scoring dataframe; also use for lookup keys |
| Stand | `dep_stand`, `arr_stand` | `leg_misc` joined by `leg_no` | Materialize as PIT source-state table or include in request if available |
| Aircraft | `ac_registration` prefix | `leg` | Keep in base dataframe; derive prefix on-demand if raw registration available |
| Flight type | `leg_type` | `leg` | Keep in base dataframe |
| Owner flag | `isLO` | `leg.ac_owner` | On-demand from `ac_owner` if available; otherwise base feature |
| Marker flags | `marker_1` ... `marker_17` | `leg.marker` | On-demand from `marker` if available; otherwise materialized in leg-status table |

### Reference features

| Feature family | Current columns | Current source | Proposed treatment |
| --- | --- | --- | --- |
| Airport timezone | `dep_utc_offset_min`, `arr_utc_offset_min` | `ap_basics` + `time_zone` | `ft_airport_timezone` or `ft_airport_reference` with validity timestamp |
| Airport coordinates | internal radian lat/lon used for distance | `ap_basics` | Same reference table; derive distance on-demand |

### Historical aggregate features

These should remain materialized, but the current table design should change.

| Current table | Current key | Current feature families | Proposed table |
| --- | --- | --- | --- |
| `fs_taxi_out_features` | `dep_ap_sched`, `event_date` | taxi-out rolling stats, dep duration ratio stats | `ft_airport_daily_taxi_out` or `ft_dep_airport_daily_stats` |
| `fs_airborne_features` | `dep_ap_sched`, `arr_ap_sched`, `event_date` | airborne, arrival delay, route duration ratio stats | `ft_route_daily_stats` with `route_id`, `event_date TIMESERIES` |
| `fs_taxi_in_features` | `arr_ap_sched`, `event_date` | taxi-in rolling stats, arr duration ratio stats | `ft_airport_daily_taxi_in` or `ft_arr_airport_daily_stats` |
| `fs_stand_out_features` | `fs_dep_ap_sched`, `fs_dep_stand`, `fs_event_date` | departure stand taxi-out stats | `ft_stand_daily_out` with `stand_id`, `event_date TIMESERIES` |
| `fs_stand_in_features` | `fs_arr_ap_sched`, `fs_arr_stand`, `fs_event_date` | arrival stand taxi-in stats | `ft_stand_daily_in` with `stand_id`, `event_date TIMESERIES` |

Recommended deterministic keys:

- `route_id = concat(dep_ap_sched, '_', arr_ap_sched)`
- `stand_id = concat(airport, '_', stand)`

This aligns with the Databricks recommendation to keep time series feature table
primary keys small for performant writes and lookups.

## Materialized feature contract proposal

| Materialized feature set | Concrete features covered | Target feature table | Lookup key | `TIMESERIES` key | Timestamp meaning | Online scoring use |
| --- | --- | --- | --- | --- | --- | --- |
| Flight source-state features | `dep_ap_sched`, `arr_ap_sched`, `ac_registration`, `leg_type`, raw inputs for `isLO`, raw `marker` for `marker_1` ... `marker_17` | `ft_leg_status` if not supplied directly in request/base dataframe | `leg_no` | `source_update_ts` or `valid_from_ts`, team decision required | Source update time or valid-from time; not flight event time | Safe only if row version is known before prediction timestamp |
| Stand source-state features | raw `dep_stand`, raw `arr_stand` used for `dep_stand`, `arr_stand` and stand lookup keys | `ft_leg_misc` | `leg_no` | `source_update_ts` or `valid_from_ts`, team decision required | Source update time or valid-from time for stand assignment | Conditional; `arr_stand` may be unavailable before some scoring stages |
| Airport reference features | timezone offset and coordinates needed for `local_hour_*`, `local_dow_*`, `distance_km`, `is_eastbound` | `ft_airport_reference` | `iata_ap_code` | `valid_from_ts` or `valid_since` | Valid-from time for reference data | Safe before ARR if reference row is valid for scheduled timestamp |
| Departure airport taxi-out history | `avg_taxi_out_7d`, `avg_taxi_out_30d`, `std_taxi_out_7d`, `std_taxi_out_30d`, `trend_taxi_out_7d`, `p90_taxi_out_7d`, `p90_taxi_out_30d`, `min_taxi_out_7d`, `min_taxi_out_30d`, `max_taxi_out_7d`, `max_taxi_out_30d`, `ema_taxi_out_7d`, `ema_taxi_out_30d`, `avg_dur_ratio_dep_7d`, `avg_dur_ratio_dep_30d`, `std_dur_ratio_dep_7d`, `std_dur_ratio_dep_30d`, `p90_dur_ratio_dep_7d`, `p90_dur_ratio_dep_30d`, `min_dur_ratio_dep_7d`, `min_dur_ratio_dep_30d`, `max_dur_ratio_dep_7d`, `max_dur_ratio_dep_30d`, `trend_dur_ratio_dep_7d`, `ema_dur_ratio_dep_7d`, `ema_dur_ratio_dep_30d`, `delta_ema_avg_taxi_out_7d`, `delta_ema_avg_taxi_out_30d`, `delta_ema_avg_dur_ratio_dep_7d`, `delta_ema_avg_dur_ratio_dep_30d`, `count_dep_7d`, `count_dep_30d`, `has_hist_dep_7d`, `has_hist_dep_30d`, `ema_confidence_dep_7d`, `ema_confidence_dep_30d` | `ft_airport_daily_taxi_out` | `dep_ap_sched` or normalized `airport_id` | `event_date` | Aggregate event date for prior historical flights. If sparse rows are used, lookup should return latest row not after observation date. | Safe only through PIT lookup; source measures are post-event OOOI labels from prior flights |
| Route history | `avg_airborne_7d`, `avg_airborne_30d`, `std_airborne_7d`, `std_airborne_30d`, `trend_airborne_7d`, `p90_airborne_7d`, `p90_airborne_30d`, `min_airborne_7d`, `min_airborne_30d`, `max_airborne_7d`, `max_airborne_30d`, `ema_airborne_7d`, `ema_airborne_30d`, `avg_arrival_delay_7d`, `avg_arrival_delay_30d`, `std_arrival_delay_7d`, `std_arrival_delay_30d`, `trend_arrival_delay_7d`, `p90_arrival_delay_7d`, `p90_arrival_delay_30d`, `min_arrival_delay_7d`, `min_arrival_delay_30d`, `max_arrival_delay_7d`, `max_arrival_delay_30d`, `ema_arrival_delay_7d`, `ema_arrival_delay_30d`, `avg_dur_ratio_route_7d`, `avg_dur_ratio_route_30d`, `std_dur_ratio_route_7d`, `std_dur_ratio_route_30d`, `p90_dur_ratio_route_7d`, `p90_dur_ratio_route_30d`, `min_dur_ratio_route_7d`, `min_dur_ratio_route_30d`, `max_dur_ratio_route_7d`, `max_dur_ratio_route_30d`, `trend_dur_ratio_route_7d`, `ema_dur_ratio_route_7d`, `ema_dur_ratio_route_30d`, `delta_ema_avg_airborne_7d`, `delta_ema_avg_airborne_30d`, `delta_ema_avg_arrival_delay_7d`, `delta_ema_avg_arrival_delay_30d`, `delta_ema_avg_dur_ratio_route_7d`, `delta_ema_avg_dur_ratio_route_30d`, `count_route_7d`, `count_route_30d`, `has_hist_route_7d`, `has_hist_route_30d`, `ema_confidence_route_7d`, `ema_confidence_route_30d` | `ft_route_daily_stats` | `route_id` | `event_date` | Aggregate event date for prior route history. With sparse rows, `event_date` is the date the aggregate became available. | Safe only through PIT lookup; depends on post-event airborne, delay, and block metrics from prior flights |
| Arrival airport taxi-in history | `avg_taxi_in_7d`, `avg_taxi_in_30d`, `std_taxi_in_7d`, `std_taxi_in_30d`, `trend_taxi_in_7d`, `p90_taxi_in_7d`, `p90_taxi_in_30d`, `min_taxi_in_7d`, `min_taxi_in_30d`, `max_taxi_in_7d`, `max_taxi_in_30d`, `ema_taxi_in_7d`, `ema_taxi_in_30d`, `avg_dur_ratio_arr_7d`, `avg_dur_ratio_arr_30d`, `std_dur_ratio_arr_7d`, `std_dur_ratio_arr_30d`, `p90_dur_ratio_arr_7d`, `p90_dur_ratio_arr_30d`, `min_dur_ratio_arr_7d`, `min_dur_ratio_arr_30d`, `max_dur_ratio_arr_7d`, `max_dur_ratio_arr_30d`, `trend_dur_ratio_arr_7d`, `ema_dur_ratio_arr_7d`, `ema_dur_ratio_arr_30d`, `delta_ema_avg_taxi_in_7d`, `delta_ema_avg_taxi_in_30d`, `delta_ema_avg_dur_ratio_arr_7d`, `delta_ema_avg_dur_ratio_arr_30d`, `count_arr_7d`, `count_arr_30d`, `has_hist_arr_7d`, `has_hist_arr_30d`, `ema_confidence_arr_7d`, `ema_confidence_arr_30d` | `ft_airport_daily_taxi_in` | `arr_ap_sched` or normalized `airport_id` | `event_date` | Aggregate event date for prior historical flights. If sparse rows are used, lookup should return latest row not after observation date. | Safe only through PIT lookup; source measures are post-event OOOI labels from prior flights |
| Departure stand taxi-out history | `stand_count_out_7d`, `stand_count_out_30d`, `stand_avg_taxi_out_7d`, `stand_avg_taxi_out_30d`, `stand_trend_taxi_out_7d`, `stand_p10_taxi_out_30d`, `stand_p50_taxi_out_30d`, `stand_p90_taxi_out_30d`, `stand_std_taxi_out_30d` | `ft_stand_daily_out` | `stand_id` | `event_date` | Aggregate event date for prior stand history | Safe only through PIT lookup once departure stand is known; depends on post-event taxi-out and PIT-correct stand assignment |
| Arrival stand taxi-in history | `stand_count_in_7d`, `stand_count_in_30d`, `stand_avg_taxi_in_7d`, `stand_avg_taxi_in_30d`, `stand_trend_taxi_in_7d`, `stand_p10_taxi_in_30d`, `stand_p50_taxi_in_30d`, `stand_p90_taxi_in_30d`, `stand_std_taxi_in_30d` | `ft_stand_daily_in` | `stand_id` | `event_date` | Aggregate event date for prior stand history | Conditional; safe only through PIT lookup once arrival stand is known |

### On-demand candidates

These are deterministic transformations and should not need persisted feature
tables if their inputs are available from the request/base dataframe or lookup
results.

| Output feature | Required inputs | Notes |
| --- | --- | --- |
| `scheduled_block_time_sec` | `dep_sched_dt`, `arr_sched_dt` | Safe on-demand for training and scoring |
| `month` | `dep_sched_dt` | Can be intermediate, not necessarily model input |
| `sin_month`, `cos_month` | `month` or `dep_sched_dt` | Good on-demand candidate |
| `local_hour_dep`, `local_dow_dep` | `dep_sched_dt`, departure UTC offset | On-demand after airport/timezone lookup |
| `local_hour_arr`, `local_dow_arr` | `arr_sched_dt`, arrival UTC offset | On-demand after airport/timezone lookup |
| `sin_local_hour_*`, `cos_local_hour_*` | local hour | Good on-demand candidate |
| `sin_local_dow_*`, `cos_local_dow_*` | local day of week | Good on-demand candidate |
| `distance_km` | dep/arr coordinates | On-demand after airport reference lookup |
| `is_eastbound` | dep/arr longitude | On-demand after airport reference lookup |
| `isLO` | `ac_owner` | On-demand if `ac_owner` stays in base dataframe |
| `marker_1` ... `marker_17` | `marker` | On-demand if raw marker is available at scoring |
| `feature_age_days` | request timestamp/date, feature row timestamp/date | Add for PIT lookup of sparse daily stats |

`duration_ratio` should not be a request-time on-demand feature for online
pre-arrival scoring because it requires actual block time. It is valid as an
input to historical aggregate computations if those aggregates are strictly
point-in-time correct.

## Proposed feature tables

### `ft_leg_status`

Purpose: base flight state and request-level attributes.

Suggested key:

- `leg_no`
- `source_update_ts TIMESERIES` or `valid_from_ts TIMESERIES`, team decision
  required

Candidate columns:

- `leg_state`
- `leg_type`
- `counter`
- `dep_ap_sched`
- `arr_ap_sched`
- `dep_ap_actual`
- `arr_ap_actual`
- `dep_sched_dt`
- `arr_sched_dt`
- `ac_registration`
- `ac_owner`
- `marker`
- `fn_carrier`
- `fn_number`

Open decision: if scoring receives these columns directly from CDF, this table
may be more important for training consistency and backfills than for online
scoring.

### `ft_leg_misc`

Purpose: stand information with SCD/PIT semantics.

Suggested key:

- `leg_no`
- `source_update_ts TIMESERIES` or `valid_from_ts TIMESERIES`, team decision
  required

Candidate columns:

- `dep_stand`
- `arr_stand`

### `ft_leg_times`

Purpose: post-event facts used for labels and historical aggregate generation.

Suggested key:

- `leg_no`
- `source_update_ts TIMESERIES` or `valid_from_ts TIMESERIES`, team decision
  required

Candidate columns:

- `offblock_dt`
- `airborne_dt`
- `landing_dt`
- `onblock_dt`

Important: do not use post-event values as pre-event online features unless
the scoring contract explicitly supports stage-aware predictions.

#### `ft_leg_times` stage decision

| Column | Current derived features / labels | Pre-departure scoring | Pre-arrival scoring | Safe use |
| --- | --- | --- | --- | --- |
| `offblock_dt` | part of `taxi_out_sec = airborne_dt - offblock_dt`, `actual_block_time_sec = onblock_dt - offblock_dt` | Not safe before off-block | Stage-specific after off-block if the prediction stage explicitly allows it | Label generation, historical aggregate generation, stage-aware model after off-block |
| `airborne_dt` | part of `taxi_out_sec`, `airborne_sec = landing_dt - airborne_dt` | Not safe | Stage-specific after airborne; still not safe before airborne | Label generation, historical aggregate generation, stage-aware model after airborne |
| `landing_dt` | part of `airborne_sec`, `taxi_in_sec = onblock_dt - landing_dt` | Not safe | Stage-specific only after landing; not safe for pre-arrival block-time prediction | Label generation, historical aggregate generation, stage-aware model after landing |
| `onblock_dt` | part of `taxi_in_sec`, `actual_block_time_sec`, `arrival_delay_sec` | Not safe | Not safe until ARR/on-block; this is effectively label/final outcome | Label-only and historical aggregate generation |

Decision: for the current model contract, treat `ft_leg_times` as label-only
and aggregate-input data for completed flights. Do not expose its raw OOOI
columns as ordinary online features for pre-departure or pre-arrival scoring.
If the team wants predictions at multiple operational stages, create separate
stage-specific feature specs and models.

### `ft_airport_reference`

Purpose: airport coordinates and timezone offsets.

Suggested key:

- `iata_ap_code`
- `valid_ts TIMESERIES`

Candidate columns:

- `utc_offset_min`
- `lat_rad`
- `lon_rad`
- `valid_since`
- `valid_until`

### `ft_route_daily_stats`

Purpose: route-level historical aggregates.

Suggested key:

- `route_id`
- `event_date TIMESERIES`

Candidate columns:

- `avg_airborne_7d`, `avg_airborne_30d`
- `std_airborne_7d`, `std_airborne_30d`
- `p90_airborne_7d`, `p90_airborne_30d`
- `min_airborne_*`, `max_airborne_*`
- `avg_arrival_delay_*`, `std_arrival_delay_*`, `p90_arrival_delay_*`
- `avg_dur_ratio_route_*`, `std_dur_ratio_route_*`, `p90_dur_ratio_route_*`
- `ema_airborne_*`, `ema_arrival_delay_*`, `ema_dur_ratio_route_*`
- `ema_confidence_route_*`
- `count_route_*`, `has_hist_route_*`

Change from current design: calculate rows only for dates with actual events,
then use PIT lookup to retrieve the latest known row. Add `feature_age_days` so
the model can learn freshness.

### `ft_airport_daily_taxi_out`

Suggested key:

- `airport_id` or `dep_ap_sched`
- `event_date TIMESERIES`

Candidate columns:

- `avg_taxi_out_*`, `std_taxi_out_*`, `p90_taxi_out_*`
- `min_taxi_out_*`, `max_taxi_out_*`
- `avg_dur_ratio_dep_*`, `std_dur_ratio_dep_*`, `p90_dur_ratio_dep_*`
- `ema_taxi_out_*`, `ema_dur_ratio_dep_*`
- `ema_confidence_dep_*`
- `count_dep_*`, `has_hist_dep_*`

### `ft_airport_daily_taxi_in`

Suggested key:

- `airport_id` or `arr_ap_sched`
- `event_date TIMESERIES`

Candidate columns:

- `avg_taxi_in_*`, `std_taxi_in_*`, `p90_taxi_in_*`
- `min_taxi_in_*`, `max_taxi_in_*`
- `avg_dur_ratio_arr_*`, `std_dur_ratio_arr_*`, `p90_dur_ratio_arr_*`
- `ema_taxi_in_*`, `ema_dur_ratio_arr_*`
- `ema_confidence_arr_*`
- `count_arr_*`, `has_hist_arr_*`

### `ft_stand_daily_out`

Suggested key:

- `stand_id`
- `event_date TIMESERIES`

Candidate columns:

- `stand_count_out_7d`, `stand_count_out_30d`
- `stand_avg_taxi_out_7d`, `stand_avg_taxi_out_30d`
- `stand_p10_taxi_out_30d`
- `stand_p50_taxi_out_30d`
- `stand_p90_taxi_out_30d`
- `stand_std_taxi_out_30d`
- `stand_trend_taxi_out_7d`

### `ft_stand_daily_in`

Suggested key:

- `stand_id`
- `event_date TIMESERIES`

Candidate columns:

- `stand_count_in_7d`, `stand_count_in_30d`
- `stand_avg_taxi_in_7d`, `stand_avg_taxi_in_30d`
- `stand_p10_taxi_in_30d`
- `stand_p50_taxi_in_30d`
- `stand_p90_taxi_in_30d`
- `stand_std_taxi_in_30d`
- `stand_trend_taxi_in_7d`

## Target / label columns

The current target columns are:

- `taxi_out_sec`
- `airborne_sec`
- `taxi_in_sec`
- `actual_block_time_sec`

These should remain label/post-event columns for training and aggregate
generation. They should not be available as ordinary online scoring inputs for
pre-arrival scoring.

Derived post-event columns such as `arrival_delay_sec`, `block_delay_sec`, and
`duration_ratio` are safe only when used to compute historical aggregate rows
that are then joined point-in-time before the prediction timestamp.

## Timestamp contract decisions required

These decisions must be made before changing production code. Without them, a
pipeline can be syntactically Feature Store-native but still semantically leak
future state into training or scoring.

1. What is the canonical observation timestamp for training rows?

   Candidate options:

   - scheduled departure timestamp,
   - CDF commit timestamp,
   - actual prediction/scoring timestamp,
   - a stage-specific timestamp such as off-block or airborne.

2. What is the canonical prediction timestamp for online scoring?

   The current scoring flow computes `hours_to_departure_at_prediction` from
   `dep_sched_dt` and `_commit_timestamp` / `current_timestamp()`. The team
   should decide whether `_commit_timestamp` is the correct PIT lookup timestamp
   for all feature tables.

3. Which source column represents availability time for `leg` rows?

   `dep_sched_dt` is flight event time, not necessarily data availability time.
   For PIT correctness, `ft_leg_status` needs a source update or valid-from
   timestamp.

4. Which source column represents availability time for `leg_misc` rows?

   This controls whether `dep_stand` and `arr_stand` are safe for historical
   training and for online scoring at different operational stages.

5. Which source column represents availability time for `leg_times` rows?

   Raw OOOI values should be label-only for the current model, but the timestamp
   still matters for generating historical aggregate rows without future data.

6. Should daily aggregate feature rows use `DATE` or `TIMESTAMP` as the
   `TIMESERIES` key?

   Current code uses `event_date`. If daily aggregate rows are sparse and PIT
   lookup returns the latest available row, the team should define whether the
   row is available at start of day, end of day, or after a daily batch completes.

7. Should historical aggregates exclude the current flight, current day, or all
   data after prediction timestamp?

   Current route and stand builders use marker rows and windows ending before
   the marker timestamp. A new sparse-row design must preserve the same
   no-future-data guarantee.

8. How should `feature_age_days` be computed?

   Candidate definition:

   - `datediff(observation_date, feature_event_date)` for daily tables,
   - timestamp difference in hours for source-state tables,
   - separate age features by table family.

9. Are `arr_stand` and arrival-side stand stats expected to be available before
   the model scores?

   If not, arrival stand features should be stage-specific or optional with a
   clear missing-feature policy.

10. Should training use the same feature availability contract as online
    scoring?

    If yes, training should be built from the same base request columns and PIT
    lookups that are available at the intended online scoring time.

## Recommended migration order

1. Freeze the feature inventory and agree on the observation timestamp.
2. Decide which source update timestamp should be the `TIMESERIES` key for
   `ft_leg_*` tables.
3. Replace current-only SCD reads in training with AS OF/PIT logic.
4. Redesign daily aggregate tables to emit rows only for dates with events.
5. Add `feature_age_days` after PIT lookups.
6. Convert Lakeflow batch outputs from `@dp.table` to `@dp.materialized_view`
   with schema constraints where they remain batch materialized views.
7. Add UC Python UDFs for on-demand features.
8. Build training datasets with `FeatureEngineeringClient.create_training_set`.
9. Log the model with `FeatureEngineeringClient.log_model`.
10. Move batch/microbatch scoring toward `FeatureEngineeringClient.score_batch`.

## Relevant documentation

- Feature tables in Unity Catalog:
  https://docs.databricks.com/gcp/en/machine-learning/feature-store/uc/feature-tables-uc
- Point-in-time feature joins:
  https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series
- On-demand feature computation:
  https://docs.databricks.com/aws/en/machine-learning/feature-store/on-demand-features
- Lakeflow Spark Declarative Pipelines Python reference:
  https://docs.databricks.com/aws/en/ldp/developer/python-ref
- Materialized views:
  https://docs.databricks.com/aws/en/ldp/materialized-views
