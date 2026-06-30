from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


ENTITY_COL = "dep_ap_sched"
DATE_COL = "event_date"
DIRTY_EVENT_DATE_COL = "dirty_event_date"
AFFECTED_OUTPUT_DATE_COL = "affected_output_date"

TARGET_COLS_DICT = {
    "taxi_out_sec": "taxi_out",
    "duration_ratio": "dur_ratio_dep",
}
COUNT_PREFIX = "dep"
ROLLING_WINDOWS_DAYS = {
    "7d": 7,
    "30d": 30,
}
CURRENT_DAY_INCLUDED = False
EMA_POLICY = "deferred"

SECONDS_IN_DAY = 60 * 60 * 24
NUMERIC_PARITY_TOLERANCE = 1e-9

NON_EMA_PARITY_COLUMNS = (
    ENTITY_COL,
    DATE_COL,
    "avg_taxi_out_7d",
    "std_taxi_out_7d",
    "p90_taxi_out_7d",
    "min_taxi_out_7d",
    "max_taxi_out_7d",
    "avg_dur_ratio_dep_7d",
    "std_dur_ratio_dep_7d",
    "p90_dur_ratio_dep_7d",
    "min_dur_ratio_dep_7d",
    "max_dur_ratio_dep_7d",
    "count_dep_7d",
    "avg_taxi_out_30d",
    "std_taxi_out_30d",
    "p90_taxi_out_30d",
    "min_taxi_out_30d",
    "max_taxi_out_30d",
    "avg_dur_ratio_dep_30d",
    "std_dur_ratio_dep_30d",
    "p90_dur_ratio_dep_30d",
    "min_dur_ratio_dep_30d",
    "max_dur_ratio_dep_30d",
    "count_dep_30d",
    "trend_taxi_out_7d",
    "trend_dur_ratio_dep_7d",
    "has_hist_dep_7d",
    "has_hist_dep_30d",
    "days_since_last_event",
)

PARITY_KEY_COLUMNS = (ENTITY_COL, DATE_COL)
COMPARABLE_VALUE_COLUMNS = tuple(col for col in NON_EMA_PARITY_COLUMNS if col not in PARITY_KEY_COLUMNS)


@dataclass(frozen=True)
class TaxiOutCandidatePlan:
    target_table: str = "ft_airport_daily_taxi_out"
    source_shape: str = "cleaned_flight_data_full_table"
    scope: str = "entity-scoped"
    parity_scope: str = "non-EMA"
    output_date_rule: str = "D+1 through D+30"
    write_policy: str = "deferred"


def _pyspark_sql():
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    return F, Window


def get_taxi_out_candidate_plan() -> TaxiOutCandidatePlan:
    return TaxiOutCandidatePlan()


def expand_dirty_taxi_out_events_to_affected_outputs(dirty_events_df: "DataFrame") -> "DataFrame":
    """Expand dirty taxi-out event dates to unique affected entity/output-date pairs."""
    F, _ = _pyspark_sql()
    expanded = (
        dirty_events_df.withColumn("_stage30b_offset", F.explode(F.sequence(F.lit(1), F.lit(30))))
        .withColumn(AFFECTED_OUTPUT_DATE_COL, F.date_add(F.col(DIRTY_EVENT_DATE_COL), F.col("_stage30b_offset")))
        .withColumn("affects_rolling_7d", F.col("_stage30b_offset") <= F.lit(7))
        .withColumn("affects_rolling_30d", F.lit(True))
    )

    metadata_exprs = [
        F.countDistinct("leg_no").alias("dirty_leg_count"),
        F.collect_set(DIRTY_EVENT_DATE_COL).alias("dirty_event_dates"),
        F.max("affects_rolling_7d").alias("affects_rolling_7d"),
        F.max("affects_rolling_30d").alias("affects_rolling_30d"),
    ]
    if "dirty_source_aliases" in dirty_events_df.columns:
        metadata_exprs.append(
            F.array_distinct(F.flatten(F.collect_set("dirty_source_aliases"))).alias("dirty_source_aliases")
        )

    selected_cols = [
        ENTITY_COL,
        AFFECTED_OUTPUT_DATE_COL,
        "affects_rolling_7d",
        "affects_rolling_30d",
        "dirty_leg_count",
        "dirty_event_dates",
    ]
    if "dirty_source_aliases" in dirty_events_df.columns:
        selected_cols.append("dirty_source_aliases")

    return (
        expanded.groupBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL)
        .agg(*metadata_exprs)
        .select(*selected_cols)
    )


def build_taxi_out_non_ema_candidate_features(cleaned_flight_df: "DataFrame") -> "DataFrame":
    """Build non-EMA taxi-out daily candidate features with production-equivalent rolling windows."""
    F, Window = _pyspark_sql()
    df = cleaned_flight_df.withColumn(
        "duration_ratio",
        F.when(
            F.col("scheduled_block_time_sec") > 0,
            (F.col("actual_block_time_sec") / F.col("scheduled_block_time_sec")).cast("double"),
        ),
    )

    agg_exprs = []
    for src_col, prefix in TARGET_COLS_DICT.items():
        agg_exprs.extend(
            [
                F.sum(F.col(src_col).cast("double")).alias(f"_sum_{prefix}"),
                F.count(src_col).alias(f"_cnt_{prefix}"),
                F.min(F.col(src_col).cast("double")).alias(f"_min_{prefix}"),
                F.max(F.col(src_col).cast("double")).alias(f"_max_{prefix}"),
                F.sum(F.col(src_col).cast("double") * F.col(src_col).cast("double")).alias(f"_sumsq_{prefix}"),
                F.expr(f"percentile_approx(CAST({src_col} AS DOUBLE), 0.9)").alias(f"_p90_{prefix}"),
            ]
        )
    agg_exprs.append(F.count("*").alias("_fcnt"))

    daily = df.groupBy(DATE_COL, ENTITY_COL).agg(*agg_exprs)
    daily = daily.withColumn("_ets", F.unix_timestamp(DATE_COL))

    windows = {
        name: Window.partitionBy(ENTITY_COL)
        .orderBy("_ets")
        .rangeBetween(-days * SECONDS_IN_DAY, -1)
        for name, days in ROLLING_WINDOWS_DAYS.items()
    }

    for window_name, window in windows.items():
        for _, prefix in TARGET_COLS_DICT.items():
            rolling_sum = F.sum(f"_sum_{prefix}").over(window)
            rolling_count = F.sum(f"_cnt_{prefix}").over(window)
            rolling_sumsq = F.sum(f"_sumsq_{prefix}").over(window)
            daily = daily.withColumn(f"avg_{prefix}_{window_name}", rolling_sum / rolling_count)
            daily = daily.withColumn(
                f"std_{prefix}_{window_name}",
                F.sqrt(F.abs(rolling_sumsq / rolling_count - F.pow(rolling_sum / rolling_count, 2))),
            )
            daily = daily.withColumn(f"p90_{prefix}_{window_name}", F.max(f"_p90_{prefix}").over(window))
            daily = daily.withColumn(f"min_{prefix}_{window_name}", F.min(f"_min_{prefix}").over(window))
            daily = daily.withColumn(f"max_{prefix}_{window_name}", F.max(f"_max_{prefix}").over(window))
        daily = daily.withColumn(
            f"count_{COUNT_PREFIX}_{window_name}",
            F.sum("_fcnt").over(window).cast("double"),
        )

    for _, prefix in TARGET_COLS_DICT.items():
        daily = daily.withColumn(f"trend_{prefix}_7d", F.col(f"avg_{prefix}_7d") - F.col(f"avg_{prefix}_30d"))

    daily = (
        daily.withColumn("has_hist_dep_7d", F.when(F.col("count_dep_7d") > 0, 1.0).otherwise(0.0))
        .withColumn("has_hist_dep_30d", F.when(F.col("count_dep_30d") > 0, 1.0).otherwise(0.0))
    )

    entity_window = Window.partitionBy(ENTITY_COL).orderBy(DATE_COL)
    daily = (
        daily.withColumn("_prev", F.lag(DATE_COL).over(entity_window))
        .withColumn(
            "days_since_last_event",
            F.when(F.col("_prev").isNull(), F.lit(0.0)).otherwise(F.datediff(F.col(DATE_COL), F.col("_prev")).cast("double")),
        )
        .drop("_prev")
    )

    internal_cols = [col for col in daily.columns if col.startswith("_")]
    return daily.drop(*internal_cols).select(*NON_EMA_PARITY_COLUMNS)


def build_taxi_out_candidate_for_affected_outputs(
    cleaned_flight_df: "DataFrame",
    affected_outputs_df: "DataFrame",
    *,
    history_start: str | None = None,
    data_cutoff_date: str | None = None,
) -> "DataFrame":
    """Entity-scoped non-EMA candidate recompute filtered to affected output pairs."""
    F, _ = _pyspark_sql()
    affected_entities = affected_outputs_df.select(ENTITY_COL).dropDuplicates()
    scoped = cleaned_flight_df.join(affected_entities, on=ENTITY_COL, how="inner")

    if history_start is not None:
        scoped = scoped.filter(F.col(DATE_COL) >= F.to_date(F.lit(history_start)))
    if data_cutoff_date is not None:
        scoped = scoped.filter(F.col(DATE_COL) < F.to_date(F.lit(data_cutoff_date)))

    candidate = build_taxi_out_non_ema_candidate_features(scoped)
    affected_pairs = affected_outputs_df.select(
        ENTITY_COL,
        F.col(AFFECTED_OUTPUT_DATE_COL).alias(DATE_COL),
    ).dropDuplicates()

    return candidate.join(affected_pairs, on=[ENTITY_COL, DATE_COL], how="inner").select(*NON_EMA_PARITY_COLUMNS)


def compare_taxi_out_candidate_to_current_mv(
    candidate_df: "DataFrame",
    current_mv_df: "DataFrame",
    *,
    tolerance: float = NUMERIC_PARITY_TOLERANCE,
) -> "DataFrame":
    """Compare non-EMA candidate rows with the current materialized view rows."""
    F, _ = _pyspark_sql()
    candidate = candidate_df.select(*NON_EMA_PARITY_COLUMNS).withColumn("_candidate_present", F.lit(True)).alias("candidate")
    current = current_mv_df.select(*NON_EMA_PARITY_COLUMNS).withColumn("_current_present", F.lit(True)).alias("current")
    joined = candidate.join(current, on=list(PARITY_KEY_COLUMNS), how="full_outer")

    mismatch_expr = F.lit(False)
    for col_name in COMPARABLE_VALUE_COLUMNS:
        candidate_col = F.col(f"candidate.{col_name}")
        current_col = F.col(f"current.{col_name}")
        value_mismatch = (
            (candidate_col.isNull() & current_col.isNotNull())
            | (candidate_col.isNotNull() & current_col.isNull())
            | (
                candidate_col.isNotNull()
                & current_col.isNotNull()
                & (F.abs(candidate_col.cast("double") - current_col.cast("double")) > F.lit(tolerance))
            )
        )
        mismatch_expr = mismatch_expr | value_mismatch

    status = (
        F.when(F.col("candidate._candidate_present").isNull(), F.lit("missing_in_candidate"))
        .when(F.col("current._current_present").isNull(), F.lit("missing_in_current"))
        .when(mismatch_expr, F.lit("value_mismatch"))
        .otherwise(F.lit("matched"))
    )

    diff_cols = [F.col(ENTITY_COL), F.col(DATE_COL), status.alias("parity_status")]
    for col_name in COMPARABLE_VALUE_COLUMNS:
        diff_cols.extend(
            [
                F.col(f"candidate.{col_name}").alias(f"candidate_{col_name}"),
                F.col(f"current.{col_name}").alias(f"current_{col_name}"),
            ]
        )

    return joined.select(*diff_cols)
