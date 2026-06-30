from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


class DirtyKeyStrategy(str, Enum):
    UPDATE_KEY = "update_key"


@dataclass(frozen=True)
class SourceSpec:
    alias: str
    table_name: str
    candidate_columns: tuple[str, ...]
    role: str
    required_for_taxi_out_poc: bool


@dataclass(frozen=True)
class AffectedOutputDate:
    affected_output_date: date
    affects_rolling_7d: bool
    affects_rolling_30d: bool


SOURCE_SPECS: dict[str, SourceSpec] = {
    "leg": SourceSpec(
        alias="leg",
        table_name="netline___schedops__leg",
        candidate_columns=("update_key", "entry_dt", "__START_AT", "__END_AT"),
        role="required for taxi-out POC",
        required_for_taxi_out_poc=True,
    ),
    "leg_times": SourceSpec(
        alias="leg_times",
        table_name="netline___schedops__leg_times",
        candidate_columns=("update_key", "__START_AT", "__END_AT"),
        role="required for taxi-out POC",
        required_for_taxi_out_poc=True,
    ),
    "leg_misc": SourceSpec(
        alias="leg_misc",
        table_name="netline___schedops__leg_misc",
        candidate_columns=("update_key", "__START_AT", "__END_AT"),
        role="deferred for stand POCs",
        required_for_taxi_out_poc=False,
    ),
}

TAXI_OUT_POC_SOURCE_ALIASES = ("leg", "leg_times")
DEFERRED_SOURCE_ALIASES = ("leg_misc",)

CURRENT_STREAM_TABLES_NOT_PRIMARY_SOURCES = (
    "ft_leg_status",
    "ft_leg_times",
    "ft_leg_misc",
)

PRIMARY_DIRTY_KEY_STRATEGY = DirtyKeyStrategy.UPDATE_KEY
DIRTY_KEY_STRATEGY_NOTES = {
    DirtyKeyStrategy.UPDATE_KEY: (
        "Filter source rows with update_key greater than the last checkpoint. "
        "update_key equals __START_AT in the verified sources, is monotonic, "
        "and is batch-level rather than row-level, so dirty outputs must be "
        "deduplicated after filtering."
    ),
}

EMA_POLICY = "deferred"
EMA_POLICY_NOTES = (
    "EMA can propagate beyond D+30. Stage 30B-1 does not solve partial EMA "
    "recompute; later options include full entity recompute from the dirty date "
    "onward, bounded approximation, or explicit EMA state."
)

TAXI_OUT_ENTITY_COL = "dep_ap_sched"
TAXI_OUT_EVENT_TS_COL = "dep_sched_dt"
TAXI_OUT_EVENT_DATE_COL = "dirty_event_date"
TAXI_OUT_INCLUDED_LEG_TYPES = ("J", "C", "G")


def _pyspark_sql():
    from pyspark.sql import Window
    from pyspark.sql import functions as F

    return F, Window


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value)
    raise TypeError(f"Unsupported date value: {value!r}")


def expand_taxi_out_affected_output_dates(dirty_event_date: date | datetime | str) -> tuple[AffectedOutputDate, ...]:
    """Return affected taxi-out output dates for a dirty source event date.

    Current production rolling windows exclude same-day data, so event date D
    affects output dates D+1 through D+30, inclusive.
    """
    base_date = _as_date(dirty_event_date)
    return tuple(
        AffectedOutputDate(
            affected_output_date=base_date + timedelta(days=offset),
            affects_rolling_7d=offset <= 7,
            affects_rolling_30d=True,
        )
        for offset in range(1, 31)
    )


def select_current_latest(
    source_df: "DataFrame",
    partition_cols: Iterable[str] = ("leg_no",),
    *,
    end_col: str = "__END_AT",
    update_col: str = "update_key",
    tie_breaker_cols: Iterable[str] = (),
) -> "DataFrame":
    """Select current/open source rows, then latest row per key as a transform."""
    F, Window = _pyspark_sql()
    partitions = tuple(partition_cols)
    if not partitions:
        raise ValueError("partition_cols must not be empty")

    order_cols = [F.col(update_col).desc_nulls_last()]
    order_cols.extend(F.col(col_name).desc_nulls_last() for col_name in tie_breaker_cols)

    window = Window.partitionBy(*partitions).orderBy(*order_cols)
    return (
        source_df.filter(F.col(end_col).isNull())
        .withColumn("_stage30b_current_rank", F.row_number().over(window))
        .filter(F.col("_stage30b_current_rank") == F.lit(1))
        .drop("_stage30b_current_rank")
    )


def extract_dirty_leg_keys(
    source_df: "DataFrame",
    last_seen_update_key: Any,
    source_alias: str,
    *,
    leg_col: str = "leg_no",
    update_col: str = "update_key",
) -> "DataFrame":
    """Extract distinct dirty leg candidates using update_key checkpointing."""
    F, _ = _pyspark_sql()
    return (
        source_df.filter(F.col(update_col) > F.lit(last_seen_update_key))
        .select(
            F.col(leg_col).alias("leg_no"),
            F.lit(source_alias).alias("dirty_source_alias"),
            F.col(update_col).alias("_stage30b_update_key"),
        )
        .groupBy("leg_no", "dirty_source_alias")
        .agg(F.max("_stage30b_update_key").alias("max_update_key"))
    )


def map_dirty_legs_to_taxi_out_events(
    dirty_leg_df: "DataFrame",
    current_leg_df: "DataFrame",
    *,
    history_start: str | None = None,
    data_cutoff_date: str | None = None,
) -> "DataFrame":
    """Map dirty leg_no values to taxi-out entity/date candidates."""
    F, _ = _pyspark_sql()
    current_leg = select_current_latest(current_leg_df, partition_cols=("leg_no",))
    event_date = F.to_date(F.col(TAXI_OUT_EVENT_TS_COL))

    filtered_leg = (
        current_leg.filter(F.col("counter") == F.lit(0))
        .filter(F.col("leg_type").isin(*TAXI_OUT_INCLUDED_LEG_TYPES))
        .filter(F.col("leg_state") == F.lit("ARR"))
        .filter(F.col(TAXI_OUT_ENTITY_COL).isNotNull())
        .filter(F.col(TAXI_OUT_EVENT_TS_COL).isNotNull())
    )

    if history_start is not None:
        filtered_leg = filtered_leg.filter(event_date >= F.to_date(F.lit(history_start)))
    if data_cutoff_date is not None:
        filtered_leg = filtered_leg.filter(event_date < F.to_date(F.lit(data_cutoff_date)))

    dirty = dirty_leg_df.select("leg_no", "dirty_source_alias").dropDuplicates()
    return (
        dirty.join(filtered_leg, on="leg_no", how="inner")
        .select(
            F.col("leg_no"),
            F.col(TAXI_OUT_ENTITY_COL),
            event_date.alias(TAXI_OUT_EVENT_DATE_COL),
            F.col("dirty_source_alias"),
        )
        .groupBy("leg_no", TAXI_OUT_ENTITY_COL, TAXI_OUT_EVENT_DATE_COL)
        .agg(F.collect_set("dirty_source_alias").alias("dirty_source_aliases"))
        .withColumn("dirty_reason", F.lit("taxi_out_dirty_source_change"))
    )
