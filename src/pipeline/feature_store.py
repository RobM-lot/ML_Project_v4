import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, StringType, StructField, StructType
from pyspark.sql.window import Window

_TIME_WINDOWS = ("7d", "30d")
_STAT_NAMES = ("avg", "std", "p90", "min", "max")


def _format_ddl(columns: Sequence[str], constraint: str) -> str:
    body = ",\n    ".join(columns)
    return f"\n    {body},\n    {constraint}\n"


def route_schema_ddl(
    *,
    entity_cols: Sequence[str],
    target_cols_dict: Dict[str, str],
    count_prefix: str,
    extra_pk_col: str | None = None,
) -> str:
    """DDL dla output `_build_route_feature_store(...)`."""
    columns: List[str] = []

    if extra_pk_col:
        columns.append(f"{extra_pk_col} STRING NOT NULL")
        columns.append("event_date DATE NOT NULL")
        for col in entity_cols:
            columns.append(f"{col} STRING")
        pk_business_col = extra_pk_col
    else:
        columns.append(f"{entity_cols[0]} STRING NOT NULL")
        columns.append("event_date DATE NOT NULL")
        for col in entity_cols[1:]:
            columns.append(f"{col} STRING")
        pk_business_col = entity_cols[0]

    prefixes = list(target_cols_dict.values())

    for window in _TIME_WINDOWS:
        for prefix in prefixes:
            for stat in _STAT_NAMES:
                columns.append(f"{stat}_{prefix}_{window} DOUBLE")
        columns.append(f"count_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        columns.append(f"trend_{prefix}_7d DOUBLE")

    for window in _TIME_WINDOWS:
        columns.append(f"has_hist_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        for window in _TIME_WINDOWS:
            columns.append(f"ema_{prefix}_{window} DOUBLE")

    for window in _TIME_WINDOWS:
        columns.append(f"ema_confidence_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        for window in _TIME_WINDOWS:
            columns.append(f"delta_ema_avg_{prefix}_{window} DOUBLE")

    constraint_name = f"fs_{count_prefix}_features_pk"
    constraint = (
        f"CONSTRAINT {constraint_name} "
        f"PRIMARY KEY ({pk_business_col}, event_date TIMESERIES)"
    )
    return _format_ddl(columns, constraint)


def stand_schema_ddl(*, is_taxi_out: bool) -> str:
    """DDL dla output `_build_stand_features(..., is_taxi_out=...)`."""
    side = "dep" if is_taxi_out else "arr"
    prefix = "out" if is_taxi_out else "in"

    columns: List[str] = [
        "stand_id STRING NOT NULL",
        "event_date DATE NOT NULL",
        f"fs_{side}_ap_sched STRING",
        f"fs_{side}_stand STRING",
        f"stand_count_{prefix}_7d DOUBLE",
        f"stand_avg_taxi_{prefix}_7d DOUBLE",
        f"stand_count_{prefix}_30d DOUBLE",
        f"stand_avg_taxi_{prefix}_30d DOUBLE",
        f"stand_std_taxi_{prefix}_30d DOUBLE",
        f"stand_p10_taxi_{prefix}_30d DOUBLE",
        f"stand_p50_taxi_{prefix}_30d DOUBLE",
        f"stand_p90_taxi_{prefix}_30d DOUBLE",
        f"stand_trend_taxi_{prefix}_7d DOUBLE",
    ]

    constraint = (
        f"CONSTRAINT fs_stand_{prefix}_features_pk "
        f"PRIMARY KEY (stand_id, event_date TIMESERIES)"
    )
    return _format_ddl(columns, constraint)


# =====================================================================================
# Iter2 — DDL dla nowego layoutu ft_*  (B2: EMA bez densyfikacji + days_since_last_event)
# =====================================================================================

def leg_status_schema_ddl() -> str:
    """DDL dla `ft_leg_status` — streaming ingest statusu lotu z df_labels.

    TIMESERIES = event_ts (dep_sched_dt lotu). PK (leg_no, event_ts).
    """
    columns: List[str] = [
        "leg_no LONG NOT NULL",
        "event_ts TIMESTAMP NOT NULL",
        "event_date DATE",
        "leg_state STRING",
        "leg_type STRING",
        "marker STRING",
        "ac_owner STRING",
        "ac_registration STRING",
        "ac_subtype STRING",
        "commercial_carrier STRING",
        "dep_ap_sched STRING",
        "arr_ap_sched STRING",
        "dep_sched_dt TIMESTAMP",
        "arr_sched_dt TIMESTAMP",
        "counter INT",
        "isLO INT",
        "fn_full_number STRING",
    ]
    return _format_ddl(columns, "CONSTRAINT ft_leg_status_pk PRIMARY KEY (leg_no, event_ts TIMESERIES)")


def leg_times_schema_ddl() -> str:
    """DDL dla `ft_leg_times` — streaming ingest pomiarów OOOI z df_leg_times.

    Brak TIMESERIES (tabela nie jest używana w PIT lookupach). PK = leg_no.
    """
    columns: List[str] = [
        "leg_no LONG NOT NULL",
        "offblock_dt TIMESTAMP",
        "airborne_dt TIMESTAMP",
        "landing_dt TIMESTAMP",
        "onblock_dt TIMESTAMP",
    ]
    return _format_ddl(columns, "CONSTRAINT ft_leg_times_pk PRIMARY KEY (leg_no)")


def leg_misc_schema_ddl() -> str:
    """DDL dla `ft_leg_misc` — streaming ingest stand assignment z df_leg_misc.

    Brak TIMESERIES (tabela nie jest używana w PIT lookupach). PK = leg_no.
    """
    columns: List[str] = [
        "leg_no LONG NOT NULL",
        "dep_stand STRING",
        "arr_stand STRING",
    ]
    return _format_ddl(columns, "CONSTRAINT ft_leg_misc_pk PRIMARY KEY (leg_no)")


def airport_timezone_schema_ddl() -> str:
    """DDL dla `ft_airport_timezone` — strefa czasowa + lat/lon (radiany) per lotnisko.

    TIMESERIES = valid_ts (= valid_since). PK (iata_ap_code, valid_ts).
    """
    columns: List[str] = [
        "iata_ap_code STRING NOT NULL",
        "valid_ts TIMESTAMP NOT NULL",
        "lat_deg DOUBLE",
        "lon_deg DOUBLE",
        "utc_offset_min INT",
        "valid_until DATE",
    ]
    return _format_ddl(columns, "CONSTRAINT ft_airport_timezone_pk PRIMARY KEY (iata_ap_code, valid_ts TIMESERIES)")


def daily_stats_schema_ddl(
    *,
    entity_cols: Sequence[str],
    target_cols_dict: Dict[str, str],
    count_prefix: str,
    extra_pk_col: str | None = None,
) -> str:
    """DDL dla `ft_*_daily_*` (route/airport). Jak `route_schema_ddl` + `days_since_last_event`."""
    columns: List[str] = []

    if extra_pk_col:
        columns.append(f"{extra_pk_col} STRING NOT NULL")
        columns.append("event_date DATE NOT NULL")
        for col in entity_cols:
            columns.append(f"{col} STRING")
        pk_business_col = extra_pk_col
    else:
        columns.append(f"{entity_cols[0]} STRING NOT NULL")
        columns.append("event_date DATE NOT NULL")
        for col in entity_cols[1:]:
            columns.append(f"{col} STRING")
        pk_business_col = entity_cols[0]

    prefixes = list(target_cols_dict.values())

    for window in _TIME_WINDOWS:
        for prefix in prefixes:
            for stat in _STAT_NAMES:
                columns.append(f"{stat}_{prefix}_{window} DOUBLE")
        columns.append(f"count_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        columns.append(f"trend_{prefix}_7d DOUBLE")

    for window in _TIME_WINDOWS:
        columns.append(f"has_hist_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        for window in _TIME_WINDOWS:
            columns.append(f"ema_{prefix}_{window} DOUBLE")

    for window in _TIME_WINDOWS:
        columns.append(f"ema_confidence_{count_prefix}_{window} DOUBLE")

    for prefix in prefixes:
        for window in _TIME_WINDOWS:
            columns.append(f"delta_ema_avg_{prefix}_{window} DOUBLE")

    columns.append("days_since_last_event DOUBLE")  # B2: luka dni od poprzedniego eventu encji

    constraint = (
        f"CONSTRAINT ft_{count_prefix}_daily_pk "
        f"PRIMARY KEY ({pk_business_col}, event_date TIMESERIES)"
    )
    return _format_ddl(columns, constraint)


def stand_daily_schema_ddl(*, is_taxi_out: bool) -> str:
    """DDL dla `ft_stand_daily_*`. Jak `stand_schema_ddl` + `days_since_last_event`."""
    side = "dep" if is_taxi_out else "arr"
    prefix = "out" if is_taxi_out else "in"

    columns: List[str] = [
        "stand_id STRING NOT NULL",
        "event_date DATE NOT NULL",
        f"fs_{side}_ap_sched STRING",
        f"fs_{side}_stand STRING",
        f"stand_count_{prefix}_7d DOUBLE",
        f"stand_avg_taxi_{prefix}_7d DOUBLE",
        f"stand_count_{prefix}_30d DOUBLE",
        f"stand_avg_taxi_{prefix}_30d DOUBLE",
        f"stand_std_taxi_{prefix}_30d DOUBLE",
        f"stand_p10_taxi_{prefix}_30d DOUBLE",
        f"stand_p50_taxi_{prefix}_30d DOUBLE",
        f"stand_p90_taxi_{prefix}_30d DOUBLE",
        f"stand_trend_taxi_{prefix}_7d DOUBLE",
        "days_since_last_event DOUBLE",
    ]
    constraint = (
        f"CONSTRAINT ft_stand_{prefix}_daily_pk "
        f"PRIMARY KEY (stand_id, event_date TIMESERIES)"
    )
    return _format_ddl(columns, constraint)


SECONDS_IN_DAY = 60 * 60 * 24
HALF_LIFE_DAYS = {"7d": 7, "30d": 30}

MIN_VALID_TIME_SEC = 0
MAX_VALID_SCHED_BLOCK_SEC = 20 * 3600
MAX_VALID_BLOCK_SEC = 20 * 3600
MAX_VALID_TAXI_OUT_SEC = 3 * 3600
MAX_VALID_AIRBORNE_SEC = 18 * 3600
MAX_VALID_TAXI_IN_SEC = 2 * 3600
MAX_SEGMENT_SUM_GAP_SEC = 15 * 60

DLT_TABLE_PROPERTIES = {"delta.enableDeletionVectors": "true"}


def _conf(key, default):
    return spark.conf.get(key, default)


def _source_table(table_name):
    """Buduje pełną nazwę tabeli ŹRÓDŁOWEJ (raw schedops) jako catalog.schema.name.

    JEDYNE miejsce, w którym budujemy nazwę tabeli źródłowej — katalog/schemat biorą się
    z konfiguracji runtime (`ml.source_catalog` / `ml.source_schema`), NIE hardkodujemy ich
    nigdzie indziej. Dotyczy wyłącznie trwałych tabel UC; wewnętrzne widoki DLT
    (np. "df_labels") czytamy po gołej nazwie i NIE przepuszczamy przez ten helper.
    """
    return f"{_conf('ml.source_catalog', 'panda_silver_prod')}.{_conf('ml.source_schema', 'occ_ops')}.{table_name}"


def _silver_catalog():
    return _conf("ml.silver_catalog", "panda_silver_dev")


def _silver_schema():
    return _conf("ml.silver_schema", "ml_ops")


def _fs_table(table_name):
    """Buduje pełną nazwę tabeli FEATURE STORE (silver) jako catalog.schema.name.

    JEDYNE miejsce, w którym budujemy nazwę tabeli FS — katalog/schemat biorą się z
    konfiguracji runtime (`ml.silver_catalog` / `ml.silver_schema`), NIE hardkodujemy ich
    nigdzie indziej. Używane zarówno przez `@dp.materialized_view(name=_fs_table(...))`,
    jak i przy odczytach `spark.read.table(_fs_table(...))`.
    """
    return f"{_silver_catalog()}.{_silver_schema()}.{table_name}"


def _latest_window(df, partition_cols):
    if "update_key" in df.columns:
        order_col = F.col("update_key").desc()
    elif "entry_dt" in df.columns:
        order_col = F.col("entry_dt").desc()
    else:
        order_col = F.col(partition_cols[0]).asc()
    return Window.partitionBy(*partition_cols).orderBy(order_col)


@dp.temporary_view()
def df_labels():
    return spark.read.table(_source_table("netline___schedops__leg"))


@dp.temporary_view()
def df_leg_times():
    return spark.read.table(_source_table("netline___schedops__leg_times"))


@dp.temporary_view()
def df_leg_remark():
    return spark.read.table(_source_table("netline___schedops__leg_remark"))


@dp.temporary_view()
def df_leg_misc():
    return spark.read.table(_source_table("netline___schedops__leg_misc"))


@dp.temporary_view()
def df_ap_basics():
    return spark.read.table(_source_table("netline___schedops__ap_basics"))


@dp.temporary_view()
def df_time_zone():
    return spark.read.table(_source_table("netline___schedops__time_zone"))


def _add_marker_columns(df):
    for i in range(1, 18):
        df = df.withColumn(
            f"marker_{i}",
            F.when(
                F.length(F.col("marker")) >= i,
                F.when(F.substring(F.col("marker"), i, 1) == "Y", 0)
                .when(F.substring(F.col("marker"), i, 1) == "N", 1)
                .otherwise(F.lit(None).cast("double")),
            ).otherwise(F.lit(None).cast("double")),
        )
    return df


@dp.temporary_view()
def v_cleaned_flight_data_full_table():
    df_labels = spark.read.table("df_labels")
    df_leg_times = spark.read.table("df_leg_times")
    df_leg_remark = spark.read.table("df_leg_remark")
    df_leg_misc = spark.read.table("df_leg_misc")

    # TODO Iter2: usunąć filtr __END_AT po migracji na readStream
    base = (
        df_labels.filter(F.col("__END_AT").isNull())
        .filter(F.col("counter") == 0)
        .filter(F.col("leg_type").isin(["J", "C", "G"]))
        .filter(F.col("leg_state") == "ARR")
        .filter(F.to_date(F.col("dep_sched_dt")) >= F.to_date(F.lit(_conf("ml.history_start", "2023-07-01"))))
        .filter(F.to_date(F.col("dep_sched_dt")) < F.to_date(F.lit(_conf("ml.data_cutoff_date", "2027-01-01"))))
        .withColumn("event_ts", F.col("dep_sched_dt"))
        .withColumn("event_date", F.to_date("dep_sched_dt"))
        .withColumn("ac_registration", F.substring(F.col("ac_registration"), 1, 4))
        .withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
        .withColumn("fn_full_number", F.concat(F.col("fn_carrier"), F.col("fn_number")))
    )
    base = _add_marker_columns(base)

    # TODO Iter2: usunąć filtr po migracji na readStream
    leg_times_raw = df_leg_times.filter(F.col("__END_AT").isNull())
    w_times = _latest_window(leg_times_raw, ["leg_no"])
    leg_times_latest = (
        leg_times_raw
        .withColumn("rn", F.row_number().over(w_times))
        .filter(F.col("rn") == 1)
        .select("leg_no", "offblock_dt", "airborne_dt", "landing_dt", "onblock_dt")
    )

    # leg_remark NIE ma __END_AT (brak SCD2 tracking) — bez filtra wersji.
    leg_remark_raw = df_leg_remark.filter(F.col("usage") == "F")
    w_remark = _latest_window(leg_remark_raw, ["leg_no", "usage"])
    leg_remark_latest = (
        leg_remark_raw
        .withColumn("rn", F.row_number().over(w_remark))
        .filter(F.col("rn") == 1)
        .withColumn("eet_str", F.regexp_extract(F.col("text"), r"EET\s*:\s*(\d+)", 1))
        .withColumn(
            "netline_eet_duration_min",
            F.when(
                F.col("eet_str") != "",
                (F.substring(F.lpad(F.col("eet_str"), 4, "0"), 1, 2).cast("int") * 60)
                + F.substring(F.lpad(F.col("eet_str"), 4, "0"), 3, 2).cast("int"),
            ).otherwise(F.lit(None).cast("int")),
        )
        .select("leg_no", "netline_eet_duration_min")
    )

    # leg_misc NIE ma __END_AT (brak SCD2 tracking) — bez filtra wersji.
    leg_misc_base = df_leg_misc
    leg_misc_current = (
        leg_misc_base
        .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
        .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
        .select("leg_no", "dep_stand", "arr_stand")
    )

    return (
        base.join(leg_times_latest, on="leg_no", how="left")
        .join(leg_remark_latest, on="leg_no", how="left")
        .join(leg_misc_current, on="leg_no", how="left")
    )


for prefix in ["dep", "arr"]:

    @dp.temporary_view(name=f"airport_features_{prefix}")
    def airport_features(prefix=prefix):
        apb = spark.read.table("df_ap_basics")
        tzd = spark.read.table("df_time_zone")

        apt = apb.join(tzd, F.col("time_zone") == F.col("time_zone_code"), "left")

        lat_deg = F.when(
            F.substring(F.col("coord_latitude"), 1, 1).isin(["N", "S"]),
            F.when(F.substring(F.col("coord_latitude"), 1, 1) == "N", 1).otherwise(-1)
            * (
                F.substring(F.col("coord_latitude"), 2, 2).cast("int")
                + F.substring(F.col("coord_latitude"), 4, 2).cast("int") / 60.0
            ),
        ).otherwise(F.lit(None))

        lon_deg = F.when(
            F.substring(F.col("coord_longitude"), 1, 1).isin(["E", "W"]),
            F.when(F.substring(F.col("coord_longitude"), 1, 1) == "E", 1).otherwise(-1)
            * (
                F.substring(F.col("coord_longitude"), 2, 3).cast("int")
                + F.substring(F.col("coord_longitude"), 5, 2).cast("int") / 60.0
            ),
        ).otherwise(F.lit(None))

        apt = apt.withColumn(f"{prefix}_lat_rad", F.radians(lat_deg))
        apt = apt.withColumn(f"{prefix}_lon_rad", F.radians(lon_deg))

        return apt.select(
            F.col("iata_ap_code").alias(f"{prefix}_ap_code"),
            F.col(f"{prefix}_lat_rad"),
            F.col(f"{prefix}_lon_rad"),
            F.coalesce(F.col("diff_utc_lst"), F.lit(0)).alias(f"{prefix}_utc_offset_min"),
            F.to_date(apb["valid_since"]).alias(f"{prefix}_valid_since"),
            F.to_date(apb["valid_until"]).alias(f"{prefix}_valid_until"),
        ).distinct()


@dp.materialized_view(name=_fs_table("enriched"), table_properties=DLT_TABLE_PROPERTIES)
def enriched():
    df = spark.read.table("v_cleaned_flight_data_full_table")
    dep_apt = F.broadcast(spark.read.table("airport_features_dep"))
    arr_apt = F.broadcast(spark.read.table("airport_features_arr"))

    df = df.join(
        dep_apt,
        (F.col("dep_ap_sched") == F.col("dep_ap_code"))
        & (F.to_date(F.col("dep_sched_dt")) >= F.col("dep_valid_since"))
        & (F.to_date(F.col("dep_sched_dt")) <= F.coalesce(F.col("dep_valid_until"), F.to_date(F.lit("2099-12-31")))),
        "left",
    )
    df = df.join(
        arr_apt,
        (F.col("arr_ap_sched") == F.col("arr_ap_code"))
        & (F.to_date(F.col("arr_sched_dt")) >= F.col("arr_valid_since"))
        & (F.to_date(F.col("arr_sched_dt")) <= F.coalesce(F.col("arr_valid_until"), F.to_date(F.lit("2099-12-31")))),
        "left",
    )

    df = df.withColumn("dep_local_ts", F.to_timestamp(F.unix_timestamp("dep_sched_dt") + (F.col("dep_utc_offset_min") * 60)))
    df = df.withColumn("arr_local_ts", F.to_timestamp(F.unix_timestamp("arr_sched_dt") + (F.col("arr_utc_offset_min") * 60)))

    df = df.withColumn("local_hour_dep", F.hour("dep_local_ts").cast("int"))
    df = df.withColumn("local_dow_dep", ((F.dayofweek("dep_local_ts") + 5) % 7).cast("int"))
    df = df.withColumn("local_hour_arr", F.hour("arr_local_ts").cast("int"))
    df = df.withColumn("local_dow_arr", ((F.dayofweek("arr_local_ts") + 5) % 7).cast("int"))

    df = df.withColumn("dlat", F.col("arr_lat_rad") - F.col("dep_lat_rad"))
    df = df.withColumn("dlon", F.col("arr_lon_rad") - F.col("dep_lon_rad"))
    df = df.withColumn("is_eastbound", F.when(F.col("dlon") > 0, 1).otherwise(0))

    df = df.withColumn("month", F.month("dep_sched_dt").cast("int"))
    df = df.withColumn("sin_month", F.sin(2.0 * math.pi * F.col("month") / 12.0))
    df = df.withColumn("cos_month", F.cos(2.0 * math.pi * F.col("month") / 12.0))

    df = df.withColumn("sin_local_hour_dep", F.sin(2.0 * math.pi * F.col("local_hour_dep") / 24.0))
    df = df.withColumn("cos_local_hour_dep", F.cos(2.0 * math.pi * F.col("local_hour_dep") / 24.0))
    df = df.withColumn("sin_local_hour_arr", F.sin(2.0 * math.pi * F.col("local_hour_arr") / 24.0))
    df = df.withColumn("cos_local_hour_arr", F.cos(2.0 * math.pi * F.col("local_hour_arr") / 24.0))
    df = df.withColumn("sin_local_dow_dep", F.sin(2.0 * math.pi * F.col("local_dow_dep") / 7.0))
    df = df.withColumn("cos_local_dow_dep", F.cos(2.0 * math.pi * F.col("local_dow_dep") / 7.0))
    df = df.withColumn("sin_local_dow_arr", F.sin(2.0 * math.pi * F.col("local_dow_arr") / 7.0))
    df = df.withColumn("cos_local_dow_arr", F.cos(2.0 * math.pi * F.col("local_dow_arr") / 7.0))

    df = df.withColumn(
        "a",
        F.pow(F.sin(F.col("dlat") / 2.0), 2)
        + F.cos(F.col("dep_lat_rad")) * F.cos(F.col("arr_lat_rad")) * F.pow(F.sin(F.col("dlon") / 2.0), 2),
    )
    df = df.withColumn("c", 2.0 * F.atan2(F.sqrt(F.col("a")), F.sqrt(F.greatest(F.lit(0.0), 1.0 - F.col("a")))))
    df = df.withColumn("distance_km", F.round(6371.0 * F.col("c"), 2))
    df = df.withColumn("distance_km", F.coalesce(F.col("distance_km"), F.lit(0.0)))

    cols_to_drop = [
        "dep_ap_code",
        "dep_lat_rad",
        "dep_lon_rad",
        "dep_valid_since",
        "dep_valid_until",
        "arr_ap_code",
        "arr_lat_rad",
        "arr_lon_rad",
        "arr_valid_since",
        "arr_valid_until",
        "dep_local_ts",
        "arr_local_ts",
        "dlat",
        "dlon",
        "a",
        "c",
    ]
    return df.drop(*cols_to_drop)


def _apply_data_quality_rules(df):
    cond_invalid_sched = F.lit(False)
    cond_missing_keys = F.lit(False)
    cond_invalid_actuals = F.lit(False)
    cond_airport_mismatch = F.lit(False)
    cond_same_airport = F.lit(False)
    cond_sequence_invalid = F.lit(False)
    cond_outlier_segments = F.lit(False)
    cond_segment_gap = F.lit(False)

    df_dq = df.withColumn(
        "scheduled_block_time_sec",
        F.col("arr_sched_dt").cast("long") - F.col("dep_sched_dt").cast("long"),
    )
    cond_invalid_sched = (
        F.col("scheduled_block_time_sec").isNull()
        | (F.col("scheduled_block_time_sec") <= MIN_VALID_TIME_SEC)
        | (F.col("scheduled_block_time_sec") > MAX_VALID_SCHED_BLOCK_SEC)
    )

    for key_col in ["dep_ap_sched", "arr_ap_sched", "ac_registration"]:
        cond_missing_keys = cond_missing_keys | F.col(key_col).isNull() | (F.trim(F.col(key_col)) == "")

    cond_same_airport = F.col("dep_ap_sched") == F.col("arr_ap_sched")
    cond_airport_mismatch = (
        (F.col("arr_ap_actual").isNotNull() & (F.col("arr_ap_actual") != F.col("arr_ap_sched")))
        | (F.col("dep_ap_actual").isNotNull() & (F.col("dep_ap_actual") != F.col("dep_ap_sched")))
    )

    df_dq = (
        df_dq
        .withColumn("taxi_out_sec", F.col("airborne_dt").cast("long") - F.col("offblock_dt").cast("long"))
        .withColumn("airborne_sec", F.col("landing_dt").cast("long") - F.col("airborne_dt").cast("long"))
        .withColumn("taxi_in_sec", F.col("onblock_dt").cast("long") - F.col("landing_dt").cast("long"))
        .withColumn("actual_block_time_sec", F.col("onblock_dt").cast("long") - F.col("offblock_dt").cast("long"))
        .withColumn("arrival_delay_sec", F.col("onblock_dt").cast("long") - F.col("arr_sched_dt").cast("long"))
        .withColumn("block_delay_sec", F.col("actual_block_time_sec") - F.col("scheduled_block_time_sec"))
    )

    cond_sequence_invalid = (
        F.col("offblock_dt").isNull()
        | F.col("airborne_dt").isNull()
        | F.col("landing_dt").isNull()
        | F.col("onblock_dt").isNull()
        | (F.col("offblock_dt") > F.col("airborne_dt"))
        | (F.col("airborne_dt") > F.col("landing_dt"))
        | (F.col("landing_dt") > F.col("onblock_dt"))
    )
    cond_invalid_actuals = (
        F.col("taxi_out_sec").isNull()
        | F.col("airborne_sec").isNull()
        | F.col("taxi_in_sec").isNull()
        | (F.col("taxi_out_sec") < MIN_VALID_TIME_SEC)
        | (F.col("airborne_sec") < MIN_VALID_TIME_SEC)
        | (F.col("taxi_in_sec") < MIN_VALID_TIME_SEC)
        | (F.col("actual_block_time_sec") <= MIN_VALID_TIME_SEC)
        | (F.col("actual_block_time_sec") > MAX_VALID_BLOCK_SEC)
    )
    cond_outlier_segments = (
        (F.col("taxi_out_sec") > MAX_VALID_TAXI_OUT_SEC)
        | (F.col("airborne_sec") > MAX_VALID_AIRBORNE_SEC)
        | (F.col("taxi_in_sec") > MAX_VALID_TAXI_IN_SEC)
    )
    cond_segment_gap = (
        F.abs(
            (
                F.coalesce(F.col("taxi_out_sec"), F.lit(0))
                + F.coalesce(F.col("airborne_sec"), F.lit(0))
                + F.coalesce(F.col("taxi_in_sec"), F.lit(0))
            )
            - F.coalesce(F.col("actual_block_time_sec"), F.lit(0))
        )
        > F.lit(MAX_SEGMENT_SUM_GAP_SEC)
    )

    should_inactivate = (
        cond_missing_keys
        | cond_invalid_sched
        | cond_same_airport
        | cond_airport_mismatch
        | cond_sequence_invalid
        | cond_invalid_actuals
        | cond_outlier_segments
        | cond_segment_gap
    )

    return (
        df_dq
        .withColumn("dq_missing_keys", cond_missing_keys.cast("int"))
        .withColumn("dq_invalid_sched", cond_invalid_sched.cast("int"))
        .withColumn("dq_same_airport", cond_same_airport.cast("int"))
        .withColumn("dq_airport_mismatch", cond_airport_mismatch.cast("int"))
        .withColumn("dq_sequence_invalid", cond_sequence_invalid.cast("int"))
        .withColumn("dq_invalid_actuals", cond_invalid_actuals.cast("int"))
        .withColumn("dq_outlier_segments", cond_outlier_segments.cast("int"))
        .withColumn("dq_segment_gap", cond_segment_gap.cast("int"))
        .withColumn("dq_any_flag", should_inactivate.cast("int"))
        .withColumn("is_active", F.when(should_inactivate, F.lit(False)).otherwise(F.lit(True)))
        .withColumn(
            "inactive_reason",
            F.when(cond_missing_keys, F.lit("MISSING_CRITICAL_KEYS"))
            .when(cond_invalid_sched, F.lit("INVALID_SCHED_BLOCK_TIME"))
            .when(cond_same_airport, F.lit("SAME_DEP_ARR_AIRPORT"))
            .when(cond_airport_mismatch, F.lit("AIRPORT_MISMATCH"))
            .when(cond_sequence_invalid, F.lit("BROKEN_OOOI_SEQUENCE"))
            .when(cond_invalid_actuals, F.lit("INVALID_ACTUAL_TIMES"))
            .when(cond_outlier_segments, F.lit("OUTLIER_SEGMENT_DURATION"))
            .when(cond_segment_gap, F.lit("SEGMENT_SUM_MISMATCH"))
            .otherwise(F.lit("ACTIVE")),
        )
    )


@dp.materialized_view(name=_fs_table("data_quality"), table_properties=DLT_TABLE_PROPERTIES)
def data_quality():
    return _apply_data_quality_rules(spark.read.table(_fs_table("enriched")))


@dp.materialized_view(name=_fs_table("cleaned_flight_data_full_table"), table_properties=DLT_TABLE_PROPERTIES)
def cleaned_flight_data_full_table():
    return spark.read.table(_fs_table("data_quality")).filter(F.col("is_active"))


def _create_ema_schema(entity_cols, target_cols_dict, count_prefix, half_life_days):
    fields = [StructField("event_date", DateType(), True)]
    for col_name in entity_cols:
        fields.append(StructField(col_name, StringType(), True))

    for prefix in target_cols_dict.values():
        for window_name in half_life_days.keys():
            fields.append(StructField(f"ema_{prefix}_{window_name}", DoubleType(), True))

    for window_name in half_life_days.keys():
        fields.append(StructField(f"ema_confidence_{count_prefix}_{window_name}", DoubleType(), True))

    return StructType(fields)


def _get_ema_compute_function(entity_cols, target_cols_dict, count_prefix, ema_schema, half_life_days):
    def compute_ema_dynamic(pdf: pd.DataFrame) -> pd.DataFrame:
        pdf = pdf.sort_values("event_date").reset_index(drop=True)
        lambdas = {k: math.log(2) / v for k, v in half_life_days.items()}

        states = {prefix: {k: np.nan for k in lambdas.keys()} for prefix in target_cols_dict.values()}
        conf = {k: 0.0 for k in lambdas.keys()}
        last_day = None

        out = {field.name: [] for field in ema_schema.fields}

        for _, row in pdf.iterrows():
            day = int(row["day_num"])
            cnt = float(row["daily_cnt"])
            dt = max(0, day - last_day) if last_day is not None else 0

            current_morning_conf = {}
            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                current_morning_conf[k] = conf[k] * decay if last_day is not None else 0.0

            out["event_date"].append(row["event_date"])
            for col_name in entity_cols:
                out[col_name].append(row[col_name])

            for _, prefix in target_cols_dict.items():
                for k in lambdas.keys():
                    out[f"ema_{prefix}_{k}"].append(states[prefix][k])

            for k in lambdas.keys():
                out[f"ema_confidence_{count_prefix}_{k}"].append(current_morning_conf[k])

            for k, lam in lambdas.items():
                decay = math.exp(-lam * dt) if last_day is not None else 0.0
                conf[k] = current_morning_conf[k] + cnt

                for _, prefix in target_cols_dict.items():
                    y = row[f"daily_avg_{prefix}"]
                    if np.isnan(states[prefix][k]):
                        states[prefix][k] = float(y) if pd.notnull(y) else np.nan
                    elif pd.notnull(y):
                        states[prefix][k] = (states[prefix][k] * decay) + (1.0 - decay) * float(y)
                    else:
                        states[prefix][k] = states[prefix][k] * decay

            last_day = day

        return pd.DataFrame(out)

    return compute_ema_dynamic


def _build_route_feature_store(df, entity_cols, target_cols_dict, count_prefix):
    df = df.withColumn(
        "duration_ratio",
        F.when(
            F.col("scheduled_block_time_sec") > 0,
            (F.col("actual_block_time_sec") / F.col("scheduled_block_time_sec")).cast("double"),
        ),
    )

    entities = df.select(*entity_cols).distinct()
    date_bounds = df.select(F.min("event_date").alias("min_date"), F.max("event_date").alias("max_date"))
    calendar = date_bounds.select(
        F.explode(F.sequence(F.col("min_date"), F.col("max_date"), F.expr("INTERVAL 1 DAY"))).alias("event_date")
    )

    markers = (
        entities.crossJoin(calendar)
        .withColumn("event_ts", F.to_timestamp(F.col("event_date")))
        .withColumn("is_marker", F.lit(1))
        .withColumn("_flight_row", F.lit(None).cast("int"))
    )
    flights = (
        df.select("event_ts", "event_date", *entity_cols, *target_cols_dict.keys())
        .withColumn("is_marker", F.lit(0))
        .withColumn("_flight_row", F.lit(1))
    )

    for col_name in target_cols_dict.keys():
        markers = markers.withColumn(col_name, F.lit(None).cast("double"))

    union_df = flights.unionByName(markers.select(flights.columns), allowMissingColumns=True)
    union_df = union_df.repartition(*entity_cols)
    order_col = F.col("event_ts").cast("long")

    windows = {
        "7d": Window.partitionBy(*entity_cols).orderBy(order_col).rangeBetween(-7 * SECONDS_IN_DAY, -1),
        "30d": Window.partitionBy(*entity_cols).orderBy(order_col).rangeBetween(-30 * SECONDS_IN_DAY, -1),
    }

    window_exprs = []
    for window_name, window_spec in windows.items():
        for src_col, prefix in target_cols_dict.items():
            window_exprs.extend(
                [
                    F.avg(src_col).over(window_spec).alias(f"avg_{prefix}_{window_name}"),
                    F.stddev(src_col).over(window_spec).alias(f"std_{prefix}_{window_name}"),
                    F.expr(f"percentile_approx({src_col}, 0.9)").over(window_spec).alias(f"p90_{prefix}_{window_name}"),
                    F.min(src_col).over(window_spec).alias(f"min_{prefix}_{window_name}"),
                    F.max(src_col).over(window_spec).alias(f"max_{prefix}_{window_name}"),
                ]
            )
        window_exprs.append(F.count("_flight_row").over(window_spec).cast("double").alias(f"count_{count_prefix}_{window_name}"))

    feat_df = union_df.select("*", *window_exprs)
    for _, prefix in target_cols_dict.items():
        feat_df = feat_df.withColumn(
            f"trend_{prefix}_7d",
            F.col(f"avg_{prefix}_7d") - F.col(f"avg_{prefix}_30d"),
        )

    feat_df = (
        feat_df
        .withColumn(f"has_hist_{count_prefix}_7d", F.when(F.col(f"count_{count_prefix}_7d") > 0, 1.0).otherwise(0.0))
        .withColumn(f"has_hist_{count_prefix}_30d", F.when(F.col(f"count_{count_prefix}_30d") > 0, 1.0).otherwise(0.0))
    )

    daily_cols = [F.avg(src_col).alias(f"daily_avg_{prefix}") for src_col, prefix in target_cols_dict.items()]
    daily_agg_flights = df.groupBy("event_date", *entity_cols).agg(*daily_cols, F.count("*").alias("daily_cnt"))
    daily_agg = (
        entities.crossJoin(calendar)
        .join(daily_agg_flights, on=["event_date", *entity_cols], how="left")
        .fillna({"daily_cnt": 0})
        .withColumn("day_num", F.datediff(F.col("event_date"), F.lit("1970-01-01")))
    )

    ema_schema = _create_ema_schema(entity_cols, target_cols_dict, count_prefix, HALF_LIFE_DAYS)
    ema_func = _get_ema_compute_function(entity_cols, target_cols_dict, count_prefix, ema_schema, HALF_LIFE_DAYS)
    ema_df = daily_agg.groupBy(*entity_cols).applyInPandas(ema_func, schema=ema_schema)
    feat_df = feat_df.join(ema_df, on=["event_date", *entity_cols], how="left")

    for _, prefix in target_cols_dict.items():
        for window_name in HALF_LIFE_DAYS.keys():
            feat_df = feat_df.withColumn(
                f"delta_ema_avg_{prefix}_{window_name}",
                F.col(f"ema_{prefix}_{window_name}") - F.col(f"avg_{prefix}_{window_name}"),
            )

    return feat_df.filter(F.col("is_marker") == 1).drop("event_ts", "is_marker", "_flight_row", *target_cols_dict.keys())


def _build_stand_features(df, is_taxi_out):
    ap_col = "dep_ap_sched" if is_taxi_out else "arr_ap_sched"
    stand_col = "dep_stand" if is_taxi_out else "arr_stand"
    target_col = "taxi_out_sec" if is_taxi_out else "taxi_in_sec"
    prefix = "out" if is_taxi_out else "in"

    clean_df = df.filter(F.col(stand_col).isNotNull() & (F.col(stand_col) != ""))
    markers = (
        clean_df.select("event_date", ap_col, stand_col).distinct()
        .withColumn("event_ts_unix", F.unix_timestamp("event_date"))
        .withColumn("target_val", F.lit(None).cast("double"))
        .withColumn("is_marker", F.lit(1))
    )
    flights = (
        clean_df.select("event_date", ap_col, stand_col, F.col(target_col).cast("double").alias("target_val"))
        .withColumn("event_ts_unix", F.unix_timestamp("event_date"))
        .withColumn("is_marker", F.lit(0))
    )

    union_df = flights.unionByName(markers)
    w_7d = Window.partitionBy(ap_col, stand_col).orderBy("event_ts_unix").rangeBetween(-7 * SECONDS_IN_DAY, -1)
    w_30d = Window.partitionBy(ap_col, stand_col).orderBy("event_ts_unix").rangeBetween(-30 * SECONDS_IN_DAY, -1)

    feat_df = (
        union_df
        .withColumn(f"stand_count_{prefix}_7d", F.count("target_val").over(w_7d).cast("double"))
        .withColumn(f"stand_avg_taxi_{prefix}_7d", F.avg("target_val").over(w_7d))
        .withColumn(f"stand_count_{prefix}_30d", F.count("target_val").over(w_30d).cast("double"))
        .withColumn(f"stand_avg_taxi_{prefix}_30d", F.avg("target_val").over(w_30d))
        .withColumn(f"stand_std_taxi_{prefix}_30d", F.stddev("target_val").over(w_30d))
        .withColumn(f"stand_p10_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.1)").over(w_30d))
        .withColumn(f"stand_p50_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.5)").over(w_30d))
        .withColumn(f"stand_p90_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.9)").over(w_30d))
        .withColumn(
            f"stand_trend_taxi_{prefix}_7d",
            F.col(f"stand_avg_taxi_{prefix}_7d") - F.col(f"stand_avg_taxi_{prefix}_30d"),
        )
    )

    rename_pairs = [
        (ap_col, f"fs_{'dep' if is_taxi_out else 'arr'}_ap_sched"),
        (stand_col, f"fs_{'dep' if is_taxi_out else 'arr'}_stand"),
    ]
    for old_name, new_name in rename_pairs:
        feat_df = feat_df.withColumnRenamed(old_name, new_name)

    return feat_df.filter(F.col("is_marker") == 1).drop("event_ts_unix", "target_val", "is_marker")


# =====================================================================================
# LEGACY fs_* (Iter1). TODO Iter2 cleanup: do usunięcia po przejściu na ft_* i retreningu.
# Zostawione obok nowych ft_* żeby nie zerwać obecnego treningu/scoringu przed migracją.
# =====================================================================================

_FS_TAXI_OUT_SPEC = dict(
    entity_cols=["dep_ap_sched"],
    target_cols_dict={"taxi_out_sec": "taxi_out", "duration_ratio": "dur_ratio_dep"},
    count_prefix="dep",
)

_FS_AIRBORNE_SPEC = dict(
    entity_cols=["dep_ap_sched", "arr_ap_sched"],
    target_cols_dict={
        "airborne_sec": "airborne",
        "arrival_delay_sec": "arrival_delay",
        "duration_ratio": "dur_ratio_route",
    },
    count_prefix="route",
)

_FS_TAXI_IN_SPEC = dict(
    entity_cols=["arr_ap_sched"],
    target_cols_dict={"taxi_in_sec": "taxi_in", "duration_ratio": "dur_ratio_arr"},
    count_prefix="arr",
)


@dp.materialized_view(
    name=_fs_table("fs_taxi_out_features"),
    schema=route_schema_ddl(**_FS_TAXI_OUT_SPEC),
    table_properties=DLT_TABLE_PROPERTIES,
)
def fs_taxi_out_features():
    return _build_route_feature_store(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FS_TAXI_OUT_SPEC,
    )


@dp.materialized_view(
    name=_fs_table("fs_airborne_features"),
    schema=route_schema_ddl(**_FS_AIRBORNE_SPEC, extra_pk_col="route_id"),
    table_properties=DLT_TABLE_PROPERTIES,
)
def fs_airborne_features():
    return _build_route_feature_store(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FS_AIRBORNE_SPEC,
    ).withColumn(
        "route_id",
        F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")),
    )


@dp.materialized_view(
    name=_fs_table("fs_taxi_in_features"),
    schema=route_schema_ddl(**_FS_TAXI_IN_SPEC),
    table_properties=DLT_TABLE_PROPERTIES,
)
def fs_taxi_in_features():
    return _build_route_feature_store(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FS_TAXI_IN_SPEC,
    )


@dp.materialized_view(
    name=_fs_table("fs_stand_out_features"),
    schema=stand_schema_ddl(is_taxi_out=True),
    table_properties=DLT_TABLE_PROPERTIES,
)
def fs_stand_out_features():
    return _build_stand_features(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")), is_taxi_out=True
    ).withColumn(
        "stand_id",
        F.concat_ws("_", F.col("fs_dep_ap_sched"), F.col("fs_dep_stand")),
    )


@dp.materialized_view(
    name=_fs_table("fs_stand_in_features"),
    schema=stand_schema_ddl(is_taxi_out=False),
    table_properties=DLT_TABLE_PROPERTIES,
)
def fs_stand_in_features():
    return _build_stand_features(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")), is_taxi_out=False
    ).withColumn(
        "stand_id",
        F.concat_ws("_", F.col("fs_arr_ap_sched"), F.col("fs_arr_stand")),
    )


_SCD2_VERSION_TS = "__START_AT"


def _stream_source(table_name):
    """readStream ze źródłowej tabeli SCD2 (źródło bywa aktualizowane -> skipChangeCommits)."""
    return (
        spark.readStream
        .option("skipChangeCommits", "true")
        .table(_source_table(table_name))
    )


@dp.table(name=_fs_table("ft_leg_status"), schema=leg_status_schema_ddl(), table_properties=DLT_TABLE_PROPERTIES)
def ft_leg_status():
    src = _stream_source("netline___schedops__leg")
    return (
        src
        .filter(F.col("counter") == 0)
        .filter(F.col("leg_type").isin(["J", "C", "G"]))
        .filter(F.col("leg_state") == "ARR")
        .withColumn("event_ts", F.col("dep_sched_dt").cast("timestamp"))
        .withColumn("event_date", F.to_date("dep_sched_dt"))
        .withColumn("ac_registration", F.substring(F.col("ac_registration"), 1, 4))
        .withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
        .withColumn("fn_full_number", F.concat(F.col("fn_carrier"), F.col("fn_number")))
        .select(
            "leg_no", "event_ts", "event_date", "leg_state", "leg_type", "marker",
            "ac_owner", "ac_registration", "ac_subtype", "commercial_carrier",
            "dep_ap_sched", "arr_ap_sched", "dep_sched_dt", "arr_sched_dt",
            "counter", "isLO", "fn_full_number",
        )
    )


@dp.table(name=_fs_table("ft_leg_times"), schema=leg_times_schema_ddl(), table_properties=DLT_TABLE_PROPERTIES)
def ft_leg_times():
    src = _stream_source("netline___schedops__leg_times")
    return src.select(
        "leg_no", "offblock_dt", "airborne_dt", "landing_dt", "onblock_dt",
    )


@dp.table(name=_fs_table("ft_leg_misc"), schema=leg_misc_schema_ddl(), table_properties=DLT_TABLE_PROPERTIES)
def ft_leg_misc():
    src = _stream_source("netline___schedops__leg_misc")
    return (
        src
        .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
        .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
        .select("leg_no", "dep_stand", "arr_stand")
    )


def _parse_coord_lat(col):
    return F.when(
        F.substring(col, 1, 1).isin(["N", "S"]),
        F.when(F.substring(col, 1, 1) == "N", 1).otherwise(-1)
        * (F.substring(col, 2, 2).cast("int") + F.substring(col, 4, 2).cast("int") / 60.0),
    ).otherwise(F.lit(None))


def _parse_coord_lon(col):
    return F.when(
        F.substring(col, 1, 1).isin(["E", "W"]),
        F.when(F.substring(col, 1, 1) == "E", 1).otherwise(-1)
        * (F.substring(col, 2, 3).cast("int") + F.substring(col, 5, 2).cast("int") / 60.0),
    ).otherwise(F.lit(None))


@dp.table(name=_fs_table("ft_airport_timezone"), schema=airport_timezone_schema_ddl(), table_properties=DLT_TABLE_PROPERTIES)
def ft_airport_timezone():
    # stream-static join: ap_basics jako stream, time_zone jako statyczny broadcast.
    apb = _stream_source("netline___schedops__ap_basics")
    tzd = F.broadcast(spark.read.table(_source_table("netline___schedops__time_zone")))
    apt = apb.join(tzd, F.col("time_zone") == F.col("time_zone_code"), "left")
    # lat/lon w STOPNIACH (degrees) — zgodnie z UDF haversine_km/is_eastbound (math.radians w środku).
    return (
        apt
        .withColumn("valid_ts", F.to_timestamp(F.col("valid_since")))
        .withColumn("lat_deg", _parse_coord_lat(F.col("coord_latitude")).cast("double"))
        .withColumn("lon_deg", _parse_coord_lon(F.col("coord_longitude")).cast("double"))
        .withColumn("utc_offset_min", F.coalesce(F.col("diff_utc_lst"), F.lit(0)).cast("int"))
        .withColumn("valid_until", F.to_date(F.col("valid_until")))
        .select(
            F.col("iata_ap_code"), "valid_ts", "lat_deg", "lon_deg", "utc_offset_min", "valid_until",
        )
    )


def _build_daily_stats(df, entity_cols, target_cols_dict, count_prefix):
    """B2 — statystyki dzienne BEZ densyfikacji kalendarza.

    Różnice względem legacy `_build_route_feature_store`:
      - markery TYLKO na dniach z eventami (brak `entities.crossJoin(calendar)`),
      - EMA liczona po dniach z eventami (daily_agg bez crossJoin),
      - dochodzi `days_since_last_event` (luka dni od poprzedniego eventu encji),
        spójna z PIT lookup po timestamp_lookup_key.
    """
    df = df.withColumn(
        "duration_ratio",
        F.when(
            F.col("scheduled_block_time_sec") > 0,
            (F.col("actual_block_time_sec") / F.col("scheduled_block_time_sec")).cast("double"),
        ),
    )

    # Markery TYLKO na dniach z eventami (B2: brak crossJoin(calendar))
    markers = (
        df.select("event_date", *entity_cols).distinct()
        .withColumn("event_ts", F.to_timestamp(F.col("event_date")))
        .withColumn("is_marker", F.lit(1))
        .withColumn("_flight_row", F.lit(None).cast("int"))
    )
    flights = (
        df.select("event_ts", "event_date", *entity_cols, *target_cols_dict.keys())
        .withColumn("is_marker", F.lit(0))
        .withColumn("_flight_row", F.lit(1))
    )
    for col_name in target_cols_dict.keys():
        markers = markers.withColumn(col_name, F.lit(None).cast("double"))

    union_df = flights.unionByName(markers.select(flights.columns), allowMissingColumns=True)
    union_df = union_df.repartition(*entity_cols)
    order_col = F.col("event_ts").cast("long")

    windows = {
        "7d": Window.partitionBy(*entity_cols).orderBy(order_col).rangeBetween(-7 * SECONDS_IN_DAY, -1),
        "30d": Window.partitionBy(*entity_cols).orderBy(order_col).rangeBetween(-30 * SECONDS_IN_DAY, -1),
    }

    window_exprs = []
    for window_name, window_spec in windows.items():
        for src_col, prefix in target_cols_dict.items():
            window_exprs.extend(
                [
                    F.avg(src_col).over(window_spec).alias(f"avg_{prefix}_{window_name}"),
                    F.stddev(src_col).over(window_spec).alias(f"std_{prefix}_{window_name}"),
                    F.expr(f"percentile_approx({src_col}, 0.9)").over(window_spec).alias(f"p90_{prefix}_{window_name}"),
                    F.min(src_col).over(window_spec).alias(f"min_{prefix}_{window_name}"),
                    F.max(src_col).over(window_spec).alias(f"max_{prefix}_{window_name}"),
                ]
            )
        window_exprs.append(F.count("_flight_row").over(window_spec).cast("double").alias(f"count_{count_prefix}_{window_name}"))

    feat_df = union_df.select("*", *window_exprs)
    for _, prefix in target_cols_dict.items():
        feat_df = feat_df.withColumn(
            f"trend_{prefix}_7d",
            F.col(f"avg_{prefix}_7d") - F.col(f"avg_{prefix}_30d"),
        )

    feat_df = (
        feat_df
        .withColumn(f"has_hist_{count_prefix}_7d", F.when(F.col(f"count_{count_prefix}_7d") > 0, 1.0).otherwise(0.0))
        .withColumn(f"has_hist_{count_prefix}_30d", F.when(F.col(f"count_{count_prefix}_30d") > 0, 1.0).otherwise(0.0))
    )

    # Daily agg BEZ crossJoin(calendar) — tylko dni z eventami (B2)
    daily_cols = [F.avg(src_col).alias(f"daily_avg_{prefix}") for src_col, prefix in target_cols_dict.items()]
    daily_agg_flights = df.groupBy("event_date", *entity_cols).agg(*daily_cols, F.count("*").alias("daily_cnt"))

    # days_since_last_event: luka dni względem poprzedniego eventu encji
    w_gap = Window.partitionBy(*entity_cols).orderBy("event_date")
    daily_agg_flights = (
        daily_agg_flights
        .withColumn("prev_event_date", F.lag("event_date").over(w_gap))
        .withColumn(
            "days_since_last_event",
            F.when(F.col("prev_event_date").isNull(), F.lit(0.0))
            .otherwise(F.datediff(F.col("event_date"), F.col("prev_event_date")).cast("double")),
        )
        .drop("prev_event_date")
    )

    daily_agg = daily_agg_flights.withColumn("day_num", F.datediff(F.col("event_date"), F.lit("1970-01-01")))

    ema_schema = _create_ema_schema(entity_cols, target_cols_dict, count_prefix, HALF_LIFE_DAYS)
    ema_func = _get_ema_compute_function(entity_cols, target_cols_dict, count_prefix, ema_schema, HALF_LIFE_DAYS)
    ema_df = daily_agg.groupBy(*entity_cols).applyInPandas(ema_func, schema=ema_schema)
    feat_df = feat_df.join(ema_df, on=["event_date", *entity_cols], how="left")

    for _, prefix in target_cols_dict.items():
        for window_name in HALF_LIFE_DAYS.keys():
            feat_df = feat_df.withColumn(
                f"delta_ema_avg_{prefix}_{window_name}",
                F.col(f"ema_{prefix}_{window_name}") - F.col(f"avg_{prefix}_{window_name}"),
            )

    # Dołącz days_since_last_event do wierszy feature (per event-day)
    days_since = daily_agg_flights.select("event_date", *entity_cols, "days_since_last_event")
    feat_df = feat_df.join(days_since, on=["event_date", *entity_cols], how="left")

    return feat_df.filter(F.col("is_marker") == 1).drop("event_ts", "is_marker", "_flight_row", *target_cols_dict.keys())


def _build_stand_daily(df, is_taxi_out):
    """B2 — stand-level statystyki + days_since_last_event. Markery już są tylko na
    dniach z eventami (stand features nigdy nie używały densyfikacji kalendarza)."""
    ap_col = "dep_ap_sched" if is_taxi_out else "arr_ap_sched"
    stand_col = "dep_stand" if is_taxi_out else "arr_stand"
    target_col = "taxi_out_sec" if is_taxi_out else "taxi_in_sec"
    prefix = "out" if is_taxi_out else "in"

    clean_df = df.filter(F.col(stand_col).isNotNull() & (F.col(stand_col) != ""))
    markers = (
        clean_df.select("event_date", ap_col, stand_col).distinct()
        .withColumn("event_ts_unix", F.unix_timestamp("event_date"))
        .withColumn("target_val", F.lit(None).cast("double"))
        .withColumn("is_marker", F.lit(1))
    )
    flights = (
        clean_df.select("event_date", ap_col, stand_col, F.col(target_col).cast("double").alias("target_val"))
        .withColumn("event_ts_unix", F.unix_timestamp("event_date"))
        .withColumn("is_marker", F.lit(0))
    )

    union_df = flights.unionByName(markers)
    w_7d = Window.partitionBy(ap_col, stand_col).orderBy("event_ts_unix").rangeBetween(-7 * SECONDS_IN_DAY, -1)
    w_30d = Window.partitionBy(ap_col, stand_col).orderBy("event_ts_unix").rangeBetween(-30 * SECONDS_IN_DAY, -1)

    feat_df = (
        union_df
        .withColumn(f"stand_count_{prefix}_7d", F.count("target_val").over(w_7d).cast("double"))
        .withColumn(f"stand_avg_taxi_{prefix}_7d", F.avg("target_val").over(w_7d))
        .withColumn(f"stand_count_{prefix}_30d", F.count("target_val").over(w_30d).cast("double"))
        .withColumn(f"stand_avg_taxi_{prefix}_30d", F.avg("target_val").over(w_30d))
        .withColumn(f"stand_std_taxi_{prefix}_30d", F.stddev("target_val").over(w_30d))
        .withColumn(f"stand_p10_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.1)").over(w_30d))
        .withColumn(f"stand_p50_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.5)").over(w_30d))
        .withColumn(f"stand_p90_taxi_{prefix}_30d", F.expr("percentile_approx(target_val, 0.9)").over(w_30d))
        .withColumn(
            f"stand_trend_taxi_{prefix}_7d",
            F.col(f"stand_avg_taxi_{prefix}_7d") - F.col(f"stand_avg_taxi_{prefix}_30d"),
        )
    )

    # days_since_last_event per (ap, stand) względem poprzedniego event-day
    w_gap = Window.partitionBy(ap_col, stand_col).orderBy("event_date")
    gaps = (
        clean_df.select("event_date", ap_col, stand_col).distinct()
        .withColumn("prev_event_date", F.lag("event_date").over(w_gap))
        .withColumn(
            "days_since_last_event",
            F.when(F.col("prev_event_date").isNull(), F.lit(0.0))
            .otherwise(F.datediff(F.col("event_date"), F.col("prev_event_date")).cast("double")),
        )
        .select("event_date", ap_col, stand_col, "days_since_last_event")
    )
    feat_df = feat_df.join(gaps, on=["event_date", ap_col, stand_col], how="left")

    rename_pairs = [
        (ap_col, f"fs_{'dep' if is_taxi_out else 'arr'}_ap_sched"),
        (stand_col, f"fs_{'dep' if is_taxi_out else 'arr'}_stand"),
    ]
    for old_name, new_name in rename_pairs:
        feat_df = feat_df.withColumnRenamed(old_name, new_name)

    return feat_df.filter(F.col("is_marker") == 1).drop("event_ts_unix", "target_val", "is_marker")


_FT_TAXI_OUT_SPEC = dict(
    entity_cols=["dep_ap_sched"],
    target_cols_dict={"taxi_out_sec": "taxi_out", "duration_ratio": "dur_ratio_dep"},
    count_prefix="dep",
)

_FT_AIRBORNE_SPEC = dict(
    entity_cols=["dep_ap_sched", "arr_ap_sched"],
    target_cols_dict={
        "airborne_sec": "airborne",
        "arrival_delay_sec": "arrival_delay",
        "duration_ratio": "dur_ratio_route",
    },
    count_prefix="route",
)

_FT_TAXI_IN_SPEC = dict(
    entity_cols=["arr_ap_sched"],
    target_cols_dict={"taxi_in_sec": "taxi_in", "duration_ratio": "dur_ratio_arr"},
    count_prefix="arr",
)

@dp.materialized_view(
    name=_fs_table("ft_airport_daily_taxi_out"),
    schema=daily_stats_schema_ddl(**_FT_TAXI_OUT_SPEC),
    table_properties=DLT_TABLE_PROPERTIES,
)
@dp.expect("valid_event_date", "event_date >= '2023-01-01'")
@dp.expect("non_negative_avg", "avg_taxi_out_7d IS NULL OR avg_taxi_out_7d >= 0")
def ft_airport_daily_taxi_out():
    return _build_daily_stats(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FT_TAXI_OUT_SPEC,
    )

@dp.materialized_view(
    name=_fs_table("ft_route_daily_stats"),
    schema=daily_stats_schema_ddl(**_FT_AIRBORNE_SPEC, extra_pk_col="route_id"),
    table_properties=DLT_TABLE_PROPERTIES,
)
@dp.expect("valid_event_date", "event_date >= '2023-01-01'")
@dp.expect("non_negative_avg", "avg_taxi_out_7d IS NULL OR avg_taxi_out_7d >= 0")
def ft_route_daily_stats():
    return _build_daily_stats(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FT_AIRBORNE_SPEC,
    ).withColumn(
        "route_id",
        F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")),
    )

@dp.materialized_view(
    name=_fs_table("ft_airport_daily_taxi_in"),
    schema=daily_stats_schema_ddl(**_FT_TAXI_IN_SPEC),
    table_properties=DLT_TABLE_PROPERTIES,
)
@dp.expect("valid_event_date", "event_date >= '2023-01-01'")
@dp.expect("non_negative_avg", "avg_taxi_out_7d IS NULL OR avg_taxi_out_7d >= 0")
def ft_airport_daily_taxi_in():
    return _build_daily_stats(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")),
        **_FT_TAXI_IN_SPEC,
    )

@dp.materialized_view(
    name=_fs_table("ft_stand_daily_out"),
    schema=stand_daily_schema_ddl(is_taxi_out=True),
    table_properties=DLT_TABLE_PROPERTIES,
)
@dp.expect("valid_event_date", "event_date >= '2023-01-01'")
@dp.expect("non_negative_avg", "avg_taxi_out_7d IS NULL OR avg_taxi_out_7d >= 0")
def ft_stand_daily_out():
    return _build_stand_daily(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")), is_taxi_out=True
    ).withColumn(
        "stand_id",
        F.concat_ws("_", F.col("fs_dep_ap_sched"), F.col("fs_dep_stand")),
    )

@dp.materialized_view(
    name=_fs_table("ft_stand_daily_in"),
    schema=stand_daily_schema_ddl(is_taxi_out=False),
    table_properties=DLT_TABLE_PROPERTIES,
)
@dp.expect("valid_event_date", "event_date >= '2023-01-01'")
@dp.expect("non_negative_avg", "avg_taxi_out_7d IS NULL OR avg_taxi_out_7d >= 0")
def ft_stand_daily_in():
    return _build_stand_daily(
        spark.read.table(_fs_table("cleaned_flight_data_full_table")), is_taxi_out=False
    ).withColumn(
        "stand_id",
        F.concat_ws("_", F.col("fs_arr_ap_sched"), F.col("fs_arr_stand")),
    )
