from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from typing import Any, Mapping

from pyspark.sql import functions as F
from pyspark.sql.types import DateType, DoubleType, StringType, StructField, StructType


_RUNTIME = {}


def configure_runtime(settings: Any, spark=None) -> None:
    global _RUNTIME

    if is_dataclass(settings):
        values = asdict(settings)
    elif isinstance(settings, Mapping):
        values = dict(settings)
    else:
        raise TypeError(f"Nieobsługiwany typ settings: {type(settings)!r}")

    _RUNTIME = values
    globals().update(values)

    if spark is not None and values.get("SPARK_TIMEZONE"):
        spark.conf.set("spark.sql.session.timeZone", values["SPARK_TIMEZONE"])


def _assert_runtime_ready() -> None:
    if not _RUNTIME:
        raise RuntimeError(
            "ml_project.common nie został skonfigurowany. "
            "Uruchom configure_runtime(settings, spark) przed użyciem helperów."
        )

def apply_data_quality_rules(df):
    """
    Twarde Quality Gates dla treningu i scoringu.
    """
    _assert_runtime_ready()
    cols = set(df.columns)
    df_dq = df

    cond_invalid_sched = F.lit(False)
    cond_missing_keys = F.lit(False)
    cond_invalid_actuals = F.lit(False)
    cond_airport_mismatch = F.lit(False)
    cond_same_airport = F.lit(False)
    cond_sequence_invalid = F.lit(False)
    cond_outlier_segments = F.lit(False)
    cond_segment_gap = F.lit(False)

    if {"arr_sched_dt", "dep_sched_dt"}.issubset(cols):
        df_dq = df_dq.withColumn(
            "scheduled_block_time_sec",
            F.col("arr_sched_dt").cast("long") - F.col("dep_sched_dt").cast("long")
        )
        cond_invalid_sched = (
            F.col("scheduled_block_time_sec").isNull() |
            (F.col("scheduled_block_time_sec") <= MIN_VALID_TIME_SEC) |
            (F.col("scheduled_block_time_sec") > MAX_VALID_SCHED_BLOCK_SEC)
        )

    for key_col in ["dep_ap_sched", "arr_ap_sched", "ac_registration"]:
        if key_col in cols:
            cond_missing_keys = cond_missing_keys | F.col(key_col).isNull() | (F.trim(F.col(key_col)) == "")

    if {"dep_ap_sched", "arr_ap_sched"}.issubset(cols):
        cond_same_airport = F.col("dep_ap_sched") == F.col("arr_ap_sched")

    if {"arr_ap_actual", "arr_ap_sched"}.issubset(cols):
        cond_airport_mismatch = cond_airport_mismatch | (
            F.col("arr_ap_actual").isNotNull() & (F.col("arr_ap_actual") != F.col("arr_ap_sched"))
        )
    if {"dep_ap_actual", "dep_ap_sched"}.issubset(cols):
        cond_airport_mismatch = cond_airport_mismatch | (
            F.col("dep_ap_actual").isNotNull() & (F.col("dep_ap_actual") != F.col("dep_ap_sched"))
        )

    if {"offblock_dt", "airborne_dt", "landing_dt", "onblock_dt", "arr_sched_dt"}.issubset(cols):
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
            F.col("offblock_dt").isNull() |
            F.col("airborne_dt").isNull() |
            F.col("landing_dt").isNull() |
            F.col("onblock_dt").isNull() |
            (F.col("offblock_dt") > F.col("airborne_dt")) |
            (F.col("airborne_dt") > F.col("landing_dt")) |
            (F.col("landing_dt") > F.col("onblock_dt"))
        )

        cond_invalid_actuals = (
            F.col("taxi_out_sec").isNull() |
            F.col("airborne_sec").isNull() |
            F.col("taxi_in_sec").isNull() |
            (F.col("taxi_out_sec") < MIN_VALID_TIME_SEC) |
            (F.col("airborne_sec") < MIN_VALID_TIME_SEC) |
            (F.col("taxi_in_sec") < MIN_VALID_TIME_SEC) |
            (F.col("actual_block_time_sec") <= MIN_VALID_TIME_SEC) |
            (F.col("actual_block_time_sec") > MAX_VALID_BLOCK_SEC)
        )

        cond_outlier_segments = (
            (F.col("taxi_out_sec") > MAX_VALID_TAXI_OUT_SEC) |
            (F.col("airborne_sec") > MAX_VALID_AIRBORNE_SEC) |
            (F.col("taxi_in_sec") > MAX_VALID_TAXI_IN_SEC)
        )

        cond_segment_gap = (
            F.abs(
                (F.coalesce(F.col("taxi_out_sec"), F.lit(0)) +
                 F.coalesce(F.col("airborne_sec"), F.lit(0)) +
                 F.coalesce(F.col("taxi_in_sec"), F.lit(0))) -
                F.coalesce(F.col("actual_block_time_sec"), F.lit(0))
            ) > F.lit(MAX_SEGMENT_SUM_GAP_SEC)
        )

    should_inactivate = (
        cond_missing_keys |
        cond_invalid_sched |
        cond_same_airport |
        cond_airport_mismatch |
        cond_sequence_invalid |
        cond_invalid_actuals |
        cond_outlier_segments |
        cond_segment_gap
    )

    df_dq = (
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
             .otherwise(F.lit("ACTIVE"))
        )
    )
    return df_dq

def add_marker_columns(df, max_marker_length: int = 17):
    """Rozkłada kolumnę `marker` na binarne kolumny marker_1..marker_N.

    Y → 0, N → 1, brak/inny → NaN. Shared utility dla scoring.py i common.py.
    (feature_store.py ma własną kopię ze względu na izolację DLT pipeline.)
    """
    for i in range(1, max_marker_length + 1):
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

def get_airport_features(spark, prefix):
    """Pobiera lotniska i wyciąga przesunięcie czasowe oraz współrzędne"""
    _assert_runtime_ready()
    apb = spark.read.table(AP_BASICS_TABLE)
    tzd = spark.read.table(TIME_ZONE_TABLE)
    apt = apb.join(tzd, F.col("time_zone") == F.col("time_zone_code"), "left")

    lat_deg = F.when(
        F.substring(F.col("coord_latitude"), 1, 1).isin(["N", "S"]),
        F.when(F.substring(F.col("coord_latitude"), 1, 1) == "N", 1).otherwise(-1)
        * (F.substring(F.col("coord_latitude"), 2, 2).cast("int") + F.substring(F.col("coord_latitude"), 4, 2).cast("int") / 60.0)
    ).otherwise(F.lit(None))

    lon_deg = F.when(
        F.substring(F.col("coord_longitude"), 1, 1).isin(["E", "W"]),
        F.when(F.substring(F.col("coord_longitude"), 1, 1) == "E", 1).otherwise(-1)
        * (F.substring(F.col("coord_longitude"), 2, 3).cast("int") + F.substring(F.col("coord_longitude"), 5, 2).cast("int") / 60.0)
    ).otherwise(F.lit(None))

    apt = apt.withColumn(f"{prefix}_lat_rad", F.radians(lat_deg))
    apt = apt.withColumn(f"{prefix}_lon_rad", F.radians(lon_deg))

    res = apt.select(
        F.col("iata_ap_code").alias(f"{prefix}_ap_code"),
        F.col(f"{prefix}_lat_rad"),
        F.col(f"{prefix}_lon_rad"),
        F.coalesce(F.col("diff_utc_lst"), F.lit(0)).alias(f"{prefix}_utc_offset_min"),
        F.to_date(apb["valid_since"]).alias(f"{prefix}_valid_since"),
        F.to_date(apb["valid_until"]).alias(f"{prefix}_valid_until")
    )
    return res.distinct()

def enrich_with_local_context(df, spark):
    """Oblicza dystans, lokalny czas oraz zmienne trygonometryczne i wektor wiatru."""
    _assert_runtime_ready()
    dep_apt = F.broadcast(get_airport_features(spark, "dep"))
    arr_apt = F.broadcast(get_airport_features(spark, "arr"))

    df = df.join(
        dep_apt,
        (F.col("dep_ap_sched") == F.col("dep_ap_code")) &
        (F.to_date(F.col("dep_sched_dt")) >= F.col("dep_valid_since")) &
        (F.to_date(F.col("dep_sched_dt")) <= F.coalesce(F.col("dep_valid_until"), F.to_date(F.lit("2099-12-31")))),
        "left"
    )
    df = df.join(
        arr_apt,
        (F.col("arr_ap_sched") == F.col("arr_ap_code")) &
        (F.to_date(F.col("arr_sched_dt")) >= F.col("arr_valid_since")) &
        (F.to_date(F.col("arr_sched_dt")) <= F.coalesce(F.col("arr_valid_until"), F.to_date(F.lit("2099-12-31")))),
        "left"
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
        F.pow(F.sin(F.col("dlat") / 2.0), 2) +
        F.cos(F.col("dep_lat_rad")) * F.cos(F.col("arr_lat_rad")) * F.pow(F.sin(F.col("dlon") / 2.0), 2)
    )
    df = df.withColumn("c", 2.0 * F.atan2(F.sqrt(F.col("a")), F.sqrt(F.greatest(F.lit(0.0), 1.0 - F.col("a")))))
    df = df.withColumn("distance_km", F.round(6371.0 * F.col("c"), 2))
    df = df.withColumn("distance_km", F.coalesce(F.col("distance_km"), F.lit(0.0)))

    cols_to_drop = [
        "dep_ap_code", "dep_lat_rad", "dep_lon_rad", "dep_valid_since", "dep_valid_until",
        "arr_ap_code", "arr_lat_rad", "arr_lon_rad", "arr_valid_since", "arr_valid_until",
        "dep_local_ts", "arr_local_ts", "dlat", "dlon", "a", "c"
    ]
    return df.drop(*cols_to_drop)

def get_cleaned_flight_data(spark_session, start_date_str, active_only=True):
    """
    Łączy tabele źródłowe, wykonuje DQ check i zwraca dane wzbogacone
    o lokalny kontekst. active_only=True zwraca tylko clean ops,
    active_only=False zwraca cały świat z flagami DQ.
    """
    _assert_runtime_ready()
    from pyspark.sql.window import Window

    base = (
        spark_session.read.table(LABELS_TABLE)
        .filter(F.col("__END_AT").isNull())
        .filter(F.col("counter") == 0)
        .filter(F.col("leg_type").isin(INCLUDED_LEG_TYPES))
        .filter(F.col("leg_state") == LEG_STATE_ARR)
        .filter(F.to_date(F.col("dep_sched_dt")) >= F.to_date(F.lit(start_date_str)))
        .filter(F.to_date(F.col("dep_sched_dt")) < F.to_date(F.lit(DATA_CUTOFF_DATE)))
        .withColumn("event_ts", F.col("dep_sched_dt"))
        .withColumn("event_date", F.to_date("dep_sched_dt"))
        .withColumn("ac_registration", F.substring(F.col("ac_registration"), 1, 4))
        .withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
        .withColumn("fn_full_number", F.concat(F.col("fn_carrier"), F.col("fn_number"))) # NUMER LOTU
    )

    for i in range(1, MAX_MARKER_LENGTH + 1):
        base = base.withColumn(
            f"marker_{i}",
            F.when(
                F.length(F.col("marker")) >= i,
                F.when(F.substring(F.col("marker"), i, 1) == "Y", 0)
                 .when(F.substring(F.col("marker"), i, 1) == "N", 1)
                 .otherwise(F.lit(None).cast("double"))
            ).otherwise(F.lit(None).cast("double"))
        )

    leg_times_raw = spark_session.read.table(LEG_TIMES_TABLE)
    if "__END_AT" in leg_times_raw.columns:
        leg_times_raw = leg_times_raw.filter(F.col("__END_AT").isNull())
    w_times = Window.partitionBy("leg_no").orderBy(F.col("update_key").desc())

    leg_times_latest = (
        leg_times_raw
        .withColumn("rn", F.row_number().over(w_times))
        .filter(F.col("rn") == 1)
        .select("leg_no", "offblock_dt", "airborne_dt", "landing_dt", "onblock_dt")
    )

    leg_remark_raw = spark_session.read.table(LEG_REMARK_TABLE)
    
    if "update_key" in leg_remark_raw.columns:
        w_remark = Window.partitionBy("leg_no").orderBy(F.col("update_key").desc())
    else:
        w_remark = Window.partitionBy("leg_no").orderBy(F.col("entry_dt").desc())
    
    leg_remark_filtered = leg_remark_raw.filter(F.col("usage") == 'F')
    
    if "__END_AT" in leg_remark_raw.columns:
        leg_remark_filtered = leg_remark_filtered.filter(F.col("__END_AT").isNull())
        
    leg_remark_latest = (
        leg_remark_filtered
        .withColumn("rn", F.row_number().over(w_remark))
        .filter(F.col("rn") == 1)
        .withColumn("eet_str", F.regexp_extract(F.col("text"), r'EET\s*:\s*(\d+)', 1))
        .withColumn(
            "netline_eet_duration_min",
            F.when(
                F.col("eet_str") != "",
                (F.substring(F.lpad(F.col("eet_str"), 4, '0'), 1, 2).cast("int") * 60) + 
                 F.substring(F.lpad(F.col("eet_str"), 4, '0'), 3, 2).cast("int")
            ).otherwise(F.lit(None).cast("int"))
        )
        .select("leg_no", "netline_eet_duration_min")
    )
    leg_misc_current = spark_session.read.table(LEG_MISC_TABLE)
    if "__END_AT" in leg_misc_current.columns:
        leg_misc_current = leg_misc_current.filter(F.col("__END_AT").isNull())

    leg_misc_current = (
        leg_misc_current
        .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
        .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
        .select("leg_no", "dep_stand", "arr_stand")
    )
    df_joined = (
        base.join(leg_times_latest, on="leg_no", how="left")
        .join(leg_remark_latest, on="leg_no", how="left")
        .join(leg_misc_current, on="leg_no", how="left")
    )
    df_dq = apply_data_quality_rules(df_joined)
    df_final = enrich_with_local_context(df_dq, spark_session)

    if active_only:
        return df_final.filter(F.col("is_active") == True)
    return df_final
