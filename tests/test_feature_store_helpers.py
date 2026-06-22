"""Testy generatorów DDL z `feature_store.py` (route_schema_ddl / stand_schema_ddl).

`feature_store.py` importuje pyspark i używa globalnego `spark` (wstrzykiwanego przez
Databricks) w dekoratorach `@dp.materialized_view` ewaluowanych przy imporcie modułu.
Mockujemy pyspark + `spark`, żeby zaimportować moduł lokalnie. Testowane funkcje budują
wyłącznie stringi DDL i nie dotykają Sparka, więc wynik jest realny (nie zmockowany).
"""
import builtins
import sys
from unittest.mock import MagicMock

for _mod in (
    "pyspark",
    "pyspark.pipelines",
    "pyspark.sql",
    "pyspark.sql.functions",
    "pyspark.sql.types",
    "pyspark.sql.window",
):
    sys.modules.setdefault(_mod, MagicMock())
builtins.spark = MagicMock()

import feature_store as fs


def test_route_schema_ddl_taxi_out_contract():
    ddl = fs.route_schema_ddl(
        entity_cols=["dep_ap_sched"],
        target_cols_dict={"taxi_out_sec": "taxi_out", "duration_ratio": "dur_ratio_dep"},
        count_prefix="dep",
    )
    assert "PRIMARY KEY" in ddl
    assert "TIMESERIES" in ddl
    assert "DOUBLE" in ddl
    assert "dep_ap_sched STRING NOT NULL" in ddl
    assert "event_date DATE NOT NULL" in ddl
    # count i has_hist są DOUBLE (fix dla kompatybilności z MLflow signature)
    assert "count_dep_7d DOUBLE" in ddl
    assert "count_dep_30d DOUBLE" in ddl
    assert "has_hist_dep_7d DOUBLE" in ddl
    assert "has_hist_dep_30d DOUBLE" in ddl


def test_route_schema_ddl_airborne_extra_pk():
    ddl = fs.route_schema_ddl(
        entity_cols=["dep_ap_sched", "arr_ap_sched"],
        target_cols_dict={"airborne_sec": "airborne"},
        count_prefix="route",
        extra_pk_col="route_id",
    )
    assert "route_id STRING NOT NULL" in ddl
    assert "PRIMARY KEY (route_id, event_date TIMESERIES)" in ddl
    # entity_cols są dołączane jako zwykłe kolumny STRING
    assert "dep_ap_sched STRING" in ddl
    assert "arr_ap_sched STRING" in ddl


def test_route_schema_ddl_taxi_in():
    ddl = fs.route_schema_ddl(
        entity_cols=["arr_ap_sched"],
        target_cols_dict={"taxi_in_sec": "taxi_in"},
        count_prefix="arr",
    )
    assert "PRIMARY KEY (arr_ap_sched, event_date TIMESERIES)" in ddl
    assert "count_arr_7d DOUBLE" in ddl


def test_stand_schema_ddl_out_contract():
    ddl = fs.stand_schema_ddl(is_taxi_out=True)
    assert "stand_id STRING NOT NULL" in ddl
    assert "PRIMARY KEY (stand_id, event_date TIMESERIES)" in ddl
    assert "stand_count_out_7d DOUBLE" in ddl
    assert "stand_count_out_30d DOUBLE" in ddl
    assert "fs_dep_ap_sched STRING" in ddl
    assert "fs_dep_stand STRING" in ddl


def test_stand_schema_ddl_in_contract():
    ddl = fs.stand_schema_ddl(is_taxi_out=False)
    assert "PRIMARY KEY (stand_id, event_date TIMESERIES)" in ddl
    assert "stand_count_in_7d DOUBLE" in ddl
    assert "stand_count_in_30d DOUBLE" in ddl
    assert "fs_arr_ap_sched STRING" in ddl
    assert "fs_arr_stand STRING" in ddl


def test_stand_ddl_side_specific():
    out = fs.stand_schema_ddl(is_taxi_out=True)
    inn = fs.stand_schema_ddl(is_taxi_out=False)
    assert "_out_" in out and "_out_" not in inn
    assert "_in_" in inn and "_in_" not in out


# ===== Iter2 — nowe ft_* DDL buildery =====

def test_leg_status_ddl():
    ddl = fs.leg_status_schema_ddl()
    assert "leg_no LONG NOT NULL" in ddl  # leg_no jest LONG (nie STRING)
    assert "event_ts TIMESTAMP NOT NULL" in ddl
    assert "PRIMARY KEY (leg_no, event_ts TIMESERIES)" in ddl


def test_leg_times_and_misc_ddl():
    t = fs.leg_times_schema_ddl()
    assert "leg_no LONG NOT NULL" in t
    assert "PRIMARY KEY (leg_no, event_ts TIMESERIES)" in t
    assert "offblock_dt TIMESTAMP" in t and "onblock_dt TIMESTAMP" in t
    m = fs.leg_misc_schema_ddl()
    assert "leg_no LONG NOT NULL" in m
    assert "PRIMARY KEY (leg_no, event_ts TIMESERIES)" in m
    assert "dep_stand STRING" in m and "arr_stand STRING" in m


def test_airport_timezone_ddl_degrees():
    ddl = fs.airport_timezone_schema_ddl()
    assert "PRIMARY KEY (iata_ap_code, valid_ts TIMESERIES)" in ddl
    # lat/lon w STOPNIACH (parytet z UDF haversine/is_eastbound), NIE radiany
    assert "lat_deg DOUBLE" in ddl and "lon_deg DOUBLE" in ddl
    assert "lat_rad" not in ddl and "lon_rad" not in ddl
    assert "utc_offset_min INT" in ddl


def test_daily_stats_ddl_has_days_since():
    ddl = fs.daily_stats_schema_ddl(
        entity_cols=["dep_ap_sched"],
        target_cols_dict={"taxi_out_sec": "taxi_out", "duration_ratio": "dur_ratio_dep"},
        count_prefix="dep",
    )
    assert "PRIMARY KEY (dep_ap_sched, event_date TIMESERIES)" in ddl
    assert "days_since_last_event DOUBLE" in ddl  # B2
    assert "count_dep_7d DOUBLE" in ddl and "has_hist_dep_7d DOUBLE" in ddl


def test_daily_stats_ddl_route_extra_pk():
    ddl = fs.daily_stats_schema_ddl(
        entity_cols=["dep_ap_sched", "arr_ap_sched"],
        target_cols_dict={"airborne_sec": "airborne"},
        count_prefix="route",
        extra_pk_col="route_id",
    )
    assert "route_id STRING NOT NULL" in ddl
    assert "PRIMARY KEY (route_id, event_date TIMESERIES)" in ddl
    assert "days_since_last_event DOUBLE" in ddl


def test_stand_daily_ddl_has_days_since():
    out = fs.stand_daily_schema_ddl(is_taxi_out=True)
    assert "PRIMARY KEY (stand_id, event_date TIMESERIES)" in out
    assert "stand_count_out_7d DOUBLE" in out
    assert "days_since_last_event DOUBLE" in out
    inn = fs.stand_daily_schema_ddl(is_taxi_out=False)
    assert "stand_count_in_7d DOUBLE" in inn and "days_since_last_event DOUBLE" in inn
