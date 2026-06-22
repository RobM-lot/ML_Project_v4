"""Testy on-demand UDF (local time + geo) — czysty Python, bez Spark.

Import jak w test_feature_store_helpers: conftest dokłada `src/pipeline` na sys.path,
a on_demand_functions importuje tylko stdlib (math/datetime), więc ładuje się bez mocków.
"""
import math
from datetime import datetime

import on_demand_functions as o


def test_local_hour_basic():
    # 14:00 UTC + 120 min (UTC+2) -> 16:00 local
    dt = datetime(2026, 6, 9, 14, 0, 0)
    assert o.local_hour(dt, 120) == 16
    # null-safe
    assert o.local_hour(None, 120) == 0
    assert o.local_hour(dt, None) == 0


def test_local_dow_basic():
    # 2026-06-09 to wtorek -> weekday()=1
    dt = datetime(2026, 6, 9, 14, 0, 0)
    assert o.local_dow(dt, 0) == 1
    # +12h (720 min) -> 2026-06-10 02:00 (środa) -> 2
    assert o.local_dow(dt, 720) == 2
    assert o.local_dow(None, 0) == 0


def test_month_of():
    assert o.month_of(datetime(2026, 6, 9)) == 6
    assert o.month_of(None) == 1


def test_sin_cos_consistency():
    dt = datetime(2026, 6, 9, 14, 0, 0)
    s = o.sin_local_hour(dt, 0)
    c = o.cos_local_hour(dt, 0)
    assert abs(s * s + c * c - 1.0) < 1e-10

    s = o.sin_local_dow(dt, 0)
    c = o.cos_local_dow(dt, 0)
    assert abs(s * s + c * c - 1.0) < 1e-10

    s = o.sin_month_of(dt)
    c = o.cos_month_of(dt)
    assert abs(s * s + c * c - 1.0) < 1e-10


def test_cyclical_matches_raw():
    # sin/cos UDF muszą być spójne z raw local_hour/dow/month (ten sam wzór co enriched()).
    dt = datetime(2026, 3, 15, 8, 30, 0)
    off = 60
    h = o.local_hour(dt, off)
    assert abs(o.sin_local_hour(dt, off) - math.sin(2 * math.pi * h / 24.0)) < 1e-12
    assert abs(o.cos_local_hour(dt, off) - math.cos(2 * math.pi * h / 24.0)) < 1e-12
    d = o.local_dow(dt, off)
    assert abs(o.sin_local_dow(dt, off) - math.sin(2 * math.pi * d / 7.0)) < 1e-12
    m = o.month_of(dt)
    assert abs(o.sin_month_of(dt) - math.sin(2 * math.pi * m / 12.0)) < 1e-12


def test_geo_functions_still_work():
    # haversine WAW->KRK ~250 km; is_eastbound; duration_ratio (orphan, ale nadal poprawny)
    assert 200 < o.haversine_km(52.16, 20.97, 50.07, 19.79) < 300
    assert o.is_eastbound(19.79, 20.97) == 1 and o.is_eastbound(20.97, 19.79) == 0
    assert o.duration_ratio(3600, 3000) == 1.2 and o.duration_ratio(0, 3000) is None
