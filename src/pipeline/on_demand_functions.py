"""Iter2 — on-demand feature functions (UC Python UDF).

Te funkcje zostaną zarejestrowane w Unity Catalog jako Python UDF
(notebook notebooks/10_register_on_demand_functions.ipynb). FeatureFunction w
training/scoring odwoła się do nich po nazwie UC — wartości NIE są materializowane,
liczone w `fe.create_training_set` i `fe.score_batch`.

Definicje tutaj są źródłem prawdy ciała UDF (notebook wstrzykuje je do
CREATE FUNCTION ... LANGUAGE PYTHON). Czyste Python (tylko stdlib `math`),
żeby były testowalne lokalnie i 1:1 z tym, co rejestrujemy w UC.
"""
import math
from datetime import datetime, timedelta


def sin_cos_hour(hour: int) -> dict:
    """Zwraca {'sin': sin, 'cos': cos} dla godziny 0-23."""
    rad = 2.0 * math.pi * (hour or 0) / 24.0
    return {"sin": math.sin(rad), "cos": math.cos(rad)}


def sin_cos_dow(dow: int) -> dict:
    """Zwraca {'sin': sin, 'cos': cos} dla dnia tygodnia 0-6."""
    rad = 2.0 * math.pi * (dow or 0) / 7.0
    return {"sin": math.sin(rad), "cos": math.cos(rad)}


def sin_cos_month(month: int) -> dict:
    """Zwraca {'sin': sin, 'cos': cos} dla miesiąca 1-12."""
    rad = 2.0 * math.pi * (month or 1) / 12.0
    return {"sin": math.sin(rad), "cos": math.cos(rad)}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Odległość Haversine w km."""
    if None in (lat1, lon1, lat2, lon2):
        return 0.0
    R = 6371.0
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def is_eastbound(lon1: float, lon2: float) -> int:
    """1 jeśli arr_lon > dep_lon, inaczej 0."""
    if lon1 is None or lon2 is None:
        return 0
    return 1 if (lon2 - lon1) > 0 else 0


def duration_ratio(actual_sec: int, scheduled_sec: int) -> float:
    """actual/scheduled lub None gdy brak danych / scheduled == 0."""
    if not actual_sec or not scheduled_sec or scheduled_sec == 0:
        return None
    return float(actual_sec) / float(scheduled_sec)


# =====================================================================================
# Iter2.5 (Opcja A) — local time features jako on-demand UDF.
# Parytet z legacy enriched(): dep_local_ts = dep_sched_dt + utc_offset_min*60s.
# UWAGA: w rejestracji UC (notebook 10) ciała sin/cos są INLINE (UC UDF nie woła innych
# Python-symboli) — tutaj wołamy helper local_hour/local_dow/month_of dla czytelności/testów,
# ale wynik jest tożsamy.
# =====================================================================================


def local_hour(scheduled_dt: datetime, utc_offset_min: int) -> int:
    """Lokalna godzina (0-23) w strefie czasowej lotniska."""
    if scheduled_dt is None or utc_offset_min is None:
        return 0
    local_dt = scheduled_dt + timedelta(minutes=utc_offset_min)
    return local_dt.hour


def local_dow(scheduled_dt: datetime, utc_offset_min: int) -> int:
    """Lokalny dzień tygodnia (0=poniedziałek..6=niedziela).

    Parytet z enriched(): Spark `(dayofweek + 5) % 7` (Sun=1->Mon=0) == Python `weekday()` (Mon=0).
    """
    if scheduled_dt is None or utc_offset_min is None:
        return 0
    local_dt = scheduled_dt + timedelta(minutes=utc_offset_min)
    return local_dt.weekday()


def month_of(scheduled_dt: datetime) -> int:
    """Miesiąc 1-12 z dep_sched_dt (bez offsetu — month globalnie)."""
    if scheduled_dt is None:
        return 1
    return scheduled_dt.month


def sin_local_hour(scheduled_dt: datetime, utc_offset_min: int) -> float:
    """sin(2π·local_hour/24)."""
    h = local_hour(scheduled_dt, utc_offset_min)
    return math.sin(2.0 * math.pi * h / 24.0)


def cos_local_hour(scheduled_dt: datetime, utc_offset_min: int) -> float:
    """cos(2π·local_hour/24)."""
    h = local_hour(scheduled_dt, utc_offset_min)
    return math.cos(2.0 * math.pi * h / 24.0)


def sin_local_dow(scheduled_dt: datetime, utc_offset_min: int) -> float:
    """sin(2π·local_dow/7)."""
    d = local_dow(scheduled_dt, utc_offset_min)
    return math.sin(2.0 * math.pi * d / 7.0)


def cos_local_dow(scheduled_dt: datetime, utc_offset_min: int) -> float:
    """cos(2π·local_dow/7)."""
    d = local_dow(scheduled_dt, utc_offset_min)
    return math.cos(2.0 * math.pi * d / 7.0)


def sin_month_of(scheduled_dt: datetime) -> float:
    """sin(2π·month/12)."""
    m = month_of(scheduled_dt)
    return math.sin(2.0 * math.pi * m / 12.0)


def cos_month_of(scheduled_dt: datetime) -> float:
    """cos(2π·month/12)."""
    m = month_of(scheduled_dt)
    return math.cos(2.0 * math.pi * m / 12.0)
