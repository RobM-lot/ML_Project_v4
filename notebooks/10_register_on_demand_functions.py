# Databricks notebook source
# Komórka 1 — Bootstrap
import sys, glob
_hits = glob.glob("/Workspace/Users/30002818@lot.pl/.bundle/**/src/ml_project/settings.py", recursive=True)
SRC_PATH = [h for h in _hits if "/dev/" in h][0][:-len("/ml_project/settings.py")]
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
import ml_project.settings as st
SETTINGS = st.load_settings("dev", project_root=SRC_PATH[:-len("/src")],
    source_catalog_override="panda_silver_prod", source_schema_override="occ_ops")
# Nazwy funkcji budujemy z silver catalog/schema (niezależnie od pól UC_FN_* w settings)
SILVER_FULL = f"{SETTINGS.SILVER_CATALOG}.{SETTINGS.SILVER_SCHEMA}"
print(f"✅ Bootstrap OK | target schema: {SILVER_FULL}")

# COMMAND ----------

# Komórka 2 — Definicje funkcji (ciała 1:1 z src/pipeline/on_demand_functions.py;
# UC UDF nie woła innych UDF -> ciała sin/cos są INLINE, ale tożsame z helperami w .py)
FUNCTIONS = [
    {
        "name": "sin_cos_hour",
        "params": "hour INT",
        "returns": "STRUCT<sin: DOUBLE, cos: DOUBLE>",
        "body": "import math\nrad = 2.0 * math.pi * (hour or 0) / 24.0\nreturn {\"sin\": math.sin(rad), \"cos\": math.cos(rad)}",
    },
    {
        "name": "sin_cos_dow",
        "params": "dow INT",
        "returns": "STRUCT<sin: DOUBLE, cos: DOUBLE>",
        "body": "import math\nrad = 2.0 * math.pi * (dow or 0) / 7.0\nreturn {\"sin\": math.sin(rad), \"cos\": math.cos(rad)}",
    },
    {
        "name": "sin_cos_month",
        "params": "month INT",
        "returns": "STRUCT<sin: DOUBLE, cos: DOUBLE>",
        "body": "import math\nrad = 2.0 * math.pi * (month or 1) / 12.0\nreturn {\"sin\": math.sin(rad), \"cos\": math.cos(rad)}",
    },
    {
        "name": "haversine_km",
        "params": "lat1 DOUBLE, lon1 DOUBLE, lat2 DOUBLE, lon2 DOUBLE",
        "returns": "DOUBLE",
        "body": "import math\nif None in (lat1, lon1, lat2, lon2):\n    return 0.0\nR = 6371.0\nlat1_r = math.radians(lat1)\nlat2_r = math.radians(lat2)\ndlat = math.radians(lat2 - lat1)\ndlon = math.radians(lon2 - lon1)\na = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2\nreturn 2 * R * math.asin(math.sqrt(a))",
    },
    {
        "name": "is_eastbound",
        "params": "lon1 DOUBLE, lon2 DOUBLE",
        "returns": "INT",
        "body": "if lon1 is None or lon2 is None:\n    return 0\nreturn 1 if (lon2 - lon1) > 0 else 0",
    },
    {
        "name": "duration_ratio",
        "params": "actual_sec INT, scheduled_sec INT",
        "returns": "DOUBLE",
        "body": "if not actual_sec or not scheduled_sec or scheduled_sec == 0:\n    return None\nreturn float(actual_sec) / float(scheduled_sec)",
    },
    # ===== Iter2.5 Opcja A — local time (bodies self-contained / inline) =====
    {
        "name": "local_hour",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "INT",
        "body": "from datetime import timedelta\nif scheduled_dt is None or utc_offset_min is None:\n    return 0\nreturn (scheduled_dt + timedelta(minutes=utc_offset_min)).hour",
    },
    {
        "name": "local_dow",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "INT",
        "body": "from datetime import timedelta\nif scheduled_dt is None or utc_offset_min is None:\n    return 0\nreturn (scheduled_dt + timedelta(minutes=utc_offset_min)).weekday()",
    },
    {
        "name": "month_of",
        "params": "scheduled_dt TIMESTAMP",
        "returns": "INT",
        "body": "if scheduled_dt is None:\n    return 1\nreturn scheduled_dt.month",
    },
    {
        "name": "sin_local_hour",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "DOUBLE",
        "body": "import math\nfrom datetime import timedelta\nh = 0 if (scheduled_dt is None or utc_offset_min is None) else (scheduled_dt + timedelta(minutes=utc_offset_min)).hour\nreturn math.sin(2.0 * math.pi * h / 24.0)",
    },
    {
        "name": "cos_local_hour",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "DOUBLE",
        "body": "import math\nfrom datetime import timedelta\nh = 0 if (scheduled_dt is None or utc_offset_min is None) else (scheduled_dt + timedelta(minutes=utc_offset_min)).hour\nreturn math.cos(2.0 * math.pi * h / 24.0)",
    },
    {
        "name": "sin_local_dow",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "DOUBLE",
        "body": "import math\nfrom datetime import timedelta\nd = 0 if (scheduled_dt is None or utc_offset_min is None) else (scheduled_dt + timedelta(minutes=utc_offset_min)).weekday()\nreturn math.sin(2.0 * math.pi * d / 7.0)",
    },
    {
        "name": "cos_local_dow",
        "params": "scheduled_dt TIMESTAMP, utc_offset_min INT",
        "returns": "DOUBLE",
        "body": "import math\nfrom datetime import timedelta\nd = 0 if (scheduled_dt is None or utc_offset_min is None) else (scheduled_dt + timedelta(minutes=utc_offset_min)).weekday()\nreturn math.cos(2.0 * math.pi * d / 7.0)",
    },
    {
        "name": "sin_month_of",
        "params": "scheduled_dt TIMESTAMP",
        "returns": "DOUBLE",
        "body": "import math\nm = 1 if scheduled_dt is None else scheduled_dt.month\nreturn math.sin(2.0 * math.pi * m / 12.0)",
    },
    {
        "name": "cos_month_of",
        "params": "scheduled_dt TIMESTAMP",
        "returns": "DOUBLE",
        "body": "import math\nm = 1 if scheduled_dt is None else scheduled_dt.month\nreturn math.cos(2.0 * math.pi * m / 12.0)",
    },
]
print(f"{len(FUNCTIONS)} funkcji do rejestracji")

# COMMAND ----------

# Komórka 3 — Rejestracja (CREATE OR REPLACE FUNCTION ... LANGUAGE PYTHON)
results = {}
for fn in FUNCTIONS:
    full_name = f"{SILVER_FULL}.{fn['name']}"
    ddl = (
        f"CREATE OR REPLACE FUNCTION {full_name}({fn['params']})\n"
        f"RETURNS {fn['returns']}\n"
        f"LANGUAGE PYTHON\n"
        f"AS $$\n{fn['body']}\n$$"
    )
    try:
        spark.sql(ddl)
        print(f"✅ zarejestrowano: {full_name}")
        results[full_name] = "ok"
    except Exception as e:
        print(f"❌ FAIL {full_name}: {str(e)[:200]}")
        results[full_name] = f"error: {e}"

print("\n=== PODSUMOWANIE ===")
for k, v in results.items():
    print(f"  {k.split('.')[-1]}: {v}")

# COMMAND ----------

# Komórka 4 — Weryfikacja: każda funkcja istnieje i liczy (test SELECT)
smoke = {
    "sin_cos_hour": "SELECT {fn}(6).sin AS s, {fn}(6).cos AS c",
    "sin_cos_dow": "SELECT {fn}(3).sin AS s, {fn}(3).cos AS c",
    "sin_cos_month": "SELECT {fn}(7).sin AS s, {fn}(7).cos AS c",
    "haversine_km": "SELECT {fn}(52.16, 20.97, 50.07, 19.79) AS km",
    "is_eastbound": "SELECT {fn}(20.97, 19.79) AS eb",
    "duration_ratio": "SELECT {fn}(3600, 3000) AS r",
    # Iter2.5 local time: 2026-06-09 14:00 UTC + 120 min (UTC+2) -> 16:00, wtorek(1), czerwiec(6)
    "local_hour": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS h",
    "local_dow": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS d",
    "month_of": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00') AS m",
    "sin_local_hour": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS v",
    "cos_local_hour": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS v",
    "sin_local_dow": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS v",
    "cos_local_dow": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00', 120) AS v",
    "sin_month_of": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00') AS v",
    "cos_month_of": "SELECT {fn}(TIMESTAMP'2026-06-09 14:00:00') AS v",
}
all_ok = True
for name, q in smoke.items():
    full_name = f"{SILVER_FULL}.{name}"
    try:
        row = spark.sql(q.format(fn=full_name)).first()
        print(f"✅ {name}: {row}")
    except Exception as e:
        print(f"❌ {name}: {str(e)[:200]}")
        all_ok = False

# Oczekiwane sanity: local_hour=16, local_dow=1 (wtorek), month_of=6
print("\n>>> WSZYSTKIE UDF DZIAŁAJĄ ✅" if all_ok else ">>> SPRAWDŹ BŁĘDY ❌")