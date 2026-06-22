# Databricks notebook source
import sys, glob
_hits = glob.glob("/Workspace/Users/30002818@lot.pl/.bundle/**/src/ml_project/settings.py", recursive=True)
SRC_PATH = [h for h in _hits if "/dev/" in h][0][:-len("/ml_project/settings.py")]
if SRC_PATH not in sys.path: sys.path.insert(0, SRC_PATH)
import ml_project.settings as st
import ml_project.common as cm
SETTINGS = st.load_settings("dev", project_root=SRC_PATH[:-len("/src")],
    source_catalog_override="panda_silver_prod", source_schema_override="occ_ops")
cm.configure_runtime(SETTINGS, spark=spark)
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
print("✅ Bootstrap OK")
print(f"Silver ft_*: {SETTINGS.FT_ROUTE_DAILY_STATS_TABLE}")

# COMMAND ----------

tables_to_register = [
    {"name": SETTINGS.FT_LEG_STATUS_TABLE,            "primary_keys": ["leg_no"],        "timestamp_keys": ["event_ts"],   "description": "Status lotu (streaming ingest z df_labels)"},
    {"name": SETTINGS.FT_LEG_TIMES_TABLE,             "primary_keys": ["leg_no"],        "timestamp_keys": ["event_ts"],   "description": "Pomiary OOOI (streaming ingest z df_leg_times)"},
    {"name": SETTINGS.FT_LEG_MISC_TABLE,              "primary_keys": ["leg_no"],        "timestamp_keys": ["event_ts"],   "description": "Stand assignment (streaming ingest z df_leg_misc)"},
    {"name": SETTINGS.FT_AIRPORT_TIMEZONE_TABLE,      "primary_keys": ["iata_ap_code"],  "timestamp_keys": ["valid_ts"],   "description": "Strefa czasowa + lat/lon (stopnie) per lotnisko"},
    {"name": SETTINGS.FT_ROUTE_DAILY_STATS_TABLE,     "primary_keys": ["route_id"],      "timestamp_keys": ["event_date"], "description": "Statystyki trasy (airborne/arrival_delay/dur_ratio) + days_since"},
    {"name": SETTINGS.FT_AIRPORT_DAILY_TAXI_OUT_TABLE,"primary_keys": ["dep_ap_sched"],  "timestamp_keys": ["event_date"], "description": "Statystyki taxi-out na lotnisko + days_since"},
    {"name": SETTINGS.FT_AIRPORT_DAILY_TAXI_IN_TABLE, "primary_keys": ["arr_ap_sched"],  "timestamp_keys": ["event_date"], "description": "Statystyki taxi-in na lotnisko + days_since"},
    {"name": SETTINGS.FT_STAND_DAILY_OUT_TABLE,       "primary_keys": ["stand_id"],      "timestamp_keys": ["event_date"], "description": "Statystyki stand-out + days_since (stand_id = concat ap_stand)"},
    {"name": SETTINGS.FT_STAND_DAILY_IN_TABLE,        "primary_keys": ["stand_id"],      "timestamp_keys": ["event_date"], "description": "Statystyki stand-in + days_since"},
]


legacy_tables_to_deregister = [
    SETTINGS.FS_TAXI_OUT_TABLE,
    SETTINGS.FS_AIRBORNE_TABLE,
    SETTINGS.FS_TAXI_IN_TABLE,
    SETTINGS.FS_STAND_OUT_TABLE,
    SETTINGS.FS_STAND_IN_TABLE,
]
print(f"Do rejestracji: {len(tables_to_register)} ft_* | legacy fs_* (rollback, NIE deregister): {len(legacy_tables_to_deregister)}")

for t in tables_to_register:
    try:
        info = fe.get_table(name=t["name"])
        print(f"✅ Już zarejestrowana: {t['name']}")
        print(f"   PK: {info.primary_keys}, TS: {info.timestamp_keys}")
    except Exception:
        print(f"⬜ Niezarejestrowana: {t['name']}")

# COMMAND ----------

# Rejestracja
results = {}
for t in tables_to_register:
    name = t["name"]
    try:
        fe.get_table(name=name)
        print(f"⏭️  Skip (już istnieje): {name}")
        results[name] = "already_registered"
    except Exception:
        try:
            fe.create_feature_table(
                name=name,
                primary_keys=t["primary_keys"],
                timestamp_keys=t["timestamp_keys"],
                description=t["description"],
            )
            print(f"✅ Zarejestrowano: {name}")
            results[name] = "registered"
        except Exception as e:
            print(f"❌ FAIL {name}: {e}")
            results[name] = f"error: {e}"

print("\n=== PODSUMOWANIE ===")
for name, status in results.items():
    print(f"  {name.split('.')[-1]}: {status}")

# COMMAND ----------

# Weryfikacja po rejestracji
print("=== Weryfikacja finalna ===")
all_ok = True
for t in tables_to_register:
    try:
        info = fe.get_table(name=t["name"])
        pk_ok = set(t["primary_keys"]) <= set(info.primary_keys)
        ts_ok = set(t["timestamp_keys"]) <= set(info.timestamp_keys or [])
        status = "✅" if pk_ok and ts_ok else "⚠️"
        print(f"{status} {t['name'].split('.')[-1]}: PK={info.primary_keys}, TS={info.timestamp_keys}")
        if not (pk_ok and ts_ok):
            all_ok = False
    except Exception as e:
        print(f"❌ {t['name']}: {e}")
        all_ok = False

print("\n>>> WSZYSTKIE TABELE ZAREJESTROWANE ✅" if all_ok else ">>> SPRAWDŹ BŁĘDY ❌")
