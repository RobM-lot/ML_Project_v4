# Databricks notebook source
from pathlib import Path
import importlib
import sys

from pyspark.sql import functions as F


def _get_notebook_path(dbutils_obj) -> str:
    try:
        notebook_path = (
            dbutils_obj.notebook.entry_point.getDbutils() 
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
    except Exception:
        return ""

    if notebook_path and not notebook_path.startswith("/Workspace"):
        notebook_path = f"/Workspace{notebook_path}"
    return notebook_path


def _resolve_project_root(dbutils_obj) -> Path:
    notebook_path = _get_notebook_path(dbutils_obj)

    candidates = []
    if notebook_path:
        notebook_file = Path(notebook_path)
        candidates.extend([notebook_file.parent, *notebook_file.parent.parents])

    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])

    checked = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        checked.append(candidate_str)

        if (candidate / "src" / "ml_project" / "settings.py").exists():
            return candidate

    raise FileNotFoundError(
        "Nie udało się odnaleźć root projektu zawierającego src/ml_project/settings.py. "
        f"Sprawdzone lokalizacje: {checked}"
    )


PROJECT_ROOT = _resolve_project_root(dbutils)
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

import ml_project as mp
mp = importlib.reload(mp)

ENV = mp.ensure_env_widget(dbutils, default="dev")

mp.ensure_text_widget(dbutils, "SOURCE_CATALOG", "panda_silver_prod", "Source catalog")
mp.ensure_text_widget(dbutils, "SOURCE_SCHEMA", "occ_ops", "Source schema")
mp.ensure_text_widget(dbutils, "SILVER_CATALOG", "", "Silver catalog")
mp.ensure_text_widget(dbutils, "SILVER_SCHEMA", "", "Silver schema")
mp.ensure_text_widget(dbutils, "GOLD_CATALOG", "", "Gold catalog")
mp.ensure_text_widget(dbutils, "GOLD_SCHEMA", "", "Gold schema")

SOURCE_CATALOG_WIDGET = mp.get_widget_value(dbutils, "SOURCE_CATALOG", "panda_silver_prod").strip()
SOURCE_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "SOURCE_SCHEMA", "occ_ops").strip()
SILVER_CATALOG_WIDGET = mp.get_widget_value(dbutils, "SILVER_CATALOG", "").strip()
SILVER_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "SILVER_SCHEMA", "").strip()
GOLD_CATALOG_WIDGET = mp.get_widget_value(dbutils, "GOLD_CATALOG", "").strip()
GOLD_SCHEMA_WIDGET = mp.get_widget_value(dbutils, "GOLD_SCHEMA", "").strip()

SETTINGS = mp.load_settings(
    ENV,
    project_root=str(PROJECT_ROOT),
    source_catalog_override=SOURCE_CATALOG_WIDGET or "panda_silver_prod",
    source_schema_override=SOURCE_SCHEMA_WIDGET or "occ_ops",
    silver_catalog_override=SILVER_CATALOG_WIDGET or None,
    silver_schema_override=SILVER_SCHEMA_WIDGET or None,
    gold_catalog_override=GOLD_CATALOG_WIDGET or None,
    gold_schema_override=GOLD_SCHEMA_WIDGET or None,
)

mp.configure_runtime(SETTINGS, spark=spark)

print("ENV:", SETTINGS.ENV)
print("SOURCE:", f"{SETTINGS.SOURCE_CATALOG}.{SETTINGS.SOURCE_SCHEMA}")
print("SILVER:", f"{SETTINGS.SILVER_CATALOG}.{SETTINGS.SILVER_SCHEMA}")
print("GOLD:", f"{SETTINGS.GOLD_CATALOG}.{SETTINGS.GOLD_SCHEMA}")

# COMMAND ----------

hard_failures = []
warnings = []


def table_exists(table_name: str) -> bool:
    try:
        return spark.catalog.tableExists(table_name)
    except Exception:
        return False


def require_table(table_name: str):
    if not table_exists(table_name):
        hard_failures.append(f"Brak tabeli: {table_name}")


SOURCE_TABLES = {
    "LABELS_TABLE": SETTINGS.LABELS_TABLE,
    "LEG_MISC_TABLE": SETTINGS.LEG_MISC_TABLE,
    "LEG_REMARK_TABLE": SETTINGS.LEG_REMARK_TABLE,
    "LEG_TIMES_TABLE": SETTINGS.LEG_TIMES_TABLE,
    "AP_BASICS_TABLE": SETTINGS.AP_BASICS_TABLE,
    "TIME_ZONE_TABLE": SETTINGS.TIME_ZONE_TABLE,
}

SILVER_TABLES = {
    "FS_TAXI_OUT_TABLE": SETTINGS.FS_TAXI_OUT_TABLE,
    "FS_AIRBORNE_TABLE": SETTINGS.FS_AIRBORNE_TABLE,
    "FS_TAXI_IN_TABLE": SETTINGS.FS_TAXI_IN_TABLE,
    "FS_STAND_OUT_TABLE": SETTINGS.FS_STAND_OUT_TABLE,
    "FS_STAND_IN_TABLE": SETTINGS.FS_STAND_IN_TABLE,
    "SHADOW_TABLE": SETTINGS.SHADOW_TABLE,
}

GOLD_TABLES = {
    "SINK_TABLE": SETTINGS.SINK_TABLE,
    "EVENTS_SINK_TABLE": SETTINGS.EVENTS_SINK_TABLE,
}

for _, table_name in SOURCE_TABLES.items():
    require_table(table_name)

for _, table_name in SILVER_TABLES.items():
    require_table(table_name)

for _, table_name in GOLD_TABLES.items():
    require_table(table_name)

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

print("[OK] Wszystkie wymagane tabele istnieją")

# COMMAND ----------

def assert_required_columns(table_name: str, required_columns: list[str]):
    df_cols = {c.lower() for c in spark.table(table_name).columns}
    missing = [c for c in required_columns if c.lower() not in df_cols]
    if missing:
        hard_failures.append(f"{table_name}: brak kolumn {missing}")


assert_required_columns(
    SETTINGS.LABELS_TABLE,
    [
        "leg_no",
        "dep_sched_dt",
        "arr_sched_dt",
        "dep_ap_sched",
        "arr_ap_sched",
        "leg_state",
        "leg_type",
        "counter",
        "__END_AT",
    ],
)

assert_required_columns(
    SETTINGS.LEG_MISC_TABLE,
    [
        "leg_no",
        "dep_stand",
        "arr_stand",
    ],
)

assert_required_columns(
    SETTINGS.LEG_TIMES_TABLE,
    [
        "leg_no",
    ],
)

assert_required_columns(
    SETTINGS.FS_TAXI_OUT_TABLE,
    list(SETTINGS.PK_TAXI_OUT) + ["event_date"],
)

assert_required_columns(
    SETTINGS.FS_AIRBORNE_TABLE,
    list(SETTINGS.PK_AIRBORNE) + ["event_date"],
)

assert_required_columns(
    SETTINGS.FS_TAXI_IN_TABLE,
    list(SETTINGS.PK_TAXI_IN) + ["event_date"],
)

assert_required_columns(
    SETTINGS.FS_STAND_OUT_TABLE,
    list(SETTINGS.PK_STAND_OUT) + ["event_date"],
)

assert_required_columns(
    SETTINGS.FS_STAND_IN_TABLE,
    list(SETTINGS.PK_STAND_IN) + ["event_date"],
)

assert_required_columns(
    SETTINGS.SHADOW_TABLE,
    [
        SETTINGS.SINK_PRIMARY_KEY,
        "dep_sched_dt",
        "arr_sched_dt",
        "leg_state",
        "leg_type",
        "dep_ap_sched",
        "arr_ap_sched",
    ],
)

assert_required_columns(
    SETTINGS.SINK_TABLE,
    [
        SETTINGS.SINK_PRIMARY_KEY,
        "prediction_status",
        "is_operationally_active",
        "effective_actual_block_time_sec",
        "effective_block_delay_sec",
        "model_uri",
        "source_commit_version",
        "source_commit_timestamp",
        "missing_feature_count",
    ],
)

assert_required_columns(
    SETTINGS.EVENTS_SINK_TABLE,
    [
        SETTINGS.SINK_PRIMARY_KEY,
        "logged_at",
        "prediction_status",
        "model_uri",
        "source_commit_version",
        "last_change_type",
    ],
)

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

print("[OK] Wymagane kolumny są obecne")

# COMMAND ----------

def check_duplicate_keys(table_name: str, key_cols: list[str], label: str, max_allowed: int = 0):
    df = spark.table(table_name)
    dup_cnt = (
        df.groupBy(*key_cols)
        .count()
        .filter(F.col("count") > 1)
        .count()
    )
    print(f"{label} duplicate key groups =", dup_cnt)
    if dup_cnt > max_allowed:
        hard_failures.append(f"{label}: znaleziono duplikaty po kluczu {key_cols}")


check_duplicate_keys(
    SETTINGS.FS_TAXI_OUT_TABLE,
    list(SETTINGS.PK_TAXI_OUT) + ["event_date"],
    "FS_TAXI_OUT_TABLE",
)

check_duplicate_keys(
    SETTINGS.FS_AIRBORNE_TABLE,
    list(SETTINGS.PK_AIRBORNE) + ["event_date"],
    "FS_AIRBORNE_TABLE",
)

check_duplicate_keys(
    SETTINGS.FS_TAXI_IN_TABLE,
    list(SETTINGS.PK_TAXI_IN) + ["event_date"],
    "FS_TAXI_IN_TABLE",
)

check_duplicate_keys(
    SETTINGS.SHADOW_TABLE,
    [SETTINGS.SINK_PRIMARY_KEY],
    "SHADOW_TABLE",
)

check_duplicate_keys(
    SETTINGS.SINK_TABLE,
    [SETTINGS.SINK_PRIMARY_KEY],
    "SINK_TABLE",
)

check_duplicate_keys(
    SETTINGS.EVENTS_SINK_TABLE,
    [SETTINGS.SINK_PRIMARY_KEY, "source_commit_version", "last_change_type"],
    "EVENTS_SINK_TABLE",
)

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

print("[OK] Brak krytycznych duplikatów po kluczach")

# COMMAND ----------

labels_current = spark.table(SETTINGS.LABELS_TABLE).filter(F.col("__END_AT").isNull())
cutoff_ts = F.to_timestamp(F.lit(f"{SETTINGS.DATA_CUTOFF_DATE} 00:00:00"))

labels_stats = labels_current.agg(
    F.count("*").alias("rows_cnt"),
    F.max("dep_sched_dt").alias("max_dep_sched_dt"),
    F.min("dep_sched_dt").alias("min_dep_sched_dt"),
    F.avg(F.when(F.col("leg_no").isNull(), 1.0).otherwise(0.0)).alias("leg_no_null_share"),
    F.avg(F.when(F.col("dep_sched_dt").isNull(), 1.0).otherwise(0.0)).alias("dep_sched_dt_null_share"),
    F.avg(F.when(F.col("arr_sched_dt").isNull(), 1.0).otherwise(0.0)).alias("arr_sched_dt_null_share"),
    F.avg(F.when(F.col("dep_ap_sched").isNull(), 1.0).otherwise(0.0)).alias("dep_ap_sched_null_share"),
    F.avg(F.when(F.col("arr_ap_sched").isNull(), 1.0).otherwise(0.0)).alias("arr_ap_sched_null_share"),
    F.sum(F.when(F.col("dep_sched_dt") >= cutoff_ts, 1).otherwise(0)).alias("rows_ge_cutoff"),
).first()

print("LABELS current rows =", labels_stats["rows_cnt"])
print("LABELS min dep_sched_dt =", labels_stats["min_dep_sched_dt"])
print("LABELS max dep_sched_dt =", labels_stats["max_dep_sched_dt"])
print("LABELS leg_no_null_share =", labels_stats["leg_no_null_share"])
print("LABELS dep_sched_dt_null_share =", labels_stats["dep_sched_dt_null_share"])
print("LABELS arr_sched_dt_null_share =", labels_stats["arr_sched_dt_null_share"])
print("LABELS dep_ap_sched_null_share =", labels_stats["dep_ap_sched_null_share"])
print("LABELS arr_ap_sched_null_share =", labels_stats["arr_ap_sched_null_share"])
print("LABELS rows_ge_cutoff =", labels_stats["rows_ge_cutoff"])
print("DATA_CUTOFF_DATE =", SETTINGS.DATA_CUTOFF_DATE)

critical_null_checks = {
    "leg_no_null_share": labels_stats["leg_no_null_share"],
    "dep_sched_dt_null_share": labels_stats["dep_sched_dt_null_share"],
    "arr_sched_dt_null_share": labels_stats["arr_sched_dt_null_share"],
    "dep_ap_sched_null_share": labels_stats["dep_ap_sched_null_share"],
    "arr_ap_sched_null_share": labels_stats["arr_ap_sched_null_share"],
}

for metric_name, metric_value in critical_null_checks.items():
    if float(metric_value or 0.0) > 0.001:
        hard_failures.append(f"LABELS_TABLE: zbyt wysoki {metric_name} = {metric_value:.4f}")

if labels_stats["rows_cnt"] == 0:
    hard_failures.append("LABELS_TABLE: brak bieżących rekordów (__END_AT IS NULL)")

if int(labels_stats["rows_ge_cutoff"] or 0) > 0:
    warnings.append(
        f"LABELS_TABLE zawiera rekordy z dep_sched_dt >= {SETTINGS.DATA_CUTOFF_DATE}. "
        "To nie blokuje notebooka, ale surowe source wykracza poza projektowy cutoff."
    )

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

print("[OK] LABELS_TABLE przechodzi podstawowy contract")

# COMMAND ----------

def fs_freshness_check(table_name: str, label: str):
    row = spark.table(table_name).agg(F.max("event_date").alias("max_event_date")).first()
    max_event_date = row["max_event_date"]
    print(f"{label} max_event_date =", max_event_date)

    if max_event_date is None:
        hard_failures.append(f"{label}: brak event_date")
        return

    lag_days = spark.sql(
        f"SELECT datediff(current_date(), DATE('{max_event_date}')) AS lag_days"
    ).first()["lag_days"]

    print(f"{label} lag_days =", lag_days)

    if lag_days is not None and lag_days > 14:
        warnings.append(f"{label}: event_date opóźnione o {lag_days} dni")


def dep_sched_cutoff_check(table_name: str, label: str, fail_on_cutoff_breach: bool):
    row = spark.table(table_name).agg(
        F.count("*").alias("rows_cnt"),
        F.min("dep_sched_dt").alias("min_dep_sched_dt"),
        F.max("dep_sched_dt").alias("max_dep_sched_dt"),
        F.sum(
            F.when(
                F.col("dep_sched_dt") >= F.to_timestamp(F.lit(f"{SETTINGS.DATA_CUTOFF_DATE} 00:00:00")),
                1
            ).otherwise(0)
        ).alias("rows_ge_cutoff"),
    ).first()

    print(f"{label} rows =", row["rows_cnt"])
    print(f"{label} min dep_sched_dt =", row["min_dep_sched_dt"])
    print(f"{label} max dep_sched_dt =", row["max_dep_sched_dt"])
    print(f"{label} rows_ge_cutoff =", row["rows_ge_cutoff"])

    if int(row["rows_cnt"] or 0) == 0:
        warnings.append(f"{label}: brak rekordów")
        return

    if int(row["rows_ge_cutoff"] or 0) > 0:
        msg = (
            f"{label}: znaleziono {int(row['rows_ge_cutoff'])} rekordów z dep_sched_dt >= "
            f"{SETTINGS.DATA_CUTOFF_DATE}"
        )
        if fail_on_cutoff_breach:
            hard_failures.append(msg)
        else:
            warnings.append(msg)


fs_freshness_check(SETTINGS.FS_TAXI_OUT_TABLE, "FS_TAXI_OUT_TABLE")
fs_freshness_check(SETTINGS.FS_AIRBORNE_TABLE, "FS_AIRBORNE_TABLE")
fs_freshness_check(SETTINGS.FS_TAXI_IN_TABLE, "FS_TAXI_IN_TABLE")

dep_sched_cutoff_check(SETTINGS.SHADOW_TABLE, "SHADOW_TABLE", fail_on_cutoff_breach=True)

dep_sched_cutoff_check(SETTINGS.LABELS_TABLE, "LABELS_TABLE", fail_on_cutoff_breach=False)

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

# COMMAND ----------

sink_stats = spark.table(SETTINGS.SINK_TABLE).agg(
    F.count("*").alias("rows_cnt"),
    F.max("scored_at").alias("max_scored_at"),
    F.max("source_commit_version").alias("max_source_commit_version"),
    F.avg(F.when(F.col("prediction_status").isNull(), 1.0).otherwise(0.0)).alias("prediction_status_null_share"),
    F.avg(F.when(F.col("model_uri").isNull(), 1.0).otherwise(0.0)).alias("model_uri_null_share"),
).first()

events_stats = spark.table(SETTINGS.EVENTS_SINK_TABLE).agg(
    F.count("*").alias("rows_cnt"),
    F.max("logged_at").alias("max_logged_at"),
    F.max("source_commit_version").alias("max_source_commit_version"),
).first()

print("SINK rows =", sink_stats["rows_cnt"])
print("SINK max_scored_at =", sink_stats["max_scored_at"])
print("SINK max_source_commit_version =", sink_stats["max_source_commit_version"])
print("SINK prediction_status_null_share =", sink_stats["prediction_status_null_share"])
print("SINK model_uri_null_share =", sink_stats["model_uri_null_share"])

print("EVENTS rows =", events_stats["rows_cnt"])
print("EVENTS max_logged_at =", events_stats["max_logged_at"])
print("EVENTS max_source_commit_version =", events_stats["max_source_commit_version"])

if float(sink_stats["prediction_status_null_share"] or 0.0) > 0.001:
    hard_failures.append("SINK_TABLE: prediction_status ma zbyt wysoki null share")

if float(sink_stats["model_uri_null_share"] or 0.0) > 0.001:
    warnings.append("SINK_TABLE: model_uri ma niezerowy null share")

if events_stats["rows_cnt"] < sink_stats["rows_cnt"]:
    warnings.append("EVENTS_SINK_TABLE ma mniej rekordów niż SINK_TABLE")

if hard_failures:
    raise RuntimeError(" | ".join(hard_failures))

# COMMAND ----------

print("=== WARNINGS ===")
if warnings:
    for w in warnings:
        print("-", w)
else:
    print("Brak ostrzeżeń")

print("\n[OK] Data/schema contract passed.")