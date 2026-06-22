# Databricks notebook source
import sys
import platform
import importlib

print("Python:", sys.version)
print("Platform:", platform.platform())
print("Spark version:", spark.version)
print("DBR:", spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown"))

modules = ["pyspark", "mlflow", "delta", "pandas", "numpy", "sklearn"]
for module_name in modules:
    try:
        module = importlib.import_module(module_name)
        print(f"[OK] {module_name}: {getattr(module, '__version__', 'no __version__')}")
    except Exception as exc:
        print(f"[FAIL] {module_name}: {exc}")
        raise

from delta.tables import DeltaTable
import mlflow

print("DeltaTable import OK")
print("MLflow import OK")

# COMMAND ----------

from pathlib import Path

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

globals().update(mp.settings_to_globals(SETTINGS))
mp.configure_runtime(SETTINGS, spark=spark)

print(f"ENV={ENV}")
print(f"SOURCE={SOURCE_CATALOG}.{SOURCE_SCHEMA}")
print(f"SILVER={SILVER_CATALOG}.{SILVER_SCHEMA}")
print(f"GOLD={GOLD_CATALOG}.{GOLD_SCHEMA}")
print(f"PROJECT_ROOT={PROJECT_ROOT}")

# COMMAND ----------

from mlflow import MlflowClient
import mlflow

hard_failures = []
warnings = []

print("=== ENVIRONMENT CONTRACT CHECKS ===")

expected_source_prefix = f"{SOURCE_CATALOG}.{SOURCE_SCHEMA}."
expected_silver_prefix = f"{SILVER_CATALOG}.{SILVER_SCHEMA}."
expected_gold_prefix = f"{GOLD_CATALOG}.{GOLD_SCHEMA}."

source_objects = {
    "LABELS_TABLE": LABELS_TABLE,
    "LEG_TIMES_TABLE": LEG_TIMES_TABLE,
    "AP_BASICS_TABLE": AP_BASICS_TABLE,
    "TIME_ZONE_TABLE": TIME_ZONE_TABLE,
    "LEG_MISC_TABLE": LEG_MISC_TABLE,
    "LEG_REMARK_TABLE": LEG_REMARK_TABLE,
}

silver_objects = {
    "FS_TAXI_OUT_TABLE": FS_TAXI_OUT_TABLE,
    "FS_AIRBORNE_TABLE": FS_AIRBORNE_TABLE,
    "FS_TAXI_IN_TABLE": FS_TAXI_IN_TABLE,
    "FS_STAND_OUT_TABLE": FS_STAND_OUT_TABLE,
    "FS_STAND_IN_TABLE": FS_STAND_IN_TABLE,
    "SHADOW_TABLE": SHADOW_TABLE,
    "TRAINING_DATASET_TABLE": TRAINING_DATASET_TABLE,
    "EVAL_CLEAN_DATASET_TABLE": EVAL_CLEAN_DATASET_TABLE,
    "EVAL_ALL_DATASET_TABLE": EVAL_ALL_DATASET_TABLE,
}

gold_objects = {
    "SINK_TABLE": SINK_TABLE,
    "EVENTS_SINK_TABLE": EVENTS_SINK_TABLE,
}

def _check_prefix(obj_name: str, obj_value: str, expected_prefix: str):
    if not str(obj_value).startswith(expected_prefix):
        hard_failures.append(
            f"{obj_name} ma zły namespace. expected_prefix={expected_prefix}, actual={obj_value}"
        )

for obj_name, obj_value in source_objects.items():
    _check_prefix(obj_name, obj_value, expected_source_prefix)

for obj_name, obj_value in silver_objects.items():
    _check_prefix(obj_name, obj_value, expected_silver_prefix)

for obj_name, obj_value in gold_objects.items():
    _check_prefix(obj_name, obj_value, expected_gold_prefix)

if not str(UC_MODEL_NAME).startswith(expected_gold_prefix):
    hard_failures.append(
        f"UC_MODEL_NAME ma zły namespace. expected_prefix={expected_gold_prefix}, actual={UC_MODEL_NAME}"
    )

expected_model_uri_prefix = f"models:/{UC_MODEL_NAME}@"
if not str(MODEL_URI).startswith(expected_model_uri_prefix):
    hard_failures.append(
        f"MODEL_URI nie zgadza się z UC_MODEL_NAME. expected_prefix={expected_model_uri_prefix}, actual={MODEL_URI}"
    )

def assert_table_readable(table_name: str, required_cols=None, min_rows: int = 1):
    if not spark.catalog.tableExists(table_name):
        hard_failures.append(f"Brak tabeli: {table_name}")
        return

    try:
        df = spark.table(table_name)
        row_cnt = df.limit(min_rows).count()
    except Exception as e:
        hard_failures.append(f"Nie można odczytać tabeli {table_name}: {e}")
        return

    if row_cnt < min_rows:
        warnings.append(f"Tabela {table_name} istnieje, ale wygląda na pustą lub prawie pustą.")

    if required_cols:
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            hard_failures.append(f"Tabela {table_name} nie zawiera wymaganych kolumn: {missing}")

assert_table_readable(
    LABELS_TABLE,
    required_cols=["leg_no", "dep_sched_dt", "arr_sched_dt", "dep_ap_sched", "arr_ap_sched", "leg_state"],
)
assert_table_readable(LEG_TIMES_TABLE)
assert_table_readable(AP_BASICS_TABLE)
assert_table_readable(TIME_ZONE_TABLE)
assert_table_readable(LEG_MISC_TABLE)
assert_table_readable(LEG_REMARK_TABLE)

def write_probe(table_name: str):
    try:
        spark.sql(f"DROP TABLE IF EXISTS {table_name}")

        probe_df = spark.createDataFrame(
            [(1, "ok", ENV)],
            "probe_id INT, status STRING, env STRING"
        )

        (
            probe_df.write
            .format("delta")
            .mode("overwrite")
            .saveAsTable(table_name)
        )

        cnt = spark.table(table_name).count()
        if cnt != 1:
            hard_failures.append(f"Write probe do {table_name} zakończył się niepoprawnie. count={cnt}")

    except Exception as e:
        hard_failures.append(f"Nie udał się write probe dla {table_name}: {e}")
    finally:
        try:
            spark.sql(f"DROP TABLE IF EXISTS {table_name}")
        except Exception as cleanup_e:
            warnings.append(f"Nie udało się posprzątać probe table {table_name}: {cleanup_e}")

silver_probe = f"{SILVER_CATALOG}.{SILVER_SCHEMA}.__env_contract_probe_{ENV}"
gold_probe = f"{GOLD_CATALOG}.{GOLD_SCHEMA}.__env_contract_probe_{ENV}"

write_probe(silver_probe)
write_probe(gold_probe)

try:
    client = MlflowClient()
    exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)

    if exp is None:
        warnings.append(
            f"Eksperyment MLflow jeszcze nie istnieje: {EXPERIMENT_PATH}. To warning, nie hard fail."
        )
    else:
        print(f"MLflow experiment exists: {exp.experiment_id} | {exp.name}")

    try:
        model_meta = client.get_registered_model(UC_MODEL_NAME)
        print(f"Registered model exists: {model_meta.name}")
    except Exception as e:
        warnings.append(f"Registered model jeszcze nie istnieje albo brak dostępu: {UC_MODEL_NAME} | {e}")

except Exception as e:
    hard_failures.append(f"Problem z MLflow / registry client: {e}")

print("\n=== SUMMARY ===")
print(f"hard_failures={len(hard_failures)}")
print(f"warnings={len(warnings)}")

if warnings:
    print("\n--- WARNINGS ---")
    for w in warnings:
        print(f"- {w}")

if hard_failures:
    print("\n--- HARD FAILURES ---")
    for f in hard_failures:
        print(f"- {f}")
    raise RuntimeError("Environment contract checks failed. Zobacz hard_failures powyżej.")

print("\n[OK] Environment contract checks passed.")