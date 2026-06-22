# Databricks notebook source
import sys
import math
import importlib
from pathlib import Path

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, StructType, StructField, DateType, StringType


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

import ml_project.settings as st
import ml_project.widgets as wd
import ml_project.common as cm

st = importlib.reload(st)
wd = importlib.reload(wd)
cm = importlib.reload(cm)

load_settings = st.load_settings
settings_to_globals = st.settings_to_globals

ensure_env_widget = wd.ensure_env_widget
ensure_text_widget = wd.ensure_text_widget
get_widget_value = wd.get_widget_value

configure_runtime = cm.configure_runtime
apply_data_quality_rules = cm.apply_data_quality_rules
get_airport_features = cm.get_airport_features
enrich_with_local_context = cm.enrich_with_local_context
get_cleaned_flight_data = cm.get_cleaned_flight_data

try:
    ENV = ensure_env_widget(dbutils, default="dev")
except Exception as exc:
    raise ValueError(
        "[ERROR] BŁĄD KRYTYCZNY: Nie podano lub podano błędny parametr 'ENV'. "
        "Skonfiguruj widget lub parametry Joba."
    ) from exc

ensure_text_widget(dbutils, "SOURCE_CATALOG", "panda_silver_prod", "Source catalog")
ensure_text_widget(dbutils, "SOURCE_SCHEMA", "occ_ops", "Source schema")

ensure_text_widget(dbutils, "SILVER_CATALOG", "", "Silver catalog override")
ensure_text_widget(dbutils, "SILVER_SCHEMA", "", "Silver schema override")

ensure_text_widget(dbutils, "GOLD_CATALOG", "", "Gold catalog override")
ensure_text_widget(dbutils, "GOLD_SCHEMA", "", "Gold schema override")

SOURCE_CATALOG_WIDGET = get_widget_value(dbutils, "SOURCE_CATALOG", "panda_silver_prod").strip()
SOURCE_SCHEMA_WIDGET = get_widget_value(dbutils, "SOURCE_SCHEMA", "occ_ops").strip()

SILVER_CATALOG_WIDGET = get_widget_value(dbutils, "SILVER_CATALOG", "").strip()
SILVER_SCHEMA_WIDGET = get_widget_value(dbutils, "SILVER_SCHEMA", "").strip()

GOLD_CATALOG_WIDGET = get_widget_value(dbutils, "GOLD_CATALOG", "").strip()
GOLD_SCHEMA_WIDGET = get_widget_value(dbutils, "GOLD_SCHEMA", "").strip()

SETTINGS = load_settings(
    ENV,
    project_root=str(PROJECT_ROOT),
    source_catalog_override=SOURCE_CATALOG_WIDGET or "panda_silver_prod",
    source_schema_override=SOURCE_SCHEMA_WIDGET or "occ_ops",
    silver_catalog_override=SILVER_CATALOG_WIDGET or None,
    silver_schema_override=SILVER_SCHEMA_WIDGET or None,
    gold_catalog_override=GOLD_CATALOG_WIDGET or None,
    gold_schema_override=GOLD_SCHEMA_WIDGET or None,
)

globals().update(settings_to_globals(SETTINGS))
configure_runtime(SETTINGS, spark=spark)

print(f" Konfiguracja załadowana dla środowiska: {ENV.upper()}")
print(f"📂 Source: {SOURCE_CATALOG}.{SOURCE_SCHEMA}")
print(f"🥈 Silver: {SILVER_CATALOG}.{SILVER_SCHEMA}")
print(f"🥇 Gold: {GOLD_CATALOG}.{GOLD_SCHEMA}")
print(f" Checkpoint: {CHECKPOINT_PATH}")
print(f"🧩 Project root: {PROJECT_ROOT}")
print(f"🧩 src path: {SRC_PATH}")
if not ALLOW_CHECKPOINT_RESET:
    print(f"🔒 Środowisko {ENV.upper()}: BLOKADA resetowania checkpointów")

# COMMAND ----------

_ = (
    SETTINGS,
    apply_data_quality_rules,
    get_airport_features,
    enrich_with_local_context,
    get_cleaned_flight_data,
    F,
    DoubleType,
    StructType,
    StructField,
    DateType,
    StringType,
    math,
)