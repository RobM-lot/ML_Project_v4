# Databricks notebook source
# DBTITLE 1,Cell 1
from pathlib import Path
import importlib
import sys


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
import ml_project.registry as rg

mp = importlib.reload(mp)
rg = importlib.reload(rg)

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
print(f"EXPERIMENT_PATH={EXPERIMENT_PATH}")
print(f"UC_MODEL_NAME={UC_MODEL_NAME}")
print(f"MODEL_URI={MODEL_URI}")

# COMMAND ----------

rg.ensure_register_widgets(dbutils, default_promote=False)

try:
    dbutils.widgets.remove("PROMOTE_IF_PASS")
except Exception:
    pass

dbutils.widgets.dropdown(
    "PROMOTE_IF_PASS",
    "False",
    ["False"],
    "1. Promote to champion if gates pass (locked to False in quality gate)"
)

run_id_from_widget = dbutils.widgets.get("RUN_ID").strip()

if not run_id_from_widget:
    try:
        upstream_run_id = dbutils.jobs.taskValues.get(
            taskKey="train_compare_models",
            key="run_id",
            debugValue=""
        )
        upstream_run_id = str(upstream_run_id).strip()
    except Exception:
        upstream_run_id = ""

    if upstream_run_id:
        try:
            dbutils.widgets.remove("RUN_ID")
        except Exception:
            pass
        dbutils.widgets.text(
            "RUN_ID",
            upstream_run_id,
            "0. Candidate run_id (automatycznie pobrany z taska train_compare_models)"
        )
        print(f"[OK] RUN_ID automatycznie pobrany z taska train_compare_models: {upstream_run_id}")
    else:
        print("[INFO] RUN_ID pozostaje pusty — quality gate wybierze najlepszy zgodny run z eksperymentu.")
else:
    print(f"[INFO] RUN_ID podany jawnie w widgetcie: {run_id_from_widget}")

# COMMAND ----------

result = rg.run_register_best(spark, dbutils, SETTINGS)

print("=== QUALITY GATE RESULT ===")
for k, v in result.items():
    if k not in {"checks", "candidate_metrics", "champion_metrics", "promotion"}:
        print(f"{k}: {v}")

print("\n=== CHECKS ===")
for k, v in result["checks"].items():
    print(f"{k}: {v}")

print("\n=== CANDIDATE METRICS ===")
for k, v in result["candidate_metrics"].items():
    print(f"{k}: {v}")

print("\n=== CHAMPION METRICS ===")
if result["champion_metrics"]:
    for k, v in result["champion_metrics"].items():
        print(f"{k}: {v}")
else:
    print("None")

# COMMAND ----------

gates_passed = bool(result["gates_passed"])
decision = str(result["decision"])

if not gates_passed:
    raise RuntimeError(
        f"Quality gate failed. decision={decision}, run_id={result['candidate_run_id']}"
    )

print(f"[OK] Quality gate passed. decision={decision}, run_id={result['candidate_run_id']}")