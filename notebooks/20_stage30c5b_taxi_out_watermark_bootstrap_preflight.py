# Databricks notebook source
# ruff: noqa: F821
# Stage 30C-5b read-only taxi-out watermark bootstrap preflight.

import json
from pathlib import Path
import sys

from pyspark.sql import functions as F

print("=" * 100)
print(
    "Stage 30C-5b taxi-out watermark bootstrap preflight. "
    "Candidate-only output, read-only, no watermark changes."
)
print("=" * 100)

# COMMAND ----------

RUN_BOOTSTRAP_PREFLIGHT = False
ALLOW_WATERMARK_BOOTSTRAP = False
DRY_RUN_ONLY = True

SOURCE_LEG = "panda_silver_prod.occ_ops.netline___schedops__leg"
SOURCE_LEG_TIMES = "panda_silver_prod.occ_ops.netline___schedops__leg_times"
SHADOW_TAXI_OUT_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"


def _display_value(value):
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _display_metric_rows(mapping):
    return [(str(key), _display_value(value)) for key, value in mapping.items()]


CONFIG_ROWS = [
    ("RUN_BOOTSTRAP_PREFLIGHT", str(RUN_BOOTSTRAP_PREFLIGHT)),
    ("ALLOW_WATERMARK_BOOTSTRAP", str(ALLOW_WATERMARK_BOOTSTRAP)),
    ("DRY_RUN_ONLY", str(DRY_RUN_ONLY)),
    ("SOURCE_LEG", SOURCE_LEG),
    ("SOURCE_LEG_TIMES", SOURCE_LEG_TIMES),
    ("SHADOW_TAXI_OUT_TABLE", SHADOW_TAXI_OUT_TABLE),
    ("WATERMARK_TABLE", WATERMARK_TABLE),
]

display(spark.createDataFrame(CONFIG_ROWS, ["parameter", "value"]))

if not RUN_BOOTSTRAP_PREFLIGHT:
    print("RUN_BOOTSTRAP_PREFLIGHT is False. Configuration displayed only; exiting before reads.")
    dbutils.notebook.exit("RUN_BOOTSTRAP_PREFLIGHT_FALSE")
if not DRY_RUN_ONLY:
    raise ValueError("Bootstrap preflight must run with DRY_RUN_ONLY=True.")
if ALLOW_WATERMARK_BOOTSTRAP:
    raise ValueError("Bootstrap preflight is read-only; keep ALLOW_WATERMARK_BOOTSTRAP=False.")

# COMMAND ----------


def _get_notebook_path() -> str:
    try:
        path = (
            dbutils.notebook.entry_point.getDbutils()
            .notebook()
            .getContext()
            .notebookPath()
            .get()
        )
    except Exception:
        return ""
    return f"/Workspace{path}" if path and not path.startswith("/Workspace") else path


def _resolve_project_root() -> Path:
    candidates = []
    notebook_path = _get_notebook_path()
    if notebook_path:
        notebook_file = Path(notebook_path)
        candidates.extend([notebook_file.parent, *notebook_file.parent.parents])
    cwd = Path.cwd()
    candidates.extend([cwd, *cwd.parents])

    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if (candidate / "src" / "ml_project" / "stage30c_taxi_out_watermark.py").exists():
            return candidate
    raise FileNotFoundError("Cannot locate repository root containing src/ml_project/stage30c_taxi_out_watermark.py")


PROJECT_ROOT = _resolve_project_root()
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from ml_project.stage30c_taxi_out_watermark import (  # noqa: E402
    SOURCE_ALIASES,
    build_bootstrap_version_candidate,
    detect_missing_watermark_columns,
    earliest_delta_history_entry,
    latest_delta_history_entry,
    quote_table_name,
    summarize_delta_history_entry,
)

print(f"Project root: {PROJECT_ROOT}")
print(f"Loaded local helper modules from: {SRC_PATH}")

# COMMAND ----------


def _delta_history_rows(table_name: str):
    history_df = (
        spark.sql(f"DESCRIBE HISTORY {quote_table_name(table_name)}")
        .select("version", "timestamp", "operation", "operationParameters")
        .orderBy(F.col("version").asc())
    )
    return [row.asDict(recursive=True) for row in history_df.collect()]


def _watermark_rows_and_schema():
    if not spark.catalog.tableExists(WATERMARK_TABLE):
        return [], [], ["watermark table does not exist"]

    watermark_df = spark.table(WATERMARK_TABLE)
    columns = watermark_df.columns
    missing_columns = list(detect_missing_watermark_columns(columns))
    messages = [f"missing required columns: {missing_columns}"] if missing_columns else []
    rows = []
    if "source_alias" in columns:
        rows = [
            row.asDict(recursive=True)
            for row in watermark_df.where(F.col("source_alias").isin(*SOURCE_ALIASES)).collect()
        ]
    return rows, columns, messages


shadow_history = _delta_history_rows(SHADOW_TAXI_OUT_TABLE)
shadow_earliest = earliest_delta_history_entry(shadow_history)
shadow_latest = latest_delta_history_entry(shadow_history)
shadow_earliest_summary = summarize_delta_history_entry(shadow_earliest)
shadow_latest_summary = summarize_delta_history_entry(shadow_latest)

source_histories = {
    "leg": _delta_history_rows(SOURCE_LEG),
    "leg_times": _delta_history_rows(SOURCE_LEG_TIMES),
}
candidate_rows = []
for source_alias, history_rows in source_histories.items():
    candidate = build_bootstrap_version_candidate(
        source_alias=source_alias,
        source_history_rows=history_rows,
        shadow_baseline_timestamp=shadow_earliest["timestamp"],
    )
    candidate_rows.append(
        {
            "source_alias": candidate.source_alias,
            "candidate_version": candidate.candidate_version,
            "candidate_timestamp": candidate.candidate_timestamp,
            "candidate_operation": candidate.candidate_operation,
            "shadow_baseline_timestamp": candidate.shadow_baseline_timestamp,
            "status": candidate.status,
        }
    )

watermark_rows, watermark_schema, watermark_schema_messages = _watermark_rows_and_schema()

summary = {
    "status": "candidate_only_requires_human_confirmation",
    "shadow_table": SHADOW_TAXI_OUT_TABLE,
    "shadow_earliest_history_operation": shadow_earliest_summary,
    "shadow_latest_history_operation": shadow_latest_summary,
    "candidate_bootstrap_versions": candidate_rows,
    "current_watermark_schema": watermark_schema,
    "current_watermark_schema_messages": watermark_schema_messages,
    "current_watermark_rows": watermark_rows,
    "next_step": "Only after confirming these baseline versions, run a separate gated bootstrap insert.",
}

print("CANDIDATE ONLY - requires human confirmation.")
print("These versions are selected at or immediately before the shadow baseline timestamp.")
print("Do not infer baseline from validation windows or latest source versions.")
print(summary["next_step"])

display(spark.createDataFrame(_display_metric_rows(summary), ["metric", "value"]))
display(
    spark.createDataFrame(
        [
            (row["source_alias"], _display_value(key), _display_value(value))
            for row in candidate_rows
            for key, value in row.items()
        ],
        ["source_alias", "metric", "value"],
    )
)

dbutils.notebook.exit("bootstrap_preflight_candidate_only")
