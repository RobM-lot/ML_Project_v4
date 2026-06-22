# Databricks notebook source
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
mlflow.set_tracking_uri("databricks")
from pyspark.sql import functions as F
from sklearn.metrics import mean_absolute_error
from IPython.display import display

sns.set_theme(style="whitegrid")

try:
    dbutils.widgets.dropdown(
        "EVAL_MODEL_SOURCE",
        "champion",
        ["champion", "run_id"],
        "Źródło modelu do ewaluacji"
    )
    dbutils.widgets.text(
        "EVAL_RUN_ID",
        "",
        "Run ID (jeśli EVAL_MODEL_SOURCE=run_id)"
    )
except Exception:
    pass

# COMMAND ----------

# MAGIC %run ./config

# COMMAND ----------

print("📦 Źródło danych do ewaluacji: materializowane tabele pośrednie z builda")

# COMMAND ----------

from mlflow.tracking import MlflowClient

mlflow.set_experiment(EXPERIMENT_PATH)

try:
    EVAL_MODEL_SOURCE = dbutils.widgets.get("EVAL_MODEL_SOURCE")
except Exception:
    EVAL_MODEL_SOURCE = "champion"

try:
    EVAL_RUN_ID = dbutils.widgets.get("EVAL_RUN_ID").strip()
except Exception:
    EVAL_RUN_ID = ""

print(f" Pobieranie danych do ewaluacji (ostatnie {EVAL_DAYS} dni)...")
print(f"📦 Eval clean table: {EVAL_CLEAN_DATASET_TABLE}")
print(f"📦 Eval all table:   {EVAL_ALL_DATASET_TABLE}")

training_df_eval_clean = spark.table(EVAL_CLEAN_DATASET_TABLE)
training_df_eval_all = spark.table(EVAL_ALL_DATASET_TABLE)

training_sdf = training_df_eval_clean.orderBy("event_ts")
max_event_ts = training_sdf.agg(F.max("event_ts")).collect()[0][0]
if max_event_ts is None:
    raise RuntimeError(f"Tabela ewaluacyjna jest pusta: {EVAL_CLEAN_DATASET_TABLE}")

split_date = pd.to_datetime(max_event_ts) - pd.Timedelta(days=EVAL_DAYS)

eval_clean_sdf = training_df_eval_clean.filter(F.col("event_ts") >= F.lit(split_date))
eval_all_sdf = training_df_eval_all.filter(F.col("event_ts") >= F.lit(split_date))

eval_clean_pdf = eval_clean_sdf.toPandas()
eval_all_pdf = eval_all_sdf.toPandas()

for frame in [eval_clean_pdf, eval_all_pdf]:
    for col in CATEGORICAL_FEATURES:
        if col in frame.columns:
            frame[col] = frame[col].fillna("UNKNOWN").astype(str)

print(f"[OK] Zbiory testowe gotowe. Clean={len(eval_clean_pdf)} | All={len(eval_all_pdf)} | split={split_date}")

experiment = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
if experiment is None:
    raise RuntimeError(f"Nie znaleziono eksperymentu MLflow: {EXPERIMENT_PATH}")
client = MlflowClient()

def resolve_model_reference():
    if EVAL_MODEL_SOURCE == "run_id":
        if not EVAL_RUN_ID:
            raise ValueError("Wybrano EVAL_MODEL_SOURCE=run_id, ale nie podano EVAL_RUN_ID.")
        return EVAL_RUN_ID, f"runs:/{EVAL_RUN_ID}/model", "explicit_run_id"

    if EVAL_MODEL_SOURCE == "champion":
        try:
            champion_version = client.get_model_version_by_alias(UC_MODEL_NAME, "champion")
            return champion_version.run_id, f"models:/{UC_MODEL_NAME}@champion", "uc_alias_champion"
        except Exception as exc:
            raise RuntimeError(
                f"Nie udało się pobrać aliasu @champion dla modelu {UC_MODEL_NAME}: {exc}"
            ) from exc

    raise ValueError(f"Nieobsługiwane EVAL_MODEL_SOURCE: {EVAL_MODEL_SOURCE}")

run_id, model_uri, model_source_resolved = resolve_model_reference()
print(f"Pobieranie modelu do ewaluacji | source={model_source_resolved} | run_id={run_id}")
model = mlflow.pyfunc.load_model(model_uri)

model_info = mlflow.models.get_model_info(model_uri)
MODEL_INPUT_COLS = [spec.name for spec in model_info.signature.inputs.inputs]

def prepare_model_input(pdf: pd.DataFrame) -> pd.DataFrame:
    prepared = pdf.copy()

    for spec in model_info.signature.inputs.inputs:
        col = spec.name
        expected_type = str(spec.type).lower()

        if col not in prepared.columns:
            prepared[col] = np.nan

        if "long" in expected_type:
            prepared[col] = pd.to_numeric(prepared[col], errors='coerce').fillna(0).astype("int64")
        elif "integer" in expected_type or "int" in expected_type:
            prepared[col] = pd.to_numeric(prepared[col], errors='coerce').fillna(0).astype("int32")
        elif "double" in expected_type or "float" in expected_type:
            prepared[col] = pd.to_numeric(prepared[col], errors='coerce').astype("float64")
        elif "string" in expected_type:
            prepared[col] = prepared[col].fillna("UNKNOWN").astype(str)

    return prepared[MODEL_INPUT_COLS].copy()

def score_frame(pdf, scope_name):
    if pdf.empty:
        return pdf

    model_input = prepare_model_input(pdf)
    preds = model.predict(model_input)

    eval_df = pdf.copy()

    for col in preds.columns:
        eval_df[col] = preds[col]

    if "pred_reconciled_segment_sum_sec" not in eval_df.columns:
        required_seg_cols = ["pred_taxi_out_sec", "pred_airborne_sec", "pred_taxi_in_sec"]
        if all(c in eval_df.columns for c in required_seg_cols):
            eval_df["pred_reconciled_segment_sum_sec"] = eval_df[required_seg_cols].sum(axis=1)

    if "pred_block_model_sec" not in eval_df.columns and "pred_actual_block_time_sec" in eval_df.columns:
        eval_df["pred_block_model_sec"] = eval_df["pred_actual_block_time_sec"]

    eval_df["eval_scope"] = scope_name
    eval_df["evaluated_model_uri"] = model_uri
    eval_df["evaluated_run_id"] = run_id

    return eval_df

eval_clean_df = score_frame(eval_clean_pdf, "clean_ops")
eval_all_df = score_frame(eval_all_pdf, "all_ops")

print("[OK] Predykcje zakończone sukcesem")

# COMMAND ----------

ERROR_SIGN_CONVENTION = "pred_minus_actual"

def safe_pct(numerator, denominator):
    if denominator in [0, None] or pd.isna(denominator):
        return np.nan
    return (numerator / denominator) * 100.0

def prepare_eval_frame(df):
    if df.empty:
        return df.copy()

    eval_df = df.dropna(subset=["actual_block_time_sec", "scheduled_block_time_sec", "pred_actual_block_time_sec"]).copy()
    
    eval_df["model_error_vs_actual_min"] = (eval_df["pred_actual_block_time_sec"] - eval_df["actual_block_time_sec"]) / 60.0
    eval_df["model_abs_error_vs_actual_min"] = eval_df["model_error_vs_actual_min"].abs()

    eval_df["schedule_error_vs_actual_min"] = (eval_df["scheduled_block_time_sec"] - eval_df["actual_block_time_sec"]) / 60.0
    eval_df["schedule_abs_error_vs_actual_min"] = eval_df["schedule_error_vs_actual_min"].abs()

    eval_df["pred_delay_vs_schedule_min"] = (eval_df["pred_actual_block_time_sec"] - eval_df["scheduled_block_time_sec"]) / 60.0
    eval_df["actual_delay_vs_schedule_min"] = (eval_df["actual_block_time_sec"] - eval_df["scheduled_block_time_sec"]) / 60.0
    eval_df["delay_prediction_error_min"] = eval_df["pred_delay_vs_schedule_min"] - eval_df["actual_delay_vs_schedule_min"]

    eval_df["actual_block_time_min"] = eval_df["actual_block_time_sec"] / 60.0
    eval_df["pred_actual_block_time_min"] = eval_df["pred_actual_block_time_sec"] / 60.0
    eval_df["scheduled_block_time_min"] = eval_df["scheduled_block_time_sec"] / 60.0

    eval_df["relative_abs_error_pct"] = np.where(
        eval_df["scheduled_block_time_sec"] > 0,
        np.abs(eval_df["pred_actual_block_time_sec"] - eval_df["actual_block_time_sec"]) / eval_df["scheduled_block_time_sec"] * 100.0,
        np.nan
    )
    if "pred_actual_block_time_p90_sec" in eval_df.columns:
        eval_df["p90_covered"] = eval_df["pred_actual_block_time_p90_sec"] >= eval_df["actual_block_time_sec"]
    else:
        eval_df["p90_covered"] = np.nan

    bins = [0, 90, 180, 360, np.inf]
    labels = ["Krótkie (<1.5h)", "Średnie (1.5h - 3h)", "Długie (3h - 6h)", "Ultra-Długie (>6h)"]
    eval_df["flight_length_category"] = pd.cut(eval_df["scheduled_block_time_min"], bins=bins, labels=labels)

    if "local_hour_dep" in eval_df.columns:
        eval_df["hour_of_day"] = eval_df["local_hour_dep"]
    else:
        eval_df["hour_of_day"] = pd.to_datetime(eval_df["event_ts"]).dt.hour

    return eval_df

def compute_actual_block_scorecard(df, pred_col="pred_actual_block_time_sec", actual_col="actual_block_time_sec",
                                   baseline_col="scheduled_block_time_sec", p90_col="pred_actual_block_time_p90_sec",
                                   label="MODEL", calc_ci=False):
    work = df.dropna(subset=[pred_col, actual_col]).copy()
    if work.empty:
        return pd.Series(dtype="object")

    signed_error_min = (work[pred_col] - work[actual_col]) / 60.0
    abs_error_min = signed_error_min.abs()
    
    mae_ci_low = mae_ci_up = p90_ci_low = p90_ci_up = np.nan
    
    if calc_ci and len(abs_error_min) >= 30:
        n_bootstraps = 1000
        values = abs_error_min.values
        np.random.seed(42)
        boot_idx = np.random.randint(0, len(values), size=(n_bootstraps, len(values)))
        boot_samples = values[boot_idx]
        mae_boot = np.mean(boot_samples, axis=1)
        p90_boot = np.percentile(boot_samples, 90, axis=1)
        mae_ci_low, mae_ci_up = np.percentile(mae_boot, 2.5), np.percentile(mae_boot, 97.5)
        p90_ci_low, p90_ci_up = np.percentile(p90_boot, 2.5), np.percentile(p90_boot, 97.5)
        
    metrics = {
        "Wariant": label,
        "Liczba_lotów": int(len(work)),
        "MAE_min": abs_error_min.mean(),
        "MAE_95CI_Low": mae_ci_low,
        "MAE_95CI_Up": mae_ci_up,
        "MedianAE_min": abs_error_min.median(),
        "P90_AE_min": abs_error_min.quantile(0.90),
        "P90_95CI_Low": p90_ci_low,
        "P90_95CI_Up": p90_ci_up,
        "Bias_min": signed_error_min.mean(),
    }
    
    if baseline_col and baseline_col in work.columns:
        baseline_abs_error_min = ((work[baseline_col] - work[actual_col]) / 60.0).abs()
        valid_sched = work[baseline_col].gt(0) & work[baseline_col].notna() 
        
        if valid_sched.any():
            metrics["WinRate_vs_schedule_pct"] = (abs_error_min[valid_sched] < baseline_abs_error_min[valid_sched]).mean() * 100.0
            metrics["Relative_AE_vs_sched_time_pct"] = (
                ((work.loc[valid_sched, pred_col] - work.loc[valid_sched, actual_col]).abs() / work.loc[valid_sched, baseline_col]).mean() * 100.0
            )
        else:
            metrics["WinRate_vs_schedule_pct"] = np.nan
            metrics["Relative_AE_vs_sched_time_pct"] = np.nan
            
    if p90_col and p90_col in work.columns:
        metrics["P90_Coverage_pct"] = (work[p90_col] >= work[actual_col]).mean() * 100.0
    return pd.Series(metrics)

def compute_delay_scorecard(df, label="MODEL"):
    work = df.dropna(subset=["pred_delay_vs_schedule_min", "actual_delay_vs_schedule_min"]).copy()
    if work.empty:
        return pd.Series(dtype="object")
    err = work["pred_delay_vs_schedule_min"] - work["actual_delay_vs_schedule_min"]
    abs_err = err.abs()
    return pd.Series({
        "Wariant": label,
        "Liczba_lotów": int(len(work)),
        "Delay_MAE_min": abs_err.mean(),
        "Delay_MedianAE_min": abs_err.median(),
        "Delay_P90_AE_min": abs_err.quantile(0.90),
        "Delay_Bias_min": err.mean(),
        "Precision_delay_gt15_pct": safe_pct(
            ((work["pred_delay_vs_schedule_min"] > 15) & (work["actual_delay_vs_schedule_min"] > 15)).sum(),
            (work["pred_delay_vs_schedule_min"] > 15).sum()
        ),
        "Recall_delay_gt15_pct": safe_pct(
            ((work["pred_delay_vs_schedule_min"] > 15) & (work["actual_delay_vs_schedule_min"] > 15)).sum(),
            (work["actual_delay_vs_schedule_min"] > 15).sum()
        ),
    })

def build_slice_report(df, slice_col, scorecard_fn, min_count=MIN_SLICE_COUNT, top_n=None, sort_by="Liczba_lotów"):
    if slice_col not in df.columns:
        print(f"Brak kolumny {slice_col} w zbiorze.")
        return None
    work = df.dropna(subset=[slice_col]).copy()
    if work.empty:
        return None

    counts = work[slice_col].value_counts(dropna=False)
    if top_n is not None:
        selected = counts.nlargest(top_n).index
        work = work[work[slice_col].isin(selected)]

    grouped = []
    for key, grp in work.groupby(slice_col, dropna=False):
        score = scorecard_fn(grp, label=str(key))
        if score.empty:
            continue
        grouped.append(score)

    if not grouped:
        return None

    report = pd.DataFrame(grouped)
    report["slice_value"] = report["Wariant"]
    report["is_low_sample"] = report["Liczba_lotów"] < min_count

    ascending = False if sort_by == "Liczba_lotów" else True
    secondary = "MAE_min" if "MAE_min" in report.columns else report.columns[1]
    report = report.sort_values([sort_by, secondary], ascending=[ascending, True])
    return report.reset_index(drop=True)

def format_and_display_scorecard(title, scorecard_df):
    print(f"\n{'=' * 90}\n{title}\n{'=' * 90}")
    display(scorecard_df.round(2))

eval_clean_df = prepare_eval_frame(eval_clean_df)
eval_all_df = prepare_eval_frame(eval_all_df)

benchmark_snapshot = pd.DataFrame([{
    "run_id": run_id,
    "model_uri": model_uri,
    "eval_days": EVAL_DAYS,
    "split_date": split_date,
    "rows_clean_eval": len(eval_clean_df),
    "rows_all_eval": len(eval_all_df),
    "error_sign_convention": ERROR_SIGN_CONVENTION,
    "created_at_utc": pd.Timestamp.utcnow()
}])

format_and_display_scorecard("📌 ZAMROŻONY SNAPSHOT EWALUACJI", benchmark_snapshot)
print("Interpretacja znaku błędu: pred_minus_actual -> wartości ujemne oznaczają niedoszacowanie czasu przez model, dodatnie oznaczają przeszacowanie.")

# COMMAND ----------

global_actual_scorecard = pd.DataFrame([
    compute_actual_block_scorecard(eval_clean_df, label="ML_FINAL__clean_ops", calc_ci=True),
    compute_actual_block_scorecard(eval_all_df, label="ML_FINAL__all_ops", calc_ci=True),
    
    compute_actual_block_scorecard(eval_clean_df, pred_col="scheduled_block_time_sec", baseline_col=None, p90_col=None, label="SCHEDULE_BASELINE__clean_ops"),
    compute_actual_block_scorecard(eval_all_df, pred_col="scheduled_block_time_sec", baseline_col=None, p90_col=None, label="SCHEDULE_BASELINE__all_ops"),
])

global_delay_scorecard = pd.DataFrame([
    compute_delay_scorecard(eval_clean_df, label="Delay__clean_ops"),
    compute_delay_scorecard(eval_all_df, label="Delay__all_ops"),
])

format_and_display_scorecard("📉 GLOBALNY SCORECARD — ACTUAL BLOCK", global_actual_scorecard)
format_and_display_scorecard("🧭 GLOBALNY SCORECARD — DELAY VS SCHEDULE", global_delay_scorecard)

# COMMAND ----------

slice_airport = build_slice_report(eval_clean_df, "dep_ap_sched", compute_actual_block_scorecard, top_n=TOP_AIRPORTS_N, min_count=MIN_SLICE_COUNT)
slice_hour = build_slice_report(eval_clean_df, "hour_of_day", compute_actual_block_scorecard, top_n=24, min_count=MIN_SLICE_COUNT, sort_by="slice_value")
slice_type = build_slice_report(eval_clean_df, "leg_type", compute_actual_block_scorecard, top_n=10, min_count=MIN_SLICE_COUNT)
slice_length = build_slice_report(eval_clean_df, "flight_length_category", compute_actual_block_scorecard, min_count=MIN_SLICE_COUNT, sort_by="slice_value")

for title, report in [
    (" SLICE REPORT — LOTNISKO WYLOTU", slice_airport),
    ("🕒 SLICE REPORT — LOKALNA GODZINA WYLOTU", slice_hour),
    ("🧾 SLICE REPORT — LEG_TYPE", slice_type),
    ("📏 SLICE REPORT — DŁUGOŚĆ LOTU", slice_length),
]:
    if report is not None:
        format_and_display_scorecard(title, report)

print("[WARN] Kolumna is_low_sample=True oznacza grupy, których nie należy interpretować jako twardy insight produkcyjny.")

# COMMAND ----------

variant_rows = [
    compute_actual_block_scorecard(eval_clean_df, pred_col="scheduled_block_time_sec", baseline_col=None, p90_col=None, label="SCHEDULE_BASELINE"),
    compute_actual_block_scorecard(eval_clean_df, pred_col="pred_block_model_sec", p90_col="pred_actual_block_time_p90_sec", label="BLOCK_MODEL")
]

if "pred_raw_segment_sum_sec" in eval_clean_df.columns:
    variant_rows.append(compute_actual_block_scorecard(eval_clean_df, pred_col="pred_raw_segment_sum_sec", p90_col=None, label="RAW_SEGMENT_SUM"))

variant_rows.append(compute_actual_block_scorecard(eval_clean_df, pred_col="pred_reconciled_segment_sum_sec", p90_col=None, label="RECONCILED_SEGMENT_SUM"))
variant_rows.append(compute_actual_block_scorecard(eval_clean_df, pred_col="pred_actual_block_time_sec", p90_col="pred_actual_block_time_p90_sec", label="FINAL_OUTPUT"))

variant_scorecard = pd.DataFrame(variant_rows).drop_duplicates(subset=["Wariant"]).reset_index(drop=True)
format_and_display_scorecard("🧪 PORÓWNANIE WARIANTÓW PREDYKCJI — CLEAN OPS", variant_scorecard)

# COMMAND ----------

fig, axes = plt.subplots(1, 2, figsize=(18, 6))

sns.histplot(eval_clean_df["model_error_vs_actual_min"], bins=80, kde=True, ax=axes[0], label="Model ML", alpha=0.70)
sns.histplot(eval_clean_df["schedule_error_vs_actual_min"], bins=80, kde=True, ax=axes[0], label="Schedule", alpha=0.45)
axes[0].set_xlim(-60, 60)
axes[0].axvline(0, color="red", linestyle="--")
axes[0].set_title("Rozkład signed error (pred - actual)")
axes[0].set_xlabel("Błąd [min] | <0 niedoszacowanie, >0 przeszacowanie")
axes[0].legend()

hour_bias = (
    eval_clean_df.groupby("hour_of_day")
    .agg(
        Bias_min=("model_error_vs_actual_min", "mean"),
        MAE_min=("model_abs_error_vs_actual_min", "mean"),
        flight_count=("hour_of_day", "count")
    )
    .reset_index()
)
hour_bias_filtered = hour_bias[hour_bias["flight_count"] >= MIN_SLICE_COUNT]

sns.lineplot(data=hour_bias_filtered, x="hour_of_day", y="Bias_min", marker="o", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title(f"Bias modelu po lokalnej godzinie odlotu (n ≥ {MIN_SLICE_COUNT})")
axes[1].set_xlabel("Lokalna godzina odlotu")
axes[1].set_ylabel("Bias [min]")
axes[1].set_xticks(range(0, 24))

plt.tight_layout()
plt.show()

coverage_pct = eval_clean_df["p90_covered"].mean() * 100.0
print(f" Global coverage P90 (clean ops): {coverage_pct:.1f}% (cel około 90%).")

# COMMAND ----------

length_stats = (
    eval_clean_df.groupby("flight_length_category", observed=False)
    .agg(
        MAE_min=("model_abs_error_vs_actual_min", "mean"),
        Rel_MAE_pct=("relative_abs_error_pct", "mean"),
        Bias_min=("model_error_vs_actual_min", "mean"),
        P90_AE_min=("model_abs_error_vs_actual_min", lambda x: x.quantile(0.90)),
        P90_Coverage_pct=("p90_covered", "mean"),
        flight_count=("flight_length_category", "count")
    )
    .reset_index()
)
length_stats["P90_Coverage_pct"] = length_stats["P90_Coverage_pct"] * 100.0

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.barplot(data=length_stats, x="flight_length_category", y="MAE_min", ax=axes[0])
axes[0].set_title("MAE vs długość lotu")
axes[0].set_xlabel("Kategoria długości lotu")
axes[0].set_ylabel("MAE [min]")

sns.barplot(data=length_stats, x="flight_length_category", y="Rel_MAE_pct", ax=axes[1])
axes[1].set_title("Relative MAE vs schedule")
axes[1].set_xlabel("Kategoria długości lotu")
axes[1].set_ylabel("Relative MAE [% schedule]")
plt.tight_layout()
plt.show()

format_and_display_scorecard("📏 SCORECARD — DŁUGOŚĆ LOTU", length_stats)

segment_scorecard = pd.DataFrame([
    compute_actual_block_scorecard(eval_clean_df.dropna(subset=["taxi_out_sec", "pred_taxi_out_sec"]), pred_col="pred_taxi_out_sec", actual_col="taxi_out_sec", baseline_col=None, p90_col="pred_taxi_out_p90_sec", label="TAXI_OUT"),
    compute_actual_block_scorecard(eval_clean_df.dropna(subset=["airborne_sec", "pred_airborne_sec"]), pred_col="pred_airborne_sec", actual_col="airborne_sec", baseline_col=None, p90_col="pred_airborne_p90_sec", label="AIRBORNE"),
    compute_actual_block_scorecard(eval_clean_df.dropna(subset=["taxi_in_sec", "pred_taxi_in_sec"]), pred_col="pred_taxi_in_sec", actual_col="taxi_in_sec", baseline_col=None, p90_col="pred_taxi_in_p90_sec", label="TAXI_IN")
])
format_and_display_scorecard("🧩 SCORECARD — SEGMENTY LOTU (bez schedule baseline)", segment_scorecard)

# COMMAND ----------

def build_monthly_scorecard(df, scope_name):
    if df.empty:
        return pd.DataFrame()
    work = df.dropna(subset=["event_ts"]).copy()
    work["year_month"] = pd.to_datetime(work["event_ts"]).dt.to_period("M")
    rows = []
    for ym, grp in work.groupby("year_month"):
        score = compute_actual_block_scorecard(grp, label=str(ym))
        if score.empty:
            continue
        score["year_month"] = str(ym)
        score["eval_scope"] = scope_name
        rows.append(score)
    return pd.DataFrame(rows)

monthly_clean = build_monthly_scorecard(eval_clean_df, "clean_ops")
monthly_all = build_monthly_scorecard(eval_all_df, "all_ops")
monthly_metrics = pd.concat([monthly_clean, monthly_all], ignore_index=True)

format_and_display_scorecard(
    " MIESIĘCZNY BACKTEST",
    monthly_metrics[["year_month", "eval_scope", "Liczba_lotów", "MAE_min", "P90_AE_min", "Bias_min", "WinRate_vs_schedule_pct", "P90_Coverage_pct"]]
)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.lineplot(data=monthly_metrics, x="year_month", y="MAE_min", hue="eval_scope", marker="o", ax=axes[0])
axes[0].set_title("MAE miesiąc po miesiącu")
axes[0].tick_params(axis="x", rotation=45)

sns.lineplot(data=monthly_metrics, x="year_month", y="Bias_min", hue="eval_scope", marker="o", ax=axes[1])
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title("Bias miesiąc po miesiącu")
axes[1].tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.show()

coverage_hour = (
    eval_clean_df.groupby("hour_of_day")
    .agg(P90_Coverage_pct=("p90_covered", "mean"), flight_count=("hour_of_day", "count"))
    .reset_index()
)
coverage_hour = coverage_hour[coverage_hour["flight_count"] >= MIN_SLICE_COUNT]
coverage_hour["P90_Coverage_pct"] = coverage_hour["P90_Coverage_pct"] * 100.0

plt.figure(figsize=(12, 6))
sns.lineplot(data=coverage_hour, x="hour_of_day", y="P90_Coverage_pct", marker="o")
plt.axhline(90.0, color="red", linestyle="--", label="Cel 90%")
plt.title(f"Coverage P90 po lokalnej godzinie odlotu (clean ops, n ≥ {MIN_SLICE_COUNT})")
plt.xlabel("Lokalna godzina odlotu")
plt.ylabel("Coverage P90 [%]")
plt.xticks(range(0, 24))
plt.legend()
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error

plot_df = eval_clean_df.dropna(subset=["actual_block_time_sec", "scheduled_block_time_sec"]).copy()

plot_df["actual_block_time_min"] = plot_df["actual_block_time_sec"] / 60.0
plot_df["pred_actual_block_time_min"] = plot_df["pred_actual_block_time_sec"] / 60.0

plot_df["model_error_min"] = (plot_df["pred_actual_block_time_sec"] - plot_df["actual_block_time_sec"]) / 60.0

plot_df["model_abs_error_min"] = plot_df["model_error_min"].abs()

plot_df["baseline_error_min"] = (plot_df["scheduled_block_time_sec"] - plot_df["actual_block_time_sec"]) / 60.0
plot_df["baseline_abs_error_min"] = plot_df["baseline_error_min"].abs()

mae = plot_df["model_abs_error_min"].mean()
median_err = plot_df["model_abs_error_min"].median()
p90_err = plot_df["model_abs_error_min"].quantile(0.90)
bias = plot_df["model_error_min"].mean()
p90_coverage = (plot_df["pred_actual_block_time_p90_sec"] >= plot_df["actual_block_time_sec"]).mean() * 100
win_rate = (plot_df["model_abs_error_min"] < plot_df["baseline_abs_error_min"]).mean() * 100

print(f"{'='*50}")
print(" GLOBALNE METRYKI BŁĘDU (TAIL KPI & BIAS)")
print(f"{'='*50}")
print(f"Średni Błąd (MAE):           {mae:.2f} min")
print(f"Mediana Błędu (P50):         {median_err:.2f} min")
print("-" * 50)
print(f"Błąd 90. percentyla (P90):   {p90_err:.2f} min  <-- Biznesowy ogon opóźnień")
print(f"Bias (Kierunek Błędu):       {bias:.2f} min  (> 0 to bezpieczne przeszacowanie)")
print("-" * 50)
print(f"Pokrycie bufora Ryzyka P90:  {p90_coverage:.1f} %   (Cel: ~90.0%)")
print(f"Wskaźnik Wygranej ML:        {win_rate:.1f} %   (Tyle razy ML był bliżej prawdy niż rozkład)")
print(f"{'='*50}")

# COMMAND ----------

sns.set_theme(style="whitegrid")

plt.figure(figsize=(10, 6))
plt.scatter(plot_df["actual_block_time_min"], plot_df["pred_actual_block_time_min"], alpha=0.3, color='#1f77b4', s=20)
max_val = min(plot_df["actual_block_time_min"].max(), 800)
plt.plot([0, max_val], [0, max_val], color='red', linestyle='--', linewidth=2, label='Idealna Predykcja')
plt.xlim(0, max_val)
plt.ylim(0, max_val)
plt.title("Całkowity Czas Lotu: Rzeczywistość vs Predykcja", fontsize=16, fontweight='bold')
plt.xlabel("Rzeczywisty czas lotu (minuty)", fontsize=14)
plt.ylabel("Przewidywany czas lotu (minuty)", fontsize=14)
plt.legend()
plt.tight_layout()
plt.show()

if "local_hour_dep" in plot_df.columns:
    plt.figure(figsize=(12, 6))
    sns.boxplot(x="local_hour_dep", y="model_error_min", data=plot_df, showfliers=False, color="lightblue")
    plt.axhline(0, color="red", linestyle="--")
    plt.title("Bias Modelu ML w przekroju Pory Dnia (Local Hour)", fontsize=16, fontweight='bold')
    plt.xlabel("Lokalna godzina odlotu", fontsize=14)
    plt.ylabel("Błąd (min)", fontsize=14)
    plt.tight_layout()
    plt.show()

# COMMAND ----------

plt.figure(figsize=(14, 7))

ax = sns.violinplot(
    data=plot_df, 
    y="flight_length_category", 
    x="model_error_min", 
    palette="coolwarm", 
    inner="quartile",
    linewidth=1.5
)

means = plot_df.groupby("flight_length_category", observed=False)["model_error_min"].mean().reset_index()

sns.scatterplot(
    data=means, 
    y="flight_length_category", 
    x="model_error_min", 
    ax=ax,
    color="white", 
    edgecolor="black",
    s=100,
    marker="D",
    zorder=3,
    label="Średnia (Mean)",
)

plt.axvline(x=0, color='black', linestyle='-', linewidth=2, label='Idealna predykcja (0 min)')
plt.axvline(x=15, color='gray', linestyle='--', alpha=0.5, label='Tolerancja +/- 15 min')
plt.axvline(x=-15, color='gray', linestyle='--', alpha=0.5)

plt.xlim(-60, 60)
plt.title("Gęstość Błędów Predykcji wg Długości Lotu (Mediany i Średnie)", fontsize=16, fontweight='bold')
plt.xlabel("Błąd Predykcji: [Predykcja - Rzeczywistość] (minuty)\n(> 0 to bezpieczne przeszacowanie, < 0 to niedoszacowanie)", fontsize=14)
plt.ylabel("Kategoria Trasy", fontsize=14)

handles, labels = ax.get_legend_handles_labels()
if len(handles) >= 4:
    plt.legend(handles[-4:], labels[-4:], loc='upper right') 
else:
    plt.legend(loc='upper right')

plt.grid(axis='x', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.show()

print("\n STATYSTYKI BŁĘDU (w minutach) DLA POSZCZEGÓLNYCH TRAS:")
stats = plot_df.groupby("flight_length_category", observed=False)["model_error_min"].agg(['mean', 'median', 'count'])
stats.columns = ['Średnia błędu', 'Mediana błędu', 'Liczba lotów']
print(stats.round(2))

# COMMAND ----------

airport_stats = plot_df.groupby("dep_ap_sched").agg(
    MAE_min=("model_abs_error_min", "mean"),
    flight_count=("model_abs_error_min", "count")
).reset_index()

top_airports = airport_stats.sort_values("flight_count", ascending=False).head(20).sort_values("MAE_min", ascending=True)

plt.figure(figsize=(14, 8))
sns.barplot(x="MAE_min", y="dep_ap_sched", data=top_airports, palette="Blues_r")
plt.title("Średni Błąd Predykcji (MAE) na Top 20 Lotniskach Wylotowych", fontsize=16, fontweight='bold')
plt.xlabel("Średni Błąd MAE (Minuty)", fontsize=14)
plt.ylabel("Lotnisko Wylotu", fontsize=14)
for index, value in enumerate(top_airports["MAE_min"]):
    plt.text(value + 0.1, index, f"{value:.1f} min", va='center', fontsize=11)
plt.tight_layout()
plt.show()

plot_df["sched_min"] = plot_df["scheduled_block_time_sec"] / 60.0
bins = [0, 90, 180, 360, np.inf]
labels = ["Krótkie (<1.5h)", "Średnie (1.5h - 3h)", "Długie (3h - 6h)", "Ultra-Długie (>6h)"]
plot_df["flight_length_category"] = pd.cut(plot_df["sched_min"], bins=bins, labels=labels)

length_stats = plot_df.groupby("flight_length_category", observed=False).agg(
    MAE_min=("model_abs_error_min", "mean")
).reset_index()

plt.figure(figsize=(12, 6))
sns.barplot(x="flight_length_category", y="MAE_min", data=length_stats, palette="magma")
plt.title("Wielkość Błędu vs Zaplanowana Długość Trasy", fontsize=16, fontweight='bold')
plt.xlabel("Kategoria Trasy", fontsize=14)
plt.ylabel("Średni Błąd MAE (Minuty)", fontsize=14)
for index, value in enumerate(length_stats["MAE_min"]):
    plt.text(index, value + 0.2, f"{value:.1f} min", ha='center', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

plot_df["actual_min"] = plot_df["actual_block_time_sec"] / 60.0
plot_df["pred_min"] = plot_df["pred_actual_block_time_sec"] / 60.0

plot_df["MPE_pct"] = ((plot_df["pred_min"] - plot_df["actual_min"]) / plot_df["actual_min"]) * 100.0
plot_df["MAPE_pct"] = plot_df["MPE_pct"].abs()

airport_stats_pct = plot_df.groupby("dep_ap_sched").agg(
    MAPE_pct=("MAPE_pct", "mean"),
    MPE_pct=("MPE_pct", "mean"),
    flight_count=("MAPE_pct", "count")
).reset_index()

top_airports_pct = airport_stats_pct.sort_values("flight_count", ascending=False).head(20).sort_values("MAPE_pct", ascending=True)

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

sns.barplot(x="MAPE_pct", y="dep_ap_sched", data=top_airports_pct, palette="Blues_r", ax=axes[0])
axes[0].set_title("Średni Błąd Procentowy (MAPE) - Top 20 Lotnisk", fontsize=15, fontweight='bold')
axes[0].set_xlabel("MAPE (%) - Wielkość Błędu", fontsize=13)
axes[0].set_ylabel("Lotnisko Wylotu", fontsize=13)
for index, value in enumerate(top_airports_pct["MAPE_pct"]):
    axes[0].text(value + 0.1, index, f"{value:.1f}%", va='center', fontsize=11)

sns.barplot(x="MPE_pct", y="dep_ap_sched", data=top_airports_pct, palette="coolwarm", ax=axes[1])
axes[1].set_title("Kierunek Błędu Procentowego (MPE) - Top 20 Lotnisk", fontsize=15, fontweight='bold')
axes[1].set_xlabel("MPE (%)  [ Plus = Zapas / Minus = Spóźnienie ]", fontsize=13)
axes[1].set_ylabel("")
axes[1].axvline(0, color='black', linewidth=1.5)

for index, value in enumerate(top_airports_pct["MPE_pct"]):
    offset = 0.2 if value >= 0 else -0.2
    ha_align = 'left' if value >= 0 else 'right'
    axes[1].text(value + offset, index, f"{value:.1f}%", va='center', ha=ha_align, fontsize=11)

plt.tight_layout()
plt.show()

plot_df["sched_min"] = plot_df["scheduled_block_time_sec"] / 60.0
bins = [0, 90, 180, 360, np.inf]
labels = ["Krótkie (<1.5h)", "Średnie (1.5h - 3h)", "Długie (3h - 6h)", "Ultra-Długie (>6h)"]
plot_df["flight_length_category"] = pd.cut(plot_df["sched_min"], bins=bins, labels=labels)

length_stats_pct = plot_df.groupby("flight_length_category", observed=False).agg(
    MAPE_pct=("MAPE_pct", "mean"),
    MPE_pct=("MPE_pct", "mean")
).reset_index()

fig2, axes2 = plt.subplots(1, 2, figsize=(18, 6))

sns.barplot(x="flight_length_category", y="MAPE_pct", data=length_stats_pct, palette="magma", ax=axes2[0])
axes2[0].set_title("Wielkość Błędu Procentowego (MAPE) vs Kategoria Trasy", fontsize=15, fontweight='bold')
axes2[0].set_xlabel("Kategoria Trasy", fontsize=13)
axes2[0].set_ylabel("MAPE (%)", fontsize=13)
for index, value in enumerate(length_stats_pct["MAPE_pct"]):
    axes2[0].text(index, value + 0.2, f"{value:.1f}%", ha='center', fontsize=12, fontweight='bold')

sns.barplot(x="flight_length_category", y="MPE_pct", data=length_stats_pct, palette="coolwarm", ax=axes2[1])
axes2[1].set_title("Kierunek Błędu (MPE) vs Kategoria Trasy", fontsize=15, fontweight='bold')
axes2[1].set_xlabel("Kategoria Trasy", fontsize=13)
axes2[1].set_ylabel("MPE (%)", fontsize=13)
axes2[1].axhline(0, color='black', linewidth=1.5)

for index, value in enumerate(length_stats_pct["MPE_pct"]):
    offset = 0.2 if value >= 0 else -0.2
    va_align = 'bottom' if value >= 0 else 'top'
    axes2[1].text(index, value + offset, f"{value:.1f}%", ha='center', va=va_align, fontsize=12, fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

segment_df = plot_df.dropna(subset=["taxi_out_sec", "airborne_sec", "taxi_in_sec"]).copy()

segment_df["err_out_min"] = (segment_df["taxi_out_sec"] - segment_df["pred_taxi_out_sec"]).abs() / 60.0
segment_df["err_air_min"] = (segment_df["airborne_sec"] - segment_df["pred_airborne_sec"]).abs() / 60.0
segment_df["err_in_min"]  = (segment_df["taxi_in_sec"] - segment_df["pred_taxi_in_sec"]).abs() / 60.0

phase_length_stats = segment_df.groupby("flight_length_category", observed=False).agg(
    Taxi_Out=("err_out_min", "mean"),
    Airborne=("err_air_min", "mean"),
    Taxi_In=("err_in_min", "mean")
).reset_index()

melted_stats = pd.melt(
    phase_length_stats, 
    id_vars=["flight_length_category"], 
    value_vars=["Taxi_Out", "Airborne", "Taxi_In"],
    var_name="Segment Lotu", 
    value_name="MAE_min"
)

melted_stats["Segment Lotu"] = melted_stats["Segment Lotu"].replace({
    "Taxi_Out": "Taxi-Out (Wylot)",
    "Airborne": "Airborne (W powietrzu)",
    "Taxi_In": "Taxi-In (Przylot)"
})

plt.figure(figsize=(14, 7))
sns.barplot(data=melted_stats, x="flight_length_category", y="MAE_min", hue="Segment Lotu", palette="crest")

plt.title("Średni Błąd Predykcji (MAE) wg Fazy Lotu oraz Długości Trasy", fontsize=16, fontweight='bold')
plt.xlabel("Kategoria Trasy (Zaplanowany czas lotu)", fontsize=14)
plt.ylabel("Błąd MAE (Minuty)", fontsize=14)
plt.legend(title="Faza Lotu", fontsize=12, title_fontsize=13)
plt.grid(axis='y', linestyle='--', alpha=0.7)

for container in plt.gca().containers:
    plt.gca().bar_label(container, fmt='%.1f', padding=3, fontsize=11)

plt.tight_layout()
plt.show()

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

segment_df = plot_df.dropna(subset=[
    "taxi_out_sec", "airborne_sec", "taxi_in_sec",
    "pred_taxi_out_sec", "pred_airborne_sec", "pred_taxi_in_sec"
]).copy()

segment_df["mpe_out"] = ((segment_df["pred_taxi_out_sec"] - segment_df["taxi_out_sec"]) / segment_df["taxi_out_sec"]) * 100.0
segment_df["mpe_air"] = ((segment_df["pred_airborne_sec"] - segment_df["airborne_sec"]) / segment_df["airborne_sec"]) * 100.0
segment_df["mpe_in"]  = ((segment_df["pred_taxi_in_sec"] - segment_df["taxi_in_sec"]) / segment_df["taxi_in_sec"]) * 100.0

segment_df.replace([np.inf, -np.inf], np.nan, inplace=True)
segment_df.dropna(subset=["mpe_out", "mpe_air", "mpe_in"], inplace=True)

segment_df["mape_out"] = segment_df["mpe_out"].abs()
segment_df["mape_air"] = segment_df["mpe_air"].abs()
segment_df["mape_in"]  = segment_df["mpe_in"].abs()

phase_length_stats = segment_df.groupby("flight_length_category", observed=False).agg(
    MAPE_out=("mape_out", "mean"),
    MAPE_air=("mape_air", "mean"),
    MAPE_in=("mape_in", "mean"),
    MPE_out=("mpe_out", "mean"),
    MPE_air=("mpe_air", "mean"),
    MPE_in=("mpe_in", "mean")
).reset_index()

melted_mape = pd.melt(
    phase_length_stats, 
    id_vars=["flight_length_category"], 
    value_vars=["MAPE_out", "MAPE_air", "MAPE_in"],
    var_name="Segment Lotu", 
    value_name="Wartosc_Procentowa"
)
melted_mape["Segment Lotu"] = melted_mape["Segment Lotu"].replace({
    "MAPE_out": "Taxi-Out (Wylot)", "MAPE_air": "Airborne (W powietrzu)", "MAPE_in": "Taxi-In (Przylot)"
})

melted_mpe = pd.melt(
    phase_length_stats, 
    id_vars=["flight_length_category"], 
    value_vars=["MPE_out", "MPE_air", "MPE_in"],
    var_name="Segment Lotu", 
    value_name="Wartosc_Procentowa"
)
melted_mpe["Segment Lotu"] = melted_mpe["Segment Lotu"].replace({
    "MPE_out": "Taxi-Out (Wylot)", "MPE_air": "Airborne (W powietrzu)", "MPE_in": "Taxi-In (Przylot)"
})

fig, axes = plt.subplots(1, 2, figsize=(22, 8))

sns.barplot(data=melted_mape, x="flight_length_category", y="Wartosc_Procentowa", hue="Segment Lotu", palette="crest", ax=axes[0])
axes[0].set_title("Wielkość Błędu Procentowego (MAPE) wg Fazy Lotu i Trasy", fontsize=15, fontweight='bold')
axes[0].set_xlabel("Kategoria Trasy (Zaplanowany czas lotu)", fontsize=13)
axes[0].set_ylabel("Błąd MAPE (%)", fontsize=13)
axes[0].grid(axis='y', linestyle='--', alpha=0.7)
axes[0].legend(title="Faza Lotu", fontsize=11, title_fontsize=12)

for container in axes[0].containers:
    axes[0].bar_label(container, fmt='%.1f%%', padding=3, fontsize=11)

sns.barplot(data=melted_mpe, x="flight_length_category", y="Wartosc_Procentowa", hue="Segment Lotu", palette="coolwarm", ax=axes[1])
axes[1].set_title("Kierunek Błędu (MPE) wg Fazy Lotu i Trasy", fontsize=15, fontweight='bold')
axes[1].set_xlabel("Kategoria Trasy (Zaplanowany czas lotu)", fontsize=13)
axes[1].set_ylabel("Błąd MPE (%)", fontsize=13)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)
axes[1].axhline(0, color='black', linewidth=1.5)
axes[1].legend(title="Faza Lotu", fontsize=11, title_fontsize=12)

for container in axes[1].containers:
    labels = [f'{v.get_height():.1f}%' if v.get_height() != 0 else '' for v in container]
    axes[1].bar_label(container, labels=labels, padding=3, fontsize=11)

plt.tight_layout()
plt.show()

# COMMAND ----------

cols_to_show = [
    "leg_no", "fn_full_number", "event_date", "dep_ap_sched", "arr_ap_sched",
    "ac_registration", "flight_length_category", "leg_type", "eval_scope",
    "inactive_reason", "dq_any_flag",
    "scheduled_block_time_sec", "actual_block_time_sec", "pred_actual_block_time_sec",
    "actual_block_time_min", "pred_actual_block_time_min", "netline_eet_duration_min"
    "model_error_vs_actual_min", "model_abs_error_vs_actual_min",
    "taxi_out_sec", "airborne_sec", "taxi_in_sec"
]
cols_to_show = [c for c in cols_to_show if c in eval_all_df.columns]

print("🔴 TOP 20 NAJWIĘKSZYCH NIEDOSZACOWAŃ")
underestimated = eval_all_df.sort_values(by="model_error_vs_actual_min", ascending=True).head(20)
display(underestimated[cols_to_show])

print("\n" + "=" * 80 + "\n")
print("🔵 TOP 20 NAJWIĘKSZYCH PRZESZACOWAŃ")
overestimated = eval_all_df.sort_values(by="model_error_vs_actual_min", ascending=False).head(20)
display(overestimated[cols_to_show])

print("\n" + "=" * 80 + "\n")
print("[WARN] TOP 20 NAJWIĘKSZYCH BŁĘDÓW BEZWZGLĘDNYCH")
absolute_worst = eval_all_df.sort_values(by="model_abs_error_vs_actual_min", ascending=False).head(20)
display(absolute_worst[cols_to_show])

# COMMAND ----------


eval_all_df["error_plan_vs_pred_min"] = eval_all_df["pred_actual_block_time_min"] - eval_all_df["scheduled_block_time_min"]
eval_all_df["abs_error_plan_vs_pred_min"] = eval_all_df["error_plan_vs_pred_min"].abs()

cols_plan_vs_pred = [
    "leg_no", "fn_full_number", "event_date", "dep_ap_sched", "arr_ap_sched", "ac_registration",
    "scheduled_block_time_min", "pred_actual_block_time_min", "actual_block_time_min", "netline_eet_duration_min"
    "error_plan_vs_pred_min", 
    "model_error_vs_actual_min",
    "eval_scope", "inactive_reason"
]
cols_plan_vs_pred = [c for c in cols_plan_vs_pred if c in eval_all_df.columns]

print(" TOP 20: Największe konflikty między Modelem a Rozkładem Lotów (wg wartości bezwzględnej)")
worst_plan_vs_pred = eval_all_df.sort_values(by="abs_error_plan_vs_pred_min", ascending=False).head(20)
display(worst_plan_vs_pred[cols_plan_vs_pred])

# COMMAND ----------

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

print(" Uruchamiam diagnostykę dla zmiennych: distance_km oraz local_hour_dep...")

diag_df = eval_clean_df.dropna(subset=["actual_block_time_sec", "distance_km", "local_hour_dep"]).copy()

diag_df["error_min"] = (diag_df["actual_block_time_sec"] - diag_df["pred_actual_block_time_sec"]) / 60.0
diag_df["abs_error_min"] = diag_df["error_min"].abs()

bins = np.arange(0, diag_df["distance_km"].max() + 500, 500)
labels = [f"{int(b)}-{int(b+500)}km" for b in bins[:-1]]
diag_df["distance_bin"] = pd.cut(diag_df["distance_km"], bins=bins, labels=labels)

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
sns.set_theme(style="whitegrid")

hourly_mae = diag_df.groupby("local_hour_dep")["abs_error_min"].mean().reset_index()
sns.barplot(data=hourly_mae, x="local_hour_dep", y="abs_error_min", ax=axes[0, 0], palette="viridis")
axes[0, 0].set_title("Wielkość błędu (MAE) vs Lokalna Godzina Wylotu", fontweight='bold')
axes[0, 0].set_ylabel("MAE (minuty)")
axes[0, 0].set_xlabel("Lokalna godzina (0-23)")

sns.boxplot(data=diag_df, x="local_hour_dep", y="error_min", ax=axes[0, 1], showfliers=False, palette="coolwarm")
axes[0, 1].axhline(0, color="black", linewidth=2, linestyle="-")
axes[0, 1].set_title("Kierunek błędu (Bias) vs Lokalna Godzina Wylotu", fontweight='bold')
axes[0, 1].set_ylabel("Bias (min) [>0: Przeszacowanie, <0: Niedoszacowanie]")
axes[0, 1].set_xlabel("Lokalna godzina (0-23)")

dist_mae = diag_df.groupby("distance_bin", observed=True)["abs_error_min"].mean().reset_index()
sns.barplot(data=dist_mae, x="distance_bin", y="abs_error_min", ax=axes[1, 0], palette="crest")
axes[1, 0].set_title("Wielkość błędu (MAE) vs Dystans (km)", fontweight='bold')
axes[1, 0].set_ylabel("MAE (minuty)")
axes[1, 0].set_xlabel("Dystans trasy")
axes[1, 0].tick_params(axis='x', rotation=45)

sns.boxplot(data=diag_df, x="distance_bin", y="error_min", ax=axes[1, 1], showfliers=False, palette="coolwarm")
axes[1, 1].axhline(0, color="black", linewidth=2, linestyle="-")
axes[1, 1].set_title("Kierunek błędu (Bias) vs Dystans (km)", fontweight='bold')
axes[1, 1].set_ylabel("Bias (min)")
axes[1, 1].set_xlabel("Dystans trasy")
axes[1, 1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------






    
    
    
    



