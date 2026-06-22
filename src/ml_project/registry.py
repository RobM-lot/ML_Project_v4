from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

import mlflow
from mlflow import MlflowClient

from .common import configure_runtime
from .widgets import ensure_dropdown_widget, ensure_text_widget, get_bool_widget, get_widget_value


def ensure_register_widgets(dbutils, default_promote: bool = True) -> None:
    ensure_text_widget(
        dbutils,
        "RUN_ID",
        "",
        "0. Candidate run_id (zostaw puste, aby wybrać najlepszy run z eksperymentu)",
    )
    ensure_dropdown_widget(
        dbutils,
        "PROMOTE_IF_PASS",
        "True" if default_promote else "False",
        ["True", "False"],
        "1. Promote to champion if gates pass",
    )


def _settings_dict(settings: Any) -> Dict[str, Any]:
    if is_dataclass(settings):
        return asdict(settings)
    if isinstance(settings, Mapping):
        return dict(settings)
    raise TypeError(f"Nieobsługiwany typ settings: {type(settings)!r}")


def _safe_metric(run, key: str) -> Optional[float]:
    value = run.data.metrics.get(key)
    return float(value) if value is not None else None


def _safe_param(run, key: str) -> Optional[str]:
    value = run.data.params.get(key)
    return str(value) if value is not None else None


def _get_experiment_id(experiment_path: str) -> str:
    experiment = mlflow.get_experiment_by_name(experiment_path)
    if experiment is None:
        raise RuntimeError(
            f"Nie znaleziono eksperymentu MLflow: {experiment_path}. "
            "Uruchom najpierw notebook treningowy."
        )
    return experiment.experiment_id


def _required_metric_keys(settings: Any) -> Sequence[str]:
    return tuple(getattr(settings, "REGISTER_REQUIRED_METRICS", ()))


def _required_param_keys(settings: Any) -> Sequence[str]:
    return tuple(getattr(settings, "REGISTER_REQUIRED_PARAMS", ()))


def _run_matches_contract(run, settings: Any) -> bool:
    for key in _required_metric_keys(settings):
        if _safe_metric(run, key) is None:
            return False

    for key in _required_param_keys(settings):
        if _safe_param(run, key) is None:
            return False

    expected_training_table = getattr(settings, "TRAINING_DATASET_TABLE", None)
    if expected_training_table:
        training_table = _safe_param(run, "training_dataset_table")
        if training_table != expected_training_table:
            return False

    expected_aircraft_contract = getattr(settings, "MODEL_AIRCRAFT_FEATURE_COL", None)
    if expected_aircraft_contract:
        aircraft_contract = _safe_param(run, "model_aircraft_feature_col")
        if aircraft_contract != expected_aircraft_contract:
            return False

    expected_run_name = getattr(settings, "TRAINING_RUN_NAME", None)
    run_name = getattr(run.data, "tags", {}).get("mlflow.runName")
    if expected_run_name and run_name and run_name != expected_run_name:
        return False

    return True


def _candidate_sort_key(run) -> tuple:
    mae = _safe_metric(run, "TOTAL_MAE_actual_block_time")
    abs_bias = _safe_metric(run, "TOTAL_ABS_BIAS_actual_block_time")
    coverage_gap = _safe_metric(run, "TOTAL_P90_COVERAGE_GAP_pct")
    win_rate = _safe_metric(run, "HOLDOUT_WIN_RATE_VS_SCHEDULE_pct")
    overfit = _safe_metric(run, "OVERFIT_DIFF_actual_block_time_sec")
    valid_rows = _safe_param(run, "valid_rows_holdout_split") or _safe_param(run, "valid_rows")
    try:
        valid_rows_num = int(valid_rows) if valid_rows is not None else -1
    except Exception:
        valid_rows_num = -1

    return (
        mae if mae is not None else float("inf"),
        abs_bias if abs_bias is not None else float("inf"),
        coverage_gap if coverage_gap is not None else float("inf"),
        -(win_rate if win_rate is not None else float("-inf")),
        overfit if overfit is not None else float("inf"),
        -valid_rows_num,
    )


def _find_best_candidate_run(client: MlflowClient, experiment_id: str, settings: Any):
    runs = client.search_runs(
        experiment_ids=[experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=200,
        order_by=["attribute.start_time DESC"],
    )

    eligible = []
    rejected = []
    for run in runs:
        if not _run_matches_contract(run, settings):
            rejected.append(run.info.run_id)
            continue
        eligible.append(run)

    if not eligible:
        raise RuntimeError(
            "Nie znaleziono żadnego zakończonego runa zgodnego z aktualnym kontraktem treningowym. "
            f"Sprawdzono {len(runs)} runów, odrzucono {len(rejected)}."
        )

    eligible.sort(key=_candidate_sort_key)
    return eligible[0]


def _get_run_by_id(client: MlflowClient, run_id: str):
    try:
        return client.get_run(run_id)
    except Exception as e:
        raise RuntimeError(f"Nie udało się pobrać run_id={run_id}: {e}") from e


def _champion_info(client: MlflowClient, model_name: str) -> Optional[Dict[str, Any]]:
    try:
        mv = client.get_model_version_by_alias(model_name, "champion")
    except Exception:
        return None

    result = {
        "model_name": model_name,
        "version": mv.version,
        "run_id": getattr(mv, "run_id", None),
        "source": getattr(mv, "source", None),
        "current_stage": getattr(mv, "current_stage", None),
    }

    if result["run_id"]:
        try:
            run = client.get_run(result["run_id"])
            result["mae_sec"] = _safe_metric(run, "TOTAL_MAE_actual_block_time")
            result["abs_bias_sec"] = _safe_metric(run, "TOTAL_ABS_BIAS_actual_block_time")
            result["coverage_pct"] = _safe_metric(run, "TOTAL_P90_COVERAGE_actual_block_time")
            result["win_rate_pct"] = _safe_metric(run, "HOLDOUT_WIN_RATE_VS_SCHEDULE_pct")
        except Exception:
            pass
    return result


def _gate_checks(run, settings: Any, champion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    mae_sec = _safe_metric(run, "TOTAL_MAE_actual_block_time")
    abs_bias_sec = _safe_metric(run, "TOTAL_ABS_BIAS_actual_block_time")
    coverage_pct = _safe_metric(run, "TOTAL_P90_COVERAGE_actual_block_time")
    baseline_mae_sec = _safe_metric(run, "BASELINE_MAE_actual_block_time")
    win_rate_pct = _safe_metric(run, "HOLDOUT_WIN_RATE_VS_SCHEDULE_pct")
    overfit_diff_sec = _safe_metric(run, "OVERFIT_DIFF_actual_block_time_sec")
    monthly_mae_std_sec = _safe_metric(run, "HOLDOUT_MONTHLY_MAE_STD_sec")

    valid_rows_raw = _safe_param(run, "valid_rows_holdout_split") or _safe_param(run, "valid_rows")
    try:
        valid_rows = int(valid_rows_raw) if valid_rows_raw is not None else 0
    except Exception:
        valid_rows = 0

    checks = {
        "has_mae": mae_sec is not None,
        "has_abs_bias": abs_bias_sec is not None,
        "has_coverage": coverage_pct is not None,
        "has_win_rate": win_rate_pct is not None,
        "min_valid_rows": valid_rows >= int(settings.PROMOTION_MIN_VALID_ROWS),
        "baseline_beat": True if not bool(settings.PROMOTION_REQUIRE_BASELINE_BEAT) else (
            baseline_mae_sec is not None and mae_sec is not None and mae_sec < baseline_mae_sec
        ),
        "abs_bias_ok": abs_bias_sec is not None and (abs_bias_sec / 60.0) <= float(settings.PROMOTION_MAX_ABS_BIAS_MIN),
        "coverage_min_ok": coverage_pct is not None and coverage_pct >= float(settings.PROMOTION_MIN_P90_COVERAGE_PCT),
        "coverage_max_ok": coverage_pct is not None and coverage_pct <= float(settings.PROMOTION_MAX_P90_COVERAGE_PCT),
        "win_rate_ok": win_rate_pct is not None and win_rate_pct >= float(settings.PROMOTION_MIN_HOLDOUT_WIN_RATE_PCT),
        "overfit_ok": overfit_diff_sec is not None and (overfit_diff_sec / 60.0) <= float(settings.PROMOTION_MAX_OVERFIT_DIFF_MIN),
        "monthly_stability_ok": (
            monthly_mae_std_sec is None or (monthly_mae_std_sec / 60.0) <= float(settings.PROMOTION_MAX_MONTHLY_MAE_STD_MIN)
        ),
    }

    champion_beat = True
    if bool(settings.PROMOTION_REQUIRE_CHAMPION_BEAT) and champion and champion.get("mae_sec") is not None:
        champion_beat = mae_sec is not None and mae_sec < float(champion["mae_sec"])
    checks["champion_beat"] = champion_beat

    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "candidate_metrics": {
            "mae_sec": mae_sec,
            "abs_bias_sec": abs_bias_sec,
            "coverage_pct": coverage_pct,
            "baseline_mae_sec": baseline_mae_sec,
            "win_rate_pct": win_rate_pct,
            "overfit_diff_sec": overfit_diff_sec,
            "monthly_mae_std_sec": monthly_mae_std_sec,
            "valid_rows": valid_rows,
        },
        "champion_metrics": champion,
    }


def _register_and_promote(client: MlflowClient, run_id: str, settings: Any) -> Dict[str, Any]:
    model_uri = f"runs:/{run_id}/model"
    registered = mlflow.register_model(model_uri=model_uri, name=settings.UC_MODEL_NAME)

    try:
        client.set_registered_model_alias(
            name=settings.UC_MODEL_NAME,
            alias=settings.MODEL_ALIAS,
            version=registered.version,
        )
    except Exception as e:
        raise RuntimeError(
            f"Model został zarejestrowany jako wersja {registered.version}, "
            f"ale nie udało się ustawić aliasu {settings.MODEL_ALIAS!r}: {e}"
        ) from e

    return {
        "registered_model_name": settings.UC_MODEL_NAME,
        "registered_model_version": registered.version,
        "registered_model_alias": settings.MODEL_ALIAS,
        "registered_model_uri": model_uri,
    }


def run_register_best(spark, dbutils, settings: Any) -> Dict[str, Any]:
    configure_runtime(settings, spark=spark)
    ensure_register_widgets(dbutils, default_promote=bool(settings.PROMOTE_IF_PASS_DEFAULT))

    candidate_run_id = get_widget_value(dbutils, "RUN_ID", settings.REGISTER_RUN_ID_WIDGET_DEFAULT).strip()
    promote_if_pass = get_bool_widget(dbutils, "PROMOTE_IF_PASS", bool(settings.PROMOTE_IF_PASS_DEFAULT))

    client = MlflowClient()
    experiment_id = _get_experiment_id(settings.EXPERIMENT_PATH)

    if candidate_run_id:
        candidate_run = _get_run_by_id(client, candidate_run_id)
        selection_mode = "explicit_run_id"
    else:
        candidate_run = _find_best_candidate_run(client, experiment_id, settings)
        selection_mode = "best_finished_run_from_experiment"

    champion = _champion_info(client, settings.UC_MODEL_NAME)
    gate_result = _gate_checks(candidate_run, settings, champion)

    result = {
        "selection_mode": selection_mode,
        "candidate_run_id": candidate_run.info.run_id,
        "experiment_path": settings.EXPERIMENT_PATH,
        "model_name": settings.UC_MODEL_NAME,
        "model_alias": settings.MODEL_ALIAS,
        "promote_if_pass": promote_if_pass,
        "gates_passed": gate_result["passed"],
        "checks": gate_result["checks"],
        "candidate_metrics": gate_result["candidate_metrics"],
        "champion_metrics": gate_result["champion_metrics"],
    }

    champion_run_id = champion.get("run_id") if champion else None
    candidate_is_current_champion = bool(champion_run_id) and candidate_run.info.run_id == champion_run_id

    if gate_result["passed"] and promote_if_pass and candidate_is_current_champion:
        result["promotion"] = None
        result["decision"] = "ALREADY_CHAMPION"
    elif gate_result["passed"] and promote_if_pass:
        result["promotion"] = _register_and_promote(client, candidate_run.info.run_id, settings)
        result["decision"] = "PROMOTED"
    elif gate_result["passed"]:
        result["promotion"] = None
        result["decision"] = "PASSED_GATES_NOT_PROMOTED"
    else:
        result["promotion"] = None
        result["decision"] = "REJECTED_BY_GATES"

    return result