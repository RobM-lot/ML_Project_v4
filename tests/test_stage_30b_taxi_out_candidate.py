import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30b_taxi_out_candidate.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage30b_taxi_out_candidate_standalone", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


candidate = _load_module()


def _read_module_source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_module_imports_without_pyspark_installed():
    assert candidate.ENTITY_COL == "dep_ap_sched"
    assert candidate.DATE_COL == "event_date"


def test_taxi_out_candidate_specs_match_poc_target():
    assert candidate.ENTITY_COL == "dep_ap_sched"
    assert candidate.DATE_COL == "event_date"
    assert candidate.DIRTY_EVENT_DATE_COL == "dirty_event_date"
    assert candidate.AFFECTED_OUTPUT_DATE_COL == "affected_output_date"
    assert candidate.TARGET_COLS_DICT == {
        "taxi_out_sec": "taxi_out",
        "duration_ratio": "dur_ratio_dep",
    }
    assert candidate.COUNT_PREFIX == "dep"
    assert candidate.ROLLING_WINDOWS_DAYS == {"7d": 7, "30d": 30}
    assert candidate.CURRENT_DAY_INCLUDED is False


def test_non_ema_parity_columns_include_expected_feature_families():
    cols = set(candidate.NON_EMA_PARITY_COLUMNS)

    for col in (
        "dep_ap_sched",
        "event_date",
        "avg_taxi_out_7d",
        "std_taxi_out_7d",
        "p90_taxi_out_7d",
        "min_taxi_out_7d",
        "max_taxi_out_7d",
        "avg_dur_ratio_dep_7d",
        "std_dur_ratio_dep_7d",
        "p90_dur_ratio_dep_7d",
        "min_dur_ratio_dep_7d",
        "max_dur_ratio_dep_7d",
        "avg_taxi_out_30d",
        "std_taxi_out_30d",
        "p90_taxi_out_30d",
        "min_taxi_out_30d",
        "max_taxi_out_30d",
        "avg_dur_ratio_dep_30d",
        "std_dur_ratio_dep_30d",
        "p90_dur_ratio_dep_30d",
        "min_dur_ratio_dep_30d",
        "max_dur_ratio_dep_30d",
        "count_dep_7d",
        "count_dep_30d",
        "trend_taxi_out_7d",
        "trend_dur_ratio_dep_7d",
        "has_hist_dep_7d",
        "has_hist_dep_30d",
        "days_since_last_event",
    ):
        assert col in cols


def test_non_ema_parity_columns_exclude_ema_and_delta_ema():
    assert all(not col.startswith("ema_") for col in candidate.NON_EMA_PARITY_COLUMNS)
    assert all(not col.startswith("delta_ema_avg_") for col in candidate.NON_EMA_PARITY_COLUMNS)
    assert candidate.EMA_POLICY == "deferred"


def test_affected_output_expansion_uses_d_plus_1_through_d_plus_30():
    source = _read_module_source()

    assert "F.sequence(F.lit(1), F.lit(30))" in source
    assert "F.date_add(F.col(DIRTY_EVENT_DATE_COL), F.col(\"_stage30b_offset\"))" in source
    assert "F.col(\"_stage30b_offset\") <= F.lit(7)" in source


def test_affected_output_uniqueness_is_entity_and_affected_output_date_not_leg():
    source = _read_module_source()

    assert ".groupBy(ENTITY_COL, AFFECTED_OUTPUT_DATE_COL)" in source
    assert ".groupBy(\"leg_no\", ENTITY_COL, AFFECTED_OUTPUT_DATE_COL)" not in source
    assert "F.countDistinct(\"leg_no\").alias(\"dirty_leg_count\")" in source


def test_candidate_recompute_is_entity_scoped_for_parity_safety():
    source = _read_module_source()

    assert "affected_entities = affected_outputs_df.select(ENTITY_COL).dropDuplicates()" in source
    assert "scoped = cleaned_flight_df.join(affected_entities, on=ENTITY_COL, how=\"inner\")" in source
    assert "candidate.join(affected_pairs, on=[ENTITY_COL, DATE_COL], how=\"inner\")" in source


def test_candidate_builder_has_no_ema_apply_in_pandas_or_pipeline_import():
    source = _read_module_source()

    assert "applyInPandas" not in source
    assert "ema_" not in "\n".join(candidate.NON_EMA_PARITY_COLUMNS)
    assert "delta_ema_avg_" not in "\n".join(candidate.NON_EMA_PARITY_COLUMNS)
    assert "src.pipeline.feature_store" not in source
    assert "pipeline.feature_store" not in source


def test_compare_helper_uses_non_ema_columns_and_full_outer_statuses():
    source = _read_module_source()

    assert "current_mv_df.select(*NON_EMA_PARITY_COLUMNS)" in source
    assert "how=\"full_outer\"" in source
    assert "missing_in_candidate" in source
    assert "missing_in_current" in source
    assert "value_mismatch" in source
    assert "matched" in source


def test_module_contains_no_production_write_strategy_or_mutating_patterns():
    source = _read_module_source()

    for forbidden in (
        "foreachBatch",
        "MERGE INTO",
        ".write",
        ".saveAsTable",
        "CREATE TABLE",
        "ALTER TABLE",
        "INSERT INTO",
        "UPDATE",
        "DELETE",
        "VACUUM",
        "OPTIMIZE",
        "dirty_key_table",
        "applyInPandas",
        "src.pipeline.feature_store",
        "pipeline.feature_store",
    ):
        assert forbidden not in source
