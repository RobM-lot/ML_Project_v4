from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "15_stage30b5_end_to_end_batch_cdf_parity.py"

REQUIRED_SOURCE_TABLE_PARTS = {
    "panda_silver_prod",
    "occ_ops",
    "netline___schedops__leg",
    "netline___schedops__leg_times",
}

REQUIRED_SILVER_TABLE_NAMES = {
    "panda_silver_dev.ml_ops.cleaned_flight_data_full_table",
    "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out",
}

REQUIRED_VERSION_CONTROLS = {
    "LEG_CDF_STARTING_VERSION",
    "LEG_CDF_ENDING_VERSION",
    "LEG_TIMES_CDF_STARTING_VERSION",
    "LEG_TIMES_CDF_ENDING_VERSION",
}

REQUIRED_COLUMNS = {
    "leg_no",
    "update_key",
    "__START_AT",
    "__END_AT",
    "dep_sched_dt",
    "dep_ap_sched",
    "leg_state",
    "leg_type",
    "counter",
    "offblock_dt",
    "airborne_dt",
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
}

REQUIRED_HELPERS = {
    "map_dirty_legs_to_taxi_out_events",
    "select_current_latest",
    "expand_dirty_taxi_out_events_to_affected_outputs",
    "build_taxi_out_candidate_for_affected_outputs",
    "compare_taxi_out_candidate_to_current_mv",
}

FORBIDDEN_PATTERNS = {
    "spark.readStream",
    "writeStream",
    "foreachBatch",
    "foreach_batch",
    ".write",
    ".saveAsTable",
    ".insertInto",
    ".toTable",
    ".format(\"delta\")",
    ".format('delta')",
    "ALTER TABLE",
    "SET TBLPROPERTIES",
    "enableChangeDataFeed",
    "delta.enableChangeDataFeed",
    "CREATE TABLE",
    "CREATE OR REPLACE",
    "DROP TABLE",
    "DELETE FROM",
    "UPDATE ",
    "MERGE INTO",
    "databricks bundle",
    "databricks pipelines",
    "src.pipeline.feature_store",
    "pipeline.feature_store",
}


def _read_notebook() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_stage_30b5_batch_cdf_parity_notebook_exists_and_is_safe_gated():
    source = _read_notebook()

    assert NOTEBOOK_PATH.exists()
    assert "Stage 30B-5 read-only end-to-end batch CDF parity POC" in source
    assert "This notebook must not mutate production tables or workspace resources." in source
    assert "RUN_PARITY = False" in source
    assert "if not RUN_PARITY:" in source
    assert "RUN_PARITY_FALSE" in source


def test_stage_30b5_batch_cdf_parity_notebook_references_expected_tables_and_versions():
    source = _read_notebook()

    for table_name in REQUIRED_SOURCE_TABLE_PARTS | REQUIRED_SILVER_TABLE_NAMES:
        assert table_name in source

    for control in REQUIRED_VERSION_CONTROLS:
        assert control in source

    assert "At least one source-specific CDF starting version must be set." in source
    assert ".option(\"startingVersion\", str(starting_version))" in source
    assert ".option(\"endingVersion\", str(ending_version))" in source


def test_stage_30b5_batch_cdf_parity_notebook_uses_batch_cdf_only():
    source = _read_notebook()

    assert "spark.read.option(\"readChangeFeed\", \"true\")" in source
    assert "_read_batch_cdf" in source
    assert "_read_cdf_stream_table" not in source
    assert "RUN_STREAM" not in source


def test_stage_30b5_batch_cdf_parity_notebook_extracts_dirty_keys_with_cdf_metadata():
    source = _read_notebook()

    for column_name in REQUIRED_COLUMNS:
        assert column_name in source

    assert "\"leg\"" in source
    assert "\"leg_times\"" in source
    assert "dirty_source_alias" in source
    assert "cdf_change_types" in source
    assert "latest_commit_version" in source


def test_stage_30b5_batch_cdf_parity_notebook_runs_end_to_end_non_ema_parity_flow():
    source = _read_notebook()

    for helper_name in REQUIRED_HELPERS:
        assert helper_name in source

    dirty_events_pos = source.index("dirty_events = dirty_events_limited.join")
    affected_outputs_pos = source.index("affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs")
    candidate_pos = source.index("candidate_scoped = build_taxi_out_candidate_for_affected_outputs")
    current_scoped_pos = source.index("current_scoped = (")
    compare_pos = source.index("compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped")

    assert dirty_events_pos < affected_outputs_pos < candidate_pos < current_scoped_pos < compare_pos
    assert "NON_EMA_PARITY_COLUMNS" in source
    assert "current_mv.select(*NON_EMA_PARITY_COLUMNS)" in source
    assert ".join(affected_pairs, on=[ENTITY_COL, DATE_COL], how=\"inner\")" in source
    assert "REQUIRE_FULL_AFFECTED_WINDOW = True" in source
    assert "MAX_DIRTY_EVENTS = 1000" in source


def test_stage_30b5_batch_cdf_parity_notebook_contains_no_stream_or_mutating_patterns():
    source = _read_notebook()

    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in source
