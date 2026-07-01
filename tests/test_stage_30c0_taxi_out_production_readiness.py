from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DESIGN_PATH = REPO_ROOT / "docs" / "stage_30b6_30c0_taxi_out_production_design.md"
HANDOFF_PATH = REPO_ROOT / "docs" / "stage_30b_dirty_key_poc_design.md"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "16_stage30c0_taxi_out_production_readiness.py"

REQUIRED_TABLE_REFERENCES = {
    "panda_silver_prod",
    "occ_ops",
    "netline___schedops__leg",
    "netline___schedops__leg_times",
    "panda_silver_dev.ml_ops.cleaned_flight_data_full_table",
    "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out",
}

REQUIRED_NOTEBOOK_COLUMNS = {
    "leg_no",
    "update_key",
    "__START_AT",
    "__END_AT",
    "dep_ap_sched",
    "dep_sched_dt",
    "leg_state",
    "leg_type",
    "counter",
    "offblock_dt",
    "airborne_dt",
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
    "taxi_out_sec",
    "scheduled_block_time_sec",
    "actual_block_time_sec",
}

FINAL_BOOLEAN_SUMMARY_FIELDS = {
    "source_leg_schema_ok",
    "source_leg_times_schema_ok",
    "cleaned_flight_schema_ok",
    "current_mv_schema_ok",
    "current_mv_key_unique",
    "current_mv_key_non_null",
    "cdf_versions_configured",
    "cdf_leg_read_ok",
    "cdf_leg_times_read_ok",
    "dirty_preview_available",
    "read_only_no_writes",
}

FORBIDDEN_NOTEBOOK_PATTERNS = {
    "spark.readStream",
    "readStream",
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


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_stage_30b6_30c0_design_doc_exists_and_has_production_flow():
    source = _read(DESIGN_PATH)

    assert DESIGN_PATH.exists()
    assert "Stage 30B-6 / 30C-0 Taxi-out Partial Recompute Production Design" in source
    assert "Stage 30B-5 proved the read-only end-to-end flow" in source
    assert "The next production target is only `ft_airport_daily_taxi_out`" in source
    assert "read source-specific CDF windows" in source
    assert "extract dirty leg_no/source" in source
    assert "expand D+1...D+30 affected output dates" in source
    assert "advance watermarks only after successful write + validation" in source
    assert "This stage does not implement the production write mechanism" in source


def test_stage_30b6_30c0_design_doc_covers_watermarks_validation_and_deferred_scope():
    source = _read(DESIGN_PATH)

    for table_name in REQUIRED_TABLE_REFERENCES:
        assert table_name in source

    assert "source-specific commit versions" in source
    assert "Never use one global CDF version" in source
    assert "\"leg\": \"<last_processed_leg_commit_version>\"" in source
    assert "\"leg_times\": \"<last_processed_leg_times_commit_version>\"" in source
    assert "If a run fails, rerun the same source-specific version range" in source
    assert "Watermarks must advance only after" in source
    assert "Validation Gates" in source
    assert "affected output pairs are unique" in source
    assert "Future Write Strategy" in source
    assert "Option A" in source
    assert "Option B" in source
    assert "Option C" in source
    assert "Recommended next implementation: Option A" in source
    assert "Idempotency And Retry" in source
    assert "update_preimage" in source
    assert "update_postimage" in source
    assert "D+1 ... D+30" in source
    assert "stream/readStream implementation" in source
    assert "EMA remains deferred" in source
    assert "no stream aggregations" in source
    assert "read-only diagnostic" in source


def test_stage_30b_dirty_key_doc_links_to_30c0_handoff():
    source = _read(HANDOFF_PATH)

    assert "30B-6 / 30C-0 Handoff" in source
    assert "batch CDF polling by source-specific commit" in source
    assert "docs/stage_30b6_30c0_taxi_out_production_design.md" in source
    assert "notebooks/16_stage30c0_taxi_out_production_readiness.py" in source


def test_stage_30c0_readiness_notebook_exists_and_is_safe_gated():
    source = _read(NOTEBOOK_PATH)

    assert NOTEBOOK_PATH.exists()
    assert "Stage 30C-0 taxi-out production readiness diagnostics" in source
    assert "This notebook is read-only and must not mutate production tables or workspace resources." in source
    assert "RUN_READINESS = False" in source
    assert "ENTITY_FILTER = \"\"" in source
    assert "if not RUN_READINESS:" in source
    assert "RUN_READINESS_FALSE" in source


def test_stage_30c0_readiness_notebook_references_required_tables_and_columns():
    source = _read(NOTEBOOK_PATH)

    for table_name in REQUIRED_TABLE_REFERENCES:
        assert table_name in source

    for column_name in REQUIRED_NOTEBOOK_COLUMNS:
        assert column_name in source

    assert "LEG_CDF_STARTING_VERSION" in source
    assert "LEG_CDF_ENDING_VERSION" in source
    assert "LEG_TIMES_CDF_STARTING_VERSION" in source
    assert "LEG_TIMES_CDF_ENDING_VERSION" in source
    assert "spark.read.option(\"readChangeFeed\", \"true\")" in source


def test_stage_30c0_readiness_notebook_checks_readiness_without_writes():
    source = _read(NOTEBOOK_PATH)

    assert "current_mv.groupBy(ENTITY_COL, DATE_COL)" in source
    assert "max_current_mv_event_date" in source
    assert "cleaned_horizon" in source
    assert "key columns non-null" in source
    assert "null dep_ap_sched/event_date keys" in source
    assert "taxi_out recompute null counts" in source
    assert "Dirty event date D affects output dates D+1...D+30" in source
    assert "EMA remains deferred" in source
    assert "map_dirty_legs_to_taxi_out_events" in source
    assert "expand_dirty_taxi_out_events_to_affected_outputs" in source
    assert "unique affected output key" in source
    assert "Recommended future target key columns: dep_ap_sched, event_date" in source
    assert "final boolean summary" in source
    for field in FINAL_BOOLEAN_SUMMARY_FIELDS:
        assert field in source
    assert "readiness failed checks" in source


def test_stage_30c0_readiness_notebook_contains_no_stream_or_mutating_patterns():
    source = _read(NOTEBOOK_PATH)

    for pattern in FORBIDDEN_NOTEBOOK_PATTERNS:
        assert pattern not in source
