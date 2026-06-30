from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DOC_PATH = REPO_ROOT / "docs" / "stage_30b_dirty_key_poc_design.md"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "12_stage30b_dirty_key_source_discovery.py"
FEATURE_STORE_PATH = REPO_ROOT / "src" / "pipeline" / "feature_store.py"
DATABRICKS_PATH = REPO_ROOT / "databricks.yml"
PIPELINE_PATH = REPO_ROOT / "resources" / "pipeline.yml"

SOURCE_TABLES = {
    "netline___schedops__leg",
    "netline___schedops__leg_times",
    "netline___schedops__leg_misc",
}

CURRENT_STREAM_TABLES = {
    "ft_leg_status",
    "ft_leg_times",
    "ft_leg_misc",
}

DOC_REQUIRED_TERMS = {
    "skipChangeCommits",
    "ft_airport_daily_taxi_out",
    "dirty-key",
    "D+1",
    "D+30",
    "EMA",
    "30B-0",
    "30B-1",
    "30B-2",
    "30B-3",
    "30B-4",
}

FORBIDDEN_NOTEBOOK_PATTERNS = {
    ".write",
    ".saveAsTable",
    "foreachBatch",
    "foreach_batch_sink",
    "MERGE INTO",
    "readChangeFeed",
    "read_change_feed",
    "CREATE TABLE",
    "ALTER TABLE",
    "INSERT INTO",
    "UPDATE ",
    "DELETE ",
    "VACUUM",
    "OPTIMIZE",
}

FORBIDDEN_PRODUCTION_PATTERNS = {
    "foreachBatch",
    "foreach_batch_sink",
    "MERGE INTO",
    "readChangeFeed",
    "read_change_feed",
    "dirty_key",
    "dirty-key",
    "watermark",
    "partial_recompute",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_stage_30b_design_doc_and_diagnostic_notebook_exist():
    assert DOC_PATH.exists()
    assert NOTEBOOK_PATH.exists()


def test_stage_30b_notebook_mentions_required_candidate_tables():
    source = _read(NOTEBOOK_PATH)

    for table_name in SOURCE_TABLES | CURRENT_STREAM_TABLES:
        assert table_name in source


def test_stage_30b_design_doc_covers_required_topics():
    doc = _read(DOC_PATH)

    for term in DOC_REQUIRED_TERMS:
        assert term in doc


def test_stage_30b_notebook_contains_no_executable_mutating_patterns():
    source = _read(NOTEBOOK_PATH)

    for pattern in FORBIDDEN_NOTEBOOK_PATTERNS:
        assert pattern not in source


def test_stage_30b_production_files_do_not_contain_dirty_key_logic():
    feature_store_source = _read(FEATURE_STORE_PATH)
    databricks_source = _read(DATABRICKS_PATH)
    pipeline_source = _read(PIPELINE_PATH)

    for pattern in FORBIDDEN_PRODUCTION_PATTERNS:
        assert pattern not in feature_store_source

    assert "30B" not in databricks_source
    assert "dirty" not in databricks_source.lower()
    assert "30B" not in pipeline_source
    assert "dirty" not in pipeline_source.lower()
