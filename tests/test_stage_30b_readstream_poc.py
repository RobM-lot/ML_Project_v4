import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "14_stage30b_readstream_dirty_key_poc.py"

SOURCE_TABLES = {
    "netline___schedops__leg",
    "netline___schedops__leg_times",
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
}

CDF_COLUMNS = {
    "_change_type",
    "_commit_version",
    "_commit_timestamp",
}

FORBIDDEN_SOURCE_MUTATION_PATTERNS = {
    "ALTER TABLE",
    "SET TBLPROPERTIES",
    "enableChangeDataFeed",
    "delta.enableChangeDataFeed = true",
    "CREATE TABLE",
    "DROP TABLE",
    "DELETE FROM",
    "UPDATE ",
    "MERGE INTO",
    ".saveAsTable",
    ".insertInto",
    ".toTable",
    ".format(\"delta\")",
    ".format('delta')",
}


def _read_notebook() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_stage_30b_readstream_poc_notebook_exists_and_is_safe_gated():
    source = _read_notebook()

    assert NOTEBOOK_PATH.exists()
    assert "Stage 30B-4b CDF readStream dirty-key detection POC" in source
    assert "This notebook must not mutate production tables or workspace resources." in source
    assert "RUN_STREAM = False" in source
    assert "RUN_BATCH_CDF = False" in source
    assert "if not RUN_STREAM and not RUN_BATCH_CDF:" in source
    assert "RUN_CDF_DIAGNOSTICS_FALSE" in source


def test_stage_30b_readstream_poc_uses_cdf_available_now_and_checkpoint_path():
    source = _read_notebook()

    assert "spark.readStream" in source
    assert "spark.read" in source
    assert ".option(\"readChangeFeed\", \"true\")" in source
    assert "trigger(availableNow=True)" in source
    assert "STREAM_TRIGGER_AVAILABLE_NOW = True" in source
    assert "CHECKPOINT_BASE_PATH = \"\"" in source
    assert ".option(\"checkpointLocation\", checkpoint_path)" in source
    assert "CHECKPOINT_BASE_PATH must be set for RUN_STREAM mode" in source
    assert "starts with /Volumes/" in source


def test_stage_30b_readstream_poc_references_required_sources_columns_and_aliases():
    source = _read_notebook()

    for table_name in SOURCE_TABLES:
        assert table_name in source

    for column_name in REQUIRED_COLUMNS | CDF_COLUMNS:
        assert column_name in source

    assert "\"leg\"" in source
    assert "\"leg_times\"" in source
    assert "source_alias" in source


def test_stage_30b_readstream_poc_uses_memory_sink_only_for_streaming_output():
    source = _read_notebook()

    assert "raw_stream_df.writeStream.format(\"memory\")" in source
    assert ".outputMode(\"append\")" in source

    write_stream_format_calls = set(
        re.findall(r"\.writeStream\.format\(([^)]+)\)", source)
    )
    assert write_stream_format_calls == {"\"memory\""}


def test_stage_30b_readstream_poc_emits_raw_stream_rows_without_stream_aggregation():
    source = _read_notebook()

    raw_select_pos = source.index("def _select_raw_dirty_candidates")
    write_stream_pos = source.index("raw_stream_df.writeStream")
    writer_start_pos = source.index("query = writer.start()")
    memory_read_pos = source.index("memory_df = spark.table(query_name)")

    assert raw_select_pos < write_stream_pos < memory_read_pos
    assert ".select(" in source[raw_select_pos:write_stream_pos]
    assert ".agg(" not in source[raw_select_pos:writer_start_pos]
    assert ".groupBy(" not in source[raw_select_pos:writer_start_pos]
    assert "_summarize_static_cdf_rows(memory_df, source_alias, \"stream\")" in source


def test_stage_30b_readstream_poc_contains_no_source_mutation_or_production_write_patterns():
    source = _read_notebook()

    for pattern in FORBIDDEN_SOURCE_MUTATION_PATTERNS:
        assert pattern not in source

    assert ".foreachBatch(" not in source
    assert "foreach_batch" not in source
    assert "feature_store.py" not in source
    assert "src.pipeline.feature_store" not in source
    assert "pipeline.feature_store" not in source
