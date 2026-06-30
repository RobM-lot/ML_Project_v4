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

FORBIDDEN_CDF_PATTERNS = {
    "readChangeFeed",
    "changeDataFeed",
    "table_changes",
}

FORBIDDEN_PRODUCTION_WRITE_PATTERNS = {
    ".saveAsTable",
    ".insertInto",
    ".toTable",
    ".format(\"delta\")",
    ".format('delta')",
    ".format(\"parquet\")",
    ".format('parquet')",
    ".format(\"orc\")",
    ".format('orc')",
    ".format(\"json\")",
    ".format('json')",
    ".format(\"csv\")",
    ".format('csv')",
    "CREATE TABLE",
    "ALTER TABLE",
    "DROP TABLE",
    "DELETE FROM",
    "MERGE INTO",
    "dbutils.fs.",
    "checkpointLocation",
}


def _read_notebook() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_stage_30b_readstream_poc_notebook_exists_and_is_safe_gated():
    source = _read_notebook()

    assert NOTEBOOK_PATH.exists()
    assert "Stage 30B-4 readStream dirty-key detection POC" in source
    assert "This notebook must not mutate production tables or workspace resources." in source
    assert "RUN_STREAM = False" in source
    assert "if not RUN_STREAM:" in source
    assert "RUN_STREAM_FALSE" in source


def test_stage_30b_readstream_poc_references_required_sources_columns_and_aliases():
    source = _read_notebook()

    for table_name in SOURCE_TABLES:
        assert table_name in source

    for column_name in REQUIRED_COLUMNS:
        assert column_name in source

    assert "\"leg\"" in source
    assert "\"leg_times\"" in source
    assert "source_alias" in source


def test_stage_30b_readstream_poc_uses_readstream_skip_change_commits_and_memory_sink_only():
    source = _read_notebook()

    assert "spark.readStream" in source
    assert ".option(\"skipChangeCommits\", \"true\")" in source
    assert ".writeStream.format(\"memory\")" in source

    format_calls = set(re.findall(r"\.format\(([^)]+)\)", source))
    assert format_calls == {"\"memory\""}


def test_stage_30b_readstream_poc_emits_raw_rows_without_stream_aggregation():
    source = _read_notebook()

    raw_select_pos = source.index("def _select_raw_dirty_candidates")
    write_stream_pos = source.index("raw_stream_df.writeStream")
    memory_read_pos = source.index("memory_df = spark.table(query_name)")

    assert raw_select_pos < write_stream_pos < memory_read_pos
    assert ".select(" in source[raw_select_pos:write_stream_pos]

    aggregation_positions = [
        match.start()
        for match in re.finditer(r"\.(?:agg|groupBy)\(", source)
    ]
    assert aggregation_positions
    assert all(position > memory_read_pos for position in aggregation_positions)


def test_stage_30b_readstream_poc_contains_no_cdf_or_production_write_patterns():
    source = _read_notebook()

    for pattern in FORBIDDEN_CDF_PATTERNS | FORBIDDEN_PRODUCTION_WRITE_PATTERNS:
        assert pattern not in source

    assert ".foreachBatch(" not in source
    assert "foreach_batch" not in source
    assert "feature_store.py" not in source
    assert "src.pipeline.feature_store" not in source
    assert "pipeline.feature_store" not in source
