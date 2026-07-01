import importlib.util
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30c_taxi_out_shadow.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "17_stage30c1_taxi_out_shadow_partial_recompute.py"
DOC_PATH = REPO_ROOT / "docs" / "stage_30c1_taxi_out_shadow_partial_recompute.md"
HANDOFF_DOC_PATH = REPO_ROOT / "docs" / "stage_30b6_30c0_taxi_out_production_design.md"

SOURCE_TABLES = {
    "panda_silver_prod.occ_ops.netline___schedops__leg",
    "panda_silver_prod.occ_ops.netline___schedops__leg_times",
}
CURRENT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
SHADOW_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

WRITE_PATTERNS = (
    "CREATE TABLE",
    "MERGE INTO",
    ".write",
    "saveAsTable",
    "INSERT",
    "DELETE",
)

_SPEC = importlib.util.spec_from_file_location("stage30c_taxi_out_shadow", MODULE_PATH)
assert _SPEC is not None
shadow = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = shadow
_SPEC.loader.exec_module(shadow)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _combined_source() -> str:
    return "\n".join((_read(MODULE_PATH), _read(NOTEBOOK_PATH)))


def _has_write_target(source: str, table_name: str) -> bool:
    quoted = re.escape(table_name)
    backtick_quoted = re.escape("`" + "`.`".join(table_name.split(".")) + "`")
    target = f"(?:{quoted}|{backtick_quoted})"
    patterns = [
        rf"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?{target}",
        rf"MERGE\s+INTO\s+{target}",
        rf"INSERT\s+INTO\s+{target}",
        rf"DELETE\s+FROM\s+{target}",
        rf"UPDATE\s+{target}",
    ]
    return any(re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL) for pattern in patterns)


def test_stage_30c1_files_exist():
    assert MODULE_PATH.exists()
    assert NOTEBOOK_PATH.exists()
    assert DOC_PATH.exists()


def test_stage_30c1_notebook_defaults_are_safe():
    source = _read(NOTEBOOK_PATH)

    assert "RUN_SHADOW_PIPELINE = False" in source
    assert "DRY_RUN_ONLY = True" in source
    assert "ALLOW_CREATE_SHADOW_TABLE = False" in source
    assert "ALLOW_CREATE_WATERMARK_TABLE = False" in source
    assert "ALLOW_SHADOW_MERGE = False" in source
    assert "ALLOW_WATERMARK_ADVANCE = False" in source
    assert 'WRITE_CONFIRMATION = ""' in source
    assert 'REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"' in source
    assert "RUN_SHADOW_PIPELINE_FALSE" in source
    assert "No source, current MV, shadow, or control tables were read." in source


def test_stage_30c1_has_no_stream_or_foreach_batch_paths():
    source = _combined_source()

    forbidden = ("readStream", "writeStream", "foreachBatch", "foreach_batch")
    for pattern in forbidden:
        assert pattern not in source


def test_stage_30c1_does_not_target_sources_or_current_mv_for_writes():
    source = _combined_source()

    for table_name in SOURCE_TABLES:
        assert not _has_write_target(source, table_name)

    assert not _has_write_target(source, CURRENT_MV)
    assert CURRENT_MV in source
    assert SHADOW_TABLE in source
    assert WATERMARK_TABLE in source


def test_stage_30c1_write_patterns_are_gated_to_shadow_or_control_targets():
    source = _combined_source()

    assert any(pattern in source for pattern in WRITE_PATTERNS)
    assert "ALLOW_CREATE_SHADOW_TABLE = False" in source
    assert "ALLOW_CREATE_WATERMARK_TABLE = False" in source
    assert "ALLOW_SHADOW_MERGE = False" in source
    assert "ALLOW_WATERMARK_ADVANCE = False" in source
    assert "REQUIRED_WRITE_CONFIRMATION" in source
    assert "require_shadow_write_confirmation" in source
    assert "validate_dev_shadow_table_name" in source
    assert SHADOW_TABLE in source
    assert WATERMARK_TABLE in source
    assert ".write" not in source
    assert "saveAsTable" not in source


def test_stage_30c1_table_name_validation_allows_only_dev_shadow_or_watermark():
    shadow.validate_dev_shadow_table_name(SHADOW_TABLE, expected_suffix_or_token="shadow")
    shadow.validate_dev_shadow_table_name(WATERMARK_TABLE, expected_suffix_or_token="watermark")

    with pytest.raises(ValueError):
        shadow.validate_dev_shadow_table_name(CURRENT_MV, expected_suffix_or_token="shadow")
    with pytest.raises(ValueError):
        shadow.validate_dev_shadow_table_name(
            "panda_silver_prod.occ_ops.netline___schedops__leg",
            expected_suffix_or_token="shadow",
        )
    with pytest.raises(ValueError):
        shadow.validate_dev_shadow_table_name(
            "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out",
            expected_suffix_or_token="shadow",
        )


def test_stage_30c1_source_specific_cdf_windows_and_batch_cdf_are_required():
    notebook = _read(NOTEBOOK_PATH)
    module = _read(MODULE_PATH)

    assert "LEG_CDF_STARTING_VERSION = None" in notebook
    assert "LEG_CDF_ENDING_VERSION = None" in notebook
    assert "LEG_TIMES_CDF_STARTING_VERSION = None" in notebook
    assert "LEG_TIMES_CDF_ENDING_VERSION = None" in notebook
    assert "validate_source_cdf_windows" in module
    assert "There is deliberately no global CDF version input" in module
    assert "GLOBAL_CDF" not in notebook
    assert "spark.read.option(\"readChangeFeed\", \"true\")" in notebook
    assert ".option(\"startingVersion\", str(starting_version))" in notebook
    assert ".option(\"endingVersion\", str(ending_version))" in notebook


def test_stage_30c1_shadow_helpers_define_key_and_merge_source_semantics():
    module = _read(MODULE_PATH)

    assert 'TARGET_KEY_COLS = ("dep_ap_sched", "event_date")' in module
    assert "build_shadow_replace_source" in module
    assert "_stage30c_has_candidate" in module
    assert "affected_output_date" in module
    assert "left join" in module.lower() or 'how="left"' in module


def test_stage_30c1_sql_builders_use_single_merge_with_update_insert_delete():
    create_shadow_sql = shadow.build_create_shadow_table_sql(
        shadow_table=SHADOW_TABLE,
        current_mv_table=CURRENT_MV,
    )
    create_watermark_sql = shadow.build_create_watermark_table_sql(watermark_table=WATERMARK_TABLE)
    merge_sql = shadow.build_shadow_merge_sql(
        shadow_table=SHADOW_TABLE,
        merge_source_view="stage30c1_taxi_out_shadow_merge_source",
        candidate_columns=("dep_ap_sched", "event_date", "avg_taxi_out_7d"),
    )

    assert "CREATE TABLE IF NOT EXISTS `panda_silver_dev`.`ml_ops`.`stage30c_ft_airport_daily_taxi_out_shadow`" in (
        create_shadow_sql
    )
    assert "AS SELECT * FROM `panda_silver_dev`.`ml_ops`.`ft_airport_daily_taxi_out`" in create_shadow_sql
    assert "CREATE TABLE IF NOT EXISTS `panda_silver_dev`.`ml_ops`.`stage30c_taxi_out_watermarks`" in (
        create_watermark_sql
    )
    assert "MERGE INTO `panda_silver_dev`.`ml_ops`.`stage30c_ft_airport_daily_taxi_out_shadow`" in merge_sql
    assert "_stage30c_has_candidate" in merge_sql
    assert "WHEN MATCHED" in merge_sql
    assert "UPDATE SET" in merge_sql
    assert "THEN DELETE" in merge_sql
    assert "WHEN NOT MATCHED" in merge_sql
    assert "THEN INSERT" in merge_sql
    assert "ft_airport_daily_taxi_out` AS target" not in merge_sql


def test_stage_30c1_notebook_runs_expected_shadow_flow_order():
    source = _read(NOTEBOOK_PATH)

    dirty_pos = source.index("dirty_legs = dirty_legs.dropDuplicates")
    mapped_pos = source.index("mapped_dirty_events = map_dirty_legs_to_taxi_out_events")
    affected_pos = source.index("affected_outputs = expand_dirty_taxi_out_events_to_affected_outputs")
    candidate_pos = source.index("candidate_scoped = build_taxi_out_candidate_for_affected_outputs")
    merge_source_pos = source.index("shadow_merge_source = build_shadow_replace_source")
    write_plan_pos = source.index("Dry-run write plan")
    merge_pos = source.index("if ALLOW_SHADOW_MERGE:")
    watermark_pos = source.index("if ALLOW_WATERMARK_ADVANCE:")

    assert dirty_pos < mapped_pos < affected_pos < candidate_pos < merge_source_pos < write_plan_pos
    assert write_plan_pos < merge_pos < watermark_pos
    assert "D+1...D+30" in source
    assert "EMA remains deferred" in source
    assert "update_preimage" in source
    assert "update_postimage" in source
    assert "insert" in source


def test_stage_30c1_watermark_advancement_is_source_specific_and_validation_gated():
    source = _read(NOTEBOOK_PATH)
    module = _read(MODULE_PATH)

    assert "build_advance_watermark_sqls" in module
    assert "for source_alias in SOURCE_ALIASES" in module
    assert '"leg": source_table(SOURCE_LEG_TABLE)' in source
    assert '"leg_times": source_table(SOURCE_LEG_TIMES_TABLE)' in source
    assert "if not (shadow_merge_succeeded and shadow_post_merge_validation_ok):" in source
    assert "Watermark advancement requires a successful shadow merge and post-merge validation." in source
    assert source.index("shadow_post_merge_validation_ok") < source.index("if ALLOW_WATERMARK_ADVANCE:")


def test_stage_30c1_final_summary_flags_are_present():
    source = _read(NOTEBOOK_PATH)

    for field in (
        "read_cdf_leg_ok",
        "read_cdf_leg_times_ok",
        "dirty_events_available",
        "affected_pairs_available",
        "candidate_rows_available",
        "candidate_key_unique",
        "candidate_key_non_null",
        "current_overlap_compare_ok",
        "shadow_table_name_safe",
        "watermark_table_name_safe",
        "dry_run_only",
        "shadow_table_created",
        "shadow_merge_executed",
        "shadow_post_merge_validation_ok",
        "watermarks_advanced",
        "read_only_or_dev_shadow_only",
    ):
        assert field in source


def test_stage_30c1_docs_cover_shadow_design_and_limitations():
    source = _read(DOC_PATH)

    for phrase in (
        "shadow-first",
        "current MV remains a read-only reference",
        "dev-only shadow/control tables",
        "all writes are off by default",
        "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY",
        "dep_ap_sched",
        "event_date",
        "single MERGE",
        "source-specific",
        "Idempotency And Retry",
        "D+1...D+30",
        "EMA remains deferred",
        "CDF retention is a limitation",
        "Full-window parity",
        "Safe config run",
        "Dry run with explicit CDF versions",
        "Optional watermark advancement",
    ):
        assert phrase in source


def test_stage_30c1_handoff_note_added_to_30c0_design():
    source = _read(HANDOFF_DOC_PATH)

    assert "Stage 30C-1 Handoff" in source
    assert "shadow-first path" in source
    assert "materialized view still remains untouched and read-only" in source
    assert "watermark advancement remains gated" in source
