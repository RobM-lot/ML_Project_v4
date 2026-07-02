import importlib.util
import re
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30c_taxi_out_watermark.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "19_stage30c5_taxi_out_watermark_advance.py"
BOOTSTRAP_PREFLIGHT_PATH = (
    REPO_ROOT / "notebooks" / "20_stage30c5b_taxi_out_watermark_bootstrap_preflight.py"
)
DOC_PATH = REPO_ROOT / "docs" / "stage_30c5_taxi_out_watermark_advance.md"
RUNBOOK_PATH = REPO_ROOT / "docs" / "stage_30c2_taxi_out_job_runbook.md"
JOBIZATION_PATH = REPO_ROOT / "docs" / "stage_30c4_taxi_out_shadow_jobization.md"

SOURCE_TABLES = {
    "panda_silver_prod.occ_ops.netline___schedops__leg",
    "panda_silver_prod.occ_ops.netline___schedops__leg_times",
}
CURRENT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
SHADOW_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

_SPEC = importlib.util.spec_from_file_location("stage30c_taxi_out_watermark", MODULE_PATH)
assert _SPEC is not None
watermark = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = watermark
_SPEC.loader.exec_module(watermark)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


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


def test_stage_30c5_files_exist():
    assert MODULE_PATH.exists()
    assert NOTEBOOK_PATH.exists()
    assert BOOTSTRAP_PREFLIGHT_PATH.exists()
    assert DOC_PATH.exists()


def test_stage_30c5_notebook_defaults_are_safe():
    source = _read(NOTEBOOK_PATH)

    assert "RUN_WATERMARK_ADVANCE = False" in source
    assert "DRY_RUN_ONLY = True" in source
    assert "ALLOW_SHADOW_MERGE = False" in source
    assert "ALLOW_WATERMARK_ADVANCE = False" in source
    assert "ALLOW_WATERMARK_SCHEMA_MIGRATION = False" in source
    assert "ALLOW_WATERMARK_BOOTSTRAP = False" in source
    assert 'WRITE_CONFIRMATION = ""' in source
    assert 'REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"' in source
    assert "RUN_WATERMARK_ADVANCE_FALSE" in source
    assert "exiting before reads or writes" in source


def test_watermark_schema_requires_explicit_column_names():
    assert watermark.validate_watermark_schema(watermark.REQUIRED_WATERMARK_COLUMNS)

    with pytest.raises(ValueError, match="Do not silently infer alternate column names"):
        watermark.validate_watermark_schema(
            [
                "source_alias",
                "last_processed_version",
                "updated_at",
            ]
        )


def test_watermark_schema_migration_allows_only_additive_metadata_columns():
    missing = watermark.detect_missing_watermark_columns(
        [
            "source_alias",
            "last_processed_version",
            "last_processed_timestamp",
            "updated_at",
        ]
    )
    assert missing == ("updated_by_stage", "run_id")

    assert watermark.validate_watermark_schema_migration_gates(
        table_name=WATERMARK_TABLE,
        missing_columns=missing,
        allow_schema_migration=True,
        dry_run_only=False,
        write_confirmation=watermark.REQUIRED_WRITE_CONFIRMATION,
    )

    sql = watermark.build_watermark_schema_migration_sql(
        table_name=WATERMARK_TABLE,
        missing_columns=missing,
    )
    assert "ALTER TABLE `panda_silver_dev`.`ml_ops`.`stage30c_taxi_out_watermarks`" in sql
    assert "ADD COLUMNS" in sql
    assert "`updated_by_stage` STRING" in sql
    assert "`run_id` STRING" in sql
    assert CURRENT_MV not in sql
    for source_table in SOURCE_TABLES:
        assert source_table not in sql


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"allow_schema_migration": False}, "ALLOW_WATERMARK_SCHEMA_MIGRATION"),
        ({"dry_run_only": True}, "DRY_RUN_ONLY=False"),
        ({"write_confirmation": ""}, "confirmation"),
    ],
)
def test_watermark_schema_migration_requires_explicit_write_gates(kwargs, message):
    params = {
        "table_name": WATERMARK_TABLE,
        "missing_columns": ("updated_by_stage", "run_id"),
        "allow_schema_migration": True,
        "dry_run_only": False,
        "write_confirmation": watermark.REQUIRED_WRITE_CONFIRMATION,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        watermark.validate_watermark_schema_migration_gates(**params)


def test_watermark_schema_migration_blocks_core_or_unknown_columns():
    with pytest.raises(ValueError, match="core watermark columns"):
        watermark.validate_watermark_schema_migration_gates(
            table_name=WATERMARK_TABLE,
            missing_columns=("source_alias", "updated_by_stage"),
            allow_schema_migration=True,
            dry_run_only=False,
            write_confirmation=watermark.REQUIRED_WRITE_CONFIRMATION,
        )

    with pytest.raises(ValueError, match="only additive metadata columns"):
        watermark.validate_watermark_schema_migration_gates(
            table_name=WATERMARK_TABLE,
            missing_columns=("unexpected_column",),
            allow_schema_migration=True,
            dry_run_only=False,
            write_confirmation=watermark.REQUIRED_WRITE_CONFIRMATION,
        )

    with pytest.raises(ValueError, match="dev control table"):
        watermark.build_watermark_schema_migration_sql(
            table_name=CURRENT_MV,
            missing_columns=("updated_by_stage",),
        )


def test_notebook_revalidates_schema_after_optional_migration_before_continuing():
    source = _read(NOTEBOOK_PATH)

    migration_pos = source.index("missing_watermark_columns = detect_missing_watermark_columns")
    sql_pos = source.index("spark.sql(migration_sql)")
    reread_pos = source.index("watermark_df = spark.table(WATERMARK_TABLE)", sql_pos)
    validate_pos = source.index("validate_watermark_schema(watermark_df.columns)", reread_pos)
    rows_pos = source.index("watermark_rows = [row.asDict()")
    latest_pos = source.index("latest_available_versions = {")

    assert migration_pos < sql_pos < reread_pos < validate_pos < rows_pos < latest_pos


def test_missing_watermark_rows_trigger_bootstrap_required_status():
    assert (
        watermark.classify_watermark_run_status(
            watermark_rows_present=False,
            any_new_versions=False,
            all_windows_contiguous=False,
        )
        == watermark.WATERMARK_BOOTSTRAP_REQUIRED
    )

    with pytest.raises(ValueError, match="Missing watermark rows"):
        watermark.validate_watermark_rows(
            [{"source_alias": "leg", "last_processed_version": 34620}],
            require_all_sources=True,
        )


def test_watermark_mode_builds_source_specific_next_windows():
    rows = {
        "leg": {"source_alias": "leg", "last_processed_version": 34620},
        "leg_times": {"source_alias": "leg_times", "last_processed_version": 34538},
    }
    windows = watermark.build_next_source_windows_from_watermarks(
        rows,
        {"leg": 34700, "leg_times": 34618},
        max_cdf_version_span_per_source=50,
    )

    assert windows["leg"].starting_version == 34621
    assert windows["leg"].ending_version == 34670
    assert windows["leg_times"].starting_version == 34539
    assert windows["leg_times"].ending_version == 34588


def test_explicit_window_must_start_at_previous_watermark_plus_one_by_default():
    assert watermark.validate_explicit_window_against_watermark(
        source_alias="leg",
        previous_watermark_version=34620,
        starting_version=34621,
        ending_version=34630,
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        watermark.validate_explicit_window_against_watermark(
            source_alias="leg",
            previous_watermark_version=34620,
            starting_version=34680,
            ending_version=34700,
        )


def test_non_contiguous_30c4_validation_samples_are_blocked_from_advance():
    windows = (
        watermark.SourceWindow(
            source_alias="leg",
            starting_version=34600,
            ending_version=34620,
            previous_watermark_version=34599,
        ),
        watermark.SourceWindow(
            source_alias="leg",
            starting_version=34680,
            ending_version=34700,
            previous_watermark_version=34679,
        ),
    )

    assert watermark.detect_non_contiguous_validation_sample_windows(windows)
    with pytest.raises(ValueError, match="non-contiguous"):
        watermark.validate_contiguous_source_window(
            source_alias="leg",
            previous_watermark_version=34620,
            starting_version=34680,
            ending_version=34700,
        )


def test_source_specific_advance_rows_include_both_sources_and_no_global_watermark():
    windows = {
        "leg": watermark.SourceWindow("leg", 34621, 34670, 34620),
        "leg_times": watermark.SourceWindow("leg_times", 34539, 34588, 34538),
    }
    rows = watermark.build_watermark_advance_rows(
        source_windows=windows,
        processed_timestamps_by_alias={
            "leg": "2026-07-01T01:00:00",
            "leg_times": "2026-07-01T01:05:00",
        },
        run_id="run-123",
    )

    assert {row["source_alias"] for row in rows} == {"leg", "leg_times"}
    assert {row["last_processed_version"] for row in rows} == {34670, 34588}
    module_source = _read(MODULE_PATH)
    assert "GLOBAL_WATERMARK" not in module_source


def test_bootstrap_candidate_uses_source_version_at_or_before_shadow_baseline():
    shadow_history = [
        {"version": 0, "timestamp": "2026-07-01 10:00:00", "operation": "CREATE TABLE"},
        {"version": 1, "timestamp": "2026-07-01 11:00:00", "operation": "WRITE"},
    ]
    source_history = [
        {"version": 99, "timestamp": "2026-07-01 09:59:00", "operation": "WRITE"},
        {"version": 100, "timestamp": "2026-07-01 10:00:00", "operation": "WRITE"},
        {"version": 101, "timestamp": "2026-07-01 10:01:00", "operation": "WRITE"},
    ]

    baseline = watermark.earliest_delta_history_entry(shadow_history)
    latest_shadow = watermark.latest_delta_history_entry(shadow_history)
    candidate = watermark.build_bootstrap_version_candidate(
        source_alias="leg",
        source_history_rows=source_history,
        shadow_baseline_timestamp=baseline["timestamp"],
    )

    assert baseline["version"] == 0
    assert latest_shadow["version"] == 1
    assert candidate.candidate_version == 100
    assert candidate.candidate_timestamp == "2026-07-01 10:00:00"
    assert candidate.status == watermark.BOOTSTRAP_PREFLIGHT_CANDIDATE_ONLY


def test_bootstrap_candidate_is_missing_when_no_source_history_precedes_baseline():
    candidate = watermark.build_bootstrap_version_candidate(
        source_alias="leg_times",
        source_history_rows=[
            {"version": 1, "timestamp": "2026-07-01 10:01:00", "operation": "WRITE"},
        ],
        shadow_baseline_timestamp="2026-07-01 10:00:00",
    )

    assert candidate.candidate_version is None
    assert candidate.status == watermark.BOOTSTRAP_PREFLIGHT_MISSING_CANDIDATE


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shadow_merge_executed": False}, "shadow merge did not run"),
        ({"post_merge_validation_ok": False}, "post-merge validation failed"),
        ({"candidate_duplicate_key_count": 1}, "duplicate candidate"),
        ({"candidate_null_key_count": 1}, "null candidate"),
        ({"shadow_duplicate_key_count": 1}, "duplicate shadow"),
        ({"shadow_null_key_count": 1}, "null shadow"),
        ({"candidate_current_mismatch_count": 1}, "candidate/current mismatches"),
        ({"candidate_shadow_mismatch_count": 1}, "candidate/shadow mismatches"),
        ({"write_confirmation": ""}, "confirmation"),
        ({"windows_contiguous": False}, "non-contiguous"),
        ({"source_versions_present": False}, "source-specific versions"),
    ],
)
def test_watermark_advance_gates_block_failed_preconditions(kwargs, message):
    params = {
        "shadow_merge_executed": True,
        "post_merge_validation_ok": True,
        "write_confirmation": watermark.REQUIRED_WRITE_CONFIRMATION,
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        watermark.validate_watermark_advance_gates(**params)


def test_watermark_advance_merge_sql_targets_only_dev_control_table():
    rows = (
        {
            "source_alias": "leg",
            "last_processed_version": 34670,
            "last_processed_timestamp": "2026-07-01T01:00:00",
            "updated_by_stage": watermark.DEFAULT_STAGE_NAME,
            "run_id": "run-123",
        },
        {
            "source_alias": "leg_times",
            "last_processed_version": 34588,
            "last_processed_timestamp": "2026-07-01T01:05:00",
            "updated_by_stage": watermark.DEFAULT_STAGE_NAME,
            "run_id": "run-123",
        },
    )

    sql = watermark.build_watermark_advance_merge_sql(
        watermark_table=WATERMARK_TABLE,
        advance_rows=rows,
    )

    assert "MERGE INTO `panda_silver_dev`.`ml_ops`.`stage30c_taxi_out_watermarks`" in sql
    assert "'leg' AS source_alias" in sql
    assert "'leg_times' AS source_alias" in sql
    assert "updated_by_stage" in sql
    assert "run_id" in sql
    for source_table in SOURCE_TABLES:
        assert source_table not in sql
    assert CURRENT_MV not in sql


def test_stage_30c5_notebook_has_no_stream_or_forbidden_mutation_paths():
    source = _read(NOTEBOOK_PATH)

    assert "readStream" not in source
    assert "writeStream" not in source
    assert "foreachBatch" not in source
    assert "foreach_batch" not in source
    assert "resources/pipeline.yml" not in source
    assert "databricks.yml" not in source
    assert "src/pipeline/feature_store.py" not in source

    for source_table in SOURCE_TABLES:
        assert source_table in source
        assert not _has_write_target(source, source_table)
    assert CURRENT_MV in source
    assert not _has_write_target(source, CURRENT_MV)
    assert SHADOW_TABLE in source
    assert WATERMARK_TABLE in source


def test_stage_30c5b_bootstrap_preflight_is_read_only_candidate_output():
    source = _read(BOOTSTRAP_PREFLIGHT_PATH)

    assert "RUN_BOOTSTRAP_PREFLIGHT = False" in source
    assert "ALLOW_WATERMARK_BOOTSTRAP = False" in source
    assert "DRY_RUN_ONLY = True" in source
    assert "candidate_only_requires_human_confirmation" in source
    assert "CANDIDATE ONLY - requires human confirmation." in source
    assert "shadow_earliest_history_operation" in source
    assert "shadow_latest_history_operation" in source
    assert "candidate_bootstrap_versions" in source
    assert "current_watermark_rows" in source
    assert "current_watermark_schema" in source
    assert "Only after confirming these baseline versions" in source

    for forbidden in (
        "MERGE INTO",
        "INSERT INTO",
        "ALTER TABLE",
        "CREATE TABLE",
        "DROP TABLE",
        "DELETE FROM",
        "UPDATE ",
        ".write",
        "saveAsTable",
        "insertInto",
        ".toTable",
    ):
        assert forbidden not in source
    assert "readStream" not in source
    assert "writeStream" not in source
    assert "foreachBatch" not in source
    assert "foreach_batch" not in source
    assert "ALLOW_WATERMARK_ADVANCE" not in source
    assert CURRENT_MV not in source
    for source_table in SOURCE_TABLES:
        assert source_table in source
        assert not _has_write_target(source, source_table)
    assert SHADOW_TABLE in source
    assert WATERMARK_TABLE in source
    assert not _has_write_target(source, SHADOW_TABLE)
    assert not _has_write_target(source, WATERMARK_TABLE)


def test_stage_30c5b_bootstrap_preflight_does_not_use_validation_windows_as_baseline():
    source = _read(BOOTSTRAP_PREFLIGHT_PATH)

    assert "Stage 30C-4" not in source
    for validation_window_version in ("34600", "34620", "34680", "34700"):
        assert validation_window_version not in source


def test_stage_30c5_docs_cover_watermark_safety_and_bootstrap():
    doc = _read(DOC_PATH)
    runbook = _read(RUNBOOK_PATH)
    jobization = _read(JOBIZATION_PATH)

    for phrase in (
        "Watermark Meaning",
        "all CDF changes for that source up to",
        "Stage 30C-4 validation windows",
        "non-contiguous samples",
        "source-specific",
        "Bootstrap Requirement",
        "Earlier Stage 30C-1 dev watermark tables",
        "ALLOW_WATERMARK_SCHEMA_MIGRATION = False",
        "additive schema migration path",
        "watermark_bootstrap_required",
        "do not infer baseline versions from validation windows",
        "contiguous",
        "shadow merge did not run",
        "post-merge validation failed",
        "Rollback And Rerun",
        "dev-shadow only",
        "EMA remains deferred",
    ):
        assert phrase in doc

    assert "Stage 30C-5 Handoff" in runbook
    assert "complete contiguous source window" in runbook
    assert "Stage 30C-5 Handoff" in jobization
    assert "must not advance source-specific watermarks" in jobization
