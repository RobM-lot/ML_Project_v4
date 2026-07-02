import importlib.util
import json
import re
import sys
from datetime import date
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30c_taxi_out_job.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "18_stage30c4_taxi_out_shadow_job.py"
DOC_PATH = REPO_ROOT / "docs" / "stage_30c4_taxi_out_shadow_jobization.md"
RUNBOOK_PATH = REPO_ROOT / "docs" / "stage_30c2_taxi_out_job_runbook.md"

SOURCE_TABLES = {
    "panda_silver_prod.occ_ops.netline___schedops__leg",
    "panda_silver_prod.occ_ops.netline___schedops__leg_times",
}
CURRENT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
SHADOW_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

_SPEC = importlib.util.spec_from_file_location("stage30c_taxi_out_job", MODULE_PATH)
assert _SPEC is not None
job = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = job
_SPEC.loader.exec_module(job)


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


def test_stage_30c4_files_exist():
    assert MODULE_PATH.exists()
    assert NOTEBOOK_PATH.exists()
    assert DOC_PATH.exists()


def test_stage_30c4_notebook_defaults_are_safe():
    source = _read(NOTEBOOK_PATH)

    assert "RUN_JOB = False" in source
    assert 'JOB_MODE = "validation"' in source
    assert 'SOURCE_VERSION_MODE = "explicit"' in source
    assert "DRY_RUN_ONLY = True" in source
    assert "ALLOW_SHADOW_MERGE = False" in source
    assert "ALLOW_WATERMARK_ADVANCE = False" in source
    assert 'WRITE_CONFIRMATION = ""' in source
    assert 'REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"' in source
    assert "RUN_JOB_FALSE" in source
    assert "exiting before reads or writes" in source


def test_stage_30c4_multi_window_defaults_and_json_override_exist():
    source = _read(NOTEBOOK_PATH)

    for window_id in ("A_KRK", "B_WAW", "C_MULTI_OVERLAP_AWARE"):
        assert f'"window_id": "{window_id}"' in source

    for key in ("leg_start", "leg_end", "leg_times_start", "leg_times_end"):
        assert key in source

    assert "VALIDATION_WINDOWS_JSON = \"\"" in source
    assert "parse_validation_windows(VALIDATION_WINDOWS, VALIDATION_WINDOWS_JSON)" in source
    assert "GLOBAL_CDF" not in source

    windows = job.parse_validation_windows(None)
    assert [window.window_id for window in windows] == ["A_KRK", "B_WAW", "C_MULTI_OVERLAP_AWARE"]

    override = json.dumps(
        [
            {
                "window_id": "OVERRIDE",
                "leg_start": 1,
                "leg_end": 2,
                "leg_times_start": 10,
                "leg_times_end": 11,
                "entity_filter": "WAW",
            }
        ]
    )
    parsed = job.parse_validation_windows(None, override)
    assert len(parsed) == 1
    assert parsed[0].window_id == "OVERRIDE"
    assert parsed[0].configured_source_aliases == ("leg", "leg_times")


def test_stage_30c4_window_config_rejects_global_cdf_only():
    with pytest.raises(ValueError, match="source-specific CDF"):
        job.validate_validation_window_config(
            {
                "window_id": "BAD",
                "starting_version": 1,
                "ending_version": 2,
            }
        )


def test_stage_30c4_overlap_tagging_and_entity_selection_prioritizes_real_overlap():
    affected = [
        {"dep_ap_sched": "AAA", "event_date": date(2026, 7, 1)},
        {"dep_ap_sched": "BBB", "event_date": date(2026, 7, 1)},
        {"dep_ap_sched": "CCC", "event_date": date(2026, 7, 1)},
    ]
    candidate = [
        {"dep_ap_sched": "AAA", "event_date": date(2026, 7, 1)},
        {"dep_ap_sched": "BBB", "event_date": date(2026, 7, 1)},
    ]
    current = [
        {"dep_ap_sched": "BBB", "event_date": date(2026, 7, 1)},
        {"dep_ap_sched": "CCC", "event_date": date(2026, 7, 1)},
    ]

    tagged = job.tag_validation_overlap(affected, candidate, current)
    tagged_by_entity = {row["dep_ap_sched"]: row for row in tagged}

    assert tagged_by_entity["AAA"]["has_candidate"] is True
    assert tagged_by_entity["AAA"]["has_current_mv_key"] is False
    assert tagged_by_entity["AAA"]["has_validation_overlap"] is False
    assert tagged_by_entity["BBB"]["has_validation_overlap"] is True
    assert tagged_by_entity["CCC"]["has_candidate"] is False

    assert job.select_overlap_aware_entities(tagged, max_entities=1) == ("BBB",)


def test_stage_30c4_no_overlap_window_does_not_count_as_parity_pass():
    result = job.summarize_window_result(
        window_id="C_MULTI_OVERLAP_AWARE",
        selected_entities=("WAW", "KRK"),
        dirty_events_count=10,
        affected_pairs_count=60,
        candidate_rows_count=0,
        current_scoped_rows_count=0,
        candidate_current_status_counts={},
        require_validation_overlap=True,
    )

    assert result["validation_status"] == job.STATUS_VALIDATION_NO_OVERLAP
    assert result["matched_count"] == 0

    summary = job.summarize_job_result([result])
    assert summary["number_of_parity_pass_windows"] == 0
    assert summary["overall_pass"] is False


def test_stage_30c4_overall_pass_requires_two_parity_windows_and_two_entities():
    one_window = job.summarize_window_result(
        window_id="A_KRK",
        selected_entities=("KRK",),
        candidate_rows_count=3,
        current_scoped_rows_count=3,
        candidate_current_status_counts={"matched": 3},
    )
    two_windows = [
        one_window,
        job.summarize_window_result(
            window_id="B_WAW",
            selected_entities=("WAW",),
            candidate_rows_count=1,
            current_scoped_rows_count=1,
            candidate_current_status_counts={"matched": 1},
        ),
    ]

    assert job.summarize_job_result([one_window])["overall_pass"] is False
    summary = job.summarize_job_result(two_windows)
    assert summary["parity_windows_required_met"] is True
    assert summary["entities_required_met"] is True
    assert summary["overall_pass"] is True


def test_stage_30c4_validation_failed_when_mismatch_or_duplicate_keys_exist():
    result = job.summarize_window_result(
        window_id="BAD",
        selected_entities=("WAW",),
        candidate_rows_count=2,
        current_scoped_rows_count=2,
        candidate_current_status_counts={"matched": 1, "value_mismatch": 1},
        candidate_duplicate_key_count=1,
    )

    assert result["validation_status"] == job.STATUS_VALIDATION_FAILED
    summary = job.summarize_job_result([result], require_at_least_two_parity_windows=False)
    assert summary["current_overlap_compare_ok_all"] is False
    assert summary["candidate_keys_unique_all"] is False
    assert summary["overall_pass"] is False


def test_stage_30c4_write_gates_block_unsafe_modes():
    assert job.validate_job_write_gates(
        job_mode="validation",
        dry_run_only=True,
        allow_shadow_merge=False,
        allow_watermark_advance=False,
        write_confirmation="",
    )

    with pytest.raises(ValueError, match="validation mode"):
        job.validate_job_write_gates(
            job_mode="validation",
            dry_run_only=False,
            allow_shadow_merge=True,
            allow_watermark_advance=False,
            write_confirmation=job.REQUIRED_WRITE_CONFIRMATION,
        )

    with pytest.raises(ValueError, match="DRY_RUN_ONLY"):
        job.validate_job_write_gates(
            job_mode="shadow_merge",
            dry_run_only=True,
            allow_shadow_merge=True,
            allow_watermark_advance=False,
            write_confirmation=job.REQUIRED_WRITE_CONFIRMATION,
        )

    with pytest.raises(ValueError, match="confirmation"):
        job.validate_job_write_gates(
            job_mode="watermark_advance",
            dry_run_only=False,
            allow_shadow_merge=True,
            allow_watermark_advance=True,
            write_confirmation="",
        )


def test_stage_30c4_watermark_success_requires_versions_and_validation_success():
    good_windows = [
        job.summarize_window_result(
            window_id="A_KRK",
            selected_entities=("KRK",),
            candidate_rows_count=3,
            current_scoped_rows_count=3,
            candidate_current_status_counts={"matched": 3},
            shadow_merge_executed=True,
            candidate_shadow_status_counts={"matched": 3},
        ),
        job.summarize_window_result(
            window_id="B_WAW",
            selected_entities=("WAW",),
            candidate_rows_count=1,
            current_scoped_rows_count=1,
            candidate_current_status_counts={"matched": 1},
            shadow_merge_executed=True,
            candidate_shadow_status_counts={"matched": 1},
        ),
    ]
    summary = job.summarize_job_result(good_windows)

    assert job.validate_job_success_preconditions(
        job_summary=summary,
        job_mode="watermark_advance",
        allow_watermark_advance=True,
        processed_versions_by_alias={"leg": 34700, "leg_times": 34618},
        write_confirmation=job.REQUIRED_WRITE_CONFIRMATION,
    )

    with pytest.raises(ValueError, match="source-specific latest versions"):
        job.validate_job_success_preconditions(
            job_summary=summary,
            job_mode="watermark_advance",
            allow_watermark_advance=True,
            processed_versions_by_alias={"leg": 34700},
            write_confirmation=job.REQUIRED_WRITE_CONFIRMATION,
        )


def test_stage_30c4_notebook_has_final_boolean_summary_fields():
    source = _read(NOTEBOOK_PATH)

    for field in (
        "run_job",
        "job_mode",
        "source_version_mode_explicit",
        "read_cdf_all_windows_ok",
        "dirty_events_any_window",
        "affected_pairs_any_window",
        "overlap_aware_selection_enabled",
        "parity_windows_required_met",
        "entities_required_met",
        "candidate_keys_unique_all",
        "candidate_keys_non_null_all",
        "current_overlap_compare_ok_all",
        "shadow_table_name_safe",
        "watermark_table_name_safe",
        "shadow_merge_executed_any",
        "shadow_post_merge_validation_ok_all",
        "watermarks_advanced",
        "read_only_or_dev_shadow_only",
        "overall_pass",
    ):
        assert field in source


def test_stage_30c4_has_no_stream_paths_or_global_cdf_version():
    source = "\n".join((_read(MODULE_PATH), _read(NOTEBOOK_PATH)))

    assert "readStream" not in source
    assert "writeStream" not in source
    assert "foreachBatch" not in source
    assert "foreach_batch" not in source
    assert "GLOBAL_CDF" not in source
    assert "global_cdf_version" in source


def test_stage_30c4_current_mv_and_sources_are_read_only_targets():
    source = "\n".join((_read(MODULE_PATH), _read(NOTEBOOK_PATH)))

    for table_name in SOURCE_TABLES:
        assert table_name in source
        assert not _has_write_target(source, table_name)

    assert CURRENT_MV in source
    assert not _has_write_target(source, CURRENT_MV)
    assert SHADOW_TABLE in source
    assert WATERMARK_TABLE in source


def test_stage_30c4_docs_capture_jobization_and_run_c_lesson():
    doc = _read(DOC_PATH)
    runbook = _read(RUNBOOK_PATH)

    for phrase in (
        "Databricks Job / Workflow",
        "not for the Lakeflow pipeline definition",
        "batch CDF polling",
        "overlap-aware validation",
        "shadow-first",
        "materialized view remains untouched",
        "Watermark advancement remains disabled by default",
        "Watermarks advance only after full success",
        "Run C lesson",
        "caps without overlap can create a false validation pass",
        "has_validation_overlap",
    ):
        assert phrase in doc

    assert "Stage 30C-4 Handoff" in runbook
    assert "VALIDATION_WINDOWS_JSON" in runbook
    assert "overlap-aware validation" in runbook
