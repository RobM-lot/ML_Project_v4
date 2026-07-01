import importlib.util
import re
import sys
from datetime import date
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30c_taxi_out_shadow.py"
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "17_stage30c1_taxi_out_shadow_partial_recompute.py"
GUARDRAILS_DOC_PATH = REPO_ROOT / "docs" / "stage_30c2_taxi_out_production_guardrails.md"
RUNBOOK_DOC_PATH = REPO_ROOT / "docs" / "stage_30c2_taxi_out_job_runbook.md"
STAGE_30C1_DOC_PATH = REPO_ROOT / "docs" / "stage_30c1_taxi_out_shadow_partial_recompute.md"

CURRENT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
SOURCE_TABLES = {
    "panda_silver_prod.occ_ops.netline___schedops__leg",
    "panda_silver_prod.occ_ops.netline___schedops__leg_times",
}

_SPEC = importlib.util.spec_from_file_location("stage30c_taxi_out_shadow", MODULE_PATH)
assert _SPEC is not None
shadow = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = shadow
_SPEC.loader.exec_module(shadow)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _leg_row(
    *,
    leg_no: int = 100,
    change_type: str,
    leg_state: str,
    dep_ap_sched: str = "WAW",
    dep_sched_dt: str = "2026-06-01T10:00:00",
    leg_type: str = "J",
    counter: int = 0,
):
    return {
        "leg_no": leg_no,
        "_change_type": change_type,
        "leg_state": leg_state,
        "dep_ap_sched": dep_ap_sched,
        "dep_sched_dt": dep_sched_dt,
        "leg_type": leg_type,
        "counter": counter,
        "update_key": 123,
    }


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


def test_stage_30c2_files_exist():
    assert MODULE_PATH.exists()
    assert GUARDRAILS_DOC_PATH.exists()
    assert RUNBOOK_DOC_PATH.exists()


def test_arr_to_non_arr_marks_old_event_for_recompute_and_possible_delete():
    requirements = shadow.build_dirty_event_requirements_from_leg_cdf(
        [
            _leg_row(change_type="update_preimage", leg_state="ARR", dep_ap_sched="WAW"),
            _leg_row(change_type="update_postimage", leg_state="SKD", dep_ap_sched="WAW"),
        ]
    )

    assert len(requirements) == 1
    assert requirements[0].dep_ap_sched == "WAW"
    assert requirements[0].dirty_event_date == date(2026, 6, 1)
    assert requirements[0].dirty_sides == ("old",)
    assert requirements[0].cdf_change_types == ("update_preimage",)

    affected_pairs = shadow.expand_dirty_event_requirements_to_affected_pairs(requirements)
    assert len(affected_pairs) == 30
    merge_source = shadow.build_shadow_replace_source_rows(affected_pairs[:1], candidate_rows=[])
    assert merge_source[0][shadow.SHADOW_CANDIDATE_FLAG_COL] is False
    assert shadow.validate_shadow_merge_source(merge_source, affected_pairs=affected_pairs).delete_row_count == 1


def test_non_arr_to_arr_marks_new_event_for_recompute():
    requirements = shadow.build_dirty_event_requirements_from_leg_cdf(
        [
            _leg_row(change_type="update_preimage", leg_state="SKD", dep_ap_sched="WAW"),
            _leg_row(change_type="update_postimage", leg_state="ARR", dep_ap_sched="WAW"),
        ]
    )

    assert len(requirements) == 1
    assert requirements[0].dirty_sides == ("new",)
    assert requirements[0].cdf_change_types == ("update_postimage",)


def test_airport_move_marks_old_and_new_entities():
    requirements = shadow.build_dirty_event_requirements_from_leg_cdf(
        [
            _leg_row(change_type="update_preimage", leg_state="ARR", dep_ap_sched="WAW"),
            _leg_row(change_type="update_postimage", leg_state="ARR", dep_ap_sched="KRK"),
        ]
    )

    assert {(req.dep_ap_sched, req.dirty_event_date, req.dirty_sides) for req in requirements} == {
        ("WAW", date(2026, 6, 1), ("old",)),
        ("KRK", date(2026, 6, 1), ("new",)),
    }


def test_date_move_marks_old_and_new_event_dates():
    requirements = shadow.build_dirty_event_requirements_from_leg_cdf(
        [
            _leg_row(change_type="update_preimage", leg_state="ARR", dep_sched_dt="2026-06-01T10:00:00"),
            _leg_row(change_type="update_postimage", leg_state="ARR", dep_sched_dt="2026-06-03T10:00:00"),
        ]
    )

    assert {(req.dep_ap_sched, req.dirty_event_date, req.dirty_sides) for req in requirements} == {
        ("WAW", date(2026, 6, 1), ("old",)),
        ("WAW", date(2026, 6, 3), ("new",)),
    }


def test_insert_and_delete_change_types_are_dirty_side_inputs():
    requirements = shadow.build_dirty_event_requirements_from_leg_cdf(
        [
            _leg_row(leg_no=1, change_type="insert", leg_state="ARR", dep_ap_sched="WAW"),
            _leg_row(leg_no=2, change_type="delete", leg_state="ARR", dep_ap_sched="KRK"),
        ]
    )

    assert {(req.leg_no, req.dep_ap_sched, req.dirty_sides, req.cdf_change_types) for req in requirements} == {
        (1, "WAW", ("new",), ("insert",)),
        (2, "KRK", ("old",), ("delete",)),
    }


def test_leg_times_only_mapping_uses_current_leg_and_documents_limitation():
    requirements = shadow.build_dirty_event_requirements_from_leg_times_cdf(
        cdf_rows=[
            {"leg_no": 300, "_change_type": "update_postimage", "update_key": 700},
        ],
        current_leg_rows=[
            _leg_row(
                leg_no=300,
                change_type="update_postimage",
                leg_state="ARR",
                dep_ap_sched="WAW",
                dep_sched_dt="2026-06-04T10:00:00",
            )
        ],
    )

    assert len(requirements) == 1
    assert requirements[0].source_alias == "leg_times"
    assert requirements[0].dirty_sides == ("current",)
    assert requirements[0].limitation == shadow.LEG_TIMES_ONLY_MAPPING_LIMITATION
    assert "cannot recover an old dep_ap_sched/event_date mapping" in shadow.LEG_TIMES_ONLY_MAPPING_LIMITATION


def test_shadow_merge_source_has_candidate_flag_and_unique_keys():
    affected_pairs = [
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2)},
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 3)},
    ]
    candidate_rows = [
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2), "avg_taxi_out_7d": 12.0},
    ]

    merge_source = shadow.build_shadow_replace_source_rows(affected_pairs, candidate_rows)
    validation = shadow.validate_shadow_merge_source(merge_source, affected_pairs=affected_pairs)

    assert all(shadow.SHADOW_CANDIDATE_FLAG_COL in row for row in merge_source)
    assert validation.row_count == 2
    assert validation.key_count == 2
    assert validation.candidate_row_count == 1
    assert validation.delete_row_count == 1


def test_delete_branch_is_limited_to_affected_pairs():
    merge_source = [
        {
            "dep_ap_sched": "KRK",
            "event_date": date(2026, 6, 2),
            shadow.SHADOW_CANDIDATE_FLAG_COL: False,
        }
    ]
    affected_pairs = [{"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2)}]

    with pytest.raises(ValueError, match="outside affected pairs"):
        shadow.validate_shadow_merge_source(merge_source, affected_pairs=affected_pairs)


def test_idempotency_same_inputs_produce_same_merge_source_and_no_duplicate_keys():
    affected_pairs = [
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2)},
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2)},
        {"dep_ap_sched": "KRK", "event_date": date(2026, 6, 3)},
    ]
    candidate_rows = [{"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2), "avg_taxi_out_7d": 12.0}]

    first = shadow.build_shadow_replace_source_rows(affected_pairs, candidate_rows)
    second = shadow.build_shadow_replace_source_rows(affected_pairs, candidate_rows)
    validation = shadow.validate_shadow_merge_source(first, affected_pairs=affected_pairs)

    assert first == second
    assert validation.row_count == 2
    assert validation.key_count == 2


def test_duplicate_candidate_keys_are_rejected_before_merge_source_idempotency_breaks():
    affected_pairs = [{"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2)}]
    candidate_rows = [
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2), "avg_taxi_out_7d": 12.0},
        {"dep_ap_sched": "WAW", "event_date": date(2026, 6, 2), "avg_taxi_out_7d": 13.0},
    ]

    with pytest.raises(ValueError, match="Duplicate candidate key"):
        shadow.build_shadow_replace_source_rows(affected_pairs, candidate_rows)


def test_watermark_advancement_requires_source_specific_versions_and_validations():
    assert shadow.validate_watermark_advance_preconditions(
        shadow_merge_executed=True,
        shadow_post_merge_validation_ok=True,
        processed_versions_by_alias={"leg": 34545, "leg_times": 34465},
        write_confirmation="I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY",
        required_write_confirmation="I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY",
    )

    sqls = shadow.build_advance_watermark_sqls(
        stage_name="stage30c2",
        source_tables_by_alias={
            "leg": "panda_silver_prod.occ_ops.netline___schedops__leg",
            "leg_times": "panda_silver_prod.occ_ops.netline___schedops__leg_times",
        },
        processed_versions_by_alias={"leg": 34545, "leg_times": 34465},
        processed_timestamps_by_alias={"leg": "2026-06-30T01:00:00", "leg_times": "2026-06-30T01:05:00"},
        last_successful_run_id="run-1",
    )
    assert len(sqls) == 2
    assert "'leg' AS source_alias" in sqls[0]
    assert "'leg_times' AS source_alias" in sqls[1]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"shadow_merge_executed": False}, "shadow_merge_executed"),
        ({"shadow_post_merge_validation_ok": False}, "post-merge"),
        ({"processed_versions_by_alias": {"leg": 34545}}, "source-specific latest versions"),
        ({"candidate_duplicate_key_count": 1}, "duplicate candidate"),
        ({"shadow_duplicate_key_count": 1}, "duplicate shadow"),
        ({"shadow_null_key_count": 1}, "null shadow"),
        ({"compare_failed": True}, "failed compare"),
        ({"write_confirmation": ""}, "confirmation"),
    ],
)
def test_watermark_advancement_is_blocked_on_failed_preconditions(kwargs, message):
    params = {
        "shadow_merge_executed": True,
        "shadow_post_merge_validation_ok": True,
        "processed_versions_by_alias": {"leg": 34545, "leg_times": 34465},
        "write_confirmation": "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY",
        "required_write_confirmation": "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY",
    }
    params.update(kwargs)

    with pytest.raises(ValueError, match=message):
        shadow.validate_watermark_advance_preconditions(**params)


def test_stage_30c2_has_no_global_watermark_or_stream_paths():
    source = "\n".join((_read(MODULE_PATH), _read(NOTEBOOK_PATH)))

    assert "GLOBAL_WATERMARK" not in source
    assert "readStream" not in source
    assert "writeStream" not in source
    assert "foreachBatch" not in source
    assert "foreach_batch" not in source


def test_stage_30c2_does_not_write_to_source_tables_or_current_mv():
    source = "\n".join((_read(MODULE_PATH), _read(NOTEBOOK_PATH)))

    for table_name in SOURCE_TABLES:
        assert not _has_write_target(source, table_name)
    assert not _has_write_target(source, CURRENT_MV)
    assert CURRENT_MV in source


def test_stage_30c2_docs_capture_job_decision_guardrails_and_deferred_scope():
    guardrails = _read(GUARDRAILS_DOC_PATH)
    runbook = _read(RUNBOOK_DOC_PATH)
    stage30c1 = _read(STAGE_30C1_DOC_PATH)

    for phrase in (
        "separate Databricks Job",
        "not Lakeflow pipeline",
        "GO for the shadow-to-target pattern",
        "NO-GO for direct production target mutation",
        "Batch CDF polling",
        "ARR -> non-ARR",
        "non-ARR -> ARR",
        "airport moves",
        "date moves",
        "leg_times-only",
        "There is no global watermark",
        "EMA remains deferred",
        "Multi-window Validation Plan",
    ):
        assert phrase in guardrails

    for phrase in (
        "Run the taxi-out shadow partial recompute as a separate batch Databricks Job",
        "not inside the Lakeflow pipeline definition",
        "readStream investigation",
        "Do not advance watermarks on failure",
        "dirty leg/source count",
        "No global watermark",
    ):
        assert phrase in runbook

    assert "Stage 30C-2 Handoff" in stage30c1
    assert "current MV read-only" in stage30c1
