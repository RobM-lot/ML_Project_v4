from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = REPO_ROOT / "notebooks" / "13_stage30b_taxi_out_parity_read_only.py"

REQUIRED_TABLE_NAMES = {
    "netline___schedops__leg",
    "netline___schedops__leg_times",
    "cleaned_flight_data_full_table",
    "ft_airport_daily_taxi_out",
}

REQUIRED_HELPERS = {
    "extract_dirty_leg_keys",
    "select_current_latest",
    "map_dirty_legs_to_taxi_out_events",
    "expand_dirty_taxi_out_events_to_affected_outputs",
    "build_taxi_out_candidate_for_affected_outputs",
    "compare_taxi_out_candidate_to_current_mv",
}

SAMPLING_CONTROLS = {
    "LATEST_UPDATE_KEY_BATCHES",
    "MAX_DIRTY_LEGS",
    "MAX_AFFECTED_ENTITIES",
    "LAST_SEEN_UPDATE_KEY",
    "REQUIRE_FULL_AFFECTED_WINDOW",
}

FORBIDDEN_PATTERNS = {
    "foreachBatch",
    "MERGE INTO",
    ".write",
    ".saveAsTable",
    "CREATE TABLE",
    "CREATE VIEW",
    "CREATE OR REPLACE",
    "ALTER TABLE",
    "INSERT INTO",
    "UPDATE ",
    "DELETE ",
    "VACUUM",
    "OPTIMIZE",
    "DROP TABLE",
    "databricks bundle",
    "databricks pipelines",
    "dbutils.fs.put",
    "dbutils.fs.rm",
    "src.pipeline.feature_store",
    "pipeline.feature_store",
}


def _read_notebook() -> str:
    return NOTEBOOK_PATH.read_text(encoding="utf-8")


def test_stage_30b_parity_notebook_exists_and_has_read_only_safety_gate():
    source = _read_notebook()

    assert NOTEBOOK_PATH.exists()
    assert "Stage 30B-3 read-only parity validation" in source
    assert "This notebook must not mutate tables or workspace resources." in source
    assert "RUN_PARITY = False" in source
    assert "if not RUN_PARITY:" in source
    assert "RUN_PARITY_FALSE" in source


def test_stage_30b_parity_notebook_references_required_tables_and_helpers():
    source = _read_notebook()

    for name in REQUIRED_TABLE_NAMES | REQUIRED_HELPERS:
        assert name in source


def test_stage_30b_parity_notebook_has_sampling_controls():
    source = _read_notebook()

    for name in SAMPLING_CONTROLS:
        assert name in source

    assert "ENTITY_FILTER" in source
    assert "MAX_AFFECTED_ENTITIES" in source
    assert "REQUIRE_FULL_AFFECTED_WINDOW = True" in source


def test_stage_30b_parity_notebook_requires_full_affected_window_by_default():
    source = _read_notebook()

    assert "MAX_CURRENT_MV_EVENT_DATE" in source
    assert "current_mv.agg(F.max(DATE_COL).alias(\"max_current_mv_event_date\")).first()" in source
    assert "date_add(to_date(dep_sched_dt), 30) <= MAX_CURRENT_MV_EVENT_DATE" in source
    assert "F.date_add(F.to_date(F.col(\"dep_sched_dt\")), 30) <= F.lit(MAX_CURRENT_MV_EVENT_DATE)" in source
    assert "NO_FULL_WINDOW_ELIGIBLE_DIRTY_EVENTS" in source
    assert "selected lower bound/entity/window" in source


def test_stage_30b_parity_notebook_maps_and_filters_before_dirty_leg_cap():
    source = _read_notebook()

    dirty_legs_pos = source.index("dirty_legs = dirty_leg.unionByName")
    map_pos = source.index("mapped_dirty_events = map_dirty_legs_to_taxi_out_events")
    entity_filter_pos = source.index("if ENTITY_FILTER:")
    full_window_pos = source.index("if REQUIRE_FULL_AFFECTED_WINDOW:")
    final_cap_pos = source.index("dirty_events_limited = dirty_events_with_updates.orderBy")

    assert dirty_legs_pos < map_pos < entity_filter_pos < full_window_pos < final_cap_pos
    assert "dirty_legs_limited = dirty_legs.orderBy" not in source
    assert "dirty_legs_limited" not in source


def test_stage_30b_parity_notebook_reports_eligibility_first_sampling_counts():
    source = _read_notebook()

    expected_count_messages = {
        "dirty leg/source candidates before eligibility",
        "mapped dirty taxi-out events before eligibility",
        "mapped dirty taxi-out events after ENTITY_FILTER",
        "mapped dirty taxi-out events after full-window eligibility",
        "mapped dirty taxi-out events after cap",
    }

    for message in expected_count_messages:
        assert message in source


def test_stage_30b_parity_notebook_filters_current_mv_to_affected_pairs_before_comparison():
    source = _read_notebook()

    affected_pairs_pos = source.index("affected_pairs = affected_outputs.select")
    current_scoped_pos = source.index("current_scoped = (")
    compare_pos = source.index("compare_taxi_out_candidate_to_current_mv(candidate_scoped, current_scoped")

    assert affected_pairs_pos < current_scoped_pos < compare_pos
    assert "current_mv.select(*NON_EMA_PARITY_COLUMNS)" in source
    assert ".join(affected_pairs, on=[ENTITY_COL, DATE_COL], how=\"inner\")" in source


def test_stage_30b_parity_notebook_compares_non_ema_candidate_rows_only():
    source = _read_notebook()

    assert "NON_EMA_PARITY_COLUMNS" in source
    assert "candidate_scoped = build_taxi_out_candidate_for_affected_outputs" in source
    assert "current_mv.select(*NON_EMA_PARITY_COLUMNS)" in source


def test_stage_30b_parity_notebook_contains_no_mutating_or_deploy_patterns():
    source = _read_notebook()

    for pattern in FORBIDDEN_PATTERNS:
        assert pattern not in source
