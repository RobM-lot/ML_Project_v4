import importlib.util
import sys
from datetime import date
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "ml_project" / "stage30b_dirty_keys.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("stage30b_dirty_keys_standalone", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


dk = _load_module()


def _read_module_source() -> str:
    return MODULE_PATH.read_text(encoding="utf-8")


def test_source_specs_cover_stage_30b_candidate_sources():
    table_names = {spec.table_name for spec in dk.SOURCE_SPECS.values()}

    assert "netline___schedops__leg" in table_names
    assert "netline___schedops__leg_times" in table_names
    assert "netline___schedops__leg_misc" in table_names


def test_taxi_out_poc_uses_leg_and_leg_times_not_leg_misc():
    assert dk.TAXI_OUT_POC_SOURCE_ALIASES == ("leg", "leg_times")
    assert "leg_misc" not in dk.TAXI_OUT_POC_SOURCE_ALIASES
    assert dk.SOURCE_SPECS["leg"].required_for_taxi_out_poc is True
    assert dk.SOURCE_SPECS["leg_times"].required_for_taxi_out_poc is True
    assert dk.SOURCE_SPECS["leg_misc"].required_for_taxi_out_poc is False


def test_source_specs_include_expected_candidate_columns():
    assert dk.SOURCE_SPECS["leg"].candidate_columns == (
        "update_key",
        "entry_dt",
        "__START_AT",
        "__END_AT",
    )
    assert dk.SOURCE_SPECS["leg_times"].candidate_columns == (
        "update_key",
        "__START_AT",
        "__END_AT",
    )
    assert dk.SOURCE_SPECS["leg_misc"].candidate_columns == (
        "update_key",
        "__START_AT",
        "__END_AT",
    )


def test_current_ft_leg_tables_are_not_primary_dirty_key_sources():
    assert set(dk.CURRENT_STREAM_TABLES_NOT_PRIMARY_SOURCES) == {
        "ft_leg_status",
        "ft_leg_times",
        "ft_leg_misc",
    }


def test_primary_strategy_is_update_key_with_batch_level_semantics():
    assert dk.PRIMARY_DIRTY_KEY_STRATEGY is dk.DirtyKeyStrategy.UPDATE_KEY

    notes = dk.DIRTY_KEY_STRATEGY_NOTES[dk.DirtyKeyStrategy.UPDATE_KEY]
    assert "update_key" in notes
    assert "__START_AT" in notes
    assert "monotonic" in notes
    assert "batch-level" in notes
    assert "row-level" in notes


def test_taxi_out_affected_dates_are_d_plus_1_through_d_plus_30():
    affected = dk.expand_taxi_out_affected_output_dates(date(2026, 6, 1))
    dates = [record.affected_output_date for record in affected]

    assert len(affected) == 30
    assert len(set(dates)) == 30
    assert date(2026, 6, 1) not in dates
    assert dates[0] == date(2026, 6, 2)
    assert dates[-1] == date(2026, 7, 1)


def test_taxi_out_affected_dates_mark_rolling_windows():
    affected = dk.expand_taxi_out_affected_output_dates("2026-06-01")

    seven_day_dates = {
        record.affected_output_date
        for record in affected
        if record.affects_rolling_7d
    }
    thirty_day_dates = {
        record.affected_output_date
        for record in affected
        if record.affects_rolling_30d
    }

    assert seven_day_dates == {date(2026, 6, day) for day in range(2, 9)}
    assert thirty_day_dates == {record.affected_output_date for record in affected}


def test_ema_policy_is_explicitly_deferred():
    assert dk.EMA_POLICY == "deferred"
    assert "beyond D+30" in dk.EMA_POLICY_NOTES
    assert "does not solve" in dk.EMA_POLICY_NOTES


def test_taxi_out_event_mapping_preserves_sources_as_metadata_after_entity_date_dedup():
    source = _read_module_source()

    assert '.groupBy("leg_no", TAXI_OUT_ENTITY_COL, TAXI_OUT_EVENT_DATE_COL)' in source
    assert 'F.collect_set("dirty_source_alias").alias("dirty_source_aliases")' in source
    assert 'dropDuplicates(["leg_no", TAXI_OUT_ENTITY_COL, TAXI_OUT_EVENT_DATE_COL, "dirty_source_alias"])' not in source


def test_module_contains_no_production_write_strategy_or_mutating_patterns():
    source = _read_module_source()

    for forbidden in (
        "foreachBatch",
        "MERGE INTO",
        ".write",
        ".saveAsTable",
        "CREATE TABLE",
        "ALTER TABLE",
        "INSERT INTO",
        "UPDATE ",
        "DELETE ",
        "VACUUM",
        "OPTIMIZE",
        "dirty_key_table",
    ):
        assert forbidden not in source
