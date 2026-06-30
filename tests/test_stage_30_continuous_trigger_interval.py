from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DATABRICKS_PATH = REPO_ROOT / "databricks.yml"
PIPELINE_PATH = REPO_ROOT / "resources" / "pipeline.yml"
FEATURE_STORE_PATH = REPO_ROOT / "src" / "pipeline" / "feature_store.py"
TRAINING_PATH = REPO_ROOT / "src" / "ml_project" / "training.py"
SCORING_PATH = REPO_ROOT / "src" / "ml_project" / "scoring.py"
SETTINGS_PATH = REPO_ROOT / "src" / "ml_project" / "settings.py"

_SETTINGS_MOD_NAME = "ml_project_settings_stage_30_standalone"
_settings_spec = importlib.util.spec_from_file_location(_SETTINGS_MOD_NAME, SETTINGS_PATH)
_settings_mod = importlib.util.module_from_spec(_settings_spec)
sys.modules[_SETTINGS_MOD_NAME] = _settings_mod
_settings_spec.loader.exec_module(_settings_mod)
load_settings = _settings_mod.load_settings

FINAL_DAILY_TABLES = {
    "ft_airport_daily_taxi_out",
    "ft_route_daily_stats",
    "ft_airport_daily_taxi_in",
    "ft_stand_daily_out",
    "ft_stand_daily_in",
}

NON_FINAL_TABLES = {
    "enriched",
    "data_quality",
    "cleaned_flight_data_full_table",
    "ft_leg_status",
    "ft_leg_times",
    "ft_leg_misc",
    "ft_airport_timezone",
}

FINAL_FT_SETTINGS = {
    "FT_AIRPORT_DAILY_TAXI_OUT_TABLE": "panda_silver_prod.ml_ops.ft_airport_daily_taxi_out",
    "FT_ROUTE_DAILY_STATS_TABLE": "panda_silver_prod.ml_ops.ft_route_daily_stats",
    "FT_AIRPORT_DAILY_TAXI_IN_TABLE": "panda_silver_prod.ml_ops.ft_airport_daily_taxi_in",
    "FT_STAND_DAILY_OUT_TABLE": "panda_silver_prod.ml_ops.ft_stand_daily_out",
    "FT_STAND_DAILY_IN_TABLE": "panda_silver_prod.ml_ops.ft_stand_daily_in",
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(_read(path))


def _pipeline_resource_block() -> list[str]:
    lines = _read(PIPELINE_PATH).splitlines()
    start = lines.index("    pipeline_ml_feature_store:")
    block: list[str] = []
    for line in lines[start + 1:]:
        if line.startswith("    ") and not line.startswith("      ") and line.strip():
            break
        block.append(line)
    return block


def _assignments(tree: ast.Module) -> dict[str, ast.AST]:
    out: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    out[target.id] = node.value
    return out


def _fs_table_name(value: ast.AST) -> str | None:
    if not isinstance(value, ast.Call):
        return None
    if not isinstance(value.func, ast.Name) or value.func.id != "_fs_table":
        return None
    if not value.args or not isinstance(value.args[0], ast.Constant):
        return None
    table_name = value.args[0].value
    return table_name if isinstance(table_name, str) else None


def _decorated_dlt_tables() -> dict[str, dict[str, str | None]]:
    tree = ast.parse(_read(FEATURE_STORE_PATH))
    tables: dict[str, dict[str, str | None]] = {}

    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if not isinstance(decorator, ast.Call):
                continue
            if not isinstance(decorator.func, ast.Attribute):
                continue
            if decorator.func.attr not in {"materialized_view", "table"}:
                continue

            name_kw = next((kw for kw in decorator.keywords if kw.arg == "name"), None)
            props_kw = next((kw for kw in decorator.keywords if kw.arg == "table_properties"), None)
            spark_conf_kw = next((kw for kw in decorator.keywords if kw.arg == "spark_conf"), None)
            if name_kw is None or props_kw is None:
                continue

            table_name = _fs_table_name(name_kw.value)
            if table_name is None:
                continue
            if not isinstance(props_kw.value, ast.Name):
                raise AssertionError(f"{table_name} uses non-name table_properties")

            spark_conf_name = None
            if spark_conf_kw is not None:
                if not isinstance(spark_conf_kw.value, ast.Name):
                    raise AssertionError(f"{table_name} uses non-name spark_conf")
                spark_conf_name = spark_conf_kw.value.id

            tables[table_name] = {
                "decorator": decorator.func.attr,
                "function": node.name,
                "table_properties": props_kw.value.id,
                "spark_conf": spark_conf_name,
            }

    return tables


def test_pipeline_feature_store_resource_is_continuous_without_development_mode():
    block = _pipeline_resource_block()
    pipeline_text = "\n".join(block)
    pipeline = _load_yaml(PIPELINE_PATH)["resources"]["pipelines"]["pipeline_ml_feature_store"]

    assert pipeline["continuous"] is True
    assert pipeline.get("development") is not True
    assert "      serverless: true" in block
    assert "      channel: PREVIEW" in block
    assert "pipelines.trigger.interval" not in pipeline_text


def test_dev_target_overrides_pipeline_to_continuous_non_development_mode():
    config = _load_yaml(DATABRICKS_PATH)
    dev = config["targets"]["dev"]
    pipeline = dev["resources"]["pipelines"]["pipeline_ml_feature_store"]

    assert dev["mode"] == "development"
    assert dev["presets"]["pipelines_development"] is False
    assert pipeline["development"] is False
    assert pipeline["continuous"] is True


def test_dev_target_keeps_dev_catalog_schema_variables():
    dev_variables = _load_yaml(DATABRICKS_PATH)["targets"]["dev"]["variables"]

    assert dev_variables["runtime_env"] == "dev"
    assert dev_variables["source_catalog"] == "panda_silver_prod"
    assert dev_variables["source_schema"] == "occ_ops"
    assert dev_variables["silver_catalog"] == "panda_silver_dev"
    assert dev_variables["silver_schema"] == "ml_ops"
    assert dev_variables["gold_catalog"] == "panda_gold_dev"
    assert dev_variables["gold_schema"] == "ml_ops"


def test_final_daily_feature_trigger_interval_config_is_defined():
    tree = ast.parse(_read(FEATURE_STORE_PATH))
    assignments = _assignments(tree)

    assert isinstance(assignments["DLT_TABLE_PROPERTIES"], ast.Dict)
    dlt_properties = assignments["DLT_TABLE_PROPERTIES"]
    assert len(dlt_properties.keys) == 1
    assert isinstance(dlt_properties.keys[0], ast.Constant)
    assert dlt_properties.keys[0].value == "delta.enableDeletionVectors"
    assert isinstance(dlt_properties.values[0], ast.Constant)
    assert dlt_properties.values[0].value == "true"

    interval = assignments["FINAL_DAILY_FEATURE_TRIGGER_INTERVAL"]
    assert isinstance(interval, ast.Constant)
    assert interval.value == "1 hour"

    final_properties = assignments["FINAL_DAILY_FEATURE_TABLE_PROPERTIES"]
    assert isinstance(final_properties, ast.Dict)
    assert None in final_properties.keys
    assert all(
        not (isinstance(key, ast.Constant) and key.value == "pipelines.trigger.interval")
        for key in final_properties.keys
    )

    spark_conf = assignments["FINAL_DAILY_FEATURE_SPARK_CONF"]
    assert isinstance(spark_conf, ast.Dict)
    trigger_key_index = next(
        idx
        for idx, key in enumerate(spark_conf.keys)
        if isinstance(key, ast.Constant) and key.value == "pipelines.trigger.interval"
    )
    trigger_value = spark_conf.values[trigger_key_index]
    assert isinstance(trigger_value, ast.Name)
    assert trigger_value.id == "FINAL_DAILY_FEATURE_TRIGGER_INTERVAL"


def test_only_final_daily_feature_mvs_use_interval_properties():
    tables = _decorated_dlt_tables()

    assert FINAL_DAILY_TABLES <= tables.keys()
    assert NON_FINAL_TABLES <= tables.keys()

    for table_name in FINAL_DAILY_TABLES:
        assert tables[table_name]["table_properties"] == "FINAL_DAILY_FEATURE_TABLE_PROPERTIES"
        assert tables[table_name]["spark_conf"] == "FINAL_DAILY_FEATURE_SPARK_CONF"

    for table_name in NON_FINAL_TABLES:
        assert tables[table_name]["table_properties"] == "DLT_TABLE_PROPERTIES"
        assert tables[table_name]["spark_conf"] is None


def test_final_daily_feature_objects_remain_materialized_views():
    tables = _decorated_dlt_tables()

    for table_name in FINAL_DAILY_TABLES:
        assert tables[table_name]["decorator"] == "materialized_view"
        assert tables[table_name]["function"] == table_name


def test_exactly_final_daily_feature_mvs_use_trigger_spark_conf():
    tables = _decorated_dlt_tables()
    trigger_conf_functions = {
        metadata["function"]
        for metadata in tables.values()
        if metadata["spark_conf"] == "FINAL_DAILY_FEATURE_SPARK_CONF"
    }

    assert len(trigger_conf_functions) == 5
    assert trigger_conf_functions == FINAL_DAILY_TABLES


def test_feature_store_has_no_stage_30b_or_cdf_logic():
    source = _read(FEATURE_STORE_PATH)

    for forbidden in (
        "readChangeFeed",
        "foreachBatch",
        "foreach_batch_sink",
        "MERGE",
        "dirty_key",
        "dirty-key",
        "dirty key",
        "watermark",
        "partial_recompute",
        "partial recompute",
    ):
        assert forbidden not in source


def test_final_ft_settings_keep_existing_prod_table_names():
    settings = load_settings("prod")

    for attr, expected_name in FINAL_FT_SETTINGS.items():
        table_name = getattr(settings, attr)
        assert table_name == expected_name
        assert "dirty" not in table_name.lower()
        assert "helper" not in table_name.lower()


def test_training_and_scoring_do_not_reference_dirty_key_tables():
    training_source = _read(TRAINING_PATH)
    scoring_source = _read(SCORING_PATH)

    for source in (training_source, scoring_source):
        assert "dirty_key" not in source
        assert "dirty-key" not in source
        assert "_dirty" not in source
        assert "FINAL_DAILY_FEATURE_TABLE_PROPERTIES" not in source
        assert "FINAL_DAILY_FEATURE_SPARK_CONF" not in source

    for attr in FINAL_FT_SETTINGS:
        assert f"settings.{attr}" in training_source
