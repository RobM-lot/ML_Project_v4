"""Testy `load_settings()` — czysta logika Pythona, bez Spark/Databricks/MLflow.

Sprawdzają, że nazwy tabel są w pełni kwalifikowane (catalog.schema.name) — czyli że
SETTINGS są jedynym miejscem budowania nazw, a kod nie hardkoduje katalogu/schematu.
"""
import importlib.util
import sys
from pathlib import Path

_MOD_NAME = "ml_project_settings_standalone"
_SETTINGS_PATH = Path(__file__).resolve().parents[1] / "src" / "ml_project" / "settings.py"
_spec = importlib.util.spec_from_file_location(_MOD_NAME, _SETTINGS_PATH)
_settings_mod = importlib.util.module_from_spec(_spec)
sys.modules[_MOD_NAME] = _settings_mod
_spec.loader.exec_module(_settings_mod)
load_settings = _settings_mod.load_settings

_QUALIFIED_TABLE_ATTRS = [
    "LABELS_TABLE",
    "LEG_MISC_TABLE",
    "SHADOW_TABLE",
    "TRAINING_DATASET_TABLE",
    "FS_TAXI_OUT_TABLE",
    "FS_AIRBORNE_TABLE",
    "FS_TAXI_IN_TABLE",
    "FS_STAND_OUT_TABLE",
    "FS_STAND_IN_TABLE",
]


def test_settings_table_names_are_qualified():
    s = load_settings("dev")
    for attr in _QUALIFIED_TABLE_ATTRS:
        name = getattr(s, attr, None)
        assert name, f"{attr} jest puste/None"
        parts = name.split(".")
        assert len(parts) == 3, f"{attr}={name!r} nie ma formatu catalog.schema.name"
        assert all(parts), f"{attr}={name!r} ma pustą część"


def test_settings_dev_silver_defaults():
    s = load_settings("dev")
    assert s.FS_TAXI_OUT_TABLE == "panda_silver_dev.ml_ops.fs_taxi_out_features"
    assert s.FS_STAND_IN_TABLE == "panda_silver_dev.ml_ops.fs_stand_in_features"


def test_settings_source_override_applies():
    s = load_settings(
        "dev",
        source_catalog_override="panda_silver_prod",
        source_schema_override="occ_ops",
    )
    assert s.LABELS_TABLE.startswith("panda_silver_prod.occ_ops.")


def test_uc_model_name_qualified():
    s = load_settings("dev")
    assert s.UC_MODEL_NAME == "panda_gold_dev.ml_ops.flight_delay_model"
    assert s.MODEL_URI == "models:/panda_gold_dev.ml_ops.flight_delay_model@champion"
