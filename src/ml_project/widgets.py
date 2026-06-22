from __future__ import annotations

from typing import Iterable, Optional


def ensure_dropdown_widget(dbutils, name: str, default: str, choices: Iterable[str], label: str) -> None:
    try:
        dbutils.widgets.dropdown(name, default, list(choices), label)
    except Exception:
        pass


def ensure_text_widget(dbutils, name: str, default: str, label: str) -> None:
    try:
        dbutils.widgets.text(name, default, label)
    except Exception:
        pass


def get_widget_value(dbutils, name: str, default: Optional[str] = None) -> str:
    try:
        return dbutils.widgets.get(name)
    except Exception:
        if default is None:
            raise
        return default


def ensure_env_widget(dbutils, default: str = "dev") -> str:
    ensure_dropdown_widget(dbutils, "ENV", default, ["dev", "prod"], "Środowisko (ENV)")
    value = get_widget_value(dbutils, "ENV", default).strip().lower()
    if value not in {"dev", "prod"}:
        raise ValueError(f"Nieobsługiwane środowisko: {value!r}")
    return value


def get_bool_widget(dbutils, name: str, default: bool = False) -> bool:
    default_str = "True" if default else "False"
    value = get_widget_value(dbutils, name, default_str).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Widget {name!r} musi mieć wartość True/False, otrzymano: {value!r}")
    return value == "true"


def ensure_cdf_stream_widgets(dbutils) -> None:
    ensure_dropdown_widget(dbutils, "RUN_BOOTSTRAP", "True", ["True", "False"], "0. Bootstrap tables + shadow sync")
    ensure_dropdown_widget(dbutils, "RESET_CHECKPOINT", "False", ["True", "False"], "1. Reset Checkpoint (DEV ONLY)")
    ensure_text_widget(dbutils, "STARTING_VERSION", "", "2. Starting Version (zostaw puste by użyć checkpointu/now)")
