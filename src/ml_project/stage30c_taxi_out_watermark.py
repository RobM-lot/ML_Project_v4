from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping


SOURCE_ALIASES = ("leg", "leg_times")
DEFAULT_WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"
DEFAULT_STAGE_NAME = "stage30c5_taxi_out_watermark_advance"
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"

CORE_WATERMARK_COLUMNS = (
    "source_alias",
    "last_processed_version",
    "last_processed_timestamp",
    "updated_at",
)
ADDITIVE_WATERMARK_METADATA_COLUMNS = {
    "updated_by_stage": "STRING",
    "run_id": "STRING",
}
REQUIRED_WATERMARK_COLUMNS = (
    *CORE_WATERMARK_COLUMNS,
    "updated_by_stage",
    "run_id",
)

WATERMARK_READY = "watermark_ready"
WATERMARK_NOOP = "watermark_noop"
WATERMARK_BOOTSTRAP_REQUIRED = "watermark_bootstrap_required"
WATERMARK_BLOCKED_NON_CONTIGUOUS = "watermark_blocked_non_contiguous"
WATERMARK_BLOCKED_MISSING_SOURCE = "watermark_blocked_missing_source"
WATERMARK_BLOCKED_VALIDATION_FAILED = "watermark_blocked_validation_failed"
WATERMARK_SCHEMA_INCOMPATIBLE = "watermark_schema_incompatible"
WATERMARK_ADVANCE_PASS = "watermark_advance_pass"
BOOTSTRAP_PREFLIGHT_CANDIDATE_ONLY = "candidate_only_requires_human_confirmation"
BOOTSTRAP_PREFLIGHT_MISSING_CANDIDATE = "candidate_missing_requires_manual_review"
BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_VIEW = "bootstrap_preflight_blocked_source_history_view"
BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_UNAVAILABLE = "bootstrap_preflight_blocked_source_history_unavailable"
SOURCE_HISTORY_UNAVAILABLE_FOR_VIEW = "source_history_unavailable_for_view"
SOURCE_HISTORY_READY = "source_history_ready"
SOURCE_HISTORY_UNAVAILABLE = "source_history_unavailable"
UC_OBJECT_TABLE = "table"
UC_OBJECT_VIEW = "view"
UC_OBJECT_UNKNOWN = "unknown"

__all__ = (
    "ADDITIVE_WATERMARK_METADATA_COLUMNS",
    "BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_UNAVAILABLE",
    "BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_VIEW",
    "BOOTSTRAP_PREFLIGHT_CANDIDATE_ONLY",
    "BOOTSTRAP_PREFLIGHT_MISSING_CANDIDATE",
    "BootstrapVersionCandidate",
    "CORE_WATERMARK_COLUMNS",
    "DEFAULT_STAGE_NAME",
    "DEFAULT_WATERMARK_TABLE",
    "REQUIRED_WATERMARK_COLUMNS",
    "REQUIRED_WRITE_CONFIRMATION",
    "SOURCE_ALIASES",
    "SOURCE_HISTORY_READY",
    "SOURCE_HISTORY_UNAVAILABLE",
    "SOURCE_HISTORY_UNAVAILABLE_FOR_VIEW",
    "SourceWindow",
    "UC_OBJECT_TABLE",
    "UC_OBJECT_UNKNOWN",
    "UC_OBJECT_VIEW",
    "WATERMARK_ADVANCE_PASS",
    "WATERMARK_BLOCKED_MISSING_SOURCE",
    "WATERMARK_BLOCKED_NON_CONTIGUOUS",
    "WATERMARK_BLOCKED_VALIDATION_FAILED",
    "WATERMARK_BOOTSTRAP_REQUIRED",
    "WATERMARK_NOOP",
    "WATERMARK_READY",
    "WATERMARK_SCHEMA_INCOMPATIBLE",
    "build_bootstrap_preflight_status_for_history_sources",
    "build_bootstrap_version_candidate",
    "build_next_source_windows_from_watermarks",
    "build_watermark_advance_merge_sql",
    "build_watermark_advance_rows",
    "build_watermark_schema_migration_sql",
    "classify_uc_object_type",
    "classify_watermark_run_status",
    "detect_missing_watermark_columns",
    "detect_non_contiguous_validation_sample_windows",
    "earliest_delta_history_entry",
    "latest_delta_history_entry",
    "quote_table_name",
    "source_history_entry_at_or_before_timestamp",
    "summarize_delta_history_entry",
    "summarize_watermark_run",
    "validate_contiguous_source_window",
    "validate_dev_watermark_table_name",
    "validate_explicit_window_against_watermark",
    "validate_history_source_is_table",
    "validate_watermark_advance_gates",
    "validate_watermark_rows",
    "validate_watermark_schema",
    "validate_watermark_schema_migration_gates",
)


@dataclass(frozen=True)
class SourceWindow:
    source_alias: str
    starting_version: int | None
    ending_version: int | None
    previous_watermark_version: int | None

    @property
    def has_new_versions(self) -> bool:
        return self.starting_version is not None and self.ending_version is not None


@dataclass(frozen=True)
class BootstrapVersionCandidate:
    source_alias: str
    candidate_version: int | None
    candidate_timestamp: str | None
    candidate_operation: str | None
    shadow_baseline_timestamp: str
    status: str


def _mapping_get(row: Mapping[str, Any], key: str) -> Any:
    return row.get(key)


def _coerce_history_timestamp(value: Any) -> datetime:
    if value is None:
        raise ValueError("Delta history row is missing timestamp.")
    if isinstance(value, datetime):
        timestamp = value
    elif isinstance(value, str):
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    else:
        raise ValueError(f"Unsupported Delta history timestamp type: {type(value).__name__}")
    return timestamp.replace(tzinfo=None)


def _history_sort_key(row: Mapping[str, Any]) -> tuple[datetime, int]:
    version = _mapping_get(row, "version")
    if version is None:
        raise ValueError("Delta history row is missing version.")
    return (_coerce_history_timestamp(_mapping_get(row, "timestamp")), int(version))


def summarize_delta_history_entry(row: Mapping[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "version": None,
            "timestamp": None,
            "operation": None,
        }
    timestamp = _coerce_history_timestamp(_mapping_get(row, "timestamp"))
    return {
        "version": int(_mapping_get(row, "version")),
        "timestamp": timestamp.isoformat(sep=" "),
        "operation": _mapping_get(row, "operation"),
    }


def earliest_delta_history_entry(history_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(history_rows)
    if not rows:
        raise ValueError("Delta history is empty; cannot identify a shadow baseline timestamp.")
    return dict(min(rows, key=_history_sort_key))


def latest_delta_history_entry(history_rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = tuple(history_rows)
    if not rows:
        raise ValueError("Delta history is empty; cannot identify the latest table operation.")
    return dict(max(rows, key=_history_sort_key))


def source_history_entry_at_or_before_timestamp(
    history_rows: Iterable[Mapping[str, Any]],
    baseline_timestamp: Any,
) -> dict[str, Any] | None:
    baseline = _coerce_history_timestamp(baseline_timestamp)
    eligible = [
        row
        for row in history_rows
        if _coerce_history_timestamp(_mapping_get(row, "timestamp")) <= baseline
    ]
    if not eligible:
        return None
    return dict(max(eligible, key=_history_sort_key))


def build_bootstrap_version_candidate(
    *,
    source_alias: str,
    source_history_rows: Iterable[Mapping[str, Any]],
    shadow_baseline_timestamp: Any,
) -> BootstrapVersionCandidate:
    if source_alias not in SOURCE_ALIASES:
        raise ValueError(f"Unsupported source alias: {source_alias!r}")
    baseline = _coerce_history_timestamp(shadow_baseline_timestamp)
    candidate = source_history_entry_at_or_before_timestamp(source_history_rows, baseline)
    if candidate is None:
        return BootstrapVersionCandidate(
            source_alias=source_alias,
            candidate_version=None,
            candidate_timestamp=None,
            candidate_operation=None,
            shadow_baseline_timestamp=baseline.isoformat(sep=" "),
            status=BOOTSTRAP_PREFLIGHT_MISSING_CANDIDATE,
        )

    summary = summarize_delta_history_entry(candidate)
    return BootstrapVersionCandidate(
        source_alias=source_alias,
        candidate_version=summary["version"],
        candidate_timestamp=summary["timestamp"],
        candidate_operation=summary["operation"],
        shadow_baseline_timestamp=baseline.isoformat(sep=" "),
        status=BOOTSTRAP_PREFLIGHT_CANDIDATE_ONLY,
    )


def classify_uc_object_type(object_type: Any) -> str:
    if object_type is None:
        return UC_OBJECT_UNKNOWN
    normalized = str(object_type).strip().lower().replace("_", " ")
    if "view" in normalized:
        return UC_OBJECT_VIEW
    if normalized in {"table", "managed", "external", "managed table", "external table"}:
        return UC_OBJECT_TABLE
    if "table" in normalized:
        return UC_OBJECT_TABLE
    return UC_OBJECT_UNKNOWN


def validate_history_source_is_table(
    *,
    source_alias: str,
    logical_source_name: str,
    history_source_name: str,
    object_type: Any,
) -> dict[str, Any]:
    classified_type = classify_uc_object_type(object_type)
    status = SOURCE_HISTORY_READY
    can_describe_history = True
    message = "History object is a physical table."

    if classified_type == UC_OBJECT_VIEW:
        status = SOURCE_HISTORY_UNAVAILABLE_FOR_VIEW
        can_describe_history = False
        message = (
            "Configured source history object is a VIEW. Provide physical Delta table in "
            "SOURCE_LEG_HISTORY_TABLE / SOURCE_LEG_TIMES_HISTORY_TABLE."
        )
    elif classified_type != UC_OBJECT_TABLE:
        status = SOURCE_HISTORY_UNAVAILABLE
        can_describe_history = False
        message = (
            "Configured source history object type is not known to support DESCRIBE HISTORY. "
            "Provide physical Delta table in SOURCE_LEG_HISTORY_TABLE / SOURCE_LEG_TIMES_HISTORY_TABLE."
        )

    return {
        "source_alias": source_alias,
        "logical_source_name": logical_source_name,
        "history_source_name": history_source_name,
        "object_type": object_type,
        "classified_object_type": classified_type,
        "can_describe_history": can_describe_history,
        "status": status,
        "message": message,
    }


def build_bootstrap_preflight_status_for_history_sources(
    history_source_statuses: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    statuses = tuple(dict(status) for status in history_source_statuses)
    blocked = [status for status in statuses if not status.get("can_describe_history")]
    if not blocked:
        return {
            "status": BOOTSTRAP_PREFLIGHT_CANDIDATE_ONLY,
            "history_sources": statuses,
            "message": "History sources are physical tables; candidate bootstrap versions can be computed.",
        }

    view_blocked = [
        status for status in blocked if status.get("status") == SOURCE_HISTORY_UNAVAILABLE_FOR_VIEW
    ]
    if view_blocked:
        status = BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_VIEW
        message = (
            "Configured source history object is a VIEW. Provide physical Delta table in "
            "SOURCE_LEG_HISTORY_TABLE / SOURCE_LEG_TIMES_HISTORY_TABLE."
        )
    else:
        status = BOOTSTRAP_PREFLIGHT_BLOCKED_SOURCE_HISTORY_UNAVAILABLE
        message = "Source history is unavailable; provide physical Delta history table overrides."

    return {
        "status": status,
        "history_sources": statuses,
        "blocked_history_sources": tuple(blocked),
        "message": message,
    }


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


def quote_table_name(table_name: str) -> str:
    return ".".join(_quote_identifier(part) for part in table_name.split("."))


def _sql_string(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_int(value: int | None) -> str:
    return "CAST(NULL AS BIGINT)" if value is None else str(int(value))


def _sql_timestamp(value: str | None) -> str:
    if value is None:
        return "CAST(NULL AS TIMESTAMP)"
    escaped = value.replace("'", "''")
    return f"CAST('{escaped}' AS TIMESTAMP)"


def validate_dev_watermark_table_name(table_name: str) -> bool:
    if table_name != DEFAULT_WATERMARK_TABLE:
        raise ValueError(
            "Watermark schema migration can only target the dev control table "
            f"{DEFAULT_WATERMARK_TABLE}. Got: {table_name}"
        )
    return True


def detect_missing_watermark_columns(columns: Iterable[str]) -> tuple[str, ...]:
    available = set(columns)
    return tuple(column for column in REQUIRED_WATERMARK_COLUMNS if column not in available)


def validate_watermark_schema(columns: Iterable[str]) -> bool:
    missing = list(detect_missing_watermark_columns(columns))
    if missing:
        raise ValueError(
            "Watermark table schema is missing required columns. "
            f"Missing={missing}. Do not silently infer alternate column names."
        )
    return True


def validate_watermark_schema_migration_gates(
    *,
    table_name: str,
    missing_columns: Iterable[str],
    allow_schema_migration: bool,
    dry_run_only: bool,
    write_confirmation: str,
    required_write_confirmation: str = REQUIRED_WRITE_CONFIRMATION,
) -> bool:
    validate_dev_watermark_table_name(table_name)
    missing = tuple(missing_columns)
    if not missing:
        return True

    core_missing = [column for column in missing if column in CORE_WATERMARK_COLUMNS]
    if core_missing:
        raise ValueError(
            "watermark_schema_incompatible: core watermark columns are missing and cannot be auto-migrated. "
            f"Missing core columns={core_missing}."
        )

    unsupported = [column for column in missing if column not in ADDITIVE_WATERMARK_METADATA_COLUMNS]
    if unsupported:
        raise ValueError(
            "watermark_schema_incompatible: only additive metadata columns can be auto-migrated. "
            f"Unsupported missing columns={unsupported}."
        )

    if not allow_schema_migration:
        raise ValueError(
            "Watermark schema migration is required but ALLOW_WATERMARK_SCHEMA_MIGRATION is False."
        )
    if dry_run_only:
        raise ValueError("Watermark schema migration requires DRY_RUN_ONLY=False.")
    if write_confirmation != required_write_confirmation:
        raise ValueError("Watermark schema migration requires the exact dev-shadow write confirmation string.")
    return True


def build_watermark_schema_migration_sql(
    *,
    table_name: str = DEFAULT_WATERMARK_TABLE,
    missing_columns: Iterable[str],
) -> str:
    validate_dev_watermark_table_name(table_name)
    missing = tuple(missing_columns)
    if not missing:
        raise ValueError("No missing watermark columns were provided for schema migration.")
    validate_watermark_schema_migration_gates(
        table_name=table_name,
        missing_columns=missing,
        allow_schema_migration=True,
        dry_run_only=False,
        write_confirmation=REQUIRED_WRITE_CONFIRMATION,
    )
    column_specs = ",\n  ".join(
        f"{_quote_identifier(column)} {ADDITIVE_WATERMARK_METADATA_COLUMNS[column]}"
        for column in missing
    )
    return f"""ALTER TABLE {quote_table_name(table_name)}
ADD COLUMNS (
  {column_specs}
)"""


def validate_watermark_rows(
    rows: Iterable[Mapping[str, Any]],
    *,
    require_all_sources: bool = True,
) -> dict[str, Mapping[str, Any]]:
    rows_by_alias: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        source_alias = row.get("source_alias")
        if source_alias not in SOURCE_ALIASES:
            raise ValueError(f"Unsupported watermark source_alias: {source_alias!r}")
        if source_alias in rows_by_alias:
            raise ValueError(f"Duplicate watermark row for source_alias={source_alias!r}")
        version = row.get("last_processed_version")
        if version is None:
            raise ValueError(f"Watermark row for source_alias={source_alias!r} has no last_processed_version.")
        rows_by_alias[source_alias] = row

    missing = [source_alias for source_alias in SOURCE_ALIASES if source_alias not in rows_by_alias]
    if require_all_sources and missing:
        raise ValueError(f"Missing watermark rows for source aliases: {missing}")
    return rows_by_alias


def classify_watermark_run_status(
    *,
    watermark_rows_present: bool,
    any_new_versions: bool,
    all_windows_contiguous: bool,
    all_sources_present: bool = True,
    validation_ok: bool = True,
    advanced: bool = False,
) -> str:
    if not watermark_rows_present:
        return WATERMARK_BOOTSTRAP_REQUIRED
    if not all_sources_present:
        return WATERMARK_BLOCKED_MISSING_SOURCE
    if not all_windows_contiguous:
        return WATERMARK_BLOCKED_NON_CONTIGUOUS
    if not validation_ok:
        return WATERMARK_BLOCKED_VALIDATION_FAILED
    if advanced:
        return WATERMARK_ADVANCE_PASS
    if not any_new_versions:
        return WATERMARK_NOOP
    return WATERMARK_READY


def build_next_source_windows_from_watermarks(
    watermark_rows: Mapping[str, Mapping[str, Any]],
    latest_available_versions: Mapping[str, int | None],
    *,
    max_cdf_version_span_per_source: int,
) -> dict[str, SourceWindow]:
    if max_cdf_version_span_per_source <= 0:
        raise ValueError("max_cdf_version_span_per_source must be positive.")

    rows_by_alias = validate_watermark_rows(watermark_rows.values(), require_all_sources=True)
    windows: dict[str, SourceWindow] = {}
    for source_alias in SOURCE_ALIASES:
        previous_version = int(rows_by_alias[source_alias]["last_processed_version"])
        latest_version = latest_available_versions.get(source_alias)
        if latest_version is None or int(latest_version) <= previous_version:
            windows[source_alias] = SourceWindow(
                source_alias=source_alias,
                starting_version=None,
                ending_version=None,
                previous_watermark_version=previous_version,
            )
            continue

        starting_version = previous_version + 1
        ending_version = min(int(latest_version), starting_version + max_cdf_version_span_per_source - 1)
        windows[source_alias] = SourceWindow(
            source_alias=source_alias,
            starting_version=starting_version,
            ending_version=ending_version,
            previous_watermark_version=previous_version,
        )
    return windows


def validate_contiguous_source_window(
    *,
    source_alias: str,
    previous_watermark_version: int,
    starting_version: int | None,
    ending_version: int | None,
    require_contiguous: bool = True,
) -> bool:
    if source_alias not in SOURCE_ALIASES:
        raise ValueError(f"Unsupported source alias: {source_alias!r}")
    if starting_version is None and ending_version is None:
        return True
    if starting_version is None or ending_version is None:
        raise ValueError(f"{source_alias} source window requires both starting and ending versions.")
    if int(ending_version) < int(starting_version):
        raise ValueError(f"{source_alias} ending version must be >= starting version.")
    expected_start = int(previous_watermark_version) + 1
    if require_contiguous and int(starting_version) != expected_start:
        raise ValueError(
            f"{source_alias} source window is non-contiguous: expected start {expected_start}, "
            f"got {starting_version}."
        )
    return True


def validate_explicit_window_against_watermark(
    *,
    source_alias: str,
    previous_watermark_version: int,
    starting_version: int | None,
    ending_version: int | None,
    allow_non_watermark_explicit_window: bool = False,
) -> bool:
    return validate_contiguous_source_window(
        source_alias=source_alias,
        previous_watermark_version=previous_watermark_version,
        starting_version=starting_version,
        ending_version=ending_version,
        require_contiguous=not allow_non_watermark_explicit_window,
    )


def detect_non_contiguous_validation_sample_windows(windows: Iterable[SourceWindow]) -> bool:
    previous_by_alias: dict[str, int | None] = {}
    for window in sorted(windows, key=lambda item: (item.source_alias, item.starting_version or -1)):
        if not window.has_new_versions:
            continue
        previous_end = previous_by_alias.get(window.source_alias)
        if previous_end is not None and window.starting_version != previous_end + 1:
            return True
        previous_by_alias[window.source_alias] = window.ending_version
    return False


def validate_watermark_advance_gates(
    *,
    shadow_merge_executed: bool,
    post_merge_validation_ok: bool,
    candidate_duplicate_key_count: int = 0,
    candidate_null_key_count: int = 0,
    shadow_duplicate_key_count: int = 0,
    shadow_null_key_count: int = 0,
    candidate_current_mismatch_count: int = 0,
    candidate_shadow_mismatch_count: int = 0,
    write_confirmation: str = "",
    required_write_confirmation: str = REQUIRED_WRITE_CONFIRMATION,
    windows_contiguous: bool = True,
    watermark_rows_valid: bool = True,
    source_versions_present: bool = True,
) -> bool:
    if write_confirmation != required_write_confirmation:
        raise ValueError("Watermark advance blocked: write confirmation is missing or incorrect.")
    if not watermark_rows_valid:
        raise ValueError("Watermark advance blocked: watermark table state is missing or invalid.")
    if not source_versions_present:
        raise ValueError("Watermark advance blocked: source-specific versions are missing.")
    if not windows_contiguous:
        raise ValueError("Watermark advance blocked: source windows are non-contiguous.")
    if not shadow_merge_executed:
        raise ValueError("Watermark advance blocked: shadow merge did not run.")
    if not post_merge_validation_ok:
        raise ValueError("Watermark advance blocked: post-merge validation failed.")
    if candidate_duplicate_key_count:
        raise ValueError("Watermark advance blocked: duplicate candidate keys detected.")
    if candidate_null_key_count:
        raise ValueError("Watermark advance blocked: null candidate keys detected.")
    if shadow_duplicate_key_count:
        raise ValueError("Watermark advance blocked: duplicate shadow keys detected.")
    if shadow_null_key_count:
        raise ValueError("Watermark advance blocked: null shadow keys detected.")
    if candidate_current_mismatch_count:
        raise ValueError("Watermark advance blocked: candidate/current mismatches detected.")
    if candidate_shadow_mismatch_count:
        raise ValueError("Watermark advance blocked: candidate/shadow mismatches detected.")
    return True


def build_watermark_advance_rows(
    *,
    source_windows: Mapping[str, SourceWindow],
    processed_timestamps_by_alias: Mapping[str, str | None],
    run_id: str,
    updated_by_stage: str = DEFAULT_STAGE_NAME,
) -> tuple[dict[str, Any], ...]:
    rows = []
    for source_alias in SOURCE_ALIASES:
        window = source_windows[source_alias]
        if not window.has_new_versions:
            continue
        rows.append(
            {
                "source_alias": source_alias,
                "last_processed_version": window.ending_version,
                "last_processed_timestamp": processed_timestamps_by_alias.get(source_alias),
                "updated_at": datetime.utcnow().isoformat(timespec="seconds"),
                "updated_by_stage": updated_by_stage,
                "run_id": run_id,
            }
        )
    return tuple(rows)


def build_watermark_advance_merge_sql(
    *,
    watermark_table: str = DEFAULT_WATERMARK_TABLE,
    advance_rows: Iterable[Mapping[str, Any]],
) -> str:
    rows = tuple(advance_rows)
    if not rows:
        raise ValueError("Cannot build watermark advance SQL without advance rows.")

    selects = []
    for row in rows:
        selects.append(
            "SELECT "
            f"{_sql_string(str(row['source_alias']))} AS source_alias, "
            f"{_sql_int(row['last_processed_version'])} AS last_processed_version, "
            f"{_sql_timestamp(row.get('last_processed_timestamp'))} AS last_processed_timestamp, "
            "current_timestamp() AS updated_at, "
            f"{_sql_string(str(row['updated_by_stage']))} AS updated_by_stage, "
            f"{_sql_string(str(row['run_id']))} AS run_id"
        )
    source_sql = "\nUNION ALL\n".join(selects)
    target = quote_table_name(watermark_table)
    return f"""MERGE INTO {target} AS target
USING (
{source_sql}
) AS source
ON target.source_alias = source.source_alias
WHEN MATCHED THEN UPDATE SET
    target.last_processed_version = source.last_processed_version,
    target.last_processed_timestamp = source.last_processed_timestamp,
    target.updated_at = source.updated_at,
    target.updated_by_stage = source.updated_by_stage,
    target.run_id = source.run_id
WHEN NOT MATCHED THEN INSERT (
    source_alias,
    last_processed_version,
    last_processed_timestamp,
    updated_at,
    updated_by_stage,
    run_id
) VALUES (
    source.source_alias,
    source.last_processed_version,
    source.last_processed_timestamp,
    source.updated_at,
    source.updated_by_stage,
    source.run_id
)"""


def summarize_watermark_run(
    *,
    status: str,
    source_windows: Mapping[str, SourceWindow],
    shadow_merge_executed: bool,
    post_merge_validation_ok: bool,
    watermarks_advanced: bool,
    duplicate_or_null_key_count: int = 0,
    mismatch_count: int = 0,
) -> dict[str, Any]:
    return {
        "status": status,
        "source_windows": {
            source_alias: {
                "starting_version": window.starting_version,
                "ending_version": window.ending_version,
                "previous_watermark_version": window.previous_watermark_version,
            }
            for source_alias, window in source_windows.items()
        },
        "shadow_merge_executed": shadow_merge_executed,
        "post_merge_validation_ok": post_merge_validation_ok,
        "watermarks_advanced": watermarks_advanced,
        "duplicate_or_null_key_count": duplicate_or_null_key_count,
        "mismatch_count": mismatch_count,
    }
