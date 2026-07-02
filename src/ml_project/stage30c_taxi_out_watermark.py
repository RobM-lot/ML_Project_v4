from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Mapping


SOURCE_ALIASES = ("leg", "leg_times")
DEFAULT_WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"
DEFAULT_STAGE_NAME = "stage30c5_taxi_out_watermark_advance"
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"

REQUIRED_WATERMARK_COLUMNS = (
    "source_alias",
    "last_processed_version",
    "last_processed_timestamp",
    "updated_at",
    "updated_by_stage",
    "run_id",
)

WATERMARK_READY = "watermark_ready"
WATERMARK_NOOP = "watermark_noop"
WATERMARK_BOOTSTRAP_REQUIRED = "watermark_bootstrap_required"
WATERMARK_BLOCKED_NON_CONTIGUOUS = "watermark_blocked_non_contiguous"
WATERMARK_BLOCKED_MISSING_SOURCE = "watermark_blocked_missing_source"
WATERMARK_BLOCKED_VALIDATION_FAILED = "watermark_blocked_validation_failed"
WATERMARK_ADVANCE_PASS = "watermark_advance_pass"


@dataclass(frozen=True)
class SourceWindow:
    source_alias: str
    starting_version: int | None
    ending_version: int | None
    previous_watermark_version: int | None

    @property
    def has_new_versions(self) -> bool:
        return self.starting_version is not None and self.ending_version is not None


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


def validate_watermark_schema(columns: Iterable[str]) -> bool:
    available = set(columns)
    missing = [column for column in REQUIRED_WATERMARK_COLUMNS if column not in available]
    if missing:
        raise ValueError(
            "Watermark table schema is missing required columns. "
            f"Missing={missing}. Do not silently infer alternate column names."
        )
    return True


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
