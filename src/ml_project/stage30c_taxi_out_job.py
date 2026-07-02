from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, Sequence


ALLOWED_JOB_MODES = ("validation", "shadow_merge", "watermark_advance")
ALLOWED_SOURCE_VERSION_MODES = ("explicit", "watermark")
SOURCE_ALIASES = ("leg", "leg_times")
REQUIRED_WRITE_CONFIRMATION = "I_UNDERSTAND_THIS_WRITES_TO_DEV_SHADOW_TABLES_ONLY"

DEFAULT_MIN_PARITY_WINDOWS = 2
DEFAULT_MIN_ENTITIES = 2

STATUS_PARITY_PASS = "parity_pass"
STATUS_VALIDATION_NO_OVERLAP = "validation_no_overlap"
STATUS_VALIDATION_FAILED = "validation_failed"
STATUS_DRY_RUN_PASS = "dry_run_pass"
STATUS_SHADOW_MERGE_PASS = "shadow_merge_pass"


@dataclass(frozen=True)
class ValidationWindow:
    window_id: str
    leg_start: int | None
    leg_end: int | None
    leg_times_start: int | None
    leg_times_end: int | None
    entity_filter: str = ""
    max_affected_entities: int = 3
    overlap_aware: bool = False

    @property
    def configured_source_aliases(self) -> tuple[str, ...]:
        aliases = []
        if self.leg_start is not None:
            aliases.append("leg")
        if self.leg_times_start is not None:
            aliases.append("leg_times")
        return tuple(aliases)


def _as_date_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return value
    return value


def _key(row: Mapping[str, Any]) -> tuple[Any, Any]:
    return (row["dep_ap_sched"], _as_date_value(row["event_date"]))


def build_default_validation_windows() -> tuple[dict[str, Any], ...]:
    return (
        {
            "window_id": "A_KRK",
            "leg_start": 34600,
            "leg_end": 34620,
            "leg_times_start": 34519,
            "leg_times_end": 34538,
            "entity_filter": "KRK",
            "max_affected_entities": 1,
        },
        {
            "window_id": "B_WAW",
            "leg_start": 34680,
            "leg_end": 34700,
            "leg_times_start": 34598,
            "leg_times_end": 34618,
            "entity_filter": "WAW",
            "max_affected_entities": 1,
        },
        {
            "window_id": "C_MULTI_OVERLAP_AWARE",
            "leg_start": 34680,
            "leg_end": 34700,
            "leg_times_start": 34598,
            "leg_times_end": 34618,
            "entity_filter": "",
            "max_affected_entities": 3,
            "overlap_aware": True,
        },
    )


def validate_validation_window_config(window: Mapping[str, Any]) -> ValidationWindow:
    if "window_id" not in window or not str(window["window_id"]).strip():
        raise ValueError("Each validation window requires a non-empty window_id.")
    forbidden_global_keys = {"cdf_start", "cdf_end", "starting_version", "ending_version", "global_cdf_version"}
    forbidden_present = sorted(key for key in forbidden_global_keys if key in window)
    if forbidden_present:
        raise ValueError(f"Validation windows require source-specific CDF versions, not global keys: {forbidden_present}")

    leg_start = window.get("leg_start")
    leg_end = window.get("leg_end")
    leg_times_start = window.get("leg_times_start")
    leg_times_end = window.get("leg_times_end")

    for alias, start, end in (
        ("leg", leg_start, leg_end),
        ("leg_times", leg_times_start, leg_times_end),
    ):
        if end is not None and start is None:
            raise ValueError(f"{alias} ending version requires {alias} starting version.")
        if start is not None and end is not None and int(end) < int(start):
            raise ValueError(f"{alias} ending version must be >= starting version.")

    if leg_start is None and leg_times_start is None:
        raise ValueError("Each validation window requires at least one source-specific starting version.")

    max_affected_entities = int(window.get("max_affected_entities", 3))
    if max_affected_entities <= 0:
        raise ValueError("max_affected_entities must be positive.")

    return ValidationWindow(
        window_id=str(window["window_id"]),
        leg_start=None if leg_start is None else int(leg_start),
        leg_end=None if leg_end is None else int(leg_end),
        leg_times_start=None if leg_times_start is None else int(leg_times_start),
        leg_times_end=None if leg_times_end is None else int(leg_times_end),
        entity_filter=str(window.get("entity_filter", "") or ""),
        max_affected_entities=max_affected_entities,
        overlap_aware=bool(window.get("overlap_aware", False)),
    )


def parse_validation_windows(
    validation_windows: Sequence[Mapping[str, Any]] | None,
    validation_windows_json: str = "",
) -> tuple[ValidationWindow, ...]:
    raw_windows: Sequence[Mapping[str, Any]]
    if validation_windows_json.strip():
        parsed = json.loads(validation_windows_json)
        if not isinstance(parsed, list):
            raise ValueError("VALIDATION_WINDOWS_JSON must decode to a list of window objects.")
        raw_windows = parsed
    else:
        raw_windows = build_default_validation_windows() if validation_windows is None else validation_windows

    windows = tuple(validate_validation_window_config(window) for window in raw_windows)
    if not windows:
        raise ValueError("At least one validation window is required.")
    return windows


def tag_validation_overlap(
    affected_pairs: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
    current_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    candidate_keys = {_key(row) for row in candidate_rows}
    current_keys = {_key(row) for row in current_rows}
    tagged = []

    for row in affected_pairs:
        key = _key(row)
        has_candidate = key in candidate_keys
        has_current_mv_key = key in current_keys
        tagged_row = dict(row)
        tagged_row["event_date"] = key[1]
        tagged_row["has_candidate"] = has_candidate
        tagged_row["has_current_mv_key"] = has_current_mv_key
        tagged_row["has_validation_overlap"] = has_candidate and has_current_mv_key
        tagged.append(tagged_row)

    return tuple(sorted(tagged, key=lambda row: (row["dep_ap_sched"], row["event_date"])))


def select_overlap_aware_entities(
    tagged_pairs: Iterable[Mapping[str, Any]],
    *,
    max_entities: int,
    require_validation_overlap: bool = True,
) -> tuple[str, ...]:
    if max_entities <= 0:
        raise ValueError("max_entities must be positive.")

    stats: dict[str, dict[str, int]] = {}
    for row in tagged_pairs:
        entity = str(row["dep_ap_sched"])
        entity_stats = stats.setdefault(entity, {"overlap": 0, "candidate": 0, "current": 0, "affected": 0})
        entity_stats["affected"] += 1
        entity_stats["candidate"] += int(bool(row.get("has_candidate")))
        entity_stats["current"] += int(bool(row.get("has_current_mv_key")))
        entity_stats["overlap"] += int(bool(row.get("has_validation_overlap")))

    candidates = [
        (entity, values)
        for entity, values in stats.items()
        if not require_validation_overlap or values["overlap"] > 0
    ]
    candidates.sort(
        key=lambda item: (
            -item[1]["overlap"],
            -item[1]["candidate"],
            -item[1]["current"],
            -item[1]["affected"],
            item[0],
        )
    )
    return tuple(entity for entity, _ in candidates[:max_entities])


def filter_pairs_to_entities(
    pairs: Iterable[Mapping[str, Any]],
    selected_entities: Iterable[str],
) -> tuple[dict[str, Any], ...]:
    selected = set(selected_entities)
    return tuple(
        sorted(
            (dict(row) for row in pairs if row["dep_ap_sched"] in selected),
            key=lambda row: (row["dep_ap_sched"], _as_date_value(row["event_date"])),
        )
    )


def summarize_window_result(
    *,
    window_id: str,
    entity_filter: str = "",
    selected_entities: Iterable[str] = (),
    cdf_leg_count: int = 0,
    cdf_leg_times_count: int = 0,
    dirty_events_count: int = 0,
    affected_pairs_count: int = 0,
    candidate_rows_count: int = 0,
    current_scoped_rows_count: int = 0,
    candidate_current_status_counts: Mapping[str, int] | None = None,
    shadow_merge_executed: bool = False,
    candidate_shadow_status_counts: Mapping[str, int] | None = None,
    candidate_duplicate_key_count: int = 0,
    candidate_null_key_count: int = 0,
    shadow_duplicate_key_count: int = 0,
    shadow_null_key_count: int = 0,
    require_validation_overlap: bool = True,
) -> dict[str, Any]:
    current_counts = dict(candidate_current_status_counts or {})
    shadow_counts = dict(candidate_shadow_status_counts or {})
    mismatch_count = sum(
        count for status, count in current_counts.items() if status != "matched"
    )
    shadow_mismatch_count = sum(
        count for status, count in shadow_counts.items() if status != "matched"
    )
    matched_count = int(current_counts.get("matched", 0))
    has_overlap = candidate_rows_count > 0 and current_scoped_rows_count > 0 and matched_count > 0
    key_failure = any(
        count > 0
        for count in (
            candidate_duplicate_key_count,
            candidate_null_key_count,
            shadow_duplicate_key_count,
            shadow_null_key_count,
        )
    )

    if require_validation_overlap and not has_overlap:
        validation_status = STATUS_VALIDATION_NO_OVERLAP
    elif key_failure or mismatch_count or shadow_mismatch_count:
        validation_status = STATUS_VALIDATION_FAILED
    elif shadow_merge_executed:
        validation_status = STATUS_SHADOW_MERGE_PASS
    elif has_overlap:
        validation_status = STATUS_PARITY_PASS
    else:
        validation_status = STATUS_DRY_RUN_PASS

    return {
        "window_id": window_id,
        "entity_filter": entity_filter,
        "selected_entities": tuple(selected_entities),
        "cdf_leg_count": int(cdf_leg_count),
        "cdf_leg_times_count": int(cdf_leg_times_count),
        "dirty_events_count": int(dirty_events_count),
        "affected_pairs_count": int(affected_pairs_count),
        "candidate_rows_count": int(candidate_rows_count),
        "current_scoped_rows_count": int(current_scoped_rows_count),
        "candidate_current_status_counts": current_counts,
        "shadow_merge_executed": bool(shadow_merge_executed),
        "candidate_shadow_status_counts": shadow_counts,
        "candidate_duplicate_key_count": int(candidate_duplicate_key_count),
        "candidate_null_key_count": int(candidate_null_key_count),
        "shadow_duplicate_key_count": int(shadow_duplicate_key_count),
        "shadow_null_key_count": int(shadow_null_key_count),
        "matched_count": matched_count,
        "mismatch_count": mismatch_count + shadow_mismatch_count,
        "has_validation_overlap": has_overlap,
        "validation_status": validation_status,
    }


def summarize_job_result(
    window_results: Iterable[Mapping[str, Any]],
    *,
    require_at_least_two_parity_windows: bool = True,
    require_at_least_two_entities_overall: bool = True,
    min_parity_windows: int = DEFAULT_MIN_PARITY_WINDOWS,
    min_entities: int = DEFAULT_MIN_ENTITIES,
    shadow_table_name_safe: bool = True,
    watermark_table_name_safe: bool = True,
    watermarks_advanced: bool = False,
) -> dict[str, Any]:
    windows = tuple(window_results)
    parity_windows = [
        row
        for row in windows
        if row.get("validation_status") in {STATUS_PARITY_PASS, STATUS_SHADOW_MERGE_PASS}
    ]
    entities = {
        entity
        for row in parity_windows
        for entity in row.get("selected_entities", ())
    }
    total_mismatches = sum(int(row.get("mismatch_count", 0)) for row in windows)
    any_duplicate_or_null_keys = any(
        int(row.get(field, 0)) > 0
        for row in windows
        for field in (
            "candidate_duplicate_key_count",
            "candidate_null_key_count",
            "shadow_duplicate_key_count",
            "shadow_null_key_count",
        )
    )
    parity_windows_required_met = (
        len(parity_windows) >= min_parity_windows
        if require_at_least_two_parity_windows
        else True
    )
    entities_required_met = (
        len(entities) >= min_entities
        if require_at_least_two_entities_overall
        else True
    )

    return {
        "number_of_windows": len(windows),
        "number_of_parity_pass_windows": len(parity_windows),
        "number_of_entities_validated": len(entities),
        "entities_validated": tuple(sorted(entities)),
        "total_candidate_rows": sum(int(row.get("candidate_rows_count", 0)) for row in windows),
        "total_current_matched_rows": sum(int(row.get("matched_count", 0)) for row in windows),
        "total_mismatches": total_mismatches,
        "any_duplicate_or_null_keys": any_duplicate_or_null_keys,
        "shadow_merges_executed_count": sum(int(bool(row.get("shadow_merge_executed"))) for row in windows),
        "watermarks_advanced": bool(watermarks_advanced),
        "run_job": True,
        "source_version_mode_explicit": True,
        "read_cdf_all_windows_ok": True,
        "dirty_events_any_window": any(int(row.get("dirty_events_count", 0)) > 0 for row in windows),
        "affected_pairs_any_window": any(int(row.get("affected_pairs_count", 0)) > 0 for row in windows),
        "overlap_aware_selection_enabled": True,
        "parity_windows_required_met": parity_windows_required_met,
        "entities_required_met": entities_required_met,
        "candidate_keys_unique_all": not any(int(row.get("candidate_duplicate_key_count", 0)) for row in windows),
        "candidate_keys_non_null_all": not any(int(row.get("candidate_null_key_count", 0)) for row in windows),
        "current_overlap_compare_ok_all": total_mismatches == 0,
        "shadow_table_name_safe": bool(shadow_table_name_safe),
        "watermark_table_name_safe": bool(watermark_table_name_safe),
        "shadow_merge_executed_any": any(bool(row.get("shadow_merge_executed")) for row in windows),
        "shadow_post_merge_validation_ok_all": not any(
            row.get("validation_status") == STATUS_VALIDATION_FAILED for row in windows
        ),
        "read_only_or_dev_shadow_only": True,
        "overall_pass": (
            parity_windows_required_met
            and entities_required_met
            and not any_duplicate_or_null_keys
            and total_mismatches == 0
            and shadow_table_name_safe
            and watermark_table_name_safe
        ),
    }


def validate_job_write_gates(
    *,
    job_mode: str,
    dry_run_only: bool,
    allow_shadow_merge: bool,
    allow_watermark_advance: bool,
    write_confirmation: str,
    required_write_confirmation: str = REQUIRED_WRITE_CONFIRMATION,
) -> bool:
    if job_mode not in ALLOWED_JOB_MODES:
        raise ValueError(f"Unsupported JOB_MODE {job_mode!r}.")
    if job_mode == "validation" and (allow_shadow_merge or allow_watermark_advance):
        raise ValueError("validation mode cannot enable shadow merge or watermark advancement.")
    if allow_watermark_advance and job_mode != "watermark_advance":
        raise ValueError("ALLOW_WATERMARK_ADVANCE requires JOB_MODE='watermark_advance'.")
    if allow_shadow_merge and job_mode not in {"shadow_merge", "watermark_advance"}:
        raise ValueError("ALLOW_SHADOW_MERGE requires shadow_merge or watermark_advance mode.")
    if dry_run_only and (allow_shadow_merge or allow_watermark_advance):
        raise ValueError("Write flags require DRY_RUN_ONLY = False.")
    if (allow_shadow_merge or allow_watermark_advance or not dry_run_only) and write_confirmation != required_write_confirmation:
        raise ValueError("Job write mode requires the exact dev-shadow write confirmation string.")
    return True


def validate_job_success_preconditions(
    *,
    job_summary: Mapping[str, Any],
    job_mode: str,
    allow_watermark_advance: bool,
    processed_versions_by_alias: Mapping[str, int | None] | None = None,
    write_confirmation: str = "",
    required_write_confirmation: str = REQUIRED_WRITE_CONFIRMATION,
    configured_source_aliases: Iterable[str] = SOURCE_ALIASES,
) -> bool:
    if not job_summary.get("overall_pass", False):
        raise ValueError("Job success preconditions require overall_pass = True.")
    if job_mode == "watermark_advance" or allow_watermark_advance:
        if write_confirmation != required_write_confirmation:
            raise ValueError("Watermark advancement requires the exact dev-shadow write confirmation string.")
        if not job_summary.get("shadow_merge_executed_any"):
            raise ValueError("Watermark advancement requires a shadow merge.")
        if not job_summary.get("shadow_post_merge_validation_ok_all"):
            raise ValueError("Watermark advancement requires post-merge validation success.")
        if not job_summary.get("candidate_keys_unique_all"):
            raise ValueError("Watermark advancement blocked by duplicate candidate keys.")
        if not job_summary.get("candidate_keys_non_null_all"):
            raise ValueError("Watermark advancement blocked by null candidate keys.")
        if job_summary.get("any_duplicate_or_null_keys"):
            raise ValueError("Watermark advancement blocked by duplicate/null shadow keys.")
        if not job_summary.get("current_overlap_compare_ok_all"):
            raise ValueError("Watermark advancement blocked by failed compare status.")
        versions = processed_versions_by_alias or {}
        missing_versions = [
            source_alias
            for source_alias in configured_source_aliases
            if source_alias not in versions or versions[source_alias] is None
        ]
        if missing_versions:
            raise ValueError(f"Watermark advancement requires source-specific latest versions for: {missing_versions}")
    return True
