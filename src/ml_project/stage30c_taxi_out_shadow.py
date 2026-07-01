from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:
    from pyspark.sql import DataFrame


TARGET_KEY_COLS = ("dep_ap_sched", "event_date")
SOURCE_ALIASES = ("leg", "leg_times")
DEFAULT_SHADOW_TAXI_OUT_TABLE = "panda_silver_dev.ml_ops.stage30c_ft_airport_daily_taxi_out_shadow"
DEFAULT_WATERMARK_TABLE = "panda_silver_dev.ml_ops.stage30c_taxi_out_watermarks"

DEV_ML_OPS_PREFIX = "panda_silver_dev.ml_ops."
CURRENT_TAXI_OUT_MV = "panda_silver_dev.ml_ops.ft_airport_daily_taxi_out"
PRODUCTION_SOURCE_TABLES = (
    "panda_silver_prod.occ_ops.netline___schedops__leg",
    "panda_silver_prod.occ_ops.netline___schedops__leg_times",
)

WATERMARK_COLUMNS = (
    "stage_name",
    "source_alias",
    "source_table",
    "last_processed_version",
    "last_processed_timestamp",
    "last_successful_run_id",
    "updated_at",
    "status",
)

SHADOW_CANDIDATE_FLAG_COL = "_stage30c_has_candidate"
TAXI_OUT_INCLUDED_LEG_TYPES = ("J", "C", "G")
SUPPORTED_LEG_CDF_CHANGE_TYPES = ("insert", "update_preimage", "update_postimage", "delete")
LEG_TIMES_ONLY_MAPPING_LIMITATION = (
    "leg_times-only dirty keys can map to the current eligible leg row, but they cannot recover an old "
    "dep_ap_sched/event_date mapping unless leg CDF preimage data or a historical leg snapshot is available."
)


@dataclass(frozen=True)
class SourceCdfWindow:
    source_alias: str
    starting_version: int | None
    ending_version: int | None

    @property
    def is_configured(self) -> bool:
        return self.starting_version is not None


@dataclass(frozen=True)
class DirtyEventRequirement:
    leg_no: Any
    dep_ap_sched: str
    dirty_event_date: date
    source_alias: str
    dirty_sides: tuple[str, ...]
    cdf_change_types: tuple[str, ...]
    dirty_reason: str
    limitation: str | None = None

    @property
    def key(self) -> tuple[str, date]:
        return (self.dep_ap_sched, self.dirty_event_date)


@dataclass(frozen=True)
class ShadowMergeSourceValidation:
    row_count: int
    key_count: int
    candidate_row_count: int
    delete_row_count: int


def _pyspark_sql():
    from pyspark.sql import functions as F

    return F


def _normalise_table_name(table_name: str) -> str:
    return table_name.strip().lower()


def _quote_identifier(identifier: str) -> str:
    escaped = identifier.replace("`", "``")
    return f"`{escaped}`"


def quote_table_name(table_name: str) -> str:
    return ".".join(_quote_identifier(part) for part in table_name.split("."))


def _sql_string(value: str) -> str:
    escaped = value.replace("'", "''")
    return f"'{escaped}'"


def _sql_int_or_null(value: int | None) -> str:
    return "NULL" if value is None else str(int(value))


def _sql_timestamp_or_null(value: str | None) -> str:
    if value is None:
        return "CAST(NULL AS TIMESTAMP)"
    return f"CAST({_sql_string(value)} AS TIMESTAMP)"


def _row_value(row: Mapping[str, Any], column_name: str, default: Any = None) -> Any:
    return row.get(column_name, default)


def _as_event_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return date.fromisoformat(value[:10])
    raise TypeError(f"Unsupported event-date value: {value!r}")


def _normalise_key_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.date()
    return value


def _row_key(row: Mapping[str, Any]) -> tuple[Any, Any]:
    missing = [column for column in TARGET_KEY_COLS if column not in row]
    if missing:
        raise ValueError(f"Row is missing target key columns: {missing}")
    return (_row_value(row, "dep_ap_sched"), _normalise_key_value(_row_value(row, "event_date")))


def _sorted_change_types(change_types: Iterable[str]) -> tuple[str, ...]:
    order = {change_type: index for index, change_type in enumerate(SUPPORTED_LEG_CDF_CHANGE_TYPES)}
    return tuple(sorted(set(change_types), key=lambda value: (order.get(value, 999), value)))


def _sorted_dirty_sides(dirty_sides: Iterable[str]) -> tuple[str, ...]:
    order = {"old": 0, "new": 1, "current": 2}
    return tuple(sorted(set(dirty_sides), key=lambda value: (order.get(value, 999), value)))


def is_taxi_out_eligible_leg_row(row: Mapping[str, Any] | None) -> bool:
    if row is None:
        return False
    return (
        _row_value(row, "counter") == 0
        and _row_value(row, "leg_type") in TAXI_OUT_INCLUDED_LEG_TYPES
        and _row_value(row, "leg_state") == "ARR"
        and _row_value(row, "dep_ap_sched") is not None
        and _row_value(row, "dep_sched_dt") is not None
    )


def classify_leg_cdf_dirty_sides(cdf_rows: Iterable[Mapping[str, Any]]) -> tuple[DirtyEventRequirement, ...]:
    """Classify leg CDF rows into old/new dirty taxi-out event requirements.

    `update_preimage` and `delete` rows represent old-side requirements.
    `insert` and `update_postimage` rows represent new-side requirements.
    Both sides are intentionally considered so removal and move cases are not
    reduced to current/postimage-only dirty extraction.
    """
    return build_dirty_event_requirements_from_leg_cdf(cdf_rows)


def build_dirty_event_requirements_from_leg_cdf(
    cdf_rows: Iterable[Mapping[str, Any]],
) -> tuple[DirtyEventRequirement, ...]:
    grouped: dict[tuple[Any, str, date], dict[str, Any]] = {}

    for row in cdf_rows:
        change_type = _row_value(row, "_change_type")
        if change_type not in SUPPORTED_LEG_CDF_CHANGE_TYPES:
            raise ValueError(f"Unsupported leg CDF change type: {change_type!r}")
        if not is_taxi_out_eligible_leg_row(row):
            continue

        side = "old" if change_type in {"update_preimage", "delete"} else "new"
        leg_no = _row_value(row, "leg_no")
        dep_ap_sched = _row_value(row, "dep_ap_sched")
        dirty_event_date = _as_event_date(_row_value(row, "dep_sched_dt"))
        key = (leg_no, dep_ap_sched, dirty_event_date)
        bucket = grouped.setdefault(
            key,
            {
                "dirty_sides": set(),
                "cdf_change_types": set(),
            },
        )
        bucket["dirty_sides"].add(side)
        bucket["cdf_change_types"].add(change_type)

    requirements = [
        DirtyEventRequirement(
            leg_no=leg_no,
            dep_ap_sched=dep_ap_sched,
            dirty_event_date=dirty_event_date,
            source_alias="leg",
            dirty_sides=_sorted_dirty_sides(bucket["dirty_sides"]),
            cdf_change_types=_sorted_change_types(bucket["cdf_change_types"]),
            dirty_reason="taxi_out_dirty_leg_cdf_change",
        )
        for (leg_no, dep_ap_sched, dirty_event_date), bucket in grouped.items()
    ]
    return tuple(sorted(requirements, key=lambda req: (str(req.leg_no), req.dep_ap_sched, req.dirty_event_date)))


def build_dirty_event_requirements_from_leg_times_cdf(
    cdf_rows: Iterable[Mapping[str, Any]],
    current_leg_rows: Iterable[Mapping[str, Any]],
) -> tuple[DirtyEventRequirement, ...]:
    """Map leg_times-only dirty leg_no values through current eligible leg rows.

    This is a guarded fallback for leg_times-only CDF. It cannot recover old
    entity/date mappings without leg CDF preimage data or a historical leg
    snapshot; that limitation is carried in each returned requirement.
    """
    change_types_by_leg: dict[Any, set[str]] = {}
    for row in cdf_rows:
        change_type = _row_value(row, "_change_type")
        if change_type not in SUPPORTED_LEG_CDF_CHANGE_TYPES:
            raise ValueError(f"Unsupported leg_times CDF change type: {change_type!r}")
        leg_no = _row_value(row, "leg_no")
        if leg_no is None:
            continue
        change_types_by_leg.setdefault(leg_no, set()).add(change_type)

    current_by_leg: dict[Any, Mapping[str, Any]] = {}
    for row in current_leg_rows:
        leg_no = _row_value(row, "leg_no")
        if leg_no is None or not is_taxi_out_eligible_leg_row(row):
            continue
        existing = current_by_leg.get(leg_no)
        if existing is None or (_row_value(row, "update_key", -1) or -1) > (_row_value(existing, "update_key", -1) or -1):
            current_by_leg[leg_no] = row

    requirements = []
    for leg_no, change_types in change_types_by_leg.items():
        current_row = current_by_leg.get(leg_no)
        if current_row is None:
            continue
        requirements.append(
            DirtyEventRequirement(
                leg_no=leg_no,
                dep_ap_sched=_row_value(current_row, "dep_ap_sched"),
                dirty_event_date=_as_event_date(_row_value(current_row, "dep_sched_dt")),
                source_alias="leg_times",
                dirty_sides=("current",),
                cdf_change_types=_sorted_change_types(change_types),
                dirty_reason="taxi_out_dirty_leg_times_change_current_mapping",
                limitation=LEG_TIMES_ONLY_MAPPING_LIMITATION,
            )
        )

    return tuple(sorted(requirements, key=lambda req: (str(req.leg_no), req.dep_ap_sched, req.dirty_event_date)))


def expand_dirty_event_requirements_to_affected_pairs(
    requirements: Iterable[DirtyEventRequirement],
) -> tuple[dict[str, Any], ...]:
    affected: dict[tuple[str, date], dict[str, Any]] = {}
    for requirement in requirements:
        for offset in range(1, 31):
            affected_date = requirement.dirty_event_date + timedelta(days=offset)
            key = (requirement.dep_ap_sched, affected_date)
            bucket = affected.setdefault(
                key,
                {
                    "dep_ap_sched": requirement.dep_ap_sched,
                    "event_date": affected_date,
                    "dirty_event_dates": set(),
                    "dirty_leg_nos": set(),
                    "dirty_source_aliases": set(),
                },
            )
            bucket["dirty_event_dates"].add(requirement.dirty_event_date)
            bucket["dirty_leg_nos"].add(requirement.leg_no)
            bucket["dirty_source_aliases"].add(requirement.source_alias)

    rows = []
    for row in affected.values():
        rows.append(
            {
                "dep_ap_sched": row["dep_ap_sched"],
                "event_date": row["event_date"],
                "dirty_event_dates": tuple(sorted(row["dirty_event_dates"])),
                "dirty_leg_nos": tuple(sorted(row["dirty_leg_nos"], key=str)),
                "dirty_source_aliases": tuple(sorted(row["dirty_source_aliases"])),
            }
        )
    return tuple(sorted(rows, key=lambda row: (row["dep_ap_sched"], row["event_date"])))


def validate_dev_shadow_table_name(table_name: str, *, expected_suffix_or_token: str) -> None:
    """Validate that a write target is a dev-only shadow/control table."""
    if not table_name or not table_name.strip():
        raise ValueError("table_name must not be empty.")

    normalised = _normalise_table_name(table_name)
    expected = expected_suffix_or_token.lower()

    if not normalised.startswith(DEV_ML_OPS_PREFIX):
        raise ValueError(f"Stage 30C write targets must be under {DEV_ML_OPS_PREFIX}*. Got: {table_name}")
    if normalised == CURRENT_TAXI_OUT_MV:
        raise ValueError("The current ft_airport_daily_taxi_out materialized view is read-only for Stage 30C-1.")
    if any(source_table in normalised for source_table in PRODUCTION_SOURCE_TABLES):
        raise ValueError("Stage 30C-1 must never target source tables.")
    if expected not in normalised:
        raise ValueError(f"Expected token {expected_suffix_or_token!r} in dev shadow/control table: {table_name}")
    if "shadow" in expected and "shadow" not in normalised:
        raise ValueError("Shadow target table names must contain 'shadow'.")
    if "watermark" in expected and "watermark" not in normalised:
        raise ValueError("Watermark table names must contain 'watermark'.")


def validate_source_cdf_window(
    source_alias: str,
    starting_version: int | None,
    ending_version: int | None,
) -> SourceCdfWindow:
    """Validate one source-specific CDF window."""
    if source_alias not in SOURCE_ALIASES:
        raise ValueError(f"Unsupported source alias {source_alias!r}. Expected one of {SOURCE_ALIASES}.")
    if ending_version is not None and starting_version is None:
        raise ValueError(f"{source_alias} ending version requires a starting version.")
    if starting_version is not None and int(starting_version) < 0:
        raise ValueError(f"{source_alias} starting version must be non-negative.")
    if ending_version is not None and int(ending_version) < 0:
        raise ValueError(f"{source_alias} ending version must be non-negative.")
    if starting_version is not None and ending_version is not None and int(ending_version) < int(starting_version):
        raise ValueError(f"{source_alias} ending version must be >= starting version.")

    return SourceCdfWindow(
        source_alias=source_alias,
        starting_version=None if starting_version is None else int(starting_version),
        ending_version=None if ending_version is None else int(ending_version),
    )


def validate_source_cdf_windows(
    *,
    leg_starting_version: int | None,
    leg_ending_version: int | None,
    leg_times_starting_version: int | None,
    leg_times_ending_version: int | None,
    require_configured: bool = False,
    require_all_sources: bool = False,
) -> dict[str, SourceCdfWindow]:
    """Validate independent CDF windows for leg and leg_times.

    There is deliberately no global CDF version input because Delta commit
    versions are source-table specific.
    """
    windows = {
        "leg": validate_source_cdf_window("leg", leg_starting_version, leg_ending_version),
        "leg_times": validate_source_cdf_window(
            "leg_times",
            leg_times_starting_version,
            leg_times_ending_version,
        ),
    }
    configured = [window.source_alias for window in windows.values() if window.is_configured]
    if require_configured and not configured:
        raise ValueError("At least one source-specific CDF starting version must be set.")
    if require_all_sources and set(configured) != set(SOURCE_ALIASES):
        raise ValueError("Both leg and leg_times require separate source-specific CDF versions.")
    return windows


def configured_cdf_windows(windows: Mapping[str, SourceCdfWindow]) -> dict[str, SourceCdfWindow]:
    return {source_alias: window for source_alias, window in windows.items() if window.is_configured}


def build_next_starting_versions(windows: Mapping[str, SourceCdfWindow]) -> dict[str, int]:
    """Return next starting versions for a future watermark-driven run."""
    next_versions: dict[str, int] = {}
    for source_alias, window in windows.items():
        if not window.is_configured:
            continue
        processed_version = window.ending_version if window.ending_version is not None else window.starting_version
        if processed_version is None:
            continue
        next_versions[source_alias] = int(processed_version) + 1
    return next_versions


def _normalise_affected_pairs(affected_pairs_df: "DataFrame") -> "DataFrame":
    F = _pyspark_sql()
    columns = set(affected_pairs_df.columns)
    if all(column in columns for column in TARGET_KEY_COLS):
        return affected_pairs_df.select(*TARGET_KEY_COLS).dropDuplicates()
    if "dep_ap_sched" in columns and "affected_output_date" in columns:
        return (
            affected_pairs_df.select(
                F.col("dep_ap_sched"),
                F.col("affected_output_date").alias("event_date"),
            )
            .select(*TARGET_KEY_COLS)
            .dropDuplicates()
        )
    raise ValueError("affected_pairs_df must contain dep_ap_sched and event_date or affected_output_date.")


def build_shadow_replace_source(affected_pairs_df: "DataFrame", candidate_df: "DataFrame") -> "DataFrame":
    """Build the single-MERGE source for affected-key replacement into the shadow table."""
    F = _pyspark_sql()
    missing_candidate_keys = [column for column in TARGET_KEY_COLS if column not in candidate_df.columns]
    if missing_candidate_keys:
        raise ValueError(f"candidate_df is missing target key columns: {missing_candidate_keys}")

    affected_pairs = _normalise_affected_pairs(affected_pairs_df)
    candidate_with_marker = candidate_df.withColumn("_stage30c_candidate_present", F.lit(True))
    merge_source = affected_pairs.join(candidate_with_marker, on=list(TARGET_KEY_COLS), how="left")
    return (
        merge_source.withColumn(
            SHADOW_CANDIDATE_FLAG_COL,
            F.coalesce(F.col("_stage30c_candidate_present"), F.lit(False)),
        )
        .drop("_stage30c_candidate_present")
        .select(
            *TARGET_KEY_COLS,
            *[column for column in candidate_df.columns if column not in TARGET_KEY_COLS],
            SHADOW_CANDIDATE_FLAG_COL,
        )
    )


def build_shadow_replace_source_rows(
    affected_pairs: Iterable[Mapping[str, Any]],
    candidate_rows: Iterable[Mapping[str, Any]],
) -> tuple[dict[str, Any], ...]:
    """Pure-Python deterministic equivalent of the shadow merge source builder."""
    affected_by_key: dict[tuple[Any, Any], dict[str, Any]] = {}
    for row in affected_pairs:
        key = _row_key(row)
        affected_by_key[key] = {
            "dep_ap_sched": key[0],
            "event_date": key[1],
        }

    candidate_by_key: dict[tuple[Any, Any], Mapping[str, Any]] = {}
    for row in candidate_rows:
        key = _row_key(row)
        if key in candidate_by_key:
            raise ValueError(f"Duplicate candidate key in shadow merge source: {key}")
        candidate_by_key[key] = row

    merge_rows = []
    for key in sorted(affected_by_key, key=lambda item: (str(item[0]), item[1])):
        candidate = candidate_by_key.get(key)
        if candidate is None:
            merge_rows.append(
                {
                    "dep_ap_sched": key[0],
                    "event_date": key[1],
                    SHADOW_CANDIDATE_FLAG_COL: False,
                }
            )
            continue

        row = dict(candidate)
        row["dep_ap_sched"] = key[0]
        row["event_date"] = key[1]
        row[SHADOW_CANDIDATE_FLAG_COL] = True
        merge_rows.append(row)

    return tuple(merge_rows)


def validate_shadow_merge_source(
    merge_source_rows: Iterable[Mapping[str, Any]],
    *,
    affected_pairs: Iterable[Mapping[str, Any]] | None = None,
) -> ShadowMergeSourceValidation:
    """Validate the merge source used for update/insert/delete affected-key replacement."""
    rows = tuple(merge_source_rows)
    affected_keys = None
    if affected_pairs is not None:
        affected_keys = {_row_key(row) for row in affected_pairs}

    seen_keys: set[tuple[Any, Any]] = set()
    delete_count = 0
    candidate_count = 0
    for row in rows:
        if SHADOW_CANDIDATE_FLAG_COL not in row:
            raise ValueError(f"Merge source row is missing {SHADOW_CANDIDATE_FLAG_COL}.")
        key = _row_key(row)
        if key in seen_keys:
            raise ValueError(f"Duplicate shadow merge source key: {key}")
        seen_keys.add(key)
        if affected_keys is not None and key not in affected_keys:
            raise ValueError(f"Shadow merge source key is outside affected pairs: {key}")

        has_candidate = bool(_row_value(row, SHADOW_CANDIDATE_FLAG_COL))
        if has_candidate:
            candidate_count += 1
        else:
            delete_count += 1
            if affected_keys is not None and key not in affected_keys:
                raise ValueError(f"Delete branch key is outside affected pairs: {key}")

    return ShadowMergeSourceValidation(
        row_count=len(rows),
        key_count=len(seen_keys),
        candidate_row_count=candidate_count,
        delete_row_count=delete_count,
    )


def build_create_shadow_table_sql(
    *,
    shadow_table: str = DEFAULT_SHADOW_TAXI_OUT_TABLE,
    current_mv_table: str = CURRENT_TAXI_OUT_MV,
    if_not_exists: bool = True,
) -> str:
    validate_dev_shadow_table_name(shadow_table, expected_suffix_or_token="shadow")
    if _normalise_table_name(current_mv_table) != CURRENT_TAXI_OUT_MV:
        raise ValueError("Stage 30C-1 shadow initialization must copy from the current taxi-out MV only.")
    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    return (
        f"CREATE TABLE {exists_clause}{quote_table_name(shadow_table)}\n"
        f"AS SELECT * FROM {quote_table_name(current_mv_table)}"
    )


def build_create_watermark_table_sql(
    *,
    watermark_table: str = DEFAULT_WATERMARK_TABLE,
    if_not_exists: bool = True,
) -> str:
    validate_dev_shadow_table_name(watermark_table, expected_suffix_or_token="watermark")
    exists_clause = "IF NOT EXISTS " if if_not_exists else ""
    return f"""CREATE TABLE {exists_clause}{quote_table_name(watermark_table)} (
  stage_name STRING,
  source_alias STRING,
  source_table STRING,
  last_processed_version BIGINT,
  last_processed_timestamp TIMESTAMP,
  last_successful_run_id STRING,
  updated_at TIMESTAMP,
  status STRING
)
USING DELTA"""


def validate_merge_columns(candidate_columns: Sequence[str]) -> tuple[str, ...]:
    columns = tuple(candidate_columns)
    missing = [column for column in TARGET_KEY_COLS if column not in columns]
    if missing:
        raise ValueError(f"candidate_columns must include target key columns: {missing}")
    if SHADOW_CANDIDATE_FLAG_COL in columns:
        raise ValueError(f"{SHADOW_CANDIDATE_FLAG_COL} is generated by the merge source and is not a feature column.")
    return columns


def build_shadow_merge_sql(
    *,
    shadow_table: str = DEFAULT_SHADOW_TAXI_OUT_TABLE,
    merge_source_view: str,
    candidate_columns: Sequence[str],
) -> str:
    validate_dev_shadow_table_name(shadow_table, expected_suffix_or_token="shadow")
    columns = validate_merge_columns(candidate_columns)
    non_key_columns = [column for column in columns if column not in TARGET_KEY_COLS]
    if not non_key_columns:
        raise ValueError("candidate_columns must include at least one non-key feature column.")

    target = quote_table_name(shadow_table)
    source = quote_table_name(merge_source_view)
    merge_condition = " AND ".join(f"target.{_quote_identifier(column)} = source.{_quote_identifier(column)}" for column in TARGET_KEY_COLS)
    update_assignments = ",\n    ".join(
        f"target.{_quote_identifier(column)} = source.{_quote_identifier(column)}" for column in non_key_columns
    )
    insert_columns = ", ".join(_quote_identifier(column) for column in columns)
    insert_values = ", ".join(f"source.{_quote_identifier(column)}" for column in columns)

    return f"""MERGE INTO {target} AS target
USING {source} AS source
ON {merge_condition}
WHEN MATCHED AND source.{_quote_identifier(SHADOW_CANDIDATE_FLAG_COL)} = true THEN UPDATE SET
    {update_assignments}
WHEN MATCHED AND source.{_quote_identifier(SHADOW_CANDIDATE_FLAG_COL)} = false THEN DELETE
WHEN NOT MATCHED AND source.{_quote_identifier(SHADOW_CANDIDATE_FLAG_COL)} = true THEN INSERT ({insert_columns})
VALUES ({insert_values})"""


def build_advance_watermark_sql(
    *,
    watermark_table: str = DEFAULT_WATERMARK_TABLE,
    stage_name: str,
    source_alias: str,
    source_table: str,
    last_processed_version: int,
    last_processed_timestamp: str | None,
    last_successful_run_id: str,
    status: str = "success",
) -> str:
    validate_dev_shadow_table_name(watermark_table, expected_suffix_or_token="watermark")
    if source_alias not in SOURCE_ALIASES:
        raise ValueError(f"Unsupported source alias {source_alias!r}.")

    target = quote_table_name(watermark_table)
    return f"""MERGE INTO {target} AS target
USING (
  SELECT
    {_sql_string(stage_name)} AS stage_name,
    {_sql_string(source_alias)} AS source_alias,
    {_sql_string(source_table)} AS source_table,
    {_sql_int_or_null(last_processed_version)} AS last_processed_version,
    {_sql_timestamp_or_null(last_processed_timestamp)} AS last_processed_timestamp,
    {_sql_string(last_successful_run_id)} AS last_successful_run_id,
    current_timestamp() AS updated_at,
    {_sql_string(status)} AS status
) AS source
ON target.stage_name = source.stage_name AND target.source_alias = source.source_alias
WHEN MATCHED THEN UPDATE SET
    target.source_table = source.source_table,
    target.last_processed_version = source.last_processed_version,
    target.last_processed_timestamp = source.last_processed_timestamp,
    target.last_successful_run_id = source.last_successful_run_id,
    target.updated_at = source.updated_at,
    target.status = source.status
WHEN NOT MATCHED THEN INSERT (
    stage_name,
    source_alias,
    source_table,
    last_processed_version,
    last_processed_timestamp,
    last_successful_run_id,
    updated_at,
    status
) VALUES (
    source.stage_name,
    source.source_alias,
    source.source_table,
    source.last_processed_version,
    source.last_processed_timestamp,
    source.last_successful_run_id,
    source.updated_at,
    source.status
)"""


def build_advance_watermark_sqls(
    *,
    watermark_table: str = DEFAULT_WATERMARK_TABLE,
    stage_name: str,
    source_tables_by_alias: Mapping[str, str],
    processed_versions_by_alias: Mapping[str, int],
    processed_timestamps_by_alias: Mapping[str, str | None],
    last_successful_run_id: str,
) -> tuple[str, ...]:
    sqls = []
    for source_alias in SOURCE_ALIASES:
        if source_alias not in processed_versions_by_alias:
            continue
        sqls.append(
            build_advance_watermark_sql(
                watermark_table=watermark_table,
                stage_name=stage_name,
                source_alias=source_alias,
                source_table=source_tables_by_alias[source_alias],
                last_processed_version=processed_versions_by_alias[source_alias],
                last_processed_timestamp=processed_timestamps_by_alias.get(source_alias),
                last_successful_run_id=last_successful_run_id,
            )
        )
    return tuple(sqls)


def require_shadow_write_confirmation(
    *,
    dry_run_only: bool,
    write_confirmation: str,
    required_write_confirmation: str,
    write_flags: Mapping[str, bool],
) -> None:
    enabled_flags = [flag_name for flag_name, enabled in write_flags.items() if enabled]
    if not enabled_flags and dry_run_only:
        return
    if dry_run_only and enabled_flags:
        raise ValueError(f"Write flags require DRY_RUN_ONLY = False. Enabled flags: {enabled_flags}")
    if write_confirmation != required_write_confirmation:
        raise ValueError(
            "Stage 30C-1 dev-shadow writes require the exact confirmation string. "
            f"Enabled write flags: {enabled_flags}"
        )


def validate_watermark_advance_preconditions(
    *,
    shadow_merge_executed: bool,
    shadow_post_merge_validation_ok: bool,
    processed_versions_by_alias: Mapping[str, int | None],
    write_confirmation: str,
    required_write_confirmation: str,
    configured_source_aliases: Iterable[str] = SOURCE_ALIASES,
    candidate_duplicate_key_count: int = 0,
    candidate_null_key_count: int = 0,
    shadow_duplicate_key_count: int = 0,
    shadow_null_key_count: int = 0,
    compare_failed: bool = False,
    watermark_advance_requested: bool = True,
) -> bool:
    """Validate that source-specific watermarks may advance after shadow success."""
    if not watermark_advance_requested:
        return True
    if write_confirmation != required_write_confirmation:
        raise ValueError("Watermark advancement requires the exact dev-shadow write confirmation string.")
    if not shadow_merge_executed:
        raise ValueError("Watermark advancement requires shadow_merge_executed = True.")
    if not shadow_post_merge_validation_ok:
        raise ValueError("Watermark advancement requires successful post-merge shadow validation.")
    if candidate_duplicate_key_count:
        raise ValueError("Watermark advancement blocked by duplicate candidate keys.")
    if candidate_null_key_count:
        raise ValueError("Watermark advancement blocked by null candidate keys.")
    if shadow_duplicate_key_count:
        raise ValueError("Watermark advancement blocked by duplicate shadow keys.")
    if shadow_null_key_count:
        raise ValueError("Watermark advancement blocked by null shadow keys.")
    if compare_failed:
        raise ValueError("Watermark advancement blocked by failed compare status.")

    configured = tuple(configured_source_aliases)
    if not configured:
        raise ValueError("Watermark advancement requires at least one configured source alias.")
    unsupported = [source_alias for source_alias in configured if source_alias not in SOURCE_ALIASES]
    if unsupported:
        raise ValueError(f"Unsupported configured source aliases for watermark advancement: {unsupported}")
    missing_versions = [
        source_alias
        for source_alias in configured
        if source_alias not in processed_versions_by_alias or processed_versions_by_alias[source_alias] is None
    ]
    if missing_versions:
        raise ValueError(f"Watermark advancement requires source-specific latest versions for: {missing_versions}")

    return True


def ensure_expected_columns(available_columns: Iterable[str], required_columns: Iterable[str]) -> tuple[str, ...]:
    available = set(available_columns)
    return tuple(column for column in required_columns if column not in available)


def is_shadow_or_control_table(table_name: str) -> bool:
    normalised = _normalise_table_name(table_name)
    return normalised.startswith(DEV_ML_OPS_PREFIX) and ("shadow" in normalised or "watermark" in normalised)


def describe_shadow_write_plan(
    *,
    shadow_table: str,
    watermark_table: str,
    write_flags: Mapping[str, bool],
    dry_run_only: bool,
) -> list[dict[str, Any]]:
    validate_dev_shadow_table_name(shadow_table, expected_suffix_or_token="shadow")
    validate_dev_shadow_table_name(watermark_table, expected_suffix_or_token="watermark")
    return [
        {
            "operation": "create_shadow_table",
            "target": shadow_table,
            "enabled": bool(write_flags.get("ALLOW_CREATE_SHADOW_TABLE", False)),
            "dry_run_only": dry_run_only,
        },
        {
            "operation": "create_watermark_table",
            "target": watermark_table,
            "enabled": bool(write_flags.get("ALLOW_CREATE_WATERMARK_TABLE", False)),
            "dry_run_only": dry_run_only,
        },
        {
            "operation": "shadow_merge_replace_affected_keys",
            "target": shadow_table,
            "enabled": bool(write_flags.get("ALLOW_SHADOW_MERGE", False)),
            "dry_run_only": dry_run_only,
        },
        {
            "operation": "advance_source_specific_watermarks",
            "target": watermark_table,
            "enabled": bool(write_flags.get("ALLOW_WATERMARK_ADVANCE", False)),
            "dry_run_only": dry_run_only,
        },
    ]
