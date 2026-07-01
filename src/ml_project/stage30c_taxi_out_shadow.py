from __future__ import annotations

from dataclasses import dataclass
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


@dataclass(frozen=True)
class SourceCdfWindow:
    source_alias: str
    starting_version: int | None
    ending_version: int | None

    @property
    def is_configured(self) -> bool:
        return self.starting_version is not None


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
