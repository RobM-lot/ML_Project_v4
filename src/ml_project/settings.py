from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import PurePosixPath
from typing import Any, Dict, List, Optional

_ALLOWED_ENVS = {"dev", "prod"}


@dataclass
class FlightDelaySettings:
    # Runtime
    ENV: str
    ALLOW_CHECKPOINT_RESET: bool

    # Workspace / project metadata
    PROJECT_ROOT: str
    SRC_PATH: str
    WORKSPACE_OWNER: str
    PROJECT_NAME: str

    # Catalogs / schemas
    CATALOG: str
    SCHEMA: str

    SOURCE_CATALOG: str
    SOURCE_SCHEMA: str

    SILVER_CATALOG: str
    SILVER_SCHEMA: str

    GOLD_CATALOG: str
    GOLD_SCHEMA: str

    # Paths / model registry
    CHECKPOINT_PATH: str
    UC_MODEL_NAME: str
    MODEL_URI: str
    EXPERIMENTS_ROOT: str
    EXPERIMENT_PATH: str
    MODEL_ALIAS: str = "champion"

    # Tables
    LABELS_TABLE: str = ""
    LEG_TIMES_TABLE: str = ""
    AP_BASICS_TABLE: str = ""
    TIME_ZONE_TABLE: str = ""
    LEG_MISC_TABLE: str = ""
    LEG_REMARK_TABLE: str = ""
    FS_TAXI_OUT_TABLE: str = ""
    FS_AIRBORNE_TABLE: str = ""
    FS_TAXI_IN_TABLE: str = ""
    FS_STAND_OUT_TABLE: str = ""
    FS_STAND_IN_TABLE: str = ""
    SINK_TABLE: str = ""
    EVENTS_SINK_TABLE: str = ""
    SHADOW_TABLE: str = ""
    TRAINING_DATASET_TABLE: str = ""
    EVAL_CLEAN_DATASET_TABLE: str = ""
    EVAL_ALL_DATASET_TABLE: str = ""

    # Iter2 - nowy layout ft_*
    FT_LEG_STATUS_TABLE: str = ""
    FT_LEG_TIMES_TABLE: str = ""
    FT_LEG_MISC_TABLE: str = ""
    FT_AIRPORT_TIMEZONE_TABLE: str = ""
    FT_ROUTE_DAILY_STATS_TABLE: str = ""
    FT_AIRPORT_DAILY_TAXI_OUT_TABLE: str = ""
    FT_AIRPORT_DAILY_TAXI_IN_TABLE: str = ""
    FT_STAND_DAILY_OUT_TABLE: str = ""
    FT_STAND_DAILY_IN_TABLE: str = ""


    # Iter2.5: on-demand UDF dla local time (Opcja A — parytet z enriched() v9)
    UC_FN_LOCAL_HOUR: str = ""
    UC_FN_LOCAL_DOW: str = ""
    UC_FN_MONTH_OF: str = ""
    UC_FN_SIN_LOCAL_HOUR: str = ""
    UC_FN_COS_LOCAL_HOUR: str = ""
    UC_FN_SIN_LOCAL_DOW: str = ""
    UC_FN_COS_LOCAL_DOW: str = ""
    UC_FN_SIN_MONTH_OF: str = ""
    UC_FN_COS_MONTH_OF: str = ""

    # Iter2.5: on-demand UDF geo (haversine + eastbound)
    UC_FN_HAVERSINE_KM: str = ""
    UC_FN_IS_EASTBOUND: str = ""

    # Keys / business rules
    PK_TAXI_OUT: List[str] = field(default_factory=lambda: ["dep_ap_sched"])
    PK_AIRBORNE: List[str] = field(default_factory=lambda: ["route_id"])
    PK_TAXI_IN: List[str] = field(default_factory=lambda: ["arr_ap_sched"])
    PK_STAND_OUT: List[str] = field(default_factory=lambda: ["stand_id"])
    PK_STAND_IN: List[str] = field(default_factory=lambda: ["stand_id"])

    # Iter2 - PK dla nowego layoutu ft_*
    PK_FT_AIRPORT_TAXI_OUT: List[str] = field(default_factory=lambda: ["dep_ap_sched"])
    PK_FT_AIRPORT_TAXI_IN: List[str] = field(default_factory=lambda: ["arr_ap_sched"])
    PK_FT_ROUTE: List[str] = field(default_factory=lambda: ["route_id"])
    PK_FT_STAND_OUT: List[str] = field(default_factory=lambda: ["stand_id_out"])
    PK_FT_STAND_IN: List[str] = field(default_factory=lambda: ["stand_id_in"])

    FS_TIMESTAMP_KEY: str = "event_date"
    SINK_PRIMARY_KEY: str = "leg_no"
    SPARK_TIMEZONE: str = "UTC"

    PREFERRED_AIRCRAFT_COL: str = "ac_registration"
    MODEL_AIRCRAFT_FEATURE_COL: str = "ac_registration"
    AC_REGISTRATION_PREFIX_LEN: int = 4

    LEG_STATE_ARR: str = "ARR"
    INCLUDED_LEG_TYPES: List[str] = field(default_factory=lambda: ["J", "C", "G"])
    TARGET_COLS: List[str] = field(default_factory=lambda: ["taxi_out_sec", "airborne_sec", "taxi_in_sec", "actual_block_time_sec"])
    REQUIRES_DAYS_HISTORY: int = 180
    HISTORY_START: str = "2023-07-01"
    LABEL_START: str = "2024-01-01"
    DATA_CUTOFF_DATE: str = "2027-01-01"

    # CDF / scoring controls
    SHADOW_SYNC_LOOKBACK_DAYS: int = 2
    SHADOW_SYNC_LOOKAHEAD_MONTHS: int = 3
    CDF_MAX_FILES_PER_TRIGGER: int = 1000

    # Data quality thresholds
    MIN_VALID_TIME_SEC: int = 0
    MAX_VALID_SCHED_BLOCK_SEC: int = 20 * 3600
    MAX_VALID_BLOCK_SEC: int = 20 * 3600
    MAX_VALID_TAXI_OUT_SEC: int = 3 * 3600
    MAX_VALID_AIRBORNE_SEC: int = 18 * 3600
    MAX_VALID_TAXI_IN_SEC: int = 2 * 3600
    MAX_SEGMENT_SUM_GAP_SEC: int = 15 * 60
    MAX_MISSING_FEATURES: int = 10
    MAX_TRAINING_DQ_DROP_RATIO: float = 0.20

    # Modelowanie i walidacja
    EVAL_DAYS: int = 30
    MIN_SLICE_COUNT: int = 30
    TOP_AIRPORTS_N: int = 12
    BIAS_PENALTY_WEIGHT: float = 0.10
    CV_PARAM_GRID: List[Dict[str, Any]] = field(default_factory=list)
    CV_N_SPLITS: int = 3
    REQUIRE_BASELINE_BEAT: bool = True
    REQUIRE_NO_DRIFT: bool = False

    # Training runtime guards
    MAX_TRAINING_PANDAS_ROWS: int = 1_500_000
    MIN_CALIBRATION_ROWS: int = 100
    CALIBRATION_DAYS: int = 30
    BLOCK_CQR_USE_SCHED_BUCKETS: bool = True
    BLOCK_CQR_MIN_BUCKET_ROWS: int = 100
    BLOCK_CQR_BUCKETS: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "short_lt_90m", "min_sec": 0, "max_sec": 90 * 60},
        {"name": "medium_90m_180m", "min_sec": 90 * 60, "max_sec": 180 * 60},
        {"name": "long_180m_360m", "min_sec": 180 * 60, "max_sec": 360 * 60},
        {"name": "ultra_ge_360m", "min_sec": 360 * 60, "max_sec": None},
    ])
    TOP_K_MODEL_AIRCRAFT: int = 250
    TOP_K_DEP_AIRPORTS: int = 50
    TOP_K_ARR_AIRPORTS: int = 50
    TOP_K_DEP_STANDS: int = 150
    TOP_K_ARR_STANDS: int = 150
    TRAINING_RUN_NAME: str = "Segmented_CV_Quantile_v4"
    REGISTER_REQUIRED_METRICS: List[str] = field(default_factory=lambda: [
        "TOTAL_MAE_actual_block_time",
        "TOTAL_ABS_BIAS_actual_block_time",
        "TOTAL_P90_COVERAGE_actual_block_time",
        "HOLDOUT_WIN_RATE_VS_SCHEDULE_pct",
        "OVERFIT_DIFF_actual_block_time_sec",
    ])
    REGISTER_REQUIRED_PARAMS: List[str] = field(default_factory=lambda: [
        "training_dataset_table",
        "model_aircraft_feature_col",
        "valid_rows_holdout_split",
    ])

    # Registry / promotion gates
    REGISTER_RUN_ID_WIDGET_DEFAULT: str = ""
    PROMOTE_IF_PASS_DEFAULT: bool = True
    PROMOTION_MAX_ABS_BIAS_MIN: float = 3.0
    PROMOTION_MIN_P90_COVERAGE_PCT: float = 85.0
    PROMOTION_MAX_P90_COVERAGE_PCT: float = 97.0
    PROMOTION_MIN_HOLDOUT_WIN_RATE_PCT: float = 55.0
    PROMOTION_MAX_OVERFIT_DIFF_MIN: float = 3.0
    PROMOTION_MAX_MONTHLY_MAE_STD_MIN: float = 4.0
    PROMOTION_MIN_VALID_ROWS: int = 500
    PROMOTION_REQUIRE_BASELINE_BEAT: bool = True
    PROMOTION_REQUIRE_CHAMPION_BEAT: bool = False

    # Features
    TIME_WINDOWS: List[str] = field(default_factory=lambda: ["7d", "30d"])
    MAX_MARKER_LENGTH: int = 17
    CATEGORICAL_FEATURES: List[str] = field(default_factory=list)
    MARKER_COLS: List[str] = field(default_factory=list)
    FEATURES_TAXI_OUT: List[str] = field(default_factory=list)
    FEATURES_AIRBORNE: List[str] = field(default_factory=list)
    FEATURES_TAXI_IN: List[str] = field(default_factory=list)
    ALL_FS_FEATURES: List[str] = field(default_factory=list)
    SECONDS_IN_DAY: int = 86400
    HALF_LIFE_DAYS: Dict[str, int] = field(default_factory=lambda: {"7d": 7, "30d": 30})


def validate_env(env: str) -> str:
    normalized = (env or "").strip().lower()
    if normalized not in _ALLOWED_ENVS:
        raise ValueError(f"Nieobsługiwane środowisko: {env!r}. Dozwolone: {sorted(_ALLOWED_ENVS)}")
    return normalized


def _marker_cols(max_marker_length: int) -> List[str]:
    return [f"marker_{i}" for i in range(1, max_marker_length + 1)]


def _derive_workspace_metadata(project_root: Optional[str]) -> Dict[str, str]:
    if not project_root:
        return {
            "PROJECT_ROOT": "",
            "SRC_PATH": "",
            "WORKSPACE_OWNER": "",
            "PROJECT_NAME": "",
            "EXPERIMENTS_ROOT": "",
        }

    path = PurePosixPath(project_root)
    parts = list(path.parts)

    owner = ""
    if "Users" in parts:
        idx = parts.index("Users")
        if idx + 1 < len(parts):
            owner = parts[idx + 1]

    project_name = path.name

    if ".bundle" in parts:
        bidx = parts.index(".bundle")
        if bidx + 1 < len(parts):
            project_name = parts[bidx + 1]

    experiments_root = ""
    if owner and project_name:
        experiments_root = str(PurePosixPath("/Users") / owner / project_name / "Experiments")

    return {
        "PROJECT_ROOT": str(path),
        "SRC_PATH": str(path / "src"),
        "WORKSPACE_OWNER": owner,
        "PROJECT_NAME": project_name,
        "EXPERIMENTS_ROOT": experiments_root,
    }

def load_settings(
    env: str,
    project_root: Optional[str] = None,
    *,
    source_catalog_override: Optional[str] = None,
    source_schema_override: Optional[str] = None,
    target_catalog: Optional[str] = None,
    target_schema: Optional[str] = None,
    silver_catalog_override: Optional[str] = None,
    silver_schema_override: Optional[str] = None,
    gold_catalog_override: Optional[str] = None,
    gold_schema_override: Optional[str] = None,
) -> FlightDelaySettings:
    env = validate_env(env)
    metadata = _derive_workspace_metadata(project_root)

    source_catalog = (source_catalog_override or "panda_silver_prod").strip()
    source_schema = (source_schema_override or "occ_ops").strip()

    if env == "prod":
        _prod_targets = {
            "silver_catalog": silver_catalog_override or target_catalog or "",
            "silver_schema": silver_schema_override or target_schema or "",
            "gold_catalog": gold_catalog_override or "",
            "gold_schema": gold_schema_override or "",
        }
        _bad_placeholders = {
            name: value for name, value in _prod_targets.items()
            if value and "__TODO" in value.upper()
        }
        if _bad_placeholders:
            raise RuntimeError(
                f"Wykryto niewypełnione placeholdery w konfiguracji prod: "
                f"{_bad_placeholders}. "
                "Uzupełnij databricks.yml (target: prod) przed deploymentem. "
                "NIE zgaduj wartości — potwierdź z zespołem DBA/platform."
            )

    silver_catalog = (silver_catalog_override or target_catalog or f"panda_silver_{env}").strip()
    silver_schema = (silver_schema_override or target_schema or "ml_ops").strip()

    gold_catalog = (gold_catalog_override or f"panda_gold_{env}").strip()
    gold_schema = (gold_schema_override or "ml_ops").strip()

    catalog = silver_catalog
    schema = silver_schema

    checkpoint_path = f"/Volumes/{silver_catalog}/{silver_schema}/ml_checkpoints/flight_delay_cdf_{env}"

    uc_model_name = f"{gold_catalog}.{gold_schema}.flight_delay_model"
    model_alias = "champion"
    model_uri = f"models:/{uc_model_name}@{model_alias}"

    experiment_path = (
        f"{metadata['EXPERIMENTS_ROOT']}/block_time_auto_segmented_{env}_v4"
        if metadata["EXPERIMENTS_ROOT"]
        else f"block_time_auto_segmented_{env}_v4"
    )

    marker_cols = _marker_cols(17)
    preferred_aircraft_col = "ac_registration"
    model_aircraft_feature_col = "ac_registration"

    features_taxi_out = [
        "dep_ap_sched", "dep_stand", model_aircraft_feature_col, "leg_type", "avg_taxi_out_7d", "avg_taxi_out_30d",
        "std_taxi_out_7d", "std_taxi_out_30d", "trend_taxi_out_7d",
        "delta_ema_avg_taxi_out_7d", "delta_ema_avg_taxi_out_30d",
        "p90_taxi_out_7d", "p90_taxi_out_30d", "avg_dur_ratio_dep_7d",
        "avg_dur_ratio_dep_30d", "count_dep_30d", "has_hist_dep_30d",
        "count_dep_7d", "delta_ema_avg_dur_ratio_dep_30d", "delta_ema_avg_dur_ratio_dep_7d",
        "ema_confidence_dep_30d", "ema_confidence_dep_7d", "ema_dur_ratio_dep_30d",
        "ema_dur_ratio_dep_7d", "ema_taxi_out_30d", "ema_taxi_out_7d",
        "has_hist_dep_7d", "max_dur_ratio_dep_30d", "max_dur_ratio_dep_7d",
        "max_taxi_out_30d", "max_taxi_out_7d", "min_dur_ratio_dep_30d",
        "min_dur_ratio_dep_7d", "min_taxi_out_30d", "min_taxi_out_7d",
        "p90_dur_ratio_dep_30d", "p90_dur_ratio_dep_7d", "std_dur_ratio_dep_30d",
        "std_dur_ratio_dep_7d", "trend_dur_ratio_dep_7d",
        "stand_count_out_7d", "stand_avg_taxi_out_7d", "stand_trend_taxi_out_7d",
        "stand_count_out_30d", "stand_avg_taxi_out_30d", "stand_p10_taxi_out_30d",
        "stand_p50_taxi_out_30d", "stand_p90_taxi_out_30d", "stand_std_taxi_out_30d",
        "isLO", "local_hour_dep", "local_dow_dep",
        "sin_local_hour_dep", "cos_local_hour_dep", "sin_local_dow_dep", "cos_local_dow_dep", "sin_month", "cos_month",
    ] + marker_cols

    features_airborne = [
        "dep_ap_sched", "arr_ap_sched", model_aircraft_feature_col, "leg_type", "avg_airborne_7d", "avg_airborne_30d",
        "std_airborne_7d", "std_airborne_30d", "trend_airborne_7d",
        "delta_ema_avg_airborne_7d", "delta_ema_avg_airborne_30d",
        "avg_dur_ratio_route_7d", "avg_dur_ratio_route_30d", "avg_arrival_delay_7d",
        "avg_arrival_delay_30d", "std_arrival_delay_7d", "std_arrival_delay_30d",
        "count_route_30d", "has_hist_route_30d", "count_route_7d",
        "delta_ema_avg_arrival_delay_30d", "delta_ema_avg_arrival_delay_7d",
        "delta_ema_avg_dur_ratio_route_30d", "delta_ema_avg_dur_ratio_route_7d",
        "ema_airborne_30d", "ema_airborne_7d", "ema_arrival_delay_30d",
        "ema_arrival_delay_7d", "ema_confidence_route_30d", "ema_confidence_route_7d",
        "ema_dur_ratio_route_30d", "ema_dur_ratio_route_7d", "has_hist_route_7d",
        "max_airborne_30d", "max_airborne_7d", "max_arrival_delay_30d",
        "max_arrival_delay_7d", "max_dur_ratio_route_30d", "max_dur_ratio_route_7d",
        "min_airborne_30d", "min_airborne_7d", "min_arrival_delay_30d",
        "min_arrival_delay_7d", "min_dur_ratio_route_30d", "min_dur_ratio_route_7d",
        "p90_airborne_30d", "p90_airborne_7d", "p90_arrival_delay_30d",
        "p90_arrival_delay_7d", "p90_dur_ratio_route_30d", "p90_dur_ratio_route_7d",
        "std_dur_ratio_route_30d", "std_dur_ratio_route_7d", "trend_arrival_delay_7d",
        "trend_dur_ratio_route_7d",
        "isLO", "distance_km", "local_hour_dep", "local_dow_dep", "local_hour_arr", "local_dow_arr",
        "is_eastbound", "sin_local_hour_dep", "cos_local_hour_dep", "sin_local_hour_arr", "cos_local_hour_arr",
        "sin_local_dow_dep", "cos_local_dow_dep", "sin_local_dow_arr", "cos_local_dow_arr", "sin_month", "cos_month",
    ] + marker_cols

    features_taxi_in = [
        "arr_ap_sched", "arr_stand", model_aircraft_feature_col, "leg_type", "avg_taxi_in_7d", "avg_taxi_in_30d",
        "std_taxi_in_7d", "std_taxi_in_30d", "trend_taxi_in_7d",
        "delta_ema_avg_taxi_in_7d", "delta_ema_avg_taxi_in_30d",
        "p90_taxi_in_7d", "p90_taxi_in_30d", "avg_dur_ratio_arr_7d", "avg_dur_ratio_arr_30d",
        "count_arr_30d", "has_hist_arr_30d", "count_arr_7d",
        "delta_ema_avg_dur_ratio_arr_30d", "delta_ema_avg_dur_ratio_arr_7d",
        "ema_confidence_arr_30d", "ema_confidence_arr_7d", "ema_dur_ratio_arr_30d",
        "ema_dur_ratio_arr_7d", "ema_taxi_in_30d", "ema_taxi_in_7d",
        "has_hist_arr_7d", "max_dur_ratio_arr_30d", "max_dur_ratio_arr_7d",
        "max_taxi_in_30d", "max_taxi_in_7d", "min_dur_ratio_arr_30d",
        "min_dur_ratio_arr_7d", "min_taxi_in_30d", "min_taxi_in_7d",
        "p90_dur_ratio_arr_30d", "p90_dur_ratio_arr_7d", "p90_taxi_in_30d",
        "std_dur_ratio_arr_30d", "std_dur_ratio_arr_7d", "trend_dur_ratio_arr_7d",
        "stand_count_in_7d", "stand_avg_taxi_in_7d", "stand_trend_taxi_in_7d",
        "stand_count_in_30d", "stand_avg_taxi_in_30d", "stand_p10_taxi_in_30d",
        "stand_p50_taxi_in_30d", "stand_p90_taxi_in_30d", "stand_std_taxi_in_30d",
        "isLO", "local_hour_arr", "local_dow_arr",
        "sin_local_hour_arr", "cos_local_hour_arr", "sin_local_dow_arr", "cos_local_dow_arr", "sin_month", "cos_month",
    ] + marker_cols

    all_fs_features = list(dict.fromkeys(features_taxi_out + features_airborne + features_taxi_in))

    return FlightDelaySettings(
        ENV=env,
        ALLOW_CHECKPOINT_RESET=(env == "dev"),
        PROJECT_ROOT=metadata["PROJECT_ROOT"],
        SRC_PATH=metadata["SRC_PATH"],
        WORKSPACE_OWNER=metadata["WORKSPACE_OWNER"],
        PROJECT_NAME=metadata["PROJECT_NAME"],

        CATALOG=catalog,
        SCHEMA=schema,

        SOURCE_CATALOG=source_catalog,
        SOURCE_SCHEMA=source_schema,

        SILVER_CATALOG=silver_catalog,
        SILVER_SCHEMA=silver_schema,

        GOLD_CATALOG=gold_catalog,
        GOLD_SCHEMA=gold_schema,

        CHECKPOINT_PATH=checkpoint_path,
        UC_MODEL_NAME=uc_model_name,
        MODEL_URI=model_uri,
        EXPERIMENTS_ROOT=metadata["EXPERIMENTS_ROOT"],
        EXPERIMENT_PATH=experiment_path,
        MODEL_ALIAS=model_alias,

        # Source tables
        LABELS_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__leg",
        LEG_TIMES_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__leg_times",
        AP_BASICS_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__ap_basics",
        TIME_ZONE_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__time_zone",
        LEG_MISC_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__leg_misc",
        LEG_REMARK_TABLE=f"{source_catalog}.{source_schema}.netline___schedops__leg_remark",

        # silver
        FS_TAXI_OUT_TABLE=f"{silver_catalog}.{silver_schema}.fs_taxi_out_features",
        FS_AIRBORNE_TABLE=f"{silver_catalog}.{silver_schema}.fs_airborne_features",
        FS_TAXI_IN_TABLE=f"{silver_catalog}.{silver_schema}.fs_taxi_in_features",
        FS_STAND_OUT_TABLE=f"{silver_catalog}.{silver_schema}.fs_stand_out_features",
        FS_STAND_IN_TABLE=f"{silver_catalog}.{silver_schema}.fs_stand_in_features",

        # Iter2 - nowy layout ft_*
        FT_LEG_STATUS_TABLE=f"{silver_catalog}.{silver_schema}.ft_leg_status",
        FT_LEG_TIMES_TABLE=f"{silver_catalog}.{silver_schema}.ft_leg_times",
        FT_LEG_MISC_TABLE=f"{silver_catalog}.{silver_schema}.ft_leg_misc",
        FT_AIRPORT_TIMEZONE_TABLE=f"{silver_catalog}.{silver_schema}.ft_airport_timezone",
        FT_ROUTE_DAILY_STATS_TABLE=f"{silver_catalog}.{silver_schema}.ft_route_daily_stats",
        FT_AIRPORT_DAILY_TAXI_OUT_TABLE=f"{silver_catalog}.{silver_schema}.ft_airport_daily_taxi_out",
        FT_AIRPORT_DAILY_TAXI_IN_TABLE=f"{silver_catalog}.{silver_schema}.ft_airport_daily_taxi_in",
        FT_STAND_DAILY_OUT_TABLE=f"{silver_catalog}.{silver_schema}.ft_stand_daily_out",
        FT_STAND_DAILY_IN_TABLE=f"{silver_catalog}.{silver_schema}.ft_stand_daily_in",

        # Iter2 - UC functions (on-demand)
        UC_FN_LOCAL_HOUR=f"{silver_catalog}.{silver_schema}.local_hour",
        UC_FN_LOCAL_DOW=f"{silver_catalog}.{silver_schema}.local_dow",
        UC_FN_MONTH_OF=f"{silver_catalog}.{silver_schema}.month_of",
        UC_FN_SIN_LOCAL_HOUR=f"{silver_catalog}.{silver_schema}.sin_local_hour",
        UC_FN_COS_LOCAL_HOUR=f"{silver_catalog}.{silver_schema}.cos_local_hour",
        UC_FN_SIN_LOCAL_DOW=f"{silver_catalog}.{silver_schema}.sin_local_dow",
        UC_FN_COS_LOCAL_DOW=f"{silver_catalog}.{silver_schema}.cos_local_dow",
        UC_FN_SIN_MONTH_OF=f"{silver_catalog}.{silver_schema}.sin_month_of",
        UC_FN_COS_MONTH_OF=f"{silver_catalog}.{silver_schema}.cos_month_of",

        # Iter2.5: geo UDFs
        UC_FN_HAVERSINE_KM=f"{silver_catalog}.{silver_schema}.haversine_km",
        UC_FN_IS_EASTBOUND=f"{silver_catalog}.{silver_schema}.is_eastbound",

        SHADOW_TABLE=f"{silver_catalog}.{silver_schema}.netline_leg_shadow_cdf_v3",
        TRAINING_DATASET_TABLE=f"{silver_catalog}.{silver_schema}.block_time_training_dataset_v1",
        EVAL_CLEAN_DATASET_TABLE=f"{silver_catalog}.{silver_schema}.block_time_eval_clean_dataset_v1",
        EVAL_ALL_DATASET_TABLE=f"{silver_catalog}.{silver_schema}.block_time_eval_all_dataset_v1",

        # Final / gold
        SINK_TABLE=f"{gold_catalog}.{gold_schema}.block_time_predictions_v3",
        EVENTS_SINK_TABLE=f"{gold_catalog}.{gold_schema}.block_time_predictions_events_v3",

        CV_PARAM_GRID=[
            {"learning_rate": 0.05, "max_leaf_nodes": 63, "min_samples_leaf": 20, "l2_regularization": 0.0, "max_bins": 255},
            {"learning_rate": 0.03, "max_leaf_nodes": 63, "min_samples_leaf": 30, "l2_regularization": 0.10, "max_bins": 255},
        ],
        CATEGORICAL_FEATURES=[model_aircraft_feature_col, "leg_type", "commercial_carrier", "dep_ap_sched", "arr_ap_sched", "dep_stand", "arr_stand"],
        MARKER_COLS=marker_cols,
        FEATURES_TAXI_OUT=features_taxi_out,
        FEATURES_AIRBORNE=features_airborne,
        FEATURES_TAXI_IN=features_taxi_in,
        ALL_FS_FEATURES=all_fs_features,
        PREFERRED_AIRCRAFT_COL=preferred_aircraft_col,
        MODEL_AIRCRAFT_FEATURE_COL=model_aircraft_feature_col,
        REGISTER_REQUIRED_METRICS=[
            "TOTAL_MAE_actual_block_time",
            "TOTAL_ABS_BIAS_actual_block_time",
            "TOTAL_P90_COVERAGE_actual_block_time",
            "HOLDOUT_WIN_RATE_VS_SCHEDULE_pct",
            "OVERFIT_DIFF_actual_block_time_sec",
        ],
        REGISTER_REQUIRED_PARAMS=[
            "training_dataset_table",
            "model_aircraft_feature_col",
            "valid_rows_holdout_split",
        ],
    )


def settings_to_globals(settings: FlightDelaySettings) -> Dict[str, Any]:
    if not isinstance(settings, FlightDelaySettings):
        raise TypeError(f"Nieobsługiwany typ settings: {type(settings)!r}")
    return asdict(settings)
