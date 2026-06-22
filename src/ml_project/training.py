from __future__ import annotations

import json
import os
import pickle
import tempfile
from dataclasses import asdict
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, TYPE_CHECKING

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from pyspark.sql import DataFrame
from pyspark.sql import Window as W
from pyspark.sql import functions as F
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from pathlib import PurePosixPath

try:
    import sklearn
except Exception:
    sklearn = None

from .common import configure_runtime, get_cleaned_flight_data
if TYPE_CHECKING:
    from .settings import FlightDelaySettings

_RAW_LABEL_KEEP = [
    "leg_type", "ac_registration", "ac_subtype", "commercial_carrier",
    "dep_stand", "arr_stand",
    "event_ts", "event_date", "dow", "month", "hour",
    "local_hour_dep", "local_dow_dep", "local_hour_arr", "local_dow_arr",
    "distance_km", "scheduled_block_time_sec", "actual_block_time_sec",
    "block_delay_sec", "arrival_delay_sec", "dep_sched_dt", "arr_sched_dt",
    "dep_ap_actual", "arr_ap_actual", "inactive_reason", "is_active",
    "dq_any_flag", "dq_missing_keys", "dq_invalid_sched", "dq_same_airport",
    "dq_airport_mismatch", "dq_sequence_invalid", "dq_invalid_actuals",
    "dq_outlier_segments", "dq_segment_gap", "dep_utc_offset_min", "arr_utc_offset_min",
]

_EVAL_ONLY_COLS = ["netline_eet_duration_min", "est_taxitime_in", "est_taxitime_out", "fn_full_number"]

_LABEL_COLS_TO_DROP = [
    "actual_block_time_sec", "block_delay_sec", "arrival_delay_sec",
    "dep_ap_actual", "arr_ap_actual", "inactive_reason",
    "dq_any_flag", "dq_missing_keys", "dq_invalid_sched",
    "dq_same_airport", "dq_airport_mismatch", "dq_sequence_invalid",
    "dq_invalid_actuals", "dq_outlier_segments", "dq_segment_gap",
]

def _build_sched_block_bucket_labels(
    scheduled_block_time_sec: pd.Series,
    bucket_defs: List[Dict[str, Any]],
) -> pd.Series:
    sec = pd.to_numeric(scheduled_block_time_sec, errors="coerce")
    labels = pd.Series("UNKNOWN", index=sec.index, dtype="object")

    for bucket in bucket_defs:
        bucket_name = str(bucket["name"])
        min_sec = float(bucket.get("min_sec", 0))
        max_sec = bucket.get("max_sec", None)

        mask = sec >= min_sec
        if max_sec is not None:
            mask = mask & (sec < float(max_sec))

        labels.loc[mask.fillna(False)] = bucket_name

    return labels


def _compute_conditional_block_cqr(
    df_calib: pd.DataFrame,
    preds_p90: np.ndarray,
    settings: FlightDelaySettings,
) -> Dict[str, Any]:
    residuals_p90 = pd.Series(
        pd.to_numeric(df_calib["actual_block_time_sec"], errors="coerce").values - preds_p90,
        index=df_calib.index,
        dtype="float64",
    )

    valid_global = residuals_p90.dropna()
    if valid_global.empty:
        global_shift = 0.0
    else:
        global_shift = max(0.0, float(np.quantile(valid_global, 0.90)))

    if (
        not getattr(settings, "BLOCK_CQR_USE_SCHED_BUCKETS", True)
        or "scheduled_block_time_sec" not in df_calib.columns
    ):
        return {
            "global_shift": global_shift,
            "shift_by_bucket": {},
            "bucket_rows": {},
        }

    bucket_defs = list(getattr(settings, "BLOCK_CQR_BUCKETS", []))
    min_bucket_rows = int(getattr(settings, "BLOCK_CQR_MIN_BUCKET_ROWS", 100))

    bucket_labels = _build_sched_block_bucket_labels(
        df_calib["scheduled_block_time_sec"],
        bucket_defs,
    )

    shift_by_bucket: Dict[str, float] = {}
    bucket_rows: Dict[str, int] = {}

    for bucket in bucket_defs:
        bucket_name = str(bucket["name"])
        bucket_mask = bucket_labels == bucket_name
        bucket_residuals = residuals_p90.loc[bucket_mask].dropna()

        bucket_rows[bucket_name] = int(len(bucket_residuals))

        if len(bucket_residuals) >= min_bucket_rows:
            shift_by_bucket[bucket_name] = max(
                0.0,
                float(np.quantile(bucket_residuals, 0.90)),
            )

    return {
        "global_shift": global_shift,
        "shift_by_bucket": shift_by_bucket,
        "bucket_rows": bucket_rows,
    }


class UltimateSegmentedModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["payload"], "r", encoding="utf-8") as f:
            self.p = json.load(f)

        self.feat_out = self.p["features_out"]
        self.feat_air = self.p["features_air"]
        self.feat_in = self.p["features_in"]
        self.feat_block = self.p["features_block"]

        self.model_aircraft_feature_col = self.p.get("model_aircraft_feature_col", "ac_registration")
        self.top_model_aircraft = self.p.get("top_model_aircraft", [])
        self.top_deps = self.p.get("top_deps", [])
        self.top_arrs = self.p.get("top_arrs", [])

        def _load_pkl(artifact_key: str):
            with open(context.artifacts[artifact_key], "rb") as fh:
                return pickle.load(fh)

        self.m_out_p50 = _load_pkl("model_out_p50")
        self.m_air_p50 = _load_pkl("model_air_p50")
        self.m_in_p50 = _load_pkl("model_in_p50")
        self.m_block_p50 = _load_pkl("model_block_p50")

        self.m_out_p90 = _load_pkl("model_out_p90")
        self.m_air_p90 = _load_pkl("model_air_p90")
        self.m_in_p90 = _load_pkl("model_in_p90")
        self.m_block_p90 = _load_pkl("model_block_p90")

        self.known_leg_types = self.p.get("known_leg_types", ["J", "C", "G"])
        self.known_carriers = self.p.get("known_carriers", ["LO"])
        self.known_dep_stands = self.p.get("known_dep_stands", [])
        self.known_arr_stands = self.p.get("known_arr_stands", [])

        self.w_out = float(self.p.get("weight_out", 0.33))
        self.w_air = float(self.p.get("weight_air", 0.33))
        self.w_in = float(self.p.get("weight_in", 0.34))

        self.cqr_out = float(self.p.get("cqr_out", 0.0))
        self.cqr_air = float(self.p.get("cqr_air", 0.0))
        self.cqr_in = float(self.p.get("cqr_in", 0.0))
        self.cqr_block = float(self.p.get("cqr_block", 0.0))
        self.block_cqr_by_sched_bucket = {
            str(k): float(v)
            for k, v in self.p.get("block_cqr_by_sched_bucket", {}).items()
        }
        self.block_cqr_buckets = list(self.p.get("block_cqr_buckets", []))

    def _predict_segment(self, df, model_obj, feature_list):
        X = pd.DataFrame(index=df.index)
        cat_cols = {
            self.model_aircraft_feature_col,
            "leg_type",
            "commercial_carrier",
            "dep_ap_sched",
            "arr_ap_sched",
            "dep_stand",
            "arr_stand",
        }

        for col in feature_list:
            if col in cat_cols:
                if col not in df.columns:
                    temp = pd.Series("UNKNOWN", index=df.index)
                else:
                    temp = df[col].astype(object).fillna("UNKNOWN")

                if col == self.model_aircraft_feature_col and self.top_model_aircraft:
                    temp = temp.where(temp.isin(self.top_model_aircraft), "OTHER")
                    known_cats = self.top_model_aircraft + ["OTHER", "UNKNOWN"]
                elif col == "dep_ap_sched" and self.top_deps:
                    temp = temp.where(temp.isin(self.top_deps), "OTHER")
                    known_cats = self.top_deps + ["OTHER", "UNKNOWN"]
                elif col == "arr_ap_sched" and self.top_arrs:
                    temp = temp.where(temp.isin(self.top_arrs), "OTHER")
                    known_cats = self.top_arrs + ["OTHER", "UNKNOWN"]
                elif col == "leg_type":
                    known_cats = self.known_leg_types + ["UNKNOWN"]
                elif col == "commercial_carrier":
                    known_cats = self.known_carriers + ["UNKNOWN"]
                elif col == "dep_stand":
                    if self.known_dep_stands:
                        temp = temp.where(temp.isin(self.known_dep_stands), "OTHER")
                    known_cats = self.known_dep_stands + ["OTHER", "UNKNOWN"]
                elif col == "arr_stand":
                    if self.known_arr_stands:
                        temp = temp.where(temp.isin(self.known_arr_stands), "OTHER")
                    known_cats = self.known_arr_stands + ["OTHER", "UNKNOWN"]
                else:
                    known_cats = ["UNKNOWN"]

                known_cats = list(dict.fromkeys(known_cats))
                X[col] = temp.astype(pd.CategoricalDtype(categories=known_cats, ordered=False))
            else:
                X[col] = pd.to_numeric(df.get(col, np.nan), errors="coerce")
        return model_obj.predict(X)
    

    def _resolve_scheduled_block_time_sec(self, df: pd.DataFrame) -> pd.Series:
        if "scheduled_block_time_sec" in df.columns:
            return pd.to_numeric(df["scheduled_block_time_sec"], errors="coerce")

        if {"dep_sched_dt", "arr_sched_dt"}.issubset(df.columns):
            dep_ts = pd.to_datetime(df["dep_sched_dt"], errors="coerce")
            arr_ts = pd.to_datetime(df["arr_sched_dt"], errors="coerce")
            return (arr_ts - dep_ts).dt.total_seconds()

        return pd.Series(np.nan, index=df.index, dtype="float64")


    def _get_block_cqr_shift_vector(self, df: pd.DataFrame) -> np.ndarray:
        if not self.block_cqr_by_sched_bucket or not self.block_cqr_buckets:
            return np.full(len(df), float(self.cqr_block), dtype="float64")

        scheduled_sec = self._resolve_scheduled_block_time_sec(df)
        bucket_labels = _build_sched_block_bucket_labels(
            scheduled_sec,
            self.block_cqr_buckets,
        )

        mapped = bucket_labels.map(self.block_cqr_by_sched_bucket)
        mapped = pd.to_numeric(mapped, errors="coerce").fillna(float(self.cqr_block))

        return mapped.to_numpy(dtype="float64")

    def predict(self, context, model_input: pd.DataFrame):
        df = model_input.copy()

        out_raw = self._predict_segment(df, self.m_out_p50, self.feat_out)
        air_raw = self._predict_segment(df, self.m_air_p50, self.feat_air)
        in_raw = self._predict_segment(df, self.m_in_p50, self.feat_in)
        block_p50 = self._predict_segment(df, self.m_block_p50, self.feat_block)

        sum_raw = out_raw + air_raw + in_raw
        error_to_distribute = block_p50 - sum_raw

        out_adj = np.maximum(0, out_raw + (error_to_distribute * self.w_out))
        air_adj = np.maximum(0, air_raw + (error_to_distribute * self.w_air))
        in_adj = np.maximum(0, in_raw + (error_to_distribute * self.w_in))

        out_p90_raw = self._predict_segment(df, self.m_out_p90, self.feat_out) + self.cqr_out
        air_p90_raw = self._predict_segment(df, self.m_air_p90, self.feat_air) + self.cqr_air
        in_p90_raw = self._predict_segment(df, self.m_in_p90, self.feat_in) + self.cqr_in

        block_cqr_shift = self._get_block_cqr_shift_vector(df)
        block_p90_raw = self._predict_segment(df, self.m_block_p90, self.feat_block) + block_cqr_shift

        out_p90 = np.maximum(out_adj, out_p90_raw)
        air_p90 = np.maximum(air_adj, air_p90_raw)
        in_p90 = np.maximum(in_adj, in_p90_raw)
        block_p90 = np.maximum(block_p50, block_p90_raw)

        return pd.DataFrame(
            {
                "pred_taxi_out_sec": out_adj,
                "pred_airborne_sec": air_adj,
                "pred_taxi_in_sec": in_adj,
                "pred_actual_block_time_sec": block_p50,
                "pred_taxi_out_p90_sec": out_p90,
                "pred_airborne_p90_sec": air_p90,
                "pred_taxi_in_p90_sec": in_p90,
                "pred_actual_block_time_p90_sec": block_p90,
                "pred_raw_segment_sum_sec": sum_raw,
                "pred_reconciled_segment_sum_sec": out_adj + air_adj + in_adj,
                "pred_block_model_sec": block_p50,
                "reconciliation_diff_sec": error_to_distribute,
            },
            index=df.index,
        )


def _build_keep_cols(joined_all: DataFrame, settings: FlightDelaySettings) -> list[str]:
    keep_cols = list(
        dict.fromkeys(
            ["leg_no", "event_date", "dep_ap_sched", "arr_ap_sched"]
            + _RAW_LABEL_KEEP
            + _EVAL_ONLY_COLS
            + list(settings.TARGET_COLS)
            + list(settings.ALL_FS_FEATURES)
        )
    )
    return [c for c in keep_cols if c in joined_all.columns]


def finalize_model_df(df: DataFrame, keep_cols: list[str], target_cols: list[str], *, is_for_training: bool) -> DataFrame:
    out = df.select(*keep_cols)

    for c in [c for c in out.columns if c.startswith("has_hist_")]:
        out = out.withColumn(c, F.coalesce(F.col(c).cast("int"), F.lit(0)))

    if "dq_any_flag" in out.columns:
        out = out.withColumn(
            "label_scope",
            F.when(F.col("dq_any_flag") == 1, F.lit("flagged")).otherwise(F.lit("clean")),
        )
    else:
        out = out.withColumn("label_scope", F.lit("clean"))

    if is_for_training:
        cols_to_drop_eval = [c for c in _EVAL_ONLY_COLS if c in out.columns]
        if cols_to_drop_eval:
            out = out.drop(*cols_to_drop_eval)

        cols_to_drop_leakage = [
            c for c in _LABEL_COLS_TO_DROP if c in out.columns and c not in target_cols
        ]
        if cols_to_drop_leakage:
            out = out.drop(*cols_to_drop_leakage)

    return out



def _build_feature_lookups(spark, settings: FlightDelaySettings) -> List[Any]:
    """FeatureLookups for rolling stats from ft_*_daily_* tables.

    Only tables with simple list-form lookup_key are included here.
    Stand tables (which need dict-form key mapping) are joined manually
    in _join_stand_features() to avoid SDK bug (unhashable type: dict).
    """
    ts_key = settings.FS_TIMESTAMP_KEY  # event_date

    # Route: exclude PK cols + days_since_last_event
    route_exclude = {"route_id", "event_date", "dep_ap_sched", "arr_ap_sched", "days_since_last_event"}
    route_features = [
        c for c in spark.read.table(settings.FT_ROUTE_DAILY_STATS_TABLE).columns if c not in route_exclude
    ]

    return [
        # ===== Airport taxi out (PK: dep_ap_sched, event_date TIMESERIES) =====
        FeatureLookup(
            table_name=settings.FT_AIRPORT_DAILY_TAXI_OUT_TABLE,
            lookup_key=list(settings.PK_FT_AIRPORT_TAXI_OUT),
            timestamp_lookup_key=ts_key,
            rename_outputs={"days_since_last_event": "days_since_last_event_dep"},
        ),
        # ===== Route (PK: route_id, event_date TIMESERIES) =====
        FeatureLookup(
            table_name=settings.FT_ROUTE_DAILY_STATS_TABLE,
            lookup_key=list(settings.PK_FT_ROUTE),
            feature_names=route_features,
            timestamp_lookup_key=ts_key,
        ),
        # ===== Airport taxi in (PK: arr_ap_sched, event_date TIMESERIES) =====
        FeatureLookup(
            table_name=settings.FT_AIRPORT_DAILY_TAXI_IN_TABLE,
            lookup_key=list(settings.PK_FT_AIRPORT_TAXI_IN),
            timestamp_lookup_key=ts_key,
            rename_outputs={"days_since_last_event": "days_since_last_event_arr"},
        ),
    ]


def _join_stand_features(df: DataFrame, spark, settings: FlightDelaySettings) -> DataFrame:
    """Manual PIT join for stand tables (avoids SDK dict lookup_key bug).

    Joins ft_stand_daily_out on (stand_id_out, event_date) and
    ft_stand_daily_in on (stand_id_in, event_date) using asof join logic:
    feature event_date <= base event_date (latest available before flight).
    """
    from pyspark.sql import Window

    def _pit_join_stand(base_df, stand_table, base_stand_col, suffix):
        """PIT join: get latest stand features where stand.event_date <= base.event_date."""
        stand_df = spark.read.table(stand_table)

        # Exclude PK and days_since from stand features
        feature_cols = [c for c in stand_df.columns if c not in ("stand_id", "event_date", "days_since_last_event")]

        # Join on stand_id match + stand.event_date <= base.event_date
        joined = base_df.join(
            stand_df.select("stand_id", "event_date", *feature_cols).withColumnRenamed("event_date", "_stand_event_date"),
            on=(F.col(base_stand_col) == F.col("stand_id")) & (F.col("_stand_event_date") <= F.col("event_date")),
            how="left",
        )

        # Keep only the LATEST stand row (max _stand_event_date per base row)
        w = Window.partitionBy(base_df.columns).orderBy(F.col("_stand_event_date").desc())
        joined = joined.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn", "_stand_event_date", "stand_id")

        return joined

    df = _pit_join_stand(df, settings.FT_STAND_DAILY_OUT_TABLE, "stand_id_out", "out")
    df = _pit_join_stand(df, settings.FT_STAND_DAILY_IN_TABLE, "stand_id_in", "in")
    return df


_FS_LOOKUP_HELPER_COLS = ["route_id", "stand_id_out", "stand_id_in"]

# Columns that exist in base df (from enriched()) but are RE-ADDED by FeatureLookup
# (ft_airport_timezone). Must be dropped from base df to avoid duplicate output names.
_FS_COLS_TO_DROP_FROM_BASE = [
    # No longer need to drop geo/time columns — they stay in base df
    # (FeatureLookup only adds rolling stats which don't exist in enriched)
]
_FS_REQUIRED_LOOKUP_KEYS = ["dep_ap_sched", "route_id", "arr_ap_sched", "stand_id_out", "stand_id_in"]


def _add_fs_lookup_keys(df: DataFrame) -> DataFrame:
    return (
        df.withColumn("route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")))
        .withColumn("stand_id_out", F.concat_ws("_", F.col("dep_ap_sched"), F.col("dep_stand")))
        .withColumn("stand_id_in", F.concat_ws("_", F.col("arr_ap_sched"), F.col("arr_stand")))
    )


def _base_training_df(spark, settings: FlightDelaySettings) -> DataFrame:
    """Kanoniczny base df dla FeatureLookup."""
    base = get_cleaned_flight_data(spark, settings.LABEL_START, active_only=False)
    base = _add_fs_lookup_keys(base)
    from pyspark.sql.types import ByteType, ShortType, IntegerType
    for f in base.schema.fields:
        if isinstance(f.dataType, (ByteType, ShortType)):
            base = base.withColumn(f.name, F.col(f.name).cast(IntegerType()))
    return base


def _create_fs_training_set(fe: FeatureEngineeringClient, base_df: DataFrame, settings: FlightDelaySettings, spark):
    # Step 1: FeatureLookup for airport/route tables (list-key, no SDK issues)
    training_set = fe.create_training_set(
        df=base_df,
        feature_lookups=_build_feature_lookups(spark, settings),
        label=None,
        exclude_columns=_FS_LOOKUP_HELPER_COLS,
    )
    return training_set


def _add_stand_features_post_lookup(df: DataFrame, spark, settings: FlightDelaySettings) -> DataFrame:
    """Add stand features via manual PIT join (post create_training_set)."""
    return _join_stand_features(df, spark, settings)


def _assert_lookup_contract(base_df: DataFrame, settings: FlightDelaySettings) -> None:
    cols = set(base_df.columns)
    required = set(_FS_REQUIRED_LOOKUP_KEYS) | {settings.FS_TIMESTAMP_KEY}
    missing = required - cols
    if missing:
        raise RuntimeError(
            "base df nie zawiera wymaganych kluczy FeatureLookup / event_date: "
            f"{sorted(missing)}. feature_spec.yaml rozjechałby się ze scoringiem."
        )

def build_training_datasets(spark, settings: FlightDelaySettings) -> Dict[str, Any]:
    configure_runtime(settings, spark=spark)

    print("Route FS join mode: FeatureEngineeringClient.create_training_set (PIT lookup)")
    print("Pobieranie etykiet i weryfikacja Global Data Quality...")
    labels_all = get_cleaned_flight_data(spark, settings.LABEL_START, active_only=False)

    dq_summary = (
        labels_all.groupBy("inactive_reason")
        .agg(
            F.count("*").alias("rows"),
            F.avg(F.col("is_active").cast("double")).alias("share_active"),
        )
        .orderBy(F.desc("rows"))
    )

    total_rows = labels_all.count()
    clean_rows = labels_all.filter(F.col("is_active") == True).count()
    flagged_rows = total_rows - clean_rows
    drop_ratio = 1.0 - (clean_rows / total_rows) if total_rows else 0.0

    print(
        f"DQ summary: clean={clean_rows:,} | flagged={flagged_rows:,} | "
        f"drop_ratio={drop_ratio:.1%}"
    )

    if total_rows > 0 and drop_ratio > float(settings.MAX_TRAINING_DQ_DROP_RATIO):
        raise RuntimeError(
            "🛑 STOP THE LINE: Zbyt duży odsetek danych oznaczonych jako flagged "
            f"({drop_ratio:.1%}). Zweryfikuj OOOI i logikę DQ przed treningiem."
        )

    print("[OK] Etykiety pobrane. Tworzę zbiory: clean ops do treningu i all ops do audytu.")
    labels_all = _add_fs_lookup_keys(labels_all)

    print("Dołączanie route i stand FS przez FeatureLookup z timestamp_lookup_key=event_date...")
    _assert_lookup_contract(labels_all, settings)
    fe = FeatureEngineeringClient()
    row_id_col = "_training_row_id"
    labels_with_id = labels_all.withColumn(row_id_col, F.monotonically_increasing_id())

    training_set = _create_fs_training_set(fe, labels_with_id, settings, spark)
    joined_all = training_set.load_df()


    cardinality = joined_all.agg(
        F.count("*").alias("rows_after_join"),
        F.countDistinct(row_id_col).alias("distinct_rows"),
    ).first()
    rows_after_join = int(cardinality["rows_after_join"])
    distinct_rows = int(cardinality["distinct_rows"])
    base_count = labels_all.count()
    if rows_after_join != distinct_rows or rows_after_join != base_count:
        raise RuntimeError(
            "FeatureLookup naruszył kardynalność zbioru treningowego: "
            f"base={base_count}, rows_after_join={rows_after_join}, "
            f"distinct_rows={distinct_rows}."
        )
    joined_all = joined_all.drop(row_id_col)
    print("[OK] FeatureLookup join zakończony sukcesem.")

    joined_clean = joined_all.filter(F.col("is_active") == True)

    keep_cols = _build_keep_cols(joined_all, settings)

    training_df_model = finalize_model_df(
        joined_clean,
        keep_cols,
        list(settings.TARGET_COLS),
        is_for_training=True,
    )
    training_df_eval_clean = finalize_model_df(
        joined_clean,
        keep_cols,
        list(settings.TARGET_COLS),
        is_for_training=False,
    )
    training_df_eval_all = finalize_model_df(
        joined_all,
        keep_cols,
        list(settings.TARGET_COLS),
        is_for_training=False,
    )

    rejected_cols = sorted(set(joined_all.columns) - set(training_df_eval_all.columns))
    counts = {
        "labels_all": total_rows,
        "labels_clean": clean_rows,
        "training_df_model": clean_rows,
        "training_df_eval_clean": clean_rows,
        "training_df_eval_all": total_rows,
    }

    max_event_row = training_df_model.agg(F.max("event_ts").alias("max_event_ts")).first()
    max_event_ts = max_event_row["max_event_ts"] if max_event_row else None

    summary = {
        "dq_drop_ratio": drop_ratio,
        "counts": counts,
        "max_event_ts": str(max_event_ts) if max_event_ts is not None else None,
        "training_table": settings.TRAINING_DATASET_TABLE,
        "eval_clean_table": settings.EVAL_CLEAN_DATASET_TABLE,
        "eval_all_table": settings.EVAL_ALL_DATASET_TABLE,
        "aircraft_feature_business_preference": settings.PREFERRED_AIRCRAFT_COL,
        "aircraft_feature_model_contract": settings.MODEL_AIRCRAFT_FEATURE_COL,
        "rejected_cols": rejected_cols,
    }

    return {
        "dq_summary": dq_summary,
        "labels_all": labels_all,
        "joined_all": joined_all,
        "joined_clean": joined_clean,
        "training_df_model": training_df_model,
        "training_df_eval_clean": training_df_eval_clean,
        "training_df_eval_all": training_df_eval_all,
        "summary": summary,
    }


def materialize_training_datasets(spark, settings: FlightDelaySettings, datasets: Mapping[str, Any], *, mode: str = "overwrite") -> Dict[str, str]:
    overwrite_schema = str(mode).lower() == "overwrite"

    def _write(df: DataFrame, table_name: str) -> None:
        writer = df.write.format("delta").mode(mode)
        if overwrite_schema:
            writer = writer.option("overwriteSchema", "true")
        writer.saveAsTable(table_name)

    _write(datasets["training_df_model"], settings.TRAINING_DATASET_TABLE)
    _write(datasets["training_df_eval_clean"], settings.EVAL_CLEAN_DATASET_TABLE)
    _write(datasets["training_df_eval_all"], settings.EVAL_ALL_DATASET_TABLE)

    return {
        "training_df_model": settings.TRAINING_DATASET_TABLE,
        "training_df_eval_clean": settings.EVAL_CLEAN_DATASET_TABLE,
        "training_df_eval_all": settings.EVAL_ALL_DATASET_TABLE,
    }


def expose_legacy_training_globals(namespace: Dict[str, Any], datasets: Mapping[str, Any]) -> None:
    namespace.update(
        {
            "labels_all": datasets["labels_all"],
            "joined_all": datasets["joined_all"],
            "joined_clean": datasets["joined_clean"],
            "training_df_model": datasets["training_df_model"],
            "training_df_eval_clean": datasets["training_df_eval_clean"],
            "training_df_eval_all": datasets["training_df_eval_all"],
        }
    )


def load_training_dataset_pdf(spark, settings: FlightDelaySettings) -> Dict[str, Any]:
    configure_runtime(settings, spark=spark)
    training_sdf_raw = spark.read.table(settings.TRAINING_DATASET_TABLE)
    total_count = training_sdf_raw.count()

    if total_count == 0:
        raise RuntimeError(f"Tabela treningowa jest pusta: {settings.TRAINING_DATASET_TABLE}")

    max_rows = int(getattr(settings, "MAX_TRAINING_PANDAS_ROWS", 1_500_000))
    was_truncated = False
    min_ts = None
    max_ts = None

    if total_count > max_rows:
        print(
            f"[WARN] UWAGA: Zbiór ({total_count:,} wierszy) przekracza limit pamięci ({max_rows:,})."
        )
        print("Zachowuję najnowsze ciągłe okno czasowe zamiast losowego próbkowania.")
        training_sdf = (
            training_sdf_raw.orderBy(F.col("event_ts").desc())
            .limit(max_rows)
            .orderBy("event_ts")
        )
        was_truncated = True
        total_count = training_sdf.count()
        min_ts, max_ts = training_sdf.agg(F.min("event_ts"), F.max("event_ts")).first()
        print(f"[OK] Zredukowano zbiór do {total_count:,} wierszy | zakres czasu: {min_ts} -> {max_ts}")
    else:
        training_sdf = training_sdf_raw.orderBy("event_ts")

    print("Pobieranie danych do pamięci węzła...")
    full_pdf = training_sdf.toPandas()
    full_pdf = prepare_training_pdf(full_pdf, settings)

    return {
        "training_sdf": training_sdf,
        "full_pdf": full_pdf,
        "row_count": total_count,
        "was_truncated": was_truncated,
        "min_ts": str(min_ts) if min_ts is not None else None,
        "max_ts": str(max_ts) if max_ts is not None else None,
        "training_table": settings.TRAINING_DATASET_TABLE,
    }


def prepare_training_pdf(full_pdf: pd.DataFrame, settings: FlightDelaySettings) -> pd.DataFrame:
    out = full_pdf.copy()
    if "event_ts" not in out.columns:
        raise RuntimeError("Brakuje kolumny event_ts w treningowym DataFrame.")

    out["event_ts"] = pd.to_datetime(out["event_ts"], errors="coerce")
    if out["event_ts"].isna().all():
        raise RuntimeError("Kolumna event_ts jest całkowicie pusta po konwersji do datetime.")
    
    out = out.sort_values("event_ts", kind="mergesort").reset_index(drop=True)

    categorical_features = [c for c in settings.CATEGORICAL_FEATURES if c in out.columns]
    for col in categorical_features:
        out[col] = out[col].fillna("UNKNOWN")

    numeric_exclusions = {"event_date", "event_ts", *categorical_features}
    for c in out.columns:
        if c not in numeric_exclusions:
            out[c] = pd.to_numeric(out[c], errors="coerce")

    return out


def get_segment_feature_sets(full_pdf: pd.DataFrame, settings: FlightDelaySettings) -> Dict[str, List[str]]:
    return {
        "features_out": [c for c in settings.FEATURES_TAXI_OUT if c in full_pdf.columns],
        "features_air": [c for c in settings.FEATURES_AIRBORNE if c in full_pdf.columns],
        "features_in": [c for c in settings.FEATURES_TAXI_IN if c in full_pdf.columns],
        "features_block": [c for c in settings.ALL_FS_FEATURES if c in full_pdf.columns],
    }


def _get_top_k(df: pd.DataFrame, col: str, k: int) -> List[str]:
    if col not in df.columns:
        return []
    return df[col].dropna().value_counts().nlargest(k).index.tolist()


def _build_top_k_mappings(df: pd.DataFrame, settings: FlightDelaySettings) -> Dict[str, List[str]]:
    mappings = {
        settings.MODEL_AIRCRAFT_FEATURE_COL: _get_top_k(df, settings.MODEL_AIRCRAFT_FEATURE_COL, int(settings.TOP_K_MODEL_AIRCRAFT)),
        "dep_ap_sched": _get_top_k(df, "dep_ap_sched", int(settings.TOP_K_DEP_AIRPORTS)),
        "arr_ap_sched": _get_top_k(df, "arr_ap_sched", int(settings.TOP_K_ARR_AIRPORTS)),
        "dep_stand": _get_top_k(df, "dep_stand", int(settings.TOP_K_DEP_STANDS)),
        "arr_stand": _get_top_k(df, "arr_stand", int(settings.TOP_K_ARR_STANDS)),
    }
    return {k: v for k, v in mappings.items() if k in df.columns}


def _apply_top_k(df: pd.DataFrame, mappings: Mapping[str, List[str]], settings: FlightDelaySettings) -> pd.DataFrame:
    out = df.copy()
    for col, top_cats in mappings.items():
        if col in out.columns:
            out[col] = np.where(out[col].isin(top_cats), out[col], "OTHER")
            out[col] = out[col].astype("category")

    for col in settings.CATEGORICAL_FEATURES:
        if col in out.columns and col not in mappings:
            out[col] = out[col].astype("category")
    return out


def _evaluate_param_set(X: pd.DataFrame, y: pd.Series, params: Mapping[str, Any], settings: FlightDelaySettings) -> tuple[pd.DataFrame, float]:
    n_splits = int(settings.CV_N_SPLITS)
    if len(X) <= n_splits:
        raise RuntimeError(
            f"Za mało wierszy ({len(X)}) do TimeSeriesSplit z n_splits={n_splits}."
        )

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_rows: list[dict[str, float]] = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        mappings = _build_top_k_mappings(X_train, settings)
        X_train_clean = _apply_top_k(X_train, mappings, settings)
        X_test_clean = _apply_top_k(X_test, mappings, settings)

        model = HistGradientBoostingRegressor(
            loss="quantile",
            quantile=0.5,
            random_state=42,
            categorical_features="from_dtype",
            **params,
        )

        model.fit(X_train_clean, y_train)
        preds = model.predict(X_test_clean)

        fold_rows.append(
            {
                "fold": fold,
                "mae_sec": float(mean_absolute_error(y_test, preds)),
                "bias_sec": float(np.mean(preds - y_test)),
            }
        )

        del model, X_train, X_test, X_train_clean, X_test_clean, y_train, y_test, preds

    fold_df = pd.DataFrame(fold_rows)
    objective = float(
        fold_df["mae_sec"].mean() + (float(settings.BIAS_PENALTY_WEIGHT) * fold_df["bias_sec"].abs().mean())
    )
    import gc; gc.collect()
    return fold_df, objective


def train_and_evaluate_segment(target_col: str, features: List[str], df: pd.DataFrame, segment_name: str, settings: FlightDelaySettings) -> Dict[str, Any]:
    print(f"\nRozpoczynam tuning i walidację kroczącą (CV) dla: {segment_name} ...")

    actual_features = [c for c in features if c in df.columns]
    if not actual_features:
        raise RuntimeError(f"Brak cech wejściowych dla segmentu {segment_name}.")

    df_valid = df.dropna(subset=[target_col, "event_ts"])
    if df_valid.empty:
        raise RuntimeError(f"Brak danych po dropna dla targetu {target_col} ({segment_name}).")

    X = df_valid[actual_features]
    y = df_valid[target_col]

    best_result = None
    tuning_rows = []

    for idx, params in enumerate(settings.CV_PARAM_GRID, start=1):
        fold_df, score = _evaluate_param_set(X, y, params, settings)
        tuning_rows.append(
            {
                "segment": segment_name,
                "candidate_id": idx,
                "params_json": json.dumps(params, sort_keys=True),
                "mean_mae_min": float(fold_df["mae_sec"].mean()) / 60.0,
                "objective": float(score) / 60.0,
            }
        )
        if best_result is None or score < best_result["score"]:
            best_result = {"params": params, "score": score, "fold_df": fold_df.copy()}

    best_params = dict(best_result["params"])
    final_mappings = _build_top_k_mappings(X, settings)

    del X, y
    import gc; gc.collect()

    calib_days = int(getattr(settings, "CALIBRATION_DAYS", 30))
    max_ts = df_valid["event_ts"].max()
    calib_threshold = max_ts - pd.Timedelta(days=calib_days)

    df_fit = df_valid[df_valid["event_ts"] < calib_threshold]
    df_calib = df_valid[df_valid["event_ts"] >= calib_threshold]

    if len(df_calib) < int(getattr(settings, "MIN_CALIBRATION_ROWS", 100)):
        split_idx = int(len(df_valid) * 0.9)
        df_fit = df_valid.iloc[:split_idx]
        df_calib = df_valid.iloc[split_idx:]

    if df_fit.empty or df_calib.empty:
        raise RuntimeError(
            f"Nie udało się zbudować zbioru fit/calibration dla segmentu {segment_name}."
        )

    X_fit = _apply_top_k(df_fit[actual_features].copy(), final_mappings, settings)
    y_fit = df_fit[target_col].copy()
    X_calib = _apply_top_k(df_calib[actual_features].copy(), final_mappings, settings)
    y_calib = df_calib[target_col].copy()

    ml_p50 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.5,
        random_state=42,
        categorical_features="from_dtype",
        **best_params,
    )
    ml_p90 = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=0.9,
        random_state=42,
        categorical_features="from_dtype",
        **best_params,
    )

    ml_p50.fit(X_fit, y_fit)
    ml_p90.fit(X_fit, y_fit)

    preds_p90 = ml_p90.predict(X_calib)

    cqr_shift = 0.0
    cqr_shift_by_bucket: Dict[str, float] = {}
    cqr_bucket_rows: Dict[str, int] = {}

    if target_col == "actual_block_time_sec":
        block_cqr = _compute_conditional_block_cqr(df_calib, preds_p90, settings)
        cqr_shift = float(block_cqr["global_shift"])
        cqr_shift_by_bucket = dict(block_cqr["shift_by_bucket"])
        cqr_bucket_rows = dict(block_cqr["bucket_rows"])
    else:
        residuals_p90 = y_calib - preds_p90
        cqr_shift = float(np.quantile(residuals_p90, 0.90))
        cqr_shift = max(0.0, cqr_shift)

    fit_rows = int(len(df_fit))
    calib_rows = int(len(df_calib))
    del df_valid, df_fit, df_calib
    import gc; gc.collect()

    return {
        "final_features": actual_features,
        "final_mappings": final_mappings,
        "ml_p50": ml_p50,
        "ml_p90": ml_p90,
        "best_params": best_params,
        "best_cv_mae_sec": float(best_result["fold_df"]["mae_sec"].mean()),
        "best_cv_bias_sec": float(best_result["fold_df"]["bias_sec"].mean()),
        "cqr_shift": cqr_shift,
        "cqr_shift_by_bucket": cqr_shift_by_bucket,
        "cqr_bucket_rows": cqr_bucket_rows,
        "fit_rows": fit_rows,
        "calib_rows": calib_rows,
        "tuning_rows": tuning_rows,
    }


def _compute_known_categories(train_pdf: pd.DataFrame, settings: FlightDelaySettings) -> Dict[str, List[str]]:
    def _safe_unique(col: str) -> List[str]:
        if col not in train_pdf.columns:
            return []
        return train_pdf[col].dropna().astype(str).unique().tolist()

    model_aircraft = settings.MODEL_AIRCRAFT_FEATURE_COL
    return {
        "top_model_aircraft": _get_top_k(train_pdf, model_aircraft, int(settings.TOP_K_MODEL_AIRCRAFT)),
        "top_deps": _get_top_k(train_pdf, "dep_ap_sched", int(settings.TOP_K_DEP_AIRPORTS)),
        "top_arrs": _get_top_k(train_pdf, "arr_ap_sched", int(settings.TOP_K_ARR_AIRPORTS)),
        "known_dep_stands": _get_top_k(train_pdf, "dep_stand", int(settings.TOP_K_DEP_STANDS)),
        "known_arr_stands": _get_top_k(train_pdf, "arr_stand", int(settings.TOP_K_ARR_STANDS)),
        "known_leg_types": _safe_unique("leg_type"),
        "known_carriers": _safe_unique("commercial_carrier"),
    }


def _make_payload(best_out, best_air, best_in, best_block, train_pdf: pd.DataFrame, settings: FlightDelaySettings) -> Dict[str, Any]:
    mae_out = float(best_out["best_cv_mae_sec"])
    mae_air = float(best_air["best_cv_mae_sec"])
    mae_in = float(best_in["best_cv_mae_sec"])
    total_mae = max(mae_out + mae_air + mae_in, 1e-9)

    known = _compute_known_categories(train_pdf, settings)
    return {
        "features_out": best_out["final_features"],
        "features_air": best_air["final_features"],
        "features_in": best_in["final_features"],
        "features_block": best_block["final_features"],
        "model_aircraft_feature_col": settings.MODEL_AIRCRAFT_FEATURE_COL,
        **known,
        "weight_out": mae_out / total_mae,
        "weight_air": mae_air / total_mae,
        "weight_in": mae_in / total_mae,
        "cqr_out": best_out["cqr_shift"],
        "cqr_air": best_air["cqr_shift"],
        "cqr_in": best_in["cqr_shift"],
        "cqr_block": best_block["cqr_shift"],
        "block_cqr_by_sched_bucket": best_block.get("cqr_shift_by_bucket", {}),
        "block_cqr_buckets": list(getattr(settings, "BLOCK_CQR_BUCKETS", [])),
    }


def _log_segment_metrics(best_obj: Mapping[str, Any], prefix: str) -> None:
    mlflow.log_metric(f"CV_MAE_{prefix}_sec", float(best_obj["best_cv_mae_sec"]))
    mlflow.log_metric(f"CV_BIAS_{prefix}_sec", float(best_obj["best_cv_bias_sec"]))
    mlflow.log_param(f"BEST_PARAMS_{prefix}", json.dumps(best_obj["best_params"], sort_keys=True))

def _ensure_experiment_parent_exists(experiment_path: str) -> None:
    if not experiment_path or not experiment_path.startswith("/"):
        return

    parent_dir = str(PurePosixPath(experiment_path).parent)

    try:
        from databricks.sdk import WorkspaceClient
        WorkspaceClient().workspace.mkdirs(parent_dir)
    except Exception as e:
        raise RuntimeError(
            f"Nie udało się utworzyć folderu eksperymentów MLflow: {parent_dir}. Błąd: {e}"
        ) from e

def run_train_compare_models(spark, settings: FlightDelaySettings) -> Dict[str, Any]:
    configure_runtime(settings, spark=spark)
    _ensure_experiment_parent_exists(settings.EXPERIMENT_PATH)
    mlflow.set_experiment(settings.EXPERIMENT_PATH)
    print("Experiment:", settings.EXPERIMENT_PATH)

    loaded = load_training_dataset_pdf(spark, settings)
    full_pdf = loaded["full_pdf"]
    feature_sets = get_segment_feature_sets(full_pdf, settings)

    split_date = pd.to_datetime(full_pdf["event_ts"].max()) - pd.Timedelta(days=int(settings.EVAL_DAYS))
    train_pdf = full_pdf[full_pdf["event_ts"] < split_date].copy()
    valid_pdf = full_pdf[full_pdf["event_ts"] >= split_date].copy()

    if train_pdf.empty or valid_pdf.empty:
        raise RuntimeError(
            f"Niepoprawny split czasowy: train_rows={len(train_pdf)}, valid_rows={len(valid_pdf)}, split_date={split_date}."
        )

    print(
        f"Dane do treningu: do {split_date.date()} | Zbiór wstrzymany do ewaluacji: ostatnie {settings.EVAL_DAYS} dni."
    )
    print(f"Train rows: {len(train_pdf):,} | Eval rows: {len(valid_pdf):,}")

    best_out = train_and_evaluate_segment("taxi_out_sec", feature_sets["features_out"], train_pdf, "TAXI-OUT", settings)
    import gc; gc.collect()
    best_air = train_and_evaluate_segment("airborne_sec", feature_sets["features_air"], train_pdf, "AIRBORNE", settings)
    import gc; gc.collect()
    best_in = train_and_evaluate_segment("taxi_in_sec", feature_sets["features_in"], train_pdf, "TAXI-IN", settings)
    import gc; gc.collect()
    best_block = train_and_evaluate_segment("actual_block_time_sec", feature_sets["features_block"], train_pdf, "TOTAL-BLOCK", settings)

    payload_dict = _make_payload(best_out, best_air, best_in, best_block, train_pdf, settings)

    with mlflow.start_run(run_name=settings.TRAINING_RUN_NAME) as run:
        mlflow.log_param("training_dataset_table", settings.TRAINING_DATASET_TABLE)
        mlflow.log_param("training_eval_clean_table", settings.EVAL_CLEAN_DATASET_TABLE)
        mlflow.log_param("training_eval_all_table", settings.EVAL_ALL_DATASET_TABLE)
        mlflow.log_param("preferred_aircraft_col", settings.PREFERRED_AIRCRAFT_COL)
        mlflow.log_param("model_aircraft_feature_col", settings.MODEL_AIRCRAFT_FEATURE_COL)
        mlflow.log_param("max_training_pandas_rows", int(settings.MAX_TRAINING_PANDAS_ROWS))
        mlflow.log_param("training_rows_pdf", int(len(full_pdf)))
        mlflow.log_param("train_rows_holdout_split", int(len(train_pdf)))
        mlflow.log_param("valid_rows_holdout_split", int(len(valid_pdf)))
        mlflow.log_param("holdout_split_date", str(split_date))
        mlflow.log_param("training_data_was_truncated", str(loaded["was_truncated"]))

        for prefix, best_obj in [("taxi_out", best_out), ("airborne", best_air), ("taxi_in", best_in), ("block", best_block)]:
            _log_segment_metrics(best_obj, prefix)

        mlflow.log_param(
            "block_cqr_bucket_defs_json",
            json.dumps(list(getattr(settings, "BLOCK_CQR_BUCKETS", [])), sort_keys=True),
        )
        mlflow.log_param(
            "block_cqr_min_bucket_rows",
            int(getattr(settings, "BLOCK_CQR_MIN_BUCKET_ROWS", 100)),
        )
        mlflow.log_param(
            "block_cqr_shift_by_bucket_json",
            json.dumps(best_block.get("cqr_shift_by_bucket", {}), sort_keys=True),
        )
        mlflow.log_param(
            "block_cqr_bucket_rows_json",
            json.dumps(best_block.get("cqr_bucket_rows", {}), sort_keys=True),
        )

        
        _importance_payload = {}
        for _seg_name, _best_obj in [("taxi_out", best_out), ("airborne", best_air), ("taxi_in", best_in), ("block", best_block)]:
            _feat_list = _best_obj["final_features"]
            for _q_name, _model_key in [("p50", "ml_p50"), ("p90", "ml_p90")]:
                _model_obj = _best_obj[_model_key]
                if hasattr(_model_obj, "feature_importances_"):
                    _importance_payload[f"{_seg_name}_{_q_name}"] = dict(
                        zip(_feat_list, _model_obj.feature_importances_.tolist())
                    )
        if _importance_payload:
            mlflow.log_dict(_importance_payload, "feature_importances.json")


        with tempfile.TemporaryDirectory() as td:
            artifacts = {}
            payload_path = os.path.join(td, "payload.json")
            with open(payload_path, "w", encoding="utf-8") as f:
                json.dump(payload_dict, f)
            artifacts["payload"] = payload_path

            for name, best_obj in [("out", best_out), ("air", best_air), ("in", best_in), ("block", best_block)]:
                m50_path = os.path.join(td, f"m_{name}_p50.pkl")
                with open(m50_path, "wb") as f:
                    pickle.dump(best_obj["ml_p50"], f)
                artifacts[f"model_{name}_p50"] = m50_path

                m90_path = os.path.join(td, f"m_{name}_p90.pkl")
                with open(m90_path, "wb") as f:
                    pickle.dump(best_obj["ml_p90"], f)
                artifacts[f"model_{name}_p90"] = m90_path

            wrapper_model = UltimateSegmentedModel()
            wrapper_model.load_context(SimpleNamespace(artifacts=artifacts))

            final_preds = wrapper_model.predict(None, valid_pdf)
            valid_clean = valid_pdf.dropna(subset=["actual_block_time_sec", "scheduled_block_time_sec"]).copy()
            final_preds_valid = final_preds.loc[valid_clean.index].copy()

            valid_clean["pred_actual_block_time_sec"] = final_preds_valid["pred_actual_block_time_sec"].values
            valid_clean["pred_actual_block_time_p90_sec"] = final_preds_valid["pred_actual_block_time_p90_sec"].values
            valid_clean["model_abs_error_sec"] = np.abs(valid_clean["pred_actual_block_time_sec"] - valid_clean["actual_block_time_sec"])
            valid_clean["schedule_abs_error_sec"] = np.abs(valid_clean["scheduled_block_time_sec"] - valid_clean["actual_block_time_sec"])

            mae_total = float(mean_absolute_error(valid_clean["actual_block_time_sec"], valid_clean["pred_actual_block_time_sec"]))
            bias_total = float(np.mean(valid_clean["actual_block_time_sec"] - valid_clean["pred_actual_block_time_sec"]))
            abs_bias_total = abs(bias_total)
            p90_coverage = float(np.mean(valid_clean["pred_actual_block_time_p90_sec"] >= valid_clean["actual_block_time_sec"])) * 100.0
            p90_coverage_gap = abs(p90_coverage - 90.0)
            baseline_mae = float(mean_absolute_error(valid_clean["actual_block_time_sec"], valid_clean["scheduled_block_time_sec"]))
            holdout_win_rate = float(np.mean(valid_clean["model_abs_error_sec"] < valid_clean["schedule_abs_error_sec"])) * 100.0

            monthly_mae_std = np.nan
            if "event_ts" in valid_clean.columns:
                valid_clean["eval_month"] = pd.to_datetime(valid_clean["event_ts"]).dt.to_period("M").astype(str)
                monthly_mae = valid_clean.groupby("eval_month")["model_abs_error_sec"].mean()
                monthly_mae_std = float(monthly_mae.std(ddof=0)) if len(monthly_mae) > 1 else 0.0

            mlflow.log_metric("TOTAL_MAE_actual_block_time", mae_total)
            mlflow.log_metric("TOTAL_BIAS_actual_block_time", bias_total)
            mlflow.log_metric("TOTAL_ABS_BIAS_actual_block_time", abs_bias_total)
            mlflow.log_metric("TOTAL_P90_COVERAGE_actual_block_time", p90_coverage)
            mlflow.log_metric("TOTAL_P90_COVERAGE_GAP_pct", p90_coverage_gap)
            mlflow.log_metric("BASELINE_DELTA", mae_total - baseline_mae)
            mlflow.log_metric("BASELINE_MAE_actual_block_time", baseline_mae)
            mlflow.log_metric("HOLDOUT_WIN_RATE_VS_SCHEDULE_pct", holdout_win_rate)
            cv_mae_total = float(best_block["best_cv_mae_sec"])
            overfit_diff_sec = mae_total - cv_mae_total
            mlflow.log_metric("OOT_MAE_actual_block_time_sec", mae_total)
            mlflow.log_metric("OVERFIT_DIFF_actual_block_time_sec", overfit_diff_sec)
            if pd.notna(monthly_mae_std):
                mlflow.log_metric("HOLDOUT_MONTHLY_MAE_STD_sec", float(monthly_mae_std))

            try:
                hist = spark.sql(f"DESCRIBE HISTORY {settings.TRAINING_DATASET_TABLE}").limit(1).collect()
                if hist:
                    mlflow.log_param("training_dataset_table_version", hist[0]["version"])
                    mlflow.log_param("training_dataset_table_timestamp", str(hist[0]["timestamp"]))
            except Exception:
                pass

            mlflow.log_param(
                "databricks_runtime",
                spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion", "unknown"),
            )

            pip_requirements = []
            if sklearn is not None:
                pip_requirements.append(f"scikit-learn=={sklearn.__version__}")
            pip_requirements.extend([
                f"pandas=={pd.__version__}",
                f"numpy=={np.__version__}",
            ])

            fe = FeatureEngineeringClient()
            base_for_spec = _base_training_df(spark, settings).limit(256)
            _assert_lookup_contract(base_for_spec, settings)
            training_set_meta = _create_fs_training_set(fe, base_for_spec, settings, spark)

            from mlflow.models import ModelSignature
            from mlflow.types import Schema, ColSpec, DataType

            _S2M = {
                "string": DataType.string, "boolean": DataType.boolean,
                "int": DataType.integer, "integer": DataType.integer,
                "tinyint": DataType.integer, "smallint": DataType.integer,
                "bigint": DataType.long, "long": DataType.long,
                "float": DataType.float, "double": DataType.double,
                "date": DataType.datetime, "timestamp": DataType.datetime,
            }
            _ldf = training_set_meta.load_df()
            _in_specs = []
            for f in _ldf.schema.fields:
                _st = f.dataType.simpleString()
                if _st not in _S2M:
                    print(f"[WARN] signature: dtype '{_st}' kolumny '{f.name}' spoza mapy -> fallback double")
                _in_specs.append(
                    ColSpec(_S2M.get(_st, DataType.double), f.name, required=not f.nullable)
                )
            _out_specs = [
                ColSpec(DataType.double, "pred_actual_block_time_sec"),
                ColSpec(DataType.double, "pred_actual_block_time_p90_sec"),
            ]
            full_sig = ModelSignature(inputs=Schema(_in_specs), outputs=Schema(_out_specs))

            fe.log_model(
                model=wrapper_model,
                artifact_path="model",
                flavor=mlflow.pyfunc,
                training_set=training_set_meta,
                signature=full_sig,
                infer_input_example=True,
                pip_requirements=pip_requirements,
                artifacts=artifacts,
            )

            logged = mlflow.last_logged_model()
            model_uri = logged.model_uri

            mlflow.models.set_signature(model_uri, full_sig)

            mlflow.register_model(model_uri, settings.UC_MODEL_NAME)

        print(
            f"Wynik (Out-of-Sample): MAE={mae_total/60:.2f} min | "
            f"Bias={bias_total/60:.2f} min | |Bias|={abs_bias_total/60:.2f} min | "
            f"P90 coverage={p90_coverage:.2f}% | Win rate vs schedule={holdout_win_rate:.2f}% | "
            f"Poprawa vs schedule={(baseline_mae - mae_total)/60:.2f} min"
        )
        if pd.notna(monthly_mae_std):
            print(f"📆 Monthly MAE std (holdout): {monthly_mae_std/60:.2f} min")

        result = {
            "run_id": run.info.run_id,
            "experiment_path": settings.EXPERIMENT_PATH,
            "training_dataset_table": settings.TRAINING_DATASET_TABLE,
            "eval_clean_dataset_table": settings.EVAL_CLEAN_DATASET_TABLE,
            "eval_all_dataset_table": settings.EVAL_ALL_DATASET_TABLE,
            "holdout_split_date": str(split_date),
            "training_rows_pdf": int(len(full_pdf)),
            "train_rows": int(len(train_pdf)),
            "valid_rows": int(len(valid_pdf)),
            "mae_total_sec": mae_total,
            "bias_total_sec": bias_total,
            "abs_bias_total_sec": abs_bias_total,
            "p90_coverage_pct": p90_coverage,
            "baseline_mae_sec": baseline_mae,
            "holdout_win_rate_pct": holdout_win_rate,
            "monthly_mae_std_sec": None if pd.isna(monthly_mae_std) else float(monthly_mae_std),
            "best_params": {
                "taxi_out": best_out["best_params"],
                "airborne": best_air["best_params"],
                "taxi_in": best_in["best_params"],
                "block": best_block["best_params"],
            },
        }
        return result
