from __future__ import annotations

from typing import Any, Dict, TYPE_CHECKING

import mlflow
from databricks.feature_engineering import FeatureEngineeringClient
from delta.tables import DeltaTable
from pyspark.sql import functions as F, Window as W
from pyspark.sql.types import (
    BooleanType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from .common import add_marker_columns, apply_data_quality_rules, configure_runtime, enrich_with_local_context
from .widgets import ensure_cdf_stream_widgets, get_bool_widget, get_widget_value
if TYPE_CHECKING:
    from .settings import FlightDelaySettings


def run_cdf_scoring(spark, dbutils, settings: "FlightDelaySettings") -> Dict[str, Any]:
    """
    Refaktoryzowany entrypoint scoringu CDF.
    Zachowuje istniejącą logikę biznesową notebooka, ale przenosi ją do modułu.
    """
    configure_runtime(settings, spark=spark)

    KEY_COL = settings.SINK_PRIMARY_KEY
    DEP_TS_COL = "dep_sched_dt"
    ARR_TS_COL = "arr_sched_dt"
    LEG_STATE_COL = "leg_state"

    LOOKBACK_DAYS = int(getattr(settings, "SHADOW_SYNC_LOOKBACK_DAYS", 2))
    LOOKAHEAD_MONTHS = int(getattr(settings, "SHADOW_SYNC_LOOKAHEAD_MONTHS", 3))
    MAX_FILES_PER_TRIGGER = int(getattr(settings, "CDF_MAX_FILES_PER_TRIGGER", 1000))
    MODEL_AIRCRAFT_FEATURE_COL = getattr(settings, "MODEL_AIRCRAFT_FEATURE_COL", "ac_registration")
    AC_REGISTRATION_PREFIX_LEN = int(getattr(settings, "AC_REGISTRATION_PREFIX_LEN", 4) or 0)

    ensure_cdf_stream_widgets(dbutils)
    run_bootstrap = get_bool_widget(dbutils, "RUN_BOOTSTRAP", True)
    reset_chkpt = get_bool_widget(dbutils, "RESET_CHECKPOINT", False)
    starting_version = get_widget_value(dbutils, "STARTING_VERSION", "").strip()

    def ensure_table_columns(table_name: str, expected_columns: dict) -> None:
        existing_cols = {field.name.lower() for field in spark.read.table(table_name).schema.fields}
        missing_defs = [f"{col} {dtype}" for col, dtype in expected_columns.items() if col.lower() not in existing_cols]
        if missing_defs:
            spark.sql(f"ALTER TABLE {table_name} ADD COLUMNS ({', '.join(missing_defs)})")
            print(
                f"Dodano brakujące kolumny do {table_name}: "
                f"{', '.join([d.split()[0] for d in missing_defs])}"
            )

    EXPECTED_SINK_COLUMNS = {
        KEY_COL: "BIGINT",
        "pred_taxi_out_sec": "DOUBLE",
        "pred_airborne_sec": "DOUBLE",
        "pred_taxi_in_sec": "DOUBLE",
        "pred_actual_block_time_sec": "DOUBLE",
        "pred_taxi_out_p90_sec": "DOUBLE",
        "pred_airborne_p90_sec": "DOUBLE",
        "pred_taxi_in_p90_sec": "DOUBLE",
        "pred_actual_block_time_p90_sec": "DOUBLE",
        "pred_block_delay_sec": "DOUBLE",
        "scheduled_block_time_sec": "DOUBLE",
        "is_active": "BOOLEAN",
        "inactive_reason": "STRING",
        "scored_at": "TIMESTAMP",
        "source_commit_version": "BIGINT",
        "source_commit_timestamp": "TIMESTAMP",
        "model_uri": "STRING",
        "dep_sched_dt": "TIMESTAMP",
        "arr_sched_dt": "TIMESTAMP",
        "dep_ap_sched": "STRING",
        "arr_ap_sched": "STRING",
        "dep_stand": "STRING",
        "arr_stand": "STRING",
        "ac_subtype": "STRING",
        "ac_registration": "STRING",
        "last_change_type": "STRING",
        "missing_feature_count": "INT",
        "batch_id": "BIGINT",
        "is_operationally_active": "BOOLEAN",
        "prediction_status": "STRING",
        "effective_actual_block_time_sec": "DOUBLE",
        "effective_block_delay_sec": "DOUBLE",
        "model_pred_actual_block_time_sec_raw": "DOUBLE",
        "model_pred_block_delay_sec_raw": "DOUBLE",
        "hours_to_departure_at_prediction": "DOUBLE",
        "missing_feature_count_with_stands": "INT",

    }

    EXPECTED_EVENTS_COLUMNS = {
        "logged_at": "TIMESTAMP",
        **EXPECTED_SINK_COLUMNS,
    }

    def ensure_scoring_tables() -> None:
        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.SHADOW_TABLE} (
              {KEY_COL} BIGINT,
              leg_state STRING,
              leg_type STRING,
              dep_sched_dt TIMESTAMP,
              arr_sched_dt TIMESTAMP,
              dep_ap_sched STRING,
              arr_ap_sched STRING,
              dep_ap_actual STRING,
              arr_ap_actual STRING,
              ac_subtype STRING,
              ac_registration STRING,
              ac_owner STRING,
              marker STRING,
              row_hash BIGINT,
              __END_AT TIMESTAMP
            )
            USING DELTA
            TBLPROPERTIES (
              'delta.enableChangeDataFeed' = 'true'
            )
            """
        )

        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.SINK_TABLE} (
              {', '.join([f'{col} {dtype}' for col, dtype in EXPECTED_SINK_COLUMNS.items()])}
            )
            USING DELTA
            """
        )
        ensure_table_columns(settings.SINK_TABLE, EXPECTED_SINK_COLUMNS)

        spark.sql(
            f"""
            CREATE TABLE IF NOT EXISTS {settings.EVENTS_SINK_TABLE} (
              {', '.join([f'{col} {dtype}' for col, dtype in EXPECTED_EVENTS_COLUMNS.items()])}
            )
            USING DELTA
            """
        )
        ensure_table_columns(settings.EVENTS_SINK_TABLE, EXPECTED_EVENTS_COLUMNS)

    def sync_source_to_shadow() -> None:
        print(f"Synchronizacja: {settings.LABELS_TABLE} -> {settings.SHADOW_TABLE} ...")

        source_df = spark.read.table(settings.LABELS_TABLE)
        if "__END_AT" in source_df.columns:
            source_df = source_df.filter(F.col("__END_AT").isNull())

        source_df = source_df.filter(
            F.col(DEP_TS_COL) >= F.expr(f"current_timestamp() - INTERVAL {LOOKBACK_DAYS} DAYS")
        )
        source_df = source_df.filter(
            F.col(DEP_TS_COL) <= F.expr(f"current_timestamp() + INTERVAL {LOOKAHEAD_MONTHS} MONTHS")
        )
        source_df = source_df.filter(F.col("counter") == 0)

        shadow_cols = spark.read.table(settings.SHADOW_TABLE).columns
        final_cols = [c for c in shadow_cols if c in source_df.columns and c != "row_hash"]
        source_df = source_df.select(*final_cols)

        hash_cols = [F.col(c) for c in final_cols if c not in [KEY_COL, "__END_AT"]]
        source_df = source_df.withColumn("row_hash", F.xxhash64(*hash_cols))

        delta_shadow = DeltaTable.forName(spark, settings.SHADOW_TABLE)
        delete_cond = (
            f"t.{DEP_TS_COL} >= current_timestamp() - INTERVAL {LOOKBACK_DAYS} DAYS "
            f"AND t.{DEP_TS_COL} <= current_timestamp() + INTERVAL {LOOKAHEAD_MONTHS} MONTHS"
        )

        (
            delta_shadow.alias("t")
            .merge(source_df.alias("s"), f"t.{KEY_COL} = s.{KEY_COL}")
            .whenMatchedUpdateAll(condition="t.row_hash IS NULL OR t.row_hash != s.row_hash")
            .whenNotMatchedBySourceDelete(condition=delete_cond)
            .whenNotMatchedInsertAll()
            .execute()
        )
        print("Synchronizacja zakończona")

    def dtype_name(t) -> str:
        return str(t).split(".")[-1].lower()

    def mlflow_type_to_spark_dtype(type_name: str):
        t = (type_name or "double").lower()
        if "string" in t:
            return StringType()
        if "bool" in t:
            return BooleanType()
        if "int" in t and "long" not in t:
            return IntegerType()
        if "long" in t:
            return LongType()
        if "date" in t:
            return DateType()
        if "timestamp" in t or "datetime" in t:
            return TimestampType()
        return DoubleType()

    def _schema_specs(schema_obj):
        if schema_obj is None:
            return []
        return getattr(schema_obj, "inputs", schema_obj)

    def get_model_signature_io(model_uri: str):
        info = mlflow.models.get_model_info(model_uri)
        input_specs = _schema_specs(info.signature.inputs)
        output_specs = _schema_specs(info.signature.outputs)

        input_cols = [s.name for s in input_specs]
        input_types = {s.name: dtype_name(s.type) for s in input_specs}
        output_cols = [s.name for s in output_specs]
        output_types = {s.name: dtype_name(s.type) for s in output_specs}
        return input_cols, input_types, output_cols, output_types

    INPUT_COLS, INPUT_TYPES, OUTPUT_COLS, OUTPUT_TYPES = get_model_signature_io(settings.MODEL_URI)

    # Identify columns that come from feature_store (populated by score_batch via feature lookup).
    # These must NOT be pre-added to the df — score_batch skips lookup if column already exists.
    import yaml as _yaml
    _fe_spec_path = mlflow.artifacts.download_artifacts(
        artifact_uri=f"{settings.MODEL_URI}/data/feature_store/feature_spec.yaml"
    )
    with open(_fe_spec_path, "r") as _f:
        _fe_spec = _yaml.safe_load(_f.read())
    _FS_COLUMNS = set()
    for _col_dict in _fe_spec.get("input_columns", []):
        for _col_name, _col_info in _col_dict.items():
            if _col_info.get("source") == "feature_store":
                _FS_COLUMNS.add(_col_name)
    del _fe_spec, _fe_spec_path

    out_schema = StructType(
        [
            StructField(col_name, mlflow_type_to_spark_dtype(OUTPUT_TYPES.get(col_name, "double")), True)
            for col_name in OUTPUT_COLS
        ]
    )
    fe = FeatureEngineeringClient()

    def ensure_signature_columns(df):
        """Pre-coalesce base columns before fe.score_batch.

        Plan A. Iterujemy po kolumnach OBECNYCH w batch_prep, żeby:
          - policzyć missing_feature_count z kolumn bazowych (przed dołożeniem braków;
            cond_too_many_missing/final_cols nadal tego używają downstream),
          - castować każdą kolumnę do typu, który NAPRAWDĘ ma signature (nie hard-coded
            double), i fillna dla INT/LONG. Po retreningu inner signature ma LONG dla
            leg_no, INT dla counter itd. — pre-cast wszystkiego na double dałby float64
            w pandas i konflikt z LONG/INT w signature.
        Następnie dokładamy brakujące kolumny signature jako NULL — score_batch wymaga
        WSZYSTKICH kolumn z training df, a SHADOW_TABLE bywa węższa.
        """
        out = df

        base_present = [
            c for c in INPUT_COLS
            if c in out.columns and not c.startswith("marker_") and "stand_" not in c
            and c not in _FS_COLUMNS
        ]
        if base_present:
            null_checks = [F.when(F.col(c).isNull(), 1).otherwise(0) for c in base_present]
            missing_count_expr = null_checks[0]
            for expr in null_checks[1:]:
                missing_count_expr = missing_count_expr + expr
            out = out.withColumn("missing_feature_count", missing_count_expr.cast("int"))
        else:
            out = out.withColumn("missing_feature_count", F.lit(0).cast("int"))

        for c in list(out.columns):
            if c not in INPUT_TYPES:
                continue
            t = INPUT_TYPES[c].lower()

            if "long" in t:
                out = out.withColumn(c, F.coalesce(F.col(c).cast("long"), F.lit(0).cast("long")))
            elif "int" in t and "long" not in t:
                out = out.withColumn(c, F.coalesce(F.col(c).cast("int"), F.lit(0).cast("int")))
            elif "double" in t or "float" in t:
                out = out.withColumn(c, F.col(c).cast("double"))
            elif "string" in t:
                default = "UNKNOWN" if c in ("dep_ap_sched", "arr_ap_sched", "dep_stand", "arr_stand") else None
                if default:
                    out = out.withColumn(c, F.coalesce(F.col(c), F.lit(default)))
            elif "bool" in t:
                out = out.withColumn(c, F.col(c).cast("boolean"))
            elif "date" in t and "datetime" not in t and "timestamp" not in t:
                out = out.withColumn(c, F.col(c).cast("date"))
            elif "timestamp" in t or "datetime" in t:
                out = out.withColumn(c, F.col(c).cast("timestamp"))


        # Skip feature_store columns — score_batch will populate them via feature lookup.
        # Adding them here as NULL would prevent score_batch from querying feature tables.
        for c, type_str in INPUT_TYPES.items():
            if c in out.columns:
                continue
            if c in _FS_COLUMNS:
                continue
            t = type_str.lower()
            if "long" in t:
                out = out.withColumn(c, F.lit(0).cast("long"))
            elif "int" in t and "long" not in t:
                out = out.withColumn(c, F.lit(0).cast("int"))
            elif "double" in t or "float" in t:
                out = out.withColumn(c, F.lit(None).cast("double"))
            elif "bool" in t:
                out = out.withColumn(c, F.lit(None).cast("boolean"))
            elif "timestamp" in t or "datetime" in t:
                out = out.withColumn(c, F.lit(None).cast("timestamp"))
            elif "date" in t:
                out = out.withColumn(c, F.lit(None).cast("date"))
            else:
                out = out.withColumn(c, F.lit(None).cast("string"))

        if "event_date" in out.columns:
            out = out.withColumn("event_date", F.to_date(F.col("event_date")))

        return out

    def _pred_expr(field_name: str):
        return F.col(f"prediction.{field_name}") if field_name in OUTPUT_COLS else F.lit(None).cast("double")

    def add_derived_cols(df):
        df_derived = (
            df.withColumn("event_ts", F.col(DEP_TS_COL).cast("timestamp"))
            .withColumn("event_date", F.to_date(F.col(DEP_TS_COL)))
            .withColumn(
                "scheduled_block_time_sec",
                (F.col(ARR_TS_COL).cast("long") - F.col(DEP_TS_COL).cast("long")).cast("double"),
            )
            .withColumn("isLO", F.when(F.col("ac_owner") == "LO", 1).otherwise(0))
        )

        df_derived = add_marker_columns(df_derived, max_marker_length=settings.MAX_MARKER_LENGTH)

        return df_derived

    def apply_inactivation_rules(df):
        df_dq = apply_data_quality_rules(df)

        is_arr = F.col(LEG_STATE_COL) == settings.LEG_STATE_ARR
        is_div = (
            (F.col("arr_ap_actual").isNotNull() & (F.col("arr_ap_actual") != F.col("arr_ap_sched")))
            | (F.col("dep_ap_actual").isNotNull() & (F.col("dep_ap_actual") != F.col("dep_ap_sched")))
        )
        is_excl = ~F.col("leg_type").isin(settings.INCLUDED_LEG_TYPES)
        is_del = F.col("_change_type") == "delete"
        is_too_far_future = F.col(DEP_TS_COL) > F.current_timestamp() + F.expr(
            f"INTERVAL {LOOKAHEAD_MONTHS} MONTHS"
        )

        should_inactivate_stream = is_arr | is_div | is_excl | is_del | is_too_far_future

        return (
            df_dq.withColumn("should_inactivate", (~F.col("is_active")) | should_inactivate_stream).withColumn(
                "inactive_reason",
                F.when(is_del, F.lit("DELETED"))
                .when(is_excl, F.lit("EXCLUDED_LEG"))
                .when(is_arr, F.lit("ARR_STATE"))
                .when(is_div, F.lit("DIVERSION"))
                .when(is_too_far_future, F.lit("TOO_FAR_FUTURE"))
                .otherwise(F.col("inactive_reason")),
            )
        )

    def microbatch_upsert(batch_df, batch_id):
        if batch_df.isEmpty():
            return

        batch_df = batch_df.filter(F.col("_change_type") != "update_preimage")
        batch_df = batch_df.filter(
            F.col(DEP_TS_COL) >= F.expr(f"current_timestamp() - INTERVAL {LOOKBACK_DAYS} DAYS")
        )

        batch_count = batch_df.count()
        if batch_count == 0:
            return

        print(f"[Batch {batch_id}] Pobrano z CDF {batch_count} czystych zdarzeń. Przetwarzam...")

        batch_dq = apply_inactivation_rules(batch_df)
        batch_prep = enrich_with_local_context(batch_dq, spark)
        batch_prep = add_derived_cols(batch_prep)

        if (
            MODEL_AIRCRAFT_FEATURE_COL == "ac_registration"
            and MODEL_AIRCRAFT_FEATURE_COL in batch_prep.columns
            and AC_REGISTRATION_PREFIX_LEN > 0
        ):
            batch_prep = batch_prep.withColumn(
                MODEL_AIRCRAFT_FEATURE_COL,
                F.when(
                    F.col(MODEL_AIRCRAFT_FEATURE_COL).isNull(),
                    F.lit(None).cast("string"),
                ).otherwise(F.substring(F.col(MODEL_AIRCRAFT_FEATURE_COL), 1, AC_REGISTRATION_PREFIX_LEN)),
            )

        leg_misc_raw = spark.read.table(settings.LEG_MISC_TABLE)
        if "__END_AT" in leg_misc_raw.columns:
            leg_misc_raw = leg_misc_raw.filter(F.col("__END_AT").isNull())

        leg_misc_current = (
            leg_misc_raw
            .withColumn("dep_stand", F.upper(F.trim(F.col("dep_stand"))))
            .withColumn("arr_stand", F.upper(F.trim(F.col("arr_stand"))))
            .select("leg_no", "dep_stand", "arr_stand")
        )

        batch_prep = batch_prep.join(leg_misc_current, on="leg_no", how="left")
        batch_prep = batch_prep.fillna(
            {
                "dep_ap_sched": "UNKNOWN",
                "arr_ap_sched": "UNKNOWN",
                "dep_stand": "UNKNOWN",
                "arr_stand": "UNKNOWN",
            }
        )
        batch_prep = (
            batch_prep.withColumn(
                "route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched"))
            )
            .withColumn(
                "stand_id_out", F.concat_ws("_", F.col("dep_ap_sched"), F.col("dep_stand"))
            )
            .withColumn(
                "stand_id_in", F.concat_ws("_", F.col("arr_ap_sched"), F.col("arr_stand"))
            )
            .withColumn(
                "event_date", F.coalesce(F.col("event_date"), F.to_date(F.lit("1970-01-01")))
            )
        )

        # Cap event_date to MIN of max dates across ALL required feature tables.
        # score_batch uses exact-match on event_date (TIMESERIES PIT not recognized for MVs).
        # Using MIN ensures every table has a matching row for the capped date.
        _max_dates = []
        for _ft_table in [
            settings.FT_AIRPORT_DAILY_TAXI_OUT_TABLE,
            settings.FT_ROUTE_DAILY_STATS_TABLE,
            settings.FT_AIRPORT_DAILY_TAXI_IN_TABLE,
        ]:
            _d = spark.read.table(_ft_table).agg(F.max("event_date")).first()[0]
            if _d:
                _max_dates.append(_d)
        if _max_dates:
            _safe_max_date = min(_max_dates)
            batch_prep = batch_prep.withColumn(
                "event_date",
                F.least(F.col("event_date"), F.lit(_safe_max_date)),
            )

        try:
            batch_prep = ensure_signature_columns(batch_prep)

            scored_df = fe.score_batch(
                model_uri=settings.MODEL_URI,
                df=batch_prep,
                result_type=out_schema,
                env_manager="local",
            )

            scored_df = (
                scored_df.withColumn("pred_taxi_out_sec", _pred_expr("pred_taxi_out_sec"))
                .withColumn("pred_airborne_sec", _pred_expr("pred_airborne_sec"))
                .withColumn("pred_taxi_in_sec", _pred_expr("pred_taxi_in_sec"))
                .withColumn("pred_actual_block_time_sec", _pred_expr("pred_actual_block_time_sec"))
                .withColumn("pred_taxi_out_p90_sec", _pred_expr("pred_taxi_out_p90_sec"))
                .withColumn("pred_airborne_p90_sec", _pred_expr("pred_airborne_p90_sec"))
                .withColumn("pred_taxi_in_p90_sec", _pred_expr("pred_taxi_in_p90_sec"))
                .withColumn("pred_actual_block_time_p90_sec", _pred_expr("pred_actual_block_time_p90_sec"))
            )



            scored_df = scored_df.withColumn(
                "pred_block_delay_sec", F.col("pred_actual_block_time_sec") - F.col("scheduled_block_time_sec")
            )
            scored_df = (
                scored_df.withColumn(
                    "model_pred_actual_block_time_sec_raw", F.col("pred_actual_block_time_sec").cast("double")
                ).withColumn("model_pred_block_delay_sec_raw", F.col("pred_block_delay_sec").cast("double"))
            )
            scored_df = scored_df.withColumn(
                "hours_to_departure_at_prediction",
                (
                    F.col("dep_sched_dt").cast("long") - F.coalesce(F.col("_commit_timestamp"), F.current_timestamp()).cast("long")
                ) / F.lit(3600.0)
            )

        except Exception as e:
            print(f"[ERROR] KRYTYCZNY BŁĄD PODCZAS score_batch w paczce {batch_id}:")
            raise e

        cond_too_many_missing = F.col("missing_feature_count") > settings.MAX_MISSING_FEATURES
        is_cold_start = (
            (F.coalesce(F.col("has_hist_dep_30d").cast("int"), F.lit(0)) == 0)
            | (F.coalesce(F.col("has_hist_route_30d").cast("int"), F.lit(0)) == 0)
            | (F.coalesce(F.col("has_hist_arr_30d").cast("int"), F.lit(0)) == 0)
        )
        cond_fallback = cond_too_many_missing | is_cold_start

        scored_df = (
            scored_df.withColumn(
                "is_active",
                F.when(cond_fallback | F.col("should_inactivate"), F.lit(False)).otherwise(F.col("is_active")),
            )
            .withColumn(
                "inactive_reason",
                F.when(F.col("should_inactivate"), F.col("inactive_reason"))
                .when(is_cold_start, F.lit("COLD_START_FALLBACK"))
                .when(cond_too_many_missing, F.lit("TOO_MANY_MISSING_FEATURES"))
                .otherwise(F.col("inactive_reason")),
            )
            .withColumn(
                "pred_actual_block_time_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.col("scheduled_block_time_sec"))
                .otherwise(F.col("pred_actual_block_time_sec")),
            )
            .withColumn(
                "pred_actual_block_time_p90_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.col("scheduled_block_time_sec"))
                .otherwise(F.col("pred_actual_block_time_p90_sec")),
            )
            .withColumn(
                "pred_block_delay_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(0.0).cast("double"))
                .otherwise(F.col("pred_block_delay_sec")),
            )
            .withColumn(
                "pred_taxi_out_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_taxi_out_sec")),
            )
            .withColumn(
                "pred_airborne_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_airborne_sec")),
            )
            .withColumn(
                "pred_taxi_in_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_taxi_in_sec")),
            )
            .withColumn(
                "pred_taxi_out_p90_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_taxi_out_p90_sec")),
            )
            .withColumn(
                "pred_airborne_p90_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_airborne_p90_sec")),
            )
            .withColumn(
                "pred_taxi_in_p90_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(None).cast("double"))
                .otherwise(F.col("pred_taxi_in_p90_sec")),
            )
            .withColumn("is_operationally_active", (~F.col("should_inactivate")).cast("boolean"))
            .withColumn(
                "prediction_status",
                F.when(F.col("should_inactivate"), F.lit("INACTIVE_OPERATIONAL"))
                .when(is_cold_start, F.lit("COLD_START_FALLBACK"))
                .when(cond_too_many_missing, F.lit("TOO_MANY_MISSING_FEATURES_FALLBACK"))
                .otherwise(F.lit("MODEL_OK")),
            )
            .withColumn(
                "effective_actual_block_time_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.col("scheduled_block_time_sec").cast("double"))
                .otherwise(F.col("model_pred_actual_block_time_sec_raw").cast("double")),
            )
            .withColumn(
                "effective_block_delay_sec",
                F.when(F.col("should_inactivate"), F.lit(None).cast("double"))
                .when(cond_fallback, F.lit(0.0).cast("double"))
                .otherwise(F.col("model_pred_block_delay_sec_raw").cast("double")),
            )
        )

        try:
            if "_commit_version" in scored_df.columns:
                scored_df = scored_df.withColumn("source_commit_version", F.col("_commit_version"))
            if "_commit_timestamp" in scored_df.columns:
                scored_df = scored_df.withColumn("source_commit_timestamp", F.col("_commit_timestamp"))
            if "_change_type" in scored_df.columns:
                scored_df = scored_df.withColumn("last_change_type", F.col("_change_type"))

            scored_df = scored_df.withColumn("model_uri", F.lit(settings.MODEL_URI))

            final_cols = [
                KEY_COL,
                "pred_taxi_out_sec",
                "pred_airborne_sec",
                "pred_taxi_in_sec",
                "pred_actual_block_time_sec",
                "pred_taxi_out_p90_sec",
                "pred_airborne_p90_sec",
                "pred_taxi_in_p90_sec",
                "pred_actual_block_time_p90_sec",
                "pred_block_delay_sec",
                "model_pred_actual_block_time_sec_raw",
                "model_pred_block_delay_sec_raw",
                "hours_to_departure_at_prediction",
                "scheduled_block_time_sec",
                "missing_feature_count",
                "is_active",
                "inactive_reason",
                "scored_at",
                "source_commit_version",
                "source_commit_timestamp",
                "model_uri",
                "dep_sched_dt",
                "arr_sched_dt",
                "dep_ap_sched",
                "arr_ap_sched",
                "dep_stand",
                "arr_stand",
                "ac_subtype",
                "ac_registration",
                "last_change_type",
                "batch_id",
                "is_operationally_active",
                "prediction_status",
                "effective_actual_block_time_sec",
                "effective_block_delay_sec",
            ]

            actions = scored_df.select(*[c for c in final_cols if c in scored_df.columns])
            for c in final_cols:
                if c not in actions.columns:
                    actions = actions.withColumn(c, F.lit(None))

            actions = actions.withColumn("batch_id", F.lit(batch_id).cast("long"))
            actions = actions.withColumn(
                "scored_at",
                F.current_timestamp(),
            )
            actions_with_ts = actions.withColumn("logged_at", F.current_timestamp())

            win_dedup = W.partitionBy(KEY_COL).orderBy(
                F.col("source_commit_version").desc(), F.col("source_commit_timestamp").desc()
            )
            actions_for_merge = actions.withColumn("rn", F.row_number().over(win_dedup)).filter(
                F.col("rn") == 1
            ).drop("rn")

            tgt = DeltaTable.forName(spark, settings.SINK_TABLE)
            merge_update_cols = [c for c in actions_for_merge.columns if c != KEY_COL]
            safe_set_map = {c: f"s.`{c}`" for c in merge_update_cols}
            insert_map = {c: f"s.`{c}`" for c in actions_for_merge.columns}

            update_condition = """
            (t.source_commit_version IS NULL)
            OR (s.source_commit_version > t.source_commit_version)
            OR (
                s.source_commit_version = t.source_commit_version
                AND COALESCE(t.batch_id, -1) < COALESCE(s.batch_id, -1)
            )
            OR (
                s.source_commit_version = t.source_commit_version
                AND COALESCE(t.batch_id, -1) = COALESCE(s.batch_id, -1)
                AND COALESCE(t.scored_at, TIMESTAMP '1970-01-01') < COALESCE(s.scored_at, TIMESTAMP '1970-01-01')
            )
            """

            (
                tgt.alias("t")
                .merge(actions_for_merge.alias("s"), f"t.{KEY_COL} = s.{KEY_COL}")
                .whenMatchedUpdate(condition=update_condition, set=safe_set_map)
                .whenNotMatchedInsert(values=insert_map)
                .execute()
            )

            events_tgt = DeltaTable.forName(spark, settings.EVENTS_SINK_TABLE)
            (
                events_tgt.alias("t")
                .merge(
                    actions_with_ts.alias("s"),
                    f"t.{KEY_COL} = s.{KEY_COL} "
                    "AND t.source_commit_version = s.source_commit_version "
                    "AND t.last_change_type = s.last_change_type",
                )
                .whenNotMatchedInsert(values={c: f"s.{c}" for c in actions_with_ts.columns})
                .execute()
            )

            print(f"[OK] [Batch {batch_id}] Zapisano poprawnie!")
        except Exception as e:
            import traceback

            print(f"[ERROR] KRYTYCZNY BŁĄD W MIKROBACZU {batch_id} (ETAP ZAPISU):")
            traceback.print_exc()
            raise e

    if run_bootstrap:
        print("RUN_BOOTSTRAP=True -> tworzę/uzupełniam tabele i synchronizuję shadow.")
        ensure_scoring_tables()
        sync_source_to_shadow()
    else:
        print("RUN_BOOTSTRAP=False -> pomijam bootstrap tabel i sync shadow.")

    if reset_chkpt:
        if not settings.ALLOW_CHECKPOINT_RESET:
            raise PermissionError(
                f"[ERROR] CRITICAL SECURITY ERROR: Zresetowanie checkpointu na środowisku "
                f"{settings.ENV.upper()}! Widget 'RESET_CHECKPOINT' musi być False."
            )

        print("[WARN] UWAGA: Tryb deweloperski. Trwa resetowanie checkpointu strumienia i czyszczenie tabel...")
        dbutils.fs.rm(settings.CHECKPOINT_PATH, True)
        spark.sql(f"TRUNCATE TABLE {settings.SINK_TABLE}")
        spark.sql(f"TRUNCATE TABLE {settings.EVENTS_SINK_TABLE}")
        print("[OK] Tabele SINK i EVENTS zostały wyzerowane do czysta.")

        if not starting_version:
            starting_version = "0"
            print(
                "[AUDYT FIX] Automatycznie wymuszono STARTING_VERSION = 0 po resecie, "
                "aby uniknąć 'fresh start' i pustych tabel."
            )

    stream_reader = (
        spark.readStream.format("delta")
        .option("readChangeFeed", "true")
        .option("maxFilesPerTrigger", MAX_FILES_PER_TRIGGER)
    )

    if starting_version:
        print(f"Wymuszony start strumienia od wersji: {starting_version}")
        stream_reader = stream_reader.option("startingVersion", starting_version)
    else:
        print("Start strumienia z użyciem istniejącego checkpointu.")

    stream_df = stream_reader.table(settings.SHADOW_TABLE)

    print(f"Uruchamiam scoring stream w trybie AvailableNow (Środowisko: {settings.ENV.upper()})...")
    (
        stream_df.writeStream.foreachBatch(microbatch_upsert)
        .option("checkpointLocation", settings.CHECKPOINT_PATH)
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )

    print("[OK] Scoring zakończony pomyślnie. Tabele SINK oraz EVENTS zostały zaktualizowane.")
    return {
        "env": settings.ENV,
        "run_bootstrap": run_bootstrap,
        "reset_checkpoint": reset_chkpt,
        "starting_version": starting_version or None,
        "checkpoint_path": settings.CHECKPOINT_PATH,
        "shadow_table": settings.SHADOW_TABLE,
        "sink_table": settings.SINK_TABLE,
        "events_table": settings.EVENTS_SINK_TABLE,
        "model_uri": settings.MODEL_URI,
    }
