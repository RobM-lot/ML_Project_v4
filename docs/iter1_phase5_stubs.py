PLAN_A_MICROBATCH_UPSERT = '''
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

    print(f" [Batch {batch_id}] Pobrano z CDF {batch_count} czystych zdarzeń. Przetwarzam (Plan A / score_batch)...")

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
            F.when(F.col(MODEL_AIRCRAFT_FEATURE_COL).isNull(), F.lit(None).cast("string"))
             .otherwise(F.substring(F.col(MODEL_AIRCRAFT_FEATURE_COL), 1, AC_REGISTRATION_PREFIX_LEN)),
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

    batch_prep = (
        batch_prep
        .withColumn("route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")))
        .withColumn("stand_id_out", F.concat_ws("_", F.col("dep_ap_sched"), F.col("dep_stand")))
        .withColumn("stand_id_in", F.concat_ws("_", F.col("arr_ap_sched"), F.col("arr_stand")))
    )


    _base_int_cols = [
        c for c in INPUT_COLS
        if c in batch_prep.columns
        and INPUT_TYPES.get(c, "string").lower() in ("int", "integer", "long")
    ]
    for c in _base_int_cols:
        batch_prep = batch_prep.withColumn(c, F.coalesce(F.col(c), F.lit(0)))
    print(f"[Plan A] base-INT coalesce (signature int/long, non-FS): {_base_int_cols}")

    try:
        pred_df = fe.score_batch(
            model_uri=settings.MODEL_URI,
            df=batch_prep,
            result_type=out_schema,
            env_manager="local",
        )
        pred_df = pred_df.withColumnRenamed("prediction", "preds")
    except Exception as e:
        print(f"[ERROR] KRYTYCZNY BŁĄD score_batch w paczce {batch_id}:")
        raise e

    scored_df = ensure_signature_columns(pred_df)

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
        scored_df.withColumn("model_pred_actual_block_time_sec_raw", F.col("pred_actual_block_time_sec").cast("double"))
        .withColumn("model_pred_block_delay_sec_raw", F.col("pred_block_delay_sec").cast("double"))
    )
    scored_df = scored_df.withColumn(
        "hours_to_departure_at_prediction",
        (F.col("dep_sched_dt").cast("long") - F.coalesce(F.col("_commit_timestamp"), F.current_timestamp()).cast("long")) / F.lit(3600.0)
    )

'''


PLAN_B_REFERENCE_CURRENT_STATE = '''

    batch_prep = ensure_signature_columns(batch_prep)


    batch_prep = join_fs_exact(batch_prep, fs_out, settings.PK_TAXI_OUT, "event_date")
    batch_prep = join_fs_exact(batch_prep, fs_air, settings.PK_AIRBORNE, "event_date")
    batch_prep = join_fs_exact(batch_prep, fs_in,  settings.PK_TAXI_IN,  "event_date")

    fs_stand_out = spark.table(settings.FS_STAND_OUT_TABLE).withColumnRenamed("stand_id", "stand_id_out")
    batch_prep = join_fs_asof_latest(batch_prep, fs_stand_out, ["stand_id_out"], "event_date")
    fs_stand_in = spark.table(settings.FS_STAND_IN_TABLE).withColumnRenamed("stand_id", "stand_id_in")
    batch_prep = join_fs_asof_latest(batch_prep, fs_stand_in, ["stand_id_in"], "event_date")

    preds_struct = pred_udf(*[F.col(c) for c in INPUT_COLS])
    scored_df = batch_prep.withColumn("preds", preds_struct)

'''
