
# COMMAND ----------

print("=== [0] FS-miss + stand-only-miss precheck ===")
from datetime import date, timedelta
from pyspark.sql import functions as F
from ml_project.common import get_cleaned_flight_data
from ml_project.training import _add_fs_lookup_keys

recent_window_start = (date.today() - timedelta(days=7)).isoformat()
base = _add_fs_lookup_keys(get_cleaned_flight_data(spark, recent_window_start, active_only=True))

fs_out = spark.read.table(SETTINGS.FS_TAXI_OUT_TABLE)
fs_air = spark.read.table(SETTINGS.FS_AIRBORNE_TABLE)
fs_in = spark.read.table(SETTINGS.FS_TAXI_IN_TABLE)
fs_stand_out = spark.read.table(SETTINGS.FS_STAND_OUT_TABLE).withColumnRenamed("stand_id", "stand_id_out")
fs_stand_in = spark.read.table(SETTINGS.FS_STAND_IN_TABLE).withColumnRenamed("stand_id", "stand_id_in")

mo = fs_out.groupBy("dep_ap_sched").agg(F.min("event_date").alias("min_o"))
ma = fs_air.groupBy("route_id").agg(F.min("event_date").alias("min_a"))
mi = fs_in.groupBy("arr_ap_sched").agg(F.min("event_date").alias("min_i"))
mso = fs_stand_out.groupBy("stand_id_out").agg(F.min("event_date").alias("min_so"))
msi = fs_stand_in.groupBy("stand_id_in").agg(F.min("event_date").alias("min_si"))

j = (
    base.select("leg_no", "dep_ap_sched", "arr_ap_sched", "route_id", "stand_id_out", "stand_id_in", "event_date")
    .join(mo, "dep_ap_sched", "left").join(ma, "route_id", "left").join(mi, "arr_ap_sched", "left")
    .join(mso, "stand_id_out", "left").join(msi, "stand_id_in", "left")
)

def _hit(mincol):
    return F.col(mincol).isNotNull() & (F.col(mincol) < F.col("event_date"))

j = (
    j.withColumn("hit_out", _hit("min_o")).withColumn("hit_air", _hit("min_a"))
    .withColumn("hit_in", _hit("min_i")).withColumn("hit_so", _hit("min_so")).withColumn("hit_si", _hit("min_si"))
)

total = j.count()
miss_any = ~(F.col("hit_out") & F.col("hit_air") & F.col("hit_in") & F.col("hit_so") & F.col("hit_si"))
is_cold_start = ~F.col("hit_out") | ~F.col("hit_air") | ~F.col("hit_in")
route_airport_all_hit = F.col("hit_out") & F.col("hit_air") & F.col("hit_in")
stand_miss = ~F.col("hit_so") | ~F.col("hit_si")
stand_only_miss = route_airport_all_hit & stand_miss

miss_any_n = j.filter(miss_any).count()
cold_n = j.filter(is_cold_start).count()
stand_only_n = j.filter(stand_only_miss).count()

FS_MISS_RATE = miss_any_n / total if total else 0.0
STAND_ONLY_MISS_RATE = stand_only_n / total if total else 0.0

print(f"Total rows (last 7d, scoring-repr. as-of prior): {total}")
print(f"  FS_MISS_RATE (≥1 miss):           {miss_any_n:>6} ({100*FS_MISS_RATE:.2f}%)  [kontekst, skala rozjazdu RAW]")
print(f"  cold-start (dep/route/arr miss):  {cold_n:>6} ({100*cold_n/total:.2f}%)  [fallback nadpisuje served]")
print(f"  STAND_ONLY_MISS (escape vector):  {stand_only_n:>6} ({100*STAND_ONLY_MISS_RATE:.2f}%)  ← KLUCZOWA dla A1")
print(f"     (dep/route/arr trafiają, stand chybia -> no cold_start -> served divergence JEŚLI stand_count=LONG)")
print()
print(f"FS_MISS_RATE = {FS_MISS_RATE:.4f}   STAND_ONLY_MISS_RATE = {STAND_ONLY_MISS_RATE:.4f}")
print("""
INTERPRETACJA — STAND_ONLY_MISS_RATE jest realny TYLKO jeśli stand_count ma signature dtype=long.
To rozstrzyga [1c-iii] dtype-probe (po treningu). Bramka decyzyjna = STAND_ONLY_MISS_RATE AND STAND_COUNT_COALESCED:
  - STAND_COUNT_COALESCED=False (signature double) -> brak rozjazdu niezależnie od miss rate -> Plan A safe.
  - STAND_COUNT_COALESCED=True + STAND_ONLY_MISS_RATE < 0.01 -> Plan A (rozjazd marginalny, weryfikuj Faza 7).
  - STAND_COUNT_COALESCED=True + STAND_ONLY_MISS_RATE ≥ 0.01 -> escape realny -> Plan B albo A3 (patrz stuby).
FS_MISS_RATE tylko szacuje skalę rozjazdu RAW-diagnostic (model_pred_*_raw), nie served. Ostateczna
weryfikacja: Faza 7 regression na SERVED kolumnach Plan A vs Plan B.
""")


# COMMAND ----------

import os
import yaml
from mlflow.artifacts import download_artifacts

model_uri = SETTINGS.MODEL_URI
local_dir = download_artifacts(artifact_uri=model_uri)

spec_path = None
for root, _dirs, files in os.walk(local_dir):
    for fn in files:
        if fn == "feature_spec.yaml":
            spec_path = os.path.join(root, fn)
            break

assert spec_path is not None, (
    "BRAK feature_spec.yaml w artefaktach modelu — model NIE został zalogowany przez "
    "fe.log_model(training_set=...). To blocker: score_batch (Plan A) nie zadziała."
)

spec = yaml.safe_load(open(spec_path))
print("feature_spec.yaml:", spec_path)
print(yaml.safe_dump(spec, sort_keys=False)[:4000])

EXPECTED_TABLES = {
    SETTINGS.FS_TAXI_OUT_TABLE.split(".")[-1],
    SETTINGS.FS_AIRBORNE_TABLE.split(".")[-1],
    SETTINGS.FS_TAXI_IN_TABLE.split(".")[-1],
    SETTINGS.FS_STAND_OUT_TABLE.split(".")[-1],
    SETTINGS.FS_STAND_IN_TABLE.split(".")[-1],
}
spec_text = yaml.safe_dump(spec)
missing_tables = [t for t in EXPECTED_TABLES if t not in spec_text]
missing_keys = [
    k for k in ("dep_ap_sched", "route_id", "arr_ap_sched", "stand_id_out", "stand_id_in", "event_date")
    if k not in spec_text
]
print("\nBrakujące tabele w spec:", missing_tables or "NONE [OK]")
print("Brakujące klucze w spec:", missing_keys or "NONE [OK]")
assert not missing_tables and not missing_keys, "STOP: feature_spec niespójny — napraw przed spike 0.3."
print("\n[1c-i] feature_spec kontrakt OK [OK]")


# COMMAND ----------

import mlflow
from ml_project.training import _base_training_df

info = mlflow.models.get_model_info(model_uri)
sig_inputs = {s.name for s in info.signature.inputs.inputs}

base_cols = set(_base_training_df(spark, SETTINGS).columns)

FS_TABLES_KEYS = [
    (SETTINGS.FS_TAXI_OUT_TABLE, set(SETTINGS.PK_TAXI_OUT) | {"event_date"}),
    (SETTINGS.FS_AIRBORNE_TABLE, set(SETTINGS.PK_AIRBORNE) | {"event_date"}),
    (SETTINGS.FS_TAXI_IN_TABLE,  set(SETTINGS.PK_TAXI_IN)  | {"event_date"}),
    (SETTINGS.FS_STAND_OUT_TABLE, {"stand_id", "event_date"}),
    (SETTINGS.FS_STAND_IN_TABLE,  {"stand_id", "event_date"}),
]
lookup_outputs = set()
for tbl, keys in FS_TABLES_KEYS:
    cols = set(spark.table(tbl).columns)
    lookup_outputs |= (cols - keys)

resolved = base_cols | lookup_outputs

missing = sig_inputs - resolved
extras = resolved - sig_inputs

print(f"signature inputs:        {len(sig_inputs)}")
print(f"resolved (base+lookup):  {len(resolved)}")
print(f"MISSING (sig - resolved): {sorted(missing) if missing else 'NONE [OK]'}")
print(f"extras  (resolved - sig): {len(extras)} kolumn (oczekiwane, info)")
assert not missing, (
    f"STOP przed spike 0.3: model deklaruje wejścia których score_batch nie dostarczy: {sorted(missing)}. "
    "To rozjazd kontraktu (train zawęża inaczej niż feature_spec/base resolve). Pogodzić zanim Plan A."
)
print("\n[1c-ii] signature ⊆ resolved — kontrakt inferencji spójny [OK]")


# COMMAND ----------

sig = mlflow.models.get_model_info(model_uri).signature
sig_types = {s.name: str(s.type).split(".")[-1].lower() for s in sig.inputs.inputs}

def _is_coalesced(dtype):
    return not ("float" in dtype or "double" in dtype)

int_long_sig = {c for c, t in sig_types.items() if t in ("long", "integer", "int")}

cold_start_tables = [SETTINGS.FS_TAXI_OUT_TABLE, SETTINGS.FS_AIRBORNE_TABLE, SETTINGS.FS_TAXI_IN_TABLE]
stand_tables = [SETTINGS.FS_STAND_OUT_TABLE, SETTINGS.FS_STAND_IN_TABLE]
cold_start_cols = set().union(*[set(spark.read.table(t).columns) for t in cold_start_tables])
stand_cols = set().union(*[set(spark.read.table(t).columns) for t in stand_tables])
fs_provided = cold_start_cols | stand_cols

escape_fs_neutralized = sorted(int_long_sig & cold_start_cols)
ESCAPE_VECTORS_FS = sorted(int_long_sig & stand_cols)
BASE_ESCAPE_VECTORS = sorted(int_long_sig - fs_provided)

_overlap = int_long_sig & cold_start_cols & stand_cols
assert not _overlap, (
    f"INT/LONG w route I stand table — klasyfikacja dwuznaczna: {sorted(_overlap)}. "
    f"Taka kolumna escape'uje przez stand-miss bez route-miss; A3 musiałby ją objąć. STOP."
)
_partition = set(escape_fs_neutralized) | set(ESCAPE_VECTORS_FS) | set(BASE_ESCAPE_VECTORS)
assert _partition == int_long_sig, (
    f"Niepełna partycja INT/LONG: {sorted(int_long_sig - _partition)} poza wszystkimi klasami "
    f"(literówka w table sets?). STOP."
)
print(f"[OK] Partycja rozłączna+pełna: {len(int_long_sig)} INT/LONG = "
      f"{len(escape_fs_neutralized)} neutr + {len(ESCAPE_VECTORS_FS)} FS-escape + {len(BASE_ESCAPE_VECTORS)} base-escape")

print("INT/LONG w signature:", sorted(int_long_sig))
print(f"  klasa1 FS neutralizowane (cold_start): {escape_fs_neutralized}")
print(f"  klasa1 FS ESCAPE (stand):              {ESCAPE_VECTORS_FS}")
print(f"  klasa2 BASE ESCAPE (passthrough):      {BASE_ESCAPE_VECTORS}")

expected_fs = sorted(c for c in int_long_sig & stand_cols if c.startswith("stand_count_"))
STAND_COUNT_COALESCED = len(ESCAPE_VECTORS_FS) > 0
if ESCAPE_VECTORS_FS == expected_fs:
    print(f"\n[OK] FS escape = stand_count_* ({len(expected_fs)} kol). A3 (fillna 0) obejmuje pełny zbiór FS.")
else:
    extra = sorted(set(ESCAPE_VECTORS_FS) - set(expected_fs))
    print(f"\n[WARN]  FS escape MA DODATKOWE kolumny: {extra} — A3 musi je objąć ALBO inna semantyka -> STOP, raportuj.")

print(f"\nSTAND_COUNT_COALESCED = {STAND_COUNT_COALESCED}   ESCAPE_VECTORS_FS = {ESCAPE_VECTORS_FS}")
print(f"BASE_ESCAPE_VECTORS = {BASE_ESCAPE_VECTORS}")
print("""
ROZSTRZYGNIĘCIE:
  - ESCAPE_VECTORS_FS puste (stand_count=double) -> brak rozjazdu FS. Klasa 1 OK.
  - ESCAPE_VECTORS_FS = stand_count_* -> waga = STAND_ONLY_MISS_RATE [0]; fix A3 (fillna 0 w _predict_segment).
  - BASE_ESCAPE_VECTORS niepuste (local_hour_*/local_dow_* int) -> Plan A MUSI coalesce base-INT PRZED
    score_batch (replikuje ensure_signature_columns na base cols). Inaczej cichy rozjazd na nowych lotniskach.
    To NIE jest objęte przez A3 — to osobny krok w stubie Plan A.
Pełna parity Plan A = (base-INT coalesce przed score_batch) + (A3 fillna na ESCAPE_VECTORS_FS).
""")


# COMMAND ----------

from ml_project.common import get_cleaned_flight_data

base = (
    get_cleaned_flight_data(spark, SETTINGS.LABEL_START, active_only=False)
    .withColumn("route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")))
    .select("leg_no", "dep_ap_sched", "arr_ap_sched", "route_id", "event_date")
    .limit(50000)
)

fs_air = spark.table(SETTINGS.FS_AIRBORNE_TABLE)
print("fs_airborne schema:")
fs_air.printSchema()

fs_air_j = (
    fs_air.withColumnRenamed("event_date", "_fs_date")
    .drop("dep_ap_sched", "arr_ap_sched")
)
joined = base.join(
    fs_air_j,
    (base.route_id == fs_air_j.route_id) & (base.event_date == fs_air_j._fs_date),
    "left",
).drop(fs_air_j.route_id)

total = base.count()
matched = joined.filter(F.col("_fs_date").isNotNull()).count()
print(f"\nbase rows: {total:,} | matched (route_id+event_date): {matched:,} "
      f"| match rate: {matched/total:.1%}")
print("Oczekiwane: wysoki match rate (porównywalny z poprzednim dep+arr+event_date joinem).")
print("Jeśli ~0% -> route_id się nie matchuje, STOP i sprawdź concat_ws po obu stronach.")


# COMMAND ----------

import mlflow
import mlflow.pyfunc
from pyspark.sql import functions as F
from pyspark.sql.types import (
    StructType, StructField, StringType, BooleanType, IntegerType,
    LongType, DateType, TimestampType, DoubleType,
)
from ml_project.common import get_cleaned_flight_data

def _mlflow_type_to_spark(type_name):
    t = (type_name or "double").lower()
    if "string" in t: return StringType()
    if "bool" in t: return BooleanType()
    if "int" in t and "long" not in t: return IntegerType()
    if "long" in t: return LongType()
    if "date" in t: return DateType()
    if "timestamp" in t or "datetime" in t: return TimestampType()
    return DoubleType()

def _signature_io(model_uri):
    info = mlflow.models.get_model_info(model_uri)
    si = getattr(info.signature.inputs, "inputs", info.signature.inputs)
    so = getattr(info.signature.outputs, "inputs", info.signature.outputs)
    in_cols = [s.name for s in si]
    in_types = {s.name: str(s.type).split(".")[-1].lower() for s in si}
    out_cols = [s.name for s in so]
    out_types = {s.name: str(s.type).split(".")[-1].lower() for s in so}
    return in_cols, in_types, out_cols, out_types

INPUT_COLS, INPUT_TYPES, OUTPUT_COLS, OUTPUT_TYPES = _signature_io(SETTINGS.MODEL_URI)
out_schema = StructType([
    StructField(c, _mlflow_type_to_spark(OUTPUT_TYPES.get(c, "double")), True) for c in OUTPUT_COLS
])

def _build_manual_joined_static_batch_for_check(spark, settings, input_cols):
    fs_dates = (
        spark.read.table(settings.FS_TAXI_OUT_TABLE)
        .agg(F.min("event_date").alias("min_d"), F.max("event_date").alias("max_d"))
        .collect()[0]
    )
    print(f"   FS coverage (fs_taxi_out): {fs_dates['min_d']} -> {fs_dates['max_d']}")
    base = (
        get_cleaned_flight_data(spark, settings.LABEL_START, active_only=True)
        .filter(F.col("event_date").between(fs_dates["min_d"], fs_dates["max_d"]))
        .withColumn("route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")))
        .withColumn("stand_id_out", F.concat_ws("_", F.col("dep_ap_sched"), F.col("dep_stand")))
        .withColumn("stand_id_in", F.concat_ws("_", F.col("arr_ap_sched"), F.col("arr_stand")))
        .limit(50)
    )

    def _exact_join(b, tbl, key_in_base, fs_key_name=None):
        fs = spark.table(tbl)
        fs_key = fs_key_name or key_in_base
        fs = fs.withColumnRenamed("event_date", "_fsdate")
        drop_collisions = [c for c in fs.columns if c in b.columns and c != fs_key]
        fs = fs.drop(*drop_collisions)
        cond = (F.col(key_in_base) == F.col(fs_key)) & (F.col("event_date") == F.col("_fsdate"))
        out = b.join(fs, on=cond, how="left").drop("_fsdate")
        if fs_key != key_in_base and fs_key in out.columns:
            out = out.drop(fs_key)
        return out

    b = base
    b = _exact_join(b, settings.FS_TAXI_OUT_TABLE, settings.PK_TAXI_OUT[0])
    b = _exact_join(b, settings.FS_AIRBORNE_TABLE, "route_id")
    b = _exact_join(b, settings.FS_TAXI_IN_TABLE, settings.PK_TAXI_IN[0])
    b = _exact_join(b, settings.FS_STAND_OUT_TABLE, "stand_id_out", "stand_id")
    b = _exact_join(b, settings.FS_STAND_IN_TABLE,  "stand_id_in",  "stand_id")

    for c in input_cols:
        if c not in b.columns:
            b = b.withColumn(c, F.lit(None).cast(_mlflow_type_to_spark(INPUT_TYPES.get(c, "string"))))

    if "avg_taxi_out_7d" in b.columns:
        hit = b.filter(F.col("avg_taxi_out_7d").isNotNull()).count()
        total = b.count()
        print(f"   FS hit rate w test batch: {hit}/{total} wierszy ma realne FS values (avg_taxi_out_7d)")
        if hit == 0:
            print("   [WARN]  Wszystkie FS NULL — viability check będzie niemiarodajny przy NaN predykcjach")
    return b

print("=== [4-pre] Plan B viability check ===")
try:
    test_udf = mlflow.pyfunc.spark_udf(spark, SETTINGS.MODEL_URI, result_type=out_schema)
    test_df = _build_manual_joined_static_batch_for_check(spark, SETTINGS, INPUT_COLS)
    pred_df = test_df.withColumn("pred", test_udf(*[F.col(c) for c in INPUT_COLS]))
    sample = pred_df.select("pred").limit(3).collect()
    PLAN_B_VIABLE = True
    print("[OK] Plan B VIABLE: pred_udf ładuje fe.log_model model i predykuje (brak wyjątku).")
    print(f"   Sample predictions: {sample}")
    def _leaf_vals(v):
        if hasattr(v, "asDict"):
            return list(v.asDict().values())
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    def _is_real(v):
        return v is not None and not (isinstance(v, float) and v != v)
    flat = [x for row in sample for x in _leaf_vals(row["pred"])]
    has_real_pred = any(_is_real(x) for x in flat)
    if has_real_pred:
        print("   Predictions look real (non-NaN sample present).")
    else:
        print("   [WARN]  Predictions all NaN/null — viable, ale test helper miał luki w joinach.")
        print("   To NIE jest blocker dla Plan B, tylko ograniczenie tego testu.")
        print("   Correctness weryfikowana bit-exact w Fazie 7.")
except Exception as e:
    PLAN_B_VIABLE = False
    print(f"[ERROR] Plan B NIE viable: {type(e).__name__}")
    print(f"   Message: {str(e)[:600]}")
    print("   → Plan A (score_batch) MUSI się udać. Brak fallbacku w iteracji 1.")
    print("   → Jeśli spike [4] też padnie -> eskalacja:")
    print("     (a) rollback do mlflow.pyfunc.log_model (tracimy lineage UC↔model, ale PK+TIMESERIES")
    print("         w tabelach + FeatureLookup w trainingu zostają — ~50% wartości),")
    print("     (b) zamrożenie iteracji 1 do rozwiązania serializacji w spike.")


# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()
spike_result = {"status": None, "exc_type": None, "exc_msg": None, "latency_sec": None}

def _add_keys(df):
    return (
        df.withColumn("route_id", F.concat_ws("_", F.col("dep_ap_sched"), F.col("arr_ap_sched")))
        .withColumn("stand_id_out", F.concat_ws("_", F.col("dep_ap_sched"), F.col("dep_stand")))
        .withColumn("stand_id_in", F.concat_ws("_", F.col("arr_ap_sched"), F.col("arr_stand")))
    )

def spike_microbatch(batch_df, batch_id):
    import time
    if batch_id > 0 or batch_df.rdd.isEmpty():
        return
    prep = _add_keys(batch_df).withColumn("event_date", F.to_date(F.col("dep_sched_dt")))
    t0 = time.time()
    try:
        pred = fe.score_batch(
            model_uri=SETTINGS.MODEL_URI,
            df=prep,
            result_type="double",
            env_manager="local",
        )
        n = pred.count()
        spike_result.update(status="OK", latency_sec=round(time.time() - t0, 2))
        print(f"[spike] fe.score_batch OK — {n} rows, {spike_result['latency_sec']}s")
    except Exception as e:
        spike_result.update(status="FAIL", exc_type=type(e).__name__, exc_msg=str(e)[:1500])
        print(f"[spike] FAIL: {spike_result['exc_type']}: {spike_result['exc_msg']}")

spike_ckpt = SETTINGS.CHECKPOINT_PATH.rstrip("/") + "_spike03"
(
    spark.readStream.option("readChangeFeed", "true").table(SETTINGS.SHADOW_TABLE)
    .writeStream.foreachBatch(spike_microbatch)
    .option("checkpointLocation", spike_ckpt)
    .trigger(availableNow=True)
    .start()
    .awaitTermination()
)

dbutils.fs.rm(spike_ckpt, recurse=True)

print("\n=== SPIKE 0.3 WYNIK ===")
print("spike_result:", spike_result)
print("PLAN_B_VIABLE:", PLAN_B_VIABLE)

print("""
JEŚLI spike FAIL z exc_type ~ Pickling/Serialization/Py4J — NIE deklaruj wyniku od razu:
   1) potwierdź env_manager='local'
   2) mlflow.pyfunc.load_model RAZ poza microbatch + broadcast/reuse, dopiero potem oceń
   (serializacja to problem do obejścia, nie blocker API)
JEŚLI spike OK: porównaj predykcje vs pred_udf na tej samej kohorcie (<1e-6) + latencja (>2x? raportuj).
""")

_spike_ok = spike_result["status"] == "OK"
_expected_fs = sorted(c for c in ESCAPE_VECTORS_FS if c.startswith("stand_count_"))
_fs_escape_known = (ESCAPE_VECTORS_FS == _expected_fs)
_parity_risk = STAND_ONLY_MISS_RATE if STAND_COUNT_COALESCED else 0.0
print("=== DECYZJA (macierz 3D: PLAN_B_VIABLE × spike × parity_risk) ===")
print(f"   PLAN_B_VIABLE={PLAN_B_VIABLE}  spike_ok={_spike_ok}  STAND_COUNT_COALESCED={STAND_COUNT_COALESCED}")
print(f"   ESCAPE_VECTORS_FS={ESCAPE_VECTORS_FS}  BASE_ESCAPE_VECTORS={BASE_ESCAPE_VECTORS}")
print(f"   STAND_ONLY_MISS_RATE={STAND_ONLY_MISS_RATE:.4f}  -> parity_risk={_parity_risk:.4f}")
if BASE_ESCAPE_VECTORS:
    print(f"   [WARN] Plan A WYMAGA base-INT coalesce przed score_batch dla: {BASE_ESCAPE_VECTORS} (krok w stubie A).")
if not _fs_escape_known:
    print(f"-> STOP/ESKALACJA: nieznany FS escape {sorted(set(ESCAPE_VECTORS_FS)-set(_expected_fs))} — zbadaj semantykę przed A/B.")
elif _spike_ok and _parity_risk < 0.01:
    _why = "stand_count=double (escape nieaktywny)" if not STAND_COUNT_COALESCED else "stand-only-miss < 1%"
    print(f"-> PLAN A (score_batch) + base-coalesce. Served-safe FS ({_why}). Weryfikuj Faza 7.")
elif _spike_ok and _parity_risk >= 0.01 and PLAN_B_VIABLE:
    print("-> PLAN B. FS escape realny (stand_count=long + stand-only-miss≥1%).")
    print("   A3 (fillna(0) na ESCAPE_VECTORS_FS w _predict_segment, bez retreningu) = osobny PR domykający A.")
elif _spike_ok and _parity_risk >= 0.01 and not PLAN_B_VIABLE:
    print("-> ESKALACJA. parity_risk≥0.01 + brak Plan B -> A3 (lekki) ALBO A2 (retrain) blockerem iter1.")
elif (not _spike_ok) and PLAN_B_VIABLE:
    print("-> PLAN B (manual join + pred_udf, stan po Fazie 5-rename). parity nieistotne (B=baseline).")
else:
    print("-> TWARDY BLOCKER (spike FAIL + brak Plan B). Eskalacja:")
    print("   (a) rollback do mlflow.pyfunc.log_model (PK+TIMESERIES+FeatureLookup w trainingu zostają), albo")
    print("   (b) zamrożenie iteracji 1 do rozwiązania serializacji w spike.")
print("\n>>> Zaraportuj prowadzącemu: [0] FS_MISS_RATE + STAND_ONLY_MISS_RATE,")
print(">>> [1c-iii] ESCAPE_VECTORS_FS + BASE_ESCAPE_VECTORS + STAND_COUNT_COALESCED, [1c-ii] MISSING,")
print(">>> [3] match rate, [4-pre] PLAN_B_VIABLE, [4] spike exc_type/predykcje/latencja.")
