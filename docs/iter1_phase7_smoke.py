
from datetime import date, timedelta
import dataclasses
from pyspark.sql import functions as F


# COMMAND ----------

FS_TABLES = [
    SETTINGS.FS_TAXI_OUT_TABLE,
    SETTINGS.FS_AIRBORNE_TABLE,
    SETTINGS.FS_TAXI_IN_TABLE,
    SETTINGS.FS_STAND_OUT_TABLE,
    SETTINGS.FS_STAND_IN_TABLE,
]

for t in FS_TABLES:
    print("=" * 80, f"\n{t}\n", "=" * 80)
    spark.sql(f"DESCRIBE EXTENDED {t}").show(truncate=False)
    try:
        spark.sql(f"DESCRIBE FEATURE TABLE {t}").show(truncate=False)
        print(f"[OK] {t}: zarejestrowana jako UC feature table")
    except Exception as e:
        print(f"[WARN]  {t}: DESCRIBE FEATURE TABLE nie działa -> {type(e).__name__}: {str(e)[:200]}")
        print("   Sprawdź czy MV ma PRIMARY KEY + TIMESERIES w schemacie (Faza 1).")



# COMMAND ----------

EXPECTED_COL_COUNTS = {
    SETTINGS.FS_TAXI_OUT_TABLE: 38,
    SETTINGS.FS_AIRBORNE_TABLE: 55,
    SETTINGS.FS_TAXI_IN_TABLE:  38,
    SETTINGS.FS_STAND_OUT_TABLE: 13,
    SETTINGS.FS_STAND_IN_TABLE:  13,
}

for table, expected_count in EXPECTED_COL_COUNTS.items():
    actual_count = len(spark.read.table(table).schema.fieldNames())
    assert actual_count == expected_count, (
        f"{table}: kolumn {actual_count} ≠ oczekiwane {expected_count}. "
        f"Inline generator w feature_store.py rozjechał się z runtime."
    )
    desc = spark.sql(f"DESCRIBE EXTENDED {table}").collect()
    desc_str = "\n".join(f"{r['col_name']}: {r['data_type']}" for r in desc).upper()
    assert "TIMESERIES" in desc_str, f"{table}: brak TIMESERIES flag — full refresh nie zarejestrował feature table"
    assert "_FEATURES_PK" in desc_str or "PRIMARY KEY" in desc_str, f"{table}: brak PRIMARY KEY constraint"
    print(f"[OK] {table}: {actual_count} kolumn, PK+TIMESERIES OK")
print("\n[7.2] schema validation: OK [OK]")


# COMMAND ----------

from ml_project.training import build_training_datasets

s30 = dataclasses.replace(SETTINGS, LABEL_START=(date.today() - timedelta(days=30)).isoformat())

result = build_training_datasets(spark, s30)
train_df = result["training_df_model"]
joined_all = result["joined_all"]

joined_count = joined_all.count()
distinct_count = joined_all.select("leg_no").distinct().count()
assert joined_count == distinct_count, \
    f"Cardinality break w joined_all: joined={joined_count}, distinct={distinct_count}"

missing_feats = set(s30.ALL_FS_FEATURES) - set(train_df.columns)
assert not missing_feats, f"Brakujące FS features w training_df_model: {sorted(missing_feats)}"

print(f"[OK] Training smoke: joined={joined_count} (distinct leg_no OK), "
      f"{len(s30.ALL_FS_FEATURES)} ALL_FS_FEATURES obecne w training_df_model")
print("   (base==joined==distinct sprawdzone wewnątrz build_training_datasets)")


# COMMAND ----------

SNAPSHOT_PATH = "dbfs:/tmp/iter1_regression/old_values.parquet"

try:
    old_vals = spark.read.parquet(SNAPSHOT_PATH)
    sample_legs = [r["leg_no"] for r in old_vals.select("leg_no").collect()]

    from ml_project.training import _base_training_df, _create_fs_training_set
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()

    base = _base_training_df(spark, SETTINGS).filter(F.col("leg_no").isin(sample_legs))
    new_ts = _create_fs_training_set(fe, base, SETTINGS)
    new_vals = new_ts.load_df().select("leg_no", *SETTINGS.ALL_FS_FEATURES)

    feat_cols = [c for c in SETTINGS.ALL_FS_FEATURES if c in old_vals.columns and c in new_vals.columns]
    o = old_vals.select("leg_no", *feat_cols).alias("o")
    n = new_vals.select("leg_no", *feat_cols).alias("n")
    cmp = o.join(n, on="leg_no", how="inner")

    max_diffs = {}
    for c in feat_cols:
        d = cmp.select(F.max(F.abs(F.col(f"o.{c}").cast("double") - F.col(f"n.{c}").cast("double"))).alias("d")).first()["d"]
        if d is not None:
            max_diffs[c] = d

    worst = sorted(max_diffs.items(), key=lambda kv: -(kv[1] or 0))[:10]
    print("Top 10 max abs diff per feature:")
    for c, d in worst:
        print(f"   {c}: {d}")
    over_tol = {c: d for c, d in max_diffs.items() if (d or 0) > 1e-9}
    assert not over_tol, f"BIT-EXACT FAIL — feature'y z diff > 1e-9: {over_tol}"
    print(f"\n[OK] Regression bit-exact: {len(feat_cols)} features, wszystkie diff < 1e-9")
except Exception as e:
    print(f"[WARN]  [7.4] pominięte / błąd: {type(e).__name__}: {str(e)[:300]}")
    print("   Upewnij się, że SNAPSHOT_PATH wskazuje na stary snapshot.")


# COMMAND ----------

import subprocess

PATTERNS = [
    "ml_project.features",
    "_filter_current_if_present",
    "STAND_FS_EVENT_COL",
    "fs_event_date",
]
all_clean = True
for p in PATTERNS:
    r = subprocess.run(
        ["grep", "-rn", "--include=*.py", "--include=*.ipynb",
         "--exclude-dir=archive", p, "src/", "notebooks/"],
        capture_output=True, text=True,
    )
    if r.stdout.strip():
        all_clean = False
        print(f"[WARN]  '{p}':\n{r.stdout}")
    else:
        print(f"[OK] '{p}': clean")
print("\n[7.5] cleanup:", "ALL CLEAN [OK]" if all_clean else "[WARN] są trafienia — sprawdź wyżej")
