"""
Data Understanding: Verify Data Quality (duplicates + outliers).

Reads the already-cleaned parquet from 02_clean_data.py -- this script only
inspects and reports, it doesn't overwrite that file. Decisions on how to
handle anything flagged here get made in a follow-up pass once reviewed.
"""
import pandas as pd
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns\n")

    # --- Duplicates ---
    print("=== DUPLICATES ===")
    full_dupes = df.duplicated().sum()
    print(f"Fully identical rows: {full_dupes:,}")

    key_cols = ["pct", "ser_num", "datestop"]
    key_dupes = df.duplicated(subset=key_cols).sum()
    print(f"Duplicate (precinct, serial number, date) -- the natural record key: {key_dupes:,}")
    if key_dupes:
        example = df[df.duplicated(subset=key_cols, keep=False)].sort_values(key_cols).head(6)
        print(example[key_cols + ["timestop", "race", "age"]].to_string(index=False))

    # --- Outliers / implausible values ---
    print("\n=== AGE ===")
    print(df["age"].describe().round(1).to_string())
    print("value counts, extremes:")
    print(df["age"].value_counts().sort_index().head(5).to_string())
    print("...")
    print(df["age"].value_counts().sort_index().tail(5).to_string())

    print("\n=== HEIGHT (ht_feet / ht_inch) ===")
    print("ht_feet range:", df["ht_feet"].min(), "-", df["ht_feet"].max())
    print("ht_inch range:", df["ht_inch"].min(), "-", df["ht_inch"].max())
    bad_inch = (df["ht_inch"] < 0) | (df["ht_inch"] > 11)
    print(f"ht_inch outside 0-11: {bad_inch.sum():,}")

    print("\n=== WEIGHT ===")
    print(df["weight"].describe().round(1).to_string())
    print(f"weight == 0: {(df['weight'] == 0).sum():,}")
    print(f"weight > 400: {(df['weight'] > 400).sum():,}")

    print("\n=== DOB ===")
    print("distinct dob values with count > 1000 (likely placeholders, not real birthdates):")
    dob_counts = df["dob"].value_counts()
    print(dob_counts[dob_counts > 1000].to_string())

    print("\n=== TIMESTOP / PEROBS / PERSTOP ===")
    print(f"timestop outside 0-2359: {((df['timestop'] < 0) | (df['timestop'] > 2359)).sum():,}")
    print("perobs (min of observation) describe:")
    print(df["perobs"].describe().round(1).to_string())

    perstop_num = pd.to_numeric(df["perstop"], errors="coerce")
    non_numeric = df["perstop"].notna() & perstop_num.isna()
    print(f"perstop: {non_numeric.sum():,} non-numeric entries (e.g. {df.loc[non_numeric, 'perstop'].unique()[:5].tolist()})")
    print("perstop describe (non-numeric entries excluded):")
    print(perstop_num.describe().round(1).to_string())

    print("\n=== XCOORD / YCOORD ===")
    print(f"xcoord == 0 or missing: {(df['xcoord'].isna() | (df['xcoord'] == 0)).sum():,}")
    print(f"ycoord == 0 or missing: {(df['ycoord'].isna() | (df['ycoord'] == 0)).sum():,}")