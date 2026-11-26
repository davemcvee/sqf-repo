"""
Initial data load + structure check for the NYC SQF 2012 dataset.

CRISP-DM: Data Understanding > Collect Initial Data / Describe Data.
"""
from sqf_data import load_codebook, load_sqf

if __name__ == "__main__":
    spec, labels = load_codebook()
    df = load_sqf()

    print(f"Rows: {len(df):,}   Columns: {df.shape[1]}")
    print(f"Spec rows: {len(spec)}   Value-label rows: {len(labels)}   Fields with labels: {labels['field'].nunique()}")

    print("\n--- columns in CSV but missing from spec ---")
    spec_vars = set(spec["variable"].astype(str))
    print(sorted(set(df.columns) - spec_vars) or "(none)")

    print("\n--- columns in spec but missing from CSV ---")
    print(sorted(spec_vars - set(df.columns)) or "(none)")

    print("\n--- % missing/blank per column (top 20) ---")
    blank_like = df.apply(lambda s: s.astype(str).str.strip().isin(["", "nan", "NaN"]))
    pct_missing = (df.isna() | blank_like).mean().sort_values(ascending=False) * 100
    print(pct_missing.head(20).round(1).to_string())

    print("\n--- dtypes summary ---")
    print(df.dtypes.value_counts().to_string())