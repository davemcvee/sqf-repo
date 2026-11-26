"""
Sweep every column for '*'-style sentinel placeholders (the pattern found
in perstop='**' and beat='*'), so we catch all of them at once instead of
one column at a time.
"""
import pandas as pd
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    n = len(df)

    print(f"{'column':12s} {'star-like count':>15s} {'% of rows':>10s}  example values")
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            continue
        vals = df[col].dropna().astype(str)
        is_star = vals.str.fullmatch(r"\*+")
        count = int(is_star.sum())
        if count > 0:
            examples = sorted(vals[is_star].unique())[:3]
            pct = count / n * 100
            print(f"{col:12s} {count:15,d} {pct:9.1f}%  {examples}")