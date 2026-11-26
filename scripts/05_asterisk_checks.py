"""
Verify the two hypotheses about the '*' sentinel raised in research:
1. sumoffen == '*'  <->  sumissue == 'N' (no summons issued)?
2. beat == '*'  concentrated in specific precincts, or spread evenly?
"""
import pandas as pd
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)

    print("=== sumoffen == '*' vs sumissue ===")
    star_rows = df["sumoffen"] == "*"
    print(f"sumoffen == '*': {star_rows.sum():,} rows")
    print(df.loc[star_rows, "sumissue"].value_counts(dropna=False))
    print()
    print("for comparison, sumissue distribution across ALL rows:")
    print(df["sumissue"].value_counts(dropna=False))

    print("\n=== beat == '*' by precinct (pct) ===")
    beat_star = df["beat"] == "*"
    by_pct = df.groupby("pct").agg(
        n_stops=("beat", "size"),
        n_star=("beat", lambda s: (s == "*").sum()),
    )
    by_pct["pct_star"] = (by_pct["n_star"] / by_pct["n_stops"] * 100).round(1)
    by_pct = by_pct.sort_values("pct_star", ascending=False)

    print(f"Overall: {beat_star.sum():,} of {len(df):,} rows ({beat_star.mean()*100:.1f}%) have beat == '*'")
    print(f"\nPrecincts (out of {len(by_pct)}) where beat=='*' is 0%: {(by_pct['pct_star'] == 0).sum()}")
    print(f"Precincts where beat=='*' is 100%: {(by_pct['pct_star'] == 100).sum()}")
    print("\nTop 10 precincts by % of stops with beat=='*':")
    print(by_pct.head(10).to_string())
    print("\nBottom 10 precincts by % of stops with beat=='*':")
    print(by_pct.tail(10).to_string())

    print("\n=== beat == '*' by month (from datestop, format MMDDYYYY) ===")
    month = df["datestop"].astype(str).str.zfill(8).str[:2]
    by_month = df.assign(month=month).groupby("month").agg(
        n_stops=("beat", "size"),
        n_star=("beat", lambda s: (s == "*").sum()),
    )
    by_month["pct_star"] = (by_month["n_star"] / by_month["n_stops"] * 100).round(1)
    print(by_month.to_string())