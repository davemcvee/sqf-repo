"""
Data Understanding: summary statistics for the most important attributes.

Splits attributes into three shapes since one stats table doesn't fit all of
them sensibly:
- numeric (age, weight, height, timing fields): mean/median/mode/std/range
- Y/N flags (actions, suspicion factors, force types, weapons found): rate of Y
- multi-category fields (sex, race, build, etc.): value counts / mode
"""
import pandas as pd
from pathlib import Path

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

NUMERIC = ["age", "weight", "ht_feet", "ht_inch", "perobs", "perstop", "timestop"]

BINARY_FLAGS = [
    "frisked", "searched", "contrabn", "arstmade", "sumissue", "offunif",
    "cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth", "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other",
    "rf_vcrim", "rf_othsw", "rf_attir", "rf_vcact", "rf_rfcmp", "rf_verbl", "rf_knowl", "rf_furt", "rf_bulg",
    "ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc", "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other",
    "sb_hdobj", "sb_outln", "sb_admis", "sb_other",
    "pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep", "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other",
    "pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap",
]

CATEGORICAL = ["sex", "race", "build", "city", "inout", "trhsloc", "forceuse"]


def hhmm_to_minutes(s):
    """timestop is encoded HHMM (e.g. 2100 = 9:00 PM), not linear minutes --
    the numeric gap from 1259 to 1300 is 41, not 1, so mean/std need this
    converted to real minutes-since-midnight first. Median/mode are rank-based
    and are unaffected (HHMM is monotonic with time-of-day), so those still
    use the raw column."""
    return (s // 100) * 60 + (s % 100)


def numeric_stats(df):
    rows = []
    for col in NUMERIC:
        s = df[col].dropna()
        mean_basis = hhmm_to_minutes(s) if col == "timestop" else s
        rows.append({
            "variable": col, "count": len(s), "mean": round(mean_basis.mean(), 2),
            "median": s.median(), "mode": s.mode().iloc[0], "std": round(mean_basis.std(), 2),
            "min": s.min(), "max": s.max(), "range": s.max() - s.min(),
        })
    return pd.DataFrame(rows)


def binary_rates(df):
    rows = []
    for col in BINARY_FLAGS:
        n_yes = (df[col] == "Y").sum()
        rate = n_yes / len(df)
        rows.append({"variable": col, "n_yes": n_yes, "rate": round(rate, 4)})
    return pd.DataFrame(rows).sort_values("rate", ascending=False)


if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)

    print("=== NUMERIC ATTRIBUTES ===")
    print("(note: timestop mean/std are in real minutes-since-midnight, not raw HHMM -- see hhmm_to_minutes)")
    print(numeric_stats(df).to_string(index=False))

    mean_min = hhmm_to_minutes(df["timestop"].dropna()).mean()
    print(f"\ntimestop mean, converted back to clock time: {int(mean_min // 60):02d}:{int(mean_min % 60):02d}")

    print("\n=== BINARY FLAGS, sorted by rate (highest first) ===")
    print(binary_rates(df).to_string(index=False))

    print("\n=== CATEGORICAL ATTRIBUTES (mode + top values) ===")
    for col in CATEGORICAL:
        vc = df[col].value_counts(dropna=False, normalize=True).round(4)
        print(f"\n{col} (mode = {df[col].mode().iloc[0]}):")
        print(vc.head(6).to_string())