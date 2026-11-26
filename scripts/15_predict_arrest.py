"""
Report 3, Task 2: predict whether an arrest was made.

Arrest is the terminal event in a stop's sequence (stop -> reasons cited ->
frisk/search -> weapon/contraband found -> force -> arrest decision), so
unlike Task 1, every earlier-stage field is a legitimate predictor here --
using frisked/searched/weapon-found/force as inputs isn't leakage, it's
the actual causal chain leading to an arrest.

One feature configuration (the full "kitchen sink"), 3 algorithms, since
the interesting comparison for this task is model-to-model, not
config-to-config the way Task 1 and Task 3 are.
"""
import pandas as pd
from pathlib import Path
from model_pipeline import run_classification_suite

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

CS_COLS = ["cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
           "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]
AC_COLS = ["ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc",
           "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other"]
RF_COLS = ["rf_vcrim", "rf_othsw", "rf_attir", "rf_vcact", "rf_rfcmp",
           "rf_verbl", "rf_knowl", "rf_furt", "rf_bulg"]
SB_COLS = ["sb_hdobj", "sb_outln", "sb_admis", "sb_other"]
WEAPON_COLS = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]
FLAG_COLS = CS_COLS + AC_COLS + RF_COLS + SB_COLS

FORCE_TIERS_RANKED = [
    ("Weapon-level", ["pf_drwep", "pf_ptwep", "pf_baton", "pf_pepsp"]),
    ("Physical restraint", ["pf_wall", "pf_grnd", "pf_hcuff"]),
    ("Hands", ["pf_hands"]),
    ("Other", ["pf_other"]),
]


def force_tier(row):
    for label, cols in FORCE_TIERS_RANKED:
        if any(row[c] == "Y" for c in cols):
            return label
    return "No force"


def prepare(df):
    df = df.copy()
    df["any_weapon"] = df[WEAPON_COLS].eq("Y").any(axis=1).astype(int)
    df["any_contraband"] = (df["contrabn"] == "Y").astype(int)
    df["frisked_flag"] = (df["frisked"] == "Y").astype(int)
    df["searched_flag"] = (df["searched"] == "Y").astype(int)
    for c in FLAG_COLS:
        df[c] = (df[c] == "Y").astype(int)
    df["hour"] = (df["timestop"] // 100).astype(int)
    df["force_tier"] = df.apply(force_tier, axis=1)
    df["arrest_made"] = (df["arstmade"] == "Y").astype(int)
    return df


if __name__ == "__main__":
    df = prepare(pd.read_parquet(IN_PATH))
    print(f"n={len(df):,}, arrest rate={df['arrest_made'].mean():.3%}")
    print(df["force_tier"].value_counts())

    numeric_cols = FLAG_COLS + ["hour", "perstop", "perobs", "age", "ht_feet", "ht_inch", "weight",
                                 "any_weapon", "any_contraband", "frisked_flag", "searched_flag"]
    categorical_cols = ["pct", "race", "sex", "build", "force_tier"]

    metrics, fitted, importances, _ = run_classification_suite(
        df, numeric_cols, categorical_cols, "arrest_made",
        task_label="Arrest made (full feature set)",
        fig_prefix="fig24_arrest", class_names=["No arrest", "Arrest"])
    print("\n=== Metrics ===")
    print(metrics.round(4).to_string())

    print("\n--- Feature importance (top 12), Random Forest ---")
    print(importances["Random Forest"].head(12).round(4).to_string())
    print("\n--- Feature importance (top 12), Logistic Regression ---")
    print(importances["Logistic Regression"].head(12).round(4).to_string())