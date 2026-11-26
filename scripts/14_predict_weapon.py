"""
Report 3, Task 1: predict whether a weapon was found.

Two feature configurations are run and compared, not just one:
  Model A: behavior/context only (why the stop happened, what was observed
           during it, when/where) -- no demographics.
  Model B: Model A + demographics (race, sex, age, build, height, weight).

The comparison is the actual point of this task -- Reports 1-2 already
established that cs_bulge is the single strongest behavioral predictor of
a weapon find. What hasn't been tested is whether demographics add any
predictive power on top of behavior. If Model B doesn't meaningfully beat
Model A, that's evidence targeting is driven by behavior, not demographics.

frisked/searched are deliberately excluded from both configurations: a
weapon can typically only be recorded as "found" once one of those has
happened, so including them would mean predicting the outcome from a
near-mechanical gate on the outcome, not from anything informative.

rf_* (reason for frisk) is excluded for the same reason, not just cs_*/
ac_*: verified that any_rf (any rf_* flag set) matches frisked in 532,908
of 532,911 rows (99.9994%) -- it's functionally a duplicate of the
frisked field with extra detail about why, not independent information.
Weapon-found rate is ~9x higher among frisked/any_rf=True stops (2.0% vs
0.2%), so including rf_* would silently reintroduce the same
near-mechanical-gate leakage that excluding frisked itself was meant to
avoid. (This mirrors the rf_*/sb_* exclusion already applied in Task 3's
Bucket 1 -- an earlier version of this script inconsistently excluded
frisked/searched directly but still included rf_*, which Dave caught and
was fixed before the figures below were finalized.)
"""
import pandas as pd
from pathlib import Path
from model_pipeline import run_classification_suite

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

CS_COLS = ["cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
           "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]
AC_COLS = ["ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc",
           "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other"]
WEAPON_COLS = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]
FLAG_COLS = CS_COLS + AC_COLS


def prepare(df):
    df = df.copy()
    df["any_weapon"] = df[WEAPON_COLS].eq("Y").any(axis=1).astype(int)
    for c in FLAG_COLS:
        df[c] = (df[c] == "Y").astype(int)
    df["hour"] = (df["timestop"] // 100).astype(int)
    return df


if __name__ == "__main__":
    df = prepare(pd.read_parquet(IN_PATH))
    print(f"n={len(df):,}, weapon-found rate={df['any_weapon'].mean():.3%}")

    behavior_numeric = FLAG_COLS + ["hour", "perstop", "perobs"]
    behavior_cat = ["pct"]

    print("\n=== Model A: behavior/context only ===")
    metrics_a, fitted_a, imp_a, _ = run_classification_suite(
        df, behavior_numeric, behavior_cat, "any_weapon",
        task_label="Weapon found -- Model A (behavior only)",
        fig_prefix="fig23a_weapon_behavior", class_names=["No weapon", "Weapon"])
    print(metrics_a.round(4).to_string())

    demo_numeric = behavior_numeric + ["age", "ht_feet", "ht_inch", "weight"]
    demo_cat = behavior_cat + ["race", "sex", "build"]

    print("\n=== Model B: behavior + demographics ===")
    metrics_b, fitted_b, imp_b, _ = run_classification_suite(
        df, demo_numeric, demo_cat, "any_weapon",
        task_label="Weapon found -- Model B (behavior + demographics)",
        fig_prefix="fig23b_weapon_demo", class_names=["No weapon", "Weapon"])
    print(metrics_b.round(4).to_string())

    print("\n=== A vs B: does adding demographics improve F1? ===")
    compare = pd.DataFrame({
        "Model A F1 (behavior only)": metrics_a["F1-Score"],
        "Model B F1 (+ demographics)": metrics_b["F1-Score"],
    })
    compare["Delta"] = compare["Model B F1 (+ demographics)"] - compare["Model A F1 (behavior only)"]
    print(compare.round(4).to_string())

    print("\n--- Model B feature importance (top 10), Random Forest ---")
    print(imp_b["Random Forest"].head(10).round(4).to_string())