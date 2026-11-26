"""
Report 3, Task 3: predict the race of the person stopped -- the
"outside the box" task, testing a fairness question directly rather than
an operational one.

Two feature configurations, compared the same way as Task 1:
  Bucket 1 ("guilt"/evidence): cs_* + ac_* + weapon found + contraband
           found -- suspicion grounds and actual recovered evidence, not
           anything the officer chose to do in response.
  Bucket 2 (officer response): frisked, searched, force tier, arrest made,
           stop duration -- added on top of Bucket 1.

The question isn't "can race be predicted" in isolation (trivially yes,
if you hand the model precinct -- NYC neighborhoods are demographically
stratified, per Report 2). It's whether Bucket 2 predicts race BETTER
than Bucket 1 alone explains: if officer response adds real predictive
power on top of the underlying evidence, that's a disparate-treatment
signal that survives controlling for what was actually found. If it adds
nothing, escalation is proportionate to evidence, not race.

Deliberately excluded from both buckets: location (pct/xcoord/ycoord --
a near-direct race proxy in this dataset per Report 2's precinct
clusters, and not the question being asked here) and officer-recorded
physical descriptors (build, height, weight -- appearance fields that
would make race trivially recoverable and test nothing interesting).
rf_*/sb_* are excluded too: both are only recorded once frisked/searched
already happened, so they're subordinate detail of Bucket 2's actions,
not independent evidence -- including them would blur the two buckets.
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
BUCKET1_FLAGS = CS_COLS + AC_COLS

FORCE_TIERS_RANKED = [
    ("Weapon-level", ["pf_drwep", "pf_ptwep", "pf_baton", "pf_pepsp"]),
    ("Physical restraint", ["pf_wall", "pf_grnd", "pf_hcuff"]),
    ("Hands", ["pf_hands"]),
    ("Other", ["pf_other"]),
]

RACE_MAP = {"B": "Black", "Q": "White-Hisp.", "W": "White", "P": "Black-Hisp."}


def force_tier(row):
    for label, cols in FORCE_TIERS_RANKED:
        if any(row[c] == "Y" for c in cols):
            return label
    return "No force"


def prepare(df):
    df = df.copy()
    df["race_simple"] = df["race"].map(RACE_MAP)
    df = df[df["race_simple"].notna()].copy()

    df["any_weapon"] = df[WEAPON_COLS].eq("Y").any(axis=1).astype(int)
    df["any_contraband"] = (df["contrabn"] == "Y").astype(int)
    for c in BUCKET1_FLAGS:
        df[c] = (df[c] == "Y").astype(int)
    df["frisked_flag"] = (df["frisked"] == "Y").astype(int)
    df["searched_flag"] = (df["searched"] == "Y").astype(int)
    df["force_tier"] = df.apply(force_tier, axis=1)
    df["arrest_flag"] = (df["arstmade"] == "Y").astype(int)
    return df


if __name__ == "__main__":
    df = prepare(pd.read_parquet(IN_PATH))
    print(f"n={len(df):,} (4-category race subset)")
    print(df["race_simple"].value_counts(normalize=True).round(3))

    bucket1_numeric = BUCKET1_FLAGS + ["any_weapon", "any_contraband"]
    bucket2_numeric = bucket1_numeric + ["frisked_flag", "searched_flag", "arrest_flag", "perstop"]
    bucket2_cat = ["force_tier"]

    # must match sklearn's alphabetical label order: sorted(df["race_simple"].unique())
    class_order = sorted(df["race_simple"].unique().tolist())
    print("class order (alphabetical, matches confusion-matrix axes):", class_order)

    print("\n=== Bucket 1 only: suspicion/evidence grounds ===")
    metrics_1, fitted_1, imp_1, _ = run_classification_suite(
        df, bucket1_numeric, [], "race_simple",
        task_label="Predict race -- Bucket 1 (evidence only)",
        fig_prefix="fig25a_race_bucket1", class_names=class_order)
    print(metrics_1.round(4).to_string())

    print("\n=== Bucket 1 + 2: + officer response/escalation ===")
    metrics_2, fitted_2, imp_2, _ = run_classification_suite(
        df, bucket2_numeric, bucket2_cat, "race_simple",
        task_label="Predict race -- Bucket 1+2 (+ officer response)",
        fig_prefix="fig25b_race_bucket12", class_names=class_order)
    print(metrics_2.round(4).to_string())

    print("\n=== Bucket 1 vs Bucket 1+2: does officer response add predictive power? ===")
    compare = pd.DataFrame({
        "Bucket 1 F1 (macro)": metrics_1["F1-Score"],
        "Bucket 1+2 F1 (macro)": metrics_2["F1-Score"],
    })
    compare["Delta"] = compare["Bucket 1+2 F1 (macro)"] - compare["Bucket 1 F1 (macro)"]
    print(compare.round(4).to_string())

    print("\n--- Bucket 1+2 feature importance (top 10), Random Forest ---")
    print(imp_2["Random Forest"].head(10).round(4).to_string())