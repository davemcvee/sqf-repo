"""
Data Preparation: Select Data + Clean Data + Construct Data.

All decisions confirmed with Dave (2026-08-12):
- Normalize blank-ish text (whitespace-only strings, literal "nan") to real NaN
  across all text columns first, so nothing downstream treats " " as a category.
- Drop the 6 columns that are 100% blank across all 532,911 stops.
- Keep arstoffn and sumoffen despite high blank rates (93.9% / 96.1%) -- blank
  there means "no arrest" / "no summons", i.e. a real outcome, not missing data.
  Same logic for forceuse (87.4% blank = no notable force used): blank is
  recoded to the explicit label "NONE" so it reads as a real category.
- officrid/offverb/offshld ("(IF NOT IN UNIFORM)" fields) are explained by the
  existing offunif column: 100% blank whenever offunif == 'Y'. Rather than
  deriving a redundant "presumed uniform" flag, derive officer_id_provided:
  among officers NOT in uniform, did they identify themselves any of the 3
  ways (badge / verbal / ID card)? NaN when offunif == 'Y' (question doesn't
  apply). Raw officrid/offverb/offshld columns are kept too, since they still
  distinguish *which* method was used.
- Three fields use a literal '*' as an undocumented keypunch placeholder
  (confirmed 2026-08-13, not an official NYPD code -- see scripts note below):
    * perstop ('**', 243 rows): numeric field, coerced to NaN -- unambiguous.
    * sumoffen ('*', 673 rows): verified these are NOT "no summons" (that's
      already handled by the blank->NONE fill above) -- every one of these
      673 rows has sumissue == 'Y', meaning a summons WAS issued but the
      offense text specifically wasn't captured. Real gap -> NaN, kept
      distinct from the "NONE" category.
    * beat ('*', 68,491 rows, 12.9%): NOT random -- 0% of rows in October,
      November, and December 2012, vs. 14-16.5% in every month before that.
      Clean process cutover, not scattered missingness. -> NaN, but this is
      worth calling out explicitly in the data quality write-up: beat is
      fully reliable for Q4 2012 stops and unreliable/absent before that.
- Sentinel numeric values:
    * age: 999 is an "unknown" placeholder -> NaN. Ages 0-9 (53/712/53/37/20/
      95/17/6/4/13 rows) have no coherent shape and are implausible for this
      program -> also NaN. Real distribution looks genuine from ~age 10-12,
      ramping smoothly from there (12: 616, 13: 2429, 14: 7682, ...).
      Also found (2026-08-13, while checking an odd bump in the age-vs-outcome
      chart): age 99 behaves as a second, softer sentinel -- ages 90-98 have
      only 8 people total, but age exactly 99 has 362, i.e. "very old, exact
      age not really known" rather than a real recorded age -> also NaN.
      Found a THIRD instance of the same pattern (2026-08-14, after the age-99
      fix still left a smaller bump at 80+): age 100 has 77 rows vs. only 9
      across ages 91-98 combined, and those 77 are 96.1% housing-authority
      stops (trhsloc='H') vs. 8.8% for the rest of the 80s/90s -- a specific
      context pattern, not a real age-100 population. Excluding it, the 80-98
      frisk rate (27.5%) fits the smooth decline from the 70s (25.4%)
      perfectly, confirming age 100 -- not genuine old age -- was producing
      the uptick. -> also NaN.
    * weight: 999 -> NaN (same convention). weight == 0 -> NaN (impossible).
      The 380-918 lb tail (129 rows) has no repeated value/pattern -- left as
      real data, not truncated. Low end: checked whether low weight tracks
      young age (it would, if these were real small people) -- it doesn't;
      the 40-60 lb band includes ages 21/29/44/60/70, so these are scattered
      entry errors, not real light people. Distribution only gets dense
      starting ~90-95 lbs (e.g. 148 rows at exactly 95). weight < 80 -> NaN.
    * dob: 12311900 (214,245 rows, ~40%) is a placeholder for "unknown DOB"
      -> NaN. Use `age` for analysis, not dob -- too contaminated to trust.
    * age, high end: 99/100 look like real (if soft-rounded) ages, but above
      100 it scatters thinly all the way to 851 -- physically impossible.
      age > 100 -> NaN.
    * xcoord/ycoord loaded as text but contain nothing but clean numbers
      (verified -- no stray characters in any non-null value) -- cast to
      numeric so they're usable for the Report 2 DBSCAN work.

Columns with high blank rates that are genuine data-quality gaps, not
structural blanks (post/beat/addrnum/stname), are left as real NaN after the
whitespace normalization -- no synthetic recoding, since a location field
being unrecorded isn't the same as "not applicable".
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sqf_data import load_codebook, load_sqf, label_for

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUT_PATH = OUT_DIR / "sqf_2012_clean.parquet"

DROP_COLS = ["zip", "premtype", "state", "rescode", "aptnum", "othfeatr"]
BLANK_MEANS_NONE_COLS = ["arstoffn", "sumoffen", "forceuse"]
ID_FIELDS = ["officrid", "offverb", "offshld"]


def normalize_blanks(df):
    obj_cols = df.select_dtypes(include=["object", "string"]).columns
    for col in obj_cols:
        stripped = df[col].astype(str).str.strip()
        is_blank = stripped.isin(["", "nan", "NaN"])
        df[col] = stripped.mask(is_blank, np.nan)
    return df


def clean(df, spec):
    df = normalize_blanks(df)

    print("--- dropping 100%-blank columns ---")
    for col in DROP_COLS:
        print(f"  {col:10s} : {label_for(spec, col)}")
    df = df.drop(columns=DROP_COLS)

    print("\n--- recoding blank -> 'NONE' (not-applicable, not missing) ---")
    for col in BLANK_MEANS_NONE_COLS:
        n_blank = df[col].isna().sum()
        df[col] = df[col].fillna("NONE")
        print(f"  {col:10s} ({label_for(spec, col)}): {n_blank:,} blanks -> 'NONE'")

    print("\n--- cleaning '*' keypunch placeholders ---")
    n_perstop_star = (df["perstop"] == "**").sum()
    df["perstop"] = pd.to_numeric(df["perstop"], errors="coerce")
    print(f"  perstop   : {n_perstop_star:,} '**' -> NaN, converted to numeric")

    n_sumoffen_star = (df["sumoffen"] == "*").sum()
    df.loc[df["sumoffen"] == "*", "sumoffen"] = np.nan
    print(f"  sumoffen  : {n_sumoffen_star:,} '*' -> NaN (summons issued, offense text not captured)")

    n_beat_star = (df["beat"] == "*").sum()
    df.loc[df["beat"] == "*", "beat"] = np.nan
    print(f"  beat      : {n_beat_star:,} '*' -> NaN (0% in Oct-Dec 2012, all pre-October)")

    print("\n--- cleaning sentinel numeric placeholders ---")
    n_age_999 = (df["age"] == 999).sum()
    n_age_low = df["age"].lt(10).sum()
    n_age_99 = (df["age"] == 99).sum()
    df.loc[(df["age"] == 999) | (df["age"] < 10) | (df["age"] == 99), "age"] = np.nan
    print(f"  age       : {n_age_999:,} at 999 (unknown) + {n_age_low:,} under 10 (implausible) "
          f"+ {n_age_99:,} at 99 (soft 'very old' sentinel) -> NaN")

    n_weight_999 = (df["weight"] == 999).sum()
    n_weight_low = df["weight"].lt(80).sum()
    df.loc[(df["weight"] == 999) | (df["weight"] < 80), "weight"] = np.nan
    print(f"  weight    : {n_weight_999:,} at 999 (unknown) + {n_weight_low:,} under 80 (implausible) -> NaN")

    n_dob_placeholder = (df["dob"] == 12311900).sum()
    df.loc[df["dob"] == 12311900, "dob"] = np.nan
    print(f"  dob       : {n_dob_placeholder:,} at 12311900 (unknown-DOB placeholder) -> NaN")

    n_age_high = df["age"].gt(100).sum()
    n_age_100 = (df["age"] == 100).sum()
    df.loc[df["age"] >= 100, "age"] = np.nan
    print(f"  age (high): {n_age_high:,} over 100 (impossible) + {n_age_100:,} at exactly 100 "
          f"(96.1% housing-related, not a real age cluster) -> NaN")

    for coord_col in ["xcoord", "ycoord"]:
        df[coord_col] = pd.to_numeric(df[coord_col], errors="coerce")
    print("  xcoord/ycoord cast to numeric")

    print("\n--- deriving officer_id_provided (offunif == 'N' rows only) ---")
    not_uniformed = df["offunif"] == "N"
    any_id_given = df[ID_FIELDS].notna().any(axis=1)
    df["officer_id_provided"] = np.where(not_uniformed, any_id_given, np.nan)
    n_applicable = not_uniformed.sum()
    n_provided = (df["officer_id_provided"] == True).sum()  # noqa: E712
    print(f"  applicable rows (offunif == 'N'): {n_applicable:,}")
    print(f"  provided some form of ID: {n_provided:,} ({n_provided / n_applicable:.1%})")

    return df


if __name__ == "__main__":
    spec, labels = load_codebook()
    df = load_sqf()
    print(f"Loaded {len(df):,} rows x {df.shape[1]} columns\n")

    df_clean = clean(df, spec)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_clean.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved cleaned dataset -> {OUT_PATH}  ({df_clean.shape[0]:,} rows x {df_clean.shape[1]} columns)")

    print("\n--- remaining blank rates, post-normalization (location fields) ---")
    for col in ["post", "beat", "addrnum", "stname"]:
        pct = df_clean[col].isna().mean() * 100
        print(f"  {col:10s} ({label_for(spec, col)}): {pct:.1f}% blank")