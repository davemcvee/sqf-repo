"""
Data Understanding: data dictionary for a curated selection of attributes.

Report 1 requires describing >= 20 attributes (meaning/type/scale). Rather
than dump all 106 remaining columns, this picks ~28 that span the categories
we'll actually use across all three reports: demographics, stop context,
suspicion/action fields, officer conduct, and location -- so this dictionary
does double duty as a reference for later reports too.
"""
import pandas as pd
from pathlib import Path
from sqf_data import load_codebook, label_for

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"

SELECTED = [
    # demographics
    "sex", "race", "age", "ht_feet", "ht_inch", "weight", "build",
    # stop context
    "pct", "datestop", "timestop", "city", "inout", "trhsloc", "perobs", "perstop",
    # suspicion / procedure
    "crimsusp", "explnstp", "othpers",
    # actions taken
    "frisked", "searched", "contrabn", "arstmade", "sumissue", "forceuse", "arstoffn", "sumoffen",
    # officer conduct
    "offunif", "officer_id_provided",
    # location (numeric)
    "xcoord", "ycoord",
    # reason for stop (cs_) -- checked BEFORE the stop
    "cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
    "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other",
    # reason for frisk (rf_) -- checked AFTER the stop, before frisking
    "rf_vcrim", "rf_othsw", "rf_attir", "rf_vcact", "rf_rfcmp",
    "rf_verbl", "rf_knowl", "rf_furt", "rf_bulg",
    # additional circumstances (ac_)
    "ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc",
    "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other",
    # basis of search (sb_) -- checked before searching, after frisk
    "sb_hdobj", "sb_outln", "sb_admis", "sb_other",
    # type of force actually used by officer (pf_)
    "pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep",
    "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other",
    # weapon actually found on suspect (the outcome to test suspicion factors against)
    "pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap",
]


def describe(df, spec, labels, col):
    is_derived = col not in spec["variable"].values
    label = "Derived in cleaning (see 02_clean_data.py)" if is_derived else label_for(spec, col)
    level = "-" if is_derived else spec.loc[spec["variable"] == col, "level"].iloc[0]

    s = df[col]
    dtype = str(s.dtype)
    pct_missing = s.isna().mean() * 100
    n_unique = s.nunique()

    if pd.api.types.is_numeric_dtype(s):
        value_info = f"range {s.min():.0f}-{s.max():.0f}"
    else:
        value_info = f"{n_unique} distinct values"

    coded_values = ""
    field_labels = labels.loc[labels["field"].str.lower() == col.lower()]
    if len(field_labels) and not is_derived:
        pairs = [f"{v}={l}" for v, l in zip(field_labels["value"], field_labels["label"])]
        coded_values = "; ".join(pairs[:8]) + (" ..." if len(pairs) > 8 else "")

    return {
        "variable": col,
        "label": label,
        "level": level,
        "dtype": dtype,
        "pct_missing": round(pct_missing, 1),
        "values": value_info,
        "coded_values": coded_values,
    }


if __name__ == "__main__":
    spec, labels = load_codebook()
    df = pd.read_parquet(IN_PATH)

    rows = [describe(df, spec, labels, col) for col in SELECTED]
    out = pd.DataFrame(rows)

    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print(out.drop(columns=["coded_values"]).to_string(index=False))

    print("\n--- coded value labels (for categorical fields) ---")
    for r in rows:
        if r["coded_values"]:
            print(f"{r['variable']:10s}: {r['coded_values']}")

    out_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "data_dictionary.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")