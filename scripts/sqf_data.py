"""
Shared loaders for the NYC SQF 2012 dataset, used across all report scripts.
"""
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
SPEC_PATH = DATA_DIR / "2012 SQF File Spec.xlsx"
CSV_PATH = DATA_DIR / "2012 SQF Data.csv"


def load_codebook():
    spec = pd.read_excel(SPEC_PATH, sheet_name="SQF2012 File Spec", skiprows=3)
    spec = spec.rename(columns={"Variable": "variable", "Label": "label", "Measurement Level": "level"})
    spec = spec.dropna(subset=["variable"])

    labels = pd.read_excel(SPEC_PATH, sheet_name="Value Labels", skiprows=4)
    labels = labels.rename(columns={"Field Name": "field", "Value": "value", "Label": "label"})
    labels["field"] = labels["field"].ffill()

    return spec, labels


def load_sqf():
    df = pd.read_csv(CSV_PATH, low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def label_for(spec, variable):
    row = spec.loc[spec["variable"] == variable]
    return row["label"].iloc[0] if len(row) else "(no label found)"