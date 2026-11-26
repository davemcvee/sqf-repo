"""
Report 1 requirement: "Compare the reasons for an SQF and what type of
force was used by the officer."

10 stop reasons (cs_*) x 9 raw force flags (pf_*) would be a 90-cell grid --
too much to read at once. Collapsed the force side into 4 meaningful tiers
(an escalation hierarchy, not an arbitrary grouping):
  Hands            -> pf_hands
  Physical restraint -> pf_wall, pf_grnd, pf_hcuff
  Weapon-level     -> pf_drwep, pf_ptwep, pf_baton, pf_pepsp
  Other            -> pf_other
Each cell = rate of that force tier, among stops where that reason was cited.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sqf_data import load_codebook, label_for
from viz_style import COLOR, SEQUENTIAL_CMAP

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

REASON_COLS = ["cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
               "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]

FORCE_TIERS = {
    "Hands": ["pf_hands"],
    "Physical restraint": ["pf_wall", "pf_grnd", "pf_hcuff"],
    "Weapon-level": ["pf_drwep", "pf_ptwep", "pf_baton", "pf_pepsp"],
    "Other": ["pf_other"],
}


def fig13_reason_vs_force(df, spec):
    reason_labels = [label_for(spec, c).replace("REASON FOR STOP - ", "").title() for c in REASON_COLS]

    rows = []
    for reason_col in REASON_COLS:
        cited = df[reason_col] == "Y"
        row = {}
        for tier_name, cols in FORCE_TIERS.items():
            row[tier_name] = (df.loc[cited, cols] == "Y").any(axis=1).mean()
        rows.append(row)
    table = pd.DataFrame(rows, index=reason_labels)
    table = table.sort_values("Hands", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    im = ax.imshow(table.values, cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Type of force used, by reason cited for the stop", loc="left", fontsize=12, pad=12)
    vmax = table.values.max()
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            text_color = "white" if val > vmax * 0.6 else COLOR["ink"]
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=9, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.text(0.02, -0.02, "each cell: rate of that force tier among stops where the row's reason was cited",
              fontsize=9, color=COLOR["ink_muted"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig13_reason_vs_force.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    spec, labels = load_codebook()
    df = pd.read_parquet(IN_PATH)

    table = fig13_reason_vs_force(df, spec)
    print("fig13_reason_vs_force.png saved\n")
    print(table.round(3).to_string())

    print("\n--- baseline (all stops, any reason) ---")
    row = {}
    for tier_name, cols in FORCE_TIERS.items():
        row[tier_name] = (df[cols] == "Y").any(axis=1).mean()
    print(pd.Series(row).round(3).to_string())