"""
Data Understanding: visualize >= 5 attributes (Report 1 requirement).

6 figures, following the dataviz skill's form/color rules:
  1. Demographics -- race (bar) + age (histogram) + sex (stacked bar)
  2. Reasons for stop (cs_* rates)
  3. Outcome sequence: frisked -> searched -> arrest/summons/contraband
  4. Type of force used (pf_* rates)
  5. Weapons found on suspect
  6. Officer uniform / ID-provided compliance (meters)

Palette + form choices from the dataviz skill: sequential blue for single-
measure magnitude comparisons, categorical slots (blue/orange/aqua) only
where categories are genuinely different series (sex), ordinal blue steps
for the procedurally-ordered outcome chart. All PNGs saved at 300 DPI for
embedding directly into the Word report.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sqf_data import load_codebook, label_for
from viz_style import COLOR, SEQ_ORDINAL, style_axes, hbar_rate_chart

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"


def fig1_demographics(df, spec):
    fig, axes = plt.subplots(1, 3, figsize=(13, 5), gridspec_kw={"width_ratios": [2.2, 2.2, 1]})

    race_labels_raw, _ = load_codebook()
    race_map = {"B": "Black", "Q": "White-Hispanic", "W": "White", "P": "Black-Hispanic",
                "A": "Asian/Pacific Isl.", "Z": "Other", "U": "Unknown", "I": "Amer. Indian/AK Native"}
    race_counts = df["race"].map(race_map).value_counts()
    race_rates = (race_counts / len(df)).values
    hbar_rate_chart(axes[0], race_counts.index, race_rates, "(a) Race of person stopped")

    ax = axes[1]
    ax.hist(df["age"].dropna(), bins=range(10, 105, 5), color=COLOR["blue"], zorder=3)
    style_axes(ax, x_is_pct=False)
    ax.set_title("(b) Age of person stopped", loc="left", fontsize=12, color=COLOR["ink"], pad=12)
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Number of stops")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    ax = axes[2]
    sex_map = {"M": "Male", "F": "Female", "Z": "Unknown"}
    sex_counts = df["sex"].map(sex_map).value_counts()
    sex_rates = sex_counts / len(df)
    sex_colors = [COLOR["blue"], COLOR["orange"], COLOR["aqua"]]
    left = 0
    for (cat, rate), color in zip(sex_rates.items(), sex_colors):
        ax.barh([0], [rate], left=left, color=color, height=0.5, zorder=3,
                label=f"{cat} ({rate:.1%})")
        left += rate
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.2, 1.2)
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("(c) Sex", loc="left", fontsize=12, color=COLOR["ink"], pad=12)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig1_demographics.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2_reasons_for_stop(df, spec):
    cs_cols = ["cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
               "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]
    labels = [label_for(spec, c).replace("REASON FOR STOP - ", "").title() for c in cs_cols]
    rates = np.array([(df[c] == "Y").mean() for c in cs_cols])

    fig, ax = plt.subplots(figsize=(9, 6.3))
    title = ("Reasons cited for the stop (share of all 532,911 stops)\n"
             "officers may cite more than one reason per stop -- bars do not sum to 100%")
    hbar_rate_chart(ax, labels, rates, title)
    ax.title.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2_reasons_for_stop.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig2b_additional_circumstances(df, spec):
    ac_cols = ["ac_rept", "ac_inves", "ac_proxm", "ac_evasv", "ac_assoc",
               "ac_cgdir", "ac_incid", "ac_time", "ac_stsnd", "ac_other"]
    labels = [label_for(spec, c).replace("ADDITIONAL CIRCUMSTANCES - ", "").title() for c in ac_cols]
    rates = np.array([(df[c] == "Y").mean() for c in ac_cols])

    fig, ax = plt.subplots(figsize=(9, 6.3))
    title = ("Additional circumstances cited (share of all 532,911 stops)\n"
             "officers may cite more than one circumstance per stop -- bars do not sum to 100%")
    hbar_rate_chart(ax, labels, rates, title)
    ax.title.set_fontsize(12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig2b_additional_circumstances.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig3_outcome_sequence(df):
    pf_cols = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep",
               "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other"]
    any_force = (df[pf_cols] == "Y").any(axis=1)
    weapon_cols = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]
    any_weapon = (df[weapon_cols] == "Y").any(axis=1)

    stages = ["Stopped", "Frisked", "Force used (any type)", "Searched",
              "Weapon found", "Contraband found", "Arrest made", "Summons issued"]
    rates = np.array([
        1.0,
        (df["frisked"] == "Y").mean(),
        any_force.mean(),
        (df["searched"] == "Y").mean(),
        any_weapon.mean(),
        (df["contrabn"] == "Y").mean(),
        (df["arstmade"] == "Y").mean(),
        (df["sumissue"] == "Y").mean(),
    ])
    colors = [SEQ_ORDINAL[min(i, len(SEQ_ORDINAL) - 1)] for i in range(len(stages))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(stages[::-1], rates[::-1], color=colors[::-1], height=0.6, zorder=3)
    style_axes(ax)
    ax.set_title("What happens after a stop (procedural order, not strictly nested)",
                 loc="left", fontsize=12, color=COLOR["ink"], pad=12)
    for bar, rate in zip(bars, rates[::-1]):
        ax.text(bar.get_width() + 0.015, bar.get_y() + bar.get_height() / 2,
                 f"{rate:.1%}", va="center", ha="left", fontsize=9, color=COLOR["ink_secondary"])
    ax.set_xlim(0, 1.15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig3_outcome_sequence.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig4_force_type(df, spec):
    pf_cols = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep",
               "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other"]
    labels = [label_for(spec, c).replace("PHYSICAL FORCE USED BY OFFICER - ", "").title() for c in pf_cols]
    rates = np.array([(df[c] == "Y").mean() for c in pf_cols])
    ns = np.array([(df[c] == "Y").sum() for c in pf_cols])

    fig, ax = plt.subplots(figsize=(9, 5.5))
    hbar_rate_chart(ax, labels, rates, "Type of physical force used (share of all 532,911 stops)", n=ns)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig4_force_type.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig5_weapons_found(df, spec):
    weapon_cols = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]
    label_overrides = {"othrweap": "Other Weapon Type"}
    labels = [label_overrides.get(c) or label_for(spec, c).replace("WAS A ", "").replace("WAS AN ", "")
              .replace(" FOUND ON SUSPECT ?", "").replace(" FOUND ON SUSPECT", "").title() for c in weapon_cols]
    rates = np.array([(df[c] == "Y").mean() for c in weapon_cols])
    ns = np.array([(df[c] == "Y").sum() for c in weapon_cols])

    fig, ax = plt.subplots(figsize=(9, 5))
    hbar_rate_chart(ax, labels, rates, "Weapons found on suspect (share of all 532,911 stops)", n=ns)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig5_weapons_found.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig6_officer_compliance(df):
    uniform_rate = (df["offunif"] == "Y").mean()
    id_rate = (df.loc[df["offunif"] == "N", "officer_id_provided"] == 1).mean()

    fig, axes = plt.subplots(2, 1, figsize=(7, 3.2))
    meters = [
        (axes[0], uniform_rate, "Stops where the officer was in uniform"),
        (axes[1], id_rate, "Plainclothes officers who identified themselves (badge/verbal/ID)"),
    ]
    for ax, rate, title in meters:
        ax.barh([0], [1], color=COLOR["blue_light"], height=0.5, zorder=2)
        ax.barh([0], [rate], color=COLOR["blue"], height=0.5, zorder=3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-1.2, 1.2)
        ax.set_yticks([])
        ax.set_xticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.text(1.02, 0, f"{rate:.1%}", va="center", ha="left", fontsize=14,
                 color=COLOR["ink"], fontweight="bold")
        ax.set_title(title, loc="left", fontsize=10, color=COLOR["ink_secondary"], pad=6)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig6_officer_compliance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    spec, labels = load_codebook()
    df = pd.read_parquet(IN_PATH)

    fig1_demographics(df, spec)
    print("fig1_demographics.png saved")
    fig2_reasons_for_stop(df, spec)
    print("fig2_reasons_for_stop.png saved")
    fig2b_additional_circumstances(df, spec)
    print("fig2b_additional_circumstances.png saved")
    fig3_outcome_sequence(df)
    print("fig3_outcome_sequence.png saved")
    fig4_force_type(df, spec)
    print("fig4_force_type.png saved")
    fig5_weapons_found(df, spec)
    print("fig5_weapons_found.png saved")
    fig6_officer_compliance(df)
    print("fig6_officer_compliance.png saved")
    print(f"\nAll figures saved to {OUT_DIR}")