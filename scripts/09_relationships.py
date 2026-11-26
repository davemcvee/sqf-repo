"""
Data Understanding: explore relationships between attributes (Report 1 requirement).

5 relationship explorations, using scatter/density, correlation, cross-tab,
and group-wise-average forms as appropriate to each attribute pair -- not
forcing a scatter plot onto data that's mostly categorical:
  1. Correlation heatmap: age, height, weight, perobs, perstop
  2. Spatial density: xcoord/ycoord (hexbin, not a raw 532k-point scatter)
  3. Race x outcome-rate heatmap: frisked/searched/arrested/force-used
  4. Suspicion factor -> weapon-found lift (dumbbell): cited vs not-cited
  5. Age group x outcome rate (multi-line)
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sqf_data import load_codebook, label_for
from viz_style import COLOR, style_axes, style_axes_y, SEQUENTIAL_CMAP, DIVERGING_CMAP

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

PF_COLS = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep", "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other"]
WEAPON_COLS = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]


def fig8_correlation_heatmap(df):
    """Numeric attributes x real outcomes (point-biserial correlation, i.e.
    ordinary Pearson correlation with each outcome coded 0/1 -- statistically
    valid, not a hack). Chosen over a numeric-x-numeric-only matrix because
    that version showed nothing but the trivial height/weight relationship;
    this version directly tests whether background/circumstance attributes
    predict what happens to someone, which is the question the rest of the
    report is actually asking."""
    any_force = (df[PF_COLS] == "Y").any(axis=1)
    any_weapon = (df[WEAPON_COLS] == "Y").any(axis=1)

    d = pd.DataFrame({
        "Age": df["age"],
        "Height (in)": df["ht_feet"] * 12 + df["ht_inch"],
        "Weight (lb)": df["weight"],
        "Observation time (min)": df["perobs"],
        "Stop duration (min)": df["perstop"],
        "Frisked": (df["frisked"] == "Y").astype(int),
        "Searched": (df["searched"] == "Y").astype(int),
        "Arrested": (df["arstmade"] == "Y").astype(int),
        "Force used": any_force.astype(int),
        "Weapon found": any_weapon.astype(int),
    })
    rows = ["Age", "Height (in)", "Weight (lb)", "Observation time (min)", "Stop duration (min)"]
    cols = ["Frisked", "Searched", "Arrested", "Force used", "Weapon found"]
    corr = d.corr().loc[rows, cols]

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    im = ax.imshow(corr, cmap=DIVERGING_CMAP, vmin=-0.3, vmax=0.3)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows)
    ax.set_title("Correlation: numeric attributes vs. outcomes", loc="left", fontsize=12, pad=12)
    for i in range(len(rows)):
        for j in range(len(cols)):
            val = corr.iloc[i, j]
            text_color = "white" if abs(val) > 0.18 else COLOR["ink"]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.outline.set_visible(False)
    fig.text(0.02, -0.02, "note: scale is -0.3 to 0.3, not -1 to 1 -- no pair here is strongly "
              "correlated, but the relative differences are the point", fontsize=8.5, color=COLOR["ink_muted"])
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig8_correlation_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return corr


def fig14_hour_of_day(df):
    hour = (df["timestop"] // 100).astype(int)
    any_force = (df[PF_COLS] == "Y").any(axis=1)
    d = df.assign(hour=hour, any_force=any_force)

    outcomes = {
        "Frisked": (d["frisked"] == "Y"), "Searched": (d["searched"] == "Y"),
        "Arrested": (d["arstmade"] == "Y"), "Force used": d["any_force"],
    }
    table = pd.DataFrame({name: mask.groupby(d["hour"]).mean() for name, mask in outcomes.items()})

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    colors = [COLOR["blue"], COLOR["orange"], COLOR["aqua"], COLOR["red"]]
    for (name, series), color in zip(table.items(), colors):
        ax.plot(series.index, series.values, marker="o", markersize=4, linewidth=2, color=color, label=name)
    style_axes_y(ax)
    ax.set_xlabel("Hour of day (24h clock, stop start time)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_title("Outcome rate by hour of day", loc="left", fontsize=12, pad=12)
    ax.legend(loc="upper center", frameon=False, ncol=4, bbox_to_anchor=(0.5, 1.16))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig14_hour_of_day.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


def fig9_spatial_density(df):
    d = df.dropna(subset=["xcoord", "ycoord"])
    fig, ax = plt.subplots(figsize=(7, 8))
    hb = ax.hexbin(d["xcoord"], d["ycoord"], gridsize=90, cmap=SEQUENTIAL_CMAP, mincnt=1)
    ax.set_aspect("equal")
    ax.set_title("Where stops happen (density of all 532,911 stops)", loc="left", fontsize=12, pad=12)
    ax.set_xlabel("X coordinate (NY State Plane, ft)")
    ax.set_ylabel("Y coordinate (NY State Plane, ft)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.setp(ax.get_xticklabels(), rotation=40, ha="right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.7)
    cbar.set_label("Stops per hex cell", color=COLOR["ink_secondary"])
    cbar.outline.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig9_spatial_density.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig10_race_outcomes(df):
    race_map = {"B": "Black", "Q": "White-Hispanic", "W": "White", "P": "Black-Hispanic",
                "A": "Asian/Pacific Isl.", "Z": "Other", "U": "Unknown", "I": "Amer. Indian/AK Native"}
    d = df.assign(
        race_label=df["race"].map(race_map),
        any_force=(df[PF_COLS] == "Y").any(axis=1),
    )
    outcomes = {
        "Frisked": d["frisked"] == "Y",
        "Searched": d["searched"] == "Y",
        "Arrested": d["arstmade"] == "Y",
        "Force used": d["any_force"],
    }
    table = pd.DataFrame({name: mask.groupby(d["race_label"]).mean() for name, mask in outcomes.items()})
    table = table.sort_values("Frisked", ascending=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(table.values, cmap=SEQUENTIAL_CMAP, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels(table.columns)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(table.index)
    ax.set_title("Outcome rates by race", loc="left", fontsize=12, pad=12)
    vmax = table.values.max()
    for i in range(len(table.index)):
        for j in range(len(table.columns)):
            val = table.values[i, j]
            text_color = "white" if val > vmax * 0.6 else COLOR["ink"]
            ax.text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=9, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig10_race_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


def _dumbbell(ax, table, title):
    y = np.arange(len(table))
    ax.hlines(y, table["not_cited"], table["cited"], color=COLOR["baseline"], linewidth=2, zorder=2)
    ax.scatter(table["not_cited"], y, color=COLOR["blue_light"], s=80, zorder=3,
               edgecolor=COLOR["blue"], linewidth=1, label="Not cited")
    ax.scatter(table["cited"], y, color=COLOR["blue"], s=80, zorder=3, label="Cited")
    ax.set_yticks(y)
    ax.set_yticklabels(table["factor"])
    style_axes(ax, decimals=1)
    ax.set_title(title, loc="left", fontsize=12, pad=12)


def fig11_suspicion_weapon_lift(df, spec):
    factor_labels = {
        "cs_bulge": "Suspicious bulge (at stop)",
        "rf_bulg": "Suspicious bulge (at frisk)",
        "rf_othsw": "Other weapon suspicion (at frisk)",
        "cs_furtv": "Furtive movements (at stop)",
        "rf_furt": "Furtive movements (at frisk)",
        "cs_vcrim": "Violent crime actions (at stop)",
        "rf_vcrim": "Violent crime suspected (at frisk)",
        "rf_vcact": "Violent crime actions (at frisk)",
        "cs_drgtr": "Drug transaction indicators (at stop)",
        "ac_incid": "High-crime area (context)",
    }
    any_weapon = (df[WEAPON_COLS] == "Y").any(axis=1)
    contrabn = df["contrabn"] == "Y"

    weapon_rows, contrabn_rows = [], []
    for f, label in factor_labels.items():
        cited = df[f] == "Y"
        weapon_rows.append({"factor": label, "not_cited": any_weapon[~cited].mean(), "cited": any_weapon[cited].mean()})
        contrabn_rows.append({"factor": label, "not_cited": contrabn[~cited].mean(), "cited": contrabn[cited].mean()})
    weapon_table = pd.DataFrame(weapon_rows).sort_values("cited")
    # keep the same row order as the weapon panel for direct visual comparison
    contrabn_table = pd.DataFrame(contrabn_rows).set_index("factor").loc[weapon_table["factor"]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)
    _dumbbell(axes[0], weapon_table, "Weapon found")
    _dumbbell(axes[1], contrabn_table, "Contraband found (any -- weapons + drugs + other)")
    axes[1].tick_params(labelleft=False)  # redundant with axes[0] under sharey
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=2, frameon=False)
    fig.suptitle("Rate of finding something: when a suspicion factor is cited vs. not", x=0.02, ha="left", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 0.94])
    fig.savefig(OUT_DIR / "fig11_suspicion_weapon_lift.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return weapon_table, contrabn_table


def fig12_age_outcomes(df):
    # 80s (n=82) and 90+ (n=448) collapsed into one "80+" bin -- kept separate
    # they're too small individually to trust (e.g. 0 arrests in 82 people
    # reads as a rate, but it's really just small-sample noise)
    bins = [10, 20, 30, 40, 50, 60, 70, 80, 101]
    bin_labels = ["10s", "20s", "30s", "40s", "50s", "60s", "70s", "80+"]
    d = df.assign(
        age_group=pd.cut(df["age"], bins=bins, labels=bin_labels, right=False),
        any_force=(df[PF_COLS] == "Y").any(axis=1),
    )
    outcomes = {
        "Frisked": (d["frisked"] == "Y"), "Searched": (d["searched"] == "Y"),
        "Arrested": (d["arstmade"] == "Y"), "Force used": d["any_force"],
    }
    table = pd.DataFrame({name: mask.groupby(d["age_group"], observed=True).mean() for name, mask in outcomes.items()})

    fig, ax = plt.subplots(figsize=(9, 5.5))
    colors = [COLOR["blue"], COLOR["orange"], COLOR["aqua"], COLOR["red"]]
    for (name, series), color in zip(table.items(), colors):
        ax.plot(series.index, series.values, marker="o", markersize=5, linewidth=2, color=color, label=name)
    style_axes_y(ax)
    ax.set_title("Outcome rate by age group", loc="left", fontsize=12, pad=12)
    ax.legend(loc="upper right", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig12_age_outcomes.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return table


if __name__ == "__main__":
    OUT_DIR.mkdir(exist_ok=True)
    spec, labels = load_codebook()
    df = pd.read_parquet(IN_PATH)

    corr_table = fig8_correlation_heatmap(df)
    print("fig8_correlation_heatmap.png saved")
    print(corr_table.round(3).to_string())

    hour_table = fig14_hour_of_day(df)
    print("\nfig14_hour_of_day.png saved")
    print(hour_table.round(3).to_string())

    fig9_spatial_density(df)
    print("fig9_spatial_density.png saved")

    race_table = fig10_race_outcomes(df)
    print("fig10_race_outcomes.png saved")
    print(race_table.round(3).to_string())

    weapon_table, contrabn_table = fig11_suspicion_weapon_lift(df, spec)
    print("\nfig11_suspicion_weapon_lift.png saved")
    print("-- weapon found --")
    print(weapon_table.round(4).to_string(index=False))
    print("-- contraband found (any) --")
    print(contrabn_table.round(4).to_string(index=False))

    age_table = fig12_age_outcomes(df)
    print("\nfig12_age_outcomes.png saved")
    print(age_table.round(3).to_string())

    print(f"\nAll figures saved to {OUT_DIR}")