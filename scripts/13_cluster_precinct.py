"""
Report 2, Modelling task C (open-ended): a third use of cluster analysis.

Changes altitude rather than re-slicing individual stops a third way:
aggregates each of the 76 precincts into a profile (outcome rates + race
composition + stop volume) and clusters the precincts themselves, asking
whether there are distinct "types" of precincts in how stops are conducted --
extends the spatial thread from fig9/fig15/fig16 to a new level.

Features are standardized (z-score) before clustering since rates (0-1) and
raw stop volume are on very different scales.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from viz_style import COLOR, SEQUENTIAL_CMAP

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

PF_COLS = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep", "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other"]
WEAPON_COLS = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]
RACE_MAP = {"B": "Black", "Q": "White-Hisp.", "W": "White", "P": "Black-Hisp."}


def build_precinct_profiles(df):
    any_force = (df[PF_COLS] == "Y").any(axis=1)
    any_weapon = (df[WEAPON_COLS] == "Y").any(axis=1)
    race_simplified = df["race"].map(RACE_MAP).fillna("Other")

    g = df.groupby("pct")
    profile = pd.DataFrame({
        "n_stops": g.size(),
        "frisk_rate": (df["frisked"] == "Y").groupby(df["pct"]).mean(),
        "search_rate": (df["searched"] == "Y").groupby(df["pct"]).mean(),
        "arrest_rate": (df["arstmade"] == "Y").groupby(df["pct"]).mean(),
        "force_rate": any_force.groupby(df["pct"]).mean(),
        "weapon_rate": any_weapon.groupby(df["pct"]).mean(),
        "x": df["xcoord"].groupby(df["pct"]).mean(),
        "y": df["ycoord"].groupby(df["pct"]).mean(),
    })
    race_pct = pd.crosstab(df["pct"], race_simplified, normalize="index")
    profile = profile.join(race_pct)
    return profile


FEATURE_COLS = ["n_stops", "frisk_rate", "search_rate", "arrest_rate", "force_rate",
                "weapon_rate", "Black", "White-Hisp.", "White", "Black-Hisp."]


def choose_k(Xs, k_range=range(2, 11)):
    inertias = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=44, n_init=10).fit(Xs)
        inertias.append(km.inertia_)
    k_list = list(k_range)
    x_norm = (np.array(k_list) - min(k_list)) / (max(k_list) - min(k_list))
    y = np.array(inertias)
    y_norm = (y - y.min()) / (y.max() - y.min())
    line_dist = np.abs(y_norm - (1 - x_norm)) / np.sqrt(2)
    knee_k = k_list[int(np.argmax(line_dist))]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(k_list, inertias, marker="o", color=COLOR["blue"], linewidth=2)
    ax.axvline(knee_k, color=COLOR["red"], linewidth=1.5, linestyle=(0, (1, 1)))
    ax.set_title(f"Elbow method (precinct profiles) -- knee at k={knee_k}", loc="left", fontsize=11, pad=10)
    ax.set_xlabel("k")
    ax.set_ylabel("Inertia")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(color=COLOR["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig20_precinct_kselect.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return knee_k, dict(zip(k_range, inertias))


def characterize_and_plot(profile, k):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(profile[FEATURE_COLS])
    km = KMeans(n_clusters=k, random_state=44, n_init=10).fit(Xs)
    profile = profile.copy()
    profile["cluster"] = km.labels_

    # order clusters by frisk_rate for a readable, non-arbitrary label order
    order = profile.groupby("cluster")["frisk_rate"].mean().sort_values().index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    profile["cluster"] = profile["cluster"].map(remap)

    display_cols = {"n_stops": "Stops (n)", "frisk_rate": "Frisk rate", "search_rate": "Search rate",
                     "arrest_rate": "Arrest rate", "force_rate": "Force rate", "weapon_rate": "Weapon rate",
                     "Black": "% Black", "White-Hisp.": "% White-Hisp.", "White": "% White", "Black-Hisp.": "% Black-Hisp."}
    cluster_means = profile.groupby("cluster")[FEATURE_COLS].mean().rename(columns=display_cols)
    cluster_means.index = [f"Cluster {i+1} (n={ (profile['cluster']==i).sum()} precincts)" for i in cluster_means.index]

    # --- centroid table (real units, not standardized, for readability) ---
    fig, ax = plt.subplots(figsize=(11, 0.6 * k + 2))
    display_tbl = cluster_means.copy()
    rate_cols = [c for c in display_tbl.columns if "rate" in c or "%" in c]
    fmt_tbl = display_tbl.copy()
    for c in rate_cols:
        fmt_tbl[c] = display_tbl[c]
    norm_tbl = display_tbl.copy()
    for c in display_tbl.columns:
        norm_tbl[c] = (display_tbl[c] - display_tbl[c].min()) / (display_tbl[c].max() - display_tbl[c].min() + 1e-9)
    im = ax.imshow(norm_tbl.values, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(display_tbl.columns)))
    ax.set_xticklabels(display_tbl.columns, rotation=30, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(cluster_means.index)
    ax.set_title("Precinct cluster profiles (color = relative rank per column; labels = actual values)",
                 loc="left", fontsize=11.5, pad=12)
    for i in range(k):
        for j, c in enumerate(display_tbl.columns):
            val = display_tbl.iloc[i, j]
            label = f"{val:,.0f}" if c == "Stops (n)" else f"{val:.1%}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8.5,
                     color="white" if norm_tbl.iloc[i, j] > 0.6 else COLOR["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig21_precinct_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- geographic view: precinct centroid, colored by cluster ---
    fig, ax = plt.subplots(figsize=(7, 8))
    for i in range(k):
        sub = profile[profile["cluster"] == i]
        ax.scatter(sub["x"], sub["y"], s=sub["n_stops"] / 40, color=CAT_COLORS[i % len(CAT_COLORS)],
                   label=f"Cluster {i+1}", alpha=0.85, edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_aspect("equal")
    ax.set_title("Precincts by cluster (marker size = stop volume)", loc="left", fontsize=12, pad=12)
    ax.set_xlabel("X coordinate (NY State Plane, ft)")
    ax.set_ylabel("Y coordinate (NY State Plane, ft)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", frameon=False, fontsize=9, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig22_precinct_map.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return profile, cluster_means


if __name__ == "__main__":
    df = pd.read_parquet(IN_PATH)
    profile = build_precinct_profiles(df)
    print(f"{len(profile)} precincts, stop counts range {profile['n_stops'].min()}-{profile['n_stops'].max()}")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(profile[FEATURE_COLS])
    best_k, inertias = choose_k(Xs)
    print("inertias:", inertias)
    print("chosen k (inertia knee):", best_k)

    profile, cluster_means = characterize_and_plot(profile, best_k)
    print("\n--- cluster means ---")
    print(cluster_means.round(3).to_string())
    print("\n--- precinct membership ---")
    for c in sorted(profile["cluster"].unique()):
        members = sorted(profile[profile["cluster"] == c].index.tolist())
        print(f"Cluster {c+1}: {members}")