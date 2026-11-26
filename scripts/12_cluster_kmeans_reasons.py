"""
Report 2, Modelling task B (mandatory): "Cluster stopped people by
reasons for stop."

Feature set: the 10 cs_* binary reason-for-stop flags (kept as raw 0/1, not
standardized -- standardizing would distort the "which factors co-occur"
interpretation that's the actual point of this clustering).

Clustering the reasons alone only produces groups of stops that cite similar
combinations of factors together -- it doesn't answer anything by itself.
The actual question is answered in two follow-up steps: characterize each
cluster (a) by real outcomes (does a "weapon-suspected" profile actually
find more weapons than a "vague/generic" profile?) and (b) by demographics
(does one profile skew toward a particular race/age/sex group?). Both are
computed and visualized here alongside the clustering itself.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sqf_data import load_codebook, label_for
from viz_style import COLOR, SEQUENTIAL_CMAP, style_axes_y

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

CS_COLS = ["cs_objcs", "cs_descr", "cs_casng", "cs_lkout", "cs_cloth",
           "cs_drgtr", "cs_furtv", "cs_vcrim", "cs_bulge", "cs_other"]
PF_COLS = ["pf_hands", "pf_wall", "pf_grnd", "pf_drwep", "pf_ptwep", "pf_baton", "pf_hcuff", "pf_pepsp", "pf_other"]
WEAPON_COLS = ["pistol", "riflshot", "asltweap", "knifcuti", "machgun", "othrweap"]


def choose_k(X, k_range=range(2, 19)):
    """Elbow (inertia) + silhouette, both plotted -- but k is chosen from a
    geometric knee search on inertia, not from the silhouette peak.

    Silhouette was tried first and rejected as the decision rule: on this
    sparse, multi-select binary feature set it increases monotonically
    across the entire tested range (2 through 18) with no plateau, which
    means "pick the max" would just keep fragmenting the data into smaller,
    less meaningful slices rather than finding a real structural elbow --
    silhouette is known to reward exactly this kind of over-fragmentation
    on non-continuous data. Inertia's knee is found the same way as eps was
    chosen for DBSCAN (Section: crime-location clustering) -- max distance
    from the chord connecting the first and last point of the curve -- for
    a reproducible choice rather than an eyeballed one."""
    inertias, sils = [], []
    # silhouette on a subsample -- exact silhouette on 532,911 rows x up to
    # 18 k's is expensive; a consistent random subsample is standard practice
    rng = np.random.RandomState(44)
    sample_idx = rng.choice(len(X), size=20000, replace=False)

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=44, n_init=10).fit(X)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(X[sample_idx], km.labels_[sample_idx]))

    k_list = list(k_range)
    x_norm = (np.array(k_list) - min(k_list)) / (max(k_list) - min(k_list))
    y = np.array(inertias)
    y_norm = (y - y.min()) / (y.max() - y.min())
    # chord runs from (0,1) to (1,0) since inertia decreases with k
    line_dist = np.abs(y_norm - (1 - x_norm)) / np.sqrt(2)
    knee_k = k_list[int(np.argmax(line_dist))]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(k_list, inertias, marker="o", color=COLOR["blue"], linewidth=2)
    axes[0].axvline(knee_k, color=COLOR["red"], linewidth=1.5, linestyle=(0, (1, 1)))
    axes[0].set_title(f"Elbow method (inertia) -- knee at k={knee_k}", loc="left", fontsize=11, pad=10)
    axes[0].set_xlabel("k")
    axes[1].plot(k_list, sils, marker="o", color=COLOR["orange"], linewidth=2)
    axes[1].set_title("Silhouette -- rises monotonically, not used to pick k", loc="left", fontsize=10.5, pad=10)
    axes[1].set_xlabel("k")
    for ax in axes:
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        ax.grid(color=COLOR["grid"], linewidth=1, zorder=0)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig17_kmeans_reasons_kselect.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return knee_k, dict(zip(k_range, inertias)), dict(zip(k_range, sils))


def fit_and_characterize(df, spec, k):
    X = df[CS_COLS].apply(lambda s: (s == "Y").astype(int)).values
    km = KMeans(n_clusters=k, random_state=44, n_init=10).fit(X)
    df = df.copy()
    df["cluster"] = km.labels_

    labels = [label_for(spec, c).replace("REASON FOR STOP - ", "").title() for c in CS_COLS]
    centroids = pd.DataFrame(km.cluster_centers_, columns=labels)
    centroids.index = [f"Cluster {i+1} (n={ (df['cluster']==i).sum():,})" for i in range(k)]

    # --- centroid heatmap: which reasons define each cluster ---
    fig, ax = plt.subplots(figsize=(10, 0.6 * k + 2))
    im = ax.imshow(centroids.values, cmap=SEQUENTIAL_CMAP, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(k))
    ax.set_yticklabels(centroids.index)
    ax.set_title("Cluster profiles: average value of each reason (0-1) per cluster", loc="left", fontsize=12, pad=12)
    for i in range(k):
        for j in range(len(labels)):
            val = centroids.values[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8,
                     color="white" if val > 0.55 else COLOR["ink"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig18_kmeans_reasons_profiles.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- outcomes by cluster ---
    any_force = (df[PF_COLS] == "Y").any(axis=1)
    any_weapon = (df[WEAPON_COLS] == "Y").any(axis=1)
    outcome_tbl = pd.DataFrame({
        "Frisked": (df["frisked"] == "Y").groupby(df["cluster"]).mean(),
        "Searched": (df["searched"] == "Y").groupby(df["cluster"]).mean(),
        "Arrested": (df["arstmade"] == "Y").groupby(df["cluster"]).mean(),
        "Force used": any_force.groupby(df["cluster"]).mean(),
        "Weapon found": any_weapon.groupby(df["cluster"]).mean(),
    })
    outcome_tbl.index = centroids.index

    # --- demographics by cluster ---
    race_map = {"B": "Black", "Q": "White-Hisp.", "W": "White", "P": "Black-Hisp.",
                "A": "Asian/PI", "Z": "Other", "U": "Unknown", "I": "Amer. Indian"}
    race_tbl = pd.crosstab(df["cluster"], df["race"].map(race_map), normalize="index")
    race_tbl.index = centroids.index
    race_tbl = race_tbl[sorted(race_tbl.columns, key=lambda c: -race_tbl[c].mean())]

    fig, axes = plt.subplots(1, 2, figsize=(14, 0.7 * k + 2))
    im0 = axes[0].imshow(outcome_tbl.values, cmap=SEQUENTIAL_CMAP, aspect="auto")
    axes[0].set_xticks(range(len(outcome_tbl.columns)))
    axes[0].set_xticklabels(outcome_tbl.columns, rotation=30, ha="right")
    axes[0].set_yticks(range(k))
    axes[0].set_yticklabels(outcome_tbl.index)
    axes[0].set_title("Outcome rate by cluster", loc="left", fontsize=11, pad=10)
    vmax0 = outcome_tbl.values.max()
    for i in range(k):
        for j in range(len(outcome_tbl.columns)):
            val = outcome_tbl.values[i, j]
            axes[0].text(j, i, f"{val:.1%}", ha="center", va="center", fontsize=8,
                          color="white" if val > vmax0 * 0.6 else COLOR["ink"])

    im1 = axes[1].imshow(race_tbl.values, cmap=SEQUENTIAL_CMAP, aspect="auto")
    axes[1].set_xticks(range(len(race_tbl.columns)))
    axes[1].set_xticklabels(race_tbl.columns, rotation=30, ha="right")
    axes[1].set_yticks(range(k))
    axes[1].set_yticklabels([""] * k)
    axes[1].set_title("Race composition by cluster", loc="left", fontsize=11, pad=10)
    for i in range(k):
        for j in range(len(race_tbl.columns)):
            val = race_tbl.values[i, j]
            axes[1].text(j, i, f"{val:.0%}", ha="center", va="center", fontsize=8,
                          color="white" if val > 0.4 else COLOR["ink"])
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig19_kmeans_reasons_outcomes_demo.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return centroids, outcome_tbl, race_tbl, km


if __name__ == "__main__":
    spec, labels_cb = load_codebook()
    df = pd.read_parquet(IN_PATH)
    X = df[CS_COLS].apply(lambda s: (s == "Y").astype(int)).values

    best_k, inertias, sils = choose_k(X)
    print("inertias:", inertias)
    print("silhouettes:", sils)
    print("chosen k (inertia knee, not silhouette -- see choose_k docstring):", best_k)

    centroids, outcome_tbl, race_tbl, km = fit_and_characterize(df, spec, best_k)
    print("\n--- cluster profiles ---")
    print(centroids.round(2).to_string())
    print("\n--- outcomes by cluster ---")
    print(outcome_tbl.round(3).to_string())
    print("\n--- race composition by cluster ---")
    print(race_tbl.round(3).to_string())