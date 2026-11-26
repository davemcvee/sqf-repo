"""
Report 2, Modelling task A  
Chose CPW (Criminal Possession of Weapon) -- ties into the
weapon-focused thread running through Report 1, and crimsusp (not arstoffn,
which is a downstream-outcome field only populated for the 6% of stops that
end in arrest) is the right column for "what crime was suspected".

Run DBSCAN separately for a night window (20:00-05:59) and a day window
(06:00-19:59), chosen from the U-shaped outcome-rate-by-hour pattern found
in Report 1 (fig14) -- night is the high-frisk/high-force period, day the
trough. eps is chosen from a k-distance plot (not guessed), min_samples=15.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from viz_style import COLOR, SEQ_ORDINAL

IN_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "sqf_2012_clean.parquet"
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"
MIN_SAMPLES = 15

# categorical palette (from the dataviz skill reference palette), fixed order
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


def load_cpw():
    df = pd.read_parquet(IN_PATH)
    cpw = df[(df["crimsusp"] == "CPW") & df["xcoord"].notna() & df["ycoord"].notna()].copy()
    hour = (cpw["timestop"] // 100).astype(int)
    cpw["period"] = np.where(hour.isin([20, 21, 22, 23, 0, 1, 2, 3, 4, 5]), "Night (20:00-05:59)", "Day (06:00-19:59)")
    return cpw


def choose_eps(coords, min_samples, label, percentile=90):
    """k-distance plot, eps chosen at a percentile of the distribution
    rather than a geometric knee search.

    A geometric single-chord knee search was tried first and failed: a
    meaningful fraction of points sit at literally identical coordinates
    (the same corner stopped repeatedly), and the k-distance curve turns out
    to have structure at two very different scales at once -- tight
    same-address clumps (a few feet) and genuine neighborhood-level density
    (hundreds to thousands of feet). A single chord-distance knee search
    latched onto the small-scale structure (eps ~4 ft), producing ~85%
    noise and hundreds of trivial micro-clusters. Since a clean single
    elbow isn't reliably identifiable here, eps is instead set at the 90th
    percentile of the k-distance distribution -- a standard, documented
    fallback for exactly this situation: it keeps roughly the densest 90%
    of points as core-eligible while still requiring real neighborhood-scale
    proximity, not just an identical address."""
    nbrs = NearestNeighbors(n_neighbors=min_samples).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    k_dist = np.sort(distances[:, -1])
    zero_run = int((k_dist == 0).sum())
    eps = float(np.percentile(k_dist, percentile))
    knee_idx_full = int(np.searchsorted(k_dist, eps))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(np.arange(len(k_dist)), k_dist, color=COLOR["blue"], linewidth=2)
    ax.axhline(eps, color=COLOR["red"], linewidth=1.5, linestyle=(0, (1, 1)))
    ax.scatter([knee_idx_full], [eps], color=COLOR["red"], zorder=5, s=50)
    ax.annotate(f"{percentile}th percentile: eps = {eps:.0f} ft\n({zero_run:,} points at exactly 0 ft)",
                (knee_idx_full, eps), textcoords="offset points",
                xytext=(-160, 15), color=COLOR["red"], fontsize=9)
    ax.set_xlabel(f"Points, sorted by distance to {min_samples}th nearest neighbor")
    ax.set_ylabel("Distance (ft)")
    ax.set_title(f"k-distance plot -- {label}", loc="left", fontsize=12, pad=12)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.grid(color=COLOR["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    fig.tight_layout()
    safe_label = label.split(" ")[0].lower()
    fig.savefig(OUT_DIR / f"fig15_kdist_{safe_label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return eps


def run_dbscan(cpw_period, label, eps):
    coords = cpw_period[["xcoord", "ycoord"]].values
    model = DBSCAN(eps=eps, min_samples=MIN_SAMPLES).fit(coords)
    labels = model.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_pct = (labels == -1).mean()
    sil = np.nan
    if n_clusters >= 2:
        mask = labels != -1
        if mask.sum() > 1:
            sil = silhouette_score(coords[mask], labels[mask])

    # plot: largest 8 clusters get distinct colors, rest folded to "Other", noise as light gray
    sizes = pd.Series(labels[labels != -1]).value_counts()
    top_clusters = sizes.head(len(CAT_COLORS)).index.tolist()

    fig, ax = plt.subplots(figsize=(7, 8))
    noise_mask = labels == -1
    ax.scatter(coords[noise_mask, 0], coords[noise_mask, 1], s=3, color="#d8d7d0", alpha=0.5, label="Noise", zorder=1)
    other_mask = (~noise_mask) & (~pd.Series(labels).isin(top_clusters).values)
    if other_mask.any():
        ax.scatter(coords[other_mask, 0], coords[other_mask, 1], s=4, color="#a8a79f", alpha=0.6, label="Other clusters", zorder=2)
    for i, cl in enumerate(top_clusters):
        m = labels == cl
        ax.scatter(coords[m, 0], coords[m, 1], s=5, color=CAT_COLORS[i], label=f"Cluster {i+1} (n={m.sum():,})", zorder=3)

    ax.set_aspect("equal")
    ax.set_title(f"CPW stop locations -- {label}\n{n_clusters} clusters, {noise_pct:.1%} noise, silhouette={sil:.2f}",
                 loc="left", fontsize=12, pad=12)
    ax.set_xlabel("X coordinate (NY State Plane, ft)")
    ax.set_ylabel("Y coordinate (NY State Plane, ft)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.legend(loc="upper left", fontsize=7, frameon=False, markerscale=3, ncol=1, bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    safe_label = label.split(" ")[0].lower()
    fig.savefig(OUT_DIR / f"fig16_dbscan_{safe_label}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"period": label, "eps": eps, "min_samples": MIN_SAMPLES, "n_points": len(labels),
            "n_clusters": n_clusters, "noise_pct": noise_pct, "silhouette": sil}


if __name__ == "__main__":
    cpw = load_cpw()
    results = []
    for period in ["Night (20:00-05:59)", "Day (06:00-19:59)"]:
        sub = cpw[cpw["period"] == period]
        coords = sub[["xcoord", "ycoord"]].values
        eps = choose_eps(coords, MIN_SAMPLES, period)
        print(f"{period}: n={len(sub):,}, chosen eps={eps:.0f} ft")
        res = run_dbscan(sub, period, eps)
        results.append(res)
        print(res)

    print()
    print(pd.DataFrame(results).to_string(index=False))