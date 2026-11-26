"""
Shared chart style, extracted from 08_visualizations.py so every later
report reuses the same palette/mark rules instead of redefining them.
Values come from the dataviz skill's reference palette (references/palette.md).
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

COLOR = {
    "surface": "#fcfcfb",
    "ink": "#0b0b0b",
    "ink_secondary": "#52514e",
    "ink_muted": "#898781",
    "grid": "#e1e0d9",
    "baseline": "#c3c2b7",
    "blue": "#2a78d6",
    "blue_light": "#cde2fb",
    "orange": "#eb6834",
    "aqua": "#1baf7a",
    "red": "#e34948",
    "diverging_mid": "#f0efec",
}
# ordinal sequential steps, light -> dark (100 through 700)
SEQ_ORDINAL = ["#86b6ef", "#5598e7", "#2a78d6", "#184f95"]
# full sequential ramp, light -> dark, for heatmaps/density (100 through 700)
SEQ_FULL = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
            "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]

SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_FULL)
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", [COLOR["blue"], COLOR["diverging_mid"], COLOR["red"]]
)


def apply_rcparams():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
        "text.color": COLOR["ink"],
        "axes.edgecolor": COLOR["baseline"],
        "axes.labelcolor": COLOR["ink_secondary"],
        "xtick.color": COLOR["ink_muted"],
        "ytick.color": COLOR["ink_muted"],
        "figure.facecolor": COLOR["surface"],
        "axes.facecolor": COLOR["surface"],
        "savefig.facecolor": COLOR["surface"],
        "font.size": 11,
    })


def style_axes(ax, x_is_pct=True, decimals=0):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["baseline"])
    ax.grid(axis="x", color=COLOR["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    if x_is_pct:
        ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=decimals))


def style_axes_y(ax, decimals=0):
    """Like style_axes but for charts where the rate/percentage is on the
    y-axis (e.g. a line chart with a categorical x-axis)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_color(COLOR["baseline"])
    ax.spines["left"].set_color(COLOR["baseline"])
    ax.grid(axis="y", color=COLOR["grid"], linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=decimals))


def hbar_rate_chart(ax, labels, rates, title, n=None):
    """Horizontal bar chart for a single-measure magnitude comparison."""
    order = np.argsort(rates)
    labels = np.array(labels)[order]
    rates = np.array(rates)[order]
    bars = ax.barh(labels, rates, color=COLOR["blue"], height=0.6, zorder=3)
    style_axes(ax)
    ax.set_title(title, loc="left", fontsize=12, color=COLOR["ink"], pad=12)
    max_rate = max(rates) if len(rates) else 0
    for i, (bar, rate) in enumerate(zip(bars, rates)):
        label = f"{rate:.1%}"
        if n is not None:
            label += f"  (n={n[order][i]:,})"
        ax.text(bar.get_width() + max_rate * 0.02, bar.get_y() + bar.get_height() / 2,
                 label, va="center", ha="left", fontsize=9, color=COLOR["ink_secondary"])
    ax.set_xlim(0, max_rate * 1.25)


apply_rcparams()