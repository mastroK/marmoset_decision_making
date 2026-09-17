"""Panel rendering for Supplementary Figure 1 (panels d-e: pre-block-change
accuracy, P(break) after win/loss and at block transitions).

Source: _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb.

- `plot_pre_block_change_accuracy` (panel d, STAT 1 / "Supp Fig 1F"): no
  dedicated publication-figure cell for this single scalar was found in the
  source notebook (its own STATS EXTRACTION section, lines 5272-5314, only
  recomputes the number -- it does not plot). Styled here to match the
  house convention used throughout that notebook for equivalent
  bar+SEM+per-animal-scatter summaries (e.g. its
  `Asymptotic_Performance_Summary.svg` cell, ~lines 1036-1085, and the
  per-animal break-analysis cell below): FONTSIZE/LINE_WIDTH/TICK_LENGTH,
  dotted y-grid, spines removed, jittered per-animal points over a mean+SEM
  bar, gray dashed chance line at 0.5.

- `plot_p_break_comparison` (panel e, STAT 3 & 4 / "Supp Fig 1G"): matches
  the "Break Analysis - Per Animal with Error Bars" cell, specifically its
  Panel A/B (lines 5121-5232 in the nbconvert'd script) -- the version with
  per-animal paired Wilcoxon tests and per-animal scatter points, NOT the
  earlier population-only 3-panel version (`Break_Analysis_2ABT.svg`,
  lines 4993-5070) which has no per-animal breakdown to match our stats.
  The combined heatmap third panel from that source cell is out of scope
  (not one of Supp Fig 1's panels d/e) and is not reproduced.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes
from ..features.history_encoding import sort_animals


def plot_engaged_across_transitions(per_animal_summary, overall_summary, config, out_path):
    """Panel c: p(high-prob choice) at each trial position aligned to a
    block transition, for each individual marmoset (colored lines) and the
    population average (black line +/- SEM shading). Styled the same as
    Fig 1 panel b's own within-block learning curve
    (src/plotting/fig1.py::plot_within_block_learning) -- individual
    animals faint/colored, group mean bold black with SEM shading -- since
    this is the same "individual + average" style applied to a
    transition-aligned x-axis instead of a within-block one.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    animals = sort_animals(per_animal_summary["animal"].unique(), animal_order)

    fig, ax = plt.subplots(figsize=(4, 3))
    for animal in animals:
        d = per_animal_summary[per_animal_summary["animal"] == animal].sort_values("position")
        ax.plot(d["position"], d["accuracy"], color=animal_colors.get(animal, "#888"),
                alpha=0.5, linewidth=1.2)

    d = overall_summary.sort_values("position")
    ax.plot(d["position"], d["mean"], color="black", linewidth=lw * 1.5, zorder=10)
    ax.fill_between(d["position"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                     color="black", alpha=0.2, zorder=9)

    ax.axvline(0, color="gray", linestyle=":", linewidth=lw, alpha=0.5)
    ax.set_xlabel("Block position", fontsize=fontsize)
    ax.set_ylabel("p(high prob choice)", fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    clean_axes(ax)
    ax.tick_params(axis="both", labelsize=fontsize, length=config["plotting"]["tick_length"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_p_break_heatmap(result, config, out_path):
    """Panel e: 2x2 heatmap of p(break) by previous outcome (rows:
    Unrewarded/Rewarded) x block status (columns: Same Block/Transition),
    matching the manuscript's own panel e layout/orientation."""
    fontsize = config["plotting"]["fontsize"]
    cells = result["cells"]

    row_labels = ["unrewarded", "rewarded"]
    col_labels = ["same_block", "transition"]
    row_display = ["Unrewarded", "Rewarded"]
    col_display = ["Same Block", "Transition"]

    grid = np.array([
        [cells[f"{row}_{col}"]["mean"] for col in col_labels]
        for row in row_labels
    ])

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    im = ax.imshow(grid, cmap="YlOrRd", vmin=0, vmax=max(0.05, grid.max() * 1.15))
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=fontsize)

    ax.set_xticks(range(len(col_display)))
    ax.set_xticklabels(col_display, fontsize=fontsize)
    ax.set_yticks(range(len(row_display)))
    ax.set_yticklabels(row_display, fontsize=fontsize)
    ax.set_xlabel("Block status", fontsize=fontsize)
    ax.set_ylabel("Previous outcome", fontsize=fontsize)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("p(Break)", fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize - 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pre_block_change_accuracy(pre_result, config, out_path):
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    tick_length = config["plotting"]["tick_length"]
    animal_colors = config["plotting"]["animal_colors"]

    mean = pre_result["mean"]
    sem = pre_result["sem"]
    per_animal = pre_result["per_animal_accuracy"]

    fig, ax = plt.subplots(figsize=(2, 3))

    ax.bar([0], [mean], width=0.6, color="slategray", alpha=0.7,
           edgecolor="black", linewidth=lw)
    ax.errorbar([0], [mean], yerr=[sem], fmt="none", color="black", linewidth=lw,
                capsize=3, capthick=lw, zorder=10)

    animals = list(per_animal.keys())
    values = [per_animal[a] for a in animals]
    x_jitter = 0 + np.random.uniform(-0.15, 0.15, size=len(animals))
    ax.scatter(x_jitter, values, s=15, color=[animal_colors.get(a, "gray") for a in animals],
               alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(["Pre-transition"], fontsize=fontsize)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    ax.tick_params(axis="both", labelsize=fontsize, length=tick_length)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_p_break_comparison(categories, xlabel, p_value, config, out_path):
    """categories: list of dicts (in the exact left-to-right draw order,
    matching the source's alphabetically-groupby-sorted category order),
    each with keys "label", "color", "mean", "sem", "per_animal" (dict of
    animal -> value)."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    tick_length = config["plotting"]["tick_length"]
    animal_colors = config["plotting"]["animal_colors"]

    x_pos = np.arange(len(categories))
    width = 0.6
    means = [c["mean"] for c in categories]
    sems = [c["sem"] for c in categories]
    colors = [c["color"] for c in categories]
    labels = [c["label"] for c in categories]

    fig, ax = plt.subplots(figsize=(2, 2))

    ax.bar(x_pos, means, width, color=colors, alpha=0.7, edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, means, yerr=sems, fmt="none", color="black", linewidth=lw,
                capsize=3, capthick=lw, zorder=10)

    for idx, cat in enumerate(categories):
        animals = list(cat["per_animal"].keys())
        values = [cat["per_animal"][a] for a in animals]
        x_jitter = idx + np.random.uniform(-0.15, 0.15, size=len(animals))
        ax.scatter(x_jitter, values, s=40, color=[animal_colors.get(a, "gray") for a in animals],
                   alpha=0.7, edgecolor="black", linewidth=0.5, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fontsize)
    ax.set_ylabel("P(Break)", fontsize=fontsize + 1)
    ax.set_xlabel(xlabel, fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    ax.tick_params(axis="both", labelsize=fontsize, length=tick_length)
    ax.set_ylim(0, 0.5)

    ax.text(0.5, ax.get_ylim()[1] * 0.95, f"p={p_value:.10f}", ha="center", fontsize=fontsize,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
