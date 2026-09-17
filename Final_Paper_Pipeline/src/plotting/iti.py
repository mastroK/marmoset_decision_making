"""Panel rendering for Figure 1, stage 2 (panels e-f: ITI distributions,
performance vs. ITI, performance after long breaks).

Ported to match the source notebook's own actual publication-figure code
(_source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb):

- plot_iti_ridge <- the standalone ridge-plot cell at lines 1391-1425
  (In[21], saves 'ITI_Distributions_ridge.svg'), NOT CELL 11's embedded
  Panel A version (lines 1152-1223, part of the 4-panel 'ITI_Analysis.svg').
  The standalone cell is later in the notebook, has its own dedicated
  single-panel output filename, and differs in several details (alpha 0.5
  not 0.7, solid not dashed median line, no legend/title) -- by the
  "later cell + own dedicated output filename wins" rule used elsewhere in
  this pipeline (e.g. Fig 6's K-Means figure), this is the authoritative
  version.
- plot_iti_performance <- the standalone cell at lines 1431-1499 (In[79],
  "Performance vs ITI - Standalone with Individual Animals", saves
  'Performance_vs_ITI_All_Animals.svg'), NOT CELL 11's embedded Panel C.
  This version overlays a per-animal line on top of gray group-mean bars,
  which needed a new (purely additive) per-animal aggregation function,
  `iti_analysis.performance_by_iti_bin_by_animal`.
- plot_long_break_performance <- CELL 11's Panel D (lines 1310-1364, part
  of 'ITI_Analysis.svg'). No standalone single-panel cell exists for this
  specific panel anywhere else in the notebook (confirmed by search), so
  CELL 11's embedded version is the only -- and therefore authoritative --
  source for this panel's styling.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes
from ..features.history_encoding import sort_animals


def plot_iti_ridge(df_timing_filtered, config, out_path):
    """Ported from the standalone ridge-plot cell, lines 1391-1425."""
    fontsize = config["plotting"]["fontsize"]
    tick_length = config["plotting"]["tick_length"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    fig, ax = plt.subplots(figsize=(4, 2))
    offset_scale = 0.15
    animals = sort_animals(df_timing_filtered["Animal_Name"].unique(), animal_order)

    for idx, animal in enumerate(animals):
        data = df_timing_filtered[df_timing_filtered["Animal_Name"] == animal]["ITI"].values
        counts, bins = np.histogram(data, bins=50, range=(0, 50), density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        baseline = idx * offset_scale
        ax.fill_between(bin_centers, baseline, baseline + counts, color=animal_colors.get(animal, "#888"),
                         alpha=0.5, edgecolor="black", linewidth=0.3, label=animal)
        median_iti = np.median(data)
        ax.plot([median_iti, median_iti], [baseline, baseline + counts.max()],
                color=animal_colors.get(animal, "#888"), linestyle="-", linewidth=2, alpha=0.8)

    ax.set_xlabel("intertrial interval (secs)", fontsize=fontsize)
    ax.set_ylabel("Animal", fontsize=fontsize)
    ax.set_xlim(0, 15)
    ax.set_yticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="x", labelsize=fontsize, length=tick_length)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_iti_performance(iti_performance, animal_iti_performance, config, out_path):
    """Ported from the standalone "Performance vs ITI - Standalone with
    Individual Animals" cell, lines 1452-1499.

    `animal_iti_performance` is the new (purely additive) per-animal
    accuracy-by-ITI-bin frame from
    `behavior_metrics.iti_analysis.performance_by_iti_bin_by_animal` --
    needed to draw the per-animal line overlay this panel has in the
    source, which the previous version of this function did not receive.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    tick_length = config["plotting"]["tick_length"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    iti_labels = list(iti_performance["ITI_bin"])

    fig, ax = plt.subplots(figsize=(2, 2))
    x_pos = np.arange(len(iti_performance))

    ax.bar(x_pos, iti_performance["accuracy"], color="gray", alpha=0.5,
           edgecolor="black", linewidth=lw, label="Group Mean", zorder=1)
    ax.errorbar(x_pos, iti_performance["accuracy"], yerr=iti_performance["sem"],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=2)

    for animal in sort_animals(animal_iti_performance["animal"].unique(), animal_order):
        animal_data = animal_iti_performance[animal_iti_performance["animal"] == animal]
        if len(animal_data) >= 3:
            x_positions = [iti_labels.index(b) for b in animal_data["ITI_bin"] if b in iti_labels]
            y_values = animal_data["accuracy"].values[:len(x_positions)]
            ax.plot(x_positions, y_values, "o-", color=animal_colors.get(animal, "#888"),
                    linewidth=1.5, markersize=5, alpha=0.7, label=animal, zorder=3)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(iti_performance["ITI_bin"], rotation=30, ha="right")
    ax.set_ylabel("accuracy", fontsize=fontsize)
    ax.set_xlabel("intertrial interval", fontsize=fontsize)
    ax.set_ylim(0.4, 1.0)
    clean_axes(ax)
    ax.tick_params(axis="both", labelsize=fontsize, length=tick_length)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_long_break_performance(break_result, long_break_threshold, config, out_path):
    """Ported from CELL 11's Panel D, lines 1310-1364."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    tick_length = config["plotting"]["tick_length"]
    trial_level = break_result["trial_level"]

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = [f"Short ITI\n(<{long_break_threshold}s)", f"Long ITI\n(≥{long_break_threshold}s)"]
    row_false = trial_level[trial_level["Long_Break"] == False].iloc[0]
    row_true = trial_level[trial_level["Long_Break"] == True].iloc[0]
    accs = [row_false["accuracy"], row_true["accuracy"]]
    sems = [row_false["sem"], row_true["sem"]]
    ns = [row_false["n"], row_true["n"]]

    x_pos = np.arange(2)
    ax.bar(x_pos, accs, color=["steelblue", "coral"], alpha=0.7, edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, accs, yerr=sems, fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw)

    for i in range(2):
        ax.text(i, accs[i] + sems[i] + 0.02, f"n={ns[i]:,}", ha="center", fontsize=fontsize - 1)

    ax.text(0.5, 0.95, f"t={break_result['t_stat_trial_level']:.2f}, p={break_result['p_trial_level']:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=fontsize - 1,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8, linewidth=lw))

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.set_xlabel("Break Duration", fontsize=fontsize)
    ax.set_title("Performance After Breaks", fontsize=fontsize + 1, fontweight="bold")
    ax.set_ylim(0.4, 1.0)
    clean_axes(ax)
    ax.tick_params(axis="both", labelsize=fontsize, length=tick_length)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
