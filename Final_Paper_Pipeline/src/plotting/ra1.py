"""Reviewer-response addition (RA1_exploration_states.ipynb, R1-C1/R2-1/R1-s).

PAPER-candidate panel (psychometric slope by state) follows this
project's established paired-comparison style exactly
(`fig8.py::plot_panel_a`'s per-animal scatter: grey connecting lines,
state-colored endpoints, significance label from `_sig_label`) since the
same 5 animals contribute a value to both states, same as Fig 8's
same-subject condition comparisons. LETTER-ONLY panels (reclassification
crosstab, sensitivity sweep) are simpler diagnostic renderings, not
manuscript-styled.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .style import clean_axes

COL_RANDOM = "#D65F5F"
COL_DIRECTED = "#4878CF"
GREY = "#AAAAAA"
FONTSIZE = 8


def _sig_label(p):
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "n.d."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_psychometric_slope_by_state(animal_df, states, paired_ttest_p, out_path):
    """PAPER candidate. Per-animal paired psychometric slope (Random vs.
    Directed Exploration), grey connecting lines + state-colored dots,
    plus each state's mean +/- 95% CI as a horizontal bar -- same visual
    grammar as `fig8.py::plot_panel_a`'s paired accuracy scatter.
    """
    random_state, directed_state = states
    wide = animal_df.pivot(index="Animal_Name", columns="Behavioral_State", values="slope").dropna(
        subset=[random_state, directed_state]
    )

    fig, ax = plt.subplots(figsize=(2.6, 2.6))
    x = np.array([0, 1])
    for animal in wide.index:
        y = [wide.loc[animal, random_state], wide.loc[animal, directed_state]]
        ax.plot(x, y, color=GREY, linewidth=0.8, zorder=1)
        ax.scatter(x, y, color=[COL_RANDOM, COL_DIRECTED], s=24, zorder=2, edgecolor="black", linewidth=0.4)

    means = [wide[random_state].mean(), wide[directed_state].mean()]
    sems = [wide[random_state].sem(), wide[directed_state].sem()]
    ax.errorbar(x, means, yerr=[1.96 * s for s in sems], fmt="none", ecolor="black", elinewidth=1.2, capsize=3, zorder=3)
    ax.scatter(x, means, color="black", marker="_", s=200, zorder=4)

    ax.axhline(0, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(["Random\nExploration", "Directed\nExploration"], fontsize=FONTSIZE)
    ax.set_ylabel("choice-vs-ΔQ psychometric slope", fontsize=FONTSIZE)
    ax.set_title(_sig_label(paired_ttest_p), fontsize=FONTSIZE + 1)
    ax.set_xlim(-0.4, 1.4)
    clean_axes(ax)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_reclassification_crosstab(crosstab, out_path):
    """LETTER-ONLY. Heatmap of the original-vs-reordered state crosstab."""
    fig, ax = plt.subplots(figsize=(4.5, 4.0))
    mat = crosstab.values.astype(float)
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(crosstab.columns)))
    ax.set_xticklabels(crosstab.columns, rotation=45, ha="right", fontsize=FONTSIZE - 1)
    ax.set_yticks(range(len(crosstab.index)))
    ax.set_yticklabels(crosstab.index, fontsize=FONTSIZE - 1)
    ax.set_xlabel("Reordered (Directed-before-Random)", fontsize=FONTSIZE)
    ax.set_ylabel("Original ordering", fontsize=FONTSIZE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=6,
                        color="white" if mat[i, j] > mat.max() * 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="n trials")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_threshold_sensitivity(threshold_sweep_results, out_path):
    """LETTER-ONLY. Random/Directed Exploration occupancy at baseline vs.
    +/-25% perturbation, one small panel per swept threshold key."""
    keys = sorted({k.rsplit("__", 1)[0] for k in threshold_sweep_results if k != "baseline"})
    fig, axes = plt.subplots(1, len(keys), figsize=(2.2 * len(keys), 2.4), sharey=True)
    if len(keys) == 1:
        axes = [axes]

    baseline = threshold_sweep_results["baseline"]
    for ax, key in zip(axes, keys):
        minus = threshold_sweep_results[f"{key}__minus25pct"]
        plus = threshold_sweep_results[f"{key}__plus25pct"]
        for state, color in (("Random Exploration", COL_RANDOM), ("Directed Exploration", COL_DIRECTED)):
            ys = [minus[state], baseline[state], plus[state]]
            ax.plot([0, 1, 2], ys, "o-", color=color, markersize=4, label=state)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["-25%", "base", "+25%"], fontsize=6)
        ax.set_title(key.replace("_", " "), fontsize=6)
        clean_axes(ax)
    axes[0].set_ylabel("occupancy (proportion)", fontsize=FONTSIZE)
    axes[0].legend(fontsize=6, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
