"""Panel rendering for Fig 8 -- "Deterministic reward contingencies
reorganize behavioral strategy through state-specific changes in reward
sensitivity" (manuscript Fig 8 legend, ~page 29 of
2026.06.30.734629v1.full (4).pdf).

FULL REWRITE (Part 3 of the Fig 7/Fig 8 build): the previous version of this
module rendered the earlier build's two output files
(`Figure_8f_transition_heatmap`, `Figure_8fg`), based on `classify_state_updated`
and covering only the transition-heatmap + Left/Right-Bias-switching panels.
That scope has been superseded -- the figure is rebuilt here on the manuscript's
canonical 7-state taxonomy (`state_classifier.classify_state_v7`) with its real
6 panels (a-f, per the manuscript legend): (a) p(high-prob choice) aligned to
reversal + paired accuracy scatter; (b) p(switch) aligned to reversal;
(c) per-state trial proportion (5 non-Adaptation states), paired; (d) win-stay/
lose-switch for Exploitation + Directed Exploration, paired; (e) p(switch) vs.
consecutive losses within Exploitation; (f) Delta transition-probability 2x2
heatmap (Exploitation <-> Directed Exploration only), with significance
asterisks -- this last panel keeps the earlier build's heatmap rendering
approach (colormap, cell annotation style), since that part of the design
still matches the manuscript's own panel f exactly.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_80 = "#4878CF"
COL_100 = "#D65F5F"
GREY = "#AAAAAA"


def _sig_label(p):
    if p is None or np.isnan(p):
        return "n.d."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def plot_panel_a(pos_summary_80, pos_summary_100, animal_acc_result, out_path):
    """Fig 8a: (left) p(high-prob choice) aligned to block reversal for
    80-20 (blue, dashed) vs. 100-0 (purple/red, solid); (right) paired
    per-animal mean session accuracy scatter, paired t-test."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), width_ratios=[1.6, 1])
    ax_line, ax_scatter = axes

    for summary, color, label, style in (
        (pos_summary_80, COL_80, "80-20", "--"),
        (pos_summary_100, COL_100, "100-0", "-"),
    ):
        ax_line.plot(summary["position"], summary["mean"], style, color=color, label=label, linewidth=1.2)
        ax_line.fill_between(summary["position"], summary["mean"] - summary["sem"],
                              summary["mean"] + summary["sem"], color=color, alpha=0.2)
    ax_line.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax_line.set_xlabel("trial relative to reversal", fontsize=8)
    ax_line.set_ylabel("p(high prob choice)", fontsize=8)
    ax_line.set_ylim(0, 1.0)
    ax_line.legend(fontsize=7, frameon=False)
    clean_axes(ax_line)

    v80, v100 = animal_acc_result["values_80_20"], animal_acc_result["values_100_0"]
    x = np.array([0, 1])
    for i in range(len(v80)):
        ax_scatter.plot(x, [v80[i], v100[i]], color=GREY, linewidth=0.8, zorder=1)
        ax_scatter.scatter(x, [v80[i], v100[i]], color=[COL_80, COL_100], s=24, zorder=2)
    ax_scatter.set_xticks(x)
    ax_scatter.set_xticklabels(["80-20", "100-0"], fontsize=8)
    ax_scatter.set_ylabel("mean session accuracy", fontsize=8)
    ax_scatter.set_title(_sig_label(animal_acc_result["p"]), fontsize=9)
    ax_scatter.set_xlim(-0.4, 1.4)
    clean_axes(ax_scatter)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_b(pos_summary_80, pos_summary_100, out_path):
    """Fig 8b: mean p(switch) aligned to block reversal, 80-20 vs. 100-0."""
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for summary, color, label, style in (
        (pos_summary_80, COL_80, "80-20", "--"),
        (pos_summary_100, COL_100, "100-0", "-"),
    ):
        ax.plot(summary["position"], summary["mean"], style, color=color, label=label, linewidth=1.2)
        ax.fill_between(summary["position"], summary["mean"] - summary["sem"],
                         summary["mean"] + summary["sem"], color=color, alpha=0.2)
    ax.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlabel("trial relative to reversal", fontsize=8)
    ax.set_ylabel("p(switch)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_c(proportions_result, states_order, out_path):
    """Fig 8c: proportion of trials in each of 5 behavioral states
    (excluding Adapting) for 80-20 vs. 100-0, one small panel per state,
    paired per-animal dots+lines, paired t-test per state."""
    states = [s for s in states_order if s in proportions_result]
    fig, axes = plt.subplots(1, len(states), figsize=(2.0 * len(states), 2.4), sharex=True)
    if len(states) == 1:
        axes = [axes]
    x = np.array([0, 1])

    for ax, state in zip(axes, states):
        r = proportions_result[state]
        v80, v100 = r["values_80_20"], r["values_100_0"]
        for i in range(len(v80)):
            ax.plot(x, [v80[i], v100[i]], color=GREY, linewidth=0.8, zorder=1)
            ax.scatter(x, [v80[i], v100[i]], color=[COL_80, COL_100], s=22, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(["80-20", "100-0"], fontsize=7)
        ax.set_title(f"{state}\n{_sig_label(r['p'])}", fontsize=7.5)
        ax.set_xlim(-0.4, 1.4)
        clean_axes(ax)
    axes[0].set_ylabel("proportion of trials", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_d(wsls_result, states_order, out_path):
    """Fig 8d: win-stay (left) and lose-switch (right) probabilities for
    Exploitation and Directed Exploration, 80-20 vs. 100-0, paired per-animal."""
    states = [s for s in states_order if s in wsls_result["win_stay"]]
    fig, axes = plt.subplots(1, 2, figsize=(2.2 * len(states), 2.6))
    x_base = np.arange(len(states))
    width = 0.3

    for ax, metric, title in ((axes[0], "win_stay", "win-stay"), (axes[1], "lose_switch", "lose-switch")):
        for i, state in enumerate(states):
            r = wsls_result[metric][state]
            v80, v100 = r["values_80_20"], r["values_100_0"]
            m80, sem80 = np.nanmean(v80), np.nanstd(v80, ddof=1) / np.sqrt(len(v80))
            m100, sem100 = np.nanmean(v100), np.nanstd(v100, ddof=1) / np.sqrt(len(v100))

            ax.bar(i - width / 2, m80, width, yerr=sem80, color=COL_80, capsize=2, edgecolor="white", linewidth=0.5)
            ax.bar(i + width / 2, m100, width, yerr=sem100, color=COL_100, capsize=2, edgecolor="white", linewidth=0.5)
            for j in range(len(v80)):
                ax.plot([i - width / 2, i + width / 2], [v80[j], v100[j]], color=GREY, linewidth=0.6, zorder=1)
            ax.text(i, max(m80 + sem80, m100 + sem100) + 0.03, _sig_label(r["p"]), ha="center", fontsize=8)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_xticks(x_base)
        ax.set_xticklabels(states, fontsize=7.5, rotation=15, ha="right")
        ax.set_title(metric.replace("_", "-"), fontsize=8)
        ax.set_ylim(0, 1.15)
        clean_axes(ax)
    axes[0].set_ylabel("probability", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_e(streak_summary_80, streak_summary_100, out_path, state_label="Exploitation"):
    """Fig 8e: probability of switching as a function of consecutive losses
    within the Exploitation state, 80-20 (dashed) vs. 100-0 (solid), +/- SEM."""
    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    for summary, color, label, style in (
        (streak_summary_80, COL_80, "80-20", "o--"),
        (streak_summary_100, COL_100, "100-0", "o-"),
    ):
        ax.plot(summary["Loss_Streak"], summary["mean"], style, color=color, label=label,
                linewidth=1.2, markersize=4)
        ax.fill_between(summary["Loss_Streak"], summary["mean"] - summary["sem"],
                         summary["mean"] + summary["sem"], color=color, alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel(f"consecutive losses in {state_label}", fontsize=7.5)
    ax.set_ylabel("p(switch)", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_f(diff_result, out_path):
    """Fig 8f: change in trial-to-trial transition probability (100-0 minus
    80-20), restricted to Exploitation<->Directed Exploration transitions,
    as a 2x2 heatmap with significance asterisks. Rendering approach carried
    over from the earlier build's transition-heatmap cell (colormap, cell
    annotation style), still matching the manuscript's own panel f."""
    states_plot = [s.replace(" ", "\n", 1) if s == "Directed Exploration" else s
                   for s in diff_result["states"]]
    diff_matrix = diff_result["diff_matrix"]
    pval_matrix = diff_result["pval_matrix"]
    n = len(diff_result["states"])

    fig, ax = plt.subplots(figsize=(2.6, 2.4))
    vmax = np.nanmax(np.abs(diff_matrix))
    im = ax.imshow(diff_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Delta transition probability\n(100-0 - 80-20)", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    for i in range(n):
        for j in range(n):
            v = diff_matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "n.d.", ha="center", va="center", fontsize=7, color="gray")
                continue
            sig = "*" if pval_matrix[i, j] < 0.05 else ""
            tcol = "white" if abs(v) > vmax * 0.55 else "black"
            ax.text(j, i, f"{v:+.2f}{sig}", ha="center", va="center", fontsize=8,
                     fontweight="bold" if sig else "normal", color=tcol)

    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.8)
        ax.axvline(k - 0.5, color="white", linewidth=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(states_plot, fontsize=8)
    ax.set_yticklabels(states_plot, fontsize=8)
    ax.set_xlabel("To state", fontsize=8)
    ax.set_ylabel("From state", fontsize=8)
    ax.text(n - 0.45, n - 0.3, "* p < 0.05", fontsize=7, color="gray", ha="right")

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
