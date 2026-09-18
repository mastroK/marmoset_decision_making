"""Reviewer-response addition (RA4_condition_wise_RL.ipynb, R2-5, R1-D).
PAPER-candidate panels: parameter recovery (true vs. fitted), predictability
gradient (Exploit dwell time vs. predictability). LETTER-ONLY: the primary
80-20-vs-90-10 paired comparison, with boundary-pinned animals marked.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_8020 = "#4878CF"
COL_9010 = "#59A14F"
COL_1000 = "#D65F5F"
COND_COLORS = {"80-20": COL_8020, "90-10": COL_9010, "100-0": COL_1000}
FONTSIZE = 8


def plot_parameter_recovery(detail_df, conditions, out_path):
    """PAPER candidate. True vs. fitted alpha, one panel per condition --
    the clearest visual for the 100-0 identifiability story: points should
    fall on the y=x line if recovery is good, and scatter/pin at the
    bounds if not.
    """
    fig, axes = plt.subplots(1, len(conditions), figsize=(2.4 * len(conditions), 2.6), sharex=True, sharey=True)
    if len(conditions) == 1:
        axes = [axes]

    for ax, cond in zip(axes, conditions):
        sub = detail_df[detail_df["prob_condition"] == cond]
        ax.plot([0, 1], [0, 1], color="black", linewidth=0.6, linestyle=":", zorder=1)
        ax.scatter(sub["true_alpha"], sub["fitted_alpha"], color=COND_COLORS.get(cond, "#888888"),
                    s=14, alpha=0.6, edgecolor="none", zorder=2)
        ax.set_title(cond, fontsize=FONTSIZE)
        ax.set_xlabel("true alpha", fontsize=FONTSIZE)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        clean_axes(ax)
    axes[0].set_ylabel("fitted alpha", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_predictability_gradient(exploit_dwell_df, out_path):
    """PAPER candidate. Exploit-state mean dwell time (trials) vs. reward
    predictability (high-probability value)."""
    fig, ax = plt.subplots(figsize=(2.8, 2.6))
    df_sorted = exploit_dwell_df.sort_values("predictability")
    ax.plot(df_sorted["predictability"], df_sorted["mean_dwell_trials"], "o-", color=COL_8020, markersize=6)
    for _, row in df_sorted.iterrows():
        ax.annotate(row["prob_condition"], (row["predictability"], row["mean_dwell_trials"]),
                     textcoords="offset points", xytext=(4, 4), fontsize=6)
    ax.set_xlabel("predictability (high-reward probability)", fontsize=FONTSIZE)
    ax.set_ylabel("Exploit mean dwell (trials)", fontsize=FONTSIZE)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_primary_8020_vs_9010(values_80_20, values_90_10, animals, pinned_animals, out_path):
    """LETTER-ONLY. Paired per-animal alpha, 80-20 vs. 90-10, with
    boundary-pinned animals (in EITHER condition) marked with an open
    (unfilled) marker instead of solid, rather than dropped."""
    fig, ax = plt.subplots(figsize=(2.4, 2.6))
    x = np.array([0, 1])
    for a in animals:
        y = [values_80_20.get(a, np.nan), values_90_10.get(a, np.nan)]
        if any(np.isnan(y)):
            continue
        is_pinned = a in pinned_animals
        ax.plot(x, y, color="#AAAAAA", linewidth=0.8, zorder=1)
        ax.scatter(x, y, s=28, zorder=2,
                    facecolor="none" if is_pinned else [COL_8020, COL_9010],
                    edgecolor=[COL_8020, COL_9010], linewidth=1.2 if is_pinned else 0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["80-20", "90-10"], fontsize=FONTSIZE)
    ax.set_ylabel("fitted alpha\n(open = boundary-pinned)", fontsize=FONTSIZE)
    ax.set_ylim(-0.05, 1.05)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
