"""Reviewer-response addition (RA3_iti_and_transitions.ipynb, R1-C3, R2-4,
R2-9). PAPER-candidate panels: ITI by state, pooled transition matrix with
bootstrap CI. LETTER-ONLY: ITI preceding transitions vs. within-state.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_BAR = "#4878CF"
COL_TRANSITION = "#D65F5F"
FONTSIZE = 8


def plot_iti_by_state(state_summary, out_path):
    """PAPER candidate. Mean +/- SEM ITI per behavioral state, states
    ordered by mean ITI. `state_summary` is a list of records with
    Behavioral_State/mean/sem/n."""
    rows = sorted(state_summary, key=lambda r: r["mean"])
    states = [r["Behavioral_State"] for r in rows]
    means = [r["mean"] for r in rows]
    sems = [r["sem"] for r in rows]

    fig, ax = plt.subplots(figsize=(4.2, 2.6))
    x = np.arange(len(states))
    ax.bar(x, means, yerr=sems, color=COL_BAR, width=0.6, edgecolor="black",
           linewidth=0.6, error_kw={"elinewidth": 1.0, "capsize": 3})
    ax.set_xticks(x)
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=FONTSIZE - 1)
    ax.set_ylabel("mean ITI (s)\n± SEM across animals", fontsize=FONTSIZE)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_transition_matrix(point_df, ci_low_df, ci_high_df, out_path):
    """PAPER candidate. Pooled state-transition probability matrix, cell
    color = point estimate, cell text = point [CI low-high] from the
    animal-cluster bootstrap."""
    states = list(point_df.index)
    mat = point_df.values

    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, rotation=45, ha="right", fontsize=FONTSIZE - 1)
    ax.set_yticks(range(len(states)))
    ax.set_yticklabels(states, fontsize=FONTSIZE - 1)
    ax.set_xlabel("to state", fontsize=FONTSIZE)
    ax.set_ylabel("from state", fontsize=FONTSIZE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isnan(mat[i, j]):
                continue
            lo, hi = ci_low_df.values[i, j], ci_high_df.values[i, j]
            ax.text(j, i, f"{mat[i, j]:.2f}\n[{lo:.2f}-{hi:.2f}]", ha="center", va="center", fontsize=5,
                     color="white" if mat[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="p(transition)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_iti_preceding_transitions(preceding_mean, within_mean, animal_level, p_value, out_path):
    """LETTER-ONLY. Paired per-animal ITI, preceding-a-transition vs.
    within-state, with connecting lines (same 5 animals contribute both)."""
    fig, ax = plt.subplots(figsize=(2.2, 2.6))
    x = np.array([0, 1])
    for _, row in animal_level.iterrows():
        y = [row["within_state"], row["preceding_transition"]]
        ax.plot(x, y, color="#AAAAAA", linewidth=0.8, zorder=1)
        ax.scatter(x, y, color=[COL_BAR, COL_TRANSITION], s=24, zorder=2, edgecolor="black", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["within\nstate", "preceding\ntransition"], fontsize=FONTSIZE)
    ax.set_ylabel("mean ITI (s)", fontsize=FONTSIZE)
    ax.set_title(f"p={p_value:.3g}" if p_value is not None else "n.d.", fontsize=FONTSIZE)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
