"""Reviewer-response addition (RA5_small_items.ipynb, R1-f, R2-3).
PAPER-candidate panels: 3-arm feature-ablation ARI comparison, the
no-Signed_Deviation mechanism crosstab. LETTER-ONLY: choice-history
variance-given-reward-history.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_KMEANS = "#4878CF"
COL_HMM = "#D65F5F"
FONTSIZE = 8


def plot_ablation_ari_comparison(ari_by_arm, out_path):
    """PAPER candidate. K-Means/HMM ARI across three feature-set arms:
    full, without Signed_Deviation, without Choice_Deviation.
    `ari_by_arm` = {"full": {"kmeans": v, "hmm": v}, "no_signed": {...}, "no_choice": {...}}.
    """
    arms = ["full", "no_signed", "no_choice"]
    labels = ["Full", "No\nSigned_Dev.", "No\nChoice_Dev."]

    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(arms))
    width = 0.35
    kmeans_vals = [ari_by_arm[a]["kmeans"] for a in arms]
    hmm_vals = [ari_by_arm[a]["hmm"] for a in arms]
    ax.bar(x - width / 2, kmeans_vals, width, color=COL_KMEANS, edgecolor="black", linewidth=0.6, label="K-Means")
    ax.bar(x + width / 2, hmm_vals, width, color=COL_HMM, edgecolor="black", linewidth=0.6, label="HMM")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=FONTSIZE)
    ax.set_ylabel("held-out ARI", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_mechanism_crosstab(crosstab_dict, out_path):
    """PAPER candidate. True-state x predicted-cluster heatmap for the
    no-Signed_Deviation run, showing the Left/Right Bias merge. Takes the
    manifest's own `crosstab_no_signed_deviation` dict-of-dicts (row-major:
    {true_state: {pred_cluster: n}})."""
    true_states = list(crosstab_dict.keys())
    pred_clusters = sorted({k for row in crosstab_dict.values() for k in row})
    mat = np.array([[crosstab_dict[t].get(c, 0) for c in pred_clusters] for t in true_states], dtype=float)
    row_sums = mat.sum(axis=1, keepdims=True)
    norm = np.divide(mat, row_sums, out=np.zeros_like(mat), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(pred_clusters)))
    ax.set_xticklabels(pred_clusters, rotation=45, ha="right", fontsize=FONTSIZE - 1)
    ax.set_yticks(range(len(true_states)))
    ax.set_yticklabels(true_states, fontsize=FONTSIZE - 1)
    ax.set_xlabel("predicted K-Means cluster (no Signed_Deviation)", fontsize=FONTSIZE - 1)
    ax.set_ylabel("true Behavioral_State_v3", fontsize=FONTSIZE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if mat[i, j] > 0:
                ax.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", fontsize=6,
                         color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalized")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_choice_history_variance(variance_df, out_path):
    """LETTER-ONLY. p_switch range across choice-history codes, per
    reward-history sequence."""
    df_sorted = variance_df.sort_values("p_switch_range", ascending=False)
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    x = np.arange(len(df_sorted))
    ax.bar(x, df_sorted["p_switch_range"], color="#888888", width=0.6, edgecolor="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(df_sorted["reward_seq"], fontsize=FONTSIZE - 1)
    ax.set_xlabel("reward_seq (last 3 trials)", fontsize=FONTSIZE)
    ax.set_ylabel("p(switch) range across\nchoice-history codes", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.05)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
