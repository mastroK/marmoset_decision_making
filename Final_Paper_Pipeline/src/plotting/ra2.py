"""Reviewer-response addition (RA2_validation_statistics.ipynb, R2-Maj1,
R2-2, R2-8, R1-C2). PAPER-candidate panels: cross-validated ARI, k-selection
curves, PCA loadings, RF confusion matrix. LETTER-ONLY: control-clustering
ARI comparison.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_KMEANS = "#4878CF"
COL_HMM = "#D65F5F"
FONTSIZE = 8


def plot_cross_validated_ari(kmeans_summary, hmm_summary, out_path):
    """PAPER candidate. K-Means vs. HMM cross-validated ARI, mean +/- 95% CI
    across the 5 folds."""
    fig, ax = plt.subplots(figsize=(2.2, 2.6))
    x = np.array([0, 1])
    means = [kmeans_summary["mean"], hmm_summary["mean"]]
    los = [kmeans_summary["mean"] - kmeans_summary["ci95_low"], hmm_summary["mean"] - hmm_summary["ci95_low"]]
    his = [kmeans_summary["ci95_high"] - kmeans_summary["mean"], hmm_summary["ci95_high"] - hmm_summary["mean"]]
    colors = [COL_KMEANS, COL_HMM]
    ax.bar(x, means, color=colors, width=0.6, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["K-Means\n(K=4)", "HMM\n(K=3)"], fontsize=FONTSIZE)
    ax.set_ylabel("cross-validated ARI\n(mean ± 95% CI, 5 folds)", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_k_selection(k_selection_table, k_used, out_path):
    """PAPER candidate. Silhouette and BIC vs. k (mean across folds), with
    the actually-used k marked -- shows BIC saturating at the tested
    range's edge rather than turning over."""
    ks = [row["k"] for row in k_selection_table]
    sil = [row["mean_silhouette"] for row in k_selection_table]
    bic = [row["mean_bic"] for row in k_selection_table]

    fig, axes = plt.subplots(1, 2, figsize=(4.6, 2.4))
    axes[0].plot(ks, sil, "o-", color=COL_KMEANS, markersize=4)
    axes[0].axvline(k_used, color="black", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("k", fontsize=FONTSIZE)
    axes[0].set_ylabel("mean silhouette", fontsize=FONTSIZE)
    clean_axes(axes[0])

    axes[1].plot(ks, bic, "o-", color=COL_HMM, markersize=4)
    axes[1].axvline(k_used, color="black", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("k", fontsize=FONTSIZE)
    axes[1].set_ylabel("mean BIC", fontsize=FONTSIZE)
    clean_axes(axes[1])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pca_loadings(loadings_df, out_path):
    """PAPER candidate. Feature x component loading heatmap (answers R2-8)."""
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    mat = loadings_df.values
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(mat.shape[1]))
    ax.set_xticklabels(loadings_df.columns, fontsize=FONTSIZE)
    ax.set_yticks(range(mat.shape[0]))
    ax.set_yticklabels(loadings_df.index, fontsize=FONTSIZE)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if abs(mat[i, j]) > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="loading")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_rf_confusion_matrix(cm_df, out_path):
    """PAPER candidate. Row-normalized (recall-view) RF confusion matrix,
    summed across the 5 folds."""
    row_sums = cm_df.values.sum(axis=1, keepdims=True)
    norm = cm_df.values / row_sums

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cm_df.columns)))
    ax.set_xticklabels(cm_df.columns, rotation=45, ha="right", fontsize=FONTSIZE - 1)
    ax.set_yticks(range(len(cm_df.index)))
    ax.set_yticklabels(cm_df.index, fontsize=FONTSIZE - 1)
    ax.set_xlabel("RF-predicted", fontsize=FONTSIZE)
    ax.set_ylabel("true (Behavioral_State_v3)", fontsize=FONTSIZE)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=6,
                     color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, shrink=0.8, label="row-normalized (recall)")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_control_clustering_comparison(base_summary, control_summary, out_path):
    """LETTER-ONLY. Base 4-feature vs. control (+WSLS/violation) K-Means
    ARI, mean +/- 95% CI across the same 5 folds."""
    fig, ax = plt.subplots(figsize=(2.0, 2.6))
    x = np.array([0, 1])
    means = [base_summary["mean"], control_summary["mean"]]
    los = [base_summary["mean"] - base_summary["ci95_low"], control_summary["mean"] - control_summary["ci95_low"]]
    his = [base_summary["ci95_high"] - base_summary["mean"], control_summary["ci95_high"] - control_summary["mean"]]
    ax.bar(x, means, color=[COL_KMEANS, "#888888"], width=0.6, edgecolor="black", linewidth=0.6)
    ax.errorbar(x, means, yerr=[los, his], fmt="none", ecolor="black", elinewidth=1.2, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["4 features\n(base)", "+ WSLS/\nviolation"], fontsize=FONTSIZE)
    ax.set_ylabel("K-Means ARI\n(mean ± 95% CI, 5 folds)", fontsize=FONTSIZE)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
