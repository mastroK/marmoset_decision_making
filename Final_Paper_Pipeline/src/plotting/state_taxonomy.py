"""Panel rendering for Fig 6 (K-Means + HMM clustering, the manuscript's real
panels a-f) and Supp Fig 4 (their model-selection curves / cluster / HMM
mean panels, the manuscript's real panels a-c, plus panel d -- four
per-state behavioral metrics on the canonical 7-state taxonomy built for
Fig 7, see `plot_panel_d_behavioral_metrics` below).

SUPERSEDES an earlier version of this module (`plot_kmeans_publication_figure`,
`plot_hmm_publication_figure`, `plot_ari_summary`, `plot_hmm_bcd_panels`,
now removed) that was ported directly from
`_source_archive/by_figure/Fig6/07b_KMEANS_Final.ipynb`'s own internal
"PUBLICATION FIGURE" cells on the assumption those cells corresponded 1:1
to the manuscript's real Fig 6 / Supp Fig 4 figures. Cross-checking against
the actual submitted manuscript PDF found they do not:
  - the source's K-Means figure bundled silhouette/BIC/AIC model-selection
    curves and a cluster-mean heatmap together with the PCA scatter -- the
    former two belong to Supp Fig 4 (its real panels a/b), not Fig 6.
  - the source's 5-panel HMM figure's panel A was an example session
    traced by HMM state, and panel E was per-animal occupancy -- neither
    matches Fig 6's real panel a (a K-MEANS-colored example session) or
    panel c (occupancy PER STATE averaged across animals with individual
    ANIMAL dots, not a per-animal grouped bar). Its B/C/D (transition
    matrix / emission means / dwell time) were split too: transition
    matrix and dwell time are Fig 6's real d/e; emission means alone is
    Supp Fig 4's real panel c.
  - Fig 6's real panel f (win-stay/lose-switch by state) and panel c
    (occupancy per state) were not built by the source cells at all.

Each function below is named for the manuscript panel(s) it renders, not
for the source notebook's own cell/output-file names.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns

from .reversal_alignment import BLOCK_COLORS

FONTSIZE = 8


def plot_kmeans_example_session(df_ex, state_colors, display_labels, classes_v3, out_path):
    """Fig 6 panel a: "Representative sessions from a single animal
    (Knuckles) showing block structure (top), rewarded and unrewarded
    choices and rolling accuracy computed over a five-trial window
    (middle), and trial-by-trial K-means state classification (bottom).
    Colors correspond to the four states defined in the legend."

    `df_ex` comes from state_clustering.build_example_session(...,
    animal_col="Animal_Name", animal_name="Knuckles") called on the
    K-Means result's `df_test_classified` (has a `KMeans_State` column,
    NOT `HMM_4State` -- this is the K-Means panel, not the HMM one).

    Visual structure matches this manuscript's own established
    "exemplar session" convention used across its other representative-
    session panels (see src/plotting/reversal_alignment.py's
    `BLOCK_COLORS` constant and _source_archive/by_figure/Fig2/
    05_Fig2_Reversal_v2.ipynb's "Exemplar Session" cell): a 4-row
    gridspec, blocks colored by alternating BLOCK_COLORS with a bold
    in-block text label, a choice-location scatter colored by outcome,
    and dashed vertical lines at block transitions -- with the bottom
    row (that source convention's own ITI row) replaced here by this
    figure's own K-means `Behavioral_State_v3` classification coloring.

    The middle accuracy row deliberately reuses the pipeline's own
    `Rolling_Accuracy` FEATURE column (trailing 5-trial window,
    min_periods=3 -- the exact quantity `classify_state_v3` itself
    consumes), rather than recomputing a fresh CENTERED rolling window
    the way the source exemplar-session cells do: the point of this
    panel is to show why particular trials were classified into
    particular states, so the accuracy trace must be the one actually
    driving the state row directly below it, not a differently-smoothed
    lookalike.
    """
    fig = plt.figure(figsize=(7.5, 4.4))
    gs = fig.add_gridspec(4, 1, hspace=0.4, height_ratios=[0.5, 1, 1.5, 1])

    blocks = sorted(int(b) for b in df_ex["BlockCount"].unique())
    block_starts = df_ex.groupby("BlockCount")["t"].min()

    # Row 1: block structure, alternating BLOCK_COLORS with a block-number label
    ax1 = fig.add_subplot(gs[0])
    for block in blocks:
        bdf = df_ex[df_ex["BlockCount"] == block]
        start, end = bdf["t"].min(), bdf["t"].max()
        color = BLOCK_COLORS[block % len(BLOCK_COLORS)]
        ax1.barh(0, end - start + 1, left=start, height=0.5, color=color, edgecolor="black",
                  linewidth=0.5, alpha=0.7)
        ax1.text((start + end) / 2, 0, f"B{block}", ha="center", va="center", fontsize=FONTSIZE - 1,
                  fontweight="bold", color="white")
    ax1.set_xlim(-1, len(df_ex))
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_yticks([])
    ax1.set_xticks([])
    ax1.set_ylabel("Block", fontsize=FONTSIZE)
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Row 2: choice location, colored by outcome (green filled = rewarded,
    # red hollow = unrewarded), dashed lines at block transitions
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    unique_locs = sorted(df_ex["PhysicalChoice"].dropna().unique())
    loc_to_y = {loc: i for i, loc in enumerate(unique_locs)}
    for _, trial in df_ex.iterrows():
        if pd.notna(trial["PhysicalChoice"]):
            y_val = loc_to_y[trial["PhysicalChoice"]]
            face = "#2ecc71" if trial["Outcome_Binary"] == 1 else "none"
            edge = "#2ecc71" if trial["Outcome_Binary"] == 1 else "#e74c3c"
            ax2.scatter(trial["t"], y_val, s=10, marker="o", facecolors=face, edgecolors=edge,
                        alpha=0.7, linewidth=0.3)
    for _, st in block_starts.items():
        if st > 0:
            ax2.axvline(st - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.6, zorder=10)
    ax2.set_yticks(range(len(unique_locs)))
    ax2.set_yticklabels(unique_locs, fontsize=FONTSIZE - 2)
    ax2.set_ylim(-0.5, len(unique_locs) - 0.5)
    ax2.set_ylabel("Choice", fontsize=FONTSIZE)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(axis="x", labelbottom=False, length=0)

    # Row 3: rolling accuracy, colored per block (see docstring re: which
    # rolling-accuracy column this is and why)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for _, st in block_starts.items():
        if st > 0:
            ax3.axvline(st - 0.5, color="black", linestyle="--", linewidth=0.8, alpha=0.4, zorder=1)
    for block in blocks:
        bdf = df_ex[df_ex["BlockCount"] == block].sort_values("t")
        color = BLOCK_COLORS[block % len(BLOCK_COLORS)]
        ax3.plot(bdf["t"], bdf["Rolling_Accuracy"], "o-", color=color, linewidth=1, markersize=2,
                  alpha=0.8, zorder=5)
    ax3.axhline(0.5, color="gray", linestyle=":", linewidth=0.7, alpha=0.5)
    ax3.axhline(0.8, color="gray", linestyle="--", linewidth=0.7, alpha=0.4)
    ax3.set_ylim(0, 1.05)
    ax3.set_ylabel("Rolling\naccuracy", fontsize=FONTSIZE)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.tick_params(axis="x", labelbottom=False, length=0)

    # Row 4 (this figure's own content, replacing the source convention's
    # ITI row): trial-by-trial K-means state classification
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for idx, row in df_ex.iterrows():
        ax4.axvspan(idx - 0.5, idx + 0.5, color=state_colors.get(row["KMeans_State"], "grey"), alpha=0.9, linewidth=0)
    for _, st in block_starts.items():
        if st > 0:
            ax4.axvline(st - 0.5, color="white", linestyle=":", linewidth=0.5, alpha=0.6)
    handles = [mpatches.Patch(color=state_colors[s], label=display_labels.get(s, s)) for s in classes_v3]
    ax4.legend(handles=handles, fontsize=FONTSIZE - 2, frameon=False, ncol=4, loc="upper center",
               bbox_to_anchor=(0.5, -0.55))
    ax4.set_yticks([])
    ax4.set_ylabel("State", fontsize=FONTSIZE)
    ax4.set_xlabel("Trial", fontsize=FONTSIZE)
    ax4.spines[["top", "right"]].set_visible(False)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_pca_scatter(kmeans_result, classes, state_colors, display_labels, out_path):
    """Fig 6 panel b: "Principal component projections of the behavioral
    feature space colored by K-Means cluster identity. Left: PC1 versus
    PC2 ... Right: PC1 versus PC3 ... Small Gaussian jitter (sigma=0.03)
    added for visualization." Two scatter panels only -- no silhouette/
    BIC/AIC or cluster-mean heatmap (those are Supp Fig 4's real panels
    a/b, see plot_kmeans_model_selection / plot_kmeans_cluster_heatmap).
    """
    df_plot = kmeans_result["pca_df_plot"]
    target_col = [c for c in df_plot.columns if c not in ("PC1_j", "PC2_j", "PC3_j")][0]
    explained = kmeans_result["pca_explained_variance"]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.2, 3.4))

    axL.scatter(df_plot["PC1_j"], df_plot["PC2_j"], c=df_plot[target_col].map(state_colors),
                s=6, alpha=0.4, linewidths=0)
    for state in classes:
        axL.scatter([], [], color=state_colors[state], label=display_labels.get(state, state), s=30, alpha=0.9)
    axL.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=FONTSIZE)
    axL.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)", fontsize=FONTSIZE)
    axL.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.3)
    axL.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.3)
    axL.spines[["top", "right"]].set_visible(False)
    axL.legend(fontsize=FONTSIZE - 1, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    axL.text(0.02, 0.03, display_labels.get("Exploration", "Exploration"), transform=axL.transAxes,
              color=state_colors["Exploration"], fontsize=FONTSIZE - 1, style="italic", alpha=0.9,
              va="bottom", ha="left")
    axL.text(0.98, 0.03, display_labels.get("Engaged", "Exploitation"), transform=axL.transAxes,
              color=state_colors["Engaged"], fontsize=FONTSIZE - 1, style="italic", alpha=0.9,
              va="bottom", ha="right")

    axR.scatter(df_plot["PC1_j"], df_plot["PC3_j"], c=df_plot[target_col].map(state_colors),
                s=6, alpha=0.4, linewidths=0)
    axR.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)", fontsize=FONTSIZE)
    axR.set_ylabel(f"PC3 ({explained[2] * 100:.1f}% variance)", fontsize=FONTSIZE)
    axR.axhline(0, color="gray", ls="--", lw=0.5, alpha=0.3)
    axR.axvline(0, color="gray", ls="--", lw=0.5, alpha=0.3)
    axR.spines[["top", "right"]].set_visible(False)

    mean_pc3_left = df_plot[df_plot[target_col] == "Left Bias"]["PC3_j"].mean()
    mean_pc3_right = df_plot[df_plot[target_col] == "Right Bias"]["PC3_j"].mean()
    top_state = "Left Bias" if mean_pc3_left > mean_pc3_right else "Right Bias"
    bottom_state = "Right Bias" if top_state == "Left Bias" else "Left Bias"
    top_label = "Leftward\nActions" if top_state == "Left Bias" else "Rightward\nactions"
    bottom_label = "Rightward\nactions" if top_state == "Left Bias" else "Leftward\nActions"
    axR.text(0.98, 0.97, top_label, transform=axR.transAxes, color=state_colors[top_state],
              fontsize=FONTSIZE - 1, style="italic", va="top", ha="right", alpha=0.85)
    axR.text(0.02, 0.03, bottom_label, transform=axR.transAxes, color=state_colors[bottom_state],
              fontsize=FONTSIZE - 1, style="italic", va="bottom", ha="left", alpha=0.85)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_state_occupancy_bar(occ_df, classes, state_colors, display_labels, out_path, seed=42):
    """Fig 6 panel c: "State occupancy (% of trials) per behavioral state.
    Bars indicate the mean across animals; dots indicate individual
    animal means (n=5)." `occ_df` is one row per (animal, state) with an
    `Occupancy` (%) column -- e.g.
    state_clustering.state_occupancy_by_animal(df, classes,
    state_col="Behavioral_State_v3") called on the FULL classified
    dataset (Unknown trials excluded beforehand by the caller), not just
    a held-out split -- occupancy of this rule-based classifier's own
    labels is a descriptive statistic, not part of the K-Means/HMM
    train/test validation.
    """
    rng = np.random.default_rng(seed)
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    x = np.arange(len(classes))

    means = [occ_df.loc[occ_df["State"] == s, "Occupancy"].mean() for s in classes]
    sems = [occ_df.loc[occ_df["State"] == s, "Occupancy"].sem() for s in classes]
    ax.bar(x, means, yerr=sems, capsize=3, color=[state_colors[s] for s in classes],
           edgecolor="black", linewidth=0.5, alpha=0.85)

    for i, s in enumerate(classes):
        vals = occ_df.loc[occ_df["State"] == s, "Occupancy"].values
        jitter = rng.normal(0, 0.05, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals, color="black", s=18, zorder=5,
                   edgecolor="white", linewidth=0.4)

    ax.set_xticks(x)
    ax.set_xticklabels([display_labels.get(s, s) for s in classes], fontsize=FONTSIZE - 1, rotation=20, ha="right")
    ax.set_ylabel("State occupancy (% of trials)", fontsize=FONTSIZE)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hmm_transition_and_dwell(hmm_result, classes_3, state_colors, display_labels, out_path):
    """Fig 6 panels d ("Transition probability matrix from a Gaussian
    Hidden Markov Model fit to behavioral features...") and e ("Mean
    dwell time per HMM state, derived analytically from self-transition
    probabilities."). No emission-means panel here -- that heatmap is
    Supp Fig 4's real panel c (see plot_hmm_emission_means).
    """
    fig, (axB, axD) = plt.subplots(1, 2, figsize=(6.8, 3.3))
    labels = [display_labels.get(c, c) for c in classes_3]

    trans_mat = pd.DataFrame(hmm_result["transition_matrix"]).T.reindex(index=classes_3, columns=classes_3).values
    im = axB.imshow(trans_mat, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    axB.set_xticks(range(len(classes_3)))
    axB.set_yticks(range(len(classes_3)))
    axB.set_xticklabels(labels, fontsize=FONTSIZE - 1, rotation=20, ha="right")
    axB.set_yticklabels(labels, fontsize=FONTSIZE - 1)
    axB.set_xlabel("To state", fontsize=FONTSIZE)
    axB.set_ylabel("From state", fontsize=FONTSIZE)
    for i in range(len(classes_3)):
        for j in range(len(classes_3)):
            axB.text(j, i, f"{trans_mat[i, j]:.3f}", ha="center", va="center", fontsize=FONTSIZE - 1,
                      color="white" if trans_mat[i, j] > 0.55 else "black")
    plt.colorbar(im, ax=axB, shrink=0.75, label="Transition probability")

    dwells = [hmm_result["mean_dwell_trials"][c] for c in classes_3]
    bar_colors = [state_colors.get(c, "#7f8c8d") for c in classes_3]
    axD.bar(range(len(classes_3)), dwells, color=bar_colors, edgecolor="black", linewidth=0.5, alpha=0.88)
    axD.set_xticks(range(len(classes_3)))
    axD.set_xticklabels(labels, fontsize=FONTSIZE - 1, rotation=20, ha="right")
    axD.set_ylabel("Mean dwell time (trials)", fontsize=FONTSIZE)
    for i, d in enumerate(dwells):
        axD.text(i, d + 0.2, f"{d:.1f}", ha="center", fontsize=FONTSIZE - 1, color="black")
    axD.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_wsls_state_scatter(wsls_state_df, display_labels, out_path):
    """Fig 6 panel f: "Win-stay versus lose-switch probability for
    Exploitation and Exploration trials. Each point represents one
    session colored by rolling accuracy within state (Exploitation:
    teal-to-purple; Exploration: orange-to-purple). Circles indicate
    Exploitation trials; squares indicate Exploration trials."
    `wsls_state_df` comes from
    wsls.win_stay_lose_switch_by_state_session(...), one row per
    (session, state) with win_stay/lose_switch/rolling_accuracy_in_state.
    """
    cmap_exploit = LinearSegmentedColormap.from_list("exploit_grad", ["#1abc9c", "#8e44ad"])
    cmap_explore = LinearSegmentedColormap.from_list("explore_grad", ["#f39c12", "#8e44ad"])

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    cbar_pad = 0.03
    for state, cmap, marker in [("Engaged", cmap_exploit, "o"), ("Exploration", cmap_explore, "s")]:
        sdf = wsls_state_df[wsls_state_df["state"] == state]
        if len(sdf) == 0:
            continue
        sca = ax.scatter(sdf["lose_switch"], sdf["win_stay"], c=sdf["rolling_accuracy_in_state"], cmap=cmap,
                          marker=marker, s=24, edgecolor="black", linewidth=0.3, alpha=0.85,
                          vmin=sdf["rolling_accuracy_in_state"].min(), vmax=sdf["rolling_accuracy_in_state"].max())
        cbar = plt.colorbar(sca, ax=ax, shrink=0.45, pad=cbar_pad)
        cbar.set_label(f"Rolling accuracy\n({display_labels.get(state, state)})", fontsize=FONTSIZE - 2)
        cbar.ax.tick_params(labelsize=FONTSIZE - 2)
        cbar_pad += 0.30

    ax.set_xlabel("Lose-switch probability", fontsize=FONTSIZE)
    ax.set_ylabel("Win-stay probability", fontsize=FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#7f8c8d", markeredgecolor="black",
               label=display_labels.get("Engaged", "Exploitation"), markersize=6),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#7f8c8d", markeredgecolor="black",
               label="Exploration", markersize=6),
    ]
    ax.legend(handles=handles, fontsize=FONTSIZE - 1, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_model_selection(kmeans_result, out_path):
    """Supp Fig 4 panel a: "Silhouette score (left) and Bayesian
    Information Criterion (BIC) and Akaike Information Criterion (AIC)
    scores (right) across K-means cluster solutions from K=2 to K=8."
    A small 1x2 line-plot figure -- NOT the full multi-panel
    publication-figure mega-function this pipeline used to reuse here.
    """
    k_range = kmeans_result["k_range"]
    k_best = kmeans_result["k_used"]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(6.6, 3.0))

    axA.plot(k_range, kmeans_result["silhouette_by_k"], "o-", color="#2c3e50", linewidth=1.5, markersize=5)
    axA.axvline(k_best, color="#e74c3c", ls="--", lw=1, alpha=0.7, label=f"K={k_best} (used)")
    axA.set_xlabel("Number of clusters K", fontsize=FONTSIZE)
    axA.set_ylabel("Silhouette score", fontsize=FONTSIZE)
    axA.legend(fontsize=FONTSIZE - 1, frameon=False)
    axA.spines[["top", "right"]].set_visible(False)

    axB.plot(k_range, kmeans_result["bic_by_k"], "o-", color="#2980b9", linewidth=1.5, markersize=5, label="BIC")
    axB.plot(k_range, kmeans_result["aic_by_k"], "s--", color="#e67e22", linewidth=1.5, markersize=5, label="AIC")
    axB.axvline(k_best, color="#e74c3c", ls="--", lw=1, alpha=0.7, label=f"K={k_best} (used)")
    axB.set_xlabel("Number of clusters K", fontsize=FONTSIZE)
    axB.set_ylabel("Score (lower = better)", fontsize=FONTSIZE)
    axB.legend(fontsize=FONTSIZE - 1, frameon=False)
    axB.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_kmeans_cluster_heatmap(kmeans_result, classes, display_labels, out_path):
    """Supp Fig 4 panel b: "Mean feature values for each of four
    behavioral clusters identified by K-Means clustering (K=4) ...
    Cluster profiles were computed from train-set centroids and
    evaluated on held-out sessions." A single small heatmap, not part of
    the full K-Means publication-figure mega-function.
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    profiles = pd.DataFrame(kmeans_result["cluster_profiles"]).T.reindex(classes)
    feat_labels = ["Signed\nDeviation", "Choice\nDeviation", "Rolling\nAccuracy", "Switch\nrate"]
    sns.heatmap(profiles, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Mean value"},
                xticklabels=feat_labels, yticklabels=[display_labels.get(c, c) for c in classes], ax=ax)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hmm_emission_means(hmm_result, classes_3, display_labels, out_path):
    """Supp Fig 4 panel c: "HMM emission means for each state in the
    three-state solution, confirming the behavioral signatures of Bias
    (high absolute deviation, low accuracy), Exploitation (high accuracy,
    near-zero switching), and Exploration (moderate accuracy, high
    switching), derived independently of the K-Means solution." A single
    small heatmap (this pipeline's HMM uses the same 4-feature set as
    K-Means -- Signed_Deviation/Choice_Deviation/Rolling_Accuracy/
    Rolling_Switch_Rate, see config.yaml's `state_taxonomy.features` --
    so columns are labeled for what they actually are here rather than
    copying the source manuscript's possibly-differently-ordered column
    headers).
    """
    fig, ax = plt.subplots(figsize=(4.4, 3.0))
    means_df = pd.DataFrame(hmm_result["emission_means"]).T.reindex(classes_3)
    feat_labels = ["Signed\nDeviation", "Choice\nDeviation", "Accuracy", "Switch\nRate"]
    sns.heatmap(means_df, annot=True, fmt=".3f", cmap="RdBu_r", center=0, linewidths=0.5,
                cbar_kws={"shrink": 0.8, "label": "Mean value"},
                xticklabels=feat_labels, yticklabels=[display_labels.get(c, c) for c in classes_3], ax=ax)
    ax.tick_params(axis="both", labelsize=FONTSIZE)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel_d_behavioral_metrics(session_df, summary, states_order, state_colors, display_labels, out_path, seed=42):
    """Supp Fig 4 panel d: "Behavioral metrics by state classification
    across seven behavioral states ... (top, left) probability of
    selecting the high probability choice, (top, right) switch rate,
    (bottom, left) probability of selecting the rightward option, (bottom,
    right) choice entropy (bits). Bars indicate the mean across all
    animals; dots represent individual session averages." A 2x2 grid,
    one subplot per metric; `session_df`/`summary` are
    `state_taxonomy_v7.state_behavioral_metrics_by_state`'s outputs.
    """
    rng = np.random.default_rng(seed)
    panels = [
        ("p_high_prob", "P(high-prob choice)", 0, 0),
        ("switch_rate", "Switch rate", 0, 1),
        ("p_right", "P(right)", 1, 0),
        ("entropy_bits", "Choice entropy (bits)", 1, 1),
    ]
    present = set(session_df["Behavioral_State"])
    states = [s for s in states_order if s in present]
    x = np.arange(len(states))

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.5))
    for metric, ylabel, r, c in panels:
        ax = axes[r, c]
        summ = summary[metric].set_index("Behavioral_State")
        means = [summ.loc[s, "mean"] if s in summ.index else np.nan for s in states]
        sems = [summ.loc[s, "sem"] if s in summ.index else np.nan for s in states]
        ax.bar(x, means, yerr=sems, capsize=3, color=[state_colors.get(s, "#888") for s in states],
               edgecolor="black", linewidth=0.5, alpha=0.85, zorder=2)

        for i, s in enumerate(states):
            vals = session_df.loc[session_df["Behavioral_State"] == s, metric].values
            jitter = rng.uniform(-0.18, 0.18, size=len(vals))
            ax.scatter(i + jitter, vals, color="black", s=6, alpha=0.35, linewidth=0, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels([display_labels.get(s, s) for s in states], fontsize=FONTSIZE - 2, rotation=30, ha="right")
        ax.set_ylabel(ylabel, fontsize=FONTSIZE)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
