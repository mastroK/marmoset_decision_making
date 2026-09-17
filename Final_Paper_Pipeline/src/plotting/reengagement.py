"""Panel rendering for Figure 1, stage 3 (panels f, h, i: post-disengagement
accuracy, within-block bout start/end accuracy, post-transition
re-engagement speed).

Values come from the source notebook's "STATS EXTRACTION" section (STAT 2,
5, 6, 7), which is the authoritative spec for the NUMBERS (see
behavior_metrics/reengagement.py's docstring) -- but that section is purely
numeric (no plt/savefig calls at all), so it cannot be the source for these
panels' VISUAL styling. Styling below is ported from wherever the actual
publication-figure code for each panel lives instead:

- plot_disengagement_accuracy: no dedicated source cell exists for this
  exact panel; it's the same >60s ITI data point already shown as the last
  bin of panel f's ITI-binned-accuracy plot (`iti.performance_by_iti_bin`),
  just computed independently (per-animal Wilcoxon vs. chance) as a
  cross-check -- kept as a manifest-only statistic, not its own lettered
  manuscript panel.
- plot_representative_session (panel g): ported VERBATIM from the
  manuscript's own "CELL 16: Single Session Schematic - Bout Structure"
  code (provided directly by the user, not reconstructed) -- see its own
  docstring for the exact structure (bout-gradient-colored bars, dual
  block-change/ITI-break boundary lines, centered rolling accuracy).
- plot_reengagement_composite (panels h, i): built directly from the
  manuscript's own printed figure legend (no source-notebook plotting
  cell matches this one) -- see config.yaml's
  reengagement_line_max_trials for the exact legend-derived parameter.
  Each panel is a 3-part composite: a line plot of accuracy across the
  first N trials of a bout (one line per bout rank), a bar plot of
  start-vs-end accuracy per bout rank, and a boxplot of bout lengths per
  rank -- shared by both panels, applied to within-block bouts (h) and
  block-transition bouts (i) respectively.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats as scipy_stats

from .style import clean_axes
from ..features.history_encoding import sort_animals


def plot_disengagement_accuracy(df_high_iti_outcome, config, out_path):
    """No dedicated source cell exists for this panel (see module
    docstring) -- rendered as a per-animal bar + jittered-scatter plot in
    the notebook's own established style for single summary statistics
    (CELL 7 / CELL 8's template), using the same "coral" color the source
    uses for the long-ITI/disengaged condition in CELL 11's Panel D.

    `df_high_iti_outcome` here is the trial-level Series as before (no
    animal breakdown available at that call site), so unlike the CELL 7/8
    template this remains a single bar; n= is still annotated to match the
    source's own sample-size-labeling convention (CELL 11, lines 1294-1296).
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    data = df_high_iti_outcome

    fig, ax = plt.subplots(figsize=(2, 3))
    mean, sem = data.mean(), scipy_stats.sem(data)
    ax.bar([0], [mean], yerr=[sem], color="coral",
           alpha=0.7, edgecolor="black", linewidth=lw, capsize=3)
    ax.text(0, mean + sem + 0.02, f"n={len(data):,}", ha="center", fontsize=fontsize - 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.set_xticks([0])
    ax.set_xticklabels(["ITI > 60s"], fontsize=fontsize)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


_BOUT_RANK_COLORS = {1: "#3498db", 2: "#e67e22", 3: "#2ecc71"}


def plot_reengagement_composite(position_df, start_end_df, bout_df, bout_ranks, title, config, out_path):
    """3-part composite matching the manuscript's actual panels h/i exactly:
    (left) line plot of performance across the first N trials of each bout,
    one line per bout rank; (middle) bar plot of aggregate performance at
    the start (first `start_end_window` trials) and end (last
    `start_end_window`+ trials) of each bout rank; (right) boxplot of bout
    lengths per rank. Shared by panel h (within-block bouts) and panel i
    (block-transition bouts) -- same layout, different bout population.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]

    fig, (ax_line, ax_bar, ax_box) = plt.subplots(1, 3, figsize=(9, 3))

    # Left: performance across first N trials, one line per bout rank
    for rank in bout_ranks:
        d = position_df[position_df["bout_rank"] == rank]
        color = _BOUT_RANK_COLORS.get(rank, "gray")
        ax_line.plot(d["trial_in_bout"], d["mean"], color=color, linewidth=1.5, label=f"bout {rank}")
        ax_line.fill_between(d["trial_in_bout"], d["mean"] - d["sem"], d["mean"] + d["sem"], color=color, alpha=0.2)
    ax_line.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax_line.set_xlabel("Trial in bout", fontsize=fontsize)
    ax_line.set_ylabel("Accuracy", fontsize=fontsize)
    ax_line.set_ylim(0, 1.05)
    ax_line.legend(fontsize=fontsize - 1, frameon=False)
    clean_axes(ax_line)

    # Middle: aggregate start vs. end accuracy per bout rank (existing template)
    x_pos = np.arange(len(start_end_df))
    width = 0.35
    ax_bar.bar(x_pos - width / 2, start_end_df["start_mean"], width, yerr=start_end_df["start_sem"],
               color="#3498db", alpha=0.7, edgecolor="black", linewidth=lw, label="first 3 trials", capsize=3)
    ax_bar.bar(x_pos + width / 2, start_end_df["end_mean"], width, yerr=start_end_df["end_sem"],
               color="#e74c3c", alpha=0.7, edgecolor="black", linewidth=lw, label="last 3+ trials", capsize=3)
    ax_bar.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"bout {r}" for r in start_end_df["bout_rank"]])
    ax_bar.set_ylabel("Accuracy", fontsize=fontsize)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.legend(fontsize=fontsize - 2, frameon=False)
    clean_axes(ax_bar)

    # Right: boxplot of bout lengths (n_trials) per rank
    box_data = [bout_df[bout_df["bout_rank"] == r]["n_trials"].values for r in bout_ranks]
    bp = ax_box.boxplot(box_data, positions=range(len(bout_ranks)), patch_artist=True, widths=0.6,
                         tick_labels=[f"bout {r}" for r in bout_ranks], showmeans=True)
    for patch, rank in zip(bp["boxes"], bout_ranks):
        patch.set_facecolor(_BOUT_RANK_COLORS.get(rank, "gray"))
        patch.set_alpha(0.6)
    ax_box.set_ylabel("Bout length (# trials)", fontsize=fontsize)
    clean_axes(ax_box)

    for ax in (ax_line, ax_bar, ax_box):
        ax.tick_params(axis="both", labelsize=fontsize, length=config["plotting"]["tick_length"])
    fig.suptitle(title, fontsize=fontsize + 1, fontweight="bold")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_representative_session(session_df, bout_iti_threshold, config, out_path):
    """Panel g: representative session showing bout structure, ITI, and
    rolling accuracy. Ported verbatim (structure and styling) from the
    manuscript's OWN "CELL 16: Single Session Schematic - Bout Structure"
    code (provided directly by the user -- not a reconstruction), adapted
    only to this pipeline's existing `_add_bout_ids`-derived `session_df`
    (which already carries ITI/bout_num/block_change per trial) in place
    of the source's separately-built `df_multi_bout`.

    3 stacked rows: (1) bout structure -- each bout is a bar colored along
    a continuous blue-to-red gradient by its position among this session's
    bouts, labeled "B{n}"; (2) ITI, colored the same per-bout gradient,
    with a RED DASHED vertical line at each pure-ITI-triggered bout
    boundary and a BLACK SOLID vertical line at each boundary that is also
    a real block change (both are already bout boundaries by construction
    -- this just distinguishes which trigger caused each one), plus a
    horizontal red dashed line at `bout_iti_threshold`; (3) rolling
    accuracy over a CENTERED 5-trial window, same per-bout gradient and
    bout-boundary lines, horizontal gray dashed line at 0.5.
    """
    d = session_df.reset_index(drop=True).copy()
    d["continuous_trial"] = range(len(d))
    d["rolling_accuracy"] = d["Outcome"].rolling(window=5, min_periods=1, center=True).mean()

    # Local 1-indexed bout rank for this session only (gradient position),
    # independent of any other bout numbering used elsewhere in the pipeline.
    bout_order = {b: i + 1 for i, b in enumerate(d["bout_num"].unique())}
    d["Bout_Number"] = d["bout_num"].map(bout_order)
    n_bouts = d["Bout_Number"].max()
    animal = d["Animal_Name"].iloc[0]

    bout_cmap = LinearSegmentedColormap.from_list("bout_gradient", ["#3498db", "#e74c3c"])

    fig = plt.figure(figsize=(7, 5))
    gs = fig.add_gridspec(3, 1, hspace=0.4, height_ratios=[0.5, 1, 1])

    # Row 1: bout structure
    ax1 = fig.add_subplot(gs[0])
    for bout_num in sorted(d["Bout_Number"].unique()):
        bout_trials = d[d["Bout_Number"] == bout_num]
        start, end = bout_trials["continuous_trial"].min(), bout_trials["continuous_trial"].max()
        color = bout_cmap((bout_num - 1) / max(n_bouts - 1, 1))
        ax1.barh(0, end - start + 1, left=start, height=0.5, color=color, edgecolor="black", linewidth=0.5)
        ax1.text((start + end) / 2, 0, f"B{bout_num}", ha="center", va="center", fontsize=7, fontweight="bold")
    ax1.set_xlim(-1, len(d))
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_yticks([])
    ax1.set_ylabel("Bout ID", fontsize=7)
    ax1.set_title(f"Single Session Schematic: {animal} ({n_bouts} bouts, {len(d)} trials)",
                  fontsize=9, fontweight="bold")
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    block_change_trials = d[d["block_change"]]["continuous_trial"].tolist()

    # Row 2: ITI
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for bout_num in sorted(d["Bout_Number"].unique()):
        if bout_num > 1:
            bout_start = d[d["Bout_Number"] == bout_num]["continuous_trial"].min()
            ax2.axvline(bout_start - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.7, zorder=10)
    for i, trans_trial in enumerate(block_change_trials):
        ax2.axvline(trans_trial - 0.5, color="black", linestyle="-", linewidth=2, alpha=0.8, zorder=5,
                    label="Block Change" if i == 0 else "")
    for bout_num in sorted(d["Bout_Number"].unique()):
        bout_trials = d[d["Bout_Number"] == bout_num]
        color = bout_cmap((bout_num - 1) / max(n_bouts - 1, 1))
        ax2.plot(bout_trials["continuous_trial"], bout_trials["ITI"], "o-", color=color,
                  linewidth=1.5, markersize=4, alpha=0.7)
    ax2.axhline(bout_iti_threshold, color="red", linestyle="--", linewidth=1, alpha=0.5,
                label=f"Break threshold ({bout_iti_threshold}s)")
    ax2.set_ylabel("ITI (seconds)", fontsize=7)
    finite_iti = d["ITI"].dropna()
    ax2.set_ylim(0, min(finite_iti.max() * 1.2, 60) if len(finite_iti) else 30)
    ax2.legend(fontsize=6, frameon=False, loc="upper right")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    # Row 3: rolling accuracy (centered 5-trial window)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for bout_num in sorted(d["Bout_Number"].unique()):
        bout_trials = d[d["Bout_Number"] == bout_num]
        color = bout_cmap((bout_num - 1) / max(n_bouts - 1, 1))
        ax3.plot(bout_trials["continuous_trial"], bout_trials["rolling_accuracy"], "o-", color=color,
                  linewidth=2, markersize=4, alpha=0.8, label=f"Bout {bout_num}")
        if bout_num > 1:
            bout_start = bout_trials["continuous_trial"].min()
            ax3.axvline(bout_start - 0.5, color="red", linestyle="--", linewidth=1.5, alpha=0.5, zorder=1)
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax3.set_ylabel("Accuracy\n(5-trial avg)", fontsize=7)
    ax3.set_xlabel("Trial Number", fontsize=8)
    ax3.set_ylim(0, 1.05)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
