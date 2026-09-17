"""Panel rendering for Fig 2/Fig 3 panel b (accuracy aligned to trial
position relative to a block reversal) and panel a's representative-session
("exemplar session") trace.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from .style import clean_axes

BLOCK_COLORS = ["#e91e63", "#9b59b6"]  # alternate by block % 2 -- matches the manuscript's own exemplar-session code


def plot_reversal_alignment(summary_df, config, out_path, split_by_type=True):
    """Panel b: p(high-prob choice) by trial position relative to reversal.
    `split_by_type=True` (Fig 2): one line per transition_type (CUED solid
    blue, UNCUED dashed orange). `split_by_type=False` (Fig 3, no cued/
    uncued distinction): a single line.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    if split_by_type:
        styles = {"CUED": ("#3498db", "-"), "UNCUED": ("#e67e22", "--")}
        for ttype, (color, ls) in styles.items():
            d = summary_df[summary_df["transition_type"] == ttype].sort_values("position")
            if len(d) == 0:
                continue
            ax.plot(d["position"], d["mean"], color=color, linestyle=ls, linewidth=1.5, label=ttype.title())
            ax.fill_between(d["position"], d["mean"] - d["sem"], d["mean"] + d["sem"], color=color, alpha=0.2)
        ax.legend(fontsize=fontsize - 1, frameon=False)
    else:
        d = summary_df.sort_values("position")
        ax.plot(d["position"], d["mean"], color="#3498db", linewidth=1.5)
        ax.fill_between(d["position"], d["mean"] - d["sem"], d["mean"] + d["sem"], color="#3498db", alpha=0.2)

    ax.axvline(0, color="gray", linestyle=":", linewidth=lw, alpha=0.5)
    ax.set_xlabel("Trial relative to reversal", fontsize=fontsize)
    ax.set_ylabel("p(high-prob choice)", fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    clean_axes(ax)
    ax.tick_params(axis="both", labelsize=fontsize, length=config["plotting"]["tick_length"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_exemplar_session(session_data, accuracy_col, config, out_path, outcome_col="Outcome"):
    """Panel a's data-derived "exemplar session" trace, ported verbatim
    (structure and styling) from the manuscript's OWN exemplar-session
    plotting code (provided directly by the user -- this is not a
    reconstruction/approximation, it is the real source code, adapted only
    to this pipeline's config/style conventions and to accept either
    accuracy column ('Outcome' for Fig 2, 'High_Prob' for Fig 3):

    4 stacked rows: (1) block structure -- alternating 2-color bar by
    Block % 2 with the block number labeled in white on each contiguous
    segment; (2) choice location (raw `PhysicalChoice` value) scatter,
    colored green=rewarded / red=unrewarded, with a black dashed vertical
    line at each block transition; (3) ITI, plotted as a separate
    per-block-colored line (matching row 1's 2-color alternation), same
    transition lines; (4) rolling accuracy over a CENTERED 5-trial window
    (`.rolling(5, min_periods=1, center=True)`, not trailing), also
    per-block-colored, with reference lines at 0.5/0.8 and a legend
    (Rewarded / Unrewarded / Block Transition).

    Note the source's own `ax1.set_xlim(-1, 100)` (shared across all rows
    via sharex) is kept as-is: the real code caps the displayed window at
    the first 100 trials regardless of the chosen session's total length.
    """
    d = session_data.sort_values("Trial").reset_index(drop=True).copy()
    d["continuous_trial"] = range(len(d))
    if d[outcome_col].isna().any():
        d[outcome_col] = d["Reward"].map({"Rewarded": 1, "Unrewarded": 0})
    d["Prev_StartTime"] = d["AbsoluteTrialStartTime"].shift(1)
    d["ITI_calc"] = (d["AbsoluteTrialStartTime"] - d["Prev_StartTime"]) / 1000
    d["rolling_accuracy"] = d[accuracy_col].rolling(window=5, min_periods=1, center=True).mean()

    blocks = sorted(d["Block"].unique())
    animal = d["Animal_Name"].iloc[0]

    fig = plt.figure(figsize=(7, 6))
    gs = fig.add_gridspec(4, 1, hspace=0.4, height_ratios=[0.5, 1, 1.5, 1])

    # Row 1: block structure
    ax1 = fig.add_subplot(gs[0])
    for _, trial in d.iterrows():
        block_color = BLOCK_COLORS[int(trial["Block"]) % 2]
        ax1.barh(0, 1, left=trial["continuous_trial"], height=0.5, color=block_color, edgecolor="none", alpha=0.7)
    for block in blocks:
        block_trials = d[d["Block"] == block]
        start = None
        rows = list(block_trials.iterrows())
        for i, (_, trial) in enumerate(rows):
            if start is None:
                start = trial["continuous_trial"]
            if i < len(rows) - 1:
                next_trial = rows[i + 1][1]
                if next_trial["continuous_trial"] != trial["continuous_trial"] + 1:
                    ax1.text((start + trial["continuous_trial"]) / 2, 0, f"{block}", ha="center", va="center",
                             fontsize=6, fontweight="bold", color="white")
                    start = None
            else:
                ax1.text((start + trial["continuous_trial"]) / 2, 0, f"{block}", ha="center", va="center",
                         fontsize=6, fontweight="bold", color="white")
    ax1.set_xlim(-1, 100)
    ax1.set_ylim(-0.3, 0.3)
    ax1.set_yticks([])
    ax1.set_ylabel("Block", fontsize=7)
    ax1.set_title(f"Exemplar Session: {animal}", fontsize=9, fontweight="bold")
    ax1.spines[["top", "right", "left", "bottom"]].set_visible(False)

    # Row 2: choice location, colored by outcome
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    unique_locs = sorted(d["PhysicalChoice"].dropna().unique())
    loc_to_y = {loc: i for i, loc in enumerate(unique_locs)}
    for _, trial in d.iterrows():
        if pd.notna(trial["PhysicalChoice"]):
            color = "#2ecc71" if trial[outcome_col] == 1 else "#e74c3c"
            ax2.scatter(trial["continuous_trial"], loc_to_y[trial["PhysicalChoice"]], s=40, color=color,
                        alpha=0.7, edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(unique_locs)))
    ax2.set_yticklabels(unique_locs, fontsize=6)
    ax2.set_ylabel("Choice", fontsize=7)
    ax2.set_ylim(-0.5, len(unique_locs) - 0.5)
    ax2.spines[["top", "right"]].set_visible(False)

    for idx in range(len(blocks) - 1):
        next_block = blocks[idx + 1]
        trans_trial = d[d["Block"] == next_block]["continuous_trial"].iloc[0]
        for ax in (ax2,):
            ax.axvline(trans_trial - 0.5, color="black", linestyle="--", linewidth=2, alpha=0.8, zorder=10)

    # Row 3: ITI, colored per-block
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    for idx in range(len(blocks) - 1):
        next_block = blocks[idx + 1]
        trans_trial = d[d["Block"] == next_block]["continuous_trial"].iloc[0]
        ax3.axvline(trans_trial - 0.5, color="black", linestyle="--", linewidth=2, alpha=0.8, zorder=10)
    for idx, block in enumerate(blocks):
        block_trials = d[d["Block"] == block]
        ax3.plot(block_trials["continuous_trial"], block_trials["ITI_calc"], "o-",
                  color=BLOCK_COLORS[idx % 2], linewidth=1.5, markersize=4, alpha=0.7)
    ax3.set_ylabel("ITI (seconds)", fontsize=7)
    finite_iti = d["ITI_calc"].dropna()
    ax3.set_ylim(0, min(finite_iti.max() * 1.2, 60) if len(finite_iti) else 30)
    ax3.spines[["top", "right"]].set_visible(False)
    ax3.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    # Row 4: rolling accuracy (centered 5-trial window), colored per-block
    ax4 = fig.add_subplot(gs[3], sharex=ax1)
    for idx in range(len(blocks) - 1):
        next_block = blocks[idx + 1]
        trans_trial = d[d["Block"] == next_block]["continuous_trial"].iloc[0]
        ax4.axvline(trans_trial - 0.5, color="black", linestyle="--", linewidth=2, alpha=0.5, zorder=1)
    for idx, block in enumerate(blocks):
        block_trials = d[d["Block"] == block]
        ax4.plot(block_trials["continuous_trial"], block_trials["rolling_accuracy"], "o-",
                  color=BLOCK_COLORS[idx % 2], linewidth=2, markersize=4, alpha=0.8, zorder=5)
    ax4.axhline(0.5, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
    ax4.axhline(0.8, color="gray", linestyle=":", linewidth=0.5, alpha=0.5)
    ax4.set_ylabel("Accuracy\n(5-trial avg)", fontsize=7)
    ax4.set_xlabel("Trial Number", fontsize=8)
    ax4.set_ylim(0, 1.05)
    ax4.spines[["top", "right"]].set_visible(False)
    ax4.grid(axis="y", alpha=0.3, linestyle=":", linewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor="#2ecc71", label="Rewarded", alpha=0.7, edgecolor="black"),
        mpatches.Patch(facecolor="#e74c3c", label="Unrewarded", alpha=0.7, edgecolor="black"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label="Block Transition"),
    ]
    ax4.legend(handles=legend_elements, fontsize=7, frameon=False, loc="upper center",
               bbox_to_anchor=(0.5, -0.3), ncol=3)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
