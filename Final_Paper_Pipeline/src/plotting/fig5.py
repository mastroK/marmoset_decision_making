"""Panel rendering for Figure 5 (Q-guided choice: marmosets adhere to
learned Q-values on the majority of trials, with violations occurring
preferentially under conditions of uncertainty).

This figure reuses the SAME per-trial Chose_Higher_Q quantity already fit
and computed for Supp Fig 3 (src/behavior_metrics/q_following_model.py) --
"violation" is 1 - Chose_Higher_Q, the complementary framing of "follows
Q-values". No new model fit or follow/violate classification is done here;
see q_following_model.py's ADDITIVE functions (violation_rate_by_animal,
violation_rate_by_trial_in_block, select_representative_sessions, and the
new paired-t-test fields on q_following_by_outcome) for the data prep this
module consumes.

Panels:
  - plot_fig5a: representative low/high violation-rate sessions for a
    single animal -- top: trial-by-trial Q_right-Q_left with block
    background shaded by which side was high-probability; bottom: choice
    sequence colored by adherence (green) / violation (red).
  - plot_fig5b: violation rate by animal (session dots + per-animal mean +
    overall dashed mean line).
  - plot_fig5c: Q-value adherence rate on unrewarded vs. rewarded trials
    (paired per-session dots + grand means); ported directly from Supp
    Fig 3's Panel C visual style (src/plotting/q_following.py), same
    underlying session_outcome dataframe, just re-labeled and with the
    paired t-test (not Wilcoxon) reported per the manuscript's own stats.
  - plot_fig5d: violation rate vs. trial-in-block, one line per animal +
    black mean line; ported from Supp Fig 3's Panel B visual style, same
    underlying grouping, reframed as violation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

from .style import clean_axes
from ..features.history_encoding import sort_animals


def _shade_blocks(ax, df, colors):
    """Background-shade each contiguous block by which side was the
    high-probability option (Figure 5a legend: "red: right high-prob;
    blue: left high-prob")."""
    for _, bdf in df.groupby("BlockCount"):
        s, e = bdf["Trial"].min(), bdf["Trial"].max()
        color = colors["high_prob_right"] if bdf["High_Prob_Is_Right"].iloc[0] else colors["high_prob_left"]
        ax.axvspan(s - 0.5, e + 0.5, color=color, alpha=0.15, linewidth=0, zorder=0)


def plot_fig5a(low_df, high_df, low_rate, high_rate, config, out_path):
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    colors = config["fig5"]["colors"]

    fig, axes = plt.subplots(2, 2, figsize=(7.5, 3.4), sharex="col",
                              gridspec_kw={"height_ratios": [1.3, 1], "hspace": 0.15, "wspace": 0.3})

    panels = [(low_df, low_rate, "Low Violation"), (high_df, high_rate, "High Violation")]
    for col, (df, rate, title) in enumerate(panels):
        df = df.sort_values("Trial")
        ax_top, ax_bot = axes[0, col], axes[1, col]

        _shade_blocks(ax_top, df, colors)
        ax_top.plot(df["Trial"], df["Q_diff"], color="black", linewidth=lw * 1.2, zorder=3)
        ax_top.axhline(0, color="gray", linestyle=":", linewidth=lw, alpha=0.6, zorder=2)
        ax_top.set_ylim(-1.05, 1.05)
        ax_top.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])
        ax_top.set_title(f"{title} (violation rate={rate:.2f})", fontsize=fontsize, loc="left")
        clean_axes(ax_top)
        ax_top.tick_params(labelsize=fontsize, labelbottom=False, length=config["plotting"]["tick_length"])

        _shade_blocks(ax_bot, df, colors)
        follow = df[df["Chose_Higher_Q"] == 1.0]
        violate = df[df["Chose_Higher_Q"] == 0.0]
        ax_bot.vlines(df["Trial"], 0, df["Choice_Binary"], color="gray", linewidth=0.4, alpha=0.4, zorder=2)
        ax_bot.scatter(follow["Trial"], follow["Choice_Binary"], color=colors["follow"], s=7, zorder=4)
        ax_bot.scatter(violate["Trial"], violate["Choice_Binary"], color=colors["violation"], s=7, zorder=4)
        ax_bot.set_yticks([0, 1])
        ax_bot.set_yticklabels(["Left", "Right"], fontsize=fontsize)
        ax_bot.set_ylim(-0.3, 1.3)
        ax_bot.set_xlabel("trial", fontsize=fontsize + 1)
        clean_axes(ax_bot)
        ax_bot.tick_params(labelsize=fontsize, length=config["plotting"]["tick_length"])

    axes[0, 0].set_ylabel(r"$Q_{right} - Q_{left}$", fontsize=fontsize + 1)
    axes[1, 0].set_ylabel("choice", fontsize=fontsize + 1)

    handles = [
        mlines.Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors["follow"],
                      markeredgecolor="none", markersize=5, label="Follow Q"),
        mlines.Line2D([], [], marker="o", linestyle="none", markerfacecolor=colors["violation"],
                      markeredgecolor="none", markersize=5, label="Violation"),
    ]
    # Figure-level legend placed above both panels so it never overlaps the
    # Q-diff line or choice-sequence dots in either column.
    fig.legend(handles=handles, fontsize=fontsize - 1, frameon=False,
               loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=2)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5b(violation_by_animal, config, out_path):
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    session_animal = violation_by_animal["session_animal"]
    animals = sort_animals(list(session_animal["Animal_Name"].unique()), animal_order)
    overall_mean = violation_by_animal["overall_mean_pooled_sessions"]

    rng = np.random.default_rng(config["random_seed"])
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for i, animal in enumerate(animals):
        sdf = session_animal[session_animal["Animal_Name"] == animal]
        color = animal_colors.get(animal, "gray")
        ax.scatter(rng.normal(i, 0.07, size=len(sdf)), sdf["Violation_Rate"].values,
                   color=color, alpha=0.5, s=18, zorder=3)
        ax.scatter([i], [sdf["Violation_Rate"].mean()], color=color, s=80,
                   edgecolor="black", linewidth=0.8, zorder=5)
    ax.axhline(overall_mean, color="gray", linestyle="--", linewidth=lw, alpha=0.8,
               label=f"Overall mean={overall_mean:.3f}")
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, rotation=45, ha="right", fontsize=fontsize)
    ax.set_ylabel("violation rate", fontsize=fontsize + 1)
    ax.set_ylim(0, max(0.6, session_animal["Violation_Rate"].max() * 1.1))
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="upper right")
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=config["plotting"]["tick_length"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5c(by_outcome, config, out_path):
    """Adherence rate on unrewarded vs. rewarded trials. Ported directly from
    Supp Fig 3's Panel C (src/plotting/q_following.py) visual style, same
    session_outcome dataframe -- relabeled "adherence rate" and with the
    manuscript's paired t-test annotated instead of the Wilcoxon test."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    session_outcome = by_outcome["session_outcome"]
    outcome_labels = {0: "unrewarded", 1: "rewarded"}
    outcome_colors = {0: "#e74c3c", 1: "#1abc9c"}

    rng = np.random.default_rng(config["random_seed"])
    fig, ax = plt.subplots(figsize=(2.8, 2.8))
    for outcome in [0, 1]:
        sdf = session_outcome[session_outcome["Outcome_Binary"] == outcome]
        mean = sdf["Chose_Higher_Q"].mean()
        ax.scatter(rng.normal(outcome, 0.07, size=len(sdf)), sdf["Chose_Higher_Q"].values,
                   color=outcome_colors[outcome], alpha=0.4, s=16, zorder=3)
        ax.scatter([outcome], [mean], color=outcome_colors[outcome], s=80,
                   edgecolor="black", linewidth=0.8, zorder=5)
    for _, sdf in session_outcome.groupby("Session_ID"):
        if len(sdf) == 2:
            vals = sdf.sort_values("Outcome_Binary")
            ax.plot([0, 1], vals["Chose_Higher_Q"].values, color="gray", linewidth=0.5, alpha=0.25, zorder=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([outcome_labels[0], outcome_labels[1]], fontsize=fontsize)
    ax.set_ylabel("adherence rate", fontsize=fontsize + 1)
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=config["plotting"]["tick_length"])
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig5d(violation_by_trial_in_block, config, out_path):
    """Violation rate vs. trial-in-block. Ported from Supp Fig 3's Panel B
    (src/plotting/q_following.py) visual style, same underlying grouping,
    reframed as violation rather than following."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    xlim = config["fig5"]["trial_in_block_xlim"]

    animal_pos = (
        violation_by_trial_in_block.groupby(["Animal_Name", "Trial_in_Block"])["Violation_Rate"]
        .mean().reset_index()
    )
    pos_mean = violation_by_trial_in_block.groupby("Trial_in_Block")["Violation_Rate"].mean()
    animals = sort_animals(list(animal_pos["Animal_Name"].unique()), animal_order)

    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    for animal in animals:
        sdf = animal_pos[animal_pos["Animal_Name"] == animal].sort_values("Trial_in_Block")
        ax.plot(sdf["Trial_in_Block"], sdf["Violation_Rate"], color=animal_colors.get(animal, "gray"),
                linewidth=1.0, alpha=0.7, label=animal, zorder=3)
    ax.plot(pos_mean.index, pos_mean.values, color="black", linewidth=lw * 1.5, label="Mean", zorder=5)
    ax.set_xlabel("trial in block", fontsize=fontsize + 1)
    ax.set_ylabel("violation rate", fontsize=fontsize + 1)
    ax.set_xlim(-0.5, xlim)
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=config["plotting"]["tick_length"])
    ax.legend(fontsize=fontsize - 2, frameon=False, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
