"""Panel rendering for Supplementary Figure 3 (Q-following validation:
P(follows Q-values) by animal, by trial-in-block, by previous outcome).

Source: _source_archive/by_figure/SuppFig3/SuppFig3_candidate_07_GLM_HMM_v1_cleaned.ipynb,
the single cell saving `QValue_Following_Session_Level.svg`
(`fig, axes = plt.subplots(1, 3, figsize=(8, 3))`).

Notable fidelity points ported from that cell (not merely approximated):
- Panel A's animal order is `session_animal['Animal_Name'].unique()`, i.e.
  whatever order falls out of pandas' (alphabetically-sorted-by-default)
  groupby -- NOT `plotting.animal_order`'s canonical left-to-right list used
  elsewhere in this pipeline. Reproduced verbatim rather than "fixed", per
  the pixel-fidelity requirement.
- Panel B plots each animal's trace with `label=animal` and ends with
  `ax.legend(fontsize=FONTSIZE-1, frameon=False)` -- a 10-entry animal-name
  legend is part of the published panel.
- Panel C draws a faint gray line connecting each session's paired
  Rewarded/Unrewarded points (`color='gray', linewidth=0.5, alpha=0.2`).
- All three panels set explicit y-ticks at [0.25, 0.5, 0.75, 1.0].
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from .style import clean_axes
from ..features.history_encoding import sort_animals


def plot_q_following(by_animal, by_trial_in_block, by_outcome, config, out_path):
    # This source cell's own local FONTSIZE/TICK_LENGTH (11/4) differ from
    # the shared pipeline defaults (8/2) used elsewhere -- see config.yaml.
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]

    fig, axes = plt.subplots(1, 3, figsize=(8, 3))

    # Panel A: by animal
    ax = axes[0]
    session_animal = by_animal["session_animal"]
    # Matches source exactly: plain `.unique()` order (alphabetical, a side
    # effect of the upstream groupby), not the canonical animal_order.
    animals = list(session_animal["Animal_Name"].unique())
    for i, animal in enumerate(animals):
        sdf = session_animal[session_animal["Animal_Name"] == animal]
        mean = sdf["Chose_Higher_Q"].mean()
        sem = sdf["Chose_Higher_Q"].sem()
        color = animal_colors.get(animal, "gray")
        ax.scatter(np.random.normal(i, 0.07, size=len(sdf)), sdf["Chose_Higher_Q"].values,
                   color=color, alpha=0.4, s=20, zorder=3)
        ax.errorbar(i, mean, yerr=sem, fmt="o", color=color, markersize=8, capsize=4,
                    linewidth=lw * 1.5, markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks(range(len(animals)))
    ax.set_xticklabels(animals, fontsize=fontsize, rotation=45)
    ax.set_ylabel("p(follows Q-values)", fontsize=fontsize + 1)
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)

    # Panel B: by trial in block
    ax = axes[1]
    animal_pos = (
        by_trial_in_block.groupby(["Animal_Name", "Trial_in_Block"])["Chose_Higher_Q"]
        .mean().reset_index()
    )
    pos_mean = by_trial_in_block.groupby("Trial_in_Block")["Chose_Higher_Q"].mean()
    pos_sem = by_trial_in_block.groupby("Trial_in_Block")["Chose_Higher_Q"].sem()
    for animal, sdf in animal_pos.groupby("Animal_Name"):
        sdf = sdf.sort_values("Trial_in_Block")
        ax.plot(sdf["Trial_in_Block"], sdf["Chose_Higher_Q"],
                color=animal_colors.get(animal, "gray"), linewidth=1.2, alpha=0.6, zorder=2,
                label=animal)
    ax.fill_between(pos_mean.index, pos_mean.values - pos_sem.values, pos_mean.values + pos_sem.values,
                     alpha=0.25, color="black")
    ax.plot(pos_mean.index, pos_mean.values, "o-", color="black", linewidth=lw * 1.5, markersize=5, zorder=5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Trial in Block", fontsize=fontsize + 1)
    ax.set_ylabel("p(follows Q-values)", fontsize=fontsize + 1)
    ax.set_ylim(0.25, 1.0)
    ax.set_xlim(-0.5, 20)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)
    ax.legend(fontsize=fontsize - 1, frameon=False)

    # Panel C: by previous outcome
    ax = axes[2]
    session_outcome = by_outcome["session_outcome"]
    outcome_labels = {0: "Unrewarded", 1: "Rewarded"}
    outcome_colors = {0: "#e74c3c", 1: "#1abc9c"}
    for outcome in [0, 1]:
        sdf = session_outcome[session_outcome["Outcome_Binary"] == outcome]
        mean = sdf["Chose_Higher_Q"].mean()
        sem = sdf["Chose_Higher_Q"].sem()
        ax.scatter(np.random.normal(outcome, 0.07, size=len(sdf)), sdf["Chose_Higher_Q"].values,
                   color=outcome_colors[outcome], alpha=0.4, s=20, zorder=3)
        ax.errorbar(outcome, mean, yerr=sem, fmt="o", color=outcome_colors[outcome], markersize=8,
                    capsize=4, linewidth=lw * 1.5, markeredgecolor="black", markeredgewidth=0.8, zorder=5)
    # Connect session-matched (Unrewarded, Rewarded) pairs.
    for session_id, sdf in session_outcome.groupby("Session_ID"):
        if len(sdf) == 2:
            vals = sdf.sort_values("Outcome_Binary")
            ax.plot([0, 1], vals["Chose_Higher_Q"].values, color="gray", linewidth=0.5, alpha=0.2, zorder=2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([outcome_labels[0], outcome_labels[1]], fontsize=fontsize)
    ax.set_ylabel("p(follows Q-values)", fontsize=fontsize + 1)
    ax.set_ylim(0.25, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_observed_vs_predicted_adherence(result, config, out_path):
    """Panel b: observed vs. policy-predicted adherence rate, one dot per
    animal, dashed identity line."""
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]
    animal_colors = config["plotting"]["animal_colors"]

    fig, ax = plt.subplots(figsize=(3, 3))
    for animal in result["animal_observed"]:
        ax.scatter(result["animal_predicted"][animal], result["animal_observed"][animal],
                   color=animal_colors.get(animal, "gray"), s=60, edgecolor="black",
                   linewidth=0.6, zorder=5, label=animal)

    lims = [0.4, 1.0]
    ax.plot(lims, lims, "--", color="gray", linewidth=1, zorder=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Policy-predicted adherence rate", fontsize=fontsize)
    ax.set_ylabel("Observed adherence rate", fontsize=fontsize)
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)
    ax.legend(fontsize=fontsize - 2, frameon=False, loc="lower right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_adherence_diff_by_qdiff(binned_df, config, out_path):
    """Panel c: observed-minus-predicted adherence as a function of
    |Q_right - Q_left|, split by whether the previous trial was rewarded
    (blue) or not (red), with SEM shading."""
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]
    colors = {1: ("#1f77b4", "after reward"), 0: ("#e74c3c", "after loss")}

    fig, ax = plt.subplots(figsize=(3.5, 3))
    for outcome, (color, label) in colors.items():
        d = binned_df[binned_df["prev_outcome"] == outcome].sort_values("bin_center")
        ax.plot(d["bin_center"], d["mean"], color=color, linewidth=1.5, label=label)
        ax.fill_between(d["bin_center"], d["mean"] - d["sem"], d["mean"] + d["sem"],
                         color=color, alpha=0.25)

    ax.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel(r"|$Q_{right} - Q_{left}$|", fontsize=fontsize)
    ax.set_ylabel("observed - predicted adherence", fontsize=fontsize)
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)
    ax.legend(fontsize=fontsize - 1, frameon=False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violation_direction(result, config, out_path):
    """Panel d: direction of Q-value violations by animal -- stacked bar of
    proportion left (opaque) vs. right (transparent), population + per
    animal."""
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    animals = sort_animals(list(result["per_animal_prop_right"].keys()), animal_order)
    labels = ["Population"] + animals
    prop_right = [result["population_prop_right"]] + [result["per_animal_prop_right"][a] for a in animals]
    prop_left = [result["population_prop_left"]] + [result["per_animal_prop_left"][a] for a in animals]
    bar_colors = ["gray"] + [animal_colors.get(a, "gray") for a in animals]

    fig, ax = plt.subplots(figsize=(4.5, 3))
    x = np.arange(len(labels))
    ax.bar(x, prop_left, color=bar_colors, alpha=1.0, edgecolor="black", linewidth=0.5, label="Left")
    ax.bar(x, prop_right, bottom=prop_left, color=bar_colors, alpha=0.35, edgecolor="black",
           linewidth=0.5, label="Right")
    for i, (pl, pr) in enumerate(zip(prop_left, prop_right)):
        ax.text(i, pl / 2, f"{pl:.2f}", ha="center", va="center", fontsize=fontsize - 2)
        ax.text(i, pl + pr / 2, f"{pr:.2f}", ha="center", va="center", fontsize=fontsize - 2)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=fontsize, rotation=45, ha="right")
    ax.set_ylabel("Proportion of violations", fontsize=fontsize)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)
    ax.legend(fontsize=fontsize - 1, frameon=False, loc="upper right")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_paired_violation_comparison(result, ylabel, config, out_path):
    """Shared renderer for panels e/f: paired per-session dots+lines
    comparing a quantity after a Q-violation vs. after adherence, with the
    across-animal mean +/- SEM as open circles."""
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]

    pairs = result["session_pairs"]
    after_violation = [r["after_violation"] for r in pairs]
    after_adherence = [r["after_adherence"] for r in pairs]

    fig, ax = plt.subplots(figsize=(2.5, 3))
    for av, aa in zip(after_violation, after_adherence):
        ax.plot([0, 1], [av, aa], color="gray", linewidth=0.4, alpha=0.25, zorder=2)
    ax.scatter(np.zeros(len(after_violation)), after_violation, color="#e74c3c", alpha=0.5, s=18, zorder=3)
    ax.scatter(np.ones(len(after_adherence)), after_adherence, color="#2ecc71", alpha=0.5, s=18, zorder=3)

    ax.errorbar(0, result["after_violation_mean"], yerr=result["after_violation_sem"],
                fmt="o", markerfacecolor="none", markeredgecolor="black", color="black",
                markersize=9, capsize=4, linewidth=1.2, zorder=5)
    ax.errorbar(1, result["after_adherence_mean"], yerr=result["after_adherence_sem"],
                fmt="o", markerfacecolor="none", markeredgecolor="black", color="black",
                markersize=9, capsize=4, linewidth=1.2, zorder=5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["after violation", "after adhering"], fontsize=fontsize, rotation=20, ha="right")
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    ax.tick_params(labelsize=fontsize, length=tick_length)
    ax.text(0.5, 0.95, f"per-animal paired t-test p={result['per_animal_ttest_p']:.4g}",
            ha="center", fontsize=fontsize - 2, transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_violation_vs_accuracy_by_animal(results, config, out_path):
    """Panel g: session-level violation rate vs. accuracy (p(High Prob)),
    one panel per animal, dots colored light-to-dark by session date
    (early to late), with a linear regression fit line and its r/p."""
    fontsize = config["plotting"]["suppfig3_fontsize"]
    tick_length = config["plotting"]["suppfig3_tick_length"]
    animal_order = config["plotting"]["animal_order"]

    animals = sort_animals(list(results.keys()), animal_order)
    fig, axes = plt.subplots(1, len(animals), figsize=(3 * len(animals), 3), sharey=True)
    if len(animals) == 1:
        axes = [axes]

    for ax, animal in zip(axes, animals):
        r = results[animal]
        sessions = r["sessions"]
        n = len(sessions)
        order = np.array([s["session_order"] for s in sessions])
        colors = ScalarMappable(norm=Normalize(vmin=0, vmax=max(n - 1, 1)), cmap="Greys").to_rgba(order)
        vr = np.array([s["violation_rate"] for s in sessions])
        acc = np.array([s["p_high_prob"] for s in sessions])
        ax.scatter(vr, acc, color=colors, edgecolor="black", linewidth=0.4, s=30, zorder=3)

        xs = np.linspace(vr.min(), vr.max(), 50) if n > 1 else vr
        ax.plot(xs, r["intercept"] + r["slope"] * xs, "--", color="black", linewidth=1, zorder=2)

        ax.set_title(f"{animal}\nr={r['r']:.2f}, p={r['p']:.3g}", fontsize=fontsize)
        ax.set_xlabel("violation rate", fontsize=fontsize)
        clean_axes(ax)
        ax.tick_params(labelsize=fontsize, length=tick_length)
    axes[0].set_ylabel("p(High Prob)", fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
