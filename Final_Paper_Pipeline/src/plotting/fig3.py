"""Panel rendering for Figure 3 (2-armed bandit acquisition).

Ported to match the ACTUAL rendered-figure code in the source notebook
(`_source_archive/by_figure/Fig3/06_Fig3_2ABT_v2.ipynb`, lines 1-1487 only
-- everything past that belongs to later figures and is out of scope).
Source cell line numbers cited per panel below. As in fig2.py, this only
changes rendered SVGs -- every function's returned stats dict (and
therefore the values manifest) is byte-identical to the previous version.

Same caveat as fig2.py: the qualitative exemplar-session schematic (panel
a) is not ported -- out of scope for this visual-fidelity pass.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from .style import clean_axes
from ..features.history_encoding import sort_animals
from ..behavior_metrics.wsls import summarize_by_animal


def _tick_params(ax, config):
    p = config["plotting"]
    ax.tick_params(axis="both", labelsize=p["fontsize"], length=p["tick_length"])


def plot_trials_to_criterion(ttc_clean, config, out_path):
    """Per-animal bars + jittered points (06, In[29] / lines 1226-1280)."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    data = ttc_clean["trials_to_criterion"]
    animals = sort_animals(ttc_clean["animal"].unique().tolist(), animal_order)
    summary = {}
    for animal in animals:
        animal_data = ttc_clean[ttc_clean["animal"] == animal]["trials_to_criterion"]
        summary[animal] = {"mean": animal_data.mean(), "sem": scipy_stats.sem(animal_data), "n": len(animal_data)}

    rng = np.random.default_rng(config["random_seed"])
    x_pos = np.arange(len(animals))
    width = 0.6

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(x_pos, [summary[a]["mean"] for a in animals], width,
           color=[animal_colors[a] for a in animals], alpha=0.7,
           edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, [summary[a]["mean"] for a in animals], yerr=[summary[a]["sem"] for a in animals],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    for idx, animal in enumerate(animals):
        animal_data = ttc_clean[ttc_clean["animal"] == animal]
        if len(animal_data) > 0:
            x_jitter = idx + rng.uniform(-0.15, 0.15, size=len(animal_data))
            ax.scatter(x_jitter, animal_data["trials_to_criterion"], s=15, color=animal_colors[animal],
                       alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("Trials to Criterion", fontsize=fontsize + 1)
    ax.set_xlabel("Animal", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {"mean": data.mean(), "sem": scipy_stats.sem(data), "n": len(data)}


def plot_late_accuracy(late_acc_df, config, out_path):
    """Per-animal bars + jittered points (06, In[39] / lines 1447-1487)."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    data = late_acc_df["late_accuracy"].dropna()
    animals = sort_animals(late_acc_df["animal"].unique().tolist(), animal_order)
    summary = {}
    for animal in animals:
        animal_data = late_acc_df[late_acc_df["animal"] == animal]["late_accuracy"].dropna()
        summary[animal] = {"mean": animal_data.mean(), "sem": scipy_stats.sem(animal_data), "n": len(animal_data)}

    rng = np.random.default_rng(config["random_seed"])
    x_pos = np.arange(len(animals))
    width = 0.6

    fig, ax = plt.subplots(figsize=(3, 2))
    ax.bar(x_pos, [summary[a]["mean"] for a in animals], width,
           color=[animal_colors[a] for a in animals], alpha=0.7,
           edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, [summary[a]["mean"] for a in animals], yerr=[summary[a]["sem"] for a in animals],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    for idx, animal in enumerate(animals):
        animal_data = late_acc_df[late_acc_df["animal"] == animal]
        if len(animal_data) > 0:
            x_jitter = idx + rng.uniform(-0.15, 0.15, size=len(animal_data))
            ax.scatter(x_jitter, animal_data["late_accuracy"], s=15, color=animal_colors[animal],
                       alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)

    ax.axhline(0.8, color="gray", linestyle="--", linewidth=lw, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("Late Accuracy", fontsize=fontsize + 1)
    ax.set_xlabel("Animal", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {"mean": data.mean(), "sem": scipy_stats.sem(data), "n": len(data)}


def plot_wsls(wsls_sessions, pooled_wsls, config, out_path):
    """Per-animal win-stay/lose-shift side-by-side bars (06, In[27] / lines
    1057-1129). Unlike Fig 2's equivalent panel, the source does not filter
    the scatter overlay here -- all sessions are shown.

    `pooled_wsls` still drives the values manifest and is untouched; the
    by-animal breakdown for the plot comes from the already-existing,
    unmodified `summarize_by_animal` in src/behavior_metrics/wsls.py.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    per_animal = summarize_by_animal(wsls_sessions).set_index("animal")
    animals = sort_animals(per_animal.index.tolist(), animal_order)

    rng = np.random.default_rng(config["random_seed"])
    x_pos = np.arange(len(animals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.bar(x_pos - width / 2, per_animal.loc[animals, "win_stay_mean"], width,
           color=[animal_colors[a] for a in animals], alpha=0.7,
           edgecolor="black", linewidth=lw, label="Win-Stay")
    ax.errorbar(x_pos - width / 2, per_animal.loc[animals, "win_stay_mean"],
                yerr=per_animal.loc[animals, "win_stay_sem"],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    ax.bar(x_pos + width / 2, per_animal.loc[animals, "lose_shift_mean"], width,
           color=[animal_colors[a] for a in animals], alpha=0.4,
           edgecolor="black", linewidth=lw, label="Lose-Shift")
    ax.errorbar(x_pos + width / 2, per_animal.loc[animals, "lose_shift_mean"],
                yerr=per_animal.loc[animals, "lose_shift_sem"],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    for idx, animal in enumerate(animals):
        animal_sessions = wsls_sessions[wsls_sessions["animal"] == animal]
        if len(animal_sessions) > 0:
            x_jitter = (idx - width / 2) + rng.uniform(-0.1, 0.1, size=len(animal_sessions))
            ax.scatter(x_jitter, animal_sessions["win_stay"], s=15, color=animal_colors[animal],
                       alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)
            x_jitter = (idx + width / 2) + rng.uniform(-0.1, 0.1, size=len(animal_sessions))
            ax.scatter(x_jitter, animal_sessions["lose_shift"], s=15, color=animal_colors[animal],
                       alpha=0.4, edgecolor="black", linewidth=0.3, zorder=5)

    ax.axhline(1.0, color="red", linestyle="--", linewidth=lw, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("Probability", fontsize=fontsize + 1)
    ax.set_xlabel("Animal", fontsize=fontsize + 1)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=fontsize, frameon=False, loc="upper right")
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_switch_by_history(per_animal, overall, config, out_path):
    """P(switch) vs. reward history, one line per animal + group average
    (06, In[24] / lines 942-981)."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    history_n = config["fig3"]["history_n"]

    sorted_seqs = overall["reward_seq"].values
    fig, ax = plt.subplots(figsize=(4, 2))

    animals = sort_animals(per_animal["Animal_Name"].unique().tolist(), animal_order)
    for animal in animals:
        animal_data = per_animal[per_animal["Animal_Name"] == animal].set_index(
            "reward_seq").reindex(sorted_seqs).reset_index()
        color = animal_colors.get(animal, "#888888")
        ax.plot(range(len(sorted_seqs)), animal_data["p_switch"], "-", color=color,
                alpha=0.6, linewidth=1.5, markersize=4, label=animal)

    overall_sorted = overall.set_index("reward_seq").reindex(sorted_seqs).reset_index()
    ax.plot(range(len(sorted_seqs)), overall_sorted["p_switch"], "o-", color="black",
            linewidth=lw, markersize=2, label="Group Average", zorder=10)

    ax.set_xticks(range(len(sorted_seqs)))
    ax.set_xticklabels(sorted_seqs, rotation=45, ha="right", fontsize=fontsize - 1)
    ax.set_xlabel(f"Reward History (Last {history_n} Trials)", fontsize=fontsize + 1)
    ax.set_ylabel("P(Switch Choice)", fontsize=fontsize + 1)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.legend(fontsize=fontsize - 1, frameon=False, bbox_to_anchor=(1.05, 1), loc="upper left")
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
