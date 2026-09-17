"""Panel rendering for Figure 2 (reversal learning: cued vs. uncued).

These four panels are ported to match, as closely as possible, the ACTUAL
rendered-figure code in the source notebook
(`_source_archive/by_figure/Fig2/05_Fig2_Reversal_v2.ipynb`) -- not just the
statistics, but the literal layout/style: figure sizes, colors, per-animal
breakdowns, error bars, jittered scatter overlays, reference lines,
significance brackets, legends and axis text. Source cell line numbers are
cited per panel below. None of this changes any computed value -- the
values returned by each function (and therefore the values manifest) are
byte-identical to the previous version of this module; only the rendered
SVGs differ.

NOTE: the qualitative "exemplar session" schematic panel (panel a in the
manuscript) is still NOT ported here -- it's a custom trial-by-trial
illustration built from raw per-trial arrays the current pipeline stage
doesn't load for this figure, and is out of scope for this pass (which is
about matching the visual style of the four already-computed quantitative
panels, not adding new ones).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats
from scipy.stats import mannwhitneyu

from .style import clean_axes
from ..features.history_encoding import sort_animals
from ..behavior_metrics.wsls import summarize_by_animal


def _tick_params(ax, config):
    p = config["plotting"]
    ax.tick_params(axis="both", labelsize=p["fontsize"], length=p["tick_length"])


def plot_trials_to_criterion(ttc_clean, config, out_path):
    """Cued vs. uncued violin plot (05, In[22] / lines 1021-1072): violin +
    jittered points + explicit black mean bars + a significance bracket."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]

    cued = ttc_clean[ttc_clean["transition_type"] == "CUED"]["trials_to_criterion"]
    uncued = ttc_clean[ttc_clean["transition_type"] == "UNCUED"]["trials_to_criterion"]
    u_stat, p_val = mannwhitneyu(cued, uncued, alternative="less")

    rng = np.random.default_rng(config["random_seed"])

    fig, ax = plt.subplots(figsize=(2, 2))
    parts = ax.violinplot([cued.values, uncued.values], positions=[0, 1],
                           widths=0.6, showmeans=True, showmedians=True)
    for pc, color in zip(parts["bodies"], ["#3498db", "#e74c3c"]):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    for i, (data, color) in enumerate([(cued, "#3498db"), (uncued, "#e74c3c")]):
        x = rng.normal(i, 0.04, size=len(data))
        ax.scatter(x, data, alpha=0.4, s=10, color=color, edgecolor="black", linewidth=0.3)

    # Explicit black mean bars (05, line 1044) -- drawn on top of the
    # violin's own showmeans line, matching the source exactly.
    means = [cued.mean(), uncued.mean()]
    ax.hlines(means, [-0.2, 0.8], [0.2, 1.2], colors="black", linewidth=2, label="Mean")

    # Significance bracket (05, lines 1047-1058)
    if p_val < 0.05:
        y_max = max(cued.max(), uncued.max()) + 2
        ax.plot([0, 0, 1, 1], [y_max, y_max + 1, y_max + 1, y_max], "k-", linewidth=1)
        if p_val < 0.001:
            sig_text = "***"
        elif p_val < 0.01:
            sig_text = "**"
        else:
            sig_text = "*"
        ax.text(0.5, y_max + 1.5, sig_text, ha="center", fontsize=fontsize + 2, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["cued", "uncued"], fontsize=fontsize + 1)
    ax.set_ylabel("Trials to Criterion", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {"u_stat": u_stat, "p_value": p_val,
            "cued_mean": cued.mean(), "cued_median": cued.median(), "cued_n": len(cued),
            "uncued_mean": uncued.mean(), "uncued_median": uncued.median(), "uncued_n": len(uncued)}


def plot_late_accuracy(late_acc_df, config, out_path):
    """Per-animal cued-vs-uncued side-by-side bars (05, In[27] / lines
    1484-1569). The manuscript's reported number (and this function's
    returned stats) are still the pooled mean and the animal-paired
    Wilcoxon test -- see the correctness note below -- only the rendered
    panel now matches the source's by-animal breakdown."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    cued = late_acc_df[late_acc_df["transition_type"] == "CUED"]["late_accuracy"].dropna()
    uncued = late_acc_df[late_acc_df["transition_type"] == "UNCUED"]["late_accuracy"].dropna()

    # Paired test at the ANIMAL level (n=animals, not n=blocks): each animal
    # contributes one cued mean and one uncued mean. Pairing raw per-block
    # values positionally (as an earlier version of this function did) is
    # not a valid paired comparison when the two groups have different
    # block counts -- fixed here rather than left in, since this is a
    # correctness bug in this pipeline's own code, independent of whatever
    # the manuscript reports.
    per_animal_paired = late_acc_df.groupby(["animal", "transition_type"])["late_accuracy"].mean().unstack()
    paired = (per_animal_paired.dropna(subset=["CUED", "UNCUED"])
              if {"CUED", "UNCUED"}.issubset(per_animal_paired.columns) else per_animal_paired.iloc[0:0])
    if len(paired) > 0:
        w_stat, p_val = scipy_stats.wilcoxon(paired["CUED"], paired["UNCUED"])
    else:
        w_stat, p_val = np.nan, np.nan

    # Per-animal summary for the PLOT ONLY (05, lines 1484-1502): mirrors
    # the source's own per-animal cued/uncued mean+SEM table, including its
    # SEM-falls-back-to-0-for-n<=1 convention (display-only; does not feed
    # the pooled stats returned below).
    animals = sort_animals(late_acc_df["animal"].unique().tolist(), animal_order)
    rows = []
    for animal in animals:
        animal_data = late_acc_df[late_acc_df["animal"] == animal]
        c = animal_data[animal_data["transition_type"] == "CUED"]["late_accuracy"]
        u = animal_data[animal_data["transition_type"] == "UNCUED"]["late_accuracy"]
        rows.append({
            "animal": animal,
            "cued_mean": c.mean() if len(c) > 0 else np.nan,
            "cued_sem": scipy_stats.sem(c) if len(c) > 1 else 0,
            "uncued_mean": u.mean() if len(u) > 0 else np.nan,
            "uncued_sem": scipy_stats.sem(u) if len(u) > 1 else 0,
        })
    summary = {r["animal"]: r for r in rows}

    rng = np.random.default_rng(config["random_seed"])
    x_pos = np.arange(len(animals))
    width = 0.35

    fig, ax = plt.subplots(figsize=(3, 2))

    ax.bar(x_pos - width / 2, [summary[a]["cued_mean"] for a in animals], width,
           color=[animal_colors[a] for a in animals], alpha=0.7,
           edgecolor="black", linewidth=lw, label="Cued")
    ax.errorbar(x_pos - width / 2, [summary[a]["cued_mean"] for a in animals],
                yerr=[summary[a]["cued_sem"] for a in animals],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    ax.bar(x_pos + width / 2, [summary[a]["uncued_mean"] for a in animals], width,
           color=[animal_colors[a] for a in animals], alpha=0.4,
           edgecolor="black", linewidth=lw, label="Uncued")
    ax.errorbar(x_pos + width / 2, [summary[a]["uncued_mean"] for a in animals],
                yerr=[summary[a]["uncued_sem"] for a in animals],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)

    for idx, animal in enumerate(animals):
        animal_cued = late_acc_df[(late_acc_df["animal"] == animal) & (late_acc_df["transition_type"] == "CUED")]
        if len(animal_cued) > 0:
            x_jitter = (idx - width / 2) + rng.uniform(-0.1, 0.1, size=len(animal_cued))
            ax.scatter(x_jitter, animal_cued["late_accuracy"], s=15, color=animal_colors[animal],
                       alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)
        animal_uncued = late_acc_df[(late_acc_df["animal"] == animal) & (late_acc_df["transition_type"] == "UNCUED")]
        if len(animal_uncued) > 0:
            x_jitter = (idx + width / 2) + rng.uniform(-0.1, 0.1, size=len(animal_uncued))
            ax.scatter(x_jitter, animal_uncued["late_accuracy"], s=15, color=animal_colors[animal],
                       alpha=0.4, edgecolor="black", linewidth=0.3, zorder=5)

    ax.axhline(0.9, color="gray", linestyle="--", linewidth=lw, alpha=0.5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(animals, rotation=45, ha="right")
    ax.set_ylabel("late accuracy", fontsize=fontsize + 1)
    ax.set_xlabel("animal", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close(fig)

    return {"cued_mean": cued.mean(), "cued_sem": scipy_stats.sem(cued), "cued_n": len(cued),
            "uncued_mean": uncued.mean(), "uncued_sem": scipy_stats.sem(uncued), "uncued_n": len(uncued),
            "wilcoxon_stat": w_stat, "p_value": p_val}


def plot_wsls(wsls_sessions, pooled_wsls, config, out_path):
    """Per-animal win-stay/lose-shift side-by-side bars (05, In[127] / lines
    2244-2320 -- the LATER of two cells writing 'WinStay_LoseShift_*.svg',
    so the one that wins per this pipeline's own already-established
    later-cell-wins convention). The scatter overlay only (not the bar
    heights) additionally excludes low-trial-count sessions, matching the
    source's own display-only filter at line 2344-2348.

    `pooled_wsls` (from summarize_pooled) still drives the values manifest
    and is untouched here; the by-animal breakdown used for the plot comes
    from the already-existing, unmodified `summarize_by_animal` in
    src/behavior_metrics/wsls.py.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    min_trials = config["fig2"]["wsls_scatter_min_trials"]

    per_animal = summarize_by_animal(wsls_sessions).set_index("animal")
    animals = sort_animals(per_animal.index.tolist(), animal_order)

    sessions_filtered = wsls_sessions[
        (wsls_sessions["n_win"] >= min_trials) & (wsls_sessions["n_lose"] >= min_trials)
    ]

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
        animal_sessions = sessions_filtered[sessions_filtered["animal"] == animal]
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
    (05, In[103] / lines 1952-1991)."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    history_n = config["fig2"]["history_n"]

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
