"""Panel rendering for Figure 1, stage 1 (panels b-d: block-position
accuracy, trials-to-criterion, learning-across-sessions).

Ported to match the source notebook's own actual publication-figure code
(_source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb) as closely
as possible -- layout, colors, titles, legends, annotations -- not just the
underlying values (which were already verified separately and are NOT
touched here).

- plot_within_block_learning <- CELL 4 "Plot Learning Curves (Binned) -
  UPDATED TITLE" (lines 269-322), which saves 'Learning_Within_Blocks.svg'.
- plot_trials_to_criterion_and_late_accuracy: panel c. NOT ported from the
  source notebook's own CELL 5-8 (that cell implements a "5 consecutive
  correct" rule and 20-trial late window with block-level dots) -- cross-
  checked against the manuscript's own printed legend for this panel and
  found it describes a different rule entirely (rolling accuracy>=0.8-
  over-5-trials, 10-trial late window, session-level dots). Built to match
  that legend instead (see config.yaml's fig1.criterion_accuracy/window/
  late_accuracy_window and block_learning.py's module docstring). Renders
  as a single figure, top=trials-to-criterion / bottom=late accuracy, both
  per-animal bar+SEM with a per-session jittered-scatter overlay -- the
  same visual template the source notebook uses for its own (different)
  per-animal summary panels, applied here to the corrected statistics.
- plot_session_bins <- the boxplot cell at lines 891-921 (In[17], right
  after CELL 10), which saves 'Session_Bins_Boxplot.svg' -- the only cell
  writing that filename, so unambiguous.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as scipy_stats

from .style import clean_axes
from ..features.history_encoding import sort_animals


def plot_within_block_learning(binned_learning, overall_learning, config, out_path):
    """Ported from CELL 4, lines 269-322."""
    fontsize = config["plotting"]["fontsize"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    cfg = config["fig1"]

    animals = sort_animals(binned_learning["animal"].unique(), animal_order)

    fig, ax = plt.subplots(figsize=(4, 3))

    # Individual animals (faint) -- source line 292-296
    for animal in animals:
        d = binned_learning[binned_learning["animal"] == animal]
        ax.plot(d["trial_bin"], d["accuracy"], color=animal_colors.get(animal, "#888"),
                alpha=0.3, linewidth=2, label=animal)

    # Overall mean -- source line 299-304
    ax.plot(overall_learning["trial_bin"], overall_learning["accuracy"],
            color="black", linewidth=1, label="Group Mean", zorder=10)
    ax.fill_between(overall_learning["trial_bin"],
                     overall_learning["accuracy"] - overall_learning["sem"],
                     overall_learning["accuracy"] + overall_learning["sem"],
                     color="black", alpha=0.2, zorder=9)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Chance")
    ax.set_xlabel("trial #", fontsize=fontsize)
    ax.set_ylabel("p(highport)", fontsize=fontsize)
    ax.set_title(f"Learning Within Blocks (≥{cfg['min_block_length']} trials, "
                 f"≥{cfg['min_sessions_per_animal']} sessions/animal)\n"
                 f"WhereTask 100-0 | n={len(animals)} animals", fontsize=fontsize)
    ax.set_ylim(0, 1.05)
    ax.set_xlim(0, 100)
    ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.tick_params(axis="both", labelsize=fontsize, length=config["plotting"]["tick_length"])
    clean_axes(ax)
    ax.grid(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _bar_with_session_dots(ax, summary, per_session, value_col, animal_colors, lw, rng):
    x_pos = np.arange(len(summary))
    ax.bar(x_pos, summary[f"mean_{value_col}"], 0.6,
           color=[animal_colors.get(a, "#888") for a in summary["animal"]],
           alpha=0.7, edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, summary[f"mean_{value_col}"], yerr=summary[f"sem_{value_col}"],
                fmt="none", color="black", linewidth=lw, capsize=3, capthick=lw, zorder=10)
    for idx, animal in enumerate(summary["animal"]):
        animal_data = per_session[per_session["animal"] == animal][value_col]
        x_jitter = idx + rng.uniform(-0.15, 0.15, size=len(animal_data))
        ax.scatter(x_jitter, animal_data, s=15, color=animal_colors.get(animal, "#888"),
                   alpha=0.6, edgecolor="black", linewidth=0.3, zorder=5)
    return x_pos


def plot_trials_to_criterion_and_late_accuracy(ttc_df, late_acc_df, config, out_path):
    """Panel c: top=trials-to-criterion, bottom=late accuracy, both
    per-animal bar+SEM with a per-session jittered-scatter overlay. See
    module docstring for why this replaced a direct port of the source
    notebook's own (different) CELL 5-8.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    cfg = config["fig1"]
    rng = np.random.default_rng(config["random_seed"])

    ttc_df = ttc_df.rename(columns={"trials_to_criterion": "ttc"})
    late_acc_df = late_acc_df.rename(columns={"asymptote_accuracy": "late_acc"})

    ttc_summary = ttc_df.groupby("animal")["ttc"].agg(mean_ttc="mean", sem_ttc="sem", n="count").reset_index()
    late_summary = late_acc_df.groupby("animal")["late_acc"].agg(
        mean_late_acc="mean", sem_late_acc="sem", n="count").reset_index()

    animals_ordered = [a for a in animal_order if a in ttc_summary["animal"].values]
    ttc_summary = ttc_summary.set_index("animal").loc[animals_ordered].reset_index()
    late_summary = late_summary.set_index("animal").loc[
        [a for a in animals_ordered if a in late_summary["animal"].values]].reset_index()

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(3.5, 5), sharex=True)

    _bar_with_session_dots(ax_top, ttc_summary, ttc_df, "ttc", animal_colors, lw, rng)
    ax_top.set_ylabel("Trials to criterion", fontsize=fontsize)
    ax_top.set_title(f"Trials to criterion (accuracy ≥ {cfg['criterion_accuracy']}, "
                      f"{cfg['criterion_window']} trials)", fontsize=fontsize)
    ax_top.set_ylim(bottom=0)
    clean_axes(ax_top)
    ax_top.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)

    x_pos = _bar_with_session_dots(ax_bot, late_summary, late_acc_df, "late_acc", animal_colors, lw, rng)
    ax_bot.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax_bot.set_ylabel("Late accuracy", fontsize=fontsize)
    ax_bot.set_title(f"Accuracy, final {cfg['late_accuracy_window']} trials of a block", fontsize=fontsize)
    ax_bot.set_ylim(0, 1.05)
    ax_bot.set_xticks(x_pos)
    ax_bot.set_xticklabels(late_summary["animal"], rotation=45, ha="right")
    ax_bot.set_xlabel("Animal", fontsize=fontsize)
    clean_axes(ax_bot)
    ax_bot.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)

    for ax in (ax_top, ax_bot):
        ax.tick_params(axis="both", labelsize=fontsize, length=config["plotting"]["tick_length"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_session_bins(all_sessions_df, session_bin_size, config, out_path):
    """Panel d: per-session accuracy scatter (individual dots, colored by
    animal) with a per-animal binned trend line (bin size = session_bin_size),
    matching the manuscript legend exactly ("individual dots represent a
    marmoset's performance per session (bin size = 5), and the colored
    lines correspond to the performance averaged across multiple days").
    This replaces an earlier, incorrect two-box boxplot version of this
    panel. The underlying reported statistic (accuracy after ~10 sessions,
    tested vs. chance) is UNCHANGED -- computed the same way as before, from
    the same first/last session bins -- only the rendering changed.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    tick_length = config["plotting"]["tick_length"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    d = all_sessions_df.copy()
    d["session_bin"] = (d["session_number"] // session_bin_size) * session_bin_size
    bin0 = d[d["session_bin"] == 0]["accuracy"]
    bin1 = d[d["session_bin"] == session_bin_size]["accuracy"]

    # Plot restricted to the first 10 sessions ("over the course of 10
    # sessions" per the manuscript legend, and the actual figure's x-axis
    # runs 0-10) -- the STATS below still use the full bin0/bin1 data
    # (unchanged), only the plotted range is capped.
    max_session = 2 * session_bin_size
    d_plot = d[d["session_number"] < max_session]

    fig, ax = plt.subplots(figsize=(4, 3))
    animals = sort_animals(d_plot["animal"].unique().tolist(), animal_order)
    for animal in animals:
        adf = d_plot[d_plot["animal"] == animal]
        color = animal_colors.get(animal, "#888")
        ax.scatter(adf["session_number"], adf["accuracy"], color=color, alpha=0.5, s=15, zorder=3)
        binned = adf.groupby("session_bin")["accuracy"].mean().reset_index()
        ax.plot(binned["session_bin"] + session_bin_size / 2, binned["accuracy"],
                color=color, linewidth=1.5, alpha=0.9, zorder=5)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=lw)
    ax.set_ylabel("Accuracy", fontsize=fontsize)
    ax.set_xlabel("Session #", fontsize=fontsize)
    ax.set_xlim(-0.5, max_session)
    ax.set_ylim(0, 1.05)
    clean_axes(ax)
    ax.tick_params(axis="both", labelsize=fontsize, length=tick_length)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Per-session values are NOT independent observations (n=50 sessions
    # come from only ~10 animals) -- testing against chance at the session
    # level would pseudoreplicate and inflate significance. Aggregate to
    # one mean per ANIMAL first (matching the manuscript's stated n=10),
    # then test on those.
    bin1_per_animal = d[d["session_bin"] == session_bin_size].groupby("animal")["accuracy"].mean()
    bin0_per_animal = d[d["session_bin"] == 0].groupby("animal")["accuracy"].mean()

    _, p_val = scipy_stats.mannwhitneyu(bin0, bin1) if len(bin0) > 0 and len(bin1) > 0 else (np.nan, np.nan)
    _, p_vs_chance = scipy_stats.wilcoxon(bin1_per_animal - 0.5) if len(bin1_per_animal) > 1 else (np.nan, np.nan)
    return {
        "bin0_mean": bin0.mean(), "bin0_sem": scipy_stats.sem(bin0), "bin0_n": len(bin0),
        "bin1_mean": bin1.mean(), "bin1_sem": scipy_stats.sem(bin1), "bin1_n": len(bin1),
        "bin1_per_animal_mean": bin1_per_animal.mean(), "bin1_per_animal_sem": scipy_stats.sem(bin1_per_animal),
        "n_animals": len(bin1_per_animal),
        "mannwhitney_p_bin0_vs_bin1": p_val,
        "wilcoxon_p_bin1_vs_chance_per_animal": p_vs_chance,
    }
