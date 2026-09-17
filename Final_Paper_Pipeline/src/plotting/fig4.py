"""Panel rendering for Figure 4 (Q-learning model with choice kernel).

Panels a, c, d, e, f are rendered here (panel b is the model-equations box,
rendered as a markdown cell directly in the notebook -- not a matplotlib
figure, per the manuscript's own panel b, which is a text/equation box, not
a data plot).

All panels use the SAME sticky Q-learning + choice-kernel model already
fitted by q_following_model.fit_sticky_qlearning/compute_qvalues_sticky
(identical call to Supp Fig 3, same data) -- this module only renders
already-computed quantities; see src/behavior_metrics/fig4_choice_kernel.py
for the Fig4-specific derived analyses (action-outcome history, log-odds
binning, sigmoid/Gaussian-bump fits) and its module docstring for the
Log_Odds_Left sign-convention finding used throughout these panels.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .style import clean_axes
from ..features.history_encoding import sort_animals
from ..behavior_metrics.fig4_choice_kernel import (
    select_example_session, high_prob_shading_spans, sigmoid, gaussian_bump,
)


def _tick_params(ax, config):
    p = config["plotting"]
    ax.tick_params(axis="both", labelsize=p["fontsize"], length=p["tick_length"])


def plot_example_session(df_q, config, out_path):
    """Panel a: example session choice-outcome trace with the model's
    log-odds prediction overlaid, shaded by which side is currently the
    high-probability choice.

    Manuscript legend: "Example session depicting the choice-outcome
    (Choice: Top/Bottom, Outcome: Rewarded = filled, unrewarded = unfilled)
    across a subset of trials. Shaded regions indicate that the bottom is
    the high probability choice. Overlaid is the updated q-values across
    the session (magenta line)."
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    cfg = config["fig4"]

    session_id = select_example_session(df_q, min_trials=cfg["example_session_min_trials"])
    sdf = df_q[df_q["Session_ID"] == session_id].sort_values("Trial").reset_index(drop=True)
    sdf = sdf.head(cfg["example_session_n_trials_shown"])
    animal = sdf["Animal_Name"].iloc[0]

    fig, ax = plt.subplots(figsize=(6, 2))

    for start, end in high_prob_shading_spans(sdf):
        ax.axvspan(start - 0.5, end + 0.5, color="gray", alpha=0.25, linewidth=0, zorder=0)

    y_bottom, y_top = -3.3, 3.3
    top_choice = sdf[sdf["Choice_Binary"] == 1]
    bottom_choice = sdf[sdf["Choice_Binary"] == 0]
    for sub, y in [(top_choice, y_top), (bottom_choice, y_bottom)]:
        rewarded = sub[sub["Outcome_Binary"] == 1]
        unrewarded = sub[sub["Outcome_Binary"] == 0]
        ax.scatter(rewarded["Trial"], [y] * len(rewarded), facecolor="black",
                   edgecolor="black", s=14, zorder=4)
        ax.scatter(unrewarded["Trial"], [y] * len(unrewarded), facecolor="none",
                   edgecolor="black", s=14, zorder=4)

    ax.plot(sdf["Trial"], sdf["Log_Odds_Left"], color="#c2185b", linewidth=lw * 1.5, zorder=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=lw, alpha=0.5, zorder=1)

    ax.set_ylim(-4, 4)
    ax.set_xlabel("Trial Number", fontsize=fontsize + 1)
    ax.set_ylabel("Log-Odds Prediction\n(Left vs. Right)", fontsize=fontsize + 1)
    ax.set_title(f"Example Session ({animal}, choice: Top/Bottom)", fontsize=fontsize + 1)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="black",
                   markeredgecolor="black", label="Rewarded", markersize=6),
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                   markeredgecolor="black", label="Unrewarded", markersize=6),
        plt.Line2D([0], [0], color="#c2185b", linewidth=lw * 1.5, label="Log-Odds"),
        Patch(facecolor="gray", alpha=0.25, label="Bottom High-Prob (Block Type)"),
    ]
    ax.legend(handles=legend_elements, fontsize=fontsize - 1, frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.05))
    clean_axes(ax)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {"session_id": session_id, "animal": animal, "n_trials_shown": int(len(sdf))}


def plot_params_bar(sticky_params_df, config, out_path):
    """Panel c: average model parameters (alpha, beta, kappa) with
    individual animal means overlaid.

    Manuscript legend: "Average of model parameters (grays bars) overlaid
    with individual animal means (dots) across alpha (learning rate), beta
    (inverse temperature) and kappa (sticky term)."
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]

    params = ["alpha", "beta", "kappa"]
    labels = ["α", "β", "κ"]
    means = [sticky_params_df[p].mean() for p in params]
    sems = [sticky_params_df[p].sem() for p in params]
    animals = sort_animals(sticky_params_df["animal"].unique().tolist(), animal_order)

    rng = np.random.default_rng(config["random_seed"])
    x_pos = np.arange(len(params))

    fig, ax = plt.subplots(figsize=(2, 2))
    ax.bar(x_pos, means, 0.6, color="lightgray", alpha=0.7, edgecolor="black", linewidth=lw)
    ax.errorbar(x_pos, means, yerr=sems, fmt="none", color="black", linewidth=lw,
                capsize=3, capthick=lw, zorder=10)

    for i, param in enumerate(params):
        for animal in animals:
            value = sticky_params_df.loc[sticky_params_df["animal"] == animal, param].iloc[0]
            x_jitter = i + rng.uniform(-0.15, 0.15)
            ax.scatter(x_jitter, value, s=18, color=animal_colors.get(animal, "gray"),
                       alpha=0.9, edgecolor="black", linewidth=0.5, zorder=5)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=fontsize + 2)
    ax.set_ylabel("weight", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_history_logodds(history_summary, config, out_path):
    """Panel d: bar plot of the model's log-odds predictions binned by the
    3-trial action-outcome history (e.g. 'RlR'), sorted ascending.

    Manuscript legend: "Bar plot depicting the log-odds predictions binned
    by the action-outcome histories"
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]

    x_pos = np.arange(len(history_summary))
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.bar(x_pos, history_summary["mean_logodds"], color="black", alpha=0.75,
           edgecolor="white", linewidth=lw)
    ax.errorbar(x_pos, history_summary["mean_logodds"], yerr=history_summary["sem"],
                fmt="none", color="black", linewidth=lw, capsize=2, capthick=lw, zorder=10)
    ax.axhline(0, color="gray", linestyle="--", linewidth=lw, alpha=0.5)

    max_labels = 20
    step = max(1, int(np.ceil(len(x_pos) / max_labels)))
    tick_idx = x_pos[::step]
    ax.set_xticks(tick_idx)
    ax.set_xticklabels(history_summary["Action_Outcome_History"].iloc[::step],
                       rotation=45, ha="right", fontsize=fontsize - 2)
    ax.set_ylabel("log-odds prediction", fontsize=fontsize + 1)
    clean_axes(ax)
    ax.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_psychometric(binned_e, fits_e, config, out_path):
    """Panel e: p(choose left) binned across the log-odds prediction for
    individual marmosets (dots), sigmoid fits (solid lines), plus a bar plot
    of the fitted |bias| and sensitivity parameters.

    Manuscript legend: "Probability of choosing the left choice binned
    across the log-odds prediction for individual marmosets (dots) and the
    sigmoid function fits across values (solid lines) (left). Bar plot of
    the average (bars) and individual (dots) values of the bias and
    sensitivity derived from the sigmoid fit parameters."
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    clip = config["fig4"]["logodds_clip"]

    fig = plt.figure(figsize=(4.5, 2), layout="constrained")
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.5)
    ax1 = fig.add_subplot(gs[0])

    animals = sort_animals(binned_e["Animal_Name"].unique().tolist(), animal_order)
    x_smooth = np.linspace(-clip, clip, 100)
    for animal in animals:
        adata = binned_e[binned_e["Animal_Name"] == animal]
        color = animal_colors.get(animal, "gray")
        ax1.scatter(adata["bin_center"], adata["mean"], color=color, s=14, alpha=0.7,
                    edgecolor="black", linewidth=0.4, zorder=4)
        fit_row = fits_e[fits_e["Animal_Name"] == animal]
        if len(fit_row) > 0:
            x0, k = fit_row["bias"].iloc[0], fit_row["sensitivity"].iloc[0]
            ax1.plot(x_smooth, sigmoid(x_smooth, x0, k), color=color, linewidth=1.5,
                     alpha=0.85, label=animal, zorder=3)

    ax1.axhline(0.5, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax1.axvline(0, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax1.set_xlim(-clip, clip)
    ax1.set_ylim(0, 1)
    ax1.set_xlabel("log-odds prediction", fontsize=fontsize + 1)
    ax1.set_ylabel("p(Choose Left)", fontsize=fontsize + 1)
    ax1.legend(fontsize=fontsize - 2, frameon=False, loc="upper left")
    clean_axes(ax1)
    ax1.grid(axis="both", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax1, config)

    ax2 = fig.add_subplot(gs[1])
    abs_fits = fits_e.copy()
    abs_fits["bias"] = abs_fits["bias"].abs()
    param_cols = ["bias", "sensitivity"]
    param_labels = ["|bias|", "sensitivity"]
    means = [abs_fits[p].mean() for p in param_cols]
    sems = [abs_fits[p].sem() for p in param_cols]
    x_pos = np.arange(len(param_cols))

    rng = np.random.default_rng(config["random_seed"])
    ax2.bar(x_pos, means, 0.6, color="lightgray", alpha=0.7, edgecolor="black", linewidth=lw)
    ax2.errorbar(x_pos, means, yerr=sems, fmt="none", color="black", linewidth=lw,
                 capsize=3, capthick=lw, zorder=10)
    for i, param in enumerate(param_cols):
        for _, row in abs_fits.iterrows():
            x_jitter = i + rng.uniform(-0.15, 0.15)
            ax2.scatter(x_jitter, row[param], s=16, color=animal_colors.get(row["Animal_Name"], "gray"),
                       alpha=0.9, edgecolor="black", linewidth=0.4, zorder=5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(param_labels, rotation=45, ha="right", fontsize=fontsize - 1)
    ax2.set_ylabel("parameter value", fontsize=fontsize)
    clean_axes(ax2)
    ax2.grid(axis="y", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax2, config)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_switch_curve(binned_f, fits_f, config, out_path):
    """Panel f: p(switch) binned across the log-odds prediction for
    individual marmosets (dots) and curve fits across values (solid lines).

    Manuscript legend: "Probability of switching binned across the log-odds
    prediction for individual marmosets (dots) and the sigmoid function fits
    across values (solid lines)" -- see fig4_choice_kernel.gaussian_bump's
    docstring for why a symmetric Gaussian bump (not a literal monotonic
    logistic) is fit here, given the empirically inverted-U shape.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    animal_colors = config["plotting"]["animal_colors"]
    animal_order = config["plotting"]["animal_order"]
    clip = config["fig4"]["logodds_clip"]

    fig, ax = plt.subplots(figsize=(3, 2))
    animals = sort_animals(binned_f["Animal_Name"].unique().tolist(), animal_order)
    x_smooth = np.linspace(-clip, clip, 100)
    for animal in animals:
        adata = binned_f[binned_f["Animal_Name"] == animal]
        color = animal_colors.get(animal, "gray")
        ax.scatter(adata["bin_center"], adata["mean"], color=color, s=14, alpha=0.7,
                   edgecolor="black", linewidth=0.4, zorder=4)
        fit_row = fits_f[fits_f["Animal_Name"] == animal]
        if len(fit_row) > 0:
            x0, sigma, amp, base = (fit_row["peak_center"].iloc[0], fit_row["width"].iloc[0],
                                     fit_row["amplitude"].iloc[0], fit_row["baseline"].iloc[0])
            ax.plot(x_smooth, gaussian_bump(x_smooth, x0, sigma, amp, base), color=color,
                    linewidth=1.5, alpha=0.85, label=animal, zorder=3)

    ax.axvline(0, color="gray", linestyle="--", linewidth=lw, alpha=0.5)
    ax.set_xlim(-clip, clip)
    ax.set_ylim(0, 1)
    ax.set_xlabel("log-odds prediction", fontsize=fontsize + 1)
    ax.set_ylabel("p(switch)", fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize - 2, frameon=False, loc="upper right")
    clean_axes(ax)
    ax.grid(axis="both", alpha=0.3, linestyle=":", linewidth=lw)
    _tick_params(ax, config)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
