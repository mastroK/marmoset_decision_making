"""Panel rendering for Supplementary Figure 2 (4-model Q-learning
comparison: pure QL / +sticky / +forgetting / +adaptive forgetting).

Panels a-d share one layout: top-left = mean p(high-prob choice) aligned
to block reversal (marmoset grey vs. model in that model's color, +/-SEM
shading); bottom-left = same for p(switch); right = session-level
scatter of model-predicted vs. observed p(switch), dashed unity line,
mean fitted parameters annotated (manuscript legend text for panel a,
reused verbatim for b-d: "As shown in panel (a) but for the...").
Panel e = 5 bar charts (test NLL, r(p_high), r(p_switch), trial-by-trial
r(switch), composite), one group of 4 bars (one per model) each, mean+/-SEM
across animals (n=5) -- see rl_model_comparison.model_comparison_table's
docstring for why this pipeline treats the per-animal grouping as the
real unit here rather than per-session (pseudoreplication).
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes
from ..behavior_metrics.rl_model_comparison import MODEL_LABELS, MODEL_COLORS, MODEL_NAMES


def _tick_params(ax, config):
    p = config["plotting"]
    ax.tick_params(axis="both", labelsize=p["fontsize"], length=p["tick_length"])


def _param_annotation(model, param_row):
    labels = {"alpha": "α", "beta": "β", "kappa": "κ",
              "tau": "τ", "tau_max": "τ_max", "decay": "decay"}
    order = {
        "pure": ["alpha", "beta"],
        "sticky": ["alpha", "beta", "kappa"],
        "forgetting": ["alpha", "beta", "kappa", "tau"],
        "adaptive_forgetting": ["alpha", "beta", "kappa", "tau_max", "decay"],
    }[model]
    lines = [f"{labels[p]}={param_row[p]:.2f}" for p in order]
    return "\n".join(lines)


def plot_model_panel(curves, scatter_df, mean_params, model, config, out_path):
    """One panel (a/b/c/d): top p(high), bottom p(switch), right scatter."""
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]
    color = MODEL_COLORS[model]

    fig = plt.figure(figsize=(5.5, 2.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], wspace=0.45, hspace=0.5)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bottom = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[:, 1])

    pos = curves["position"].values

    ax = ax_top
    ax.plot(pos, curves["phigh_marmoset_mean"], color="gray", linewidth=lw * 1.5, label="marmoset")
    ax.fill_between(pos, curves["phigh_marmoset_mean"] - curves["phigh_marmoset_sem"],
                     curves["phigh_marmoset_mean"] + curves["phigh_marmoset_sem"], color="gray", alpha=0.3)
    ax.plot(pos, curves["phigh_model_mean"], color=color, linewidth=lw * 1.5, label="model")
    ax.fill_between(pos, curves["phigh_model_mean"] - curves["phigh_model_sem"],
                     curves["phigh_model_mean"] + curves["phigh_model_sem"], color=color, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=lw, alpha=0.6)
    ax.set_ylabel("p(high prob)", fontsize=fontsize)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=fontsize - 2, frameon=False, loc="lower right")
    clean_axes(ax)
    _tick_params(ax, config)

    ax = ax_bottom
    ax.plot(pos, curves["pswitch_marmoset_mean"], color="gray", linewidth=lw * 1.5)
    ax.fill_between(pos, curves["pswitch_marmoset_mean"] - curves["pswitch_marmoset_sem"],
                     curves["pswitch_marmoset_mean"] + curves["pswitch_marmoset_sem"], color="gray", alpha=0.3)
    ax.plot(pos, curves["pswitch_model_mean"], color=color, linewidth=lw * 1.5)
    ax.fill_between(pos, curves["pswitch_model_mean"] - curves["pswitch_model_sem"],
                     curves["pswitch_model_mean"] + curves["pswitch_model_sem"], color=color, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=lw, alpha=0.6)
    ax.set_xlabel("block position", fontsize=fontsize)
    ax.set_ylabel("p(switch)", fontsize=fontsize)
    ax.set_ylim(0, max(0.6, curves["pswitch_marmoset_mean"].max() * 1.2))
    clean_axes(ax)
    _tick_params(ax, config)

    ax = ax_scatter
    ax.scatter(scatter_df["observed"], scatter_df["predicted"], s=14, color=color, alpha=0.6,
               edgecolor="black", linewidth=0.3)
    lo = min(scatter_df["observed"].min(), scatter_df["predicted"].min(), 0)
    hi = max(scatter_df["observed"].max(), scatter_df["predicted"].max(), 1)
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=lw, alpha=0.6)
    ax.set_xlabel("p(switch) marmoset", fontsize=fontsize)
    ax.set_ylabel("p(switch) model", fontsize=fontsize)
    ax.text(0.03, 0.97, _param_annotation(model, mean_params), transform=ax.transAxes,
            fontsize=fontsize - 1, va="top", ha="left")
    clean_axes(ax)
    _tick_params(ax, config)

    fig.suptitle(MODEL_LABELS[model], fontsize=fontsize + 2, y=1.03)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_model_comparison(comparison_table, config, out_path):
    """Panel e: 5 bar charts (test NLL, r(p_high), r(p_switch), trial-by-
    trial r(switch), composite), mean+/-SEM across animals, one group of
    4 bars (models) per chart.
    """
    fontsize = config["plotting"]["fontsize"]
    lw = config["plotting"]["line_width"]

    metrics = [
        ("test_nll", "test NLL"), ("r_phigh", "r(p high prob)"), ("r_pswitch", "r(p switch)"),
        ("r_switch_trial", "r(switch, trial-by-trial)"), ("composite", "composite score"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(3.2 * len(metrics), 2.6))

    for ax, (col, label) in zip(axes, metrics):
        means = [comparison_table.loc[comparison_table["model"] == m, col].mean() for m in MODEL_NAMES]
        sems = [comparison_table.loc[comparison_table["model"] == m, col].sem() for m in MODEL_NAMES]
        x = np.arange(len(MODEL_NAMES))
        ax.bar(x, means, yerr=sems, color=[MODEL_COLORS[m] for m in MODEL_NAMES],
               alpha=0.85, edgecolor="black", linewidth=lw, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels([MODEL_LABELS[m] for m in MODEL_NAMES], rotation=45, ha="right", fontsize=fontsize - 1)
        ax.set_ylabel(label, fontsize=fontsize)
        clean_axes(ax)
        _tick_params(ax, config)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
