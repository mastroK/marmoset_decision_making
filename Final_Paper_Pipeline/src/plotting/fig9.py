"""Fig 9 -- mouse vs. marmoset, 80-20 only, styled directly after Fig 8's
own panel design (`fig8.py`) -- REBUILT (2026-09-18) per direct user
request, replacing an earlier state-color/species-marker, multi-condition
design that didn't work visually ("this figure is horrific").

Both species are classified with the SAME `classify_state_updated_noAdapting`
rule set, but each against its OWN threshold set (marmoset:
run_pipeline_marmoset.py --thresholds marmoset; mouse: run_pipeline.py
--thresholds mouse_calibrated) -- species-relative thresholds, not one
species' thresholds forced onto the other. See config.yaml's fig9 section
and notebooks/Fig9.ipynb's intro cell for the full account of why (a direct
mouse-vs-marmoset comparison at 80-20 first showed mice have genuinely
higher raw accuracy / lower raw switch-rate than marmosets at this
condition -- real, not a threshold artifact -- so a shared absolute
classifier boundary would conflate that performance-level difference with
a strategy difference).

Unlike Fig 8's own panel a/c (paired connecting lines across the SAME 5
marmosets' two conditions), there is no meaningful pairing here -- mouse
and marmoset are different individuals of different species entirely --
so per-subject dots below are unpaired (jittered clouds, no connecting
lines) and every comparison uses an unpaired t-test (scipy.stats.ttest_ind),
not Fig 8's paired one. All stats (t/p) are computed upstream in the
notebook, exactly as Fig 8 does -- this module only renders precomputed
results, same separation of concerns as fig8.py/fig7.py.
"""

import numpy as np
import matplotlib.pyplot as plt

from .style import clean_axes

COL_MARMOSET = "#8e44ad"
COL_MOUSE = "#16a085"
FONTSIZE = 8


def _sig_label(p):
    if p is None or np.isnan(p):
        return "n.d."
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _unpaired_scatter(ax, marmoset_values, mouse_values, ylabel, sig_p=None, seed=0):
    """Two unpaired jittered dot-clouds (marmoset, mouse) + group mean bar,
    the species-comparison analogue of Fig 8's paired per-animal scatter
    (no connecting lines here -- different individuals, not the same
    subject across conditions)."""
    rng = np.random.default_rng(seed)
    x = np.array([0, 1])
    for xi, values, color in ((0, marmoset_values, COL_MARMOSET), (1, mouse_values, COL_MOUSE)):
        values = np.asarray(values, dtype=float)
        values = values[~np.isnan(values)]
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(xi + jitter, values, color=color, s=22, edgecolor="black", linewidth=0.4, zorder=2)
        mean = values.mean()
        ax.plot([xi - 0.16, xi + 0.16], [mean, mean], color=color, linewidth=2.2, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["marmoset", "mouse"], fontsize=FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if sig_p is not None:
        ax.set_title(_sig_label(sig_p), fontsize=FONTSIZE + 1)
    ax.set_xlim(-0.5, 1.5)
    clean_axes(ax)


def plot_panel_a(marm_pos_summary, mouse_pos_summary, marm_perf, mouse_perf, acc_ttest, out_path):
    """Panel a: (left) p(high-prob choice) aligned to block reversal,
    marmoset (solid) vs. mouse (dashed); (right) per-subject mean session
    accuracy, unpaired. Mirrors Fig 8 panel a's layout exactly, with the
    80-20-vs-100-0 condition pair swapped for a marmoset-vs-mouse species
    pair at 80-20 only."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), width_ratios=[1.6, 1])
    ax_line, ax_scatter = axes

    for summary, color, label, style in (
        (marm_pos_summary, COL_MARMOSET, "marmoset", "-"),
        (mouse_pos_summary, COL_MOUSE, "mouse", "--"),
    ):
        ax_line.plot(summary["position"], summary["mean"], style, color=color, label=label, linewidth=1.2)
        ax_line.fill_between(summary["position"], summary["mean"] - summary["sem"],
                              summary["mean"] + summary["sem"], color=color, alpha=0.2)
    ax_line.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax_line.set_xlabel("trial relative to reversal", fontsize=FONTSIZE)
    ax_line.set_ylabel("p(high prob choice)", fontsize=FONTSIZE)
    ax_line.set_ylim(0, 1.0)
    ax_line.legend(fontsize=7, frameon=False)
    clean_axes(ax_line)

    _unpaired_scatter(ax_scatter, marm_perf["accuracy"], mouse_perf["accuracy"],
                       "mean session accuracy", sig_p=acc_ttest["p"])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_b(marm_pos_summary_switch, mouse_pos_summary_switch, marm_perf, mouse_perf, switch_ttest, out_path):
    """Panel b: (left) p(switch) aligned to block reversal; (right)
    per-subject mean switch rate, unpaired. Same two-part layout as panel
    a, applied to switch rate instead of accuracy."""
    fig, axes = plt.subplots(1, 2, figsize=(5.0, 2.4), width_ratios=[1.6, 1])
    ax_line, ax_scatter = axes

    for summary, color, label, style in (
        (marm_pos_summary_switch, COL_MARMOSET, "marmoset", "-"),
        (mouse_pos_summary_switch, COL_MOUSE, "mouse", "--"),
    ):
        ax_line.plot(summary["position"], summary["mean"], style, color=color, label=label, linewidth=1.2)
        ax_line.fill_between(summary["position"], summary["mean"] - summary["sem"],
                              summary["mean"] + summary["sem"], color=color, alpha=0.2)
    ax_line.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax_line.set_xlabel("trial relative to reversal", fontsize=FONTSIZE)
    ax_line.set_ylabel("p(switch)", fontsize=FONTSIZE)
    ax_line.legend(fontsize=7, frameon=False)
    clean_axes(ax_line)

    _unpaired_scatter(ax_scatter, marm_perf["switch_rate"], mouse_perf["switch_rate"],
                       "mean switch rate", sig_p=switch_ttest["p"])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_params_80_20(param_results, out_path):
    """New panel (no Fig 8 equivalent): sticky Q-learning model parameters
    (alpha/beta/kappa) at 80-20 only, marmoset vs. mouse, unpaired --
    `param_results[param] = {"marmoset_values", "mouse_values", "p"}`."""
    params = [("alpha", "learning rate (alpha)"), ("beta", "inverse temp. (beta)"), ("kappa", "stickiness (kappa)")]
    fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.4))

    for ax, (param, label) in zip(axes, params):
        r = param_results[param]
        _unpaired_scatter(ax, r["marmoset_values"], r["mouse_values"], label, sig_p=r["p"])

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_states_80_20(proportions_result, states_order, out_path):
    """Panel: proportion of trials in each behavioral state at 80-20,
    marmoset vs. mouse, one small panel per state -- direct analogue of
    Fig 8 panel c, unpaired (different species/individuals, so no
    connecting lines). `proportions_result[state] = {"marmoset_values",
    "mouse_values", "p"}`."""
    states = [s for s in states_order if s in proportions_result]
    fig, axes = plt.subplots(1, len(states), figsize=(2.0 * len(states), 2.4), sharex=True)
    if len(states) == 1:
        axes = [axes]

    for ax, state in zip(axes, states):
        r = proportions_result[state]
        _unpaired_scatter(ax, r["marmoset_values"], r["mouse_values"], "")
        ax.set_title(f"{state}\n{_sig_label(r['p'])}", fontsize=7.5)
    axes[0].set_ylabel("proportion of trials", fontsize=FONTSIZE)

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
