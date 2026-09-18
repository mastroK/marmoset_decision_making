"""Plotting for the new Fig 8 companion figure -- 80-20 vs. 90-10 vs. 100-0
(three conditions, now that 90-10 sessions exist), built for the 2026
reviewer response. Does NOT modify `fig8.py` -- Fig 8 itself (80-20 vs.
100-0 only) is untouched; this is a new, separate figure.

STYLE: deliberately mirrors `fig8.py`'s own established panel layouts
(reversal-aligned line + paired scatter for a/b, per-state small multiples
for c, grouped bars for d, loss-streak line for e, diff heatmap for f) --
those layouts already work fine for a handful of x-positions and don't have
Fig 9's oversplit-`sharey`-facet problem, so they're extended from two
conditions to three rather than redesigned. Significance is now a Friedman
test (3+ repeated-measures groups) instead of Fig 8's paired t-test (exactly
2 groups) -- see `cross_condition_3way.py`.

Panel f is walked in two single-step diffs (80-20->90-10, 90-10->100-0)
rather than one 80-20-vs-100-0 diff, so the gradient itself is visible
instead of collapsing it to its two endpoints.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import clean_axes
from ..features.history_encoding import sort_animals

FONTSIZE = 8
COL_80 = "#4878CF"
COL_90 = "#9B59B6"
COL_100 = "#D65F5F"
COND_STYLE = {
    "80-20": {"color": COL_80, "linestyle": "--", "marker": "o"},
    "90-10": {"color": COL_90, "linestyle": ":", "marker": "s"},
    "100-0": {"color": COL_100, "linestyle": "-", "marker": "^"},
}
GREY = "#AAAAAA"


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


def _animal_legend(ax, animals, animal_colors, loc="upper left", bbox=(1.02, 1.0)):
    """Per-animal color legend -- this project's own established convention
    for tracking an individual animal across x-positions (e.g. fig1.py's
    within-block-learning panel), applied here so an animal's line color is
    the same at 80-20/90-10/100-0.

    Always the LAST legend() call on its axes in this module -- does NOT
    call ax.add_artist() itself (that's only needed to protect an EARLIER
    legend from being replaced by a later ax.legend() call; doing it here
    too double-registers this same legend as both `ax.legend_` and a plain
    child artist, rendering it twice)."""
    handles = [Line2D([0], [0], color=animal_colors.get(a, "#888888"), marker="o",
                       linestyle="-", markersize=4.5, linewidth=1.3, label=a) for a in animals]
    return ax.legend(handles=handles, fontsize=FONTSIZE - 2, frameon=False,
                      loc=loc, bbox_to_anchor=bbox, title="Animal", title_fontsize=FONTSIZE - 2)


def plot_panel_a(pos_summaries, animal_acc_result, conditions, animal_colors, animal_order, out_path):
    """Panel a: p(high-prob choice) aligned to block reversal, one line per
    condition (left); paired per-animal mean session accuracy across all
    three conditions, one connected line per animal, colored by this
    project's own established per-animal palette (config.yaml's
    plotting.animal_colors) so a given animal is the same color at every
    condition -- lets a reader track one animal's performance across the
    80-20 -> 90-10 -> 100-0 gradient (right)."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.4), width_ratios=[1.6, 1])
    ax_line, ax_scatter = axes

    for c in conditions:
        summary = pos_summaries[c]
        st = COND_STYLE[c]
        ax_line.plot(summary["position"], summary["mean"], st["linestyle"], color=st["color"],
                     label=c, linewidth=1.2)
        ax_line.fill_between(summary["position"], summary["mean"] - summary["sem"],
                              summary["mean"] + summary["sem"], color=st["color"], alpha=0.2)
    ax_line.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax_line.set_xlabel("trial relative to reversal", fontsize=8)
    ax_line.set_ylabel("p(high prob choice)", fontsize=8)
    ax_line.set_ylim(0, 1.0)
    ax_line.legend(fontsize=7, frameon=False)
    clean_axes(ax_line)

    x = np.arange(len(conditions))
    animals = sort_animals(animal_acc_result["used_animals"], animal_order)
    values = {c: dict(zip(animal_acc_result["animals"], animal_acc_result[f"values_{c.replace('-', '_')}"]))
              for c in conditions}
    for a in animals:
        y = [values[c][a] for c in conditions]
        color = animal_colors.get(a, "#888888")
        ax_scatter.plot(x, y, color=color, linewidth=1.2, marker="o", markersize=4, zorder=2)
    ax_scatter.set_xticks(x)
    ax_scatter.set_xticklabels(conditions, fontsize=8)
    ax_scatter.set_ylabel("mean session accuracy", fontsize=8)
    ax_scatter.set_title(f"Friedman {_sig_label(animal_acc_result['p'])}", fontsize=8.5)
    ax_scatter.set_xlim(-0.4, len(conditions) - 0.6)
    clean_axes(ax_scatter)
    leg = _animal_legend(ax_scatter, animals, animal_colors, loc="upper left", bbox=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300, bbox_extra_artists=(leg,))
    plt.close(fig)


def plot_panel_b(pos_summaries, conditions, out_path):
    """Panel b: mean p(switch) aligned to block reversal, one line per condition."""
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    for c in conditions:
        summary = pos_summaries[c]
        st = COND_STYLE[c]
        ax.plot(summary["position"], summary["mean"], st["linestyle"], color=st["color"], label=c, linewidth=1.2)
        ax.fill_between(summary["position"], summary["mean"] - summary["sem"],
                         summary["mean"] + summary["sem"], color=st["color"], alpha=0.2)
    ax.axvline(-0.5, color="black", linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlabel("trial relative to reversal", fontsize=8)
    ax.set_ylabel("p(switch)", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_c(proportions_result, states_order, conditions, animal_colors, animal_order, out_path):
    """Panel c: proportion of trials in each behavioral state, one small
    panel per state, paired per-animal dots+lines across all three
    conditions -- colored per-animal (plotting.animal_colors) so the same
    animal is traceable across states and conditions."""
    states = [s for s in states_order if s in proportions_result]
    fig, axes = plt.subplots(1, len(states), figsize=(2.1 * len(states), 2.4), sharex=True)
    if len(states) == 1:
        axes = [axes]
    x = np.arange(len(conditions))

    all_animals = sort_animals(
        sorted(set().union(*[proportions_result[s]["used_animals"] for s in states])), animal_order
    )
    for ax, state in zip(axes, states):
        r = proportions_result[state]
        animals = sort_animals(r["used_animals"], animal_order)
        values = {c: dict(zip(r["animals"], r[f"values_{c.replace('-', '_')}"])) for c in conditions}
        for a in animals:
            y = [values[c][a] for c in conditions]
            ax.plot(x, y, color=animal_colors.get(a, "#888888"), linewidth=1.0,
                    marker="o", markersize=3.5, zorder=2)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=6.5, rotation=20, ha="right")
        ax.set_title(f"{state}\nFriedman {_sig_label(r['p'])}", fontsize=7.5)
        ax.set_xlim(-0.4, len(conditions) - 0.6)
        clean_axes(ax)
    axes[0].set_ylabel("proportion of trials", fontsize=8)
    leg = _animal_legend(axes[-1], all_animals, animal_colors, loc="upper left", bbox=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300, bbox_extra_artists=(leg,))
    plt.close(fig)


def plot_panel_d(wsls_result, states_order, conditions, animal_colors, animal_order, out_path):
    """Panel d: win-stay (left) and lose-switch (right) for Exploitation and
    Directed Exploration, grouped bars (colored by condition), with each
    animal's own paired trend overlaid in that animal's own color
    (plotting.animal_colors) instead of a color-anonymous grey line."""
    states = [s for s in states_order if s in wsls_result["win_stay"]]
    n_cond = len(conditions)
    width = 0.8 / n_cond
    fig, axes = plt.subplots(1, 2, figsize=(2.6 * len(states), 2.6))
    x_base = np.arange(len(states))

    all_animals = sort_animals(
        sorted(set().union(*[wsls_result[m][s]["used_animals"] for m in ("win_stay", "lose_switch") for s in states])),
        animal_order,
    )
    for ax, metric, title in ((axes[0], "win_stay", "win-stay"), (axes[1], "lose_switch", "lose-switch")):
        for i, state in enumerate(states):
            r = wsls_result[metric][state]
            values = {c: dict(zip(r["animals"], r[f"values_{c.replace('-', '_')}"])) for c in conditions}
            offsets = (np.arange(n_cond) - (n_cond - 1) / 2) * width
            means, sems = [], []
            for k, c in enumerate(conditions):
                v = np.array(list(values[c].values()), dtype=float)
                v = v[~np.isnan(v)]
                m, sem = np.nanmean(v), (np.nanstd(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
                means.append(m)
                sems.append(sem)
                ax.bar(i + offsets[k], m, width, yerr=sem, color=COND_STYLE[c]["color"],
                       capsize=2, edgecolor="white", linewidth=0.5, label=c if i == 0 else None)
            animals = sort_animals(r["used_animals"], animal_order)
            for a in animals:
                y = [values[c][a] for c in conditions]
                ax.plot(i + offsets, y, color=animal_colors.get(a, "#888888"),
                        linewidth=0.8, marker="o", markersize=2.5, zorder=2, alpha=0.85)
            ax.text(i, max(m + s for m, s in zip(means, sems)) + 0.03, _sig_label(r["p"]), ha="center", fontsize=8)

        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
        ax.set_xticks(x_base)
        ax.set_xticklabels(states, fontsize=7.5, rotation=15, ha="right")
        ax.set_title(metric.replace("_", "-"), fontsize=8)
        ax.set_ylim(0, 1.25)
        clean_axes(ax)
    axes[0].set_ylabel("probability", fontsize=8)
    leg1 = axes[1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0),
                           title="Condition", title_fontsize=7)
    axes[1].add_artist(leg1)  # promote out of the single ax.legend_ slot before the 2nd legend() call
    leg2 = _animal_legend(axes[1], all_animals, animal_colors, loc="upper left", bbox=(1.02, 0.55))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300, bbox_extra_artists=(leg1, leg2))
    plt.close(fig)


def plot_panel_e(streak_summaries, conditions, out_path, state_label="Exploitation"):
    """Panel e: p(switch) vs. consecutive losses within Exploitation, one line per condition."""
    fig, ax = plt.subplots(figsize=(2.8, 2.4))
    for c in conditions:
        summary = streak_summaries[c]
        st = COND_STYLE[c]
        ax.plot(summary["Loss_Streak"], summary["mean"], st["linestyle"], color=st["color"], label=c,
                marker=st["marker"], linewidth=1.2, markersize=4)
        ax.fill_between(summary["Loss_Streak"], summary["mean"] - summary["sem"],
                         summary["mean"] + summary["sem"], color=st["color"], alpha=0.2)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xlabel(f"consecutive losses in {state_label}", fontsize=7.5)
    ax.set_ylabel("p(switch)", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_f(diff_ab, diff_bc, out_path):
    """Panel f: two single-step Delta transition-probability heatmaps
    (Exploitation <-> Directed Exploration), walking the 80-20 -> 90-10 ->
    100-0 gradient in two steps instead of collapsing it to one 80-20-vs-
    100-0 diff."""
    fig, axes = plt.subplots(1, 2, figsize=(5.6, 2.6))
    all_vals = np.concatenate([diff_ab["diff_matrix"].ravel(), diff_bc["diff_matrix"].ravel()])
    vmax = np.nanmax(np.abs(all_vals))

    for ax, diff_result in zip(axes, (diff_ab, diff_bc)):
        states_plot = [s.replace(" ", "\n", 1) if s == "Directed Exploration" else s
                       for s in diff_result["states"]]
        diff_matrix = diff_result["diff_matrix"]
        pval_matrix = diff_result["pval_matrix"]
        n = len(diff_result["states"])

        im = ax.imshow(diff_matrix, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        for i in range(n):
            for j in range(n):
                v = diff_matrix[i, j]
                if np.isnan(v):
                    ax.text(j, i, "n.d.", ha="center", va="center", fontsize=6.5, color="gray")
                    continue
                sig = "*" if pval_matrix[i, j] < 0.05 else ""
                tcol = "white" if abs(v) > vmax * 0.55 else "black"
                ax.text(j, i, f"{v:+.2f}{sig}", ha="center", va="center", fontsize=7.5,
                         fontweight="bold" if sig else "normal", color=tcol)
        for k in range(n + 1):
            ax.axhline(k - 0.5, color="white", linewidth=0.8)
            ax.axvline(k - 0.5, color="white", linewidth=0.8)

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(states_plot, fontsize=7)
        ax.set_yticklabels(states_plot, fontsize=7)
        ax.set_xlabel("To state", fontsize=7.5)
        ax.set_title(f"{diff_result['cond_b']} - {diff_result['cond_a']}", fontsize=8)
    axes[0].set_ylabel("From state", fontsize=7.5)

    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.03)
    cbar.set_label("Delta transition probability", fontsize=7)
    cbar.ax.tick_params(labelsize=7)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_g_sticky_model(comparison, conditions, animal_colors, animal_order, out_path):
    """New panel (not in Fig 8 itself): sticky Q-learning params
    (alpha/beta/kappa) vs. condition -- the marmoset-side analogue of Fig 9
    panel b, restricted to this figure's three conditions and Fig 8's own
    single-init model fit (not Fig 9's 10-restart cross-species version).
    Per-animal fits (thin, animal-colored lines) are shown underneath the
    group mean +/- SEM (bold black line) so an animal's own model
    parameters can be traced across conditions, not just the group trend."""
    params = [("alpha", "learning rate (alpha)"), ("beta", "inverse temp. (beta)"), ("kappa", "stickiness (kappa)")]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.6))
    x = np.arange(len(conditions))

    all_animals = sort_animals(comparison["alpha"]["animals"], animal_order)
    for ax, (param, label) in zip(axes, params):
        r = comparison[param]
        values = {c: dict(zip(r["animals"], r[f"values_{c.replace('-', '_')}"])) for c in conditions}
        for a in all_animals:
            y = [values[c][a] for c in conditions]
            if any(np.isnan(y)):
                continue
            ax.plot(x, y, color=animal_colors.get(a, "#888888"), linewidth=0.9,
                    marker="o", markersize=3, alpha=0.75, zorder=2)

        means, sems = [], []
        for c in conditions:
            v = np.array(r[f"values_{c.replace('-', '_')}"], dtype=float)
            v = v[~np.isnan(v)]
            means.append(np.nanmean(v))
            sems.append(np.nanstd(v, ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
        ax.errorbar(x, means, yerr=sems, color="black", marker="o", markersize=5, linewidth=1.6, capsize=3, zorder=3)
        ax.set_xticks(x)
        ax.set_xticklabels(conditions, fontsize=7.5)
        ax.set_ylabel(label, fontsize=8)
        ax.set_title(f"Friedman {_sig_label(r['p'])}", fontsize=8)
        clean_axes(ax)

    leg = _animal_legend(axes[-1], all_animals, animal_colors, loc="upper left", bbox=(1.05, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300, bbox_extra_artists=(leg,))
    plt.close(fig)
