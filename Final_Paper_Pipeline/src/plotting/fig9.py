"""Fig 9 -- cross-species comparison (marmoset vs. mouse), NEW figure for
the reviewer response. No manuscript legend to match -- panel layout
agreed directly with the user (see notebooks/Fig9.ipynb's own intro cell).

Both species were classified with the identical `classify_state_updated_noAdapting`
rule set (cross_species_comparison/run_pipeline.py / run_pipeline_marmoset.py)
-- these functions only render the already-computed per-condition summary
tables both scripts produce; they don't touch the classifier or model fit
themselves.

STYLE: matches this project's own established convention for a
state-colored comparison (see fig7.py's plot_panel_b) -- one compact axis
per logical comparison, state identity carried by color (not by faceting
into a separate subplot per state, which wastes most of the figure as
blank space once states span very different value ranges, e.g.
Exploitation's ~0.4-0.8 vs. every other state's ~0-0.3). Species identity
is carried by marker shape + line style (marmoset: filled circle, solid;
mouse: open square, dashed), consistent across every panel.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from .style import clean_axes

FONTSIZE = 9
SPECIES_MARKER = {"marmoset": "o", "mouse": "s"}
SPECIES_FILL = {"marmoset": True, "mouse": False}


def _prob_axis(conditions, condition_to_prob):
    return [condition_to_prob[c] for c in conditions]


def _plot_species_state_line(ax, df, state, color, species, order, condition_to_prob, ycol, yerrcol=None):
    sdf = df[df["Behavioral_State"] == state].set_index("Condition")
    present = [c for c in order if c in sdf.index]
    if not present:
        return
    x = _prob_axis(present, condition_to_prob)
    y = sdf.loc[present, ycol].values
    yerr = sdf.loc[present, yerrcol].values if yerrcol else None
    filled = SPECIES_FILL[species]
    ax.errorbar(
        x, y, yerr=yerr, color=color, marker=SPECIES_MARKER[species],
        linestyle="-" if species == "marmoset" else "--",
        markersize=6.5, markerfacecolor=color if filled else "white",
        markeredgecolor=color, markeredgewidth=1.3, linewidth=1.6,
        capsize=3, elinewidth=1.1, zorder=3,
    )


def _state_legend(ax, states, state_colors, loc="upper left", bbox=(1.02, 1.0)):
    handles = [Line2D([0], [0], color=state_colors[s], marker="o", linestyle="",
                        markersize=6, markeredgecolor=state_colors[s]) for s in states]
    leg = ax.legend(handles, states, fontsize=FONTSIZE - 2, frameon=False,
                     loc=loc, bbox_to_anchor=bbox, title="State", title_fontsize=FONTSIZE - 2)
    ax.add_artist(leg)
    return leg


def _species_legend(ax, loc="lower left", bbox=(1.02, 0.0)):
    handles = [
        Line2D([0], [0], color="black", marker="o", linestyle="-", markersize=6,
               markerfacecolor="black", label="marmoset"),
        Line2D([0], [0], color="black", marker="s", linestyle="--", markersize=6,
               markerfacecolor="white", markeredgecolor="black", label="mouse"),
    ]
    return ax.legend(handles, ["marmoset", "mouse"], fontsize=FONTSIZE - 2, frameon=False,
                      loc=loc, bbox_to_anchor=bbox, title="Species", title_fontsize=FONTSIZE - 2)


def _finish(ax, ylabel, ylim=None):
    ax.set_xlabel("P(reward | better option)", fontsize=FONTSIZE)
    ax.set_xticks([0.7, 0.8, 0.9, 1.0])
    ax.set_ylabel(ylabel, fontsize=FONTSIZE)
    if ylim:
        ax.set_ylim(*ylim)
    ax.tick_params(labelsize=FONTSIZE - 1)
    clean_axes(ax)


def plot_panel_a_state_proportions(marmoset_prop, mouse_prop, cfg, out_path):
    """Panel a: state proportion vs. reward probability of the better
    option. One compact axis, state = color, species = marker/linestyle.
    """
    states = cfg["states_order"]
    colors = cfg["state_colors"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    for state in states:
        _plot_species_state_line(ax, marmoset_prop, state, colors[state], "marmoset",
                                   cfg["marmoset_conditions_order"], cfg["condition_to_prob"], "proportion")
        _plot_species_state_line(ax, mouse_prop, state, colors[state], "mouse",
                                   cfg["mouse_conditions_order"], cfg["condition_to_prob"], "proportion")

    _finish(ax, "Proportion of trials", ylim=(0, 1.0))
    leg1 = _state_legend(ax, states, colors)
    leg2 = _species_legend(ax, loc="upper left", bbox=(1.02, 0.32))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)


def plot_panel_b_model_params(marmoset_fits, mouse_fits, cfg, out_path):
    """Panel b: sticky Q-learning model parameters (alpha/beta/kappa),
    mean +/- SEM across animals/mice, vs. reward probability. Three
    subplots (different units -- unlike panel a/c/d, faceting here is
    appropriate), each compact with both species overlaid directly.
    """
    params = [("alpha", "learning rate (alpha)"), ("beta", "inverse temp. (beta)"), ("kappa", "stickiness (kappa)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    for ax, (param, label) in zip(axes, params):
        for species, fits, order in [
            ("marmoset", marmoset_fits, cfg["marmoset_conditions_order"]),
            ("mouse", mouse_fits, cfg["mouse_conditions_order"]),
        ]:
            summary = fits.groupby("Condition")[param].agg(["mean", "sem"])
            present = [c for c in order if c in summary.index]
            x = _prob_axis(present, cfg["condition_to_prob"])
            means = summary.loc[present, "mean"].values
            sems = summary.loc[present, "sem"].values
            filled = SPECIES_FILL[species]
            color = cfg["species_style"][species]["color"]
            ax.errorbar(
                x, means, yerr=sems, color=color, marker=SPECIES_MARKER[species],
                linestyle="-" if species == "marmoset" else "--",
                markersize=7, markerfacecolor=color if filled else "white",
                markeredgecolor=color, markeredgewidth=1.4, linewidth=1.8,
                capsize=3.5, elinewidth=1.2, label=species, zorder=3,
            )
        ax.set_xlabel("P(reward | better option)", fontsize=FONTSIZE)
        ax.set_ylabel(label, fontsize=FONTSIZE)
        ax.set_xticks([0.7, 0.8, 0.9, 1.0])
        ax.tick_params(labelsize=FONTSIZE - 1)
        clean_axes(ax)

    axes[0].legend(fontsize=FONTSIZE - 1, frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_panel_c_winstay_loseswitch(marmoset_wsls, mouse_wsls, cfg, out_path):
    """Panel c: win-stay (left) and lose-switch (right), one compact axis
    each, state = color, species = marker/linestyle. Highlights the
    Exploitation lose-switch jump at marmoset's deterministic 100-0 (a
    loss is a 100%-reliable "you picked wrong" signal there, unlike the
    probabilistic conditions).
    """
    states = cfg["states_order"]
    colors = cfg["state_colors"]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))

    for ax, metric, label in zip(axes, ["win_stay", "lose_switch"], ["win-stay", "lose-switch"]):
        for state in states:
            _plot_species_state_line(ax, marmoset_wsls, state, colors[state], "marmoset",
                                       cfg["marmoset_conditions_order"], cfg["condition_to_prob"], metric)
            _plot_species_state_line(ax, mouse_wsls, state, colors[state], "mouse",
                                       cfg["mouse_conditions_order"], cfg["condition_to_prob"], metric)
        _finish(ax, label, ylim=(0, 1.0))

    leg1 = _state_legend(axes[1], states, colors)
    leg2 = _species_legend(axes[1], loc="upper left", bbox=(1.02, 0.32))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)


def plot_panel_d_switch_rate(marmoset_switch, mouse_switch, cfg, out_path):
    """Panel d: raw single-trial switch rate, one compact axis, state =
    color, species = marker/linestyle."""
    states = cfg["states_order"]
    colors = cfg["state_colors"]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))

    for state in states:
        _plot_species_state_line(ax, marmoset_switch, state, colors[state], "marmoset",
                                   cfg["marmoset_conditions_order"], cfg["condition_to_prob"], "switch_rate")
        _plot_species_state_line(ax, mouse_switch, state, colors[state], "mouse",
                                   cfg["mouse_conditions_order"], cfg["condition_to_prob"], "switch_rate")

    _finish(ax, "Switch rate", ylim=(0, 0.7))
    leg1 = _state_legend(ax, states, colors)
    leg2 = _species_legend(ax, loc="upper left", bbox=(1.02, 0.32))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
