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
from ..features.history_encoding import sort_animals

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


def _animal_legend(ax, animals, animal_colors, loc="upper left", bbox=(1.05, 1.0)):
    """Per-animal color legend (config.yaml's plotting.animal_colors/
    animal_order -- this project's own established per-animal palette,
    same one used throughout Fig1-Fig5 and fig8_3cond.py). Always the
    LAST legend() call on its axes here -- no add_artist needed (see
    fig8_3cond.py's own _animal_legend docstring for why calling it
    unconditionally double-renders the legend)."""
    handles = [Line2D([0], [0], color=animal_colors.get(a, "#888888"), marker="o",
                       linestyle="-", markersize=4.5, linewidth=1.3, label=a) for a in animals]
    return ax.legend(handles=handles, fontsize=FONTSIZE - 2, frameon=False,
                      loc=loc, bbox_to_anchor=bbox, title="Animal", title_fontsize=FONTSIZE - 2)


def plot_marmoset_per_animal_by_state(data_by_animal, states_order, conditions_order, condition_to_prob,
                                        value_col, ylabel, animal_colors, animal_order, out_path, ylim=None):
    """Supplementary, marmoset-only per-animal breakdown for a panel whose
    main view already uses color for STATE identity (mice have no
    established per-individual color convention, so this is marmoset-only,
    same scope as the user's request). Overlaying per-animal color directly
    on the main state-colored axis would double-encode the color channel,
    so this is a separate small-multiples-by-state view instead (one
    subplot per state, animal = color) -- mirrors fig8_3cond.py's own
    panel c design, not a replacement for the main species/state panel.
    """
    states = [s for s in states_order if s in data_by_animal["Behavioral_State"].unique()]
    fig, axes = plt.subplots(1, len(states), figsize=(2.2 * len(states), 2.6), sharex=True)
    if len(states) == 1:
        axes = [axes]

    all_animals = sort_animals(data_by_animal["Animal_Name"].unique().tolist(), animal_order)
    xticks = _prob_axis(conditions_order, condition_to_prob)

    for ax, state in zip(axes, states):
        sdf = data_by_animal[data_by_animal["Behavioral_State"] == state]
        for animal in all_animals:
            adf = sdf[sdf["Animal_Name"] == animal].set_index("Condition")
            present = [c for c in conditions_order if c in adf.index]
            if len(present) < 2:
                continue
            xa = _prob_axis(present, condition_to_prob)
            y = adf.loc[present, value_col].values
            ax.plot(xa, y, color=animal_colors.get(animal, "#888888"), linewidth=1.0,
                    marker="o", markersize=3, zorder=2)
        ax.set_xlabel("P(reward)", fontsize=FONTSIZE - 2)
        ax.set_xticks(xticks)
        ax.set_title(state, fontsize=FONTSIZE - 1)
        if ylim:
            ax.set_ylim(*ylim)
        ax.tick_params(labelsize=FONTSIZE - 2)
        clean_axes(ax)
    axes[0].set_ylabel(ylabel, fontsize=FONTSIZE)

    leg = _animal_legend(axes[-1], all_animals, animal_colors)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=(leg,))
    plt.close(fig)


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


def plot_panel_b_model_params(marmoset_fits, mouse_fits, cfg, out_path,
                                marmoset_fits_by_animal=None, animal_colors=None, animal_order=None):
    """Panel b: sticky Q-learning model parameters (alpha/beta/kappa),
    mean +/- SEM across animals/mice, vs. reward probability. Three
    subplots (different units -- unlike panel a/c/d, faceting here is
    appropriate), each compact with both species overlaid directly.

    If `marmoset_fits_by_animal` is given, each marmoset's own per-condition
    fit is drawn underneath the aggregate as a thin, animal-colored line
    (config.yaml's plotting.animal_colors) -- no color-channel conflict
    here (unlike panel a/c/d) since color is otherwise only used for
    species, not state.
    """
    params = [("alpha", "learning rate (alpha)"), ("beta", "inverse temp. (beta)"), ("kappa", "stickiness (kappa)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))

    all_animals = []
    if marmoset_fits_by_animal is not None:
        all_animals = sort_animals(marmoset_fits_by_animal["Animal_Name"].unique().tolist(), animal_order)

    for ax, (param, label) in zip(axes, params):
        if marmoset_fits_by_animal is not None:
            for animal in all_animals:
                adf = marmoset_fits_by_animal[marmoset_fits_by_animal["Animal_Name"] == animal].set_index("Condition")
                present = [c for c in cfg["marmoset_conditions_order"] if c in adf.index]
                if len(present) < 2:
                    continue
                xa = _prob_axis(present, cfg["condition_to_prob"])
                y = adf.loc[present, param].values
                ax.plot(xa, y, color=animal_colors.get(animal, "#888888"), linewidth=0.8,
                        marker="o", markersize=2.5, alpha=0.75, zorder=2)

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
    extra_artists = ()
    if marmoset_fits_by_animal is not None:
        leg = _animal_legend(axes[-1], all_animals, animal_colors)
        extra_artists = (leg,)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight", bbox_extra_artists=extra_artists)
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
