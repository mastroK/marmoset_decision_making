"""Panel rendering for Fig 7 -- "A behavioral state framework reveals
distinct modes of reward-guided decision making in marmosets" (manuscript
Results section of the same name; Fig 7 legend, ~page 26-27 of
2026.06.30.734629v1.full.pdf). Built from scratch for this new canonical
7-state taxonomy (`state_classifier.classify_state_v7`) -- no prior
publication-figure cell for this exact 7-state version was found in the
source archive (see config.yaml's fig7 section / state_classifier.py's
module docstring for the closest prior art, the orphaned
`07c_FinalClassifier_Fig6.ipynb`).

Panel-a's block-structure row (alternating color per block, block number
labeled in white) follows this project's own general "exemplar session"
convention (seen across several of its exploratory notebooks, e.g.
`07_GLM_HMM_v1_cleaned.ipynb`) -- kept only for that cosmetic element; the
row *contents* (3 rows: block structure / rolling accuracy / state strip,
no choice-outcome scatter or ITI row) follow the manuscript's own Fig 7a
legend verbatim, which specifies exactly those three.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from .style import clean_axes

BLOCK_COLORS = ["#e91e63", "#9b59b6"]


def _state_order_present(states_order, present):
    return [s for s in states_order if s in present]


def plot_panel_a(df_example, state_colors, out_path, animal_name=""):
    """Fig 7a: representative session -- block structure (top), rolling
    accuracy over a five-trial window (middle), trial-by-trial 7-state
    classification (bottom).
    """
    d = df_example.sort_values("Trial").reset_index(drop=True)
    n = len(d)
    x = np.arange(n)

    fig = plt.figure(figsize=(7.5, 3.2))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.35, 1.4, 0.35], hspace=0.15)
    ax_block, ax_acc, ax_state = (fig.add_subplot(gs[i]) for i in range(3))

    # -- Row 1: block structure --
    blocks = d["BlockCount"].values
    for b in np.unique(blocks):
        idx = np.where(blocks == b)[0]
        color = BLOCK_COLORS[int(b) % 2]
        ax_block.barh(0, len(idx), left=idx[0], height=0.8, color=color, alpha=0.75)
        ax_block.text(idx.mean(), 0, f"{int(b)}", ha="center", va="center",
                      color="white", fontweight="bold", fontsize=6)
    ax_block.set_xlim(0, n)
    ax_block.set_yticks([])
    ax_block.set_ylabel("Block", fontsize=7, rotation=0, ha="right", va="center")
    ax_block.set_xticks([])
    for spine in ax_block.spines.values():
        spine.set_visible(False)

    # -- Row 2: rolling accuracy (same trailing 5-trial window used for classification) --
    ax_acc.plot(x, d["Rolling_Accuracy"], color="#e67e22", linewidth=1.0, zorder=3)
    ax_acc.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.7)
    ax_acc.axhline(0.8, color="gray", linestyle=":", linewidth=0.6, alpha=0.5)
    for b in np.unique(blocks)[1:]:
        trans_idx = np.where(blocks == b)[0][0]
        ax_acc.axvline(trans_idx - 0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_acc.set_xlim(0, n)
    ax_acc.set_ylim(0, 1.02)
    ax_acc.set_ylabel("rolling\naccuracy", fontsize=7)
    ax_acc.set_xticks([])
    clean_axes(ax_acc)

    # -- Row 3: trial-by-trial 7-state classification strip --
    for i, state in enumerate(d["Behavioral_State"]):
        color = state_colors.get(state, "#cccccc")
        ax_state.barh(0, 1, left=i, height=1.0, color=color)
    for b in np.unique(blocks)[1:]:
        trans_idx = np.where(blocks == b)[0][0]
        ax_state.axvline(trans_idx - 0.5, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax_state.set_xlim(0, n)
    ax_state.set_yticks([])
    ax_state.set_ylabel("state", fontsize=7, rotation=0, ha="right", va="center")
    ax_state.set_xlabel("trial", fontsize=8)
    for spine in ax_state.spines.values():
        spine.set_visible(False)

    present_states = [s for s in state_colors if s in set(d["Behavioral_State"])]
    handles = [Patch(color=state_colors[s], label=s) for s in present_states]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=6.5,
               frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(f"Representative session -- {animal_name}", fontsize=8, y=1.12)

    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_b(summary_alignment, summary_loseswitch, state_colors, out_path):
    """Fig 7b: each state positioned in the space defined by lose-switch
    probability (x) and Q-value alignment (y), +/- SEM in both directions.
    """
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    align = summary_alignment.set_index("Behavioral_State")
    ls = summary_loseswitch.set_index("Behavioral_State")
    states = [s for s in align.index if s in ls.index]

    for state in states:
        x, xe = ls.loc[state, "mean"], ls.loc[state, "sem"]
        y, ye = align.loc[state, "mean"], align.loc[state, "sem"]
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="o", color=state_colors.get(state, "#333"),
                    markersize=7, markeredgecolor="black", markeredgewidth=0.5,
                    capsize=2, elinewidth=1, label=state, zorder=3)

    ax.set_xlabel("lose-switch probability", fontsize=8)
    ax.set_ylabel("Q-value alignment", fontsize=8)
    ax.set_xlim(left=0)
    clean_axes(ax)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_c(composition_df, state_colors, out_path):
    """Fig 7c: state composition in the first trial following a fast versus
    slow block reversal (proportion of block transitions, stacked bar)."""
    fig, ax = plt.subplots(figsize=(2.2, 2.8))
    speeds = ["fast", "slow"]
    labels = {"fast": "Fast", "slow": "Slow"}
    all_states = list(dict.fromkeys(composition_df["post_adaptation_state"]))
    # Consistent stacking order matching the panel-a/e/f state ordering, restricted to
    # whichever non-Adaptation states are actually present here.
    order = [s for s in ["Exploit", "Directed Exploration", "Random Exploration", "Left Bias", "Right Bias"]
             if s in all_states]

    bottoms = np.zeros(len(speeds))
    for state in order:
        vals = []
        for speed in speeds:
            row = composition_df[(composition_df["adaptation_speed"] == speed) &
                                  (composition_df["post_adaptation_state"] == state)]
            vals.append(float(row["proportion"].iloc[0]) if len(row) else 0.0)
        ax.bar(range(len(speeds)), vals, bottom=bottoms, color=state_colors.get(state, "#ccc"),
               width=0.6, label=state, edgecolor="white", linewidth=0.5)
        bottoms += np.array(vals)

    ax.set_xticks(range(len(speeds)))
    ax.set_xticklabels([labels[s] for s in speeds], fontsize=8)
    ax.set_xlabel("Adaptation", fontsize=8)
    ax.set_ylabel("proportion of block transitions", fontsize=8)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    ax.set_title("Post-reversal state", fontsize=8, style="italic")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_d(probs, states, out_path):
    """Fig 7d: trial-to-trial state transition probability matrix (5x5,
    non-Adaptation states), row-normalized, pooled across animals/sessions."""
    n = len(states)
    fig, ax = plt.subplots(figsize=(4.0, 3.6))
    im = ax.imshow(probs, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("probability", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    for i in range(n):
        for j in range(n):
            v = probs[i, j]
            if np.isnan(v):
                continue
            color = "white" if v > 0.6 or v < 0.2 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color=color)

    labels = [s.replace(" ", "\n", 1) for s in states]
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("To state", fontsize=8)
    ax.set_ylabel("From state", fontsize=8)
    for k in range(n + 1):
        ax.axhline(k - 0.5, color="white", linewidth=0.8)
        ax.axvline(k - 0.5, color="white", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_e(win_summary, lose_summary, win_animal, lose_animal, states_order, out_path):
    """Fig 7e: win-stay / lose-switch probability by state (7 states x 2 bars)."""
    win = win_summary.set_index("Behavioral_State")
    lose = lose_summary.set_index("Behavioral_State")
    states = _state_order_present(states_order, set(win.index) | set(lose.index))

    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    x = np.arange(len(states))
    width = 0.35

    win_means = [win.loc[s, "mean"] if s in win.index else np.nan for s in states]
    win_sems = [win.loc[s, "sem"] if s in win.index else np.nan for s in states]
    lose_means = [lose.loc[s, "mean"] if s in lose.index else np.nan for s in states]
    lose_sems = [lose.loc[s, "sem"] if s in lose.index else np.nan for s in states]

    ax.bar(x - width / 2, win_means, width, yerr=win_sems, color="#34495e", label="win-stay",
           capsize=2, edgecolor="white", linewidth=0.5)
    ax.bar(x + width / 2, lose_means, width, yerr=lose_sems, color="#bdc3c7", label="lose-switch",
           capsize=2, edgecolor="white", linewidth=0.5)

    for i, s in enumerate(states):
        wa = win_animal[win_animal["Behavioral_State"] == s]["win_stay"].values
        la = lose_animal[lose_animal["Behavioral_State"] == s]["lose_switch"].values
        ax.scatter(np.full(len(wa), i - width / 2), wa, color="black", s=6, zorder=4)
        ax.scatter(np.full(len(la), i + width / 2), la, color="black", s=6, zorder=4)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(states, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("probability", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_f(session_df, summary_df, states_order, state_colors, out_path):
    """Fig 7f: violation rate by state -- session-level dots (colored by
    state), black-outlined circles = animal means."""
    states = _state_order_present(states_order, set(session_df["Behavioral_State"]))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    rng = np.random.default_rng(42)

    for i, s in enumerate(states):
        sdf = session_df[session_df["Behavioral_State"] == s]
        jitter = rng.uniform(-0.18, 0.18, size=len(sdf))
        ax.scatter(i + jitter, sdf["violation_rate"], color=state_colors.get(s, "#888"),
                   s=10, alpha=0.6, zorder=2)

        adf = summary_df[summary_df["Behavioral_State"] == s]
        animal_means = None
        # summary_df here is the animal-level table (one row per animal, state) --
        # plotted as black-outlined circles per the manuscript legend.
        if "Animal_Name" in summary_df.columns:
            animal_means = summary_df[summary_df["Behavioral_State"] == s]["violation_rate"]
        if animal_means is not None and len(animal_means):
            ax.scatter(np.full(len(animal_means), i), animal_means, facecolor="none",
                       edgecolor="black", linewidth=1.0, s=40, zorder=4)

    ax.set_xticks(range(len(states)))
    ax.set_xticklabels(states, fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("violation rate", fontsize=8)
    ax.set_ylim(0, 1.0)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_g(fit_results, out_path):
    """Fig 7g: violation rate vs. Q-value difference, linear fit, separately
    for Directed and Random Exploration."""
    colors = {"Directed Exploration": "#f39c12", "Random Exploration": "#7f8c8d"}
    fig, ax = plt.subplots(figsize=(3.2, 2.8))

    for state, result in fit_results.items():
        if result is None:
            continue
        color = colors.get(state, "black")
        binned = result["binned"]
        ax.errorbar(binned["x_mean"], binned["y_mean"], yerr=binned["y_sem"], fmt="o",
                    color=color, markersize=5, capsize=2, elinewidth=1, label=state, zorder=3)
        xs = np.linspace(0, binned["x_mean"].max(), 50)
        ax.plot(xs, result["intercept"] + result["slope"] * xs, color=color,
                linestyle="--", linewidth=1.2, zorder=2)

    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.6, alpha=0.6)
    ax.set_xlabel("|Q$_{right}$ - Q$_{left}$|", fontsize=8)
    ax.set_ylabel("violation rate", fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, frameon=False)
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_panel_h(occupancy_pct, state_colors, out_path):
    """Fig 7h: state occupancy (% of trials) per behavioral state for each
    individual animal, stacked bar."""
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    animals = list(occupancy_pct.index)
    states = list(occupancy_pct.columns)
    x = np.arange(len(animals))
    bottoms = np.zeros(len(animals))

    for state in states:
        vals = occupancy_pct[state].values
        ax.bar(x, vals, bottom=bottoms, color=state_colors.get(state, "#ccc"),
               width=0.6, label=state, edgecolor="white", linewidth=0.5)
        bottoms += vals

    ax.set_xticks(x)
    ax.set_xticklabels(animals, fontsize=8)
    ax.set_ylabel("% of trials", fontsize=8)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    clean_axes(ax)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
