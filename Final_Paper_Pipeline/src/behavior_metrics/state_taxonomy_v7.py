"""Shared analysis helpers for the manuscript's canonical 7-state taxonomy
(`state_classifier.classify_state_v7`), used by both Fig 7 (single 80-20
condition) and the Fig 8 rebuild (80-20 vs. 100-0 comparison).

DISCLOSED DEVIATION FROM THE TASK'S FILE LIST: the task brief names
`state_classifier.py` (additive), `cross_condition.py`, `config.yaml`,
`notebooks/Fig7.ipynb`, `notebooks/Fig8.ipynb`, `src/plotting/fig7.py` (new)
and `src/plotting/fig8.py` (rewrite) as the only files to touch. None of
those is a natural home for the non-classifier, non-plotting analysis code
Fig 7's panels b/c/d/e/f/g/h need (Q-value alignment, pooled transition
matrices, win-stay/lose-switch by state, violation rate by state, state
occupancy, post-adaptation state composition) -- `state_classifier.py` is
scoped to classifiers only (per its own module docstring), and
`cross_condition.py` is scoped to the 80-20-vs-100-0 reconstruction, not
general single-condition state analysis. This project's own established
architecture already separates classifier logic from figure-support
analysis logic this way (see `state_clustering.py`, which plays exactly
this role for Fig 6/Supp Fig 4's K-Means/HMM analysis, alongside
`state_classifier.py`'s classifiers) -- this module is the same pattern
applied to the 7-state taxonomy. Flagged here explicitly rather than
silently adding an unlisted file.
"""

import numpy as np
import pandas as pd
from scipy import stats


def block_transitions_grouped(df):
    """A dict {(Session_ID, BlockCount): sub-dataframe}, built once, reused
    across the block-level helpers below to avoid re-filtering the full
    dataframe per block."""
    return {key: g for key, g in df.groupby(["Session_ID", "BlockCount"], sort=False, observed=True)}


def post_adaptation_state_composition(df, block_labels, adaptation_window):
    """Fig 7 panel c. For each block transition (one row of `block_labels`,
    from `state_classifier.label_block_adaptation_speed`), find the state
    classification of the first trial AFTER the adaptation window closes
    (`Trials_Since_Transition == adaptation_window` -- the first trial the
    ordinary Exploit/Exploration/Bias rules, rather than Adaptation, ever
    apply to within that block). Manuscript Results text: "In block
    transitions during which animals quickly adapted to the new contingency
    (Fast Adaptation)... most often transitioned directly into
    Exploitation... In block transitions during which adaptation was slower
    (Slow adaptation)... rarely transitioned directly into Exploitation,
    instead entering Directed Exploration" -- i.e. this panel is about the
    state reached immediately after the adaptation period, not the state
    label carried by the adaptation window's own trials (which is always
    literally 'Adapting (fast/slow)' by construction).

    Returns (per_block_df, composition_df) -- per_block_df has one row per
    block transition with its post-adaptation state (blocks that end before
    reaching `adaptation_window` trials post-transition are dropped, noted
    in the caller's printed count); composition_df has one row per
    (adaptation_speed, post_adaptation_state) with the proportion of that
    speed group's block transitions landing in that state.
    """
    grouped = block_transitions_grouped(df)
    records = []
    for row in block_labels.itertuples():
        block_df = grouped.get((row.Session_ID, row.BlockCount))
        if block_df is None:
            continue
        post_trial = block_df[block_df["Trials_Since_Transition"] == adaptation_window]
        if post_trial.empty:
            continue
        records.append({
            "Session_ID": row.Session_ID, "BlockCount": row.BlockCount,
            "Animal_Name": row.Animal_Name, "adaptation_speed": row.adaptation_speed,
            "post_adaptation_state": post_trial["Behavioral_State"].iloc[0],
        })
    per_block_df = pd.DataFrame(records)

    comp = (
        per_block_df.groupby(["adaptation_speed", "post_adaptation_state"]).size()
        .rename("n").reset_index()
    )
    totals = per_block_df.groupby("adaptation_speed").size().rename("n_total")
    comp = comp.merge(totals, on="adaptation_speed")
    comp["proportion"] = comp["n"] / comp["n_total"]
    return per_block_df, comp


def q_value_alignment_by_state(df_q, states, min_valid_trials=10):
    """Fig 7 panel b (y-axis). "Q-value alignment" -- per session, per
    state: P(chose right | Q_right > Q_left) minus P(chose right | Q_left >
    Q_right), restricted to trials with both Q-values and a choice recorded.
    This is the same construction as the orphaned
    `07c_FinalClassifier_Fig6.ipynb`'s `compute_q_alignment_marmoset` helper
    (_source_archive/by_figure/Fig7/07c_FinalClassifier_Fig6.ipynb, cell 10)
    -- reused here as the definition consistent with this project's own
    prior exploration of this exact figure, corroborating the manuscript's
    "Q-value alignment" terminology (ranges 0 = choices uncorrelated with
    which side has the higher fitted Q-value, to 1 = choice always tracks
    the higher-Q side).

    Per this pipeline's standing pseudoreplication-correction convention:
    computed per session, averaged to one value per (animal, state), then
    the reported summary is the mean +/- SEM ACROSS ANIMALS (n up to 5),
    with the per-session/per-animal tables also returned for transparency.
    """
    records = []
    for (animal, session, state), g in df_q.groupby(
        ["Animal_Name", "Session_ID", "Behavioral_State"], sort=False, observed=True
    ):
        if state not in states:
            continue
        valid = g.dropna(subset=["Q_left", "Q_right", "Choice_Binary"])
        if len(valid) < min_valid_trials:
            continue
        higher_right = valid["Q_right"] > valid["Q_left"]
        if higher_right.sum() < 2 or (~higher_right).sum() < 2:
            continue
        p_r_qr = valid.loc[higher_right, "Choice_Binary"].mean()
        p_r_ql = valid.loc[~higher_right, "Choice_Binary"].mean()
        records.append({
            "Animal_Name": animal, "Session_ID": session, "Behavioral_State": state,
            "alignment": p_r_qr - p_r_ql, "n_trials": int(len(valid)),
        })
    session_df = pd.DataFrame(records)
    if session_df.empty:
        return session_df, pd.DataFrame(), pd.DataFrame()

    animal_df = session_df.groupby(["Animal_Name", "Behavioral_State"])["alignment"].mean().reset_index()
    summary = animal_df.groupby("Behavioral_State")["alignment"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n="count",
    ).reset_index()
    return session_df, animal_df, summary


def lose_switch_probability_by_state(df, states, min_trials=3):
    """Fig 7 panel b (x-axis) and Fig 7f's companion lose-switch axis.
    Lose-switch: among trials of a given state where the outcome was
    unrewarded, P(the very next trial is a physical switch). State is
    assigned by the OUTCOME trial (the trial that just received the loss),
    matching this pipeline's win-stay/lose-shift convention elsewhere
    (`wsls.py`) of pairing an outcome trial with the following trial's
    stay/switch decision. Per-session, then per-animal, then across-animal
    mean +/- SEM (n animals).
    """
    d = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).copy()
    d["Next_Switch"] = d.groupby(["Animal_Name", "Session_ID"])["PhysicalSwitch"].shift(-1)
    d = d[d["Behavioral_State"].isin(states) & (d["Outcome_Binary"] == 0)]
    d = d.dropna(subset=["Next_Switch"])

    session_df = (
        d.groupby(["Animal_Name", "Session_ID", "Behavioral_State"])["Next_Switch"]
        .agg(lose_switch="mean", n_trials="count").reset_index()
    )
    session_df = session_df[session_df["n_trials"] >= min_trials]
    animal_df = session_df.groupby(["Animal_Name", "Behavioral_State"])["lose_switch"].mean().reset_index()
    summary = animal_df.groupby("Behavioral_State")["lose_switch"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n="count",
    ).reset_index()
    return session_df, animal_df, summary


def win_stay_lose_switch_by_state(df, states, min_trials=3):
    """Fig 7 panel e / Fig 8 panel d. Win-stay: P(this trial is a stay |
    the PREVIOUS trial in this animal/session's OWN state-filtered
    subsequence was rewarded). Lose-switch: P(this trial is a switch |
    that previous state-filtered trial was unrewarded).

    CORRECTED (2026-09-17) to match the real source cells
    (09b_CrossProb_StickyModel_v1.ipynb, cells 36-37) exactly, after a
    direct comparison against Fig 8's own real reconstructed data showed
    the previous version of this function silently washed out a large,
    real effect. The previous version filtered to `states` AFTER computing
    `Next_Switch = PhysicalSwitch.shift(-1)` on the full chronological
    per-session sequence, and conditioned on the state-labeled trial's OWN
    `Outcome_Binary` -- i.e. "given this state-labeled trial's own outcome,
    does the NEXT chronological trial switch". The source instead filters
    to EACH state FIRST, then computes `Prev_Outcome` as a shift(1) WITHIN
    that per-state-filtered subsequence (so "previous" means the previous
    trial that was ALSO classified this state -- gaps from other-state
    trials are closed, not treated as missing), then uses the filtered
    row's own `PhysicalSwitch`. Verified: Exploit/100-0 lose-switch is
    0.385 with this corrected method (matching the published figure) vs.
    0.362 with the old method; Exploit/80-20 is 0.071 (corrected) vs. 0.249
    (old) -- the old method reported "ns" where the real effect is ~5x.

    Unlike the old version, there is no per-session averaging step here
    (the source pools all of an animal's state-filtered trials directly,
    not session-by-session) -- per-animal, then across-animal mean +/- SEM.
    `min_trials` (per animal, per win/lose bucket) is this pipeline's own
    safeguard against an unstable estimate from very few trials, not
    something the source itself applies at this granularity.
    """
    win_rows, lose_rows = [], []
    for state in states:
        sdf = df[df["Behavioral_State"] == state].copy()
        if sdf.empty:
            continue
        sdf = sdf.sort_values(["Animal_Name", "Session_ID", "Trial"])
        sdf["Prev_Outcome"] = sdf.groupby(["Animal_Name", "Session_ID"])["Outcome_Binary"].shift(1)
        sdf = sdf.dropna(subset=["PhysicalSwitch"])

        win_trials = sdf[sdf["Prev_Outcome"] == 1]
        lose_trials = sdf[sdf["Prev_Outcome"] == 0]

        win_counts = win_trials.groupby("Animal_Name").size()
        lose_counts = lose_trials.groupby("Animal_Name").size()
        ws = (1 - win_trials.groupby("Animal_Name")["PhysicalSwitch"].mean())[win_counts >= min_trials]
        ls = lose_trials.groupby("Animal_Name")["PhysicalSwitch"].mean()[lose_counts >= min_trials]

        for animal, v in ws.items():
            win_rows.append({"Animal_Name": animal, "Behavioral_State": state, "win_stay": v})
        for animal, v in ls.items():
            lose_rows.append({"Animal_Name": animal, "Behavioral_State": state, "lose_switch": v})

    win_animal = pd.DataFrame(win_rows, columns=["Animal_Name", "Behavioral_State", "win_stay"])
    lose_animal = pd.DataFrame(lose_rows, columns=["Animal_Name", "Behavioral_State", "lose_switch"])

    def _summarize(animal_df, col):
        if animal_df.empty:
            return pd.DataFrame(columns=["Behavioral_State", "mean", "sem", "n"])
        return animal_df.groupby("Behavioral_State")[col].agg(
            mean="mean",
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
            n="count",
        ).reset_index()

    win_summary = _summarize(win_animal, "win_stay")
    lose_summary = _summarize(lose_animal, "lose_switch")
    return {
        "win_animal": win_animal, "lose_animal": lose_animal,
        "win_summary": win_summary, "lose_summary": lose_summary,
    }


def violation_rate_by_state(df_valid, states):
    """Fig 7 panel f. Violation rate = proportion of trials on which the
    animal chose the LOWER model-estimated Q-value option, restricted to
    trials with a defined higher/lower Q-value (`df_valid` = the output of
    `q_following_model.add_chose_higher_q`, which already excludes
    `Q_diff == 0` ties). Session-level estimates (dots) plus per-animal
    means (circles), per state, matching the manuscript legend exactly.
    """
    d = df_valid[df_valid["Behavioral_State"].isin(states)].copy()
    d["Violation"] = 1.0 - d["Chose_Higher_Q"]
    session_df = (
        d.groupby(["Animal_Name", "Session_ID", "Behavioral_State"])["Violation"]
        .agg(violation_rate="mean", n_trials="count").reset_index()
    )
    animal_df = session_df.groupby(["Animal_Name", "Behavioral_State"])["violation_rate"].mean().reset_index()
    summary = animal_df.groupby("Behavioral_State")["violation_rate"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n="count",
    ).reset_index()
    return session_df, animal_df, summary


def violation_vs_qdiff_by_state(df_valid, states, n_bins=10):
    """Fig 7 panel g. Per state (Directed vs. Random Exploration),
    violation rate (1 - Chose_Higher_Q) as a function of |Q_right -
    Q_left|, binned, with a linear regression fit (violation rate on the
    bin midpoints, and separately on raw trial-level data for the reported
    slope/intercept/p-value).
    """
    out = {}
    for state in states:
        d = df_valid[df_valid["Behavioral_State"] == state].copy()
        d["Violation"] = 1.0 - d["Chose_Higher_Q"]
        d["Abs_Q_Diff"] = d["Q_diff"].abs()
        if len(d) < n_bins:
            out[state] = None
            continue

        slope, intercept, r, p, se = stats.linregress(d["Abs_Q_Diff"], d["Violation"])

        bins = np.linspace(0, d["Abs_Q_Diff"].quantile(0.99), n_bins + 1)
        d["bin"] = pd.cut(d["Abs_Q_Diff"], bins=bins, include_lowest=True)
        binned = d.groupby("bin", observed=True).agg(
            x_mean=("Abs_Q_Diff", "mean"),
            y_mean=("Violation", "mean"),
            y_sem=("Violation", lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan),
            n=("Violation", "count"),
        ).reset_index(drop=True)

        out[state] = {
            "slope": float(slope), "intercept": float(intercept),
            "r": float(r), "p": float(p), "se": float(se),
            "n_trials": int(len(d)), "binned": binned,
        }
    return out


def pooled_transition_matrix(df, states):
    """Fig 7 panel d / Fig 8's transition-probability inputs. Trial-to-trial
    state transition probability matrix, computed from ALL consecutive
    trial pairs pooled across animals and sessions (per the manuscript's
    own Fig 7d legend -- explicitly a pooled, not per-animal-corrected,
    quantity), row-normalized.
    """
    d = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).copy()
    d["next_state"] = d.groupby(["Animal_Name", "Session_ID"])["Behavioral_State"].shift(-1)
    d = d.dropna(subset=["next_state"])

    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    counts = np.zeros((n, n))
    grouped_counts = d.groupby(["Behavioral_State", "next_state"]).size()
    for (fs, ts), c in grouped_counts.items():
        if fs in idx and ts in idx:
            counts[idx[fs], idx[ts]] += c

    row_sums = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        probs = np.where(row_sums > 0, counts / row_sums, np.nan)
    return probs, counts


def state_occupancy_by_animal(df, states, animal_order=None):
    """Fig 7 panel h. % of trials (pooled across sessions) per behavioral
    state, for each individual animal.
    """
    d = df[df["Behavioral_State"].isin(states)].copy()
    counts = d.groupby(["Animal_Name", "Behavioral_State"]).size().unstack("Behavioral_State", fill_value=0)
    counts = counts.reindex(columns=states, fill_value=0)
    pct = counts.div(counts.sum(axis=1), axis=0) * 100.0
    if animal_order is not None:
        present = [a for a in animal_order if a in pct.index]
        pct = pct.reindex(present)
    return pct


def state_occupancy_animal_means(df, states):
    """Fig 8 panel c inputs / group-level occupancy summary: per-animal
    proportion of trials (pooled across that animal's own sessions) in
    each state, restricted to `states` (the denominator is trials in
    `states` only -- callers wanting Adapting excluded should pass
    `STATES_V7_NON_ADAPTING`).
    """
    d = df[df["Behavioral_State"].isin(states)].copy()
    counts = d.groupby(["Animal_Name", "Behavioral_State"]).size().unstack("Behavioral_State", fill_value=0)
    counts = counts.reindex(columns=states, fill_value=0)
    prop = counts.div(counts.sum(axis=1), axis=0)
    return prop


def _binary_entropy_bits(p):
    """Shannon entropy, in bits, of a Bernoulli variable with P(=1)=p.
    0 at p in {0, 1} (a deterministic choice carries no entropy); NaN input
    passes through as NaN.
    """
    if pd.isna(p) or p <= 0.0 or p >= 1.0:
        return 0.0
    return float(-p * np.log2(p) - (1.0 - p) * np.log2(1.0 - p))


def state_behavioral_metrics_by_state(df, states, min_trials=5):
    """Supp Fig 4 panel d. Four per-state behavioral metrics, computed on
    `state_features.build_state_features_df`'s own trial-level columns:
      - P(high-prob choice)  = mean(High_Prob)        (top-left)
      - switch rate          = mean(PhysicalSwitch)   (top-right)
      - P(right)             = mean(Choice_Binary)    (bottom-left)
      - choice entropy (bits) = binary entropy of that same (animal,
        session, state)'s P(right)                     (bottom-right)

    Computed per (Animal_Name, Session_ID, Behavioral_State) first (a
    session/state pair with fewer than `min_trials` trials is dropped, to
    avoid a degenerate single-trial 0.0/1.0 estimate feeding the entropy
    calculation in particular). The manuscript legend states the
    aggregation directly -- "Bars indicate the mean across all animals;
    dots represent individual session averages" -- so unlike most of this
    project's other by-state panels (which plot per-ANIMAL dots, e.g. Fig 7
    panel f's session-dots-plus-animal-mean-circles), this panel's dots are
    the raw per-session values themselves, not first collapsed to one point
    per animal. The bars still follow this project's standing per-animal-
    then-across-animal convention (mean of animal-level means, not a pooled
    session-level mean), so a heavily-sampled animal can't dominate the bar
    height -- consistent with every other summary statistic in this
    pipeline, just with session-level (not animal-level) dots layered on
    top per this panel's own specific legend wording.
    """
    d = df[df["Behavioral_State"].isin(states)].copy()

    session_df = (
        d.groupby(["Animal_Name", "Session_ID", "Behavioral_State"])
        .agg(
            p_high_prob=("High_Prob", "mean"),
            switch_rate=("PhysicalSwitch", "mean"),
            p_right=("Choice_Binary", "mean"),
            n_trials=("High_Prob", "count"),
        )
        .reset_index()
    )
    session_df = session_df[session_df["n_trials"] >= min_trials].copy()
    session_df["entropy_bits"] = session_df["p_right"].apply(_binary_entropy_bits)

    metrics = ["p_high_prob", "switch_rate", "p_right", "entropy_bits"]
    animal_df = session_df.groupby(["Animal_Name", "Behavioral_State"])[metrics].mean().reset_index()

    def _summarize(col):
        return animal_df.groupby("Behavioral_State")[col].agg(
            mean="mean",
            sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
            n="count",
        ).reset_index()

    summary = {m: _summarize(m) for m in metrics}
    return session_df, animal_df, summary
