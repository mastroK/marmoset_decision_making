"""Inter-trial interval (ITI) distributions and performance vs. ITI (Fig 1
stage 2: panels e-f).

Ported from _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb,
"CELL 11: Inter-Trial Interval (ITI) Analysis" (lines 1120-1385).

NOTE: an earlier, abandoned ITI calculation exists at lines 935-951 of the
source notebook, using a nonsensical MAX_ITI (6e15) left over from an
earlier draft -- superseded by CELL 11's version (MAX_ITI=1000 seconds) and
not ported here.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_iti(df):
    """Per-trial ITI (seconds) = this trial's start time minus the previous
    trial's start time, within each session. First trial of each session has
    no ITI and is dropped. Ported from lines 1126-1135."""
    d = df.sort_values(["Session_ID", "Trial"]).reset_index(drop=True).copy()
    d["Prev_StartTime"] = d.groupby("Session_ID")["AbsoluteTrialStartTime"].shift(1)
    d["ITI"] = (d["AbsoluteTrialStartTime"] - d["Prev_StartTime"]) / 1000
    return d.dropna(subset=["ITI"])


def filter_iti(df_timing, max_iti):
    """Drop implausible ITIs above max_iti seconds. Ported from lines 1137-1139."""
    return df_timing[df_timing["ITI"] <= max_iti].copy()


def iti_summary_by_animal(df_timing_filtered, animal_order):
    """Median/mean/std ITI per animal. Ported from lines 1229-1239."""
    from ..features.history_encoding import sort_animals

    records = []
    for animal in sort_animals(df_timing_filtered["Animal_Name"].unique(), animal_order):
        d = df_timing_filtered[df_timing_filtered["Animal_Name"] == animal]
        records.append({
            "animal": animal,
            "median_iti": d["ITI"].median(),
            "mean_iti": d["ITI"].mean(),
            "std_iti": d["ITI"].std(),
        })
    return pd.DataFrame(records)


def performance_by_iti_bin(df_timing_filtered, iti_bins, iti_labels, outcome_col="Outcome"):
    """Accuracy binned by ITI duration. Ported from lines 1271-1281."""
    d = df_timing_filtered.copy()
    d["ITI_bin"] = pd.cut(d["ITI"], bins=iti_bins, labels=iti_labels)
    perf = d.groupby("ITI_bin", observed=True)[outcome_col].agg(["mean", "sem", "count"]).reset_index()
    perf.columns = ["ITI_bin", "accuracy", "sem", "n"]
    return perf


def performance_by_iti_bin_by_animal(df_timing_filtered, iti_bins, iti_labels, outcome_col="Outcome"):
    """Per-animal accuracy binned by ITI duration -- an addition (not present
    in the original stage-2 port) needed to reproduce the source notebook's
    own publication figure for this panel, which overlays a per-animal line
    on top of the group-mean bars. Ported from the standalone "Performance
    vs ITI - Standalone with Individual Animals" cell (In[79], lines
    1446-1450: `animal_iti_performance`), which is the same per-ITI-bin
    accuracy already computed by `performance_by_iti_bin` above, just grouped
    by animal as well. Does not change any existing field/value -- purely
    additive."""
    d = df_timing_filtered.copy()
    d["ITI_bin"] = pd.cut(d["ITI"], bins=iti_bins, labels=iti_labels)
    perf = d.groupby(["Animal_Name", "ITI_bin"], observed=True)[outcome_col].agg(["mean", "count"]).reset_index()
    perf.columns = ["animal", "ITI_bin", "accuracy", "n"]
    return perf


def performance_by_long_break(df_timing_filtered, long_break_threshold, outcome_col="Outcome"):
    """Accuracy on trials following a "long break" (ITI above threshold) vs.
    a short one. Ported from lines 1316-1348.

    The source notebook's t-test compares raw trial-level Outcome values
    (short-ITI trials vs. long-ITI trials) -- these are not independent
    observations (all trials come from ~10 animals), so alongside that
    trial-level test we also aggregate to one short-break mean and one
    long-break mean per animal and run a paired Wilcoxon signed-rank test
    on those, matching the per-animal correction already applied to the
    other pseudoreplication-prone tests in this pipeline (Fig1 stage 1's
    session-bin test, Fig2's late-accuracy test).
    """
    d = df_timing_filtered.copy()
    d["Long_Break"] = d["ITI"] > long_break_threshold

    trial_level = d.groupby("Long_Break")[outcome_col].agg(["mean", "sem", "count"]).reset_index()
    trial_level.columns = ["Long_Break", "accuracy", "sem", "n"]

    short = d[~d["Long_Break"]][outcome_col]
    long = d[d["Long_Break"]][outcome_col]
    t_stat, p_trial_level = scipy_stats.ttest_ind(short, long)

    per_animal = d.groupby(["Animal_Name", "Long_Break"])[outcome_col].mean().unstack()
    per_animal = per_animal.dropna()
    if per_animal.shape[0] > 1 and per_animal.shape[1] == 2:
        w_stat, p_per_animal = scipy_stats.wilcoxon(per_animal[False], per_animal[True])
    else:
        w_stat, p_per_animal = np.nan, np.nan

    return {
        "trial_level": trial_level,
        "t_stat_trial_level": t_stat,
        "p_trial_level": p_trial_level,
        "per_animal_short_mean": per_animal[False].mean() if per_animal.shape[0] else np.nan,
        "per_animal_long_mean": per_animal[True].mean() if per_animal.shape[0] else np.nan,
        "n_animals_paired": int(per_animal.shape[0]),
        "wilcoxon_stat_per_animal": w_stat,
        "p_per_animal": p_per_animal,
    }


# ---------------------------------------------------------------------------
# Reviewer-response additions (RA3_iti_and_transitions.ipynb, R1-C3, R2-4).
# `df_timing` throughout below is `compute_iti`'s own output (already has
# ITI, Session_ID, Animal_Name) additionally carrying `Behavioral_State`
# (from a `classify_state_v7` call) and `Outcome_Binary`/`Trial` -- i.e. the
# canonical state-classified dataframe, run through `compute_iti` first.
# ---------------------------------------------------------------------------

def iti_by_state(df_timing, state_col="Behavioral_State", animal_col="Animal_Name"):
    """ITI distribution by behavioral state, and post-win vs. post-loss ITI
    within each state. Per this pipeline's standing pseudoreplication
    convention: session-level means first, then animal-level means, then
    the reported summary is mean +/- SEM ACROSS ANIMALS.
    """
    d = df_timing.sort_values([animal_col, "Session_ID", "Trial"]).copy()
    d["Prev_Outcome_Binary"] = d.groupby("Session_ID")["Outcome_Binary"].shift(1)

    session_state = (
        d.groupby([animal_col, "Session_ID", state_col])["ITI"].mean().reset_index()
    )
    animal_state = session_state.groupby([animal_col, state_col])["ITI"].mean().reset_index()
    state_summary = animal_state.groupby(state_col)["ITI"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n="count",
    ).reset_index()

    d_valid = d.dropna(subset=["Prev_Outcome_Binary"])
    session_winloss = (
        d_valid.groupby([animal_col, "Session_ID", state_col, "Prev_Outcome_Binary"])["ITI"]
        .mean().reset_index()
    )
    animal_winloss = (
        session_winloss.groupby([animal_col, state_col, "Prev_Outcome_Binary"])["ITI"]
        .mean().unstack("Prev_Outcome_Binary").rename(columns={0.0: "post_loss", 1.0: "post_win"})
        .reset_index()
    )
    winloss_summary_rows = []
    for state, sdf in animal_winloss.groupby(state_col):
        sdf = sdf.dropna(subset=["post_win", "post_loss"])
        if len(sdf) > 1:
            stat, p = scipy_stats.wilcoxon(sdf["post_win"], sdf["post_loss"])
        else:
            stat, p = np.nan, np.nan
        winloss_summary_rows.append({
            state_col: state,
            "post_win_mean": float(sdf["post_win"].mean()) if len(sdf) else np.nan,
            "post_loss_mean": float(sdf["post_loss"].mean()) if len(sdf) else np.nan,
            "n_animals": int(len(sdf)),
            "wilcoxon_stat": float(stat) if pd.notna(stat) else None,
            "wilcoxon_p": float(p) if pd.notna(p) else None,
        })

    return {
        "state_summary": state_summary,
        "animal_state": animal_state,
        "post_win_vs_post_loss_by_state": pd.DataFrame(winloss_summary_rows),
        "animal_winloss_by_state": animal_winloss,
    }


def iti_preceding_transitions(df_timing, state_col="Behavioral_State", animal_col="Animal_Name"):
    """ITI on trials immediately preceding a behavioral-state transition
    (current trial's state differs from the previous trial's) vs. ITI on
    trials that stay within the same state as the previous trial. Same
    per-animal-then-across-animals summary convention as `iti_by_state`.
    """
    d = df_timing.sort_values([animal_col, "Session_ID", "Trial"]).copy()
    d["Prev_State"] = d.groupby("Session_ID")[state_col].shift(1)
    d = d.dropna(subset=["Prev_State"])
    d["Is_State_Transition"] = d[state_col] != d["Prev_State"]

    session_level = (
        d.groupby([animal_col, "Session_ID", "Is_State_Transition"])["ITI"].mean().reset_index()
    )
    animal_level = (
        session_level.groupby([animal_col, "Is_State_Transition"])["ITI"]
        .mean().unstack("Is_State_Transition").rename(columns={True: "preceding_transition", False: "within_state"})
        .dropna()
    )
    if animal_level.shape[0] > 1:
        stat, p = scipy_stats.wilcoxon(animal_level["preceding_transition"], animal_level["within_state"])
    else:
        stat, p = np.nan, np.nan

    return {
        "session_level": session_level,
        "animal_level": animal_level.reset_index(),
        "preceding_transition_mean": float(animal_level["preceding_transition"].mean()),
        "within_state_mean": float(animal_level["within_state"].mean()),
        "n_animals": int(animal_level.shape[0]),
        "wilcoxon_stat": float(stat) if pd.notna(stat) else None,
        "wilcoxon_p": float(p) if pd.notna(p) else None,
    }
