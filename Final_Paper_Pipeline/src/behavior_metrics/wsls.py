"""Win-stay / lose-shift rates.

Ported from _source_archive/by_figure/Fig2/05_Fig2_Reversal_v2.ipynb, lines
2085-2126. Computed per session (win-stay = P(stay | previous trial
rewarded), lose-shift = P(switch | previous trial unrewarded)), then
averaged per animal. This is a single pooled calculation across the whole
dataset passed in -- it is NOT split by cued/uncued transition type in the
source notebook, so callers wanting a cued/uncued breakdown should call
this separately on pre-filtered subsets.
"""

import numpy as np
import pandas as pd
from scipy import stats


def win_stay_lose_shift_by_session(df):
    """df must have Session_ID, Animal_Name, Block, Trial, Outcome,
    PhysicalSwitch. Returns one row per session: session_id, animal,
    win_stay, lose_shift, n_win, n_lose.
    """
    d = df[df["PhysicalSwitch"].notna()].copy()
    d = d.sort_values(["Session_ID", "Block", "Trial"]).reset_index(drop=True)
    d["Prev_Outcome"] = d.groupby("Session_ID")["Outcome"].shift(1)
    d = d[d["Prev_Outcome"].notna()].copy()

    records = []
    for session_id in d["Session_ID"].unique():
        session_data = d[d["Session_ID"] == session_id]
        animal = session_data["Animal_Name"].iloc[0]

        win_trials = session_data[session_data["Prev_Outcome"] == 1]
        win_stay_rate = (win_trials["PhysicalSwitch"] == 0).mean() if len(win_trials) > 0 else np.nan

        lose_trials = session_data[session_data["Prev_Outcome"] == 0]
        lose_shift_rate = (lose_trials["PhysicalSwitch"] == 1).mean() if len(lose_trials) > 0 else np.nan

        records.append({
            "session_id": session_id,
            "animal": animal,
            "win_stay": win_stay_rate,
            "lose_shift": lose_shift_rate,
            "n_win": len(win_trials),
            "n_lose": len(lose_trials),
        })

    return pd.DataFrame(records)


def summarize_by_animal(wsls_sessions_df):
    """Per-animal mean +/- SEM of win_stay/lose_shift across that animal's sessions."""
    records = []
    for animal, animal_data in wsls_sessions_df.groupby("animal"):
        records.append({
            "animal": animal,
            "win_stay_mean": animal_data["win_stay"].mean(),
            "win_stay_sem": stats.sem(animal_data["win_stay"].dropna()) if animal_data["win_stay"].notna().sum() > 1 else np.nan,
            "lose_shift_mean": animal_data["lose_shift"].mean(),
            "lose_shift_sem": stats.sem(animal_data["lose_shift"].dropna()) if animal_data["lose_shift"].notna().sum() > 1 else np.nan,
            "n_sessions": len(animal_data),
        })
    return pd.DataFrame(records)


def win_stay_lose_switch_by_state_session(df, state_col, states, min_trials,
                                            session_col="Session_ID", animal_col="Animal_Name",
                                            accuracy_col="Rolling_Accuracy"):
    """State-conditioned win-stay / lose-switch, one row per (session, state)
    pair -- for Fig 6 panel f ("Win-stay versus lose-switch probability for
    Exploitation and Exploration trials. Each point represents one session
    colored by rolling accuracy within state.").

    Not in the source notebook (07b_KMEANS_Final.ipynb's own win-stay/
    lose-shift cell, reused unmodified above for Fig 2/3, is a single
    pooled calculation with no state conditioning) -- this is new
    computation for this pipeline, built by restricting
    `win_stay_lose_shift_by_session`'s same PhysicalSwitch/Outcome logic to
    only the CURRENT trial's state. The previous-trial outcome used to
    decide "win" vs. "lose" (and PhysicalSwitch itself) still comes from
    the FULL, unfiltered per-session trial sequence already carried by `df`
    (both are already trial-to-trial, session-sorted columns from
    state_features.build_state_features_df) -- state conditioning only
    decides which state's bucket the CURRENT trial's outcome is counted
    into; it does not require the previous trial to share that state.

    `min_trials` drops (session, state) rows with too few trials to give a
    stable estimate (a judgment call, not tuned to any manuscript number --
    without it a session with e.g. 1 Exploration trial can only ever show
    win_stay/lose_switch of exactly 0.0 or 1.0).
    """
    d = df[df["PhysicalSwitch"].notna()].sort_values([session_col, "Trial"]).copy()
    d["Prev_Outcome"] = d.groupby(session_col)["Outcome_Binary"].shift(1)
    d = d[d["Prev_Outcome"].notna()]
    d = d[d[state_col].isin(states)]

    records = []
    for (session_id, state), sdf in d.groupby([session_col, state_col]):
        if len(sdf) < min_trials:
            continue
        win_trials = sdf[sdf["Prev_Outcome"] == 1]
        lose_trials = sdf[sdf["Prev_Outcome"] == 0]
        if len(win_trials) == 0 or len(lose_trials) == 0:
            continue
        records.append({
            "session_id": session_id,
            "animal": sdf[animal_col].iloc[0],
            "state": state,
            "win_stay": float((win_trials["PhysicalSwitch"] == 0).mean()),
            "lose_switch": float((lose_trials["PhysicalSwitch"] == 1).mean()),
            "rolling_accuracy_in_state": float(sdf[accuracy_col].mean()),
            "n_trials": int(len(sdf)),
            "n_win": int(len(win_trials)),
            "n_lose": int(len(lose_trials)),
        })
    return pd.DataFrame(records)


def summarize_pooled(wsls_sessions_df):
    """Single pooled mean +/- SEM across ALL sessions (all animals combined) --
    this is the number the manuscript reports (e.g. Fig 2d: 0.852+/-0.026 /
    0.401+/-0.023), not a per-animal breakdown."""
    ws = wsls_sessions_df["win_stay"].dropna()
    ls = wsls_sessions_df["lose_shift"].dropna()
    return {
        "win_stay_mean": ws.mean(),
        "win_stay_sem": stats.sem(ws),
        "lose_shift_mean": ls.mean(),
        "lose_shift_sem": stats.sem(ls),
        "n_sessions_win": len(ws),
        "n_sessions_lose": len(ls),
    }
