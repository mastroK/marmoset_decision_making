"""Shared preprocessing + derived features for the behavioral-state-taxonomy
figures (Fig 6, Fig 8, Supp Fig 4).

This reconstructs the columns of the black-box intermediate file
`df_with_states.csv` (see _source_archive/INDEX.md -- no generating script
for that exact file was ever found on Drive or in git). The columns it
needs (Choice_Binary, Outcome_Binary, High_Prob_Is_Right, Rolling_Accuracy,
Rolling_PRight, Expected_PRight, Rolling_Switch_Rate) are the same ones
computed by _source_archive/by_figure/SuppFig3/SuppFig3_candidate_07_GLM_HMM_v1_cleaned.ipynb's
STEP 3/7/7B (lines 104-378) -- that notebook's preprocessing is reused here
(independently from src/behavior_metrics/q_following_model.py's smaller
subset of the same steps, which only needed Choice_Binary/Outcome_Binary/
Trial_in_Block for Supp Fig 3's own panels).

Also fixes one of the REVIEWER_AUDIT.md-documented defects: the source
notebooks computed the same absolute-deviation quantity under two different
names (`Abs_Deviation` in 07b_KMEANS_Final.ipynb, `Choice_Deviation` in the
07_GLM_HMM preprocessing notebook -- both equal to
abs(Rolling_PRight - Expected_PRight)), then inconsistently fed the K-Means
step ['Signed_Deviation','Choice_Deviation',...] but the HMM step
['Abs_Deviation','Choice_Deviation',...] (i.e. the same column twice, and a
different feature set from K-Means). Here there is only one such column,
`Choice_Deviation`, and both K-Means and HMM are given the same 4-feature
set (see config.yaml's `state_taxonomy.kmeans_features` /
`hmm_features` -- both point at the same list).

CORRECTED (2026-09-13): `Rolling_Accuracy` -- reward-based, not choice-based.
This project's own source notebooks disagree on what "Rolling_Accuracy"
means, and this function originally ported the WRONG one for state
CLASSIFICATION purposes (it still matches the notebook cited above for
everything else). Three notebooks were checked directly:
  - `07_GLM_HMM_v1_cleaned.ipynb` (Supp Fig 3's source, cited above for
    this function's preprocessing steps in general) computes
    `Rolling_Accuracy` as the rolling mean of `High_Prob` (P(chose the
    high-probability side)) -- choice-based. This function originally
    copied that.
  - `09b_CrossProb_StickyModel_v1.ipynb` (Fig 7/Fig 8's actual source,
    STEP 7B) and `07c_FinalClassifier_Fig6.ipynb` (the orphaned notebook
    with the fast/slow Adaptation split Fig 7 is built on, STEP 7B) BOTH
    instead compute `Rolling_Accuracy` as the rolling mean of
    `Outcome_Binary` (P(the trial was actually REWARDED)) -- reward-based,
    not choice-based. 07c's own comment is explicit: "keep as reward-
    based, thresholds were tuned to this."
This matters because `classify_state_v7`/`classify_state_updated`'s
Random Exploration branch (`Rolling_Accuracy < 0.4`) and the Adapting
fast/slow split (`accuracy >= 0.6`) are exactly the "thresholds" that
comment refers to. With the choice-based version, `Rolling_Accuracy` is
tightly algebraically coupled to `Choice_Deviation` (they become the
same event whenever `High_Prob_Is_Right` is stable), so the Bias branch
(checked first) swallows essentially every trial that could ever qualify
as Random Exploration -- confirmed to reproduce ZERO Random Exploration
trials, vs. 09b's own recorded output of 1,455 trials (4.0%) on the same
80-20 data, and vs. the manuscript's own Results text, which reports real,
non-trivial per-state statistics for Random Exploration. Reward-based
accuracy is genuinely noisy at a 5-trial window even when choice behavior
is stable, so it isn't mechanically tied to `Choice_Deviation` the same
way -- this restores a non-empty, real Random Exploration category. Fixed
here to reward-based, matching 09b/07c (the confirmed classification-figure
sources) rather than 07_GLM_HMM_v1_cleaned (a Q-following-validation
notebook whose own state-classification cell is not actually exercised by
anything in that notebook's own downstream outputs). Both versions are kept
(`Rolling_Accuracy` = reward-based, `Rolling_Accuracy_High_Prob` = choice-
based), matching 09b/07c's own convention of computing both side by side.
"""

import numpy as np
import pandas as pd


def _code_choice(physical_choice):
    if physical_choice in ("BottomLeft", "LeftCenter", "TopLeft"):
        return 0
    elif physical_choice in ("BottomRight", "RightCenter", "TopRight"):
        return 1
    return np.nan


def build_state_features_df(df, task_type, prob_condition, rolling_window):
    """Filter to the task/probability condition and add every column the
    state classifier and its K-Means/HMM validation need.
    """
    df_filtered = df[
        (df["Task_Type"] == task_type) & (df["prob_Condition"] == prob_condition)
    ].copy()
    df_filtered = df_filtered.sort_values(
        ["Animal_Name", "Session_ID", "Trial"]
    ).reset_index(drop=True)

    # Choice_Binary, incl. vertical-session Top/Bottom recoding
    df_filtered["Choice_Binary"] = df_filtered["PhysicalChoice"].map(_code_choice)
    session_sides = df_filtered.groupby("Session_ID").apply(
        lambda x: pd.Series({
            "has_left": x["PhysicalChoice"].str.contains("Left").any(),
            "has_right": x["PhysicalChoice"].str.contains("Right").any(),
        })
    ).reset_index()
    vertical_sessions = session_sides[
        ~session_sides["has_left"] | ~session_sides["has_right"]
    ]["Session_ID"].tolist()
    if vertical_sessions:
        vmask = df_filtered["Session_ID"].isin(vertical_sessions)
        df_filtered.loc[vmask, "Choice_Binary"] = (
            df_filtered.loc[vmask, "PhysicalChoice"].str.contains("Top").astype(int)
        )

    df_filtered["PhysicalSwitch"] = (
        df_filtered.groupby("Session_ID")["Choice_Binary"].diff().fillna(0).abs().astype(int)
    )
    df_filtered["Outcome_Binary"] = df_filtered["Outcome"].astype(int)

    # Block-level variables
    df_filtered["High_Prob"] = df_filtered["ProbReward"].map({0.8: 1, 0.2: 0})
    df_filtered["Trial_in_Block"] = (
        df_filtered.groupby(["Session_ID", "BlockCount"]).cumcount()
    )
    df_filtered["High_Prob_Is_Right"] = df_filtered.apply(
        lambda row: (row["ProbReward"] == 0.8 and row["Choice_Binary"] == 1)
        or (row["ProbReward"] == 0.2 and row["Choice_Binary"] == 0),
        axis=1,
    )
    df_filtered["Block_Transition"] = (
        df_filtered.groupby("Session_ID")["BlockCount"].diff().fillna(0).abs().astype(bool).astype(int)
    )
    df_filtered["Trials_Since_Transition"] = 0
    for session in df_filtered["Session_ID"].unique():
        session_mask = df_filtered["Session_ID"] == session
        counter = 0
        trials_since = []
        for is_trans in df_filtered.loc[session_mask, "Block_Transition"]:
            if is_trans:
                counter = 0
            trials_since.append(counter)
            counter += 1
        df_filtered.loc[session_mask, "Trials_Since_Transition"] = trials_since

    # Rolling-window state-classification features.
    #
    # CORRECTED (see module docstring's "Rolling_Accuracy: reward-based, not
    # choice-based" section): `Rolling_Accuracy` is the rolling mean of
    # `Outcome_Binary` (was the trial actually REWARDED), not `High_Prob`
    # (did the animal CHOOSE the deterministically-better side). The
    # choice-based version is also kept, as `Rolling_Accuracy_High_Prob`,
    # matching the two source notebooks that most directly correspond to
    # the actual classification figures (09b_CrossProb_StickyModel_v1.ipynb
    # STEP 7B, 07c_FinalClassifier_Fig6.ipynb's own STEP 7B) -- both compute
    # this exact same pair of columns under these exact same two names.
    df_filtered["Rolling_Accuracy"] = (
        df_filtered.groupby("Session_ID")["Outcome_Binary"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df_filtered["Rolling_Accuracy_High_Prob"] = (
        df_filtered.groupby("Session_ID")["High_Prob"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df_filtered["Rolling_PRight"] = (
        df_filtered.groupby("Session_ID")["Choice_Binary"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df_filtered["Expected_PRight"] = df_filtered["High_Prob_Is_Right"].map({True: 0.8, False: 0.2})
    df_filtered["Rolling_Switch_Rate"] = (
        df_filtered.groupby("Session_ID")["PhysicalSwitch"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )

    # Deviation features -- see module docstring for the duplication fix
    df_filtered["Signed_Deviation"] = df_filtered["Rolling_PRight"] - df_filtered["Expected_PRight"]
    df_filtered["Choice_Deviation"] = df_filtered["Signed_Deviation"].abs()

    return df_filtered
