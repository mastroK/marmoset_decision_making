"""Within-block learning, trials-to-criterion, and cross-session learning
trajectory.

Ported from _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb,
lines 109-1086 (Fig 1 panels b-d). Operates on "long blocks": for each
session, the single longest block, filtered to a plausible trial-count
range, restricted to animals with enough qualifying sessions -- this is a
DIFFERENT block-selection strategy than Fig 2/3's (which used every
transition, not just the longest block per session), specific to this
notebook.

NOTE on trials-to-criterion (panel c): the source notebook's own CELL 5-7
implements a "5 consecutive correct trials" rule and was initially ported
here as `trials_to_criterion_consecutive` -- but cross-checking against the
manuscript's own printed legend for this panel found no match (the legend
describes a rolling accuracy>=0.8-over-5-trials rule, a 10-trial late-
accuracy window, and session-level, not block-level, dots -- none of which
CELL 5-7 implements). CELL 5-7 turned out to be a different, unrelated
exploratory analysis in the same notebook. `trials_to_criterion_consecutive`
was removed; `trials_to_criterion_rolling` (below) reuses Fig 2/3's shared
rolling-window rule (behavior_metrics/criterion.py) instead, which is what
the manuscript's own legend actually describes.
"""

import numpy as np
import pandas as pd

from . import criterion as _criterion


def summarize_blocks(df):
    """One row per (session, block): n_trials, accuracy, animal, date,
    high_prob_location. Ported from lines 109-142."""
    records = []
    for session_id in df["Session_ID"].unique():
        session_df = df[df["Session_ID"] == session_id].sort_values("Trial")
        for block_num in session_df["Block"].unique():
            block_df = session_df[session_df["Block"] == block_num]
            records.append({
                "session_id": session_id,
                "animal": block_df["Animal_Name"].iloc[0],
                "date": block_df["Date"].iloc[0],
                "block": block_num,
                "n_trials": len(block_df),
                "accuracy": block_df["Outcome"].mean(),
                "high_prob_location": block_df["Target1_location"].iloc[0],
            })
    return pd.DataFrame(records)


def longest_block_per_session(block_df):
    """Ported from line 145."""
    return block_df.loc[block_df.groupby("session_id")["n_trials"].idxmax()]


def filter_long_blocks(longest_blocks, min_block_length, max_block_length, min_sessions_per_animal):
    """Ported from lines 203-224: restrict to blocks in a plausible trial-
    count range, then to animals with enough qualifying sessions."""
    long_blocks = longest_blocks[
        (longest_blocks["n_trials"] >= min_block_length)
        & (longest_blocks["n_trials"] <= max_block_length)
    ].copy()

    sessions_per_animal = long_blocks.groupby("animal")["session_id"].nunique()
    animals_to_include = sessions_per_animal[sessions_per_animal >= min_sessions_per_animal].index.tolist()
    return long_blocks[long_blocks["animal"].isin(animals_to_include)].copy(), animals_to_include


def within_block_learning_curve(df, long_blocks_filtered, trial_bin_size):
    """Per-trial-position accuracy within the qualifying long blocks,
    binned by trial_bin_size. Ported from lines 233-286.

    Returns (learning_df [trial-level], binned_learning [per animal],
    overall_learning [group average]).
    """
    records = []
    for _, block_info in long_blocks_filtered.iterrows():
        block_trials = df[
            (df["Session_ID"] == block_info["session_id"]) & (df["Block"] == block_info["block"])
        ].sort_values("Trial").copy()
        block_trials["trial_in_block"] = range(len(block_trials))
        for _, trial in block_trials.iterrows():
            records.append({
                "session_id": block_info["session_id"],
                "animal": block_info["animal"],
                "block": block_info["block"],
                "trial_in_block": trial["trial_in_block"],
                "outcome": trial["Outcome"],
            })
    learning_df = pd.DataFrame(records)
    learning_df["trial_bin"] = (learning_df["trial_in_block"] // trial_bin_size) * trial_bin_size

    binned_learning = learning_df.groupby(["animal", "trial_bin"]).agg(
        accuracy=("outcome", "mean"), sem=("outcome", "sem"), n=("outcome", "count")
    ).reset_index()
    overall_learning = learning_df.groupby("trial_bin").agg(
        accuracy=("outcome", "mean"), sem=("outcome", "sem"), n=("outcome", "count")
    ).reset_index()

    return learning_df, binned_learning, overall_learning


def trials_to_criterion_rolling(df, long_blocks_filtered, criterion_accuracy, criterion_window,
                                  accuracy_col="Outcome"):
    """Trials to criterion using the same rolling-window accuracy rule as
    Fig 2/3 (behavior_metrics/criterion.py::trials_to_criterion), applied to
    each session's single qualifying long block -- one row per session,
    since `long_blocks_filtered` already has at most one block per session
    (see longest_block_per_session/filter_long_blocks above). This is what
    the manuscript's own legend for Fig 1c describes (see module docstring).
    """
    transitions_df = long_blocks_filtered.rename(columns={"block": "to_block"})[
        ["session_id", "to_block", "animal"]
    ]
    return _criterion.trials_to_criterion(
        df, transitions_df, accuracy_col, criterion_accuracy, criterion_window, block_col="Block"
    )


def asymptotic_performance(df, long_blocks_filtered, asymptote_window):
    """Mean accuracy over the trailing `asymptote_window` trials of each
    qualifying block. Ported from lines 958-995."""
    records = []
    for _, block_info in long_blocks_filtered.iterrows():
        block_trials = df[
            (df["Session_ID"] == block_info["session_id"]) & (df["Block"] == block_info["block"])
        ].sort_values("Trial")
        if len(block_trials) >= asymptote_window:
            asymptote_trials = block_trials.tail(asymptote_window)
            records.append({
                "session_id": block_info["session_id"],
                "animal": block_info["animal"],
                "block": block_info["block"],
                "block_length": block_info["n_trials"],
                "asymptote_accuracy": asymptote_trials["Outcome"].mean(),
            })
    return pd.DataFrame(records)


def session_accuracy_trajectory(df, long_blocks_filtered):
    """Chronological per-session accuracy (restricted to trials from
    qualifying long blocks), per animal, with a 0-indexed session_number.
    Ported from lines 761-802."""
    records = []
    for animal in long_blocks_filtered["animal"].unique():
        animal_blocks = long_blocks_filtered[long_blocks_filtered["animal"] == animal].copy()
        animal_blocks = animal_blocks.merge(
            df[["Session_ID", "Date"]].drop_duplicates(),
            left_on="session_id", right_on="Session_ID", how="left",
        )
        animal_blocks = animal_blocks.sort_values("Date")

        unique_sessions = animal_blocks["session_id"].unique()
        session_to_number = {s: i for i, s in enumerate(unique_sessions)}
        animal_blocks["session_number"] = animal_blocks["session_id"].map(session_to_number)

        for session_id in unique_sessions:
            session_blocks = animal_blocks[animal_blocks["session_id"] == session_id]
            session_trials = df[
                (df["Session_ID"] == session_id) & (df["Block"].isin(session_blocks["block"]))
            ]
            records.append({
                "animal": animal,
                "session_id": session_id,
                "session_number": session_blocks["session_number"].iloc[0],
                "date": session_blocks["date"].iloc[0] if "date" in session_blocks else session_blocks["Date"].iloc[0],
                "accuracy": session_trials["Outcome"].mean(),
                "n_trials": len(session_trials),
            })
    return pd.DataFrame(records)
