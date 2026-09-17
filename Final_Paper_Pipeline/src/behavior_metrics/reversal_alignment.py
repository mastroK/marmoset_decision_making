"""Accuracy aligned to trial position relative to a block transition --
Fig 2/Fig 3 panel b ("Probability of selecting the high probability choice
at each trial position ... across a reversal"), and reusable by any other
figure needing the same "trial position relative to reversal" alignment
(e.g. Fig 8 panel a/b).
"""

import numpy as np
import pandas as pd


def accuracy_by_position_relative_to_transition(df, transitions_df, accuracy_col,
                                                   window_before, window_after, block_col="Block"):
    """For each transition in `transitions_df` (needs session_id, to_block,
    and optionally transition_type), collect `accuracy_col` for trials from
    `window_before` before the transition through `window_after` after it
    (transition's first trial = position 0). Returns a long dataframe:
    session_id, animal, transition_type (if present), position, accuracy.
    """
    has_transition_type = "transition_type" in transitions_df.columns
    records = []

    for _, trans in transitions_df.iterrows():
        session_id = trans["session_id"]
        to_block = trans["to_block"]
        session_data = df[df["Session_ID"] == session_id].sort_values("Trial").reset_index(drop=True)

        to_block_idx = session_data.index[session_data[block_col] == to_block]
        if len(to_block_idx) == 0:
            continue
        transition_pos = to_block_idx[0]

        for rel_pos in range(-window_before, window_after + 1):
            abs_pos = transition_pos + rel_pos
            if 0 <= abs_pos < len(session_data):
                record = {
                    "session_id": session_id,
                    "animal": trans["animal"],
                    "position": rel_pos,
                    "accuracy": session_data.iloc[abs_pos][accuracy_col],
                }
                if has_transition_type:
                    record["transition_type"] = trans["transition_type"]
                records.append(record)

    return pd.DataFrame(records)


def summarize_by_position(aligned_df, group_cols=()):
    """Mean +/- SEM accuracy at each position, optionally split by
    additional group_cols (e.g. transition_type)."""
    cols = ["position", *group_cols]
    return aligned_df.groupby(list(cols))["accuracy"].agg(mean="mean", sem="sem", n="count").reset_index()


def pick_representative_session(df, min_trials=150):
    """Pick an example session for the exemplar-session panel, matching the
    manuscript's OWN selection rule exactly (from the user-provided source
    code): the first session with more than `min_trials` trials. Falls back
    to the single longest session if none clears the threshold.
    """
    session_lengths = df.groupby("Session_ID").size()
    good_sessions = session_lengths[session_lengths > min_trials].index
    return good_sessions[0] if len(good_sessions) else session_lengths.idxmax()
