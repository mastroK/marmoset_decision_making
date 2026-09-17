"""Late-block accuracy: mean accuracy over the trailing N trials of each block.

Shared logic for Fig 2 (ported from 05_Fig2_Reversal_v2.ipynb, lines
1445-1479; accuracy_col='Outcome', late_window=10) and Fig 3 (ported from
06_Fig3_2ABT_v2.ipynb, lines 1410-1430; accuracy_col='High_Prob',
late_window=8). The two figures use different windows and different
accuracy columns for the same reason described in criterion.py -- Fig 2's
task has deterministic reward, Fig 3's does not.
"""

import pandas as pd


def late_accuracy(df, transitions_df, accuracy_col, late_window, block_col="Block"):
    """Returns a per-block dataframe: session_id, animal, transition_type
    (if present), late_accuracy, block_length. Blocks shorter than
    late_window are excluded (can't compute a trailing window that long).
    """
    has_transition_type = "transition_type" in transitions_df.columns
    records = []

    for _, trans in transitions_df.iterrows():
        session_id = trans["session_id"]
        to_block = trans["to_block"]

        session_data = df[df["Session_ID"] == session_id]
        new_block_trials = session_data[session_data[block_col] == to_block].sort_values(
            "Abs_Trial_Pos" if "Abs_Trial_Pos" in session_data.columns else "Trial"
        )

        if len(new_block_trials) < late_window:
            continue

        late_trials = new_block_trials.tail(late_window)
        record = {
            "session_id": session_id,
            "animal": trans["animal"],
            "late_accuracy": late_trials[accuracy_col].mean(),
            "block_length": len(new_block_trials),
        }
        if has_transition_type:
            record["transition_type"] = trans["transition_type"]
        records.append(record)

    return pd.DataFrame(records)
