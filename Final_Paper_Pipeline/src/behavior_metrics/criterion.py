"""Trials-to-criterion: how many trials into a new block until rolling
accuracy first reaches a threshold.

Shared logic for Fig 2, Fig 3, and Fig 1 (panel c), but note the figures
call it with DIFFERENT parameters (see config.yaml's fig1/fig2/fig3
sections) and different accuracy columns:
  - Fig 1 (panel c, deterministic VDT-reversal reward): accuracy_col='Outcome',
    applied to each session's single qualifying "long block" (see
    block_learning.py::trials_to_criterion_rolling) -- confirmed against the
    manuscript's own printed legend for this panel, which describes exactly
    this rolling-window rule (not the source notebook's own, unrelated
    CELL 5-7 consecutive-correct exploratory analysis)
  - Fig 2 (deterministic VDT-reversal reward): accuracy_col='Outcome'
    (ported from 05_Fig2_Reversal_v2.ipynb, lines 944-992)
  - Fig 3 (probabilistic 2-ABT reward): accuracy_col='High_Prob' -- whether
    the animal chose the higher-probability option, since a "correct" choice
    isn't always rewarded (ported from 06_Fig3_2ABT_v2.ipynb, lines
    1175-1220)

`transitions_df` must have columns: session_id, to_block, and optionally
transition_type (Fig 2 only -- CUED/UNCUED; omit for Fig 3, which has no
such split).
"""

import numpy as np
import pandas as pd


def trials_to_criterion(df, transitions_df, accuracy_col, criterion_accuracy,
                          criterion_window, max_trials=30, block_col="Block"):
    """Returns a per-block dataframe: session_id, animal, transition_type
    (if present in transitions_df), trials_to_criterion, block_length.

    trials_to_criterion is NaN if the block never reached criterion within
    its own length; such rows are already excluded from `_clean`, but kept
    in the full returned frame for transparency.
    """
    has_transition_type = "transition_type" in transitions_df.columns
    records = []

    for _, trans in transitions_df.iterrows():
        session_id = trans["session_id"]
        to_block = trans["to_block"]

        session_data = df[df["Session_ID"] == session_id].sort_values("Trial")
        new_block_trials = session_data[session_data[block_col] == to_block].copy()

        if len(new_block_trials) < criterion_window:
            continue

        new_block_trials["rolling_acc"] = new_block_trials[accuracy_col].rolling(
            window=criterion_window, min_periods=criterion_window
        ).mean()

        criterion_met = new_block_trials[new_block_trials["rolling_acc"] >= criterion_accuracy]
        if len(criterion_met) > 0:
            ttc = criterion_met.index[0] - new_block_trials.index[0] + 1
        else:
            ttc = np.nan

        record = {
            "session_id": session_id,
            "animal": trans["animal"],
            "trials_to_criterion": ttc,
            "block_length": len(new_block_trials),
        }
        if has_transition_type:
            record["transition_type"] = trans["transition_type"]
        records.append(record)

    ttc_df = pd.DataFrame(records)
    return ttc_df


def clean_ttc(ttc_df, max_trials=30):
    """Drop blocks that never reached criterion, and unreasonably long outliers."""
    clean = ttc_df[ttc_df["trials_to_criterion"].notna()]
    clean = clean[clean["trials_to_criterion"] <= max_trials]
    return clean


def aggregate_to_session(block_df, value_col, extra_group_cols=()):
    """Collapse block/transition-level rows (this module's and
    late_accuracy.py's natural unit -- one row per block/transition) to one
    row per SESSION (mean across that session's blocks/transitions).

    This is a correctness fix, not a cosmetic one: Fig 2/3's own manuscript
    figure legends describe "sessions (dots)" for these panels, but a
    single session can contain several blocks/transitions (confirmed: Fig 1's
    WhereTask sessions average ~6.6 blocks/session), so plotting -- or
    computing summary statistics from -- the raw block-level dataframe both
    overcounts dots and pseudoreplicates any test run on it. `extra_group_cols`
    additionally splits by, e.g., Fig 2's transition_type (CUED/UNCUED).
    """
    group_cols = ["session_id", "animal", *extra_group_cols]
    return block_df.groupby(group_cols, as_index=False)[value_col].mean()
