"""Block-transition classification: cued vs. uncued reversals.

Fig 2-specific (the 2-ABT task in Fig 3 has no cued/uncued distinction --
its blocks are all a single probabilistic-reversal type).

Ported from _source_archive/by_figure/Fig2/05_Fig2_Reversal_v2.ipynb. That
notebook tries two different classification rules before settling on this
one (see REVIEWER_AUDIT.md-style provenance note below); this is the
SECOND, final rule -- the one whose output (`trans_df`) is actually
consumed downstream by the criterion/late-accuracy/WSLS panels.

Provenance note: an earlier cell in 05 (lines ~201-206, not ported here)
classified a transition as CUED only when BOTH the physical target
locations changed AND the stimulus source changed, else UNCUED/PARTIAL.
That three-way rule's output was never used again in the notebook. The
rule below -- based solely on whether the set of chosen physical locations
changed between the current and next block -- is the one whose `trans_df`
feeds every subsequent Fig 2 panel, so it is the canonical one ported here.
"""

import pandas as pd


def classify_transitions(df):
    """For each session, classify each block-to-block transition as CUED
    (the set of physical response locations used changed) or UNCUED (same
    locations, i.e. only the rewarded side within the same physical layout
    flipped).

    Returns one row per transition: session_id, from_block, to_block,
    transition_type ('CUED'/'UNCUED'), animal, date.
    """
    all_transitions = []

    for session_id in df["Session_ID"].unique():
        session_df = df[df["Session_ID"] == session_id].sort_values("Trial")
        blocks = sorted(session_df["Block"].unique())

        for i in range(len(blocks) - 1):
            curr_block, next_block = blocks[i], blocks[i + 1]
            curr_trials = session_df[session_df["Block"] == curr_block]
            next_trials = session_df[session_df["Block"] == next_block]

            curr_locs = set(curr_trials["PhysicalChoice"].dropna().unique())
            next_locs = set(next_trials["PhysicalChoice"].dropna().unique())

            trans_type = "UNCUED" if curr_locs == next_locs else "CUED"

            all_transitions.append({
                "session_id": session_id,
                "from_block": curr_block,
                "to_block": next_block,
                "transition_type": trans_type,
                "animal": session_df["Animal_Name"].iloc[0],
                "date": session_df["Date"].iloc[0],
            })

    return pd.DataFrame(all_transitions)
