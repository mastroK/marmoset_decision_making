"""Pre-block-change accuracy and break-taking behavior (win/loss, block
transitions) for Supp Fig 1 panels d-e.

Ported from _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb's
"STATS EXTRACTION: Results Section 1" section, STAT 1 (lines 5272-5314),
STAT 3 (lines 5343-5374), and STAT 4 (lines 5376-5403) -- the same
authoritative section stage 3 (Fig 1 f/h/i) was built from. These three
stats are the ones that section itself labels "Supp Fig 1F"/"Supp Fig 1G".
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def pre_block_change_accuracy(df, pre_window):
    """Accuracy on the `pre_window` trials immediately before each block
    change (trial_relative in [-pre_window, -1]), trial-level and
    per-animal-vs-chance. Ported from lines 5276-5314 (STAT 1, Supp Fig 1F).

    NOTE: the source builds a wider transition-aligned window (-5 to +15)
    here, but only the negative (pre-transition) side is used by this
    stat -- reproduced here computing only the needed pre-transition side.
    """
    records = []
    for session_id in df["Session_ID"].unique():
        session_data = df[df["Session_ID"] == session_id].sort_values("Trial")
        block_arr = session_data["Block"].values
        for i in range(len(session_data) - 1):
            if block_arr[i] != block_arr[i + 1]:
                for rel_pos in range(-pre_window, 0):
                    abs_pos = i + rel_pos
                    if 0 <= abs_pos < len(session_data):
                        trial = session_data.iloc[abs_pos]
                        records.append({
                            "session_id": session_id,
                            "animal": trial["Animal_Name"],
                            "trial_relative": rel_pos,
                            "outcome": trial["Outcome"],
                        })
    pre_df = pd.DataFrame(records)
    pre_block = pre_df["outcome"]
    pre_per_animal = pre_df.groupby("animal")["outcome"].mean()
    _, p_val = scipy_stats.wilcoxon(pre_per_animal - 0.5)

    return {
        "n_trials": int(len(pre_block)),
        "n_animals": int(len(pre_per_animal)),
        "mean": float(pre_block.mean()),
        "sem": float(scipy_stats.sem(pre_block)),
        "wilcoxon_p_vs_chance_per_animal": float(p_val),
        # Added for plotting fidelity only (per-animal points overlaid on the
        # summary bar, matching the per-animal-scatter-over-bar convention
        # used throughout 04_LocationBasedShifting.ipynb -- e.g. its
        # Asymptotic_Performance_Summary.svg and Break_Analysis_PerAnimal_*
        # cells). Does not change any value already returned above.
        "per_animal_accuracy": {str(k): float(v) for k, v in pre_per_animal.items()},
    }


def prepare_break_analysis_df(df, bout_iti_threshold):
    """Trial-level dataframe with ITI, is_break, previous-trial outcome,
    and block-transition flags -- shared prep for STAT 3 and STAT 4.
    Ported from lines 5261-5266, 5320, 5347-5351, 5380-5381."""
    d = df.sort_values(["Session_ID", "Trial"]).copy()
    d["Prev_StartTime"] = d.groupby("Session_ID")["AbsoluteTrialStartTime"].shift(1)
    d["ITI_calc"] = (d["AbsoluteTrialStartTime"] - d["Prev_StartTime"]) / 1000
    d = d[d["ITI_calc"].notna()].copy()

    d["is_break"] = (d["ITI_calc"] >= bout_iti_threshold).astype(int)
    d["prev_outcome"] = d.groupby("Session_ID")["Outcome"].shift(1)
    d = d[d["prev_outcome"].notna()].copy()

    d["prev_block"] = d.groupby("Session_ID")["Block"].shift(1)
    d["block_transition"] = (d["Block"] != d["prev_block"]).astype(int)
    return d


def p_break_by_prev_outcome(df_iti_clean):
    """P(break) following a win vs. following a loss, paired per animal.
    Ported from lines 5353-5374 (STAT 3, Supp Fig 1G)."""
    win_per_animal = df_iti_clean[df_iti_clean["prev_outcome"] == 1].groupby("Animal_Name")["is_break"].mean()
    loss_per_animal = df_iti_clean[df_iti_clean["prev_outcome"] == 0].groupby("Animal_Name")["is_break"].mean()
    common = win_per_animal.index.intersection(loss_per_animal.index)
    _, p_val = scipy_stats.wilcoxon(win_per_animal[common], loss_per_animal[common])

    return {
        "n_trials": int(len(df_iti_clean)),
        "n_animals": int(len(common)),
        "p_break_win": float(win_per_animal[common].mean()),
        "p_break_loss": float(loss_per_animal[common].mean()),
        "wilcoxon_p": float(p_val),
        # Added for plotting fidelity only (per-animal bar + SEM + jittered
        # scatter, matching source's Break_Analysis_PerAnimal_WHERE_100-0.svg
        # cell, lines 5121-5232). Does not change any value already returned
        # above -- these are the same per-animal series those values were
        # already averaged from.
        "p_break_win_sem": float(win_per_animal[common].sem()),
        "p_break_loss_sem": float(loss_per_animal[common].sem()),
        "win_per_animal": {str(k): float(v) for k, v in win_per_animal[common].items()},
        "loss_per_animal": {str(k): float(v) for k, v in loss_per_animal[common].items()},
    }


def p_break_by_outcome_and_transition(df_iti_clean):
    """P(break) jointly split by previous-trial outcome (rewarded/
    unrewarded) and block status (transition vs. same block) -- Supp Fig 1
    panel e's 2x2 heatmap. ADDITIVE: this combines the two conditions
    already computed separately (as marginals) by p_break_by_prev_outcome
    and p_break_by_block_transition above into one joint groupby; those two
    functions and the panel d values they feed are untouched.
    """
    cell_defs = [
        ("rewarded", "transition", 1, 1),
        ("rewarded", "same_block", 1, 0),
        ("unrewarded", "transition", 0, 1),
        ("unrewarded", "same_block", 0, 0),
    ]
    per_animal_by_cell = {}
    for outcome_label, trans_label, outcome_val, trans_val in cell_defs:
        sub = df_iti_clean[
            (df_iti_clean["prev_outcome"] == outcome_val)
            & (df_iti_clean["block_transition"] == trans_val)
        ]
        per_animal_by_cell[(outcome_label, trans_label)] = sub.groupby("Animal_Name")["is_break"].mean()

    common = None
    for pa in per_animal_by_cell.values():
        common = pa.index if common is None else common.intersection(pa.index)

    cells = {}
    per_animal_out = {}
    for (outcome_label, trans_label), pa in per_animal_by_cell.items():
        pa_common = pa[common]
        key = f"{outcome_label}_{trans_label}"
        cells[key] = {
            "mean": float(pa_common.mean()),
            "sem": float(pa_common.sem()),
            "n_animals": int(len(pa_common)),
        }
        per_animal_out[key] = {str(a): float(v) for a, v in pa_common.items()}

    return {
        "n_trials": int(len(df_iti_clean)),
        "n_animals_common": int(len(common)),
        "cells": cells,
        "per_animal": per_animal_out,
    }


def p_break_by_block_transition(df_iti_clean):
    """P(break) at a block transition vs. within a block, paired per
    animal. Ported from lines 5383-5403 (STAT 4, Supp Fig 1G)."""
    trans_per_animal = df_iti_clean[df_iti_clean["block_transition"] == 1].groupby("Animal_Name")["is_break"].mean()
    same_per_animal = df_iti_clean[df_iti_clean["block_transition"] == 0].groupby("Animal_Name")["is_break"].mean()
    common = trans_per_animal.index.intersection(same_per_animal.index)
    _, p_val = scipy_stats.wilcoxon(trans_per_animal[common], same_per_animal[common])

    return {
        "n_trials": int(len(df_iti_clean)),
        "n_animals": int(len(common)),
        "p_break_transition": float(trans_per_animal[common].mean()),
        "p_break_same_block": float(same_per_animal[common].mean()),
        "wilcoxon_p": float(p_val),
        # Added for plotting fidelity only -- see p_break_by_prev_outcome
        # above for rationale. Does not change any value already returned.
        "p_break_transition_sem": float(trans_per_animal[common].sem()),
        "p_break_same_block_sem": float(same_per_animal[common].sem()),
        "transition_per_animal": {str(k): float(v) for k, v in trans_per_animal[common].items()},
        "same_block_per_animal": {str(k): float(v) for k, v in same_per_animal[common].items()},
    }
