"""Choice/reward history encoding, shared by Fig 2 and Fig 3.

Ported (with no behavior change) from _source_archive/by_figure/Fig2/05_Fig2_Reversal_v2.ipynb
(lines 1817-1896: add_history_cols_marmoset, list_to_str, encode_as_ab) and an
identical copy at _source_archive/by_figure/Fig3/06_Fig3_2ABT_v2.ipynb (lines
809-872). Both notebooks defined byte-for-byte the same three functions
independently -- consolidated here as the single canonical version.
"""

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


def list_to_str(seq):
    """Convert a list of ints/floats to a concatenated string, e.g. [1,0,1] -> '101'."""
    return "".join(str(el) for el in seq)


def encode_as_ab(row, symm):
    """Convert a (decision_seq, reward_seq) pair into a single character code
    per trial.

    symm=True: symmetrical A/B encoding relative to the FIRST choice in the
    window (so 'which side' is normalized out -- used for Fig 2/3's
    stay/switch-style history encoding).
    symm=False: raw left/right encoding (labeled 'RL_history' upstream; not
    currently used by Fig 2/3's own panels but ported for parity with source).
    """
    if int(row.decision_seq[0]) & symm:
        mapping = {("0", "0"): "b", ("0", "1"): "B", ("1", "0"): "a", ("1", "1"): "A"}
    elif (int(row.decision_seq[0]) == 0) & symm:
        mapping = {("0", "0"): "a", ("0", "1"): "A", ("1", "0"): "b", ("1", "1"): "B"}
    else:
        mapping = {("0", "0"): "r", ("0", "1"): "R", ("1", "0"): "l", ("1", "1"): "L"}
    return "".join(mapping[(c, r)] for c, r in zip(row.decision_seq, row.reward_seq))


def add_history_cols_marmoset(df, n, choice_col="test_choice", outcome_col="test_outcome"):
    """Add trailing N-trial choice/reward history columns, computed
    separately within each session.

    Adds: decision_seq, reward_seq (raw 0/1 strings), history (symmetrical
    A/B-encoded), RL_history (raw L/R-encoded).

    df must already have `choice_col`/`outcome_col` as 0/1 int-castable
    columns (Fig 2/3 build these upstream as
    df['test_choice'] = df['StimuliChosen'].map({'TargetA': 1, 'TargetB': 0})
    df['test_outcome'] = df['Reward'].map({'Rewarded': 1, 'Unrewarded': 0})
    ).
    """
    df = df.copy()
    df["decision_seq"] = pd.Series(np.nan, index=df.index).astype(object)
    df["reward_seq"] = pd.Series(np.nan, index=df.index).astype(object)
    df["history"] = pd.Series(np.nan, index=df.index).astype(object)
    df["RL_history"] = pd.Series(np.nan, index=df.index).astype(object)
    df = df.reset_index(drop=True)

    for session in df["Session_ID"].unique():
        d = df.loc[df["Session_ID"] == session].copy()
        if len(d) <= n:
            continue

        decision_arr = d[choice_col].fillna(0).astype("int").values
        reward_arr = d[outcome_col].fillna(0).astype("int").values

        decision_sequences = list(map(list_to_str, sliding_window_view(decision_arr, n)))[:-1]
        reward_sequences = list(map(list_to_str, sliding_window_view(reward_arr, n)))[:-1]

        df.loc[d.index.values[n:], "decision_seq"] = decision_sequences
        df.loc[d.index.values[n:], "reward_seq"] = reward_sequences
        df.loc[d.index.values[n:], "history"] = df.loc[d.index.values[n:]].apply(
            encode_as_ab, args=(True,), axis=1
        )
        df.loc[d.index.values[n:], "RL_history"] = df.loc[d.index.values[n:]].apply(
            encode_as_ab, args=(False,), axis=1
        )

    return df


def get_ordered_colors(animals, animal_colors):
    return [animal_colors[a] for a in animals]


def sort_animals(animal_list, animal_order):
    """Sort animals per the canonical plotting order; unlisted animals sort last, alphabetically."""
    order_index = {a: i for i, a in enumerate(animal_order)}
    return sorted(animal_list, key=lambda a: (order_index.get(a, len(animal_order)), a))
