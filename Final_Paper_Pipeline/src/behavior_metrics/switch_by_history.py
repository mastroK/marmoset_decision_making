"""P(switch) conditioned on the trailing N-trial reward history.

Ported from _source_archive/by_figure/Fig2/05_Fig2_Reversal_v2.ipynb, lines
1898-1946 (identical logic also in Fig 3's notebook). Requires
history_encoding.add_history_cols_marmoset to have already been run on the
input dataframe (needs a 'reward_seq' column, e.g. '111'/'000'/'110').
"""

import pandas as pd


def switch_probability_by_history(df_with_history):
    """df_with_history must already have 'reward_seq' (from
    add_history_cols_marmoset) and 'PhysicalSwitch'.

    Returns (per_animal_df, overall_df):
      per_animal_df: one row per (Animal_Name, reward_seq) with p_switch, sem, n_trials
      overall_df: one row per reward_seq, averaged across animals (this is
        what the manuscript reports, e.g. Fig 2e / Fig 3e), sorted by p_switch
    """
    d = df_with_history[df_with_history["reward_seq"].notna()].copy()

    per_animal = d.groupby(["Animal_Name", "reward_seq"]).agg(
        p_switch=("PhysicalSwitch", "mean"),
        sem=("PhysicalSwitch", "sem"),
        n_trials=("PhysicalSwitch", "count"),
    ).reset_index()

    from scipy import stats as scipy_stats

    overall = per_animal.groupby("reward_seq").agg(
        p_switch=("p_switch", "mean"),
        sem=("p_switch", lambda x: scipy_stats.sem(x) if len(x) > 1 else float("nan")),
        n_animals=("p_switch", "count"),
        n_trials=("n_trials", "sum"),
    ).reset_index().sort_values("p_switch")
    # `sem` here is the across-animal SEM of each animal's own p_switch
    # (matching how the manuscript reports these, e.g. Fig 2e's
    # 0.350+/-0.033) -- distinct from `per_animal`'s `sem` column, which is
    # each animal's own within-animal across-trial SEM.

    return per_animal, overall
