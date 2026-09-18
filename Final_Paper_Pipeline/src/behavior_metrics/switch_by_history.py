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


def switch_probability_by_choice_and_reward_history(df_with_history):
    """Reviewer-response addition (RA5_small_items.ipynb, R1-f): P(switch)
    conditioned on BOTH the trailing choice sequence (`decision_seq`) and
    reward sequence (`reward_seq`) jointly, not reward history alone --
    generalizes `switch_probability_by_history` (which is Fig 2e/Fig 3e's
    actual published panel, unchanged, conditioning on `reward_seq` only)
    to check whether prior CHOICE identity adds explanatory power beyond
    outcome history. `df_with_history` must already have both
    `decision_seq`/`reward_seq` (from `add_history_cols_marmoset`) and
    `PhysicalSwitch`.
    """
    d = df_with_history[df_with_history["reward_seq"].notna() & df_with_history["decision_seq"].notna()].copy()
    d["choice_reward_code"] = d["decision_seq"] + "|" + d["reward_seq"]

    per_animal = d.groupby(["Animal_Name", "choice_reward_code"]).agg(
        p_switch=("PhysicalSwitch", "mean"),
        sem=("PhysicalSwitch", "sem"),
        n_trials=("PhysicalSwitch", "count"),
    ).reset_index()

    from scipy import stats as scipy_stats

    overall = per_animal.groupby("choice_reward_code").agg(
        p_switch=("p_switch", "mean"),
        sem=("p_switch", lambda x: scipy_stats.sem(x) if len(x) > 1 else float("nan")),
        n_animals=("p_switch", "count"),
        n_trials=("n_trials", "sum"),
    ).reset_index().sort_values("p_switch")
    overall[["decision_seq", "reward_seq"]] = overall["choice_reward_code"].str.split("|", expand=True)

    return per_animal, overall


def choice_history_variance_given_reward_history(joint_overall):
    """Reviewer-response addition (RA5_small_items.ipynb, R1-f): for each
    reward_seq value, the spread (max - min, and SD) of p_switch across
    the different choice-sequence (decision_seq) codes that share that
    same reward_seq -- a small spread means prior choice identity adds
    little beyond outcome history alone (supporting Fig 2e's own choice to
    condition on reward_seq only); a large spread means it doesn't.
    Takes `joint_overall` (the `overall` table from
    `switch_probability_by_choice_and_reward_history`).
    """
    records = []
    for reward_seq, g in joint_overall.groupby("reward_seq"):
        records.append({
            "reward_seq": reward_seq,
            "n_choice_codes": int(len(g)),
            "p_switch_min": float(g["p_switch"].min()),
            "p_switch_max": float(g["p_switch"].max()),
            "p_switch_range": float(g["p_switch"].max() - g["p_switch"].min()),
            "p_switch_sd_across_choice_codes": float(g["p_switch"].std(ddof=0)) if len(g) > 1 else 0.0,
        })
    return pd.DataFrame(records).sort_values("p_switch_range", ascending=False)
