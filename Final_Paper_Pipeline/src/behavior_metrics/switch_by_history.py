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


def choice_history_variance_given_reward_history(joint_overall, min_trials=30):
    """Reviewer-response addition (RA5_small_items.ipynb, R1-f): for each
    reward_seq value, the spread (max - min, and SD) of p_switch across
    the different choice-sequence (decision_seq) codes that share that
    same reward_seq -- a small spread means prior choice identity adds
    little beyond outcome history alone (supporting Fig 2e's own choice to
    condition on reward_seq only); a large spread means it doesn't.
    Takes `joint_overall` (the `overall` table from
    `switch_probability_by_choice_and_reward_history`).

    `min_trials` excludes choice_reward_code rows with too few pooled
    trials before computing the spread -- found necessary directly: of
    the 15 (decision_seq, reward_seq) combinations that occur at all in
    Fig 2's own data, only 8 (all with decision_seq == reward_seq, each
    with 250-6148 trials across all 6 animals) have any real data; the
    other 7 are single-animal artifacts with 2-9 trials each, and it was
    exactly those 7 driving an apparent p_switch_range of up to 1.000
    (e.g. a single 2-trial code giving p_switch=0.0 or 1.0 outright).
    Dropping them is not cherry-picking a result -- the same 7 codes are
    excluded regardless of which reward_seq group they'd fall into, and
    the threshold (30) matches this project's own min-trial convention
    elsewhere, not tuned to this specific outcome.
    """
    filtered = joint_overall[joint_overall["n_trials"] >= min_trials]
    dropped = joint_overall[joint_overall["n_trials"] < min_trials]

    records = []
    for reward_seq, g in filtered.groupby("reward_seq"):
        records.append({
            "reward_seq": reward_seq,
            "n_choice_codes": int(len(g)),
            "p_switch_min": float(g["p_switch"].min()),
            "p_switch_max": float(g["p_switch"].max()),
            "p_switch_range": float(g["p_switch"].max() - g["p_switch"].min()) if len(g) > 1 else None,
            "p_switch_sd_across_choice_codes": float(g["p_switch"].std(ddof=0)) if len(g) > 1 else None,
        })
    summary = pd.DataFrame(records)
    if len(summary) and summary["p_switch_range"].notna().any():
        summary = summary.sort_values("p_switch_range", ascending=False, na_position="last")
    return summary, dropped[["choice_reward_code", "decision_seq", "reward_seq", "n_trials", "n_animals"]]


def switch_probability_by_physical_side_persistence(df_with_physical_history, min_animals=2):
    """Reviewer-response addition (RA5_small_items.ipynb, R1-f -- corrected
    operationalization per direct user request 2026-09-18): within each
    reward_seq, compares P(switch) between trials where the animal
    persisted on the physical RIGHT side for all 3 trailing trials
    (`decision_seq == "111"`) vs. the physical LEFT side
    (`decision_seq == "000"`) -- e.g. RRR vs. LLL when reward_seq == "111"
    (three consecutive rewards either way). This isolates whether
    absolute spatial side identity matters, independent of both reward
    and of `StimuliChosen`'s target-identity framing (the EARLIER,
    now-superseded version of this check used `decision_seq` built from
    `test_choice`/`StimuliChosen` -- TargetA/TargetB, a probability-based
    label, NOT the physical side -- which is a different question and
    was also nearly collinear with reward_seq in this data; this version
    uses `PhysicalChoice`-derived choice history instead).

    `df_with_physical_history` must already have `decision_seq`/
    `reward_seq` computed via `add_history_cols_marmoset` called with a
    PHYSICAL-side (Left=0/Right=1) `choice_col`, not a target-identity one.

    Returns (per_animal_df, summary_df):
      per_animal_df: one row per (Animal_Name, reward_seq, persisted_side)
        with p_switch, n_trials.
      summary_df: one row per reward_seq with each side's mean p_switch
        across animals and a paired t-test (same animals contribute both
        sides, matching this project's standard paired-comparison
        convention) -- only computed where both sides have
        >= min_animals with data.
    """
    d = df_with_physical_history[
        df_with_physical_history["decision_seq"].isin(["111", "000"])
        & df_with_physical_history["reward_seq"].notna()
    ].copy()
    d["persisted_side"] = d["decision_seq"].map({"111": "Right", "000": "Left"})

    per_animal = d.groupby(["Animal_Name", "reward_seq", "persisted_side"]).agg(
        p_switch=("PhysicalSwitch", "mean"),
        n_trials=("PhysicalSwitch", "count"),
    ).reset_index()

    from scipy import stats as scipy_stats

    records = []
    for reward_seq, g in per_animal.groupby("reward_seq"):
        wide = g.pivot(index="Animal_Name", columns="persisted_side", values="p_switch").dropna()
        n_trials_by_side = g.groupby("persisted_side")["n_trials"].sum()
        if len(wide) >= min_animals:
            t, p = scipy_stats.ttest_rel(wide["Right"], wide["Left"])
        else:
            t, p = float("nan"), float("nan")
        records.append({
            "reward_seq": reward_seq,
            "n_animals": int(len(wide)),
            "n_trials_right": int(n_trials_by_side.get("Right", 0)),
            "n_trials_left": int(n_trials_by_side.get("Left", 0)),
            "mean_p_switch_right": float(wide["Right"].mean()) if len(wide) else None,
            "mean_p_switch_left": float(wide["Left"].mean()) if len(wide) else None,
            "paired_t": float(t) if pd.notna(t) else None,
            "paired_p": float(p) if pd.notna(p) else None,
        })
    summary = pd.DataFrame(records).sort_values("reward_seq")
    return per_animal, summary


def physical_side_persistence_pooled_test(per_animal_df, min_animals_per_reward_seq=6):
    """Reviewer-response addition (RA5_small_items.ipynb, R1-f): a single
    overall test of Right- vs. Left-persistence, pooling across the
    per-reward_seq breakdown from
    `switch_probability_by_physical_side_persistence` (8 separate small
    tests, none individually well-powered/corrected for multiple
    comparisons) into one answer. `per_animal_df` is that function's own
    `per_animal_df` return (Animal_Name, reward_seq, persisted_side,
    p_switch, n_trials).

    Two pooling methods, reported side by side (this project's standing
    convention of disclosing both a naive and a corrected version rather
    than picking one silently):

      - "stratified" (primary): for each animal, average p_switch across
        only the reward_seq values with data for >= `min_animals_per_reward_seq`
        animals (excludes reward_seq='010'/'101', each from a single
        animal) -- each reward_seq contributes EQUALLY to an animal's
        Right/Left mean regardless of how many trials it had, avoiding a
        confound where one side's overall pooled rate is dominated by
        whichever reward_seq happened to contribute the most trials.
      - "naive": trial-count-weighted pooling per animal (equivalent to
        pooling raw trials directly, ignoring reward_seq entirely) --
        simpler, but a real reward_seq-composition confound is possible
        if, e.g., an animal's Right-persistence trials skew toward a
        reward_seq with a systematically different baseline switch rate
        than its Left-persistence trials.

    Both are then a single paired t-test across animals (n up to 6).
    """
    from scipy import stats as scipy_stats

    # Eligibility must require BOTH sides present for the same animal
    # (a matched pair), not merely that the animal has a row for either
    # side -- counting any row overcounts reward_seq values where most
    # animals only ever persisted one side (e.g. '010'/'101', which have
    # rows from several animals but a real Right-AND-Left pair for only 1).
    reward_seq_pair_counts = {
        rs: int(g.pivot(index="Animal_Name", columns="persisted_side", values="p_switch").dropna().shape[0])
        for rs, g in per_animal_df.groupby("reward_seq")
    }
    eligible_reward_seqs = [rs for rs, n in reward_seq_pair_counts.items() if n >= min_animals_per_reward_seq]
    d_elig = per_animal_df[per_animal_df["reward_seq"].isin(eligible_reward_seqs)]

    # Stratified: equal weight per reward_seq, per animal.
    stratified_wide = (
        d_elig.groupby(["Animal_Name", "persisted_side"])["p_switch"].mean()
        .unstack("persisted_side").dropna()
    )
    t_strat, p_strat = scipy_stats.ttest_rel(stratified_wide["Right"], stratified_wide["Left"])

    # Naive: trial-count-weighted pooling per animal (== pooling raw trials).
    d_elig = d_elig.copy()
    d_elig["n_switch"] = d_elig["p_switch"] * d_elig["n_trials"]
    naive = d_elig.groupby(["Animal_Name", "persisted_side"]).agg(
        n_switch=("n_switch", "sum"), n_trials=("n_trials", "sum")
    ).reset_index()
    naive["p_switch"] = naive["n_switch"] / naive["n_trials"]
    naive_wide = naive.pivot(index="Animal_Name", columns="persisted_side", values="p_switch").dropna()
    t_naive, p_naive = scipy_stats.ttest_rel(naive_wide["Right"], naive_wide["Left"])

    return {
        "eligible_reward_seqs": eligible_reward_seqs,
        "stratified": {
            "n_animals": int(len(stratified_wide)),
            "mean_right": float(stratified_wide["Right"].mean()),
            "mean_left": float(stratified_wide["Left"].mean()),
            "t": float(t_strat), "p": float(p_strat),
            "per_animal": stratified_wide.reset_index().to_dict(orient="records"),
        },
        "naive_trial_weighted": {
            "n_animals": int(len(naive_wide)),
            "mean_right": float(naive_wide["Right"].mean()),
            "mean_left": float(naive_wide["Left"].mean()),
            "t": float(t_naive), "p": float(p_naive),
            "per_animal": naive_wide.reset_index().to_dict(orient="records"),
        },
    }
