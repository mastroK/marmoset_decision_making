"""Per-animal Q-learning + sticky choice-kernel model, and the "does the
animal follow Q-values" panels it feeds (Supplementary Figure 3).

Ported from _source_archive/by_figure/SuppFig3/SuppFig3_candidate_07_GLM_HMM_v1_cleaned.ipynb,
STEP 1-9 (lines 1-538: data prep through the sticky-model Q-value computation)
and the Q-following aggregation cells (In[17]/In[18], lines 999-1045). The
notebook also contains a later, explicitly abandoned "OLD STEP 8" forgetting-
model variant (line 544, commented "DO NOT USE ... NO LONGER USING THIS
BECAUSE IT LEADS TO AN UNINTERPRETABLE TAU") -- not ported.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy import stats as scipy_stats


def _code_choice(physical_choice):
    if physical_choice in ("BottomLeft", "LeftCenter", "TopLeft"):
        return 0
    elif physical_choice in ("BottomRight", "RightCenter", "TopRight"):
        return 1
    return np.nan


def prepare_2abt_dataframe(df, task_type, prob_condition):
    """Filter to the task/probability condition and add the minimal set of
    derived columns the Q-following panels need: Choice_Binary (with the
    vertical-session Top/Bottom recoding), Outcome_Binary, Trial_in_Block.

    (lines 86-98 filter/sort; 111-159 choice coding incl. vertical-session
    detection; 171-172 Outcome_Binary; 285-291 Trial_in_Block via BlockCount)
    """
    df_filtered = df[
        (df["Task_Type"] == task_type) & (df["prob_Condition"] == prob_condition)
    ].copy()
    df_filtered = df_filtered.sort_values(
        ["Animal_Name", "Session_ID", "Trial"]
    ).reset_index(drop=True)

    df_filtered["Choice_Binary"] = df_filtered["PhysicalChoice"].map(_code_choice)

    session_sides = df_filtered.groupby("Session_ID").apply(
        lambda x: pd.Series({
            "has_left": x["PhysicalChoice"].str.contains("Left").any(),
            "has_right": x["PhysicalChoice"].str.contains("Right").any(),
        })
    ).reset_index()
    vertical_sessions = session_sides[
        ~session_sides["has_left"] | ~session_sides["has_right"]
    ]["Session_ID"].tolist()
    if vertical_sessions:
        vmask = df_filtered["Session_ID"].isin(vertical_sessions)
        df_filtered.loc[vmask, "Choice_Binary"] = (
            df_filtered.loc[vmask, "PhysicalChoice"].str.contains("Top").astype(int)
        )

    df_filtered["Outcome_Binary"] = df_filtered["Outcome"].astype(int)
    df_filtered["Trial_in_Block"] = (
        df_filtered.groupby(["Session_ID", "BlockCount"]).cumcount()
    )

    # ADDITIVE (Figure 5): whether the right option was the high-reward-
    # probability side on this trial, needed for panel a's block-background
    # shading. Same formula as src/features/state_features.py's
    # build_state_features_df (same Reversal/80-20 snapshot, same ProbReward
    # semantics -- ProbReward is the CHOSEN option's own reward probability,
    # so together with Choice_Binary it identifies which side was high-prob
    # regardless of which side was actually chosen). Does not alter any
    # existing column; Supp Fig 3 does not read this field.
    df_filtered["High_Prob_Is_Right"] = (
        ((df_filtered["ProbReward"] == 0.8) & (df_filtered["Choice_Binary"] == 1))
        | ((df_filtered["ProbReward"] == 0.2) & (df_filtered["Choice_Binary"] == 0))
    )
    return df_filtered


def _qlearn_sticky_nll(params, choices, rewards):
    alpha, beta, kappa = params
    if not (0 < alpha < 1) or beta <= 0:
        return np.inf

    Q = np.array([0.5, 0.5], dtype=float)
    nll = 0.0
    prev_choice = None
    for c, r in zip(choices, rewards):
        stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
        z = beta * (Q[1] - Q[0]) + kappa * stick
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        p = p1 if c == 1 else (1.0 - p1)
        nll -= np.log(p + 1e-12)
        Q[c] += alpha * (r - Q[c])
        prev_choice = c
    return nll


def fit_sticky_qlearning(df, x0, bounds):
    """Fit the per-animal Q-learning + sticky choice-kernel model
    (alpha=learning rate, beta=inverse temperature, kappa=stickiness).
    (lines 391-453)
    """
    scipy_bounds = [
        tuple(bounds["alpha"]), tuple(bounds["beta"]), tuple(bounds["kappa"]),
    ]
    rows = []
    for animal in df["Animal_Name"].unique():
        animal_data = df[df["Animal_Name"] == animal].sort_values(["Session_ID", "Trial"])
        choices = animal_data["Choice_Binary"].values.astype(int)
        rewards = animal_data["Outcome_Binary"].values.astype(int)

        res = minimize(
            _qlearn_sticky_nll, x0=np.array(x0), args=(choices, rewards),
            method="L-BFGS-B", bounds=scipy_bounds,
        )
        rows.append({
            "animal": animal, "alpha": res.x[0], "beta": res.x[1],
            "kappa": res.x[2], "nll": res.fun,
        })
    return pd.DataFrame(rows)


def compute_qvalues_sticky(df, params_df):
    """Run the fitted sticky Q-learning model forward on each animal's
    actual choices/outcomes to generate trial-by-trial Q-values.
    (lines 472-532)

    ADDITIVE (Figure 4): also returns "Model_Log_Odds", the raw pre-sigmoid
    decision variable z = beta*(Q_right-Q_left) + kappa*stick -- i.e.
    logit(Model_P_Right), and exactly the manuscript's Fig 4 V(t) ("Action
    Value: V(t) = beta*[Q(right,t)-Q(left,t)] + kappa*C(t)"). Kept in its
    raw (unsquashed, clipped) form rather than re-derived via
    log(p/(1-p)) from Model_P_Right, which would lose precision once
    Model_P_Right saturates near 0/1. Every previously-existing field
    (Q_left, Q_right, Q_diff, Model_P_Right) is computed identically to
    before this was added -- Supp Fig 3, which depends on this function, is
    unaffected.
    """
    all_rows = []
    for _, param_row in params_df.iterrows():
        animal = param_row["animal"]
        alpha, beta, kappa = param_row["alpha"], param_row["beta"], param_row["kappa"]

        animal_data = df[df["Animal_Name"] == animal].sort_values(["Session_ID", "Trial"])
        for session_id, session_data in animal_data.groupby("Session_ID", sort=False):
            session_data = session_data.sort_values("Trial")
            Q = np.array([0.5, 0.5], dtype=float)
            prev_choice = None
            for _, trial in session_data.iterrows():
                c = int(trial["Choice_Binary"])
                r = int(trial["Outcome_Binary"])
                stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
                z = beta * (Q[1] - Q[0]) + kappa * stick
                z_clipped = np.clip(z, -50, 50)
                p_right = 1.0 / (1.0 + np.exp(-z_clipped))
                all_rows.append({
                    "Animal_Name": animal, "Session_ID": session_id, "Trial": trial["Trial"],
                    "Q_left": Q[0], "Q_right": Q[1], "Q_diff": Q[1] - Q[0],
                    "Model_P_Right": p_right, "Model_Log_Odds": z_clipped,
                })
                Q[c] += alpha * (r - Q[c])
                prev_choice = c

    qval_df = pd.DataFrame(all_rows)
    df_q = df.merge(
        qval_df[["Animal_Name", "Session_ID", "Trial", "Q_left", "Q_right", "Q_diff",
                 "Model_P_Right", "Model_Log_Odds"]],
        on=["Animal_Name", "Session_ID", "Trial"], how="left",
    )
    return df_q


def add_chose_higher_q(df_q):
    """Whether the animal's choice matched the model's higher-valued option.
    Ties (Q_diff == 0) are excluded, matching the source. (lines 1002-1009)
    """
    df_q = df_q.copy()
    df_q["Chose_Higher_Q"] = (
        ((df_q["Choice_Binary"] == 1) & (df_q["Q_diff"] > 0))
        | ((df_q["Choice_Binary"] == 0) & (df_q["Q_diff"] < 0))
    ).astype(float)
    # ADDITIVE (Figure 5): "violation" = the complementary framing of the
    # same boolean ("chose the LOWER-Q option" rather than "chose the
    # higher-Q option") -- computed from Chose_Higher_Q without changing it.
    df_q["Violation"] = 1.0 - df_q["Chose_Higher_Q"]
    return df_q[df_q["Q_diff"] != 0].copy()


def q_following_by_animal(df_valid):
    """Per-animal, per-session P(follow Q-values); animal-level mean/sem
    across sessions, plus a per-animal Wilcoxon test vs. chance (0.5).
    (Panel A, lines 1026-1031)
    """
    session_animal = (
        df_valid.groupby(["Animal_Name", "Session_ID"])["Chose_Higher_Q"].mean().reset_index()
    )
    animal_means = session_animal.groupby("Animal_Name")["Chose_Higher_Q"].mean()
    stat, p = scipy_stats.wilcoxon(animal_means - 0.5)
    return {
        "session_animal": session_animal,
        "animal_means": animal_means.to_dict(),
        "n_animals": int(animal_means.shape[0]),
        "mean": float(animal_means.mean()),
        "sem": float(animal_means.sem()),
        "wilcoxon_stat_vs_chance": float(stat),
        "wilcoxon_p_vs_chance": float(p),
    }


def q_following_by_trial_in_block(df_valid):
    """Per-animal, per-session, per-block-position P(follow Q-values).
    (Panel B, lines 1034-1038)
    """
    return (
        df_valid.groupby(["Animal_Name", "Session_ID", "Trial_in_Block"])["Chose_Higher_Q"]
        .mean().reset_index()
    )


def q_following_by_outcome(df_valid):
    """Per-session P(follow Q-values) split by previous-trial outcome
    (win/loss), plus a per-animal paired Wilcoxon test (win vs. loss).
    (Panel C, lines 1040-1045; the source only plots session-level dots --
    the per-animal paired test is added here, matching the pseudoreplication
    fix applied throughout this pipeline for win/loss-type comparisons.)
    """
    session_outcome = (
        df_valid.groupby(["Animal_Name", "Session_ID", "Outcome_Binary"])["Chose_Higher_Q"]
        .mean().reset_index()
    )
    animal_outcome = (
        session_outcome.groupby(["Animal_Name", "Outcome_Binary"])["Chose_Higher_Q"]
        .mean().unstack("Outcome_Binary")
    )
    animal_outcome = animal_outcome.dropna()
    stat, p = scipy_stats.wilcoxon(animal_outcome[1], animal_outcome[0])

    # ADDITIVE (Figure 5c): the manuscript's Fig 5c reports this same
    # rewarded-vs-unrewarded adherence comparison with a *paired t-test*
    # (not the Wilcoxon signed-rank test Supp Fig 3 already uses) --
    # added as new keys alongside the existing wilcoxon_stat/wilcoxon_p,
    # which are untouched. Per this pipeline's standing pseudoreplication
    # fix, both a naive session-level paired test (n=sessions, animal
    # membership ignored -- pseudoreplicated) and the corrected per-animal
    # paired test (n=5 animals) are reported; the per-animal one is treated
    # as the real result.
    ttest_stat_by_animal, ttest_p_by_animal = scipy_stats.ttest_rel(
        animal_outcome[1], animal_outcome[0]
    )
    session_pivot = (
        session_outcome.pivot_table(index="Session_ID", columns="Outcome_Binary",
                                     values="Chose_Higher_Q")
        .dropna()
    )
    naive_ttest_stat, naive_ttest_p = scipy_stats.ttest_rel(
        session_pivot[1], session_pivot[0]
    )

    return {
        "session_outcome": session_outcome,
        "n_animals": int(animal_outcome.shape[0]),
        "p_follow_win": float(animal_outcome[1].mean()),
        "p_follow_loss": float(animal_outcome[0].mean()),
        "wilcoxon_stat": float(stat),
        "wilcoxon_p": float(p),
        "paired_ttest_stat_by_animal": float(ttest_stat_by_animal),
        "paired_ttest_p_by_animal": float(ttest_p_by_animal),
        "paired_ttest_df_by_animal": int(animal_outcome.shape[0] - 1),
        "paired_ttest_stat_naive_session_level": float(naive_ttest_stat),
        "paired_ttest_p_naive_session_level": float(naive_ttest_p),
        "n_sessions_naive": int(session_pivot.shape[0]),
    }


def violation_rate_by_animal(df_valid):
    """Violation rate (1 - Chose_Higher_Q) per session/animal -- Figure 5b.
    ADDITIVE: reuses q_following_by_animal's own per-session/per-animal
    grouping and simply reframes it as violation rather than following (the
    same underlying per-session quantity, complementary framing)."""
    follow = q_following_by_animal(df_valid)
    session_animal = follow["session_animal"].copy()
    session_animal["Violation_Rate"] = 1.0 - session_animal["Chose_Higher_Q"]
    animal_means = session_animal.groupby("Animal_Name")["Violation_Rate"].mean()
    return {
        "session_animal": session_animal,
        "animal_means": animal_means.to_dict(),
        "n_animals": int(animal_means.shape[0]),
        # "Overall mean ... across all animals and sessions" (Figure 5b
        # legend, dashed line) -- pooled equally over all sessions,
        # NOT animal-weighted (see mean_of_animal_means below for the
        # animal-weighted alternative, reported alongside for transparency).
        "overall_mean_pooled_sessions": float(session_animal["Violation_Rate"].mean()),
        "mean_of_animal_means": float(animal_means.mean()),
        "sem_of_animal_means": float(animal_means.sem()),
    }


def violation_rate_by_trial_in_block(df_valid):
    """Violation rate (1 - Chose_Higher_Q) per animal/session/trial-in-block
    -- Figure 5d. ADDITIVE: reuses q_following_by_trial_in_block's own
    grouping, reframed as violation."""
    by_pos = q_following_by_trial_in_block(df_valid).copy()
    by_pos["Violation_Rate"] = 1.0 - by_pos["Chose_Higher_Q"]
    return by_pos


def add_policy_predicted_adherence(df_valid):
    """Per-trial "policy-predicted adherence" -- the theoretical quantity
    the manuscript's Results text calls the policy-predicted adherence rate
    (Supp Fig 3 panel b). The Results text gives two phrasings in the same
    sentence, presented as equivalent ("...or, in other words..."), that are
    NOT actually the same quantity for a binary softmax policy:
      (1) "the probability that a single draw from the policy produces the
          argmax choice" = max(p, 1-p)
      (2) "the probability that two independent draws from the policy
          agree" = p^2 + (1-p)^2
    where p = Model_P_Right. This is a genuine ambiguity in the manuscript
    text itself, not resolved by the text alone -- but empirically
    resolved here: computed per-animal and compared against the
    manuscript's own printed cross-check numbers (observed 0.704+/-0.017,
    policy-predicted 0.722+/-0.013, t(4)=-0.871, p=0.433), definition (2)
    reproduces those numbers almost exactly (0.721+/-0.013, t(4)=-0.974,
    p=0.385) while definition (1) does not (0.818+/-0.012, t(4)=-5.520,
    p=0.0053). Definition (2), p^2+(1-p)^2, is therefore used as the
    primary "Policy_Predicted_Adherence" column (this is picking the
    textually-intended formula using independent evidence, not tuning a
    free parameter to match a value); definition (1) is also computed as
    "Policy_Argmax_Single_Draw" and reported alongside as a cross-check,
    not silently discarded. ADDITIVE: adds two new columns; does not touch
    Chose_Higher_Q/Violation or any existing column.
    """
    df_valid = df_valid.copy()
    p = df_valid["Model_P_Right"]
    df_valid["Policy_Predicted_Adherence"] = p**2 + (1.0 - p) ** 2
    df_valid["Policy_Argmax_Single_Draw"] = np.maximum(p, 1.0 - p)
    return df_valid


def observed_vs_predicted_adherence_by_animal(df_valid_pred):
    """Per-animal observed P(Q-adherence) vs. policy-predicted adherence
    rate (Supp Fig 3 panel b) -- same per-session-then-per-animal
    aggregation as q_following_by_animal, applied to Chose_Higher_Q
    (observed), Policy_Predicted_Adherence (predicted, primary "two draws
    agree" definition), and Policy_Argmax_Single_Draw (predicted,
    alternative "single draw produces the argmax" reading -- see
    add_policy_predicted_adherence docstring), plus paired t-tests across
    animals (observed vs. each predicted definition; matching the
    manuscript's own t(4) comparison in form, Results text/Supp Fig 3b
    legend)."""
    session_animal = (
        df_valid_pred.groupby(["Animal_Name", "Session_ID"])[
            ["Chose_Higher_Q", "Policy_Predicted_Adherence", "Policy_Argmax_Single_Draw"]
        ].mean().reset_index()
    )
    animal_means = session_animal.groupby("Animal_Name")[
        ["Chose_Higher_Q", "Policy_Predicted_Adherence", "Policy_Argmax_Single_Draw"]
    ].mean()
    stat, p = scipy_stats.ttest_rel(
        animal_means["Chose_Higher_Q"], animal_means["Policy_Predicted_Adherence"]
    )
    stat_alt, p_alt = scipy_stats.ttest_rel(
        animal_means["Chose_Higher_Q"], animal_means["Policy_Argmax_Single_Draw"]
    )
    return {
        "animal_observed": animal_means["Chose_Higher_Q"].to_dict(),
        "animal_predicted": animal_means["Policy_Predicted_Adherence"].to_dict(),
        "n_animals": int(animal_means.shape[0]),
        "observed_mean": float(animal_means["Chose_Higher_Q"].mean()),
        "observed_sem": float(animal_means["Chose_Higher_Q"].sem()),
        "predicted_mean": float(animal_means["Policy_Predicted_Adherence"].mean()),
        "predicted_sem": float(animal_means["Policy_Predicted_Adherence"].sem()),
        "paired_ttest_df": int(animal_means.shape[0] - 1),
        "paired_ttest_stat": float(stat),
        "paired_ttest_p": float(p),
        # Cross-check using the sentence's other ("single draw produces the
        # argmax") reading -- see docstring above. Not the primary result.
        "animal_predicted_argmax_single_draw": animal_means["Policy_Argmax_Single_Draw"].to_dict(),
        "predicted_argmax_single_draw_mean": float(animal_means["Policy_Argmax_Single_Draw"].mean()),
        "predicted_argmax_single_draw_sem": float(animal_means["Policy_Argmax_Single_Draw"].sem()),
        "paired_ttest_stat_argmax_single_draw": float(stat_alt),
        "paired_ttest_p_argmax_single_draw": float(p_alt),
    }


def add_prev_outcome(df_valid):
    """Previous-trial binary outcome within (animal, session) -- needed to
    split panel c's observed-minus-predicted-adherence curve by whether the
    prior trial was rewarded. ADDITIVE: adds Prev_Outcome_Binary only."""
    d = df_valid.sort_values(["Animal_Name", "Session_ID", "Trial"]).copy()
    d["Prev_Outcome_Binary"] = d.groupby(["Animal_Name", "Session_ID"])["Outcome_Binary"].shift(1)
    return d


def adherence_diff_by_qdiff_and_outcome(df_valid_pred_prev, n_bins):
    """Observed-minus-predicted adherence (Chose_Higher_Q -
    Policy_Predicted_Adherence) as a function of |Q_right - Q_left|,
    split by whether the previous trial was rewarded -- Supp Fig 3 panel c.
    Bins |Q_diff| into `n_bins` equal-width bins over its observed range.
    """
    d = df_valid_pred_prev[df_valid_pred_prev["Prev_Outcome_Binary"].notna()].copy()
    d["Abs_Q_Diff"] = d["Q_diff"].abs()
    d["Adherence_Minus_Predicted"] = d["Chose_Higher_Q"] - d["Policy_Predicted_Adherence"]
    d["qdiff_bin"] = pd.cut(d["Abs_Q_Diff"], bins=n_bins)

    records = []
    for prev_outcome, sub in d.groupby("Prev_Outcome_Binary"):
        grouped = sub.groupby("qdiff_bin", observed=True)["Adherence_Minus_Predicted"].agg(
            mean="mean", sem="sem", n="count"
        ).reset_index()
        grouped["bin_center"] = grouped["qdiff_bin"].apply(lambda iv: iv.mid).astype(float)
        grouped["prev_outcome"] = int(prev_outcome)
        records.append(grouped)
    return pd.concat(records, ignore_index=True)


def violation_direction_by_animal(df_valid):
    """Direction of Q-value violations (left vs. right side chosen on
    violation trials) by animal -- Supp Fig 3 panel d. Choice_Binary is
    1=right, 0=left, so the mean of Choice_Binary among violation trials
    is the proportion directed right. Includes both the population-pooled
    test (matching the manuscript's own binomial test, pseudoreplicated
    across sessions/animals) and a per-animal Wilcoxon-vs-chance test (the
    pseudoreplication-corrected version), per this pipeline's standing
    convention -- both reported side by side.
    """
    viol = df_valid[df_valid["Violation"] == 1]
    per_animal_prop_right = viol.groupby("Animal_Name")["Choice_Binary"].mean()

    population_prop_right = float(viol["Choice_Binary"].mean())
    population_n = int(len(viol))
    population_binom = scipy_stats.binomtest(
        int(viol["Choice_Binary"].sum()), population_n, p=0.5
    )

    per_animal_stat, per_animal_p = scipy_stats.wilcoxon(per_animal_prop_right - 0.5)

    return {
        "n_violations_total": population_n,
        "n_violations_per_animal": {
            str(k): int(v) for k, v in viol.groupby("Animal_Name").size().items()
        },
        "population_prop_right": population_prop_right,
        "population_prop_left": 1.0 - population_prop_right,
        "population_binomial_p_naive": float(population_binom.pvalue),
        "per_animal_prop_right": {str(k): float(v) for k, v in per_animal_prop_right.items()},
        "per_animal_prop_left": {str(k): float(1.0 - v) for k, v in per_animal_prop_right.items()},
        "per_animal_wilcoxon_stat_vs_chance": float(per_animal_stat),
        "per_animal_wilcoxon_p_vs_chance": float(per_animal_p),
        "n_animals": int(per_animal_prop_right.shape[0]),
    }


def add_next_trial_fields(df_valid):
    """Next-trial (within animal/session) outcome and Q-following status --
    shared prep for Supp Fig 3 panels e (reward on next trial) and f
    (Q-following on next trial), both conditioned on the CURRENT trial's
    Violation status. ADDITIVE: adds Next_Outcome_Binary/Next_Chose_Higher_Q
    only."""
    d = df_valid.sort_values(["Animal_Name", "Session_ID", "Trial"]).copy()
    d["Next_Outcome_Binary"] = d.groupby(["Animal_Name", "Session_ID"])["Outcome_Binary"].shift(-1)
    d["Next_Chose_Higher_Q"] = d.groupby(["Animal_Name", "Session_ID"])["Chose_Higher_Q"].shift(-1)
    return d


def paired_by_violation(df_with_next, value_col):
    """Session-level (naive, matching the manuscript legend's own paired
    t-test -- pseudoreplicated across animals) and per-animal-corrected
    paired comparison of `value_col` on trials following a Q-violation vs.
    trials following adherence. Shared aggregation for Supp Fig 3 panel e
    (value_col="Next_Outcome_Binary") and panel f
    (value_col="Next_Chose_Higher_Q").
    """
    d = df_with_next[df_with_next[value_col].notna()].copy()
    session_level = (
        d.groupby(["Animal_Name", "Session_ID", "Violation"])[value_col]
        .mean().reset_index()
    )
    session_pivot = session_level.pivot_table(
        index=["Animal_Name", "Session_ID"], columns="Violation", values=value_col
    ).dropna()
    naive_stat, naive_p = scipy_stats.ttest_rel(session_pivot[1.0], session_pivot[0.0])

    animal_level = (
        session_level.groupby(["Animal_Name", "Violation"])[value_col]
        .mean().unstack("Violation").dropna()
    )
    animal_stat, animal_p = scipy_stats.ttest_rel(animal_level[1.0], animal_level[0.0])

    return {
        "session_pairs": session_pivot.rename(
            columns={0.0: "after_adherence", 1.0: "after_violation"}
        ).reset_index().to_dict(orient="records"),
        "n_sessions": int(session_pivot.shape[0]),
        "after_violation_mean": float(session_pivot[1.0].mean()),
        "after_violation_sem": float(session_pivot[1.0].sem()),
        "after_adherence_mean": float(session_pivot[0.0].mean()),
        "after_adherence_sem": float(session_pivot[0.0].sem()),
        "naive_session_level_ttest_stat": float(naive_stat),
        "naive_session_level_ttest_p": float(naive_p),
        "naive_session_level_df": int(session_pivot.shape[0] - 1),
        "n_animals": int(animal_level.shape[0]),
        "per_animal_ttest_stat": float(animal_stat),
        "per_animal_ttest_p": float(animal_p),
        "per_animal_df": int(animal_level.shape[0] - 1),
    }


def violation_rate_vs_accuracy_by_session(df_valid):
    """Session-level violation rate vs. accuracy (p(High Prob)) per animal,
    with per-animal linear regression (r, p) -- Supp Fig 3 panel g.
    "High Prob" here is (ProbReward == 0.8), i.e. whether the CHOSEN option
    was the higher-reward-probability side that trial (same ProbReward
    semantics used elsewhere in this module -- see prepare_2abt_dataframe's
    High_Prob_Is_Right docstring). Sessions are ordered chronologically by
    Date within animal (for the legend's early-to-late color gradient).
    """
    d = df_valid.copy()
    d["High_Prob"] = (d["ProbReward"] == 0.8).astype(int)
    session_level = d.groupby(["Animal_Name", "Session_ID"]).agg(
        violation_rate=("Violation", "mean"),
        p_high_prob=("High_Prob", "mean"),
        date=("Date", "first"),
    ).reset_index()

    results = {}
    for animal, sdf in session_level.groupby("Animal_Name"):
        sdf = sdf.sort_values("date").reset_index(drop=True)
        slope, intercept, r, p, se = scipy_stats.linregress(sdf["violation_rate"], sdf["p_high_prob"])
        results[animal] = {
            "n_sessions": int(len(sdf)),
            "r": float(r),
            "p": float(p),
            "slope": float(slope),
            "intercept": float(intercept),
            "sessions": sdf[["Session_ID", "violation_rate", "p_high_prob", "date"]].assign(
                session_order=range(len(sdf))
            ).to_dict(orient="records"),
        }
    return results


def select_representative_sessions(df_valid, animal):
    """Lowest- and highest-violation-rate sessions for one animal --
    Figure 5a's representative low/high violation sessions.

    JUDGMENT CALL: the manuscript names the animal (Fiona, Figure 5a
    legend) but does not give an exact session-selection rule beyond "low"
    and "high" violation rate sessions. This picks the single lowest- and
    single highest- per-session violation-rate session for that animal
    among its own sessions in the dataset (ties broken by Session_ID sort
    order, which does not occur in practice since violation rate is a
    continuous per-session mean over many trials).
    """
    sub = df_valid[df_valid["Animal_Name"] == animal]
    session_rate = sub.groupby("Session_ID")["Violation"].mean().sort_values()
    low_session, high_session = session_rate.index[0], session_rate.index[-1]
    return {
        "animal": animal,
        "low_session": low_session,
        "low_violation_rate": float(session_rate.loc[low_session]),
        "high_session": high_session,
        "high_violation_rate": float(session_rate.loc[high_session]),
        "n_sessions": int(session_rate.shape[0]),
        "session_rates": session_rate.to_dict(),
    }
