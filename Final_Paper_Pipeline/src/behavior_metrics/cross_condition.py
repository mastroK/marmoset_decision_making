"""Fig 8 -- 80-20 vs. 100-0 condition comparison.

REBUILD NOTE: this figure has been rebuilt on the manuscript's canonical
7-state taxonomy (`state_classifier.classify_state_v7`), replacing the
earlier build's `classify_state_updated`-based Figure_8f/Figure_8fg-only
scope. The 100-0 data-reconstruction machinery below (`build_100_0_features_df`,
`create_correct_pseudo_sessions`) is unchanged and reused as-is -- only the
classifier applied to its output (in `notebooks/Fig8.ipynb`) and the
downstream panels (`src/plotting/fig8.py`, rewritten) have changed. New
cross-condition comparison helpers for the rebuilt panels c/d/e
(`paired_animal_comparison`, `compare_state_proportions`,
`compare_win_stay_lose_switch`, `switch_vs_consecutive_losses`) are added
at the bottom of this module; `compute_transition_probabilities`/
`compute_diff_matrix` (panel f) and `fit_sticky_and_compare` are unchanged
and still classifier-agnostic (they only read the `Behavioral_State` column
already computed, whichever classifier produced it).

Originally ported from _source_archive/by_figure/Fig8/09b_CrossProb_StickyModel_v1.ipynb,
scoped to exactly the cells feeding its two explicit output filenames,
`Figure_8f_transition_heatmap` (In[185], lines ~3450-3552) and `Figure_8fg`
(In[197], lines ~3585-3759), plus every upstream cell those two depend on.

80-20 side
----------
The source's `df_80_corrected` is just the already-reconstructed 80-20
state-features dataframe (`src/features/state_features.build_state_features_df`,
same one Fig 6/Supp Fig 4 use) filtered to the 5 animals who ran both
conditions -- no additional pseudo-session splitting is applied to it in the
source (`df_80_mixed` is None, lines 149-266, so `df_80_pseudo == df_80`
unchanged; `df_80_corrected = df_80_pseudo.copy()`, line 1906). See
`Fig8.ipynb` for that side; this module only builds the 100-0 side.

100-0 side
----------
The source's 100-0 preprocessing (lines 92-1120ish) is the messiest part of
this notebook and contains a genuinely abandoned implementation:

- TWO competing pseudo-session-splitting functions. `create_pseudo_sessions`/
  `detect_transitions` (lines 159-240) is an earlier, cruder attempt (splits
  wherever `PhysicalChoice` sets differ block-to-block) and is NOT ported
  here. `create_correct_pseudo_sessions` (lines 361-395), named "_correct" by
  the original author and applied later using the `Has_Cued_Transitions`/
  `Original_Session_ID`/`Target1_location`/`Target2_location` columns already
  present in the raw snapshot, is the one used here.

- High_Prob_Is_Right / Expected_PRight are computed TWICE: an earlier
  block-majority-vote version (lines 302-331, discarded) and a later version
  right after `create_correct_pseudo_sessions` (lines 409-447) that is what
  actually feeds everything downstream in the source's own execution order.
  Only the later version is implemented here.

- A THIRD session-identity concept, `Fit_Session_ID` (source line 1111 on:
  `Original_Session_ID` for sessions with no cued transitions,
  `New_Pseudo_Session_ID` otherwise), is introduced later and used for a
  separate per-session Q-learning refit (STEP 8/9) that feeds other,
  out-of-scope panels in that notebook (state-occupancy, adaptation-curve,
  raster panels) -- NOT Figure_8f/8fg. That refit and `Fit_Session_ID` are
  not reconstructed here; see config.yaml's fig8 section for why.

- A subtlety worth flagging plainly rather than silently matching: the
  source's own Figure_8f/8fg cells group next-state shifts by plain
  `Session_ID` (line 3392, `df.groupby(['Animal_Name','Session_ID'])`). For
  80-20 that is unambiguously the real session id. For 100-0, at that point
  in the source notebook `Session_ID` is actually a stale leftover from the
  superseded, un-ported `create_pseudo_sessions` call (that function
  literally overwrites `Session_ID` with its own cruder pseudo-session id,
  line 239, and `create_correct_pseudo_sessions` never touches `Session_ID`
  afterward -- it only adds `New_Pseudo_Session_ID`). Since this pipeline
  deliberately does not implement the superseded splitter, there is no stale
  column to (accidentally) reproduce here; instead, `Session_ID` for the
  100-0 side is set to the `create_correct_pseudo_sessions`-derived segment
  id throughout -- the only legitimate session-like grouping actually built.
  This is a judgment call, not a claim that it exactly reproduces the
  source's (arguably buggy) execution -- flagged for the user to weigh in on.

Sticky Q-learning model (STEP 6, lines 475-548): reuses
`q_following_model.fit_sticky_qlearning`/`compute_qvalues_sticky` verbatim
(not reimplemented) -- same model form, same bounds as
`config.yaml`'s `suppfig3.sticky_model_bounds`. This does not feed
Figure_8f/8fg (state classification only needs Rolling_Accuracy/
Choice_Deviation/Rolling_Switch_Rate/Trials_Since_Transition/
High_Prob_Is_Right/Rolling_PRight, none of which depend on Q-values) --
it is included because point 4 of the task brief asks for it explicitly, as
a parallel sanity-check / cross-condition comparison (mirroring the source's
own STEP 7 `Parameter_Comparison_80_100.svg`, which is not one of the two
confirmed Fig 8 output files and so is not re-plotted here).
"""

import numpy as np
import pandas as pd
from scipy import stats


def _code_choice(physical_choice):
    if physical_choice in ("BottomLeft", "LeftCenter", "TopLeft"):
        return 0
    elif physical_choice in ("BottomRight", "RightCenter", "TopRight"):
        return 1
    return np.nan


def is_cued_transition(row, reversal_locations):
    """A block transition is "cued" (target set changed) when the pair of
    target locations is NOT a subset of the reversal-only location set.
    (line 361-363)
    """
    locs = {row["Target1_location"], row["Target2_location"]}
    return not locs.issubset(reversal_locations)


def create_correct_pseudo_sessions(df, reversal_locations):
    """Split each raw session at its CUED block transitions only (sessions
    with no cued transitions at all stay intact as one segment). Ported
    unchanged in logic from `create_correct_pseudo_sessions`, lines 365-395 --
    the "_correct" (author's own naming) pseudo-session splitter, the only
    one this pipeline uses (see module docstring for the superseded sibling
    this deliberately does NOT port).

    Requires `Original_Session_ID`, `Has_Cued_Transitions`, `BlockCount`,
    `Block_Transition`, `Target1_location`, `Target2_location` already
    present on `df`.
    """
    records = []
    for orig_session, sdf in df.groupby("Original_Session_ID"):
        sdf = sdf.sort_values("Trial").copy()

        if not sdf["Has_Cued_Transitions"].iloc[0]:
            sdf["New_Pseudo_Session_ID"] = orig_session + "_SEG0"
            records.append(sdf)
            continue

        transition_rows = sdf[sdf["Block_Transition"] == 1].copy()
        cued_rows = transition_rows[
            transition_rows.apply(lambda r: is_cued_transition(r, reversal_locations), axis=1)
        ]
        cued_block_counts = set(cued_rows["BlockCount"].values)

        seg = 0
        for block_count, block_df in sdf.groupby("BlockCount"):
            if block_count in cued_block_counts:
                seg += 1
            sdf.loc[block_df.index, "New_Pseudo_Session_ID"] = f"{orig_session}_SEG{seg}"

        records.append(sdf)

    return pd.concat(records).reset_index(drop=True)


def build_100_0_features_df(df_raw, overlapping_animals, reversal_locations, rolling_window):
    """Reconstruct the 100-0 dataframe with every column
    `state_classifier.classify_state_v7` (Fig 8 rebuild; previously
    `classify_state_updated`) needs, from the raw `100_reversal_data_only.csv`
    snapshot. See module docstring for the exact
    scope/ordering decisions (which of the source's two computations of each
    quantity is authoritative, and what is deliberately not ported).
    """
    df = df_raw[df_raw["Animal_Name"].isin(overlapping_animals)].copy()
    df = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).reset_index(drop=True)

    # Choice_Binary, incl. vertical-session Top/Bottom recoding (same pattern
    # as state_features.py / q_following_model.py -- the raw 100-0 snapshot
    # has 42/52 overlapping-animal sessions using vertical Top/Bottom targets).
    df["Choice_Binary"] = df["PhysicalChoice"].map(_code_choice)
    session_sides = df.groupby("Session_ID").apply(
        lambda x: pd.Series({
            "has_left": x["PhysicalChoice"].str.contains("Left", na=False).any(),
            "has_right": x["PhysicalChoice"].str.contains("Right", na=False).any(),
        })
    ).reset_index()
    vertical_sessions = session_sides[
        ~session_sides["has_left"] | ~session_sides["has_right"]
    ]["Session_ID"].tolist()
    if vertical_sessions:
        vmask = df["Session_ID"].isin(vertical_sessions)
        df.loc[vmask, "Choice_Binary"] = (
            df.loc[vmask, "PhysicalChoice"].str.contains("Top", na=False).astype(int)
        )

    # Outcome_Binary: the raw snapshot already carries a 0/1 `Outcome` column
    # (equivalent to the source's `Reward`.map({'Rewarded':1,'Unrewarded':0}),
    # line 131) -- using it directly matches state_features.py's convention.
    df["Outcome_Binary"] = df["Outcome"].astype(int)

    # Original_Session_ID = raw Session_ID (task brief point 3; no
    # create_pseudo_sessions call, so there is no pre-existing pseudo id).
    df["Original_Session_ID"] = df["Session_ID"]

    # Trial_in_Block / Block_Transition / Trials_Since_Transition, computed
    # directly on the raw per-session/per-BlockCount data -- same pattern as
    # state_features.build_state_features_df (lines 75-96 there).
    df["Trial_in_Block"] = df.groupby(["Original_Session_ID", "BlockCount"]).cumcount()
    df["Block_Transition"] = (
        df.groupby("Original_Session_ID")["BlockCount"].diff().fillna(0).abs().astype(bool).astype(int)
    )
    df["Trials_Since_Transition"] = 0
    for session in df["Original_Session_ID"].unique():
        session_mask = df["Original_Session_ID"] == session
        counter = 0
        trials_since = []
        for is_trans in df.loc[session_mask, "Block_Transition"]:
            if is_trans:
                counter = 0
            trials_since.append(counter)
            counter += 1
        df.loc[session_mask, "Trials_Since_Transition"] = trials_since

    # Pseudo-session correction (create_correct_pseudo_sessions ONLY -- see
    # module docstring) -> New_Pseudo_Session_ID becomes this pipeline's
    # `Session_ID` for everything downstream.
    df = create_correct_pseudo_sessions(df, reversal_locations)
    df["Session_ID"] = df["New_Pseudo_Session_ID"]
    df = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).reset_index(drop=True)

    # High_Prob / High_Prob_Is_Right -- LATER (authoritative) computation,
    # lines 409-415: High_Prob is the raw per-trial ProbReward of the CHOSEN
    # option (1.0 if the chosen side was the deterministically-rewarded one
    # this block, 0.0 otherwise), and High_Prob_Is_Right is recovered from
    # Choice_Binary == ProbReward. Flagged per task brief point 3 as unusual
    # -- see the empirical P(High_Prob) check this function's caller prints.
    df["High_Prob"] = df["ProbReward"]
    df["High_Prob_Is_Right"] = (df["Choice_Binary"] == df["ProbReward"]).astype(float)

    # Rolling-window features + Expected_PRight/Choice_Deviation, computed
    # AFTER the pseudo-session split, grouped by the corrected segment id
    # (see module docstring for why this differs from the source's own
    # Session_ID/Fit_Session_ID grouping at the equivalent step).
    df["PhysicalSwitch"] = (
        df.groupby("Session_ID")["Choice_Binary"].diff().fillna(0).abs().astype(int)
    )
    # CORRECTED (2026-09-13, see src/features/state_features.py's docstring
    # for the full account): `Rolling_Accuracy` must be reward-based
    # (Outcome_Binary), not choice-based (High_Prob), to match what
    # classify_state_v7's thresholds were actually tuned against and what
    # 09b_CrossProb_StickyModel_v1.ipynb's own STEP 7B computes. For this
    # 100-0 condition specifically this is a NO-OP numerically -- reward is
    # deterministic here (Expected_PRight is 1.0/0.0, not 0.8/0.2), so
    # Outcome_Binary == High_Prob on every valid trial -- but using the same
    # source column as the 80-20 side keeps the two conditions' features
    # built the same way rather than only "happening" to agree.
    df["Rolling_Accuracy"] = (
        df.groupby("Session_ID")["Outcome_Binary"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df["Rolling_Accuracy_High_Prob"] = (
        df.groupby("Session_ID")["High_Prob"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df["Rolling_PRight"] = (
        df.groupby("Session_ID")["Choice_Binary"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    df["Rolling_Switch_Rate"] = (
        df.groupby("Session_ID")["PhysicalSwitch"]
        .transform(lambda x: x.rolling(window=rolling_window, min_periods=3).mean())
    )
    # Expected_PRight for 100-0: 1.0/0.0 (not 0.8/0.2) -- correct choice is
    # reinforced 100% of the time in this condition (task brief point 3).
    df["Expected_PRight"] = df["High_Prob_Is_Right"].map({1.0: 1.0, 0.0: 0.0})
    df["Choice_Deviation"] = (df["Rolling_PRight"] - df["Expected_PRight"]).abs()

    return df


def fit_sticky_and_compare(df_80, df_100, x0, bounds, fit_sticky_qlearning):
    """Per-animal sticky Q-learning fit on both conditions (STEP 6, lines
    475-548), then a paired t-test on alpha/beta/kappa across the shared
    animals (STEP 7, lines 578-660) -- `fit_sticky_qlearning` is passed in
    (from `q_following_model`) rather than imported, per the "reuse those
    exact functions" instruction.
    """
    params_80 = fit_sticky_qlearning(df_80, x0, bounds).set_index("animal")
    params_100 = fit_sticky_qlearning(df_100, x0, bounds).set_index("animal")

    shared = sorted(set(params_80.index) & set(params_100.index))
    params_80 = params_80.loc[shared]
    params_100 = params_100.loc[shared]

    comparison = {}
    for param in ("alpha", "beta", "kappa"):
        v80 = params_80[param].values
        v100 = params_100[param].values
        t, p = stats.ttest_rel(v80, v100)
        comparison[param] = {
            "mean_80_20": float(v80.mean()), "sem_80_20": float(stats.sem(v80)),
            "mean_100_0": float(v100.mean()), "sem_100_0": float(stats.sem(v100)),
            "t": float(t), "p": float(p),
        }

    return {
        "animals": shared,
        "params_80_20": params_80.reset_index().to_dict(orient="records"),
        "params_100_0": params_100.reset_index().to_dict(orient="records"),
        "comparison": comparison,
    }


def compute_transition_probabilities(df_80, df_100, states, shared_animals, min_trials):
    """Per-animal, per-condition state-transition probability matrix (STATE
    TRANSITION PROBABILITIES: 80-20 vs 100-0, In[182], lines 3367-3413).
    """
    results = []
    for label, df in (("80-20", df_80), ("100-0", df_100)):
        d = df[df["Animal_Name"].isin(shared_animals)].copy()
        d = d.sort_values(["Animal_Name", "Session_ID", "Trial"])
        d["next_state"] = d.groupby(["Animal_Name", "Session_ID"])["Behavioral_State"].shift(-1)

        for animal in sorted(shared_animals):
            adf = d[d["Animal_Name"] == animal]
            for from_state in states:
                from_df = adf[adf["Behavioral_State"] == from_state].dropna(subset=["next_state"])
                if len(from_df) < min_trials:
                    continue
                for to_state in states:
                    p_trans = (from_df["next_state"] == to_state).mean()
                    results.append({
                        "condition": label, "animal": animal,
                        "from_state": from_state, "to_state": to_state,
                        "p_transition": p_trans, "n_trials": len(from_df),
                    })
    return pd.DataFrame(results)


def compute_diff_matrix(trans_df, states_key, shared_animals):
    """Per-cell (100-0 minus 80-20) mean transition-probability difference
    and paired t-test p-value across shared animals, for a given ordered
    subset of states (In[197] lines 3613-3637, and the earlier, non-figure
    print-only version at In[185] lines 3464-3496 which is the same
    computation over all 5 `states_full`).
    """
    n = len(states_key)
    diff_matrix = np.full((n, n), np.nan)
    pval_matrix = np.full((n, n), np.nan)
    mean_80_matrix = np.full((n, n), np.nan)
    mean_100_matrix = np.full((n, n), np.nan)

    for i, from_state in enumerate(states_key):
        for j, to_state in enumerate(states_key):
            v80 = (trans_df[(trans_df["condition"] == "80-20") &
                             (trans_df["from_state"] == from_state) &
                             (trans_df["to_state"] == to_state)]
                   .set_index("animal")["p_transition"].reindex(shared_animals))
            v100 = (trans_df[(trans_df["condition"] == "100-0") &
                              (trans_df["from_state"] == from_state) &
                              (trans_df["to_state"] == to_state)]
                    .set_index("animal")["p_transition"].reindex(shared_animals))
            mask = v80.notna() & v100.notna()
            v80, v100 = v80[mask].values, v100[mask].values
            if len(v80) < 3:
                continue
            t, p = stats.ttest_rel(v80, v100)
            diff_matrix[i, j] = v100.mean() - v80.mean()
            pval_matrix[i, j] = p
            mean_80_matrix[i, j] = v80.mean()
            mean_100_matrix[i, j] = v100.mean()

    return {
        "states": states_key,
        "diff_matrix": diff_matrix,
        "pval_matrix": pval_matrix,
        "mean_80_20_matrix": mean_80_matrix,
        "mean_100_0_matrix": mean_100_matrix,
    }


def paired_animal_comparison(values_80, values_100, animals):
    """Generic per-animal paired 80-20 vs 100-0 comparison: returns the
    per-animal value pairs plus a paired t-test across `animals` (this
    pipeline's standard pseudoreplication-correction pattern -- one value
    per animal, not per session). `values_80`/`values_100` are dicts or
    pd.Series keyed by animal name.
    """
    v80 = np.array([values_80[a] for a in animals], dtype=float)
    v100 = np.array([values_100[a] for a in animals], dtype=float)
    mask = ~(np.isnan(v80) | np.isnan(v100))
    v80m, v100m = v80[mask], v100[mask]
    if mask.sum() < 2:
        return {"animals": list(animals), "values_80_20": v80, "values_100_0": v100,
                "t": np.nan, "p": np.nan, "n": int(mask.sum())}
    t, p = stats.ttest_rel(v80m, v100m)
    return {
        "animals": list(animals), "values_80_20": v80, "values_100_0": v100,
        "t": float(t), "p": float(p), "n": int(mask.sum()),
    }


def compare_state_proportions(df_80, df_100, states, shared_animals):
    """Fig 8 panel c: per-animal proportion of trials in each of `states`
    (the 5 non-Adaptation states), 80-20 vs 100-0, paired t-test per state.
    Uses `state_taxonomy_v7.state_occupancy_animal_means` (denominator =
    trials in `states` only, i.e. Adapting excluded per the manuscript's
    own panel c legend) for each condition separately, then pairs by animal.
    """
    from . import state_taxonomy_v7 as sx7

    prop_80 = sx7.state_occupancy_animal_means(df_80[df_80["Animal_Name"].isin(shared_animals)], states)
    prop_100 = sx7.state_occupancy_animal_means(df_100[df_100["Animal_Name"].isin(shared_animals)], states)

    results = {}
    for state in states:
        v80 = prop_80[state].reindex(shared_animals) if state in prop_80.columns else pd.Series(np.nan, index=shared_animals)
        v100 = prop_100[state].reindex(shared_animals) if state in prop_100.columns else pd.Series(np.nan, index=shared_animals)
        results[state] = paired_animal_comparison(v80.to_dict(), v100.to_dict(), shared_animals)
    return results


def compare_win_stay_lose_switch(df_80, df_100, states, shared_animals):
    """Fig 8 panel d: win-stay / lose-switch probability for `states`
    (Exploitation, Directed Exploration), 80-20 vs 100-0, per-animal paired.
    Reuses `state_taxonomy_v7.win_stay_lose_switch_by_state` (per-session ->
    per-animal aggregation) independently on each condition, then pairs.
    """
    from . import state_taxonomy_v7 as sx7

    wsls_80 = sx7.win_stay_lose_switch_by_state(df_80[df_80["Animal_Name"].isin(shared_animals)], states)
    wsls_100 = sx7.win_stay_lose_switch_by_state(df_100[df_100["Animal_Name"].isin(shared_animals)], states)

    results = {"win_stay": {}, "lose_switch": {}}
    for state in states:
        win80 = wsls_80["win_animal"][wsls_80["win_animal"]["Behavioral_State"] == state].set_index("Animal_Name")["win_stay"]
        win100 = wsls_100["win_animal"][wsls_100["win_animal"]["Behavioral_State"] == state].set_index("Animal_Name")["win_stay"]
        lose80 = wsls_80["lose_animal"][wsls_80["lose_animal"]["Behavioral_State"] == state].set_index("Animal_Name")["lose_switch"]
        lose100 = wsls_100["lose_animal"][wsls_100["lose_animal"]["Behavioral_State"] == state].set_index("Animal_Name")["lose_switch"]

        results["win_stay"][state] = paired_animal_comparison(
            win80.reindex(shared_animals).to_dict(), win100.reindex(shared_animals).to_dict(), shared_animals
        )
        results["lose_switch"][state] = paired_animal_comparison(
            lose80.reindex(shared_animals).to_dict(), lose100.reindex(shared_animals).to_dict(), shared_animals
        )
    return results


def switch_vs_consecutive_losses(df, state, max_streak):
    """Fig 8 panel e ingredient (called once per condition): probability
    that a trial classified `state`, immediately preceded (within that
    same state-filtered subsequence) by N consecutive losses, is itself a
    switch. Ported from 09b_CrossProb_StickyModel_v1.ipynb cell 38
    (mislabeled `Figure7_LossStreak_*.svg` in the source despite comparing
    80-20 vs 100-0 -- functionally this IS Fig 8 panel e's source cell).
    Returns (per_animal_df, summary_df) with columns Loss_Streak / mean / sem / n.

    CORRECTED (2026-09-17) to match that source cell exactly, after a
    direct comparison against Fig 8's own real reconstructed data showed
    the previous version of this function silently washed out a large,
    real effect. The previous version filtered to `state` FIRST but then
    walked every trial in the session and reset the streak whenever the
    state didn't match, and checked the NEXT chronological trial's switch
    (`PhysicalSwitch.shift(-1)`) rather than the state-filtered row's own
    switch. The source instead filters to `state` FIRST and counts the
    loss streak WITHIN that filtered subsequence only (gaps from
    other-state trials are closed, not streak-resetting), then checks the
    filtered row's own `PhysicalSwitch`. Verified: Exploit/100-0 goes from
    p(switch)=0.060 at streak 0 to 0.387 at streak 1 with this corrected
    method (matching the published figure), vs. 0.081 -> 0.128 (ns) with
    the old method.
    """
    d = df[df["Behavioral_State"] == state].copy()
    d = d.sort_values(["Animal_Name", "Session_ID", "Trial"])

    loss_streak_col = []
    for _, grp in d.groupby(["Animal_Name", "Session_ID"], sort=False):
        streak = 0
        for outcome in grp["Outcome_Binary"].values:
            loss_streak_col.append(streak)
            streak = streak + 1 if outcome == 0 else 0
    d["Loss_Streak"] = loss_streak_col
    d = d[d["Loss_Streak"] <= max_streak].dropna(subset=["PhysicalSwitch"])

    per_animal = (
        d.groupby(["Animal_Name", "Loss_Streak"])["PhysicalSwitch"]
        .agg(mean="mean", n="count").reset_index()
    )
    summary = per_animal.groupby("Loss_Streak")["mean"].agg(
        mean="mean",
        sem=lambda x: x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan,
        n="count",
    ).reset_index()
    return per_animal, summary


def compute_cross_bias_rates(df_80, df_100, shared_animals):
    """Per-animal direct Left<->Right Bias cross-transition rate (pooled
    L->R + R->L, normalized by total Bias-state trials), for both conditions,
    plus a paired t-test -- Figure 8g (In[197], lines 3677-3707).
    """
    out = {}
    for label, df in (("80-20", df_80), ("100-0", df_100)):
        d = df.copy().sort_values(["Animal_Name", "Session_ID", "Trial"])
        d["next_state"] = d.groupby(["Animal_Name", "Session_ID"])["Behavioral_State"].shift(-1)

        rates = []
        for animal in shared_animals:
            adf = d[d["Animal_Name"] == animal].dropna(subset=["next_state"])
            lb_total = len(adf[adf["Behavioral_State"] == "Left Bias"])
            rb_total = len(adf[adf["Behavioral_State"] == "Right Bias"])
            total_bias = lb_total + rb_total
            lr = len(adf[(adf["Behavioral_State"] == "Left Bias") & (adf["next_state"] == "Right Bias")])
            rl = len(adf[(adf["Behavioral_State"] == "Right Bias") & (adf["next_state"] == "Left Bias")])
            rate = (lr + rl) / total_bias if total_bias > 0 else 0.0
            rates.append(rate)
        out[label] = np.array(rates)

    t, p = stats.ttest_rel(out["80-20"], out["100-0"])
    return {
        "animals": list(shared_animals),
        "cross_80_20": out["80-20"],
        "cross_100_0": out["100-0"],
        "t": float(t), "p": float(p),
    }
