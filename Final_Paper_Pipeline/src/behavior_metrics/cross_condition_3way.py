"""New, standalone companion to Fig 8 -- 80-20 vs. 90-10 vs. 100-0 (three
conditions instead of Fig 8's two). Built for the 2026 reviewer response now
that 90-10 sessions exist (`preprocessing/`'s new bhv2-conversion pipeline).

Does NOT modify `fig8.py`, `Fig8.ipynb`, or config.yaml's existing `fig8:`
section -- Fig 8's own panel layout/scope is unchanged. This module adds
N-way (here, 3-way) generalizations of `cross_condition.py`'s pairwise
comparison helpers, using the identical statistics pattern (per-animal
pseudoreplication correction, then a repeated-measures test across the
shared animals) with a Friedman test in place of a paired t-test, since a
paired t-test only compares two groups.

NOTE (2026-09-17): while building this companion's panel d/e, a real bug
was found in `cross_condition.switch_vs_consecutive_losses` and
`state_taxonomy_v7.win_stay_lose_switch_by_state` -- both washed out a
large, real, already-published effect (Exploit-state lose-switch/loss-streak
jump at 100-0) by filtering to state AFTER computing the switch/prev-outcome
relationship on the full chronological sequence, instead of filtering to
state FIRST as the real source notebook does. Per the user's explicit
direction, BOTH functions were fixed in place (see their own docstrings for
the exact correction and verification against Fig 8's real data) -- this
DOES change Fig 8's and Fig 7's own output, deliberately. This module now
just calls those corrected shared functions directly rather than
maintaining its own duplicate copies.

DATA SOURCE: unlike Fig 8's two dedicated single-condition snapshot CSVs,
all three conditions here come from ONE source -- the new merged master file
built by `preprocessing/` (`Merged_Data/preprocessed_master_merged_data_with_metadata_20260917.csv`),
already validated for exactly this condition-generic feature construction by
`cross_species_comparison/run_pipeline_marmoset.py` (that script's
`load_and_prepare`/`add_rolling_features` are imported and reused here
unchanged, not reimplemented). Using one source for all three conditions
(rather than mixing Fig 8's own 80-20/100-0 snapshots with a new 90-10-only
file) keeps the three conditions built the same way.

DISCLOSED DESIGN CHOICE: Fig 8's 100-0 side uses a bespoke pseudo-session
split (`cross_condition.build_100_0_features_df` /
`create_correct_pseudo_sessions`) to handle that snapshot's own
multi-block-per-raw-session quirk. The new merged master file's 100-0 rows
are NOT run through that splitter here -- `Block_Transition`/
`Trials_Since_Transition` are instead computed directly from `BlockCount`
resets within the raw `Session_ID` (the same construction
`run_pipeline_marmoset.py` already uses for all three conditions, verified
there to produce sensible state proportions/model fits). This is a
simplification relative to Fig 8's own 100-0 handling, not a claim that it
is equivalent -- flagged here for the user to weigh in on, per this
project's standing convention of disclosing rather than silently matching.
"""

import os
import sys
import functools

import numpy as np
import pandas as pd
from scipy import stats

from . import state_taxonomy_v7 as sx7
from . import state_classifier
from . import cross_condition


def load_and_classify_3way(master_csv_path, cross_species_dir, conditions, animals, st_cfg):
    """Build the classify_state_v7-ready feature dataframe for each of
    `conditions` from the new merged master CSV, reusing
    `cross_species_comparison/run_pipeline_marmoset.py`'s already-validated
    `load_and_prepare`/`add_rolling_features` (condition-generic
    High_Prob/Expected_PRight formulas -- see that module's docstring for
    why those, not `state_features.py`'s hardcoded-0.8/0.2 ones, are needed
    here) for the feature-engineering step, then applies Fig 8's OWN
    `state_classifier.classify_state_v7` (not the cross-species
    `classify_state_updated_noAdapting` classifier) so this stays a genuine
    subset/companion of Fig 8, not a repeat of the Fig 9 cross-species
    build. Returns (dfs_by_condition, block_labels_by_condition) dicts keyed
    by condition label.
    """
    sys.path.insert(0, os.path.abspath(cross_species_dir))
    from run_pipeline_marmoset import load_and_prepare, add_rolling_features  # noqa: E402

    df_all = load_and_prepare()
    df_all = add_rolling_features(df_all)
    df_all = df_all[df_all["Animal_Name"].isin(animals)].copy()

    dfs_by_condition, block_labels_by_condition = {}, {}
    for c in conditions:
        df = df_all[df_all["prob_Condition"] == c].copy().reset_index(drop=True)

        block_labels = state_classifier.label_block_adaptation_speed(
            df, st_cfg["adaptation_window"], st_cfg["adaptation_fast_accuracy_threshold"]
        )
        adaptation_labels = state_classifier.compute_adaptation_labels(
            df, st_cfg["adaptation_window"], st_cfg["adaptation_fast_accuracy_threshold"],
            block_labels=block_labels,
        )
        classify_v7 = functools.partial(
            state_classifier.classify_state_v7, adaptation_labels=adaptation_labels,
            config=st_cfg["classify_updated"],
        )
        df["Behavioral_State"] = df.apply(classify_v7, axis=1)

        dfs_by_condition[c] = df
        block_labels_by_condition[c] = block_labels

    return dfs_by_condition, block_labels_by_condition


def tag_cohort_and_session_order(df, animal_col="Animal_Name", session_col="Session_ID",
                                   date_col="Date", filename_col="Session_Filename"):
    """Reviewer-response addition (RA4_condition_wise_RL.ipynb, R2-5/R1-D):
    tags each trial with `Cohort` ("2026" if its Session_Filename contains
    '2abt', else "original") and `Session_Order` (that session's ordinal
    position, 0-indexed, within its own animal's full chronological
    session history) -- the cumulative-experience covariate. Applied to
    the output of `load_and_classify_3way` (one call per condition's
    dataframe), NOT inside that function itself, so Fig8_3cond.ipynb/
    Fig9.ipynb's own existing calls are completely unaffected.

    The '2abt' filename rule and the choice not to use a date cutoff or
    Has_Cued_Transitions instead were verified empirically earlier this
    session (see run_pipeline_marmoset.py's own docstring/commit history):
    0 pre-2024 sessions contain '2abt', all 167 later 2026 sessions do,
    and Has_Cued_Transitions is False for both cohorts so cannot
    discriminate between them.
    """
    d = df.copy()
    d["Cohort"] = np.where(
        d[filename_col].str.lower().str.contains("2abt", na=False), "2026", "original"
    )
    session_dates = (
        d[[animal_col, session_col, date_col]].drop_duplicates(subset=session_col)
        .sort_values([animal_col, date_col])
    )
    session_dates["Session_Order"] = session_dates.groupby(animal_col).cumcount()
    d = d.merge(session_dates[[session_col, "Session_Order"]], on=session_col, how="left")
    return d


def friedman_paired_comparison(values_by_condition, animals, conditions):
    """N-way generalization of `cross_condition.paired_animal_comparison`:
    per-animal values across `conditions` (>= 2), repeated-measures
    significance via Friedman's test (non-parametric, matches the paired/
    n=small-sample design without assuming normality) in place of a paired
    t-test, which only handles exactly two groups. `values_by_condition` is
    {condition: {animal: value}}.
    """
    values = {c: np.array([values_by_condition[c].get(a, np.nan) for a in animals], dtype=float)
              for c in conditions}
    stacked = np.vstack([values[c] for c in conditions])
    complete = ~np.isnan(stacked).any(axis=0)
    used_animals = [a for a, keep in zip(animals, complete) if keep]

    result = {"animals": list(animals), "used_animals": used_animals,
              "n": int(complete.sum())}
    for c in conditions:
        result[f"values_{c.replace('-', '_')}"] = values[c]

    if complete.sum() < 3:
        result["chi2"], result["p"] = np.nan, np.nan
        return result

    complete_vals = [values[c][complete] for c in conditions]
    chi2, p = stats.friedmanchisquare(*complete_vals)
    result["chi2"], result["p"] = float(chi2), float(p)
    return result


def compare_state_proportions_nway(dfs_by_condition, states, animals, conditions):
    """3-way generalization of `cross_condition.compare_state_proportions`
    (Fig 8 panel c)."""
    props = {
        c: sx7.state_occupancy_animal_means(dfs_by_condition[c][dfs_by_condition[c]["Animal_Name"].isin(animals)], states)
        for c in conditions
    }
    results = {}
    for state in states:
        values_by_condition = {}
        for c in conditions:
            prop = props[c]
            v = prop[state].reindex(animals) if state in prop.columns else pd.Series(np.nan, index=animals)
            values_by_condition[c] = v.to_dict()
        results[state] = friedman_paired_comparison(values_by_condition, animals, conditions)
    return results


def compare_win_stay_lose_switch_nway(dfs_by_condition, states, animals, conditions):
    """3-way generalization of `cross_condition.compare_win_stay_lose_switch`
    (Fig 8 panel d) -- calls the now-corrected
    `state_taxonomy_v7.win_stay_lose_switch_by_state` directly (see that
    function's docstring for the 2026-09-17 fix and its verification
    against Fig 8's real data)."""
    wsls = {c: sx7.win_stay_lose_switch_by_state(dfs_by_condition[c], states) for c in conditions}
    results = {"win_stay": {}, "lose_switch": {}}
    for metric, key in (("win_stay", "win_animal"), ("lose_switch", "lose_animal")):
        for state in states:
            values_by_condition = {}
            for c in conditions:
                tbl = wsls[c][key]
                sub = tbl[tbl["Behavioral_State"] == state].set_index("Animal_Name")[metric]
                values_by_condition[c] = sub.reindex(animals).to_dict()
            results[metric][state] = friedman_paired_comparison(values_by_condition, animals, conditions)
    return results


def compare_loss_streak_nway(dfs_by_condition, state, animals, conditions, max_streak):
    """3-way generalization of the now-corrected
    `cross_condition.switch_vs_consecutive_losses` (Fig 8 panel e):
    per-condition per-animal p(switch) by loss-streak length, plus a
    Friedman test per streak length across conditions."""
    per_animal_by_condition, summary_by_condition = {}, {}
    for c in conditions:
        per_animal, summary = cross_condition.switch_vs_consecutive_losses(dfs_by_condition[c], state, max_streak)
        per_animal_by_condition[c] = per_animal
        summary_by_condition[c] = summary

    comparison = {}
    for streak in range(0, max_streak + 1):
        values_by_condition = {}
        for c in conditions:
            pa = per_animal_by_condition[c]
            sub = pa[pa["Loss_Streak"] == streak].set_index("Animal_Name")["mean"]
            values_by_condition[c] = sub.reindex(animals).to_dict()
        r = friedman_paired_comparison(values_by_condition, animals, conditions)
        if r["n"] >= 3:
            comparison[streak] = r
    return summary_by_condition, comparison


def compute_transition_probabilities_nway(dfs_by_condition, states, animals, min_trials, conditions):
    """3-way generalization of `cross_condition.compute_transition_probabilities`:
    `dfs_by_condition` is {condition_label: df}, not fixed to '80-20'/'100-0'."""
    rows = []
    for label in conditions:
        df = dfs_by_condition[label]
        d = df[df["Animal_Name"].isin(animals)].copy()
        d = d.sort_values(["Animal_Name", "Session_ID", "Trial"])
        d["next_state"] = d.groupby(["Animal_Name", "Session_ID"])["Behavioral_State"].shift(-1)

        for animal in sorted(animals):
            adf = d[d["Animal_Name"] == animal]
            for from_state in states:
                from_df = adf[adf["Behavioral_State"] == from_state].dropna(subset=["next_state"])
                if len(from_df) < min_trials:
                    continue
                for to_state in states:
                    p_trans = (from_df["next_state"] == to_state).mean()
                    rows.append({
                        "condition": label, "animal": animal,
                        "from_state": from_state, "to_state": to_state,
                        "p_transition": p_trans, "n_trials": len(from_df),
                    })
    return pd.DataFrame(rows)


def compute_diff_matrix_pair(trans_df, states_key, animals, cond_a, cond_b):
    """Generalization of `cross_condition.compute_diff_matrix` to an
    arbitrary ordered pair of condition labels (cond_b minus cond_a),
    rather than hardcoding '80-20'/'100-0'. Used twice here to walk the
    80-20 -> 90-10 -> 100-0 gradient in two steps, mirroring Fig 8 panel f's
    own single-step diff design."""
    n = len(states_key)
    diff_matrix = np.full((n, n), np.nan)
    pval_matrix = np.full((n, n), np.nan)

    for i, from_state in enumerate(states_key):
        for j, to_state in enumerate(states_key):
            va = (trans_df[(trans_df["condition"] == cond_a) &
                            (trans_df["from_state"] == from_state) &
                            (trans_df["to_state"] == to_state)]
                  .set_index("animal")["p_transition"].reindex(animals))
            vb = (trans_df[(trans_df["condition"] == cond_b) &
                            (trans_df["from_state"] == from_state) &
                            (trans_df["to_state"] == to_state)]
                  .set_index("animal")["p_transition"].reindex(animals))
            mask = va.notna() & vb.notna()
            va, vb = va[mask].values, vb[mask].values
            if len(va) < 3:
                continue
            t, p = stats.ttest_rel(va, vb)
            diff_matrix[i, j] = vb.mean() - va.mean()
            pval_matrix[i, j] = p

    return {"states": states_key, "diff_matrix": diff_matrix, "pval_matrix": pval_matrix,
            "cond_a": cond_a, "cond_b": cond_b}


def fit_sticky_nway(dfs_by_condition, x0, bounds, fit_sticky_qlearning, animals, conditions):
    """3-way generalization of `cross_condition.fit_sticky_and_compare`:
    per-animal sticky Q-learning fit on each condition (same
    `q_following_model.fit_sticky_qlearning` Fig 8/Supp Fig 3 already use,
    single-init, not the cross-species script's 10-restart version), then a
    Friedman test on alpha/beta/kappa across the animals with a fit in all
    `conditions`."""
    params = {}
    for c in conditions:
        p = fit_sticky_qlearning(dfs_by_condition[c][dfs_by_condition[c]["Animal_Name"].isin(animals)], x0, bounds)
        params[c] = p.set_index("animal")

    comparison = {}
    for param in ("alpha", "beta", "kappa"):
        values_by_condition = {c: params[c][param].to_dict() if param in params[c].columns else {} for c in conditions}
        comparison[param] = friedman_paired_comparison(values_by_condition, animals, conditions)

    return {
        "animals": list(animals),
        "params_by_condition": {c: params[c].reset_index().to_dict(orient="records") for c in conditions},
        "comparison": comparison,
    }


def paired_two_condition_comparison(values_a, values_b, animals, label_a, label_b):
    """Reviewer-response addition (RA4_condition_wise_RL.ipynb, R2-5):
    generic paired per-animal comparison between exactly two conditions,
    correctly labeled by whichever `label_a`/`label_b` are passed in --
    unlike `cross_condition.paired_animal_comparison` (hardcoded to
    "80-20"/"100-0" key names) or `friedman_paired_comparison` (Friedman's
    test itself requires >= 3 related samples, so it cannot be called with
    only two conditions). Same paired t-test as
    `cross_condition.paired_animal_comparison` -- this project's standard
    pseudoreplication-correction pattern, one value per animal.
    `values_a`/`values_b` are dicts or pd.Series keyed by animal name.
    """
    va = np.array([values_a.get(a, np.nan) if hasattr(values_a, "get") else values_a[a] for a in animals], dtype=float)
    vb = np.array([values_b.get(a, np.nan) if hasattr(values_b, "get") else values_b[a] for a in animals], dtype=float)
    mask = ~(np.isnan(va) | np.isnan(vb))
    used_animals = [a for a, keep in zip(animals, mask) if keep]
    result = {
        "animals": list(animals), "used_animals": used_animals, "n": int(mask.sum()),
        f"values_{label_a.replace('-', '_')}": va.tolist(), f"values_{label_b.replace('-', '_')}": vb.tolist(),
    }
    if mask.sum() < 2:
        result["t"], result["p"] = None, None
        return result
    t, p = stats.ttest_rel(va[mask], vb[mask])
    result["t"], result["p"] = float(t), float(p)
    return result
