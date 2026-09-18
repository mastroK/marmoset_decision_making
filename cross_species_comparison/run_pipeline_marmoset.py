"""
Cross-species replication: apply the SAME classifier/model pipeline
`run_pipeline.py` uses for the mouse (Beron) data to marmoset data, across
its own 100-0 / 90-10 / 80-20 Reversal-task conditions.

WHY A SEPARATE SCRIPT, NOT `state_features.py`
------------------------------------------------
`Final_Paper_Pipeline/src/features/state_features.py::build_state_features_df`
already does something similar, but its `High_Prob`/`Expected_PRight`
formulas are HARDCODED to the 80-20 condition (`ProbReward.map({0.8: 1,
0.2: 0})`) -- correct for Fig 6/7/8/Supp Fig 4's own single-condition
80-20 data, but silently wrong (all-NaN) for 90-10 or 100-0 data. Rather
than generalize a Final_Paper_Pipeline module for a use case outside that
pipeline's own scope, this script -- like `run_pipeline.py` before it --
computes its own condition-generic features directly, importing the
classifier/model/win-stay-lose-switch logic from `run_pipeline.py` so
BOTH species are run through the literal same functions, not
independently-reimplemented equivalents.

FEATURE ENGINEERING (condition-generic; see `run_pipeline.py` for the
mouse-side version this mirrors)
------------------------------------------------------------------------
`ProbReward` is the CHOSEN option's own reward probability this trial
(e.g. 0.8 if the animal chose the high-prob side of an 80-20 block, 0.2 if
it chose the low-prob side) -- this project's established convention
throughout (see `state_features.py`'s own docstring). Rather than assume
a specific pair of values:
  - High_Prob            = ProbReward > 0.5           (chose the better option)
  - high_prob_value       = max(ProbReward, 1 - ProbReward)   (this block's
                             actual high-probability value: 0.8/0.9/1.0)
  - High_Prob_Is_Right    = (ProbReward > 0.5 and chose Right) or
                             (ProbReward < 0.5 and chose Left)
  - Expected_PRight       = high_prob_value if High_Prob_Is_Right else
                             (1 - high_prob_value)
This reduces to `state_features.py`'s own formulas exactly when
high_prob_value happens to be 0.8, and generalizes correctly to 0.9/1.0
for 90-10/100-0 -- verified against `state_features.py`'s 80-20 output
before this script was used for anything (see README.md).

Everything else (rolling window/min_periods, the classifier, the sticky
model, win-stay/lose-switch, output shape) is imported unchanged from
`run_pipeline.py` -- see that file's own docstring for what "verbatim
port" means for the classifier/model themselves.
"""

import os
import sys
import json
import argparse
import datetime

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_pipeline import (  # noqa: E402
    THRESHOLD_SETS,
    ROLLING_WINDOW,
    MIN_PERIODS,
    N_RESTARTS,
    make_classify_state_updated_noAdapting,
    fit_sticky_model,
    compute_win_stay_lose_switch,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# No implicit "most recent file" guessing -- same convention as
# preprocessing/python/01_session_transition_classification.ipynb's own
# INPUT_MASTER_FILE: pick the Step 2/3 output explicitly.
DATA_PATH = (
    "/Users/kevinmastro/Library/CloudStorage/GoogleDrive-kmastro@broadinstitute.org/"
    "Shared drives/Broad Marmoset/2-Experiments/5 - Monkey Logic Task/2 - Raw Data/"
    "Merged_Data/preprocessed_master_merged_data_with_metadata_20260918.csv"
)

CONDITIONS = ["100-0", "90-10", "80-20"]


def _code_choice(physical_choice):
    """Same mapping as state_features.py's own `_code_choice`."""
    if physical_choice in ("BottomLeft", "LeftCenter", "TopLeft", "Left"):
        return 0
    elif physical_choice in ("BottomRight", "RightCenter", "TopRight", "Right"):
        return 1
    return np.nan


def load_and_prepare():
    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df[(df["Task_Type"] == "Reversal") & (df["prob_Condition"].isin(CONDITIONS))].copy()

    # CRITICAL FIX (2026-09-18): exclude cued-transition sessions entirely.
    # Every other snapshot this whole pipeline uses (Fig3 onward, including
    # Fig 8's own filtered_df_Reversal_80-20_2-ABT_100.csv -- verified
    # directly: Has_Cued_Transitions is False for all 36,673 of its rows)
    # was ALREADY pre-filtered to uncued-only sessions upstream, before
    # ever reaching this pipeline, representing "pure" 2-armed-bandit
    # behavior (a cued transition tells the animal directly that the
    # contingency changed via a visual/location cue, which is a
    # fundamentally different task than inferring a reversal from
    # experienced outcomes alone). This new preprocessing pipeline's own
    # merged master file was never filtered this way -- confirmed
    # directly: 192 of 437 Reversal sessions (44%, 51,198 of 125,745
    # trials, 41%) have at least one cued transition. Has_Cued_Transitions
    # is a session-level flag (constant within a Session_ID, verified),
    # so the correct fix -- matching the existing snapshots' own
    # convention exactly -- is to drop the ENTIRE session, not just the
    # individual cued-transition trials within it.
    cued_session = df.groupby("Session_ID")["Has_Cued_Transitions"].transform("any")
    df = df[~cued_session].copy()

    df = df.dropna(subset=["ProbReward", "PhysicalChoice"])
    df = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).reset_index(drop=True)

    df["Choice_Binary"] = df["PhysicalChoice"].map(_code_choice)
    df = df.dropna(subset=["Choice_Binary"]).copy()
    df["Choice_Binary"] = df["Choice_Binary"].astype(int)
    df["Outcome_Binary"] = df["Outcome"].astype(int)

    df["High_Prob"] = (df["ProbReward"] > 0.5).astype(int)
    high_prob_value = np.maximum(df["ProbReward"], 1 - df["ProbReward"])
    df["High_Prob_Is_Right"] = (
        ((df["ProbReward"] > 0.5) & (df["Choice_Binary"] == 1))
        | ((df["ProbReward"] < 0.5) & (df["Choice_Binary"] == 0))
    )
    df["Expected_PRight"] = np.where(df["High_Prob_Is_Right"], high_prob_value, 1 - high_prob_value)

    df["PhysicalSwitch"] = df["PhysicalSwitch"].astype(float)

    # Block transitions / within-block trial count, from BlockCount resets --
    # same construction as state_features.py's Block_Transition/Trials_Since_Transition.
    df["Block_Transition"] = (
        df.groupby("Session_ID")["BlockCount"].diff().fillna(0).abs().astype(bool).astype(int)
    )
    trials_since = []
    for _, session_df in df.groupby("Session_ID", sort=False):
        counter = 0
        for is_trans in session_df["Block_Transition"]:
            if is_trans:
                counter = 0
            trials_since.append(counter)
            counter += 1
    df["Trials_Since_Transition"] = trials_since

    return df


def add_rolling_features(df):
    g = df.groupby(["Animal_Name", "Session_ID"])

    df["Rolling_Accuracy"] = g["Outcome_Binary"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
    )
    df["Rolling_PRight"] = g["Choice_Binary"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
    )
    df["Rolling_Switch_Rate"] = g["PhysicalSwitch"].transform(
        lambda x: x.rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS).mean()
    )
    df["Choice_Deviation"] = (df["Rolling_PRight"] - df["Expected_PRight"]).abs()

    return df


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--thresholds", required=True, choices=sorted(THRESHOLD_SETS),
        help="Same threshold-set registry as run_pipeline.py -- required, "
             "no default, so marmoset results always land in the same "
             "outputs/<name>_thresholds/ directory as the matching mouse run.",
    )
    p.add_argument(
        "--skip-fit", action="store_true",
        help="Skip the sticky-model fit stage (reuse the existing "
             "marmoset_sticky_model_params_*.csv in this threshold set's output dir).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    thresholds = THRESHOLD_SETS[args.thresholds]
    classify_fn = make_classify_state_updated_noAdapting(thresholds)

    out_dir = os.path.join(HERE, "outputs", f"{args.thresholds}_thresholds")
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)

    print(f"Threshold set : {args.thresholds}")
    print(f"Output dir    : {out_dir}  (marmoset_* files)")
    print("Loading data...")
    df = load_and_prepare()
    print(f"  {len(df):,} trials, {df['Animal_Name'].nunique()} animals, "
          f"{df['Session_ID'].nunique()} sessions, "
          f"conditions: {sorted(df['prob_Condition'].unique())}")

    print("Computing rolling features...")
    df = add_rolling_features(df)

    print(f"Applying classify_state_updated_noAdapting with '{args.thresholds}' thresholds...")
    df["Behavioral_State"] = df.apply(classify_fn, axis=1)

    provenance = {
        "species": "marmoset",
        "threshold_set": args.thresholds,
        "thresholds": thresholds,
        "classifier": "classify_state_updated_noAdapting (imported from run_pipeline.py)",
        "input_file": DATA_PATH,
        "conditions": CONDITIONS,
        "script": os.path.basename(__file__),
        "invocation": "python3 " + " ".join(sys.argv),
        "generated_at": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(out_dir, "marmoset_provenance.json"), "w") as f:
        json.dump(provenance, f, indent=2, default=str)
    print(f"  Saved: {os.path.join(out_dir, 'marmoset_provenance.json')}")

    classified_path = os.path.join(out_dir, "marmoset_df_classified.csv")
    df.to_csv(classified_path, index=False)
    print(f"  Saved: {classified_path}")

    # ---- (1) state proportions by condition -------------------------------
    print("\nComputing state proportions by condition...")
    prop = (
        df.groupby("prob_Condition")["Behavioral_State"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
        .rename(columns={"prob_Condition": "Condition"})
    )
    prop_path = os.path.join(out_dir, "marmoset_state_proportions_by_condition.csv")
    prop.to_csv(prop_path, index=False)
    print(prop.pivot(index="Behavioral_State", columns="Condition", values="proportion").round(3))
    print(f"  Saved: {prop_path}")

    prop_by_animal = (
        df.groupby(["prob_Condition", "Animal_Name"])["Behavioral_State"]
        .value_counts(normalize=True)
        .rename("proportion")
        .reset_index()
        .rename(columns={"prob_Condition": "Condition"})
    )
    prop_by_animal.to_csv(
        os.path.join(out_dir, "marmoset_state_proportions_by_condition_and_animal.csv"),
        index=False,
    )

    # ---- (2) switch behavior within each state, by condition --------------
    print("\nComputing switch-rate by state x condition...")
    switch_rate = (
        df.groupby(["prob_Condition", "Behavioral_State"])["PhysicalSwitch"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "switch_rate", "count": "n_trials"})
        .reset_index()
        .rename(columns={"prob_Condition": "Condition"})
    )
    switch_path = os.path.join(out_dir, "marmoset_switch_rate_by_state_by_condition.csv")
    switch_rate.to_csv(switch_path, index=False)
    print(switch_rate.pivot(index="Behavioral_State", columns="Condition", values="switch_rate").round(3))
    print(f"  Saved: {switch_path}")

    print("\nComputing win-stay / lose-switch by state x condition...")
    # NOTE: don't rename prob_Condition -> "Condition" here -- the raw
    # MonkeyLogic data already has its own unrelated "Condition" column (an
    # internal condition-file index), so that rename creates a duplicate
    # column name and groupby("Condition") silently returns a DataFrame
    # instead of a Series ("Grouper for 'Condition' not 1-dimensional").
    # compute_win_stay_lose_switch only needs Mouse/Session renamed.
    df_for_wsls = df.rename(columns={"Animal_Name": "Mouse", "Session_ID": "Session"})
    wsls_rows = []
    for cond, cdf in df_for_wsls.groupby("prob_Condition"):
        for state, sdf in cdf.groupby("Behavioral_State"):
            if len(sdf) < 30:
                continue
            ws, ls, n_win, n_lose = compute_win_stay_lose_switch(sdf, cdf)
            wsls_rows.append({
                "Condition": cond, "Behavioral_State": state,
                "win_stay": ws, "lose_switch": ls,
                "n_win_trials": n_win, "n_lose_trials": n_lose,
                "n_trials": len(sdf),
            })
    wsls = pd.DataFrame(wsls_rows)
    wsls_path = os.path.join(out_dir, "marmoset_winstay_loseswitch_by_state_by_condition.csv")
    wsls.to_csv(wsls_path, index=False)
    print(wsls.to_string(index=False))
    print(f"  Saved: {wsls_path}")

    # ---- (3) fit sticky Q-learning model per animal x condition ------------
    if args.skip_fit:
        print("\n--skip-fit passed: leaving existing marmoset_sticky_model_params_*.csv untouched.")
        print("\nDone.")
        return

    print("\nFitting sticky Q-learning + choice-kernel model per animal x condition...")
    n_animals = df["Animal_Name"].nunique()
    n_conditions = df["prob_Condition"].nunique()
    print(f"  ({n_animals} animals x {n_conditions} conditions, {N_RESTARTS} restarts each)")
    fit_rows = []
    seed_counter = 0
    for (animal, cond), adf in df.groupby(["Animal_Name", "prob_Condition"]):
        adf = adf.sort_values(["Session_ID", "Trial"])
        choices = adf["Choice_Binary"].values.astype(int)
        rewards = adf["Outcome_Binary"].values.astype(int)
        if len(choices) < 30:
            print(f"  {animal:>10s} | {cond:>5s} | n={len(adf):6d} | skipped (too few trials)")
            continue
        res = fit_sticky_model(choices, rewards, seed_base=seed_counter)
        seed_counter += 1
        fit_rows.append({
            "Animal_Name": animal, "Condition": cond,
            "alpha": res.x[0], "beta": res.x[1], "kappa": res.x[2],
            "nll": res.fun, "n_trials": len(adf), "converged": res.success,
        })
        print(f"  {animal:>10s} | {cond:>5s} | n={len(adf):6d} | "
              f"alpha={res.x[0]:.3f} beta={res.x[1]:.3f} kappa={res.x[2]:+.3f}")

    fits = pd.DataFrame(fit_rows)
    fits_path = os.path.join(out_dir, "marmoset_sticky_model_params_by_animal_by_condition.csv")
    fits.to_csv(fits_path, index=False)
    print(f"  Saved: {fits_path}")

    print("\nParameter summary across animals, by condition (mean +/- SEM):")
    summary = fits.groupby("Condition")[["alpha", "beta", "kappa"]].agg(["mean", "sem"])
    print(summary.round(3))
    summary.to_csv(os.path.join(out_dir, "marmoset_sticky_model_params_summary_by_condition.csv"))

    print("\nDone.")


if __name__ == "__main__":
    main()
