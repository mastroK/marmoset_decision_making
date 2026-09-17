"""Four-model Q-learning comparison for Supplementary Figure 2: pure
Q-learning, +sticky choice kernel, +forgetting, +adaptive forgetting --
each fit per-animal on a held-out train/test session split, on the same
80-20 2-ABT reversal data used by Fig 3 / Supp Fig 3.

Source: _source_archive/by_figure/SuppFig2/01_Reversal_80_20_model.ipynb
(see config.yaml's `suppfig2` section for full provenance/line-number
notes and the discrepancies already flagged there -- Q-value
initialization, single-x0-vs-50-restart, 70/30-vs-80/20 split). The
"+sticky" model (panel b) is NOT reimplemented here -- it reuses
q_following_model.fit_sticky_qlearning / compute_qvalues_sticky directly,
per the same model already fit and verified for Fig 4 / Supp Fig 3.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import q_following_model as qmodel

MODEL_NAMES = ["pure", "sticky", "forgetting", "adaptive_forgetting"]
MODEL_LABELS = {
    "pure": "Pure QL",
    "sticky": "w/Sticky",
    "forgetting": "w/Forgetting",
    "adaptive_forgetting": "w/Adaptive",
}
# Matches the source's own model_colors dict (SYSTEMATIC MODEL COMPARISON
# cell) and the manuscript's own panel a-e coloring (blue/red/green/orange).
MODEL_COLORS = {
    "pure": "#3498db",
    "sticky": "#e74c3c",
    "forgetting": "#2ecc71",
    "adaptive_forgetting": "#f39c12",
}


# ---------------------------------------------------------------------------
# Data prep
# ---------------------------------------------------------------------------

def prepare_dataframe(df_raw, task_type, prob_condition):
    """Reuse q_following_model's prep (Choice_Binary/Outcome_Binary/
    Trial_in_Block), then add what this figure additionally needs:
    High_Prob (whether the trial's chosen option was the 80% side --
    source's STEP 7, `df['ProbReward'].map({0.8: 1, 0.2: 0})`) and Switch
    (trial-level choice-repeat indicator, NaN at each session's first trial).
    """
    df = qmodel.prepare_2abt_dataframe(df_raw, task_type, prob_condition)
    df["High_Prob"] = (df["ProbReward"] == 0.8).astype(float)
    df["Switch"] = (
        df.groupby(["Animal_Name", "Session_ID"])["Choice_Binary"].diff().abs()
    )
    return _add_high_side(df)


def _add_high_side(df):
    """Per (Session_ID, BlockCount) block, which Choice_Binary value (0/1)
    was the 80%-reward side -- the mode of Choice_Binary among that
    block's High_Prob==1 trials. NaN if a block has no such trial.
    """
    def _mode_high_side(g):
        chosen_high = g.loc[g["High_Prob"] == 1, "Choice_Binary"]
        if len(chosen_high) == 0:
            return np.nan
        return float(chosen_high.mode().iloc[0])

    high_side = (
        df.groupby(["Session_ID", "BlockCount"])
        .apply(_mode_high_side, include_groups=False)
        .rename("High_Side_Binary")
        .reset_index()
    )
    return df.merge(high_side, on=["Session_ID", "BlockCount"], how="left")


def split_train_test_sessions(df, test_size, seed):
    """Session-level train/test split, done independently per animal so
    every animal contributes to both splits (matches the source's
    per-animal `train_test_split(mouse_data['Session'].unique(), ...)`,
    but with the manuscript's stated 80/20 ratio -- see config.yaml note).
    """
    rng = np.random.default_rng(seed)
    train_sessions, test_sessions = [], []
    for animal in sorted(df["Animal_Name"].unique()):
        sessions = np.array(sorted(df.loc[df["Animal_Name"] == animal, "Session_ID"].unique()))
        perm = rng.permutation(len(sessions))
        n_test = max(1, int(round(len(sessions) * test_size)))
        test_sessions.extend(sessions[perm[:n_test]].tolist())
        train_sessions.extend(sessions[perm[n_test:]].tolist())
    return set(train_sessions), set(test_sessions)


def _session_sequences(df_animal):
    """List of (choices, rewards) int arrays, one per session."""
    seqs = []
    for _, sess in df_animal.groupby("Session_ID", sort=False):
        sess = sess.sort_values("Trial")
        seqs.append((sess["Choice_Binary"].values.astype(int),
                     sess["Outcome_Binary"].values.astype(int)))
    return seqs


def _session_sequences_with_blocks(df_animal):
    """Same as `_session_sequences`, plus each session's BlockCount array
    (used by adaptive forgetting to reset its within-block trial counter
    at the TRUE block boundary -- see config.yaml note on this vs. the
    source's own every-20-trials approximation).
    """
    seqs = []
    for _, sess in df_animal.groupby("Session_ID", sort=False):
        sess = sess.sort_values("Trial")
        seqs.append((sess["Choice_Binary"].values.astype(int),
                     sess["Outcome_Binary"].values.astype(int),
                     sess["BlockCount"].values))
    return seqs


# ---------------------------------------------------------------------------
# Model A: pure Q-learning (alpha, beta). Q initialized to 0.0, matching the
# source's fit_qlearning_pure (line 2363) exactly -- NOT 0.5, despite the
# manuscript Methods text stating all models init Q to 0.5. Reported as-is.
# ---------------------------------------------------------------------------

def _pure_nll(params, training_data):
    alpha, beta = params
    if not (0 < alpha < 1) or beta <= 0:
        return np.inf
    total_nll, total_trials = 0.0, 0
    for choices, rewards in training_data:
        Q = np.zeros(2, dtype=float)
        for c, r in zip(choices, rewards):
            z = beta * (Q[1] - Q[0])
            p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            p = p1 if c == 1 else (1.0 - p1)
            total_nll -= np.log(p + 1e-12)
            Q[c] += alpha * (r - Q[c])
            total_trials += 1
    return total_nll / total_trials


def fit_qlearning_pure(training_data, x0, bounds):
    scipy_bounds = [tuple(bounds["alpha"]), tuple(bounds["beta"])]
    res = minimize(_pure_nll, x0=np.array(x0), args=(training_data,),
                    method="L-BFGS-B", bounds=scipy_bounds)
    return res.x, res.fun


def qlearning_pure_probs(test_data, params):
    alpha, beta = params
    out = []
    for choices, rewards in test_data:
        Q = np.zeros(2, dtype=float)
        probs = []
        for c, r in zip(choices, rewards):
            z = beta * (Q[1] - Q[0])
            p_right = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            probs.append(p_right)
            Q[c] += alpha * (r - Q[c])
        out.append(np.array(probs))
    return out


# ---------------------------------------------------------------------------
# Model C: +forgetting (alpha, beta, kappa, tau). Unchosen Q decays toward
# 0.5 by factor tau each trial. Q initialized to 0.5, matching the source's
# fit_qlearning_forgetting (line 2270) exactly.
# ---------------------------------------------------------------------------

def _forgetting_nll(params, training_data):
    alpha, beta, kappa, tau = params
    if not (0 < alpha < 1) or beta <= 0 or not (0 <= tau < 1):
        return np.inf
    total_nll, total_trials = 0.0, 0
    for choices, rewards in training_data:
        Q = np.zeros(2, dtype=float) + 0.5
        prev_choice = None
        for c, r in zip(choices, rewards):
            stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
            z = beta * (Q[1] - Q[0]) + kappa * stick
            p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            p = p1 if c == 1 else (1.0 - p1)
            total_nll -= np.log(p + 1e-12)
            Q[c] += alpha * (r - Q[c])
            Q[1 - c] = Q[1 - c] * (1 - tau) + tau * 0.5
            prev_choice = c
            total_trials += 1
    return total_nll / total_trials


def fit_qlearning_forgetting(training_data, x0, bounds):
    scipy_bounds = [tuple(bounds["alpha"]), tuple(bounds["beta"]),
                     tuple(bounds["kappa"]), tuple(bounds["tau"])]
    res = minimize(_forgetting_nll, x0=np.array(x0), args=(training_data,),
                    method="L-BFGS-B", bounds=scipy_bounds)
    return res.x, res.fun


def qlearning_forgetting_probs(test_data, params):
    alpha, beta, kappa, tau = params
    out = []
    for choices, rewards in test_data:
        Q = np.zeros(2, dtype=float) + 0.5
        prev_choice = None
        probs = []
        for c, r in zip(choices, rewards):
            stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
            z = beta * (Q[1] - Q[0]) + kappa * stick
            p_right = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            probs.append(p_right)
            Q[c] += alpha * (r - Q[c])
            Q[1 - c] = Q[1 - c] * (1 - tau) + tau * 0.5
            prev_choice = c
        out.append(np.array(probs))
    return out


# ---------------------------------------------------------------------------
# Model D: +adaptive forgetting (alpha, beta, kappa, tau_max, decay).
# tau = tau_max * exp(-trial_in_block / decay), recomputed each trial from
# a within-block trial counter reset at the TRUE block boundary (this
# pipeline's BlockCount) rather than the source's every-20-trials
# approximation -- see config.yaml note. Q initialized to 0.5.
# ---------------------------------------------------------------------------

def _adaptive_forgetting_nll(params, training_data):
    alpha, beta, kappa, tau_max, decay = params
    if not (0 < alpha < 1) or beta <= 0 or not (0 <= tau_max < 1) or decay <= 0:
        return np.inf
    total_nll, total_trials = 0.0, 0
    for choices, rewards, blocks in training_data:
        Q = np.zeros(2, dtype=float) + 0.5
        prev_choice = None
        prev_block = None
        trial_in_block = 0
        for c, r, b in zip(choices, rewards, blocks):
            if b != prev_block:
                trial_in_block = 0
                prev_block = b
            stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
            z = beta * (Q[1] - Q[0]) + kappa * stick
            p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            p = p1 if c == 1 else (1.0 - p1)
            total_nll -= np.log(p + 1e-12)
            tau = tau_max * np.exp(-trial_in_block / decay)
            Q[c] += alpha * (r - Q[c])
            Q[1 - c] = Q[1 - c] * (1 - tau) + tau * 0.5
            prev_choice = c
            trial_in_block += 1
            total_trials += 1
    return total_nll / total_trials


def fit_qlearning_adaptive_forgetting(training_data, x0, bounds):
    scipy_bounds = [tuple(bounds["alpha"]), tuple(bounds["beta"]), tuple(bounds["kappa"]),
                     tuple(bounds["tau_max"]), tuple(bounds["decay"])]
    res = minimize(_adaptive_forgetting_nll, x0=np.array(x0), args=(training_data,),
                    method="L-BFGS-B", bounds=scipy_bounds)
    return res.x, res.fun


def qlearning_adaptive_forgetting_probs(test_data, params):
    alpha, beta, kappa, tau_max, decay = params
    out = []
    for choices, rewards, blocks in test_data:
        Q = np.zeros(2, dtype=float) + 0.5
        prev_choice = None
        prev_block = None
        trial_in_block = 0
        probs = []
        for c, r, b in zip(choices, rewards, blocks):
            if b != prev_block:
                trial_in_block = 0
                prev_block = b
            stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
            z = beta * (Q[1] - Q[0]) + kappa * stick
            p_right = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            probs.append(p_right)
            tau = tau_max * np.exp(-trial_in_block / decay)
            Q[c] += alpha * (r - Q[c])
            Q[1 - c] = Q[1 - c] * (1 - tau) + tau * 0.5
            prev_choice = c
            trial_in_block += 1
        out.append(np.array(probs))
    return out


# ---------------------------------------------------------------------------
# Orchestration: fit all 4 models per animal on train sessions, predict
# Model_P_Right for every trial (train+test), then derive the two
# quantities every panel actually needs (Model_P_High, Model_P_Switch).
# ---------------------------------------------------------------------------

def _predict_and_assign(df_animal, pred_func, params, colname, with_blocks=False):
    df_animal = df_animal.sort_values(["Session_ID", "Trial"]).copy()
    for _, sess in df_animal.groupby("Session_ID", sort=False):
        sess = sess.sort_values("Trial")
        if with_blocks:
            seq = (sess["Choice_Binary"].values.astype(int),
                   sess["Outcome_Binary"].values.astype(int),
                   sess["BlockCount"].values)
        else:
            seq = (sess["Choice_Binary"].values.astype(int),
                   sess["Outcome_Binary"].values.astype(int))
        probs = pred_func([seq], params)[0]
        df_animal.loc[sess.index, colname] = probs
    return df_animal


def fit_and_predict_all_models(df, cfg, suppfig3_cfg, train_sessions, test_sessions):
    """Fit all 4 models per animal on train sessions; predict Model_P_Right
    for every trial (train+test). Returns (df_with_preds, params_by_model)
    where params_by_model[model] is a per-animal DataFrame of fitted
    parameters + train NLL.
    """
    df = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).reset_index(drop=True)
    df["is_test"] = df["Session_ID"].isin(test_sessions)

    frames = []
    params_records = {m: [] for m in MODEL_NAMES}

    for animal, df_animal in df.groupby("Animal_Name", sort=False):
        df_train = df_animal[df_animal["Session_ID"].isin(train_sessions)]
        train_seqs = _session_sequences(df_train)
        train_seqs_blocks = _session_sequences_with_blocks(df_train)

        # --- pure ---
        pure_params, pure_nll = fit_qlearning_pure(train_seqs, cfg["pure_x0"], cfg["pure_bounds"])
        df_animal = _predict_and_assign(df_animal, qlearning_pure_probs, pure_params, "Model_P_Right_pure")
        params_records["pure"].append({"animal": animal, "alpha": pure_params[0], "beta": pure_params[1],
                                        "train_nll": pure_nll})

        # --- sticky: reuse q_following_model directly, not reimplemented ---
        sticky_params_df = qmodel.fit_sticky_qlearning(
            df_train, suppfig3_cfg["sticky_model_x0"], suppfig3_cfg["sticky_model_bounds"]
        )
        sp = sticky_params_df.iloc[0]
        df_animal_q = qmodel.compute_qvalues_sticky(df_animal, sticky_params_df)
        df_animal = df_animal.copy()
        df_animal["Model_P_Right_sticky"] = df_animal_q["Model_P_Right"].values
        params_records["sticky"].append({"animal": animal, "alpha": sp["alpha"], "beta": sp["beta"],
                                          "kappa": sp["kappa"], "train_nll": sp["nll"]})

        # --- forgetting ---
        forg_params, forg_nll = fit_qlearning_forgetting(train_seqs, cfg["forgetting_x0"], cfg["forgetting_bounds"])
        df_animal = _predict_and_assign(df_animal, qlearning_forgetting_probs, forg_params, "Model_P_Right_forgetting")
        params_records["forgetting"].append({"animal": animal, "alpha": forg_params[0], "beta": forg_params[1],
                                              "kappa": forg_params[2], "tau": forg_params[3], "train_nll": forg_nll})

        # --- adaptive forgetting ---
        adap_params, adap_nll = fit_qlearning_adaptive_forgetting(
            train_seqs_blocks, cfg["adaptive_forgetting_x0"], cfg["adaptive_forgetting_bounds"]
        )
        df_animal = _predict_and_assign(df_animal, qlearning_adaptive_forgetting_probs, adap_params,
                                         "Model_P_Right_adaptive_forgetting", with_blocks=True)
        params_records["adaptive_forgetting"].append({
            "animal": animal, "alpha": adap_params[0], "beta": adap_params[1], "kappa": adap_params[2],
            "tau_max": adap_params[3], "decay": adap_params[4], "train_nll": adap_nll,
        })

        frames.append(df_animal)

    df_out = pd.concat(frames, ignore_index=True)
    df_out = _add_model_high_and_switch(df_out, MODEL_NAMES)
    params_by_model = {m: pd.DataFrame(recs).set_index("animal") for m, recs in params_records.items()}
    return df_out, params_by_model


def _add_model_high_and_switch(df, model_names):
    """For each model's Model_P_Right column, derive:
    - Model_P_High: probability the model chooses the 80%-reward side
      (using the block-level High_Side_Binary computed in prepare_dataframe)
    - Model_P_Switch: probability the model switches relative to the
      animal's ACTUAL previous choice (teacher-forced, analytic -- same
      computation as the source's own trial-by-trial r_switch_trial cell:
      P(switch) = P(right) if prev actual choice was left, else 1-P(right))
    """
    df = df.sort_values(["Animal_Name", "Session_ID", "Trial"]).copy()
    prev_choice = df.groupby(["Animal_Name", "Session_ID"])["Choice_Binary"].shift(1)
    for m in model_names:
        p_right = df[f"Model_P_Right_{m}"]
        high = np.where(df["High_Side_Binary"] == 1, p_right, 1 - p_right)
        df[f"Model_P_High_{m}"] = np.where(df["High_Side_Binary"].isna(), np.nan, high)
        switch = np.where(prev_choice == 0, p_right, 1 - p_right)
        df[f"Model_P_Switch_{m}"] = np.where(prev_choice.isna(), np.nan, switch)
    return df


# ---------------------------------------------------------------------------
# Panels a-d: reversal-aligned p(high)/p(switch) curves, session-level
# scatter, and the four per-animal comparison metrics behind panel e.
# ---------------------------------------------------------------------------

def build_reversal_windows(df, pre, post):
    """Long-format table: one row per (session, reversal event, position),
    position in [-pre..post]. position>=0 trials come from the NEW block
    (position == Trial_in_Block); position<0 trials come from the tail of
    the OLD block (position -1 = its last trial, etc.). A trial can appear
    in more than one reversal's window when blocks are shorter than
    pre+post -- expected for a peri-event alignment, not a bug.
    """
    model_cols = [c for c in df.columns if c.startswith("Model_P_High_") or c.startswith("Model_P_Switch_")]
    keep_cols = ["Animal_Name", "Session_ID", "Trial", "High_Prob", "Switch", "is_test"] + model_cols
    records = []
    for session_id, sess in df.groupby("Session_ID", sort=False):
        sess = sess.sort_values("Trial").reset_index(drop=True)
        blocks = sess["BlockCount"].values
        change_idx = np.where(np.diff(blocks) != 0)[0] + 1
        for reversal_num, idx0 in enumerate(change_idx):
            lo, hi = max(0, idx0 - pre), min(len(sess), idx0 + post + 1)
            window = sess.iloc[lo:hi][keep_cols].copy()
            window["position"] = np.arange(lo, hi) - idx0
            window["reversal_id"] = f"{session_id}_{reversal_num}"
            records.append(window)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(columns=keep_cols + ["position", "reversal_id"])


def block_position_curves(windows, model, test_only=True):
    """Mean +/- SEM of High_Prob/Switch (marmoset) and Model_P_High/Switch
    (model) by position, pooled across all animals -- panels a-d's top/
    bottom line plots.
    """
    w = windows[windows["is_test"]] if test_only else windows
    g = w.groupby("position")
    return pd.DataFrame({
        "phigh_marmoset_mean": g["High_Prob"].mean(),
        "phigh_marmoset_sem": g["High_Prob"].sem(),
        "phigh_model_mean": g[f"Model_P_High_{model}"].mean(),
        "phigh_model_sem": g[f"Model_P_High_{model}"].sem(),
        "pswitch_marmoset_mean": g["Switch"].mean(),
        "pswitch_marmoset_sem": g["Switch"].sem(),
        "pswitch_model_mean": g[f"Model_P_Switch_{model}"].mean(),
        "pswitch_model_sem": g[f"Model_P_Switch_{model}"].sem(),
    }).reset_index()


def session_switch_scatter(df, model, test_only=True):
    """Session-level mean observed vs. model-predicted p(switch) -- panels
    a-d's right scatter. One dot per (held-out test) session, pooled
    across animals.
    """
    d = df[df["is_test"]] if test_only else df
    d = d.dropna(subset=["Switch", f"Model_P_Switch_{model}"])
    return (
        d.groupby("Session_ID")
        .agg(observed=("Switch", "mean"), predicted=(f"Model_P_Switch_{model}", "mean"))
        .reset_index()
    )


def per_animal_metrics(df, windows, model):
    """The four per-animal metrics behind panel e (test NLL, r(p_high),
    r(p_switch), trial-by-trial r(switch)), test-sessions only, matching
    the source's `SYSTEMATIC MODEL COMPARISON` cell's per-animal loop.
    """
    rows = []
    for animal, d in df.groupby("Animal_Name", sort=False):
        test_d = d[d["is_test"]]
        p_right = test_d[f"Model_P_Right_{model}"].values
        choice = test_d["Choice_Binary"].values
        p_actual = np.where(choice == 1, p_right, 1 - p_right)
        test_nll = float(-np.mean(np.log(np.clip(p_actual, 1e-12, 1))))

        w = windows[(windows["Animal_Name"] == animal) & (windows["is_test"])]
        curve = w.groupby("position").agg(
            phigh_m=("High_Prob", "mean"), phigh_mod=(f"Model_P_High_{model}", "mean"),
            pswitch_m=("Switch", "mean"), pswitch_mod=(f"Model_P_Switch_{model}", "mean"),
        ).dropna()
        r_phigh = float(np.corrcoef(curve["phigh_m"], curve["phigh_mod"])[0, 1]) if len(curve) > 1 else np.nan
        r_pswitch = float(np.corrcoef(curve["pswitch_m"], curve["pswitch_mod"])[0, 1]) if len(curve) > 1 else np.nan

        trial_d = test_d.dropna(subset=["Switch", f"Model_P_Switch_{model}"])
        r_switch_trial = (
            float(np.corrcoef(trial_d["Switch"], trial_d[f"Model_P_Switch_{model}"])[0, 1])
            if len(trial_d) > 1 else np.nan
        )

        rows.append({"animal": animal, "model": model, "test_nll": test_nll,
                     "r_phigh": r_phigh, "r_pswitch": r_pswitch, "r_switch_trial": r_switch_trial})
    return pd.DataFrame(rows)


def naive_session_level_metrics(df, model):
    """Session-level (NOT per-animal) test NLL and trial-by-trial
    r(switch), treating each held-out test session as an independent
    unit. Included ONLY to demonstrate the pseudoreplication this
    introduces -- sessions from the same animal are not independent
    draws, so n here is n_sessions, not n_animals. The per-animal version
    (`per_animal_metrics` / `model_comparison_table`, n=5 animals) is the
    one treated as real, per this pipeline's standing convention for
    session-vs-animal comparisons (see e.g. Fig 2/Fig 3 WSLS, Supp Fig 3
    outcome panel).
    """
    rows = []
    test_d = df[df["is_test"]]
    for session_id, d in test_d.groupby("Session_ID"):
        p_right = d[f"Model_P_Right_{model}"].values
        choice = d["Choice_Binary"].values
        p_actual = np.where(choice == 1, p_right, 1 - p_right)
        test_nll = float(-np.mean(np.log(np.clip(p_actual, 1e-12, 1))))
        trial_d = d.dropna(subset=["Switch", f"Model_P_Switch_{model}"])
        r_switch_trial = (
            float(np.corrcoef(trial_d["Switch"], trial_d[f"Model_P_Switch_{model}"])[0, 1])
            if len(trial_d) > 1 else np.nan
        )
        rows.append({"session_id": session_id, "model": model, "test_nll": test_nll,
                     "r_switch_trial": r_switch_trial})
    return pd.DataFrame(rows)


def model_comparison_table(df, windows):
    """All 4 models' per-animal metrics, plus the composite score (mean of
    the 4 normalized metrics, matching the source's own normalization:
    nll_score = 1 - test_nll / max(test_nll) computed GLOBALLY across all
    model/animal rows, then composite = mean(nll_score, r_phigh,
    r_switch_trial, r_pswitch)).
    """
    all_rows = pd.concat([per_animal_metrics(df, windows, m) for m in MODEL_NAMES], ignore_index=True)
    max_nll = all_rows["test_nll"].max()
    all_rows["nll_score"] = 1 - all_rows["test_nll"] / max_nll
    all_rows["composite"] = all_rows[["nll_score", "r_phigh", "r_switch_trial", "r_pswitch"]].mean(axis=1)
    return all_rows
