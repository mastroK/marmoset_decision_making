"""Reviewer-response addition (RA4_condition_wise_RL.ipynb, R2-5/R1-D):
sticky Q-learning + choice-kernel parameter-recovery simulation. Simulates
an agent with KNOWN (alpha, beta, kappa) under a given reward schedule
(100-0 / 90-10 / 80-20), then refits it with the exact same fitting
routine used on real data (`q_following_model.fit_sticky_qlearning`, not a
reimplementation) -- demonstrating whether 100-0 is unidentifiable in
principle, independent of any real data-quality issue.

A single stationary block (no reversals) is simulated -- sufficient for
this identifiability question, since it is about whether the LIKELIHOOD
itself constrains the parameters under a given reward schedule, not about
block-transition dynamics. At 100-0, the reward is a deterministic
function of choice, so once the policy converges the agent always picks
the same (rewarded) side; the choice-kernel term (kappa) and the value
term (beta) become observationally near-equivalent at conveying "keep
picking this side," and alpha only matters transiently before convergence
-- this is the concrete mechanism the recovery-quality gap below is
expected to show, not asserted without checking.
"""

import numpy as np
import pandas as pd

from . import q_following_model as qmodel


def simulate_sticky_agent(alpha, beta, kappa, n_trials, prob_condition, seed):
    """Simulate one agent's Choice_Binary/Outcome_Binary trial sequence
    under the sticky Q-learning + choice-kernel policy
    (`q_following_model._qlearn_sticky_nll`'s own generative model, run
    forward instead of evaluated), for reward schedule `prob_condition`
    (e.g. "100-0"/"90-10"/"80-20" -- parsed as high/low reward
    probabilities). WLOG the right (1) option is the high-probability
    side for the simulated agent.
    """
    rng = np.random.default_rng(seed)
    high_str, low_str = prob_condition.split("-")
    high_prob, low_prob = int(high_str) / 100, int(low_str) / 100

    Q = np.array([0.5, 0.5])
    choices, rewards = np.empty(n_trials, dtype=int), np.empty(n_trials, dtype=int)
    prev_choice = None
    for t in range(n_trials):
        stick = 0.0 if prev_choice is None else (1.0 if prev_choice == 1 else -1.0)
        z = beta * (Q[1] - Q[0]) + kappa * stick
        p_right = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
        c = int(rng.random() < p_right)
        r_prob = high_prob if c == 1 else low_prob
        r = int(rng.random() < r_prob)
        choices[t], rewards[t] = c, r
        Q[c] += alpha * (r - Q[c])
        prev_choice = c

    return choices, rewards


def recover_parameters(true_params, n_trials, prob_condition, x0, bounds, seed):
    """Simulate one agent at `true_params` under `prob_condition`, refit
    with `q_following_model.fit_sticky_qlearning` (the same routine used
    on real data), and return true vs. fitted parameters + recovery
    error. `true_params` = {"alpha", "beta", "kappa"}.
    """
    choices, rewards = simulate_sticky_agent(
        true_params["alpha"], true_params["beta"], true_params["kappa"], n_trials, prob_condition, seed
    )
    synthetic_df = pd.DataFrame({
        "Animal_Name": "sim", "Session_ID": "sim_session",
        "Trial": np.arange(n_trials), "Choice_Binary": choices, "Outcome_Binary": rewards,
    })
    fit_df = qmodel.fit_sticky_qlearning(synthetic_df, x0, bounds)
    fitted = fit_df.iloc[0]
    fitted_params = {"alpha": float(fitted["alpha"]), "beta": float(fitted["beta"]), "kappa": float(fitted["kappa"])}
    return {
        "true": true_params,
        "fitted": fitted_params,
        "error": {p: fitted_params[p] - true_params[p] for p in ("alpha", "beta", "kappa")},
        "nll": float(fitted["nll"]),
        "prob_condition": prob_condition,
        "n_trials": n_trials,
        "seed": seed,
    }


def recovery_quality_by_condition(true_param_sets, conditions, n_trials, x0, bounds, n_seeds_per_set, base_seed):
    """Runs `recover_parameters` for every (true_param_set, condition,
    seed replicate) combination and summarizes mean absolute recovery
    error per parameter per condition -- the table that shows whether
    100-0 is genuinely harder to identify than 90-10/80-20, or not.
    `true_param_sets` = list of {"alpha", "beta", "kappa"} dicts (e.g. the
    5 animals' own already-fitted Fig 4 params, reused as realistic
    ground truth).
    """
    records = []
    for cond in conditions:
        for set_idx, true_params in enumerate(true_param_sets):
            for rep in range(n_seeds_per_set):
                seed = base_seed + 1000 * set_idx + rep
                result = recover_parameters(true_params, n_trials, cond, x0, bounds, seed)
                records.append({
                    "prob_condition": cond, "param_set": set_idx, "replicate": rep,
                    **{f"true_{p}": true_params[p] for p in ("alpha", "beta", "kappa")},
                    **{f"fitted_{p}": result["fitted"][p] for p in ("alpha", "beta", "kappa")},
                    **{f"abs_error_{p}": abs(result["error"][p]) for p in ("alpha", "beta", "kappa")},
                })
    detail_df = pd.DataFrame(records)
    summary = detail_df.groupby("prob_condition")[
        ["abs_error_alpha", "abs_error_beta", "abs_error_kappa"]
    ].agg(["mean", "sem"])
    return detail_df, summary
