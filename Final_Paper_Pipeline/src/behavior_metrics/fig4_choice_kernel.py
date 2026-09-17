"""Figure 4 -- Q-learning model with choice kernel: analyses specific to this
figure that build on top of q_following_model.py's already-fitted sticky
Q-learning model (see that module's docstring for the model itself, which is
reused unmodified here -- this file adds no new model fitting, only new
derived quantities from its existing output).

Covers:
  - action-outcome history encoding for panel d (log-odds by 3-trial
    action-outcome history, e.g. 'RlR')
  - generic log-odds binning for panels e/f
  - the sigmoid (panel e) and peaked/Gaussian (panel f) curve_fit wrappers

JUDGMENT CALL on the history encoding (panel d): the manuscript's Methods/
Results describe encoding each trial as R/L for choice (upper/lower case for
rewarded/unrewarded). src/features/history_encoding.py::add_history_cols_marmoset
already builds exactly this kind of trailing-N-trial history (its
`decision_seq`/`reward_seq` columns are reused below, unmodified, for their
correct per-session sliding-window construction), but that module's own
symm=False letter mapping (its `RL_history` column) turns out to be inverted
relative to true choice side: empirically, decision digit '1' (this
pipeline's Choice_Binary==1, i.e. actually Right/Top) maps to letter 'L', and
digit '0' (actually Left/Bottom) maps to letter 'R' -- confirmed by checking
RL_history against decision_seq digit-by-digit on the real data. That
function is shared with Fig 2/3 and is not touched here (out of scope, and
Fig2/3 never actually use its RL_history output -- see that module's own
docstring). This module instead builds the corrected letter directly from
decision_seq/reward_seq's raw digits so panel d's bar labels ('RlR', 'rlr',
etc.) refer to true choice side.

FINDING on sign convention (panels a/d/e/f): the manuscript's Fig 4b box
defines V(t) = beta*[Q(right,t)-Q(left,t)] + kappa*C(t), i.e. logit(P(right))
-- and q_following_model.py's "Model_Log_Odds" is exactly that quantity.
But the manuscript's own worked examples for panel d ("positive log-odds are
dominated by recently rewarded choices on the left side (e.g., 'LLL', 'rLL'),
whereas negative log-odds are dominated by histories with recent rewards on
the right side (e.g., 'RRR', 'RlR')") only reproduce with the OPPOSITE sign,
i.e. logit(P(left)) = -Model_Log_Odds -- confirmed empirically: computing
history_logodds_summary on the raw Model_Log_Odds puts 'LLL'/'rLL' at the
very negative end and 'RRR'/'RlR' at the very positive end (backwards from
the text); negating it reproduces the manuscript's stated examples exactly.
This also matches panel e's axis being "p(Choose Left)" rather than "p(Choose
Right)". The callers below (Fig4 notebook) therefore build a
`Log_Odds_Left = -Model_Log_Odds` column and pass that as `logodds_col`
throughout panels a/d/e/f; q_following_model.py itself is untouched (V(t) as
literally defined in the manuscript's equation box remains logit(P(right))).
"""

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from ..features.history_encoding import add_history_cols_marmoset


def add_action_outcome_history(df, n, choice_col="Choice_Binary", outcome_col="Outcome_Binary"):
    """Add an `Action_Outcome_History` column: the trailing `n`-trial
    choice+outcome code (e.g. 'RlR'), built from the correctly-signed
    Choice_Binary/Outcome_Binary digits -- choice digit 1 -> 'R', 0 -> 'L';
    uppercase if that trial was rewarded, lowercase if not. Reuses
    add_history_cols_marmoset for the per-session sliding-window
    decision_seq/reward_seq construction (see module docstring for why the
    letter itself is rebuilt rather than reusing that function's own
    RL_history column).

    For the small number of "vertical" (Top/Bottom) sessions in this
    dataset (3/100, all one animal -- Choice_Binary==1 there means Top, not
    Right), the resulting code still uses R/L letters generically, pooling
    them with the Left/Right sessions under the same option-1-vs-option-0
    convention. This matches how the fitted model itself treats "option 1"
    vs "option 0" (q_following_model.py has no separate vertical-session
    log-odds convention either), and is a negligible fraction of trials
    (~2.5%).
    """
    df_hist = add_history_cols_marmoset(df, n, choice_col=choice_col, outcome_col=outcome_col)
    has_hist = df_hist["decision_seq"].notna()

    def _code(decision_seq, reward_seq):
        letters = []
        for d, r in zip(decision_seq, reward_seq):
            letter = "R" if d == "1" else "L"
            letters.append(letter if r == "1" else letter.lower())
        return "".join(letters)

    df_hist["Action_Outcome_History"] = pd.Series(np.nan, index=df_hist.index).astype(object)
    df_hist.loc[has_hist, "Action_Outcome_History"] = [
        _code(d, r) for d, r in zip(
            df_hist.loc[has_hist, "decision_seq"], df_hist.loc[has_hist, "reward_seq"]
        )
    ]
    return df_hist


def history_logodds_summary(df_hist, logodds_col="Model_Log_Odds",
                              history_col="Action_Outcome_History", min_trials=100):
    """Mean/SEM/n log-odds prediction per action-outcome history code,
    filtered to codes with >= min_trials observations, sorted ascending by
    mean log-odds (panel d)."""
    d = df_hist[df_hist[history_col].notna() & df_hist[logodds_col].notna()]
    summary = d.groupby(history_col)[logodds_col].agg(["mean", "sem", "count"]).reset_index()
    summary.columns = [history_col, "mean_logodds", "sem", "n"]
    summary = summary[summary["n"] >= min_trials].sort_values("mean_logodds").reset_index(drop=True)
    return summary


def bin_by_logodds(df, value_col, logodds_col="Model_Log_Odds", n_bins=30, clip=5.0,
                    min_trials_per_bin=5, group_col="Animal_Name"):
    """Bin `value_col` (e.g. Chose_Left, PhysicalSwitch) by `logodds_col`,
    separately per `group_col` (animal), for the psychometric-style panels
    (e/f). log-odds are clipped to +/-clip before binning (matches the
    Fig 4 candidate notebook's own preprocessing,
    `_source_archive/by_figure/Fig4/06_Fig3_2ABT_v2_candidate.ipynb`
    line ~4264/4377/4497: `np.clip(log_odds, -5, 5)`, `bins = np.linspace(-5, 5, 30)`,
    bins with n<5 dropped).

    Returns a dataframe with columns [group_col, 'bin_center', 'mean', 'sem', 'n'].
    """
    d = df[df[logodds_col].notna() & df[value_col].notna()].copy()
    d["_logodds_clipped"] = d[logodds_col].clip(-clip, clip)
    bin_edges = np.linspace(-clip, clip, n_bins)
    d["_bin"] = pd.cut(d["_logodds_clipped"], bins=bin_edges)

    rows = []
    for group, gdf in d.groupby(group_col):
        binned = gdf.groupby("_bin", observed=True)[value_col].agg(["mean", "sem", "count"]).reset_index()
        binned = binned[binned["count"] >= min_trials_per_bin].dropna(subset=["mean"])
        for _, row in binned.iterrows():
            rows.append({
                group_col: group,
                "bin_center": float(row["_bin"].mid),
                "mean": float(row["mean"]),
                "sem": float(row["sem"]) if pd.notna(row["sem"]) else np.nan,
                "n": int(row["count"]),
            })
    return pd.DataFrame(rows)


def sigmoid(x, x0, k):
    """Standard logistic function: 1 / (1 + exp(-k*(x-x0)))."""
    return 1.0 / (1.0 + np.exp(-k * (x - x0)))


def fit_sigmoid_per_animal(binned_df, group_col="Animal_Name", p0=(0.0, -1.0),
                             bounds=((-10.0, -20.0), (10.0, 20.0))):
    """curve_fit a standard logistic per animal against its own binned
    (bin_center, mean) points (panel e: p(choose left) vs. log-odds
    prediction -- a monotonically DECREASING function of log-odds, hence
    p0/bounds allow negative k, unlike a p(choose right)-style fit)."""
    rows = []
    for animal, gdf in binned_df.groupby(group_col):
        if len(gdf) < 4:
            continue
        try:
            popt, _ = curve_fit(sigmoid, gdf["bin_center"].values, gdf["mean"].values,
                                 p0=p0, bounds=bounds, maxfev=10000)
        except RuntimeError:
            continue
        rows.append({group_col: animal, "bias": float(popt[0]), "sensitivity": float(popt[1]),
                     "n_bins": int(len(gdf))})
    return pd.DataFrame(rows)


def gaussian_bump(x, x0, sigma, amplitude, baseline):
    """Symmetric peaked function: baseline + amplitude * exp(-(x-x0)^2 / (2*sigma^2)).

    JUDGMENT CALL: panel f (p(switch) vs. log-odds prediction) is an
    inverted-U -- highest switch probability near log-odds=0, lowest at the
    extremes -- not a monotonic function of log-odds, so the "standard
    logistic" used for panel e cannot fit it (a logistic is monotonic by
    construction). A symmetric Gaussian bump is the simplest curve_fit-able
    function that can actually represent a single-peaked, symmetric-around-
    zero relationship, so it is used here instead, still via
    scipy.optimize.curve_fit. This is flagged explicitly since it departs
    from a literal "logistic fit" for this one panel.
    """
    return baseline + amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


def select_example_session(df, min_trials=100):
    """Pick panel a's example session: the manuscript's legend explicitly
    describes THIS example as "Choice: Top/Bottom" -- i.e. one of the small
    number of "vertical" sessions in this dataset (3/100, all one animal;
    see q_following_model.prepare_2abt_dataframe's own vertical-session
    detection). Among those, picks the one with the most trials (>=
    min_trials), ties broken by Session_ID sort order -- a documented,
    data-driven judgment call (the manuscript does not name the specific
    session), not tuned to reproduce any particular reported number.
    """
    vertical_mask = ~(
        df["PhysicalChoice"].str.contains("Left") & df["PhysicalChoice"].str.contains("Right")
    ) & (df["PhysicalChoice"].str.contains("Top") | df["PhysicalChoice"].str.contains("Bottom"))
    session_sides = df.groupby("Session_ID")["PhysicalChoice"].agg(
        lambda s: s.str.contains("Left").any() and s.str.contains("Right").any()
    )
    vertical_sessions = session_sides[~session_sides].index
    counts = df[df["Session_ID"].isin(vertical_sessions)].groupby("Session_ID").size()
    counts = counts[counts >= min_trials].sort_values(ascending=False)
    if len(counts) == 0:
        raise ValueError("No vertical (Top/Bottom) session with >= min_trials trials found.")
    return counts.index[0]


def high_prob_shading_spans(session_df, trial_col="Trial", high_prob_col="High_Prob_Is_Right"):
    """Contiguous (start, end) trial spans where `high_prob_col` is False
    (i.e. the OTHER coded option -- Bottom, for a vertical/Top-Bottom
    session -- is the high-probability choice), for panel a's background
    shading ("Shaded regions indicate that the bottom is the high
    probability choice.")."""
    d = session_df.sort_values(trial_col).reset_index(drop=True)
    shade = ~d[high_prob_col].astype(bool)
    spans = []
    start = None
    for i, val in enumerate(shade):
        if val and start is None:
            start = d[trial_col].iloc[i]
        elif not val and start is not None:
            spans.append((start, d[trial_col].iloc[i - 1]))
            start = None
    if start is not None:
        spans.append((start, d[trial_col].iloc[-1]))
    return spans


def fit_gaussian_bump_per_animal(binned_df, group_col="Animal_Name",
                                   p0=(0.0, 2.0, 0.3, 0.1),
                                   bounds=((-5.0, 0.1, 0.0, 0.0), (5.0, 20.0, 1.0, 1.0))):
    """curve_fit a Gaussian bump per animal against its own binned
    (bin_center, mean) points (panel f)."""
    rows = []
    for animal, gdf in binned_df.groupby(group_col):
        if len(gdf) < 5:
            continue
        try:
            popt, _ = curve_fit(gaussian_bump, gdf["bin_center"].values, gdf["mean"].values,
                                 p0=p0, bounds=bounds, maxfev=10000)
        except RuntimeError:
            continue
        rows.append({
            group_col: animal, "peak_center": float(popt[0]), "width": float(popt[1]),
            "amplitude": float(popt[2]), "baseline": float(popt[3]), "n_bins": int(len(gdf)),
        })
    return pd.DataFrame(rows)
