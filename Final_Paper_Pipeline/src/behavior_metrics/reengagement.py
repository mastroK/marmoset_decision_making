"""Within-block bout accuracy (start vs. end) and post-block-transition
re-engagement speed (Fig 1 panels h-i).

Ported from _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb's
"STATS EXTRACTION: Results Section 1" section (STAT 5, 6, 7 -- lines
5406-5544), NOT from the earlier exploratory bout-analysis cells (CELL 12-16,
lines 2350-3800ish) that precede it in the notebook. Those earlier cells are
iterative exploration -- e.g. "CELL 13 REVISED", two differently-coded
"CELL 16" cells, and a same-named output file (`Prior_Losses_by_ITI.svg`)
overwritten by three different cells in turn -- and the STATS EXTRACTION
section is the one place in the notebook that is explicitly labeled as
generating the manuscript's reported Results-section numbers, so it is
treated as the authoritative spec here rather than the earlier drafts.

Bout definition here (a THIRD distinct "engagement" boundary rule in this
figure, alongside stage 2's long_break_threshold and Fig1's other criterion
rules): a new bout starts whenever ITI >= bout_iti_threshold OR the block
changes. A "within-block bout" is one where the block never changes inside
it; a "block-transition bout" is one that starts exactly at a block change.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def _add_bout_ids(session_data, bout_iti_threshold):
    """One session's trials (sorted by Trial), with ITI, block_change, and
    a bout_num that increments at every ITI break or block change.

    The session's first trial always starts bout_num=1 (there's nothing
    before it to compare ITI/block against), but its `block_change` is
    explicitly forced to False: comparing `Block` to a shifted-NaN
    previous value otherwise evaluates to True (pandas' `!=` treats NaN as
    unequal to everything), which would spuriously mark the first trial of
    every session as if it were a genuine block transition -- creating a
    fake 1-trial "block-transition bout" at the start of every session and
    simultaneously truncating the session's real first within-block bout
    by one trial. Fixed here since it's a real bug in bout-type
    classification (found while building Fig 1 panels h/i's per-rank
    breakdown), not something safe to leave as a pre-existing artifact.
    """
    d = session_data.sort_values("Trial").copy()
    d["ITI"] = (d["AbsoluteTrialStartTime"] - d["AbsoluteTrialStartTime"].shift(1)) / 1000
    d["block_change"] = d["Block"] != d["Block"].shift(1)
    d.iloc[0, d.columns.get_loc("block_change")] = False
    d["new_bout"] = (d["ITI"] >= bout_iti_threshold) | d["block_change"]
    d["bout_num"] = d["new_bout"].cumsum()
    return d


def within_block_bouts(df, bout_iti_threshold, start_end_window):
    """One row per within-block bout (bouts that never cross a block
    change): start/end accuracy over the first/last `start_end_window`
    trials, ranked by order of occurrence within the session. Ported from
    lines 5410-5437."""
    records = []
    for session_id in df["Session_ID"].unique():
        d = _add_bout_ids(df[df["Session_ID"] == session_id], bout_iti_threshold)
        for bout_num in d["bout_num"].unique():
            bout = d[d["bout_num"] == bout_num]
            if len(bout) == 0 or bout["block_change"].any():
                continue
            records.append({
                "session_id": session_id,
                "bout_num": bout_num,
                "animal": bout["Animal_Name"].iloc[0],
                "n_trials": len(bout),
                "start_acc": bout.iloc[:start_end_window]["Outcome"].mean(),
                "end_acc": bout.iloc[-start_end_window:]["Outcome"].mean(),
            })
    wb_df = pd.DataFrame(records)
    wb_df["bout_rank"] = wb_df.groupby("session_id").cumcount() + 1
    return wb_df


def start_vs_end_accuracy_by_rank(wb_df, bout_ranks):
    """For each bout rank (1st, 2nd, 3rd... bout in the session), a paired
    per-animal Wilcoxon test of start accuracy vs. end accuracy. Ported from
    lines 5439-5466 (Fig 1H / STAT 5)."""
    records = []
    for rank in bout_ranks:
        bout_data = wb_df[wb_df["bout_rank"] == rank]
        start_per_animal = bout_data.groupby("animal")["start_acc"].mean()
        end_per_animal = bout_data.groupby("animal")["end_acc"].mean()
        common = start_per_animal.index.intersection(end_per_animal.index)

        if len(common) >= 3:
            _, p_val = scipy_stats.wilcoxon(start_per_animal[common], end_per_animal[common])
        else:
            p_val = np.nan

        records.append({
            "bout_rank": rank,
            "n_bouts": int(len(bout_data)),
            "n_animals": int(len(common)),
            "start_mean": float(start_per_animal.mean()),
            "start_sem": float(scipy_stats.sem(start_per_animal)),
            "end_mean": float(end_per_animal.mean()),
            "end_sem": float(scipy_stats.sem(end_per_animal)),
            "wilcoxon_p": float(p_val) if not np.isnan(p_val) else None,
        })
    return pd.DataFrame(records)


def bout_length_kruskal(wb_df, bout_ranks):
    """Kruskal-Wallis test of bout length (n_trials) across bout ranks.
    Ported from lines 5472-5489 (Fig 1H / STAT 6)."""
    groups = [wb_df[wb_df["bout_rank"] == r]["n_trials"].values for r in bout_ranks]
    h_stat, p_val = scipy_stats.kruskal(*groups)
    return {
        "bout_ranks": list(bout_ranks),
        "means": [float(np.mean(g)) for g in groups],
        "sems": [float(scipy_stats.sem(g)) for g in groups],
        "n_per_rank": [int(len(g)) for g in groups],
        "h_stat": float(h_stat),
        "p_value": float(p_val),
    }


def block_transition_bouts(df, bout_iti_threshold, criterion_accuracy, criterion_window):
    """One row per bout that starts at a block transition: whether/when it
    reaches `criterion_accuracy` over a rolling `criterion_window`-trial
    window. Ported from lines 5496-5523 (Fig 1I / STAT 7). Also carries a
    `bout_rank` (1st, 2nd, 3rd... block-transition bout in the session) for
    panel i's start/end and bout-length sub-panels, which STAT 7 itself
    doesn't need (it pools across all ranks) but the figure does.
    """
    records = []
    for session_id in df["Session_ID"].unique():
        d = _add_bout_ids(df[df["Session_ID"] == session_id], bout_iti_threshold)
        rank = 0
        for bout_num in sorted(d["bout_num"].unique()):
            bout = d[d["bout_num"] == bout_num]
            if len(bout) == 0 or not bout.iloc[0]["block_change"]:
                continue
            rank += 1
            outcomes = bout["Outcome"].values
            crit_trial = None
            for i in range(len(outcomes) - criterion_window + 1):
                if np.mean(outcomes[i:i + criterion_window]) >= criterion_accuracy:
                    crit_trial = i + criterion_window
                    break
            records.append({
                "session_id": session_id,
                "animal": bout["Animal_Name"].iloc[0],
                "bout_rank": rank,
                "n_trials": len(bout),
                "trials_to_criterion": crit_trial,
                "reached_criterion": crit_trial is not None,
                "acc_first5": np.mean(outcomes[:5]) if len(outcomes) >= 5 else np.nan,
            })
    return pd.DataFrame(records)


def reengagement_summary(bt_df):
    """Fraction reaching criterion and mean/SEM trials-to-criterion among
    those that did. Ported from lines 5527-5530 (Fig 1I / STAT 7)."""
    frac_criterion = bt_df["reached_criterion"].mean()
    reached = bt_df[bt_df["reached_criterion"]]["trials_to_criterion"]
    return {
        "n_bouts_total": int(len(bt_df)),
        "n_animals": int(bt_df["animal"].nunique()),
        "frac_reached_criterion": float(frac_criterion),
        "mean_trials_to_criterion": float(reached.mean()),
        "sem_trials_to_criterion": float(scipy_stats.sem(reached)),
    }


def _bouts_with_positions(df, bout_iti_threshold, bout_type):
    """All trials belonging to bouts of the given type ('within_block' or
    'block_transition'), with `trial_in_bout` (0-indexed position within
    the bout) and `bout_rank` (1st, 2nd, 3rd... bout of that type in the
    session) columns added. Feeds panels h/i's line-plot sub-panel
    ("performance across the first 15 trials").
    """
    records = []
    for session_id in df["Session_ID"].unique():
        d = _add_bout_ids(df[df["Session_ID"] == session_id], bout_iti_threshold)
        rank = 0
        for bout_num in sorted(d["bout_num"].unique()):
            bout = d[d["bout_num"] == bout_num]
            starts_at_transition = bool(bout["block_change"].iloc[0])
            crosses_block = bool(bout["block_change"].iloc[1:].any())
            if bout_type == "within_block" and (starts_at_transition or crosses_block):
                continue
            if bout_type == "block_transition" and not starts_at_transition:
                continue
            rank += 1
            bout = bout.copy()
            bout["bout_rank"] = rank
            bout["trial_in_bout"] = range(len(bout))
            records.append(bout)
    return pd.concat(records, ignore_index=True) if records else pd.DataFrame()


def bout_position_accuracy(df, bout_iti_threshold, bout_type, bout_ranks, max_trials):
    """Mean accuracy at each within-bout trial position (0..max_trials-1),
    for each of `bout_ranks`, averaged per-animal-then-across-animals.
    Ported from Fig 1H/I's line-plot sub-panel.
    """
    d = _bouts_with_positions(df, bout_iti_threshold, bout_type)
    d = d[d["bout_rank"].isin(bout_ranks) & (d["trial_in_bout"] < max_trials)]
    per_animal = d.groupby(["bout_rank", "trial_in_bout", "Animal_Name"])["Outcome"].mean().reset_index()
    return per_animal.groupby(["bout_rank", "trial_in_bout"])["Outcome"].agg(
        mean="mean", sem="sem", n="count"
    ).reset_index()


def representative_session_trace(df, bout_iti_threshold, rolling_window):
    """Full per-trial trace (block, bout, ITI, rolling accuracy) for one
    representative session -- Fig 1 panel g. Session selection and the
    rolling-accuracy window match the manuscript's own "CELL 16: Single
    Session Schematic" code exactly (provided directly by the user): among
    sessions with >=7 bouts (by whatever bout-numbering the session happens
    to have), take the SECOND one (index 1) in Session_ID's natural
    groupby order -- not "the session with the most bouts/blocks", which
    was this function's own earlier guess before that source code was
    available. Rolling accuracy uses a CENTERED window, matching the
    source's `.rolling(window, min_periods=1, center=True)`.
    """
    all_bout_ids = []
    for session_id in df["Session_ID"].unique():
        d = _add_bout_ids(df[df["Session_ID"] == session_id], bout_iti_threshold)
        all_bout_ids.append(d)
    df_multi_bout = pd.concat(all_bout_ids, ignore_index=True)

    session_bout_counts = df_multi_bout.groupby("Session_ID")["bout_num"].max()
    good_sessions = session_bout_counts[session_bout_counts >= 7]
    session_id = good_sessions.index[1] if len(good_sessions) > 1 else good_sessions.index[0]

    d = df_multi_bout[df_multi_bout["Session_ID"] == session_id].sort_values("Trial").reset_index(drop=True)
    d["rolling_accuracy"] = d["Outcome"].rolling(rolling_window, min_periods=1, center=True).mean()
    return d, session_id


def bouts_per_session(df, bout_iti_threshold):
    """Mean +/- SD number of bouts (of any type) per session, across all
    sessions -- the "8.8+/-5.0 bouts per session" statistic quoted in Fig 1
    panel g's legend, recomputed from scratch rather than assumed."""
    counts = []
    for session_id in df["Session_ID"].unique():
        d = _add_bout_ids(df[df["Session_ID"] == session_id], bout_iti_threshold)
        counts.append(d["bout_num"].nunique())
    counts = np.array(counts)
    return {"mean": float(counts.mean()), "std": float(counts.std(ddof=1)), "n_sessions": int(len(counts))}


def disengagement_accuracy(df, disengagement_iti_threshold):
    """Accuracy on trials preceded by a long ITI (>= disengagement_iti_threshold),
    trial-level and per-animal-vs-chance. Ported from lines 5260-5263,
    5320-5327 (Fig 1F / STAT 2)."""
    d = df.sort_values(["Session_ID", "Trial"]).copy()
    d["Prev_StartTime"] = d.groupby("Session_ID")["AbsoluteTrialStartTime"].shift(1)
    d["ITI_calc"] = (d["AbsoluteTrialStartTime"] - d["Prev_StartTime"]) / 1000
    d = d[d["ITI_calc"].notna()]

    high_iti = d[d["ITI_calc"] > disengagement_iti_threshold]["Outcome"]
    high_iti_per_animal = d[d["ITI_calc"] > disengagement_iti_threshold].groupby("Animal_Name")["Outcome"].mean()
    _, p_val = scipy_stats.wilcoxon(high_iti_per_animal - 0.5)

    return {
        "n_trials": int(len(high_iti)),
        "n_animals": int(len(high_iti_per_animal)),
        "mean": float(high_iti.mean()),
        "sem": float(scipy_stats.sem(high_iti)),
        "wilcoxon_p_vs_chance_per_animal": float(p_val),
    }
