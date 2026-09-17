"""Inter-trial interval (ITI) distributions and performance vs. ITI (Fig 1
stage 2: panels e-f).

Ported from _source_archive/by_figure/Fig1/04_LocationBasedShifting.ipynb,
"CELL 11: Inter-Trial Interval (ITI) Analysis" (lines 1120-1385).

NOTE: an earlier, abandoned ITI calculation exists at lines 935-951 of the
source notebook, using a nonsensical MAX_ITI (6e15) left over from an
earlier draft -- superseded by CELL 11's version (MAX_ITI=1000 seconds) and
not ported here.
"""

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


def compute_iti(df):
    """Per-trial ITI (seconds) = this trial's start time minus the previous
    trial's start time, within each session. First trial of each session has
    no ITI and is dropped. Ported from lines 1126-1135."""
    d = df.sort_values(["Session_ID", "Trial"]).reset_index(drop=True).copy()
    d["Prev_StartTime"] = d.groupby("Session_ID")["AbsoluteTrialStartTime"].shift(1)
    d["ITI"] = (d["AbsoluteTrialStartTime"] - d["Prev_StartTime"]) / 1000
    return d.dropna(subset=["ITI"])


def filter_iti(df_timing, max_iti):
    """Drop implausible ITIs above max_iti seconds. Ported from lines 1137-1139."""
    return df_timing[df_timing["ITI"] <= max_iti].copy()


def iti_summary_by_animal(df_timing_filtered, animal_order):
    """Median/mean/std ITI per animal. Ported from lines 1229-1239."""
    from ..features.history_encoding import sort_animals

    records = []
    for animal in sort_animals(df_timing_filtered["Animal_Name"].unique(), animal_order):
        d = df_timing_filtered[df_timing_filtered["Animal_Name"] == animal]
        records.append({
            "animal": animal,
            "median_iti": d["ITI"].median(),
            "mean_iti": d["ITI"].mean(),
            "std_iti": d["ITI"].std(),
        })
    return pd.DataFrame(records)


def performance_by_iti_bin(df_timing_filtered, iti_bins, iti_labels, outcome_col="Outcome"):
    """Accuracy binned by ITI duration. Ported from lines 1271-1281."""
    d = df_timing_filtered.copy()
    d["ITI_bin"] = pd.cut(d["ITI"], bins=iti_bins, labels=iti_labels)
    perf = d.groupby("ITI_bin", observed=True)[outcome_col].agg(["mean", "sem", "count"]).reset_index()
    perf.columns = ["ITI_bin", "accuracy", "sem", "n"]
    return perf


def performance_by_iti_bin_by_animal(df_timing_filtered, iti_bins, iti_labels, outcome_col="Outcome"):
    """Per-animal accuracy binned by ITI duration -- an addition (not present
    in the original stage-2 port) needed to reproduce the source notebook's
    own publication figure for this panel, which overlays a per-animal line
    on top of the group-mean bars. Ported from the standalone "Performance
    vs ITI - Standalone with Individual Animals" cell (In[79], lines
    1446-1450: `animal_iti_performance`), which is the same per-ITI-bin
    accuracy already computed by `performance_by_iti_bin` above, just grouped
    by animal as well. Does not change any existing field/value -- purely
    additive."""
    d = df_timing_filtered.copy()
    d["ITI_bin"] = pd.cut(d["ITI"], bins=iti_bins, labels=iti_labels)
    perf = d.groupby(["Animal_Name", "ITI_bin"], observed=True)[outcome_col].agg(["mean", "count"]).reset_index()
    perf.columns = ["animal", "ITI_bin", "accuracy", "n"]
    return perf


def performance_by_long_break(df_timing_filtered, long_break_threshold, outcome_col="Outcome"):
    """Accuracy on trials following a "long break" (ITI above threshold) vs.
    a short one. Ported from lines 1316-1348.

    The source notebook's t-test compares raw trial-level Outcome values
    (short-ITI trials vs. long-ITI trials) -- these are not independent
    observations (all trials come from ~10 animals), so alongside that
    trial-level test we also aggregate to one short-break mean and one
    long-break mean per animal and run a paired Wilcoxon signed-rank test
    on those, matching the per-animal correction already applied to the
    other pseudoreplication-prone tests in this pipeline (Fig1 stage 1's
    session-bin test, Fig2's late-accuracy test).
    """
    d = df_timing_filtered.copy()
    d["Long_Break"] = d["ITI"] > long_break_threshold

    trial_level = d.groupby("Long_Break")[outcome_col].agg(["mean", "sem", "count"]).reset_index()
    trial_level.columns = ["Long_Break", "accuracy", "sem", "n"]

    short = d[~d["Long_Break"]][outcome_col]
    long = d[d["Long_Break"]][outcome_col]
    t_stat, p_trial_level = scipy_stats.ttest_ind(short, long)

    per_animal = d.groupby(["Animal_Name", "Long_Break"])[outcome_col].mean().unstack()
    per_animal = per_animal.dropna()
    if per_animal.shape[0] > 1 and per_animal.shape[1] == 2:
        w_stat, p_per_animal = scipy_stats.wilcoxon(per_animal[False], per_animal[True])
    else:
        w_stat, p_per_animal = np.nan, np.nan

    return {
        "trial_level": trial_level,
        "t_stat_trial_level": t_stat,
        "p_trial_level": p_trial_level,
        "per_animal_short_mean": per_animal[False].mean() if per_animal.shape[0] else np.nan,
        "per_animal_long_mean": per_animal[True].mean() if per_animal.shape[0] else np.nan,
        "n_animals_paired": int(per_animal.shape[0]),
        "wilcoxon_stat_per_animal": w_stat,
        "p_per_animal": p_per_animal,
    }
