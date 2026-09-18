"""The behavioral-state classifiers used for Fig 6 / Fig 7 / Fig 8 / Supp Fig 4.

Per REVIEWER_AUDIT.md, at least 9 divergent rule-based classifier variants
exist across the project's notebooks. Most (6, in 06_Fig3_2ABT_v2.ipynb) are
confirmed exploratory dead ends whose output is never referenced elsewhere.
The ones actually used for a published figure are kept here as clearly-named
functions -- per the user's explicit clarification, these are NOT the same
classifier with a bug to unify: the manuscript uses one classifier without
an Adapting state at all (Fig 6 / Supp Fig 4's clustering-validation
target), an intermediate one with a single undifferentiated Adapting state
(the now-superseded Fig 8 build), and the manuscript's own canonical 7-state
taxonomy with a fast/slow Adaptation split (Fig 7, and Fig 8's rebuild) --
each kept as its own function, fixed for its own audited defects:

- `classify_state_v3` (Fig 6 / Supp Fig 4): ported unchanged from
  07b_KMEANS_Final.ipynb, lines 159-200 -- a 4-state classifier (Engaged,
  Exploration, Left Bias, Right Bias) used only as the internal-consistency
  target label for the K-Means/HMM validation exercise, not as a claim of
  ground truth. No if-ordering bug here (Random/Directed Exploration are
  already merged into one `Exploration` fallback, so there is no ordering
  to get wrong).

- `classify_state_updated` (kept for reference only -- no current notebook
  applies this function; Fig 6 / Supp Fig 4 use `classify_state_v3`, Fig 7 /
  Fig 8 use `classify_state_v7` below): ported from
  09b_CrossProb_StickyModel_v1.ipynb, lines ~1766-1795.

  STATUS (2026-09-13, REVISED): this function's `Random Exploration`/
  `Directed Exploration` branch order now matches the SOURCE exactly
  (`Random Exploration`, acc < 0.4, checked BEFORE `Directed Exploration`,
  switch >= 0.2 and acc <= 0.6) -- previously this pipeline swapped that
  order per REVIEWER_AUDIT.md Audit 1's recommendation (that audit found
  the source's own ordering "silently reassigned 99.86% of trials that
  should have been Random Exploration into Directed Exploration"). That
  swap has been REVERTED here, because it turned out to be the wrong fix
  for this project's purposes: verified directly against
  09b_CrossProb_StickyModel_v1.ipynb's own recorded "State distribution
  (80-20)" output comment (lines ~1845-1852), the ORIGINAL order --
  combined with a separate, genuine bug fix to `Rolling_Accuracy` itself
  (see `state_features.py`'s docstring: it must be reward-based, not
  choice-based) -- reproduces that recorded output EXACTLY, trial for
  trial, including a real, non-empty Random Exploration category (1,455 /
  36,673 trials, 4.0%) matching the manuscript's own reported per-state
  statistics for it. Swapping to Directed-before-Random (REVIEWER_AUDIT's
  recommended fix) collapses Random Exploration to 2 trials regardless of
  the accuracy-column fix, because `Directed Exploration`'s own condition
  turns out to almost always co-occur with `acc < 0.4` in this data.
  REVIEWER_AUDIT.md's underlying concern (this ordering is not a
  principled way to partition the two categories, since their conditions
  overlap) remains a legitimate, open methodological question -- the user
  has confirmed this is being discussed with a co-author (Bernardo) and
  is NOT settled by this pipeline; this is the version that reproduces
  the manuscript's actual analysis and the user's own working definition
  ("Random Exploration is the subset of exploration that is not
  Q-aligned" -- confirmed true of this classification: Fig 7's own
  Q-alignment statistic for Random Exploration comes out near zero). If
  that discussion concludes the audited ordering should be used instead,
  reverting is a one-line change (swap the two `if` blocks back) -- doing
  so will again make Random Exploration collapse to near-zero, at which
  point (per the user's own words) "all trials are simply put into a
  bucket of exploration, which is also fine."

- `classify_state_v7` / `compute_adaptation_labels` / `label_block_adaptation_speed`
  (Fig 7, Fig 8 rebuild): the manuscript's own canonical 7-state taxonomy
  (Exploitation, Directed Exploration, Random Exploration, Left Spatial
  Bias, Right Spatial Bias, Fast Adaptation, Slow Adaptation) -- see below.
  Its Directed/Random Exploration branch order and the reward-based
  `Rolling_Accuracy` fix are the same provisional decision described above
  for `classify_state_updated`, and this is the function that actually
  matters for current figure output (`classify_state_updated` itself is
  not called by any notebook any more).
"""

import numpy as np
import pandas as pd


def classify_state_v3(row, config):
    """4-state classifier: Engaged, Exploration, Left Bias, Right Bias.
    Used only as Fig 6 / Supp Fig 4's clustering-validation target label.
    """
    acc = row["Rolling_Accuracy"]
    p_right = row["Rolling_PRight"]
    exp_pr = row["Expected_PRight"]
    hpr = row["High_Prob_Is_Right"]
    switch = row["Rolling_Switch_Rate"]

    if pd.isna(acc) or pd.isna(p_right):
        return "Unknown"

    abs_dev = abs(p_right - exp_pr)
    if abs_dev >= config["bias_abs_deviation_threshold"]:
        if hpr == 1 and p_right <= config["bias_prright_threshold_low"]:
            return "Left Bias"
        if hpr == 0 and p_right >= config["bias_prright_threshold_high"]:
            return "Right Bias"

    if acc >= config["engaged_accuracy_threshold"] and switch <= config["engaged_switch_threshold"]:
        return "Engaged"

    return "Exploration"


def classify_state_updated(row, config):
    """7-state classifier (Adapting, Directed Exploration, Random Exploration,
    Left Bias, Right Bias, Exploit, Unknown), ported from
    09b_CrossProb_StickyModel_v1.ipynb lines 1766-1795. Matches the source's
    OWN branch order exactly: Random Exploration (acc < 0.4) is checked
    BEFORE Directed Exploration (switch >= 0.2 and acc <= 0.6) -- see this
    module's own docstring ("STATUS (2026-09-13, REVISED)") for why this
    was reverted back to the source's order after this pipeline had
    previously swapped it per REVIEWER_AUDIT.md Audit 1/5's recommendation:
    the swap is not what actually reproduces the manuscript's own recorded
    state distribution or reported Random Exploration statistics, and the
    ordering question itself remains an open, disclosed methodological
    question, not resolved here. Bias branch logic (lines 1780-1784) is
    unchanged: any trial with deviation above threshold is unconditionally
    Left or Right Bias, it does not fall through to Exploration/Exploit.
    """
    acc = row["Rolling_Accuracy"]
    dev = row["Choice_Deviation"]
    switch_rate = row["Rolling_Switch_Rate"]
    trials_since = row["Trials_Since_Transition"]

    if pd.isna(acc) or pd.isna(dev):
        return "Unknown"

    if trials_since <= config["adapting_trials_since_transition"]:
        return "Adapting"

    if dev > config["bias_deviation_threshold"]:
        if row["High_Prob_Is_Right"]:
            return "Left Bias" if row["Rolling_PRight"] < config["bias_prright_threshold_low"] else "Right Bias"
        else:
            return "Right Bias" if row["Rolling_PRight"] > config["bias_prright_threshold_high"] else "Left Bias"

    # Source order (see docstring): Random Exploration checked first.
    if acc < config["random_exploration_accuracy_threshold"]:
        return "Random Exploration"
    if switch_rate >= config["directed_exploration_switch_threshold"] and acc <= config["directed_exploration_accuracy_threshold"]:
        return "Directed Exploration"

    return "Exploit"


# ---------------------------------------------------------------------------
# Part 1 -- the canonical 7-state taxonomy (Fig 7; Fig 8 rebuild).
#
# Manuscript Results text ("A behavioral state framework reveals distinct
# modes of reward-guided decision making in marmosets", pages 10-11 of
# 2026.06.30.734629v1.full.pdf), quoted directly:
#
#   "...the exploration cluster was further subdivided into Directed and
#   Random exploration based on combinations of rolling accuracy and switch
#   rate... Post-reversal trials, defined as the first five trials following
#   a contingency change, were classified as a period of 'Adaptation'...
#   The speed of adaptation was based on whether the animal achieved greater
#   than chance accuracy during this period (fast: accuracy >= 0.6; slow:
#   accuracy < 0.6), yielding seven states in total: Exploitation, Directed
#   Exploration, Random Exploration, Left Spatial Bias, Right Spatial Bias,
#   Fast Adaptation, and Slow Adaptation."
#
# CRITICAL DESIGN POINT (this is the one place this taxonomy structurally
# differs from `classify_state_updated` above, and from the near-verbatim
# `classify_state_updated`/`classify_state_v5` copies found in the orphaned
# `07c_FinalClassifier_Fig6.ipynb` -- see REVIEWER_AUDIT.md's account of
# that notebook and its `git show e7607e9` recovery): the fast/slow split is
# a BLOCK-TRANSITION-LEVEL property, computed once per block transition from
# that transition's own first-`adaptation_window`-trials accuracy, not a
# per-trial rule. Every trial in that same window inherits the SAME
# fast/slow label. `classify_state_v5`/`classify_state_updated` in the
# orphaned notebook instead re-evaluate `Rolling_Accuracy` (a trailing
# rolling window that can itself drift trial-to-trial within the same
# 5-trial post-reversal window, and that -- because it is only grouped by
# Session_ID, not reset at block boundaries -- can be contaminated by
# trials from the PRECEDING block) independently on every trial with
# `Trials_Since_Transition <= 5`, which can silently split a single
# 5-trial adaptation window across both fast AND slow labels. The
# manuscript's own Results text ("classified as a period of 'Adaptation'...
# The speed of adaptation was based on whether the animal achieved greater
# than chance accuracy DURING THIS PERIOD") describes one property of one
# period, corroborating the block-level design used here over the
# per-trial one used everywhere else in this project's source notebooks.
#
# Implementation is a two-pass design:
#   1. `label_block_adaptation_speed` -- one row per block transition,
#      giving that block's own within-window accuracy and fast/slow label.
#   2. `compute_adaptation_labels` -- broadcasts each block's label back
#      onto every trial in that block's adaptation window, returning a
#      pd.Series aligned to the input dataframe's index.
# `classify_state_v7` then classifies every remaining (non-Adaptation)
# trial with the SAME Directed/Random Exploration + Left/Right Bias +
# Exploit branch logic as `classify_state_updated` above (thresholds and
# if-ordering unchanged, including the Directed-before-Random fix from
# REVIEWER_AUDIT.md Audit 1) -- only the Adapting branch differs.
#
# Judgment call, disclosed: a block is only eligible for a fast/slow label
# if it was entered via a genuine, detected block transition
# (`Block_Transition == 1` on its first trial). Each session's own opening
# block trivially has `Trials_Since_Transition` counting up from 0 as well
# (see state_features.py / cross_condition.py's shared counting logic), but
# was never "following a contingency change" -- the manuscript's own
# definition of the Adaptation period. Trials in a session's opening block
# are therefore classified by the ordinary Exploit/Exploration/Bias rules
# instead of forced into Adaptation. `classify_state_updated` above does
# not make this distinction (its per-trial `trials_since <= 5` check fires
# identically for a session's opening block); this is a deliberate
# improvement made only in this new function, not a claim that the two are
# otherwise equivalent.
# ---------------------------------------------------------------------------

def label_block_adaptation_speed(df, adaptation_window, fast_accuracy_threshold):
    """One row per block transition (Session_ID, BlockCount) that was
    entered via a genuine, detected block change (`Block_Transition == 1`
    on the block's first trial; a session's own opening block is excluded,
    see module docstring). For each such block, computes accuracy
    (mean of `Outcome_Binary` -- reward-based, i.e. was the trial actually
    rewarded, NOT `High_Prob`/choice-based -- see the CORRECTED note in
    `state_features.py`'s module docstring for why: 07c_FinalClassifier_Fig6.ipynb's
    own inline fast/slow split applies this same 0.6 threshold to its
    reward-based `Rolling_Accuracy` column, not a separately-computed
    choice-based average) over the first `adaptation_window` trials of
    that block (`Trials_Since_Transition` 0 .. adaptation_window-1), and
    labels the block 'fast' if that accuracy is >= `fast_accuracy_threshold`
    else 'slow'. Requires Session_ID, BlockCount, Animal_Name,
    Block_Transition, Trials_Since_Transition, Outcome_Binary already on
    `df` (state_features.build_state_features_df / cross_condition's 100-0
    reconstruction both provide these).
    """
    rows = []
    for (session_id, block_count), block_df in df.groupby(
        ["Session_ID", "BlockCount"], sort=False, observed=True
    ):
        if block_df["Block_Transition"].iloc[0] != 1:
            continue
        window = block_df[block_df["Trials_Since_Transition"] < adaptation_window]
        if window.empty:
            continue
        acc = float(window["Outcome_Binary"].mean())
        rows.append({
            "Session_ID": session_id,
            "BlockCount": block_count,
            "Animal_Name": block_df["Animal_Name"].iloc[0],
            "adaptation_accuracy": acc,
            "adaptation_speed": "fast" if acc >= fast_accuracy_threshold else "slow",
            "n_window_trials": int(len(window)),
        })
    return pd.DataFrame(rows, columns=[
        "Session_ID", "BlockCount", "Animal_Name", "adaptation_accuracy",
        "adaptation_speed", "n_window_trials",
    ])


def compute_adaptation_labels(df, adaptation_window, fast_accuracy_threshold, block_labels=None):
    """Broadcast each block's fast/slow label (from
    `label_block_adaptation_speed`) onto every trial in that block's
    adaptation window. Returns a pd.Series aligned to `df.index`:
    'Adapting (fast)' / 'Adapting (slow)' for every trial with
    `Trials_Since_Transition < adaptation_window` in an eligible block
    (see module docstring), NaN everywhere else (including a session's own
    opening block).

    `block_labels` can be passed in (the output of `label_block_adaptation_speed`
    on this same df) to avoid recomputing it when a caller also needs the
    block-level table directly (e.g. Fig 7 panel c).
    """
    if block_labels is None:
        block_labels = label_block_adaptation_speed(df, adaptation_window, fast_accuracy_threshold)

    speed_by_block = {
        (r.Session_ID, r.BlockCount): r.adaptation_speed for r in block_labels.itertuples()
    }
    labels = pd.Series(np.nan, index=df.index, dtype=object)
    for (session_id, block_count), block_df in df.groupby(
        ["Session_ID", "BlockCount"], sort=False, observed=True
    ):
        speed = speed_by_block.get((session_id, block_count))
        if speed is None:
            continue
        window = block_df[block_df["Trials_Since_Transition"] < adaptation_window]
        labels.loc[window.index] = "Adapting (fast)" if speed == "fast" else "Adapting (slow)"
    return labels


def classify_state_v7(row, adaptation_labels, config):
    """The manuscript's canonical 7-state taxonomy: Exploit ("Exploitation"),
    Directed Exploration, Random Exploration, Left Bias ("Left Spatial
    Bias"), Right Bias ("Right Spatial Bias"), Adapting (fast) ("Fast
    Adaptation"), Adapting (slow) ("Slow Adaptation") -- display-name
    parentheticals per the manuscript's own wording, used for figure labels
    only; the internal state strings match this project's existing
    Left Bias/Right Bias/Exploit naming convention for consistency with
    `classify_state_updated`/`cross_condition.py`.

    Must be called with `adaptation_labels` = the full-dataframe Series
    from `compute_adaptation_labels` (computed ONCE per dataframe, not
    per-row) -- rows with a non-null adaptation label there short-circuit
    here before any of the Directed/Random Exploration or Bias thresholds
    are evaluated. `config` is `state_taxonomy.classify_updated` (same
    thresholds `classify_state_updated` uses, reused unchanged here -- see
    module docstring: only the Adapting branch differs in this function).

    Typical use, given `adaptation_labels = compute_adaptation_labels(df, ...)`:
        classify = functools.partial(classify_state_v7, adaptation_labels=adaptation_labels, config=cfg)
        df["Behavioral_State"] = df.apply(classify, axis=1)
    """
    label = adaptation_labels.get(row.name)
    if isinstance(label, str):
        return label

    acc = row["Rolling_Accuracy"]
    dev = row["Choice_Deviation"]
    switch_rate = row["Rolling_Switch_Rate"]

    if pd.isna(acc) or pd.isna(dev):
        return "Unknown"

    if dev > config["bias_deviation_threshold"]:
        if row["High_Prob_Is_Right"]:
            return "Left Bias" if row["Rolling_PRight"] < config["bias_prright_threshold_low"] else "Right Bias"
        else:
            return "Right Bias" if row["Rolling_PRight"] > config["bias_prright_threshold_high"] else "Left Bias"

    # Source order (matches classify_state_updated and 09b/07c's own code):
    # Random Exploration checked BEFORE Directed Exploration.
    #
    # HISTORY (2026-09-13) -- two issues were found and fixed together;
    # see `state_classifier.py`'s module docstring ("STATUS ... REVISED")
    # and `state_features.py`'s docstring for the full account:
    #   1. `Rolling_Accuracy` (`acc` here) was CHOICE-based (P(chose the
    #      high-probability side)), which is tightly algebraically coupled
    #      to `Choice_Deviation` -- they become the same event whenever
    #      `High_Prob_Is_Right` is stable, so the Bias branch above
    #      (checked first) intercepted every trial that could ever qualify
    #      as Random Exploration. Fixed: `Rolling_Accuracy` is now
    #      REWARD-based (`Outcome_Binary`), matching
    #      `09b_CrossProb_StickyModel_v1.ipynb`/`07c_FinalClassifier_Fig6.ipynb`'s
    #      own STEP 7B ("keep as reward-based, thresholds were tuned to
    #      this").
    #   2. This function had ALSO swapped Directed-before-Random per
    #      REVIEWER_AUDIT.md Audit 1's recommendation. That swap is reverted
    #      here: combined with fix #1, the ORIGINAL order reproduces 09b's
    #      own recorded 80-20 state distribution exactly (Random Exploration
    #      1,455 / 36,673 trials, 4.0%); the swapped order collapses it to
    #      2 trials regardless of fix #1, because Directed Exploration's own
    #      condition almost always co-occurs with `acc < 0.4` in this data.
    # REVIEWER_AUDIT.md's concern that this ordering is not a principled way
    # to partition the two categories (their conditions overlap) remains a
    # legitimate, OPEN methodological question -- not resolved here, and
    # under discussion with a co-author. This is the version that (a)
    # reproduces the manuscript's actual recorded analysis and (b) matches
    # the user's own working definition of Random Exploration as "the
    # subset of exploration that is not Q-aligned" (confirmed: Fig 7's own
    # Q-alignment statistic for this state comes out near zero).
    if acc < config["random_exploration_accuracy_threshold"]:
        return "Random Exploration"
    if switch_rate >= config["directed_exploration_switch_threshold"] and acc <= config["directed_exploration_accuracy_threshold"]:
        return "Directed Exploration"

    return "Exploit"


# Display names for figure axes/legends -- manuscript wording ("Exploitation",
# "Fast Adaptation", "Slow Adaptation") vs. this codebase's internal state
# strings ("Exploit", "Adapting (fast)", "Adapting (slow)"). Left Bias/Right
# Bias/Directed Exploration/Random Exploration are used verbatim in both.
STATE_DISPLAY_NAMES = {
    "Exploit": "Exploitation",
    "Adapting (fast)": "Fast Adaptation",
    "Adapting (slow)": "Slow Adaptation",
    "Directed Exploration": "Directed Exploration",
    "Random Exploration": "Random Exploration",
    "Left Bias": "Left Bias",
    "Right Bias": "Right Bias",
    "Unknown": "Unknown",
}

STATES_V7_ALL = [
    "Exploit", "Adapting (fast)", "Adapting (slow)",
    "Directed Exploration", "Random Exploration", "Left Bias", "Right Bias",
]
STATES_V7_NON_ADAPTING = [
    "Exploit", "Directed Exploration", "Random Exploration", "Left Bias", "Right Bias",
]


# ---------------------------------------------------------------------------
# Reviewer-response addition (RA1_exploration_states.ipynb, R1-C1/R2-1/R1-s):
# the reordered Directed-before-Random variant, provided as its own function
# (not a config flag on classify_state_v7) so both orderings can be run
# side by side and cross-tabulated without touching the taxonomy actually
# used by Fig 7/Fig 8. Per this module's own docstring above ("STATUS ...
# REVISED"), this is the literal one-line swap described there -- kept here
# rather than re-derived, so RA1's reclassification-impact analysis is
# comparing against the exact alternative this pipeline already considered
# and rejected, not a fresh reimplementation of it.
# ---------------------------------------------------------------------------

def classify_state_v7_reordered(row, adaptation_labels, config):
    """Same as `classify_state_v7`, except the Directed Exploration check
    is evaluated BEFORE the Random Exploration check (the ordering
    REVIEWER_AUDIT.md Audit 1/5 originally recommended). Everything else
    (Adapting pre-pass, Bias branch, thresholds) is identical."""
    label = adaptation_labels.get(row.name)
    if isinstance(label, str):
        return label

    acc = row["Rolling_Accuracy"]
    dev = row["Choice_Deviation"]
    switch_rate = row["Rolling_Switch_Rate"]

    if pd.isna(acc) or pd.isna(dev):
        return "Unknown"

    if dev > config["bias_deviation_threshold"]:
        if row["High_Prob_Is_Right"]:
            return "Left Bias" if row["Rolling_PRight"] < config["bias_prright_threshold_low"] else "Right Bias"
        else:
            return "Right Bias" if row["Rolling_PRight"] > config["bias_prright_threshold_high"] else "Left Bias"

    # Reordered: Directed Exploration checked BEFORE Random Exploration.
    if switch_rate >= config["directed_exploration_switch_threshold"] and acc <= config["directed_exploration_accuracy_threshold"]:
        return "Directed Exploration"
    if acc < config["random_exploration_accuracy_threshold"]:
        return "Random Exploration"

    return "Exploit"


def reclassification_crosstab(df, col_a, col_b):
    """Generic before/after reclassification-impact summary between two
    state-label columns on the same dataframe (e.g. `Behavioral_State`
    from `classify_state_v7` vs. `classify_state_v7_reordered`).
    Generalizes the one-off `audit_scratch/reclassification_impact.py`
    script (REVIEWER_AUDIT.md Audit 5) into reusable src code, so it can be
    re-run on the corrected (reward-based Rolling_Accuracy) classification
    rather than only the original audit's pre-fix numbers.

    Returns (crosstab_df, moved_pct, per_state_moved_pct):
      - crosstab_df: full col_a x col_b contingency table.
      - moved_pct: % of all trials whose label differs between col_a/col_b.
      - per_state_moved_pct: for each col_a state, % of its trials that
        changed label under col_b (e.g. "99.86% of Random Exploration
        trials moved to Directed Exploration").
    """
    crosstab = pd.crosstab(df[col_a], df[col_b])
    moved = df[col_a] != df[col_b]
    moved_pct = float(moved.mean() * 100)

    per_state = {}
    for state, sdf in df.groupby(col_a, observed=True):
        state_moved = (sdf[col_a] != sdf[col_b]).mean()
        per_state[state] = {
            "n_trials": int(len(sdf)),
            "pct_moved": float(state_moved * 100),
            "destination_counts": sdf.loc[sdf[col_a] != sdf[col_b], col_b].value_counts().to_dict(),
        }
    return crosstab, moved_pct, per_state
