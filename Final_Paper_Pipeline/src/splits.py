"""Reviewer-response addition (RA2_validation_statistics.ipynb, R2-Maj1):
session-level k-fold cross-validation, stratified by animal, replacing the
manuscript's/Fig 6's single static 80/20 session holdout
(`state_clustering.train_test_split_sessions`) for the K-Means/HMM ARI
validation.

WHY THIS EXISTS (forensic finding, see RA2's own intro cell and
REVIEWER_AUDIT.md Audit 4 for the full account): the original notebook
(`07b_KMEANS_Final.ipynb`) computed its train/held-out session split fresh,
in-memory, via `sklearn.train_test_split(session_ids, test_size=0.2,
random_state=42)` every run -- never persisted to disk, not stratified by
animal, never k-fold. This module is a disclosed methodological upgrade
over that single-split design, not a reproduction of it -- RA2's manifest
says so explicitly.

`session` (not trial) is the atomic unit being split, exactly as the
original design intended ("held-out sessions") -- this is a plain
`StratifiedKFold` over the array of unique Session_IDs, stratified by each
session's own Animal_Name, NOT `StratifiedGroupKFold` (that tool is for
when a group spans multiple rows that must stay together across a
separate splitting unit; here the session already IS the splitting unit).
"""

import json

import numpy as np
from sklearn.model_selection import StratifiedKFold


def stratified_session_kfold(df, k, seed, session_col="Session_ID", animal_col="Animal_Name"):
    """k-fold split of unique sessions, stratified by animal, so every
    fold's held-out set contains sessions from every animal (not just
    the animals with the most sessions). Returns a list of k
    (train_sessions, test_sessions) tuples (arrays of Session_ID).
    """
    session_animal = (
        df[[session_col, animal_col]].drop_duplicates(subset=session_col).reset_index(drop=True)
    )
    sessions = session_animal[session_col].values
    animals = session_animal[animal_col].values

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(sessions, animals):
        folds.append((sessions[train_idx], sessions[test_idx]))
    return folds


def save_fold_assignments(folds, path):
    """Persist fold assignments to JSON (config/splits.json) -- unlike the
    original ephemeral in-memory split, this one is a durable, auditable
    artifact: {"fold_0": {"train": [...], "test": [...]}, ...}.
    """
    payload = {
        f"fold_{i}": {"train": list(map(str, train)), "test": list(map(str, test))}
        for i, (train, test) in enumerate(folds)
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return payload


def load_fold_assignments(path):
    """Load fold assignments previously saved by `save_fold_assignments`,
    returning the same list-of-tuples-of-numpy-arrays shape
    `stratified_session_kfold` returns, for exact reproducibility of a
    prior run."""
    with open(path) as f:
        payload = json.load(f)
    folds = []
    for i in range(len(payload)):
        entry = payload[f"fold_{i}"]
        folds.append((np.array(entry["train"]), np.array(entry["test"])))
    return folds
