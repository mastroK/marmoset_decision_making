"""K-Means (Fig 6) and Gaussian-HMM (Supp Fig 4) validation of the
`classify_state_v3` state-classifier target, both on a held-out test split.

Ported from _source_archive/by_figure/Fig6/07b_KMEANS_Final.ipynb,
CHUNK 2 (K-Means, lines 4033-4155) and CHUNK 3 (HMM, lines 5964-6175).

Both methods here use the SAME 4-feature set (see config.yaml's
`state_taxonomy.features`), fixing the REVIEWER_AUDIT.md-documented
inconsistency where the source's K-Means used
['Signed_Deviation','Choice_Deviation','Rolling_Accuracy','Rolling_Switch_Rate']
but its HMM used ['Abs_Deviation','Choice_Deviation','Rolling_Accuracy',
'Rolling_Switch_Rate'] -- the same absolute-deviation quantity computed
twice under different names (Abs_Deviation == Choice_Deviation; see
src/features/state_features.py's docstring), with no signed feature.
"""

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from hmmlearn import hmm as hmmlearn


def hungarian_match(true_labels, cluster_labels, label_names, n_clusters):
    """Match cluster IDs to state names via the Hungarian algorithm. (lines 94-110)"""
    n = len(label_names)
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    true_idx = np.array([label_to_idx[l] for l in true_labels])
    cost = np.zeros((n, n_clusters))
    for i in range(n):
        for j in range(n_clusters):
            cost[i, j] = -np.sum((true_idx == i) & (cluster_labels == j))
    row_ind, col_ind = linear_sum_assignment(cost)
    mapping, matched = {}, set()
    for r, c in zip(row_ind, col_ind):
        mapping[c] = label_names[r]
        matched.add(c)
    for j in range(n_clusters):
        if j not in matched:
            mapping[j] = f"Cluster_{j}"
    return mapping


def train_test_split_sessions(df, test_size, seed):
    sessions_all = df["Session_ID"].unique()
    train_sessions, test_sessions = train_test_split(sessions_all, test_size=test_size, random_state=seed)
    return train_sessions, test_sessions


def run_kmeans_validation(df, target_col, classes, features, train_sessions, test_sessions,
                            k_range, k_best, seed, pca_n_components=3, pca_subsample_per_session=30):
    """Model selection (silhouette/BIC/AIC over k_range) + final K-Means fit
    at k_best, ARI on train and (honest) held-out test, plus a PCA
    projection of the test set for the publication figure's scatter panels.
    (lines 4061-4143 for the fit/ARI; lines 5362-5678 -- the source's later,
    final revision of the same "PUBLICATION FIGURE" cell, which resamples
    30 trials/session for the PCA scatter rather than CHUNK 2's original 40
    -- for the PCA projection/subsample.)
    """
    def _prepare(sessions):
        return (
            df.loc[df["Session_ID"].isin(sessions)]
            .loc[df[target_col] != "Unknown"]
            .dropna(subset=features + [target_col])
            .copy().reset_index(drop=True)
        )

    df_train = _prepare(train_sessions)
    df_test = _prepare(test_sessions)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_test = scaler.transform(df_test[features])

    sil, bic, aic = [], [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        kml = km.fit_predict(X_train)
        sil.append(float(silhouette_score(X_train, kml, sample_size=5000, random_state=seed)))
        gmm = GaussianMixture(n_components=k, random_state=seed, n_init=5, covariance_type="full")
        gmm.fit(X_train)
        bic.append(float(gmm.bic(X_train)))
        aic.append(float(gmm.aic(X_train)))

    best_k_sil = list(k_range)[int(np.argmax(sil))]
    best_k_bic = list(k_range)[int(np.argmin(bic))]

    km_final = KMeans(n_clusters=k_best, random_state=seed, n_init=20)
    km_final.fit(X_train)

    train_labels = km_final.predict(X_train)
    mapping = hungarian_match(df_train[target_col], train_labels, classes, k_best)
    df_train = df_train.copy()
    df_train["KMeans_State"] = pd.Series(train_labels).map(mapping).values
    ari_train = float(adjusted_rand_score(df_train[target_col], df_train["KMeans_State"]))

    test_labels = km_final.predict(X_test)
    df_test = df_test.copy()
    df_test["KMeans_State"] = pd.Series(test_labels).map(mapping).values
    ari_test = float(adjusted_rand_score(df_test[target_col], df_test["KMeans_State"]))

    profiles = pd.DataFrame(scaler.inverse_transform(km_final.cluster_centers_), columns=features)
    profiles.index = [mapping[i] for i in range(k_best)]
    profiles = profiles.reindex(classes)

    # PCA projection of the test set, for the publication figure's scatter panels
    pca = PCA(n_components=pca_n_components, random_state=seed)
    pca.fit(X_train)
    X_pca_test = pca.transform(X_test)
    df_test["PC1"] = X_pca_test[:, 0]
    df_test["PC2"] = X_pca_test[:, 1]
    df_test["PC3"] = X_pca_test[:, 2]
    df_plot = (
        df_test.groupby("Session_ID", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), pca_subsample_per_session), random_state=seed))
        .reset_index(drop=True).copy()
    )
    rng = np.random.default_rng(seed)
    for col in ("PC1", "PC2", "PC3"):
        df_plot[f"{col}_j"] = df_plot[col] + rng.normal(0, 0.03, len(df_plot))

    return {
        "k_range": list(k_range),
        "silhouette_by_k": sil,
        "bic_by_k": bic,
        "aic_by_k": aic,
        "best_k_silhouette": int(best_k_sil),
        "best_k_bic": int(best_k_bic),
        "k_used": int(k_best),
        "n_train_trials": int(len(df_train)),
        "n_test_trials": int(len(df_test)),
        "ari_train": ari_train,
        "ari_test": ari_test,
        "cluster_profiles": profiles.round(4).to_dict(orient="index"),
        "pca_explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        "pca_df_plot": df_plot[[target_col, "PC1_j", "PC2_j", "PC3_j"]],
        # Full (non-subsampled) per-trial test set with its K-Means cluster
        # assignment attached -- added so callers (e.g. Fig 6 panel a's
        # example-session trace) can pick a single session and show every
        # one of its trials colored by "KMeans_State", rather than only the
        # per-session-subsampled points used for the PCA scatter above.
        "df_test_classified": df_test,
    }


def run_hmm_validation(df, target_3_col, target_v3_col, features, train_sessions, test_sessions,
                         n_states, n_trials_drop, n_restarts, seed):
    """Gaussian-HMM validation, K-Means-initialized emission means, ARI on a
    held-out test split, plus transition matrix / dwell times.
    (lines 6002-6175)
    """
    def _prepare(sessions):
        out = (
            df.loc[df["Session_ID"].isin(sessions)]
            .loc[df[target_v3_col] != "Unknown"]
            .dropna(subset=features + [target_3_col, target_v3_col, "High_Prob_Is_Right"])
            .sort_values(["Session_ID", "Trial"])
            .copy().reset_index(drop=True)
        )
        out["_tis"] = out.groupby("Session_ID").cumcount()
        out = out[out["_tis"] >= n_trials_drop].drop("_tis", axis=1).reset_index(drop=True)
        return out

    df_train = _prepare(train_sessions)
    df_test = _prepare(test_sessions)

    train_lengths = df_train.groupby("Session_ID", sort=False).size().values
    test_lengths = df_test.groupby("Session_ID", sort=False).size().values

    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[features])
    X_test = scaler.transform(df_test[features])
    rng = np.random.default_rng(seed)
    X_train_fit = X_train + rng.normal(0, 0.01, X_train.shape)
    X_test_fit = X_test + rng.normal(0, 0.01, X_test.shape)

    classes_3 = sorted(df_train[target_3_col].unique())
    km3 = KMeans(n_clusters=n_states, random_state=seed, n_init=20)
    km3.fit(X_train)
    mapping_km3 = hungarian_match(df_train[target_3_col], km3.predict(X_train), classes_3, n_states)
    order_init = [k for k, v in sorted(mapping_km3.items(), key=lambda x: classes_3.index(x[1]))]
    init_means = km3.cluster_centers_[order_init]

    best_model, best_ll = None, -np.inf
    for restart in range(n_restarts):
        try:
            model = hmmlearn.GaussianHMM(
                n_components=n_states, covariance_type="diag", n_iter=200,
                random_state=seed + restart, verbose=False, init_params="stc", params="stmc",
            )
            model.means_ = init_means.copy()
            model.fit(X_train_fit, lengths=train_lengths)
            ll = model.score(X_train_fit, lengths=train_lengths)
            if ll > best_ll:
                best_ll, best_model = ll, model
        except ValueError:
            continue

    train_raw = best_model.predict(X_train_fit, lengths=train_lengths)
    mapping_train = hungarian_match(df_train[target_3_col], train_raw, classes_3, n_states)
    df_train = df_train.copy()
    df_train["HMM_3State"] = pd.Series(train_raw).map(mapping_train).values

    test_raw = best_model.predict(X_test_fit, lengths=test_lengths)
    df_test = df_test.copy()
    df_test["HMM_3State"] = pd.Series(test_raw).map(mapping_train).values

    for d in (df_train, df_test):
        d["HMM_4State"] = d.apply(
            lambda r: (
                "Left Bias" if r["HMM_3State"] == "Bias" and r["High_Prob_Is_Right"] == 1
                else "Right Bias" if r["HMM_3State"] == "Bias" and r["High_Prob_Is_Right"] == 0
                else r["HMM_3State"]
            ), axis=1,
        )

    ari_train_3 = float(adjusted_rand_score(df_train[target_3_col], df_train["HMM_3State"]))
    ari_test_3 = float(adjusted_rand_score(df_test[target_3_col], df_test["HMM_3State"]))
    ari_test_4 = float(adjusted_rand_score(df_test[target_v3_col], df_test["HMM_4State"]))

    state_order = [k for k, v in sorted(mapping_train.items(), key=lambda x: classes_3.index(x[1]))]
    trans_mat = best_model.transmat_[np.ix_(state_order, state_order)]
    stickiness = np.diag(trans_mat)
    mean_dwells = [1 / max(1 - s, 1e-6) for s in stickiness]
    means_orig = scaler.inverse_transform(best_model.means_[state_order])

    return {
        "classes_3": classes_3,
        "n_train_trials": int(len(df_train)),
        "n_test_trials": int(len(df_test)),
        "converged": bool(best_model.monitor_.converged),
        "train_loglik": float(best_ll),
        "ari_train_3state": ari_train_3,
        "ari_test_3state": ari_test_3,
        "ari_test_4state_posthoc_direction": ari_test_4,
        "transition_matrix": pd.DataFrame(trans_mat, index=classes_3, columns=classes_3).round(4).to_dict(orient="index"),
        "state_stickiness": {classes_3[i]: float(stickiness[i]) for i in range(n_states)},
        "mean_dwell_trials": {classes_3[i]: float(mean_dwells[i]) for i in range(n_states)},
        "emission_means": pd.DataFrame(means_orig, index=classes_3, columns=features).round(4).to_dict(orient="index"),
        "df_test_classified": df_test,
    }


def build_example_session(df_test_classified, session_col="Session_ID", animal_col=None, animal_name=None):
    """Pick a representative held-out test session and prepare it for a
    per-trial example-session trace (originally the HMM publication
    figure's Panel A, lines 6228-6347 of the source notebook). No
    selection rule is given in the source (which cell picks a specific
    session by inspection, not a documented criterion) -- the longest test
    session (most trials) is used here instead, a documented judgment
    call, since it best shows block structure and state changes over time.

    `animal_col`/`animal_name` (both optional, default None -- unchanged
    behavior) restrict the candidate sessions to one animal first, e.g.
    Fig 6 panel a's manuscript legend names a specific representative
    animal ("Knuckles"): pass `animal_col="Animal_Name",
    animal_name="Knuckles"` to pick that animal's own longest session
    instead of the longest session across all animals.
    """
    candidates = df_test_classified
    if animal_col is not None and animal_name is not None:
        candidates = candidates[candidates[animal_col] == animal_name]
    session_lengths = candidates.groupby(session_col).size()
    example_session_id = session_lengths.idxmax()
    df_ex = (
        df_test_classified[df_test_classified[session_col] == example_session_id]
        .sort_values("Trial").reset_index(drop=True)
    )
    df_ex["t"] = range(len(df_ex))
    return df_ex, example_session_id


def state_occupancy_by_animal(df_test_classified, classes, animal_col="Animal_Name", state_col="HMM_4State"):
    """Per-animal fraction (%) of TEST-set trials in each of `classes`.
    (lines 6404-6430, Panel E)
    """
    records = []
    for animal, adf in df_test_classified.groupby(animal_col):
        counts = adf[state_col].value_counts(normalize=True)
        for state in classes:
            records.append({"Animal": animal, "State": state, "Occupancy": float(counts.get(state, 0.0) * 100)})
    return pd.DataFrame(records)
