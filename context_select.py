"""Nearest-neighbour TabPFN contexts, shared by context_select_test.py and run_nn_full.py.

The distance space: standardised features (one-hot categoricals, log1p delays, "not applicable" ratings
as 0), each weighted by the square root of its share of LightGBM gain importance on the reference rows.
A cluster's context is every member's nearest reference row, then every member's 2nd nearest, ... until
it is full, so the cluster's edges (where the hard rows are) are covered as well as its centre.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from preprocess import ARR_DELAY, CAT_COLS, DEP_DELAY

N_NEIGHBOURS = 128


def distance_space(X, y, ref_idx, weight=None):
    """All rows of X (train + test stacked is fine) as vectors; scaling and weights come from ref_idx
    (labels y for those rows), unless the weights are given."""
    if weight is None:
        model = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, num_threads=8,
                                   importance_type="gain", verbose=-1)
        model.fit(X.iloc[ref_idx], y[ref_idx])
        weight = np.sqrt(model.feature_importances_ / model.feature_importances_.sum())
    parts = []
    for col, w in zip(X.columns, weight):
        if col in CAT_COLS:
            parts.append(pd.get_dummies(X[col]).to_numpy(np.float32) * w / np.sqrt(2))
        else:
            v = X[col].fillna(0).to_numpy(np.float64)  # ratings: 0 = "not applicable"
            if col in (DEP_DELAY, ARR_DELAY):
                v = np.log1p(v)
            parts.append(((v - v[ref_idx].mean()) / v[ref_idx].std() * w)[:, None].astype(np.float32))
    return np.hstack(parts), weight


def nearest(D, query_idx, ref_idx, k=N_NEIGHBOURS, batch=256):
    """Positions (within ref_idx) of each query row's k nearest reference rows, nearest first."""
    T = D[ref_idx]
    t_sq = (T ** 2).sum(axis=1)
    out = np.empty((len(query_idx), k), dtype=np.int32)
    for i in range(0, len(query_idx), batch):
        dist = t_sq[None, :] - 2 * D[query_idx[i:i + batch]] @ T.T  # the query's own |q|^2 doesn't change ranking
        top = np.argpartition(dist, k, axis=1)[:, :k]
        out[i:i + batch] = np.take_along_axis(top, np.argsort(np.take_along_axis(dist, top, 1), 1), 1)
    return out


def knn_contexts(D, neighbours, query_idx, ref_idx, n_clusters, size, seed):
    """[(context rows, query rows it predicts)]: k-means clusters of the query rows, each with `size`
    reference rows taken from its members' neighbour lists."""
    km = KMeans(n_clusters, n_init=4, random_state=seed).fit(D[query_idx])
    out = []
    for c in range(n_clusters):
        members = km.labels_ == c
        order = neighbours[members].T.ravel()  # all 1st neighbours, then all 2nd, ...
        _, first = np.unique(order, return_index=True)
        ctx = order[np.sort(first)][:size]
        if len(ctx) < size:  # not enough distinct neighbours: top up with the rows nearest the centre
            dist = ((D[ref_idx] - km.cluster_centers_[c]) ** 2).sum(axis=1)
            extra = np.setdiff1d(np.argsort(dist), ctx, assume_unique=True)
            ctx = np.concatenate([ctx, extra[:size - len(ctx)]])
        out.append((ref_idx[ctx], query_idx[members]))
    return out


def cached_space(path, X, y, ref_idx, query_idx):
    """(distance space, query rows' neighbours), saved to `path` (.npz) the first time."""
    if path.exists():
        f = np.load(path)
        return distance_space(X, y, ref_idx, weight=f["weight"])[0], f["neighbours"]
    D, weight = distance_space(X, y, ref_idx)
    neighbours = nearest(D, query_idx, ref_idx)
    np.savez(path, weight=weight, neighbours=neighbours)
    return D, neighbours
