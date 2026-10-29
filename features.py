"""Named feature sets for compare_fe.py: name -> (load_data kwargs, function adding columns or None).

All extra features are built without the target, from train + test together, so they can't leak labels.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import CAT_COLS, RATING_COLS, load_original

GROUP = ["Class", "Type of Travel"]  # the passenger segment used for interactions / expectation gaps


def _both(X, X_test, make):
    """Apply make(frame) to train and test concatenated, so statistics use all rows; split back."""
    allx = pd.concat([X, X_test], keys=["train", "test"])
    new = make(allx)
    return (pd.concat([X, new.loc["train"]], axis=1),
            pd.concat([X_test, new.loc["test"]], axis=1))


def rating_aggregates(X, X_test):
    """Mean / std of the answered ratings and how many were "not applicable" (0 -> NaN)."""
    def make(a):
        r = a[RATING_COLS]
        return pd.DataFrame({"rating_mean": r.mean(axis=1), "rating_std": r.std(axis=1),
                             "rating_na_count": r.isna().sum(axis=1)}, index=a.index)
    return _both(X, X_test, make)


def expectation_gaps(X, X_test):
    """Each rating minus the average rating of that passenger's segment (Class x Type of Travel)."""
    def make(a):
        means = a.groupby(GROUP, observed=True)[RATING_COLS].transform("mean")
        return (a[RATING_COLS] - means).add_suffix(" gap")
    return _both(X, X_test, make)


def segment_crosses(X, X_test):
    """Each rating crossed with the segment as one categorical (e.g. "Business|Business travel|5")."""
    def make(a):
        seg = a["Class"].astype(str) + "|" + a["Type of Travel"].astype(str)
        out = pd.DataFrame(index=a.index)
        for col in RATING_COLS:
            out[f"{col} x segment"] = (seg + "|" + a[col].fillna(0).astype(int).astype(str)).astype("category")
        return out
    return _both(X, X_test, make)


def nearest_original(X, X_test, k=10):
    """The k most similar rows of the original (real) dataset: nearest row's label and distance, the share of
    satisfied among the k nearest and their mean distance. Distance weights and scaling come from the original
    data only (context_select.distance_space with LightGBM fitted on it), so no competition labels are used.
    Cached in results/orig_nn.npz."""
    from context_select import distance_space, nearest
    cache = Path(__file__).parent / "results" / "orig_nn.npz"
    if not cache.exists():
        X_orig, y_orig = load_original()
        for col in CAT_COLS:  # same categories as the competition frames
            X_orig[col] = pd.Categorical(X_orig[col], categories=X[col].cat.categories)
        allx = pd.concat([X, X_test, X_orig], ignore_index=True)
        n_comp = len(X) + len(X_test)
        orig_rows = np.arange(n_comp, len(allx))
        y_all = np.concatenate([np.zeros(n_comp, dtype=int), np.asarray(y_orig).astype(int)])
        D, _ = distance_space(allx, y_all, orig_rows)
        idx = nearest(D, np.arange(n_comp), orig_rows, k=k)  # positions within the original rows
        dist = np.sqrt(np.maximum(((D[:n_comp, None, :] - D[orig_rows][idx]) ** 2).sum(-1), 0))
        lab = np.asarray(y_orig).astype(int)[idx]
        np.savez(cache, nn1_label=lab[:, 0], nn1_dist=dist[:, 0], knn_share=lab.mean(1), knn_dist=dist.mean(1))
    c = np.load(cache)
    feats = pd.DataFrame({"orig nn1 label": c["nn1_label"], "orig nn1 dist": c["nn1_dist"],
                          f"orig knn{k} satisfied": c["knn_share"], f"orig knn{k} dist": c["knn_dist"]})
    n = len(X)
    return (pd.concat([X, feats.iloc[:n].set_index(X.index)], axis=1),
            pd.concat([X_test, feats.iloc[n:].set_index(X_test.index)], axis=1))


def nearest_original_emb(X, X_test, k=10):
    """Like nearest_original, but distances in TabPFN's embedding space (tabpfn_embed_orig.py): in each of
    the two cross-fitted spaces, the k nearest reference rows of the other original half; the features are
    averaged over the two spaces. Cached in results/orig_nn_emb.npz."""
    from context_select import nearest
    cache = Path(__file__).parent / "results" / "orig_nn_emb.npz"
    if not cache.exists():
        feats = []
        for name in ("A", "B"):
            f = np.load(Path(__file__).parent / "results" / "full" / "knn" / f"orig_emb_{name}.npz")
            D = np.vstack([f["comp"], f["ref"]])
            n_comp = len(f["comp"])
            ref_rows = np.arange(n_comp, len(D))
            idx = nearest(D, np.arange(n_comp), ref_rows, k=k)
            dist = np.sqrt(np.maximum(((D[:n_comp, None, :] - D[ref_rows][idx]) ** 2).sum(-1), 0))
            lab = f["ref_label"][idx]
            feats.append((lab[:, 0], dist[:, 0], lab.mean(1), dist.mean(1)))
        nn1_label, nn1_dist, knn_share, knn_dist = (np.mean(v, axis=0) for v in zip(*feats))
        np.savez(cache, nn1_label=nn1_label, nn1_dist=nn1_dist, knn_share=knn_share, knn_dist=knn_dist)
    c = np.load(cache)
    feats = pd.DataFrame({"orig emb nn1 label": c["nn1_label"], "orig emb nn1 dist": c["nn1_dist"],
                          f"orig emb knn{k} satisfied": c["knn_share"], f"orig emb knn{k} dist": c["knn_dist"]})
    n = len(X)
    return (pd.concat([X, feats.iloc[:n].set_index(X.index)], axis=1),
            pd.concat([X_test, feats.iloc[n:].set_index(X_test.index)], axis=1))


def _then(*fns):
    def combined(X, X_test):
        for fn in fns:
            X, X_test = fn(X, X_test)
        return X, X_test
    return combined


FEATURE_SETS = {
    "baseline": ({}, None),
    "teacher": ({"teacher": True}, None),
    "teacher+aggregates": ({"teacher": True}, rating_aggregates),
    "teacher+gaps": ({"teacher": True}, expectation_gaps),
    "teacher+crosses": ({"teacher": True}, segment_crosses),
    "teacher+aggregates+gaps": ({"teacher": True}, _then(rating_aggregates, expectation_gaps)),
    "orig_nn": ({}, nearest_original),
    "orig_nn_emb": ({}, nearest_original_emb),
}
