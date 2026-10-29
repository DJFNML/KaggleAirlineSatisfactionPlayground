"""Full 15-fold TabPFN with nearest-neighbour contexts (context_select.py), saved after every fold.

Usage: python run_knn_full.py [emb] neighbours                    - CPU only: cache every fold's neighbours
       python run_knn_full.py [emb] N_CLUSTERS CONTEXT_SIZE MAX_PASSES PREDICT_CHUNK [sat_only]
"sat_only": skip clusters whose context is mostly unsatisfied (by the context rows' training labels); their
rows get NaN, and the blend uses the other models alone there. Every file also saves `sat`, which rows were
in a mostly-satisfied cluster, for cluster-specific blend weights.
"emb": distances in TabPFN's embedding space (tabpfn_embed.py) instead of the LightGBM-weighted features.
Each pass: every fold's validation rows are split into N_CLUSTERS k-means clusters, each predicted from a
CONTEXT_SIZE-row context of its members' nearest training rows; then the test rows the same way, with
all training rows to draw from. Passes differ in the k-means and TabPFN seeds. Files (same layout as
run_nn_full.py, so assemble_full.py works): results/full/tabpfn_{knn|emb}{C}x{N}k_c{pass}_fold{k}.npz and
..._test.npz. Neighbour caches: results/full/knn/space_fold{k}.npz / space_test.npz (features) or
emb_fold{k}.npy / emb_test.npy (embeddings).
Creating results/HOLD_tabpfn stops it before the next fold or test cluster.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from context_select import cached_space, knn_contexts, nearest
from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

ROOT = Path(__file__).parent
FULL = ROOT / "results" / "full"
SPACES = FULL / "knn"
HOLD = ROOT / "results" / "HOLD_tabpfn"
EMB_MODE = sys.argv[1] == "emb"
ARGS = sys.argv[2:] if EMB_MODE else sys.argv[1:]

X, y_s, X_test, _ = load_data()
y = y_s.to_numpy()
X_all = pd.concat([X, X_test], ignore_index=True)  # test rows are positions len(X).. in X_all
assert all(X_all[c].dtype == "category" for c in X.select_dtypes("category"))
train_rows, test_rows = np.arange(len(X)), np.arange(len(X), len(X_all))
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
SPACES.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def space(k):
    """Distance space + neighbours for fold k (1-based) or the test set (k=None)."""
    if EMB_MODE:
        query, ref = (test_rows, train_rows) if k is None else (folds[k - 1][1], folds[k - 1][0])
        path = SPACES / f"emb_{'test' if k is None else f'fold{k}'}.npy"
        if not path.exists():
            np.save(path, nearest(EMB, query, ref))
        return EMB, np.load(path)
    if k is None:
        return cached_space(SPACES / "space_test.npz", X_all, y, train_rows, test_rows)
    tr_idx, va_idx = folds[k - 1]
    return cached_space(SPACES / f"space_fold{k}.npz", X_all, y, tr_idx, va_idx)


EMB = np.load(SPACES / "tabpfn_emb64.npz")["emb"] if EMB_MODE else None  # rows: train then test, as X_all
if ARGS[0] == "neighbours":
    for k in [*range(1, N_SPLITS + 1), None]:
        start = time.time()
        space(k)
        log(f"neighbours {'test' if k is None else f'fold {k}'}: {(time.time() - start) / 60:.1f} min")
    sys.exit(0)

from tabpfn import TabPFNClassifier  # noqa: E402

N_CLUSTERS, CONTEXT, MAX_PASSES, CHUNK = (int(a) for a in ARGS[:4])
SAT_ONLY = ARGS[4:] == ["sat_only"]
tag = f"tabpfn_{'emb' if EMB_MODE else 'knn'}{N_CLUSTERS}x{CONTEXT // 1000}k"
Xp, cat_idx = tabpfn_matrix(X_all)
log(f"TabPFN, {N_CLUSTERS} clusters x {CONTEXT}-row nearest-neighbour contexts "
    f"({'embedding' if EMB_MODE else 'feature'} space), up to {MAX_PASSES} passes")


def stop_if_held():
    if HOLD.exists():
        log(f"results/{HOLD.name} exists, stopping (finished files are kept)")
        sys.exit(0)


def predict(k, seed):
    """Predictions for fold k's validation rows (or the test rows, k=None), one context per cluster."""
    D, neighbours = space(k)
    query, ref = (test_rows, train_rows) if k is None else (folds[k - 1][1], folds[k - 1][0])
    pos = np.empty(len(X_all), dtype=np.int64)
    pos[query] = np.arange(len(query))
    pred, sat = np.full(len(query), np.nan), np.zeros(len(query), dtype=bool)
    for i, (ctx, rows) in enumerate(knn_contexts(D, neighbours, query, ref, N_CLUSTERS, CONTEXT, seed)):
        sat[pos[rows]] = y[ctx].mean() >= 0.5
        if SAT_ONLY and y[ctx].mean() < 0.5:
            continue
        stop_if_held()
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                               random_state=seed + i, fit_mode="fit_with_cache", kv_cache_precision="int8")
        clf.fit(Xp[ctx], y[ctx])
        pred[pos[rows]] = np.concatenate([clf.predict_proba(Xp[rows[j:j + CHUNK]])[:, 1]
                                          for j in range(0, len(rows), CHUNK)])
        del clf
    assert not np.isnan(pred[sat] if SAT_ONLY else pred).any()
    return pred, sat


for c in range(MAX_PASSES):
    for k in range(1, N_SPLITS + 1):
        path = FULL / f"{tag}_c{c}_fold{k}.npz"
        if path.exists():
            continue
        stop_if_held()
        start = time.time()
        va_idx = folds[k - 1][1]
        pred, sat = predict(k, seed=SEED + 100 * c + k)
        np.savez(path, val_idx=va_idx, oof=pred, sat=sat)
        ok = ~np.isnan(pred)  # sat_only: AUC on the satisfied cluster's rows only
        log(f"{tag} pass {c + 1} fold {k}/{N_SPLITS}: AUC {roc_auc_score(y[va_idx][ok], pred[ok]):.5f}"
            f"{' (satisfied cluster only)' if SAT_ONLY else ''}, "
            f"{(time.time() - start) / 60:.1f} min -> {path.name}")
    path = FULL / f"{tag}_c{c}_test.npz"
    if not path.exists():
        start = time.time()
        pred, sat = predict(None, seed=1000 + c)
        np.savez(path, test=pred, sat=sat)
        log(f"{tag} pass {c + 1} test: {(time.time() - start) / 60:.1f} min -> {path.name}")
    log(f"{tag} pass {c + 1} complete")
log(f"{tag}: all done")
