"""Do nearest-neighbour (embedding) contexts still add at today's best TabPFN settings? Fold 1 of the 5-fold split.

Same as tabpfn_route_full.py (300k-row contexts x 4 members, raw columns + the two route keys, same seed),
except the context: fold 1's validation rows are split into N_CLUSTERS k-means clusters in TabPFN's
embedding space (tabpfn_embed.py, built with the original data as context, so no competition labels), and
each cluster is predicted from its members' nearest training rows (context_select.knn_contexts), instead
of one random 300k context for all of them. ~43k query rows per context, close to the 15-fold embedding
run's ~22k per 200k context.
Paired with the random-context route model's fold 1 (results/routefull/tabpfn300k_route_fold1_val.npz),
and scored by what it adds on top of the full stack on fold 1's rows (LR on log-odds, 5 inner folds).
Writes results/localroute/cluster{c}.npz per cluster (a HOLD_tabpfn stop loses at most one cluster).
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from context_select import knn_contexts, nearest

N_CLUSTERS, CONTEXT, N_EST, CHUNK = 3, 300_000, 4, 10_000
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "localroute"
OUT.mkdir(parents=True, exist_ok=True)
HOLD = ROOT / "results" / "HOLD_tabpfn"
STACK = ["tabpfn300k_route_5fold", "tabpfn400k_e4_15fold_1ctx", "lgb_ladder_5fold", "xgb_ladder_5fold",
         "realmlp_6seeds_5fold"]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def stacked(Z, yv):
    """Out-of-fold LR stack scores on fold 1's rows (inner folds seed 7)."""
    out = np.zeros(len(yv))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=7).split(Z, yv):
        out[b] = LogisticRegression(max_iter=2000).fit(Z[a], yv[a]).decision_function(Z[b])
    return out


if __name__ == "__main__":
    import tabpfn_route_full as rf  # builds the route matrix and the folds on import (no fitting)
    from tabpfn import TabPFNClassifier

    y, Xp, cat_idx = rf.y, rf.Xp, rf.cat_idx
    tr, va = rf.folds[0]
    yv = y[va]
    emb = np.load(ROOT / "results" / "full" / "knn" / "tabpfn_emb64.npz")["emb"][:len(y)]
    nb_path = OUT / "neighbours_fold1.npy"
    if not nb_path.exists():
        start = time.time()
        np.save(nb_path, nearest(emb, va, tr))
        log(f"neighbours: {(time.time() - start) / 60:.1f} min")
    contexts = knn_contexts(emb, np.load(nb_path), va, tr, N_CLUSTERS, CONTEXT, seed=rf.SEED + 1)
    pos = np.empty(len(y), dtype=np.int64)
    pos[va] = np.arange(len(va))
    for c, (ctx, rows) in enumerate(contexts):
        path = OUT / f"cluster{c}.npz"
        if path.exists():
            continue
        if HOLD.exists():
            log(f"results/{HOLD.name} exists, stopping (finished clusters are kept)")
            sys.exit(0)
        start = time.time()
        clf = TabPFNClassifier(device=rf.DEVICE, categorical_features_indices=cat_idx, n_estimators=N_EST,
                               ignore_pretraining_limits=True, random_state=rf.SEED + 1,
                               fit_mode="fit_with_cache", kv_cache_precision="int8")
        clf.fit(Xp[ctx], y[ctx])
        pred = rf.predict(clf, Xp[rows])
        np.savez(path, rows=rows, oof=pred)
        log(f"cluster {c + 1}/{N_CLUSTERS}: {len(rows):,} rows, context satisfied rate {y[ctx].mean():.3f}, "
            f"AUC {roc_auc_score(y[rows], pred):.5f}, {(time.time() - start) / 60:.1f} min, peak "
            f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        del clf
        torch.cuda.empty_cache()

    local = np.full(len(va), np.nan)
    for c in range(N_CLUSTERS):
        f = np.load(OUT / f"cluster{c}.npz")
        local[pos[f["rows"]]] = f["oof"]
    assert not np.isnan(local).any()
    rand = np.load(ROOT / "results" / "routefull" / "tabpfn300k_route_fold1_val.npz")
    assert (rand["val_idx"] == va).all()
    rand = rand["oof"]
    Z = np.column_stack([logit(np.load(ROOT / "results" / "preds" / f"{m}.npz")["oof"][va]) for m in STACK])
    base = stacked(Z, yv)
    with_local = stacked(np.column_stack([Z, logit(local)]), yv)
    swap = stacked(np.column_stack([Z[:, 1:], logit(local)]), yv)  # local replaces the random-context route model
    rng = np.random.default_rng(0)
    boots = [rng.integers(0, len(yv), len(yv)) for _ in range(200)]

    def paired(a, b):
        d = roc_auc_score(yv, a) - roc_auc_score(yv, b)
        return d, np.std([roc_auc_score(yv[i], a[i]) - roc_auc_score(yv[i], b[i]) for i in boots])

    for name, a, b in [("local vs random context, alone", local, rand), ("stack + local", with_local, base),
                       ("stack, local instead of random route", swap, base)]:
        d, sd = paired(a, b)
        log(f"{name}: {d:+.6f} (bootstrap sd {sd:.6f})")
    log(f"alone: local {roc_auc_score(yv, local):.5f}, random {roc_auc_score(yv, rand):.5f}; "
        f"stack on fold 1 {roc_auc_score(yv, base):.5f}; rank corr local/random "
        f"{np.corrcoef(np.argsort(np.argsort(local)), np.argsort(np.argsort(rand)))[0, 1]:.4f}")
