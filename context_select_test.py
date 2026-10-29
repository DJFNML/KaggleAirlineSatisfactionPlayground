"""Smarter TabPFN contexts: do rows chosen for their similarity beat a random 200k-row context?

Fold 1 of the 15-fold split, all of its validation rows, compared with the random 200k contexts from
run_nn_full.py passes 1-2 (results/full/tabpfn200k_c{0,1}_fold1.npz). Variants:
  knn{C}x{N}  - k-means splits the validation rows into C clusters, each with an N-row context of its
                members' nearest training rows (context_select.py), in the LightGBM-weighted feature space
  emb{C}x{N}  - the same, with distances in TabPFN's own embedding space (tabpfn_embed.py); skipped if
                the embeddings haven't been computed
  segment3    - one context per passenger segment (business travel x Business class / business travel x
                other classes / personal travel): that segment's training rows, at most 200k.
Every context uses the int8 key-value cache, as in the full run. Predictions are saved per variant to
results/context_select/{variant}_fold1.npz (finished ones are skipped), with the AUC alone and averaged
with the random contexts.

Usage: python context_select_test.py [--dry] [variant ...]   (--dry: cluster and context stats only, no GPU)
With no variants given, runs the embedding ones.
"""
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from context_select import N_NEIGHBOURS, distance_space, knn_contexts, nearest
from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS, tabpfn_matrix

VARIANTS = {"emb4x100k": (4, 100_000), "emb16x50k": (16, 50_000), "emb2x200k": (2, 200_000),
            "knn4x100k": (4, 100_000), "knn16x50k": (16, 50_000), "segment3": None, "knn2x200k": (2, 200_000)}
CHUNK, MAX_CONTEXT = 10_000, 200_000
ROOT = Path(__file__).parent
FULL, OUT = ROOT / "results" / "full", ROOT / "results" / "context_select"
EMB = FULL / "knn" / "tabpfn_emb64.npz"
DRY = "--dry" in sys.argv
names = [a for a in sys.argv[1:] if a != "--dry"] or [n for n in VARIANTS if n.startswith("emb")]
if not EMB.exists() and any(n.startswith("emb") for n in names):
    print(f"{EMB.name} not found, skipping the embedding variants", flush=True)
    names = [n for n in names if not n.startswith("emb")]

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
Xp, cat_idx = tabpfn_matrix(X)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
randoms = [np.load(FULL / f"tabpfn200k_c{c}_fold1.npz") for c in (0, 1)]
assert all((r["val_idx"] == va_idx).all() for r in randoms)
rand = [r["oof"] for r in randoms]
OUT.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def contexts(name):
    """[(context training rows, validation rows it predicts)] for a variant."""
    if name == "segment3":
        seg = np.where(X["Type of Travel"] == "Personal Travel", 2, np.where(X["Class"] == "Business", 0, 1))
        out = []
        for s in range(3):
            ctx = tr_idx[seg[tr_idx] == s]
            if len(ctx) > MAX_CONTEXT:
                ctx, _ = train_test_split(ctx, train_size=MAX_CONTEXT, stratify=y[ctx], random_state=SEED + s)
            out.append((ctx, va_idx[seg[va_idx] == s]))
        return out
    n_clusters, size = VARIANTS[name]
    space, neighbours = (DE, NEIGHBOURS_E) if name.startswith("emb") else (D, NEIGHBOURS)
    return knn_contexts(space, neighbours, va_idx, tr_idx, n_clusters, size, seed=SEED)


def fit_predict(ctx, rows, seed):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device="cuda", categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                           random_state=seed, fit_mode="fit_with_cache", kv_cache_precision="int8")
    clf.fit(Xp[ctx], y[ctx])
    return np.concatenate([clf.predict_proba(Xp[rows[i:i + CHUNK]])[:, 1] for i in range(0, len(rows), CHUNK)])


log(f"fold 1: {len(va_idx)} validation rows | random 200k: pass 1 {roc_auc_score(y[va_idx], rand[0]):.5f}, "
    f"pass 2 {roc_auc_score(y[va_idx], rand[1]):.5f}, both {roc_auc_score(y[va_idx], np.mean(rand, axis=0)):.5f}")
if any(n.startswith("knn") for n in names):
    D, weight = distance_space(X, y, tr_idx)
    log("importance weights: " + ", ".join(f"{c} {w:.2f}" for c, w in sorted(zip(X.columns, weight), key=lambda t: -t[1])[:8]))
    cache = OUT / f"neighbours{N_NEIGHBOURS}_fold1.npy"
    if not cache.exists():
        start = time.time()
        np.save(cache, nearest(D, va_idx, tr_idx))
        log(f"nearest neighbours: {(time.time() - start) / 60:.1f} min")
    NEIGHBOURS = np.load(cache)
if any(n.startswith("emb") for n in names):
    DE = np.load(EMB)["emb"][:len(X)]  # train rows come first
    cache = OUT / f"emb_neighbours{N_NEIGHBOURS}_fold1.npy"
    if not cache.exists():
        start = time.time()
        np.save(cache, nearest(DE, va_idx, tr_idx))
        log(f"embedding nearest neighbours: {(time.time() - start) / 60:.1f} min")
    NEIGHBOURS_E = np.load(cache)
pos = {r: i for i, r in enumerate(va_idx)}
for name in names:
    path = OUT / f"{name}_fold1.npz"
    if path.exists() and not DRY:
        pred = np.load(path)["oof"]
    else:
        parts = contexts(name)
        for ctx, rows in parts:
            log(f"{name}: context {len(ctx)} rows ({y[ctx].mean():.2f} satisfied) -> {len(rows)} rows "
                f"({y[rows].mean():.2f} satisfied)")
        if DRY:
            continue
        start, pred = time.time(), np.full(len(va_idx), np.nan)
        for i, (ctx, rows) in enumerate(parts):
            pred[[pos[r] for r in rows]] = fit_predict(ctx, rows, seed=SEED + i)
        assert not np.isnan(pred).any()
        np.savez(path, val_idx=va_idx, oof=pred)
        log(f"{name}: {(time.time() - start) / 60:.1f} min -> {path.name}")
    auc = lambda p: roc_auc_score(y[va_idx], p)  # noqa: E731
    log(f"{name}: AUC {auc(pred):.5f} | avg with random pass 1 {auc((pred + rand[0]) / 2):.5f} | "
        f"avg with both random {auc((pred + rand[0] + rand[1]) / 3):.5f}")
