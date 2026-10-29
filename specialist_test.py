"""A hard-row specialist: TabPFN with a context made only of uncertain passengers. Fold 1 of the 15-fold split.

Hardness without leakage: a LightGBM 5-fold CV inside fold 1's training rows scores every training row; a
LightGBM on all of them scores the validation rows. The N training rows with scores nearest 0.5 are the
context; validation rows scoring inside the same band are predicted by it. Compared, on those rows and
overall, with the base: the mean of the 3 random passes and the emb2x200k run, weighted 3:1.

Usage: python specialist_test.py prep   (CPU: hardness scores, cached)
       python specialist_test.py        (GPU: the specialist contexts)
Writes results/specialist/.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

SIZES, CHUNK = [200_000, 100_000], 10_000
ROOT = Path(__file__).parent
OUT, FULL = ROOT / "results" / "specialist", ROOT / "results" / "full"
OUT.mkdir(parents=True, exist_ok=True)

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def lgb_model():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=127, min_child_samples=50,
                              num_threads=8, verbose=-1)


hard_path = OUT / "hardness_fold1.npz"
if not hard_path.exists():
    start = time.time()
    tr_score = np.empty(len(tr_idx))
    for fit, pred in StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr_idx, y[tr_idx]):
        m = lgb_model().fit(X.iloc[tr_idx[fit]], y[tr_idx[fit]])
        tr_score[pred] = m.predict_proba(X.iloc[tr_idx[pred]])[:, 1]
    va_score = lgb_model().fit(X.iloc[tr_idx], y[tr_idx]).predict_proba(X.iloc[va_idx])[:, 1]
    np.savez(hard_path, tr_score=tr_score, va_score=va_score)
    log(f"hardness scores: inner AUC {roc_auc_score(y[tr_idx], tr_score):.5f}, validation AUC "
        f"{roc_auc_score(y[va_idx], va_score):.5f}, {(time.time() - start) / 60:.1f} min")
h = np.load(hard_path)
tr_score, va_score = h["tr_score"], h["va_score"]
if sys.argv[1:] == ["prep"]:
    for n in SIZES:
        cut = np.sort(np.abs(tr_score - 0.5))[n - 1]
        band = np.abs(va_score - 0.5) <= cut
        log(f"{n} hardest training rows: scores within 0.5 +/- {cut:.3f} ({y[tr_idx][np.abs(tr_score - 0.5) <= cut].mean():.2f} "
            f"satisfied) -> {band.sum()} validation rows in the band ({y[va_idx][band].mean():.2f} satisfied)")
    sys.exit(0)

from tabpfn import TabPFNClassifier  # noqa: E402

Xp, cat_idx = tabpfn_matrix(X)
rand3 = np.mean([np.load(FULL / f"tabpfn200k_c{c}_fold1.npz")["oof"] for c in range(3)], axis=0)
emb = np.load(ROOT / "results" / "context_select" / "emb2x200k_fold1.npz")["oof"]
base = (3 * rand3 + emb) / 4
yv = y[va_idx]
for n in SIZES:
    order = np.argsort(np.abs(tr_score - 0.5))
    ctx = tr_idx[order[:n]]
    cut = np.abs(tr_score[order[n - 1]] - 0.5)
    band = np.abs(va_score - 0.5) <= cut
    path = OUT / f"specialist{n // 1000}k_fold1.npz"
    if not path.exists():
        start = time.time()
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                               random_state=SEED, fit_mode="fit_with_cache", kv_cache_precision="int8")
        clf.fit(Xp[ctx], y[ctx])
        rows = va_idx[band]
        pred = np.concatenate([clf.predict_proba(Xp[rows[i:i + CHUNK]])[:, 1] for i in range(0, len(rows), CHUNK)])
        np.savez(path, band=band, pred=pred)
        del clf
        log(f"specialist {n}: {(time.time() - start) / 60:.1f} min")
    f = np.load(path)
    band, pred = f["band"], f["pred"]
    log(f"specialist {n // 1000}k: {band.sum()} band rows ({yv[band].mean():.2f} satisfied) | AUC inside the band: "
        f"base {roc_auc_score(yv[band], base[band]):.5f}, specialist {roc_auc_score(yv[band], pred):.5f}, "
        f"half-half {roc_auc_score(yv[band], (base[band] + pred) / 2):.5f}")
    h1, h2 = next(StratifiedKFold(2, shuffle=True, random_state=0).split(yv, yv))
    gains = {}
    for w in np.round(np.arange(0, 1.01, 0.1), 1):
        mixed = base.copy()
        mixed[band] = (1 - w) * base[band] + w * pred
        gains[w] = (roc_auc_score(yv[h1], mixed[h1]), roc_auc_score(yv[h2], mixed[h2]), roc_auc_score(yv, mixed))
    w_best = max(gains, key=lambda w: gains[w][0])  # picked on half 1, scored on half 2
    log(f"    overall fold AUC: base {roc_auc_score(yv, base):.5f} | specialist weight picked on half 1: {w_best} "
        f"-> half-2 gain {gains[w_best][1] - gains[0.0][1]:+.5f} | all weights (all rows): "
        + ", ".join(f"{w}: {g[2] - gains[0.0][2]:+.5f}" for w, g in gains.items()))
