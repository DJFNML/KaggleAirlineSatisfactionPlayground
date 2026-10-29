"""Does TabPFN gain from route / original-lookup features? Fold 1, all validation rows, paired with the base.

Base: results/bigctx/ctx200k_est4_fold1.npz (200k random rows x 4 members, all 21 raw columns). Each stage
uses exactly its context rows and seed; only the columns differ:
  catfd      + Flight Distance as a categorical, + Flight Distance // 10 as a categorical (goodpjw2008's
               "catfd" / "catfd10" TabPFN inputs)
  catfd_org  + the original data's satisfied rate per value of every column (busyaprime's "org" block)
GPU memory is capped below the card's size, so a stage that does not fit fails with an out-of-memory error
instead of spilling into system RAM. Also reports the gain on top of the current best mix.
Writes results/routectx/{stage}_fold1.npz.
"""
import os
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.errors import TabPFNCUDAOutOfMemoryError

from cv import SEED
from preprocess import load_data, load_original
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

CONTEXT, N_EST, CHUNK = 200_000, 4, 5_000
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "routectx"
OUT.mkdir(parents=True, exist_ok=True)
torch.cuda.set_per_process_memory_fraction(0.92)  # out-of-memory error rather than spilling into system RAM

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
yv = y[va_idx]
base = np.load(ROOT / "results" / "bigctx" / "ctx200k_est4_fold1.npz")["oof"]
mix = np.load(ROOT / "results" / "preds" / "tabpfn_emb2x200k_mix.npz")["oof"][va_idx]
seed = SEED + 1
ctx = train_test_split(tr_idx, train_size=CONTEXT, stratify=y[tr_idx], random_state=seed)[0]

X_base, cat_base = tabpfn_matrix(X)
fd = X["Flight Distance"].to_numpy()
route = np.column_stack([pd.factorize(fd, sort=True)[0], pd.factorize(fd // 10, sort=True)[0]]).astype(np.float32)
X_orig, y_orig = load_original()
y_orig = np.asarray(y_orig).astype(int)
org = []
for c in X.columns:  # original satisfied rate per value (0 ratings are NaN in both frames: "nan" maps to "nan")
    rate = pd.Series(y_orig).groupby(X_orig[c].astype(str).to_numpy(), dropna=False).mean()
    keys = X[c].astype(str)
    print(f"  org {c}: {keys.isin(rate.index).mean():.4f} of rows have the value in the original", flush=True)
    org.append(keys.map(rate).fillna(y_orig.mean()).to_numpy(np.float32))
n = X_base.shape[1]
STAGES = {"catfd": (np.hstack([X_base, route]), cat_base + [n, n + 1]),  # the two route keys are categorical
          "catfd_org": (np.hstack([X_base, route, np.column_stack(org)]), cat_base + [n, n + 1])}


def best_on_top(pred):
    return max((roc_auc_score(yv, (1 - w) * mix + w * pred), w) for w in np.arange(0, 1.01, 0.05))


print(f"base 200k x {N_EST}: AUC {roc_auc_score(yv, base):.5f} | mix {roc_auc_score(yv, mix):.5f} | "
      f"base on top of the mix {best_on_top(base)[0] - roc_auc_score(yv, mix):+.5f}", flush=True)
for stage, (Xp, cat_idx) in STAGES.items():
    path = OUT / f"{stage}_fold1.npz"
    if not path.exists():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        try:
            clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, n_estimators=N_EST,
                                   ignore_pretraining_limits=True, random_state=seed,
                                   fit_mode="fit_with_cache", kv_cache_precision="int8")
            clf.fit(Xp[ctx], y[ctx])
            pred = np.concatenate([clf.predict_proba(Xp[va_idx[i:i + CHUNK]])[:, 1]
                                   for i in range(0, len(va_idx), CHUNK)])
        except (torch.OutOfMemoryError, TabPFNCUDAOutOfMemoryError):
            print(f"{stage}: out of GPU memory after {(time.time() - start) / 60:.1f} min, skipped", flush=True)
            continue
        np.savez(path, val_idx=va_idx, oof=pred, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                 total_min=(time.time() - start) / 60)
        del clf
    f = np.load(path)
    pred = f["oof"]
    gain = roc_auc_score(yv, pred) - roc_auc_score(yv, base)
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(yv[i], pred[i]) - roc_auc_score(yv[i], base[i])
             for i in (rng.integers(0, len(yv), len(yv)) for _ in range(200))]
    top, w = best_on_top(pred)
    print(f"{stage}: AUC {roc_auc_score(yv, pred):.5f} ({gain:+.5f} vs base, bootstrap sd {np.std(boots):.5f}) | "
          f"on top of the mix {top - roc_auc_score(yv, mix):+.5f} (w {w:.2f}) | peak {float(f['peak_gb']):.1f} GB | "
          f"{float(f['total_min']):.0f} min", flush=True)
