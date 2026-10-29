"""Check lean_patch.py: the same context and seed with and without the patch must give the same predictions.

Fold 1, a 20k-row context, 1 ensemble member, 5k predicted rows. Then, with the patch, times a fit on
FULL_ROWS context rows (default: all of fold 1's training rows) and reports peak GPU memory, with memory
capped so a context that does not fit fails instead of spilling into system RAM.
Usage: python lean_patch_check.py [FULL_ROWS]
"""
import sys
import time

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from tabpfn import TabPFNClassifier

import lean_patch
from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

torch.cuda.set_per_process_memory_fraction(0.92)
X, y_s, _, _ = load_data()
y = y_s.to_numpy()
Xp, cat_idx = tabpfn_matrix(X)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
rng = np.random.default_rng(0)
small, query = rng.choice(tr_idx, 20_000, replace=False), va_idx[:5_000]


def run(ctx, rows, chunk=5_000):
    clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, n_estimators=1,
                           ignore_pretraining_limits=True, random_state=SEED + 1,
                           fit_mode="fit_with_cache", kv_cache_precision="int8")
    clf.fit(Xp[ctx], y[ctx])
    return np.concatenate([clf.predict_proba(Xp[rows[i:i + chunk]])[:, 1] for i in range(0, len(rows), chunk)])


stock = run(small, query)
lean_patch.apply()
lean = run(small, query)
print(f"20k context: max |stock - lean| = {np.abs(stock - lean).max():.2e}, AUC stock "
      f"{roc_auc_score(y[query], stock):.5f} lean {roc_auc_score(y[query], lean):.5f}", flush=True)

n = int(sys.argv[1]) if len(sys.argv) > 1 else len(tr_idx)
ctx = tr_idx if n >= len(tr_idx) else rng.choice(tr_idx, n, replace=False)
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
start = time.time()
try:
    pred = run(ctx, va_idx)
    print(f"{len(ctx):,}-row context x 1 member, lean: fold-1 AUC {roc_auc_score(y[va_idx], pred):.5f} | "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB | {(time.time() - start) / 60:.1f} min", flush=True)
    np.savez(f"results/bigctx/lean{len(ctx) // 1000}k_est1_fold1.npz", val_idx=va_idx, oof=pred)
except torch.OutOfMemoryError:
    print(f"{len(ctx):,}-row context: out of GPU memory after {(time.time() - start) / 60:.1f} min", flush=True)
