"""Bigger TabPFN contexts by using fewer internal ensemble members. Fold 1, all validation rows.

The int8 key-value cache keeps one copy of the context per ensemble member, so halving the members should
roughly halve the cache and leave room for a bigger context. Stages (context rows x members):
  200k x 4 (the base run's exact context rows and seed: the cost of halving the ensemble), 400k x 4,
  300k x 4, all training rows x 2.
Compared with the base, run_nn_full.py's pass-1 random 200k x 8 (results/full/tabpfn200k_c0_fold1.npz).
Each stage falls back to smaller predict chunks if it runs out of GPU memory.
Writes results/bigctx/ctx{N}k_est{E}_fold1.npz.
"""
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier

from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

STAGES = [(200_000, 4), (400_000, 4), (300_000, 4), (None, 2)]  # None = every training row of the fold
CHUNKS = [10_000, 5_000, 2_500]
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "bigctx"
OUT.mkdir(parents=True, exist_ok=True)

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
Xp, cat_idx = tabpfn_matrix(X)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
yv = y[va_idx]
base = np.load(ROOT / "results" / "full" / "tabpfn200k_c0_fold1.npz")["oof"]
rand3 = np.mean([np.load(ROOT / "results" / "full" / f"tabpfn200k_c{c}_fold1.npz")["oof"] for c in range(3)], axis=0)
seed = SEED + 1  # run_nn_full.py: pass 0, fold 1
print(f"base 200k x 8: AUC {roc_auc_score(yv, base):.5f} | 3 random 200k x 8 averaged {roc_auc_score(yv, rand3):.5f}",
      flush=True)


def is_oom(e):
    return "out of memory" in str(e).lower() or "OutOfMemory" in type(e).__name__


for size, n_est in STAGES:
    n = len(tr_idx) if size is None else size
    path = OUT / f"ctx{n // 1000}k_est{n_est}_fold1.npz"
    if not path.exists():
        ctx = tr_idx if size is None else train_test_split(tr_idx, train_size=size, stratify=y[tr_idx],
                                                           random_state=seed)[0]
        pred = None
        for chunk in CHUNKS:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start, clf = time.time(), None
            try:
                clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, n_estimators=n_est,
                                       ignore_pretraining_limits=True, random_state=seed,
                                       fit_mode="fit_with_cache", kv_cache_precision="int8")
                clf.fit(Xp[ctx], y[ctx])
                fit_min = (time.time() - start) / 60
                pred = np.concatenate([clf.predict_proba(Xp[va_idx[i:i + chunk]])[:, 1]
                                       for i in range(0, len(va_idx), chunk)])
                break
            except Exception as e:  # noqa: BLE001 - CUDA OOM surfaces as several error types
                if not is_oom(e):
                    raise
                print(f"{n} x {n_est}, predict chunk {chunk}: out of GPU memory after "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
            finally:
                del clf
        if pred is None:
            print(f"{n} x {n_est}: does not fit, skipping", flush=True)
            continue
        peak = torch.cuda.max_memory_allocated() / 1e9
        np.savez(path, val_idx=va_idx, oof=pred, peak_gb=peak, fit_min=fit_min,
                 total_min=(time.time() - start) / 60)
    f = np.load(path)
    pred = f["oof"]
    print(f"{n // 1000}k x {n_est}: AUC {roc_auc_score(yv, pred):.5f} ({roc_auc_score(yv, pred) - roc_auc_score(yv, base):+.5f} "
          f"vs base) | averaged with the 3 random passes {roc_auc_score(yv, (3 * rand3 + pred) / 4):.5f} | "
          f"peak {float(f['peak_gb']):.1f} GB | fit {float(f['fit_min']):.0f} min, total {float(f['total_min']):.0f} min",
          flush=True)
