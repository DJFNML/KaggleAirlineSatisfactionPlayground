"""Does TabPFN do better without near-useless features? Fold 1, all validation rows, paired with the base run.

Uses exactly the context rows and seed of run_nn_full.py's pass-1 random 200k context for fold 1
(results/full/tabpfn200k_c0_fold1.npz, all features), so the only difference is the dropped columns.
Reports the AUC change with a bootstrap error bar, and peak GPU memory.

Usage: python tabpfn_featsel_test.py COLUMN [COLUMN ...]   (default: Gender, Food and drink, Departure Delay)
Writes results/featsel/drop_{columns}_fold1.npz.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier

from cv import SEED
from preprocess import DEP_DELAY, load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

DROP = sys.argv[1:] or ["Gender", "Food and drink", DEP_DELAY]
CONTEXT, CHUNK = 200_000, 10_000
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "featsel"
OUT.mkdir(parents=True, exist_ok=True)

X, y_s, _, _ = load_data()  # same preprocessing as the full TabPFN run (0 ratings -> NaN)
y = y_s.to_numpy()
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
base = np.load(ROOT / "results" / "full" / "tabpfn200k_c0_fold1.npz")
assert (base["val_idx"] == va_idx).all()
Xp, cat_idx = tabpfn_matrix(X.drop(columns=DROP))
seed = SEED + 1  # run_nn_full.py: pass 0, fold 1
ctx, _ = train_test_split(tr_idx, train_size=CONTEXT, stratify=y[tr_idx], random_state=seed)

path = OUT / ("drop_" + "_".join(c.split()[0].lower() for c in DROP) + "_fold1.npz")
start = time.time()
torch.cuda.reset_peak_memory_stats()
clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                       random_state=seed, fit_mode="fit_with_cache", kv_cache_precision="int8")
clf.fit(Xp[ctx], y[ctx])
pred = np.concatenate([clf.predict_proba(Xp[va_idx[i:i + CHUNK]])[:, 1] for i in range(0, len(va_idx), CHUNK)])
peak = torch.cuda.max_memory_allocated() / 1e9
np.savez(path, val_idx=va_idx, oof=pred)

yv, b = y[va_idx], base["oof"]
gain = roc_auc_score(yv, pred) - roc_auc_score(yv, b)
rng = np.random.default_rng(0)
boots = [roc_auc_score(yv[i], pred[i]) - roc_auc_score(yv[i], b[i])
         for i in (rng.integers(0, len(yv), len(yv)) for _ in range(200))]
print(f"dropped {DROP} ({Xp.shape[1]} features left): AUC {roc_auc_score(yv, pred):.5f} vs all features "
      f"{roc_auc_score(yv, b):.5f} -> {gain:+.5f} (bootstrap sd {np.std(boots):.5f}) | corr with base "
      f"{np.corrcoef(pred, b)[0, 1]:.4f} | peak GPU memory {peak:.1f} GB | {(time.time() - start) / 60:.1f} min "
      f"-> {path.name}", flush=True)
