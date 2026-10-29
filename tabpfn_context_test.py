"""How much does a bigger TabPFN context help, and what fits on a 12 GB GPU, at what speed?

Fold 1 of the 15-fold split: contexts of different sizes from its training rows, predicting the
same 10k validation rows each time. For each size it tries:
  cache    - fit_with_cache: the context is processed once into an int8 key-value cache that every
             predict call reuses (fast predictions if the cache fits in memory)
  standard - the context is reprocessed on every predict call (fit_preprocessors, memory mode "auto")
Each variant falls back to smaller predict chunks if it runs out of memory. LightGBM's OOF AUC on
the same rows is shown for reference. Writes results/tabpfn_context_test.csv.
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier

from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS, tabpfn_matrix

CONTEXT_SIZES = [50_000, 100_000, 200_000, 400_000]
N_EVAL = 10_000
CHUNKS = [10_000, 5_000, 2_500]  # try the largest predict chunk first
VARIANTS = {
    "cache": {"fit_mode": "fit_with_cache", "kv_cache_precision": "int8"},
    "standard": {"fit_mode": "fit_preprocessors"},
}
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "tabpfn_context_test.csv"

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
Xp, cat_idx = tabpfn_matrix(X)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
ev = va_idx[:N_EVAL]
lgb_oof = np.load(ROOT / "results" / "preds" / "lgb_t119_15fold_5seeds.npz")["oof"]
print(f"reference: LightGBM (5 seeds) on these {N_EVAL} rows: AUC {roc_auc_score(y[ev], lgb_oof[ev]):.5f}",
      flush=True)


def run(size, variant, chunk):
    """Fit on a context of `size` rows and predict the eval rows; None if out of memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ctx, _ = train_test_split(tr_idx, train_size=size, stratify=y[tr_idx], random_state=0)
    clf = None
    try:
        clf = TabPFNClassifier(device="cuda", categorical_features_indices=cat_idx,
                               ignore_pretraining_limits=True, random_state=0, **VARIANTS[variant])
        t = time.time()
        clf.fit(Xp[ctx], y[ctx])
        fit_s = time.time() - t
        t = time.time()
        p = np.concatenate([clf.predict_proba(Xp[ev[i:i + chunk]])[:, 1] for i in range(0, N_EVAL, chunk)])
        pred_s = time.time() - t
    except Exception as e:  # noqa: BLE001 - CUDA OOM surfaces as several error types
        if "out of memory" not in str(e).lower() and "OutOfMemory" not in type(e).__name__:
            raise
        return None
    finally:
        del clf
    return {"context": size, "variant": variant, "chunk": chunk, "auc": roc_auc_score(y[ev], p),
            "fit_s": fit_s, "predict_s_per_10k": pred_s * 10_000 / N_EVAL,
            "peak_gb": torch.cuda.max_memory_allocated() / 1e9}


rows = []
for size in CONTEXT_SIZES:
    fitted_any = False
    for variant in VARIANTS:
        for chunk in CHUNKS:
            row = run(size, variant, chunk)
            if row is None:
                print(f"context {size} {variant} chunk {chunk}: out of memory", flush=True)
                continue
            fitted_any = True
            rows.append(row)
            print(f"context {size} {variant} chunk {chunk}: AUC {row['auc']:.5f} | fit {row['fit_s']:.0f}s | "
                  f"predict {row['predict_s_per_10k']:.0f}s per 10k rows | peak {row['peak_gb']:.1f} GB", flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
            break  # largest chunk that fits is enough for this variant
    if not fitted_any:
        print(f"context {size}: nothing fits, stopping", flush=True)
        break
