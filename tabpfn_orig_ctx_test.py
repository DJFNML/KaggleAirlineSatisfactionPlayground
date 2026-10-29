"""Does TabPFN do better with the original real rows in its context? Fold 1, all validation rows.

Same context size as the 300k x 4 bigctx run (results/bigctx/ctx300k_est4_fold1.npz, 300k random rows),
so the only difference is what the rows are:
  orig:      170k random competition rows + all 129,880 original rows
  orig_flag: the same, plus an is_original column (0 for every competition row)
Also reports the gain on top of the current best mix (results/preds/tabpfn_emb2x200k_mix.npz).
Writes results/origctx/{stage}_fold1.npz.
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
from preprocess import CAT_COLS, load_data, load_original
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

N_RANDOM, N_EST, CHUNK = 170_000, 4, 10_000
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "origctx"
OUT.mkdir(parents=True, exist_ok=True)

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
X_orig, y_orig = load_original()
for col in CAT_COLS:  # same category codes as the competition frame
    X_orig[col] = pd.Categorical(X_orig[col], categories=X[col].cat.categories)
    assert X_orig[col].notna().all(), f"{col}: original values missing from the competition categories"
y_orig = np.asarray(y_orig).astype(int)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
yv = y[va_idx]
base = np.load(ROOT / "results" / "bigctx" / "ctx300k_est4_fold1.npz")["oof"]
mix = np.load(ROOT / "results" / "preds" / "tabpfn_emb2x200k_mix.npz")["oof"][va_idx]
seed = SEED + 1
rand = train_test_split(tr_idx, train_size=N_RANDOM, stratify=y[tr_idx], random_state=seed)[0]
print(f"300k random x {N_EST}: AUC {roc_auc_score(yv, base):.5f} | mix {roc_auc_score(yv, mix):.5f}", flush=True)


def best_on_top(pred):
    """Best AUC of (1-w)*mix + w*pred over a 0.05 grid, and its weight."""
    return max((roc_auc_score(yv, (1 - w) * mix + w * pred), w) for w in np.arange(0, 1.01, 0.05))


for stage in ["orig", "orig_flag"]:
    path = OUT / f"{stage}_fold1.npz"
    if not path.exists():
        frames = [X, X_orig]
        if stage == "orig_flag":
            frames = [X.assign(is_original=0.0), X_orig.assign(is_original=1.0)]
        Xc, cat_idx = tabpfn_matrix(pd.concat(frames, ignore_index=True))
        Xs, Xo = Xc[:len(X)], Xc[len(X):]
        ctx_X, ctx_y = np.concatenate([Xs[rand], Xo]), np.concatenate([y[rand], y_orig])
        torch.cuda.reset_peak_memory_stats()
        start = time.time()
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, n_estimators=N_EST,
                               ignore_pretraining_limits=True, random_state=seed,
                               fit_mode="fit_with_cache", kv_cache_precision="int8")
        clf.fit(ctx_X, ctx_y)
        pred = np.concatenate([clf.predict_proba(Xs[va_idx[i:i + CHUNK]])[:, 1]
                               for i in range(0, len(va_idx), CHUNK)])
        np.savez(path, val_idx=va_idx, oof=pred, peak_gb=torch.cuda.max_memory_allocated() / 1e9,
                 total_min=(time.time() - start) / 60)
        del clf
        torch.cuda.empty_cache()
    f = np.load(path)
    pred, gain = f["oof"], roc_auc_score(yv, f["oof"]) - roc_auc_score(yv, base)
    rng = np.random.default_rng(0)
    boots = [roc_auc_score(yv[i], pred[i]) - roc_auc_score(yv[i], base[i])
             for i in (rng.integers(0, len(yv), len(yv)) for _ in range(200))]
    top, w = best_on_top(pred)
    print(f"{stage}: AUC {roc_auc_score(yv, pred):.5f} ({gain:+.5f} vs 300k random, bootstrap sd "
          f"{np.std(boots):.5f}) | on top of the mix {top:.5f} ({top - roc_auc_score(yv, mix):+.5f}, w {w:.2f}) | "
          f"peak {float(f['peak_gb']):.1f} GB | {float(f['total_min']):.0f} min", flush=True)
