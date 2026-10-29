"""Full 5-fold TabPFN run with route features (tabpfn_route_test.py: catfd + catfd10 gave +0.00068 on fold 1).

Folds: StratifiedKFold(5, seed 42), the same as ladder.py and realmlp.py. Each fold: a CONTEXT-row random
(stratified) context from the fold's training rows, N_EST members; the fold model predicts its validation
rows and the whole test set, and the test predictions are averaged over the folds (an ensemble of 5
contexts, as hemingweb and goodpjw2008 do).
Features: the 21 raw columns (0 ratings as NaN) + Flight Distance and Flight Distance // 10 as categoricals,
and with "org" also the original data's satisfied rate per value of every column.
GPU memory is capped below the card's size, so anything that does not fit fails instead of spilling into
system RAM; prediction falls back to smaller chunks. Creating results/HOLD_tabpfn stops it before the next
fold step. Validation and test predictions are saved separately, so a stop loses at most one step.

Usage: python tabpfn_route_full.py [CONTEXT] [org] [pass=N]   (pass N > 0 draws other contexts and seeds)
Writes results/routefull/{tag}_fold{k}_val.npz / _test.npz and results/preds/{tag}_5fold.npz.
"""
import os
import sys
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
from probe_nn import DEVICE, tabpfn_matrix

CONTEXT = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 300_000
ORG = "org" in sys.argv[1:]
PASS = next((int(a[5:]) for a in sys.argv[1:] if a.startswith("pass=")), 0)  # other random contexts
N_FOLDS, N_EST, CHUNKS = 5, 4, [10_000, 5_000, 2_500]
TAG = f"tabpfn{CONTEXT // 1000}k_route{'_org' if ORG else ''}{f'_p{PASS}' if PASS else ''}"
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "routefull"
OUT.mkdir(parents=True, exist_ok=True)
HOLD = ROOT / "results" / "HOLD_tabpfn"
torch.cuda.set_per_process_memory_fraction(0.95)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def route_matrix(X, X_test):
    """Raw TabPFN matrix + the two route keys (categorical), optionally + original lookups; train then test."""
    both = pd.concat([X, X_test], ignore_index=True)
    base, cat_idx = tabpfn_matrix(both)
    fd = both["Flight Distance"].to_numpy()
    route = np.column_stack([pd.factorize(fd, sort=True)[0], pd.factorize(fd // 10, sort=True)[0]])
    parts, n = [base, route.astype(np.float32)], base.shape[1]
    if ORG:
        X_orig, y_orig = load_original()
        y_orig = np.asarray(y_orig).astype(int)
        for c in X.columns:
            rate = pd.Series(y_orig).groupby(X_orig[c].astype(str).to_numpy(), dropna=False).mean()
            parts.append(both[c].astype(str).map(rate).fillna(y_orig.mean()).to_numpy(np.float32)[:, None])
    M = np.hstack(parts)
    return M[:len(X)], M[len(X):], cat_idx + [n, n + 1]


def predict(clf, rows):
    for chunk in CHUNKS:
        try:
            return np.concatenate([clf.predict_proba(rows[i:i + chunk])[:, 1] for i in range(0, len(rows), chunk)])
        except (torch.OutOfMemoryError, TabPFNCUDAOutOfMemoryError):
            torch.cuda.empty_cache()
            log(f"  out of GPU memory at predict chunk {chunk}, retrying smaller")
    raise RuntimeError("prediction does not fit even at the smallest chunk")


X, y_s, X_test, _ = load_data()
y = y_s.to_numpy()
Xp, Xt, cat_idx = route_matrix(X, X_test)
folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
if __name__ == "__main__":
    log(f"{TAG}: {Xp.shape[1]} features, {CONTEXT:,}-row contexts x {N_EST} members, {N_FOLDS} folds")
    for k, (tr, va) in enumerate(folds, start=1):
        val_path, test_path = OUT / f"{TAG}_fold{k}_val.npz", OUT / f"{TAG}_fold{k}_test.npz"
        if val_path.exists() and test_path.exists():
            continue
        if HOLD.exists():
            log(f"results/{HOLD.name} exists, stopping (finished files are kept)")
            sys.exit(0)
        start = time.time()
        ctx = train_test_split(tr, train_size=CONTEXT, stratify=y[tr], random_state=SEED + k + 100 * PASS)[0]
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, n_estimators=N_EST,
                               ignore_pretraining_limits=True, random_state=SEED + k + 100 * PASS,
                               fit_mode="fit_with_cache", kv_cache_precision="int8")
        clf.fit(Xp[ctx], y[ctx])
        log(f"  fold {k}: context fitted, {(time.time() - start) / 60:.1f} min, peak "
            f"{torch.cuda.max_memory_allocated() / 1e9:.1f} GB")
        if not val_path.exists():
            pv = predict(clf, Xp[va])
            np.savez(val_path, val_idx=va, oof=pv)
            log(f"{TAG} fold {k}/{N_FOLDS}: AUC {roc_auc_score(y[va], pv):.5f}, {(time.time() - start) / 60:.1f} min")
        if HOLD.exists():
            log(f"results/{HOLD.name} exists, stopping before fold {k}'s test predictions (finished files are kept)")
            sys.exit(0)
        np.savez(test_path, test=predict(clf, Xt))
        log(f"{TAG} fold {k}/{N_FOLDS}: test predicted, {(time.time() - start) / 60:.1f} min total")
        del clf
        torch.cuda.empty_cache()

    oof, test = np.zeros(len(y)), np.zeros(len(Xt))
    for k in range(1, N_FOLDS + 1):
        f = np.load(OUT / f"{TAG}_fold{k}_val.npz")
        oof[f["val_idx"]] = f["oof"]
        test += np.load(OUT / f"{TAG}_fold{k}_test.npz")["test"] / N_FOLDS
    np.savez(ROOT / "results" / "preds" / f"{TAG}_5fold.npz", oof=oof, test=test)
    log(f"{TAG}: OOF AUC {roc_auc_score(y, oof):.5f} -> results/preds/{TAG}_5fold.npz | all done")
