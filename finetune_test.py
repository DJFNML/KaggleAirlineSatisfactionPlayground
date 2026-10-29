"""Does fine-tuning TabPFN on our data help, with the contexts we already use? Fold 1 of the 15-fold split.

1. Fine-tune TabPFN on fold 1's training rows only (FinetunedTabPFNClassifier, time-capped, early stopping
   on a 10% split of those rows). Small meta-datasets (FT_ROWS rows each) so it fits in 12 GB.
2. Load the best fine-tuned weights into the usual TabPFNClassifier (int8 key-value cache) and predict
   fold 1's validation rows with exactly the contexts the base model used: run_nn_full.py's pass-1 random
   200k context, and context_select_test.py's emb2x200k contexts. So any difference is the model itself.
Writes results/finetune/ (checkpoint, predictions). Usage: python finetune_test.py [smoke]
"""
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier
from tabpfn.finetuning.finetuned_classifier import FinetunedTabPFNClassifier
from tabpfn.model_loading import save_tabpfn_model

from context_select import knn_contexts
from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, tabpfn_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")  # per-epoch validation
SMOKE = sys.argv[1:] == ["smoke"]  # tiny CPU run: checks the pipeline end to end, not the result
FT_MINUTES, FT_ROWS, CONTEXT, CHUNK = (1, 1_000, 2_000, 500) if SMOKE else (120, 8_000, 200_000, 10_000)
FT_ROWS = int(os.environ.get("FT_ROWS", FT_ROWS))  # 20k-row tasks nearly filled the 12 GB and ran ~400 s/step
# The library spreads warmup (10% of all steps) and cosine decay over `epochs`, so it must be the real run
# length: with epochs=1000 the learning rate stays ~1e-7 for any run that fits in a night.
FT_EPOCHS = int(os.environ.get("FT_EPOCHS", 2))
ROOT = Path(__file__).parent
OUT = ROOT / "results" / ("finetune_smoke" if SMOKE else "finetune" + os.environ.get("FT_TAG", ""))
FULL = ROOT / "results" / "full"
OUT.mkdir(parents=True, exist_ok=True)

X, y_s, _, _ = load_data()
y = y_s.to_numpy()
Xp, cat_idx = tabpfn_matrix(X)
tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
device = "cpu" if SMOKE else DEVICE
if SMOKE:
    rng = np.random.default_rng(0)
    tr_idx, va_idx = np.sort(rng.choice(tr_idx, 5_000, replace=False)), np.sort(rng.choice(va_idx, 1_000, replace=False))


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


# 1. Fine-tune on fold 1's training rows. The library only writes a "best" checkpoint when an epoch beats the
#    base model on its validation split; the final (best-restored) weights are saved here either way.
ckpt = OUT / "finetuned.ckpt"
if not ckpt.exists():
    start = time.time()
    ft = FinetunedTabPFNClassifier(
        device=device, epochs=FT_EPOCHS, time_limit=FT_MINUTES * 60, learning_rate=1e-5,
        validation_split_ratio=0.1, n_finetune_ctx_plus_query_samples=FT_ROWS,
        n_inference_subsample_samples=min(50_000, len(tr_idx)), random_state=SEED,
        early_stopping=True, early_stopping_patience=4, eval_metric="roc_auc",
        n_estimators_final_inference=1,
        extra_classifier_kwargs={"categorical_features_indices": cat_idx, "ignore_pretraining_limits": True})
    ft.fit(Xp[tr_idx], y[tr_idx], output_dir=OUT)
    save_tabpfn_model(ft.finetuned_estimator_, ckpt)
    log(f"fine-tuned in {(time.time() - start) / 60:.1f} min; beat the base model on its validation split: "
        f"{'yes' if list(OUT.glob('*best.pth')) else 'NO (weights are effectively the base model)'}")
    del ft


def fit_predict(ctx, rows, seed, model_path=None):
    kw = {"model_path": str(model_path)} if model_path else {}
    clf = TabPFNClassifier(device=device, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                           random_state=seed, fit_mode="fit_with_cache", kv_cache_precision="int8", **kw)
    clf.fit(Xp[ctx], y[ctx])
    return np.concatenate([clf.predict_proba(Xp[rows[i:i + CHUNK]])[:, 1] for i in range(0, len(rows), CHUNK)])


# 2. The base model's exact contexts, with the fine-tuned weights
size = min(CONTEXT, len(tr_idx) - 1)
ctx, _ = train_test_split(tr_idx, train_size=size, stratify=y[tr_idx], random_state=SEED + 1)  # pass 1, fold 1
path = OUT / "ft_random_fold1.npz"
if not path.exists():
    start = time.time()
    np.savez(path, val_idx=va_idx, oof=fit_predict(ctx, va_idx, SEED + 1, ckpt))
    log(f"fine-tuned, random context: {(time.time() - start) / 60:.1f} min")
ft_rand = np.load(path)["oof"]

path = OUT / "ft_emb_fold1.npz"
if not path.exists():
    start = time.time()
    emb = np.load(FULL / "knn" / "tabpfn_emb64.npz")["emb"][:len(X)]
    neighbours = np.load(ROOT / "results" / "context_select" / "emb_neighbours128_fold1.npy")
    if SMOKE:  # neighbours are cached for the full fold only: use random contexts of the same size
        parts = [(ctx, va_idx)]
    else:
        parts = knn_contexts(emb, neighbours, va_idx, tr_idx, 2, CONTEXT, seed=SEED)
    pos = {r: i for i, r in enumerate(va_idx)}
    pred = np.full(len(va_idx), np.nan)
    for i, (c, rows) in enumerate(parts):
        pred[[pos[r] for r in rows]] = fit_predict(c, rows, SEED + i, ckpt)
    np.savez(path, val_idx=va_idx, oof=pred)
    log(f"fine-tuned, emb2x200k contexts: {(time.time() - start) / 60:.1f} min")
ft_emb = np.load(path)["oof"]

auc = lambda p: roc_auc_score(y[va_idx], p)  # noqa: E731
if SMOKE:
    log(f"smoke run OK: AUC random {auc(ft_rand):.4f}, emb {auc(ft_emb):.4f} (tiny data, not meaningful)")
    sys.exit(0)
base_rand = np.load(FULL / "tabpfn200k_c0_fold1.npz")["oof"]
base_emb = np.load(ROOT / "results" / "context_select" / "emb2x200k_fold1.npz")["oof"]
rand3 = np.mean([np.load(FULL / f"tabpfn200k_c{c}_fold1.npz")["oof"] for c in range(3)], axis=0)
log(f"random 200k context: base {auc(base_rand):.5f} -> fine-tuned {auc(ft_rand):.5f} "
    f"({auc(ft_rand) - auc(base_rand):+.5f})")
log(f"emb2x200k contexts:  base {auc(base_emb):.5f} -> fine-tuned {auc(ft_emb):.5f} ({auc(ft_emb) - auc(base_emb):+.5f})")
log(f"3 random + emb (base) {auc((3 * rand3 + base_emb) / 4):.5f} | + fine-tuned random and emb "
    f"{auc((3 * rand3 + base_emb + ft_rand + ft_emb) / 6):.5f} | fine-tuned random + emb alone "
    f"{auc((ft_rand + ft_emb) / 2):.5f}")
