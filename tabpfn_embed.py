"""TabPFN's own row embeddings, as a distance space for choosing contexts (context_select.py).

Every competition row (train + test) is embedded with the original real dataset (129,880 rows, no
competition labels) as the context, so no row's own label can leak into its embedding. One ensemble
member (1024 dimensions per row), reduced to 64 by PCA fitted on a random 50k-row sample. Rows are in
load_data() order, train then test. Writes results/full/knn/tabpfn_emb64.npz (emb, explained).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from tabpfn import TabPFNClassifier

from preprocess import CAT_COLS, load_data, load_original
from probe_nn import DEVICE, tabpfn_matrix

N_DIMS, CHUNK = 64, 10_000
SMOKE = sys.argv[1:] == ["smoke"]  # quick CPU check on a few thousand rows
OUT = Path(__file__).parent / "results" / "full" / "knn" / f"tabpfn_emb{N_DIMS}{'_smoke' if SMOKE else ''}.npz"

X, _, X_test, _ = load_data()
X_orig, y_orig = load_original()
for col in CAT_COLS:  # same category codes as the competition frames
    X_orig[col] = pd.Categorical(X_orig[col], categories=X[col].cat.categories)
rows, cat_idx = tabpfn_matrix(pd.concat([X, X_test], ignore_index=True))
ctx, _ = tabpfn_matrix(X_orig)
y_ctx = np.asarray(y_orig).astype(int)
if SMOKE:
    DEVICE, CHUNK, rows, ctx, y_ctx = "cpu", 500, rows[:3000], ctx[:1000], y_ctx[:1000]

start = time.time()
clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                       n_estimators=1, random_state=0, fit_mode="fit_with_cache", kv_cache_precision="int8")
clf.fit(ctx, y_ctx)
print(f"context fitted: {(time.time() - start) / 60:.1f} min", flush=True)


def embed(idx):
    return np.concatenate([clf.get_embeddings(rows[idx[i:i + CHUNK]])[0] for i in range(0, len(idx), CHUNK)])


sample = np.random.default_rng(0).choice(len(rows), min(50_000, len(rows)), replace=False)
pca = PCA(N_DIMS, random_state=0).fit(embed(sample))
print(f"PCA: {N_DIMS} dims keep {pca.explained_variance_ratio_.sum():.1%} of the variance", flush=True)

emb = np.empty((len(rows), N_DIMS), dtype=np.float32)
for i in range(0, len(rows), CHUNK):
    emb[i:i + CHUNK] = pca.transform(clf.get_embeddings(rows[i:i + CHUNK])[0])
    if i % 200_000 == 0:
        print(f"  {i:,} / {len(rows):,} rows, {(time.time() - start) / 60:.1f} min", flush=True)
np.savez(OUT, emb=emb, explained=pca.explained_variance_ratio_)
print(f"done: {(time.time() - start) / 60:.1f} min -> {OUT.name}", flush=True)
