"""TabPFN embedding spaces for finding each competition row's nearest original (real) rows, without leakage.

The original data is split into halves A and B. Space A: TabPFN with half A as context embeds every
competition row and every half-B row, and half B is the reference set. Space B: the same, swapped. In each
space all rows are embedded against the same context and no reference row is in it, so no reference row's
own label is in its embedding. One ensemble member, PCA to 64 dims per space.
Writes results/full/knn/orig_emb_{A,B}.npz (comp: competition rows train then test, ref: reference rows,
ref_label: their labels).
"""
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tabpfn import TabPFNClassifier

from preprocess import CAT_COLS, load_data, load_original
from probe_nn import DEVICE, tabpfn_matrix

N_DIMS, CHUNK = 64, 10_000
OUT = Path(__file__).parent / "results" / "full" / "knn"

X, _, X_test, _ = load_data()
X_orig, y_orig = load_original()
for col in CAT_COLS:  # same category codes as the competition frames
    X_orig[col] = pd.Categorical(X_orig[col], categories=X[col].cat.categories)
comp, cat_idx = tabpfn_matrix(pd.concat([X, X_test], ignore_index=True))
orig, _ = tabpfn_matrix(X_orig)
y_orig = np.asarray(y_orig).astype(int)
half_a, half_b = train_test_split(np.arange(len(orig)), train_size=0.5, stratify=y_orig, random_state=0)

for name, ctx, ref in (("A", half_a, half_b), ("B", half_b, half_a)):
    start = time.time()
    clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                           n_estimators=1, random_state=0, fit_mode="fit_with_cache", kv_cache_precision="int8")
    clf.fit(orig[ctx], y_orig[ctx])
    rows = np.vstack([comp, orig[ref]])

    def embed(r):
        return np.concatenate([clf.get_embeddings(r[i:i + CHUNK])[0] for i in range(0, len(r), CHUNK)])

    sample = np.random.default_rng(0).choice(len(rows), 50_000, replace=False)
    pca = PCA(N_DIMS, random_state=0).fit(embed(rows[sample]))
    emb = np.vstack([pca.transform(clf.get_embeddings(rows[i:i + CHUNK])[0]) for i in range(0, len(rows), CHUNK)])
    np.savez(OUT / f"orig_emb_{name}.npz", comp=emb[:len(comp)].astype(np.float32),
             ref=emb[len(comp):].astype(np.float32), ref_label=y_orig[ref])
    print(f"space {name}: context {len(ctx)} original rows, reference {len(ref)}, PCA keeps "
          f"{pca.explained_variance_ratio_.sum():.1%}, {(time.time() - start) / 60:.1f} min", flush=True)
    del clf
