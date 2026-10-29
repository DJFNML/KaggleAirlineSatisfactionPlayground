"""Does the teacher column (teacher.py) help TabPFN? Same fold, context rows and 10k eval rows as
tabpfn_context_test.py, key-value-cached, with and without the column.

Usage: python tabpfn_teacher_test.py CONTEXT_SIZE PREDICT_CHUNK
"""
import sys
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from tabpfn import TabPFNClassifier

from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS, tabpfn_matrix

SIZE, CHUNK = int(sys.argv[1]), int(sys.argv[2])
N_EVAL = 10_000

for teacher in (False, True):
    X, y_s, _, _ = load_data(teacher=teacher)
    y = y_s.to_numpy()
    Xp, cat_idx = tabpfn_matrix(X)
    tr_idx, va_idx = next(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
    ev = va_idx[:N_EVAL]
    ctx, _ = train_test_split(tr_idx, train_size=SIZE, stratify=y[tr_idx], random_state=0)
    t = time.time()
    clf = TabPFNClassifier(device="cuda", categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                           random_state=0, fit_mode="fit_with_cache", kv_cache_precision="int8")
    clf.fit(Xp[ctx], y[ctx])
    p = np.concatenate([clf.predict_proba(Xp[ev[i:i + CHUNK]])[:, 1] for i in range(0, N_EVAL, CHUNK)])
    del clf
    print(f"context {SIZE}, teacher={teacher}: AUC {roc_auc_score(y[ev], p):.5f} "
          f"({(time.time() - t) / 60:.1f} min)", flush=True)
