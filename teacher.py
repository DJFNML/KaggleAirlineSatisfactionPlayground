"""Teacher feature: a LightGBM trained only on the original (real) dataset, predicting the competition rows.

The teacher never sees competition labels, so its predictions can be used as a feature on both the
competition train and test sets without leakage. It is trained with 5-fold CV on the original data
(for early stopping) and the 5 fold models' predictions are averaged.
Cached in results/teacher_{nan|zero}.npz.
"""
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import EARLY_STOPPING, SEED
from preprocess import CAT_COLS, load_original

CACHE_DIR = Path(__file__).parent / "results"
PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "num_threads": 8,  # leave CPU for GPU jobs running alongside
    "verbose": -1,
    "seed": SEED,
}


def teacher_predictions(X, X_test, zero_ratings_as_nan=True):
    """Teacher probabilities for the competition train and test rows (cached)."""
    cache = CACHE_DIR / f"teacher_{'nan' if zero_ratings_as_nan else 'zero'}.npz"
    if cache.exists():
        c = np.load(cache)
        if len(c["train"]) == len(X) and len(c["test"]) == len(X_test):
            return c["train"], c["test"]

    X_orig, y_orig = load_original(zero_ratings_as_nan)
    for col in CAT_COLS:  # same category codes as the competition frames
        X_orig[col] = pd.Categorical(X_orig[col], categories=X[col].cat.categories)
    X_orig = X_orig[X.columns.drop("teacher", errors="ignore")]

    oof = np.zeros(len(X_orig))
    train_pred, test_pred = np.zeros(len(X)), np.zeros(len(X_test))
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for tr, va in skf.split(X_orig, y_orig):
        dtr = lgb.Dataset(X_orig.iloc[tr], y_orig.iloc[tr])
        dva = lgb.Dataset(X_orig.iloc[va], y_orig.iloc[va], reference=dtr)
        model = lgb.train(PARAMS, dtr, 10000, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        oof[va] = model.predict(X_orig.iloc[va], num_iteration=model.best_iteration)
        train_pred += model.predict(X[X_orig.columns], num_iteration=model.best_iteration) / 5
        test_pred += model.predict(X_test[X_orig.columns], num_iteration=model.best_iteration) / 5

    print(f"teacher: OOF AUC on original {roc_auc_score(y_orig, oof):.5f} "
          f"({len(X_orig)} rows) -> cached {cache.name}", flush=True)
    np.savez(cache, train=train_pred, test=test_pred)
    return train_pred, test_pred
