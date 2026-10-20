"""LightGBM and CatBoost, cross-validated on exactly the same folds as cv.run_cv (XGBoost)."""
import time

import lightgbm as lgb
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import EARLY_STOPPING, SEED
from preprocess import CAT_COLS

MAX_ROUNDS = 10000

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 127,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,  # row sampling only happens when this is > 0
    "verbose": -1,
    "seed": SEED,
}
CAT_PARAMS = {
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "learning_rate": 0.05,  # set explicitly: CatBoost's automatic rate assumes no early stopping
    "depth": 6,
    "l2_leaf_reg": 3,
    "iterations": MAX_ROUNDS,
    "od_type": "Iter",
    "od_wait": EARLY_STOPPING,
    "use_best_model": True,
    "task_type": "GPU",
    "random_seed": SEED,
    "verbose": 0,
}


def fit_lgb(params):
    """Return a fit function: (X_tr, y_tr, X_va, y_va, X_test) -> (valid_pred, test_pred, trees)."""
    def fit(X_tr, y_tr, X_va, y_va, X_test):
        dtrain = lgb.Dataset(X_tr, y_tr)
        dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
        model = lgb.train(
            params, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )
        best = model.best_iteration
        return model.predict(X_va, num_iteration=best), model.predict(X_test, num_iteration=best), best
    return fit


def fit_cat(params):
    """Return a fit function: (X_tr, y_tr, X_va, y_va, X_test) -> (valid_pred, test_pred, trees)."""
    def fit(X_tr, y_tr, X_va, y_va, X_test):
        valid = Pool(X_va, y_va, cat_features=CAT_COLS)
        model = CatBoostClassifier(**params)
        model.fit(Pool(X_tr, y_tr, cat_features=CAT_COLS), eval_set=valid)
        test = Pool(X_test, cat_features=CAT_COLS)
        return model.predict_proba(valid)[:, 1], model.predict_proba(test)[:, 1], model.get_best_iteration() + 1
    return fit


def run_cv_model(X, y, X_test, n_splits, fit, label=""):
    """Stratified K-fold with any model's fit function; same folds as cv.run_cv."""
    start = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs, trees = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        va_pred, te_pred, n_trees = fit(X.iloc[tr_idx], y.iloc[tr_idx], X.iloc[va_idx], y.iloc[va_idx], X_test)
        oof[va_idx] = va_pred
        test_pred += te_pred / n_splits
        fold_aucs.append(roc_auc_score(y.iloc[va_idx], va_pred))
        trees.append(n_trees)
        print(f"  [{label}] fold {fold}/{n_splits}: AUC={fold_aucs[-1]:.5f}, trees={n_trees}")

    return {
        "oof_auc": roc_auc_score(y, oof),
        "avg_trees": np.mean(trees),
        "seconds": time.time() - start,
        "oof": oof,
        "test_pred": test_pred,
    }
