import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

SEED = 42
PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "tree_method": "hist",
    "device": "cuda",  # use "cpu" if no GPU
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": SEED,
}
# Result of Phases 1-3 (confirmed with 5-fold CV in confirm_cv.py)
TUNED_PARAMS = {
    **PARAMS,
    "learning_rate": 0.02,
    "max_depth": 10,
    "min_child_weight": 34,
    "gamma": 0.19,
    "subsample": 0.8,
    "colsample_bytree": 0.65,
    "reg_lambda": 0.14,
    "reg_alpha": 0.002,
}
MAX_ROUNDS = 5000
EARLY_STOPPING = 100


def run_cv(X, y, X_test, n_splits=5, params=PARAMS, label="", max_rounds=MAX_ROUNDS):
    """Stratified K-fold XGBoost; return scores and fold-averaged test predictions."""
    start = time.time()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    dtest = xgb.DMatrix(X_test, enable_categorical=True)
    oof = np.zeros(len(X))
    test_pred = np.zeros(len(X_test))
    fold_aucs, best_iters = [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), start=1):
        dtrain = xgb.DMatrix(X.iloc[tr_idx], y.iloc[tr_idx], enable_categorical=True)
        dvalid = xgb.DMatrix(X.iloc[va_idx], y.iloc[va_idx], enable_categorical=True)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=max_rounds,
            evals=[(dvalid, "valid")],
            early_stopping_rounds=EARLY_STOPPING,
            verbose_eval=False,
        )
        best_range = (0, model.best_iteration + 1)
        oof[va_idx] = model.predict(dvalid, iteration_range=best_range)
        test_pred += model.predict(dtest, iteration_range=best_range) / n_splits

        fold_aucs.append(roc_auc_score(y.iloc[va_idx], oof[va_idx]))
        best_iters.append(model.best_iteration + 1)
        print(f"  [{label}] fold {fold}/{n_splits}: AUC={fold_aucs[-1]:.5f}, trees={best_iters[-1]}")

    return {
        "oof_auc": roc_auc_score(y, oof),
        "fold_auc_std": np.std(fold_aucs),
        "avg_trees": np.mean(best_iters),
        "seconds": time.time() - start,
        "oof": oof,
        "test_pred": test_pred,
    }


def make_holdout(X, y, valid_size=0.2):
    """Fixed stratified split as DMatrices, built once and reused across tuning trials."""
    X_tr, X_va, y_tr, y_va = train_test_split(
        X, y, test_size=valid_size, stratify=y, random_state=SEED
    )
    dtrain = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
    dvalid = xgb.DMatrix(X_va, y_va, enable_categorical=True)
    return dtrain, dvalid


def run_holdout(dtrain, dvalid, params, max_rounds=MAX_ROUNDS):
    """Train one model on a fixed holdout split; return validation AUC and tree count."""
    start = time.time()
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=max_rounds,
        evals=[(dvalid, "valid")],
        early_stopping_rounds=EARLY_STOPPING,
        verbose_eval=False,
    )
    return {
        "auc": model.best_score,
        "trees": model.best_iteration + 1,
        "seconds": time.time() - start,
    }
