"""Tuned XGBoost / LightGBM / CatBoost on hand-picked feature subsets, for the blend.

Subsets (the user's choice of features that are likely to act together):
  long  - long-distance flights only (Flight Distance above the LONG_QUANTILE of training flights): trained
          and applied only on those rows; in-flight experience features.
  route - all rows; route / airport / punctuality features.
Every subset uses the global 15-fold split, restricted to its rows, so its OOF predictions line up with the
other models. Rows outside a subset get NaN.

Usage: python subset_models.py tune MODEL SUBSET N_TRIALS   - Optuna on an 80/20 holdout inside the subset
       python subset_models.py cv MODEL SUBSET              - 15-fold CV with the study's best settings
MODEL: xgb | lgb | cat. Studies in results/subset_tuning.db ({subset}_{model}); predictions in
results/preds/subset_{subset}_{model}.npz (oof, test, mask, test_mask).
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv import EARLY_STOPPING, PARAMS as XGB_BASE, SEED
from models import CAT_PARAMS, LGB_PARAMS
from preprocess import ARR_DELAY, DEP_DELAY, load_data

LONG_QUANTILE = 0.75
FEATURES = {
    "long": ["Flight Distance", "Inflight wifi service", "Departure/Arrival time convenient", "Food and drink",
             "Seat comfort", "Inflight entertainment", "On-board service", "Leg room service", "Cleanliness",
             ARR_DELAY, "Class"],
    "route": ["Flight Distance", "Departure/Arrival time convenient", "Ease of Online booking", "Gate location",
              "Online boarding", "Seat comfort", "Baggage handling", "Cleanliness", DEP_DELAY, ARR_DELAY, "Class"],
}
MAX_ROUNDS = 10_000
ROOT = Path(__file__).parent
STORAGE = f"sqlite:///{ROOT / 'results' / 'subset_tuning.db'}"
N_SPLITS = 15

X_all, y_s, X_test_all, _ = load_data()
y = y_s.to_numpy()
cut = X_all["Flight Distance"].quantile(LONG_QUANTILE)
MASKS = {"long": (X_all["Flight Distance"] > cut).to_numpy(), "route": np.ones(len(y), dtype=bool)}
TEST_MASKS = {"long": (X_test_all["Flight Distance"] > cut).to_numpy(), "route": np.ones(len(X_test_all), dtype=bool)}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def suggest(trial, model):
    if model == "xgb":
        return {"max_depth": trial.suggest_int("max_depth", 3, 12),
                "min_child_weight": trial.suggest_float("min_child_weight", 1, 200, log=True),
                "gamma": trial.suggest_float("gamma", 0, 2),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0, step=0.05),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10, log=True)}
    if model == "lgb":
        return {"num_leaves": trial.suggest_int("num_leaves", 7, 1023, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 1000, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0, step=0.05),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0, step=0.05),
                "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 30, log=True),
                "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
                "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 2)}
    p = {"depth": trial.suggest_int("depth", 4, 10),
         "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
         "random_strength": trial.suggest_float("random_strength", 0.01, 10, log=True),
         "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
         "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli"])}
    if p["bootstrap_type"] == "Bayesian":
        p["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        p["subsample"] = trial.suggest_float("subsample", 0.5, 1.0, step=0.05)
    return p


def fit_predict(model, params, X_tr, y_tr, X_va, y_va, others, seed=SEED):
    """Train with early stopping on (X_va, y_va); return (valid pred, [preds for each frame in others], trees)."""
    cats = [c for c in X_tr.columns if X_tr[c].dtype == "category"]
    if model == "xgb":
        p = {**XGB_BASE, "learning_rate": 0.02, **params, "seed": seed}
        dtr = xgb.DMatrix(X_tr, y_tr, enable_categorical=True)
        dva = xgb.DMatrix(X_va, y_va, enable_categorical=True)
        m = xgb.train(p, dtr, MAX_ROUNDS, evals=[(dva, "valid")], early_stopping_rounds=EARLY_STOPPING,
                      verbose_eval=False)
        rng = (0, m.best_iteration + 1)
        pred = lambda X: m.predict(xgb.DMatrix(X, enable_categorical=True), iteration_range=rng)  # noqa: E731
        return pred(X_va), [pred(X) for X in others], m.best_iteration + 1
    if model == "lgb":
        p = {**LGB_PARAMS, **params, "seed": seed}
        dtr = lgb.Dataset(X_tr, y_tr)
        m = lgb.train(p, dtr, MAX_ROUNDS, valid_sets=[lgb.Dataset(X_va, y_va, reference=dtr)],
                      callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)])
        return (m.predict(X_va, num_iteration=m.best_iteration),
                [m.predict(X, num_iteration=m.best_iteration) for X in others], m.best_iteration)
    m = CatBoostClassifier(**{**CAT_PARAMS, **params, "random_seed": seed})
    va = Pool(X_va, y_va, cat_features=cats)
    m.fit(Pool(X_tr, y_tr, cat_features=cats), eval_set=va)
    return (m.predict_proba(va)[:, 1], [m.predict_proba(Pool(X, cat_features=cats))[:, 1] for X in others],
            m.get_best_iteration() + 1)


def tune(model, subset, n_trials):
    rows = np.where(MASKS[subset])[0]
    X = X_all[FEATURES[subset]].iloc[rows]
    X_tr, X_va, y_tr, y_va = train_test_split(X, y[rows], test_size=0.2, stratify=y[rows], random_state=SEED)
    name = f"{subset}_{model}"
    n_existing = len(optuna.load_study(study_name=name, storage=STORAGE).trials) \
        if name in optuna.get_all_study_names(STORAGE) else 0
    study = optuna.create_study(study_name=name, storage=STORAGE, load_if_exists=True, direction="maximize",
                                sampler=optuna.samplers.TPESampler(n_startup_trials=15, seed=SEED + n_existing))

    def objective(trial):
        start = time.time()
        pred, _, trees = fit_predict(model, suggest(trial, model), X_tr, y_tr, X_va, y_va, [])
        auc = roc_auc_score(y_va, pred)
        trial.set_user_attr("trees", trees)
        log(f"{name} trial {trial.number}: AUC {auc:.5f}, {trees} trees, {time.time() - start:.0f}s")
        return auc

    done = len(study.get_trials(states=(optuna.trial.TrialState.COMPLETE,)))
    study.optimize(objective, n_trials=max(n_trials - done, 0))
    log(f"{name}: best trial {study.best_trial.number} AUC {study.best_value:.5f} {study.best_params}")


def run_cv(model, subset):
    name = f"{subset}_{model}"
    params = optuna.load_study(study_name=name, storage=STORAGE).best_params
    mask, test_mask = MASKS[subset], TEST_MASKS[subset]
    X, X_test = X_all[FEATURES[subset]], X_test_all[FEATURES[subset]].iloc[np.where(test_mask)[0]]
    oof, test = np.full(len(y), np.nan), np.zeros(test_mask.sum())
    start = time.time()
    for k, (tr, va) in enumerate(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X_all, y), 1):
        tr, va = tr[mask[tr]], va[mask[va]]
        pred, (te,), trees = fit_predict(model, params, X.iloc[tr], y[tr], X.iloc[va], y[va], [X_test])
        oof[va], test = pred, test + te / N_SPLITS
        log(f"{name} fold {k}/{N_SPLITS}: AUC {roc_auc_score(y[va], pred):.5f}, {trees} trees")
    full_test = np.full(len(test_mask), np.nan)
    full_test[test_mask] = test
    out = ROOT / "results" / "preds" / f"subset_{name}.npz"
    np.savez(out, oof=oof, test=full_test, mask=mask, test_mask=test_mask)
    log(f"{name}: OOF AUC {roc_auc_score(y[mask], oof[mask]):.5f} on {mask.sum()} rows, "
        f"{(time.time() - start) / 60:.1f} min -> {out.name}")


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    mode, model, subset = sys.argv[1:4]
    log(f"{subset}: {MASKS[subset].sum()} training rows" + (f" (Flight Distance > {cut:.0f})" if subset == "long" else "")
        + f", features {FEATURES[subset]}")
    tune(model, subset, int(sys.argv[4])) if mode == "tune" else run_cv(model, subset)
