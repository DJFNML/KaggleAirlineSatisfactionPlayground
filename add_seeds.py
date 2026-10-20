"""Add seeds to a model's saved 2-seed predictions (15-fold) and save a 5-seed version.

Usage: python add_seeds.py xgb|lgb|cat
Reads results/preds/{source}.npz (seeds 42 and 1), trains seeds 2, 3, 4 with the same
settings and folds, and saves results/preds/{source}_5seeds.npz (same format).
"""
import sys
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score

from cv import TUNED_PARAMS, run_cv
from models import CAT_PARAMS, LGB_PARAMS, fit_cat, fit_lgb, run_cv_model
from preprocess import load_data

MODEL = sys.argv[1]
N_SPLITS = 15
OLD_SEEDS = [42, 1]
NEW_SEEDS = [2, 3, 4]
ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"
SOURCES = {"xgb": "xgb_tuned_15fold", "lgb": "lgb_t119_15fold", "cat": "cat_t92_15fold"}

optuna.logging.set_verbosity(optuna.logging.WARNING)


def trial_params(study_name, number):
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{ROOT / 'results' / study_name}.db")
    return study.trials[number].params


X, y, X_test, _ = load_data()


def run_seed(seed):
    label = f"{MODEL} seed {seed}"
    if MODEL == "xgb":
        return run_cv(X, y, X_test, N_SPLITS, params={**TUNED_PARAMS, "seed": seed}, label=label, max_rounds=10000)
    if MODEL == "lgb":
        params = {**LGB_PARAMS, **trial_params("lgb_tuning", 119), "seed": seed}
        return run_cv_model(X, y, X_test, N_SPLITS, fit_lgb(params), label=label)
    if MODEL == "cat":
        params = {**CAT_PARAMS, **trial_params("cat_tuning", 92), "random_seed": seed}
        return run_cv_model(X, y, X_test, N_SPLITS, fit_cat(params), label=label)
    raise ValueError(f"Unknown model: {MODEL}")


old = np.load(PREDS / f"{SOURCES[MODEL]}.npz")
runs = [run_seed(s) for s in NEW_SEEDS]

oof_seeds = np.concatenate([old["oof_seeds"], [r["oof"] for r in runs]])
# Saved test predictions are already a 2-seed mean, so weight them by 2
test = (old["test"] * len(OLD_SEEDS) + sum(r["test_pred"] for r in runs)) / (len(OLD_SEEDS) + len(NEW_SEEDS))
oof = oof_seeds.mean(axis=0)
out = PREDS / f"{SOURCES[MODEL]}_5seeds.npz"
np.savez(out, oof=oof, test=test, oof_seeds=oof_seeds)

seed_aucs = [roc_auc_score(y, o) for o in oof_seeds]
print(f"\n{MODEL}: per-seed {', '.join(f'{a:.5f}' for a in seed_aucs)}")
print(f"{MODEL}: 2-seed avg {roc_auc_score(y, old['oof']):.5f} -> 5-seed avg {roc_auc_score(y, oof):.5f} -> {out.name}")
