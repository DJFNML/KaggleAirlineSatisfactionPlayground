"""Confirm the top LightGBM tuning trials with 15-fold CV x 2 seeds against the defaults.

Each config's predictions are saved to results/preds/lgb_t{trial}_15fold.npz (same format
as train_models.py); the defaults are already in results/preds/lgb_15fold.npz.
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from models import LGB_PARAMS, fit_lgb, run_cv_model
from preprocess import load_data

N_SPLITS = 15
SEEDS = [42, 1]
TRIALS = [119, 67, 114]
ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name="lgb_tuning", storage=f"sqlite:///{ROOT / 'results' / 'lgb_tuning.db'}")
X, y, X_test, _ = load_data()

files = {"defaults": "lgb_15fold"}
for t in TRIALS:
    name = f"lgb_t{t}_{N_SPLITS}fold"
    files[f"trial {t}"] = name
    if (PREDS / f"{name}.npz").exists():
        continue
    params = {**LGB_PARAMS, **study.trials[t].params}
    runs = [
        run_cv_model(X, y, X_test, N_SPLITS, fit_lgb({**params, "seed": s}), label=f"trial {t} seed {s}")
        for s in SEEDS
    ]
    oof_seeds = np.array([r["oof"] for r in runs])
    np.savez(PREDS / f"{name}.npz", oof=oof_seeds.mean(axis=0),
             test=np.mean([r["test_pred"] for r in runs], axis=0), oof_seeds=oof_seeds)

rows = []
for label, name in files.items():
    s = np.load(PREDS / f"{name}.npz")
    seed_aucs = [roc_auc_score(y, o) for o in s["oof_seeds"]]
    rows.append({"config": label, "file": name, "oof_auc_seed42": seed_aucs[0], "oof_auc_seed1": seed_aucs[1],
                 "oof_auc_seed_avg": roc_auc_score(y, s["oof"]), "seed_spread": max(seed_aucs) - min(seed_aucs)})

summary = pd.DataFrame(rows).set_index("config")
summary.insert(4, "vs_defaults", summary["oof_auc_seed_avg"] - summary.loc["defaults", "oof_auc_seed_avg"])
summary.to_csv(ROOT / "results" / "confirm_lgb.csv")
print("\n" + summary.round(5).to_string())
