"""Confirm the top CatBoost tuning trials with 15-fold CV x 2 seeds against the defaults.

Each config's predictions are saved to results/preds/cat_t{trial}_15fold.npz; the defaults
are already in results/preds/cat_15fold.npz. Writes results/confirm_cat.csv, best first.
"""
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from models import CAT_PARAMS, fit_cat, run_cv_model
from preprocess import load_data

N_SPLITS = 15
SEEDS = [42, 1]
TOP_N = 2  # ~35 min per config, so only the best few
ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name="cat_tuning", storage=f"sqlite:///{ROOT / 'results' / 'cat_tuning.db'}")
complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.number != 0]
top = sorted(complete, key=lambda t: t.value, reverse=True)[:TOP_N]
X, y, X_test, _ = load_data()

files = {"defaults": "cat_15fold"}
for t in top:
    name = f"cat_t{t.number}_{N_SPLITS}fold"
    files[f"trial {t.number}"] = name
    if (PREDS / f"{name}.npz").exists():
        continue
    print(f"trial {t.number} (holdout {t.value:.5f}): {t.params}")
    params = {**CAT_PARAMS, **t.params}
    runs = [
        run_cv_model(X, y, X_test, N_SPLITS, fit_cat({**params, "random_seed": s}), label=f"trial {t.number} seed {s}")
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
summary = summary.sort_values("oof_auc_seed_avg", ascending=False)
summary.to_csv(ROOT / "results" / "confirm_cat.csv")
print("\n" + summary.round(5).to_string())
