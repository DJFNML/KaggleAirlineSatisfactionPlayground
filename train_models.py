"""Train LightGBM and CatBoost (two seeds each) and save OOF + test predictions for blending.

Same file format as run_xgb_final.py: results/preds/{model}_{k}fold.npz with oof, test, oof_seeds.
Models already saved are skipped.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from models import CAT_PARAMS, LGB_PARAMS, fit_cat, fit_lgb, run_cv_model
from preprocess import load_data

N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10  # must match the XGBoost predictions
SEEDS = [42, 1]
PREDS = Path(__file__).parent / "results" / "preds"

# name -> (fit factory, base params, name of the seed parameter)
MODELS = {
    "lgb": (fit_lgb, LGB_PARAMS, "seed"),
    "cat": (fit_cat, CAT_PARAMS, "random_seed"),
}

X, y, X_test, _ = load_data()
PREDS.mkdir(parents=True, exist_ok=True)

for name, (make_fit, params, seed_key) in MODELS.items():
    path = PREDS / f"{name}_{N_SPLITS}fold.npz"
    if path.exists():
        print(f"{path.name} already exists, skipping")
        continue

    runs = [
        run_cv_model(X, y, X_test, N_SPLITS, make_fit({**params, seed_key: s}), label=f"{name} seed {s}")
        for s in SEEDS
    ]
    oof_seeds = np.array([r["oof"] for r in runs])
    oof = oof_seeds.mean(axis=0)
    np.savez(path, oof=oof, test=np.mean([r["test_pred"] for r in runs], axis=0), oof_seeds=oof_seeds)

    seed_aucs = ", ".join(f"{r['oof_auc']:.5f}" for r in runs)
    print(f"{name}: seeds {seed_aucs} | seed avg {roc_auc_score(y, oof):.5f} | "
          f"trees {np.mean([r['avg_trees'] for r in runs]):.0f} | "
          f"{sum(r['seconds'] for r in runs) / 60:.1f} min -> {path.name}")
