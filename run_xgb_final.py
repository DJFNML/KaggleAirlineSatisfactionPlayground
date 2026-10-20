"""Tuned XGBoost with several fold counts (two seeds each); save OOF + test predictions for blending.

Saves the best fold count plus any within seed noise of it. Fold counts that were
already saved are loaded instead of retrained.
Predictions go to results/preds/xgb_tuned_{k}fold.npz with arrays:
  oof        - seed-averaged out-of-fold predictions, in train.csv row order
  test       - seed-averaged test predictions, in test.csv row order
  oof_seeds  - per-seed OOF predictions (for checking seed noise later)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv import TUNED_PARAMS, run_cv
from preprocess import load_data

FOLD_OPTIONS = [10, 15]
SEEDS = [42, 1]
MAX_ROUNDS = 10000
ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"

X, y, X_test, test_ids = load_data()

results = {}
for k in FOLD_OPTIONS:
    path = PREDS / f"xgb_tuned_{k}fold.npz"
    if path.exists():
        saved = np.load(path)
        oof_seeds, oof, test = saved["oof_seeds"], saved["oof"], saved["test"]
        trees = minutes = np.nan
        print(f"Loaded saved {k}-fold predictions")
    else:
        runs = [
            run_cv(X, y, X_test, k, params={**TUNED_PARAMS, "seed": s},
                   label=f"{k}-fold seed {s}", max_rounds=MAX_ROUNDS)
            for s in SEEDS
        ]
        oof_seeds = np.array([r["oof"] for r in runs])
        oof = oof_seeds.mean(axis=0)
        test = np.mean([r["test_pred"] for r in runs], axis=0)
        trees = np.mean([r["avg_trees"] for r in runs])
        minutes = sum(r["seconds"] for r in runs) / 60

    seed_aucs = [roc_auc_score(y, o) for o in oof_seeds]
    results[k] = {
        **{f"oof_auc_seed{s}": a for s, a in zip(SEEDS, seed_aucs)},
        "oof_auc_seed_avg": roc_auc_score(y, oof),
        "seed_spread": max(seed_aucs) - min(seed_aucs),
        "avg_trees": trees,
        "minutes": minutes,
        "oof": oof,
        "test": test,
        "oof_seeds": oof_seeds,
    }

summary = pd.DataFrame(results).T.drop(columns=["oof", "test", "oof_seeds"]).astype(float)
summary.index.name = "folds"
print("\n" + summary.round(5).to_string())

# Keep the best, plus any fold count whose gap to the best is within the largest seed spread
best = summary["oof_auc_seed_avg"].idxmax()
noise = summary["seed_spread"].max()
to_save = []
for k in FOLD_OPTIONS:
    gap = summary.loc[k, "oof_auc_seed_avg"] - summary.loc[best, "oof_auc_seed_avg"]
    keep = k == best or abs(gap) <= noise
    print(f"{k}-fold vs best ({best}-fold): {gap:+.5f} (noise {noise:.5f}) -> {'keep' if keep else 'drop'}")
    if keep:
        to_save.append(k)

PREDS.mkdir(parents=True, exist_ok=True)
for k in to_save:
    path = PREDS / f"xgb_tuned_{k}fold.npz"
    if path.exists():
        continue
    r = results[k]
    np.savez(path, oof=r["oof"], test=r["test"], oof_seeds=r["oof_seeds"])
    pd.DataFrame({"id": test_ids, "satisfaction": r["test"]}).to_csv(
        ROOT / "submissions" / f"submission_xgb_tuned_{k}fold.csv", index=False
    )
    print(f"Saved results/preds/xgb_tuned_{k}fold.npz and submissions/submission_xgb_tuned_{k}fold.csv")
