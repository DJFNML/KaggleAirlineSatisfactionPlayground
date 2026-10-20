"""Ratings as numbers vs categories: 5-fold CV, two seeds, tuned XGBoost settings."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv import TUNED_PARAMS, run_cv
from preprocess import load_data

N_SPLITS = 5
SEEDS = [42, 1]
MAX_ROUNDS = 10000
MODES = ["category", "both"]  # "numeric" = tuned_lr0.02 in results/confirm_cv.csv
OUT = Path(__file__).parent / "submissions"

rows = []
for mode in MODES:
    X, y, X_test, test_ids = load_data(ratings=mode)
    runs = [
        run_cv(X, y, X_test, N_SPLITS, params={**TUNED_PARAMS, "seed": s},
               label=f"ratings={mode} seed {s}", max_rounds=MAX_ROUNDS)
        for s in SEEDS
    ]
    rows.append({
        "ratings": mode,
        **{f"oof_auc_seed{s}": r["oof_auc"] for s, r in zip(SEEDS, runs)},
        "oof_auc_seed_avg": roc_auc_score(y, np.mean([r["oof"] for r in runs], axis=0)),
        "avg_trees": np.mean([r["avg_trees"] for r in runs]),
        "minutes": sum(r["seconds"] for r in runs) / 60,
    })
    test_pred = np.mean([r["test_pred"] for r in runs], axis=0)
    pd.DataFrame({"id": test_ids, "satisfaction": test_pred}).to_csv(
        OUT / f"submission_tuned_ratings_{mode}.csv", index=False
    )

# Add the numeric result from the confirmation run for comparison
numeric = pd.read_csv(Path(__file__).parent / "results" / "confirm_cv.csv", index_col="config")
numeric = numeric.loc[["tuned_lr0.02"]].rename(index={"tuned_lr0.02": "numeric"})
summary = pd.concat([numeric.drop(columns="vs_baseline"), pd.DataFrame(rows).set_index("ratings")])
summary.insert(3, "vs_numeric", summary["oof_auc_seed_avg"] - summary.loc["numeric", "oof_auc_seed_avg"])
print("\n" + summary.round(5).to_string())
