from pathlib import Path

import pandas as pd

from cv import PARAMS, run_cv
from preprocess import load_data

N_SPLITS = 5
OUT = Path(__file__).parent / "submissions"

# Each experiment = (keyword arguments for load_data(), XGBoost params)
EXPERIMENTS = {
    "baseline": ({"delay_thresholds": ()}, PARAMS),
    # Same data, different model seed: shows how much AUC moves from randomness alone
    "baseline (seed 1)": ({"delay_thresholds": ()}, {**PARAMS, "seed": 1}),
    "+delay_over_0": ({"delay_thresholds": (0,)}, PARAMS),
    "+delay_over_5": ({"delay_thresholds": (5,)}, PARAMS),
    "+delay_over_10": ({"delay_thresholds": (10,)}, PARAMS),
    "+delay_over_15": ({"delay_thresholds": (15,)}, PARAMS),
    "+delay_over_20": ({"delay_thresholds": (20,)}, PARAMS),
    "+delay_over_30": ({"delay_thresholds": (30,)}, PARAMS),
    "+delay_over_60": ({"delay_thresholds": (60,)}, PARAMS),
}

results = {}
for name, (data_kwargs, params) in EXPERIMENTS.items():
    X, y, X_test, test_ids = load_data(**data_kwargs)
    results[name] = run_cv(X, y, X_test, N_SPLITS, params=params, label=name)

summary = pd.DataFrame(results).T[["oof_auc", "fold_auc_std", "avg_trees", "seconds"]].astype(float)
summary.insert(1, "vs_baseline", summary["oof_auc"] - summary.loc["baseline", "oof_auc"])
print("\n" + summary.round(5).to_string())

# Save a submission for the best experiment
best = summary["oof_auc"].idxmax()
OUT.mkdir(exist_ok=True)
pd.DataFrame({"id": test_ids, "satisfaction": results[best]["test_pred"]}).to_csv(
    OUT / "submission_xgb_best_features.csv", index=False
)
print(f"\nBest: {best} -> submissions/submission_xgb_best_features.csv")
