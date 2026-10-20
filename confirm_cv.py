"""Final check: 5-fold CV (two model seeds each) for the baseline and the tuned configs."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from cv import PARAMS, TUNED_PARAMS as TUNED, run_cv
from preprocess import load_data

N_SPLITS = 5
SEEDS = [42, 1]
MAX_ROUNDS = 10000
OUT = Path(__file__).parent / "submissions"

CONFIGS = {
    "baseline": PARAMS,
    "tuned_lr0.02": TUNED,
    "tuned_lr0.01": {**TUNED, "learning_rate": 0.01},
    "tuned_lr0.02_sub0.9": {**TUNED, "subsample": 0.9},
}

X, y, X_test, test_ids = load_data()
OUT.mkdir(exist_ok=True)

rows = []
for name, params in CONFIGS.items():
    runs = [
        run_cv(X, y, X_test, N_SPLITS, params={**params, "seed": s},
               label=f"{name} seed {s}", max_rounds=MAX_ROUNDS)
        for s in SEEDS
    ]
    seed_avg_oof = np.mean([r["oof"] for r in runs], axis=0)
    rows.append({
        "config": name,
        **{f"oof_auc_seed{s}": r["oof_auc"] for s, r in zip(SEEDS, runs)},
        "oof_auc_seed_avg": roc_auc_score(y, seed_avg_oof),
        "avg_trees": np.mean([r["avg_trees"] for r in runs]),
        "minutes": sum(r["seconds"] for r in runs) / 60,
    })
    # Submission = test predictions averaged over both seeds (and all folds)
    test_pred = np.mean([r["test_pred"] for r in runs], axis=0)
    pd.DataFrame({"id": test_ids, "satisfaction": test_pred}).to_csv(
        OUT / f"submission_{name}.csv", index=False
    )

summary = pd.DataFrame(rows).set_index("config")
base = summary.loc["baseline", "oof_auc_seed_avg"]
summary.insert(3, "vs_baseline", summary["oof_auc_seed_avg"] - base)
summary.to_csv(Path(__file__).parent / "results" / "confirm_cv.csv")
print("\n" + summary.round(5).to_string())
