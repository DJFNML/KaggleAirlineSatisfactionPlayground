from pathlib import Path

import pandas as pd

from cv import run_cv
from preprocess import load_data

FOLD_OPTIONS = [3, 5, 10]  # number of folds to compare
OUT = Path(__file__).parent / "submissions"

X, y, X_test, test_ids = load_data()

OUT.mkdir(exist_ok=True)
results = {}
for k in FOLD_OPTIONS:
    res = run_cv(X, y, X_test, n_splits=k, label=f"{k}-fold")
    results[k] = res
    pd.DataFrame({"id": test_ids, "satisfaction": res["test_pred"]}).to_csv(
        OUT / f"submission_xgb_{k}fold.csv", index=False
    )

summary = pd.DataFrame(results).T.drop(columns=["oof", "test_pred"]).rename_axis("folds")
print("\n" + summary.astype(float).round(5).to_string())
