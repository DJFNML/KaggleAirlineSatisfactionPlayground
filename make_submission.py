"""Rank-blend chosen models: best weights on all OOF rows, an honest score, and a submission file.

Usage: python make_submission.py NAME xgb=xgb_tuned_15fold_5seeds lgb=lgb_t119_15fold_5seeds ...
Each model=file refers to results/preds/{file}.npz. Writes submissions/submission_{NAME}.csv.
The honest score picks weights on one half of the rows and scores them on the other half.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_data

STEP = 0.05
ROOT = Path(__file__).parent
NAME = sys.argv[1]
MODELS = dict(arg.split("=", 1) for arg in sys.argv[2:])

_, y_s, _, test_ids = load_data()
y = y_s.to_numpy()
saved = {m: np.load(ROOT / "results" / "preds" / f"{f}.npz") for m, f in MODELS.items()}
ranks = {m: (rankdata(s["oof"]) / len(y), rankdata(s["test"]) / len(s["test"])) for m, s in saved.items()}
P = np.column_stack([ranks[m][0] for m in MODELS])
T = np.column_stack([ranks[m][1] for m in MODELS])

for m, s in saved.items():
    print(f"{m}: OOF AUC {roc_auc_score(y, s['oof']):.5f}")

steps = round(1 / STEP)
grid = [np.array(w) / steps for w in product(range(steps + 1), repeat=len(MODELS)) if sum(w) == steps]


def best(rows):
    return max(grid, key=lambda w: roc_auc_score(y[rows], P[rows] @ w))


honest = np.zeros(len(y))
for fit_rows, score_rows in StratifiedKFold(2, shuffle=True, random_state=0).split(y, y):
    honest[score_rows] = P[score_rows] @ best(fit_rows)
w = best(np.arange(len(y)))

print(f"\nweights: {' / '.join(f'{m}={x:.2f}' for m, x in zip(MODELS, w))}")
print(f"OOF AUC {roc_auc_score(y, P @ w):.5f} | honest {roc_auc_score(y, honest):.5f}")
out = ROOT / "submissions" / f"submission_{NAME}.csv"
pd.DataFrame({"id": test_ids, "satisfaction": T @ w}).to_csv(out, index=False)
print(f"-> {out.relative_to(ROOT)}")
