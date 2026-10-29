"""Honest logistic-regression stack on the members' log-odds.

The stacker itself is cross-validated (meta folds StratifiedKFold(5, seed 7)): each row's stacked score
comes from a stacker that never saw that row. Members can come from different fold schemes (15-fold
TabPFN, 5-fold ladder models): every member column is out of fold for every row.
Reports the nested CV, each member's coefficient and what leaving it out costs, and with name=NAME writes
submissions/submission_{NAME}.csv from a stacker fitted on all rows.

Usage: python stack_lr.py member [member ...] [name=NAME] [C=1.0]   (members are results/preds/{member}.npz)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_data

ROOT = Path(__file__).parent
args = sys.argv[1:]
NAME = next((a[5:] for a in args if a.startswith("name=")), None)
C = float(next((a[2:] for a in args if a.startswith("C=")), 1.0))
MEMBERS = [a for a in args if "=" not in a]

_, y_s, _, test_ids = load_data()
y = y_s.to_numpy()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


preds = {m: np.load(ROOT / "results" / "preds" / f"{m}.npz") for m in MEMBERS}
Z = np.column_stack([logit(preds[m]["oof"]) for m in MEMBERS])
ZT = np.column_stack([logit(preds[m]["test"]) for m in MEMBERS])
META = list(StratifiedKFold(5, shuffle=True, random_state=7).split(Z, y))


def nested(cols):
    out = np.zeros(len(y))
    for a, b in META:
        out[b] = LogisticRegression(C=C, max_iter=2000).fit(Z[a][:, cols], y[a]).decision_function(Z[b][:, cols])
    return roc_auc_score(y, out), [roc_auc_score(y[b], out[b]) for _, b in META]


all_cols = list(range(len(MEMBERS)))
full, folds = nested(all_cols)
coef = LogisticRegression(C=C, max_iter=2000).fit(Z, y).coef_[0]
print(f"nested CV, {len(MEMBERS)} members: {full:.5f}  (meta folds {' '.join(f'{f:.5f}' for f in folds)})")
for i, m in enumerate(MEMBERS):
    alone = roc_auc_score(y, preds[m]["oof"])
    without, fw = nested([j for j in all_cols if j != i])
    up = sum(a > b for a, b in zip(folds, fw))
    print(f"  {m:40s} alone {alone:.5f} | coef {coef[i]:+.3f} | leaving it out {without - full:+.6f} "
          f"(it helps in {up}/5 meta folds)")
if NAME:
    m = LogisticRegression(C=C, max_iter=2000).fit(Z, y)
    p = 1 / (1 + np.exp(-m.decision_function(ZT)))
    path = ROOT / "submissions" / f"submission_{NAME}.csv"
    pd.DataFrame({"id": test_ids, "satisfaction": p}).to_csv(path, index=False)
    print(f"-> {path.name} (test mean {p.mean():.4f}, OOF positive rate {y.mean():.4f})")
