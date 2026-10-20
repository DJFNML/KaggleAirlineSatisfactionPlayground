"""Phase 1: sweep each hyperparameter on its own, holding the rest at the baseline."""
from pathlib import Path

import pandas as pd

from cv import PARAMS, make_holdout, run_holdout
from preprocess import load_data

MAX_ROUNDS = 10000  # low learning rates need many more trees
OUT = Path(__file__).parent / "results"

SWEEPS = {
    "learning_rate": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10],
    "max_depth": [4, 5, 6, 7, 8, 9, 10],
    "min_child_weight": [1, 2, 5, 10, 20, 50, 100],
    "gamma": [0, 0.05, 0.1, 0.25, 0.5, 1, 2],
    "subsample": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "reg_lambda": [0, 0.1, 1, 3, 10, 30, 100],
    "reg_alpha": [0, 0.1, 1, 3, 10, 30],
}
NOISE_SEEDS = [1, 2, 3]  # baseline with other model seeds = noise floor

X, y, _, _ = load_data()
dtrain, dvalid = make_holdout(X, y)
OUT.mkdir(exist_ok=True)

runs = [("baseline", None, PARAMS)]
runs += [("noise", s, {**PARAMS, "seed": s}) for s in NOISE_SEEDS]
runs += [(p, v, {**PARAMS, p: v}) for p, values in SWEEPS.items() for v in values]

rows = []
for i, (param, value, params) in enumerate(runs, start=1):
    res = run_holdout(dtrain, dvalid, params, MAX_ROUNDS)
    rows.append({"param": param, "value": value, **res})
    print(f"[{i}/{len(runs)}] {param}={value}: AUC={res['auc']:.5f}, "
          f"trees={res['trees']}, {res['seconds']:.0f}s")
    pd.DataFrame(rows).to_csv(OUT / "phase1_sweeps.csv", index=False)  # save as we go

df = pd.DataFrame(rows)
base = df.loc[df["param"] == "baseline", "auc"].iloc[0]
df["vs_baseline"] = df["auc"] - base
df.to_csv(OUT / "phase1_sweeps.csv", index=False)
print("\n" + df.round(5).to_string(index=False))
