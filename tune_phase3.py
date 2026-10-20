"""Phase 3: real 2D grids for the strongest hyperparameter pairs from Phase 2."""
import json
from pathlib import Path

import pandas as pd

from cv import PARAMS, make_holdout, run_holdout
from preprocess import load_data

MAX_ROUNDS = 10000
OUT = Path(__file__).parent / "results"
RESULTS = OUT / "phase3_grids.csv"

# Phase 2 best, with learning rate fixed at 0.02; each grid varies two of these
CENTER = {
    **PARAMS,
    "learning_rate": 0.02,
    "max_depth": 10,
    "min_child_weight": 34,
    "gamma": 0.19,
    "subsample": 0.8,
    "colsample_bytree": 0.65,
    "reg_lambda": 0.14,
    "reg_alpha": 0.002,
}
DEPTHS = [10, 11, 12, 13, 14]
GRIDS = {
    ("max_depth", "subsample"): (DEPTHS, [0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]),
    ("max_depth", "gamma"): (DEPTHS, [0, 0.1, 0.2, 0.4, 0.8, 1.5]),
    ("max_depth", "reg_alpha"): (DEPTHS, [0, 0.001, 0.01, 0.1, 1, 10]),
    ("max_depth", "colsample_bytree"): (DEPTHS, [0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]),
    ("max_depth", "min_child_weight"): (DEPTHS, [5, 10, 20, 30, 40, 60, 100]),
    ("colsample_bytree", "reg_alpha"): ([0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0.001, 0.01, 0.1, 1, 10]),
}
NOISE_SEEDS = [1, 2, 3]  # center config with other seeds = noise floor

# Build the run list; identical configs shared between grids are only trained once
runs = {}
for (a, b), (values_a, values_b) in GRIDS.items():
    for va in values_a:
        for vb in values_b:
            params = {**CENTER, a: va, b: vb}
            runs[json.dumps(params, sort_keys=True)] = params
for s in NOISE_SEEDS:
    params = {**CENTER, "seed": s}
    runs[json.dumps(params, sort_keys=True)] = params

# Resume: skip configs already in the results file
done = pd.read_csv(RESULTS) if RESULTS.exists() else pd.DataFrame(columns=["key"])
todo = {k: p for k, p in runs.items() if k not in set(done["key"])}
print(f"{len(runs)} unique configs, {len(todo)} to run")

X, y, _, _ = load_data()
dtrain, dvalid = make_holdout(X, y)  # same split as Phases 1-2
OUT.mkdir(exist_ok=True)

rows = done.to_dict("records")
for i, (key, params) in enumerate(todo.items(), start=1):
    res = run_holdout(dtrain, dvalid, params, MAX_ROUNDS)
    rows.append({"key": key, **params, **res})
    changed = {p: v for p, v in params.items() if v != CENTER[p]}
    print(f"[{i}/{len(todo)}] {changed or 'center'}: AUC={res['auc']:.5f}, "
          f"trees={res['trees']}, {res['seconds']:.0f}s")
    pd.DataFrame(rows).to_csv(RESULTS, index=False)  # save as we go

df = pd.DataFrame(rows)
center = df[df["key"] == json.dumps(CENTER, sort_keys=True)]["auc"].iloc[0]
noise = df[df["seed"] != CENTER["seed"]]["auc"]
print(f"\nCenter AUC {center:.5f} | other seeds: {', '.join(f'{v:.5f}' for v in noise)}")

for (a, b), (values_a, values_b) in GRIDS.items():
    others = [p for p in CENTER if p not in (a, b)]
    mask = (df[others] == pd.Series({p: CENTER[p] for p in others})).all(axis=1)
    table = df[mask].pivot_table(index=a, columns=b, values="auc")
    print(f"\n{a} x {b}  (AUC - center, x1e5)\n{((table - center) * 1e5).round(1).to_string()}")
