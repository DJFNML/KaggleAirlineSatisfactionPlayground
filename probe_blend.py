"""Compare the current GBDT blend against blends that add TabM and/or TabPFN, on the probe folds only.

Uses the 5-seed OOF predictions of xgb/lgb/cat and results/probe/{model}_fold{i}.npz.
Rank blending; weights picked on one half of the probe rows and scored on the other ("honest").
"""
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_data

ROOT = Path(__file__).parent
PREDS, PROBE = ROOT / "results" / "preds", ROOT / "results" / "probe"
GBDT = {"xgb": "xgb_tuned_15fold_5seeds", "lgb": "lgb_t119_15fold_5seeds", "cat": "cat_t92_15fold_5seeds"}
NN = ["tabm", "tabm_tuned", "tabm_tuned2", "tabpfn"]
STEP = 0.05

_, y_s, _, _ = load_data()
y_all = y_s.to_numpy()

# Probe folds available for every NN model that has results
nn_folds = {m: sorted(PROBE.glob(f"{m}_fold*.npz")) for m in NN}
nn_models = [m for m in NN if nn_folds[m]]
n_folds = min(len(nn_folds[m]) for m in nn_models)
fold_idx = [np.load(PROBE / f"{nn_models[0]}_fold{i}.npz")["val_idx"] for i in range(1, n_folds + 1)]
rows = np.concatenate(fold_idx)
fold_of_row = np.concatenate([np.full(len(f), i) for i, f in enumerate(fold_idx, start=1)])
y = y_all[rows]

preds = {m: np.load(PREDS / f"{f}.npz")["oof"][rows] for m, f in GBDT.items()}
for m in nn_models:
    parts = [np.load(PROBE / f"{m}_fold{i}.npz") for i in range(1, n_folds + 1)]
    assert all((p["val_idx"] == f).all() for p, f in zip(parts, fold_idx)), "fold rows differ between models"
    preds[m] = np.concatenate([p["pred"] for p in parts])
ranks = {m: rankdata(p) / len(p) for m, p in preds.items()}
print(f"Probe: {n_folds} folds, {len(rows)} rows | models: {list(preds)}")


def per_fold_auc(p):
    return [roc_auc_score(y[fold_of_row == i], p[fold_of_row == i]) for i in range(1, n_folds + 1)]


print("\n=== Single models on probe rows ===")
singles = pd.DataFrame({m: [roc_auc_score(y, p), *per_fold_auc(p)] for m, p in preds.items()},
                       index=["all", *[f"fold{i}" for i in range(1, n_folds + 1)]]).T
print(singles.round(5).to_string())

print("\n=== Rank correlation ===")
print(pd.DataFrame(ranks).corr().round(4).to_string())


def weight_grid(n):
    steps = round(1 / STEP)
    return [np.array(w) / steps for w in product(range(steps + 1), repeat=n) if sum(w) == steps]


halves = list(StratifiedKFold(2, shuffle=True, random_state=0).split(y, y))


def honest_blend(models):
    """Rank blend whose weights are chosen on one half and applied to the other."""
    P = np.column_stack([ranks[m] for m in models])
    grid = weight_grid(len(models))
    out, weights = np.zeros(len(y)), []
    for fit_rows, score_rows in halves:
        w = max(grid, key=lambda w: roc_auc_score(y[fit_rows], P[fit_rows] @ w))
        out[score_rows] = P[score_rows] @ w
        weights.append(w)
    return out, np.mean(weights, axis=0)


combos = {"current (xgb+lgb+cat)": ["xgb", "lgb", "cat"]}
for m in nn_models:
    combos[f"current + {m}"] = ["xgb", "lgb", "cat", m]
if "tabpfn" in nn_models:
    for m in (m for m in nn_models if m.startswith("tabm")):
        combos[f"current + {m} + tabpfn"] = ["xgb", "lgb", "cat", m, "tabpfn"]

rows_out, base = [], None
for name, models in combos.items():
    p, w = honest_blend(models)
    folds = per_fold_auc(p)
    if base is None:
        base = (roc_auc_score(y, p), folds)
    rows_out.append({
        "blend": name,
        "honest_auc": roc_auc_score(y, p),
        "vs_current": roc_auc_score(y, p) - base[0],
        **{f"fold{i}_vs_current": f - b for i, (f, b) in enumerate(zip(folds, base[1]), start=1)},
        "avg_weights": " / ".join(f"{m}={x:.2f}" for m, x in zip(models, w)),
    })
    print(f"  scored {name}")

summary = pd.DataFrame(rows_out).set_index("blend")
summary.to_csv(ROOT / "results" / "probe_blend.csv")
print("\n=== Blends (honest, rank) ===")
print(summary.round(5).to_string())
