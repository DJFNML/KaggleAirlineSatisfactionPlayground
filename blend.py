"""Blend saved OOF predictions: every single model, pair and the trio, scored side by side.

For each combination: equal weights, best weights (grid search, step 0.05), and an
"honest" score where weights are picked on one half of the rows and scored on the other.
Each is computed on probabilities and on ranks.
"""
import sys
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_data

# Usage: python blend.py 15 [lgb=lgb_t119_15fold ...] to swap in other prediction files
N_SPLITS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
MODELS = {"xgb": f"xgb_tuned_{N_SPLITS}fold", "lgb": f"lgb_{N_SPLITS}fold", "cat": f"cat_{N_SPLITS}fold"}
MODELS.update(arg.split("=", 1) for arg in sys.argv[2:])
STEP = 0.05
ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"

_, y, _, test_ids = load_data()
y = y.to_numpy()
saved = {m: np.load(PREDS / f"{f}.npz") for m, f in MODELS.items()}


def to_ranks(p):
    return rankdata(p) / len(p)


preds = {
    "prob": {m: (s["oof"], s["test"]) for m, s in saved.items()},
    "rank": {m: (to_ranks(s["oof"]), to_ranks(s["test"])) for m, s in saved.items()},
}

print("=== Single models ===")
for m, s in saved.items():
    seed_aucs = [roc_auc_score(y, o) for o in s["oof_seeds"]]
    print(f"{m}: seeds {', '.join(f'{a:.5f}' for a in seed_aucs)} | "
          f"seed avg {roc_auc_score(y, s['oof']):.5f} | spread {max(seed_aucs) - min(seed_aucs):.5f}")

print("\n=== Rank correlation of OOF predictions (lower = more different = more to gain) ===")
oof_ranks = pd.DataFrame({m: preds["rank"][m][0] for m in MODELS})
print(oof_ranks.corr().round(4).to_string())


def weight_grid(n):
    """All weight vectors for n models in steps of STEP that sum to 1."""
    steps = round(1 / STEP)
    return [np.array(w) / steps for w in product(range(steps + 1), repeat=n) if sum(w) == steps]


def best_weights(P, rows):
    """Weights maximising AUC on the given rows."""
    return max(weight_grid(P.shape[1]), key=lambda w: roc_auc_score(y[rows], P[rows] @ w))


# Two halves for the honest score: weights chosen on one half are scored on the other
halves = list(StratifiedKFold(2, shuffle=True, random_state=0).split(y, y))

rows, blends = [], {}
for size in (1, 2, 3):
    for combo in combinations(MODELS, size):
        for method in ("prob", "rank"):
            P = np.column_stack([preds[method][m][0] for m in combo])
            T = np.column_stack([preds[method][m][1] for m in combo])
            equal = np.full(size, 1 / size)
            all_rows = np.arange(len(y))
            w_best = best_weights(P, all_rows) if size > 1 else equal

            honest = np.zeros(len(y))
            for fit_rows, score_rows in halves:
                w = best_weights(P, fit_rows) if size > 1 else equal
                honest[score_rows] = P[score_rows] @ w

            name = " + ".join(combo)
            rows.append({
                "combo": name,
                "method": method,
                "equal_w": roc_auc_score(y, P @ equal),
                "best_w": roc_auc_score(y, P @ w_best),
                "honest": roc_auc_score(y, honest),
                "weights": " / ".join(f"{m}={w:.2f}" for m, w in zip(combo, w_best)),
            })
            blends[(name, method)] = T @ w_best
            print(f"  scored {name} ({method})")

df = pd.DataFrame(rows)
best_single = df[~df["combo"].str.contains(r"\+")]["equal_w"].max()
df["honest_vs_best_single"] = df["honest"] - best_single
df = df.sort_values("honest", ascending=False).reset_index(drop=True)
tag = f"{N_SPLITS}fold" + "".join(f"_{a.split('=', 1)[1]}" for a in sys.argv[2:])  # keep earlier blends
df.to_csv(ROOT / "results" / f"blend_{tag}.csv", index=False)
print("\n=== All combinations (sorted by honest score) ===")
print(df.round(5).to_string())

top = df.iloc[0]
out = ROOT / "submissions" / f"submission_blend_{tag}.csv"
pd.DataFrame({"id": test_ids, "satisfaction": blends[(top["combo"], top["method"])]}).to_csv(out, index=False)
print(f"\nBest: {top['combo']} ({top['method']}, {top['weights']}) -> {out.relative_to(ROOT)}")
