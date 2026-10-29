"""Add the feature-subset models (subset_models.py) to the current best blend.

base  = 0.05 * xgb + 0.95 * TabPFN cluster mix (the current best submission, in probability space)
route = mean of the finished route models (all rows); long = mean of the finished long models (long rows)
    p = (1 - w_route) * base + w_route * route;   on long rows: p = (1 - w_long) * p + w_long * long
The two weights are picked on one half of the OOF rows and scored on the other, both ways round.
If the honest gain is positive on both halves, writes submissions/submission_subset_blend.csv.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from preprocess import load_data

ROOT = Path(__file__).parent
PREDS = ROOT / "results" / "preds"
W = np.round(np.arange(0, 0.61, 0.025), 3)

_, y_s, _, test_ids = load_data()
y = y_s.to_numpy()
ld = lambda n: np.load(PREDS / f"{n}.npz")  # noqa: E731
base_oof = 0.05 * ld("xgb_tuned_15fold_5seeds")["oof"] + 0.95 * ld("tabpfn_emb2x200k_mix")["oof"]
base_test = 0.05 * ld("xgb_tuned_15fold_5seeds")["test"] + 0.95 * ld("tabpfn_emb2x200k_mix")["test"]
print(f"base OOF AUC {roc_auc_score(y, base_oof):.5f}")

family = {}
for subset in ("route", "long"):
    files = sorted(PREDS.glob(f"subset_{subset}_*.npz"))
    if not files:
        continue
    fs = [np.load(f) for f in files]
    mask = fs[0]["mask"]
    for f, path in zip(fs, files):
        print(f"{path.stem}: OOF AUC {roc_auc_score(y[mask], f['oof'][mask]):.5f} on its {mask.sum()} rows "
              f"(base on the same rows {roc_auc_score(y[mask], base_oof[mask]):.5f}, corr with base "
              f"{np.corrcoef(f['oof'][mask], base_oof[mask])[0, 1]:.4f})")
    family[subset] = (mask, fs[0]["test_mask"], np.mean([f["oof"] for f in fs], axis=0),
                      np.mean([f["test"] for f in fs], axis=0))
assert family, "no subset predictions yet"


def mix(base, w_route, w_long, part):
    p = base.copy()
    if "route" in family:
        p = (1 - w_route) * p + w_route * family["route"][2 if part == "oof" else 3]
    if "long" in family:
        m = family["long"][0 if part == "oof" else 1]
        other = family["long"][2 if part == "oof" else 3]
        p[m] = (1 - w_long) * p[m] + w_long * other[m]
    return p


grid = [(a, b) for a in (W if "route" in family else [0]) for b in (W if "long" in family else [0])]
h1, h2 = next(StratifiedKFold(2, shuffle=True, random_state=0).split(y, y))
gains = []
for fit, score in ((h1, h2), (h2, h1)):
    best = max(grid, key=lambda w: roc_auc_score(y[fit], mix(base_oof, *w, "oof")[fit]))
    gains.append(roc_auc_score(y[score], mix(base_oof, *best, "oof")[score]) - roc_auc_score(y[score], base_oof[score]))
    print(f"weights picked on one half: route {best[0]}, long {best[1]} -> gain on the other half {gains[-1]:+.5f}")
w = max(grid, key=lambda w: roc_auc_score(y, mix(base_oof, *w, "oof")))
oof = mix(base_oof, *w, "oof")
print(f"honest gain {np.mean(gains):+.5f} | final weights (all rows): route {w[0]}, long {w[1]} -> OOF AUC "
      f"{roc_auc_score(y, oof):.5f}")
if min(gains) > 0:
    test = mix(base_test, *w, "test")
    out = ROOT / "submissions" / "submission_subset_blend.csv"
    pd.DataFrame({"id": test_ids, "satisfaction": test}).to_csv(out, index=False)
    print(f"-> {out.name} (test mean {test.mean():.4f}, OOF mean {oof.mean():.4f}, NaN {np.isnan(test).sum()})")
else:
    print("not better on both halves: no submission written")
