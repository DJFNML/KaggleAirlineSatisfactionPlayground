"""A learned stacker vs the rank blend, both scored on the same outer 5-fold split of the OOF rows.

Inputs: OOF predictions of xgb, lgb, the 3 random TabPFN passes (mean) and the emb2x200k run (as logits),
the emb cluster flag (satisfied / unsatisfied), the first 8 TabPFN-embedding coordinates and the raw
features. The stacker is a regularised LightGBM, early-stopped on 10% of each outer-training part.
Baseline: rank blend of xgb, lgb and the per-cluster TabPFN mix (tabpfn_cluster_mix.py), weights picked
on each outer-training part by grid search. If the stacker wins, it is refitted on all rows and writes
results/preds/stack_lgb.npz and submissions/submission_stack_lgb.csv.
"""
from itertools import product
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS

ROOT = Path(__file__).parent
PREDS, FULL = ROOT / "results" / "preds", ROOT / "results" / "full"
X, y_s, X_test, test_ids = load_data()
y = y_s.to_numpy()
EMB = np.load(FULL / "knn" / "tabpfn_emb64.npz")["emb"]
base = {m: np.load(PREDS / f"{f}.npz") for m, f in [("xgb", "xgb_tuned_15fold_5seeds"), ("lgb", "lgb_t119_15fold_5seeds"),
        ("rand", "tabpfn200k_15fold_3ctx"), ("emb", "tabpfn_emb2x200k_15fold_1ctx"), ("mix", "tabpfn_emb2x200k_mix")]}


def satisfied_cluster(rows, random_pred, seed):  # as in tabpfn_cluster_mix.py
    labels = KMeans(2, n_init=4, random_state=seed).fit(EMB[rows]).labels_
    return labels == max(range(2), key=lambda c: random_pred[labels == c].mean())


folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
sat = np.zeros(len(y), dtype=bool)
for k, (_, va) in enumerate(folds, start=1):
    sat[va] = satisfied_cluster(va, base["rand"]["oof"][va], SEED if k == 1 else SEED + k)
sat_test = satisfied_cluster(np.arange(len(y), len(EMB)), base["rand"]["test"], 1000)


def features(part, frame, cluster, emb_rows):
    logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))  # noqa: E731
    f = frame.reset_index(drop=True).copy()
    for m in ("xgb", "lgb", "rand", "emb"):
        f[f"pred {m}"] = logit(base[m][part])
    f["sat cluster"] = cluster.astype(int)
    for d in range(8):
        f[f"emb {d}"] = emb_rows[:, d]
    return f


F, F_test = features("oof", X, sat, EMB[:len(y)]), features("test", X_test, sat_test, EMB[len(y):])
PARAMS = dict(n_estimators=3000, learning_rate=0.02, num_leaves=31, min_child_samples=500, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0, num_threads=8, verbose=-1)


def fit_stacker(idx):
    fit, stop = train_test_split(idx, test_size=0.1, stratify=y[idx], random_state=SEED)
    return lgb.LGBMClassifier(**PARAMS).fit(F.iloc[fit], y[fit], eval_set=[(F.iloc[stop], y[stop])],
                                             callbacks=[lgb.early_stopping(100, verbose=False)])


ranks = {m: rankdata(base[m]["oof"]) / len(y) for m in ("xgb", "lgb", "mix")}
grid = [w for w in product(np.round(np.arange(0, 1.001, 0.05), 2), repeat=2) if w[0] + w[1] <= 1]
blend = lambda idx, w: w[0] * ranks["xgb"][idx] + w[1] * ranks["lgb"][idx] + (1 - w[0] - w[1]) * ranks["mix"][idx]  # noqa: E731
rows = []
for i, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=7).split(F, y), start=1):
    w = max(grid, key=lambda w: roc_auc_score(y[tr], blend(tr, w)))
    m = fit_stacker(tr)
    a_blend, a_stack = roc_auc_score(y[va], blend(va, w)), roc_auc_score(y[va], m.predict_proba(F.iloc[va])[:, 1])
    rows.append((a_blend, a_stack))
    print(f"outer fold {i}: blend {a_blend:.5f} (xgb {w[0]:.2f} / lgb {w[1]:.2f}) | stacker {a_stack:.5f} "
          f"({a_stack - a_blend:+.5f}, {m.best_iteration_} trees)", flush=True)
d = np.array(rows)
print(f"mean: blend {d[:, 0].mean():.5f} | stacker {d[:, 1].mean():.5f} ({(d[:, 1] - d[:, 0]).mean():+.5f}, "
      f"better on {(d[:, 1] > d[:, 0]).sum()}/5)", flush=True)

if (d[:, 1] > d[:, 0]).mean() >= 0.8 and (d[:, 1] - d[:, 0]).mean() > 0:
    m = fit_stacker(np.arange(len(y)))
    test = m.predict_proba(F_test)[:, 1]
    np.savez(PREDS / "stack_lgb.npz", test=test)
    pd.DataFrame({"id": test_ids, "satisfaction": test}).to_csv(ROOT / "submissions" / "submission_stack_lgb.csv",
                                                                 index=False)
    imp = pd.Series(m.booster_.feature_importance("gain"), index=F.columns).sort_values(ascending=False)
    print(f"refitted on all rows ({m.best_iteration_} trees) -> submissions/submission_stack_lgb.csv; test mean "
          f"{test.mean():.4f} vs OOF positive rate {y.mean():.4f}\ntop inputs by gain: "
          + ", ".join(f"{k} {v / imp.sum():.1%}" for k, v in imp.head(8).items()), flush=True)
else:
    print("stacker not clearly better: no submission written", flush=True)
