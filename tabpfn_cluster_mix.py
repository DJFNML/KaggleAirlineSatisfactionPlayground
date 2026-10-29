"""Mix the random-context TabPFN passes with the embedding-context run, with one weight per cluster.

The embedding-context run (run_knn_full.py emb 2 ...) splits each fold's validation rows, and the test rows,
into 2 k-means clusters in TabPFN's embedding space: one mostly satisfied, one mostly unsatisfied. The
embedding contexts help the satisfied cluster most, so each cluster gets its own weight for them:
    mix = (1 - w_cluster) * mean(random passes) + w_cluster * embedding run
The weights are picked on one half of the OOF rows and scored on the other (both ways round, averaged).
The clusters are recomputed with the run's k-means seeds (fold 1 came from context_select_test.py, seed
SEED; fold k from the full run, SEED + k; test 1000); "satisfied" = the cluster with higher predictions.

Usage: python tabpfn_cluster_mix.py RANDOM_TAG EMB_TAG   e.g. tabpfn200k tabpfn_emb2x200k
Writes results/preds/{EMB_TAG}_mix.npz (oof, test), for make_submission.py.
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS

RANDOM_TAG, EMB_TAG = sys.argv[1:3]
ROOT = Path(__file__).parent
FULL, PREDS = ROOT / "results" / "full", ROOT / "results" / "preds"
WEIGHTS = np.round(np.arange(0, 1.001, 0.05), 2)

_, y_s, _, _ = load_data()
y = y_s.to_numpy()
EMB = np.load(FULL / "knn" / "tabpfn_emb64.npz")["emb"]  # rows: train then test
n_random = len(sorted(FULL.glob(f"{RANDOM_TAG}_c*_test.npz")))


def satisfied_cluster(rows, random_pred, seed):
    """True for rows in the cluster with the higher mean random-pass prediction."""
    labels = KMeans(2, n_init=4, random_state=seed).fit(EMB[rows]).labels_
    sat = max(range(2), key=lambda c: random_pred[labels == c].mean())
    return labels == sat


rand_oof, emb_oof = np.full(len(y), np.nan), np.full(len(y), np.nan)
sat_oof = np.zeros(len(y), dtype=bool)
for k in range(1, N_SPLITS + 1):
    e = np.load(FULL / f"{EMB_TAG}_c0_fold{k}.npz")
    va = e["val_idx"]
    r = np.mean([np.load(FULL / f"{RANDOM_TAG}_c{c}_fold{k}.npz")["oof"] for c in range(n_random)], axis=0)
    rand_oof[va], emb_oof[va] = r, e["oof"]
    sat_oof[va] = satisfied_cluster(va, r, SEED if k == 1 else SEED + k)
assert not np.isnan(rand_oof).any() and not np.isnan(emb_oof).any()

rand_test = np.mean([np.load(FULL / f"{RANDOM_TAG}_c{c}_test.npz")["test"] for c in range(n_random)], axis=0)
emb_test = np.load(FULL / f"{EMB_TAG}_c0_test.npz")["test"]
sat_test = satisfied_cluster(np.arange(len(y), len(EMB)), rand_test, 1000)


def mix(rand, emb, sat, w_sat, w_unsat):
    w = np.where(sat, w_sat, w_unsat)
    return (1 - w) * rand + w * emb


grid = [(a, b) for a in WEIGHTS for b in WEIGHTS]
h1, h2 = next(StratifiedKFold(2, shuffle=True, random_state=0).split(y, y))
honest, picks = [], []
for fit, score in ((h1, h2), (h2, h1)):
    best = max(grid, key=lambda p: roc_auc_score(y[fit], mix(rand_oof, emb_oof, sat_oof, *p)[fit]))
    picks.append(best)
    honest.append(roc_auc_score(y[score], mix(rand_oof, emb_oof, sat_oof, *best)[score])
                  - roc_auc_score(y[score], rand_oof[score]))
w_sat, w_unsat = max(grid, key=lambda p: roc_auc_score(y, mix(rand_oof, emb_oof, sat_oof, *p)))

oof = mix(rand_oof, emb_oof, sat_oof, w_sat, w_unsat)
test = mix(rand_test, emb_test, sat_test, w_sat, w_unsat)
out = PREDS / f"{EMB_TAG}_mix.npz"
np.savez(out, oof=oof, test=test, oof_seeds=oof[None])
print(f"OOF AUC: {n_random} random passes {roc_auc_score(y, rand_oof):.5f} | embedding run {roc_auc_score(y, emb_oof):.5f}")
print(f"satisfied-cluster share: OOF {sat_oof.mean():.3f}, test {sat_test.mean():.3f}")
print(f"honest gain over the random passes: {np.mean(honest):+.5f} (halves {honest[0]:+.5f}, {honest[1]:+.5f}; "
      f"weights picked {picks})")
print(f"final weights (all rows): satisfied {w_sat:.2f} / unsatisfied {w_unsat:.2f} -> OOF AUC "
      f"{roc_auc_score(y, oof):.5f} -> {out.name}")
