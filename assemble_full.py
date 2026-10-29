"""Combine the per-fold files from run_nn_full.py into results/preds/*.npz (oof, test, oof_seeds).

Usage: python assemble_full.py tabm
       python assemble_full.py tabpfn TAG   (TAG as in run_nn_full.py's file names, e.g. tabpfn100k)
  tabm   - uses every seed with all 15 folds done -> results/preds/tabm_15fold_{n}seeds.npz
  tabpfn - averages every pass with all 15 folds and the test set done -> results/preds/{TAG}_15fold_{n}ctx.npz
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from preprocess import load_data
from probe_nn import N_SPLITS

MODEL = sys.argv[1]
ROOT = Path(__file__).parent
FULL, PREDS = ROOT / "results" / "full", ROOT / "results" / "preds"
_, y_s, _, _ = load_data()
y = y_s.to_numpy()


def collect(pattern):
    """OOF from the 15 fold files matching pattern (with {k}), plus the fold-averaged test predictions."""
    oof, tests = np.full(len(y), np.nan), []
    for k in range(1, N_SPLITS + 1):
        f = np.load(FULL / pattern.format(k=k))
        oof[f["val_idx"]] = f["oof"]
        if "test" in f:
            tests.append(f["test"])
    assert not np.isnan(oof).any(), "some rows have no OOF prediction"
    return oof, (np.mean(tests, axis=0) if tests else None)


if MODEL == "tabm":
    seeds = sorted({int(p.name.split("_")[1][1:]) for p in FULL.glob("tabm_s*_fold*.npz")})
    seeds = [s for s in seeds if all((FULL / f"tabm_s{s}_fold{k}.npz").exists() for k in range(1, N_SPLITS + 1))]
    runs = [collect(f"tabm_s{s}_fold{{k}}.npz") for s in seeds]
    oof_seeds = np.array([r[0] for r in runs])
    out = PREDS / f"tabm_15fold_{len(seeds)}seeds.npz"
    np.savez(out, oof=oof_seeds.mean(axis=0), test=np.mean([r[1] for r in runs], axis=0), oof_seeds=oof_seeds)
    for s, o in zip(seeds, oof_seeds):
        print(f"tabm seed {s}: OOF AUC {roc_auc_score(y, o):.5f}")
    print(f"tabm {len(seeds)}-seed avg: OOF AUC {roc_auc_score(y, oof_seeds.mean(axis=0)):.5f} -> {out.name}")

elif MODEL == "tabpfn":  # python assemble_full.py tabpfn TAG, e.g. tabpfn100k
    tag = sys.argv[2]
    passes = sorted({int(p.name[len(tag) + 2:].split("_")[0]) for p in FULL.glob(f"{tag}_c*_test.npz")})
    passes = [c for c in passes if all((FULL / f"{tag}_c{c}_fold{k}.npz").exists() for k in range(1, N_SPLITS + 1))]
    assert passes, f"no complete {tag} passes yet"
    oof_seeds = np.array([collect(f"{tag}_c{c}_fold{{k}}.npz")[0] for c in passes])
    test = np.mean([np.load(FULL / f"{tag}_c{c}_test.npz")["test"] for c in passes], axis=0)
    out = PREDS / f"{tag}_15fold_{len(passes)}ctx.npz"
    np.savez(out, oof=oof_seeds.mean(axis=0), test=test, oof_seeds=oof_seeds)
    for c, o in zip(passes, oof_seeds):
        print(f"{tag} pass {c + 1}: OOF AUC {roc_auc_score(y, o):.5f}")
    print(f"{tag} {len(passes)}-context avg: OOF AUC {roc_auc_score(y, oof_seeds.mean(axis=0)):.5f} -> {out.name}")

else:
    raise ValueError(f"Unknown model: {MODEL}")
