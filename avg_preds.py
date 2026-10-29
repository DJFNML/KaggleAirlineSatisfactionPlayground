"""Average several prediction files (same model, other seeds or contexts) into one blend member.

Usage: python avg_preds.py OUT member [member ...]   (files are results/preds/{name}.npz with oof and test)
"""
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from preprocess import load_data

PREDS = Path(__file__).parent / "results" / "preds"
out, members = sys.argv[1], sys.argv[2:]
_, y_s, _, _ = load_data()
y = y_s.to_numpy()
files = [np.load(PREDS / f"{m}.npz") for m in members]
oof, test = np.mean([f["oof"] for f in files], axis=0), np.mean([f["test"] for f in files], axis=0)
for m, f in zip(members, files):
    print(f"  {m}: OOF AUC {roc_auc_score(y, f['oof']):.5f}")
np.savez(PREDS / f"{out}.npz", oof=oof, test=test)
print(f"{out}: average of {len(members)}, OOF AUC {roc_auc_score(y, oof):.5f}")
