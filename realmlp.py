"""RealMLP (pytabkit) with Vladimir Demidov's published S6E10 configuration and feature engineering, as
re-run in busyaprime's "Route or distance? Both" notebook.

Changes from his notebook (the same ones busyaprime made): our folds and seeds, categorical columns taken
as every non-number column, and no best-epoch selection (stop_epoch = n_epochs keeps the last of the
three epochs, so the scored fold is never used to pick anything).
Folds: StratifiedKFold(5, seed 42), the same as ladder.py. Each fold model also predicts the test set.

Usage: python realmlp.py [RUN]   (RUN > 0: other training seeds, same folds)
Writes results/realmlp[_s{RUN}]/fold{k}.npz (val_idx, oof, test) and results/preds/realmlp[_s{RUN}]_5fold.npz.
"""
import contextlib
import io
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

from cv import SEED
from preprocess import DATA, TARGET

N_FOLDS = 5
RUN = int(sys.argv[1]) if len(sys.argv) > 1 else 0  # extra runs with other training seeds
ROOT = Path(__file__).parent
OUT = ROOT / "results" / ("realmlp" if RUN == 0 else f"realmlp_s{RUN}")
NAME = "realmlp_5fold" if RUN == 0 else f"realmlp_s{RUN}_5fold"
OUT.mkdir(parents=True, exist_ok=True)
for name in ("lightning", "lightning.pytorch", "lightning.fabric", "pytorch_lightning", "lightning_fabric"):
    logging.getLogger(name).setLevel(logging.ERROR)

COMBOS = [("Class", "Type of Travel", "Gender"), ("Class", "Type of Travel", "Age"),
          ("Flight Distance", "Departure/Arrival time convenient")]
PARAMS = {
    "random_state": SEED, "verbosity": 0, "val_metric_name": "1-auc_ovr",
    "n_ens": 8, "n_epochs": 3, "batch_size": 256, "use_early_stopping": False,
    "early_stopping_additive_patience": 10, "early_stopping_multiplicative_patience": 1,
    "lr": 0.053, "wd": 0.015, "sq_mom": 0.988, "lr_sched": "flat_anneal", "wd_sched": "cos_log_15",
    "first_layer_lr_factor": 0.25,
    "embedding_size": 5, "max_one_hot_cat_size": 18, "hidden_sizes": [512, 256, 128], "act": "silu",
    "p_drop": 0.05, "p_drop_sched": "invsqrtp1e-3",
    "plr_hidden_1": 16, "plr_hidden_2": 8, "plr_act_name": "gelu", "plr_lr_factor": 0.1151, "plr_sigma": 2.33,
    "ls_eps": 0.01, "ls_eps_sched": "sqrt_cos",
    "add_front_scale": False, "bias_init_mode": "neg-uniform-dynamic-2",
    "tfms": ["one_hot", "median_center", "robust_scale", "smooth_clip", "embedding", "l2_normalize"],
    "stop_epoch": 3,  # keep the last epoch: no selection on the validation fold
}


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def feature_engineering(df, cat_cols, num_cols, maps, fit):
    """Demidov's features: numeric columns also as categories, count encodings, three interaction keys."""
    for c in cat_cols:
        df[c] = df[c].fillna("missing")
    for c in num_cols:
        df[c] = df[c].fillna(0.0)
    for c in num_cols:
        if fit:
            codes, uniques = df[c].factorize()
            maps[c] = uniques
        else:
            codes = df[c].map({v: i for i, v in enumerate(maps[c])}).fillna(-1).astype("int32")
        df[f"{c}_cat_"] = pd.Series(codes, index=df.index).astype("category")
    for c in cat_cols + ["Leg room service"]:
        if fit:
            maps[f"_{c}_count"] = df[c].value_counts()
        df[f"_{c}_count"] = df[c].map(maps[f"_{c}_count"]).fillna(0).astype("int32")
    names = []
    for cols in COMBOS:
        name = "_".join(cols) + "_"
        names.append(name)
        s = df[cols[0]].astype(str)
        for c in cols[1:]:
            s = s + "_" + df[c].astype(str)
        if fit:
            codes, uniques = pd.factorize(s, sort=False)
            maps[name] = uniques
        else:
            codes = s.map({v: i for i, v in enumerate(maps[name])}).fillna(-1).astype("int32")
        df[name] = pd.Series(codes, index=df.index).astype("category")
    return df, names


if __name__ == "__main__":
    from pytabkit import RealMLP_TD_Classifier

    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    y = train[TARGET].astype(int).to_numpy()
    X, X_test = train.drop(columns=["id", TARGET]), test.drop(columns=["id"])
    cat_cols = X.select_dtypes(exclude="number").columns.tolist()
    num_cols = X.select_dtypes(include="number").columns.tolist()
    maps = {}
    X, combos = feature_engineering(X, cat_cols, num_cols, maps, fit=True)
    X_test, _ = feature_engineering(X_test, cat_cols, num_cols, maps, fit=False)
    log(f"RealMLP inputs: {X.shape[1]} columns")
    ys = pd.Series(y)
    drop = ["Class_Type of Travel_Age_", "Flight Distance_Departure/Arrival time convenient_"]
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    for k, (ia, iv) in enumerate(folds, start=1):
        path = OUT / f"fold{k}.npz"
        if path.exists():
            continue
        start = time.time()
        torch.manual_seed(SEED + k + 1000 * RUN)
        Xa, Xv, Xt = X.iloc[ia].copy(), X.iloc[iv].copy(), X_test.copy()
        enc = TargetEncoder(cv=5, smooth="auto", shuffle=True, random_state=SEED + k)
        te = [f"_{c}TE" for c in combos]
        Xa[te] = enc.fit_transform(Xa[combos], ys.iloc[ia])
        Xv[te], Xt[te] = enc.transform(Xv[combos]), enc.transform(Xt[combos])
        Xa, Xv, Xt = Xa.drop(columns=drop), Xv.drop(columns=drop), Xt.drop(columns=drop)
        m = RealMLP_TD_Classifier(**{**PARAMS, "random_state": SEED + k + 1000 * RUN})
        with contextlib.redirect_stdout(io.StringIO()):
            m.fit(Xa, ys.iloc[ia], Xv, ys.iloc[iv])
        pv, pt = m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1]
        np.savez(path, val_idx=iv, oof=pv, test=pt)
        log(f"realmlp fold {k}/{N_FOLDS}: AUC {roc_auc_score(y[iv], pv):.5f}, {(time.time() - start) / 60:.1f} min")
        del m
        torch.cuda.empty_cache()
    if all((OUT / f"fold{k}.npz").exists() for k in range(1, N_FOLDS + 1)):
        oof, tst = np.zeros(len(y)), np.zeros(len(X_test))
        for k in range(1, N_FOLDS + 1):
            f = np.load(OUT / f"fold{k}.npz")
            oof[f["val_idx"]] = f["oof"]
            tst += f["test"] / N_FOLDS
        np.savez(ROOT / "results" / "preds" / f"{NAME}.npz", oof=oof, test=tst)
        log(f"realmlp: OOF AUC {roc_auc_score(y, oof):.5f} -> results/preds/{NAME}.npz")
