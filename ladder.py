"""Feature-ladder tree models (ideas from the public S6E10 notebooks by evgendvorkin, goodpjw2008 and busyaprime).

Blocks, all built from the raw competition columns (0 ratings kept as 0, missing arrival delay -> 0):
  org    each column's satisfied rate per value in the ORIGINAL data (a lookup; no competition labels)
  origm  a LightGBM trained on the original data only, predicting every competition row
  digit  the digits of the four numeric columns, positions 10^-4 .. 10^3 (synthetic-generator artefacts)
  freq   frequency and rarity (-log frequency) of every value, over train + test
  route  per Flight Distance value: row count, the mean of every other column, and each row's difference
  te/te2 target encodings of every column and of ten column pairs, fitted inside each fold only
         (cross-fitted within the fold's training part)
Folds: StratifiedKFold(5, seed 42). Early stopping on a 5% slice of each fold's training part, never on
the scored fold. Each fold model also predicts the test set (averaged over folds).

Usage: python ladder.py lgb|xgb
Writes results/ladder/{model}_fold{k}.npz (val_idx, oof, test, rounds) and, when all folds exist,
results/preds/{model}_ladder_5fold.npz (oof, test). Features are cached in results/ladder/features.pkl.
"""
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder

from cv import SEED
from preprocess import CAT_COLS, DATA, ORIG_DIR, RATING_COLS, TARGET

MODEL = sys.argv[1] if len(sys.argv) > 1 else "lgb"
N_FOLDS, THREADS = 5, 10
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "ladder"
OUT.mkdir(parents=True, exist_ok=True)
NUMS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
BASE = CAT_COLS + NUMS + RATING_COLS
PAIRS = [("Class", "Type of Travel"), ("Customer Type", "Type of Travel"), ("Class", "Customer Type"),
         ("Gender", "Class"), ("Inflight wifi service", "Online boarding"),
         ("Inflight wifi service", "Type of Travel"), ("Online boarding", "Type of Travel"),
         ("Seat comfort", "Inflight entertainment"), ("Class", "Inflight wifi service"),
         ("Class", "Online boarding")]


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


def raw_frames():
    train, test = pd.read_csv(DATA / "train.csv"), pd.read_csv(DATA / "test.csv")
    orig = pd.concat([pd.read_csv(ORIG_DIR / "train.csv"), pd.read_csv(ORIG_DIR / "test.csv")], ignore_index=True)
    orig[TARGET] = orig[TARGET] == "satisfied"
    for df in (train, test, orig):
        df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0)
    return train, test, orig


def build_features():
    """Label-free blocks over train + test (org/origm use only the original's labels)."""
    train, test, orig = raw_frames()
    y = train[TARGET].astype(int).to_numpy()
    full = pd.concat([train[BASE], test[BASE]], ignore_index=True)
    yo = orig[TARGET].astype(int)
    X = full.copy()

    for c in BASE:  # org
        X[f"org_{c}"] = full[c].map(yo.groupby(orig[c]).mean()).fillna(yo.mean()).astype("float32")

    Xo, Xf = orig[BASE].copy(), full.copy()  # origm
    for c in CAT_COLS:
        cat = pd.CategoricalDtype(sorted(set(Xo[c]) | set(Xf[c])))
        Xo[c], Xf[c] = Xo[c].astype(cat), Xf[c].astype(cat)
    m = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.05, num_leaves=63, colsample_bytree=0.6,
                           subsample=0.8, subsample_freq=1, reg_lambda=5.0, n_jobs=THREADS,
                           random_state=SEED, verbose=-1).fit(Xo, yo)
    X["orig_model"] = m.predict_proba(Xf)[:, 1].astype("float32")

    coded = full.copy()  # route
    for c in CAT_COLS:
        coded[c] = coded[c].astype("category").cat.codes
    fd = full["Flight Distance"]
    grp = coded.groupby(fd)
    X["fd_cnt"] = fd.map(fd.value_counts()).astype("float32")
    for c in BASE:
        if c != "Flight Distance":
            X[f"fdm_{c}"] = grp[c].transform("mean").astype("float32")
            X[f"fdd_{c}"] = (coded[c] - X[f"fdm_{c}"]).astype("float32")

    for c in NUMS:  # digit
        v = full[c].astype(float)
        for k in range(-4, 4):
            X[f"{c}_d{k}"] = (v // (10.0 ** k) % 10).astype("int8")

    for c in list(X.columns):  # freq
        if c.startswith(("org_", "orig_model")):
            continue
        f = X[c].map(X[c].value_counts(normalize=True)).astype("float32")
        X[f"freq_{c}"] = f
        X[f"rar_{c}"] = (-np.log(f.clip(lower=1 / len(X)))).astype("float32")

    for c in CAT_COLS:
        X[c] = X[c].astype("category")
    const = [c for c in X.columns if X[c].nunique() <= 1]  # e.g. fractional digits of integer columns
    X = X.drop(columns=const)
    n = len(train)
    log(f"features: {X.shape[1]} columns ({len(const)} constant ones dropped)")
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True), y


def add_encodings(Xa, ya, others, k):
    """te + te2: target encodings fitted on Xa only (cross-fitted there), applied to the other frames."""
    def keys(F):
        cols = {f"te_{c}": F[c].astype(str) for c in BASE}
        cols.update({f"te2_{a}|{b}": F[a].astype(str) + "_" + F[b].astype(str) for a, b in PAIRS})
        return pd.DataFrame(cols)
    enc = TargetEncoder(target_type="binary", smooth="auto", cv=5, random_state=SEED + k)
    Ka = keys(Xa)
    ta = enc.fit_transform(Ka, ya)
    out = [pd.concat([Xa.reset_index(drop=True), pd.DataFrame(ta, columns=Ka.columns, dtype="float32")], axis=1)]
    for F in others:
        Ko = keys(F)
        out.append(pd.concat([F.reset_index(drop=True),
                              pd.DataFrame(enc.transform(Ko), columns=Ka.columns, dtype="float32")], axis=1))
    return out


def fit_predict(Xa, ya, Xe, ye, Xv, Xt, k):
    if MODEL == "lgb":
        m = lgb.LGBMClassifier(n_estimators=8000, learning_rate=0.02, max_depth=8, num_leaves=127,
                               colsample_bytree=0.5, subsample=0.9, subsample_freq=1, min_child_samples=20,
                               reg_lambda=5.0, reg_alpha=0.05, n_jobs=THREADS, random_state=SEED + k, verbose=-1)
        m.fit(Xa, ya, eval_set=[(Xe, ye)], eval_metric="auc", callbacks=[lgb.early_stopping(200, verbose=False)])
        return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration_
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=12000, learning_rate=0.02, max_depth=8, subsample=0.9, colsample_bytree=0.5,
                          min_child_weight=5, reg_lambda=5.0, reg_alpha=0.05, tree_method="hist", device="cpu",
                          n_jobs=THREADS, enable_categorical=True, max_cat_to_onehot=8, eval_metric="auc",
                          early_stopping_rounds=300, random_state=SEED + k)
    m.fit(Xa, ya, eval_set=[(Xe, ye)], verbose=False)
    return m.predict_proba(Xv)[:, 1], m.predict_proba(Xt)[:, 1], m.best_iteration + 1


if __name__ == "__main__":
    cache = OUT / "features.pkl"
    if cache.exists():
        X, Xt, y = pd.read_pickle(cache)
    else:
        X, Xt, y = build_features()
        pd.to_pickle((X, Xt, y), cache)
    folds = list(StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED).split(X, y))
    for k, (tr, va) in enumerate(folds, start=1):
        path = OUT / f"{MODEL}_fold{k}.npz"
        if path.exists():
            continue
        start = time.time()
        fit_idx, es_idx = train_test_split(tr, test_size=0.05, stratify=y[tr], random_state=SEED + k)
        Xa, Xe, Xv, Xtk = add_encodings(X.iloc[fit_idx], y[fit_idx], [X.iloc[es_idx], X.iloc[va], Xt], k)
        pv, pt, rounds = fit_predict(Xa, y[fit_idx], Xe, y[es_idx], Xv, Xtk, k)
        np.savez(path, val_idx=va, oof=pv, test=pt, rounds=rounds)
        log(f"{MODEL} fold {k}/{N_FOLDS}: AUC {roc_auc_score(y[va], pv):.5f}, {rounds} rounds, "
            f"{(time.time() - start) / 60:.1f} min -> {path.name}")
    if all((OUT / f"{MODEL}_fold{k}.npz").exists() for k in range(1, N_FOLDS + 1)):
        oof, test = np.zeros(len(y)), np.zeros(len(Xt))
        for k in range(1, N_FOLDS + 1):
            f = np.load(OUT / f"{MODEL}_fold{k}.npz")
            oof[f["val_idx"]] = f["oof"]
            test += f["test"] / N_FOLDS
        np.savez(ROOT / "results" / "preds" / f"{MODEL}_ladder_5fold.npz", oof=oof, test=test)
        log(f"{MODEL}: OOF AUC {roc_auc_score(y, oof):.5f} -> results/preds/{MODEL}_ladder_5fold.npz")
