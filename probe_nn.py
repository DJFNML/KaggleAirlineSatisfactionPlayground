"""Probe TabM / TabPFN on a few of the 15 CV folds, to see if they add to the GBDT blend.

Usage: python probe_nn.py tabm|tabm_tuned|tabpfn [n_folds]
Uses the same StratifiedKFold(15, seed 42) split as all other models, trains/conditions on the
other 14 folds and predicts the probe fold. Saves results/probe/{model}_fold{i}.npz (val_idx, pred).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv import SEED
from preprocess import ARR_DELAY, CAT_COLS, DEP_DELAY, RATING_COLS, load_data

N_SPLITS = 15
OUT = Path(__file__).parent / "results" / "probe"
DEVICE = "cuda"


# ---------------- TabM ----------------
TABM = {
    "k": 32,  # ensemble members inside one TabM
    "n_blocks": 2,
    "d_block": 512,
    "dropout": 0.1,
    "d_embedding": 16,  # piecewise-linear embedding size per numeric feature
    "n_bins": 48,
    "lr": 2e-3,
    "weight_decay": 3e-4,
    "batch_size": 1024,
    "max_epochs": 60,
    "patience": 6,  # epochs without inner-validation improvement before stopping
    "lr_decay_patience": None,  # if set: halve the learning rate after this many epochs without improvement
}


def nn_arrays(X, ratings_as_cat=False):
    """Numeric matrix (NaN-free, delays log-scaled) and categorical codes for neural nets.

    With ratings_as_cat, the 0-5 ratings become one-hot categorical inputs instead of numbers.
    """
    num = X.drop(columns=CAT_COLS).copy()
    num[ARR_DELAY] = num[ARR_DELAY].fillna(num[DEP_DELAY])  # 292 missing; arrival ~ departure delay
    for col in (DEP_DELAY, ARR_DELAY):
        num[col] = np.log1p(num[col])
    num = num.fillna(0)  # NaN ratings = "not applicable" (was 0 originally)
    cats = [X[c].cat.codes.to_numpy() for c in CAT_COLS]
    cards = [len(X[c].cat.categories) for c in CAT_COLS]
    if ratings_as_cat:
        for c in RATING_COLS:
            cats.append(num.pop(c).to_numpy().astype(np.int64))  # values 0-5
            cards.append(6)
    return num.to_numpy(np.float32), np.stack(cats, axis=1).astype(np.int64), cards


def fit_predict_tabm(X_num, X_cat, cards, y, tr_idx, va_idx, seed, cfg=TABM, trial=None):
    """Train TabM on tr_idx (minus a 5% early-stopping slice), predict va_idx.

    With an Optuna trial, the inner-validation AUC is reported each epoch for pruning.
    """
    import optuna
    import rtdl_num_embeddings
    import tabm

    cfg = {**TABM, **cfg}
    torch.manual_seed(seed)
    # Inner validation split for early stopping (never the probe fold)
    tr_idx, es_idx = train_test_split(tr_idx, test_size=0.05, stratify=y[tr_idx], random_state=seed)

    # Standardize numeric features with training statistics
    mean, std = X_num[tr_idx].mean(0), X_num[tr_idx].std(0) + 1e-6
    to_t = lambda idx: (
        torch.as_tensor((X_num[idx] - mean) / std, device=DEVICE),
        torch.as_tensor(X_cat[idx], device=DEVICE),
    )
    (xn_tr, xc_tr), (xn_es, xc_es), (xn_va, xc_va) = to_t(tr_idx), to_t(es_idx), to_t(va_idx)
    y_tr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=DEVICE)

    bins = rtdl_num_embeddings.compute_bins(xn_tr, n_bins=cfg["n_bins"])
    model = tabm.TabM.make(
        n_num_features=X_num.shape[1],
        cat_cardinalities=cards,
        d_out=1,
        k=cfg["k"],
        n_blocks=cfg["n_blocks"],
        d_block=cfg["d_block"],
        dropout=cfg["dropout"],
        num_embeddings=rtdl_num_embeddings.PiecewiseLinearEmbeddings(
            bins, cfg["d_embedding"], activation=False, version="B"
        ),
    ).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    loss_fn = torch.nn.BCEWithLogitsLoss()

    @torch.no_grad()
    def predict(xn, xc):
        model.eval()
        out = []
        for i in range(0, len(xn), 8192):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xn[i:i + 8192], xc[i:i + 8192]).squeeze(-1).float()
            out.append(torch.sigmoid(logits).mean(1))  # average the k members' probabilities
        return torch.cat(out).cpu().numpy()

    best_auc, best_state, bad, epoch = -1.0, None, 0, 0
    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        perm = torch.randperm(len(y_tr), device=DEVICE)
        for i in range(0, len(perm), cfg["batch_size"]):
            b = perm[i:i + cfg["batch_size"]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(xn_tr[b], xc_tr[b]).squeeze(-1).float()  # (batch, k)
            # Each of the k members is trained on the same batch with its own loss
            loss = loss_fn(logits, y_tr[b, None].expand_as(logits))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        auc = roc_auc_score(y[es_idx], predict(xn_es, xc_es))
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        decay = cfg["lr_decay_patience"]
        if decay and bad and bad % decay == 0:
            for group in opt.param_groups:
                group["lr"] /= 2
        print(f"    epoch {epoch}: inner-val AUC {auc:.5f}{' *' if bad == 0 else ''}", flush=True)
        if trial is not None:
            trial.report(best_auc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if bad >= cfg["patience"]:
            break

    model.load_state_dict(best_state)
    return predict(xn_va, xc_va), {"epochs": epoch, "inner_val_auc": best_auc}


# ---------------- TabPFN ----------------
TABPFN = {
    "context_size": 50_000,  # training rows given as context per run
    "n_contexts": 4,  # different random contexts, predictions averaged
    "predict_chunk": 10_000,  # test rows per predict call (limits GPU memory)
}


def fit_predict_tabpfn(X, y, tr_idx, va_idx, seed):
    from tabpfn import TabPFNClassifier

    Xp = X.copy()
    for c in CAT_COLS:
        Xp[c] = Xp[c].cat.codes  # TabPFN takes integer codes + categorical indices
    Xp = Xp.to_numpy(np.float32)
    cat_idx = [list(X.columns).index(c) for c in CAT_COLS]

    rng = np.random.default_rng(seed)
    preds = []
    for c in range(TABPFN["n_contexts"]):
        ctx, _ = train_test_split(tr_idx, train_size=TABPFN["context_size"], stratify=y[tr_idx],
                                  random_state=int(rng.integers(1e9)))
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx,
                               ignore_pretraining_limits=True, random_state=seed + c)
        clf.fit(Xp[ctx], y[ctx])
        p = np.concatenate([
            clf.predict_proba(Xp[va_idx[i:i + TABPFN["predict_chunk"]]])[:, 1]
            for i in range(0, len(va_idx), TABPFN["predict_chunk"])
        ])
        preds.append(p)
        print(f"    context {c + 1}/{TABPFN['n_contexts']}: AUC {roc_auc_score(y[va_idx], p):.5f} "
              f"| running avg {roc_auc_score(y[va_idx], np.mean(preds, axis=0)):.5f}", flush=True)
    return np.mean(preds, axis=0), {"contexts": TABPFN["n_contexts"], "context_size": TABPFN["context_size"]}


# ---------------- Main ----------------
if __name__ == "__main__":
    MODEL = sys.argv[1]
    N_PROBE_FOLDS = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    X, y_s, _, _ = load_data(zero_ratings_as_nan=MODEL == "tabpfn")  # NNs: keep 0 = "n/a" as a value
    y = y_s.to_numpy()
    folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))[:N_PROBE_FOLDS]
    OUT.mkdir(parents=True, exist_ok=True)
    tabm_cfg, ratings_as_cat = TABM, False
    if MODEL == "tabm_tuned":  # best settings from tune_tabm.py
        import optuna
        study = optuna.load_study(study_name="tabm_tuning",
                                  storage=f"sqlite:///{Path(__file__).parent / 'results' / 'tabm_tuning.db'}")
        tabm_cfg = dict(study.best_params)
        ratings_as_cat = tabm_cfg.pop("ratings_as_cat")
        print(f"tuned TabM (trial {study.best_trial.number}, holdout {study.best_value:.5f}): "
              f"{tabm_cfg}, ratings_as_cat={ratings_as_cat}")
    if MODEL == "tabm_tuned2":  # best settings from tune_tabm2.py, with its training schedule
        import optuna
        from tune_tabm2_config import SCHEDULE
        study = optuna.load_study(study_name="tabm_tuning2",
                                  storage=f"sqlite:///{Path(__file__).parent / 'results' / 'tabm_tuning2.db'}")
        tabm_cfg = {**study.best_params, **SCHEDULE}
        print(f"tuned TabM v2 (trial {study.best_trial.number}, folds 4-5 avg {study.best_value:.5f}): {tabm_cfg}")
    if MODEL in ("tabm", "tabm_tuned", "tabm_tuned2"):
        X_num, X_cat, cards = nn_arrays(X, ratings_as_cat=ratings_as_cat)

    for i, (tr_idx, va_idx) in enumerate(folds, start=1):
        path = OUT / f"{MODEL}_fold{i}.npz"
        if path.exists():
            print(f"{path.name} exists, skipping")
            continue
        start = time.time()
        print(f"[{MODEL}] fold {i}/{N_PROBE_FOLDS}: train {len(tr_idx)}, predict {len(va_idx)}", flush=True)
        if MODEL in ("tabm", "tabm_tuned", "tabm_tuned2"):
            pred, info = fit_predict_tabm(X_num, X_cat, cards, y, tr_idx, va_idx, seed=SEED + i, cfg=tabm_cfg)
        elif MODEL == "tabpfn":
            pred, info = fit_predict_tabpfn(X, y, tr_idx, va_idx, seed=SEED + i)
        else:
            raise ValueError(f"Unknown model: {MODEL}")
        np.savez(path, val_idx=va_idx, pred=pred)
        print(f"[{MODEL}] fold {i}: AUC {roc_auc_score(y[va_idx], pred):.5f} | {info} | "
              f"{(time.time() - start) / 60:.1f} min -> {path.name}", flush=True)
