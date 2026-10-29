"""Full 15-fold runs of the tuned TabM and TabPFN, saved after every fold so they can be stopped and resumed.

Usage: python run_nn_full.py tabm
       python run_nn_full.py tabpfn CONTEXT_SIZE MAX_PASSES PREDICT_CHUNK [teacher] [est=N]
  tabm   - tune_tabm2 best settings + its schedule, 2 seeds; each fold model also predicts the test set.
           Saves results/full/tabm_s{seed}_fold{k}.npz (val_idx, oof, test).
  tabpfn - key-value-cached TabPFN on random CONTEXT_SIZE-row contexts, run in passes: each pass gives
           every fold one context of its training rows, then the test set one context of all training rows,
           so each finished pass is a complete model and later passes are averaged in. Saves
           results/full/tabpfn{K}k_c{pass}_fold{k}.npz (val_idx, oof) and tabpfn{K}k_c{pass}_test.npz.
           "teacher" adds the teacher column (teacher.py). Creating results/HOLD_tabpfn stops it before the next fold.
Finished files are skipped on rerun.
"""
import sys
import time
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from cv import SEED
from preprocess import load_data
from probe_nn import DEVICE, N_SPLITS, fit_predict_tabm, nn_arrays, tabpfn_matrix
from tune_tabm2_config import SCHEDULE

MODEL = sys.argv[1]
if MODEL == "tabpfn":  # python run_nn_full.py tabpfn CONTEXT_SIZE MAX_PASSES PREDICT_CHUNK [teacher]
    CONTEXT, MAX_CONTEXTS, CHUNK = (int(a) for a in sys.argv[2:5])
    TEACHER = "teacher" in sys.argv[5:]
    # est=N: TabPFN's internal ensemble members (default 8); 4 loses nothing at 200k and leaves room for bigger contexts
    EST = next((int(a[4:]) for a in sys.argv[5:] if a.startswith("est=")), None)
TABM_SEEDS = [42, 1]
ROOT = Path(__file__).parent
FULL = ROOT / "results" / "full"

# Delete results/HOLD_{MODEL} to allow this model's run again
if (ROOT / "results" / f"HOLD_{MODEL}").exists():
    print(f"{MODEL} is on hold (results/HOLD_{MODEL} exists), exiting", flush=True)
    sys.exit(0)

X, y_s, X_test, _ = load_data(zero_ratings_as_nan=MODEL == "tabpfn",  # NNs: keep 0 = "n/a" as a value
                              teacher=MODEL == "tabpfn" and TEACHER)
y = y_s.to_numpy()
folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
FULL.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)


if MODEL == "tabm":
    study = optuna.load_study(study_name="tabm_tuning2", storage=f"sqlite:///{ROOT / 'results' / 'tabm_tuning2.db'}")
    cfg = {**study.best_params, **SCHEDULE}
    log(f"TabM settings (trial {study.best_trial.number}): {cfg}")
    X_num, X_cat, cards = nn_arrays(X)
    test_num, test_cat, _ = nn_arrays(X_test)

    for seed in TABM_SEEDS:  # all folds of one seed first, so a stop still leaves complete seeds
        for k, (tr_idx, va_idx) in enumerate(folds, start=1):
            path = FULL / f"tabm_s{seed}_fold{k}.npz"
            if path.exists():
                continue
            start = time.time()
            pred, info = fit_predict_tabm(X_num, X_cat, cards, y, tr_idx, va_idx, seed=seed * 100 + k,
                                          cfg=cfg, X_extra=(test_num, test_cat))
            np.savez(path, val_idx=va_idx, oof=pred, test=info["extra_pred"])
            log(f"tabm seed {seed} fold {k}/{N_SPLITS}: AUC {roc_auc_score(y[va_idx], pred):.5f}, "
                f"{info['epochs']} epochs, {(time.time() - start) / 60:.1f} min -> {path.name}")

elif MODEL == "tabpfn":
    from tabpfn import TabPFNClassifier

    hold = ROOT / "results" / "HOLD_tabpfn"
    Xp, cat_idx = tabpfn_matrix(X)
    Xt, _ = tabpfn_matrix(X_test)
    tag = f"tabpfn{CONTEXT // 1000}k" + ("_teacher" if TEACHER else "") + (f"_e{EST}" if EST else "")
    log(f"TabPFN: {CONTEXT}-row contexts (key-value cache, int8), up to {MAX_CONTEXTS} passes, "
        f"predict chunk {CHUNK}")

    def fit_predict(ctx, rows, seed):
        clf = TabPFNClassifier(device=DEVICE, categorical_features_indices=cat_idx, ignore_pretraining_limits=True,
                               random_state=seed, fit_mode="fit_with_cache", kv_cache_precision="int8",
                               **({"n_estimators": EST} if EST else {}))
        clf.fit(Xp[ctx], y[ctx])
        return np.concatenate([clf.predict_proba(rows[i:i + CHUNK])[:, 1] for i in range(0, len(rows), CHUNK)])

    def sample_context(pool, seed):
        """CONTEXT random rows of pool (stratified), or the whole pool if it isn't bigger than CONTEXT."""
        if CONTEXT >= len(pool):
            return pool
        return train_test_split(pool, train_size=CONTEXT, stratify=y[pool], random_state=seed)[0]

    def stop_if_held():
        if hold.exists():
            log(f"results/{hold.name} exists, stopping (finished files are kept)")
            sys.exit(0)

    for c in range(MAX_CONTEXTS):  # one pass = one context for every fold + the test set
        for k, (tr_idx, va_idx) in enumerate(folds, start=1):
            path = FULL / f"{tag}_c{c}_fold{k}.npz"
            if path.exists():
                continue
            stop_if_held()
            start = time.time()
            ctx = sample_context(tr_idx, SEED + 100 * c + k)
            pred = fit_predict(ctx, Xp[va_idx], seed=SEED + 100 * c + k)
            np.savez(path, val_idx=va_idx, oof=pred)
            log(f"{tag} pass {c + 1} fold {k}/{N_SPLITS}: AUC {roc_auc_score(y[va_idx], pred):.5f}, "
                f"{(time.time() - start) / 60:.1f} min -> {path.name}")
        path = FULL / f"{tag}_c{c}_test.npz"
        if not path.exists():  # test context drawn from all training rows
            stop_if_held()
            start = time.time()
            ctx = sample_context(np.arange(len(y)), 1000 + c)
            np.savez(path, test=fit_predict(ctx, Xt, seed=1000 + c))
            log(f"{tag} pass {c + 1} test: {(time.time() - start) / 60:.1f} min -> {path.name}")
        log(f"{tag} pass {c + 1} complete")

else:
    raise ValueError(f"Unknown model: {MODEL}")
log(f"{MODEL}: all done")
