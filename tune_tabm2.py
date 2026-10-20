"""Careful TabM tuning: each trial is scored on 2 CV folds (different seeds), with no pruning.

Uses folds 4 and 5 of the 15-fold split, so the probe folds (1-3) stay untouched for the blend
comparison. Learning rate halves on plateaus to make early stopping less erratic.
The defaults are run twice with different seeds as a noise check.
Resumable: rerunning continues the study in results/tabm_tuning2.db.
"""
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from cv import SEED
from preprocess import load_data
from probe_nn import N_SPLITS, TABM, fit_predict_tabm, nn_arrays

N_TRIALS = 30
N_RANDOM = 10
TUNE_FOLDS = [3, 4]  # 0-based: folds 4 and 5 of the 15-fold split
SCHEDULE = {"max_epochs": 80, "patience": 10, "lr_decay_patience": 3}
OUT = Path(__file__).parent / "results"
STORAGE = f"sqlite:///{OUT / 'tabm_tuning2.db'}"

X, y_s, _, _ = load_data(zero_ratings_as_nan=False)
y = y_s.to_numpy()
X_num, X_cat, cards = nn_arrays(X, ratings_as_cat=False)  # numeric ratings won the first study
all_folds = list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED).split(X, y))
folds = [all_folds[i] for i in TUNE_FOLDS]


def objective(trial):
    cfg = {
        **SCHEDULE,
        "lr": trial.suggest_float("lr", 2e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True),
        "n_blocks": trial.suggest_int("n_blocks", 2, 5),
        "d_block": trial.suggest_categorical("d_block", [384, 512, 768]),
        "dropout": trial.suggest_float("dropout", 0, 0.2, step=0.05),
        "d_embedding": trial.suggest_categorical("d_embedding", [8, 16, 24]),
        "n_bins": trial.suggest_categorical("n_bins", [48, 64, 96, 128]),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096]),
    }
    aucs, epochs = [], []
    for j, (tr_idx, va_idx) in enumerate(folds):
        # A different seed per trial and fold, so repeated settings measure seed noise
        pred, info = fit_predict_tabm(X_num, X_cat, cards, y, tr_idx, va_idx,
                                      seed=1000 * trial.number + j, cfg=cfg)
        aucs.append(roc_auc_score(y[va_idx], pred))
        epochs.append(info["epochs"])
    for j, (a, e) in enumerate(zip(aucs, epochs)):
        trial.set_user_attr(f"auc_fold{TUNE_FOLDS[j] + 1}", a)
        trial.set_user_attr(f"epochs_fold{TUNE_FOLDS[j] + 1}", e)
    return float(np.mean(aucs))


OUT.mkdir(exist_ok=True)
n_existing = len(optuna.load_study(study_name="tabm_tuning2", storage=STORAGE).trials) \
    if "tabm_tuning2" in optuna.get_all_study_names(STORAGE) else 0
study = optuna.create_study(
    study_name="tabm_tuning2",
    storage=STORAGE,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=42 + n_existing),
)

if n_existing == 0:
    defaults = {k: TABM[k] for k in ("lr", "weight_decay", "n_blocks", "d_block", "dropout",
                                     "d_embedding", "n_bins", "batch_size")}
    study.enqueue_trial(defaults)  # trial 0: defaults
    study.enqueue_trial(defaults)  # trial 1: defaults again, other seeds = noise check
    study.enqueue_trial({  # trial 2: best of the first study
        "lr": 0.00052, "weight_decay": 1e-5, "n_blocks": 4, "d_block": 512, "dropout": 0.05,
        "d_embedding": 8, "n_bins": 96, "batch_size": 2048,
    })

finished = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
study.optimize(objective, n_trials=max(N_TRIALS - len(finished), 0))

df = study.trials_dataframe()
df.to_csv(OUT / "tabm_tuning2_trials.csv", index=False)
done = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
print(f"\n{len(done)} complete")
print(f"Defaults: trial 0 {df.loc[0, 'value']:.5f}, trial 1 {df.loc[1, 'value']:.5f} (seed noise) | "
      f"best: {study.best_value:.5f} (trial {study.best_trial.number})")
cols = ["number", "value"] + [c for c in df.columns if c.startswith(("user_attrs_", "params_"))]
print(done[cols].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", ""))
      .round(5).to_string(index=False))
