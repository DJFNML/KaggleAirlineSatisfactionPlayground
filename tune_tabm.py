"""Tune TabM with Optuna on the same fixed 80/20 holdout as the GBDT tuning.

Unpromising trials are pruned on their per-epoch early-stopping AUC.
Resumable: rerunning continues the study in results/tabm_tuning.db.
"""
from pathlib import Path

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from cv import SEED
from preprocess import load_data
from probe_nn import TABM, fit_predict_tabm, nn_arrays

N_TRIALS = 50
N_RANDOM = 15  # first trials sampled at random to cover the space
OUT = Path(__file__).parent / "results"
STORAGE = f"sqlite:///{OUT / 'tabm_tuning.db'}"

X, y_s, _, _ = load_data(zero_ratings_as_nan=False)  # NNs: keep 0 = "n/a" as a value
y = y_s.to_numpy()
# Same rows as the other models' tuning holdout (split depends only on n, y and the seed)
tr_idx, va_idx = train_test_split(np.arange(len(y)), test_size=0.2, stratify=y, random_state=SEED)
arrays = {False: nn_arrays(X, ratings_as_cat=False), True: nn_arrays(X, ratings_as_cat=True)}


def objective(trial):
    ratings_as_cat = trial.suggest_categorical("ratings_as_cat", [False, True])
    cfg = {
        "lr": trial.suggest_float("lr", 3e-4, 5e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 3e-2, log=True),
        "n_blocks": trial.suggest_int("n_blocks", 1, 4),
        "d_block": trial.suggest_categorical("d_block", [256, 384, 512, 768, 1024]),
        "dropout": trial.suggest_float("dropout", 0, 0.4, step=0.05),
        "d_embedding": trial.suggest_categorical("d_embedding", [8, 16, 24, 32]),
        "n_bins": trial.suggest_categorical("n_bins", [16, 32, 48, 64, 96]),
        "batch_size": trial.suggest_categorical("batch_size", [512, 1024, 2048]),
    }
    X_num, X_cat, cards = arrays[ratings_as_cat]
    pred, info = fit_predict_tabm(X_num, X_cat, cards, y, tr_idx, va_idx, seed=SEED, cfg=cfg, trial=trial)
    trial.set_user_attr("epochs", info["epochs"])
    return roc_auc_score(y[va_idx], pred)


OUT.mkdir(exist_ok=True)
# Offset the sampler seed by the trials already run, so a resumed study doesn't repeat them
n_existing = len(optuna.load_study(study_name="tabm_tuning", storage=STORAGE).trials) \
    if "tabm_tuning" in optuna.get_all_study_names(STORAGE) else 0
study = optuna.create_study(
    study_name="tabm_tuning",
    storage=STORAGE,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=42 + n_existing),
    # Prune a trial if it's below the median of earlier trials at the same epoch
    pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5),
)

# Trial 0 = the defaults used in the probe, so we know where tuning starts from
if n_existing == 0:
    study.enqueue_trial({"ratings_as_cat": False, **{k: TABM[k] for k in (
        "lr", "weight_decay", "n_blocks", "d_block", "dropout", "d_embedding", "n_bins", "batch_size")}})

finished = study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
study.optimize(objective, n_trials=max(N_TRIALS - len(finished), 0))

df = study.trials_dataframe()
df.to_csv(OUT / "tabm_tuning_trials.csv", index=False)
done = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
print(f"\n{len(done)} complete, {(df['state'] == 'PRUNED').sum()} pruned")
print(f"Defaults (trial 0): {df.loc[0, 'value']:.5f} | best: {study.best_value:.5f}")
cols = ["number", "value", "user_attrs_epochs"] + [c for c in df.columns if c.startswith("params_")]
print(done[cols].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", "")).round(5).to_string(index=False))
