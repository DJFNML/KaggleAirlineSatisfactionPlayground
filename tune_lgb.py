"""Tune LightGBM with Optuna on the same fixed 80/20 holdout as the XGBoost tuning.

Learning rate is fixed at 0.02; unpromising trials are pruned partway through training.
Resumable: rerunning continues the study in results/lgb_tuning.db.
"""
from pathlib import Path

import lightgbm as lgb
import optuna
import pandas as pd
from sklearn.model_selection import train_test_split

from cv import EARLY_STOPPING, SEED
from models import LGB_PARAMS, MAX_ROUNDS
from preprocess import load_data

N_TRIALS = 150
N_RANDOM = 50  # first trials sampled at random to cover the space
REPORT_EVERY = 50  # trees between pruning checks
OUT = Path(__file__).parent / "results"

X, y, _, _ = load_data()
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def pruning_callback(trial):
    """Report validation AUC to Optuna every REPORT_EVERY trees; stop the trial if it's lagging."""
    def callback(env):
        if (env.iteration + 1) % REPORT_EVERY:
            return
        auc = next(r[2] for r in env.evaluation_result_list if r[1] == "auc")
        trial.report(auc, step=env.iteration + 1)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return callback


def objective(trial):
    params = {
        **LGB_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 15, 1023, log=True),
        "max_depth": trial.suggest_categorical("max_depth", [-1, 6, 8, 10, 12, 16]),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 500, log=True),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.3, 1.0, step=0.05),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0, step=0.05),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 30, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 100, log=True),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 2),
    }
    dtrain = lgb.Dataset(X_tr, y_tr)
    dvalid = lgb.Dataset(X_va, y_va, reference=dtrain)
    model = lgb.train(
        params, dtrain, MAX_ROUNDS, valid_sets=[dvalid],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), pruning_callback(trial)],
    )
    trial.set_user_attr("trees", model.best_iteration)
    return model.best_score["valid_0"]["auc"]


OUT.mkdir(exist_ok=True)
STORAGE = f"sqlite:///{OUT / 'lgb_tuning.db'}"
# Offset the sampler seed by the trials already run, so a resumed study doesn't repeat them
n_existing = len(optuna.load_study(study_name="lgb_tuning", storage=STORAGE).trials) \
    if "lgb_tuning" in optuna.get_all_study_names(STORAGE) else 0
study = optuna.create_study(
    study_name="lgb_tuning",
    storage=STORAGE,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=42 + n_existing),
    # Prune a trial if it's below the median of earlier trials at the same tree count
    pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=200),
)

# Trial 0 = the current defaults, so we know where the tuning starts from
if len(study.trials) == 0:
    study.enqueue_trial({
        "num_leaves": LGB_PARAMS["num_leaves"], "max_depth": -1,
        "min_child_samples": LGB_PARAMS["min_child_samples"],
        "feature_fraction": LGB_PARAMS["feature_fraction"],
        "bagging_fraction": LGB_PARAMS["bagging_fraction"],
        "lambda_l1": 1e-3, "lambda_l2": 1e-3, "min_gain_to_split": 0,
    })

# Interrupted trials (e.g. the run was stopped) don't count towards N_TRIALS
finished = study.get_trials(states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED))
study.optimize(objective, n_trials=max(N_TRIALS - len(finished), 0))

df = study.trials_dataframe()
df.to_csv(OUT / "lgb_tuning_trials.csv", index=False)
done = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
print(f"\n{len(done)} complete, {(df['state'] == 'PRUNED').sum()} pruned")
print(f"Defaults (trial 0): {df.loc[0, 'value']:.5f} | best: {study.best_value:.5f}")
cols = ["number", "value", "user_attrs_trees"] + [c for c in df.columns if c.startswith("params_")]
print(done[cols].head(10).rename(columns=lambda c: c.replace("params_", "")).round(4).to_string(index=False))
