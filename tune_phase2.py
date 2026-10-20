"""Phase 2: search all 8 hyperparameters together with Optuna, to measure interactions."""
from pathlib import Path

import optuna

from cv import PARAMS, make_holdout, run_holdout
from preprocess import load_data

N_TRIALS = 250
N_RANDOM = 100  # first trials sampled uniformly at random, so the whole space is covered
MAX_ROUNDS = 10000
OUT = Path(__file__).parent / "results"

X, y, _, _ = load_data()
dtrain, dvalid = make_holdout(X, y)  # same split as Phase 1


def objective(trial):
    params = {
        **PARAMS,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.10, step=0.01),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_float("min_child_weight", 1, 100, log=True),
        "gamma": trial.suggest_float("gamma", 0, 2),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0, step=0.05),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0, step=0.05),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 100, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 30, log=True),
    }
    res = run_holdout(dtrain, dvalid, params, MAX_ROUNDS)
    trial.set_user_attr("trees", res["trees"])
    trial.set_user_attr("seconds", res["seconds"])
    return res["auc"]


OUT.mkdir(exist_ok=True)
study = optuna.create_study(
    study_name="xgb_phase2",
    storage=f"sqlite:///{OUT / 'phase2.db'}",  # resumable: rerunning continues the study
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=42),
)
remaining = N_TRIALS - len(study.trials)
study.optimize(objective, n_trials=max(remaining, 0))

study.trials_dataframe().to_csv(OUT / "phase2_trials.csv", index=False)
print(f"\nBest AUC: {study.best_value:.5f}")
print("Best params:", study.best_params)
