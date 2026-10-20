"""Tune CatBoost (GPU) with Optuna on the same fixed 80/20 holdout as the XGBoost/LightGBM tuning.

Learning rate fixed at 0.05 and symmetric trees kept (CatBoost's difference from the other two).
No pruning: CatBoost doesn't support custom callbacks on GPU, so each trial runs to early stopping.
Resumable: rerunning continues the study in results/cat_tuning.db.
"""
import time
from pathlib import Path

import optuna
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split

from cv import SEED
from models import CAT_PARAMS
from preprocess import CAT_COLS, load_data

N_TRIALS = 100
N_RANDOM = 30  # first trials sampled at random to cover the space
OUT = Path(__file__).parent / "results"
STORAGE = f"sqlite:///{OUT / 'cat_tuning.db'}"

X, y, _, _ = load_data()
X_tr, X_va, y_tr, y_va = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
train_pool = Pool(X_tr, y_tr, cat_features=CAT_COLS)
valid_pool = Pool(X_va, y_va, cat_features=CAT_COLS)


def objective(trial):
    params = {
        **CAT_PARAMS,
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 0.1, 30, log=True),
        "random_strength": trial.suggest_float("random_strength", 0.01, 10, log=True),
        "border_count": trial.suggest_categorical("border_count", [32, 64, 128, 254]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
    }
    if params["bootstrap_type"] == "Bayesian":
        params["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
    else:
        params["subsample"] = trial.suggest_float("subsample", 0.5, 1.0, step=0.05)

    start = time.time()
    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=valid_pool)
    trial.set_user_attr("trees", model.get_best_iteration() + 1)
    trial.set_user_attr("seconds", time.time() - start)
    return model.get_best_score()["validation"]["AUC"]


OUT.mkdir(exist_ok=True)
# Offset the sampler seed by the trials already run, so a resumed study doesn't repeat them
n_existing = len(optuna.load_study(study_name="cat_tuning", storage=STORAGE).trials) \
    if "cat_tuning" in optuna.get_all_study_names(STORAGE) else 0
study = optuna.create_study(
    study_name="cat_tuning",
    storage=STORAGE,
    load_if_exists=True,
    direction="maximize",
    sampler=optuna.samplers.TPESampler(n_startup_trials=N_RANDOM, seed=42 + n_existing),
)

# Trial 0 = the current defaults (GPU border_count default is 128)
if n_existing == 0:
    study.enqueue_trial({
        "depth": CAT_PARAMS["depth"], "l2_leaf_reg": CAT_PARAMS["l2_leaf_reg"], "random_strength": 1.0,
        "border_count": 128, "bootstrap_type": "Bayesian", "bagging_temperature": 1.0,
    })

# Interrupted trials (e.g. the run was stopped) don't count towards N_TRIALS
finished = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
study.optimize(objective, n_trials=max(N_TRIALS - len(finished), 0))

df = study.trials_dataframe()
df.to_csv(OUT / "cat_tuning_trials.csv", index=False)
done = df[df["state"] == "COMPLETE"].sort_values("value", ascending=False)
print(f"\n{len(done)} complete")
print(f"Defaults (trial 0): {df.loc[0, 'value']:.5f} | best: {study.best_value:.5f}")
cols = ["number", "value", "user_attrs_trees", "user_attrs_seconds"] + [c for c in df.columns if c.startswith("params_")]
print(done[cols].head(10).rename(columns=lambda c: c.replace("params_", "").replace("user_attrs_", "")).round(4).to_string(index=False))
