"""Feature experiments: tuned LightGBM (trial 119), 5-fold CV, 2 seeds, one row per feature set.

Runs on CPU with 8 threads so GPU jobs can run alongside. Each experiment is load_data() kwargs
plus an optional function that adds columns to (X, X_test). Results go to results/compare_fe.csv.
"""
import sys
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import FEATURE_SETS
from models import LGB_PARAMS, fit_lgb, run_cv_model
from preprocess import load_data

N_SPLITS = 5
SEEDS = [42, 1]
ROOT = Path(__file__).parent
EXPERIMENTS = sys.argv[1:] or ["baseline", "teacher"]  # names from features.FEATURE_SETS

optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.load_study(study_name="lgb_tuning", storage=f"sqlite:///{ROOT / 'results' / 'lgb_tuning.db'}")
PARAMS = {**LGB_PARAMS, **study.trials[119].params, "num_threads": 8}

out = ROOT / "results" / "compare_fe.csv"
rows = pd.read_csv(out).to_dict("records") if out.exists() else []
done = {r["experiment"] for r in rows}
for name in EXPERIMENTS:
    if name in done:
        print(f"{name}: already in {out.name}, skipping")
        continue
    load_kwargs, add_features = FEATURE_SETS[name]
    X, y, X_test, _ = load_data(**load_kwargs)
    if add_features:
        X, X_test = add_features(X, X_test)
    runs = [run_cv_model(X, y, X_test, N_SPLITS, fit_lgb({**PARAMS, "seed": s}), label=f"{name} seed {s}")
            for s in SEEDS]
    seed_aucs = [r["oof_auc"] for r in runs]
    rows.append({
        "experiment": name, "n_features": X.shape[1],
        **{f"seed{s}": a for s, a in zip(SEEDS, seed_aucs)},
        "seed_avg": roc_auc_score(y, np.mean([r["oof"] for r in runs], axis=0)),
        "seed_spread": max(seed_aucs) - min(seed_aucs),
        "minutes": sum(r["seconds"] for r in runs) / 60,
    })
    pd.DataFrame(rows).to_csv(out, index=False)  # save after each experiment

df = pd.DataFrame(rows).set_index("experiment")
df.insert(1, "vs_baseline", df["seed_avg"] - df.loc["baseline", "seed_avg"])
print("\n" + df.round(5).to_string())
