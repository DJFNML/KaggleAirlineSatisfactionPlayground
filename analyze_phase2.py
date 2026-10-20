"""Analyse Phase 2 trials: main effects and pairwise interactions of the hyperparameters.

A random forest is fitted to (hyperparameters -> AUC) over all trials. Its partial
dependence then shows each parameter's own effect, and for every pair how much of
their joint effect is NOT explained by adding up their separate effects (the interaction).
"""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

OUT = Path(__file__).parent / "results"
LOG_PARAMS = ["min_child_weight", "reg_lambda", "reg_alpha"]  # searched on a log scale
GRID = 10  # grid points per parameter

trials = pd.read_csv(OUT / "phase2_trials.csv")
trials = trials[trials["state"] == "COMPLETE"]
X = trials.filter(like="params_").rename(columns=lambda c: c.removeprefix("params_"))
X[LOG_PARAMS] = np.log10(X[LOG_PARAMS])
y = trials["value"].to_numpy()
params = list(X.columns)

surrogate = RandomForestRegressor(n_estimators=500, min_samples_leaf=3, random_state=42)
r2 = cross_val_score(surrogate, X, y, cv=5, scoring="r2").mean()
surrogate.fit(X, y)
print(f"{len(X)} trials | surrogate CV R^2 = {r2:.2f} (how well hyperparameters explain AUC)")

# Evenly spaced grids (quantiles would bunch up where Optuna's TPE clustered its trials)
grids = {p: np.linspace(X[p].quantile(0.02), X[p].quantile(0.98), GRID) for p in params}
grids["max_depth"] = np.arange(4, 11)
grids["learning_rate"] = np.round(np.arange(0.01, 0.105, 0.01), 2)


def partial_dependence(cols):
    """Average predicted AUC with `cols` fixed to each grid combination."""
    mesh = np.array(np.meshgrid(*[grids[c] for c in cols], indexing="ij")).reshape(len(cols), -1).T
    out = []
    for point in mesh:
        Xp = X.copy()
        Xp[list(cols)] = point
        out.append(surrogate.predict(Xp).mean())
    return np.array(out).reshape([len(grids[c]) for c in cols])


pd1 = {p: partial_dependence([p]) for p in params}

main = pd.DataFrame(
    {
        "effect_range": [pd1[p].max() - pd1[p].min() for p in params],
        "best_value": [grids[p][pd1[p].argmax()] for p in params],
    },
    index=params,
)
main.loc[LOG_PARAMS, "best_value"] = 10 ** main.loc[LOG_PARAMS, "best_value"]
main = main.sort_values("effect_range", ascending=False)
print("\n=== Main effects (AUC swing across each parameter's range) ===")
print(main.round(5).to_string())

rows, pd2 = [], {}
for a, b in combinations(params, 2):
    joint = partial_dependence([a, b])
    additive = pd1[a][:, None] + pd1[b][None, :]
    resid = (joint - joint.mean()) - (additive - additive.mean())
    pd2[(a, b)] = joint
    rows.append({
        "pair": f"{a} x {b}",
        "interaction": resid.max() - resid.min(),  # AUC swing not explained by separate effects
        "share_of_joint": resid.var() / max(joint.var(), 1e-18),  # Friedman H^2
        "joint_range": joint.max() - joint.min(),
    })

pairs = pd.DataFrame(rows).sort_values("interaction", ascending=False).reset_index(drop=True)
pairs.to_csv(OUT / "phase2_interactions.csv", index=False)
print("\n=== Pairwise interactions (strongest first) ===")
print(pairs.round(5).to_string())

print("\n=== Joint partial dependence for the top 6 pairs (AUC - 0.957, x1e5) ===")
for pair in pairs["pair"].head(6):
    a, b = pair.split(" x ")
    fmt = lambda p, v: f"{10 ** v:.3g}" if p in LOG_PARAMS else f"{v:.3g}"
    table = pd.DataFrame(
        (pd2[(a, b)] - 0.957) * 1e5,
        index=[fmt(a, v) for v in grids[a]],
        columns=[fmt(b, v) for v in grids[b]],
    ).rename_axis(index=a, columns=b)
    print(f"\n{pair}\n{table.round(1).to_string()}")
