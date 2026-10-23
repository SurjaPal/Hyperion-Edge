"""
One-shot held-out evaluation. Selection rule fixed BEFORE looking at held-out data:
 AE: among all tuning trials with F1 >= best-0.01, pick the fewest parameters.
 Baselines (PCA, IsolationForest) get the same threshold-tuning budget on the TUNE files.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, optuna
from lib import *
import tune as T

optuna.logging.set_verbosity(optuna.logging.WARNING)
files = load_files(); TUNE, HELD = split_tune_heldout(files)
study = optuna.load_study(study_name="ae_tune", storage=T.DB)
done = [t for t in study.trials if t.values]
best_f1 = max(t.values[0] for t in done)
pick = sorted([t for t in done if t.values[0] >= best_f1 - 0.01], key=lambda t: (t.values[1], -t.values[0]))[0]
p = pick.params
nl = p["n_layers"]
ae_cfg = dict(window=p["window"], hidden=tuple(p[f"h{i}"] for i in range(nl)), bottleneck=p["bottleneck"], alpha=p["alpha"],
              lr=p["lr"], batch=p["batch"], activation=p["activation"], epochs=p["epochs"], q=p["q"], scale=p["scale"], smooth=p["smooth"])
print("selected trial", pick.number, "tune F1", pick.values[0], "params", pick.values[1]); print(ae_cfg)

def tune_baseline(kind, n):
    st = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
    def obj(t):
        c = dict(q=t.suggest_float("q", .95, .999), scale=t.suggest_float("scale", 1, 3), smooth=t.suggest_categorical("smooth", [1,3,5,9,15]))
        if kind == "pca": c["k"] = t.suggest_int("k", 1, 6)
        return evaluate(kind, TUNE, c)["F1"]
    st.optimize(obj, n_trials=n); return st.best_params, st.best_value

pca_cfg, pca_tune = tune_baseline("pca", 120)
if_cfg, if_tune = tune_baseline("iforest", 25)
default_ae = dict(window=8, hidden=(32,), bottleneck=6, alpha=1e-4, lr=1e-3, epochs=120, batch=32, activation="relu", q=.99, scale=1.5, smooth=5)

def summarize(rs):
    ks = ["F1", "FAR", "MAR", "precision", "recall"]
    return {k: (round(float(np.mean([r[k] for r in rs])), 4), round(float(np.std([r[k] for r in rs])), 4)) for k in ks}

res = {"selected_trial": pick.number, "tune_F1": pick.values[0], "params": int(pick.values[1]), "ae_cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in ae_cfg.items()}}
seeds = [0, 1, 2, 3, 4]
runs = {"AE tuned": [evaluate("ae", HELD, {**ae_cfg, "seed": s}) for s in seeds],
        "AE default (untuned)": [evaluate("ae", HELD, {**default_ae, "seed": s}) for s in seeds],
        "PCA (thresholds+k tuned)": [evaluate("pca", HELD, pca_cfg)], "IsolationForest (thresholds tuned)": [evaluate("iforest", HELD, {**if_cfg, "seed": 0})]}
res["heldout_files"] = [f["name"] for f in HELD]; res["tune_files"] = [f["name"] for f in TUNE]
res["heldout"] = {k: summarize(v) for k, v in runs.items()}
res["baseline_tune_F1"] = {"pca": pca_tune, "iforest": if_tune}
# detection delay for tuned AE seed 0 (samples after onset until first alarm)
d = [r["delay"] for r in runs["AE tuned"][0]["per_file"]]
res["ae_tuned_delay_median_s"] = float(np.median([x for x in d if x is not None])); res["ae_tuned_missed_files"] = sum(x is None for x in d)
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "heldout.json"), "w"), indent=1, default=str)
for k, v in res["heldout"].items(): print(k.ljust(38), {a: b[0] for a, b in v.items()}, "F1 std", v["F1"][1])
print("tuned AE median detection delay (s):", res["ae_tuned_delay_median_s"], "missed files:", res["ae_tuned_missed_files"])
