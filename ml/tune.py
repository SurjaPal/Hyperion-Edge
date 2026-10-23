"""
Multi-objective hyperparameter search (Optuna, TPE) for the windowed autoencoder.
Objectives: maximise pooled F1 on the TUNE files, minimise parameter count (edge budget).
Held-out files are never touched here. Resumable: trials persist in results/optuna.db.
usage: python ml/tune.py <seconds_budget>
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(__file__))
import optuna
from lib import *

DB = "sqlite:///" + os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "optuna.db")
files = load_files(); TUNE, HELD = split_tune_heldout(files)

def cfg_from(t):
    n_layers = t.suggest_int("n_layers", 1, 2)
    hidden = tuple(t.suggest_categorical(f"h{i}", [16, 32, 64, 128]) for i in range(n_layers))
    return dict(window=t.suggest_categorical("window", [4, 6, 8, 12, 16]), hidden=hidden,
                bottleneck=t.suggest_categorical("bottleneck", [2, 3, 4, 6, 8]),
                alpha=t.suggest_float("alpha", 1e-6, 1e-1, log=True), lr=t.suggest_float("lr", 1e-4, 1e-2, log=True),
                batch=t.suggest_categorical("batch", [16, 32, 64]), activation=t.suggest_categorical("activation", ["relu", "tanh"]),
                epochs=t.suggest_int("epochs", 40, 200, step=20),
                q=t.suggest_float("q", 0.95, 0.999), scale=t.suggest_float("scale", 1.0, 3.0),
                smooth=t.suggest_categorical("smooth", [1, 3, 5, 9, 15]), seed=0)

def objective(t):
    cfg = cfg_from(t)
    r = evaluate("ae", TUNE, cfg)
    n_in = cfg["window"] * 8; sizes = [n_in, *cfg["hidden"], cfg["bottleneck"], *cfg["hidden"][::-1], n_in]
    params = sum(a * b + b for a, b in zip(sizes[:-1], sizes[1:]))
    t.set_user_attr("FAR", r["FAR"]); t.set_user_attr("MAR", r["MAR"])
    return r["F1"], params

if __name__ == "__main__":
    budget = float(sys.argv[1]) if len(sys.argv) > 1 else 120
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="ae_tune", storage=DB, load_if_exists=True,
        directions=["maximize", "minimize"], sampler=optuna.samplers.TPESampler(multivariate=True, seed=42))
    study.optimize(objective, timeout=budget, n_jobs=min(4, os.cpu_count() or 1))
    done = [t for t in study.trials if t.values]
    best = sorted(done, key=lambda t: -t.values[0])[:3]
    print("trials so far:", len(done), "| pareto size:", len(study.best_trials))
    for t in best: print(" F1=%.4f params=%d FAR=%.1f MAR=%.1f" % (t.values[0], t.values[1], t.user_attrs["FAR"], t.user_attrs["MAR"]))
