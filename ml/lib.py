"""
Shared ML library for the Sentinel-Edge experiments.

Protocol (adapted from the public SKAB benchmark): each file is one experiment on a real
centrifugal-pump testbed. The device sees the first N_TRAIN=400 samples of a run,
assumed normal: rows [0:300) train the model, rows [300:400) calibrate the alarm
threshold on unseen normal data. Everything after row 400 is the test stream.
No labels are ever used to fit a model or a threshold on a file being evaluated.
"""
from __future__ import annotations
import glob, os, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore")
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "skab")
COLS = ["Accelerometer1RMS","Accelerometer2RMS","Current","Pressure","Temperature","Thermocouple","Voltage","Volume Flow RateRMS"]
N_TRAIN, N_FIT = 400, 300
EXCLUDED = {"other/2.csv"}   # anomaly starts at row 104, i.e. inside the "normal" warm-up window


def load_files():
    out = []
    for f in sorted(glob.glob(os.path.join(ROOT, "*", "*.csv"))):
        name = os.path.relpath(f, ROOT).replace(os.sep, "/")
        if name.startswith("anomaly-free") or name in EXCLUDED:
            continue
        d = pd.read_csv(f, sep=";")
        out.append(dict(name=name, group=name.split("/")[0], X=d[COLS].values.astype(float),
                        y=d["anomaly"].values.astype(int)))
    return out


def load_anomaly_free():
    d = pd.read_csv(os.path.join(ROOT, "anomaly-free", "anomaly-free.csv"), sep=";")
    return d[COLS].values.astype(float)


def split_tune_heldout(files):
    """Deterministic, group-stratified: alternate files inside each group."""
    tune, held = [], []
    by = {}
    for f in files: by.setdefault(f["group"], []).append(f)
    for g in sorted(by):
        for i, f in enumerate(sorted(by[g], key=lambda r: int(r["name"].split("/")[1].split(".")[0]))):
            (tune if i % 2 == 0 else held).append(f)
    return tune, held


def windows(Z, W):
    """Flattened sliding windows; row i covers samples [i, i+W), i.e. ends at i+W-1."""
    n = len(Z) - W + 1
    idx = np.arange(W)[None, :] + np.arange(n)[:, None]
    return Z[idx].reshape(n, W * Z.shape[1])


class WindowAE:
    def __init__(self, window=8, hidden=(32,), bottleneck=6, alpha=1e-4, lr=1e-3, epochs=120,
                 batch=32, activation="relu", seed=0):
        self.W, self.seed = window, seed
        layers = tuple(hidden) + (bottleneck,) + tuple(hidden)[::-1]
        self.m = MLPRegressor(hidden_layer_sizes=layers, activation=activation, alpha=alpha,
                              learning_rate_init=lr, max_iter=epochs, batch_size=batch,
                              random_state=seed, tol=1e-7, n_iter_no_change=epochs)

    def fit(self, Zfit):
        A = windows(Zfit, self.W); self.m.fit(A, A); return self

    def errors(self, Z):
        """(n_windows, 8) per-channel reconstruction error; row i ends at sample i+W-1."""
        A = windows(Z, self.W); R = self.m.predict(A)
        return ((R - A) ** 2).reshape(len(A), self.W, Z.shape[1]).mean(axis=1)

    def n_params(self):
        return int(sum(c.size for c in self.m.coefs_) + sum(b.size for b in self.m.intercepts_))


def _scores_iforest(Z, fit_rows, seed=0):
    m = IsolationForest(n_estimators=100, random_state=seed).fit(Z[:fit_rows])
    return -m.decision_function(Z)


def _scores_pca(Z, fit_rows, k=3):
    p = PCA(n_components=k).fit(Z[:fit_rows])
    return ((Z - p.inverse_transform(p.transform(Z))) ** 2).sum(axis=1)


def _smooth(s, k):
    if k <= 1: return s
    return pd.Series(s).rolling(k, min_periods=1).median().values


def score_stream(kind, X, cfg):
    """
    Returns (score_per_sample aligned to X, errors_per_channel or None).
    Samples with no score yet (window warm-up) get 0. Normalisation is fit on the
    first N_FIT rows only (site-local calibration).
    """
    mu, sd = X[:N_FIT].mean(0), X[:N_FIT].std(0) + 1e-6
    Z = (X - mu) / sd
    if kind == "iforest":
        return _scores_iforest(Z, N_FIT, cfg.get("seed", 0)), None
    if kind == "pca":
        return _scores_pca(Z, N_FIT, cfg.get("k", 3)), None
    if kind == "ae":
        ae = cfg["_model"] if "_model" in cfg else WindowAE(**{k: v for k, v in cfg.items() if k in
              ("window","hidden","bottleneck","alpha","lr","epochs","batch","activation","seed")}).fit(Z[:N_FIT])
        E = ae.errors(Z); W = ae.W
        full = np.zeros(len(X)); full[W - 1:] = E.mean(1)
        ch = np.zeros((len(X), X.shape[1])); ch[W - 1:] = E
        return full, ch
    raise ValueError(kind)


def alarms(score, cfg):
    """Threshold from calibration slice [N_FIT:N_TRAIN) of the same stream; then smooth."""
    cal = score[N_FIT:N_TRAIN]
    thr = np.quantile(cal, cfg.get("q", 0.99)) * cfg.get("scale", 1.5)
    return (_smooth(score, cfg.get("smooth", 5)) > thr).astype(int)


def evaluate(kind, files, cfg):
    """Pooled point-wise metrics over test rows (>= N_TRAIN), plus per-file details."""
    TP = FP = FN = TN = 0; per = []
    for f in files:
        s, _ = score_stream(kind, f["X"], cfg); a = alarms(s, cfg)[N_TRAIN:]; y = f["y"][N_TRAIN:]
        tp = int(((a == 1) & (y == 1)).sum()); fp = int(((a == 1) & (y == 0)).sum())
        fn = int(((a == 0) & (y == 1)).sum()); tn = int(((a == 0) & (y == 0)).sum())
        TP += tp; FP += fp; FN += fn; TN += tn
        onset = int(np.argmax(y)); hit = np.where(a[onset:] == 1)[0]
        per.append(dict(name=f["name"], f1=2*tp/max(1, 2*tp+fp+fn), delay=int(hit[0]) if len(hit) else None,
                        false_alarms_pre=int(a[:onset].sum())))
    f1 = 2*TP/max(1, 2*TP+FP+FN)
    return dict(F1=round(f1, 4), FAR=round(100*FP/max(1, FP+TN), 2), MAR=round(100*FN/max(1, FN+TP), 2),
                precision=round(TP/max(1, TP+FP), 4), recall=round(TP/max(1, TP+FN), 4), per_file=per)
