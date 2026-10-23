"""
Sensor-fault vs process-fault discrimination using how concentrated the reconstruction error is on one channel.
Real process faults = the labelled SKAB anomalies. Sensor faults = INJECTED into real *normal* test rows (synthetic
perturbations on real data): stuck-at, dropout-to-zero, spike, drift on one random channel.
Threshold tau is chosen on the TUNE files (Youden J) and applied unchanged to held-out files.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from lib import *
RES = os.path.join(os.path.dirname(__file__), "..", "results")
cfg = json.load(open(os.path.join(RES, "heldout.json")))["ae_cfg"]; cfg["hidden"] = tuple(cfg["hidden"])
files = load_files(); TUNE, HELD = split_tune_heldout(files)
KEYS = ("window","hidden","bottleneck","alpha","lr","epochs","batch","activation")
rng = np.random.default_rng(7)

def inject(X, kind, c, t0, sd):
    Y = X.copy()
    if kind == "stuck": Y[t0:t0+60, c] = X[t0, c]
    elif kind == "dropout": Y[t0:t0+40, c] = 0.0
    elif kind == "spike": Y[t0:t0+6, c] += 8 * sd[c]
    elif kind == "drift": Y[t0:t0+60, c] += np.linspace(0, 6 * sd[c], 60)
    return Y, (t0, t0 + {"stuck": 60, "dropout": 40, "spike": 6, "drift": 60}[kind])

def top_share(E, sel):
    E = np.asarray(E)[sel]
    if len(E) == 0: return None
    m = E.mean(0); return float(m.max() / (m.sum() + 1e-12))

def collect(fs):
    proc, sens, undetected = [], [], 0
    for f in fs:
        X = f["X"]; mu, sd = X[:N_FIT].mean(0), X[:N_FIT].std(0) + 1e-6
        ae = WindowAE(**{k: cfg[k] for k in KEYS}, seed=0).fit(((X - mu) / sd)[:N_FIT]); W = ae.W
        def stream(Xm):
            E = ae.errors((Xm - mu) / sd); s = np.zeros(len(Xm)); s[W-1:] = E.mean(1)
            ch = np.zeros((len(Xm), 8)); ch[W-1:] = E; return s, ch
        s, ch = stream(X); a = alarms(s, cfg); y = f["y"]
        sel = np.where((a == 1) & (y == 1))[0]; v = top_share(ch, sel)
        if v is not None: proc.append(v)
        onset = int(np.argmax(y))
        for kind in ("stuck", "dropout", "spike", "drift"):
            for _ in range(2):
                c = int(rng.integers(0, 8)); t0 = int(rng.integers(N_TRAIN + 20, max(N_TRAIN + 21, onset - 100)))
                Xm, (i0, i1) = inject(X, kind, c, t0, sd); sm, chm = stream(Xm); am = alarms(sm, cfg)
                idx = np.arange(i0, min(len(X), i1 + W)); hit = idx[am[idx] == 1]; v = top_share(chm, hit)
                if v is None: undetected += 1
                else: sens.append((kind, v))
    return proc, sens, undetected

def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))

pt, st, ut = collect(TUNE)
grid = np.linspace(0.2, 0.95, 76); sv = np.array([v for _, v in st])
J = [(np.mean(sv >= t) - np.mean(np.array(pt) >= t)) for t in grid]; tau = float(grid[int(np.argmax(J))])
ok = [t for t in grid if np.mean(np.array(pt) >= t) <= 0.10]
tau_safe = float(ok[0]) if ok else 0.95   # smallest tau whose process-relabel rate on TUNE is <= 10%
ph, sh, uh = collect(HELD); svh = np.array([v for _, v in sh])
res = dict(tau=round(tau, 3), tau_safe=round(tau_safe, 3),
    heldout_safe_operating_point=None, tune=dict(n_process=len(pt), n_sensor=len(st), sensor_undetected=ut),
    heldout=dict(n_process_files=len(ph), n_sensor_alarmed=len(sh), sensor_faults_not_alarmed=uh,
        AUC_top_share=round(auc(svh, ph), 3), sensor_flagged_as_sensor=round(float(np.mean(svh >= tau)), 3),
        process_wrongly_flagged_as_sensor=round(float(np.mean(np.array(ph) >= tau)), 3),
        median_top_share_sensor=round(float(np.median(svh)), 3), median_top_share_process=round(float(np.median(ph)), 3),
        per_kind_flagged={k: round(float(np.mean([v >= tau for kk, v in sh if kk == k])), 3) for k in ("stuck","dropout","spike","drift") if any(kk == k for kk, _ in sh)}))
res['heldout_safe_operating_point'] = dict(tau=round(tau_safe,3), sensor_flagged_as_sensor=round(float(np.mean(svh >= tau_safe)),3), process_wrongly_flagged_as_sensor=round(float(np.mean(np.array(ph) >= tau_safe)),3))
json.dump(res, open(os.path.join(RES, "sensor_gate.json"), "w"), indent=1); print(json.dumps(res, indent=1))
