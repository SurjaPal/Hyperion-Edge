"""
Federated pre-training of the autoencoder across 5 non-IID "sites" (real pump-testbed data, TUNE files only),
then per-device fine-tuning on very little local normal data ("cold start"), evaluated on held-out files.
Arms: scratch | FedAvg | central(pooled) | FedAvg/median/trimmed under a sign-flip poisoning attacker.
Also: hash-chained HMAC update commitments (integrity / replay), tested separately.
"""
import sys, os, json, hmac, hashlib, time
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from lib import *

RES = os.path.join(os.path.dirname(__file__), "..", "results")
cfgj = json.load(open(os.path.join(RES, "heldout.json")))["ae_cfg"]
ARCH = dict(window=cfgj["window"], hidden=tuple(cfgj["hidden"]), bottleneck=cfgj["bottleneck"], alpha=cfgj["alpha"], activation=cfgj["activation"])
files = load_files(); TUNE, HELD = split_tune_heldout(files)
W = ARCH["window"]

def norm(X, n=N_FIT): mu, sd = X[:n].mean(0), X[:n].std(0) + 1e-6; return (X - mu) / sd

def site_windows():
    num = lambda f: int(f["name"].split("/")[1].split(".")[0])
    g = lambda grp: sorted([f for f in TUNE if f["group"] == grp], key=num)
    v1, v2, ot = g("valve1"), g("valve2"), g("other")
    groups = [v1[:4], v1[4:], v2, ot[:4], ot[4:]]
    af = load_anomaly_free(); chunks = np.array_split(af, 5)
    out = []
    for grp, ch in zip(groups, chunks):
        A = [windows(norm(f["X"][:N_TRAIN]), W) for f in grp] + [windows(norm(ch[i:i + 1000]), W) for i in range(0, len(ch) - 999, 1000)]
        out.append(np.vstack(A).astype(np.float64))
    return out

def new_model(seed=0, lr=2e-3):
    m = WindowAE(**ARCH, lr=lr, batch=32, seed=seed)
    d = ARCH["window"] * 8; m.m.partial_fit(np.zeros((2, d)), np.zeros((2, d))); return m
def get_w(m): return np.concatenate([a.ravel() for a in m.m.coefs_] + [b.ravel() for b in m.m.intercepts_])
def set_w(m, v):
    i = 0
    for a in m.m.coefs_: n = a.size; a[...] = v[i:i + n].reshape(a.shape); i += n
    for b in m.m.intercepts_: n = b.size; b[...] = v[i:i + n]; i += n
def train_epochs(m, A, epochs):
    for _ in range(epochs): m.m.partial_fit(A, A)

def aggregate(vs, how):
    V = np.stack(vs)
    if how == "mean": return V.mean(0)
    if how == "median": return np.median(V, 0)
    if how == "trimmed": S = np.sort(V, 0); return S[1:-1].mean(0)
    raise ValueError(how)

def federate(data, how, attacker=None, rounds=30, local_epochs=3, seed=0):
    g = get_w(new_model(seed)); models = [new_model(seed) for _ in data]
    for r in range(rounds):
        ups = []
        for i, (m, A) in enumerate(zip(models, data)):
            set_w(m, g.copy()); train_epochs(m, A, local_epochs); w = get_w(m)
            if attacker is not None and i == attacker: w = g - 5.0 * (w - g)   # sign-flipped, boosted update
            ups.append(w)
        g = aggregate(ups, how)
    return g

def central(data, seed=0, epochs=90):
    m = new_model(seed); train_epochs(m, np.vstack(data), epochs); return get_w(m)

def eval_arm(init, n_local, seed, ft_epochs=40, scratch_epochs=120):
    TP = FP = FN = TN = 0
    for f in HELD:
        X = f["X"]; Z = norm(X, n_local); A = windows(Z[:n_local], W).astype(np.float64)
        m = new_model(seed, lr=1e-3)
        if init is None: train_epochs(m, A, scratch_epochs)
        else: set_w(m, init.copy()); train_epochs(m, A, ft_epochs)
        E = m.errors(Z); s = np.zeros(len(X)); s[W - 1:] = E.mean(1)
        a = alarms(s, dict(q=cfgj["q"], scale=cfgj["scale"], smooth=cfgj["smooth"]))[N_TRAIN:]; y = f["y"][N_TRAIN:]
        TP += int(((a==1)&(y==1)).sum()); FP += int(((a==1)&(y==0)).sum()); FN += int(((a==0)&(y==1)).sum()); TN += int(((a==0)&(y==0)).sum())
    return round(2*TP/max(1, 2*TP+FP+FN), 4)

# ---- provenance: HMAC-tagged, hash-chained update commitments -----------------------------------------------
class Commit:
    def __init__(self, key): self.key, self.prev = key, b"\0" * 32
    def make(self, rnd, update_bytes, ledger_head):
        body = self.prev + hashlib.sha256(update_bytes).digest() + str(rnd).encode() + ledger_head
        tag = hmac.new(self.key, body, hashlib.sha256).digest(); self.prev = tag; return dict(round=rnd, tag=tag, ledger_head=ledger_head, prev=body[:32])
class Verifier:
    def __init__(self, keys): self.keys, self.last, self.rnd = keys, {k: b"\0" * 32 for k in keys}, {k: -1 for k in keys}
    def accept(self, site, update_bytes, c):
        body = self.last[site] + hashlib.sha256(update_bytes).digest() + str(c["round"]).encode() + c["ledger_head"]
        ok = hmac.compare_digest(hmac.new(self.keys[site], body, hashlib.sha256).digest(), c["tag"]) and c["round"] == self.rnd[site] + 1
        if ok: self.last[site] = c["tag"]; self.rnd[site] = c["round"]
        return ok
def provenance_test(trials=200):
    rng = np.random.default_rng(0); keys = {f"s{i}": os.urandom(32) for i in range(5)}
    res = dict(honest_accepted=0, honest_total=0, tamper_rejected=0, tamper_total=0, replay_rejected=0, replay_total=0, wrong_key_rejected=0, wrong_key_total=0)
    for _ in range(trials):
        v = Verifier(keys); cs = {k: Commit(keys[k]) for k in keys}; hist = {}
        for r in range(3):
            for k in keys:
                u = rng.normal(size=100).tobytes(); c = cs[k].make(r, u, os.urandom(32)); hist[(k, r)] = (u, c)
                res["honest_total"] += 1; res["honest_accepted"] += v.accept(k, u, c)
        v2 = Verifier(keys); cs2 = {k: Commit(keys[k]) for k in keys}
        u = rng.normal(size=100).tobytes(); c = cs2["s0"].make(0, u, os.urandom(32))
        res["tamper_total"] += 1; res["tamper_rejected"] += (not v2.accept("s0", (rng.normal(size=100)).tobytes(), c))   # update altered in transit
        v3 = Verifier(keys); c3 = Commit(keys["s1"]); u3 = rng.normal(size=100).tobytes(); cc = c3.make(0, u3, b"h" * 32); v3.accept("s1", u3, cc)
        res["replay_total"] += 1; res["replay_rejected"] += (not v3.accept("s1", u3, cc))                                 # same update replayed
        v4 = Verifier(keys); forged = Commit(os.urandom(32)).make(0, u3, b"h" * 32)
        res["wrong_key_total"] += 1; res["wrong_key_rejected"] += (not v4.accept("s2", u3, forged))                       # attacker without the site key
    return res

if __name__ == "__main__":
    t0 = time.time(); data = site_windows(); print("site window counts:", [len(d) for d in data])
    inits = {"scratch": None, "FedAvg (clean)": federate(data, "mean"), "central pooled": central(data),
             "FedAvg + attacker": federate(data, "mean", attacker=0), "median + attacker": federate(data, "median", attacker=0),
             "trimmed-mean + attacker": federate(data, "trimmed", attacker=0)}
    print("pretraining done %.0fs" % (time.time() - t0))
    out = {"n_local_rows": [30, 60, 120, 300], "F1": {}}
    for name, init in inits.items():
        out["F1"][name] = {}
        for n in out["n_local_rows"]:
            vals = [eval_arm(init, n, s) for s in (0, 1)]; out["F1"][name][n] = (round(float(np.mean(vals)), 4), round(float(np.std(vals)), 4))
        print(name.ljust(26), {k: v[0] for k, v in out["F1"][name].items()}, "%.0fs" % (time.time() - t0))
    out["provenance"] = provenance_test(); print(out["provenance"])
    json.dump(out, open(os.path.join(RES, "federated.json"), "w"), indent=1)
