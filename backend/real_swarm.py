"""
Robust federated rounds over the sites' locally trained autoencoder weights, with provenance verification.
Every site starts from the same shared model (so weight averaging is meaningful), fine-tunes locally, then submits its
weights with an HMAC-chained commitment. The aggregator verifies commitments, then aggregates with the coordinate-wise
median (tolerates a minority of poisoned updates). The result becomes the shared model the sites start from next cycle.
Round accuracy claims come from the offline experiments in ml/ (results/federated.json), not from this live demo.
"""
from __future__ import annotations
import hashlib, json, os, time
import numpy as np
from provenance import SiteCommitter, Verifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class RealSwarm:
    def __init__(self, sites):
        self.sites = sites; self.round_number = 0; self.model_version = "v0 (offline-pretrained)"; self.sync_log = []
        z = np.load(os.path.join(ROOT, "models", "fed_init.npz")); self.global_w = z["w"].copy()
        keys = {s.site_id: os.urandom(32) for s in sites}; self.committers = {s.site_id: SiteCommitter(keys[s.site_id]) for s in sites}
        self.verifier = Verifier(keys); self._auto_done_cycle = -1

    def shared_init(self): return self.global_w

    def maybe_auto_round(self):
        ready = [s for s in self.sites if s.pending_weights is not None]
        cyc = max(s.cycle for s in self.sites)
        if len(ready) == len(self.sites) and self._auto_done_cycle < cyc:
            self._auto_done_cycle = cyc; self.run_round()

    def _log(self, msg, kind="sync"): self.sync_log.append({"ts": time.time(), "round": self.round_number, "note": msg, "kind": kind}); self.sync_log = self.sync_log[-30:]

    def run_round(self, poison: str | None = None, tamper: str | None = None):
        ups = {s.site_id: s.latest_weights for s in self.sites if s.latest_weights is not None}
        if len(ups) < 3: self._log("Round skipped: fewer than 3 sites have a trained local model."); return
        self.round_number += 1; r = self.round_number
        honest = {k: v.copy() for k, v in ups.items()}; ref = np.median(np.stack(list(honest.values())), 0)
        sent = {k: v.copy() for k, v in honest.items()}
        if poison in sent: sent[poison] = ref - 5.0 * (sent[poison] - ref)            # attacker signs a poisoned update
        accepted, notes = {}, []
        for s in self.sites:
            k = s.site_id
            if k not in sent: continue
            b = sent[k].astype(np.float32).tobytes(); c = self.committers[k].make(r, b, s.ledger.head())
            wire = bytearray(b)
            if tamper == k: wire[0] ^= 0xFF                                            # altered in transit after signing
            ok, why = self.verifier.check(k, bytes(wire), c)
            if ok: self.committers[k].ack(c); accepted[k] = sent[k]
            else: notes.append(f"{k} REJECTED ({why})")
        if len(accepted) < 2: self._log(f"Round {r}: too few verified updates; no aggregation. " + "; ".join(notes)); return
        A = np.stack(list(accepted.values())); mean_agg, med_agg = A.mean(0), np.median(A, 0)
        clean = np.median(np.stack([honest[k] for k in accepted if k != poison] or [ref]), 0); nrm = np.linalg.norm(clean) + 1e-12
        d_mean, d_med = float(np.linalg.norm(mean_agg - clean) / nrm), float(np.linalg.norm(med_agg - clean) / nrm)
        self.global_w = med_agg; self.model_version = f"v{r}"
        digest = hashlib.sha256(med_agg.astype(np.float32).tobytes()).hexdigest()[:12]
        msg = f"Round {r}: {len(accepted)}/{len(ups)} updates verified, aggregated by median (model {digest})."
        if notes: msg += " " + "; ".join(notes) + "."
        if poison: msg += f" Poisoned update from {poison} shifted a plain mean by {d_mean:.0%} of the model norm; the median shifted it by {d_med:.0%}."
        self._log(msg, kind="alert" if (poison or notes) else "sync")
        for s in self.sites: s.ledger.append({"type": "fed_round", "round": r, "model_digest": digest, "verified": s.site_id in accepted})
        for s in self.sites: s.pending_weights = None

    def snapshot(self): return {"round_number": self.round_number, "model_version": self.model_version, "sync_log": self.sync_log[-8:]}
