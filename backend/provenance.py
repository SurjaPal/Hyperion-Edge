"""
HMAC-tagged, hash-chained commitments for federated updates.
What this gives: an update altered in transit, replayed, or forged without the site key is rejected.
What it does NOT give: protection against a *compromised site* signing a poisoned update. That is the job of the
robust aggregator (coordinate-wise median), tested separately. Production would use asymmetric signatures and
hardware-backed keys; symmetric per-site keys keep this demo dependency-free.
"""
import hashlib, hmac

ZERO = b"\0" * 32


class SiteCommitter:
    def __init__(self, key: bytes): self.key, self.prev = key, ZERO
    def make(self, rnd: int, update_bytes: bytes, ledger_head: str) -> dict:
        lh = ledger_head.encode()
        body = self.prev + hashlib.sha256(update_bytes).digest() + str(rnd).encode() + lh
        return {"round": rnd, "ledger_head": lh, "tag": hmac.new(self.key, body, hashlib.sha256).digest()}
    def ack(self, commit: dict): self.prev = commit["tag"]     # advance chain only once the aggregator accepted


class Verifier:
    def __init__(self, keys: dict):
        self.keys, self.last, self.rnd = keys, {k: ZERO for k in keys}, {k: -1 for k in keys}
    def check(self, site: str, update_bytes: bytes, c: dict) -> tuple[bool, str]:
        if c["round"] <= self.rnd[site]: return False, "replayed or out-of-order round"
        body = self.last[site] + hashlib.sha256(update_bytes).digest() + str(c["round"]).encode() + c["ledger_head"]
        if not hmac.compare_digest(hmac.new(self.keys[site], body, hashlib.sha256).digest(), c["tag"]):
            return False, "commitment mismatch (update altered, forged, or chain broken)"
        self.last[site], self.rnd[site] = c["tag"], c["round"]
        return True, "ok"
