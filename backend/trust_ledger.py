"""
trust_ledger.py
-----------------
Append-only, hash-chained local log. This is the piece that lets an
unmanned site be trusted even when nobody can watch it live and the
network link is down for hours or days at a stretch: every entry embeds
the SHA-256 hash of the entry before it, so re-ordering, deleting, or
editing any past record breaks the chain from that point forward and
verify() catches it the moment the site reconnects.

Genesis block, chained hashing and verify() are the same idea behind
the blockchain audit-ledger layer built for the Atlas Flux project,
re-applied here to equipment/anomaly events instead of grid transactions
-- and, unlike that project, deliberately NOT a real blockchain: no
consensus, no peers agreeing on state, just a local integrity guarantee
that costs almost nothing to run on-device. That trade-off is the point
for a single unmanned site.
"""

from __future__ import annotations

import hashlib
import json
import time


GENESIS_HASH = "0" * 64


class TamperEvidentLedger:
    def __init__(self, site_id: str):
        self.site_id = site_id
        self.entries: list[dict] = []
        self._last_hash = GENESIS_HASH

    def append(self, event: dict) -> dict:
        payload = {
            "site_id": self.site_id,
            "seq": len(self.entries),
            "ts": time.time(),
            "prev_hash": self._last_hash,
            "event": event,
        }
        payload["hash"] = self._hash_entry(payload)
        self.entries.append(payload)
        self._last_hash = payload["hash"]
        return payload

    @staticmethod
    def _hash_entry(payload_without_hash: dict) -> str:
        body = {k: v for k, v in payload_without_hash.items() if k != "hash"}
        blob = json.dumps(body, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()

    def verify(self) -> tuple[bool, int | None]:
        """Returns (is_intact, index_of_first_break_or_None)."""
        prev = GENESIS_HASH
        for i, entry in enumerate(self.entries):
            expected = self._hash_entry(entry)
            if entry["hash"] != expected or entry["prev_hash"] != prev:
                return False, i
            prev = entry["hash"]
        return True, None

    def head(self) -> str:
        return self._last_hash

    def tail(self, n: int = 8) -> list[dict]:
        return self.entries[-n:]

    def tamper_for_demo(self, index: int):
        """
        Deliberately corrupts one past entry -- exists purely so a demo
        can show the ledger catching real tampering on cue. Not part of
        the production code path.
        """
        if 0 <= index < len(self.entries):
            self.entries[index]["event"]["_tampered"] = True
