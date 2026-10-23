"""
server.py -- Hyperion-Edge demo server (standard library only for the web layer).
Replays REAL pump-testbed data (SKAB) at 16 rows/s through 3 simulated sites, runs the tuned autoencoder, and serves:
  GET  /api/state                    live JSON
  POST /api/tamper/<site>            corrupt one past ledger entry (tamper-detection demo)
  POST /api/fed/run                  run a federated round now
  POST /api/fed/poison/<site>        round in which <site> submits a poisoned (signed) update
  POST /api/fed/tamper/<site>        round in which <site>'s update is altered in transit
"""
from __future__ import annotations
import json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

from real_node import ROOT, RealSiteNode
from real_swarm import RealSwarm

# Fixed, non-cherry-picked rule: the first held-out file of each group (files never used for tuning or pretraining).
SITES = [("RO-KOL-014", "valve1/2.csv"), ("RO-DGP-007", "valve2/2.csv"), ("RO-SGR-022", "other/3.csv")]
ROWS_PER_TICK, TICK_SECONDS = 4, 0.25
DASHBOARD_DIR = os.path.join(ROOT, "dashboard")
PORT = int(os.environ.get("HYPERION_PORT", os.environ.get("SENTINEL_PORT", "8787")))
HOST = os.environ.get("HYPERION_HOST", os.environ.get("SENTINEL_HOST", "127.0.0.1"))   # localhost only by default; demo control endpoints are unauthenticated

_lock = threading.RLock()
_state = {"sites": {}, "swarm": {}}
_nodes, _swarm = [], None


def _loop():
    global _swarm
    data_dir = os.path.join(ROOT, "data", "skab")
    nodes = [RealSiteNode(sid, os.path.join(data_dir, f), None, seed=i) for i, (sid, f) in enumerate(SITES)]
    swarm = RealSwarm(nodes)
    with _lock:
        _nodes.extend(nodes); _swarm = swarm
    while True:
        with _lock:
            snap = {n.site_id: n.tick(ROWS_PER_TICK) for n in nodes}
            swarm.maybe_auto_round()
            _state["sites"] = snap; _state["swarm"] = swarm.snapshot()
        time.sleep(TICK_SECONDS)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/state":
            with _lock: return self._json(dict(_state))
        route = "/index.html" if path == "/" else path
        ctype = {".html": "text/html", ".js": "application/javascript", ".css": "text/css", ".woff2": "font/woff2", ".txt": "text/plain"}.get(os.path.splitext(route)[1], "application/octet-stream")
        full = os.path.normpath(os.path.join(DASHBOARD_DIR, route.lstrip("/")))
        if not full.startswith(DASHBOARD_DIR) or not os.path.isfile(full):
            self.send_response(404); self.end_headers(); return
        body = open(full, "rb").read()
        self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)

    def do_POST(self):
        parts = urlparse(self.path).path.strip("/").split("/")
        with _lock:
            if parts[:2] == ["api", "tamper"] and len(parts) == 3:
                n = next((x for x in _nodes if x.site_id == parts[2]), None)
                if n and len(n.ledger.entries) > 1:
                    n.ledger.tamper_for_demo(0); return self._json({"ok": True})
            if parts[:2] == ["api", "fed"] and _swarm is not None:
                if parts[2:3] == ["run"]: _swarm.run_round(); return self._json({"ok": True})
                if parts[2:3] == ["poison"] and len(parts) == 4: _swarm.run_round(poison=parts[3]); return self._json({"ok": True})
                if parts[2:3] == ["tamper"] and len(parts) == 4: _swarm.run_round(tamper=parts[3]); return self._json({"ok": True})
        self._json({"ok": False}, 404)


def main():
    threading.Thread(target=_loop, daemon=True).start()
    print(f"Hyperion-Edge demo server running: http://localhost:{PORT}")
    ThreadingHTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
