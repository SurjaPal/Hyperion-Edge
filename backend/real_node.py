"""
real_node.py -- one Hyperion-Edge site replaying REAL pump-testbed telemetry (SKAB) at accelerated speed.

Protocol (identical to the offline evaluation in ml/): rows [0,300) baseline learning, rows [300,400) alarm-threshold
calibration on unseen normal data, rows >= 400 live scoring. The dataset's anomaly labels are used ONLY for the
"dataset label" chip in the UI and never by the model.
"""
from __future__ import annotations
import csv, json, os

import numpy as np

from ae_core import OnnxScorer, SklearnScorer, WindowAE, windows
from vision_module import get_frames as _get_frames, get_vision as _get_vision
from trust_ledger import TamperEvidentLedger

COLS = ["Accelerometer1RMS", "Accelerometer2RMS", "Current", "Pressure", "Temperature", "Thermocouple", "Voltage", "Volume Flow RateRMS"]
N_FIT, N_TRAIN = 300, 400
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(ROOT, "results", "model_config.json")))
TAU = CFG["sensor_triage_tau"]


def load_csv(path):
    with open(path, newline="") as f:
        rd = csv.DictReader(f, delimiter=";")
        rows = list(rd)
    X = np.array([[float(r[c]) for c in COLS] for r in rows]); y = np.array([int(float(r["anomaly"])) for r in rows])
    return X, y


class RealSiteNode:
    def __init__(self, site_id: str, csv_path: str, init_weights_getter=None, seed: int = 0):
        self.site_id = site_id; self.csv_name = os.path.relpath(csv_path, os.path.join(ROOT, "data", "skab")).replace(os.sep, "/")
        self.X, self.y = load_csv(csv_path)
        self._init_getter, self.seed = init_weights_getter, seed
        self.ledger = TamperEvidentLedger(site_id)
        self.cycle = 0; self.tick_count = 0; self.total_alarms = 0
        self.pending_weights = None      # set after local training; consumed by the automatic swarm round
        self.latest_weights = None       # most recent locally trained weights (used by on-demand rounds)
        self.ledger.append({"type": "boot", "detail": f"Hyperion-Edge node online for {site_id}", "data": self.csv_name})
        self._last_vision = None; self._last_vision_counts = None
        self._reset()

    # ---------------------------------------------------------------- state
    def _reset(self):
        self.row = 0; self.phase = "learning"; self.ae = None; self.scorer = None
        self.mu = self.sd = None; self.cal = []; self.thr = None
        self.s_hist, self.e_hist = [], []; self.in_alarm = False; self.alarm_start = None; self.alarm_peak = 0.0
        self.severity = "NOMINAL"; self.note = None; self.ratio = 0.0; self.triage = None; self.share = None; self.top_channels = []
        self.latency_us = None

    def _train(self):
        Xn = self.X[:N_FIT]; self.mu, self.sd = Xn.mean(0), Xn.std(0) + 1e-6
        Z = (Xn - self.mu) / self.sd; A = windows(Z, CFG["window"]).astype(np.float64)
        # Best-validated configuration (tuned on the TUNE split, scratch training). All sites share the same random
        # initialisation (seed 0) so that their weights are averageable in a federated round.
        self.ae = WindowAE(CFG["window"], CFG["hidden"], CFG["bottleneck"], CFG["alpha"], CFG["activation"], lr=CFG["lr"], batch=CFG["batch"], seed=0)
        self.ae.train(A, CFG["epochs"]); self.init_kind = "local training from the shared initialisation"
        try:
            self.scorer = OnnxScorer(self.ae.export_onnx(), CFG["window"])
        except Exception:
            self.scorer = SklearnScorer(self.ae)
        self.pending_weights = self.ae.get_w().copy(); self.latest_weights = self.pending_weights.copy()
        self.ledger.append({"type": "baseline_learned", "detail": f"AE trained on {N_FIT} local rows ({self.init_kind})", "provider": self.scorer.provider})

    # ----------------------------------------------------------------- step
    def _step(self):
        t = self.row
        if t == N_FIT: self._train()
        if t >= N_FIT:
            import time
            W = CFG["window"]; Z = (self.X[max(0, t - W + 1):t + 1] - self.mu) / self.sd
            if len(Z) == W:
                t0 = time.perf_counter(); e = self.scorer.channel_errors(Z.ravel()); self.latency_us = round((time.perf_counter() - t0) * 1e6, 0)
            else:
                e = np.zeros(8)
            s = float(e.mean()); self.s_hist.append(s); self.e_hist.append(e)
            if t < N_TRAIN:
                self.phase = "calibrating"; self.cal.append(s)
                if t == N_TRAIN - 1:
                    self.thr = float(np.quantile(self.cal, CFG["q"]) * CFG["scale"]); self.phase = "live"
                    self.ledger.append({"type": "threshold_set", "detail": "alarm threshold calibrated on unseen normal rows"})
            else:
                k = CFG["smooth"]; sm = float(np.median(self.s_hist[-k:])); self.ratio = sm / self.thr
                self._update_alarm(sm)
        self.row += 1

    def _update_alarm(self, sm):
        alarm = self.ratio > 1.0
        if alarm:
            k = CFG["smooth"]; m = np.mean(self.e_hist[-k:], axis=0); tot = float(m.sum()) + 1e-12
            self.share = float(m.max() / tot); order = np.argsort(m)[::-1]; self.top_channels = [COLS[i] for i in order[:2]]
            self.triage = "sensor_fault_suspected" if self.share >= TAU else "process_anomaly"
            self.severity = "HIGH" if self.ratio > 4 else "ELEVATED" if self.ratio > 2 else "WATCH"
            self.alarm_peak = max(self.alarm_peak, self.ratio)
            what = ("A single channel dominates the error, consistent with a failing sensor rather than a process change"
                    if self.triage == "sensor_fault_suspected" else
                    "The error is spread across several channels, consistent with a real process disturbance")
            self.note = (f"{self.severity}: {self.top_channels[0]} and {self.top_channels[1]} deviate most from this site's learned baseline "
                         f"({self.ratio:.1f}x the alarm threshold). {what} (top-channel share {self.share:.2f}). Advisory: inspect the flagged subsystems.")
            if not self.in_alarm:
                self.in_alarm = True; self.alarm_start = self.row; self.total_alarms += 1
                self.ledger.append({"type": self.triage, "row": self.row, "top_channels": self.top_channels, "share": round(self.share, 3), "severity": self.severity})
        else:
            if self.in_alarm:
                self.ledger.append({"type": "alarm_cleared", "row": self.row, "duration_rows": self.row - self.alarm_start, "peak_ratio": round(self.alarm_peak, 2)})
            self.in_alarm = False; self.alarm_peak = 0.0; self.severity = "NOMINAL"; self.triage = None

    # ----------------------------------------------------------------- public
    def tick(self, rows: int = 4):
        for _ in range(rows):
            if self.row >= len(self.X):
                self.cycle += 1
                self.ledger.append({"type": "replay_restart", "detail": f"end of recording; restarting cycle {self.cycle} from the latest shared model"})
                self._reset()
            self._step()
        self.tick_count += 1
        self._maybe_vision()
        return self._payload()

    def _maybe_vision(self):
        vision, frames = _get_vision(), _get_frames()
        if not vision.available or not frames or self.tick_count % 4 != 0: return
        det = vision.detect(frames[(self.tick_count // 20 + self.seed) % len(frames)]); self._last_vision = det
        counts = (det["persons"], det["vehicles"])
        if counts != self._last_vision_counts:
            self._last_vision_counts = counts
            self.ledger.append({"type": "vision", "persons": det["persons"], "vehicles": det["vehicles"], "vehicle_types": det["vehicle_types"], "provider": vision.provider})

    def _payload(self):
        ok, brk = self.ledger.verify(); t = min(self.row, len(self.X) - 1)
        v = _get_vision()
        return {
            "site_id": self.site_id, "mode": "real", "warmup": self.phase != "live", "phase": self.phase,
            "data_file": self.csv_name, "tick": self.row, "rows_total": len(self.X), "cycle": self.cycle,
            "telemetry": {c: float(self.X[t, i]) for i, c in enumerate(COLS)},
            "score": round(min(1.0, self.ratio / 4.0), 3), "ratio": round(self.ratio, 2), "severity": self.severity, "note": self.note if self.in_alarm else None,
            "triage": self.triage, "share": None if self.share is None else round(self.share, 2),
            "dataset_label_anomaly": bool(self.y[t]),
            "backend": getattr(self.scorer, "provider", "not trained yet"), "telemetry_latency_us": self.latency_us,
            "model_params": self.ae.n_params() if self.ae else None,
            "vision": {"available": v.available, "provider": v.provider,
                       "latency_ms": self._last_vision["latency_ms"] if self._last_vision else None,
                       "persons": self._last_vision["persons"] if self._last_vision else None,
                       "vehicles": self._last_vision["vehicles"] if self._last_vision else None,
                       "vehicle_types": self._last_vision["vehicle_types"] if self._last_vision else []},
            "total_anomalies": self.total_alarms, "ledger_head": self.ledger.head()[:16], "ledger_len": len(self.ledger.entries),
            "ledger_intact": ok, "ledger_break_index": brk,
            "ledger_tail": [{"seq": e["seq"], "hash": e["hash"][:12], "type": e["event"].get("type")} for e in self.ledger.tail(5)],
        }
