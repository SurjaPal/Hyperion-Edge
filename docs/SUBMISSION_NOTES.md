# Submission notes -- mapped to the evaluation criteria

Detailed evidence lives in two documents; this page is the short map from each rubric heading
to where the proof is:

- `docs/RESULTS.md` -- every measured number (detection quality, quantization cost, federated
  cold-start and poisoning-robustness experiments, provenance tests, sensor-triage statistics),
  all run on real sensor recordings (SKAB), with the negative/null results kept in, not hidden.
- `docs/PROBLEM_AND_APPROACH.md` -- the failure-mode register, prior-art positioning, and the
  list of anticipated judge questions with honest answers.

## Technical Implementation
- A real pretrained open-source vision model (YOLOX-nano, Apache-2.0) executing through ONNX
  Runtime.
- A real telemetry autoencoder selected by a documented, reproducible protocol: 74 Optuna
  trials, multi-objective (F1 vs parameter count), tuned and selected on a held-out-blind
  15-file split of real pump-testbed recordings, with a fixed selection rule decided before the
  held-out files were touched (`docs/RESULTS.md` section 1).
- Both neural models are exported to ONNX and scored through the same QNN-first,
  CPU-fallback provider selection code (`docs/ARCHITECTURE.md`); the provider actually active
  is reported live, not hardcoded.
- A real federated round with HMAC-chained update commitments and coordinate-wise-median
  aggregation, measured under a poisoning attack on real (non-IID) site data
  (`docs/RESULTS.md` sections 3-4).
- A real hash-chained tamper-evident log with a working, demoable `verify()`.

## Application Use Case & Innovation
- Grounded in a documented, named operational gap (unmanned/lightly-staffed industrial sites
  with intermittent connectivity), not an abstract scenario.
- The claimed contribution is stated precisely, and no further than the evidence supports:
  federated anomaly detection and poisoning-robust aggregation are established research
  directions (`docs/PROBLEM_AND_APPROACH.md` section 4 names comparable published work); what
  this project adds is a working, reproducible, honestly-evaluated integration of those ideas
  with update provenance and a tamper-evident log for a specific offline industrial setting --
  presented as an integration, not a claim of inventing federated learning.

## Deployment & Accessibility
- Zero-connectivity operation by design for everything except the (optional, periodic)
  federated round.
- Fully offline dashboard: fonts bundled, no CDN dependency, no account, one-command launch.
- Small footprint: ~2.2k-parameter, ~10 KB telemetry model; ~3.6 MB open-source vision model.
- Known, stated gaps: not yet tested on Windows on Arm; ledger and baseline are in-memory only
  in this build (`docs/PROBLEM_AND_APPROACH.md` section 3, failure-mode register).

## Presentation & Documentation
- `docs/RESULTS.md` -- numbers, with confidence via multi-seed means/paired differences where
  relevant, and every negative result kept in.
- `docs/PROBLEM_AND_APPROACH.md` -- problem framing, architecture rationale, a structured
  failure-mode register (17 rows: handled / partial / open), and anticipated judge Q&A.
- `docs/ARCHITECTURE.md` -- the four-layer design and the reasoning behind each non-obvious
  choice (one neural model family instead of two, hash chain instead of blockchain, HMAC
  commitments alongside robust aggregation rather than instead of it).
- Live dashboard replays real sensor recordings and shows telemetry, alarms with the specific
  channels that triggered them, the sensor-fault-vs-process-fault triage hint, vision
  detections with measured latency, the tamper-evident chain, and federated round outcomes
  (including on-demand poisoning and transit-tamper demonstrations) all updating in real time.

## Honesty about what's simulated here

This dev sandbox has no Snapdragon hardware to compile against or test on, and no access to
aihub.qualcomm.com. The telemetry is a replay of real recordings (not a live sensor feed); the
camera frames are replayed sample images, not live CCTV. Every other claim above is backed by
code that runs and results that were measured, not implied. State this distinction plainly in
the presentation -- see `docs/RESULTS.md`'s final section for the explicit "not yet measured"
list, and `docs/PROBLEM_AND_APPROACH.md` section 7 for how to answer a judge who probes any of
this.
