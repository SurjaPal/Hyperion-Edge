# Sentinel-Edge

Offline-first, federated anomaly detection for unmanned industrial sites (retail fuel outlets,
warehouses, substations), built for Snapdragon-powered HP PCs and the Qualcomm Snapdragon® AI
Lab Build & Present Challenge.

One Sentinel-Edge node is designed to run entirely on-device at a site: it scores equipment
telemetry through a small neural autoencoder, runs an open-source vision model (YOLOX-nano) on
camera frames, explains flagged anomalies in terms of which measurements moved, and writes
everything to a tamper-evident local log, with no connectivity required. When a link is
available, sites exchange only small, HMAC-committed model-weight deltas (never raw telemetry)
to improve a shared model, aggregated robustly so one compromised site can't poison the fleet.

**Read `docs/RESULTS.md` before presenting.** It has every measured number -- including the
ones that didn't work as hoped -- from real experiments on a real sensor dataset, not
illustrative figures.

## Quick start

```bash
pip install -r requirements.txt
python scripts/run_demo.py     # also fetches the SKAB dataset + sample frames on first run (needs internet once)
```

Opens a live dashboard at `http://localhost:8787` replaying real pump-testbed recordings
(`valve1/2.csv`, `valve2/2.csv`, `other/3.csv` -- files never used to tune or pretrain the
model) through 3 simulated sites. Each site spends the first ~75 ticks learning its own
baseline, then calibrates an alarm threshold, then scores live, flagging real labelled
anomalies from the recording. Use the **federated learning controls** to run a round on demand,
or a round where one site's update is poisoned or altered in transit, and watch the log react.
Click **"Simulate tamper on this ledger"** on any site to see the tamper detector catch it.

## What's real vs. simulated (read before presenting)

| Component | Status in this build |
|---|---|
| Telemetry | **REAL**: replayed recordings from a physical pump testbed (SKAB). Not a live feed. |
| Telemetry model | **REAL**: tuned windowed autoencoder (~2,210 params), selected by 74 Optuna trials under a held-out-blind protocol. Measured F1/FAR/MAR in `docs/RESULTS.md`. |
| Vision model | **REAL**: YOLOX-nano (Apache-2.0) via ONNX Runtime. |
| Camera frames | Replayed sample images (`data/frames/`) -- not live CCTV. |
| NPU execution | **NOT YET MEASURED** -- requires the Snapdragon device. Provider shown is read from the live ONNX Runtime session, never hardcoded (`docs/NPU_INTEGRATION.md`). |
| INT8 quantization | **REAL** measurement of accuracy cost on real held-out data (`docs/RESULTS.md` section 2). CPU-only latency measured; NPU latency not measured. |
| Federated learning | **REAL** FedAvg/median math, measured cold-start and poisoning-robustness on real, non-IID site data. 3 sites simulated in one process for the live demo. |
| Update provenance | **REAL** HMAC-chained commitments, measured against tampering/replay/forgery. Does not defend against a site that legitimately holds its own key -- that's the aggregation robustness result, a separate mechanism. |
| Sensor-fault triage | **REAL** statistic (error concentration), measured against real process anomalies and injected sensor faults. About a third of injected faults are missed; advisory only. |
| Tamper-evident ledger | **REAL** hash chain with working verification. In-memory only in this build. |
| Incident notes | **TEMPLATE** text built from the measured deviation, not a language model. |

## Repository layout

```
backend/
  ae_core.py           windowed autoencoder + ONNX export + QNN-first/CPU-fallback scoring
  vision_module.py      YOLOX-nano via ONNX Runtime, same provider-selection pattern
  trust_ledger.py        hash-chained tamper-evident local log
  provenance.py           HMAC-chained update commitments (integrity/replay/forgery)
  real_node.py             per-site orchestrator, replays real telemetry through the pipeline
  real_swarm.py             federated rounds: verify commitments -> aggregate by median
  server.py                  demo server: drives 3 real_node instances + serves the dashboard
dashboard/                  live ops-console UI (vanilla HTML/CSS/JS, fonts bundled, no CDN)
ml/                          the experiments behind every number in docs/RESULTS.md
  lib.py                      shared data loading, models, evaluation protocol
  tune.py                      Optuna multi-objective hyperparameter search
  final_eval.py                  held-out-blind evaluation, fixed selection rule
  quantize.py                     INT8 QDQ quantization accuracy study
  federated_experiment.py          cold-start + poisoning-robustness experiments
  sensor_gate.py                    sensor-vs-process fault triage statistic
results/                    raw output of every ml/ script (JSON + the Optuna trial database)
models/                     yolox_nano.onnx (bundled) + fed_init.npz (our own derived output)
docs/
  RESULTS.md                 every measured number, including negative results
  PROBLEM_AND_APPROACH.md      failure-mode register, prior art, anticipated Q&A
  ARCHITECTURE.md                design rationale
  NPU_INTEGRATION.md               exact steps + gaps for the real Snapdragon device
  COMPLIANCE.md                    audit against the challenge's stated guidelines
  SUBMISSION_NOTES.md                short map from rubric headings to evidence
scripts/
  fetch_assets.py             pulls the SKAB dataset + vision model + sample frames (not bundled)
  run_demo.py                  one-command launcher
THIRD_PARTY_NOTICES.md      licenses for everything not written for this project
```

## Reproducing the experiments

```bash
pip install -r ml/requirements.txt
python scripts/fetch_assets.py
python ml/tune.py 600          # resumable; re-run with more seconds for more trials
python ml/final_eval.py
python ml/quantize.py
python ml/federated_experiment.py
python ml/sensor_gate.py
```
