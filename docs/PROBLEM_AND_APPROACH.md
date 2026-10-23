# Problem and approach: the full picture

## 1. The problem

Industrial sites that are unmanned or lightly staffed (fuel retail outlets, pump stations, warehouses, substations) need continuous
equipment monitoring, and the usual answers each fail in a specific way:

| Common approach | Where it breaks |
|---|---|
| Fixed alarm thresholds | Every machine is different; thresholds are either noisy or blind, and drift over seasons |
| Cloud analytics | Needs a reliable uplink; raw telemetry and video leave the site (privacy, bandwidth, cost) |
| One fleet-wide model | Sites differ (equipment, load, climate), so a single model fits none of them well |
| Trusting the log | Nobody is on site; a rewritten or deleted log is indistinguishable from a real one |
| Alerting on every deviation | A failing *sensor* looks like a failing *machine*, so technicians are sent out for nothing |

**Goal:** a device that runs entirely at the site, learns that site's own normal behaviour in minutes without labels, raises alarms it can
explain in terms of which measurements moved, can tell the fleet what it learned without sharing raw data, and leaves a record whose
tampering is detectable. It is advisory only: it never actuates equipment.

## 2. What was built, and where each piece runs

1. **Per-site learning (CPU or NPU).** A small windowed autoencoder (2,210 parameters, ~10 KB) trained on the site's first 300 samples;
   alarm threshold calibrated on the next 100 unseen normal samples. Exported to ONNX and executed through ONNX Runtime with automatic
   QNN (Hexagon NPU) provider selection when available. INT8 QDQ export costs ~0.006 F1 (measured on CPU).
2. **Explanation.** Per-channel reconstruction error names the channels that moved; the share carried by the single worst channel gives an
   advisory sensor-fault-vs-process-fault hint. Wording is template text; no language model is used.
3. **Camera model.** YOLOX-nano (open-source) counts people and vehicles from frames; this is the neural workload that most benefits from an NPU.
4. **Tamper-evident log.** SHA-256 hash chain per site with verification.
5. **Federated round.** Sites submit weights with HMAC-chained commitments; aggregation by coordinate-wise median.

## 3. Failure-mode register (what was considered, and its status)

| Failure or attack | Handling | Status |
|---|---|---|
| Uplink down for days | Everything except the federated round works offline | Done |
| No labelled faults exist | Unsupervised; trained on the site's own normal data | Done |
| New device, almost no data | Federated init helps at ~30 s of data only | Measured, modest |
| Sensor stuck, dropped out, spiking | Error-concentration hint; ~1/3 of injected faults never alarm | Partial, advisory |
| Normal behaviour drifts | Not handled. Main source of false alarms (21.5% pre-onset, 62.9% post-anomaly) | **Open** |
| Poisoned model update from a compromised site | Median / trimmed-mean aggregation; plain averaging collapses | Done for sign-flip attacks; adaptive attackers untested |
| Update altered in transit / replayed / forged | HMAC-chained commitments | Done (protocol tests); not a formal proof |
| Log edited or deleted after the fact | Hash chain detects edits and deletion inside the chain | Done. An attacker with full disk access could rewrite the *whole* chain; mitigation is to anchor chain heads off-device (heads already travel in commitments, but the aggregator does not yet store or compare them) |
| Model weights leak information about local data | Not addressed (no differential privacy) | Open |
| Site keys stolen | Symmetric per-site keys in the demo; production needs asymmetric signatures and hardware-backed key storage | Open |
| Dashboard exposed on a network | Binds to 127.0.0.1 by default; demo control endpoints are unauthenticated and must not be exposed | Mitigated for the demo |
| Wrong action taken on an alarm | Advisory only, never controls equipment | By design |
| Model too heavy for the device | ~10 KB telemetry model, ~3.6 MB camera model | Done |
| NPU not actually used | Dashboard prints the provider read from the live session | Done; NPU run itself unverified |
| Ledger and baseline lost on reboot | In memory only in this build | Open |

## 4. Prior art, stated plainly

Federated anomaly detection with autoencoders for IoT, poisoning-robust federated learning for industrial IoT, and blockchain-backed
auditability of federated learning are all published (for example FedAA, DP2Guard, and a 2026 privacy-preserving, poisoning-robust
federated framework for industrial IoT). **Do not present any single component as a world first.**

What this project contributes that is defensible:
- A working, reproducible **end-to-end edge system** on real pump data, with an evaluation protocol that keeps labels out of training
  and thresholds and keeps held-out files untouched until the end.
- **Honest negative and null results**, which most demos omit: tuning did not beat defaults on held-out data; federation helps only at
  ~30 s of data; the sensor-fault hint is unsafe as an automatic suppressor; false alarms are high because of baseline drift.
- A **deployment-oriented path to Snapdragon**: ONNX export, QNN-oriented INT8 quantization with measured accuracy cost, provider
  selection that reports what actually ran.
- Empirical comparison of aggregation rules under attack on non-IID real-data sites.

## 5. Why this can help the internship application, and what would hurt it

Helps: reproducible numbers, a live demo on real data, clear separation of measured vs simulated, tuned models with a stated selection
rule, and a candid limitations list. Hurts: any claim of novelty that a reviewer can refute in one search, any implied NPU result that
was never run, and any headline number that ignores the false-alarm rate.

## 6. Priority list for the remaining time

1. If a Snapdragon device is reachable: run the demo, confirm the header shows `QNNExecutionProvider`, watch the NPU tab in Task Manager,
   record CPU-vs-NPU latency of the camera model. One measured screenshot is worth more than any paragraph.
2. Record a 3-5 minute demo (learning, alarm with channel explanation, poisoned round, tampered update, ledger tamper).
3. Write the proposal with the four rubric headings; copy numbers only from `docs/RESULTS.md`.
4. Only then, extensions in this order: drift-aware recalibration (biggest weakness), persist the ledger, anchor ledger heads at the
   aggregator, on-device language model via Qualcomm's documented SDK.

## 7. Questions a judge may ask

- *Is this new?* The components are known; the contribution is the integrated, honestly evaluated edge system (section 4).
- *Does it run on the NPU?* The camera and telemetry models go through ONNX Runtime with QNN selection; NPU execution is not yet measured.
- *Why is the false-alarm rate high?* Baseline drift; 21.5% on normal rows before onset, 62.9% after an anomaly. It is the first thing to fix.
- *Did tuning help?* It produced a model less than half the size at equal accuracy; it did not raise accuracy.
- *Why federate at all?* Privacy and offline operation, plus a measured cold-start benefit at ~30 s of data; not an accuracy gain.
- *Can a bad site poison the fleet?* Plain averaging: yes (F1 0.77 to 0.55). Median aggregation: no, for sign-flip attacks (0.755).
- *Is the data real?* Yes, recordings from a physical pump testbed (SKAB). The camera frames are replayed sample images.
