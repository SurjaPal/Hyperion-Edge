# Measured results

Everything here comes from scripts in `ml/` and raw output in `results/*.json`. Data: **SKAB** (Skoltech Anomaly Benchmark),
recordings from a real centrifugal-pump water-circulation testbed, 8 sensors at 1 Hz, GPL-3.0, downloaded by
`scripts/fetch_assets.py` (not redistributed here). Please cite: Katser & Kozitsin, *Skoltech Anomaly Benchmark (SKAB)*, Kaggle, 2020.

## Protocol (no label ever touches training or thresholds)

- 31 files (one, `other/2.csv`, excluded: its anomaly starts at row 104, inside the "normal" warm-up window).
- Per device/run: rows 0-299 train the model, rows 300-399 calibrate the alarm threshold on unseen normal data, rows >= 400 are the test stream.
- Files are split deterministically (alternating within each group) into **17 tune files** and **14 held-out files**. Hyperparameters,
  thresholds, the triage cut-off and the federated pre-training used tune files only. Held-out files were evaluated once, after the
  selection rule (fewest parameters within 0.01 F1 of the best) was fixed.
- Metric: pooled point-wise F1 over test rows; FAR/MAR = false / missed alarm rates. Neural results are mean over 5 seeds.
- This is a **different split and protocol from the public SKAB leaderboard**, so numbers are not directly comparable to it.

## 1. Detection quality (held-out, 14 files)

| Model (same threshold-tuning budget for all) | F1 | FAR % | MAR % |
|---|---|---|---|
| Windowed autoencoder, tuned (2,210 parameters) | 0.756 +/- 0.010 | 41.4 | 17.2 |
| Windowed autoencoder, default settings (~4.6k parameters) | 0.764 +/- 0.011 | 38.6 | 17.3 |
| PCA reconstruction (k and thresholds tuned) | 0.692 | 34.6 | 31.1 |
| Isolation Forest (thresholds tuned) | 0.580 | 16.5 | 53.2 |

Reading this honestly:
- Autoencoders beat PCA and Isolation Forest on F1 here. Isolation Forest is the model used in the first version of this project; it was the weakest.
- **Hyperparameter tuning did not improve held-out accuracy** (0.756 vs 0.764, within noise). Tune-split F1 was 0.806, so tuning over-fit the
  17 tune files. What tuning did deliver: a model less than half the default size at statistically equal accuracy (74 Optuna trials,
  multi-objective: maximise F1, minimise parameters).
- **False alarms are high (41%).** Split by where they occur: 21.5% on truly normal rows before the anomaly starts (driven by a few files
  where the machine's normal behaviour drifts away from the first 300 rows, e.g. `valve1/2`), and 62.9% after an anomaly ends (the
  machine has not returned to its earlier baseline). Baseline drift is the main open weakness.
- Median detection delay after onset: 8 s; no file was missed entirely.
- Per-file F1 spread is wide: 0.18 to 0.98 across held-out files.

## 2. INT8 quantization (the format Snapdragon's NPU expects)

Pipeline per file: train -> export ONNX -> quantize with the site's own normal windows (ONNX Runtime QNN-oriented QDQ path) -> score.

| Variant | F1 | FAR % | MAR % |
|---|---|---|---|
| float32 (ONNX Runtime, matches sklearn to 1e-6) | 0.7485 | 42.6 | 17.9 |
| QDQ, uint8 activations | 0.7423 | 45.5 | 17.4 |
| QDQ, uint16 activations | 0.7426 | 45.5 | 17.4 |

Quantization costs about 0.006 F1. Model size is about 10 KB. **CPU latency is 7 us float32 vs 15-19 us QDQ: quantized is slower on CPU,
and this says nothing about NPU speed.** The telemetry model is so small that NPU offload would bring little; the neural workload that
genuinely benefits from an NPU is the camera model. Real NPU numbers require the device.

## 3. Federated learning (5 non-IID simulated sites, tune-split data only)

Cold start (how much local normal data a new device needs), paired over 6 seeds:

| Local rows available | From scratch | From federated init | Paired difference |
|---|---|---|---|
| 30 | 0.565 | 0.626 | **+0.061 +/- 0.016 (SE)** |
| 60 | 0.662 | 0.665 | +0.003 +/- 0.006 |
| 120 | 0.695 | 0.699 | +0.004 +/- 0.016 |
| 300 (median-aggregated init) | 0.751 | 0.741 | -0.011 +/- 0.004 |

**Federation helps only in extreme cold start (about 30 s of data), gives no measurable benefit from 60 s on, and is slightly worse at 300 rows.**
It is not an accuracy feature; it shortens the time until a new device can start alarming.

Poisoning (one of five sites sends a sign-flipped, 5x-boosted update), F1 at 300 local rows:

| Aggregation | F1 |
|---|---|
| plain FedAvg, no attacker | 0.774 |
| plain FedAvg, attacker | **0.546** |
| coordinate-wise median, attacker | 0.755 |
| trimmed mean, attacker | 0.732 |

Plain averaging collapses under one attacker; median and trimmed mean hold. This is a simple attack model (sign flip); adaptive
attackers are out of scope. In the live demo (3 sites) the poisoned update shifts a plain mean by 58% of the model norm and the median by 12%.

## 4. Update provenance (HMAC-chained commitments)

Protocol test, 200 trials each: 3,000/3,000 honest updates accepted; 200/200 in-transit alterations rejected; 200/200 replays rejected;
200/200 forgeries without the site key rejected. **This is an integrity mechanism, not a defence against a compromised site**: a site that
holds its key can sign a poisoned update, which is why aggregation is robust (section 3). It is also not a formal security proof.

## 5. Sensor fault vs process fault triage

Sensor faults (stuck, dropout, spike, drift on one channel) were **injected into real normal rows**; process faults are the labelled SKAB anomalies.
Statistic: share of reconstruction error carried by the single worst channel. AUC 0.867 on held-out.

| Operating point (chosen on tune files) | Sensor faults labelled "sensor" | Real process anomalies wrongly labelled "sensor" |
|---|---|---|
| tau = 0.51 (Youden) | 91.9% | 42.9% |
| tau = 0.95 (conservative, used in the demo) | 25.7% | 0.0% |

About a third of injected sensor faults never raised an alarm at all (for example a stuck channel that barely varies anyway).
Conclusion: usable as an **advisory hint, not as an automatic suppressor**.

## 6. Camera model

YOLOX-nano (Apache-2.0) runs through ONNX Runtime; about 20-27 ms per frame on this sandbox's CPU. Frames are replayed sample images, and
detections are not fused with telemetry. No accuracy was measured on real forecourt footage.

## What has NOT been measured

NPU execution, NPU latency or power, ARM64 behaviour, any real deployment, false alarms on long recordings, drift handling, adaptive attackers.
