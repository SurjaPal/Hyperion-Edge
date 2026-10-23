# Architecture

```
                         ┌───────────────────────────────────────────────┐
                         │              SITE  (one HP PC)                 │
                         │                                                 │
  vision_module.py         Sensing                                        │
                         │   camera frames -> YOLOX-nano (open-source,      │
                         │   Apache-2.0) via ONNX Runtime                    │
                         │   equipment telemetry stream (real sensor         │
                         │   recording in this build; live PLC/RTU feed       │
                         │   on a real deployment)                            │
                         │                 |                                  │
  ae_core.py                On-device inference                             │
                         │   WindowAE (small MLP autoencoder, ~2.2k           │
                         │   params) trained on the site's own first 300      │
                         │   samples; exported to ONNX; scored through        │
                         │   ONNX Runtime with QNNExecutionProvider           │
                         │   requested first, CPUExecutionProvider fallback   │
                         │                 |                                  │
  real_node.py               Explanation + triage                          │
                         │   per-channel reconstruction error names the       │
                         │   channels that moved; error-concentration ratio   │
                         │   gives an advisory sensor-fault-vs-process hint   │
                         │   (template text; no language model)               │
                         │                 |                                  │
  trust_ledger.py            Trust layer                                   │
                         │   SHA-256 hash-chained local log, tamper-evident   │
                         │   even fully offline                               │
                         └───────────────────────────────────────────────┘
                                           |
                             (only when connectivity exists: a
                              small HMAC-committed weight delta,
                              KBs not GBs)
                                           v
  provenance.py + real_swarm.py     Swarm sync layer
                             site submits weights with an HMAC-chained
                             commitment -> aggregator verifies -> aggregates
                             by coordinate-wise median (tolerates a minority
                             of poisoned updates) -> broadcasts the new
                             shared model
```

`server.py` is the demo harness: it drives 3 `RealSiteNode` instances by replaying real sensor
recordings at accelerated speed and a `RealSwarm` orchestrating rounds between them, and serves
the dashboard. On a real deployment, one `RealSiteNode`-equivalent process is the entire
application at a site; there is no "server.py" replaying anything.

## Why one neural model instead of a tree ensemble

An earlier version of this project used an Isolation Forest. `ml/final_eval.py` compares it
against PCA reconstruction and a windowed autoencoder on 14 held-out real files under an
identical tuning budget; the autoencoder won on F1 (see `docs/RESULTS.md`) and, unlike a tree
ensemble, exports cleanly to ONNX and is a genuine candidate for the Hexagon NPU path through
`QNNExecutionProvider` — the same pipeline the camera model uses. Both neural models in this
project go through the same provider-selection code (`ae_core.OnnxScorer` /
`vision_module._build_session`), so "what ran on the NPU" is a single, honestly-reported fact
(`site.backend` / `site.vision.provider` in the API), not a claim split across two different
mechanisms.

## Why federated learning here, precisely

`FederatedFaultScorer`-equivalent weights (the autoencoder's own parameters, ~2.2k floats) are
what's exchanged — never telemetry, never video, never the local ledger. `docs/RESULTS.md`
section 3 gives the measured effect: federated initialization shortens cold start at very
little local data (~30 samples) and gives no measurable benefit beyond that. It is a
bring-up-time optimization, not an accuracy feature, and the project does not claim otherwise.

## Why a hash chain and not a real blockchain, still

A single unmanned site has no peers to reach consensus with in real time, and a real
blockchain's consensus/replication overhead buys nothing here. What's needed is: prove
after-the-fact whether a site's own log was altered while offline. A hash chain gives that for
near-zero compute, which matters on a device also running continuous inference. See
`docs/PROBLEM_AND_APPROACH.md` section 3 for what this does *not* cover (an attacker with full
disk access can still rewrite an entire local chain; chain heads are not yet anchored off-device).

## Why HMAC commitments in addition to robust aggregation

`provenance.py` proves an update wasn't altered in transit, replayed, or forged by someone
without the site's key (measured in `docs/RESULTS.md` section 4: 100% detection across 600
trials of each attack). It does **not** stop a site that legitimately holds its key from
signing a poisoned update — that's a different threat, handled by median/trimmed-mean
aggregation in `real_swarm.py` (section 3 of the same doc). The two mechanisms are
complementary and neither substitutes for the other.

## Data that never leaves the device

- Raw telemetry and camera frames
- The locally fitted baseline (mean/std) and the full tamper-evident ledger
- Any confirmed-fault labels used to calibrate the site's own threshold

Only the autoencoder's weight vector is ever transmitted, and only during a federated round,
wrapped in an HMAC commitment.
