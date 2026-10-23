# Compliance audit vs. the challenge guidelines

Source: the Unstop screenshots (Eligibility, Stages, Submission Guidelines, Evaluation
Criteria). Only part of the Details/Stages pages was visible; items marked OPEN need
the remaining text checked before submitting.

| Guideline | Verdict | Evidence / what to do |
|---|---|---|
| Residents of India, 18+ | Participant check | Confirm your own eligibility. Not a project matter. |
| Individual participation | OK | Solo build; no team members are listed anywhere in the repo. |
| Solution "designed, developed, or intended to be optimised for Snapdragon-powered HP PCs" | Partial | Design targets the HP OmniBook Ultra (Snapdragon X2). ONNX Runtime provider selection is implemented. NPU execution is NOT yet measured on a device. The wording "intended to be optimised" covers intent, but a measured run is much stronger. |
| Proposal may be newly created | OK | Code written from scratch during the submission window. |
| If pre-existing, must be significantly modified to add AI models from Qualcomm AI Hub or other open-source platforms | OK | Strictly applies only to pre-existing proposals; this is new. It is also covered anyway: a real open-source model (YOLOX-nano, Apache-2.0) is integrated and executed. |
| Work/idea solely owned by the participant | Risk to manage | All code here is original. But the idea lineage touches your IOCL internship work and team projects. Do NOT include IOCL documents, tender text, data, branding or confidential details. Use only the synthetic data in this repo. If unsure, ask your mentor/IOCL whether a generic derived concept is fine to submit. |
| One submission per participant | Participant check | Submit exactly once, from one account. Duplicates: only the first counts. |
| Submission cannot be changed after submission | Process | Final dry run first; submit hours before 11:59 PM IST Sep 30. |
| "Double-check all intake-form fields" | OPEN | The form fields were not visible in the screenshots. Open the Submit dialog now, read every field, and tell me what it asks for (link, file, video, text limits). |
| Eval: Technical Implementation | Strong, one gap | Real vision model + real tuned telemetry model (74 Optuna trials, held-out-blind evaluation) + real federated round with HMAC provenance and robust aggregation, all measured on real sensor data (`docs/RESULTS.md`). Remaining gap: no on-device NPU measurement. |
| Eval: Application Use Case & Innovation | OK | Offline industrial monitoring. Prior art for federated learning and poisoning-robust aggregation is named explicitly in `docs/PROBLEM_AND_APPROACH.md` section 4 -- claim the integrated, evaluated system, not novelty of the individual techniques. |
| Eval: Deployment & Accessibility | Partial | Offline fonts bundled, no cloud dependency, one-command launch, ~10 KB telemetry model. Gaps: not tested on Windows on Arm; ledger and baseline are in memory only (full list in `docs/PROBLEM_AND_APPROACH.md` section 3). |
| Eval: Presentation & Documentation | OK, needs your output | `docs/RESULTS.md` and `docs/PROBLEM_AND_APPROACH.md` give numbers and reasoning a judge can push on. You still need the demo video and the proposal write-up. |
| "Build & Present" title, other stages | OPEN | Only the Solution Submission stage was visible. Check the Stages & Timeline tab for a later presentation round and any format rules. |

## Weak points found and fixed in this pass

- Explanations used the simulator's ground-truth fault label. Now inferred from the
  observed deviation vector via a signature library.
- Dashboard header said "Hexagon NPU" while running on CPU, and the NPU load gauge was
  a made-up formula. Both removed; the header now shows the live provider.
- Docs contained an invented GenieX API snippet and implied the tree-ensemble model
  would compile to the NPU. Both removed and corrected.
- No pretrained AI model existed in the build. YOLOX-nano now runs through ONNX Runtime.
- Dashboard loaded fonts from a CDN, contradicting offline-first. Fonts are bundled.
- Bundled third-party photos removed from the deliverable; fetched separately for tests.
- Low-contrast dim text raised; event log marked as a live region.

## Still open (cannot be closed from this sandbox)

1. Run on a real Snapdragon device and record provider, latency and Task Manager NPU use.
2. On-device language model for the notes (currently a template, stated as such).
3. Persist the ledger and baseline to disk.
4. Tell the judges plainly what is simulated (see the README table).


## Second pass: deeper research and ML rigor (this update)

What changed since the first build:
- Replaced synthetic telemetry with real recordings from a physical pump testbed (SKAB, GPL-3.0,
  fetched separately, not bundled).
- Replaced the untuned Isolation Forest with a windowed autoencoder selected via 74 Optuna trials
  under a held-out-blind protocol, compared fairly against equally-tuned PCA and Isolation Forest
  baselines. See `docs/RESULTS.md` section 1.
- Measured the real accuracy cost of INT8 quantization for the QNN-oriented path (section 2).
- Ran real federated cold-start and poisoning experiments across 5 non-IID simulated sites on
  real data, including a specific attack (sign-flip) and three aggregation rules (section 3).
- Added HMAC-chained update commitments with measured detection rates against tampering, replay,
  and forgery (section 4), and was explicit that this does not defend against a compromised site
  that holds its own key.
- Added a sensor-fault-vs-process-fault triage statistic, tested by injecting synthetic faults
  into real normal rows, with an explicit accuracy/safety trade-off between two operating points
  (section 5).
- Wrote a 17-row failure-mode register (`docs/PROBLEM_AND_APPROACH.md` section 3) naming what is
  handled, partially handled, or still open, instead of only listing what works.
- Removed an overclaim: an earlier note implied the tree-ensemble telemetry model would not
  reach the NPU; it has been replaced with a model that exports to ONNX and goes through the same
  QNN-first provider selection as the vision model, and the docs were corrected to say so.

What is still open, unchanged from before: no run on real Snapdragon hardware, no on-device
language model, ledger and baseline are in-memory only, and the main open technical weakness --
false alarms from baseline drift (measured at 21.5% pre-onset / 62.9% post-anomaly in
`docs/RESULTS.md`) -- is not yet fixed.
