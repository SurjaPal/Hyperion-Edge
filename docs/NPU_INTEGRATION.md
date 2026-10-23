# Moving from this dev build to the Hexagon NPU

**Status: not yet executed on Snapdragon hardware.** Everything below marked
*verify on device* is an untested step, not a result.

## What already runs the real NPU-candidate code path

Both neural models in this project -- the telemetry autoencoder (`ae_core.py`) and the camera
model (`vision_module.py`) -- are executed through ONNX Runtime sessions built the same way:
request `QNNExecutionProvider` (HTP backend, `QnnHtp.dll`) first when that provider is present,
fall back to `CPUExecutionProvider` otherwise. In this dev sandbox only the CPU provider
exists, so both run on CPU; the dashboard and the `/api/state` JSON print the provider actually
returned by the live session (`site.backend`, `site.vision.provider`), never a hardcoded label.

`docs/RESULTS.md` section 2 measures the accuracy cost of quantizing the autoencoder for the
QNN-oriented INT8 path (about 0.006 F1) using ONNX Runtime's own QDQ quantization tooling, on
real held-out sensor data. That is a genuine, reproducible number. CPU latency for the
quantized model is *slower* than float32 in this sandbox (no NPU to accelerate the INT8 ops),
which says nothing about NPU speed and is stated as such in that document.

## Steps to verify on the HP Snapdragon device

1. Install a QNN-enabled ONNX Runtime build for Windows on Arm (`onnxruntime-qnn`).
   *Verify on device:* which Python architecture (ARM64 vs x64) the wheel needs. Qualcomm's AI
   Hub client tooling has documented an x64-Python requirement for `qai_hub_models`, while the
   runtime package may need ARM64 Python; use separate environments if they conflict.
2. Run `python scripts/run_demo.py`. Check `site.backend` and `site.vision.provider` in
   `/api/state` (also shown in the dashboard header). If either says
   `CPUExecutionProvider`, the NPU is NOT being used for that model and nothing should be
   claimed for it.
3. Confirm real NPU activity independently in Windows Task Manager (NPU tab) while the app
   runs, and record latency for both models, CPU vs QNN, alongside the F1/FAR/MAR numbers
   already measured in `docs/RESULTS.md` -- report them together, not the accuracy numbers alone.
4. *Verify on device:* the HTP backend is designed around quantized models. `vision_module.py`
   and the quantization path in `ml/quantize.py` already enable ONNX Runtime's documented
   fp16/QDQ options; if a model still falls back to CPU, re-run `ml/quantize.py`'s approach on
   that model and re-measure the accuracy delta the same way section 2 of `docs/RESULTS.md` does.
5. Optional: compile/profile through Qualcomm AI Hub (account + API token required, cloud
   round-trip can be slow -- start this well before a live demo). Take the exact device names
   and export flags from Qualcomm's current AI Hub documentation rather than from this repo.

## On-device language model

Not implemented. The explanation text in `real_node.py` is a deterministic template built from
the per-channel error vector, and the code and docs say so throughout. Qualcomm's GenieX SDK is
the intended runtime for a small Llama/Granite/Qwen-class model on the Hexagon NPU; use its
current official documentation for the real API rather than any snippet written from memory.

## Reproducing the measured results on new hardware

`ml/` contains the exact scripts used for every number in `docs/RESULTS.md`
(`ml/final_eval.py` for detection quality, `ml/quantize.py` for the INT8 study,
`ml/federated_experiment.py` for federated/poisoning, `ml/sensor_gate.py` for the triage
statistic). Run `python scripts/fetch_assets.py` first to pull the real SKAB dataset, then any
of these scripts against a new backend by pointing `ae_core.OnnxScorer` at the target provider.
