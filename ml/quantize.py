"""
Float32 vs INT8 (QDQ) accuracy for the tuned autoencoder on the held-out real pump files.
Pipeline per file (this is exactly the on-device flow): learn baseline -> export ONNX -> quantize with the
site's own normal windows as calibration -> score the stream with ONNX Runtime.
CPU execution only here: this measures quantization error, NOT NPU speed.
"""
import sys, os, json, time, tempfile
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, onnx, onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from onnxruntime.quantization import quantize, QuantType, CalibrationDataReader
from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
from lib import *

cfg = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "heldout.json")))["ae_cfg"]
cfg["hidden"] = tuple(cfg["hidden"])
files = load_files(); _, HELD = split_tune_heldout(files)

class Reader(CalibrationDataReader):
    def __init__(self, A): self.it = iter([{"input": a[None, :].astype(np.float32)} for a in A])
    def get_next(self): return next(self.it, None)

class OnnxAE:
    def __init__(self, path, W): self.W = W; self.s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    def errors(self, Z):
        A = windows(Z, self.W).astype(np.float32); R = self.s.run(None, {"input": A})[0].reshape(len(A), -1)  # skl2onnx flattens multi-output regressors
        return ((R - A) ** 2).reshape(len(A), self.W, Z.shape[1]).mean(axis=1)

def build(f, tmp, act_type):
    X = f["X"]; mu, sd = X[:N_FIT].mean(0), X[:N_FIT].std(0) + 1e-6; Z = (X - mu) / sd
    keys = ("window","hidden","bottleneck","alpha","lr","epochs","batch","activation")
    ae = WindowAE(**{k: cfg[k] for k in keys}, seed=0).fit(Z[:N_FIT]); n_in = ae.W * 8
    fp32 = os.path.join(tmp, "f32.onnx")
    onnx.save(convert_sklearn(ae.m, initial_types=[("input", FloatTensorType([None, n_in]))], target_opset=13), fp32)
    pre = os.path.join(tmp, "pre.onnx"); qnn_preprocess_model(fp32, pre); src = pre if os.path.exists(pre) else fp32
    q = os.path.join(tmp, f"q_{act_type.name}.onnx")
    qc = get_qnn_qdq_config(src, Reader(windows(Z[:N_FIT], ae.W)), activation_type=act_type, weight_type=QuantType.QUInt8)
    quantize(src, q, qc)
    return ae, fp32, q

def run(kind_models, f):
    out = {}
    for name, m in kind_models.items():
        s, _ = score_stream("ae", f["X"], {**cfg, "_model": m}); a = alarms(s, cfg)[N_TRAIN:]; y = f["y"][N_TRAIN:]
        out[name] = (int(((a==1)&(y==1)).sum()), int(((a==1)&(y==0)).sum()), int(((a==0)&(y==1)).sum()), int(((a==0)&(y==0)).sum()))
    return out

tot = {}; sizes = {}; lat = {}
for f in HELD:
    for act in (QuantType.QUInt8, QuantType.QUInt16):
        with tempfile.TemporaryDirectory() as tmp:
            ae, fp32, q = build(f, tmp, act)
            models = {"sklearn float32": ae, "ONNX float32": OnnxAE(fp32, ae.W), f"ONNX QDQ act={act.name}": OnnxAE(q, ae.W)}
            for k, v in run(models, f).items():
                if k == "sklearn float32" and act == QuantType.QUInt16: continue
                t = tot.setdefault(k, [0, 0, 0, 0]); [t.__setitem__(i, t[i] + v[i]) for i in range(4)]
            sizes[f"float32 KB"] = round(os.path.getsize(fp32) / 1024, 1); sizes[f"QDQ {act.name} KB"] = round(os.path.getsize(q) / 1024, 1)
            if f is HELD[0]:
                x = np.random.randn(1, ae.W * 8).astype(np.float32)
                for k, path in (("float32", fp32), (f"QDQ {act.name}", q)):
                    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); s.run(None, {"input": x})
                    t0 = time.perf_counter(); [s.run(None, {"input": x}) for _ in range(2000)]; lat[f"{k} CPU us/inference"] = round((time.perf_counter() - t0) / 2000 * 1e6, 1)
res = {}
for k, (TP, FP, FN, TN) in tot.items():
    res[k] = dict(F1=round(2*TP/max(1, 2*TP+FP+FN), 4), FAR=round(100*FP/max(1, FP+TN), 2), MAR=round(100*FN/max(1, FN+TP), 2))
res["model_size_KB"] = sizes; res["cpu_latency"] = lat; res["note"] = "seed 0, 14 held-out files, CPU execution provider; measures quantization error only"
json.dump(res, open(os.path.join(os.path.dirname(__file__), "..", "results", "quantization.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
