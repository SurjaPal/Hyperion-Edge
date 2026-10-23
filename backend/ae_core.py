"""Windowed MLP autoencoder used on-device (same implementation the offline experiments in ml/ evaluate)."""
from __future__ import annotations
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


def windows(Z: np.ndarray, W: int) -> np.ndarray:
    n = len(Z) - W + 1
    idx = np.arange(W)[None, :] + np.arange(n)[:, None]
    return Z[idx].reshape(n, W * Z.shape[1])


class WindowAE:
    def __init__(self, window, hidden, bottleneck, alpha, activation, lr=1e-3, batch=32, seed=0, n_ch=8):
        self.W, self.n_ch = window, n_ch
        layers = tuple(hidden) + (bottleneck,) + tuple(hidden)[::-1]
        self.m = MLPRegressor(hidden_layer_sizes=layers, activation=activation, alpha=alpha,
                              learning_rate_init=lr, batch_size=batch, random_state=seed, tol=1e-7)
        d = window * n_ch
        self.m.partial_fit(np.zeros((2, d)), np.zeros((2, d)))   # allocate weights

    def get_w(self) -> np.ndarray:
        return np.concatenate([a.ravel() for a in self.m.coefs_] + [b.ravel() for b in self.m.intercepts_])

    def set_w(self, v: np.ndarray):
        i = 0
        for a in self.m.coefs_:
            a[...] = v[i:i + a.size].reshape(a.shape); i += a.size
        for b in self.m.intercepts_:
            b[...] = v[i:i + b.size]; i += b.size

    def train(self, A: np.ndarray, epochs: int):
        for _ in range(epochs):
            self.m.partial_fit(A, A)

    def n_params(self) -> int:
        return int(sum(c.size for c in self.m.coefs_) + sum(b.size for b in self.m.intercepts_))

    def export_onnx(self) -> bytes:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        d = self.W * self.n_ch
        return convert_sklearn(self.m, initial_types=[("input", FloatTensorType([None, d]))],
                               target_opset=13).SerializeToString()


class OnnxScorer:
    """ONNX Runtime session with QNN(NPU)-first provider selection; falls back to CPU. Reports the provider actually active."""
    def __init__(self, model_bytes: bytes, W: int, n_ch: int = 8):
        import onnxruntime as ort
        avail = ort.get_available_providers(); provs = []
        if "QNNExecutionProvider" in avail:
            provs.append(("QNNExecutionProvider", {"backend_path": "QnnHtp.dll", "enable_htp_fp16_precision": "1"}))
        provs.append("CPUExecutionProvider")
        self.s = ort.InferenceSession(model_bytes, providers=provs)
        self.provider = self.s.get_providers()[0]; self.W, self.n_ch = W, n_ch

    def channel_errors(self, win_flat: np.ndarray) -> np.ndarray:
        A = win_flat.astype(np.float32).reshape(1, -1)
        R = self.s.run(None, {"input": A})[0].reshape(1, -1)
        return ((R - A) ** 2).reshape(self.W, self.n_ch).mean(axis=0)


class SklearnScorer:
    provider = "sklearn (CPU, ONNX export unavailable)"
    def __init__(self, ae: WindowAE): self.ae, self.W, self.n_ch = ae, ae.W, ae.n_ch
    def channel_errors(self, win_flat):
        A = win_flat.reshape(1, -1); R = self.ae.m.predict(A)
        return ((R - A) ** 2).reshape(self.W, self.n_ch).mean(axis=0)
