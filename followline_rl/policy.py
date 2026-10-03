"""Deterministic PPO actor inference using NumPy only (no torch/gym/ROS)."""
import numpy as np


class NumpyActor:
    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.weights = {key: data[key].copy() for key in data.files}
        expected = {"w0": (64,7), "b0": (64,), "w1": (64,64), "b1": (64,),
                    "w2": (1,64), "b2": (1,)}
        for name, shape in expected.items():
            if name not in self.weights or self.weights[name].shape != shape:
                raise ValueError(f"Invalid actor shape: {name}, expected {shape}")

    def __call__(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        if x.shape != (7,) or not np.isfinite(x).all():
            raise ValueError("Expected seven finite observation features")
        p = self.weights
        x = np.tanh(p["w0"] @ x + p["b0"])
        x = np.tanh(p["w1"] @ x + p["b1"])
        return np.clip(p["w2"] @ x + p["b2"], -1, 1).astype(np.float32)
