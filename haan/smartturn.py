"""Smart Turn v3 inference, matching pipecat-ai/smart-turn's inference.py.

Audio is cut to the last 8 s (or left-padded with zeros), turned into Whisper
log-mel features, and scored by the ONNX model. The output is P(turn complete).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import WhisperFeatureExtractor

SR = 16_000
WINDOW_S = 8


def last_n_seconds(audio: np.ndarray, n: int = WINDOW_S, sr: int = SR) -> np.ndarray:
    keep = n * sr
    if len(audio) >= keep:
        return audio[-keep:]
    return np.pad(audio, (keep - len(audio), 0))


class SmartTurn:
    def __init__(self, onnx_path: str | Path, threads: int = 1):
        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.inter_op_num_threads = 1
        so.intra_op_num_threads = threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(onnx_path), sess_options=so)
        self.features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def featurize(self, batch: list[np.ndarray]) -> np.ndarray:
        clipped = [last_n_seconds(np.asarray(a, dtype=np.float32)) for a in batch]
        return self.features(
            clipped,
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_S * SR,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)

    def score(self, feats: np.ndarray) -> np.ndarray:
        """P(complete) for a batch of features. The graph output is named "logits"
        but already holds sigmoid probabilities (see upstream inference.py)."""
        return self.session.run(None, {"input_features": feats})[0].reshape(-1).astype(np.float32)

    def predict_proba(self, batch: list[np.ndarray]) -> np.ndarray:
        return self.score(self.featurize(batch))


class TorchSmartTurn(SmartTurn):
    """Same preprocessing, scored by a PyTorch state dict (GPU if available)."""

    def __init__(self, weights: str | Path):
        import torch

        from haan.model import SmartTurnModel

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SmartTurnModel()
        self.model.load_state_dict(torch.load(weights, map_location="cpu"))
        self.model.to(self.device).eval()
        self.features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def score(self, feats: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            x = self.torch.from_numpy(feats).to(self.device)
            return self.torch.sigmoid(self.model(x)).float().cpu().numpy()
