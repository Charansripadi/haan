"""Training data: Smart Turn parquet shards, with optional telephone-channel augmentation."""

from __future__ import annotations

import io
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import IterableDataset, get_worker_info
from transformers import WhisperFeatureExtractor

from haan.smartturn import SR, WINDOW_S, last_n_seconds
from haan.telephony import simulate

# arm -> (channel weights for wideband, narrowband, pstn)
ARMS = {
    "control": (1.0, 0.0, 0.0),
    "telephony": (1 / 3, 1 / 3, 1 / 3),
}
CHANNEL_NAMES = ("wideband", "narrowband", "pstn")


def decode(audio_cell: dict) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr).astype(np.float32)
    return wav


def augment(audio: np.ndarray, arm: str, rng: random.Random) -> tuple[np.ndarray, str]:
    channel = rng.choices(CHANNEL_NAMES, weights=ARMS[arm])[0]
    if channel != "wideband":
        # phone lines see a wide range of levels; mu-law error depends on level
        gain = 10 ** (rng.uniform(-12, 3) / 20)
        audio = np.clip(audio * gain, -1, 1)
    return simulate(audio, channel), channel


class ShardStream(IterableDataset):
    """Streams (features, label) from parquet shards. Shards are split across workers;
    rows are shuffled within a buffer. One pass = one epoch."""

    def __init__(self, shards: list[Path], arm: str, seed: int = 0, buffer: int = 2048):
        self.shards = sorted(shards)
        self.arm = arm
        self.seed = seed
        self.buffer = buffer
        self.epoch = 0
        self.features = WhisperFeatureExtractor(chunk_length=WINDOW_S)

    def _rows(self, shards, rng):
        rng.shuffle(shards)
        for shard in shards:
            for rb in pq.ParquetFile(shard).iter_batches(batch_size=256, columns=["audio", "endpoint_bool"]):
                yield from rb.to_pylist()

    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        rng = random.Random(hash((self.seed, self.epoch, wid)))
        mine = [s for i, s in enumerate(self.shards) if i % nw == wid]
        buf = []
        for row in self._rows(mine, rng):
            buf.append(row)
            if len(buf) >= self.buffer:
                yield self._example(buf.pop(rng.randrange(len(buf))), rng)
        rng.shuffle(buf)
        for row in buf:
            yield self._example(row, rng)

    def _example(self, row, rng):
        audio, _ = augment(decode(row["audio"]), self.arm, rng)
        feats = self.features(
            last_n_seconds(audio),
            sampling_rate=SR,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_S * SR,
            truncation=True,
            do_normalize=True,
        ).input_features[0]
        return torch.from_numpy(feats.astype(np.float32)), torch.tensor(float(row["endpoint_bool"]))
