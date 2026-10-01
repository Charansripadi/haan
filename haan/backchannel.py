"""Backchannel vs interruption head on top of the Smart Turn encoder.

The encoder and the turn head stay frozen, so turn detection is bit-for-bit
unchanged; only a small attention-pooling + MLP head is trained. At run time
the caller audio heard while the agent is talking sits at the end of the 8 s
window (left-padded), exactly like Smart Turn's input.

Outputs of HaanModel: (p_complete, p_backchannel).
"""

from __future__ import annotations

import random
import zlib

import numpy as np
import torch
from torch import nn

from haan.model import SmartTurnModel
from haan.smartturn import SR, last_n_seconds
from haan.telephony import CHANNELS, simulate

# speakers whose clips are never trained on (one per language)
HELDOUT_SPEAKERS = {"eng": "Jatin", "hin": "Rani", "tel": "Kiran"}


def split_of(row: dict) -> str:
    """'test' for held-out speakers or held-out phrases (every 5th phrase per
    language and label), 'val' for a further 10% of clips, else 'train'."""
    phrase_idx = int(row["id"].split("-")[2])
    if row["speaker"] == HELDOUT_SPEAKERS.get(row["lang"]) or phrase_idx % 5 == 4:
        return "test"
    return "val" if zlib.crc32(row["id"].encode()) % 10 == 0 else "train"


def make_window(utt: np.ndarray, rng: random.Random, channel: str | None = None, train: bool = True) -> np.ndarray:
    """Put an utterance at the end of an 8 s window with a noise floor and a channel."""
    audio = np.asarray(utt, dtype=np.float32)
    if train:
        audio = audio * 10 ** (rng.uniform(-15, 0) / 20)
        tail = int(rng.uniform(0.0, 0.3) * SR)  # VAD hands over a little trailing silence
        audio = np.concatenate([audio, np.zeros(tail, np.float32)])
    win = last_n_seconds(audio)
    if train:
        snr_db = rng.uniform(10, 40)
        rms = np.sqrt(np.mean(audio**2) + 1e-12)
        nprng = np.random.default_rng(rng.randrange(2**31))
        win = win + nprng.standard_normal(len(win)).astype(np.float32) * rms / 10 ** (snr_db / 20)
        channel = channel or rng.choice(CHANNELS)
    return simulate(np.clip(win, -1, 1), channel or "wideband")


class BackchannelHead(nn.Module):
    def __init__(self, hidden: int = 384):
        super().__init__()
        self.pool_attention = nn.Sequential(nn.Linear(hidden, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(nn.Linear(hidden, 128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        w = torch.softmax(self.pool_attention(hidden_states), dim=1)
        return self.classifier(torch.sum(hidden_states * w, dim=1)).squeeze(-1)


class HaanModel(nn.Module):
    """Smart Turn v3.2 plus a backchannel head; one encoder pass, two probabilities."""

    def __init__(self, turn: SmartTurnModel | None = None):
        super().__init__()
        self.turn = turn or SmartTurnModel()
        self.backchannel = BackchannelHead(self.turn.config.d_model)

    def freeze_turn(self) -> None:
        for p in self.turn.parameters():
            p.requires_grad = False
        self.turn.eval()

    def forward(self, input_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.turn.encoder(input_features=input_features).last_hidden_state
        w = torch.softmax(self.turn.pool_attention(hidden), dim=1)
        p_complete = torch.sigmoid(self.turn.classifier(torch.sum(hidden * w, dim=1)).squeeze(-1))
        p_backchannel = torch.sigmoid(self.backchannel(hidden))
        return p_complete, p_backchannel
