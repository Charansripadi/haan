"""Smart Turn v3 architecture: Whisper-tiny encoder, attention pooling, MLP head.

Copied in structure from pipecat-ai/smart-turn train.py (BSD-2-Clause,
Copyright Daily) so that weights recovered from the published ONNX file load
one-to-one. `forward` returns raw logits; apply sigmoid for P(complete).
"""

from __future__ import annotations

import torch
from torch import nn
from transformers import WhisperConfig
from transformers.models.whisper.modeling_whisper import WhisperEncoder

ENCODER_POSITIONS = 400  # 8 s of audio -> 800 mel frames -> 400 after the stride-2 conv


def whisper_tiny_config() -> WhisperConfig:
    cfg = WhisperConfig(
        d_model=384,
        encoder_layers=4,
        encoder_attention_heads=6,
        encoder_ffn_dim=1536,
        num_mel_bins=80,
        activation_function="gelu",
    )
    cfg.max_source_positions = ENCODER_POSITIONS
    return cfg


class SmartTurnModel(nn.Module):
    def __init__(self, config: WhisperConfig | None = None):
        super().__init__()
        config = config or whisper_tiny_config()
        config.max_source_positions = ENCODER_POSITIONS
        self.config = config
        self.encoder = WhisperEncoder(config)
        h = config.d_model
        self.pool_attention = nn.Sequential(nn.Linear(h, 256), nn.Tanh(), nn.Linear(256, 1))
        self.classifier = nn.Sequential(
            nn.Linear(h, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def pooled(self, input_features: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(input_features=input_features).last_hidden_state
        weights = torch.softmax(self.pool_attention(hidden), dim=1)
        return torch.sum(hidden * weights, dim=1)

    def forward(self, input_features: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pooled(input_features)).squeeze(-1)
