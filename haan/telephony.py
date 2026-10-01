"""Telephone channel simulation.

Phone calls (PSTN, SIP trunks, Twilio Media Streams) carry 8 kHz audio,
companded with G.711 mu-law. Voice agents upsample that back to 16 kHz before
running a 16 kHz model, which restores the sample rate but not the 4-8 kHz band
that was never captured. These functions reproduce that path on 16 kHz audio so
a model can be evaluated (and trained) on what it will actually hear on a call.

Channels:
    wideband   16 kHz audio, untouched (what most training data looks like)
    narrowband 16k -> 8k -> 8-bit mu-law -> 8k -> 16k
    pstn       narrowband plus the 300-3400 Hz voice band filter of a phone line
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, resample_poly, sosfilt

SR = 16_000
PHONE_SR = 8_000
MU = 255
CHANNELS = ("wideband", "narrowband", "pstn")


def mulaw_encode(x: np.ndarray, mu: int = MU) -> np.ndarray:
    """Float audio in [-1, 1] to 8-bit mu-law codes (0..255)."""
    x = np.clip(x, -1.0, 1.0)
    y = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return np.round((y + 1) / 2 * mu).astype(np.uint8)


def mulaw_decode(codes: np.ndarray, mu: int = MU) -> np.ndarray:
    """8-bit mu-law codes back to float audio in [-1, 1]."""
    y = codes.astype(np.float64) / mu * 2 - 1
    return (np.sign(y) * np.expm1(np.abs(y) * np.log1p(mu)) / mu).astype(np.float32)


def _voice_band(x: np.ndarray, sr: int) -> np.ndarray:
    sos = butter(4, [300, 3400], btype="bandpass", fs=sr, output="sos")
    return sosfilt(sos, x).astype(np.float32)


def to_phone_codes(audio: np.ndarray, sr: int = SR, voice_band: bool = False) -> np.ndarray:
    """16 kHz float audio to 8 kHz mu-law codes, as sent over a phone line."""
    x = resample_poly(audio.astype(np.float64), PHONE_SR, sr)
    if voice_band:
        x = _voice_band(x, PHONE_SR)
    return mulaw_encode(x)


def from_phone_codes(codes: np.ndarray, sr: int = SR) -> np.ndarray:
    """8 kHz mu-law codes to float audio at `sr`, as a voice agent receives it."""
    return resample_poly(mulaw_decode(codes), sr, PHONE_SR).astype(np.float32)


def simulate(audio: np.ndarray, channel: str, sr: int = SR) -> np.ndarray:
    """Pass 16 kHz audio through a channel and return 16 kHz audio of equal length."""
    if channel not in CHANNELS:
        raise ValueError(f"unknown channel {channel!r}, expected one of {CHANNELS}")
    audio = np.asarray(audio, dtype=np.float32)
    if channel == "wideband":
        return audio
    out = from_phone_codes(to_phone_codes(audio, sr, voice_band=channel == "pstn"), sr)
    # resample_poly can be off by a sample; keep alignment with the input
    if len(out) >= len(audio):
        return out[: len(audio)]
    return np.pad(out, (0, len(audio) - len(out)))
