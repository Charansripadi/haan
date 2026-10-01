import numpy as np
import pytest
from scipy.signal import welch

from haan.metrics import binary_report
from haan.telephony import CHANNELS, mulaw_decode, mulaw_encode, simulate

SR = 16_000


def tone(freq, seconds=1.0, amp=0.5):
    t = np.arange(int(SR * seconds)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def band_power(x, lo, hi):
    f, p = welch(x, fs=SR, nperseg=2048)
    return p[(f >= lo) & (f < hi)].sum()


def test_mulaw_roundtrip_is_close_and_8bit():
    x = np.linspace(-1, 1, 1001, dtype=np.float32)
    codes = mulaw_encode(x)
    assert codes.dtype == np.uint8 and codes.min() == 0 and codes.max() == 255
    assert np.max(np.abs(mulaw_decode(codes) - x)) < 0.04  # coarse at the loud end, by design


def test_mulaw_is_finer_for_quiet_samples():
    quiet = np.full(10, 0.01, dtype=np.float32)
    loud = np.full(10, 0.9, dtype=np.float32)
    err_q = abs(mulaw_decode(mulaw_encode(quiet))[0] - 0.01)
    err_l = abs(mulaw_decode(mulaw_encode(loud))[0] - 0.9)
    assert err_q < err_l


@pytest.mark.parametrize("channel", CHANNELS)
def test_simulate_keeps_length_and_dtype(channel):
    x = tone(440, 1.2345)
    y = simulate(x, channel)
    assert y.shape == x.shape and y.dtype == np.float32


def test_wideband_is_identity():
    x = tone(440)
    assert np.array_equal(simulate(x, "wideband"), x)


@pytest.mark.parametrize("channel", ["narrowband", "pstn"])
def test_phone_channels_remove_the_upper_band(channel):
    x = tone(1000) + tone(6000)
    y = simulate(x, channel)
    assert band_power(y, 5500, 6500) < 1e-3 * band_power(x, 5500, 6500)
    assert band_power(y, 900, 1100) > 0.5 * band_power(x, 900, 1100)


def test_pstn_also_removes_low_rumble():
    x = tone(100) + tone(1000)
    nb, pstn = simulate(x, "narrowband"), simulate(x, "pstn")
    assert band_power(pstn, 80, 120) < 0.1 * band_power(nb, 80, 120)


def test_unknown_channel_raises():
    with pytest.raises(ValueError):
        simulate(tone(440), "carrier-pigeon")


def test_binary_report_counts():
    y = [True, True, False, False]
    p = [0.9, 0.2, 0.8, 0.1]  # tp, fn, fp, tn
    r = binary_report(y, p)
    assert (r["tp"], r["fn"], r["fp"], r["tn"]) == (1, 1, 1, 1)
    assert r["accuracy"] == 0.5 and r["incomplete_precision"] == 0.5
