import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from haan.backchannel import HELDOUT_SPEAKERS, HaanModel, make_window, split_of  # noqa: E402
from haan.lexicon import tts_jobs  # noqa: E402
from haan.model import SmartTurnModel  # noqa: E402


def test_turn_output_is_unchanged_by_the_new_head():
    torch.manual_seed(0)
    turn = SmartTurnModel().eval()
    haan = HaanModel(turn).eval()
    x = torch.randn(2, 80, 800)
    with torch.no_grad():
        expected = torch.sigmoid(turn(x))
        p_complete, p_bc = haan(x)
    assert torch.allclose(expected, p_complete, atol=1e-6)
    assert p_bc.shape == (2,) and ((p_bc > 0) & (p_bc < 1)).all()


def test_freeze_turn_leaves_only_the_head_trainable():
    haan = HaanModel()
    haan.freeze_turn()
    trainable = {n.split(".")[0] for n, p in haan.named_parameters() if p.requires_grad}
    assert trainable == {"backchannel"}


def test_split_holds_out_speakers_and_phrases():
    jobs = tts_jobs()
    splits = {j["id"]: split_of(j) for j in jobs}
    for j in jobs:
        if j["speaker"] == HELDOUT_SPEAKERS[j["lang"]]:
            assert splits[j["id"]] == "test"
    # a held-out phrase never appears in train, whoever speaks it
    test_phrases = {(j["lang"], j["text"]) for j in jobs if int(j["id"].split("-")[2]) % 5 == 4}
    train_phrases = {(j["lang"], j["text"]) for j in jobs if splits[j["id"]] == "train"}
    assert not test_phrases & train_phrases
    assert set(splits.values()) == {"train", "val", "test"}


def test_make_window_is_8s_and_keeps_the_utterance_at_the_end():
    utt = np.ones(8000, np.float32) * 0.5
    win = make_window(utt, random.Random(0), channel="wideband", train=False)
    assert len(win) == 8 * 16000
    assert np.allclose(win[-8000:], 0.5) and np.allclose(win[:-8000], 0)
