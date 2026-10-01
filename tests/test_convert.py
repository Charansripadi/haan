from pathlib import Path

import numpy as np
import pytest

ONNX = Path(__file__).resolve().parents[1] / "models/smart-turn-v3.2-gpu.onnx"
torch = pytest.importorskip("torch")
ort = pytest.importorskip("onnxruntime")


@pytest.mark.skipif(not ONNX.exists(), reason="download models/smart-turn-v3.2-gpu.onnx first")
def test_recovered_weights_reproduce_onnx_outputs():
    from haan.convert import load_from_onnx

    model = load_from_onnx(str(ONNX))
    x = np.random.default_rng(0).standard_normal((3, 80, 800)).astype(np.float32)
    expected = ort.InferenceSession(str(ONNX)).run(None, {"input_features": x})[0].reshape(-1)
    with torch.no_grad():
        got = torch.sigmoid(model(torch.from_numpy(x))).numpy()
    assert np.abs(expected - got).max() < 1e-5
