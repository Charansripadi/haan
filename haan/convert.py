"""Recover PyTorch weights from the published FP32 Smart Turn ONNX file.

Daily publishes Smart Turn v3.2 only as ONNX. The GPU variant keeps FP32
weights. Named parameters (biases, norms, convs, embeddings) are stored under
their PyTorch names with an "inner." prefix; Linear weights were folded into
MatMul initializers named val_N, stored transposed. Each MatMul node keeps the
module path in its "pkg.torch.onnx.name_scopes" metadata, which gives the
mapping back.

    python -m haan.convert models/smart-turn-v3.2-gpu.onnx models/smart-turn-v3.2.pt
"""

from __future__ import annotations

import ast
import sys

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from haan.model import SmartTurnModel

PREFIX = "inner."


def _scopes(node) -> list[str]:
    for p in node.metadata_props:
        if p.key == "pkg.torch.onnx.name_scopes":
            return ast.literal_eval(p.value)
    return []


def onnx_state_dict(path: str) -> dict[str, torch.Tensor]:
    model = onnx.load(path)
    inits = {t.name: numpy_helper.to_array(t) for t in model.graph.initializer}
    state = {k[len(PREFIX):]: v for k, v in inits.items() if k.startswith(PREFIX)}
    for node in model.graph.node:
        if node.op_type != "MatMul":
            continue
        weights = [i for i in node.input if i in inits and not i.startswith(PREFIX) and inits[i].ndim == 2]
        if not weights:
            continue
        module = _scopes(node)[-2]  # last scope is the op ("linear"), the one before is the module
        state[module[len(PREFIX):] + ".weight"] = np.ascontiguousarray(inits[weights[0]].T)
    return {k: torch.from_numpy(np.array(v)) for k, v in state.items()}


def load_from_onnx(path: str) -> SmartTurnModel:
    model = SmartTurnModel()
    state = onnx_state_dict(path)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing or unexpected:
        raise ValueError(f"weight mapping failed: missing={missing} unexpected={unexpected}")
    for k, v in state.items():
        if tuple(v.shape) != tuple(expected[k].shape):
            raise ValueError(f"{k}: onnx {tuple(v.shape)} vs torch {tuple(expected[k].shape)}")
    model.load_state_dict(state)
    return model.eval()


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    torch.save(load_from_onnx(src).state_dict(), dst)
    print(f"saved {dst}")
