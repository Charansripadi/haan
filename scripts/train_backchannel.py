"""Train the backchannel head on TTS clips (frozen Smart Turn encoder).

    python scripts/train_backchannel.py --tts data/tts --out runs/backchannel

Splits come from haan.backchannel.split_of: held-out speakers and phrases form
'test'; the real recordings in data/real are scored separately by
scripts/eval_backchannel.py and are never used here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch import nn
from transformers import WhisperFeatureExtractor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.backchannel import HaanModel, make_window, split_of  # noqa: E402
from haan.convert import load_from_onnx  # noqa: E402
from haan.smartturn import SR, WINDOW_S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_tts(tts_dir: Path) -> list[dict]:
    with (tts_dir / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["audio"], sr = sf.read(tts_dir / "wav" / f"{r['id']}.wav", dtype="float32")
        assert sr == SR
        r["y"] = 1.0 if r["label"] == "backchannel" else 0.0
        r["split"] = split_of(r)
    return rows


def featurize(fe: WhisperFeatureExtractor, wins: list[np.ndarray]) -> torch.Tensor:
    x = fe(wins, sampling_rate=SR, return_tensors="np", padding="max_length", max_length=WINDOW_S * SR,
           truncation=True, do_normalize=True).input_features
    return torch.from_numpy(x.astype(np.float32))


@torch.no_grad()
def evaluate(model, fe, rows, device, channel="wideband") -> dict:
    model.eval()
    if not rows:
        return {"n": 0, "accuracy": float("nan"), "backchannel_ignored": float("nan"), "interrupt_caught": float("nan")}
    rng = random.Random(0)
    probs = []
    for i in range(0, len(rows), 64):
        wins = [make_window(r["audio"], rng, channel=channel, train=False) for r in rows[i:i + 64]]
        probs.append(model(featurize(fe, wins).to(device))[1].cpu().numpy())
    p = np.concatenate(probs)
    y = np.array([r["y"] for r in rows]) == 1
    pred = p > 0.5
    return {
        "n": len(rows),
        "accuracy": float(np.mean(pred == y)),
        "backchannel_ignored": float(np.mean(pred[y])),  # recall on backchannels
        "interrupt_caught": float(np.mean(~pred[~y])),  # recall on interruptions
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tts", default=ROOT / "data/tts", type=Path)
    ap.add_argument("--init", default=ROOT / "models/smart-turn-v3.2-gpu.onnx", type=Path)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    rows = load_tts(args.tts)
    train = [r for r in rows if r["split"] == "train"]
    val = [r for r in rows if r["split"] == "val"]
    test = [r for r in rows if r["split"] == "test"]
    print(f"train {len(train)}  val {len(val)}  test {len(test)}")

    model = HaanModel(load_from_onnx(str(args.init))).to(device)
    model.freeze_turn()
    fe = WhisperFeatureExtractor(chunk_length=WINDOW_S)
    opt = torch.optim.AdamW(model.backchannel.parameters(), lr=args.lr, weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss()

    best, history = -1.0, []
    for epoch in range(args.epochs):
        model.backchannel.train()
        rng.shuffle(train)
        total = 0.0
        for i in range(0, len(train), args.batch):
            chunk = train[i:i + args.batch]
            x = featurize(fe, [make_window(r["audio"], rng) for r in chunk]).to(device)
            y = torch.tensor([r["y"] for r in chunk], device=device)
            with torch.no_grad():
                hidden = model.turn.encoder(input_features=x).last_hidden_state
            loss = lossf(model.backchannel(hidden), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(chunk)
        v = evaluate(model, fe, val, device)
        history.append({"epoch": epoch + 1, "train_loss": total / len(train), **{f"val_{k}": x for k, x in v.items()}})
        print(json.dumps(history[-1]), flush=True)
        if not v["n"] or v["accuracy"] > best:
            best = v["accuracy"]
            torch.save(model.backchannel.state_dict(), args.out / "backchannel_head.pt")

    model.backchannel.load_state_dict(torch.load(args.out / "backchannel_head.pt"))
    results = {ch: evaluate(model, fe, test, device, channel=ch) for ch in ("wideband", "narrowband", "pstn")}
    by = {}
    for key in ("lang", "kind"):
        for val_ in sorted({r[key] for r in test}):
            sub = [r for r in test if r[key] == val_]
            by[f"{key}={val_}"] = evaluate(model, fe, sub, device, channel="pstn")
    out = {"test_by_channel": results, "test_pstn_by_group": by, "history": history,
           "sizes": {"train": len(train), "val": len(val), "test": len(test)}}
    (args.out / "results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out["test_by_channel"], indent=2))


if __name__ == "__main__":
    main()
