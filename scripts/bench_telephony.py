"""Score Smart Turn v3.2 on its own test set under each telephone channel.

Writes one row per clip with P(complete) per channel, so the report can be
recomputed (thresholds, per-language cuts) without re-running the model.
Resumable: shards that already have an output file are skipped.

    python scripts/bench_telephony.py --every 1          # full 31.5k clips
    python scripts/bench_telephony.py --every 10         # 1-in-10 sample, quick
"""

from __future__ import annotations

import argparse
import io
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.smartturn import SR, SmartTurn, TorchSmartTurn  # noqa: E402
from haan.telephony import CHANNELS, simulate  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def decode(audio_cell: dict) -> np.ndarray:
    wav, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != SR:
        wav = resample_poly(wav, SR, sr).astype(np.float32)
    return wav


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=ROOT / "data/st-v3.2-test/data", type=Path)
    ap.add_argument("--model", default=ROOT / "models/smart-turn-v3.2-cpu.onnx", type=Path)
    ap.add_argument("--out", default=ROOT / "results/telephony_scores", type=Path)
    ap.add_argument("--every", type=int, default=1, help="keep every k-th clip")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--torch-weights", type=Path, help="score a fine-tuned model.pt instead of the ONNX file")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = TorchSmartTurn(args.torch_weights) if args.torch_weights else SmartTurn(args.model, threads=4)
    shards = sorted(args.data.glob("*.parquet"))
    if not shards:
        sys.exit(f"no parquet shards in {args.data}")

    for shard in shards:
        dest = args.out / f"{shard.stem}.every{args.every}.parquet"
        if dest.exists():
            print(f"skip {shard.name} (done)")
            continue
        t0 = time.time()
        rows = []
        pf = pq.ParquetFile(shard)
        cols = ["audio", "id", "language", "endpoint_bool", "midfiller", "endfiller", "synthetic", "dataset"]
        seen = 0
        for rb in pf.iter_batches(batch_size=args.batch, columns=cols):
            recs = rb.to_pylist()
            keep = [r for i, r in enumerate(recs, start=seen) if i % args.every == 0]
            seen += len(recs)
            if not keep:
                continue
            audios = [decode(r["audio"]) for r in keep]
            probs = {ch: model.predict_proba([simulate(a, ch) for a in audios]) for ch in CHANNELS}
            for j, r in enumerate(keep):
                row = {k: r[k] for k in cols if k != "audio"}
                row["seconds"] = len(audios[j]) / SR
                for ch in CHANNELS:
                    row[f"p_{ch}"] = float(probs[ch][j])
                rows.append(row)
        pd.DataFrame(rows).to_parquet(dest)
        print(f"{shard.name}: {len(rows)} clips in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
