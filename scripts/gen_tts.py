"""Generate backchannel / interruption clips with Indic Parler-TTS (Apache-2.0).

Runs on a Colab GPU in its own runtime (parler-tts pins an older transformers).
Writes 16 kHz mono wavs plus metadata.csv. Resumable: existing wavs are skipped.

    python scripts/gen_tts.py --out /content/drive/MyDrive/haan/tts
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly

# parler-tts pins an older transformers, which would import Colab's TensorFlow
# (and crash on the protobuf it downgrades). TTS only needs PyTorch.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.lexicon import tts_jobs  # noqa: E402

MODEL = "ai4bharat/indic-parler-tts"
SR = 16_000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained(MODEL).to(device)
    tok = AutoTokenizer.from_pretrained(MODEL)
    desc_tok = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
    src_sr = model.config.sampling_rate

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    jobs = tts_jobs()[: args.limit or None]
    meta_path = args.out / "metadata.csv"
    fields = list(jobs[0]) + ["seconds"]
    done = set()
    if meta_path.exists():
        with meta_path.open() as f:
            done = {r["id"] for r in csv.DictReader(f)}
    with meta_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not done:
            w.writeheader()
        for i, job in enumerate(jobs):
            if job["id"] in done:
                continue
            torch.manual_seed(zlib.crc32(job["id"].encode()))
            d = desc_tok(job["description"], return_tensors="pt").to(device)
            p = tok(job["text"], return_tensors="pt").to(device)
            with torch.no_grad():
                audio = model.generate(
                    input_ids=d.input_ids, attention_mask=d.attention_mask,
                    prompt_input_ids=p.input_ids, prompt_attention_mask=p.attention_mask,
                ).cpu().numpy().squeeze()
            audio = resample_poly(audio.astype(np.float64), SR, src_sr).astype(np.float32)
            peak = np.abs(audio).max() or 1.0
            audio = 0.9 * audio / peak
            sf.write(wav_dir / f"{job['id']}.wav", audio, SR)
            w.writerow({**job, "seconds": round(len(audio) / SR, 3)})
            f.flush()
            if i % 50 == 0:
                print(f"{i}/{len(jobs)} {job['id']} {len(audio) / SR:.2f}s", flush=True)
    print("done")


if __name__ == "__main__":
    main()
