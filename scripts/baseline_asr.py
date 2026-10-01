"""Transcript-based baselines for backchannel vs interruption.

Each clip is transcribed with Whisper large-v3-turbo on Groq, then classified by
  - keywords:  backchannel if every word is a known acknowledgement word
  - min_words: backchannel if fewer than 3 words (Pipecat-style word threshold)
  - vad:       always "interrupt" (what self-hosted LiveKit falls back to)
Wall-clock transcription latency is recorded, since these baselines cannot
decide before the transcript arrives.

    python scripts/baseline_asr.py --clips data/real --out results/asr_real.csv
    python scripts/baseline_asr.py --clips data/tts --split test --out results/asr_tts_test.csv

Needs GROQ_API_KEY (read from the environment or ~/PycharmProjects/skillminer/agents/support_agent/.env).
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.backchannel import split_of  # noqa: E402

URL = "https://api.groq.com/openai/v1/audio/transcriptions"
MODEL = "whisper-large-v3-turbo"
WHISPER_LANG = {"eng": "en", "hin": "hi", "tel": "te"}
ENV_FILE = Path.home() / "PycharmProjects/skillminer/agents/support_agent/.env"

# acknowledgement words, romanised and native script
ACK = {
    "hmm", "hm", "mm", "mhm", "mmhmm", "mm-hmm", "uh-huh", "uhhuh", "yeah", "yes", "yep", "ok", "okay",
    "right", "sure", "alright", "i", "see", "got", "it", "oh", "really", "absolutely", "makes", "sense",
    "haan", "han", "ha", "haa", "ji", "achha", "acha", "theek", "thik", "hai", "sahi", "bilkul",
    "avunu", "avunandi", "sare", "alage", "alaage", "andi", "nijamaa", "artham", "ayyindi",
    "हाँ", "हां", "जी", "अच्छा", "ठीक", "है", "हम्म", "सही", "बिल्कुल", "ओके",
    "అవును", "హా", "సరే", "అలాగే", "అండి", "ఓకే", "హ్మ్", "అవునండి", "అర్థమైంది", "నిజమా",
}


def api_key() -> str:
    if os.environ.get("GROQ_API_KEY"):
        return os.environ["GROQ_API_KEY"]
    for line in ENV_FILE.read_text().splitlines():
        if line.startswith("GROQ_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"')
    sys.exit("GROQ_API_KEY not found")


def transcribe(path: Path, lang: str, key: str, retries: int = 5) -> tuple[str, float]:
    for attempt in range(retries):
        t0 = time.time()
        with path.open("rb") as f:
            r = requests.post(
                URL,
                headers={"Authorization": f"Bearer {key}"},
                files={"file": (path.name, f, "audio/wav")},
                data={"model": MODEL, "language": WHISPER_LANG[lang], "temperature": "0"},
                timeout=60,
            )
        if r.status_code == 200:
            return r.json()["text"].strip(), time.time() - t0
        if r.status_code in (429, 500, 502, 503) and attempt < retries - 1:
            time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
            continue
        r.raise_for_status()
    raise RuntimeError("unreachable")


def words(text: str) -> list[str]:
    return [w for w in re.split(r"[\s,.!?।]+", text.lower()) if w]


def classify(text: str) -> dict:
    ws = words(text)
    return {
        "keywords": "backchannel" if ws and all(w in ACK for w in ws) else "interrupt",
        "min_words": "backchannel" if len(ws) < 3 else "interrupt",
        "vad": "interrupt",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", required=True, type=Path, help="folder with metadata.csv and wav/")
    ap.add_argument("--split", choices=["train", "val", "test"], help="TTS split to keep (TTS data only)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    with (args.clips / "metadata.csv").open() as f:
        rows = list(csv.DictReader(f))
    if args.split:
        rows = [r for r in rows if split_of(r) == args.split]
    done = set()
    if args.out.exists():
        with args.out.open() as f:
            done = {r["id"] for r in csv.DictReader(f)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    key = api_key()
    fields = ["id", "lang", "label", "text", "transcript", "asr_seconds", "keywords", "min_words", "vad"]
    with args.out.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not done:
            w.writeheader()
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            text, secs = transcribe(args.clips / "wav" / f"{r['id']}.wav", r["lang"], key)
            w.writerow({**r, "transcript": text, "asr_seconds": round(secs, 3), **classify(text)})
            f.flush()
            if i % 25 == 0:
                print(f"{i}/{len(rows)} {r['id']}: {text!r} ({secs:.2f}s)", flush=True)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
