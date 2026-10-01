"""Record the real-speech test set: you (and willing friends) saying backchannels
and interruptions. This set is never used for training.

Run it in Terminal (macOS asks for microphone access the first time):

    .venv/bin/python scripts/record_clips.py --speaker charan --langs tel hin eng

For each prompt: press Enter, say the phrase the way you would on a call while
someone else is talking, and it stops by itself after the given seconds.
Type r + Enter to redo the last clip, s + Enter to skip, q + Enter to quit.
Re-running continues where you left off.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SR = 16_000

# Includes phrases that are NOT in the TTS lexicon, so the test also measures
# generalisation to unseen wording.
PROMPTS = {
    "eng": {
        "backchannel": ["hmm", "yeah", "okay", "right", "uh-huh", "I see", "got it", "sure sure",
                        "yeah okay", "absolutely", "oh really", "makes sense"],
        "interrupt": ["wait", "stop", "hold on", "actually", "no no", "can I ask something",
                      "hang on a second", "that's wrong", "okay but wait", "yeah but what about the fee",
                      "sorry to interrupt", "let me finish"],
    },
    "hin": {
        "backchannel": ["haan", "haan ji", "achha", "theek hai", "hmm", "ji", "haan haan", "sahi baat hai",
                        "bilkul", "achha achha"],
        "interrupt": ["ruko", "ek minute", "nahin nahin", "suniye", "meri baat suniye", "galat hai",
                      "haan lekin ruko", "ek second", "par mera sawaal hai", "band karo"],
    },
    "tel": {
        "backchannel": ["avunu", "haa", "sare", "okay andi", "hmm", "alaage", "avunandi", "sare sare",
                        "nijamaa", "artham ayyindi"],
        "interrupt": ["aagandi", "okka nimisham", "kaadu kaadu", "vinandi", "aapandi", "malli cheppandi",
                      "sare kaani aagandi", "adi tappu", "naa prashna vinandi", "konchem aagandi"],
    },
}


def record(seconds: float) -> np.ndarray:
    import sounddevice as sd

    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    return audio[:, 0]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="a short name, e.g. charan")
    ap.add_argument("--langs", nargs="+", default=["tel", "hin", "eng"], choices=sorted(PROMPTS))
    ap.add_argument("--takes", type=int, default=1, help="recordings per phrase")
    ap.add_argument("--seconds", type=float, default=2.5)
    ap.add_argument("--out", default=ROOT / "data/real", type=Path)
    args = ap.parse_args()

    wav_dir = args.out / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    meta = args.out / "metadata.csv"
    have = set()
    if meta.exists():
        with meta.open() as f:
            have = {r["id"] for r in csv.DictReader(f)}

    queue = [
        (lang, label, text, t)
        for lang in args.langs
        for label, texts in PROMPTS[lang].items()
        for text in texts
        for t in range(args.takes)
    ]
    random.Random(args.speaker).shuffle(queue)  # mix labels so you don't fall into one tone
    todo = [q for q in queue if f"{args.speaker}-{q[0]}-{q[1][:2]}-{q[2].replace(' ', '_')}-{q[3]}" not in have]
    print(f"{len(todo)} clips to record ({len(queue) - len(todo)} already done).")

    new_file = not meta.exists()
    with meta.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "speaker", "lang", "label", "text", "seconds"])
        if new_file:
            w.writeheader()
        i = 0
        while i < len(todo):
            lang, label, text, take = todo[i]
            cid = f"{args.speaker}-{lang}-{label[:2]}-{text.replace(' ', '_')}-{take}"
            cue = "keep them talking (acknowledge)" if label == "backchannel" else "make them stop (interrupt)"
            cmd = input(f"[{i + 1}/{len(todo)}] {lang}: \"{text}\"  ({cue})  Enter=record s=skip q=quit > ").strip()
            if cmd == "q":
                break
            if cmd == "s":
                i += 1
                continue
            audio = record(args.seconds)
            peak = float(np.abs(audio).max())
            if peak < 0.01:
                print("  too quiet, nothing recorded? trying again")
                continue
            if input("  ok? Enter=keep r=redo > ").strip() == "r":
                continue
            sf.write(wav_dir / f"{cid}.wav", audio, SR)
            w.writerow({"id": cid, "speaker": args.speaker, "lang": lang, "label": label, "text": text,
                        "seconds": args.seconds})
            f.flush()
            i += 1
    print(f"saved to {args.out}")


if __name__ == "__main__":
    sys.exit(main())
