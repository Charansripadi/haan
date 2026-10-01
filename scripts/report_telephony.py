"""Summarise results/telephony_scores into results/telephony_report.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.metrics import binary_report  # noqa: E402
from haan.telephony import CHANNELS  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
COLS = ["n", "accuracy", "f1", "fpr", "fnr", "incomplete_precision"]


def bootstrap_drop(df: pd.DataFrame, ch: str, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    """95% CI of accuracy(wideband) - accuracy(ch), resampling clips (paired)."""
    rng = np.random.default_rng(seed)
    y = df.endpoint_bool.to_numpy()
    a = (df.p_wideband.to_numpy() > 0.5) == y
    b = (df[f"p_{ch}"].to_numpy() > 0.5) == y
    idx = rng.integers(0, len(y), size=(reps, len(y)))
    diffs = a[idx].mean(1) - b[idx].mean(1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def table(df: pd.DataFrame, by: str | None) -> pd.DataFrame:
    out = []
    groups = [("all", df)] if by is None else df.groupby(by)
    for key, g in groups:
        for ch in CHANNELS:
            r = binary_report(g.endpoint_bool, g[f"p_{ch}"])
            out.append({"group": key, "channel": ch, **{c: r[c] for c in COLS}})
    return pd.DataFrame(out)


def fmt(t: pd.DataFrame) -> str:
    t = t.copy()
    for c in COLS[1:]:
        t[c] = (100 * t[c]).round(2)
    return t.to_markdown(index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", default=ROOT / "results/telephony_scores", type=Path)
    ap.add_argument("--out", default=ROOT / "results/telephony_report.md", type=Path)
    ap.add_argument("--title", default="Smart Turn v3.2 (CPU INT8 ONNX)", help="which model/arm was scored")
    args = ap.parse_args()

    df = pd.concat([pd.read_parquet(p) for p in sorted(args.scores.glob("*.parquet"))], ignore_index=True)
    lines = [
        f"# {args.title} on telephone audio",
        "",
        f"Clips: {len(df):,}. Threshold 0.5. Metrics in %.",
        "Positive class = turn complete. FPR = cut the speaker off; FNR = waited on a finished turn.",
        "",
        "## Overall",
        "",
        fmt(table(df, None)),
        "",
    ]
    for ch in CHANNELS[1:]:
        lo, hi = bootstrap_drop(df, ch)
        lines.append(f"- Accuracy drop wideband → {ch}: 95% CI [{100 * lo:.2f}, {100 * hi:.2f}] points (paired bootstrap)")
    lines += ["", "## By language", "", fmt(table(df, "language")), "", "## By dataset", "", fmt(table(df, "dataset")), ""]
    args.out.write_text("\n".join(lines))
    print("\n".join(lines[:16]))


if __name__ == "__main__":
    main()
