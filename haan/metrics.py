"""Endpointing metrics. Positive class = turn complete (endpoint_bool=True)."""

from __future__ import annotations

import numpy as np


def binary_report(y_true, p_complete, threshold: float = 0.5) -> dict:
    y = np.asarray(y_true, dtype=bool)
    pred = np.asarray(p_complete) > threshold
    tp = int(np.sum(pred & y))
    tn = int(np.sum(~pred & ~y))
    fp = int(np.sum(pred & ~y))  # said "complete" while the speaker was mid-turn: cuts them off
    fn = int(np.sum(~pred & y))  # said "incomplete" on a finished turn: adds latency
    n = len(y)

    def div(a, b):
        return a / b if b else float("nan")

    precision = div(tp, tp + fp)
    recall = div(tp, tp + fn)
    return {
        "n": n,
        "accuracy": div(tp + tn, n),
        "precision": precision,
        "recall": recall,
        "f1": div(2 * precision * recall, precision + recall),
        "fpr": div(fp, fp + tn),
        "fnr": div(fn, fn + tp),
        # the number smart-turn#41 reports: how often an "incomplete" verdict is right
        "incomplete_precision": div(tn, tn + fn),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
