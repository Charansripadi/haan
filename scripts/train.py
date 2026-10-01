"""Fine-tune Smart Turn v3.2 (weights recovered from ONNX) with or without
telephone-channel augmentation. Both arms use the same init, shards, seed,
steps and learning rate; only the channel mix differs.

    python scripts/train.py --arm telephony --shards 12 --out runs/telephony
    python scripts/train.py --arm control   --shards 12 --out runs/control

Smoke test on CPU:  python scripts/train.py --arm telephony --shards 1 --max-steps 3 --batch 4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from haan.convert import load_from_onnx  # noqa: E402
from haan.data import ARMS, ShardStream  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_REPO = "pipecat-ai/smart-turn-data-v3.2-train"
CLIPS_PER_SHARD = 3265  # 270,946 clips / 83 shards


def fetch_shards(n: int, local_dir: Path) -> list[Path]:
    files = sorted(f for f in HfApi().list_repo_files(TRAIN_REPO, repo_type="dataset") if f.endswith(".parquet"))
    # spread the picks over the whole set rather than taking the first n
    picks = [files[round(i * (len(files) - 1) / max(n - 1, 1))] for i in range(n)] if n > 1 else files[:1]
    return [Path(hf_hub_download(TRAIN_REPO, f, repo_type="dataset", local_dir=local_dir)) for f in picks]


def loss_fn(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    # same class balancing as upstream: weight positives by the batch ratio
    pos_weight = ((labels == 0).sum() / (labels == 1).sum().clamp(min=1)).clamp(0.1, 10.0)
    return nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--init", default=ROOT / "models/smart-turn-v3.2-gpu.onnx", type=Path)
    ap.add_argument("--shards", type=int, default=12)
    ap.add_argument("--data-dir", default=ROOT / "data/st-v3.2-train", type=Path)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    shards = fetch_shards(args.shards, args.data_dir)
    steps = args.max_steps or math.ceil(args.epochs * len(shards) * CLIPS_PER_SHARD / args.batch)
    model = load_from_onnx(str(args.init)).to(device).train()

    stream = ShardStream(shards, args.arm, seed=args.seed)
    loader = DataLoader(stream, batch_size=args.batch, num_workers=args.workers)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warm = max(1, int(args.warmup * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))
    )
    scaler = torch.amp.GradScaler(enabled=device == "cuda")

    log, step, t0 = [], 0, time.time()
    while step < steps:
        for feats, labels in loader:
            feats, labels = feats.to(device, non_blocking=True), labels.to(device)
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                logits = model(feats)
            loss = loss_fn(logits.float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 50 == 0 or step == steps:
                rec = {"step": step, "loss": round(loss.item(), 4), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0)}
                log.append(rec)
                print(json.dumps(rec), flush=True)
            if step >= steps:
                break
        stream.epoch += 1

    model.eval().cpu()
    torch.save(model.state_dict(), args.out / "model.pt")
    meta = {**{k: str(v) for k, v in vars(args).items()}, "steps": steps, "shards_used": [s.name for s in shards]}
    (args.out / "train_meta.json").write_text(json.dumps(meta, indent=2))
    (args.out / "train_log.json").write_text(json.dumps(log))
    print(f"saved {args.out / 'model.pt'}")


if __name__ == "__main__":
    main()
