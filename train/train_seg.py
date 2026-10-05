"""Train MapWise: U-Net (ResNet-34 encoder) on DeepGlobe land cover.

  python train/train_seg.py --data /kaggle/input/deepglobe-land-cover-classification-dataset \
      --epochs 24 --batch 8 --crop 512 --out out/mapwise
Logs per-class IoU every epoch; checkpoints on best validation mean IoU.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapwise.data import CLASSES, COLORS, DeepGlobeTiles, load_class_dict, split_ids  # noqa: E402
from mapwise.losses import DiceCELoss, IoUTracker, inverse_freq_weights  # noqa: E402
from mapwise.model import UNetResNet34  # noqa: E402


def class_pixel_counts(ds, n_classes, sample=60):
    """Sample a few masks to estimate class frequencies for the loss weights."""
    counts = np.zeros(n_classes, dtype=np.int64)
    idx = np.linspace(0, len(ds) - 1, min(sample, len(ds))).astype(int)
    for i in idx:
        _, lbl = ds[int(i)]
        counts += np.bincount(lbl.numpy().ravel(), minlength=n_classes)
    return counts


def evaluate(net, loader, dev, n_classes):
    net.eval()
    tr = IoUTracker(n_classes)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                logits = net(x)
            tr.update(logits.argmax(1).cpu(), y)
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dataset root containing train/ and class_dict.csv")
    ap.add_argument("--out", default="out/mapwise")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap training tiles")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Path(args.data)
    train_dir = data / "train"

    cd = data / "class_dict.csv"
    if cd.exists():
        names, colors = load_class_dict(cd)
    else:
        names, colors = CLASSES, COLORS
    n_classes = len(names)

    tr_ids, va_ids = split_ids(train_dir, args.val_frac)
    if args.limit:
        tr_ids = tr_ids[:args.limit]; va_ids = va_ids[:max(1, args.limit // 8)]
    ds_tr = DeepGlobeTiles(train_dir, tr_ids, colors, args.crop, train=True)
    ds_va = DeepGlobeTiles(train_dir, va_ids, colors, args.crop, train=False)
    dl_tr = DataLoader(ds_tr, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                       pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    dl_va = DataLoader(ds_va, batch_size=args.batch, shuffle=False, num_workers=args.workers,
                       pin_memory=True, persistent_workers=args.workers > 0)

    counts = class_pixel_counts(ds_va, n_classes)
    weights = inverse_freq_weights(counts)
    meta = {"classes": names, "n_train": len(tr_ids), "n_val": len(va_ids), "crop": args.crop,
            "pixel_counts": counts.tolist(), "class_weights": [round(float(w), 3) for w in weights],
            "device": str(dev)}
    print(json.dumps(meta), flush=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    net = UNetResNet34(n_classes).to(dev)
    crit = DiceCELoss(n_classes, weights).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=dev.type == "cuda")

    best = -1.0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        net.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x = x.to(dev, non_blocking=True); y = y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                loss = crit(net(x), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            tot += float(loss); nb += 1
        sched.step()

        tr = evaluate(net, dl_va, dev, n_classes)
        ious = tr.iou()
        row = {"epoch": ep, "loss": round(tot / max(nb, 1), 4), "mIoU": round(tr.mean_iou(), 4),
               "pixacc": round(tr.pixel_acc(), 4), "minutes": round((time.time() - t0) / 60, 1),
               "per_class": {n: (None if np.isnan(v) else round(float(v), 4)) for n, v in zip(names, ious)}}
        print(json.dumps(row), flush=True)
        with open(out / "log.jsonl", "a") as f:
            f.write(json.dumps(row) + "\n")

        torch.save({"state": net.state_dict(), "meta": meta}, out / "last.pt")
        if row["mIoU"] > best:
            best = row["mIoU"]
            torch.save({"state": net.state_dict(), "meta": meta, "mIoU": best,
                        "per_class": row["per_class"]}, out / "best.pt")
    print(json.dumps({"done": True, "best_mIoU": round(best, 4)}), flush=True)


if __name__ == "__main__":
    main()
