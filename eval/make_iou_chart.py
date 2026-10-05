"""Per-class IoU chart in the deck palette.

Deliberately per class, not a single mean: DeepGlobe is badly imbalanced and a mean alone
would hide which classes actually work.

  python eval/make_iou_chart.py --ckpt out/mapwise/best.pt
Writes assets/figures/iou_chart.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CREAM, INK, ACCENT, MUTED, LINE = "#E8E1D2", "#262320", "#E8622A", "#8A8176", "#CDC4B2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/mapwise/best.pt")
    ap.add_argument("--log", default="out/mapwise/log.jsonl")
    args = ap.parse_args()

    per, miou = {}, None
    if Path(args.ckpt).is_file() and Path(args.ckpt).stat().st_size > 0:
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        per, miou = ck.get("per_class") or {}, ck.get("mIoU")
    if not per:
        rows = [json.loads(l) for l in open(args.log)]
        best = max(rows, key=lambda r: r["mIoU"])
        per, miou = best["per_class"], best["mIoU"]

    names = list(per.keys())
    vals = [0.0 if per[n] is None else float(per[n]) for n in names]
    pretty = [n.replace("_land", "").replace("_", " ") for n in names]
    order = np.argsort(vals)[::-1]
    names_s = [pretty[i] for i in order]
    vals_s = [vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(13, 6.2), dpi=100)
    fig.patch.set_facecolor(CREAM); ax.set_facecolor(CREAM)
    bars = ax.barh(range(len(vals_s)), vals_s, height=.62,
                   color=[ACCENT if v >= 0.3 else MUTED for v in vals_s])
    ax.set_yticks(range(len(names_s)))
    ax.set_yticklabels([n.upper() for n in names_s], fontsize=15, color=INK, fontweight="bold")
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Intersection over Union", color=MUTED, fontsize=13)
    ax.tick_params(colors=MUTED, labelsize=12)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(LINE)
    ax.grid(axis="x", color=LINE, alpha=.55, linewidth=.9)
    ax.set_axisbelow(True)

    for b, v in zip(bars, vals_s):
        ax.text(min(v + .015, .97), b.get_y() + b.get_height() / 2,
                f"{v*100:.0f}%", va="center", fontsize=14, color=INK, fontweight="bold")

    if miou is not None:
        ax.axvline(miou, color=INK, lw=1.4, ls="--", alpha=.75)
        ax.text(miou + .012, len(vals_s) - 0.35, f"mean {miou*100:.0f}%", color=INK,
                fontsize=12, fontweight="bold", ha="left")

    ax.set_title("PER-CLASS ACCURACY", color=INK, fontsize=22, fontweight="bold", loc="left", pad=22)
    fig.tight_layout()
    out = ROOT / "assets" / "figures" / "iou_chart.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=CREAM, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
