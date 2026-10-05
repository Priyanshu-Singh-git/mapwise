"""Pick the most class-diverse tile and render the portfolio thumbnail.

Scores candidate tiles by how many land-cover classes the prediction contains (weighted by
how evenly they are spread), so the thumbnail shows the model doing something interesting
rather than painting one colour over a field.

  python eval/make_thumbnail.py --ckpt <best.pt> --data <deepglobe root>
Writes assets/figures/thumbnail.png (1920x1080).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapwise.data import index_to_rgb, load_class_dict  # noqa: E402
from mapwise.model import UNetResNet34  # noqa: E402

sys.path.insert(0, str(ROOT / "eval"))
from make_figures import predict  # reuse the sliding-window inference  # noqa: E402

PAPER, INK, MUTED, ACCENT, CARD = (251, 249, 245), (35, 35, 35), (154, 148, 138), (242, 208, 75), (244, 241, 234)
W, H = 1920, 1080


def font(name, size):
    for p in [ROOT / "fonts" / name, Path("C:/Windows/Fonts") / name]:
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def diversity(pred, k):
    """Shannon evenness x class count: high when many classes appear in real quantity."""
    c = np.bincount(pred.ravel(), minlength=k).astype(np.float64)
    p = c / c.sum()
    nz = p[p > 0.01]                       # ignore specks
    if len(nz) < 2:
        return 0.0
    ent = -(nz * np.log(nz)).sum()
    return ent * len(nz)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/mapwise/best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--scan", type=int, default=14, help="how many tiles to score")
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    data = Path(args.data)
    cd = data / "class_dict.csv"
    names, colors = load_class_dict(cd)
    net = UNetResNet34(len(names), pretrained=False, encoder=ck.get('meta',{}).get('encoder','resnet34')).to(dev).eval()
    net.load_state_dict(ck["state"])

    tiles = sorted((data / "train").glob("*_sat.jpg"))[-args.scan:]
    best = None
    for p in tiles:
        rgb = np.asarray(Image.open(p).convert("RGB"))
        pred = predict(net, rgb, args.size, dev)
        d = diversity(pred, len(names))
        print(f"{p.name}: diversity {d:.2f}", flush=True)
        if best is None or d > best[0]:
            best = (d, rgb, pred, p.name)
    _, rgb, pred, tile_name = best
    print("chosen:", tile_name, flush=True)

    # ---- compose ----
    canvas = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(canvas)
    d.rectangle([0, 0, W, 14], fill=ACCENT)

    panel = 760
    top = 196
    sat = Image.fromarray(rgb).resize((panel, panel), Image.BILINEAR)
    msk = Image.fromarray(index_to_rgb(pred, colors)).resize((panel, panel), Image.NEAREST)
    lx, rx = 118, W - 118 - panel
    canvas.paste(sat, (lx, top))
    canvas.paste(msk, (rx, top))

    f_word = font("CormorantGaramond.ttf", 46)
    f_cap = font("segoeui.ttf", 20)
    f_small = font("segoeui.ttf", 17)
    f_leg = font("segoeui.ttf", 18)

    word = "M A P W I S E"
    d.text(((W - d.textlength(word, font=f_word)) / 2, 62), word, font=f_word, fill=INK)
    sub = "EVERY PIXEL OF THE EARTH, CLASSIFIED"
    d.text(((W - d.textlength(sub, font=f_cap)) / 2, 128), sub, font=f_cap, fill=MUTED)

    for x, t in ((lx, "SATELLITE"), (rx, "MAPWISE")):
        d.text((x, top + panel + 18), t, font=f_cap, fill=INK)
    d.rectangle([lx, top + panel + 52, lx + 54, top + panel + 56], fill=ACCENT)
    d.rectangle([rx, top + panel + 52, rx + 54, top + panel + 56], fill=ACCENT)

    # legend: the classes actually present, with their share
    cnt = np.bincount(pred.ravel(), minlength=len(names)).astype(np.float64)
    share = cnt / cnt.sum()
    order = np.argsort(share)[::-1]
    present = [i for i in order if share[i] > 0.01]
    y = top + 30
    cx = lx + panel + 56
    for i in present:
        col = tuple(int(v) for v in colors[i])
        d.rectangle([cx, y, cx + 26, y + 26], fill=col, outline=(0, 0, 0))
        label = names[i].replace("_land", "").replace("_", " ")
        d.text((cx + 40, y + 2), label, font=f_leg, fill=INK)
        d.text((cx + 40, y + 26), f"{share[i]*100:.0f}% of ground", font=f_small, fill=MUTED)
        y += 74

    tail = "U-Net - ResNet-34 encoder, decoder from scratch  -  mean IoU 54%, 84% pixel accuracy  -  runs in the browser"
    d.text(((W - d.textlength(tail, font=f_small)) / 2, H - 56), tail, font=f_small, fill=MUTED)

    out = ROOT / "assets" / "figures" / "thumbnail.png"
    canvas.save(out)
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
