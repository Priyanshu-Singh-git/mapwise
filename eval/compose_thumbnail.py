"""Compose the portfolio thumbnail locally, where the display serif is installed.

Takes the satellite / prediction panels produced on Kaggle and lays them out in the deck's
minimal style: larger panels, a legend that sits under the prediction rather than squeezed
between the two, and the wordmark at a readable size.

  python eval/compose_thumbnail.py
Writes assets/figures/thumbnail.png (1920x1080).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "assets" / "figures"

PAPER = (251, 249, 245)
INK = (35, 35, 35)
INK2 = (74, 71, 66)
MUTED = (154, 148, 138)
ACCENT = (242, 208, 75)
W, H = 1920, 1080


def font(name, size):
    for p in (ROOT / "fonts" / name, Path("C:/Windows/Fonts") / name):
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size)
            except Exception:
                pass
    return ImageFont.load_default()


def main():
    sat = Image.open(ROOT / "out" / "_sat.png").convert("RGB")
    msk = Image.open(ROOT / "out" / "_msk.png").convert("RGB")
    meta = json.loads((ROOT / "docs" / "models" / "mapwise_meta.json").read_text())
    names, colors = meta["classes"], np.array(meta["colors"], np.uint8)

    # recover the class shares straight from the mask panel's pixels
    flat = np.asarray(msk).reshape(-1, 3).astype(np.int32)
    d = ((flat[:, None, :] - colors[None, :, :].astype(np.int32)) ** 2).sum(-1)
    idx = d.argmin(1)
    cnt = np.bincount(idx, minlength=len(names)).astype(np.float64)
    share = cnt / cnt.sum()

    canvas = Image.new("RGB", (W, H), PAPER)
    dr = ImageDraw.Draw(canvas)
    dr.rectangle([0, 0, W, 13], fill=ACCENT)

    f_word = font("CormorantGaramond.ttf", 54)
    f_sub = font("segoeui.ttf", 21)
    f_cap = font("segoeui.ttf", 20)
    f_leg = font("segoeui.ttf", 19)
    f_legs = font("segoeui.ttf", 16)
    f_foot = font("segoeui.ttf", 17)

    # --- header ---
    word = "M A P W I S E"
    dr.text(((W - dr.textlength(word, font=f_word)) / 2, 54), word, font=f_word, fill=INK)
    sub = "EVERY PIXEL OF THE EARTH, CLASSIFIED"
    dr.text(((W - dr.textlength(sub, font=f_sub)) / 2, 126), sub, font=f_sub, fill=MUTED)
    dr.rectangle([W // 2 - 34, 168, W // 2 + 34, 172], fill=ACCENT)

    # --- panels: bigger, tighter together, vertically centred in what remains ---
    P = 648
    gap = 46
    total = P * 2 + gap
    lx = (W - total) // 2
    rx = lx + P + gap
    top = 212
    canvas.paste(sat.resize((P, P), Image.BILINEAR), (lx, top))
    canvas.paste(msk.resize((P, P), Image.NEAREST), (rx, top))

    for x, t in ((lx, "SATELLITE"), (rx, "MAPWISE")):
        dr.rectangle([x, top + P + 16, x + 46, top + P + 20], fill=ACCENT)
        dr.text((x, top + P + 32), t, font=f_cap, fill=INK)

    # --- legend: a horizontal row beneath both panels, so nothing is cramped ---
    present = [i for i in np.argsort(share)[::-1] if share[i] > 0.005]
    y = top + P + 74
    x = lx
    for i in present:
        col = tuple(int(v) for v in colors[i])
        dr.rectangle([x, y + 4, x + 22, y + 26], fill=col, outline=(200, 196, 188))
        label = names[i].replace("_land", "").replace("_", " ")
        dr.text((x + 34, y), label, font=f_leg, fill=INK)
        dr.text((x + 34, y + 24), f"{share[i] * 100:.0f}% of ground", font=f_legs, fill=MUTED)
        x += int(dr.textlength(label, font=f_leg)) + 150

    foot = ("U-Net  \u00b7  ResNet-34 encoder, decoder written from scratch  \u00b7  "
            "mean IoU 54%, 84% pixel accuracy  \u00b7  runs in the browser")
    dr.text(((W - dr.textlength(foot, font=f_foot)) / 2, H - 48), foot, font=f_foot, fill=MUTED)

    out = FIG / "thumbnail.png"
    canvas.save(out)
    print("wrote", out, canvas.size, "| classes shown:", [names[i] for i in present])


if __name__ == "__main__":
    main()
