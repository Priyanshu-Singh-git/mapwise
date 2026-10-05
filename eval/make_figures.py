"""Build MapWise's visuals: side-by-side predictions, the hero thumbnail and the IoU chart.

  python eval/make_figures.py --ckpt out/mapwise/best.pt --data <deepglobe root> --n 4
Writes assets/figures/{compare_*.png, hero.png, iou_chart.png} and docs/samples/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapwise.data import index_to_rgb, load_class_dict, rgb_to_index  # noqa: E402
from mapwise.model import UNetResNet34  # noqa: E402

CREAM, INK, ACCENT, MUTED = "#E8E1D2", "#262320", "#E8622A", "#8A8176"
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def predict(net, rgb, size, dev):
    """rgb uint8 (H,W,3) -> predicted class indices (H,W), run at the TRAINING scale.

    The model is trained on 512px crops of the native 2448px tile. Downscaling a whole tile
    to 512 shows it ground texture ~5x smaller than anything it saw in training, and the
    predictions fall apart. So slide a 512 window over the tile at native resolution and
    stitch the logits back together, averaging the overlaps.
    """
    H, W = rgb.shape[:2]
    stride = size // 2                                   # 50% overlap avoids seams at window edges
    acc, cnt = None, None
    ys = list(range(0, max(H - size, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - size, 0) + 1, stride)) or [0]
    if ys[-1] != H - size and H > size:
        ys.append(H - size)
    if xs[-1] != W - size and W > size:
        xs.append(W - size)

    for y in ys:
        for x in xs:
            patch = rgb[y:y + size, x:x + size]
            ph, pw = patch.shape[:2]
            if (ph, pw) != (size, size):                  # pad a short edge, crop the result back
                patch = np.pad(patch, ((0, size - ph), (0, size - pw), (0, 0)), mode="reflect")
            t = (patch.astype(np.float32) / 255.0 - MEAN) / STD
            t = torch.from_numpy(t).permute(2, 0, 1)[None].to(dev)
            with torch.no_grad():
                lg = net(t).softmax(1)[0].cpu().numpy()   # (C,size,size)
            if acc is None:
                acc = np.zeros((lg.shape[0], H, W), np.float32)
                cnt = np.zeros((1, H, W), np.float32)
            acc[:, y:y + ph, x:x + pw] += lg[:, :ph, :pw]
            cnt[:, y:y + ph, x:x + pw] += 1.0
    return (acc / np.maximum(cnt, 1e-6)).argmax(0).astype(np.int64)


def label_strip(w, text, sub="", h=58):
    """Charcoal caption bar, drawn with PIL so the deck palette carries into the figures."""
    from PIL import ImageDraw, ImageFont
    img = Image.new("RGB", (w, h), INK)
    d = ImageDraw.Draw(img)
    try:
        f1 = ImageFont.truetype(str(ROOT / "fonts" / "Oswald-SemiBold.ttf"), 26)
        f2 = ImageFont.truetype(str(ROOT / "fonts" / "Oswald-Light.ttf"), 19)
    except Exception:
        f1 = f2 = ImageFont.load_default()
    d.text((16, h // 2 - 17), text, font=f1, fill=CREAM)
    if sub:
        tw = d.textlength(text, font=f1)
        d.text((16 + tw + 16, h // 2 - 12), sub, font=f2, fill=ACCENT)
    return img


def stack(panels, titles, subs):
    """Horizontal strip of equal-size panels, each with a caption bar."""
    w, h = panels[0].size
    bar = 58
    out = Image.new("RGB", (w * len(panels) + 8 * (len(panels) - 1), h + bar), CREAM)
    for i, (p, t, s) in enumerate(zip(panels, titles, subs)):
        x = i * (w + 8)
        out.paste(p, (x, 0))
        out.paste(label_strip(w, t, s), (x, h))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/mapwise/best.pt")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    meta = ck.get("meta", {})
    data = Path(args.data)
    cd = data / "class_dict.csv"
    names, colors = load_class_dict(cd) if cd.exists() else (meta["classes"], None)

    net = UNetResNet34(len(names), pretrained=False, encoder=meta.get('encoder','resnet34')).to(dev).eval()
    net.load_state_dict(ck["state"])

    FIG = ROOT / "assets" / "figures"; FIG.mkdir(parents=True, exist_ok=True)
    SAMP = ROOT / "docs" / "samples"; SAMP.mkdir(parents=True, exist_ok=True)

    tiles = sorted((data / "train").glob("*_sat.jpg"))[-40:]     # from the tail, likely unseen
    picks, sample_names = [], []
    for p in tiles:
        mp = p.with_name(p.name.replace("_sat.jpg", "_mask.png"))
        if mp.exists():
            picks.append((p, mp))
        if len(picks) >= args.n:
            break

    for i, (sp, mp) in enumerate(picks):
        rgb = np.asarray(Image.open(sp).convert("RGB"))
        gt = rgb_to_index(np.asarray(Image.open(mp).convert("RGB")), colors)
        pred = predict(net, rgb, args.size, dev)

        # pred is now full-resolution; downscale all three panels together for display
        sat = Image.fromarray(rgb).resize((args.size, args.size), Image.BILINEAR)
        gt_s = Image.fromarray(index_to_rgb(gt, colors)).resize((args.size, args.size), Image.NEAREST)
        pr_s = Image.fromarray(index_to_rgb(pred, colors)).resize((args.size, args.size), Image.NEAREST)
        agree = float((gt == pred).mean())

        fig = stack([sat, pr_s, gt_s],
                    ["SATELLITE", "MAPWISE", "GROUND TRUTH"],
                    ["input", f"{agree*100:.0f}% pixel match", "reference"])
        fig.save(FIG / f"compare_{i}.png")
        sat.save(SAMP / f"tile_{i}.jpg", quality=88)
        sample_names.append(f"tile_{i}.jpg")
        if i == 0:                                   # hero: satellite beside prediction
            stack([sat, pr_s], ["SATELLITE", "MAPWISE"], ["input", "every pixel classified"]).save(FIG / "hero.png")
        print(f"compare_{i}: {agree*100:.1f}% pixel match", flush=True)

    (SAMP / "index.json").write_text(json.dumps(sample_names))
    print("wrote", len(picks), "comparisons +", len(sample_names), "demo samples")


if __name__ == "__main__":
    main()
