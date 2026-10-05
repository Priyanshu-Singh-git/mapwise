"""DeepGlobe land-cover data: RGB masks -> class indices, random crops, flips and rotations."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# DeepGlobe's seven classes, in the order used everywhere downstream.
CLASSES = ["urban", "agriculture", "rangeland", "forest", "water", "barren", "unknown"]
# RGB colours from class_dict.csv (kept here so inference does not need the csv)
COLORS = np.array([
    [0, 255, 255],    # urban        cyan
    [255, 255, 0],    # agriculture  yellow
    [255, 0, 255],    # rangeland    magenta
    [0, 255, 0],      # forest       green
    [0, 0, 255],      # water        blue
    [255, 255, 255],  # barren       white
    [0, 0, 0],        # unknown      black
], dtype=np.uint8)


def load_class_dict(path):
    """Read class_dict.csv if present, so we use the dataset's own colours rather than assume."""
    names, cols = [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            names.append(row["name"].strip())
            cols.append([int(row["r"]), int(row["g"]), int(row["b"])])
    return names, np.array(cols, dtype=np.uint8)


def rgb_to_index(mask_rgb, colors):
    """(H,W,3) uint8 -> (H,W) int64 class indices, by nearest colour match."""
    flat = mask_rgb.reshape(-1, 3).astype(np.int32)   # int32: 255^2*3 overflows int16
    d = ((flat[:, None, :] - colors[None, :, :].astype(np.int32)) ** 2).sum(-1)
    return d.argmin(1).reshape(mask_rgb.shape[:2]).astype(np.int64)


def index_to_rgb(idx, colors):
    return colors[np.clip(idx, 0, len(colors) - 1)]


class DeepGlobeTiles(Dataset):
    """Random crops from the 2448x2448 DeepGlobe tiles."""

    def __init__(self, root, ids, colors, crop=512, train=True):
        self.root, self.ids, self.colors = Path(root), ids, colors
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        sid = self.ids[i]
        img = np.array(Image.open(self.root / f"{sid}_sat.jpg").convert("RGB"))
        msk = np.array(Image.open(self.root / f"{sid}_mask.png").convert("RGB"))

        H, W = img.shape[:2]
        c = min(self.crop, H, W)
        if self.train:
            y = np.random.randint(0, H - c + 1); x = np.random.randint(0, W - c + 1)
        else:
            y, x = (H - c) // 2, (W - c) // 2          # deterministic centre crop for val
        img, msk = img[y:y + c, x:x + c], msk[y:y + c, x:x + c]

        if self.train:
            if np.random.rand() < 0.5:
                img, msk = img[:, ::-1], msk[:, ::-1]
            if np.random.rand() < 0.5:
                img, msk = img[::-1], msk[::-1]
            k = np.random.randint(4)                    # aerial imagery has no canonical "up"
            if k:
                img, msk = np.rot90(img, k), np.rot90(msk, k)

        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        lbl = rgb_to_index(np.ascontiguousarray(msk), self.colors)
        img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        return torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(lbl)


def split_ids(root, val_frac=0.1, seed=0):
    """Validation comes out of train/ because DeepGlobe's test split has no public masks."""
    ids = sorted(p.name[:-8] for p in Path(root).glob("*_sat.jpg"))
    rng = np.random.RandomState(seed); rng.shuffle(ids)
    n = max(1, int(len(ids) * val_frac))
    return ids[n:], ids[:n]
