"""Loss and metrics for an imbalanced 7-class segmentation problem.

Plain cross-entropy on DeepGlobe produces a model that predicts agriculture almost
everywhere and still posts a flattering pixel accuracy. Dice plus inverse-frequency
weighted CE is the honest fix, and IoU is reported per class rather than as one number.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceCELoss(nn.Module):
    def __init__(self, n_classes=7, class_weights=None, dice_w=0.5, smooth=1.0):
        super().__init__()
        self.n, self.dice_w, self.smooth = n_classes, dice_w, smooth
        self.register_buffer(
            "w", torch.ones(n_classes) if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        )

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, weight=self.w.to(logits.dtype))

        probs = logits.softmax(1)
        oh = F.one_hot(target, self.n).permute(0, 3, 1, 2).to(probs.dtype)
        dims = (0, 2, 3)
        inter = (probs * oh).sum(dims)
        denom = probs.sum(dims) + oh.sum(dims)
        dice = 1.0 - ((2 * inter + self.smooth) / (denom + self.smooth)).mean()
        return (1 - self.dice_w) * ce + self.dice_w * dice


class IoUTracker:
    """Accumulates a confusion matrix so IoU is computed over the whole split, not averaged per batch."""

    def __init__(self, n_classes=7):
        self.n = n_classes
        self.cm = np.zeros((n_classes, n_classes), dtype=np.int64)

    def update(self, pred, target):
        p = pred.detach().cpu().numpy().ravel()
        t = target.detach().cpu().numpy().ravel()
        k = (t >= 0) & (t < self.n)
        self.cm += np.bincount(self.n * t[k] + p[k], minlength=self.n ** 2).reshape(self.n, self.n)

    def iou(self):
        """Per-class IoU; NaN for classes absent from both prediction and ground truth."""
        inter = np.diag(self.cm).astype(np.float64)
        union = self.cm.sum(1) + self.cm.sum(0) - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(union > 0, inter / union, np.nan)

    def mean_iou(self):
        return float(np.nanmean(self.iou()))

    def pixel_acc(self):
        tot = self.cm.sum()
        return float(np.diag(self.cm).sum() / tot) if tot else 0.0


def inverse_freq_weights(counts, power=0.5, clip=8.0):
    """Class weights from pixel counts.

    Uses (1/freq)**power rather than plain inverse frequency: on DeepGlobe the rarest class
    is several thousand times rarer than the commonest, and plain inversion drives the weight
    of the common classes to ~0, so the model stops learning them at all. The square root
    keeps rare classes boosted while leaving common ones meaningful. Weights are normalised
    to mean 1 over present classes, and absent classes get 0.
    """
    c = np.asarray(counts, dtype=np.float64)
    present = c > 0
    freq = np.where(present, c / max(c.sum(), 1.0), 0.0)
    w = np.zeros_like(freq)
    w[present] = (1.0 / freq[present]) ** power
    w[present] /= w[present].mean()
    return np.clip(w, 0.0, clip)
