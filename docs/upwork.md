# MapWise — Upwork portfolio entry

## Title (≤70 chars)
```
MapWise - Land Cover Segmentation from Satellite Imagery
```
*(56 chars — verified)*

## Role (≤100 chars)
```
AI / ML Engineer - Semantic Segmentation (U-Net from scratch), PyTorch, ONNX, Computer Vision
```
*(93 chars — verified)*

## Description (≤600 chars)
```
A U-Net that labels every pixel of a satellite image as urban, agriculture, forest, water, rangeland or barren, and reports how much ground each type covers.

ResNet-50 encoder with a decoder written from scratch, trained on DeepGlobe. Mean IoU 0.69, 85.9% pixel accuracy: agriculture 0.87, water 0.83, forest 0.79. Dice + inverse-frequency weighted loss is what makes the rare classes learnable - plain cross-entropy ignores them.

Exported to ONNX and quantised to 33 MB so it runs in the browser: no server, no upload. Built for crop monitoring, urban growth tracking and environmental risk.
```
*(594 chars — verified)*

---

## Thumbnail
`assets/figures/thumbnail.png` — a complex tile beside its colour-coded prediction, with the ground-cover legend. The
before/after read is what makes this legible at gallery size.

## Gallery order
1. `hero.png` — satellite vs. classified map (the hook)
2. `mapwise_deck_slides/slide_04.png` — predictions against ground truth
3. `slide_05.png` — per-class IoU, including what the model does not do well
4. `slide_03.png` — how the U-Net works
5. `slide_07.png` — who buys it (agriculture, urban planning, environment & risk)
6. `slide_08.png` — live demo QR

## Live demo
<https://priyanshu-singh-git.github.io/mapwise/>

## Skills to tag
Semantic Segmentation · Computer Vision · PyTorch · Deep Learning · U-Net · ONNX ·
Remote Sensing · GIS · Python · Image Processing
