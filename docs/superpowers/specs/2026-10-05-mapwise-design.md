# MapWise — design

Day 05 of the 18-day sprint. A semantic segmentation model that labels every pixel of a
satellite image with its land-cover type, plus an in-browser demo and a portfolio deck.

## Intent

Show real architecture work in supervised computer vision: a U-Net decoder written from
scratch on a pretrained encoder, trained on a genuinely imbalanced 7-class dataset, with
honest per-class metrics. The commercial output is not the mask itself but the **per-class
area breakdown** — "this tile is 34% agriculture, 12% forest" — which is what crop
monitoring, urban-sprawl tracking and deforestation alerting actually consume.

Deliberately contrasts with Day 04 (DriftMind): supervised rather than reinforcement,
GPU-bound rather than CPU-bound, no shared code.

## Data

`balraj98/deepglobe-land-cover-classification-dataset` on Kaggle (2.9 GB, verified present).

- `train/*_sat.jpg` + `train/*_mask.png`, 2448x2448 RGB tiles.
- `class_dict.csv` maps 7 class names to RGB mask colours.
- `test/` has no public masks, so the validation split is carved out of `train/`.

The dataset stays on Kaggle and is attached to the training job as a dataset source.
Nothing of that size is downloaded locally; only the trained weights (~30 MB in fp32,
smaller once quantised) and a handful of sample tiles come back.

### Classes

urban, agriculture, rangeland, forest, water, barren, unknown.

Badly imbalanced: agriculture dominates, water and barren are rare, unknown is nearly
absent. This is the central technical problem, not an incidental detail.

## Model

U-Net, encoder/decoder split:

- **Encoder** — torchvision ResNet-34, ImageNet-pretrained, used as a feature pyramid
  (5 stages, stride 2 to 32).
- **Decoder** — written from scratch: four upsampling blocks, each taking the skip
  connection from the matching encoder stage, `Upsample -> concat -> 2x (Conv-BN-ReLU)`.
  A 1x1 convolution produces 7 logit channels at full resolution.

Using a pretrained encoder is a deliberate choice: the portfolio claim is architecture and
training craft, not pretraining from zero, and ImageNet features transfer well to aerial RGB.

## Loss and metrics

- **Loss** = Dice + class-weighted cross-entropy. Plain CE on this dataset yields a model
  that predicts agriculture nearly everywhere and still reports a flattering pixel accuracy.
  Weights are inverse-frequency, computed from a sample of training masks.
- **Metric** = IoU per class, plus the mean. Per-class IoU is reported in the README and on
  the deck; a single mean would conceal failure on the rare classes.
- Pixel accuracy is recorded but never used as the headline, for the same reason.

## Training

- Random 512x512 crops from the 2448x2448 tiles; horizontal/vertical flips and 90-degree
  rotations. Aerial imagery has no canonical "up", so rotation is a safe augmentation here.
- AdamW, cosine schedule, mixed precision.
- Kaggle T4. Target ~1.5-2 h.
- Checkpoint on best validation mean IoU; log per-class IoU every epoch.

## Demo

ONNX export, then onnxruntime-web on GitHub Pages, matching the pattern established on
Days 02 and 03 (`docs/` holds the page, `docs/models/` the weights).

The page takes a dropped or sampled tile and returns the colour-coded mask overlaid at
adjustable opacity, plus the per-class area table. Inference runs in the visitor's browser;
no server, no upload.

## Deliverables

- Trained model + per-class IoU table.
- `docs/` browser demo on GitHub Pages.
- 8-slide deck in the Oswald "Project Proposal" template, exported to PDF and 1920x1080 PNGs.
- Thumbnail: satellite tile beside its colour-coded mask — the strongest visual this project has.
- README with real numbers, and Upwork title/role/description within their character limits.
- Private GitHub repository.

## Risks

- **Class imbalance** — water, barren and unknown may end with weak IoU. Reported honestly
  per class rather than hidden behind the mean.
- **Kaggle environment** — Day 04 lost time to a package that had no wheel for Kaggle's
  Python. Here the dependencies are torch and torchvision, both preinstalled, so the risk is
  low; a short probe job confirms the environment before the real run is queued.
- **4 GB local VRAM** — sufficient for ONNX export and inference checks, not for training.
  Training happens on the T4.
