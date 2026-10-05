"""Push the MapWise training job to Kaggle (T4, DeepGlobe attached as a dataset source).

  python deploy/kaggle_train.py probe    # 3-min environment + data-layout check
  python deploy/kaggle_train.py push     # the real training run
  python deploy/kaggle_train.py status
  python deploy/kaggle_train.py pull
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KAGGLE = r"F:\tools\uv\bin\kaggle.exe"
USER = "priyanshusingh308"
DATASET = "balraj98/deepglobe-land-cover-classification-dataset"
SRC = ["src/mapwise/model.py", "src/mapwise/data.py", "src/mapwise/losses.py", "train/train_seg.py"]

PROBE = r'''
import sys, glob, os, json
print("PYTHON:", sys.version.split()[0], flush=True)
import torch, torchvision
print("TORCH:", torch.__version__, "TV:", torchvision.__version__, "CUDA:", torch.cuda.is_available(), flush=True)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB", flush=True)

# What actually got mounted? Do not assume the slug is the directory name.
base = "/kaggle/input"
print("INPUT EXISTS:", os.path.isdir(base), flush=True)
mounts = sorted(os.listdir(base)) if os.path.isdir(base) else []
print("MOUNTS:", mounts, flush=True)

D = None
for m in mounts:
    cand = os.path.join(base, m)
    if glob.glob(cand + "/**/*_sat.jpg", recursive=True):
        D = cand; break
print("DATA ROOT:", D, flush=True)
if D:
    sats = glob.glob(D + "/**/*_sat.jpg", recursive=True)
    msks = glob.glob(D + "/**/*_mask.png", recursive=True)
    print("sat:", len(sats), "mask:", len(msks), flush=True)
    print("SAMPLE SAT:", sats[0] if sats else None, flush=True)
    cd = glob.glob(D + "/**/class_dict.csv", recursive=True)
    if cd:
        print("CLASS_DICT PATH:", cd[0], flush=True)
        print("CLASS_DICT:", open(cd[0]).read()[:300], flush=True)
    if sats:
        from PIL import Image
        import numpy as np
        print("TILE SIZE:", Image.open(sats[0]).size, flush=True)
        mp = sats[0].replace("_sat.jpg", "_mask.png")
        if os.path.exists(mp):
            mk = np.array(Image.open(mp).convert("RGB"))
            cols, cnt = np.unique(mk.reshape(-1,3), axis=0, return_counts=True)
            print("MASK COLOURS:", [(c.tolist(), int(n)) for c, n in zip(cols, cnt)][:10], flush=True)
print("PROBE DONE", flush=True)
'''

RUN = r'''import json, os, subprocess, sys
from pathlib import Path
W = Path("/kaggle/working/mw"); W.mkdir(parents=True, exist_ok=True)
FILES = json.loads(%FILES%)
for rel, txt in FILES.items():
    p = W / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(txt, encoding="utf-8")
os.chdir(str(W))
import torch, glob; print("CUDA:", torch.cuda.is_available(), flush=True)
# the dataset mounts nested (/kaggle/input/datasets/<owner>/<slug>), so discover it
cands = glob.glob("/kaggle/input/**/train/*_sat.jpg", recursive=True)
assert cands, "no train/*_sat.jpg found under /kaggle/input"
D = os.path.dirname(os.path.dirname(cands[0]))
print("DATA ROOT:", D, "train tiles:", len(cands), flush=True)
r = subprocess.run(f"python train/train_seg.py --data {D} --out /kaggle/working/out "
                   "--epochs %EPOCHS% --batch %BATCH% --crop %CROP% --workers 2 %EXTRA%", shell=True)
print("<<< train exit", r.returncode, flush=True)
import glob; print("outputs:", glob.glob("/kaggle/working/out/*"), flush=True)
'''


def kaggle(*a, check=True):
    r = subprocess.run([KAGGLE, *a], capture_output=True, text=True)
    out = r.stdout + r.stderr
    if check and r.returncode:
        raise SystemExit(out)
    return out


def push(name, code, gpu=True):
    d = ROOT / "kaggle_jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "run.py").write_text(code, encoding="utf-8")
    (d / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{USER}/{name}", "title": name, "code_file": "run.py", "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": gpu, "enable_internet": True,
        "dataset_sources": [DATASET], "competition_sources": [], "kernel_sources": []}, indent=2))
    print(kaggle("kernels", "push", "-p", str(d)))


def main():
    op = sys.argv[1] if len(sys.argv) > 1 else "status"
    if op == "probe":
        push("mapwise-probe", PROBE, gpu=True)
    elif op == "push":
        files = {f: (ROOT / f).read_text(encoding="utf-8") for f in SRC}
        # flatten src/mapwise/* so the training script's sys.path trick still resolves
        code = (RUN.replace("%FILES%", repr(json.dumps(files)))
                   .replace("%EPOCHS%", "24").replace("%BATCH%", "8").replace("%CROP%", "512")
                   .replace("%EXTRA%", ""))
        push("mapwise-train", code, gpu=True)
    elif op == "push-v2":
        files = {f: (ROOT / f).read_text(encoding="utf-8") for f in SRC}
        # bigger encoder + higher-resolution crops + longer schedule + TTA at validation.
        # batch drops to 4: resnet50 at 768px needs roughly 4x the activation memory of
        # resnet34 at 512px, and the T4 has 15.6 GB.
        code = (RUN.replace("%FILES%", repr(json.dumps(files)))
                   .replace("%EPOCHS%", "60").replace("%BATCH%", "4").replace("%CROP%", "768")
                   .replace("%EXTRA%", "--encoder resnet50 --tta --lr 2e-4"))
        push("mapwise-train-v2", code, gpu=True)
    elif op == "sanity":
        files = {f: (ROOT / f).read_text(encoding="utf-8") for f in SRC}
        code = (RUN.replace("%FILES%", repr(json.dumps(files)))
                   .replace("%EPOCHS%", "2").replace("%BATCH%", "8").replace("%CROP%", "384")
                   .replace("--workers 2", "--workers 2 --limit 120"))
        push("mapwise-sanity", code, gpu=True)
    elif op in ("status", "probe-status"):
        name = {"probe-status": "mapwise-probe"}.get(op, sys.argv[2] if len(sys.argv) > 2 else "mapwise-train")
        print(kaggle("kernels", "status", f"{USER}/{name}", check=False))
    elif op in ("pull", "probe-log"):
        name = {"probe-log": "mapwise-probe"}.get(op, sys.argv[2] if len(sys.argv) > 2 else "mapwise-train")
        o = ROOT / "kaggle_runs" / name; o.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{USER}/{name}", "-p", str(o), check=False))
        for f in sorted(o.glob("*.log")):
            print(f.read_text(errors="replace")[-4000:])


if __name__ == "__main__":
    main()
