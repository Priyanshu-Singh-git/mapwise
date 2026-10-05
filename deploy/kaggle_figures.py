"""Render MapWise prediction figures on Kaggle, where the dataset lives.

The 2.9 GB dataset never comes down to the laptop, so the comparison strips are rendered
on Kaggle using the trained checkpoint (mounted from the training kernel's output) and only
the finished PNGs are pulled back.

  python deploy/kaggle_figures.py push
  python deploy/kaggle_figures.py status
  python deploy/kaggle_figures.py pull
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
NAME = "mapwise-figures"
SRC = ["src/mapwise/model.py", "src/mapwise/data.py", "eval/make_figures.py", "eval/make_thumbnail.py"]

RUN = r'''
import json, os, glob, shutil, subprocess
from pathlib import Path

W = Path("/kaggle/working/mw"); W.mkdir(parents=True, exist_ok=True)
for rel, txt in json.loads(%FILES%).items():
    p = W / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_text(txt, encoding="utf-8")
# the fonts the figure captions use are optional; fall back to PIL's default if absent
os.chdir(str(W))

ck = glob.glob("/kaggle/input/**/best.pt", recursive=True)
print("CKPT:", ck[:2], flush=True)
assert ck, "no best.pt found under /kaggle/input"

sat = glob.glob("/kaggle/input/**/train/*_sat.jpg", recursive=True)
assert sat, "no train tiles found"
D = os.path.dirname(os.path.dirname(sat[0]))
print("DATA:", D, "tiles:", len(sat), flush=True)

r = subprocess.run("python eval/make_figures.py --ckpt " + ck[0] + " --data " + D + " --n 4", shell=True)
print("<<< figures exit", r.returncode, flush=True)
r2 = subprocess.run("python eval/make_thumbnail.py --ckpt " + ck[0] + " --data " + D + " --scan 14", shell=True)
print("<<< thumbnail exit", r2.returncode, flush=True)

out = Path("/kaggle/working/figs"); out.mkdir(exist_ok=True)
for f in glob.glob(str(W / "assets/figures/*.png")) + glob.glob(str(W / "docs/samples/*")):
    shutil.copy(f, out / os.path.basename(f))
print("OUT:", sorted(os.listdir(out)), flush=True)
'''


def kaggle(*a, check=True):
    r = subprocess.run([KAGGLE, *a], capture_output=True, text=True)
    o = r.stdout + r.stderr
    if check and r.returncode:
        raise SystemExit(o)
    return o


def main():
    op = sys.argv[1] if len(sys.argv) > 1 else "status"
    d = ROOT / "kaggle_jobs" / NAME
    if op == "push":
        d.mkdir(parents=True, exist_ok=True)
        files = {f: (ROOT / f).read_text(encoding="utf-8") for f in SRC}
        (d / "run.py").write_text(RUN.replace("%FILES%", repr(json.dumps(files))), encoding="utf-8")
        (d / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/{NAME}", "title": NAME, "code_file": "run.py", "language": "python",
            "kernel_type": "script", "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [DATASET], "competition_sources": [],
            "kernel_sources": [f"{USER}/mapwise-train-v2"]}, indent=2))
        print(kaggle("kernels", "push", "-p", str(d)))
    elif op == "status":
        print(kaggle("kernels", "status", f"{USER}/{NAME}", check=False))
    elif op == "pull":
        o = ROOT / "kaggle_runs" / NAME
        o.mkdir(parents=True, exist_ok=True)
        print(kaggle("kernels", "output", f"{USER}/{NAME}", "-p", str(o), check=False))


if __name__ == "__main__":
    main()
