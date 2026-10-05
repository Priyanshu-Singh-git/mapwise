"""Export the trained U-Net to ONNX for in-browser inference (onnxruntime-web).

  python deploy/export_onnx.py --ckpt out/mapwise/best.pt --out docs/models/mapwise.onnx
Fixed 512x512 input keeps the graph simple and the browser memory predictable;
the page tiles larger images itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mapwise.data import CLASSES, COLORS  # noqa: E402
from mapwise.model import UNetResNet34  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="out/mapwise/best.pt")
    ap.add_argument("--out", default="docs/models/mapwise.onnx")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--no-quant", action="store_true", help="skip int8 quantisation")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    meta = ck.get("meta", {})
    names = meta.get("classes", CLASSES)
    enc = meta.get("encoder", "resnet34")
    net = UNetResNet34(len(names), pretrained=False, encoder=enc).eval()
    print("encoder:", enc)
    net.load_state_dict(ck["state"])

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.zeros(1, 3, args.size, args.size)
    torch.onnx.export(
        net, dummy, str(out), opset_version=args.opset,
        input_names=["input"], output_names=["logits"],
        dynamic_axes=None,                      # fixed shape: simpler graph, predictable memory
    )
    # sanity: run the exported graph and compare with torch
    import numpy as np
    import onnxruntime as ort
    x = torch.randn(1, 3, args.size, args.size)
    with torch.no_grad():
        ref = net(x).numpy()
    got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
    diff = float(np.abs(ref - got).max())

    # int8 dynamic quantisation: ~4x smaller, which is the difference between a usable
    # browser demo and a 98 MB download.
    q_mb = None
    if not args.no_quant:
        from onnxruntime.quantization import QuantType, quantize_dynamic
        q = out.with_name(out.stem + "_int8.onnx")
        quantize_dynamic(str(out), str(q), weight_type=QuantType.QUInt8)
        qs = ort.InferenceSession(str(q), providers=["CPUExecutionProvider"]).run(None, {"input": x.numpy()})[0]
        agree = float((ref.argmax(1) == qs.argmax(1)).mean())
        q_mb = round(q.stat().st_size / 1e6, 1)
        print(json.dumps({"int8": str(q), "mb": q_mb, "argmax_agreement_vs_fp32": round(agree, 5)}))

    sidecar = out.parent / "mapwise_meta.json"
    sidecar.write_text(json.dumps({
        "classes": names,
        "colors": [list(map(int, c)) for c in (meta.get("colors") or COLORS)],
        "size": args.size,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "mIoU": ck.get("mIoU"), "per_class": ck.get("per_class"),
    }, indent=2))
    print(json.dumps({"onnx": str(out), "mb": round(out.stat().st_size / 1e6, 1),
                      "int8_mb": q_mb, "max_abs_diff_vs_torch": diff,
                      "meta": str(sidecar)}, indent=2))


if __name__ == "__main__":
    main()
