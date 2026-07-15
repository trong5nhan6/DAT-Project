#!/usr/bin/env python3
"""Evaluate a trained checkpoint on VisDrone2019 val or test-dev.

Always reports efficiency metrics (params/GFLOPs/latency/FPS) alongside mAP, matching
the block train.py's --test-every callback writes during training (models/base_model.py)
-- so a standalone eval after training gives the same numbers, not just mAP.

    python val.py --weights runs/detect/lwso-n/weights/best.pt --split val
    python val.py --weights runs/detect/lwso-n/weights/best.pt --split test --imgsz 960
"""

import argparse
import os
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone.yaml"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--split", default="val", choices=["val", "test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    # val.py doesn't know which --idea produced this checkpoint, so register every idea's
    # custom modules regardless -- cheap, idempotent, harmless for a checkpoint that
    # doesn't use them.
    from models.fap.register import register_fap
    from models.lwso.register import register_lwso
    from models.star.register import register_star

    register_lwso()
    register_fap()
    register_star()

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import select_device

    from models.base_model import _compute_efficiency_metrics, _format_efficiency_report

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, split=args.split, batch=args.batch, device=args.device
    )
    print(f"\nmAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print("per-class mAP50-95:", {k: round(v, 4) for k, v in zip(metrics.names.values(), metrics.box.maps)})

    eff = _compute_efficiency_metrics(model.model, args.imgsz, select_device(args.device))
    print()
    for line in _format_efficiency_report(eff):
        print(line)


if __name__ == "__main__":
    main()
