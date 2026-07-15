#!/usr/bin/env python3
"""Evaluate a trained checkpoint on VisDrone2019 val or test-dev.

    python val.py --weights runs/detect/lwso/weights/best.pt --split val
    python val.py --weights runs/detect/lwso/weights/best.pt --split test --imgsz 960
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

    from lwso import register_lwso

    register_lwso()  # checkpoints containing custom modules need the classes importable

    from ultralytics import YOLO

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data, imgsz=args.imgsz, split=args.split, batch=args.batch, device=args.device
    )
    print(f"\nmAP50: {metrics.box.map50:.4f}  mAP50-95: {metrics.box.map:.4f}")
    print("per-class mAP50-95:", {k: round(v, 4) for k, v in zip(metrics.names.values(), metrics.box.maps)})


if __name__ == "__main__":
    main()
