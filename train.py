#!/usr/bin/env python3
"""Train LWSO-YOLO (or any ablation config) on VisDrone2019.

Examples:
    # baseline YOLO11n (pretrained COCO weights), 640
    python train.py --model yolo11n.pt --imgsz 640 --name base-11n-640 --no-nwd

    # ablation: +P2/-P5 only (stock modules)
    python train.py --model cfg/ablation/yolo11n-p2-nop5.yaml --name abl-p2 --no-nwd

    # full LWSO model @960 with NWD blend loss
    python train.py --model cfg/lwso-yolo11n.yaml --imgsz 960 --name lwso-n-960

    # same, but also log test-dev mAP every 10 epochs (monitoring, not model selection)
    python train.py --model cfg/lwso-yolo11n.yaml --imgsz 960 --name lwso-n-960 --test-every 10
"""

import argparse
import os
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = Path(__file__).resolve().parent


def _build_test_eval_callback(data: str, imgsz: int, batch: int, every: int):
    """Every `every` epochs, val on the VisDrone test-dev split using the run's current
    EMA weights and log mAP to <save_dir>/test_metrics.csv.

    Uses a standalone DetectionValidator (model=... path, not trainer=...) so it never
    touches trainer.validator / trainer.stopper / best.pt selection, which stay driven by
    the ordinary val split. Test mAP here is for monitoring only, not for model selection
    (test-dev has no official public labels; treat any local labels as unofficial).

    Validates a deepcopy of the EMA weights, not the live trainer.ema.ema reference:
    AutoBackend fuses whatever nn.Module it's handed (in place, `model.fuse()`), which
    permanently changes its state_dict keys. Handing it the live EMA model corrupts it
    and crashes the next `ema.update(self.model)` call with a KeyError.
    """
    import copy

    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import RANK

    state = {"validator": None, "log_path": None, "last_epoch": None}

    def _on_fit_epoch_end(trainer):
        if RANK not in (-1, 0):
            return
        epoch = trainer.epoch + 1  # 1-indexed, matches the printed epoch column
        # trainer.final_eval() re-fires on_fit_epoch_end once more at the same epoch
        # after training ends; skip the repeat instead of double-logging/re-validating.
        if epoch % every != 0 or epoch == state["last_epoch"]:
            return
        state["last_epoch"] = epoch

        if state["validator"] is None:
            state["validator"] = DetectionValidator(
                args=dict(
                    data=data,
                    split="test",
                    imgsz=imgsz,
                    batch=batch,
                    plots=False,
                    save_json=False,
                    device=trainer.device,
                ),
                save_dir=trainer.save_dir / "test_eval",
            )
            state["log_path"] = trainer.save_dir / "test_metrics.csv"
            if not state["log_path"].exists():
                state["log_path"].write_text("epoch,mAP50,mAP50-95\n")

        print(f"\n[lwso] test-set eval @ epoch {epoch}")
        model = copy.deepcopy(trainer.ema.ema or trainer.model)
        stats = state["validator"](model=model)
        map50 = stats.get("metrics/mAP50(B)", float("nan"))
        map5095 = stats.get("metrics/mAP50-95(B)", float("nan"))
        with open(state["log_path"], "a") as f:
            f.write(f"{epoch},{map50:.5f},{map5095:.5f}\n")

    return _on_fit_epoch_end


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(ROOT / "cfg" / "lwso-yolo11n.yaml"),
                    help="model .yaml (train from scratch) or .pt (finetune)")
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone.yaml"))
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8,
                    help="P2 head @960 is VRAM-hungry; 8 fits ~12GB, raise if you can")
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu")
    ap.add_argument("--name", default="lwso", help="run name under runs/detect/")
    ap.add_argument("--weights", default=None, help="optional .pt to warm-start a .yaml model")
    ap.add_argument("--multi-scale", action="store_true", help="multi-scale training (more VRAM)")
    ap.add_argument("--close-mosaic", type=int, default=15)
    ap.add_argument("--mixup", type=float, default=0.1)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--test-every", type=int, default=20,
                     help="also val on the test-dev split every N epochs (monitoring only; "
                          "0 disables). Adds one extra full test-set pass per trigger, on top "
                          "of the normal per-epoch val — costs time, use a coarse N (e.g. 10-20).")
    # NWD blend loss (see lwso/losses.py)
    ap.add_argument("--no-nwd", action="store_true", help="use stock CIoU loss only")
    ap.add_argument("--nwd-ratio", type=float, default=0.5, help="weight of CIoU term in the blend")
    ap.add_argument("--nwd-constant", type=float, default=12.8, help="NWD normalizing constant C")
    args = ap.parse_args()

    from lwso import patch_nwd_loss, register_lwso

    register_lwso()  # needed for lwso-*.yaml; harmless for stock models
    if not args.no_nwd:
        patch_nwd_loss(ratio=args.nwd_ratio, constant=args.nwd_constant)
        print(f"[lwso] NWD blend loss active: ratio={args.nwd_ratio}, C={args.nwd_constant}")

    from ultralytics import YOLO

    model = YOLO(args.model)
    if args.weights:
        model.load(args.weights)

    if args.test_every > 0:
        model.add_callback(
            "on_fit_epoch_end",
            _build_test_eval_callback(args.data, args.imgsz, args.batch, args.test_every),
        )
        print(f"[lwso] test-set eval every {args.test_every} epochs "
              f"-> runs/detect/{args.name}/test_metrics.csv")

    model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        device=args.device,
        name=args.name,
        close_mosaic=args.close_mosaic,
        mixup=args.mixup,
        multi_scale=args.multi_scale,
        patience=args.patience,
        resume=args.resume,
        cos_lr=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
