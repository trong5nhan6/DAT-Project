#!/usr/bin/env python3
"""Train LWSO-YOLO (or baseline) on VisDrone2019, config-driven.

Config layering: configs/base.yaml (defaults) <- configs/<idea>.yaml (override
theo idea, auto-resolved từ --idea, hoặc trỏ thẳng bằng --config) <- CLI flags
(override cuối cùng, chỉ áp khi thực sự được truyền).

Idea nào cũng đăng ký trong models/ (xem models/__init__.py MODEL_REGISTRY) —
--idea nhận choices trực tiếp từ đó, thêm idea mới không cần sửa file này
(xem models/__init__.py để biết 3 bước thêm 1 idea).

Model options (--model, override cfg.model_cfg của configs/<idea>.yaml):
    yolo11n.pt                          baseline YOLO11n gốc, finetune COCO-pretrained  (idea baseline)
    cfg/base-yolo11n.yaml               baseline YOLO11n gốc, train from-scratch        (idea baseline)
    cfg/ablation/yolo11n-p2-nop5.yaml   ablation: chỉ +P2/-P5, module gốc 100%          (idea baseline)
    cfg/lwso-yolo11n.yaml               LWSO đầy đủ: 2.40M params, 21.5 GFLOPs@640      (idea lwso, mặc định)
    cfg/lwso-yolo11n-lite.yaml          LWSO cắt compute mạnh: 1.12M, 12.8 GFLOPs@640   (idea lwso)
    cfg/lwso-yolo11n-eff.yaml           LWSO đề xuất: 0.96M, 12.4 GFLOPs@640, +ECA ở P2 (idea lwso)
    cfg/lwso-yolo11s.yaml               bản s — teacher cho knowledge distillation      (idea lwso)

Examples:
    # baseline (configs/baseline.yaml: YOLO11n gốc, finetune COCO, không NWD)
    python train.py --idea baseline

    # full LWSO model @960 với NWD blend loss (configs/lwso.yaml)
    python train.py --idea lwso

    # 1 trong các model option khác (xem bảng trên), vẫn cùng idea/recipe:
    python train.py --idea lwso --model cfg/lwso-yolo11n-eff.yaml --name lwso-n-eff

    # override vài hyperparam từ CLI, đè lên config
    python train.py --idea lwso --imgsz 640 --batch 16 --name lwso-n-640

    # ablation +P2/-P5 (module gốc): trỏ thẳng model, vẫn dùng idea baseline (no-nwd)
    python train.py --idea baseline --model cfg/ablation/yolo11n-p2-nop5.yaml --name abl-p2

    # trỏ thẳng 1 experiment yaml tùy ý, bỏ qua auto-resolve theo --idea
    python train.py --config configs/lwso.yaml --epochs 50

    # log mAP tập test mỗi 10 epoch (monitoring, không dùng để chọn best.pt)
    python train.py --idea lwso --test-every 10

    # fine-tune sau khi prune.py (idea fap) — giữ sparsity mask trong suốt fine-tune
    python train.py --idea fap --weights runs/detect/fap-n/weights/best.pruned.pt \\
        --sparsity-mask runs/detect/fap-n/weights/best.pruned.mask.pt --epochs 10
"""

import argparse
import os
import sys
from pathlib import Path

# Anaconda ships its own libiomp5md.dll which clashes with torch's copy on Windows
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# Windows' default console codepage (e.g. cp1258) can't encode the Vietnamese
# diacritics in --help text / log messages below; force UTF-8 so `--help` and
# prints don't crash instead of silently mojibake-ing.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent


def load_config(base_cfg: str, exp_cfg: str):
    """Merge configs/base.yaml + configs/<idea>.yaml. Priority: exp_cfg > base_cfg."""
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(base_cfg)
    cfg = OmegaConf.merge(cfg, OmegaConf.load(exp_cfg))
    return cfg


def main():
    from models import MODEL_REGISTRY

    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", default="baseline", choices=sorted(MODEL_REGISTRY),
                     help="chọn configs/<idea>.yaml + models/<idea>.py (bỏ qua nếu dùng --config)")
    ap.add_argument("--config", default=None,
                     help="trỏ thẳng 1 experiment yaml, bỏ qua auto-resolve theo --idea")
    ap.add_argument("--base-config", default=str(ROOT / "configs" / "base.yaml"))

    # Mọi flag dưới đây override giá trị tương ứng trong config CHỈ KHI được truyền
    # (default=None => "không truyền" khác với "truyền giá trị falsy")
    ap.add_argument("--model", default=None, dest="model_cfg",
                     help="model .yaml (train from scratch) hoặc .pt (finetune) — xem "
                          "danh sách option ở docstring đầu file")
    ap.add_argument("--data", default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None,
                     help="P2 head @960 is VRAM-hungry; 8 fits ~12GB, raise if you can")
    ap.add_argument("--device", default=None, help="e.g. 0 or 0,1 or cpu")
    ap.add_argument("--name", default=None, help="run name under runs/detect/")
    ap.add_argument("--weights", default=None, help="optional .pt to warm-start a .yaml model")
    ap.add_argument("--sparsity-mask", default=None, dest="sparsity_mask",
                     help="(idea fap) <out>.mask.pt from prune.py — re-applied after every "
                          "optimizer step so a fine-tune can't undo the pruning via gradient drift")
    ap.add_argument("--multi-scale", dest="multi_scale", action="store_true", default=None,
                     help="multi-scale training (more VRAM)")
    ap.add_argument("--close-mosaic", type=int, default=None)
    ap.add_argument("--mixup", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--resume", action="store_true", default=None)
    ap.add_argument("--seed", type=int, default=None, help="random seed (reproducibility)")
    ap.add_argument("--test-every", type=int, default=None, dest="test_every",
                     help="also val on the test-dev split every N epochs (monitoring only; "
                          "0 disables). Adds one extra full test-set pass per trigger, on top "
                          "of the normal per-epoch val — costs time, use a coarse N (e.g. 10-20).")
    # NWD blend loss (see models/lwso/losses.py) — --no-nwd always forces it off regardless of config
    ap.add_argument("--no-nwd", action="store_true", help="use stock CIoU loss only")
    ap.add_argument("--nwd-ratio", type=float, default=None, help="weight of CIoU term in the blend")
    ap.add_argument("--nwd-constant", type=float, default=None, help="NWD normalizing constant C")
    args = ap.parse_args()

    exp_cfg = args.config or str(ROOT / "configs" / f"{args.idea}.yaml")
    cfg = load_config(args.base_config, exp_cfg)

    overrides = {
        "model_cfg": args.model_cfg, "data": args.data, "imgsz": args.imgsz,
        "epochs": args.epochs, "batch": args.batch, "device": args.device,
        "name": args.name, "weights": args.weights, "sparsity_mask": args.sparsity_mask,
        "multi_scale": args.multi_scale,
        "close_mosaic": args.close_mosaic, "mixup": args.mixup, "patience": args.patience,
        "resume": args.resume, "seed": args.seed, "test_every": args.test_every,
        "nwd_ratio": args.nwd_ratio, "nwd_constant": args.nwd_constant,
    }
    for k, v in overrides.items():
        if v is not None:
            cfg[k] = v
    if args.no_nwd:
        cfg.use_nwd = False

    print(f"[lwso] idea={cfg.idea}  config={exp_cfg}")

    from models import build_model

    model = build_model(cfg.idea, cfg)
    model.train(
        data=str(cfg.data),
        imgsz=int(cfg.imgsz),
        epochs=int(cfg.epochs),
        batch=int(cfg.batch),
        device=cfg.device,
        name=str(cfg.name),
        close_mosaic=int(cfg.close_mosaic),
        mixup=float(cfg.mixup),
        multi_scale=bool(cfg.multi_scale),
        patience=int(cfg.patience),
        resume=bool(cfg.resume),
        seed=int(cfg.seed),
        cos_lr=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
