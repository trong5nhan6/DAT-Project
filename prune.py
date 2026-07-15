#!/usr/bin/env python3
"""Semantic-path-aware LAMP pruning for --idea fap (FAP-YOLO12n Sec 4.2).

LAMP (Lee et al., ICLR 2021, "Layer-Adaptive Sparsity for the Magnitude-Based Pruning"):
for each weight, score(w) = w^2 / sum(w'^2 for w' in the same layer with |w'| >= |w|).
Scores from ALL target Conv2d layers are pooled and ranked *globally*, so the achieved
per-layer sparsity is layer-adaptive rather than a hand-set uniform ratio (that's M4 in
the paper's ladder -- see --no-lamp below to reproduce it for comparison).

Semantic-path-aware exception rule (this file's actual contribution over plain LAMP):
weights belonging to the P2/P3 path (protected_layers in configs/fap.yaml), any
`band_logits` (FreqMix's learnable band-mixing weights), and the Detect head are
excluded from the global candidate pool entirely -- never pruned, regardless of how low
their LAMP score is. This is M6 in the paper; drop --protect to get M5 (LAMP, no
constraint) for the ablation comparison the paper is built around (Sec 5.5).

Pipeline (paper Sec 4.2): train FreqMix dense (checkpoint) -> this script -> short
fine-tune. This script now covers all three:
  1. Prune (zero weights, LAMP + P2/P3 exception rule).
  2. Evaluate original vs pruned checkpoint on --data (so you know the accuracy cost
     immediately, not just the sparsity achieved) -- pass --no-eval to skip.
  3. Save a sparsity mask (<out>.mask.pt) alongside the checkpoint. Fine-tuning happens
     via the normal `train.py --idea fap --weights <out> --sparsity-mask <out>.mask.pt`
     -- FAPModel registers a callback that re-zeros the masked positions after every
     optimizer step, so gradients can't undo the pruning while the rest of the network
     keeps training. Without --sparsity-mask, a fine-tune would let pruned weights drift
     away from zero within the first few batches.

Usage:
    # M6: LAMP + P2/P3 exception rule (the paper's proposed method)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3

    # M5: plain LAMP, no path constraint (ablation baseline for M6)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-protect

    # M4: uniform per-layer magnitude pruning, no LAMP (ablation baseline for M5)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-lamp

    # then, short fine-tune with sparsity preserved:
    python train.py --idea fap --weights runs/detect/fap-n/weights/best.pruned.pt \\
        --sparsity-mask runs/detect/fap-n/weights/best.pruned.mask.pt \\
        --epochs 10 --name fap-n-finetuned
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent


def _lamp_scores(weight):
    """LAMP score per weight, scattered back to `weight`'s original flat layout."""
    import torch

    flat = weight.detach().flatten()
    mag2 = flat.pow(2)
    order = torch.argsort(mag2)  # ascending magnitude
    sorted_mag2 = mag2[order]
    reverse_cumsum = torch.flip(torch.cumsum(torch.flip(sorted_mag2, [0]), 0), [0])
    lamp_sorted = sorted_mag2 / reverse_cumsum.clamp_min(1e-12)
    lamp = torch.empty_like(lamp_sorted)
    lamp[order] = lamp_sorted
    return lamp.view_as(weight)


def _protected_param_ids(model, protected_layers):
    """Set of id() for parameters that must never be pruned: everything in
    protected_layers (by index into model.model), every FreqMix.band_logits, and the
    final Detect head -- regardless of protected_layers.
    """
    layers = model.model  # nn.Sequential, 1:1 with the parsed YAML lines
    protected = set()
    for i in protected_layers:
        for p in layers[i].parameters():
            protected.add(id(p))
    for name, p in model.named_parameters():
        if "band_logits" in name:
            protected.add(id(p))
    for p in layers[-1].parameters():  # Detect head is always the last layer
        protected.add(id(p))
    return protected


def prune_lamp(model, sparsity: float, protected_layers, use_lamp: bool = True, protect: bool = True):
    """Zero out `sparsity` fraction of Conv2d weights globally (excluding protected
    params if `protect`), ranked by LAMP score (or raw magnitude if `use_lamp=False`,
    matching M4's uniform/uninformed baseline). Modifies `model` in place.

    Returns (report, mask): report is overall + per-region sparsity achieved; mask is
    {param_name: bool tensor} (True = pruned, held at 0) for every pruned Conv2d weight,
    keyed to match model.named_parameters() -- consumed by train.py --sparsity-mask.
    """
    import torch
    import torch.nn as nn

    protected_ids = _protected_param_ids(model, protected_layers) if protect else set()

    # Vectorized: gather every candidate weight's score into one tensor, find the global
    # threshold via kthvalue, then mask+zero each layer in one shot -- looping per-weight
    # in Python (millions of .item() calls for a multi-million-param model) would be far
    # too slow.
    target_weights = []  # [(param_name, weight_tensor, score_tensor), ...]
    all_scores = []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight
            if id(w) in protected_ids:
                continue
            scores = _lamp_scores(w) if use_lamp else w.detach().abs()
            target_weights.append((f"{name}.weight", w, scores))
            all_scores.append(scores.flatten())

    if not target_weights:
        raise ValueError("No prunable Conv2d layers found (check protected_layers / model).")

    all_scores_flat = torch.cat(all_scores)
    n_candidates = all_scores_flat.numel()
    n_prune = int(n_candidates * sparsity)
    threshold = torch.kthvalue(all_scores_flat, n_prune).values.item() if n_prune > 0 else float("-inf")

    pruned_count = 0
    mask = {}
    with torch.no_grad():
        for name, w, scores in target_weights:
            m = scores <= threshold
            pruned_count += int(m.sum().item())
            w[m] = 0.0
            mask[name] = m.clone()

    total_params = sum(p.numel() for p in model.parameters())
    total_zero = sum((p == 0).sum().item() for p in model.parameters())
    protected_params = sum(
        p.numel() for name, p in model.named_parameters() if id(p) in protected_ids
    )
    report = {
        "candidate_weights": n_candidates,
        "pruned_weights": pruned_count,
        "overall_sparsity": total_zero / total_params,
        "protected_params": protected_params,
        "protected_frac_of_total": protected_params / total_params,
    }
    return report, mask


def _evaluate(weights: str, data: str, imgsz: int, device, split: str):
    """Fresh YOLO load + val() -- returns (mAP50, mAP50-95), or (None, None) on failure
    (e.g. no labels for `split`, common for VisDrone test-dev)."""
    from ultralytics import YOLO

    try:
        metrics = YOLO(weights).val(
            data=data, imgsz=imgsz, split=split, device=device, plots=False, verbose=False
        )
        return float(metrics.box.map50), float(metrics.box.map)
    except Exception as e:
        print(f"[fap] eval failed for {weights}: {e}")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="checkpoint from `train.py --idea fap`")
    ap.add_argument("--sparsity", type=float, default=0.3, help="target global fraction to zero")
    ap.add_argument("--config", default=str(ROOT / "configs" / "fap.yaml"),
                     help="source of protected_layers (P2/P3 exception rule)")
    ap.add_argument("--no-protect", dest="protect", action="store_false",
                     help="disable the P2/P3 exception rule (M5 ablation: plain LAMP)")
    ap.add_argument("--no-lamp", dest="use_lamp", action="store_false",
                     help="rank by raw magnitude instead of LAMP score (M4 ablation)")
    ap.add_argument("--out", default=None, help="output .pt path (default: <weights>.pruned.pt)")
    ap.add_argument("--no-eval", dest="eval", action="store_false",
                     help="skip before/after accuracy evaluation (faster, but you won't "
                          "know the accuracy cost of this prune)")
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone.yaml"), help="for --eval")
    ap.add_argument("--imgsz", type=int, default=960, help="for --eval")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default=None, help="for --eval, e.g. 0 or cpu")
    args = ap.parse_args()

    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    protected_layers = list(cfg.get("protected_layers", []))

    from models.fap.register import register_fap

    register_fap()  # checkpoint's FreqMix classes must be importable to unpickle
    from ultralytics import YOLO

    yolo = YOLO(args.weights)
    model = yolo.model

    if not args.use_lamp:
        label = "uniform-magnitude"
    elif args.protect:
        label = "LAMP+P2P3"
    else:
        label = "LAMP"
    print(f"[fap] pruning {args.weights} -> sparsity={args.sparsity}, method={label}")
    if args.protect:
        print(f"[fap] protected layers (P2/P3 path): {protected_layers} "
              f"+ all band_logits + Detect head")

    report, mask = prune_lamp(
        model, args.sparsity, protected_layers, use_lamp=args.use_lamp, protect=args.protect
    )
    print(f"[fap] candidates={report['candidate_weights']:,}  pruned={report['pruned_weights']:,}")
    print(f"[fap] overall sparsity (whole model, incl. protected/non-conv params): "
          f"{report['overall_sparsity']:.4f}")
    print(f"[fap] protected params: {report['protected_params']:,} "
          f"({report['protected_frac_of_total']:.2%} of total)")

    out = Path(args.out) if args.out else Path(args.weights).with_suffix(".pruned.pt")
    yolo.save(str(out))
    print(f"[fap] saved pruned checkpoint -> {out}")

    import torch

    mask_path = out.with_suffix(".mask.pt")
    torch.save(mask, mask_path)
    n_masked = sum(int(m.sum().item()) for m in mask.values())
    print(f"[fap] saved sparsity mask -> {mask_path} ({n_masked:,} positions held at 0)")

    if args.eval:
        print(f"\n[fap] evaluating on --{args.eval_split} (before vs after prune)...")
        map50_before, map5095_before = _evaluate(
            args.weights, args.data, args.imgsz, args.device, args.eval_split
        )
        map50_after, map5095_after = _evaluate(
            str(out), args.data, args.imgsz, args.device, args.eval_split
        )

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n[fap] {'':16s} {'mAP50':>10s}  {'mAP50-95':>10s}")
        print(f"[fap] {'before prune':16s} {_fmt(map50_before):>10s}  {_fmt(map5095_before):>10s}")
        print(f"[fap] {'after prune':16s} {_fmt(map50_after):>10s}  {_fmt(map5095_after):>10s}")
        if map50_before is not None and map50_after is not None:
            print(f"[fap] mAP50 delta: {map50_after - map50_before:+.4f}")

    print(f"\n[fap] fine-tune with sparsity preserved:\n"
          f"  python train.py --idea fap --weights {out} --sparsity-mask {mask_path} "
          f"--epochs 10 --name <ten-run-moi>")


if __name__ == "__main__":
    main()
