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
fine-tune. Fine-tune is NOT implemented here: this script only zeros weights and saves a
checkpoint; nothing keeps pruned weights at zero during a subsequent train.py run (no
persistent mask/hook), so a naive fine-tune will let gradients drift them away from zero.
That's out of scope for now -- see "hãy giúp tôi code idea này" follow-up if/when the
train->prune->fine-tune loop needs closing.

Usage:
    # M6: LAMP + P2/P3 exception rule (the paper's proposed method)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3

    # M5: plain LAMP, no path constraint (ablation baseline for M6)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-protect

    # M4: uniform per-layer magnitude pruning, no LAMP (ablation baseline for M5)
    python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-lamp
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

    Returns a report dict: overall + per-region (protected vs prunable) sparsity achieved.
    """
    import torch
    import torch.nn as nn

    protected_ids = _protected_param_ids(model, protected_layers) if protect else set()

    # Vectorized: gather every candidate weight's score into one tensor, find the global
    # threshold via kthvalue, then mask+zero each layer in one shot -- looping per-weight
    # in Python (millions of .item() calls for a multi-million-param model) would be far
    # too slow.
    target_weights = []  # [(weight_tensor, score_tensor), ...]
    all_scores = []
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            w = m.weight
            if id(w) in protected_ids:
                continue
            scores = _lamp_scores(w) if use_lamp else w.detach().abs()
            target_weights.append((w, scores))
            all_scores.append(scores.flatten())

    if not target_weights:
        raise ValueError("No prunable Conv2d layers found (check protected_layers / model).")

    all_scores_flat = torch.cat(all_scores)
    n_candidates = all_scores_flat.numel()
    n_prune = int(n_candidates * sparsity)
    threshold = torch.kthvalue(all_scores_flat, n_prune).values.item() if n_prune > 0 else float("-inf")

    pruned_count = 0
    with torch.no_grad():
        for w, scores in target_weights:
            mask = scores <= threshold
            pruned_count += int(mask.sum().item())
            w[mask] = 0.0

    total_params = sum(p.numel() for p in model.parameters())
    total_zero = sum((p == 0).sum().item() for p in model.parameters())
    protected_params = sum(
        p.numel() for name, p in model.named_parameters() if id(p) in protected_ids
    )
    return {
        "candidate_weights": n_candidates,
        "pruned_weights": pruned_count,
        "overall_sparsity": total_zero / total_params,
        "protected_params": protected_params,
        "protected_frac_of_total": protected_params / total_params,
    }


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

    report = prune_lamp(
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
    print("[fap] NOTE: pruned weights are NOT masked -- a subsequent fine-tune "
          "(train.py --weights) will let gradients drift them away from zero.")


if __name__ == "__main__":
    main()
