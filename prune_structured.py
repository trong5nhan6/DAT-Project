#!/usr/bin/env python3
"""[EXPERIMENTAL] Structured (channel-level) pruning for lwso-family checkpoints — stage
2 of the "train big -> structured prune -> distill back" pipeline (idea "pd" is stage 3).

STATUS (verified locally, 2026-07, torch-pruning 1.5.2/1.6.1 x torch 2.10/2.12): the
torch-pruning DepGraph currently CANNOT handle this architecture family correctly:
  * group expansion loops forever whenever a pruning root reaches an attention-style
    reshape (C2PSA's qkv — reproduces on STOCK yolo11n), EMA, DySample's grid_sample, or
    SPDConv(Group)/FreqMix's slice/wavelet input paths -> mitigated here by an extensive
    protection list (build_ignored_layers), which shrinks the prunable surface to block
    interiors only;
  * even then, tp builds occasional groups with corrupt concat-offset index mappings —
    some crash importance scoring (skipped defensively), others prune inconsistently and
    yield a model whose forward fails. main() therefore ALWAYS forward-checks the pruned
    graph and refuses to save a broken model (exits non-zero with guidance).
Until torch-pruning fixes these upstream, the SUPPORTED route to the efficiency target
is idea "pd" with the slim student: `python train.py --idea pd` (cfg/slim-yolo11n.yaml
is already 0.56M / 6.31 GFLOPs@640 by construction + CWD distill from lwso-eff). This
file is kept for the compression-ablation ladder once tp is usable, and as the record
of which module patterns break it.

Difference vs prune.py (idea fap's unstructured LAMP): prune.py zeroes individual weights
and keeps them at 0 through a mask — the tensors keep their shape, so measured params/
GFLOPs/latency DO NOT change (fap's result: still 2.12M/8.8G after pruning). This script
uses torch-pruning's DepGraph (Fang et al., CVPR 2023) to REMOVE whole channels and every
tensor slice that depends on them — the saved model is physically smaller and faster.

Importance & protection mirror prune.py's semantics, transplanted to channel granularity:
  * LAMP importance (Lee et al., ICLR 2021), ranked globally across layers
    (--no-lamp falls back to plain L2 magnitude).
  * Protected from pruning (never selected as pruning roots):
      - the Detect head (always),
      - the layers PRODUCING the Detect inputs (default; --no-protect-inputs disables)
        -> keeps P2/P3/P4 tap channel counts identical to the dense teacher, which is
        what lets idea pd run CWD self-distillation without adapter convs,
      - modules whose forward carries reshape/upsample semantics that channel slicing
        would corrupt or that are too cheap to be worth the risk: EMA, ECA, DySample,
        BiFPNCat, SimAM, FreqMix (attention/upsampler/fusion),
      - optional extra layer indices via --protect-layers (comma-separated indices into
        model.model, same convention as configs/fap.yaml's protected_layers).

Two stopping modes:
  * --pruning-ratio R: one-shot global channel pruning at ratio R.
  * --target-gflops G: iterative (up to --max-ratio in --steps steps), stopping at the
    first step whose GFLOPs@--flops-imgsz drops below G. Use this to land under the
    baseline budget (stock YOLO11n: 6.5 GFLOPs@640 unfused).

Like prune.py, this script evals before/after (val split by default, --no-eval to skip)
so the accuracy cost is visible immediately. Fine-tune + distill afterwards:

    python prune_structured.py --weights runs/detect/lwso-n-eff/weights/best.pt --target-gflops 6.0
    python train.py --idea pd --model runs/detect/lwso-n-eff/weights/best.pruned-structured.pt \\
        --name pd-n     # configs/pd.yaml points distill_teacher at the dense best.pt
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


def _register_all():
    """Checkpoint may reference custom modules from any idea — register every family."""
    from models.fap.register import register_fap
    from models.lwso.register import register_lwso
    from models.slim.register import register_slim
    from models.star.register import register_star

    register_lwso()
    register_fap()
    register_star()
    register_slim()


def build_ignored_layers(model, protect_inputs: bool = True, extra_layer_idxs=()):
    """Modules torch-pruning must not use as pruning roots (see module docstring)."""
    import torch.nn as nn
    from ultralytics.nn.modules.block import Attention

    from models.fap.modules import FreqMix
    from models.lwso.modules import BiFPNCat, DySample, ECA, EMA, SPDConv, SPDConvGroup
    from models.star.modules import SimAM

    layers = model.model  # nn.Sequential, 1:1 with the parsed YAML lines
    ignored = [layers[-1]]  # Detect head: nc/reg_max outputs must never shrink
    if protect_inputs:
        detect_from = layers[-1].f  # e.g. [16, 19, 22]
        ignored += [layers[i] for i in detect_from]
    ignored += [layers[i] for i in extra_layer_idxs]
    # Attention: torch-pruning's group expansion loops FOREVER on the qkv head-split
    # index-mapping pattern (verified on stock yolo11n: DG.get_pruning_group on
    # model.10.m.0.attn.qkv.conv never returns, tp 1.5.2/1.6.1, torch 2.10/2.12).
    # C2PSA's own cv1/cv2 stay prunable (verified fine) — only the attention internals
    # (qkv/proj/pe) must never be used as pruning roots.
    fragile = (Attention, EMA, ECA, DySample, BiFPNCat, SimAM, FreqMix, SPDConv, SPDConvGroup)
    ignored += [m for m in model.modules() if isinstance(m, fragile)]
    # EMA/DySample/SPDConv(Group)/FreqMix have NO leading conv barrier: a channel change
    # in their PRODUCER layer propagates straight into their reshape / grid_sample /
    # space-to-depth-slice / wavelet graph, where tp's index-mapping expansion also loops
    # forever (verified on lwso-eff: root model.4.cv2 — the C3k2Ghost feeding EMA — and
    # root model.20.conv — SPDConvGroup's own fusion conv — both hang). Freeze the
    # producers' out-channels too.
    barrier_less = (EMA, DySample, SPDConv, SPDConvGroup, FreqMix)
    for i, layer in enumerate(layers):
        if isinstance(layer, barrier_less):
            f = getattr(layer, "f", -1)
            for src in (f if isinstance(f, (list, tuple)) else [f]):
                ignored.append(layers[i + src if src < 0 else src])
    # Unwrapped nn.Parameters (BiFPNCat.w fusion weights, FreqMix.band_logits) are NOT
    # covered by module-level ignoring: DepGraph auto-detects them as prunable along their
    # last non-singleton dim, which for these is the input-count/band axis — pruning it
    # corrupts the module semantics. MetaPruner's ignored_layers accepts raw parameters
    # (routed to ignored_params internally), so exclude them explicitly.
    for m in model.modules():
        if isinstance(m, fragile):
            ignored += list(m.parameters(recurse=False))
    for name, p in model.named_parameters():
        if "band_logits" in name or name.endswith(".w"):
            ignored.append(p)
    # dedupe by identity (a detect-input layer can also be a fragile module, e.g. ECA)
    seen, out = set(), []
    for m in ignored:
        if id(m) not in seen:
            seen.add(id(m))
            out.append(m)
    return out


def prune_structured(
    model,
    example_inputs,
    pruning_ratio: float,
    ignored_layers,
    use_lamp: bool = True,
    round_to: int = 8,
    steps: int = 1,
    stop_fn=None,
):
    """Prune `model` in place. With steps>1, prunes iteratively (pruning_ratio is the
    final cumulative target) and stops early the first time stop_fn(model) is True.
    Returns the number of steps actually applied.
    """
    import torch_pruning as tp

    class _SkipBrokenGroups:
        """torch-pruning occasionally builds a group whose index mapping is corrupt on
        this architecture family (idxs beyond the layer's channel count — upstream bug in
        concat-offset bookkeeping, tp<=1.6.1). base_pruner officially skips any group
        whose importance is None, so downgrade those crashes to a skip: the healthy
        groups still get pruned, the corrupt ones are left dense (conservative).
        """

        def __init__(self, base):
            self.base = base
            self.skipped = 0

        def __call__(self, group, **kwargs):
            try:
                return self.base(group, **kwargs)
            except IndexError:
                self.skipped += 1
                return None

    imp = _SkipBrokenGroups(
        tp.importance.LAMPImportance(p=2) if use_lamp else tp.importance.GroupMagnitudeImportance(p=2)
    )
    pruner = tp.pruner.MetaPruner(
        model,
        example_inputs,
        importance=imp,
        pruning_ratio=pruning_ratio,
        iterative_steps=steps,
        global_pruning=True,
        ignored_layers=ignored_layers,
        round_to=round_to,
    )
    applied = 0
    for _ in range(steps):
        pruner.step()
        applied += 1
        if stop_fn is not None and stop_fn(model):
            break
    if imp.skipped:
        print(f"[pd] note: {imp.skipped} mis-mapped group(s) skipped (left dense) — "
              "upstream torch-pruning bug, see prune_structured.py docstring")
    return applied


def _evaluate(weights: str, data: str, imgsz: int, device, split: str):
    """Fresh YOLO load + val() — returns (mAP50, mAP50-95), or (None, None) on failure."""
    from ultralytics import YOLO

    try:
        metrics = YOLO(weights).val(
            data=data, imgsz=imgsz, split=split, device=device, plots=False, verbose=False
        )
        return float(metrics.box.map50), float(metrics.box.map)
    except Exception as e:
        print(f"[pd] eval failed for {weights}: {e}")
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="dense checkpoint (e.g. lwso-eff best.pt)")
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--pruning-ratio", type=float, default=None,
                        help="one-shot global channel pruning ratio, e.g. 0.4")
    group.add_argument("--target-gflops", type=float, default=None,
                        help="iterative: stop at first step below this GFLOPs@--flops-imgsz")
    ap.add_argument("--max-ratio", type=float, default=0.8,
                     help="(--target-gflops) cumulative ratio ceiling for the iteration")
    ap.add_argument("--steps", type=int, default=16,
                     help="(--target-gflops) number of iterative steps toward --max-ratio")
    ap.add_argument("--flops-imgsz", type=int, default=640,
                     help="imgsz for GFLOPs bookkeeping (baseline reference: 6.5@640)")
    ap.add_argument("--round-to", type=int, default=8,
                     help="pruned channel counts stay multiples of this (grouped-conv safe)")
    ap.add_argument("--no-lamp", dest="use_lamp", action="store_false",
                     help="rank by plain L2 magnitude instead of LAMP (ablation)")
    ap.add_argument("--no-protect-inputs", dest="protect_inputs", action="store_false",
                     help="allow pruning the detect-input producers (breaks adapter-free "
                          "CWD self-distill in idea pd — ablation only)")
    ap.add_argument("--protect-layers", default="",
                     help="extra comma-separated model.model indices to protect, e.g. '0,1,2'")
    ap.add_argument("--out", default=None,
                     help="output .pt (default: <weights, .pt stripped>.pruned-structured.pt)")
    ap.add_argument("--no-eval", dest="eval", action="store_false",
                     help="skip before/after accuracy evaluation")
    ap.add_argument("--data", default=str(ROOT / "data" / "visdrone.yaml"), help="for eval")
    ap.add_argument("--imgsz", type=int, default=960, help="for eval")
    ap.add_argument("--eval-split", default="val", choices=["val", "test"])
    ap.add_argument("--device", default=None, help="for eval, e.g. 0 or cpu (prune is CPU)")
    args = ap.parse_args()

    _register_all()

    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    yolo = YOLO(args.weights)
    model = yolo.model.float().cpu().eval()  # prune on CPU: deterministic, VRAM-free
    for p in model.parameters():
        p.requires_grad_(True)  # tp needs grad-capable tensors to trace the dep graph
    example = torch.zeros(1, 3, args.flops_imgsz, args.flops_imgsz)

    params0 = get_num_params(model) / 1e6
    gflops0 = get_flops(model, args.flops_imgsz)
    print(f"[pd] before: {params0:.3f}M params, {gflops0:.2f} GFLOPs@{args.flops_imgsz}")

    extra = [int(x) for x in args.protect_layers.split(",") if x.strip()]
    ignored = build_ignored_layers(model, protect_inputs=args.protect_inputs, extra_layer_idxs=extra)
    print(f"[pd] method={'LAMP' if args.use_lamp else 'L2-magnitude'} (global), "
          f"protect_inputs={args.protect_inputs}, extra_protected={extra or 'none'}, "
          f"ignored_modules={len(ignored)}")

    if args.target_gflops is not None:
        def stop_fn(m):
            g = get_flops(m, args.flops_imgsz)
            print(f"[pd]   step -> {get_num_params(m)/1e6:.3f}M, {g:.2f} GFLOPs")
            return g <= args.target_gflops

        applied = prune_structured(
            model, example, args.max_ratio, ignored, use_lamp=args.use_lamp,
            round_to=args.round_to, steps=args.steps, stop_fn=stop_fn,
        )
        print(f"[pd] applied {applied}/{args.steps} iterative steps "
              f"(cumulative ratio ~{args.max_ratio * applied / args.steps:.2f})")
        if get_flops(model, args.flops_imgsz) > args.target_gflops:
            print(f"[pd] WARNING: --max-ratio {args.max_ratio} exhausted above the "
                  f"{args.target_gflops} GFLOPs target — raise --max-ratio or accept this point.")
    else:
        prune_structured(
            model, example, args.pruning_ratio, ignored,
            use_lamp=args.use_lamp, round_to=args.round_to, steps=1,
        )

    params1 = get_num_params(model) / 1e6
    gflops1 = get_flops(model, args.flops_imgsz)
    print(f"[pd] after:  {params1:.3f}M params ({params1 / params0 - 1:+.1%}), "
          f"{gflops1:.2f} GFLOPs@{args.flops_imgsz} ({gflops1 / gflops0 - 1:+.1%})")

    # sanity: both forward modes must run on the pruned graph before we save anything —
    # torch-pruning's corrupt-group bug (see docstring) can silently produce a model
    # whose channel bookkeeping is inconsistent; never save such a checkpoint.
    try:
        with torch.no_grad():
            model(example)
        model.train()
        model(torch.zeros(1, 3, 256, 256))
        model.eval()
    except RuntimeError as e:
        print(f"\n[pd] FATAL: pruned model forward is broken ({e}).\n"
              "[pd] This is the known upstream torch-pruning index-mapping bug (see the\n"
              "[pd] docstring). NOT saving the checkpoint. Use the supported route instead:\n"
              "[pd]   python train.py --idea pd    (slim student + CWD distill)")
        sys.exit(1)
    print("[pd] pruned model forward OK (eval + train modes)")

    out = Path(args.out) if args.out else Path(args.weights).with_suffix(".pruned-structured.pt")
    yolo.model = model  # ensure the wrapper saves the mutated module
    yolo.save(str(out))
    print(f"[pd] saved structurally pruned checkpoint -> {out}")

    if args.eval:
        print(f"\n[pd] evaluating on --{args.eval_split} (before vs after prune)...")
        map50_b, map5095_b = _evaluate(args.weights, args.data, args.imgsz, args.device, args.eval_split)
        map50_a, map5095_a = _evaluate(str(out), args.data, args.imgsz, args.device, args.eval_split)

        def _fmt(v):
            return f"{v:.4f}" if v is not None else "N/A"

        print(f"\n[pd] {'':16s} {'mAP50':>10s}  {'mAP50-95':>10s}")
        print(f"[pd] {'before prune':16s} {_fmt(map50_b):>10s}  {_fmt(map5095_b):>10s}")
        print(f"[pd] {'after prune':16s} {_fmt(map50_a):>10s}  {_fmt(map5095_a):>10s}")
        if map50_b is not None and map50_a is not None:
            print(f"[pd] mAP50 delta: {map50_a - map50_b:+.4f} (recovered by the pd fine-tune)")

    print(f"\n[pd] fine-tune + CWD self-distill (teacher = the dense checkpoint):\n"
          f"  python train.py --idea pd --model {out} --name pd-n\n"
          f"  (configs/pd.yaml: distill_teacher phải trỏ đúng {args.weights})")


if __name__ == "__main__":
    main()
