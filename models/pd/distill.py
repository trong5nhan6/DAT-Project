"""CWD (Channel-Wise Distillation, Shu et al., ICCV 2021) for the "pd" idea, wired into
ultralytics purely through callbacks + forward hooks — no fork, no trainer loss rewrite.

Why hooks and not a wrapped model.loss / custom criterion: anything attached to the model
*instance* (bound methods, closures) gets deepcopied into the EMA at trainer setup and then
pickled into best.pt/last.pt — closures are unpicklable and would crash save_model. Hooks
registered on `trainer.model` AFTER `ModelEMA(self.model)` has already deepcopied the model
(EMA is created in _setup_train; our hooks attach in the `on_train_start` callback, which
fires later, inside _do_train) never reach the EMA copy, and checkpoints are always
serialized from the EMA — so best.pt stays clean. Verified against ultralytics 8.3.253
ordering: _setup_train: ModelEMA(...) -> ... -> _do_train: run_callbacks("on_train_start").

Mechanics per training batch:
  1. Feature hooks on the student layers feeding Detect (indices from `detect.f`, so this
     adapts to any architecture) stash their outputs during the normal forward.
  2. A forward hook on the top-level DetectionModel fires after the loss forward
     (`model(batch)` with a dict input returns `(loss_vec * batch_size, loss_items)`).
     Inside it: teacher forward (no_grad) on the same preprocessed batch["img"] fills the
     teacher-side stashes, CWD is computed per scale, and the hook returns
     `(loss_vec + w * cwd * bs / numel, loss_items)` — sum() in the trainer then includes
     exactly `w * cwd * bs`, matching ultralytics' per-batch loss scaling convention.
     Non-dict inputs (predict/val paths) pass through untouched.

Channel mismatch between student and teacher taps (e.g. student slim: 48/96/128 vs
teacher lwso-eff: 64/128/128) is handled by lazy 1x1 adapter convs projecting student
features to teacher width. Adapters live OUTSIDE the model (inside the callback state):
they get gradients (registered into the trainer's optimizer via add_param_group) but are
never serialized into checkpoints — best.pt stays a plain model. When channels already
match (self-distill from the dense checkpoint a student was pruned from), no adapters
are created and CWD runs directly.

Constraints:
  - Single-GPU only: DDP spawns a fresh subprocess that never runs these callbacks.
    PDModel.train() guards this.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["cwd_loss", "build_distill_callbacks"]


def cwd_loss(fs: torch.Tensor, ft: torch.Tensor, tau: float = 4.0) -> torch.Tensor:
    """Channel-wise distillation: per-channel spatial softmax + KL(teacher || student).

    fs/ft: (B, C, H, W) with identical shapes (project fs through an adapter first if the
    raw widths differ). Computed in fp32 regardless of AMP dtype. Normalized by (B * C)
    as in the paper, scaled by tau^2 (standard KD temperature term).
    """
    if fs.shape != ft.shape:
        raise ValueError(
            f"CWD needs matching student/teacher feature shapes, got {tuple(fs.shape)} vs "
            f"{tuple(ft.shape)}"
        )
    b, c = fs.shape[:2]
    ps = F.log_softmax(fs.flatten(2).float() / tau, dim=2)
    pt = F.softmax(ft.flatten(2).float() / tau, dim=2)
    return F.kl_div(ps, pt, reduction="sum") * (tau**2) / (b * c)


def build_distill_callbacks(teacher_weights: str, weight: float, tau: float) -> dict:
    """Returns {event: callback} wiring CWD distillation into an ultralytics train run.

    Events used: on_train_start (build teacher + register hooks), on_train_epoch_end
    (log the running distill loss), on_train_end (teardown). Merge into the idea's
    get_callbacks() dict.
    """
    state = {
        "teacher": None,
        "adapters": {},  # tap index -> 1x1 conv (only when student/teacher widths differ)
        "handles": [],
        "s_feats": {},
        "t_feats": {},
        "epoch_sum": 0.0,
        "epoch_n": 0,
    }

    def _on_train_start(trainer):
        import torch.nn as nn
        from ultralytics.utils.torch_utils import unwrap_model

        from models.lwso.register import register_lwso

        register_lwso()  # teacher checkpoint has lwso custom modules; needed to unpickle
        from ultralytics import YOLO

        student = unwrap_model(trainer.model)
        teacher = YOLO(teacher_weights).model.float().to(trainer.device)
        teacher.eval()
        for p in teacher.parameters():
            p.requires_grad_(False)
        state["teacher"] = teacher

        taps = list(student.model[-1].f)  # layer indices feeding Detect, e.g. [16, 19, 22]
        t_taps = list(teacher.model[-1].f)
        if len(taps) != len(t_taps):
            raise RuntimeError(
                f"[pd] student has {len(taps)} detect scales but teacher has {len(t_taps)} "
                "— pick a teacher with the same P2/P3/P4 layout."
            )

        def _stash(store, key):
            def hook(_m, _inp, out):
                store[key] = out
            return hook

        for si, ti in zip(taps, t_taps):
            state["handles"].append(student.model[si].register_forward_hook(_stash(state["s_feats"], si)))
            state["handles"].append(teacher.model[ti].register_forward_hook(_stash(state["t_feats"], ti)))

        # Probe once to learn tap widths; build 1x1 adapters where they differ. Adapters
        # are trained jointly (added to the optimizer) but never touch the model object,
        # so checkpoints stay clean.
        was_training = student.training
        with torch.no_grad():
            imgsz = max(int(trainer.args.imgsz), 64)
            probe = torch.zeros(1, 3, imgsz, imgsz, device=trainer.device)
            student.eval()(probe)
            teacher(probe)
        student.train(was_training)
        adapter_params = []
        for si, ti in zip(taps, t_taps):
            cs, ct = state["s_feats"][si].shape[1], state["t_feats"][ti].shape[1]
            if cs != ct:
                a = nn.Conv2d(cs, ct, 1, bias=False).to(trainer.device).float()
                state["adapters"][si] = a
                adapter_params += list(a.parameters())
        state["s_feats"].clear()
        state["t_feats"].clear()
        if adapter_params:
            # initial_lr must be stamped manually: the LambdaLR scheduler only stamped
            # groups that existed at _setup_train, and ultralytics' warmup loop reads
            # x["initial_lr"] for EVERY group (KeyError otherwise). The scheduler must
            # also be extended in lockstep — torch>=2.12's LRScheduler.step() zips
            # param_groups with its per-group state strict=True and raises otherwise.
            # AMP stays correct because scaler.step(trainer.optimizer) unscales all of
            # its param groups.
            lr0 = float(trainer.args.lr0)
            trainer.optimizer.add_param_group(
                {"params": adapter_params, "lr": lr0, "initial_lr": lr0}
            )
            sched = getattr(trainer, "scheduler", None)
            if sched is not None and hasattr(sched, "lr_lambdas"):
                sched.lr_lambdas.append(trainer.lf)  # same cosine schedule as the model
                sched.base_lrs.append(lr0)
            print(f"[pd] channel-adapter 1x1 convs created for taps "
                  f"{sorted(state['adapters'])} (student/teacher widths differ)")

        def _add_distill(_module, inputs, output):
            batch = inputs[0] if inputs else None
            if not isinstance(batch, dict):  # predict/val forward, not a loss forward
                return None
            with torch.no_grad():
                teacher(batch["img"])
            terms = []
            for si, ti in zip(taps, t_taps):
                fs = state["s_feats"][si]
                if si in state["adapters"]:
                    fs = state["adapters"][si](fs)
                terms.append(cwd_loss(fs, state["t_feats"][ti], tau))
            d = torch.stack(terms).mean()
            state["s_feats"].clear()  # free activations promptly
            state["t_feats"].clear()
            state["epoch_sum"] += float(d)
            state["epoch_n"] += 1
            loss_vec, loss_items = output
            bs = batch["img"].shape[0]
            # loss_vec is summed by the trainer; spread the scaled distill term across its
            # elements so sum() adds exactly weight * d * bs (ultralytics' bs convention).
            return (loss_vec + (weight * d * bs) / loss_vec.numel(), loss_items)

        state["handles"].append(student.register_forward_hook(_add_distill))
        print(f"[pd] CWD distillation active: teacher={teacher_weights}, "
              f"weight={weight}, tau={tau}, taps={taps} -> teacher taps {t_taps}")

    def _on_train_epoch_end(trainer):
        if state["epoch_n"]:
            avg = state["epoch_sum"] / state["epoch_n"]
            print(f"[pd] epoch {trainer.epoch + 1}: mean CWD distill loss = {avg:.4f}")
        state["epoch_sum"] = 0.0
        state["epoch_n"] = 0

    def _on_train_end(_trainer):
        for h in state["handles"]:
            h.remove()
        state["handles"].clear()
        state["teacher"] = None

    return {
        "on_train_start": _on_train_start,
        "on_train_epoch_end": _on_train_epoch_end,
        "on_train_end": _on_train_end,
    }
