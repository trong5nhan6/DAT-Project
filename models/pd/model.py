"""Idea: pd — "prune/distill": train ANY student with CWD distillation from a stronger
teacher checkpoint. Two student modes, selected by what model_cfg points at:

  * .pt  — a structurally pruned checkpoint from prune_structured.py ("train big ->
    structured prune -> distill back"). A pruned network has irregular per-layer channel
    counts the ultralytics YAML format cannot express; the pickled nn.Module inside the
    .pt IS the architecture. Ultralytics' stock trainer would silently destroy it:
    Model.train() calls trainer.get_model(cfg=self.model.yaml, weights=self.model),
    which rebuilds the DENSE architecture from the stale yaml dict and then load()s the
    pruned state_dict through intersect_dicts — every shape-mismatched (i.e. pruned)
    tensor is dropped and re-initialized. PDTrainer overrides get_model to return the
    already-unpickled pruned module directly (setup_model then short-circuits, since it
    returns immediately for nn.Module). Checkpointing still works unchanged: save_model
    serializes the EMA (a deepcopy of this module), and final_eval/AutoBackend load
    best.pt by unpickling.

  * .yaml — a normal architecture file (e.g. cfg/slim-yolo11n.yaml), trained from
    scratch (or warm-started via `weights:`) but WITH distillation. This is the flagship
    combo for the efficiency target: slim (0.56M / 6.31 GFLOPs@640) + CWD from the
    trained lwso-eff (mAP50 31.8). Channel-width differences at the detect taps are
    absorbed by 1x1 adapters (models/pd/distill.py). Uses the stock trainer path.

CWD is attached through callbacks + forward hooks (models/pd/distill.py) — see that
file for why hooks, and for the EMA/checkpoint-pickling safety argument.

Limitations (guarded, not silent): no --resume for .pt students, no multi-GPU (DDP
subprocess would rebuild from yaml / never run the distill callbacks), no compile (its
loss path bypasses the top-level forward hook).
"""

from __future__ import annotations

from pathlib import Path

from models.base_model import BaseModel


def _pd_trainer_cls():
    """Build the PDTrainer class lazily (keeps module import light for MODEL_REGISTRY)."""
    from ultralytics.models.yolo.detect import DetectionTrainer

    class PDTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            """Return the unpickled pruned module as-is — never rebuild from yaml."""
            if weights is None:
                raise RuntimeError(
                    "[pd] trainer got no weights — model_cfg must be a pruned .pt "
                    "checkpoint (from prune_structured.py), not a .yaml."
                )
            return weights.float()

    return PDTrainer


class PDModel(BaseModel):
    def build(self) -> None:
        from models.fap.register import register_fap
        from models.lwso.losses import patch_nwd_loss
        from models.lwso.register import register_lwso
        from models.slim.register import register_slim
        from models.star.register import register_star

        model_path = str(self.cfg.model_cfg)
        self._pruned_pt = model_path.endswith(".pt")
        if self._pruned_pt and not Path(model_path).exists():
            raise FileNotFoundError(
                f"[pd] pruned checkpoint not found: {model_path} — run prune_structured.py "
                "first, or point --model at its output (or at a .yaml like "
                "cfg/slim-yolo11n.yaml to distill-train an architecture from scratch)."
            )

        # student/teacher may come from any idea family — register everything (idempotent)
        register_lwso()
        register_fap()
        register_star()
        register_slim()
        if self.cfg.get("use_nwd", False):
            ratio = float(self.cfg.get("nwd_ratio", 0.5))
            constant = float(self.cfg.get("nwd_constant", 12.8))
            patch_nwd_loss(ratio=ratio, constant=constant)
            print(f"[pd] NWD blend loss active: ratio={ratio}, C={constant}")

        self._yolo = self._build_yolo()

    def get_callbacks(self):
        from .distill import build_distill_callbacks

        callbacks = dict(super().get_callbacks())
        teacher = self.cfg.get("distill_teacher")
        if teacher:
            callbacks.update(
                build_distill_callbacks(
                    str(teacher),
                    weight=float(self.cfg.get("distill_weight", 1.0)),
                    tau=float(self.cfg.get("distill_tau", 4.0)),
                )
            )
        else:
            print("[pd] distill_teacher not set — fine-tuning WITHOUT distillation")
        return callbacks

    def train(self, **train_kwargs) -> None:
        device = str(train_kwargs.get("device") or "")
        if "," in device:
            raise NotImplementedError(
                "[pd] multi-GPU is not supported: the DDP subprocess rebuilds the model "
                "from yaml (destroying the pruned architecture) and never runs the "
                "distill callbacks. Use a single device, e.g. --device 0."
            )
        if self._pruned_pt:
            if train_kwargs.get("resume"):
                raise NotImplementedError(
                    "[pd] --resume is not supported for pruned .pt students: ultralytics' "
                    "resume path reloads the checkpoint through the yaml rebuild that pd "
                    "exists to avoid."
                )
            # Model.train(trainer=...) hands the pruned module straight to the trainer
            train_kwargs["trainer"] = _pd_trainer_cls()
        super().train(**train_kwargs)
