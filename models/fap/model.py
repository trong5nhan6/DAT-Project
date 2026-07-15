"""Idea: fap — FAP-YOLO12n's representation stage ported to YOLO11n (FreqMix downsampling
on a stock backbone with a P2 head added, P3/P4/P5 kept). The paper's second contribution
(semantic-path-aware LAMP pruning) is a separate post-training step — see prune.py at the
repo root. When cfg.sparsity_mask points at a prune.py-produced <out>.mask.pt, this class
also re-applies that mask throughout training (see _sparsity.py) so a fine-tune after
pruning can't undo the sparsity via gradient drift.
"""

from __future__ import annotations

from models.base_model import BaseModel


class FAPModel(BaseModel):
    def build(self) -> None:
        from .register import register_fap

        register_fap()  # phải chạy trước YOLO(...) để parse_model nhận diện FreqMix
        self._yolo = self._build_yolo()

        mask_path = self.cfg.get("sparsity_mask")
        if mask_path:
            from ._sparsity import build_sparsity_mask_callbacks

            apply, _ = build_sparsity_mask_callbacks(str(mask_path))
            apply(self._yolo.model)
            print(f"[fap] sparsity mask applied at load -> {mask_path}")

    def get_callbacks(self) -> dict:
        callbacks = super().get_callbacks()
        mask_path = self.cfg.get("sparsity_mask")
        if mask_path:
            from ._sparsity import build_sparsity_mask_callbacks

            _, on_train_batch_end = build_sparsity_mask_callbacks(str(mask_path))
            print(f"[fap] sparsity mask enabled -> {mask_path} "
                  f"(re-applied after every optimizer step during fine-tune)")
            callbacks["on_train_batch_end"] = on_train_batch_end
        return callbacks
