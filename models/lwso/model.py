"""Idea: lwso — full LWSO-YOLO11n (SPDConv/C3k2Ghost/EMA/DySample/BiFPNCat)
+ optional CIoU-NWD blend loss. See modules.py/register.py/losses.py in this
package for the building blocks; this file is just the idea class train.py
drives via models.build_model().
"""

from __future__ import annotations

from models.base_model import BaseModel


class LWSOModel(BaseModel):
    def build(self) -> None:
        # Imported here, not at module top, so `from models.lwso import LWSOModel`
        # (needed just to populate MODEL_REGISTRY / --idea choices) stays cheap —
        # torch/ultralytics only load once an lwso run is actually being built.
        from .losses import patch_nwd_loss
        from .register import register_lwso

        register_lwso()  # phải chạy trước YOLO(...) để parse_model nhận diện module custom
        if self.cfg.get("use_nwd", False):
            ratio = float(self.cfg.get("nwd_ratio", 0.5))
            constant = float(self.cfg.get("nwd_constant", 12.8))
            patch_nwd_loss(ratio=ratio, constant=constant)
            print(f"[lwso] NWD blend loss active: ratio={ratio}, C={constant}")

        self._yolo = self._build_yolo()
