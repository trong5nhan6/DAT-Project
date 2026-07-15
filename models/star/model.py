"""Idea: star — StarNet-style backbone (StarBlock + RepConv reparam), parameter-free SimAM
attention, GSConv slim-neck, and Wise-IoU v3 loss. Synthesized from recent (2024-2025)
lightweight-detection literature rather than ported from one single paper (contrast with
lwso/fap) — see modules.py/register.py/losses.py in this package for the building blocks;
this file is just the idea class train.py drives via models.build_model().
"""

from __future__ import annotations

from models.base_model import BaseModel


class StarModel(BaseModel):
    def build(self) -> None:
        # Imported here, not at module top, so `from models.star import StarModel` (needed
        # just to populate MODEL_REGISTRY / --idea choices) stays cheap — torch/ultralytics
        # only load once a star run is actually being built.
        from .losses import patch_wiou_loss
        from .register import register_star

        register_star()  # phải chạy trước YOLO(...) để parse_model nhận diện module custom
        patch_wiou_loss()  # luôn bật cho idea này (không có toggle như use_nwd của lwso)

        self._yolo = self._build_yolo()
