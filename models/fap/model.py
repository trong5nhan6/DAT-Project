"""Idea: fap — FAP-YOLO12n's representation stage ported to YOLO11n (FreqMix downsampling
on a stock backbone with a P2 head added, P3/P4/P5 kept). The paper's second contribution
(semantic-path-aware LAMP pruning) is a separate post-training step — see prune.py at the
repo root, not part of build()/train() here.
"""

from __future__ import annotations

from models.base_model import BaseModel


class FAPModel(BaseModel):
    def build(self) -> None:
        from .register import register_fap

        register_fap()  # phải chạy trước YOLO(...) để parse_model nhận diện FreqMix
        self._yolo = self._build_yolo()
