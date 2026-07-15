"""Idea: baseline — stock YOLO11n, no custom modules, no NWD loss.

Reference point cho mọi idea khác. cfg.model_cfg thường là yolo11n.pt (finetune
từ COCO-pretrained) hoặc cfg/base-yolo11n.yaml (train from-scratch, cùng recipe
với các idea khác để so sánh công bằng hơn) — xem configs/baseline.yaml.
"""

from __future__ import annotations

from models.base_model import BaseModel


class BaselineModel(BaseModel):
    def build(self) -> None:
        self._yolo = self._build_yolo()
