"""Model registry — map idea -> BaseModel subclass.

Thêm idea mới (3 bước, không sửa train.py):
  1. Tạo models/<ten>.py định nghĩa 1 class kế thừa BaseModel (xem baseline.py/lwso.py) —
     implement build(), override get_callbacks() nếu cần callback riêng.
  2. Thêm entry vào MODEL_REGISTRY bên dưới.
  3. Tạo configs/<ten>.yaml kế thừa configs/base.yaml (xem configs/lwso.yaml).

train.py lấy --idea choices trực tiếp từ MODEL_REGISTRY, nên không cần sửa gì ở đó.
"""

from __future__ import annotations

from models.base_model import BaseModel
from models.baseline import BaselineModel
from models.fap import FAPModel
from models.lwso import LWSOModel
from models.pd import PDModel
from models.slim import SlimModel
from models.star import StarModel

MODEL_REGISTRY: dict[str, type[BaseModel]] = {
    "baseline": BaselineModel,
    "lwso": LWSOModel,
    "fap": FAPModel,
    "star": StarModel,
    "slim": SlimModel,
    "pd": PDModel,
}


def build_model(idea: str, cfg) -> BaseModel:
    key = idea.lower()
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY)
        raise KeyError(f"idea={idea!r} chưa được đăng ký trong MODEL_REGISTRY. Có sẵn: {available}")
    return MODEL_REGISTRY[key](cfg)


__all__ = [
    "MODEL_REGISTRY", "build_model", "BaseModel", "BaselineModel", "LWSOModel", "FAPModel",
    "StarModel", "SlimModel", "PDModel",
]
