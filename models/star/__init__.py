"""Star idea package: StarNet-style backbone + GSConv slim-neck + Wise-IoU v3.

Submodules:
    modules.py  - StarBlock/C3k2Star (backbone), SimAM (attention), GSConv/VoVGSCSP (neck)
    register.py - register_star(): runtime registration into ultralytics parse_model
    losses.py    - WiseIoULoss (Wise-IoU v3) + patch_wiou_loss()
    model.py      - StarModel: the idea class, drives build()/train() for --idea star

Only StarModel is re-exported here (import-light: no torch/ultralytics at package init),
matching models/lwso/__init__.py's and models/fap/__init__.py's rationale.
"""

from .model import StarModel

__all__ = ["StarModel"]
