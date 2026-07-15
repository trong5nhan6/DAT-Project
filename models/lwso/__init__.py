"""LWSO idea package: lightweight small-object YOLO for VisDrone2019, built on ultralytics.

Submodules:
    modules.py   - SPDConv, C3k2Ghost, EMA, DySample, BiFPNCat (custom nn.Module blocks)
    register.py  - register_lwso(): runtime registration into ultralytics parse_model
    losses.py    - NWDBboxLoss / patch_nwd_loss(): CIoU-NWD blend loss
    model.py     - LWSOModel: the idea class, drives build()/train() for --idea lwso

Only LWSOModel is re-exported here (import-light: no torch/ultralytics at package
init) so `from models import MODEL_REGISTRY` stays cheap for train.py's argparse
setup. Import modules/register/losses directly, e.g. `from models.lwso.modules
import SPDConv`, when you actually need them (tests, val.py, ...).
"""

from .model import LWSOModel

__all__ = ["LWSOModel"]
