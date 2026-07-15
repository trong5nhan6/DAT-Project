"""FAP idea package: frequency-aware downsampling (FreqMix) ported from FAP-YOLO12n.

Submodules:
    modules.py  - FreqMix (Haar band decompose + learnable softmax mixing + 1x1 proj)
    register.py - register_fap(): runtime registration into ultralytics parse_model
    model.py    - FAPModel: the idea class, drives build()/train() for --idea fap

Only FAPModel is re-exported here (import-light: no torch/ultralytics at package init),
matching models/lwso/__init__.py's rationale — see that file for why.
"""

from .model import FAPModel

__all__ = ["FAPModel"]
