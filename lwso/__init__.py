"""LWSO-YOLO: lightweight small-object YOLO for VisDrone2019, built on ultralytics."""

from .losses import NWDBboxLoss, nwd, patch_nwd_loss
from .modules import BiFPNCat, C3k2Ghost, DySample, EMA, SPDConv
from .register import register_lwso

__all__ = [
    "SPDConv",
    "C3k2Ghost",
    "EMA",
    "DySample",
    "BiFPNCat",
    "register_lwso",
    "patch_nwd_loss",
    "NWDBboxLoss",
    "nwd",
]
