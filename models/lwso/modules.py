"""Custom building blocks for LWSO-YOLO (LightWeight Small-Object YOLO).

Modules:
    SPDConv      - space-to-depth downsampling (no information loss, SPD-Conv paper).
    SPDConvGroup - SPDConv with a grouped (not dense) fusion conv; same lossless
                   space-to-depth but ~4-8x cheaper post-conv (LRDS-YOLO LAD-style).
    C3k2Ghost    - C3k2 block with GhostBottleneck inner blocks (~40% fewer FLOPs).
    EMA          - Efficient Multi-scale Attention (Ouyang et al., ICASSP 2023).
    ECA          - Efficient Channel Attention (Wang et al., CVPR 2020); near-zero
                   cost (1D conv over pooled channels, no reduction MLP).
    DySample     - dynamic learnable upsampler, 'lp' style (Liu et al., ICCV 2023).
    BiFPNCat     - weighted feature concat (BiFPN fast normalized fusion), drop-in
                   replacement for Concat in YAML.

All channel-changing modules follow the ultralytics (c1, c2, *args) convention so
they can be registered into parse_model's `base_modules` set (see register.py).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.block import C2f, GhostBottleneck
from ultralytics.nn.modules.conv import Conv

__all__ = ["SPDConv", "SPDConvGroup", "C3k2Ghost", "EMA", "ECA", "DySample", "BiFPNCat"]


def _space_to_depth(x: torch.Tensor) -> torch.Tensor:
    """Rearrange each 2x2 spatial patch into 4 channels -- halves H/W like a stride-2
    conv/pool but discards no pixels, which matters for objects only a few pixels wide.
    """
    if x.shape[-1] % 2 or x.shape[-2] % 2:  # pad odd H/W to even so the 2x2 slices align
        x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
    return torch.cat(
        [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
    )


class SPDConv(nn.Module):
    """Space-to-depth downsample, fused by a dense stride-1 conv over all 4*c1 channels."""

    def __init__(self, c1: int, c2: int, k: int = 3):
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, 1)

    def forward(self, x):
        return self.conv(_space_to_depth(x))


class SPDConvGroup(nn.Module):
    """SPDConv with a grouped fusion conv instead of dense: same lossless space-to-depth,
    but the post-rearrange conv only mixes within `groups` instead of across all 4*c1
    channels. A dense SPDConv fusion conv is the single largest compute driver in
    lwso-yolo11n.yaml (measured: +38% GFLOPs vs a stock stride-2 Conv at the same spot,
    because it runs a full conv over 4x the channels) -- grouping it recovers most of
    that cost without discarding the lossless rearrange. Falls back to fewer groups if
    4*c1 or c2 isn't evenly divisible by `groups`.
    """

    def __init__(self, c1: int, c2: int, k: int = 3, groups: int = 8):
        super().__init__()
        c4 = c1 * 4
        g = groups
        while g > 1 and (c4 % g or c2 % g):
            g -= 1
        self.conv = Conv(c4, c2, k, 1, g=g)

    def forward(self, x):
        return self.conv(_space_to_depth(x))


class C3k2Ghost(C2f):
    """C3k2 with GhostBottleneck inner blocks.

    Signature mirrors C3k2(c1, c2, n, c3k, e, g, shortcut) so YAML args stay compatible;
    `c3k` is accepted but ignored (Ghost blocks are always used).
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GhostBottleneck(self.c, self.c) for _ in range(n))


class EMA(nn.Module):
    """Efficient Multi-scale Attention. Channel-preserving (c2 must equal c1);
    the c2 arg exists only to satisfy the (c1, c2, ...) registration convention.
    """

    def __init__(self, c1: int, c2: int = None, factor: int = 8):
        super().__init__()
        assert c2 is None or c2 == c1, f"EMA requires c1 == c2, got {c1} != {c2}"
        assert c1 % factor == 0, f"EMA: channels {c1} must be divisible by factor {factor}"
        self.groups = factor
        self.softmax = nn.Softmax(-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.gn = nn.GroupNorm(c1 // self.groups, c1 // self.groups)
        self.conv1x1 = nn.Conv2d(c1 // self.groups, c1 // self.groups, 1)
        self.conv3x3 = nn.Conv2d(c1 // self.groups, c1 // self.groups, 3, padding=1)

    def forward(self, x):
        b, c, h, w = x.size()
        group_x = x.reshape(b * self.groups, -1, h, w)
        x_h = self.pool_h(group_x)
        x_w = self.pool_w(group_x).permute(0, 1, 3, 2)
        hw = self.conv1x1(torch.cat([x_h, x_w], dim=2))
        x_h, x_w = torch.split(hw, [h, w], dim=2)
        x1 = self.gn(group_x * x_h.sigmoid() * x_w.permute(0, 1, 3, 2).sigmoid())
        x2 = self.conv3x3(group_x)
        x11 = self.softmax(self.agp(x1).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x12 = x2.reshape(b * self.groups, c // self.groups, -1)
        x21 = self.softmax(self.agp(x2).reshape(b * self.groups, -1, 1).permute(0, 2, 1))
        x22 = x1.reshape(b * self.groups, c // self.groups, -1)
        weights = (torch.matmul(x11, x12) + torch.matmul(x21, x22)).reshape(
            b * self.groups, 1, h, w
        )
        return (group_x * weights.sigmoid()).reshape(b, c, h, w)


class ECA(nn.Module):
    """Efficient Channel Attention (Wang et al., CVPR 2020). Channel-preserving; c2 arg
    exists only for the (c1, c2, ...) registration convention. Kernel size is derived
    from the channel count (no SE-style reduction MLP), so cost is a single 1D conv over
    globally-pooled channels -- a few hundred params, negligible FLOPs. Meant for spots
    that currently have no attention at all (e.g. the P2 output, where small objects
    live) without meaningfully touching the compute budget.
    """

    def __init__(self, c1: int, c2: int = None, gamma: int = 2, b: int = 1):
        super().__init__()
        assert c2 is None or c2 == c1, f"ECA requires c1 == c2, got {c1} != {c2}"
        t = int(abs((math.log2(c1) + b) / gamma))
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)  # (B, C, 1, 1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))  # (B, 1, C) -> conv1d over channels
        y = y.transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * self.sigmoid(y)


def _normal_init(module: nn.Module, mean: float = 0.0, std: float = 1.0, bias: float = 0.0):
    if hasattr(module, "weight") and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


class DySample(nn.Module):
    """DySample dynamic upsampler ('lp' style). Channel-preserving; c2 exists only for
    the (c1, c2, ...) registration convention. Near-zero cost replacement for
    nn.Upsample(nearest) with learned sampling offsets.
    """

    def __init__(self, c1: int, c2: int = None, scale: int = 2, groups: int = 4):
        super().__init__()
        assert c2 is None or c2 == c1, f"DySample requires c1 == c2, got {c1} != {c2}"
        assert c1 % groups == 0, f"DySample: channels {c1} must be divisible by groups {groups}"
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(c1, 2 * groups * scale**2, 1)
        _normal_init(self.offset, std=0.001)
        self.register_buffer("init_pos", self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return (
            torch.stack(torch.meshgrid([h, h], indexing="ij"))
            .transpose(1, 2)
            .repeat(1, self.groups, 1)
            .reshape(1, -1, 1, 1)
        )

    def _sample(self, x, offset):
        b, _, h, w = offset.shape
        offset = offset.view(b, 2, -1, h, w)
        coords_h = torch.arange(h, device=x.device) + 0.5
        coords_w = torch.arange(w, device=x.device) + 0.5
        coords = (
            torch.stack(torch.meshgrid([coords_w, coords_h], indexing="ij"))
            .transpose(1, 2)
            .unsqueeze(1)
            .unsqueeze(0)
            .type(x.dtype)
        )
        normalizer = torch.tensor([w, h], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = (
            F.pixel_shuffle(coords.view(b, -1, h, w), self.scale)
            .view(b, 2, -1, self.scale * h, self.scale * w)
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .flatten(0, 1)
        )
        return F.grid_sample(
            x.reshape(b * self.groups, -1, h, w),
            coords,
            mode="bilinear",
            align_corners=False,
            padding_mode="border",
        ).view(b, -1, self.scale * h, self.scale * w)

    def forward(self, x):
        offset = self.offset(x) * 0.25 + self.init_pos
        return self._sample(x, offset)


class BiFPNCat(nn.Module):
    """Weighted concat of n feature maps (BiFPN fast normalized fusion).

    Drop-in for Concat in model YAML: `[[-1, 5], 1, BiFPNCat, [2]]` where the arg
    is the number of inputs. Output channels = sum of input channels, handled by
    the patched parse_model Concat branch.
    """

    def __init__(self, n: int = 2, dimension: int = 1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(n), requires_grad=True)
        self.eps = 1e-4

    def forward(self, x):
        w = F.relu(self.w)
        w = w / (w.sum() + self.eps)
        return torch.cat([w[i] * xi for i, xi in enumerate(x)], self.d)
