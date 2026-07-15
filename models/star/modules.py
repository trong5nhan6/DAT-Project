"""Custom building blocks for idea "star" — StarNet-style backbone block + reparam,
parameter-free attention, and GSConv-based slim-neck.

Modules:
    StarBlock  - StarNet-style element-wise-multiply block (Ma et al., "Rewrite the
                 Stars", arXiv:2403.19967), spatial-mix conv is a reparam RepConv.
    C3k2Star   - C3k2 with StarBlock inner blocks (backbone).
    SimAM      - parameter-free 3D attention (Yang et al., ICML 2021), 0 nn.Parameter.
    GSConv     - dense-half + depthwise-half conv + channel shuffle (Li et al.,
                 "Slim-neck by GSConv", arXiv:2206.02424).
    GSBottleneck / VoVGSCSP - GSConv-based CSP block (neck).

SPDConvGroup/DySample/BiFPNCat are reused unchanged from models.lwso.modules (see
register.py) — not redefined here.

All channel-changing modules follow the ultralytics (c1, c2, *args) convention so
they can be registered into parse_model's `base_modules` set (see register.py).
"""

import torch
import torch.nn as nn

from ultralytics.nn.modules.block import C2f
from ultralytics.nn.modules.conv import Conv, RepConv

__all__ = ["StarBlock", "C3k2Star", "SimAM", "GSConv", "GSBottleneck", "VoVGSCSP"]


class StarBlock(nn.Module):
    """StarNet-style block: element-wise multiplication of two 1x1-conv branches maps the
    input into an implicit high-dimensional non-linear feature space (kernel-trick-like)
    without widening any conv's actual channel count — far cheaper than reaching the same
    representational capacity by literally widening a Bottleneck/GhostBottleneck.

    Simplified vs. the paper's DemoNet block (single depthwise spatial-mix conv instead of
    one before *and* after the star operation) to keep the block cheap and the YAML args
    trivial (channel-preserving, no extra hyperparams to plumb through C3k2-style repeats).

    Spatial-mix conv is `RepConv` (ultralytics' own reparam conv: trained as 3x3 + 1x1 +
    identity branches, fused into a single 3x3 at inference via `.fuse_convs()`, which
    ultralytics' `model.fuse()` calls automatically for any module that has it) run
    depthwise (`g=c1`) — recovers capacity lost to the channel-count cut below at zero
    extra inference-time cost. Activation kept as SiLU (ultralytics' default) rather than
    the paper's GELU, for consistency with the rest of this codebase.
    """

    def __init__(self, c1: int, c2: int, mlp_ratio: int = 2):
        super().__init__()
        assert c1 == c2, f"StarBlock is channel-preserving, got c1={c1} != c2={c2}"
        c_ = c1 * mlp_ratio
        self.dw = RepConv(c1, c1, k=3, g=c1, act=False)
        self.f1 = Conv(c1, c_, 1, act=False)
        self.f2 = Conv(c1, c_, 1, act=True)
        self.proj = Conv(c_, c2, 1, act=False)

    def forward(self, x):
        y = self.dw(x)
        y = self.proj(self.f1(y) * self.f2(y))
        return x + y


class C3k2Star(C2f):
    """C3k2 with StarBlock inner blocks.

    Signature mirrors C3k2Ghost (models/lwso/modules.py) for YAML compatibility;
    `c3k` is accepted but ignored (StarBlocks are always used).
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(StarBlock(self.c, self.c) for _ in range(n))


class SimAM(nn.Module):
    """SimAM (Yang et al., ICML 2021): parameter-free 3D attention derived from a
    neuroscience-inspired energy function (a neuron with more distinctive activity from
    its spatial neighbours gets a lower "energy" / higher attention weight). No
    nn.Parameter anywhere — strictly cheaper than ECA (models/lwso/modules.py), which has
    a small Conv1d. Channel-preserving; c2 kept only for the (c1, c2, ...) registration
    convention, same as ECA/EMA/DySample.
    """

    def __init__(self, c1: int = None, c2: int = None, e_lambda: float = 1e-4):
        super().__init__()
        assert c1 is None or c2 is None or c1 == c2, f"SimAM requires c1 == c2, got {c1} != {c2}"
        self.e_lambda = e_lambda
        self.act = nn.Sigmoid()

    def forward(self, x):
        n = x.shape[2] * x.shape[3] - 1
        x_minus_mu_sq = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_sq / (4 * (x_minus_mu_sq.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        return x * self.act(y)


class GSConv(nn.Module):
    """GSConv (Li et al., "Slim-neck by GSConv", arXiv:2206.02424): a dense conv on half
    the output channels, a depthwise conv on that half, concatenated and channel-shuffled.
    Keeps most of a plain Conv(c1, c2, k, s)'s cross-channel mixing quality at roughly half
    the params/FLOPs — designed by the paper specifically for necks (already-rich
    concatenated features), not backbones.
    """

    def __init__(self, c1: int, c2: int, k: int = 1, s: int = 1):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s)
        self.cv2 = Conv(c_, c_, 5, 1, g=c_)  # depthwise

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        b, n, h, w = x2.shape
        return x2.view(b, 2, n // 2, h, w).permute(0, 2, 1, 3, 4).reshape(b, n, h, w)


class GSBottleneck(nn.Module):
    """1 GSConv stage: 1x1 GSConv projection then 3x3 GSConv refine, residual if
    c1 == c2. Inner repeat unit for VoVGSCSP, mirrors Bottleneck's role inside C2f.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.cv1 = GSConv(c1, c2, 1, 1)
        self.cv2 = GSConv(c2, c2, 3, 1)
        self.add = c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class VoVGSCSP(C2f):
    """Slim-neck's VoV-GSCSP: a C3k2-style CSP wrapper around GSBottleneck instead of
    Bottleneck/GhostBottleneck/StarBlock. Used in the neck (after each BiFPNCat concat) in
    place of C3k2Ghost/C3k2Star — GSConv was designed by its authors for necks specifically.

    Signature mirrors C3k2Ghost/C3k2Star for YAML compatibility; `c3k` accepted but ignored.
    """

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(GSBottleneck(self.c, self.c) for _ in range(n))
