"""FreqMix — frequency-aware downsampling (FAP-YOLO12n, Sec 4.1).

Haar wavelet decomposition (LL/LH/HL/HH, a fixed non-learnable transform) followed by
softmax-weighted *learnable mixing* across the 4 bands, then a 1x1 conv projection.

Why mixing beats HWD's concat+conv (paper's own reasoning, kept here for reference):
  1. concat inflates channels 4x, pushing the redundancy-resolution work onto the conv
     that follows; a weighted sum keeps the post-transform channel count at c1, so the
     projection conv is c1->c2 instead of 4*c1->c2 -- fewer params for the same job.
  2. soft mixing lets each position in the network learn its own LL/LH/HL/HH emphasis,
     instead of a fixed concat order the conv has to learn to interpret every time.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.nn.modules.conv import Conv

__all__ = ["FreqMix"]


class FreqMix(nn.Module):
    """Haar-band decompose (fixed) + learnable softmax mixing + 1x1 conv projection.

    band_logits initialized to [2.0, 0, 0, -0.5] (paper Sec 4.1): softmax(...) starts at
    roughly [0.85, 0.06, 0.06, 0.03], i.e. training starts close to a plain low-pass
    (LL/average-pool) downsample and only pulls in the LH/HL/HH detail bands as gradients
    favor it, rather than starting from an untrained, uniform mix of all 4 bands.
    """

    def __init__(self, c1: int, c2: int, k: int = 1):
        super().__init__()
        self.band_logits = nn.Parameter(torch.tensor([2.0, 0.0, 0.0, -0.5]))
        self.proj = Conv(c1, c2, k, 1)

    def _bands(self, x: torch.Tensor) -> torch.Tensor:
        """2D Haar transform over each 2x2 patch -> (4, B, C, H/2, W/2)."""
        if x.shape[-1] % 2 or x.shape[-2] % 2:  # pad odd H/W so 2x2 patches align
            x = F.pad(x, (0, x.shape[-1] % 2, 0, x.shape[-2] % 2))
        p1 = x[..., ::2, ::2]  # top-left
        p2 = x[..., 1::2, ::2]  # bottom-left
        p3 = x[..., ::2, 1::2]  # top-right
        p4 = x[..., 1::2, 1::2]  # bottom-right
        ll = 0.5 * (p1 + p2 + p3 + p4)  # approximation / low-frequency
        lh = 0.5 * (p1 - p2 + p3 - p4)  # row-gradient detail
        hl = 0.5 * (p1 + p2 - p3 - p4)  # column-gradient detail
        hh = 0.5 * (p1 - p2 - p3 + p4)  # diagonal detail
        return torch.stack([ll, lh, hl, hh], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bands = self._bands(x)
        alpha = F.softmax(self.band_logits, dim=0).view(4, 1, 1, 1, 1)
        mixed = (alpha * bands).sum(0)
        return self.proj(mixed)
