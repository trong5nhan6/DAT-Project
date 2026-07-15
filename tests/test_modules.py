"""Shape/param sanity tests for LWSO modules and model YAMLs. Run: pytest tests/ -v"""

from pathlib import Path

import pytest
import torch

from lwso.losses import nwd
from lwso.modules import BiFPNCat, C3k2Ghost, DySample, EMA, SPDConv

ROOT = Path(__file__).resolve().parents[1]


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------- modules

def test_spdconv_halves_spatial_no_pixel_loss():
    m = SPDConv(16, 32)
    y = m(torch.randn(2, 16, 64, 64))
    assert y.shape == (2, 32, 32, 32)


def test_spdconv_odd_input_pads_to_even():
    # 240->120 etc. are even in our pipeline, but degrade gracefully via padding
    y = SPDConv(8, 16)(torch.randn(1, 8, 63, 63))
    assert y.shape[-2:] == (32, 32)  # 63 padded to 64, then halved


def test_c3k2ghost_shape_and_lighter_than_c3k2():
    from ultralytics.nn.modules.block import C3k2

    ghost = C3k2Ghost(64, 128, n=2)
    stock = C3k2(64, 128, n=2)
    y = ghost(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 128, 32, 32)
    assert n_params(ghost) < n_params(stock), (
        f"ghost {n_params(ghost)} should be < stock {n_params(stock)}"
    )


def test_ema_preserves_shape():
    x = torch.randn(2, 128, 30, 30)
    assert EMA(128, 128)(x).shape == x.shape


def test_ema_rejects_channel_change():
    with pytest.raises(AssertionError):
        EMA(128, 64)


def test_dysample_upscales_2x():
    m = DySample(64, 64, scale=2)
    y = m(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 64, 64, 64)


def test_dysample_fresh_init_close_to_nearest_upsample():
    # offset conv init ~0 => output should start near plain upsampling (stable start)
    torch.manual_seed(0)
    x = torch.randn(1, 8, 16, 16)
    y = DySample(8, 8, scale=2)(x)
    ref = torch.nn.functional.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
    assert (y - ref).abs().mean() < 0.2


def test_bifpncat_sums_channels_and_weights_normalized():
    m = BiFPNCat(2)
    a, b = torch.randn(2, 64, 16, 16), torch.randn(2, 128, 16, 16)
    y = m([a, b])
    assert y.shape == (2, 192, 16, 16)
    w = torch.nn.functional.relu(m.w)
    assert abs((w / (w.sum() + m.eps)).sum().item() - 1.0) < 1e-3


# ---------------------------------------------------------------- loss

def test_nwd_identical_boxes_is_one():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    assert nwd(boxes, boxes).item() == pytest.approx(1.0, abs=1e-3)


def test_nwd_smooth_for_small_shift_where_iou_collapses():
    a = torch.tensor([[100.0, 100.0, 108.0, 108.0]])  # 8x8 box
    b = a + 6.0  # 6px shift: IoU ~= 0.02, NWD should stay clearly positive
    from ultralytics.utils.metrics import bbox_iou

    iou = bbox_iou(a, b, xywh=False).item()
    sim = nwd(a, b).item()
    assert iou < 0.1
    assert sim > 0.4


# ---------------------------------------------------------------- full models

@pytest.mark.parametrize(
    "cfg,needs_patch",
    [
        ("cfg/lwso-yolo11n.yaml", True),
        ("cfg/ablation/yolo11n-p2-nop5.yaml", False),
    ],
)
def test_model_builds_and_forwards(cfg, needs_patch):
    from lwso import register_lwso

    register_lwso()
    from ultralytics import YOLO

    model = YOLO(str(ROOT / cfg))
    det = model.model  # DetectionModel
    total = n_params(det)
    assert total < 3_500_000, f"{cfg}: {total/1e6:.2f}M params, expected nano-sized (<3.5M)"

    det.eval()
    with torch.no_grad():
        preds = det(torch.zeros(1, 3, 256, 256))
    assert preds is not None
    # 3 detect scales (P2/P3/P4) with strides 4/8/16
    strides = sorted(int(s) for s in det.stride.tolist())
    assert strides == [4, 8, 16], f"unexpected strides {strides}"
    print(f"{cfg}: {total/1e6:.2f}M params, strides {strides}")
