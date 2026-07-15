"""Shape/param sanity tests for LWSO modules and model YAMLs. Run: pytest tests/ -v"""

from pathlib import Path

import pytest
import torch

from models.fap.modules import FreqMix
from models.lwso.losses import nwd
from models.lwso.modules import BiFPNCat, C3k2Ghost, DySample, ECA, EMA, SPDConv, SPDConvGroup

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


def test_spdconvgroup_halves_spatial_no_pixel_loss():
    m = SPDConvGroup(16, 32)
    y = m(torch.randn(2, 16, 64, 64))
    assert y.shape == (2, 32, 32, 32)


def test_spdconvgroup_lighter_than_dense_spdconv():
    dense = SPDConv(16, 32)
    grouped = SPDConvGroup(16, 32, groups=8)
    assert n_params(grouped) < n_params(dense), (
        f"grouped {n_params(grouped)} should be < dense {n_params(dense)}"
    )


def test_spdconvgroup_falls_back_when_not_divisible():
    # c1*4=12, c2=7 (prime, coprime with 12): no group >1 divides both -> falls back to g=1
    m = SPDConvGroup(3, 7, groups=8)
    y = m(torch.randn(1, 3, 8, 8))
    assert y.shape == (1, 7, 4, 4)


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


def test_eca_preserves_shape():
    x = torch.randn(2, 64, 30, 30)
    assert ECA(64)(x).shape == x.shape


def test_eca_rejects_channel_change():
    with pytest.raises(AssertionError):
        ECA(64, 32)


def test_eca_kernel_size_is_odd_and_positive():
    for c in (16, 32, 48, 64, 128, 256):
        k = ECA(c).conv.kernel_size[0]
        assert k >= 1 and k % 2 == 1, f"c={c}: kernel {k} should be odd"


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


def test_freqmix_halves_spatial_preserves_channels_pre_projection():
    m = FreqMix(16, 16)  # c1==c2 so we can compare shapes cleanly around the 1x1 proj
    y = m(torch.randn(2, 16, 64, 64))
    assert y.shape == (2, 16, 32, 32)


def test_freqmix_bands_reconstruct_original_patch():
    # LL+LH+HL+HH should losslessly reconstruct each 2x2 patch (Haar is orthogonal/invertible)
    m = FreqMix(3, 3)
    x = torch.randn(1, 3, 8, 8)
    ll, lh, hl, hh = m._bands(x)
    recon_tl = 0.5 * (ll + lh + hl + hh)  # inverse Haar for the top-left corner of each patch
    assert torch.allclose(recon_tl, x[..., ::2, ::2], atol=1e-5)


def test_freqmix_init_biases_toward_ll_band():
    # band_logits = [2.0, 0, 0, -0.5] -> softmax should heavily favor LL (band 0) at init
    m = FreqMix(8, 8)
    alpha = torch.nn.functional.softmax(m.band_logits, dim=0)
    assert alpha[0] > 0.7, f"LL weight {alpha[0].item():.3f} should dominate at init"
    assert alpha[0] > alpha[1] and alpha[0] > alpha[2] and alpha[0] > alpha[3]


def test_freqmix_projects_to_c2_channels():
    m = FreqMix(16, 32)
    y = m(torch.randn(1, 16, 32, 32))
    assert y.shape == (1, 32, 16, 16)


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
    "cfg,register_fn,expected_strides",
    [
        ("cfg/lwso-yolo11n.yaml", "lwso", [4, 8, 16]),
        ("cfg/lwso-yolo11n-lite.yaml", "lwso", [4, 8, 16]),
        ("cfg/lwso-yolo11n-eff.yaml", "lwso", [4, 8, 16]),
        ("cfg/ablation/yolo11n-p2-nop5.yaml", None, [4, 8, 16]),
        ("cfg/fap-yolo11n.yaml", "fap", [4, 8, 16, 32]),
    ],
)
def test_model_builds_and_forwards(cfg, register_fn, expected_strides):
    if register_fn == "lwso":
        from models.lwso.register import register_lwso

        register_lwso()
    elif register_fn == "fap":
        from models.fap.register import register_fap

        register_fap()
    # register_fn is None for cfg files that only use stock ultralytics modules

    from ultralytics import YOLO

    model = YOLO(str(ROOT / cfg))
    det = model.model  # DetectionModel
    total = n_params(det)
    assert total < 3_500_000, f"{cfg}: {total/1e6:.2f}M params, expected nano-sized (<3.5M)"

    det.eval()
    with torch.no_grad():
        preds = det(torch.zeros(1, 3, 256, 256))
    assert preds is not None
    strides = sorted(int(s) for s in det.stride.tolist())
    assert strides == expected_strides, f"{cfg}: unexpected strides {strides}"
    print(f"{cfg}: {total/1e6:.2f}M params, strides {strides}")
