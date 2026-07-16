"""Shape/param sanity tests for LWSO modules and model YAMLs. Run: pytest tests/ -v"""

from pathlib import Path

import pytest
import torch

from models.fap.modules import FreqMix
from models.lwso.losses import nwd
from models.lwso.modules import BiFPNCat, C3k2Ghost, DySample, ECA, EMA, SPDConv, SPDConvGroup
from models.pd.distill import cwd_loss
from models.slim.modules import ConvGN, LSCDetect, Scale
from models.star.losses import wiou
from models.star.modules import GSBottleneck, GSConv, SimAM, StarBlock, VoVGSCSP, C3k2Star

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


def test_starblock_preserves_shape():
    x = torch.randn(2, 32, 30, 30)
    assert StarBlock(32, 32)(x).shape == x.shape


def test_starblock_rejects_channel_change():
    with pytest.raises(AssertionError):
        StarBlock(32, 64)


def test_c3k2star_shape_and_lighter_than_c3k2():
    from ultralytics.nn.modules.block import C3k2

    star = C3k2Star(64, 128, n=2)
    stock = C3k2(64, 128, n=2)
    y = star(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 128, 32, 32)
    assert n_params(star) < n_params(stock), (
        f"star {n_params(star)} should be < stock {n_params(stock)}"
    )


def test_simam_preserves_shape():
    x = torch.randn(2, 64, 30, 30)
    assert SimAM(64)(x).shape == x.shape


def test_simam_rejects_channel_change():
    with pytest.raises(AssertionError):
        SimAM(64, 32)


def test_simam_has_zero_parameters():
    assert n_params(SimAM(64)) == 0


def test_gsconv_shape_and_lighter_than_plain_conv():
    from ultralytics.nn.modules.conv import Conv

    gs = GSConv(64, 128)
    plain = Conv(64, 128, 1, 1)
    y = gs(torch.randn(2, 64, 16, 16))
    assert y.shape == (2, 128, 16, 16)
    assert n_params(gs) < n_params(plain), f"GSConv {n_params(gs)} should be < Conv {n_params(plain)}"


def test_gsbottleneck_residual_only_when_channels_match():
    assert GSBottleneck(32, 32).add is True
    assert GSBottleneck(32, 64).add is False


def test_vovgscsp_shape_and_lighter_than_c3k2():
    from ultralytics.nn.modules.block import C3k2

    vov = VoVGSCSP(64, 128, n=2)
    stock = C3k2(64, 128, n=2)
    y = vov(torch.randn(2, 64, 32, 32))
    assert y.shape == (2, 128, 32, 32)
    assert n_params(vov) < n_params(stock), (
        f"VoVGSCSP {n_params(vov)} should be < stock {n_params(stock)}"
    )


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


def test_wiou_identical_boxes_is_zero():
    boxes = torch.tensor([[10.0, 10.0, 20.0, 20.0]])
    assert wiou(boxes, boxes, iou_mean=0.5).item() == pytest.approx(0.0, abs=1e-4)


def test_wiou_higher_for_farther_boxes():
    a = torch.tensor([[100.0, 100.0, 108.0, 108.0]])  # 8x8 box
    near = a + 1.0
    far = a + 6.0
    loss_near = wiou(a, near, iou_mean=0.5).item()
    loss_far = wiou(a, far, iou_mean=0.5).item()
    assert loss_far > loss_near


def test_wiou_r_is_one_when_outlier_degree_equals_delta():
    # r(beta) = beta / (delta * alpha**(beta - delta)) is defined so r(beta=delta) == 1
    # exactly -- pick iou_mean so beta lands exactly on delta, and cross-check the result
    # against the v1-only (r_wiou * l_iou) formula computed independently.
    from ultralytics.utils.metrics import bbox_iou

    a = torch.tensor([[100.0, 100.0, 108.0, 108.0]])
    b = a + 3.0
    l_iou = (1.0 - bbox_iou(a, b, xywh=False)).item()
    delta = 3.0
    loss = wiou(a, b, iou_mean=l_iou / delta, delta=delta).item()

    cw = (torch.max(a[..., 2], b[..., 2]) - torch.min(a[..., 0], b[..., 0])).clamp(min=1e-7)
    ch = (torch.max(a[..., 3], b[..., 3]) - torch.min(a[..., 1], b[..., 1])).clamp(min=1e-7)
    pcx, pcy = (a[..., 0] + a[..., 2]) / 2, (a[..., 1] + a[..., 3]) / 2
    tcx, tcy = (b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2
    dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2
    r_wiou = torch.exp(dist2 / (cw**2 + ch**2)).clamp(max=3.0)
    expected_v1_only = (r_wiou * l_iou).item()

    assert loss == pytest.approx(expected_v1_only, rel=1e-3)


def test_wiou_stays_finite_for_near_degenerate_small_boxes():
    # Regression test: VisDrone-sized objects can be only a few units wide in stride-
    # normalized space. An early-training (near-random) prediction can land as a tiny,
    # nearly-degenerate box offset from an equally tiny target -- the smallest enclosing
    # box then shrinks toward the eps floor, and the un-clamped WIoU v1 ratio explodes
    # through exp() to inf and poisons the whole batch's loss (reproduced via a real CPU
    # smoke train: box_loss hit ~7000-10000 instead of the usual ~5 before max_r_wiou was
    # added). A tiny 0.5x0.5 target 2 units away from a 0.1x0.1 prediction reproduces the
    # near-zero-enclosing-box, non-trivial-distance combination that triggered it.
    pred = torch.tensor([[0.0, 0.0, 0.1, 0.1]])
    target = torch.tensor([[2.0, 2.0, 2.5, 2.5]])
    loss = wiou(pred, target, iou_mean=0.5)
    assert torch.isfinite(loss).all()
    assert loss.item() < 10.0  # r (<=~1.3 peak) * r_wiou (<=3) * l_iou (<=1) is bounded well under 10


# ---------------------------------------------------------------- slim (LSCDetect head)

def test_convgn_shape_and_uses_groupnorm_not_batchnorm():
    m = ConvGN(32, 64, 3)
    y = m(torch.randn(2, 32, 16, 16))
    assert y.shape == (2, 64, 16, 16)
    assert isinstance(m.gn, torch.nn.GroupNorm)
    assert not any(isinstance(x, torch.nn.BatchNorm2d) for x in m.modules())


def test_convgn_gn_groups_adapt_to_odd_channels():
    # 40 channels: gcd(16, 40) = 8 groups; must build and forward without error
    y = ConvGN(16, 40, 3)(torch.randn(1, 16, 8, 8))
    assert y.shape == (1, 40, 8, 8)


def test_convgn_depthwise_variant():
    m = ConvGN(48, 48, 3, g=48)
    assert m.conv.groups == 48
    assert n_params(m) < n_params(ConvGN(48, 48, 3))


def test_scale_multiplies_and_is_learnable():
    s = Scale(2.0)
    assert torch.allclose(s(torch.ones(3)), torch.full((3,), 2.0))
    assert sum(1 for _ in s.parameters()) == 1


def test_lscdetect_train_and_eval_forward():
    ch = (48, 96, 128)
    m = LSCDetect(nc=10, hidc=48, ch=ch)
    m.stride = torch.tensor([4.0, 8.0, 16.0])
    m.bias_init()  # must not need per-scale cv2/cv3 ModuleLists like stock Detect does
    feats = [torch.randn(2, c, 64 // (2**i), 64 // (2**i)) for i, c in enumerate(ch)]
    m.train()
    outs = m([f.clone() for f in feats])
    assert [o.shape[1] for o in outs] == [m.no] * 3  # 4*reg_max + nc per scale
    m.eval()
    with torch.no_grad():
        y, raw = m([f.clone() for f in feats])
    assert y.shape[1] == 4 + 10  # decoded boxes + class scores


def test_lscdetect_much_lighter_than_stock_detect():
    from ultralytics.nn.modules.head import Detect

    ch = (48, 96, 128)
    shared = LSCDetect(nc=10, hidc=48, ch=ch)
    stock = Detect(nc=10, ch=ch)
    assert n_params(shared) < 0.5 * n_params(stock), (
        f"LSCDetect {n_params(shared)} should be <50% of stock Detect {n_params(stock)}"
    )


def test_slim_yaml_stays_under_baseline_budget():
    # The whole point of idea slim: params < 2.58M AND GFLOPs@640 < 6.5 (stock YOLO11n,
    # unfused) while keeping the P2 head. Regression-guard both.
    from models.slim.register import register_slim

    register_slim()
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    det = YOLO(str(ROOT / "cfg/slim-yolo11n.yaml")).model
    assert n_params(det) < 2_580_000
    gflops = get_flops(det, 640)
    assert 0 < gflops < 6.5, f"slim GFLOPs@640 = {gflops:.2f}, must stay below baseline 6.5"


def test_slim_fuse_preserves_groupnorm_head():
    # DetectionModel.fuse() folds Conv+BatchNorm pairs; ConvGN must never be caught by
    # that isinstance net (GroupNorm can't be folded into a conv the same way).
    from models.slim.register import register_slim

    register_slim()
    from ultralytics import YOLO

    det = YOLO(str(ROOT / "cfg/slim-yolo11n.yaml")).model
    n_gn_before = sum(1 for m in det.modules() if isinstance(m, torch.nn.GroupNorm))
    det.fuse()
    n_gn_after = sum(1 for m in det.modules() if isinstance(m, torch.nn.GroupNorm))
    assert n_gn_before == n_gn_after > 0
    with torch.no_grad():
        det.eval()(torch.zeros(1, 3, 256, 256))


# ---------------------------------------------------------------- pd (CWD distillation)

def test_cwd_identical_features_give_zero_loss():
    f = torch.randn(2, 8, 16, 16)
    assert cwd_loss(f, f.clone()).abs().item() < 1e-6


def test_cwd_positive_and_finite_for_different_features():
    torch.manual_seed(0)
    fs, ft = torch.randn(2, 8, 16, 16), torch.randn(2, 8, 16, 16)
    loss = cwd_loss(fs, ft)
    assert torch.isfinite(loss) and loss.item() > 0


def test_cwd_gradients_flow_to_student_only():
    fs = torch.randn(1, 4, 8, 8, requires_grad=True)
    ft = torch.randn(1, 4, 8, 8)
    cwd_loss(fs, ft).backward()
    assert fs.grad is not None and torch.isfinite(fs.grad).all()


def test_cwd_rejects_shape_mismatch():
    with pytest.raises(ValueError):
        cwd_loss(torch.randn(1, 8, 8, 8), torch.randn(1, 16, 8, 8))


def test_cwd_half_precision_inputs_computed_in_fp32():
    # under AMP the stashed features arrive as fp16; the loss must not overflow/underflow
    fs = torch.randn(1, 4, 8, 8).half()
    ft = torch.randn(1, 4, 8, 8).half()
    assert torch.isfinite(cwd_loss(fs, ft))


def test_pd_trainer_get_model_returns_module_unchanged():
    from models.pd.model import _pd_trainer_cls

    trainer_cls = _pd_trainer_cls()
    m = torch.nn.Conv2d(3, 8, 3)
    out = trainer_cls.get_model(None, cfg={"whatever": 1}, weights=m)
    assert out is m  # same object, no yaml rebuild, float() in place
    with pytest.raises(RuntimeError):
        trainer_cls.get_model(None, cfg=None, weights=None)


# ---------------------------------------------------------------- full models

@pytest.mark.parametrize(
    "cfg,register_fn,expected_strides",
    [
        ("cfg/lwso-yolo11n.yaml", "lwso", [4, 8, 16]),
        ("cfg/lwso-yolo11n-lite.yaml", "lwso", [4, 8, 16]),
        ("cfg/lwso-yolo11n-eff.yaml", "lwso", [4, 8, 16]),
        ("cfg/ablation/yolo11n-p2-nop5.yaml", None, [4, 8, 16]),
        ("cfg/fap-yolo11n.yaml", "fap", [4, 8, 16, 32]),
        ("cfg/star-yolo11n.yaml", "star", [4, 8, 16]),
        ("cfg/slim-yolo11n.yaml", "slim", [4, 8, 16]),
    ],
)
def test_model_builds_and_forwards(cfg, register_fn, expected_strides):
    if register_fn == "lwso":
        from models.lwso.register import register_lwso

        register_lwso()
    elif register_fn == "fap":
        from models.fap.register import register_fap

        register_fap()
    elif register_fn == "star":
        from models.star.register import register_star

        register_star()
    elif register_fn == "slim":
        from models.slim.register import register_slim

        register_slim()
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
