"""Wise-IoU v3 regression loss for idea "star" (Tong et al., "Wise-IoU: Bounding Box
Regression Loss with Dynamic Focusing Mechanism", arXiv:2301.10051).

Unlike CIoU (stock) or the CIoU/NWD blend (models/lwso/losses.py), WIoU v3 uses a *dynamic
non-monotonic* focusing mechanism: an EMA-tracked running mean of the IoU loss across
training sets each anchor's "outlier degree" beta = L_IoU / mean(L_IoU). Anchors near the
mean (neither trivially easy nor a probable label-noise outlier) get down-weighted, while
genuinely hard-but-plausible anchors get up-weighted — a static loss (CIoU, or NWD's fixed
blend ratio) can't distinguish "hard but real" from "outlier/mislabeled" the way this can,
which matters more on VisDrone's small/dense/occluded boxes than on well-separated objects.
"""

import torch

from ultralytics.utils import loss as uloss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

__all__ = ["wiou", "WiseIoULoss", "patch_wiou_loss"]


def wiou(pred: torch.Tensor, target: torch.Tensor, iou_mean: float, delta: float = 3.0,
         alpha: float = 1.9, eps: float = 1e-7, max_r_wiou: float = 3.0) -> torch.Tensor:
    """Wise-IoU v3 per-box loss term: r * R_WIoU * (1 - IoU). xyxy boxes.

    R_WIoU (v1): distance between box centers, normalized by the smallest enclosing box's
    diagonal (detached from the graph — paper found gradients through this term actively
    hinder convergence rather than help it). `r_wiou` is additionally clamped to
    `max_r_wiou` (default 3x) after exp() as a defensive bound — a "distance penalty
    multiplier" beyond ~3x isn't a useful gradient signal, and on VisDrone-sized small
    objects a near-degenerate enclosing box (pred and/or target only a few stride-units
    wide) is common enough that leaving this term fully unbounded is a real risk, not
    just theoretical.

    r (v3): non-monotonic focusing coefficient from the outlier degree beta = L_IoU /
    iou_mean (iou_mean is a running average tracked by WiseIoULoss.forward across training,
    passed in here as a plain float so this function stays a pure/testable computation).

    bbox_iou() returns shape (N, 1) (keepdim), while the center/enclosing-box terms below
    are computed via `pred[..., 0]`-style indexing, which drops the last dim to shape (N,)
    -- `iou`/`l_iou` are squeezed to (N,) here so every term this function computes and
    multiplies together has a consistent shape. Skipping this was a real bug caught while
    building this idea: with l_iou left at (N, 1), `r * r_wiou * l_iou` (shapes (N,) *
    (N,) * (N, 1)) broadcasts to (N, N) instead of (N,) -- every anchor's IoU loss gets
    cross-multiplied against every *other* anchor's distance/outlier terms, and the extra
    N-fold sum inflated box_loss to ~7000-10000 instead of the usual ~5 (reproduced via a
    real CPU smoke train; fixed by squeezing here, not by the max_r_wiou clamp above,
    which was a red herring for this specific bug even though it's still worth keeping).
    """
    iou = bbox_iou(pred, target, xywh=False).squeeze(-1)
    l_iou = 1.0 - iou

    px, py = (pred[..., 0] + pred[..., 2]) / 2, (pred[..., 1] + pred[..., 3]) / 2
    tx, ty = (target[..., 0] + target[..., 2]) / 2, (target[..., 1] + target[..., 3]) / 2
    enc_x1, enc_y1 = torch.min(pred[..., 0], target[..., 0]), torch.min(pred[..., 1], target[..., 1])
    enc_x2, enc_y2 = torch.max(pred[..., 2], target[..., 2]), torch.max(pred[..., 3], target[..., 3])
    cw, ch = (enc_x2 - enc_x1).clamp(min=eps), (enc_y2 - enc_y1).clamp(min=eps)
    dist2 = (px - tx) ** 2 + (py - ty) ** 2
    r_wiou = torch.exp(dist2 / (cw**2 + ch**2).detach()).clamp(max=max_r_wiou)

    beta = l_iou.detach() / max(iou_mean, eps)
    r = beta / (delta * alpha ** (beta - delta))

    return r * r_wiou * l_iou


class WiseIoULoss(uloss.BboxLoss):
    """BboxLoss using wiou() instead of CIoU. DFL part is unchanged from the parent."""

    wiou_delta = 3.0
    wiou_alpha = 1.9
    wiou_momentum = 1e-4  # EMA momentum for the running mean of L_IoU

    def __init__(self, reg_max: int = 16):
        super().__init__(reg_max)
        self.register_buffer("iou_mean", torch.tensor(1.0))

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
        pred, target = pred_bboxes[fg_mask], target_bboxes[fg_mask]

        with torch.no_grad():
            if pred.numel() > 0:
                batch_l_iou_mean = (1.0 - bbox_iou(pred, target, xywh=False)).mean()
                if self.iou_mean.item() == 1.0:  # first real batch seeds the running mean
                    self.iou_mean.fill_(batch_l_iou_mean)
                else:
                    self.iou_mean.mul_(1 - self.wiou_momentum).add_(self.wiou_momentum * batch_l_iou_mean)

        loss_terms = wiou(pred, target, float(self.iou_mean), delta=self.wiou_delta, alpha=self.wiou_alpha)
        loss_iou = (loss_terms.unsqueeze(-1) * weight).sum() / target_scores_sum

        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(
                    pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]
                )
                * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            loss_dfl = torch.tensor(0.0).to(pred_dist.device)

        return loss_iou, loss_dfl


def patch_wiou_loss() -> None:
    """Swap ultralytics' BboxLoss for WiseIoULoss. Call before model.train().

    v8DetectionLoss instantiates `BboxLoss` from the loss module's globals, so the class
    swap takes effect for every detection trainer created afterwards.
    """
    uloss.BboxLoss = WiseIoULoss
