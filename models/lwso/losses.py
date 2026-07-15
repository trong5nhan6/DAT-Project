"""Small-object-friendly regression loss for ultralytics YOLO.

Blends CIoU with Normalized Wasserstein Distance (NWD, Wang et al. 2021):

    reg = ratio * (1 - CIoU) + (1 - ratio) * (1 - NWD)

NWD models boxes as 2D Gaussians and stays smooth under the few-pixel offsets
that make plain IoU gradients unstable on tiny objects.

Note on `constant`: the original paper uses C=12.8 for absolute-pixel boxes.
Inside v8DetectionLoss, boxes are in stride-normalized (feature-map) units, so
the effective object sizes are smaller; treat C as a tunable (try 2-16) and
ablate. Default keeps 12.8 as a conservative starting point.
"""

import torch

from ultralytics.utils import loss as uloss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist

__all__ = ["nwd", "NWDBboxLoss", "patch_nwd_loss"]


def nwd(pred: torch.Tensor, target: torch.Tensor, constant: float = 12.8, eps: float = 1e-7):
    """Normalized Wasserstein Distance similarity between xyxy boxes. Returns values in (0, 1]."""
    pcx, pcy = (pred[..., 0] + pred[..., 2]) / 2, (pred[..., 1] + pred[..., 3]) / 2
    tcx, tcy = (target[..., 0] + target[..., 2]) / 2, (target[..., 1] + target[..., 3]) / 2
    pw, ph = pred[..., 2] - pred[..., 0], pred[..., 3] - pred[..., 1]
    tw, th = target[..., 2] - target[..., 0], target[..., 3] - target[..., 1]
    dist2 = (pcx - tcx) ** 2 + (pcy - tcy) ** 2 + ((pw - tw) ** 2 + (ph - th) ** 2) / 4
    return torch.exp(-torch.sqrt(dist2.clamp(min=eps)) / constant)


class NWDBboxLoss(uloss.BboxLoss):
    """BboxLoss with the CIoU/NWD blend above; DFL part is unchanged from the parent."""

    nwd_ratio = 0.5  # weight of the CIoU term; (1 - ratio) goes to NWD
    nwd_constant = 12.8

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
        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        nwd_sim = nwd(
            pred_bboxes[fg_mask], target_bboxes[fg_mask], constant=self.nwd_constant
        ).unsqueeze(-1)
        reg = self.nwd_ratio * (1.0 - iou) + (1.0 - self.nwd_ratio) * (1.0 - nwd_sim)
        loss_iou = (reg * weight).sum() / target_scores_sum

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


def patch_nwd_loss(ratio: float = 0.5, constant: float = 12.8) -> None:
    """Swap ultralytics' BboxLoss for the NWD blend. Call before model.train().

    v8DetectionLoss instantiates `BboxLoss` from the loss module's globals, so the
    class swap takes effect for every detection trainer created afterwards.
    """
    NWDBboxLoss.nwd_ratio = float(ratio)
    NWDBboxLoss.nwd_constant = float(constant)
    uloss.BboxLoss = NWDBboxLoss
