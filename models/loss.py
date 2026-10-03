"""WIoU v3 loss component used by FA-YOLO11.

The implementation follows the manuscript equations:

    L_IoU = 1 - IoU
    R_WIoU = exp(center_distance / detached_enclosing_diagonal)
    L_WIoU_v1 = R_WIoU * L_IoU
    beta = detached(L_IoU) / EMA(L_IoU)
    r = beta / (delta_w * alpha_w ** (beta - delta_w))
    L_WIoU_v3 = r * L_WIoU_v1

Only the bounding-box regression term is replaced; classification and DFL remain
the responsibility of the surrounding YOLO training code.
"""

from __future__ import annotations

import torch
from torch import nn


def box_iou_xyxy(box1: torch.Tensor, box2: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """Pairwise IoU for aligned xyxy boxes.

    Args:
        box1: Predicted boxes with shape [..., 4].
        box2: Target boxes with shape [..., 4].
        eps: Numerical stability term.
    """

    lt = torch.max(box1[..., :2], box2[..., :2])
    rb = torch.min(box1[..., 2:], box2[..., 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]

    area1 = (box1[..., 2] - box1[..., 0]).clamp(min=0) * (box1[..., 3] - box1[..., 1]).clamp(min=0)
    area2 = (box2[..., 2] - box2[..., 0]).clamp(min=0) * (box2[..., 3] - box2[..., 1]).clamp(min=0)
    return inter / (area1 + area2 - inter + eps)


class WIoUv3Loss(nn.Module):
    """Wise-IoU v3 style non-monotonic focusing loss.

    The manuscript uses alpha_w = 1.9 and delta_w = 3.0. The returned value is a
    scalar box-regression loss that can be combined with classification and DFL
    losses using the reported weights.
    """

    def __init__(
        self,
        alpha_w: float = 1.9,
        delta_w: float = 3.0,
        momentum: float = 0.01,
        eps: float = 1e-7,
    ):
        super().__init__()
        self.alpha_w = float(alpha_w)
        self.delta_w = float(delta_w)
        self.momentum = float(momentum)
        self.eps = eps
        self.register_buffer("iou_loss_ema", torch.tensor(1.0), persistent=True)
        self.register_buffer("ema_initialized", torch.tensor(False), persistent=True)

    def _update_ema(self, batch_mean: torch.Tensor) -> None:
        batch_mean = batch_mean.detach()
        if not bool(self.ema_initialized.item()):
            self.iou_loss_ema.copy_(batch_mean)
            self.ema_initialized.fill_(True)
        else:
            self.iou_loss_ema.mul_(1.0 - self.momentum).add_(batch_mean * self.momentum)

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor,
        weight: torch.Tensor | None = None,
        normalizer: torch.Tensor | float | None = None,
        update_ema: bool = True,
    ) -> torch.Tensor:
        iou = box_iou_xyxy(pred_boxes, target_boxes, self.eps)
        l_iou = 1.0 - iou

        pred_ctr = (pred_boxes[..., :2] + pred_boxes[..., 2:]) * 0.5
        target_ctr = (target_boxes[..., :2] + target_boxes[..., 2:]) * 0.5
        center_distance = ((pred_ctr - target_ctr) ** 2).sum(dim=-1)

        enclose_lt = torch.min(pred_boxes[..., :2], target_boxes[..., :2])
        enclose_rb = torch.max(pred_boxes[..., 2:], target_boxes[..., 2:])
        enclose_wh = (enclose_rb - enclose_lt).clamp(min=0)
        enclose_diag = (enclose_wh[..., 0] ** 2 + enclose_wh[..., 1] ** 2).detach().clamp(min=self.eps)

        r_wiou = torch.exp(center_distance / enclose_diag)
        l_wiou_v1 = r_wiou * l_iou

        if update_ema and self.training:
            self._update_ema(l_iou.detach().mean())

        beta = l_iou.detach() / (self.iou_loss_ema.detach().clamp(min=self.eps))
        alpha = torch.as_tensor(self.alpha_w, device=beta.device, dtype=beta.dtype)
        focusing = beta / (self.delta_w * torch.pow(alpha, beta - self.delta_w) + self.eps)
        loss = focusing * l_wiou_v1

        if weight is not None:
            weight = weight.reshape_as(loss).to(device=loss.device, dtype=loss.dtype)
            loss = loss * weight
            if normalizer is None:
                return loss.sum() / weight.sum().clamp(min=self.eps)
            normalizer_tensor = torch.as_tensor(normalizer, device=loss.device, dtype=loss.dtype)
            return loss.sum() / normalizer_tensor.clamp(min=self.eps)

        return loss.mean()

