"""WIoU v3 loss component used by FA-YOLO11.

This file provides a compact implementation of the bounding-box regression term
used in the paper. In a full YOLO11 training codebase, this loss should replace
only the IoU regression part, while the classification and DFL terms remain
unchanged.
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

    def __init__(self, alpha_w: float = 1.9, delta_w: float = 3.0, eps: float = 1e-7):
        super().__init__()
        self.alpha_w = float(alpha_w)
        self.delta_w = float(delta_w)
        self.eps = eps

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        iou = box_iou_xyxy(pred_boxes, target_boxes, self.eps)
        base_loss = 1.0 - iou

        # The detached outlier degree avoids directly backpropagating through the
        # batch-level normalization term.
        beta = base_loss.detach() / (base_loss.detach().mean() + self.eps)
        focusing = beta / (self.delta_w * torch.pow(torch.tensor(self.alpha_w, device=beta.device), beta - self.delta_w) + self.eps)
        return (focusing * base_loss).mean()

