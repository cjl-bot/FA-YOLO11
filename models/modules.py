"""Core FA-YOLO11 network modules.

This file contains compact PyTorch implementations of the proposed module logic.
For direct use in a YOLO11 codebase, register FAD and IAFF in the model parser.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class FAD(nn.Module):
    """Frequency-Aware Downsampling based on four-sub-band Haar 2D-DWT.

    Given an input tensor x with shape [B, C, H, W], FAD first applies fixed Haar
    filters to obtain LL, LH, HL, and HH responses. The four sub-bands are then
    concatenated along the channel dimension and compressed by a 1x1 projection.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        self.proj = nn.Sequential(
            nn.Conv2d(4 * c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

        haar = torch.tensor(
            [
                [[1.0, 1.0], [1.0, 1.0]],      # LL
                [[-1.0, -1.0], [1.0, 1.0]],    # LH
                [[-1.0, 1.0], [-1.0, 1.0]],    # HL
                [[1.0, -1.0], [-1.0, 1.0]],    # HH
            ],
            dtype=torch.float32,
        ) * 0.5
        self.register_buffer("haar", haar[:, None, :, :], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")

        weight = self.haar.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        # conv2d returns [B, 4C, H/2, W/2] with sub-bands repeated per channel.
        return self.proj(y)


class IAFF(nn.Module):
    """Illumination-Adaptive Feature Fusion.

    IAFF pools the feature map along the channel axis to retain a spatial
    H x W response grid, generates a learned illumination-related mask, and
    applies bounded positive residual recalibration:

        F_IAFF = F * (1 + alpha * exp(-gamma * M_illu)).
    """

    def __init__(self, channels: int, alpha: float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = float(alpha)
        self.gamma = float(gamma)

        # A standard 3x3 convolution is used here as a portable fallback. In the
        # experiments, this branch can be replaced by DCNv2/DeformConv when the
        # target YOLO11 codebase provides it.
        self.mask_branch = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(8),
            nn.SiLU(inplace=True),
            nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        descriptor = torch.cat([avg_map, max_map], dim=1)
        m_illu = self.mask_branch(descriptor)
        residual_gain = self.alpha * torch.exp(-self.gamma * m_illu)
        return x + x * residual_gain


