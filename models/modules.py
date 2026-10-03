"""Core FA-YOLO11 network modules.

The implementations in this file follow the manuscript description:

* FAD performs four-sub-band Haar 2D-DWT downsampling followed by learnable
  1x1 channel compression.
* IAFF keeps the H x W spatial grid by pooling across channels and uses a
  deformable 3x3 mask branch before the 1x1 + Sigmoid projection.
  Its default hidden width is proportional to the incoming feature channels so
  the four IAFF insertions match the paper's parameter-count scale.

The helper in ``models.ultralytics_integration`` registers these modules before
Ultralytics parses ``configs/fa_yolo11.yaml``.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
except Exception:  # pragma: no cover - depends on the local torchvision build
    DeformConv2d = None


class FAD(nn.Module):
    """Frequency-Aware Downsampling based on four-sub-band Haar 2D-DWT.

    Given an input tensor x with shape [B, C, H, W], FAD first applies fixed Haar
    filters to obtain LL, LH, HL, and HH responses. The four sub-bands are then
    concatenated along the channel dimension and compressed by a 1x1 projection.
    """

    def __init__(self, c1: int, c2: int | None):
        super().__init__()
        if c2 is None:
            raise ValueError(
                "FAD requires both input and output channels as FAD(c1, c2). "
                "Call register_fa_yolo11_modules() before parsing configs/fa_yolo11.yaml."
            )
        self.c1 = c1
        self.c2 = c2
        self.proj = self._make_projection(c1, c2)

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

    @staticmethod
    def _make_projection(c1: int, c2: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(4 * c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        if h % 2 != 0 or w % 2 != 0:
            x = F.pad(x, (0, w % 2, 0, h % 2), mode="replicate")

        weight = self.haar.repeat(c, 1, 1, 1)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=c)
        # conv2d returns [B, 4C, H/2, W/2] with sub-bands repeated per channel.
        return self.proj(y)


class DeformableMaskBranch(nn.Module):
    """IAFF mask extractor with predicted offsets and deformable convolution."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "torchvision.ops.DeformConv2d is required for the IAFF DCN mask branch. "
                "Install a torchvision build compatible with the installed PyTorch, or "
                "instantiate IAFF(..., deformable=False) for the ablation fallback."
            )
        self.offset = nn.Conv2d(2, 2 * 3 * 3, kernel_size=3, stride=1, padding=1)
        self.dcn = DeformConv2d(2, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm = nn.BatchNorm2d(hidden_channels)
        self.act = nn.SiLU(inplace=True)
        self.project = nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.out_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        offset = self.offset(x)
        x = self.dcn(x, offset)
        x = self.act(self.norm(x))
        return self.out_act(self.project(x))


class StandardMaskBranch(nn.Module):
    """Portable 3x3-convolution mask branch used only for the IAFF ablation."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.branch = nn.Sequential(
            nn.Conv2d(2, hidden_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.branch(x)


class IAFF(nn.Module):
    """Illumination-Adaptive Feature Fusion.

    IAFF pools the feature map along the channel axis to retain a spatial
    H x W response grid, generates a learned illumination-related mask, and
    applies bounded positive residual recalibration:

        F_IAFF = F * (1 + alpha * exp(-gamma * M_illu)).
    """

    def __init__(
        self,
        channels: int | float | None = None,
        alpha: float = 0.5,
        gamma: float = 2.0,
        deformable: bool = True,
        hidden_channels: int | None = None,
        hidden_ratio: float = 2.5,
    ):
        super().__init__()
        # Ultralytics custom-module parsing may instantiate IAFF as IAFF(0.5, 2.0)
        # when no input channel is needed by the module. Interpret that form as
        # alpha=0.5, gamma=2.0 rather than channels=0.5.
        if isinstance(channels, float):
            alpha, gamma, channels = channels, alpha, None
        self.channels = channels
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.deformable = bool(deformable)
        if hidden_channels is None:
            hidden_channels = 8 if channels is None else max(8, int(round(int(channels) * hidden_ratio)))
        self.hidden_channels = int(hidden_channels)

        if self.deformable:
            try:
                self.mask_branch = DeformableMaskBranch(self.hidden_channels)
            except ImportError as exc:
                warnings.warn(
                    f"{exc} Falling back to the standard 3x3 IAFF mask branch. "
                    "This fallback corresponds to the non-DCN ablation, not the main experiment.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self.deformable = False
                self.mask_branch = StandardMaskBranch(self.hidden_channels)
        else:
            self.mask_branch = StandardMaskBranch(self.hidden_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_map = x.mean(dim=1, keepdim=True)
        max_map = x.max(dim=1, keepdim=True).values
        descriptor = torch.cat([avg_map, max_map], dim=1)
        m_illu = self.mask_branch(descriptor)
        residual_gain = self.alpha * torch.exp(-self.gamma * m_illu)
        return x + x * residual_gain


