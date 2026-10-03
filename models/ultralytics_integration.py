"""Ultralytics integration helpers for FA-YOLO11.

These helpers keep the public entry points aligned with the manuscript:

* FAD and IAFF are registered before parsing the model YAML.
* The Ultralytics parser is patched so FAD receives (c1, c2) and IAFF preserves
  the input-channel count.
* The box-regression part of the Ultralytics loss is replaced by WIoU v3 while
  leaving classification and DFL terms unchanged.

The patches are intentionally small and version-tolerant. If the installed
Ultralytics version changes its internal parser or loss signature, the helper
raises an explicit error instead of silently training a mismatched baseline.
"""

from __future__ import annotations

import inspect
import textwrap
from types import ModuleType
from typing import Any

import torch

from .loss import WIoUv3Loss
from .modules import FAD, IAFF

SUPPORTED_ULTRALYTICS_VERSION = "8.3.0"


def _load_tasks_module() -> ModuleType:
    try:
        import ultralytics
        import ultralytics.nn.tasks as tasks
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError("Ultralytics is required to register FA-YOLO11 modules.") from exc
    if getattr(ultralytics, "__version__", None) != SUPPORTED_ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"FA-YOLO11 parser patch is pinned to ultralytics=={SUPPORTED_ULTRALYTICS_VERSION}. "
            f"Detected ultralytics=={getattr(ultralytics, '__version__', 'unknown')}. "
            "Install the pinned version from requirements.txt or update the patch for your local version."
        )
    return tasks


def register_fa_yolo11_modules() -> None:
    """Register FAD and IAFF in the Ultralytics YAML parser."""

    tasks = _load_tasks_module()
    tasks.FAD = FAD
    tasks.IAFF = IAFF

    if getattr(tasks.parse_model, "_fa_yolo11_patched", False):
        return

    source = textwrap.dedent(inspect.getsource(tasks.parse_model))
    if "base_modules = frozenset({" not in source or "elif m is Concat:" not in source:
        raise RuntimeError(
            "Unable to patch ultralytics.nn.tasks.parse_model automatically. "
            "Please add FAD to base_modules and add an IAFF branch that sets "
            "args=[ch[f], *args] and c2=ch[f]."
        )

    source = source.replace(
        "base_modules = frozenset({",
        "base_modules = frozenset({\n            FAD,",
        )
    source = source.replace(
        "elif m is Concat:",
        "elif m is IAFF:\n"
        "            args = [ch[f], *args]\n"
        "            c2 = ch[f]\n"
        "        elif m is Concat:",
    )

    namespace = tasks.__dict__
    exec(compile(source, filename="<fa_yolo11_parse_model_patch>", mode="exec"), namespace)
    namespace["parse_model"]._fa_yolo11_patched = True


def patch_ultralytics_wiou_loss(alpha_w: float = 1.9, delta_w: float = 3.0, momentum: float = 0.01) -> None:
    """Replace the Ultralytics bbox IoU term with WIoU v3.

    The patch targets the common Ultralytics ``BboxLoss.forward`` signature:

    ``(pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores,
    target_scores_sum, fg_mask)``.

    The original method is still called to compute the DFL term; only the IoU
    regression term is replaced.
    """

    try:
        import ultralytics.utils.loss as ul_loss
    except ImportError as exc:  # pragma: no cover - depends on user environment
        raise ImportError("Ultralytics is required to patch the WIoU loss.") from exc

    bbox_loss_cls = getattr(ul_loss, "BboxLoss", None)
    if bbox_loss_cls is None:
        raise RuntimeError("Could not locate ultralytics.utils.loss.BboxLoss.")
    if getattr(bbox_loss_cls.forward, "_fa_yolo11_wiou_patched", False):
        return

    original_forward = bbox_loss_cls.forward

    def forward(
        self: Any,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
    ):
        _, loss_dfl = original_forward(
            self,
            pred_dist,
            pred_bboxes,
            anchor_points,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )

        if not hasattr(self, "fa_wiou_v3"):
            self.fa_wiou_v3 = WIoUv3Loss(alpha_w=alpha_w, delta_w=delta_w, momentum=momentum)

        if fg_mask.sum() == 0:
            loss_iou = pred_bboxes.sum() * 0.0
        else:
            weights = target_scores.sum(-1)[fg_mask].unsqueeze(-1)
            loss_iou = self.fa_wiou_v3(
                pred_bboxes[fg_mask],
                target_bboxes[fg_mask],
                weight=weights,
                normalizer=target_scores_sum,
            )
        return loss_iou, loss_dfl

    forward._fa_yolo11_wiou_patched = True
    bbox_loss_cls.forward = forward


def prepare_fa_yolo11_ultralytics(use_wiou: bool = True) -> None:
    """Apply all FA-YOLO11 runtime registrations before building a YOLO model."""

    register_fa_yolo11_modules()
    if use_wiou:
        patch_ultralytics_wiou_loss()
