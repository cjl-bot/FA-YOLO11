"""High-level FA-YOLO11 model loader.

The official experiments are based on a YOLO11-compatible training pipeline.
This file keeps the model-loading logic separate from the module definitions.
"""

from __future__ import annotations

from pathlib import Path


def build_model(config_path: str | Path):
    """Build an FA-YOLO11 model from a YOLO-style YAML file.

    The local YOLO11 implementation must register the custom modules FAD and IAFF
    before parsing `configs/fa_yolo11.yaml`.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install ultralytics or use a YOLO11-compatible codebase.") from exc

    return YOLO(str(config_path))

