"""High-level FA-YOLO11 model loader.

The official experiments are based on a YOLO11-compatible training pipeline.
This file keeps the model-loading logic separate from the module definitions.
"""

from __future__ import annotations

from pathlib import Path

from .ultralytics_integration import prepare_fa_yolo11_ultralytics


def build_model(config_path: str | Path):
    """Build an FA-YOLO11 model from a YOLO-style YAML file.

    The helper registers FAD/IAFF and patches WIoU v3 before parsing the YAML.
    """

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("Please install ultralytics or use a YOLO11-compatible codebase.") from exc

    prepare_fa_yolo11_ultralytics(use_wiou=True)
    return YOLO(str(config_path))

