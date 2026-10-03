from __future__ import annotations

import argparse

from models.ultralytics_integration import register_fa_yolo11_modules


def parse_args():
    parser = argparse.ArgumentParser(description="Validate FA-YOLO11.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="configs/urpc2020.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for this validation entry point.") from exc

    register_fa_yolo11_modules()

    model = YOLO(args.weights)
    model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()

