from __future__ import annotations

import argparse

from models.ultralytics_integration import register_fa_yolo11_modules


def parse_args():
    parser = argparse.ArgumentParser(description="Run FA-YOLO11 inference.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--save", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for this inference entry point.") from exc

    register_fa_yolo11_modules()

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=args.save,
    )


if __name__ == "__main__":
    main()

