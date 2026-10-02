from __future__ import annotations

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train FA-YOLO11.")
    parser.add_argument("--model", default="configs/fa_yolo11.yaml")
    parser.add_argument("--data", default="configs/urpc2020.yaml")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="0")
    parser.add_argument("--name", default="fa_yolo11")
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for this training entry point.") from exc

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        optimizer="SGD",
        lr0=0.01,
        weight_decay=0.0005,
        name=args.name,
    )


if __name__ == "__main__":
    main()

