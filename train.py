from __future__ import annotations

import argparse
from pathlib import Path

from models.ultralytics_integration import prepare_fa_yolo11_ultralytics


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
    parser.add_argument("--optimizer", default="SGD")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--no-wiou", action="store_true", help="Disable WIoU v3 patch and use the Ultralytics default bbox loss.")
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="Disable epoch-wise validation/model selection and keep the final checkpoint. Automatically enabled for configs/duo.yaml.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for this training entry point.") from exc

    use_wiou = not args.no_wiou
    prepare_fa_yolo11_ultralytics(use_wiou=use_wiou)

    data_name = Path(args.data).name.lower()
    final_only = args.final_only or data_name == "duo.yaml"

    model = YOLO(args.model)
    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        seed=args.seed,
        device=args.device,
        optimizer=args.optimizer,
        lr0=args.lr0,
        weight_decay=args.weight_decay,
        name=args.name,
        exist_ok=True,
    )
    if final_only:
        # Matches the DUO protocol in the paper: no test-set model selection.
        # Ultralytics will still save the final/last checkpoint for evaluation.
        train_kwargs.update(val=False, plots=False)

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()

