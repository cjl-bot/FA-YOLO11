from __future__ import annotations

import argparse
import time

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Measure FA-YOLO11 inference latency and FPS.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--half", action="store_true", help="Use FP16 inference. Default is FP32, matching the paper.")
    return parser.parse_args()


def resolve_device(device_arg: str) -> tuple[torch.device, str]:
    """Accept both Ultralytics-style ``0`` and PyTorch-style ``cuda:0``."""

    raw = str(device_arg).strip().lower()
    if raw == "cpu" or not torch.cuda.is_available():
        return torch.device("cpu"), "cpu"
    if raw.isdigit():
        return torch.device(f"cuda:{raw}"), raw
    if raw == "cuda":
        return torch.device("cuda:0"), "0"
    if raw.startswith("cuda:"):
        index = raw.split(":", 1)[1] or "0"
        return torch.device(f"cuda:{index}"), index
    return torch.device(raw), raw


@torch.no_grad()
def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for benchmarking.") from exc

    from models.ultralytics_integration import register_fa_yolo11_modules

    register_fa_yolo11_modules()

    device, yolo_device = resolve_device(args.device)
    model = YOLO(args.weights)
    model.model.to(device).eval()

    # In-memory uint8 frames avoid disk I/O while still exercising Ultralytics
    # preprocessing (letterbox/resize and normalization), model forward, and NMS.
    sources = [
        np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8)
        for _ in range(args.batch)
    ]

    predict_kwargs = dict(
        source=sources,
        imgsz=args.imgsz,
        batch=args.batch,
        device=yolo_device,
        conf=args.conf,
        iou=args.iou,
        half=args.half,
        verbose=False,
        save=False,
        stream=False,
    )

    for _ in range(args.warmup):
        _ = model.predict(**predict_kwargs)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.predict(**predict_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_ms = sum(times) / len(times) * 1000.0
    fps = args.batch / (sum(times) / len(times))
    std_ms = torch.tensor(times).std(unbiased=False).item() * 1000.0

    print(f"batch={args.batch}")
    print(f"imgsz={args.imgsz}")
    print(f"device={yolo_device}")
    print(f"warmup={args.warmup}")
    print(f"repeat={args.repeat}")
    print("timed_scope=ultralytics_predict_preprocess_forward_nms")
    print("excluded=disk_io_camera_capture_visualization_file_writing_robot_control")
    print(f"precision={'FP16' if args.half else 'FP32'}")
    print(f"mean_latency_ms={latency_ms:.4f}")
    print(f"std_latency_ms={std_ms:.4f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()

