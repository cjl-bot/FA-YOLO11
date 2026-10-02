from __future__ import annotations

import argparse
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Measure FA-YOLO11 inference latency and FPS.")
    parser.add_argument("--weights", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeat", type=int, default=1000)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("Ultralytics is required for benchmarking.") from exc

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = YOLO(args.weights)
    model.model.to(device).eval()

    dummy = torch.zeros(args.batch, 3, args.imgsz, args.imgsz, device=device)

    for _ in range(args.warmup):
        _ = model.model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(args.repeat):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = model.model(dummy)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    latency_ms = sum(times) / len(times) * 1000.0
    fps = args.batch / (sum(times) / len(times))
    std_ms = torch.tensor(times).std(unbiased=False).item() * 1000.0

    print(f"batch={args.batch}")
    print(f"imgsz={args.imgsz}")
    print(f"warmup={args.warmup}")
    print(f"repeat={args.repeat}")
    print(f"mean_latency_ms={latency_ms:.4f}")
    print(f"std_latency_ms={std_ms:.4f}")
    print(f"fps={fps:.2f}")


if __name__ == "__main__":
    main()

