# FA-YOLO11

## Abstract

FA-YOLO11 is a frequency-aware and illumination-adaptive underwater small object detector built on the YOLO11n framework. It is designed for challenging underwater scenes with light attenuation, scattering, blur, non-uniform illumination, and dense occlusion among benthic organisms.

Main features:

- **Frequency-Aware Downsampling (FAD):** uses Haar 2D-DWT to expose low-frequency approximation and high-frequency directional sub-band information before learnable channel compression.
- **Illumination-Adaptive Feature Fusion (IAFF):** generates a spatial response map from channel-pooled features and applies bounded residual recalibration under uneven underwater illumination.
- **WIoU v3 training loss:** improves bounding-box localization robustness for blurred, small, and mutually occluded targets.
- **Underwater benchmarks:** evaluated on URPC2020-ZJ and DUO for holothurian, echinus, scallop, and starfish detection.
- **Efficiency-oriented design:** improves detection accuracy over YOLO11n while maintaining a compact model size and real-time GPU inference speed.

This repository provides the reference implementation of the proposed modules, model configuration files, training entry points, and benchmarking script used to support the experiments in the paper.

## Public datasets

The raw datasets used in this study are public datasets and are not redistributed in this repository. Please download them from their public sources:

- URPC2020-ZJ: https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC2020_dataset/datasets
- DUO: https://github.com/chongweiliu/DUO

After downloading the datasets, update the local dataset roots in:

- `configs/urpc2020.yaml`
- `configs/duo.yaml`

## Environment

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1
- Ultralytics YOLO11-compatible training interface
- Input size: 640×640
- Batch size: 16
- Optimizer: SGD
- Epochs: 300
- Seeds: 0, 1, 2

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training

URPC2020-ZJ:

```bash
python train.py --model configs/fa_yolo11.yaml --data configs/urpc2020.yaml --epochs 300 --imgsz 640 --batch 16 --seed 0
```

DUO:

```bash
python train.py --model configs/fa_yolo11.yaml --data configs/duo.yaml --epochs 300 --imgsz 640 --batch 16 --seed 0
```

For the three-seed protocol, repeat training with `--seed 0`, `--seed 1`, and `--seed 2`.

## Benchmarking

```bash
python benchmark.py --weights runs/train/fa_yolo11/weights/best.pt --imgsz 640 --batch 1
```

