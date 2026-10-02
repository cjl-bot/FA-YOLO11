# FA-YOLO11

## Abstract

FA-YOLO11 is a frequency-aware and illumination-adaptive underwater small object detector built on the YOLO11n framework. It introduces a Frequency-Aware Downsampling (FAD) module based on Haar 2D-DWT to preserve low- and high-frequency sub-band information during downsampling, an Illumination-Adaptive Feature Fusion (IAFF) module to perform spatially varying residual feature recalibration, and WIoU v3 as the bounding-box regression loss during training. The method is evaluated on URPC2020-ZJ and DUO for underwater benthic target detection.

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

