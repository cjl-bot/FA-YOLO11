# FA-YOLO11

This repository provides the reference code organization for **FA-YOLO11: A Frequency-Aware and Illumination-Adaptive Network for Underwater Small Object Detection**.

FA-YOLO11 is built on the YOLO11n detection pipeline and introduces three main changes:

1. **Frequency-Aware Downsampling (FAD)** replaces selected stride-2 downsampling transitions with a Haar 2D-DWT decomposition followed by learnable channel compression.
2. **Illumination-Adaptive Feature Fusion (IAFF)** generates a spatial response map from channel-pooled features and applies bounded residual feature recalibration.
3. **Wise-IoU v3 (WIoU v3)** is used as the bounding-box regression loss during training.

The code is organized to document the training protocol, dataset split protocol, and the core implementation logic of the proposed modules. Depending on the local YOLO11/Ultralytics version, minor adaptation of the model parser may be needed before direct execution.

## Repository structure

```text
FA-YOLO11/
├── README.md
├── requirements.txt
├── train.py
├── val.py
├── infer.py
├── configs/
│   ├── fa_yolo11.yaml
│   ├── urpc2020.yaml
│   └── duo.yaml
├── models/
│   ├── __init__.py
│   ├── modules.py
│   └── fa_yolo11.py
├── scripts/
│   └── prepare_splits.py
├── splits/
│   ├── URPC2020/
│   │   └── README.md
│   └── DUO/
│       └── README.md
└── requirements.txt
```

## Datasets

The experiments use two public underwater object detection datasets. The original datasets were not created by the authors of this repository and are not redistributed here. Please download the raw images and annotations from their official sources, then use the split files and configuration templates in this repository to reproduce the experimental protocol.

- **URPC2020-ZJ**: 5,543 training images, 1,200 validation/model-selection images from test-B, and 800 final test images from test-A.
- **DUO**: 6,671 training images and 1,111 testing images using the official DUO split.

The exact image lists used for each split should be placed under:

```text
splits/URPC2020/train_5543.txt
splits/URPC2020/val_testB_1200.txt
splits/URPC2020/testA_800.txt
splits/DUO/train_6671.txt
splits/DUO/test_1111.txt
```

Each line should contain a relative or absolute image path. Dataset root paths are configured in `configs/urpc2020.yaml` and `configs/duo.yaml`.

## Environment

The experimental setting follows the manuscript:

- Python 3.10
- PyTorch 2.1.0
- CUDA 12.1
- Ultralytics YOLO11-compatible training interface
- Input size: 640×640
- Batch size: 16
- Optimizer: SGD
- Epochs: 300
- Seeds: 0, 1, 2

Install basic dependencies:

```bash
pip install -r requirements.txt
```

## Training

URPC2020:

```bash
python train.py \
  --model configs/fa_yolo11.yaml \
  --data configs/urpc2020.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --seed 0
```

DUO:

```bash
python train.py \
  --model configs/fa_yolo11.yaml \
  --data configs/duo.yaml \
  --epochs 300 \
  --imgsz 640 \
  --batch 16 \
  --seed 0
```

For the three-seed protocol, repeat training with `--seed 0`, `--seed 1`, and `--seed 2`.

## Validation

```bash
python val.py \
  --weights runs/train/fa_yolo11/weights/best.pt \
  --data configs/urpc2020.yaml \
  --split test \
  --imgsz 640
```

For DUO, replace the data file with `configs/duo.yaml`.

## Inference

```bash
python infer.py \
  --weights runs/train/fa_yolo11/weights/best.pt \
  --source path/to/images \
  --imgsz 640
```

## Notes on module integration

The core logic of FAD and IAFF is provided in `models/modules.py`. To run the model directly inside a specific YOLO11 codebase, register `FAD` and `IAFF` in the model parser used by that codebase. The provided `configs/fa_yolo11.yaml` follows the module placement described in the paper:

- FAD is inserted at the middle backbone downsampling transitions.
- IAFF is inserted in the neck feature fusion path.
- WIoU v3 is used only during training and does not change the inference graph.

## Data availability

This repository provides the training entry points, model configuration files, module implementations, dataset preparation script, and exact split-file locations used in the paper. The raw URPC2020-ZJ and DUO datasets are public datasets and should be obtained from their official dataset sources. After downloading the datasets, place the corresponding split files under `splits/URPC2020/` and `splits/DUO/`, and update the local dataset roots in `configs/urpc2020.yaml` and `configs/duo.yaml`.

