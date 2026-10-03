# FA-YOLO11

This repository provides a reference implementation for **FA-YOLO11**, a
frequency-aware and illumination-adaptive underwater small-object detector based
on a YOLO11-compatible training interface.

## Overview

FA-YOLO11 is designed for underwater scenes where object detection is affected by
light attenuation, scattering, blur, non-uniform illumination, dense targets, and
camouflaged benthic organisms. The implementation follows the revised manuscript
in three aspects:

- **Frequency-Aware Downsampling (FAD):** replaces the four post-stem stride-2
  downsampling transitions at `P2/4`, `P3/8`, `P4/16`, and `P5/32`. Each FAD
  layer exposes the Haar `LL`, `LH`, `HL`, and `HH` sub-bands before 1x1
  learnable channel compression.
- **Illumination-Adaptive Feature Fusion (IAFF):** obtains two spatial
  descriptors through channel-wise average and maximum pooling, then uses a
  deformable 3x3 mask branch to generate the learned response map `M_illu`.
  The default hidden width is set from the incoming feature channels so that the
  four IAFF branches match the parameter-count scale reported in the paper.
- **WIoU v3 optimization:** replaces the bbox IoU regression term during
  training while leaving the classification and DFL terms unchanged. The
  implementation includes the center-distance term, detached enclosing-box
  denominator, non-monotonic focusing coefficient, and cross-batch EMA baseline.
- **DUO protocol:** DUO has no separate validation subset in the original
  7,782-image benchmark. Training with `configs/duo.yaml` disables epoch-wise
  validation by default and evaluates the final/last checkpoint once on the test
  split.
- **Benchmarking protocol:** `benchmark.py` measures in-memory Ultralytics
  prediction, including resize/letterbox preprocessing, normalization, model
  forward, and NMS. Disk I/O, camera capture, visualization, file writing, and
  robot-control latency are excluded.

## Public datasets

The datasets used in the study are public third-party datasets and are not
redistributed in this repository.

- URPC2020-ZJ: https://openi.pcl.ac.cn/OpenOrcinus_orca/URPC2020_dataset/datasets
- DUO: https://github.com/chongweiliu/DUO

After downloading the datasets, update the local dataset roots in:

- `configs/urpc2020.yaml`
- `configs/duo.yaml`

No additional split files are provided in this repository. URPC2020-ZJ follows
the public train/test-A/test-B release structure: `train` is used for fitting,
`test-B` is used as the validation/model-selection subset, and `test-A` is used
as the final test subset. DUO follows its official train/test fine-annotation
split.

## Environment

The paper setting uses:

- Python 3.10
- PyTorch 2.1.0
- Torchvision 0.16.0 with `torchvision.ops.DeformConv2d`
- CUDA 12.1
- Ultralytics 8.3.0 YOLO11 interface
- Input size: 640 x 640
- Batch size: 16 for training
- Optimizer: SGD
- Epochs: 300
- Seeds: 0, 1, 2

Install dependencies:

```bash
pip install -r requirements.txt
```

The Ultralytics version is pinned because FA-YOLO11 registers custom modules by
patching the YOLO11 model parser. If you use another Ultralytics version, inspect
and update `models/ultralytics_integration.py` before training.

## Training

URPC2020-ZJ:

```bash
python train.py --model configs/fa_yolo11.yaml --data configs/urpc2020.yaml --epochs 300 --imgsz 640 --batch 16 --seed 0 --name fa_yolo11_urpc_seed0
```

DUO:

```bash
python train.py --model configs/fa_yolo11.yaml --data configs/duo.yaml --epochs 300 --imgsz 640 --batch 16 --seed 0 --name fa_yolo11_duo_seed0
```

For DUO, `train.py` automatically enables the final-checkpoint protocol
(`val=False`) so that the test annotations are not used for checkpoint
selection. For the three-seed protocol, repeat training with `--seed 0`,
`--seed 1`, and `--seed 2`.

To run an ablation without WIoU v3:

```bash
python train.py --model configs/fa_yolo11.yaml --data configs/urpc2020.yaml --no-wiou
```

## Validation and testing

URPC2020 final test:

```bash
python val.py --weights runs/detect/fa_yolo11_urpc_seed0/weights/best.pt --data configs/urpc2020.yaml --split test --imgsz 640 --batch 1
```

DUO final test with the final/last checkpoint:

```bash
python val.py --weights runs/detect/fa_yolo11_duo_seed0/weights/last.pt --data configs/duo.yaml --split test --imgsz 640 --batch 1
```

## Benchmarking

FP32, batch-size 1, synchronized CUDA timing:

```bash
python benchmark.py --weights runs/detect/fa_yolo11_urpc_seed0/weights/best.pt --imgsz 640 --batch 1 --warmup 100 --repeat 1000
```

The script reports mean latency, latency standard deviation, and FPS. The timed
scope is printed explicitly as:

```text
ultralytics_predict_preprocess_forward_nms
```

This means the benchmark includes in-memory preprocessing, model forward, and
NMS, but excludes disk I/O and downstream robot-control latency.
