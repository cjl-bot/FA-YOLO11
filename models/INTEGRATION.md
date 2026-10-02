# Integrating FA-YOLO11 modules into a YOLO11 codebase

The provided implementation is intentionally lightweight and framework-neutral. To run it inside a local YOLO11/Ultralytics codebase, register the custom modules before parsing `configs/fa_yolo11.yaml`.

## Required modules

- `FAD(c1, c2)`: replaces selected stride-2 convolutional downsampling layers.
- `IAFF(c1, alpha=0.5, gamma=2.0)`: recalibrates one feature tensor before feature fusion.
- `WIoUv3Loss(alpha_w=1.9, delta_w=3.0)`: replaces the bounding-box regression loss during training.

## Parser note

Most YOLO-style parsers infer the input channel number `c1` from the previous layer and read the output channel number `c2` from the YAML argument list. Therefore, the YAML entry

```yaml
- [-1, 1, FAD, [256]]
```

should be interpreted as:

```python
FAD(c1, c2=256)
```

Similarly, the YAML entry

```yaml
- [-1, 1, IAFF, [0.5, 2.0]]
```

should be interpreted as:

```python
IAFF(c1, alpha=0.5, gamma=2.0)
```

If the local parser does not support this convention, manually instantiate the modules according to the same argument order.

## Loss integration

WIoU v3 is used only during training and does not change the inference graph. In practice, replace the IoU regression term in the local YOLO11 loss file with `WIoUv3Loss`, while keeping the classification and DFL terms unchanged.

The manuscript reports the loss weights:

```text
lambda_cls = 0.5
lambda_dfl = 1.5
lambda_box = 7.5
```

and WIoU v3 hyperparameters:

```text
alpha_w = 1.9
delta_w = 3.0
```

