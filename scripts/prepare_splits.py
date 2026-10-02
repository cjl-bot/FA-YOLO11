"""Prepare split text files for FA-YOLO11 experiments.

The script is intentionally conservative: it records image paths and does not move,
rename, or delete dataset files.
"""

from __future__ import annotations

import argparse
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def collect_images(folder: Path):
    return sorted(p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS)


def write_list(images, output: Path, root: Path | None = None):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for path in images:
            item = path
            if root is not None:
                try:
                    item = path.relative_to(root)
                except ValueError:
                    item = path
            f.write(str(item).replace("\\", "/") + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Dataset root directory.")
    parser.add_argument("--train-dir", required=True, help="Training image directory.")
    parser.add_argument("--val-dir", default=None, help="Validation image directory.")
    parser.add_argument("--test-dir", required=True, help="Testing image directory.")
    parser.add_argument("--out-dir", required=True, help="Output split directory.")
    parser.add_argument("--prefix", required=True, choices=["URPC2020", "DUO"])
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir).resolve()

    train_images = collect_images((root / args.train_dir).resolve())
    test_images = collect_images((root / args.test_dir).resolve())

    if args.prefix == "URPC2020":
        val_images = collect_images((root / args.val_dir).resolve()) if args.val_dir else []
        write_list(train_images, out_dir / "train_5543.txt", root)
        write_list(val_images, out_dir / "val_testB_1200.txt", root)
        write_list(test_images, out_dir / "testA_800.txt", root)
        print(f"URPC2020 split files written to {out_dir}")
        print(f"train={len(train_images)}, val={len(val_images)}, test={len(test_images)}")
    else:
        write_list(train_images, out_dir / "train_6671.txt", root)
        write_list(test_images, out_dir / "test_1111.txt", root)
        print(f"DUO split files written to {out_dir}")
        print(f"train={len(train_images)}, test={len(test_images)}")


if __name__ == "__main__":
    main()

