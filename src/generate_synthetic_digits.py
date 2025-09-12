"""
Generate a synthetic handwritten-like digits dataset using OpenCV.

Creates an image-folder dataset:
  <out>/train/0..9/*.png
  <out>/test/0..9/*.png

Usage examples:
  python src/generate_synthetic_digits.py --out data/synth_digits
  python src/generate_synthetic_digits.py --out data/synth_digits --per-class 3000 --img-size 28
"""

from __future__ import annotations

import os
import math
import random
import argparse
from typing import Tuple

import numpy as np
import cv2


FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_PLAIN,
]


def ensure_dir(p: str) -> None:
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)


def render_digit(d: int, img_size: int = 28) -> np.ndarray:
    """Render a single digit with random style/augmentations."""
    canvas = np.zeros((img_size, img_size), dtype=np.uint8)

    font = random.choice(FONTS)
    # Random scale maps roughly to font size
    scale = random.uniform(0.7, 1.4)
    thickness = random.randint(1, 3)

    # Measure text size to center
    text = str(d)
    (tw, th), baseline = cv2.getTextSize(text, font, scale, thickness)
    x = (img_size - tw) // 2 + random.randint(-2, 2)
    y = (img_size + th) // 2 + random.randint(-2, 2)

    # Draw white digit on black
    cv2.putText(canvas, text, (x, y), font, scale, 255, thickness, lineType=cv2.LINE_AA)

    # Random rotation
    if random.random() < 0.6:
        ang = random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((img_size // 2, img_size // 2), ang, 1.0)
        canvas = cv2.warpAffine(canvas, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Random shift
    if random.random() < 0.6:
        tx, ty = random.randint(-2, 2), random.randint(-2, 2)
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        canvas = cv2.warpAffine(canvas, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # Random blur/noise
    if random.random() < 0.5:
        k = random.choice([3, 5])
        canvas = cv2.GaussianBlur(canvas, (k, k), sigmaX=0.7)
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(5, 20), canvas.shape).astype(np.float32)
        canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    # Optional inversion (keep MNIST-like polarity by default)
    if random.random() < 0.15:
        canvas = 255 - canvas

    return canvas


def generate_split(out_root: str, split: str, per_class: int, img_size: int) -> None:
    for d in range(10):
        cls_dir = os.path.join(out_root, split, str(d))
        ensure_dir(cls_dir)
        for i in range(per_class):
            img = render_digit(d, img_size=img_size)
            path = os.path.join(cls_dir, f"{d}_{i:05d}.png")
            cv2.imwrite(path, img)


def main():
    ap = argparse.ArgumentParser(description="Generate synthetic digit dataset")
    ap.add_argument("--out", required=True, help="Output dataset root directory")
    ap.add_argument("--img-size", type=int, default=28, help="Image size (square)")
    ap.add_argument("--per-class", type=int, default=2000, help="Images per class for train; test uses 1/5")
    args = ap.parse_args()

    out_root = args.out
    img_size = int(args.img_size)
    per_class_train = int(args.per_class)
    per_class_test = max(1, per_class_train // 5)

    print(f"Creating dataset at: {out_root}")
    print(f"Train per class: {per_class_train} | Test per class: {per_class_test}")

    generate_split(out_root, "train", per_class_train, img_size)
    generate_split(out_root, "test", per_class_test, img_size)

    print("Done. Example structure:")
    print(os.path.join(out_root, "train", "0", "0_00000.png"))
    print(os.path.join(out_root, "test", "0", "0_00000.png"))


if __name__ == "__main__":
    main()