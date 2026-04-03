"""
Auto-annotation script: detects the Naira note in each image using OpenCV
contour detection and generates YOLO-format label files automatically.

Each image has one note on a plain background — this approach works well
for that setup. Review a sample of outputs to verify quality.

Output structure:
  datasets/naira/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    data.yaml
"""

import shutil
from pathlib import Path
import cv2
import numpy as np

DATASET_ROOT = Path("Naira/NAIRA DATASET")
OUTPUT_ROOT = Path("datasets/naira")

DENOMINATIONS = ["5", "10", "20", "50", "100", "200", "500", "1000"]
CLASS_INDEX = {d: i for i, d in enumerate(DENOMINATIONS)}
CLASS_NAMES = [f"{d}_Naira" for d in DENOMINATIONS]

SPLITS = {"Train": "train", "val": "val", "Test": "test"}


def detect_note_bbox(img_path: Path):
    """
    Returns (cx, cy, w, h) normalized [0-1] for the largest contour in the image.
    Falls back to full-image box if detection fails.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        return None

    h, w = img.shape[:2]

    # Convert to grayscale and blur
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)

    # Threshold: notes are darker than the white/light background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Morphological closing to fill holes in the note
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        # Fallback: use 90% of image centered
        return (0.5, 0.5, 0.9, 0.9)

    # Take the largest contour
    largest = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(largest)

    # Add small padding (2%)
    pad_x = int(0.02 * w)
    pad_y = int(0.02 * h)
    x = max(0, x - pad_x)
    y = max(0, y - pad_y)
    bw = min(w - x, bw + 2 * pad_x)
    bh = min(h - y, bh + 2 * pad_y)

    # Convert to YOLO format (normalized center x, center y, width, height)
    cx = (x + bw / 2) / w
    cy = (y + bh / 2) / h
    nw = bw / w
    nh = bh / h

    return (cx, cy, nw, nh)


def process_split(src_split: str, dst_split: str):
    img_out = OUTPUT_ROOT / "images" / dst_split
    lbl_out = OUTPUT_ROOT / "labels" / dst_split
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    total = fallback = 0

    for denom in DENOMINATIONS:
        src_dir = DATASET_ROOT / src_split / denom
        if not src_dir.exists():
            continue

        class_id = CLASS_INDEX[denom]

        for img_path in src_dir.iterdir():
            if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
                continue

            bbox = detect_note_bbox(img_path)
            if bbox is None:
                continue

            cx, cy, nw, nh = bbox

            # Sanity check — if box is suspiciously small, use fallback
            if nw < 0.1 or nh < 0.1:
                cx, cy, nw, nh = 0.5, 0.5, 0.9, 0.9
                fallback += 1

            # Copy image
            dest_img = img_out / f"{denom}_{img_path.name}"
            shutil.copy2(img_path, dest_img)

            # Write label
            lbl_path = lbl_out / f"{denom}_{img_path.stem}.txt"
            lbl_path.write_text(f"{class_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

            total += 1

    print(f"[{dst_split}] {total} annotated ({fallback} fallback boxes)")


def write_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    yaml_path.write_text(
        f"path: {OUTPUT_ROOT.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names: {CLASS_NAMES}\n"
    )
    print(f"data.yaml written -> {yaml_path}")


if __name__ == "__main__":
    print("Auto-annotating dataset...\n")
    for src, dst in SPLITS.items():
        process_split(src, dst)
    write_yaml()
    print("\nDone! Run 3_train.py to start training.")
