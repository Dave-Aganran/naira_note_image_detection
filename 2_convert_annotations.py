"""
Step 2: After annotating in Label Studio and exporting in YOLO format,
this script reorganises the export into the structure YOLO training expects.

Label Studio YOLO export structure:
  <export_dir>/
    images/
      *.jpg
    labels/
      *.txt
    classes.txt

Output structure:
  datasets/naira/
    images/train/  images/val/  images/test/
    labels/train/  labels/val/  labels/test/
    data.yaml
"""

import shutil
import sys
from pathlib import Path

DENOMINATIONS = ["5", "10", "20", "50", "100", "200", "500", "1000"]
CLASS_NAMES = [f"{d}_Naira" for d in DENOMINATIONS]  # must match Label Studio label order

SPLITS = ["train", "val", "test"]
OUTPUT_ROOT = Path("datasets/naira")


def convert(export_dir: Path, split: str):
    """Move images and labels from a Label Studio export into the YOLO dataset folder."""
    if split not in SPLITS:
        print(f"ERROR: split must be one of {SPLITS}")
        sys.exit(1)

    img_src = export_dir / "images"
    lbl_src = export_dir / "labels"

    img_dst = OUTPUT_ROOT / "images" / split
    lbl_dst = OUTPUT_ROOT / "labels" / split
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    img_count = lbl_count = 0
    for img in img_src.glob("*"):
        if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
            shutil.copy2(img, img_dst / img.name)
            img_count += 1

    for lbl in lbl_src.glob("*.txt"):
        shutil.copy2(lbl, lbl_dst / lbl.name)
        lbl_count += 1

    print(f"[{split}] {img_count} images, {lbl_count} labels -> {OUTPUT_ROOT}")


def write_yaml():
    yaml_path = OUTPUT_ROOT / "data.yaml"
    lines = [
        f"path: {OUTPUT_ROOT.resolve()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"nc: {len(CLASS_NAMES)}",
        f"names: {CLASS_NAMES}",
    ]
    yaml_path.write_text("\n".join(lines))
    print(f"data.yaml written -> {yaml_path}")


if __name__ == "__main__":
    # Usage: python 2_convert_annotations.py <export_dir> <split>
    # Example: python 2_convert_annotations.py exports/train_export train
    if len(sys.argv) != 3:
        print("Usage: python 2_convert_annotations.py <label_studio_export_dir> <train|val|test>")
        sys.exit(1)

    export_dir = Path(sys.argv[1])
    split = sys.argv[2]

    if not export_dir.exists():
        print(f"ERROR: {export_dir} does not exist")
        sys.exit(1)

    convert(export_dir, split)
    write_yaml()
