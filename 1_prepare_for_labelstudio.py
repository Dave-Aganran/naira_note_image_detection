"""
Step 1: Flatten the per-class image folders into a single folder per split
so they can be easily imported into Label Studio.

Output structure:
  label_studio_import/
    train/   <- all training images (prefixed with denomination)
    val/     <- all val images
    test/    <- all test images
"""

import shutil
from pathlib import Path

DATASET_ROOT = Path("Naira/NAIRA DATASET")
OUTPUT_ROOT = Path("label_studio_import")

DENOMINATIONS = ["5", "10", "20", "50", "100", "200", "500", "1000"]
SPLITS = {"Train": "train", "val": "val", "Test": "test"}

def prepare():
    for src_split, dst_split in SPLITS.items():
        out_dir = OUTPUT_ROOT / dst_split
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for denom in DENOMINATIONS:
            src_dir = DATASET_ROOT / src_split / denom
            if not src_dir.exists():
                print(f"  WARNING: {src_dir} not found, skipping")
                continue
            for img in src_dir.iterdir():
                if img.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    # Prefix filename with denomination to avoid collisions
                    dest = out_dir / f"{denom}_{img.name}"
                    shutil.copy2(img, dest)
                    count += 1

        print(f"{dst_split}: {count} images copied -> {out_dir}")

    print("\nDone. Import the folders from 'label_studio_import/' into Label Studio.")

if __name__ == "__main__":
    prepare()
