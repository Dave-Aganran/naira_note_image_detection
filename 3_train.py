"""
Step 3: Train a YOLOv8 object detection model on the annotated Naira dataset.
Run this after 2_convert_annotations.py has built datasets/naira/.
"""

from ultralytics import YOLO
from pathlib import Path

DATA_YAML = Path("datasets/naira/data.yaml")
MODEL = "yolov8n.pt"   # nano — fast; swap to yolov8s.pt for better accuracy
EPOCHS = 50
IMG_SIZE = 640
BATCH = 16
PROJECT = "runs/naira"
RUN_NAME = "detect"


def train():
    if not DATA_YAML.exists():
        print(f"ERROR: {DATA_YAML} not found. Run 2_convert_annotations.py first.")
        return

    model = YOLO(MODEL)
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        project=PROJECT,
        name=RUN_NAME,
        exist_ok=True,
    )
    print(f"\nTraining complete. Best weights: {PROJECT}/{RUN_NAME}/weights/best.pt")
    return results


if __name__ == "__main__":
    train()
