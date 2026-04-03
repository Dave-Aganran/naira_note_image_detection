"""
Step 4: Evaluate the trained model and run inference on test images.
"""

from ultralytics import YOLO
from pathlib import Path

WEIGHTS = Path("weights/best.pt")
DATA_YAML = Path("datasets/naira/data.yaml")
TEST_IMAGES = Path("datasets/naira/images/test")


def evaluate():
    if not WEIGHTS.exists():
        print(f"ERROR: {WEIGHTS} not found. Run 3_train.py first.")
        return

    model = YOLO(str(WEIGHTS))

    # Validation metrics (mAP, precision, recall per class)
    print("=== Validation Metrics ===")
    metrics = model.val(data=str(DATA_YAML), split="test")
    print(f"mAP50:     {metrics.box.map50:.3f}")
    print(f"mAP50-95:  {metrics.box.map:.3f}")
    print(f"Precision: {metrics.box.mp:.3f}")
    print(f"Recall:    {metrics.box.mr:.3f}")

    # Per-class breakdown
    print("\n=== Per-class AP50 ===")
    for name, ap in zip(metrics.names.values(), metrics.box.ap50):
        print(f"  {name}: {ap:.3f}")

    # Visual inference on test images
    if TEST_IMAGES.exists():
        print(f"\nRunning inference on {TEST_IMAGES} ...")
        model.predict(
            source=str(TEST_IMAGES),
            save=True,
            project="runs/naira",
            name="test_predictions",
            conf=0.25,
        )
        print("Predictions saved to runs/naira/test_predictions/")


if __name__ == "__main__":
    evaluate()
