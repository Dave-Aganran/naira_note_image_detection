# Nigerian Naira Note Detection — YOLOv8

**Course:** Tools and Platform For Data Science (611)  
**Student:** Oluwaseun Aganran | Matric No. 251201330003  
**University:** Pan-Atlantic University

A YOLOv8 object detection model that detects and classifies all eight Nigerian Naira banknote denominations: ₦5, ₦10, ₦20, ₦50, ₦100, ₦200, ₦500, ₦1000.

---

## Results

| Metric | Score |
|---|---|
| Precision | 0.947 |
| Recall | 0.993 |
| mAP50 | 0.992 |
| mAP50-95 | 0.915 |
| Inference speed | ~8 ms / image |

### Per-class AP50 (test set)

| Denomination | AP50 | AP50-95 |
|---|---|---|
| ₦5 | 0.984 | 0.984 |
| ₦10 | 0.995 | 0.684 |
| ₦20 | 0.995 | 0.817 |
| ₦50 | 0.983 | 0.918 |
| ₦100 | 0.995 | 0.987 |
| ₦200 | 0.995 | 0.989 |
| ₦500 | 0.995 | 0.948 |
| ₦1000 | 0.995 | 0.995 |

---

## Project Structure

```
├── datasets/naira/
│   ├── labels/          # YOLO annotation labels (train/val/test)
│   └── data.yaml        # Class definitions for YOLO
├── weights/
│   └── best.pt          # Trained model weights (6MB)
├── auto_annotate.py     # Step 1 — auto-annotation via OpenCV contours
├── 3_train.py           # Step 2 — YOLOv8 training
├── 4_evaluate.py        # Step 3 — evaluation & test inference
└── README.md
```

> **Note:** Dataset images (~188MB) are not tracked in git due to size.  
> Labels and `data.yaml` are included so the dataset structure is reproducible.

---

## Setup

### Requirements

```bash
pip install ultralytics opencv-python pillow
```

> PyTorch with CUDA is recommended for training.  
> CPU-only inference works for evaluation.

---

## Running Evaluation (no retraining needed)

The trained weights are included at `runs/detect/runs/naira/detect/weights/best.pt`.

```bash
python 4_evaluate.py
```

This will print precision, recall, mAP50, mAP50-95, and per-class AP50 on the test set.

---

## Retraining from Scratch

If you want to retrain the model, you will need to supply the dataset images.  
Place them in:

```
datasets/naira/images/train/   — 1,736 images
datasets/naira/images/val/     — 216 images
datasets/naira/images/test/    — 90 images
```

Then run:

```bash
python 3_train.py
```

---

## Dataset Summary

| Denomination | Train | Val | Test |
|---|---|---|---|
| ₦5 | 137 | 28 | 9 |
| ₦10 | 138 | 20 | 9 |
| ₦20 | 236 | 28 | 12 |
| ₦50 | 250 | 28 | 12 |
| ₦100 | 278 | 28 | 12 |
| ₦200 | 210 | 28 | 12 |
| ₦500 | 277 | 28 | 12 |
| ₦1000 | 210 | 28 | 12 |
| **Total** | **1,736** | **216** | **90** |
