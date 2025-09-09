# Probe Detection with YOLOv8

This repository contains a deep learning pipeline to detect a single class, the **ultrasonic probe** , in Elios3 drone images.  
The project includes dataset conversion, training, evaluation, and inference utilities.

---

## Repository Structure

```
├── datasets/ # Dataset in YOLO format
│ ├── data.yaml # Defines paths and class names
│ ├── images/{train,val,test}/
│ └── labels/{train,val,test}/
├── runs_probe/ # Training outputs (checkpoints, metrics, logs)
│ └── y8s_640_e100/weights/best.pt # Best trained weights
├── predictions/ # Output folder for inference results
├── inference.py # Script for running inference (assignment requirement)
├── jsonToYOLO.py # Converts COCO JSON annotations -> YOLO format
├── previewDataset.py # Utility to visualize dataset samples
├── train_yolov8_probe_detection.ipynb # Training notebook
├── README.md # This file
└── report.md  # Final project report in markdown
└── report.pdf # Final project report in pdf
```
---

## Pretrained Weights

- Best checkpoint: runs_probe/y8s_640_e100/weights/best.pt
- These weights were trained on the provided dataset (train/val/test = 246/31/31).

---

## How to Run Inference

The main entry point is `inference.py`.  
It processes all images in a folder **one by one**, detects probes, draws bounding boxes, and saves the annotated outputs.

### Example (CPU, validation set):
```bash
python inference.py data/val/images \
  --weights runs_probe/y8s_640_e100/weights/best.pt \
  --out_dir predictions \
  --device cpu
```
Arguments:

-`path/to/images` → folder of images to process (e.g. data/val/images)
-`weights` → path to trained model (.pt)
-`out_dir` → where results will be saved (default: predictions/)
-`device` → cpu or cuda:0

Outputs:
- Annotated images with bounding boxes in predictions/
- Console log with filename, detections, confidence, and average runtime

---
## Results (summary)
- Validation: Precision 0.997, Recall 0.968, F1 0.982, mAP@50 0.991, mAP@50–95 0.909
- Test: mAP@50 0.995, mAP@50–95 0.922
- Runtime (CPU): ~123 ms/image (~8.1 FPS, including I/O + drawing)

---
## Notes
- Dataset: 308 labeled images, single class probe, no true negatives.
- Framework: Ultralytics YOLOv8
- For training details, metrics, and future improvements, see docs/report.pdf.

---
