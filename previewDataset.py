import cv2
from pathlib import Path

def save_yolo_labels(img_path, labels_path, out_path):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]

    if labels_path.exists():
        with open(labels_path) as f:
            for line in f:
                cls, cx, cy, bw, bh = map(float, line.split())
                x1, y1 = int((cx - bw/2) * w), int((cy - bh/2) * h)
                x2, y2 = int((cx + bw/2) * w), int((cy + bh/2) * h)
                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(img, str(int(cls)), (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), img)

# Example usage:
split = ["train", "val", "test"]   # change to "train" or "test"

for set in split:
    images_dir = Path("datasets/images") / set
    for img_file in images_dir.glob("*.jpg"):
        labels_dir = Path("datasets/labels") / set
        label_file = labels_dir / (img_file.stem + ".txt")
        out_dir = Path("datasets/preview") / set
        out_file   = out_dir / img_file.name
        save_yolo_labels(img_file, label_file, out_file)

print(f"Annotated images saved in {out_dir}")
