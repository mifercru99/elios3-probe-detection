#!/usr/bin/env python3
import argparse, time, os, sys, glob
from pathlib import Path
import cv2
from ultralytics import YOLO

def load_image_paths(folder):
    exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff')
    paths = []
    for e in exts:
        paths.extend(sorted(glob.glob(str(Path(folder) / e))))
    return paths

def main():
    p = argparse.ArgumentParser(description="Probe detector (YOLOv8) – flyability test")
    p.add_argument("input_dir", type=str, help="Folder with input images")
    p.add_argument("--weights", type=str, default="runs_probe/y8s_640_e100/weights/best.pt",
                   help="Path to trained weights .pt")
    p.add_argument("--out_dir", type=str, default="predictions",
                   help="Folder to save annotated images")
    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    p.add_argument("--show", action="store_true", help="Also show images in a window")
    p.add_argument("--device", type=str, default=None, help="cuda:0 or cpu (auto if omitted)")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    imgs = load_image_paths(args.input_dir)
    if not imgs:
        print(f"No images found in {args.input_dir}", file=sys.stderr)
        sys.exit(1)

    model = YOLO(args.weights)
    # Small speed bump for CPU/Jetson later if desired:
    # model.fuse()  # fuse Conv+BN at inference

    total = 0.0
    n = 0

    for img_path in imgs:
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}", file=sys.stderr)
            continue

        t0 = time.time()
        results = model.predict(
            source=img,                 # ndarray: one by one
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
            imgsz=640
        )
        dt = time.time() - t0
        total += dt
        n += 1

        # There is one image in this batch
        r = results[0]
        annotated = r.plot()  # Ultralytics renders bboxes + labels

        if len(r.boxes) == 0:
            # Overlay a clear banner if nothing detected
            text = "NO PROBE DETECTED"
            (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            cv2.rectangle(annotated, (10, 10), (20 + w, 30 + h), (0, 0, 255), -1)
            cv2.putText(annotated, text, (15, 30 + h - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        out_path = str(Path(args.out_dir) / Path(img_path).name)
        cv2.imwrite(out_path, annotated)

        print(f"{Path(img_path).name:40s}  {dt*1000:7.1f} ms  "
              f"n_boxes={len(r.boxes)}")

        if args.show:
            cv2.imshow("probe detection", annotated)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

    if n:
        print(f"\nProcessed {n} images | Avg time {1000*total/n:.1f} ms | "
              f"Throughput ~{n/total:.2f} FPS")

    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# Example usage:
# python inference.py path/to/images --weights runs_probe/y8s_640_e100/weights/best.pt  --out_dir predictions --device cpu