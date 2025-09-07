#!/usr/bin/env python3
# jsonToYOLO.py
# Convert JSON COCO-like labels file(probe_labels.json)
# to YOLOv8 format with splits train/val/test split (80/10/10 by default).
# Assumes a single class dataset (class_id=0).

import argparse
import json
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set

# ----------------------------
# Types and helpers
# ----------------------------

ImageInfo = Dict[str, object]
Annotation = Dict[str, object]
DatasetDict = Dict[str, list]

def load_dataset(json_path: Path) -> Tuple[Dict[int, ImageInfo], Dict[int, List[Annotation]]]:
    """Load JSON and group annotations by image_id."""
    try:
        data: DatasetDict = json.loads(json_path.read_text(encoding="utf-8"))
    except Exception as e:
        raise RuntimeError(f"Could not read JSON file: {json_path}\n{e}")

    if "images" not in data or "annotations" not in data:
        raise ValueError("JSON must contain 'images' and 'annotations'.")

    images = {int(im["id"]): im for im in data["images"]}
    anns_by_img: Dict[int, List[Annotation]] = {img_id: [] for img_id in images.keys()}
    for ann in data["annotations"]:
        img_id = int(ann["image_id"])
        if img_id in anns_by_img:
            anns_by_img[img_id].append(ann)
        else:
            pass
    return images, anns_by_img

def compute_splits(
    img_ids: List[int],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Set[int], Set[int], Set[int]]:
    """Shuffle and split ids into train/val/test sets."""
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0 (e.g. 0.8, 0.1, 0.1).")

    rng = random.Random(seed)
    ids = list(img_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(train_ratio * n))
    n_val = int(round(val_ratio * n))
    # ensure sum == n
    n_test = n - n_train - n_val

    train_ids = set(ids[:n_train])
    val_ids = set(ids[n_train:n_train + n_val])
    test_ids = set(ids[n_train + n_val:])
    return train_ids, val_ids, test_ids

def ensure_dirs(out_dir: Path) -> None:
    """Create the output folder structure."""
    for split in ["train", "val", "test"]:
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

def yolo_line(bbox: List[float], W: int, H: int, class_id: int = 0) -> str:
    """Convert bbox [x_min, y_min, w, h] to normalized YOLO line."""
    x, y, w, h = bbox
    xc = (x + w / 2.0) / float(W)
    yc = (y + h / 2.0) / float(H)
    wn = w / float(W)
    hn = h / float(H)
    # clamp just in case
    xc = min(max(xc, 0.0), 1.0)
    yc = min(max(yc, 0.0), 1.0)
    wn = min(max(wn, 0.0), 1.0)
    hn = min(max(hn, 0.0), 1.0)
    return f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}"

def move_image(
    src: Path,
    dst: Path,
    strategy: str = "copy",
) -> None:
    """Copy or symlink an image to the output folder."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if strategy == "copy":
        shutil.copy2(src, dst)
    elif strategy == "symlink":
        # Borramos si existe y creamos symlink relativo
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        rel = Path(os.path.relpath(src, dst.parent))  # type: ignore
        dst.symlink_to(rel)
    else:
        raise ValueError("strategy must be 'copy' or 'symlink'.")

def write_labels(
    lbl_path: Path,
    anns: List[Annotation],
    W: int,
    H: int,
    class_id: int = 0,
) -> None:
    """Write YOLO label file for one image (empty if no boxes)."""
    lines = [yolo_line(ann["bbox"], W, H, class_id) for ann in anns]
    lbl_path.parent.mkdir(parents=True, exist_ok=True)
    lbl_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

def write_yaml(out_dir: Path, class_name: str) -> None:
    """Generate data.yaml for YOLOv8."""
    content = f"""path: {out_dir}
train: images/train
val: images/val
test: images/test
names:
  0: {class_name}
"""
    (out_dir / "data.yaml").write_text(content, encoding="utf-8")

def convert(
    json_path: Path,
    images_root: Path,
    out_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    class_name: str = "object",
    class_id: int = 0,
    seed: int = 42,
    link_strategy: str = "copy",  # "copy" o "symlink"
    strict_images: bool = False,
) -> None:
    """Complete conversion pipeline."""
    images, anns_by_img = load_dataset(json_path)
    ensure_dirs(out_dir)

    img_ids = list(images.keys())
    train_ids, val_ids, test_ids = compute_splits(img_ids, train_ratio, val_ratio, test_ratio, seed)

    missing = 0
    processed = 0

    for img_id, im in images.items():
        file_name = str(im["file_name"])
        W = int(im["width"])
        H = int(im["height"])
        stem = Path(file_name).stem
        anns = anns_by_img.get(img_id, [])

        if img_id in train_ids:
            split = "train"
        elif img_id in val_ids:
            split = "val"
        else:
            split = "test"

        src_img = images_root / file_name
        dst_img = out_dir / "images" / split / Path(file_name).name
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"

        if not src_img.exists():
            missing += 1
            msg = f"[WARN] Image not found: {src_img}"
            if strict_images:
                raise FileNotFoundError(msg)
            else:
                print(msg, file=sys.stderr)
                write_labels(lbl_path, anns, W, H, class_id)
                continue

        move_image(src_img, dst_img, link_strategy)
        write_labels(lbl_path, anns, W, H, class_id)
        processed += 1

    write_yaml(out_dir, class_name)

    print(f"data.yaml written to: {out_dir/'data.yaml'}")
    print(f"Processed images: {processed} | Missing images: {missing}")
    print(f"Structure created under: {out_dir}/images/{{train,val,test}} and {out_dir}/labels/{{train,val,test}}")

# ----------------------------
# CLI
# ----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert a COCO-like JSON (without categories) + images to YOLOv8 dataset with splits."
    )
    p.add_argument("--json", type=Path, default=Path("data/probe_labels.json"),
                   help="Path to JSON annotation file.")
    p.add_argument("--images-dir", type=Path, default=Path("data/probe_images"),
                   help="Root folder containing images.")
    p.add_argument("--out-dir", type=Path, default=Path("datasets"),
                   help="Output YOLOv8 dataset directory.")
    p.add_argument("--train", type=float, default=0.8, help="Train ratio (0-1).")
    p.add_argument("--val", type=float, default=0.1, help="Val ratio (0-1).")
    p.add_argument("--test", type=float, default=0.1, help="Test ratio (0-1).")
    p.add_argument("--class-name", type=str, default="object", help="Name of the single class.")
    p.add_argument("--class-id", type=int, default=0, help="YOLO class ID (usually 0).")
    p.add_argument("--seed", type=int, default=42, help="Random seed for split reproducibility.")
    p.add_argument("--link-strategy", type=str, choices=["copy", "symlink"], default="copy",
                   help="How to handle images: 'copy' or 'symlink'.")
    p.add_argument("--strict-images", action="store_true",
                   help="If set, throw error when an image is missing instead of just warning.")
    return p.parse_args()

def main():
    args = parse_args()
    convert(
        json_path=args.json,
        images_root=args.images_dir,
        out_dir=args.out_dir,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        class_name=args.class_name,
        class_id=args.class_id,
        seed=args.seed,
        link_strategy=args.link_strategy,
        strict_images=args.strict_images,
    )

if __name__ == "__main__":
    main()
