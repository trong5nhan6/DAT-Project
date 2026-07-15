#!/usr/bin/env python3
"""Convert VisDrone2019-DET annotations to YOLO format.

VisDrone annotation line:
    <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>

Categories: 0 = ignored-regions, 1..10 = pedestrian..motor, 11 = others.
Categories 1..10 map to YOLO classes 0..9. Categories 0 and 11 are dropped;
with --mask-ignored their regions are filled with gray (114) in the image copy so
the model is neither trained on nor penalized inside them (recommended).

Usage:
    python convert_visdrone.py --src D:/datasets/visdrone_raw --mask-ignored
    # expects  <src>/VisDrone2019-DET-{train,val,test-dev}/{images,annotations}/

Output layout (matches data/visdrone.yaml):
    <dst>/images/{train,val,test}/*.jpg
    <dst>/labels/{train,val,test}/*.txt
"""

import argparse
import shutil
import sys
from pathlib import Path

SPLITS = {
    "VisDrone2019-DET-train": "train",
    "VisDrone2019-DET-val": "val",
    "VisDrone2019-DET-test-dev": "test",
}
IGNORE_CATS = {0, 11}  # ignored-regions, others
GRAY = (114, 114, 114)  # matches ultralytics letterbox padding value


def convert_split(src: Path, dst: Path, split: str, mask_ignored: bool) -> tuple[int, int]:
    img_dir, ann_dir = src / "images", src / "annotations"
    out_img = dst / "images" / split
    out_lbl = dst / "labels" / split
    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)

    if mask_ignored:
        import cv2
    from PIL import Image

    n_img, n_box = 0, 0
    for ann in sorted(ann_dir.glob("*.txt")):
        img_path = img_dir / f"{ann.stem}.jpg"
        if not img_path.exists():
            print(f"  [warn] missing image for {ann.name}, skipped", file=sys.stderr)
            continue
        with Image.open(img_path) as im:
            iw, ih = im.size

        yolo_lines, ignore_boxes = [], []
        for raw in ann.read_text().splitlines():
            parts = raw.strip().strip(",").split(",")
            if len(parts) < 6:
                continue
            x, y, w, h, _score, cat = (int(float(p)) for p in parts[:6])
            if cat in IGNORE_CATS:
                ignore_boxes.append((x, y, w, h))
                continue
            # clip to image, drop degenerate boxes
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + w, iw), min(y + h, ih)
            if x2 - x1 < 1 or y2 - y1 < 1:
                continue
            cx, cy = (x1 + x2) / 2 / iw, (y1 + y2) / 2 / ih
            bw, bh = (x2 - x1) / iw, (y2 - y1) / ih
            yolo_lines.append(f"{cat - 1} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            n_box += 1

        (out_lbl / f"{ann.stem}.txt").write_text("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        if mask_ignored and ignore_boxes:
            img = cv2.imread(str(img_path))
            for x, y, w, h in ignore_boxes:
                x1, y1 = max(x, 0), max(y, 0)
                x2, y2 = min(x + w, iw), min(y + h, ih)
                if x2 > x1 and y2 > y1:
                    img[y1:y2, x1:x2] = GRAY
            cv2.imwrite(str(out_img / img_path.name), img)
        else:
            shutil.copy2(img_path, out_img / img_path.name)
        n_img += 1
    return n_img, n_box


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, type=Path,
                    help="directory containing the VisDrone2019-DET-* split folders")
    ap.add_argument("--dst", type=Path,
                    default=Path(__file__).resolve().parents[2] / "datasets" / "VisDrone",
                    help="output dataset root (default: <repo>/datasets/VisDrone)")
    ap.add_argument("--mask-ignored", action="store_true",
                    help="gray-fill ignored-regions/others in image copies (recommended)")
    args = ap.parse_args()

    found = False
    for folder, split in SPLITS.items():
        src_split = args.src / folder
        if not src_split.exists():
            print(f"[skip] {folder} not found under {args.src}")
            continue
        found = True
        print(f"[{split}] converting {folder} ...")
        n_img, n_box = convert_split(src_split, args.dst, split, args.mask_ignored)
        print(f"[{split}] {n_img} images, {n_box} boxes -> {args.dst}")
    if not found:
        sys.exit(f"error: no VisDrone2019-DET-* folders found under {args.src}")
    print(f"\nDone. Point data/visdrone.yaml 'path:' to {args.dst}")


if __name__ == "__main__":
    main()
