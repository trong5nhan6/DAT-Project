#!/usr/bin/env python3
"""Qualitative demo for the report: side-by-side comparison of all 5 models.

  python demo.py images                      # 7-column grid: Input | GT | Baseline | LWSO-eff | SLIM | FAP | STAR
  python demo.py video --seq uav0000137_00458_v   # 2x3 grid video: GT | Baseline | LWSO-eff / SLIM | FAP | STAR

Outputs go to "final report/assets/demo/". Weights are the s-scale checkpoints in
weights/ -- swap WEIGHTS paths below if n-scale checkpoints become available.
"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "final report" / "assets" / "demo"

# display name -> checkpoint (order = column/cell order after Input/GT)
WEIGHTS = {
    "Baseline (YOLO11-s)": ROOT / "weights" / "best_base-yolo11s.pt",
    "LWSO-eff": ROOT / "weights" / "best_lwso-yolo11s-eff.pt",
    "SLIM": ROOT / "weights" / "best_slim-yolo11s.pt",
    "FAP (pruned)": ROOT / "weights" / "best_fap-yolo11s_pruned.pt",
    "STAR": ROOT / "weights" / "best_star-yolo11s.pt",
}

DEMO_IMAGE_SETS = {
    "1": [
        "0000295_02400_d_0000033",  # dense urban intersection, day
        "0000296_01001_d_0000040",  # congested road, oblique view
        "0000129_02411_d_0000138",  # top-down street/parking, vehicle-heavy
        "0000086_01954_d_0000005",  # basketball courts, pedestrian-only
        "0000117_00112_d_0000087",  # night intersection
    ],
    "2": [
        "0000291_03201_d_0000884",  # dense shopping-street intersection
        "0000213_03920_d_0000243",  # low-altitude oblique street, cars+pedestrians
        "0000313_06601_d_0000469",  # high-altitude town overview, tiny objects
        "0000155_00401_d_0000001",  # boulevard, mixed traffic with motorbikes
        "0000055_00000_d_0000109",  # night market plaza
    ],
    "3": [
        "0000001_08414_d_0000013",  # residential streets with tree cover
        "0000330_04201_d_0000821",  # large crosswalk intersection, mixed traffic
        "0000271_06001_d_0000402",  # dusk shopping street, parked cars both sides
        "0000356_00589_d_0000632",  # sunlit intersection with strong glare
        "0000115_00796_d_0000081",  # night plaza with trees
    ],
    "4": [
        "0000244_03500_d_0000008",  # long street canyon, shops both sides
        "0000216_00520_d_0000001",  # top-down crossroad
        "0000215_00000_d_0000256",  # roadside parking rows along avenue
        "0000277_02601_d_0000552",  # shopping-street intersection, vans+cars
        "0000289_05601_d_0000839",  # backlit street with lens flare, dusk
    ],
    "5": [
        "0000194_00399_d_0000121",  # wide intersection with cyclists
        "0000269_00001_d_0000348",  # tree-lined avenue, dense traffic
        "0000280_01201_d_0000618",  # busy commercial avenue
        "0000333_01765_d_0000010",  # top-down vacant lot and street
        "0000327_04001_d_0000731",  # dusk street, headlights on
    ],
}

CLASS_NAMES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
               "tricycle", "awning-tricycle", "bus", "motor"]
# BGR, one distinct color per class
CLASS_COLORS = [
    (36, 28, 237),   # pedestrian - red
    (39, 127, 255),  # people - orange
    (0, 242, 255),   # bicycle - yellow
    (76, 177, 34),   # car - green
    (232, 162, 0),   # van - azure
    (204, 72, 63),   # truck - blue
    (164, 73, 163),  # tricycle - purple
    (21, 0, 136),    # awning-tricycle - dark red
    (190, 146, 112), # bus - steel blue
    (231, 191, 200), # motor - pink
]

CONF = 0.25
IMGSZ = 960


def load_models():
    from models.fap.register import register_fap
    from models.lwso.register import register_lwso
    from models.slim.register import register_slim
    from models.star.register import register_star

    register_lwso()
    register_fap()
    register_star()
    register_slim()

    from ultralytics import YOLO

    return {name: YOLO(str(w)) for name, w in WEIGHTS.items()}


def draw_boxes(img, boxes, thickness=2):
    """boxes: iterable of (x1, y1, x2, y2, cls_id) in pixel coords."""
    out = img.copy()
    for x1, y1, x2, y2, c in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)),
                      CLASS_COLORS[int(c)], thickness)
    return out


def pred_boxes(model, img):
    r = model.predict(img, imgsz=IMGSZ, conf=CONF, verbose=False)[0]
    return [(x1, y1, x2, y2, int(c)) for (x1, y1, x2, y2), c in
            zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())]


def yolo_label_boxes(label_file, w, h):
    boxes = []
    with open(label_file) as f:
        for line in f:
            p = line.split()
            if len(p) < 5:
                continue
            c, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
            boxes.append(((xc - bw / 2) * w, (yc - bh / 2) * h,
                          (xc + bw / 2) * w, (yc + bh / 2) * h, c))
    return boxes


def title_bar(width, text, height=34, scale=0.8):
    bar = np.full((height, width, 3), 32, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    cv2.putText(bar, text, ((width - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return bar


def legend_bar(width, height=42):
    bar = np.full((height, width, 3), 32, np.uint8)
    x = 10
    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
        cv2.rectangle(bar, (x, height // 2 - 8), (x + 18, height // 2 + 8), color, -1)
        x += 24
        (tw, _), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(bar, name, (x, height // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        x += tw + 22
    return bar


# ---------------------------------------------------------------- images demo

def side_label(height, text, width=48):
    """Vertical (rotated 90°) row label strip."""
    canvas = np.full((width, height, 3), 32, np.uint8)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(canvas, text, ((height - tw) // 2, (width + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return cv2.rotate(canvas, cv2.ROTATE_90_COUNTERCLOCKWISE)


def run_images(models, stems, suffix=""):
    img_dir = ROOT / "datasets" / "VisDrone" / "images" / "val"
    lbl_dir = ROOT / "datasets" / "VisDrone" / "labels" / "val"
    panel_dir = OUT_DIR / "image_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    cols = ["Input", "Ground Truth"] + list(models)
    panels_by_image = {}  # stem -> [full-res panel per col]
    for stem in stems:
        img = cv2.imread(str(img_dir / f"{stem}.jpg"))
        h, w = img.shape[:2]
        panels = [img, draw_boxes(img, yolo_label_boxes(lbl_dir / f"{stem}.txt", w, h))]
        for name, model in models.items():
            panels.append(draw_boxes(img, pred_boxes(model, img)))
        panels_by_image[stem] = panels
        print(f"  {stem} done")

    # -- horizontal grid: one row per image, one column per method (cells 640 wide)
    cell_w = 640
    rows = []
    for stem, panels in panels_by_image.items():
        h, w = panels[0].shape[:2]
        cell_h = round(h * cell_w / w)
        resized = [cv2.resize(p, (cell_w, cell_h)) for p in panels]
        for col, p in zip(cols, resized):
            safe = col.split(" ")[0].lower().replace("(", "").replace(")", "")
            cv2.imwrite(str(panel_dir / f"{stem}_{safe}.jpg"), p,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(np.hstack(resized))
    header = np.hstack([title_bar(cell_w, c) for c in cols])
    grid = np.vstack([header] + rows + [legend_bar(cell_w * len(cols))])
    out = OUT_DIR / f"demo_images_grid{suffix}.jpg"
    cv2.imwrite(str(out), grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", out, grid.shape)

    # -- vertical grid: one row per method, one column per image (cells 420 tall,
    #    width follows each image's aspect ratio; rotated labels on the left)
    cell_h = 420
    widths = [round(panels_by_image[s][0].shape[1] * cell_h / panels_by_image[s][0].shape[0])
              for s in stems]
    v_rows = []
    for r, col in enumerate(cols):
        tiles = [cv2.resize(panels_by_image[s][r], (widths[j], cell_h))
                 for j, s in enumerate(stems)]
        v_rows.append(np.hstack([side_label(cell_h, col)] + tiles))
    v_grid = np.vstack(v_rows + [legend_bar(v_rows[0].shape[1])])
    out_v = OUT_DIR / f"demo_images_grid_vertical{suffix}.jpg"
    cv2.imwrite(str(out_v), v_grid, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print("saved", out_v, v_grid.shape)


# ----------------------------------------------------------------- video demo

def vid_gt_boxes(ann_file):
    """VisDrone-VID annotation -> {frame_id: [(x1,y1,x2,y2,cls), ...]} (cats 1-10 only)."""
    per_frame = {}
    with open(ann_file) as f:
        for line in f:
            p = line.strip().split(",")
            if len(p) < 8:
                continue
            fid, _, x, y, w, h, score, cat = map(int, p[:8])
            if score == 0 or not 1 <= cat <= 10:
                continue
            per_frame.setdefault(fid, []).append((x, y, x + w, y + h, cat - 1))
    return per_frame


def run_video(models, seq, fps=25, max_frames=None):
    seq_dir = (ROOT / "datasets" / "VisDrone2019-VID-val" / "VisDrone2019-VID-val"
               / "sequences" / seq)
    ann = seq_dir.parent.parent / "annotations" / f"{seq}.txt"
    frames = sorted(seq_dir.glob("*.jpg"))[:max_frames]
    gt = vid_gt_boxes(ann)

    # collect predictions model-by-model, one frame at a time (a list source would
    # make ultralytics pre-load every image into RAM at once)
    preds = {}
    for name, model in models.items():
        preds[name] = []
        for i, f in enumerate(frames):
            preds[name].append(pred_boxes(model, str(f)))
            if (i + 1) % 50 == 0:
                print(f"  {name}: {i + 1}/{len(frames)} frames")
        print(f"  {name}: done ({len(frames)} frames)")

    cell_w, cell_h = 672, 378
    cells = ["Ground Truth"] + list(models)  # 6 cells -> 2x3
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"demo_video_{seq}.mp4"
    legend = legend_bar(cell_w * 3)
    grid_h = 2 * (34 + cell_h) + legend.shape[0]
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                         (cell_w * 3, grid_h))
    bars = {c: title_bar(cell_w, c) for c in cells}

    for i, fpath in enumerate(frames):
        img = cv2.imread(str(fpath))
        fid = int(fpath.stem)
        tiles = []
        for c in cells:
            boxes = gt.get(fid, []) if c == "Ground Truth" else preds[c][i]
            tile = cv2.resize(draw_boxes(img, boxes), (cell_w, cell_h))
            tiles.append(np.vstack([bars[c], tile]))
        row1, row2 = np.hstack(tiles[:3]), np.hstack(tiles[3:])
        vw.write(np.vstack([row1, row2, legend]))
        if (i + 1) % 50 == 0:
            print(f"  composing: {i + 1}/{len(frames)}")
    vw.release()
    print("saved", out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["images", "video"])
    ap.add_argument("--set", default="1", choices=list(DEMO_IMAGE_SETS),
                    help="which image set to render (output suffix _<set> for set != 1)")
    ap.add_argument("--seq", default="uav0000137_00458_v")
    ap.add_argument("--fps", type=int, default=25)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    models = load_models()
    if args.mode == "images":
        suffix = "" if args.set == "1" else f"_{args.set}"
        run_images(models, DEMO_IMAGE_SETS[args.set], suffix=suffix)
    else:
        run_video(models, args.seq, fps=args.fps, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
