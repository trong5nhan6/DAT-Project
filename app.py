#!/usr/bin/env python3
"""Report demo GUI — a small Flask web app that looks exactly like the approved mockup.

    python app.py            # -> http://127.0.0.1:5000 (opens the browser)

Frontend is demo_ui.html (the mockup, made interactive). This backend is a thin
presentation layer over the trained checkpoints in weights/ (5 s-scale models):
it reuses demo.py (model registration, box drawing, class colours, GT parsing) and
models/base_model._compute_efficiency_metrics for the numbers — nothing here touches
training/eval. Efficiency figures are MEASURED live on the loaded checkpoint (s-scale),
not the n-scale figures quoted in the README table.

Endpoints
  GET  /                -> the UI page
  GET  /api/init        -> device, model list (+params), sample stems, VID sequences
  POST /api/detect      -> per-model annotated images + detections + efficiency
  POST /api/video       -> build a GT+models grid mp4, returns a token
  GET  /api/video/<tok> -> stream that mp4
"""

import base64
import os
import tempfile
import time
import uuid
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # Anaconda libiomp5md clash on Windows

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file

from demo import (
    CLASS_COLORS,
    CLASS_NAMES,
    DEMO_IMAGE_SETS,
    WEIGHTS,
    draw_boxes,
    vid_gt_boxes,
    yolo_label_boxes,
)

ROOT = Path(__file__).resolve().parent
VAL_IMG_DIR = ROOT / "datasets" / "VisDrone" / "images" / "val"
VAL_LBL_DIR = ROOT / "datasets" / "VisDrone" / "labels" / "val"
VID_ROOT = ROOT / "datasets" / "VisDrone2019-VID-val" / "VisDrone2019-VID-val"

app = Flask(__name__)

# lazy singletons ------------------------------------------------------------
_MODELS = None
_DEVICE = None
_EFF_CACHE = {}      # (name, imgsz) -> efficiency dict
_VIDEOS = {}         # token -> mp4 path


def device() -> str:
    global _DEVICE
    if _DEVICE is None:
        import torch

        _DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    return _DEVICE


def models() -> dict:
    """Register every idea's custom modules once, then load each weight that exists."""
    global _MODELS
    if _MODELS is None:
        from models.fap.register import register_fap
        from models.lwso.register import register_lwso
        from models.slim.register import register_slim
        from models.star.register import register_star

        register_lwso()
        register_fap()
        register_star()
        register_slim()

        from ultralytics import YOLO

        _MODELS = {name: YOLO(str(w)) for name, w in WEIGHTS.items() if Path(w).exists()}
    return _MODELS


def efficiency(name: str, imgsz: int) -> dict:
    key = (name, imgsz)
    if key not in _EFF_CACHE:
        import torch

        from models.base_model import _compute_efficiency_metrics

        inner = models()[name].model
        dev = torch.device(device())
        inner.to(dev)
        runs = 30 if dev.type == "cuda" else 8  # CPU forward is slow; fewer timed runs
        _EFF_CACHE[key] = _compute_efficiency_metrics(inner, imgsz, dev, warmup=5, runs=runs)
    return _EFF_CACHE[key]


# helpers --------------------------------------------------------------------
def predict(model, img_bgr, conf: float, imgsz: int):
    """Return (boxes, latency_ms). boxes: [(x1,y1,x2,y2,cls_id), ...]."""
    t0 = time.perf_counter()
    r = model.predict(img_bgr, imgsz=imgsz, conf=conf, device=device(), verbose=False)[0]
    dt = (time.perf_counter() - t0) * 1000
    boxes = [
        (x1, y1, x2, y2, int(c))
        for (x1, y1, x2, y2), c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.cls.cpu().numpy())
    ]
    return boxes, dt


def to_data_uri(img_bgr, quality=88) -> str:
    ok, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def open_writer(base: Path, fps: int, size):
    """VideoWriter with a browser-playable codec first (WebM/VP8), mp4v as fallback.

    OpenCV's mp4v (MPEG-4 Part 2) mp4 will NOT play inline in Chrome/Firefox, and
    H.264/avc1 needs libopenh264 which fails to init on this Windows build — so WebM/VP8
    (bundled ffmpeg supports it) is the reliable inline-playable choice.
    """
    for fourcc, ext in (("VP80", ".webm"), ("mp4v", ".mp4")):
        path = base.with_suffix(ext)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if vw.isOpened():
            return vw, path
        vw.release()
    raise RuntimeError("Không mở được VideoWriter với codec nào.")


def read_sample(stem: str):
    p = VAL_IMG_DIR / f"{stem}.jpg"
    if not p.exists():
        return None, None
    img = cv2.imread(str(p))
    lbl = VAL_LBL_DIR / f"{stem}.txt"
    gt = None
    if lbl.exists():
        h, w = img.shape[:2]
        gt = yolo_label_boxes(lbl, w, h)
    return img, gt


# routes ---------------------------------------------------------------------
@app.get("/")
def index():
    return send_file(ROOT / "demo_ui.html")


@app.get("/api/init")
def api_init():
    from ultralytics.utils.torch_utils import get_num_params

    ms = models()
    model_info = [{"name": n, "params_m": round(get_num_params(m.model) / 1e6, 2)}
                  for n, m in ms.items()]
    samples = []
    for group in DEMO_IMAGE_SETS.values():
        for stem in group:
            if (VAL_IMG_DIR / f"{stem}.jpg").exists():
                samples.append(stem)
    seq_root = VID_ROOT / "sequences"
    default_folder = ""
    if seq_root.exists():
        for dd in sorted(p for p in seq_root.iterdir() if p.is_dir()):
            if any(dd.glob("*.jpg")):
                default_folder = str(dd)
                break
    return jsonify({
        "device": "CUDA" if device().startswith("cuda") else "CPU",
        "models": model_info,
        "samples": samples,
        "vid_root": str(seq_root),
        "default_folder": default_folder,
        "classes": [{"name": n, "color": "#%02X%02X%02X" % (c[2], c[1], c[0])}
                    for n, c in zip(CLASS_NAMES, CLASS_COLORS)],
    })


@app.get("/api/browse")
def api_browse():
    """Server-side folder browser for the video tab (app runs locally)."""
    raw = request.args.get("path") or str(VID_ROOT / "sequences")
    p = Path(raw)
    if not p.exists() or not p.is_dir():
        p = VID_ROOT / "sequences" if (VID_ROOT / "sequences").exists() else Path(ROOT)
    try:
        dirs = sorted(d.name for d in p.iterdir() if d.is_dir())
    except (PermissionError, OSError):
        dirs = []
    frames = len(list(p.glob("*.jpg")))
    parent = str(p.parent) if p.parent != p else None
    return jsonify({"path": str(p), "parent": parent, "dirs": dirs, "frames": frames})


@app.post("/api/detect")
def api_detect():
    d = request.get_json(force=True)
    sel = d.get("models", [])
    conf, imgsz = float(d.get("conf", 0.25)), int(d.get("imgsz", 960))
    show_gt = bool(d.get("show_gt", True))

    if d.get("sample"):
        img, gt = read_sample(d["sample"])
        caption = d["sample"]
        if img is None:
            return jsonify({"error": f"Không thấy ảnh mẫu: {d['sample']}"}), 400
    elif d.get("upload"):
        raw = base64.b64decode(d["upload"].split(",", 1)[-1])
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        gt, caption = None, d.get("name", "ảnh tải lên")
        if img is None:
            return jsonify({"error": "Không đọc được ảnh tải lên."}), 400
    else:
        return jsonify({"error": "Chưa có ảnh vào."}), 400
    if not sel:
        return jsonify({"error": "Chọn ít nhất 1 model."}), 400

    ms = models()
    results = []
    for name in sel:
        boxes, _ = predict(ms[name], img, conf, imgsz)
        eff = efficiency(name, imgsz)
        results.append({
            "name": name,
            "img": to_data_uri(draw_boxes(img, boxes)),
            "det": len(boxes),
            # Show the averaged, warmed benchmark latency (same source as the table) instead
            # of a single-shot per-image timing — the latter is noisy on CPU and contradicts
            # the table's FPS (cold-start / run-to-run variance).
            "ms": round(eff["latency_ms"], 1),
            "params_m": round(eff["params_m"], 3),
            "gflops": round(eff["gflops"], 1),
            "latency_ms": round(eff["latency_ms"], 1),
            "fps": round(eff["fps"], 1),
        })

    gt_payload = None
    if show_gt and gt is not None:
        gt_payload = {"img": to_data_uri(draw_boxes(img, gt)), "count": len(gt)}

    return jsonify({
        "input": to_data_uri(img),
        "caption": caption,
        "imgsz": imgsz,
        "gt": gt_payload,
        "results": results,
    })


@app.post("/api/video")
def api_video():
    from demo import legend_bar, title_bar

    d = request.get_json(force=True)
    sel = d.get("models", [])
    folder = d.get("folder")
    conf, imgsz = float(d.get("conf", 0.25)), int(d.get("imgsz", 768))
    fps = int(d.get("fps", 25))
    max_frames = d.get("max_frames")  # None or int
    if not folder or not sel:
        return jsonify({"error": "Cần chọn thư mục video và ít nhất 1 model."}), 400

    seq_dir = Path(folder)
    if not seq_dir.is_dir():
        return jsonify({"error": f"Không phải thư mục: {folder}"}), 400
    seq = seq_dir.name
    # Ground-Truth chỉ có khi thư mục là một sequence của VisDrone-VID (khớp file annotation)
    ann = VID_ROOT / "annotations" / f"{seq}.txt"
    frames = sorted(seq_dir.glob("*.jpg"))
    if max_frames:
        frames = frames[: int(max_frames)]
    if not frames:
        return jsonify({"error": f"Thư mục không có frame (*.jpg): {folder}"}), 400
    gt = vid_gt_boxes(ann) if ann.exists() else {}

    ms = models()
    cell_w, cell_h = 480, 270
    cells = (["Ground Truth"] if gt else []) + sel  # GT column only when annotation exists
    # Layout fits the actual cell count so we don't pad with black tiles. Only 5 cells
    # (4 models + GT) can't tile a full rectangle -> one filler; every other count is exact.
    LAYOUT = {1: (1, 1), 2: (1, 2), 3: (1, 3), 4: (2, 2), 5: (2, 3), 6: (2, 3)}
    nrows, ncols = LAYOUT.get(len(cells), ((len(cells) + 2) // 3, 3))
    legend = legend_bar(cell_w * ncols)
    grid_h = nrows * (34 + cell_h) + legend.shape[0]
    bars = {c: title_bar(cell_w, c) for c in cells}
    blank = np.full((34 + cell_h, cell_w, 3), 20, np.uint8)

    vw, out_path = open_writer(Path(tempfile.mkdtemp()) / f"grid_{seq}", fps,
                               (cell_w * ncols, grid_h))
    for fpath in frames:
        img = cv2.imread(str(fpath))
        try:
            fid = int(fpath.stem)
        except ValueError:
            fid = -1  # non-numeric frame name -> no GT match
        tiles = []
        for c in cells:
            boxes = gt.get(fid, []) if c == "Ground Truth" else predict(ms[c], img, conf, imgsz)[0]
            tile = cv2.resize(draw_boxes(img, boxes), (cell_w, cell_h))
            tiles.append(np.vstack([bars[c], tile]))
        while len(tiles) % ncols:
            tiles.append(blank)
        rows = [np.hstack(tiles[r * ncols:(r + 1) * ncols]) for r in range(nrows)]
        vw.write(np.vstack(rows + [legend]))
    vw.release()

    tok = uuid.uuid4().hex[:12]
    _VIDEOS[tok] = out_path
    return jsonify({"token": tok, "frames": len(frames), "name": out_path.name})


@app.get("/api/video/<tok>")
def api_video_file(tok):
    path = _VIDEOS.get(tok)
    if not path or not Path(path).exists():
        return "not found", 404
    mime = "video/webm" if Path(path).suffix == ".webm" else "video/mp4"
    return send_file(path, mimetype=mime,
                     as_attachment=request.args.get("dl") == "1",
                     download_name=Path(path).name)


if __name__ == "__main__":
    import threading
    import webbrowser

    url = "http://127.0.0.1:5000"
    print(f"\n  LWSO-YOLO demo → {url}\n")
    threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host="127.0.0.1", port=5000, threaded=True)
