# LWSO-YOLO — LightWeight Small-Object YOLO cho VisDrone2019

Cải tiến YOLO11n cho small object trên ảnh UAV: **thêm đầu detect P2 (stride 4), bỏ tầng P5**,
SPD-Conv (downsample không mất thông tin), C3k2Ghost (nhẹ hơn ~40% FLOPs/block),
EMA attention, neck BiFPN-style + DySample, loss CIoU⊕NWD.
Không fork ultralytics — module custom được đăng ký runtime qua `lwso.register_lwso()`.

## Cấu trúc

```
lwso/                  package chính
  modules.py           SPDConv, C3k2Ghost, EMA, DySample, BiFPNCat
  register.py          patch parse_model của ultralytics (pin 8.3.x)
  losses.py            NWD blend loss (thay BboxLoss)
cfg/
  lwso-yolo11n.yaml    model đề xuất (nano)
  lwso-yolo11s.yaml    bản s — teacher cho knowledge distillation
  ablation/yolo11n-p2-nop5.yaml   chỉ +P2/−P5, module gốc 100%
data/
  visdrone.yaml        dataset config (sửa `path:` nếu chuyển máy)
  scripts/convert_visdrone.py     VisDrone → YOLO format
train.py / val.py      entry point train & eval
tests/                 pytest shape/param sanity tests
```

## Bắt đầu

```bash
pip install -r requirements.txt
pytest tests/ -v                      # sanity check trước khi train

# 1) Tải VisDrone2019-DET (train/val/test-dev) rồi convert:
python data/scripts/convert_visdrone.py --src <thư mục chứa VisDrone2019-DET-*> --mask-ignored

# 2) Baseline để so sánh (M1):
python train.py --model yolo11n.pt --imgsz 640 --batch 16 --name base-11n-640 --no-nwd
python train.py --model yolo11n.pt --imgsz 960 --batch 8  --name base-11n-960 --no-nwd

# 3) Ablation +P2/−P5 (M2):
python train.py --model cfg/ablation/yolo11n-p2-nop5.yaml --imgsz 960 --name abl-p2 --no-nwd

# 4) Model đầy đủ (M3+M4):
python train.py --model cfg/lwso-yolo11n.yaml --imgsz 960 --name lwso-n-960

# Đánh giá:
python val.py --weights runs/detect/lwso-n-960/weights/best.pt --split val
```

## Ghi chú quan trọng

- **VRAM**: đầu P2 @960 rất tốn bộ nhớ. batch 8 ≈ 12GB. Thiếu VRAM → giảm `--imgsz 768`
  hoặc `--batch 4`. Không dùng `--multi-scale` nếu <16GB.
- **NWD constant** (`--nwd-constant`): mặc định 12.8 theo paper (đơn vị pixel tuyệt đối),
  nhưng trong loss của ultralytics box ở đơn vị stride-normalized → nên ablate C ∈ {2, 4, 8, 12.8}.
- **Pin ultralytics 8.3.x**: `register.py` patch source `parse_model` bằng 4 phép thay thế
  có kiểm tra; sai version sẽ raise RuntimeError ngay khi gọi `register_lwso()` (fail sớm, không hỏng ngầm).
- **Ignored regions**: luôn convert với `--mask-ignored` — vùng ignore được tô xám 114
  để model không bị phạt oan khi có object thật trong đó.
- **Anaconda trên Windows**: nếu gặp `OMP Error #15 (libiomp5md.dll)`, các script đã tự set
  `KMP_DUPLICATE_LIB_OK=TRUE`; fix triệt để là dùng venv/conda env riêng thay vì base env.

## Lộ trình (đã bàn ở thiết kế)

- [x] M1 script convert + baseline config
- [x] M2 ablation +P2/−P5 (YAML)
- [x] M3 module custom + đăng ký runtime + unit tests
- [x] M4 NWD blend loss
- [ ] M5 chạy ablation đầy đủ, gom kết quả (cần GPU)
- [ ] M6 nén: prune (Torch-Pruning/LAMP, bảo vệ nhánh P2) → distill (teacher `lwso-yolo11s`, CWD)
      → export ONNX/TensorRT INT8

## Kết quả

Params/GFLOPs đo thực bằng `ultralytics.utils.torch_utils.get_flops` (mAP điền sau khi train):

| Run | imgsz | mAP50 | mAP50-95 | Params | GFLOPs@640 | GFLOPs@960 |
|---|---|---|---|---|---|---|
| yolo11n baseline | 640/960 | | | 2.62M | 6.5 | ~14.6 |
| +P2/−P5 (ablation) | 960 | | | **1.49M** | 16.8 | 37.8 |
| LWSO-YOLO11n | 960 | | | **2.40M** | 21.5 | 48.4 |

GFLOPs cao hơn baseline là chủ đích: compute được dồn về tầng phân giải cao (P2) — nơi
small object còn tồn tại. Để tham chiếu, LRDS-YOLO (43.6% mAP50) dùng 24.1 GFLOPs@640
với 4.17M params. Cần nhẹ hơn nữa → giảm kênh P2 64→48 trong YAML, hoặc chờ bước M6
(prune + INT8).
