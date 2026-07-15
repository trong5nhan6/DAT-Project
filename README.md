# LWSO-YOLO — LightWeight Small-Object YOLO cho VisDrone2019

Cải tiến YOLO11n cho small object trên ảnh UAV: **thêm đầu detect P2 (stride 4), bỏ tầng P5**,
SPD-Conv (downsample không mất thông tin), C3k2Ghost (nhẹ hơn ~40% FLOPs/block),
EMA attention, neck BiFPN-style + DySample, loss CIoU⊕NWD.
Không fork ultralytics — module custom được đăng ký runtime qua `models.lwso.register_lwso()`.

## Cấu trúc

```
models/                 mỗi idea = 1 module riêng, chọn qua --idea (xem "Thêm idea mới")
  base_model.py          BaseModel (ABC): build()/train()/get_callbacks() dùng chung mọi idea
                          + callback test-set monitoring mặc định (test_every)
  baseline.py            idea "baseline": stock YOLO11n, không module/loss custom
  lwso/                   idea "lwso": full LWSO model — subpackage vì nhiều thành phần
    model.py               LWSOModel: gọi register_lwso() + patch_nwd_loss() rồi build YOLO
    modules.py              SPDConv(+Group), C3k2Ghost, EMA, ECA, DySample, BiFPNCat
    register.py              patch parse_model của ultralytics (pin 8.3.x)
    losses.py                 NWD blend loss (thay BboxLoss)
  fap/                    idea "fap": FreqMix downsampling (từ paper FAP-YOLO12n)
    model.py               FAPModel: gọi register_fap() rồi build YOLO (không NWD)
    modules.py              FreqMix (Haar band decompose + learnable softmax mixing)
    register.py              patch parse_model của ultralytics (pin 8.3.x)
  _patch_utils.py         helper dùng chung cho lwso/register.py + fap/register.py — patch
                          parse_model cộng dồn đúng bất kể idea nào đăng ký trước
  __init__.py             MODEL_REGISTRY {idea: class} + build_model(idea, cfg)
cfg/                    kiến trúc model (.yaml theo format ultralytics parse_model)
  base-yolo11n.yaml     baseline: YOLO11n gốc (nc=10), train from scratch, module gốc 100%
  lwso-yolo11n.yaml     model đề xuất (nano)
  lwso-yolo11n-lite.yaml  cắt bớt compute của lwso-yolo11n (P2 ch 64→48, SPDConv chỉ ở P1→P2)
  lwso-yolo11n-eff.yaml   SPDConv→group (mọi tầng) + ECA ở P2 — nhẹ hơn & kỳ vọng mAP tốt hơn
  lwso-yolo11s.yaml     bản s — teacher cho knowledge distillation
  ablation/yolo11n-p2-nop5.yaml   chỉ +P2/−P5, module gốc 100%
  fap-yolo11n.yaml      backbone gốc YOLO11n + đầu P2 thêm vào, P3/P4/P5 giữ nguyên (4 head)
configs/                hyperparameter theo idea (train.py load, xem "Cấu hình train" bên dưới)
  base.yaml             default cho mọi run — mọi idea kế thừa file này
  baseline.yaml         idea "baseline": YOLO11n gốc, không NWD
  lwso.yaml             idea "lwso": full LWSO model + NWD
  fap.yaml              idea "fap": cfg/fap-yolo11n.yaml + protected_layers cho prune.py
data/
  visdrone.yaml        dataset config (sửa `path:` nếu chuyển máy)
  scripts/convert_visdrone.py     VisDrone → YOLO format
train.py / val.py      entry point train & eval
prune.py               semantic-path-aware LAMP pruning cho idea "fap" (xem mục riêng)
tests/                 pytest shape/param sanity tests
```

## Bắt đầu

```bash
pip install -r requirements.txt
pytest tests/ -v                      # sanity check trước khi train

# 1) Tải VisDrone2019-DET (train/val/test-dev) rồi convert:
python data/scripts/convert_visdrone.py --src <thư mục chứa VisDrone2019-DET-*> --mask-ignored

# 2) Baseline (M1) — configs/baseline.yaml: finetune yolo11n.pt từ COCO, không NWD
python train.py --idea baseline
#    train from-scratch thay vì finetune — cùng recipe với lwso để so sánh công bằng hơn:
python train.py --idea baseline --model cfg/base-yolo11n.yaml --name base-11n-scratch

# 3) Ablation +P2/−P5 (M2) — chưa có idea riêng, trỏ thẳng model, giữ recipe baseline (no-nwd):
python train.py --idea baseline --model cfg/ablation/yolo11n-p2-nop5.yaml --name abl-p2

# 4) Model đầy đủ (M3+M4) — configs/lwso.yaml: cfg/lwso-yolo11n.yaml + NWD blend loss
python train.py --idea lwso

# 4b) Bản lite (compute thấp hơn, chưa validate mAP) — cùng idea/recipe, chỉ đổi kiến trúc:
python train.py --idea lwso --model cfg/lwso-yolo11n-lite.yaml --name lwso-n-lite

# 4c) Bản eff (compute thấp hơn + ECA ở P2, kỳ vọng mAP tốt hơn — xem "Kết quả"):
python train.py --idea lwso --model cfg/lwso-yolo11n-eff.yaml --name lwso-n-eff

# 5) Idea fap (FreqMix + P2 head, giữ P3/P4/P5) — xem mục "Idea fap" bên dưới:
python train.py --idea fap
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3

# Đánh giá:
python val.py --weights runs/detect/lwso-n/weights/best.pt --split val
```

## Cấu hình train (`configs/` + `models/`)

`train.py` không hardcode hyperparameter hay kiến trúc — 2 tầng tách biệt:

- **`configs/`**: hyperparameter, merge theo thứ tự ưu tiên tăng dần —
  **`configs/base.yaml`** (default chung) → **`configs/<idea>.yaml`** (override theo `--idea`,
  hoặc trỏ thẳng bằng `--config <path>`) → **CLI flag** (override cuối cùng, chỉ áp dụng khi
  thực sự truyền — không truyền thì giữ nguyên giá trị từ config).
- **`models/`**: code build model, mỗi idea 1 class kế thừa `BaseModel`
  (`models/base_model.py`), đăng ký trong `models/__init__.py` (`MODEL_REGISTRY`).
  `--idea` của `train.py` nhận `choices` trực tiếp từ registry này.

```bash
python train.py --idea lwso                                   # dùng nguyên configs/lwso.yaml
python train.py --idea lwso --imgsz 640 --batch 16             # override vài key từ CLI
python train.py --config configs/lwso.yaml --epochs 50         # trỏ thẳng 1 file, bỏ qua --idea
```

### Thêm idea mới

Hiện có 2 idea: `baseline` (YOLO11n gốc) và `lwso` (full LWSO model). Thêm 1 idea, 3 bước,
**không cần sửa `train.py`**:

1. Tạo `models/<tên>.py` (hoặc `models/<tên>/` nếu nhiều file như `lwso/`) định nghĩa 1 class
   kế thừa `BaseModel`, implement `build()` (set `self._yolo`); override `get_callbacks()` nếu
   cần callback riêng, gọi `super().get_callbacks()` để giữ test-set monitoring mặc định.
2. Thêm entry vào `MODEL_REGISTRY` trong `models/__init__.py`.
3. Tạo `configs/<tên>.yaml` kế thừa `configs/base.yaml` (xem `configs/lwso.yaml`).

`python train.py --idea <tên>` chạy được ngay — `--help` tự liệt kê idea mới trong `choices`.

## Idea `fap` — FreqMix + semantic-path-aware pruning

Port phần representation stage của paper **FAP-YOLO12n** (`outline-final-drone.pdf`) sang
YOLO11n. Khác họ `lwso` (bỏ P5, thêm P2) — `fap` **giữ nguyên P3/P4/P5, chỉ thêm đầu P2** (4
head), vì lý luận của paper là bảo vệ P2/P3 (chi tiết) trong lúc nén mạnh P4/P5 (ngữ nghĩa),
không phải loại bỏ P4/P5 khỏi kiến trúc.

**1. FreqMix** (`models/fap/modules.py`) thay mọi downsample-conv trong `cfg/fap-yolo11n.yaml`:
Haar wavelet decompose (LL/LH/HL/HH, phép biến đổi cố định, không học) → mixing có trọng số
học được qua softmax (khởi tạo thiên về LL: `[2.0, 0, 0, -0.5]`) → conv 1x1 chiếu kênh. So với
concat+conv (HWD gốc), giữ nguyên số kênh sau biến đổi (không nhân 4) nên conv chiếu nhẹ hơn.

**2. Semantic-path-aware pruning** (`prune.py`, độc lập với train.py): LAMP importance score
(Lee et al., ICLR 2021) tính **cộng dồn toàn cục** trên mọi Conv2d, nhưng loại hẳn khỏi candidate
pool các layer thuộc đường P2/P3 (`configs/fap.yaml`'s `protected_layers`) + mọi `band_logits` +
Detect head — các layer này không bao giờ bị prune bất kể LAMP score thấp thế nào.

```bash
python train.py --idea fap                                              # train FreqMix dense
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3              # M6: LAMP+P2/P3 (đề xuất)
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-protect # M5: LAMP thường (ablation)
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-lamp --no-protect  # M4: magnitude đều (ablation)
```

**Giới hạn hiện tại**: `prune.py` chỉ zero weight + lưu checkpoint, **không** giữ mask trong
lúc fine-tune tiếp — nếu chạy `train.py --weights <pruned>.pt` sau đó, gradient sẽ làm các
weight đã prune trôi khỏi 0 dần (paper mô tả pipeline có "fine-tune ngắn" sau prune, nhưng
phần giữ sparsity trong lúc fine-tune chưa implement, ngoài phạm vi lần code này). Đã verify
build+forward+prune+reload bằng chạy thật (CPU, không cần GPU); **chưa có số liệu mAP** — cần
train thật trên GPU (Kaggle) theo ladder M1-M6 của paper để so sánh.

## Ghi chú quan trọng

- **VRAM**: đầu P2 @960 rất tốn bộ nhớ. batch 8 ≈ 12GB. Thiếu VRAM → giảm `--imgsz 768`
  hoặc `--batch 4`. Không dùng `--multi-scale` nếu <16GB.
- **NWD constant** (`--nwd-constant`): mặc định 12.8 theo paper (đơn vị pixel tuyệt đối),
  nhưng trong loss của ultralytics box ở đơn vị stride-normalized → nên ablate C ∈ {2, 4, 8, 12.8}.
- **Pin ultralytics 8.3.x**: `models/lwso/register.py` patch source `parse_model` bằng 4 phép
  thay thế có kiểm tra; sai version sẽ raise RuntimeError ngay khi gọi `register_lwso()`
  (fail sớm, không hỏng ngầm).
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
| LWSO-YOLO11n-lite | 960 | | | **1.12M** | 12.8 | 28.8 |
| LWSO-YOLO11n-eff | 960 | | | **0.96M** | 12.4 | 27.9 |
| FAP-YOLO11n (P2+P3+P4+P5) | 960 | | | 2.13M | 9.0 | 20.2 |

FAP-YOLO11n có 4 head (P2/P3/P4/P5, không bỏ P5) — kiến trúc khác họ LWSO ở trên (3 head,
bỏ P5), nên GFLOPs không so 1-1 được; đối chiếu với baseline stock (6.5 GFLOPs@640) thì FAP
chỉ tăng ~1.4x (so với 3.3x của LWSO đầy đủ) vì không dồn hết compute vào 1 tầng phân giải
cao nhất — bù lại phần nén (prune P4/P5) chưa được đo mAP.

GFLOPs cao hơn baseline là chủ đích: compute được dồn về tầng phân giải cao (P2) — nơi
small object còn tồn tại. Để tham chiếu, LRDS-YOLO (43.6% mAP50) dùng 24.1 GFLOPs@640
với 4.17M params.

**Bản lite** (`cfg/lwso-yolo11n-lite.yaml`): thu hẹp kênh P2 64→48 + chỉ dùng `SPDConv` ở
tầng P1→P2 (các tầng downsample sau quay lại `Conv` thường) — cắt GFLOPs từ 3.33x xuống
1.99x so với baseline, params thấp hơn cả baseline gốc. Đổi lại chưa có số liệu mAP để biết
ảnh hưởng độ chính xác.

**Bản eff** (`cfg/lwso-yolo11n-eff.yaml`, đề xuất hiện tại): thay `SPDConv` bằng
`SPDConvGroup` (grouped conv, giữ tính "không mất pixel" ở cả 5 tầng downsample thay vì
chỉ 1 tầng như bản lite) + thêm `ECA` (attention gần như free) ngay sau P2 — nơi model
trước đây chưa có attention nào. Nhẹ hơn cả bản lite (0.96M vs 1.12M, 12.4 vs 12.8 GFLOPs)
nhưng khác lite ở chỗ: có cơ sở lý thuyết + tiền lệ (LRDS-YOLO's LAD module — downsample
grouped tương tự cải thiện mAP50 +1.0 trong ablation của họ, không đánh đổi) để kỳ vọng
không mất — thậm chí tăng — mAP, chứ không chỉ đơn thuần cắt compute. Vẫn cần train thật
để xác nhận. Train cả 3 bản (`lwso-yolo11n`, `-lite`, `-eff`) song song để so sánh.
Bước nén tiếp theo vẫn còn ở M6 (prune + INT8, xem lộ trình).
