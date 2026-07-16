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
    model.py               FAPModel: gọi register_fap() rồi build YOLO (không NWD);
                          nếu cfg.sparsity_mask được set thì áp + giữ mask khi fine-tune
    modules.py              FreqMix (Haar band decompose + learnable softmax mixing)
    register.py              patch parse_model của ultralytics (pin 8.3.x)
    _sparsity.py             re-apply mask sau mỗi optimizer step (train.py --sparsity-mask)
  star/                   idea "star": StarNet backbone + GSConv slim-neck + Wise-IoU v3
    model.py               StarModel: gọi register_star() + patch_wiou_loss() rồi build YOLO
    modules.py              StarBlock/C3k2Star (backbone), SimAM (attention, 0 param),
                          GSConv/GSBottleneck/VoVGSCSP (neck)
    register.py              register_star() = register_lwso() + patch thêm 3 class trên
    losses.py                 Wise-IoU v3 (thay BboxLoss), luôn bật (không toggle)
  slim/                   idea "slim": khung lwso-eff + LSCDetect head chia sẻ conv+GroupNorm
    model.py               SlimModel: register_slim() + NWD (cùng recipe lwso-eff)
    modules.py              ConvGN, Scale, LSCDetect (head dùng chung conv giữa 3 scale)
    register.py              register_lwso() + patch parse_model nhận diện head LSCDetect
  pd/                     idea "pd": train student bất kỳ + CWD distillation từ teacher
    model.py               PDModel: student = .yaml (vd slim) hoặc .pt đã prune (PDTrainer
                          bypass yaml-rebuild); guard resume/multi-GPU/compile
    distill.py              cwd_loss + callback hook (adapter 1x1 khi lệch kênh tap;
                          checkpoint sạch — hook đăng ký sau khi EMA đã deepcopy)
  _patch_utils.py         helper dùng chung cho lwso/fap/star register.py — patch
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
  star-yolo11n.yaml     StarBlock+GSConv+SimAM trên khung P2/P3/P4 của lwso-yolo11n-eff.yaml
  slim-yolo11n.yaml     lwso-eff + LSCDetect + P2 neck 48ch: 0.56M, 6.31 GFLOPs@640 (< baseline)
configs/                hyperparameter theo idea (train.py load, xem "Cấu hình train" bên dưới)
  base.yaml             default cho mọi run — mọi idea kế thừa file này
  baseline.yaml         idea "baseline": YOLO11n gốc, không NWD
  lwso.yaml             idea "lwso": full LWSO model + NWD
  fap.yaml              idea "fap": cfg/fap-yolo11n.yaml + protected_layers cho prune.py
  star.yaml             idea "star": cfg/star-yolo11n.yaml + Wise-IoU v3 (không NWD)
data/
  visdrone.yaml        dataset config (sửa `path:` nếu chuyển máy)
  scripts/convert_visdrone.py     VisDrone → YOLO format
train.py / val.py      entry point train & eval
prune.py               semantic-path-aware LAMP pruning cho idea "fap" — prune + eval + lưu
                          sparsity mask, xem mục "Idea fap" bên dưới
prune_structured.py    [EXPERIMENTAL] structured channel pruning (torch-pruning DepGraph) —
                          hiện bị chặn bởi bug upstream của torch-pruning (xem docstring);
                          route được hỗ trợ cho mục tiêu efficiency là idea "pd" bên dưới
tests/                 pytest shape/param sanity tests
logs/                   text log mỗi run (<name>.log, xem "Ghi chú quan trọng") — gitignored
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

# 6) Idea star (StarBlock+GSConv+SimAM+Wise-IoU v3) — xem mục "Idea star" bên dưới:
python train.py --idea star

# 7) Idea slim (LSCDetect shared GN head, 0.56M/6.31 GFLOPs@640) — mục "Idea slim":
python train.py --idea slim

# 8) Idea pd (distill slim từ teacher lwso-eff đã train) — mục "Idea pd":
python train.py --idea pd        # cần runs/detect/lwso-n-eff/weights/best.pt có sẵn

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

Hiện có 4 idea: `baseline` (YOLO11n gốc), `lwso` (full LWSO model), `fap` (FreqMix + P2),
`star` (StarBlock + GSConv slim-neck + Wise-IoU v3) — 2 idea sau xem mục riêng bên dưới.
Thêm 1 idea, 3 bước, **không cần sửa `train.py`**:

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

`prune.py` tự làm cả 3 bước của pipeline "train dense → prune → fine-tune ngắn" (trừ bước
train dense, vẫn là `train.py` riêng):

```bash
python train.py --idea fap                                              # 1. train FreqMix dense

python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3              # M6: LAMP+P2/P3 (đề xuất)
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-protect # M5: LAMP thường (ablation)
python prune.py --weights runs/detect/fap-n/weights/best.pt --sparsity 0.3 --no-lamp --no-protect  # M4: magnitude đều (ablation)
```

Mỗi lần chạy, `prune.py` sẽ:
1. Zero weight theo LAMP score + exception rule, lưu checkpoint `<weights>.pruned.pt`.
2. Lưu sparsity mask `<weights>.pruned.mask.pt` (`{param_name: bool_tensor}`, đúng tên tham số
   theo `model.named_parameters()`).
3. **Tự eval cả trước và sau khi prune** trên `--data`/`--eval-split` (mặc định `val`), in bảng
   mAP50/mAP50-95 so sánh ngay — biết ngay chi phí accuracy của lần prune này, không cần đoán.
   Dùng `--no-eval` nếu chỉ muốn prune nhanh, bỏ qua eval.

Fine-tune tiếp theo (giữ nguyên sparsity, không cho gradient kéo weight đã prune trôi khỏi 0)
truyền thêm `--sparsity-mask` cho `train.py`:

```bash
python train.py --idea fap --weights runs/detect/fap-n/weights/best.pruned.pt \
    --sparsity-mask runs/detect/fap-n/weights/best.pruned.mask.pt \
    --epochs 10 --name fap-n-finetuned
```

Cơ chế: `FAPModel` (`models/fap/model.py` + `models/fap/_sparsity.py`) áp mask 1 lần lúc load
checkpoint (an toàn trước lệch làm tròn fp16/AMP), rồi đăng ký callback `on_train_batch_end` —
zero lại đúng các vị trí đã prune trên **cả** `trainer.model` **và** `trainer.ema.ema` sau mỗi
optimizer step. Bắt buộc phải zero cả EMA: ultralytics gọi `ema.update(model)` bên trong
`optimizer_step()`, **trước** khi `on_train_batch_end` chạy — best.pt/last.pt lưu từ EMA, không
phải từ `trainer.model`, nên nếu chỉ zero `trainer.model` thì EMA vẫn hấp thụ 1 lượng gradient
drift rất nhỏ (~1e-5) mỗi step, và số này **không** tự triệt tiêu qua nhiều step (đã verify thật:
bỏ sót bước zero-EMA làm gần như toàn bộ 473K vị trí đã prune trôi khỏi 0 sau 2 epoch; zero cả
hai thì 100% giữ đúng 0.0 tuyệt đối). Không gate theo `RANK` vì DDP chỉ all-reduce gradient chứ
không đồng bộ weight — mọi rank phải tự áp mask giống hệt nhau mỗi step, nếu không các bản sao sẽ
lệch dần. Đã verify toàn bộ pipeline (prune → eval → fine-tune → kiểm tra mask giữ đúng 0.0) bằng
chạy thật CPU, tập con 6 ảnh; **chưa có số liệu mAP thật** — cần train + prune + fine-tune trên
GPU (Kaggle), tập dữ liệu đầy đủ, theo ladder M1-M6 của paper để so sánh.

## Idea `star` — StarNet backbone + GSConv slim-neck + Wise-IoU v3

Khác `lwso`/`fap` (mỗi cái port từ 1 nguồn cụ thể — LWSO tự thiết kế, FAP từ 1 paper) — `star`
tổng hợp 4 kỹ thuật riêng biệt từ literature 2021-2025 (research trước khi code, xem lịch sử
trao đổi), mỗi kỹ thuật nhắm 1 phần khác nhau của kiến trúc `lwso-yolo11n-eff.yaml`:

1. **`StarBlock`/`C3k2Star`** (`models/star/modules.py`, backbone) — thay `C3k2Ghost`. StarNet
   (Ma et al., arXiv:2403.19967): nhân element-wise 2 nhánh 1x1-conv, ánh xạ input lên không
   gian đặc trưng phi tuyến chiều cao **ngầm** (giống kernel trick) mà không cần mở kênh thật —
   rẻ hơn nhiều so với đạt cùng năng lực biểu diễn bằng cách mở rộng `GhostBottleneck` literal.
   Conv trộn không gian bên trong là `RepConv` (reparam có sẵn của ultralytics — train 3x3+1x1+
   identity, fuse về 1 conv 3x3 lúc inference qua `.fuse_convs()`, `model.fuse()` tự gọi) chạy
   depthwise — bù lại năng lực bị mất do cắt kênh, miễn phí lúc inference.
2. **`SimAM`** (`models/star/modules.py`, attention tại P2) — thay `ECA`. Yang et al. (ICML
   2021): attention 3D dựa hàm năng lượng thần kinh học, **0 `nn.Parameter`** — rẻ hơn cả `ECA`
   (vốn đã gần free, có 1 `Conv1d` nhỏ).
3. **`GSConv`/`VoVGSCSP`** (`models/star/modules.py`, neck tại P3/P4) — thay `C3k2Ghost` sau mỗi
   `BiFPNCat` (bản thân `BiFPNCat` giữ nguyên làm phép concat, chỉ đổi khối xử lý sau đó). Li et
   al. (arXiv:2206.02424): conv thường trên nửa kênh + depthwise trên nửa còn lại + channel
   shuffle — thiết kế riêng cho neck (feature đã concat, giàu thông tin), không phải backbone.
   Chỉ dùng ở P3/P4 — ở P2 dùng `C3k2` gốc (xem đoạn GFLOPs bên dưới).
4. **Wise-IoU v3** (`models/star/losses.py`, thay `BboxLoss`) — Tong et al. (arXiv:2301.10051):
   focusing **động, non-monotonic** dựa "outlier degree" β = L_IoU / mean(L_IoU chạy theo EMA)
   — hạ trọng số cả anchor quá dễ lẫn anchor giống outlier/nhãn lỗi, tăng trọng số anchor khó-
   nhưng-hợp-lý; khác CIoU/NWD (trọng số tĩnh, đơn điệu). Luôn bật cho idea này, không có toggle
   kiểu `use_nwd` của `lwso` (2 cơ chế loss khác nhau, không blend).

```bash
python train.py --idea star
```

Kiến trúc giữ khung P2/P3/P4 (bỏ P5). Đo thật (v3, sau khi tối ưu GFLOPs — xem dưới):
**0.647M params, 7.60 GFLOPs@640** — nhẹ hơn cả `fap-yolo11n.yaml` (2.13M / 8.96 GFLOPs, dù
`fap` có 4 đầu detect P2-P5 còn `star` chỉ 3), nhẹ hơn nhiều `lwso-yolo11n.yaml` (2.40M / 21.5
GFLOPs) và `lwso-yolo11n-eff.yaml` (0.96M / 12.41 GFLOPs). **Chưa có số liệu mAP thật** — kênh
neck bị thu mỏng đáng kể so với v1 (xem dưới), cần train thật để xác nhận không đổi accuracy
lấy compute.

**Hành trình tối ưu GFLOPs** (mỗi bước đều đo thật, không đoán — bắt đầu từ yêu cầu "GFLOPs
phải thấp hơn baseline YOLOv11n gốc ~6.5, nhưng vẫn giữ đầu P2"):
- **v1** (`SPDConvGroup` downsample, `StarBlock`/`GSConv` ở mọi nơi kể cả P2, neck full-width):
  1.08M params, **13.67 GFLOPs@640** — nặng hơn cả `fap` dù `fap` có tới 4 đầu detect.
- **v2** (đổi downsample sang `FreqMix`, `StarBlock`/`GSConv` chỉ ở P3/P4, P2 dùng `C3k2` gốc):
  0.876M, **12.16 GFLOPs** — đo tách biến xác nhận `C3k2Star`/`VoVGSCSP` **không** phải thủ phạm
  (thực ra nhẹ hơn `C3k2` gốc, xem `tests/`), `DySample`/`BiFPNCat`/`SimAM` gần như free, và
  thêm lại đầu P5 (để đặt `SPPF`/`C2PSA` ở độ phân giải rẻ hơn) còn làm nặng hơn (13.26
  GFLOPs) vì phần neck top-down/bottom-up thêm cho P5 tốn hơn phần tiết kiệm được.
- **v3** (thu mỏng kênh neck ở P2/P3 theo đúng tỉ lệ `fap-yolo11n.yaml`, vd P2 out 64→32,
  P3 top-down 128→64 thay vì giữ nguyên độ rộng backbone): **7.60 GFLOPs** — hoá ra đây mới là
  yếu tố quyết định, không phải module nào được dùng. Bài học: ở độ phân giải P2/P3 (tốn 4x/16x
  hơn P4 mỗi phép toán), **độ rộng kênh của neck quan trọng hơn nhiều so với việc chọn module
  "nhẹ" gì** — `fap` rẻ chủ yếu vì giữ nguyên tỉ lệ kênh neck đã được ultralytics tune sẵn, không
  phải vì FreqMix đặc biệt rẻ.

**Bug đã bắt và sửa lúc build** (đáng ghi lại vì liên quan trực tiếp small object — đúng use-case
của cả project): `bbox_iou()` của ultralytics trả về shape `(N, 1)` (giữ `keepdim`), nhưng các
số hạng tự viết trong `wiou()` (khoảng cách tâm, enclosing box...) tính qua `pred[..., 0]` nên
có shape `(N,)` — nhân `r * r_wiou * l_iou` giữa 2 shape lệch nhau broadcast thành `(N, N)` thay
vì `(N,)`, khiến `box_loss` bị thổi phồng ~N lần (bắt được qua smoke train thật: box_loss ~7000-
10000 thay vì ~5 bình thường). Đã sửa bằng `.squeeze(-1)` ngay sau `bbox_iou()`; giữ thêm 1 chặn
phòng thủ `r_wiou.clamp(max=3.0)` vì enclosing box gần suy biến (target nhỏ vài pixel, khá phổ
biến trên VisDrone) vẫn có thể làm `exp()` tràn số dù đã hết bug shape. Có test regression riêng
(`tests/test_modules.py::test_wiou_stays_finite_for_near_degenerate_small_boxes`).

## Idea `slim` — LSCDetect shared head + neck P2 48ch (đạt budget baseline, giữ P2)

Trả lời trực tiếp mục tiêu "params & GFLOPs < baseline (2.58M / 6.5 GFLOPs@640), mAP >
baseline" mà 4 run đầu chưa config nào đạt cả 3 (xem `results/result.txt`): lwso-eff thắng
mAP (+2.2) nhưng 12.4 GFLOPs; star đạt compute nhưng mất mAP vì neck 32ch quá mỏng.

Khung `lwso-yolo11n-eff.yaml` (tổ hợp đã thắng) + đúng 2 thay đổi:

1. **`LSCDetect`** (`models/slim/modules.py`) thay stock `Detect`: mỗi scale căn về 48ch
   qua 1x1 ConvGN, rồi **dùng chung** 1 stack (3x3 dense + 3x3 depthwise, GroupNorm) + 1
   conv box + 1 conv cls cho cả 3 scale; mỗi scale chỉ thêm 1 hệ số `Scale` học được ở
   nhánh box. Stock head là ~nửa params của model sub-1M có P2 và trả FLOPs ở độ phân
   giải P2; GroupNorm còn hợp batch=8 (BN nhiễu ở batch nhỏ — FCOS/LSCD papers đều báo
   GN head ngang hoặc hơn BN). Đo tách biến: 2 conv dense chia sẻ ở mọi scale = 5.57
   GFLOPs (53% model) → bản cuối dense+depthwise, hidc=48.
2. **Neck thu vừa phải**: P2 out 64→48, P3 top-down/bottom-up 128→96 — điểm giữa có chủ
   đích giữa 32ch (star, mất mAP toàn bộ class) và 64/128ch (lwso-eff, 12.4G).

Đo thật (build+forward): **0.560M params, 6.31 GFLOPs@640 / 14.2@960** (fused 6.2) — dưới
baseline cả 2 trục, vẫn giữ P2. Loss giữ nguyên NWD blend như lwso-eff. **Chưa có mAP thật**
— cần train (một mình nó, và/hoặc kèm distill qua idea `pd` bên dưới).

```bash
python train.py --idea slim
```

## Idea `pd` — CWD distillation (train student bất kỳ kèm teacher đã train)

Stage "distill back" của kế hoạch nén: **CWD** (Channel-wise Distillation, Shu et al.,
ICCV 2021) từ teacher mạnh (mặc định: lwso-eff best.pt, mAP50 31.8) sang student nhẹ.
Cắm hoàn toàn qua callback + forward hook (`models/pd/distill.py`) — không fork
ultralytics, checkpoint sạch (hook đăng ký SAU khi EMA đã deepcopy; best.pt lưu từ EMA).

2 chế độ student theo đuôi file `model_cfg`:

- **`.yaml`** (mặc định — combo chủ lực cho mục tiêu efficiency): train
  `cfg/slim-yolo11n.yaml` từ scratch + distill. Lệch kênh tap P2/P3 (48/96 vs teacher
  64/128) được hấp thụ bằng adapter conv 1x1 tự tạo — adapter nhận gradient (add vào
  optimizer + nối dài scheduler đúng chuẩn torch>=2.12 zip strict) nhưng nằm ngoài model,
  không đi vào checkpoint.
- **`.pt`**: fine-tune checkpoint đã prune vật lý (từ `prune_structured.py` — hiện
  EXPERIMENTAL, xem docstring của file đó) với self-distill từ bản dense; kênh tap khớp
  sẵn nên không cần adapter. `PDTrainer.get_model` trả thẳng module đã unpickle — né
  đường rebuild-từ-yaml của ultralytics vốn sẽ phá kiến trúc đã prune.

```bash
# flagship: slim + distill từ lwso-eff (sửa distill_teacher trong configs/pd.yaml nếu
# run name khác lwso-n-eff):
python train.py --idea pd --name slim-distill
# ablation không distill: đặt distill_teacher: null (hoặc dùng --idea slim)
```

Đã verify bằng chạy thật CPU (dataset 4 ảnh, 2 epochs, cả 2 chế độ student): distill loss
log mỗi epoch, adapter tạo đúng tap lệch kênh, best.pt load lại sạch (0 hook, đúng 0.56M
params). Giới hạn (guard chặn hẳn, không hỏng ngầm): chưa hỗ trợ `--resume` (student .pt),
multi-GPU DDP, `compile`. **Chưa có mAP thật** — cần train GPU đầy đủ.

Kỳ vọng (từ literature, cần xác nhận): CWD trên VisDrone cho student nano +1.5–2.5 mAP50
(arXiv:2509.12918 còn báo student vượt teacher); slim một mình cần ≥29.7 để thắng
baseline, kèm distill nhắm vùng 31–32.5.

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
- **Log khi train**: mặc định chỉ in ra console (stdout), không có file log riêng — muốn giữ
  lại thì tự redirect (`python train.py ... | tee console.log`). **Ngoại lệ**: nếu bật
  `--test-every N` (mặc định 20), tự tạo `logs/<name>.log` ở thư mục gốc project (tên file =
  `cfg.name`, thư mục `logs/` tự tạo nếu chưa có, đã thêm vào `.gitignore`) — ghi **mỗi
  epoch** 1 dòng (train loss + val P/R/mAP50/mAP50-95 + lr, chỉ ghi file, không in console vì
  ultralytics đã tự in bảng epoch riêng), xen giữa là khối **EFFICIENCY METRICS** (params/
  GFLOPs/model size/latency/FPS, đo forward pass thật) sau **mỗi lần trigger** `--test-every`.
  Song song đó, `runs/detect/<name>/test_metrics.csv` (`epoch,mAP50,mAP50-95,params_m,gflops,
  model_size_mb,latency_ms,fps`) cũng được ghi mỗi lần trigger. Cả 2 chỉ xuất hiện sau lần
  trigger/epoch đầu tiên, không phải ngay lúc train bắt đầu; nếu `--test-every 0` thì không
  có file log/CSV nào cả.
- **Seed**: `configs/base.yaml` set `seed: 42` (`deterministic: True` là default có sẵn của
  ultralytics, project không đổi). Override qua `--seed <n>` nếu cần chạy nhiều seed khác
  nhau để lấy mean±std.
- **Multi-GPU (`--device 0,1`)**: đã hỗ trợ cho mọi idea (kể cả `lwso`/`fap`/`star` có module
  custom). Lý do cần patch riêng: ultralytics chạy multi-GPU bằng cách spawn **process con hoàn
  toàn mới** (`torch.distributed.run`) từ 1 file `.py` tạm tự sinh ra — process đó không import
  gì từ project này nên `register_lwso()`/`register_fap()`/`register_star()` (patch `parse_model`
  để nhận diện `SPDConv`/`FreqMix`/`StarBlock`...) không tự chạy trong đó, sẽ crash ngay khi build
  model. `models/_patch_utils.py:patch_ddp_registration()` vá thêm bước đăng ký vào chính file
  tạm đó (được `BaseModel.train()` tự gọi, không cần làm gì thêm) — cùng tinh thần "không fork
  ultralytics" như `register_lwso()`. Đã verify multi-GPU DDP thật trên Kaggle T4x2 cho `lwso`
  (log train thành công, nhận diện đúng module custom) — `fap`/`star` mới verify build trong
  process con mô phỏng, chưa chạy DDP thật trên GPU.

## Lộ trình (đã bàn ở thiết kế)

- [x] M1 script convert + baseline config
- [x] M2 ablation +P2/−P5 (YAML)
- [x] M3 module custom + đăng ký runtime + unit tests
- [x] M4 NWD blend loss
- [ ] M5 chạy ablation đầy đủ, gom kết quả (cần GPU)
- [~] M6 nén: distill CWD đã xong (idea `pd`, teacher mặc định lwso-eff — đổi sang
      `lwso-yolo11s` qua distill_teacher nếu muốn); structured prune bị chặn bởi bug
      upstream torch-pruning (xem prune_structured.py); còn lại export ONNX/TensorRT INT8
- [ ] M7 train idea `slim` (một mình + kèm `pd` distill) trên GPU, điền mAP vào bảng

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
| STAR-YOLO11n | 960 | | | **0.65M** | 7.6 | 17.1 |
| SLIM-YOLO11n | 960 | | | **0.56M** | **6.3** | 14.2 |

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
