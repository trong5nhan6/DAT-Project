"""Abstract base class for every idea (baseline, lwso, ...).

Each idea subclasses BaseModel and implements build(), which must set self._yolo
to an ultralytics.YOLO instance. train.py stays idea-agnostic: it goes through
build_model() (models/__init__.py) + BaseModel.train(), never branching on
`idea` itself. To add a new idea, add one file here — see baseline.py/lwso.py.

Kept import-light at module level (no torch/ultralytics/lwso here) so that
`from models import MODEL_REGISTRY` stays cheap — train.py needs it just to
build --idea's argparse choices, before any heavy import happens.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable


class BaseModel(ABC):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self._yolo = None
        self.build()

    @abstractmethod
    def build(self) -> None:
        """Set self._yolo (an ultralytics.YOLO instance) for this idea."""
        ...

    def _build_yolo(self):
        """Shared cfg.model_cfg (+ optional cfg.weights warm-start) loading.

        Call this from build() after any idea-specific setup that must happen
        before YOLO(...) is constructed (e.g. register_lwso() for a model .yaml
        that references custom modules).
        """
        from ultralytics import YOLO

        yolo = YOLO(str(self.cfg.model_cfg))
        if self.cfg.get("weights"):
            yolo.load(str(self.cfg.weights))
        return yolo

    def get_callbacks(self) -> dict[str, Callable]:
        """Extra ultralytics callbacks to register before train().

        Default: test-set monitoring every cfg.test_every epochs (0 disables),
        shared by every idea since it doesn't depend on architecture/loss. To
        add idea-specific callbacks in a subclass, merge with this dict rather
        than replacing it (unless you mean to drop test-set monitoring):
            def get_callbacks(self):
                return {**super().get_callbacks(), "on_train_epoch_end": my_cb}
        """
        test_every = int(self.cfg.get("test_every", 0) or 0)
        if test_every <= 0:
            return {}
        print(f"[lwso] test-set eval every {test_every} epochs enabled "
              f"-> logs/{self.cfg.name}.log")
        return {
            "on_fit_epoch_end": _build_test_eval_callback(
                str(self.cfg.data), int(self.cfg.imgsz), int(self.cfg.batch),
                test_every, str(self.cfg.name),
            )
        }

    def train(self, **train_kwargs) -> None:
        assert self._yolo is not None, "build() must run before train()"
        from models._patch_utils import patch_ddp_registration

        patch_ddp_registration()  # no-op unless device has >1 GPU (e.g. "0,1")
        for event, cb in self.get_callbacks().items():
            self._yolo.add_callback(event, cb)
        self._yolo.train(**train_kwargs)

    @property
    def yolo(self):
        return self._yolo


def _compute_efficiency_metrics(model, imgsz: int, device, warmup: int = 10, runs: int = 30) -> dict:
    """Params/GFLOPs/theoretical size + batch=1 latency/FPS for `model` on `device`.

    `model_size_mb` is params*4 bytes (fp32), not a measured file size -- this callback
    never writes model to disk, so there's no checkpoint to stat(). Latency/FPS come from
    real forward passes (warmup then timed loop), matching how yolov12n-visdrone's
    utils.metrics.compute_efficiency_metrics measures it, for cross-project comparability.
    """
    import time

    import torch
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    dev = device if isinstance(device, torch.device) else torch.device(device)
    params = get_num_params(model)
    gflops = get_flops(model, imgsz)

    dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev)
    model.eval()
    with torch.no_grad():
        for _ in range(warmup):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(runs):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

    latency_ms = (t1 - t0) / runs * 1000
    return {
        "params_m": params / 1e6,
        "gflops": gflops,
        "model_size_mb": params * 4 / 1e6,  # fp32 theoretical
        "latency_ms": latency_ms,
        "fps": 1000 / latency_ms if latency_ms > 0 else float("inf"),
        "device": str(dev),
    }


def _format_efficiency_report(eff: dict) -> list[str]:
    w = 58
    row = lambda label, value: f"  {label:<32} {value}"  # noqa: E731
    return [
        "-" * w,
        "  EFFICIENCY METRICS",
        "-" * w,
        row("Parameters", f"{eff['params_m']:.3f} M"),
        row("GFLOPs", f"{eff['gflops']:.2f} G"),
        row("Model size (fp32, theoretical)", f"{eff['model_size_mb']:.2f} MB"),
        row("Latency (batch=1)", f"{eff['latency_ms']:.2f} ms/img"),
        row("FPS (batch=1)", f"{eff['fps']:.1f} fps"),
        row("Device", eff["device"]),
    ]


def _format_epoch_line(trainer, epoch: int) -> str:
    """One-line per-epoch trace: train loss + val metrics (already computed by
    ultralytics' own per-epoch validate() by the time on_fit_epoch_end fires) + lr.
    File-only (not printed) -- ultralytics already prints an equivalent progress
    table to console every epoch, so this would just duplicate it there.
    """
    tloss = trainer.tloss
    names = getattr(trainer, "loss_names", ("box_loss", "cls_loss", "dfl_loss"))
    if tloss is not None:
        vals = tloss.tolist() if hasattr(tloss, "tolist") else list(tloss)
        loss_str = "  ".join(f"{n}={v:.4f}" for n, v in zip(names, vals))
    else:
        loss_str = "loss=N/A"

    metrics = trainer.metrics or {}

    def _fmt(v):
        return f"{v:.4f}" if v is not None else " N/A"

    metric_str = (
        f"P={_fmt(metrics.get('metrics/precision(B)'))} "
        f"R={_fmt(metrics.get('metrics/recall(B)'))} "
        f"mAP50={_fmt(metrics.get('metrics/mAP50(B)'))} "
        f"mAP50-95={_fmt(metrics.get('metrics/mAP50-95(B)'))}"
    )
    lr_vals = list((trainer.lr or {}).values())
    lr_str = f"lr={lr_vals[0]:.6f}" if lr_vals else "lr=N/A"

    total = getattr(trainer, "epochs", "?")
    return f"[EPOCH {epoch:>4}/{total}]  {loss_str}  |  {metric_str}  |  {lr_str}"


def _build_test_eval_callback(data: str, imgsz: int, batch: int, every: int, log_name: str):
    """Writes <repo_root>/logs/<log_name>.log with a per-epoch train/val trace (every
    epoch) plus, every `every` epochs, a test-dev split mAP + efficiency metrics
    (params/GFLOPs/latency/FPS) block -- printed to console too. Also appends the
    periodic block to <save_dir>/test_metrics.csv.

    Uses a standalone DetectionValidator (model=... path, not trainer=...) so it never
    touches trainer.validator / trainer.stopper / best.pt selection, which stay driven by
    the ordinary val split. Test mAP here is for monitoring only, not for model selection
    (test-dev has no official public labels; treat any local labels as unofficial).

    Validates a deepcopy of the EMA weights, not the live trainer.ema.ema reference:
    AutoBackend fuses whatever nn.Module it's handed (in place, `model.fuse()`), which
    permanently changes its state_dict keys. Handing it the live EMA model corrupts it
    and crashes the next `ema.update(self.model)` call with a KeyError. The deepcopy is
    reused for the efficiency benchmark too (fusion there is harmless/even representative
    of real deployment latency) since it's discarded at the end of this callback either way.
    """
    import copy
    from pathlib import Path

    from ultralytics.models.yolo.detect import DetectionValidator
    from ultralytics.utils import RANK

    state = {"validator": None, "csv_path": None, "log_path": None, "last_epoch": None}

    def _log(msg: str) -> None:
        print(msg)
        with open(state["log_path"], "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    def _on_fit_epoch_end(trainer):
        if RANK not in (-1, 0):
            return
        epoch = trainer.epoch + 1  # 1-indexed, matches the printed epoch column
        # trainer.final_eval() re-fires on_fit_epoch_end once more at the same epoch
        # after training ends; skip the repeat instead of double-logging/re-validating.
        if epoch == state["last_epoch"]:
            return
        state["last_epoch"] = epoch

        if state["log_path"] is None:
            logs_dir = Path("logs")
            logs_dir.mkdir(parents=True, exist_ok=True)
            state["log_path"] = logs_dir / f"{log_name}.log"
            print(f"[lwso] text log -> {state['log_path']}")

        with open(state["log_path"], "a", encoding="utf-8") as f:
            f.write(_format_epoch_line(trainer, epoch) + "\n")

        if epoch % every != 0:
            return

        if state["validator"] is None:
            state["validator"] = DetectionValidator(
                args=dict(
                    data=data,
                    split="test",
                    imgsz=imgsz,
                    batch=batch,
                    plots=False,
                    save_json=False,
                    device=trainer.device,
                ),
                save_dir=trainer.save_dir / "test_eval",
            )
            state["csv_path"] = trainer.save_dir / "test_metrics.csv"
            if not state["csv_path"].exists():
                state["csv_path"].write_text(
                    "epoch,mAP50,mAP50-95,params_m,gflops,model_size_mb,latency_ms,fps\n"
                )
            print(f"[lwso] test-set eval log -> {state['csv_path']}")

        _log(f"\n[lwso] test-set eval @ epoch {epoch}")
        model = copy.deepcopy(trainer.ema.ema or trainer.model)
        stats = state["validator"](model=model)
        map50 = stats.get("metrics/mAP50(B)", float("nan"))
        map5095 = stats.get("metrics/mAP50-95(B)", float("nan"))

        eff = _compute_efficiency_metrics(model, imgsz, trainer.device)
        for line in _format_efficiency_report(eff):
            _log(line)

        with open(state["csv_path"], "a") as f:
            f.write(
                f"{epoch},{map50:.5f},{map5095:.5f},{eff['params_m']:.5f},{eff['gflops']:.5f},"
                f"{eff['model_size_mb']:.5f},{eff['latency_ms']:.5f},{eff['fps']:.5f}\n"
            )

    return _on_fit_epoch_end
