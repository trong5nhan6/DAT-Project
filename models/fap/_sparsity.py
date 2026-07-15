"""Re-applies a prune.py sparsity mask during --idea fap fine-tuning, so pruned
weights can't drift away from 0 as gradients update the rest of the network.
"""

from __future__ import annotations


def build_sparsity_mask_callbacks(mask_path: str):
    """Loads the {param_name: bool_tensor} mask saved by prune.py and returns:
      - apply(model): zero every masked position in `model`'s matching params (call
        once right after loading a pruned checkpoint, in case fp16/AMP round-tripping
        nudged a "zero" weight a tiny epsilon off exact 0.0).
      - on_train_batch_end(trainer): re-applies the same mask after every optimizer
        step, so gradients can't let pruned weights drift back to nonzero during
        fine-tuning. Registered on on_train_batch_end (not on_fit_epoch_end) because
        that's the finest-grained hook ultralytics offers after an actual optimizer
        step (engine/trainer.py: optimizer_step() then run_callbacks("on_train_batch_end"),
        a harmless no-op on accumulation-only batches where no step was taken).

    Also re-applies the mask to trainer.ema.ema, not just trainer.model: ultralytics'
    optimizer_step() calls self.ema.update(self.model) *before* on_train_batch_end fires
    (trainer.py L682-690 vs L465), so the EMA shadow copy -- what best.pt/last.pt actually
    save -- absorbs one step's worth of gradient drift on every masked position before we
    get a chance to re-zero trainer.model. Left unfixed, that per-step residual doesn't
    cancel out over many steps (verified empirically: after a 2-epoch fine-tune, ~1e-5
    residual on effectively every masked weight). Re-zeroing both copies every step is
    required to actually hold the mask in what gets saved to disk.

    Unconditional across ranks (no RANK gate): in DDP every replica must apply the
    identical mask every step, or replicas silently diverge since only gradients (not
    weights) are all-reduced.
    """
    import torch

    state = {"mask": None, "device_cache": {}}

    def _unwrap(model):
        return model.module if hasattr(model, "module") else model

    def _mask_for_device(device):
        if device not in state["device_cache"]:
            state["device_cache"][device] = {
                name: m.to(device) for name, m in state["mask"].items()
            }
        return state["device_cache"][device]

    def apply(model):
        if state["mask"] is None:
            state["mask"] = torch.load(mask_path, map_location="cpu")
        model = _unwrap(model)
        named = dict(model.named_parameters())
        device = next(model.parameters()).device
        mask_dev = _mask_for_device(device)
        with torch.no_grad():
            for name, m in mask_dev.items():
                p = named.get(name)
                if p is not None:
                    p[m] = 0.0

    def on_train_batch_end(trainer):
        apply(trainer.model)
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            apply(ema.ema)

    return apply, on_train_batch_end
