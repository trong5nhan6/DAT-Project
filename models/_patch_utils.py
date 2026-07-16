"""Shared helper for register_lwso()/register_fap(): safely build up parse_model's patched
source across multiple register_*() calls in the same process, in either order, without
either one clobbering the other's changes.

Two problems this solves:

1. `inspect.getsource(tasks.parse_model)` breaks the moment the function has been
   re-compiled by an earlier patch: `compile(src, tasks.__file__, "exec")` produces a code
   object whose line numbers don't match what's actually on disk at those lines, so
   `linecache`-based lookup for that *specific function object* returns garbage on the next
   call (verified: it silently returns the first ~2 lines of the file instead of raising).
   Asking for the *module's* source instead sidesteps this -- linecache keys purely off the
   file path, and register_*() never writes to the file on disk.

2. Naively re-deriving from the pristine module source on every register_*() call (the
   obvious fix for #1) reintroduces a different bug: whichever register_*() runs second
   would overwrite the first one's patch instead of building on it -- e.g. register_fap()
   after register_lwso() would silently drop SPDConv/C3k2Ghost/etc. from base_modules.
   So we cache the *current* (possibly already patched) source on the tasks module itself,
   and every register_*() call reads-patches-saves that shared string, composing correctly
   regardless of call order.
"""

import ast
import inspect
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def patch_ddp_registration() -> None:
    """Make ultralytics' multi-GPU (DDP, e.g. device="0,1") subprocess spawn also
    register our custom modules before it builds a model -- otherwise multi-GPU
    training crashes for any idea whose model_cfg references SPDConv/FreqMix/etc.

    Why: `device="0,1"` makes ultralytics' BaseTrainer.train() call
    generate_ddp_command() -> generate_ddp_file(), which writes a *brand-new* temp
    .py file executed by a fresh `torch.distributed.run` subprocess per GPU. That
    file only imports `ultralytics.models.yolo.detect.train.DetectionTrainer` and
    builds a trainer from `overrides["model"]` (a path string like
    "cfg/lwso-yolo11n.yaml") -- it never imports this project's code, so
    register_lwso()/register_fap()'s runtime monkeypatch of
    ultralytics.nn.tasks.parse_model (needed to recognize our custom module names)
    never happens in that subprocess, and parse_model() raises the moment it hits
    an unrecognized module name.

    Fix, in the same "patch without forking ultralytics" spirit as register_lwso()/
    register_fap(): wrap generate_ddp_file so the generated temp file also runs our
    registration first. Always registers lwso/fap/star (idempotent, harmless for
    --idea baseline / any stock model_cfg) since we don't know here which idea is
    active. Idempotent at the process level too (safe to call from every
    BaseModel.train(), which is where this gets called from).
    """
    import ultralytics.utils.dist as dist_mod

    if getattr(dist_mod, "_lwso_ddp_patched", False):
        return

    original_generate_ddp_file = dist_mod.generate_ddp_file
    marker = 'if __name__ == "__main__":\n'
    injection = (
        "    import sys\n"
        f"    sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        "    from models.fap.register import register_fap\n"
        "    from models.lwso.register import register_lwso\n"
        "    from models.slim.register import register_slim\n"
        "    from models.star.register import register_star\n"
        "    register_lwso()\n"
        "    register_fap()\n"
        "    register_star()\n"
        "    register_slim()\n"
    )

    def generate_ddp_file_with_registration(trainer):
        path = original_generate_ddp_file(trainer)
        content = Path(path).read_text(encoding="utf-8")
        if marker not in content:
            raise RuntimeError(
                "patch_ddp_registration: generate_ddp_file template changed "
                "(marker not found) -- ultralytics version incompatible."
            )
        content = content.replace(marker, marker + injection, 1)
        Path(path).write_text(content, encoding="utf-8")
        return path

    dist_mod.generate_ddp_file = generate_ddp_file_with_registration
    dist_mod._lwso_ddp_patched = True


def get_pristine_function_source(module, func_name: str) -> str:
    module_src = inspect.getsource(module)
    tree = ast.parse(module_src)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return ast.get_source_segment(module_src, node)
    raise RuntimeError(f"{func_name!r} not found in {module.__file__}")


def get_current_parse_model_source(tasks_module) -> str:
    """Current (possibly already patched by another register_*() call) source of
    parse_model -- pristine on first call in this process, whatever the last
    save_parse_model_source() stored after that.
    """
    cached = getattr(tasks_module, "_patched_parse_model_src", None)
    return cached if cached is not None else get_pristine_function_source(tasks_module, "parse_model")


def save_parse_model_source(tasks_module, src: str) -> None:
    tasks_module._patched_parse_model_src = src
