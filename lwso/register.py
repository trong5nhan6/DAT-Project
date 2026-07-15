"""Register LWSO custom modules into ultralytics without forking the repo.

ultralytics' parse_model() resolves module names via its own module globals and
infers channels only for classes listed in two frozensets defined *inside* the
function body. We therefore:

  1. setattr() our classes onto ultralytics.nn.tasks so name lookup succeeds, and
  2. rewrite parse_model's source (4 targeted, verified substitutions) and exec it
     back into the tasks namespace so channel inference covers our modules.

Tested against ultralytics 8.3.x (pinned in requirements.txt). Every substitution
must match exactly once, otherwise a RuntimeError explains which one failed —
this is the canary for an incompatible ultralytics version.
"""

import inspect
import re

import ultralytics.nn.tasks as tasks

from .modules import BiFPNCat, C3k2Ghost, DySample, EMA, SPDConv

_CUSTOM_CLASSES = (SPDConv, C3k2Ghost, EMA, DySample, BiFPNCat)
_registered = False


def register_lwso() -> None:
    """Idempotent. Call once before building any model from an LWSO YAML."""
    global _registered
    if _registered:
        return

    for cls in _CUSTOM_CLASSES:
        setattr(tasks, cls.__name__, cls)

    src = inspect.getsource(tasks.parse_model)
    substitutions = [
        # channel-inferring modules with (c1, c2, ...) signatures
        (
            r"base_modules = frozenset\(\s*\{",
            "base_modules = frozenset(\n        {\n"
            "            SPDConv,\n            C3k2Ghost,\n            EMA,\n            DySample,",
        ),
        # modules whose 3rd arg is the repeat count n
        (
            r"repeat_modules = frozenset\([^\{]*\{",
            "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n"
            "            C3k2Ghost,",
        ),
        # BiFPNCat concatenates like Concat: c2 = sum of input channels
        (
            r"elif m is Concat:",
            "elif m is Concat or m is BiFPNCat:",
        ),
        # C3k2Ghost behaves like C3k2 (non-legacy Detect head, c3k flag on m/l/x)
        (
            r"if m is C3k2:",
            "if m is C3k2 or m is C3k2Ghost:",
        ),
    ]
    for pattern, replacement in substitutions:
        src, count = re.subn(pattern, replacement, src, count=1)
        if count != 1:
            raise RuntimeError(
                f"register_lwso: could not patch parse_model (pattern not found: {pattern!r}). "
                "Your ultralytics version is incompatible; install the range pinned in "
                "requirements.txt (ultralytics>=8.3,<8.4)."
            )

    code = compile(src, tasks.__file__, "exec")
    exec(code, tasks.__dict__)  # rebinds tasks.parse_model to the patched version
    _registered = True
