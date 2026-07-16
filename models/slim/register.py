"""Register idea "slim"'s LSCDetect head into ultralytics without forking the repo — same
source-patching trick as models/lwso/register.py (see that file for the full why).

Builds on register_lwso(): cfg/slim-yolo11n.yaml is the lwso-yolo11n-eff.yaml frame
(SPDConvGroup/C3k2Ghost/EMA/ECA/DySample/BiFPNCat all reused unchanged) with only two
changes — a thinner P2 neck output and the LSCDetect shared head — so the only NEW class
parse_model must learn about is LSCDetect.

Unlike the lwso/fap/star registrations (which extend base_modules/repeat_modules, i.e.
the *block* channel-inference sets), a Detect-family head is special-cased in two other
spots inside parse_model:

  1. the head frozenset — `elif m in frozenset({Detect, WorldDetect, ...})` — which makes
     parse_model append the per-scale input channel list to args, and
  2. the legacy-flag set — `if m in {Detect, YOLOEDetect, ...}` — which stamps
     `m.legacy = legacy` on the head class (harmless for LSCDetect, which builds its own
     shared cv2/cv3 regardless, but kept consistent with stock Detect handling).

Tested against ultralytics 8.3.x (pinned in requirements.txt); any pattern mismatch
raises RuntimeError immediately (fail fast, no silent breakage).
"""

import re

import ultralytics.nn.tasks as tasks

from models._patch_utils import get_current_parse_model_source, save_parse_model_source
from models.lwso.register import register_lwso

from .modules import LSCDetect

_registered = False


def register_slim() -> None:
    """Idempotent. Call once before building any model from a slim YAML."""
    global _registered
    if _registered:
        return

    register_lwso()  # SPDConvGroup/C3k2Ghost/EMA/ECA/DySample/BiFPNCat, reused unchanged

    setattr(tasks, LSCDetect.__name__, LSCDetect)

    src = get_current_parse_model_source(tasks)
    substitutions = [
        # 1. head frozenset: parse_model appends [ch[x] for x in f] for these classes
        (
            r"\{Detect, WorldDetect,",
            "{LSCDetect, Detect, WorldDetect,",
        ),
        # 2. legacy-flag set: m.legacy = legacy stamped on Detect-family classes
        (
            r"if m in \{Detect, YOLOEDetect,",
            "if m in {LSCDetect, Detect, YOLOEDetect,",
        ),
    ]
    for pattern, replacement in substitutions:
        src, count = re.subn(pattern, replacement, src, count=1)
        if count != 1:
            raise RuntimeError(
                f"register_slim: could not patch parse_model (pattern not found: {pattern!r}). "
                "Your ultralytics version is incompatible; install the range pinned in "
                "requirements.txt (ultralytics>=8.3,<8.4)."
            )

    code = compile(src, tasks.__file__, "exec")
    exec(code, tasks.__dict__)  # rebinds tasks.parse_model to the patched version
    save_parse_model_source(tasks, src)
    _registered = True
