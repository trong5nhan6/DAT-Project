"""Register idea "star"'s custom modules into ultralytics without forking the repo — same
trick as models/lwso/register.py and models/fap/register.py (see those for why: parse_model's
channel-inference sets are local variables inside the function body, so we setattr our
classes onto the module namespace, then patch parse_model's source and re-exec it).

Builds on register_lwso() and register_fap() rather than starting from the pristine source:
cfg/star-yolo11n.yaml reuses DySample/BiFPNCat (from models/lwso/modules.py) and FreqMix (from
models/fap/modules.py) unchanged — FreqMix measured cheaper than SPDConvGroup for P2-having
architectures (fap-yolo11n.yaml: 8.96 GFLOPs@640 with 4 detect scales vs the earlier
SPDConvGroup-based star-yolo11n.yaml draft: 13.67 GFLOPs@640 with only 3), so v2 of this idea's
cfg downsamples with FreqMix instead. Calling both register_lwso()/register_fap() first
(idempotent) covers that, and this function only patches parse_model further for the 3 classes
unique to this idea (C3k2Star, SimAM, VoVGSCSP). Composes correctly regardless of call order via
the shared source-cache in models/_patch_utils.py.

Tested against ultralytics 8.3.x (pinned in requirements.txt).
"""

import re

import ultralytics.nn.tasks as tasks

from models._patch_utils import get_current_parse_model_source, save_parse_model_source
from models.fap.register import register_fap
from models.lwso.register import register_lwso

from .modules import C3k2Star, SimAM, VoVGSCSP

_registered = False


def register_star() -> None:
    """Idempotent. Call once before building any model from a star YAML."""
    global _registered
    if _registered:
        return

    register_lwso()  # SPDConv(Group)/C3k2Ghost/EMA/ECA/DySample/BiFPNCat, reused unchanged
    register_fap()  # FreqMix, reused unchanged (cheaper P2/P3/P4 downsample than SPDConvGroup)

    for cls in (C3k2Star, SimAM, VoVGSCSP):
        setattr(tasks, cls.__name__, cls)

    src = get_current_parse_model_source(tasks)
    substitutions = [
        # channel-inferring modules with (c1, c2, ...) signatures
        (
            r"base_modules = frozenset\(\s*\{",
            "base_modules = frozenset(\n        {\n"
            "            C3k2Star,\n            SimAM,\n            VoVGSCSP,",
        ),
        # modules whose 3rd arg is the repeat count n
        (
            r"repeat_modules = frozenset\([^\{]*\{",
            "repeat_modules = frozenset(  # modules with 'repeat' arguments\n        {\n"
            "            C3k2Star,\n            VoVGSCSP,",
        ),
        # C3k2Star/VoVGSCSP behave like C3k2/C3k2Ghost (non-legacy Detect head, c3k flag
        # on m/l/x scales) — pattern already includes C3k2Ghost since register_lwso() ran first
        (
            r"if m is C3k2 or m is C3k2Ghost:",
            "if m is C3k2 or m is C3k2Ghost or m is C3k2Star or m is VoVGSCSP:",
        ),
    ]
    for pattern, replacement in substitutions:
        src, count = re.subn(pattern, replacement, src, count=1)
        if count != 1:
            raise RuntimeError(
                f"register_star: could not patch parse_model (pattern not found: {pattern!r}). "
                "Your ultralytics version is incompatible; install the range pinned in "
                "requirements.txt (ultralytics>=8.3,<8.4)."
            )

    code = compile(src, tasks.__file__, "exec")
    exec(code, tasks.__dict__)  # rebinds tasks.parse_model to the patched version
    save_parse_model_source(tasks, src)
    _registered = True
