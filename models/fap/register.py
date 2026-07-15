"""Register FreqMix into ultralytics without forking the repo — same trick as
models/lwso/register.py (see there for why: parse_model's channel-inference sets are
local variables inside the function body, so we setattr the class onto the module
namespace, then patch parse_model's source and re-exec it).

Tested against ultralytics 8.3.x (pinned in requirements.txt).
"""

import re

import ultralytics.nn.tasks as tasks

from models._patch_utils import get_current_parse_model_source, save_parse_model_source

from .modules import FreqMix

_registered = False


def register_fap() -> None:
    """Idempotent. Call once before building any model from an FAP YAML."""
    global _registered
    if _registered:
        return

    setattr(tasks, FreqMix.__name__, FreqMix)

    src = get_current_parse_model_source(tasks)
    pattern = r"base_modules = frozenset\(\s*\{"
    replacement = "base_modules = frozenset(\n        {\n            FreqMix,"
    src, count = re.subn(pattern, replacement, src, count=1)
    if count != 1:
        raise RuntimeError(
            f"register_fap: could not patch parse_model (pattern not found: {pattern!r}). "
            "Your ultralytics version is incompatible; install the range pinned in "
            "requirements.txt (ultralytics>=8.3,<8.4)."
        )

    code = compile(src, tasks.__file__, "exec")
    exec(code, tasks.__dict__)  # rebinds tasks.parse_model to the patched version
    save_parse_model_source(tasks, src)
    _registered = True
