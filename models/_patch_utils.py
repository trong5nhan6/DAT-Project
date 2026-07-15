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
