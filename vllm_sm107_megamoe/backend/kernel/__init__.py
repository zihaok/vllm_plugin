# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / package __init__.
"""CuTe-DSL device code of the SM107 DeepGEMM-style MegaMoE kernel.

Importing this package is CPU-safe (``config`` and ``layout`` are pure Python); the kernel class (which imports ``cutlass``) is
resolved lazily on first attribute access.  See ``docs/sm107_deepgemm_megamoe/DESIGN.md``.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import (  # noqa: F401
    MODE_ALL,
    MODE_COMBINE,
    MODE_DISPATCH,
    MODE_GEMM,
    KernelBuild,
)

_LAZY = {
    "DeepGemmSm107MegaMoeKernel": ("kernel", "DeepGemmSm107MegaMoeKernel"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod, attr = _LAZY[name]
        value = getattr(importlib.import_module(f".{mod}", __name__), attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["DeepGemmSm107MegaMoeKernel", "KernelBuild", "MODE_ALL", "MODE_COMBINE", "MODE_DISPATCH", "MODE_GEMM"]
