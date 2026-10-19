# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / package __init__.
"""EXPERIMENTAL: CuteDSL port of DeepGEMM's Rubin (SM107) MegaMoE kernel (MXFP8 x MXFP4 -> BF16).

Importing this package is lightweight and CPU-safe: it pulls in only the pure-Python ``config`` and ``heuristics``
modules (no torch, CUDA, cutlass or nvshmem at module scope).  Everything else is resolved lazily on first attribute
access (PEP 562), so ``import flashinfer.experimental.sm107_deepgemm_megamoe`` never compiles or allocates anything.

Public experimental APIs live in core (``flashinfer.mega_moe_deepgemm``, decorated with
``@flashinfer_experimental_api``) and hand off to this package through function-local imports.
"""

from __future__ import annotations

import importlib
from typing import Any

from .config import VALID_BLOCK_MS, Sm107KernelConfig, Sm107MegaMoeConfig
from .heuristics import (
    get_block_m_for_mega_moe,
    get_sm107_kernel_config,
    get_token_alignment_for_mega_moe,
)

# name -> (submodule, attribute) of everything that needs torch / CUDA and is therefore imported lazily
_LAZY = {
    "SymmBuffer": ("workspace", "SymmBuffer"),
    "get_symm_buffer_for_mega_moe": ("compat", "get_symm_buffer_for_mega_moe"),
    "transform_weights_for_mega_moe": ("compat", "transform_weights_for_mega_moe"),
    "fp8_fp4_mega_moe": ("compat", "fp8_fp4_mega_moe"),
}

__all__ = [
    "VALID_BLOCK_MS",
    "Sm107KernelConfig",
    "Sm107MegaMoeConfig",
    "get_block_m_for_mega_moe",
    "get_sm107_kernel_config",
    "get_token_alignment_for_mega_moe",
    *_LAZY,
]


def __getattr__(name: str) -> Any:
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{target[0]}", __name__)
    value = getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
