# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / deep_gemm/mega/__init__.py (public names).
"""Drop-in re-exports of the DeepGEMM MegaMoE names so that a DeepGEMM import can be swapped for this module::

    from flashinfer.experimental.sm107_deepgemm_megamoe import compat as deep_gemm_like
    buf = deep_gemm_like.get_symm_buffer_for_mega_moe(group, E, T, topk, H, I, num_shared_experts=1)
    w1, w2 = deep_gemm_like.transform_weights_for_mega_moe(l1, l2)
    deep_gemm_like.fp8_fp4_mega_moe(y, w1, w2, buf, ...)
    bm = deep_gemm_like.get_block_m_for_mega_moe(R, E, T, num_tokens, topk, "fp8xfp4")

and the per-die helpers of ``deep_gemm.mega.sm107_locality`` (``is_localization_available``, ``get_balanced_sm_locality_domains``, ``localize``,
``localize_mega_moe_weights``; pure Python on cuda-python, no ``deep_gemm`` needed) are re-exported as well.

The public, policy-compliant entry points for vLLM live in core (``flashinfer.mega_moe_deepgemm``) and hand off here.
This module never imports ``deep_gemm``.
"""

from __future__ import annotations

from .frontend import (
    KernelOptions,
    fp8_fp4_mega_moe,
    get_block_m_for_mega_moe,
    get_kernel_options,
    precompile_mega_moe_variants,
    set_kernel_options,
)
from .heuristics import get_token_alignment_for_mega_moe
from .locality import (
    get_balanced_sm_locality_domains,
    is_localization_available,
    localize,
    localize_mega_moe_weights,
    weights_localized,
)
from .staging import prepare_megamoe_inputs, stage_inputs
from .weights import transform_weights_for_mega_moe
from .workspace import SymmBuffer, get_symm_buffer_for_mega_moe

__all__ = [
    "KernelOptions",
    "SymmBuffer",
    "fp8_fp4_mega_moe",
    "get_balanced_sm_locality_domains",
    "get_block_m_for_mega_moe",
    "get_kernel_options",
    "get_symm_buffer_for_mega_moe",
    "get_token_alignment_for_mega_moe",
    "is_localization_available",
    "localize",
    "localize_mega_moe_weights",
    "precompile_mega_moe_variants",
    "prepare_megamoe_inputs",
    "set_kernel_options",
    "stage_inputs",
    "transform_weights_for_mega_moe",
    "weights_localized",
]
