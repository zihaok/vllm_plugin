# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project (the kernel below is adapted from vLLM's `prepare_megamoe_inputs`, Apache-2.0)
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / staging contract.
"""Single-launch Triton staging kernel of the SM107 MegaMoE inputs (the fast path of :mod:`.staging`).

One launch quantizes BF16 hidden states to FP8 E4M3 with per-32 UE8M0 scales (four exponents packed per int32), repacks the top-k
routing into the int64 / float32 layout of the symmetric buffer (``-1`` / ``0`` for padded tokens) and, when the fused shared
expert is used, writes the packed scales in the BLOCK_M-dependent shared-expert layout.  It is bit-identical to
:func:`.staging.per_token_cast_to_fp8` (DeepGEMM's cast) and to vLLM's kernel (both tested) for finite inputs and +-Inf; a NaN input is the exception (the NaN
lane stays NaN, but ``tl.max`` ignores it, so the scale byte of its group differs from DeepGEMM's; the inputs are expected to be finite).  Imported lazily:
``import triton`` happens only when this module is imported.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["num_tokens"])
def _prepare_megamoe_inputs_kernel(
    hidden_states,
    x_fp8,
    x_sf,
    shared_x_sf,
    topk_ids,
    topk_weights,
    is_padding,
    topk_idx_out,
    topk_weights_out,
    hidden_stride_m: tl.constexpr,
    hidden_stride_k: tl.constexpr,
    x_stride_m: tl.constexpr,
    x_stride_k: tl.constexpr,
    x_sf_stride_m: tl.constexpr,
    x_sf_stride_k: tl.constexpr,
    shared_x_sf_stride_m: tl.constexpr,
    shared_x_sf_stride_k: tl.constexpr,
    topk_ids_stride_m: tl.constexpr,
    topk_ids_stride_k: tl.constexpr,
    topk_weights_stride_m: tl.constexpr,
    topk_weights_stride_k: tl.constexpr,
    is_padding_stride_m: tl.constexpr,
    topk_idx_stride_m: tl.constexpr,
    topk_idx_stride_k: tl.constexpr,
    topk_weights_out_stride_m: tl.constexpr,
    topk_weights_out_stride_k: tl.constexpr,
    num_tokens,
    hidden_size: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_K: tl.constexpr,
    BLOCK_TOPK: tl.constexpr,
    SHARED_BLOCK_M: tl.constexpr,
) -> None:
    token_id = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    token_mask = token_id < num_tokens
    k_block_id = tl.program_id(1)

    k_offsets = k_block_id * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = token_mask[:, None] & (k_offsets[None, :] < hidden_size)
    hidden = tl.load(
        hidden_states
        + token_id[:, None] * hidden_stride_m
        + k_offsets[None, :] * hidden_stride_k,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    num_groups: tl.constexpr = BLOCK_K // GROUP_K
    hidden_groups = tl.reshape(tl.abs(hidden), [BLOCK_M, num_groups, GROUP_K])
    amax = tl.max(hidden_groups, axis=2)
    amax = tl.maximum(amax, 1.0e-4)

    # UE8M0 scale: 2^ceil(log2(amax / 448)) by the exponent bit trick (DeepGEMM `ceil_to_ue8m0`), exponent clamped to [1, 254]
    scale = amax / 448.0
    scale_bits = scale.to(tl.uint32, bitcast=True)
    scale_exp = ((scale_bits >> 23) & 0xFF) + ((scale_bits & 0x7FFFFF) != 0).to(
        tl.uint32
    )
    scale_exp = tl.minimum(tl.maximum(scale_exp, 1), 254)
    quant_scale = (scale_exp << 23).to(tl.float32, bitcast=True)

    hidden_groups = tl.reshape(hidden, [BLOCK_M, num_groups, GROUP_K])
    scaled = hidden_groups * (1.0 / quant_scale)[:, :, None]
    scaled = tl.reshape(scaled, [BLOCK_M, BLOCK_K])
    fp8 = scaled.to(tl.float8e4nv)
    tl.store(
        x_fp8 + token_id[:, None] * x_stride_m + k_offsets[None, :] * x_stride_k,
        fp8,
        mask=k_mask,
    )

    # four UE8M0 exponents per int32 (byte j = K group 4w + j)
    scale_offsets = tl.arange(0, num_groups)
    packed_scale = tl.sum(scale_exp << (scale_offsets[None, :] * 8), axis=1).to(
        tl.int32
    )
    tl.store(
        x_sf + token_id * x_sf_stride_m + k_block_id * x_sf_stride_k,
        packed_scale,
        mask=token_mask,
    )

    # shared-expert SF in the BLOCK_M-dependent MN-major layout (written while the packed scale is register resident)
    if shared_x_sf is not None:
        m_block_id = token_id // SHARED_BLOCK_M
        m_in_block = token_id % SHARED_BLOCK_M
        aligned_block_m: tl.constexpr = triton.cdiv(SHARED_BLOCK_M, 128) * 128
        transposed_m = (
            (m_in_block // 128) * 128 + (m_in_block % 32) * 4 + (m_in_block % 128) // 32
        )
        shared_row = m_block_id * aligned_block_m + transposed_m
        tl.store(
            shared_x_sf
            + shared_row * shared_x_sf_stride_m
            + k_block_id * shared_x_sf_stride_k,
            packed_scale,
            mask=token_mask,
        )

    if k_block_id == 0:
        topk_offsets = tl.arange(0, BLOCK_TOPK)
        topk_mask = token_mask[:, None] & (topk_offsets[None, :] < top_k)
        token_is_padding = tl.full((BLOCK_M,), False, tl.int1)
        if is_padding is not None:
            token_is_padding = tl.load(
                is_padding + token_id * is_padding_stride_m, mask=token_mask, other=True
            )

        ids = tl.load(
            topk_ids
            + token_id[:, None] * topk_ids_stride_m
            + topk_offsets[None, :] * topk_ids_stride_k,
            mask=topk_mask,
            other=0,
        ).to(tl.int64)
        ids = tl.where(token_is_padding[:, None], -1, ids)
        tl.store(
            topk_idx_out
            + token_id[:, None] * topk_idx_stride_m
            + topk_offsets[None, :] * topk_idx_stride_k,
            ids,
            mask=topk_mask,
        )

        weights = tl.load(
            topk_weights
            + token_id[:, None] * topk_weights_stride_m
            + topk_offsets[None, :] * topk_weights_stride_k,
            mask=topk_mask,
            other=0.0,
        )
        weights = tl.where(token_is_padding[:, None], 0.0, weights)
        tl.store(
            topk_weights_out
            + token_id[:, None] * topk_weights_out_stride_m
            + topk_offsets[None, :] * topk_weights_out_stride_k,
            weights,
            mask=topk_mask,
        )


def prepare_megamoe_inputs_triton(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    x_fp8: torch.Tensor,
    x_sf: torch.Tensor,
    topk_idx_out: torch.Tensor,
    topk_weights_out: torch.Tensor,
    is_padding: Optional[torch.Tensor] = None,
    shared_x_sf: Optional[torch.Tensor] = None,
    shared_block_m: Optional[int] = None,
) -> None:
    """Single-launch staging (same arguments / semantics as :func:`.staging.prepare_megamoe_inputs`; CUDA tensors only)."""
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    top_k = topk_ids.shape[1]
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and topk_ids to have the same shape."
        )
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError(
            "MegaMoE shared input staging requires both shared_x_sf and shared_block_m."
        )
    if shared_x_sf is not None:
        assert shared_block_m is not None
        if shared_block_m <= 0:
            raise ValueError("MegaMoE shared_block_m must be positive.")
        if shared_x_sf.ndim != 2 or shared_x_sf.shape[1] != hidden_size // 128:
            raise ValueError(
                f"MegaMoE shared_x_sf must have shape (*, {hidden_size // 128}), got {tuple(shared_x_sf.shape)}."
            )
        aligned_block_m = triton.cdiv(shared_block_m, 128) * 128
        required_rows = triton.cdiv(num_tokens, shared_block_m) * aligned_block_m
        if shared_x_sf.shape[0] < required_rows:
            raise ValueError(
                f"MegaMoE shared_x_sf has insufficient rows: requires {required_rows}, got {shared_x_sf.shape[0]}."
            )

    # eight-row tiles win from 64 tokens on GB200 (vLLM); keep smaller batches untiled
    block_m = 8 if num_tokens >= 64 else 1
    block_k = 128
    grid = (triton.cdiv(num_tokens, block_m), triton.cdiv(hidden_size, block_k))
    padding_stride_m = is_padding.stride(0) if is_padding is not None else 0
    _prepare_megamoe_inputs_kernel[grid](
        hidden_states, x_fp8, x_sf, shared_x_sf, topk_ids, topk_weights, is_padding, topk_idx_out, topk_weights_out,
        hidden_states.stride(0), hidden_states.stride(1), x_fp8.stride(0), x_fp8.stride(1), x_sf.stride(0), x_sf.stride(1),
        shared_x_sf.stride(0) if shared_x_sf is not None else 0, shared_x_sf.stride(1) if shared_x_sf is not None else 0,
        topk_ids.stride(0), topk_ids.stride(1), topk_weights.stride(0), topk_weights.stride(1), padding_stride_m,
        topk_idx_out.stride(0), topk_idx_out.stride(1), topk_weights_out.stride(0), topk_weights_out.stride(1),
        num_tokens, hidden_size, top_k,
        BLOCK_M=block_m, BLOCK_K=block_k, GROUP_K=32, BLOCK_TOPK=triton.next_power_of_2(top_k), SHARED_BLOCK_M=shared_block_m or 1,
        num_warps=4,
    )  # fmt: skip
