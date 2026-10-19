# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / deep_gemm/utils/math.py (per_token_cast_to_fp8) + tests/test_mega_moe.py (shared SF layout); the staging contract mirrors vLLM's prepare_megamoe_inputs (Apache-2.0).
"""Input staging of the MegaMoE: BF16 hidden states -> MXFP8 (E4M3) + packed UE8M0 scale factors + routing, written straight
into views of the symmetric buffer (``SymmBuffer.x / x_sf / topk_idx / topk_weights / shared_l1_acts_sf``).

This module is the pure-torch reference / fallback of the staging that vLLM does with its Triton kernel
``prepare_megamoe_inputs`` (one launch; the performance reference).  It is bit-exact against DeepGEMM's
``per_token_cast_to_fp8(x, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True)`` (tested for finite inputs, -0.0, denormals, fp16, +-Inf and NaN) and
uses the same function signature as vLLM's kernel (minus ``hidden_quant``) so that it can be swapped in 1:1.  The inputs are expected to be finite: the Triton
fast path matches this reference bit for bit on every finite value and on +-Inf, but a NaN input gives the NaN lane NaN and a different scale byte for its
group than DeepGEMM's cast (``tl.max`` ignores NaN); the rows and groups without NaN are unaffected (``tests/experimental/test_sm107_deepgemm_megamoe_staging_edge.py``).

Layout facts:

* ``x`` FP8 E4M3 ``(T, H)``; ``x_sf`` int32 ``(T, H/128)`` row-major, byte ``j`` of word ``w`` = UE8M0 exponent of K group
  ``4w + j`` (scale = ``2^ceil(log2(amax/448))``, amax clamped to >= 1e-4, exponent clamped to [1, 254]).
* ``topk_idx`` int64 ``(T, k)`` global expert ids, ``-1`` = masked/padded token; ``topk_weights`` float32 (0 for padded tokens).
* ``shared_l1_acts_sf`` int32 ``(N_s, H/128)`` with strides ``(1, N_s)`` (MN-major, ``N_s = ceil(T/8)*128``), written by the
  CALLER in a BLOCK_M-dependent row order: token ``t``, ``m = t % bm`` -> row
  ``(t // bm) * align(bm, 128) + (m // 128) * 128 + (m % 32) * 4 + (m % 128) // 32``.  ``bm`` is
  ``get_block_m_for_mega_moe`` of THIS call's token count, so the kernel contract includes that function.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from .heuristics import align

GROUP_K = 32  # one UE8M0 scale per 32 K elements
FP8_MAX = 448.0


def ceil_to_ue8m0(x: torch.Tensor) -> torch.Tensor:
    """Round positive float32 values up to the next power of two (bit trick; DeepGEMM ``ceil_to_ue8m0``)."""
    bits = x.abs().float().view(torch.int)
    exp = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exp.clamp(1, 254) << 23).view(torch.float)


def pack_ue8m0_to_int(x: torch.Tensor) -> torch.Tensor:
    """Pack four consecutive UE8M0 scales (float32 powers of two) into one int32 (byte j = scale j)."""
    assert x.dtype == torch.float and x.size(-1) % 4 == 0
    x_int = x.view(torch.int)
    assert (x_int >= 0).all() and (x_int & 0x7FFFFF == 0).all()
    return (x_int >> 23).to(torch.uint8).view(torch.int)


def per_token_cast_to_fp8(
    x: torch.Tensor,
    use_ue8m0: bool = True,
    gran_k: int = GROUP_K,
    use_packed_ue8m0: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact port of DeepGEMM's ``per_token_cast_to_fp8`` (``deep_gemm/utils/math.py``): ``(x_fp8 (m, n), sf)``;
    with ``use_packed_ue8m0`` the SF is int32 ``(m, ceil(n / gran_k / 4))``."""
    assert x.dim() == 2
    assert not use_packed_ue8m0 or use_ue8m0
    m, n = x.shape
    padded_n = align(n, gran_k)
    x_padded = torch.empty((m, padded_n), dtype=x.dtype, device=x.device).fill_(0)
    x_padded[:, :n] = x
    x_view = x_padded.view(m, padded_n // gran_k, gran_k)
    x_amax = x_view.abs().float().amax(dim=2).view(m, padded_n // gran_k).clamp(1e-4)
    sf = x_amax / FP8_MAX
    sf = ceil_to_ue8m0(sf) if use_ue8m0 else sf
    x_fp8 = (
        (x_view * (1.0 / sf.unsqueeze(2)))
        .to(torch.float8_e4m3fn)
        .view(m, padded_n)[:, :n]
        .contiguous()
    )
    if use_packed_ue8m0:
        num_sf = sf.size(-1)
        if (
            num_sf % 4 != 0
        ):  # pad with 1.0 (= 2^0): covers no real elements, never read by the kernel
            sf = torch.nn.functional.pad(sf, (0, align(num_sf, 4) - num_sf), value=1.0)
        return x_fp8, pack_ue8m0_to_int(sf)
    return x_fp8, sf


def get_shared_sf_rows(num_tokens: int, block_m: int, device=None) -> torch.Tensor:
    """Destination row of every token in the BLOCK_M-dependent shared-expert SF layout (vectorised
    ``_to_shared_mega_moe_sf_layout`` of DeepGEMM's test / the index math of vLLM's Triton kernel)."""
    t = torch.arange(num_tokens, dtype=torch.int64, device=device)
    m = t % block_m
    return (
        (t // block_m) * align(block_m, 128)
        + (m // 128) * 128
        + (m % 32) * 4
        + (m % 128) // 32
    )


def write_shared_sf(
    shared_x_sf: torch.Tensor, x_sf: torch.Tensor, block_m: int
) -> None:
    """Scatter the packed UE8M0 SF ``x_sf (M, H/128)`` into the shared-expert SF view ``(N_s, H/128)`` (strides ``(1, N_s)``)."""
    num_tokens = x_sf.shape[0]
    if num_tokens == 0:
        return
    assert shared_x_sf.dim() == 2 and shared_x_sf.shape[1] == x_sf.shape[1], (
        "shared_x_sf must have shape (*, H/128)"
    )
    required_rows = -(-num_tokens // block_m) * align(block_m, 128)
    assert shared_x_sf.shape[0] >= required_rows, (
        f"shared_x_sf has insufficient rows: requires {required_rows}, got {shared_x_sf.shape[0]}"
    )
    shared_x_sf.index_copy_(
        0, get_shared_sf_rows(num_tokens, block_m, x_sf.device), x_sf
    )


def to_shared_mega_moe_sf_layout(
    sf: torch.Tensor, block_m: int, num_max_sf_tokens: int
) -> torch.Tensor:
    """Fresh zero-initialised ``(num_max_sf_tokens, packed_sf_k)`` int32 tensor (strides ``(1, num_max_sf_tokens)``) holding ``sf``
    in the shared-expert layout; same result as DeepGEMM's test helper ``_to_shared_mega_moe_sf_layout``."""
    _, packed_sf_k = sf.shape
    result = torch.empty_strided(
        (num_max_sf_tokens, packed_sf_k),
        (1, num_max_sf_tokens),
        dtype=sf.dtype,
        device=sf.device,
    )
    result.zero_()
    write_shared_sf(result, sf, block_m)
    return result


_TRITON_OK: Optional[bool] = None


def triton_staging_available() -> bool:
    """Whether the single-launch Triton staging kernel (:mod:`.staging_triton`) can be imported (cached)."""
    global _TRITON_OK
    if _TRITON_OK is None:
        try:
            from . import staging_triton  # noqa: F401
        except (
            Exception
        ):  # no triton (CPU-only host) or an incompatible version: use the torch path
            _TRITON_OK = False
        else:
            _TRITON_OK = True
    return _TRITON_OK


def prepare_megamoe_inputs(
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
    *,
    use_triton: Optional[bool] = None,
) -> None:
    """Stage one call's inputs: the single-launch Triton kernel on CUDA tensors when Triton is available (default), else the torch
    reference :func:`prepare_megamoe_inputs_torch`.  Both are bit-identical (tested); ``use_triton`` forces one of them."""
    if use_triton is None:
        use_triton = hidden_states.is_cuda and triton_staging_available()
    if use_triton:
        from .staging_triton import prepare_megamoe_inputs_triton

        return prepare_megamoe_inputs_triton(
            hidden_states,
            topk_weights,
            topk_ids,
            x_fp8,
            x_sf,
            topk_idx_out,
            topk_weights_out,
            is_padding,
            shared_x_sf,
            shared_block_m,
        )
    return prepare_megamoe_inputs_torch(
        hidden_states,
        topk_weights,
        topk_ids,
        x_fp8,
        x_sf,
        topk_idx_out,
        topk_weights_out,
        is_padding,
        shared_x_sf,
        shared_block_m,
    )


def prepare_megamoe_inputs_torch(
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
    """Quantize hidden states and repack top-k routing for the MegaMoE (torch reference / fallback of vLLM's Triton kernel).

    Args mirror ``vllm...ops.prepare_megamoe.prepare_megamoe_inputs`` (UE8M0 / group 32 only): ``hidden_states`` ``[M, H]`` BF16,
    ``topk_weights`` / ``topk_ids`` ``[M, k]``; ``x_fp8`` / ``x_sf`` / ``topk_idx_out`` / ``topk_weights_out`` are the (``[:M]`` slices
    of the) symmetric-buffer views; ``is_padding`` an optional per-token mask (padded tokens are not routed: idx -1, weight 0);
    ``shared_x_sf`` + ``shared_block_m`` (given together) the shared-expert SF view and the BLOCK_M that lays it out.
    """
    num_tokens, hidden_size = hidden_states.shape
    if num_tokens == 0:
        return
    if hidden_size % 128 != 0:
        raise ValueError(
            "MegaMoE input staging requires hidden_size to be a multiple of 128."
        )
    if topk_weights.shape != topk_ids.shape:
        raise ValueError(
            "MegaMoE input staging requires topk_weights and topk_ids to have the same shape."
        )
    if (shared_x_sf is None) != (shared_block_m is None):
        raise ValueError(
            "MegaMoE shared input staging requires both shared_x_sf and shared_block_m."
        )
    if shared_block_m is not None and shared_block_m <= 0:
        raise ValueError("MegaMoE shared_block_m must be positive.")

    fp8, sf = per_token_cast_to_fp8(
        hidden_states, use_ue8m0=True, gran_k=GROUP_K, use_packed_ue8m0=True
    )
    x_fp8[:num_tokens].copy_(fp8)
    x_sf[:num_tokens].copy_(sf)
    if shared_x_sf is not None:
        write_shared_sf(shared_x_sf, sf, shared_block_m)

    ids = topk_ids.to(torch.int64)
    weights = topk_weights.to(torch.float32)
    if is_padding is not None:
        pad = is_padding[:num_tokens].to(torch.bool).view(-1, 1)
        ids = torch.where(pad, torch.full_like(ids, -1), ids)
        weights = torch.where(pad, torch.zeros_like(weights), weights)
    topk_idx_out[:num_tokens].copy_(ids)
    topk_weights_out[:num_tokens].copy_(weights)


def stage_inputs(
    sym_buffer,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    is_padding: Optional[torch.Tensor] = None,
    block_m: Optional[int] = None,
) -> int:
    """Stage one call's inputs into ``sym_buffer`` (``x``, ``x_sf``, routing and, with shared experts, the shared SF in the
    BLOCK_M layout of THIS call) and return the BLOCK_M used.  ``hidden_states.size(0)`` is the call's token count."""
    num_tokens = hidden_states.size(0)
    assert num_tokens <= sym_buffer.num_max_tokens_per_rank, (
        "num_tokens must be <= num_max_tokens_per_rank"
    )
    if block_m is None:
        from .frontend import (
            get_block_m_for_mega_moe as _block_m_of_the_launch,
        )  # lazy (frontend does not import staging); a forced BLOCK_M (KernelOptions.block_m) wins, as for the launch

        block_m = _block_m_of_the_launch(
            sym_buffer.group.size(), sym_buffer.num_experts, sym_buffer.num_max_tokens_per_rank, num_tokens,
            sym_buffer.num_topk, sym_buffer.mma_type,
        )  # fmt: skip
    shared = sym_buffer.shared_l1_acts_sf if sym_buffer.num_shared_experts > 0 else None
    prepare_megamoe_inputs(
        hidden_states, topk_weights, topk_ids, sym_buffer.x, sym_buffer.x_sf, sym_buffer.topk_idx, sym_buffer.topk_weights,
        is_padding=is_padding, shared_x_sf=shared, shared_block_m=block_m if shared is not None else None,
    )  # fmt: skip
    return block_m
