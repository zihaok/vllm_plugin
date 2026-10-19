# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / deep_gemm/mega/__init__.py (transform_weights_for_mega_moe).
"""Weight layout transform of the MegaMoE kernel in pure torch (no ``deep_gemm`` import).

DeepGEMM contract:

* L1 weights ``(E_l, 2I, H/2)`` int8 (two E2M1 values per byte, low nibble = even K) get a **gate/up 8-row interleave**:
  ``[g0..g7, u0..u7, g8..g15, u8..u15, ...]`` instead of ``[gate | up]``, so that one ``tcgen05.ld.16x256b`` gives each thread
  gate/up pairs of the same channel.  L2 weights ``(E_l, H, I/2)`` are unchanged.
* Weight scale factors: int32 ``(E_l, N, ceil(K/128))`` (four UE8M0 bytes per word, byte ``b`` of word ``w`` = K group
  ``4w + b``), MN-major (``stride(-2) == 1``; the layout ``transform_sf_into_required_layout`` produces).  L1 SF gets the same
  8-row interleave, then L1 and L2 SF get the **UTCCP 4x32 transpose** inside every 128-row block: stored position
  ``4*c + a`` of row ``32*a + c``.

Bit-exactness against DeepGEMM's function is tested in ``tests/experimental/test_sm107_deepgemm_megamoe_weights.py``.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch

_SUPPORTED_ACTIVATIONS = ("swiglu", "swiglustep", "situ")


def _validate_activation(activation: str) -> None:
    assert activation in _SUPPORTED_ACTIVATIONS, (
        f"Activation must be one of {_SUPPORTED_ACTIVATIONS}, got `{activation}`"
    )


def _align(x: int, a: int) -> int:
    return -(-x // a) * a


def sf_to_required_layout(
    sf: torch.Tensor, mn: int, k: int, num_groups: Optional[int] = None
) -> torch.Tensor:
    """Per-32-K UE8M0 scales -> the kernel's weight-SF layout (``deep_gemm.transform_sf_into_required_layout`` with recipe
    ``(1, 32)`` on SM100/SM107).

    ``sf``: float32 powers of two ``([G,] mn, ceil(k/32))`` (UE8M0 values, mantissa zero) or uint8 E8M0 bytes of the same shape.
    Returns int32 ``([G,] mn, ceil(k/128))`` (byte ``b`` of word ``w`` = group ``4w + b``; a short last word is padded with the
    UE8M0 value 1.0), MN-major and TMA-aligned: strides ``(ceil(k/128) * align(mn, 4), 1, align(mn, 4))`` (3D)."""
    assert sf.dim() == (3 if num_groups is not None else 2), (
        "expected ([G,] mn, k/32) scales"
    )
    squeeze = num_groups is None
    if squeeze:
        sf = sf.unsqueeze(0)
    g, n, k32 = sf.shape
    assert n == mn and k32 == -(-k // 32), (
        f"scale shape {tuple(sf.shape)} does not match mn={mn}, k={k}"
    )
    if num_groups is not None:
        assert g == num_groups
    if sf.dtype == torch.uint8:
        bytes_ = sf
    else:
        assert sf.dtype == torch.float32, (
            "scales must be float32 (UE8M0 values) or uint8 (E8M0 bytes)"
        )
        bits = sf.view(torch.int32)
        assert (bits >= 0).all() and (bits & 0x7FFFFF == 0).all(), (
            "float32 scales must be non-negative powers of two"
        )
        bytes_ = (bits >> 23).to(torch.uint8)
    k128 = -(-k32 // 4)
    if k32 % 4:  # pad with the UE8M0 value 1.0 (exponent 127): covers no real elements
        bytes_ = torch.nn.functional.pad(bytes_, (0, k128 * 4 - k32), value=127)
    packed = bytes_.contiguous().view(
        torch.int32
    )  # (g, n, k128), little endian: byte b = group 4w + b
    n4 = _align(mn, 4)
    out = torch.empty_strided(
        (g, mn, k128), (k128 * n4, 1, n4), dtype=torch.int32, device=sf.device
    )
    out.copy_(packed)
    return out.squeeze(0) if squeeze else out


def interleave_weights(t: torch.Tensor, gran: int = 8) -> torch.Tensor:
    """``[gate: 0..7, up: 0..7, gate: 8..15, up: 8..15, ...]`` instead of ``[gate | up]`` along dim 1 (dim 0 for 2D inputs)."""
    assert t.dim() in (2, 3)
    squeeze_group_dim = t.dim() == 2
    if squeeze_group_dim:
        t = t.unsqueeze(0)

    g, n, *rest = t.shape
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    result = torch.empty_like(t).copy_(
        torch.stack([gate, up], dim=2).reshape(g, n, *rest)
    )
    return result.squeeze(0) if squeeze_group_dim else result


def transpose_sf_for_utccp(sf: torch.Tensor) -> torch.Tensor:
    """UTCCP 4x32 transpose of int32 scale factors ``(G, MN, K/128)`` inside every 128-row block (``MN % 128 == 0``).

    ``empty_like`` keeps the (MN-major) strides of the input: the result has the layout the TMA descriptor expects."""
    assert sf.dtype == torch.int and sf.dim() in (2, 3)
    squeeze_group_dim = sf.dim() == 2
    if squeeze_group_dim:
        sf = sf.unsqueeze(0)

    num_groups, mn, packed_sf_k = sf.shape
    assert mn % 128 == 0
    result = (
        sf.reshape(num_groups, -1, 4, 32, packed_sf_k)
        .transpose(2, 3)
        .reshape(num_groups, mn, packed_sf_k)
    )
    result = torch.empty_like(sf).copy_(result)
    return result.squeeze(0) if squeeze_group_dim else result


def transform_weights_for_mega_moe(
    l1_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    l2_weights: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    activation: str = "swiglu",
) -> Tuple[
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
]:
    """DeepGEMM-compatible ``transform_weights_for_mega_moe`` (``mega/__init__.py:154-172``).

    ``l1_weights`` / ``l2_weights``: ``(weights, sf)`` tuples (FP8 / packed FP4 with int32 UE8M0 SF), or plain tensors (BF16 path).
    Shared-expert weights go through the same function with 2D tensors.
    """
    _validate_activation(activation)
    if isinstance(l1_weights, tuple):
        # FP8/FP4: interleave gate/up for weight and SF, then transpose L1 SF for UTCCP
        l1_w = interleave_weights(l1_weights[0])
        l1_sf = transpose_sf_for_utccp(interleave_weights(l1_weights[1]))
        l1_transformed = (l1_w, l1_sf)
        # L2: only transpose SF for UTCCP
        l2_transformed = (l2_weights[0], transpose_sf_for_utccp(l2_weights[1]))
    else:
        # BF16: L1 interleave gate/up, L2 unchanged
        l1_transformed = interleave_weights(l1_weights)
        l2_transformed = l2_weights
    return l1_transformed, l2_transformed
