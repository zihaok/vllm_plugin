# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / deep_gemm/mega/__init__.py (SymmBuffer) + layout/mega_moe.cuh.
"""Symmetric-memory workspace of the SM107 MegaMoE: ``SymmBuffer`` (DeepGEMM-compatible) and its byte layout.

The byte layout is an external ABI (vLLM's Triton staging kernel writes ``x``, ``x_sf``, ``topk_idx``, ``topk_weights`` and
the shared-expert SF into views of this buffer) and is therefore bit-identical to DeepGEMM's ``MegaMoEBuffer``
(``deep_gemm/include/deep_gemm/layout/mega_moe.cuh``); the pure-Python layout math lives in :mod:`.heuristics`
(``build_symm_buffer_layout``) and is verified against DeepGEMM's C++ (``get_symm_buffer_size_for_mega_moe``).

Semantics (DeepGEMM ``mega/__init__.py:26-91``): ``T = align(num_max_tokens_per_rank, 1920)``; one
``symm_mem.empty(num_bytes, int8)`` + ``symm_mem.rendezvous(buffer, group)`` (plain ``torch.empty`` and a one-entry fake handle
when ``group.size() == 1``), then ``zero_()``, ``group.barrier()``, ``torch.cuda.synchronize()``.  All ranks must allocate
identical sizes (collective).  The kernel restores the workspace to its all-zero state on exit, so no per-call reset is needed.
"""

from __future__ import annotations

import types
import warnings
from typing import Dict, List, Optional, Union

import torch

from .heuristics import (
    SymmBufferLayout,
    ViewSpec,
    align,
    build_symm_buffer_layout,
    get_token_alignment_for_mega_moe,
)

_SUPPORTED_ACTIVATIONS = ("swiglu", "swiglustep", "situ")
_DTYPES = {
    "float8_e4m3fn": torch.float8_e4m3fn,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
    "float32": torch.float32,
}


def _validate_activation(activation: str) -> None:
    assert activation in _SUPPORTED_ACTIVATIONS, (
        f"Activation must be one of {_SUPPORTED_ACTIVATIONS}, got `{activation}`"
    )


def _device_num_sms() -> int:
    return int(
        torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    )


def _make_view(buffer: torch.Tensor, spec: ViewSpec) -> torch.Tensor:
    """Typed (optionally strided) view into the int8 byte buffer (``torch_compat::from_blob`` in DeepGEMM)."""
    dtype = _DTYPES[spec.dtype]
    numel = 1
    for s in spec.shape:
        numel *= s
    nbytes = numel * torch.empty((), dtype=dtype).element_size()
    flat = buffer[spec.offset : spec.offset + nbytes].view(dtype)
    if spec.strides is None:
        return flat.view(*spec.shape)
    return torch.as_strided(flat, spec.shape, spec.strides)


class SymmBuffer:
    """DeepGEMM-compatible symmetric buffer (``deep_gemm.SymmBuffer``).

    vLLM-facing attributes: ``x``, ``x_sf``, ``topk_idx``, ``topk_weights``, ``shared_l1_acts_sf``,
    ``num_max_tokens_per_rank`` (the ALIGNED value).  Internal (consumed by ``fp8_fp4_mega_moe``): ``buffer``, ``handle``
    (``.buffer_ptrs``), ``group``, ``num_experts``, ``num_topk``, ``hidden``, ``intermediate_hidden``, ``num_shared_experts``.
    Views for tests / staging: ``shared_l1_acts`` (alias of ``x``), ``shared_l2_acts``, ``shared_l2_acts_sf``, ``l1_acts``,
    ``l1_acts_sf``, ``l2_acts``, ``l2_acts_sf``.  ``layout`` is the :class:`~.heuristics.SymmBufferLayout` (all region offsets).
    """

    def __init__(
        self,
        group,
        num_experts: int,
        num_max_tokens_per_rank: int,
        num_topk: int,
        hidden: int,
        intermediate_hidden: int,
        num_shared_experts: int = 0,
        mma_type: str = "fp8xfp4",
        activation: str = "swiglu",
        base: Optional["SymmBuffer"] = None,
    ):
        # Align token count
        num_max_tokens_per_rank = align(
            num_max_tokens_per_rank, get_token_alignment_for_mega_moe()
        )

        # Init
        _validate_activation(activation)
        self.group = group
        self.num_experts = num_experts
        self.num_max_tokens_per_rank = num_max_tokens_per_rank
        self.num_topk = num_topk
        self.hidden = hidden
        self.intermediate_hidden = intermediate_hidden
        self.num_shared_experts = num_shared_experts
        self.mma_type = mma_type
        self.activation = activation

        # Byte layout (ring tokens depend on the device SM count: kNumSMs is the whole device)
        self.num_sms = _device_num_sms()
        self.layout: SymmBufferLayout = build_symm_buffer_layout(
            group.size(), num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
            mma_type, activation, num_shared_experts, self.num_sms,
        )  # fmt: skip
        num_bytes = self.layout.num_bytes

        # Allocate or reuse a symmetric buffer
        if base is None:
            if group.size() == 1:
                self.buffer = torch.empty(num_bytes, dtype=torch.int8, device="cuda")
                self.handle = types.SimpleNamespace(
                    buffer_ptrs=[self.buffer.data_ptr()]
                )
            else:
                import torch.distributed._symmetric_memory as symm_mem  # torch >= 2.9 / 2.16 dev

                self.buffer = symm_mem.empty(num_bytes, dtype=torch.int8, device="cuda")
                self.handle = symm_mem.rendezvous(self.buffer, group=group)
            self.buffer.zero_()
            self.group.barrier()
            torch.cuda.synchronize()
        else:
            assert (
                base.buffer is not None
                and base.handle is not None
                and base.group is group
            ), "Cannot reuse an invalid symmetric buffer"
            assert num_bytes <= base.buffer.nbytes, (
                f"The reused Mega MoE config requires {num_bytes} bytes, but the symmetric buffer only has {base.buffer.nbytes} bytes"
            )  # fmt: skip
            self.buffer = base.buffer
            self.handle = base.handle

        self._build_views()

    # ------------------------------------------------------------------------------------------------------ views
    def _build_views(self) -> None:
        v: Dict[str, ViewSpec] = self.layout.views
        get = lambda name: _make_view(self.buffer, v[name]) if name in v else None  # noqa: E731
        self.x = get("x")
        self.x_sf = get("x_sf")
        self.topk_idx = get("topk_idx")
        self.topk_weights = get("topk_weights")
        self.shared_l1_acts = get("shared_l1_acts")
        self.shared_l1_acts_sf = get("shared_l1_acts_sf")
        self.shared_l2_acts = get("shared_l2_acts")
        self.shared_l2_acts_sf = get("shared_l2_acts_sf")
        self.l1_acts = get("l1_acts")
        self.l1_acts_sf = get("l1_acts_sf")
        self.l2_acts = get("l2_acts")
        self.l2_acts_sf = get("l2_acts_sf")

    @property
    def view_names(self) -> List[str]:
        return ["x", "x_sf", "topk_idx", "topk_weights", "shared_l1_acts", "shared_l1_acts_sf", "shared_l2_acts",
                "shared_l2_acts_sf", "l1_acts", "l1_acts_sf", "l2_acts", "l2_acts_sf"]  # fmt: skip

    # ------------------------------------------------------------------------------------------------ kernel ABI
    @property
    def rank(self) -> int:
        return self.group.rank()

    @property
    def num_ranks(self) -> int:
        return self.group.size()

    @property
    def base_ptr(self) -> int:
        """Address of this rank's buffer (``SymBuffer::base``)."""
        return self.buffer.data_ptr()

    @property
    def peer_offsets(self) -> List[int]:
        """``offsets[i] = ptrs[i] - ptrs[rank]`` (``SymBuffer`` constructor, ``layout/sym_buffer.cuh``): byte offset that maps a
        local pointer into rank ``i``'s buffer; ``offsets[rank] == 0``."""
        ptrs = list(self.handle.buffer_ptrs)
        me = ptrs[self.group.rank()]
        return [int(p) - int(me) for p in ptrs]

    def destroy(self) -> None:
        """Release the buffer (collective in spirit: every rank must call it after its last launch; the kernel's final NVLink
        barrier guarantees no remote access into this buffer after the launch completes locally)."""
        if getattr(self, "buffer", None) is not None:
            torch.cuda.synchronize()
        self.handle = None
        self.buffer = None
        self.group = None
        for name in self.view_names:
            setattr(self, name, None)


# TODO(DeepGEMM): remove this function
def get_symm_buffer_for_mega_moe(
    group,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    use_fp8_dispatch: Union[bool, None] = None,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
) -> SymmBuffer:
    # Backward compat: derive `mma_type` from `use_fp8_dispatch` if provided
    if use_fp8_dispatch is not None:
        assert use_fp8_dispatch == (mma_type.split("x")[0] == "fp8")
        warnings.warn(
            "`use_fp8_dispatch` will be deprecated in the future, please use `mma_type`",
            DeprecationWarning,
            stacklevel=3,
        )

    return SymmBuffer(
        group, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts,
        mma_type=mma_type, activation=activation,
    )  # fmt: skip
