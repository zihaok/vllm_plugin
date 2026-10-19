# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / csrc/jit_kernels/heuristics/mega_moe.hpp + csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp + deep_gemm/layout/mega_moe.cuh + scheduler/mega_moe.cuh.
"""Pure-Python, bit-exact port of DeepGEMM's host-side MegaMoE logic (Rubin / SM107 flavour).

No torch / CUDA / cutlass import: usable on CPU-only hosts, by the kernel code, the JIT layer, the API layer and tests.

Contents (DeepGEMM source in parentheses, branch ``rubin-megamoe-a7a9277-1016`` @ 3ec0eb4):

* BLOCK_M selection ``get_block_m_for_mega_moe`` with float32 semantics (``heuristics/mega_moe.hpp:88-131``)
* live-pool-block / ring sizing ``get_num_max_live_pool_blocks``, ``get_ring_sizes`` (``scheduler/mega_moe.cuh:15-80``,
  ``apis/mega_moe.hpp:113-155``)
* the SM100 pipeline sizing ``get_pipeline_config_for_mega_moe`` + ``get_mega_moe_config`` (what ``DG_PRINT_CONFIGS``
  prints as the ``MegaMoEConfig(...)`` line; NOT the shipped SM107 values) (``heuristics/mega_moe.hpp:133-267``)
* the SM107 re-derivation of stages / dynamic smem (``impls/sm107_fp8_fp4_mega_moe.hpp:100-144``; the shipped values,
  printed as ``SM107 MegaMoE: stages=, smem=``)
* ``SharedStorage`` byte offsets (``K:250-275``), TMEM columns (``K:283-290``), register split (``K:385-388``), combine
  chunking (``K:1652-1665``)
* the symmetric-buffer byte layout and ``MegaMoESignals`` offsets (``layout/mega_moe.cuh``)
* the :func:`get_sm107_kernel_config` factory producing :class:`~.config.Sm107KernelConfig`.

Everything is parametric in the SM count (212 on Rubin VR200; 148/208 supported) and in the model shape.
Semantics are those of the C++ code (integer truncating division, float32 arithmetic); every function cites the lines
it ports.  ``HOST_ASSERT`` failures raise ``AssertionError`` (DeepGEMM: ``DG_HOST_ASSERT``).
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .config import VALID_BLOCK_MS, Sm107KernelConfig

# ===================================================================================================================
# Constants (layout/mega_moe.cuh:13-17, heuristics/sm100.hpp:16, common/types.cuh:54, layout/sym_buffer.cuh:7)
# ===================================================================================================================
CANDIDATE_BLOCK_MS: Tuple[int, ...] = (
    8,
    16,
    32,
    64,
    96,
    128,
    192,
    240,
)  # kCandidateBlockM
MAX_CANDIDATE_BLOCK_M = 240  # kMaxCandidateBlockM
MIN_CANDIDATE_BLOCK_M = 8  # kMinCandidateBlockM
TOKEN_ALIGNMENT = 1920  # kLCMCandidateBlockM == lcm(CANDIDATE_BLOCK_MS)
SM100_SMEM_CAPACITY = (
    232448  # SM100ArchSpec::smem_capacity (also used for SM107, IMPL:112)
)
NUM_DEVICE_LOCALITY_DOMAINS = 2  # kNumDeviceLocalityDomains
MAX_RANKS = 72  # kNumMaxRanks
SMEM_ALIGNMENT = (
    1024  # kSharedMemoryAlignment (alignas of the big SharedStorage members)
)
MAX_PULL_BYTES_WITH_SF = 8192  # num_max_pull_bytes for the FP8xFP4 kinds (HEUR:228)
UMMA_K_PACKED = 64

assert math.lcm(*CANDIDATE_BLOCK_MS) == TOKEN_ALIGNMENT


def _host_assert(cond: bool, msg: str = "") -> None:
    if not cond:
        raise AssertionError(msg or "host assertion failed")


def cdiv(a: int, b: int) -> int:
    """``ceil_div`` for positive ints."""
    return -(-a // b)


def align(a: int, b: int) -> int:
    return cdiv(a, b) * b


def _idiv(a: int, b: int) -> int:
    """C++ ``int / int`` (truncates toward zero)."""
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _f32(x: float) -> float:
    """Round a Python float (double) to the nearest float32 (RN-even); the result is held in a double."""
    return ctypes.c_float(x).value


def get_token_alignment_for_mega_moe() -> int:
    """``layout::kLCMCandidateBlockM`` (== ``deep_gemm._C.get_token_alignment_for_mega_moe()``)."""
    return TOKEN_ALIGNMENT


# ===================================================================================================================
# MMA kinds
# ===================================================================================================================
MMA_BF16 = "bf16"
MMA_MXFP8FP4 = "mxfp8fp4"


def parse_mma_kind(mma_type: str) -> str:
    """``parse_mma_kind`` (HEUR:73-78): 'bf16xbf16' -> BF16, 'fp8xfp4' / 'fp8xfp8' -> MXFP8FP4."""
    if mma_type == "bf16xbf16":
        return MMA_BF16
    _host_assert(
        mma_type in ("fp8xfp4", "fp8xfp8"), f"unsupported mma_type {mma_type!r}"
    )
    return MMA_MXFP8FP4


def _num_mma_elem_bytes(kind: str) -> int:
    return 2 if kind == MMA_BF16 else 1


def _is_mma_with_sf(kind: str) -> bool:
    return kind == MMA_MXFP8FP4


# ===================================================================================================================
# BLOCK_M selection (HEUR:88-131) -- float32 semantics
# ===================================================================================================================
def get_block_config_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_tokens: int,
    mma_kind: str,
    force_block_m: int = 0,
) -> Tuple[int, int, int, int, int, int]:
    """Returns ``(cluster_size, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads)``.

    ``num_max_tokens_per_rank`` is ignored (as in C++).  ``force_block_m`` stands for the opt-in env knob
    ``DG_MEGA_MOE_SM107_BLOCK_M`` (explicit argument here, no env vars).
    """
    _host_assert(
        num_ranks > 0 and num_experts > 0, "num_ranks and num_experts must be positive"
    )
    # static_cast<float>(num_tokens) * num_ranks * num_topk / num_experts   (left to right, each op rounded to float32)
    e = _f32(_f32(_f32(float(num_tokens)) * num_ranks) * num_topk)
    e = _f32(e / num_experts)
    covered = _f32(
        e + _f32(math.sqrt(e))
    )  # num_expected_tokens + std::sqrt(num_expected_tokens)

    block_m = 16 if e <= 10 else 32
    if e > 24:
        num_blocks = int(math.ceil(_f32(covered / 240)))
        if num_blocks == 1 and covered > 192 and _idiv(num_experts, num_ranks) < 14:
            num_blocks = 2
        for candidate in (64, 128, 192, 240):
            block_m = candidate
            if _f32(float(num_blocks * candidate)) >= covered:
                break
    if force_block_m > 0:
        block_m = force_block_m

    # asymmetric store blocks of the SM100 kernel (HEUR:117-119)
    store_block_m_l1 = (
        8
        if block_m <= 16
        else 16
        if block_m <= 64
        else 32
        if block_m <= 192
        else (24 if _is_mma_with_sf(mma_kind) else 40)
    )
    store_block_m_l2 = (
        8 if (block_m == 240 and _is_mma_with_sf(mma_kind)) else store_block_m_l1
    )
    block_k = _idiv(128, _num_mma_elem_bytes(mma_kind))

    _host_assert(
        block_m in CANDIDATE_BLOCK_MS,
        f"block_m={block_m} is not a candidate {CANDIDATE_BLOCK_MS}",
    )
    return 2, block_m, store_block_m_l1, store_block_m_l2, block_k, 2 * 128


def get_block_m_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    mma_type: str = "fp8xfp4",
) -> int:
    """Bit-exact port of ``deep_gemm.get_block_m_for_mega_moe`` (``API:37-46``).

    ``num_tokens`` is the token count of the CALLING rank (``y.size(0)``); ``num_max_tokens_per_rank`` is ignored.
    """
    _host_assert(num_tokens >= 0, "num_tokens must be >= 0")
    kind = parse_mma_kind(mma_type)
    return get_block_config_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, kind
    )[1]


def get_block_m_regimes(
    num_ranks: int, num_experts: int, num_topk: int, max_tokens: int = 16384
) -> List[Tuple[int, int, int]]:
    """Maximal M intervals ``(first_m, last_m, block_m)`` with constant BLOCK_M for ``M in [1, max_tokens]``
    (the selection is NOT monotonic at large M)."""
    segs: List[Tuple[int, int, int]] = []
    prev, start = None, 1
    for m in range(1, max_tokens + 1):
        bm = get_block_m_for_mega_moe(num_ranks, num_experts, 0, m, num_topk)
        if bm != prev:
            if prev is not None:
                segs.append((start, m - 1, prev))
            start, prev = m, bm
    segs.append((start, max_tokens, prev))
    return segs


def get_block_m_variants(
    num_ranks: int, num_experts: int, num_topk: int, max_tokens: int
) -> Tuple[int, ...]:
    """Sorted distinct BLOCK_M values a rank can select for ``M in [0, max_tokens]`` (variants to precompile)."""
    seen = {get_block_m_for_mega_moe(num_ranks, num_experts, 0, 0, num_topk)}
    seen.update(
        bm
        for _, _, bm in get_block_m_regimes(
            num_ranks, num_experts, num_topk, max(1, max_tokens)
        )
    )
    return tuple(sorted(seen))


# ===================================================================================================================
# Ring sizing (scheduler/mega_moe.cuh:15-80, apis/mega_moe.hpp:123-155)
# ===================================================================================================================
def get_num_l1_warmup_waves(
    num_total_m_blocks: int,
    num_clusters: int,
    num_l1_n_clusters: int,
    num_l2_n_clusters: int,
) -> int:
    """``sched::get_num_l1_warmup_waves`` (SCH0:18-45)."""
    # The first L2 wave may touch multiple M blocks; all their L1 N tasks must be issued first.
    first_l2_wave_m_blocks = cdiv(num_clusters, num_l2_n_clusters)
    warmup_clusters_for_first_l2_wave = cdiv(
        first_l2_wave_m_blocks * num_l1_n_clusters, num_clusters
    )
    interleave_diff_per_m_block = (
        num_l1_n_clusters - num_l2_n_clusters
        if num_l1_n_clusters > num_l2_n_clusters
        else 0
    )
    warmup_waves_for_interleave = (
        cdiv(
            num_l1_n_clusters + (num_total_m_blocks - 1) * interleave_diff_per_m_block,
            num_clusters,
        )
        + 1
    )
    return max(warmup_clusters_for_first_l2_wave, warmup_waves_for_interleave)


def get_num_max_live_pool_blocks(
    num_total_m_blocks: int, num_sms: int, hidden: int, intermediate_hidden: int
) -> int:
    """``sched::get_num_max_live_pool_blocks`` (SCH0:48-80): upper bound of simultaneously live pool blocks (ring size)."""
    block_n, cluster = 128, 2
    _host_assert(
        (intermediate_hidden * 2) % (cluster * block_n) == 0,
        "2*intermediate_hidden must be a multiple of 256",
    )
    _host_assert(hidden % (cluster * block_n) == 0, "hidden must be a multiple of 256")
    num_clusters = num_sms // cluster
    num_l1_n_clusters = intermediate_hidden * 2 // (cluster * block_n)
    num_l2_n_clusters = hidden // (cluster * block_n)
    num_l1_clusters = num_total_m_blocks * num_l1_n_clusters
    num_l1_waves = cdiv(num_l1_clusters, num_clusters)
    min_warmup_waves = get_num_l1_warmup_waves(
        num_total_m_blocks, num_clusters, num_l1_n_clusters, num_l2_n_clusters
    )
    num_l1_warmup_waves = min(min_warmup_waves, num_l1_waves)
    num_l1_warmup_clusters = min(num_l1_warmup_waves * num_clusters, num_l1_clusters)
    live_after_warmup = cdiv(num_l1_warmup_clusters, num_l1_n_clusters)
    # Conservative closed-form bound for the L2/L1 alternating tail (L1 advances faster in M-block space when l2n > l1n)
    frontier_growth = (
        cdiv(
            num_total_m_blocks * (num_l2_n_clusters - num_l1_n_clusters),
            num_l2_n_clusters,
        )
        if num_l2_n_clusters > num_l1_n_clusters
        else 0
    )
    wave_margin = cdiv(num_clusters, min(num_l1_n_clusters, num_l2_n_clusters))
    return min(num_total_m_blocks, live_after_warmup + frontier_growth + wave_margin)


def get_num_max_pool_tokens(
    num_ranks: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    """``layout::get_num_max_pool_tokens`` (LAY:20-28): non-ring pool span of the token-source metadata."""
    num_max_recv_tokens = num_ranks * num_max_tokens_per_rank
    num_max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv_tokens * num_max_experts_per_token
        + num_experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1),
        TOKEN_ALIGNMENT,
    )


def get_num_sf_ring_tokens_for_block_m(num_ring_tokens: int, block_m: int) -> int:
    """``layout::get_num_sf_ring_tokens`` (LAY:31-34)."""
    return (num_ring_tokens // block_m) * align(block_m, 128)


def get_num_max_shared_sf_tokens(num_max_tokens_per_rank: int) -> int:
    """``layout::get_num_max_shared_sf_tokens`` (LAY:37-40): ``ceil(T / 8) * 128``."""
    return cdiv(num_max_tokens_per_rank, MIN_CANDIDATE_BLOCK_M) * 128


def get_num_ring_tokens(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_sms: int,
) -> int:
    """Ring capacity in tokens: worst-case live pool blocks over ALL candidate BLOCK_M (``API:123-141``)."""
    num_experts_per_rank = num_experts // num_ranks
    num_active_topk = min(num_topk, num_experts_per_rank)
    num_max_routed_tokens = num_max_tokens_per_rank * num_ranks * num_active_topk
    num_ring_tokens = 0
    for block_m in CANDIDATE_BLOCK_MS:
        num_pool_blocks = cdiv(num_max_routed_tokens, block_m) + num_experts_per_rank
        live = get_num_max_live_pool_blocks(
            num_pool_blocks, num_sms, hidden, intermediate_hidden
        )
        num_ring_tokens = max(num_ring_tokens, live * block_m)
    return align(num_ring_tokens, TOKEN_ALIGNMENT)


def get_ring_sizes(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_sms: int,
    with_sf: bool = True,
) -> Tuple[int, int]:
    """``(num_ring_tokens, num_sf_ring_tokens)`` of the symmetric buffer; ``num_max_tokens_per_rank`` must be aligned to 1920."""
    num_ring_tokens = get_num_ring_tokens(
        num_ranks,
        num_experts,
        num_max_tokens_per_rank,
        num_topk,
        hidden,
        intermediate_hidden,
        num_sms,
    )
    num_sf_ring_tokens = 0
    if with_sf:
        num_sf_ring_tokens = max(
            get_num_sf_ring_tokens_for_block_m(num_ring_tokens, bm)
            for bm in CANDIDATE_BLOCK_MS
        )
    return num_ring_tokens, num_sf_ring_tokens


# ===================================================================================================================
# C++ struct layout helper (alignment rules of SharedStorage / MegaMoESignals)
# ===================================================================================================================
def _struct_layout(
    members: Sequence[Tuple[str, int, int]], struct_align: int = 1
) -> Tuple[Dict[str, int], int, int]:
    """``members = [(name, size_bytes, member_alignment)]`` -> ``(offsets, exact_end, sizeof)`` (sizeof rounded to the struct alignment)."""
    off, offs = 0, {}
    for name, size, al in members:
        off = align(off, al)
        offs[name] = off
        off += size
    return offs, off, align(off, struct_align)


# ----- MegaMoESignals (layout/mega_moe.cuh:54-90): workspace head, 21,140,864 B -----------------------------------
_NUM_MAX_GRID_SYNC_COUNTERS = 4
_NUM_MAX_EXPERTS = 2048
_NUM_MAX_EXPERTS_PER_RANK = 512
_NUM_MAX_RING_BLOCKS = 1 << 20
_NUM_MAX_SHARED_L2_BLOCKS = 1 << 15
_D = NUM_DEVICE_LOCALITY_DOMAINS

MEGA_MOE_SIGNALS_MEMBERS: Tuple[Tuple[str, int, int], ...] = (
    ("grid_sync_count", 4 * _NUM_MAX_GRID_SYNC_COUNTERS, 4),  # u32[4]; idx 0 = dispatch grid syncs, 1 = epilogue grid sync
    ("nvl_barrier_counter", 4, 4),  # u32
    ("nvl_barrier_signals", 4 * 2, 4),  # s32[2]
    ("l1_task_count", 4 * _D, 4),  # u32[2] (one per locality domain)
    ("l2_task_count", 4 * _D, 4),
    ("shared_l1_task_count", 4 * _D, 4),
    ("shared_l2_task_count", 4 * _D, 4),
    ("combine_ready_grid_idx", 8 * MAX_RANKS, 128),  # u64[72], alignas(128)
    ("peer_grid_idx", 8 * MAX_RANKS, 128),  # u64[72], alignas(128)
    ("expert_send_count", 8 * _NUM_MAX_EXPERTS, 128),  # u64[2048], alignas(128)
    ("expert_recv_count", 8 * _NUM_MAX_EXPERTS, 8),  # u64[2048]: [src_rank][local_expert]
    ("expert_recv_count_sum", 8 * _NUM_MAX_EXPERTS_PER_RANK, 8),  # u64[512]
    ("l1_full_count", 4 * _NUM_MAX_RING_BLOCKS, 4),  # u32[1<<20]
    ("l1_empty_count", 4 * _NUM_MAX_RING_BLOCKS, 4),
    ("l2_full_mask", 8 * _NUM_MAX_RING_BLOCKS, 8),  # u64[1<<20]
    ("l2_empty_count", 4 * _NUM_MAX_RING_BLOCKS, 4),
    ("shared_l2_full_count", 4 * _NUM_MAX_SHARED_L2_BLOCKS, 4),  # u32[1<<15]
)  # fmt: skip

#: byte offset of every ``MegaMoESignals`` member, ``SIGNALS_BYTES`` = sizeof(MegaMoESignals) = 21,140,864
SIGNALS_OFFSETS, _SIGNALS_END, SIGNALS_BYTES = _struct_layout(
    MEGA_MOE_SIGNALS_MEMBERS, 128
)
SIGNALS_SIZES: Dict[str, int] = {
    name: size for name, size, _ in MEGA_MOE_SIGNALS_MEMBERS
}
assert (
    SIGNALS_OFFSETS["combine_ready_grid_idx"] == 128 and SIGNALS_BYTES == 21140864
)  # DG_STATIC_ASSERT of the header


# ===================================================================================================================
# Symmetric buffer layout (layout::MegaMoEBuffer, apis/mega_moe.hpp:113-195)
# ===================================================================================================================
@dataclass(frozen=True)
class ViewSpec:
    """A tensor view into the symmetric buffer (``slice_symm_buffer_from_layout``)."""

    offset: int  # bytes from the buffer base
    shape: Tuple[int, ...]
    dtype: str  # 'float8_e4m3fn' | 'bfloat16' | 'int32' | 'int64' | 'float32'
    strides: Optional[Tuple[int, ...]] = None  # in elements; None = contiguous


@dataclass(frozen=True)
class SymmBufferLayout:
    """Byte layout of the symmetric buffer (identical to DeepGEMM's ``SymmBufferLayoutInfo`` + ``MegaMoEBuffer``).

    ``to_int_list()`` returns exactly the 21 integers of ``SymmBufferLayoutInfo::to_int_list()``; ``regions`` additionally
    lists every region the kernel addresses (name -> (offset, nbytes)).
    """

    num_bytes: int
    input_token_base: int
    input_sf_base: int
    input_topk_idx_base: int
    input_topk_weights_base: int
    shared_l1_sf_base: int
    shared_l2_token_base: int
    shared_l2_sf_base: int
    l1_token_base: int
    l1_sf_base: int
    l2_token_base: int
    l2_sf_base: int
    with_sf: bool
    num_max_tokens_per_rank: int
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_shared_experts: int
    shared_intermediate_hidden: int
    num_ring_tokens: int
    num_sf_ring_tokens: int
    # --- extras (not part of DeepGEMM's int list)
    num_ranks: int
    num_experts: int
    num_sms: int
    workspace_bytes: int  # sizeof(MegaMoESignals) + src_token_topk_idx + token_src_metadata, aligned to 16
    num_max_pool_tokens: int
    num_max_shared_sf_tokens: int
    l1_topk_weights_base: int
    combine_token_base: int
    num_combine_slots: int  # num_topk + (S > 0)
    regions: Dict[str, Tuple[int, int]]
    views: Dict[str, ViewSpec]

    def to_int_list(self) -> List[int]:
        return [
            self.num_bytes, self.input_token_base, self.input_sf_base, self.input_topk_idx_base, self.input_topk_weights_base,
            self.shared_l1_sf_base, self.shared_l2_token_base, self.shared_l2_sf_base,
            self.l1_token_base, self.l1_sf_base, self.l2_token_base, self.l2_sf_base,
            int(self.with_sf), self.num_max_tokens_per_rank, self.num_topk, self.hidden, self.intermediate_hidden,
            self.num_shared_experts, self.shared_intermediate_hidden, self.num_ring_tokens, self.num_sf_ring_tokens,
        ]  # fmt: skip

    # workspace-internal byte offsets (from the buffer base == the MegaMoESignals base)
    @property
    def src_token_topk_idx_offset(self) -> int:
        """``[local expert][source rank][token]`` u32 table (``Workspace::get_src_token_topk_idx_ptr``)."""
        return SIGNALS_BYTES

    @property
    def token_src_metadata_offset(self) -> int:
        """``TokenSrcMetadata`` (12 B ``{rank, token, topk}``) per pool token (``get_token_src_metadata_ptr``)."""
        return SIGNALS_BYTES + self.num_experts * self.num_max_tokens_per_rank * 4


def build_symm_buffer_layout(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
    num_shared_experts: int = 0,
    num_sms: int = 212,
) -> SymmBufferLayout:
    """Port of ``build_symm_buffer_layout`` / ``MegaMoEBuffer`` (``API:113-195``, ``LAY:345-457``).

    ``num_max_tokens_per_rank`` must already be aligned to :func:`get_token_alignment_for_mega_moe` (DeepGEMM's
    ``SymmBuffer.__init__`` aligns it in Python before calling C++).  ``num_sms`` is the device SM count.
    """
    _host_assert(
        num_experts % num_ranks == 0, "num_experts must be divisible by num_ranks"
    )
    _host_assert(
        activation in ("swiglu", "swiglustep", "situ"),
        f"unsupported activation {activation!r}",
    )
    _host_assert(num_shared_experts >= 0, "num_shared_experts must be >= 0")
    T = num_max_tokens_per_rank
    epr = num_experts // num_ranks
    shared_intermediate_hidden = intermediate_hidden * num_shared_experts

    kind = parse_mma_kind(mma_type)
    with_sf = _is_mma_with_sf(kind)
    num_ring_tokens, num_sf_ring_tokens = get_ring_sizes(
        num_ranks, num_experts, T, num_topk, hidden, intermediate_hidden, num_sms, with_sf
    )  # fmt: skip

    # Workspace asserts (LAY:119-125)
    _host_assert(0 < num_ranks <= MAX_RANKS, "num_ranks out of range")
    _host_assert(num_experts <= _NUM_MAX_EXPERTS, "too many experts")
    _host_assert(epr <= _NUM_MAX_EXPERTS_PER_RANK, "too many experts per rank")
    _host_assert(
        num_ring_tokens <= _NUM_MAX_RING_BLOCKS * MIN_CANDIDATE_BLOCK_M,
        "ring too large",
    )
    _host_assert(
        cdiv(T, MIN_CANDIDATE_BLOCK_M) <= _NUM_MAX_SHARED_L2_BLOCKS,
        "too many shared L2 pool blocks",
    )
    num_max_pool_tokens = get_num_max_pool_tokens(num_ranks, T, num_topk, epr)
    workspace_bytes = align(
        SIGNALS_BYTES + num_experts * T * 4 + num_max_pool_tokens * 12, 16
    )

    if with_sf:  # SF buffer requirements (API:166-170)
        _host_assert(
            hidden % 128 == 0 and intermediate_hidden % 128 == 0,
            "hidden and intermediate_hidden must be multiples of 128",
        )
        _host_assert(
            shared_intermediate_hidden % 128 == 0,
            "shared intermediate hidden must be a multiple of 128",
        )
        _host_assert(
            num_sf_ring_tokens % 4 == 0, "num_sf_ring_tokens must be a multiple of 4"
        )

    # Layouts (LAY:388-398): bytes per row
    elem = 1 if with_sf else 2
    input_token_row = hidden * elem
    bf16_token_row = hidden * 2
    intermediate_token_row = intermediate_hidden * elem
    shared_intermediate_token_row = shared_intermediate_hidden * elem
    input_sf_row = hidden // 32 if with_sf else 0
    intermediate_sf_row = intermediate_hidden // 32 if with_sf else 0
    shared_intermediate_sf_row = shared_intermediate_hidden // 32 if with_sf else 0
    for row in (
        input_token_row,
        bf16_token_row,
        intermediate_token_row,
        shared_intermediate_token_row,
    ):
        _host_assert(row % 16 == 0, "token rows must be 16 B aligned (TMA)")
    nsf = get_num_max_shared_sf_tokens(T) if with_sf else 0
    S = num_shared_experts

    regions: Dict[str, Tuple[int, int]] = {}
    cur = 0

    def add(name: str, nbytes: int) -> int:
        nonlocal cur
        base = cur
        regions[name] = (base, nbytes)
        cur += nbytes
        return base

    add("workspace", workspace_bytes)
    input_token_base = add("input_token", T * input_token_row)
    input_sf_base = add("input_sf", T * input_sf_row)
    input_topk_idx_base = add("input_topk_idx", T * num_topk * 8)
    input_topk_weights_base = add("input_topk_weights", T * num_topk * 4)
    shared_l1_sf_base = add("shared_l1_sf", (nsf if S > 0 else 0) * input_sf_row)
    shared_l2_token_base = add(
        "shared_l2_token", (T if S > 0 else 0) * shared_intermediate_token_row
    )
    shared_l2_sf_base = add(
        "shared_l2_sf", (nsf if S > 0 else 0) * shared_intermediate_sf_row
    )
    l1_token_base = add("l1_token", num_ring_tokens * input_token_row)
    l1_sf_base = add("l1_sf", num_sf_ring_tokens * input_sf_row)
    l1_topk_weights_base = add("l1_topk_weights", num_ring_tokens * 4)
    l2_token_base = add("l2_token", num_ring_tokens * intermediate_token_row)
    l2_sf_base = add("l2_sf", num_sf_ring_tokens * intermediate_sf_row)
    num_combine_slots = num_topk + (1 if S > 0 else 0)
    combine_token_base = add("combine_token", num_combine_slots * T * bf16_token_row)
    # workspace-internal sub-regions
    regions["signals"] = (0, SIGNALS_BYTES)
    regions["src_token_topk_idx"] = (SIGNALS_BYTES, num_experts * T * 4)
    regions["token_src_metadata"] = (
        SIGNALS_BYTES + num_experts * T * 4,
        num_max_pool_tokens * 12,
    )

    act_dtype = "float8_e4m3fn" if with_sf else "bfloat16"
    sf_cols, isf_cols, ssf_cols = (
        hidden // 128,
        intermediate_hidden // 128,
        shared_intermediate_hidden // 128,
    )
    views: Dict[str, ViewSpec] = {
        "x": ViewSpec(input_token_base, (T, hidden), act_dtype),
        "topk_idx": ViewSpec(input_topk_idx_base, (T, num_topk), "int64"),
        "topk_weights": ViewSpec(input_topk_weights_base, (T, num_topk), "float32"),
        "l1_acts": ViewSpec(l1_token_base, (num_ring_tokens, hidden), act_dtype),
        "l2_acts": ViewSpec(l2_token_base, (num_ring_tokens, intermediate_hidden), act_dtype),
    }  # fmt: skip
    views["shared_l1_acts"] = views[
        "x"
    ]  # shared L1 tokens reuse the input token buffer
    if with_sf:
        views["x_sf"] = ViewSpec(input_sf_base, (T, sf_cols), "int32")
        views["l1_acts_sf"] = ViewSpec(
            l1_sf_base, (num_sf_ring_tokens, sf_cols), "int32", (1, num_sf_ring_tokens)
        )
        views["l2_acts_sf"] = ViewSpec(
            l2_sf_base, (num_sf_ring_tokens, isf_cols), "int32", (1, num_sf_ring_tokens)
        )
        if S > 0:
            views["shared_l1_acts_sf"] = ViewSpec(
                shared_l1_sf_base, (nsf, sf_cols), "int32", (1, nsf)
            )
            views["shared_l2_acts_sf"] = ViewSpec(
                shared_l2_sf_base, (nsf, ssf_cols), "int32", (1, nsf)
            )
    if S > 0:
        views["shared_l2_acts"] = ViewSpec(
            shared_l2_token_base, (T, shared_intermediate_hidden), act_dtype
        )

    return SymmBufferLayout(
        num_bytes=cur,
        input_token_base=input_token_base, input_sf_base=input_sf_base,
        input_topk_idx_base=input_topk_idx_base, input_topk_weights_base=input_topk_weights_base,
        shared_l1_sf_base=shared_l1_sf_base, shared_l2_token_base=shared_l2_token_base, shared_l2_sf_base=shared_l2_sf_base,
        l1_token_base=l1_token_base, l1_sf_base=l1_sf_base, l2_token_base=l2_token_base, l2_sf_base=l2_sf_base,
        with_sf=with_sf, num_max_tokens_per_rank=T, num_topk=num_topk, hidden=hidden, intermediate_hidden=intermediate_hidden,
        num_shared_experts=S, shared_intermediate_hidden=shared_intermediate_hidden,
        num_ring_tokens=num_ring_tokens, num_sf_ring_tokens=num_sf_ring_tokens,
        num_ranks=num_ranks, num_experts=num_experts, num_sms=num_sms,
        workspace_bytes=workspace_bytes, num_max_pool_tokens=num_max_pool_tokens, num_max_shared_sf_tokens=nsf,
        l1_topk_weights_base=l1_topk_weights_base, combine_token_base=combine_token_base, num_combine_slots=num_combine_slots,
        regions=regions, views=views,
    )  # fmt: skip


def get_symm_buffer_size_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
    num_shared_experts: int = 0,
    num_sms: int = 212,
) -> Tuple[int, List[int]]:
    """Mirror of ``deep_gemm._C.get_symm_buffer_size_for_mega_moe``'s torch op: ``(num_bytes, layout_info int list)``."""
    layout = build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden,
        mma_type, activation, num_shared_experts, num_sms,
    )  # fmt: skip
    return layout.num_bytes, layout.to_int_list()


# ===================================================================================================================
# Pipeline sizing: SM100 sizing function (HEUR:133-199) and the SM100-sized config (HEUR:201-267)
# ===================================================================================================================
def get_pipeline_config_for_mega_moe(
    smem_capacity: int,
    num_experts: int,
    hidden: int,
    block_m: int,
    block_n: int,
    block_k: int,
    num_bytes_per_pull: int,
    store_block_m_l1: int,
    store_block_m_l2: int,
    sf_block_m: int,
    sf_block_n: int,
    gran_k: int,
    num_dispatch_warps: int,
    num_epilogue_warps: int,
    mma_kind: str,
) -> Tuple[int, int]:
    """``get_pipeline_config_for_mega_moe`` (HEUR:133-199): ``(num_stages, smem_size)`` of the SM100 kernel's sizing."""
    k_alignment = SMEM_ALIGNMENT
    num_epilogue_stages = 2
    num_tma_store_stages = 2
    elem = _num_mma_elem_bytes(mma_kind)
    load_block_m = block_m // 2  # always multicast on A

    # Dispatch region
    smem_expert_count_size = align(num_experts * 4, k_alignment)
    smem_send_buffers_size = align(
        num_bytes_per_pull * num_dispatch_warps, k_alignment
    )  # Buffer(Data(pull), warps, 1).get_num_bytes()
    smem_dispatch_size = smem_expert_count_size + smem_send_buffers_size

    # C/D output region: max of L1 output staging and L2 BF16 staging
    num_epilogue_warpgroups = num_epilogue_warps // 4
    smem_cd_l1 = (
        num_epilogue_warpgroups
        * store_block_m_l1
        * (block_n // 2)
        * num_tma_store_stages
        * elem
    )
    smem_cd_l2 = (
        num_epilogue_warpgroups * store_block_m_l2 * block_n * 2
    )  # sizeof(nv_bfloat16)
    smem_cd = align(max(smem_cd_l1, smem_cd_l2), k_alignment)

    # Schedule task payloads: kNumScheduleStages * sizeof(TaskInfo) (32 B)
    smem_task_info = 2 * 32

    # Barriers (stage-independent): dispatch + tmem full/empty + combine (2 per epilogue warp) + schedule full/empty
    smem_barriers = (
        num_dispatch_warps + num_epilogue_stages * 2 + num_epilogue_warps * 2 + 2 * 2
    ) * 8

    # Amax warp-pair reduction buffer; tensor memory pointer
    smem_amax_reduction = (
        store_block_m_l1 * num_epilogue_warps * 4 if _is_mma_with_sf(mma_kind) else 0
    )
    smem_tmem_ptr = 4

    # SF is aligned to UTCCP 128-element granularity
    smem_sfa_per_stage = (
        sf_block_m * (block_k // gran_k) if _is_mma_with_sf(mma_kind) else 0
    )
    smem_sfb_per_stage = (
        sf_block_n * (block_k // gran_k) if _is_mma_with_sf(mma_kind) else 0
    )

    # Per-stage: A tile + B tile + optional SF tiles + full/empty barriers
    smem_a_size_per_stage = load_block_m * block_k * elem
    smem_b_size_per_stage = block_n * block_k * elem
    _host_assert(
        smem_a_size_per_stage % k_alignment == 0,
        "smem_a stage must be a multiple of 1024 B",
    )
    _host_assert(
        smem_b_size_per_stage % k_alignment == 0,
        "smem_b stage must be a multiple of 1024 B",
    )
    smem_size_per_stage = (
        smem_a_size_per_stage
        + smem_b_size_per_stage
        + smem_sfa_per_stage
        + smem_sfb_per_stage
        + 2 * 8
    )

    smem_fixed = (
        smem_dispatch_size
        + smem_cd
        + smem_amax_reduction
        + smem_barriers
        + smem_task_info
        + smem_tmem_ptr
    )
    num_stages = _idiv(smem_capacity - smem_fixed, smem_size_per_stage)
    _host_assert(num_stages >= 2, "fewer than 2 pipeline stages fit")
    return num_stages, smem_fixed + num_stages * smem_size_per_stage


@dataclass(frozen=True)
class MegaMoEConfig:
    """``deep_gemm::MegaMoEConfig`` (HEUR:23-64): the SM100-SIZED config (``num_stages`` / ``smem_size`` are the SM100
    kernel's own and are NOT what the SM107 launcher ships; see :func:`get_sm107_pipeline_config`)."""

    block_m: int
    block_n: int
    block_k: int
    load_block_m: int
    load_block_n: int
    store_block_m_l1: int
    store_block_m_l2: int
    sf_block_m: int
    sf_block_n: int
    num_ring_tokens: int
    num_sf_ring_tokens: int
    swizzle_acts_mode: int
    swizzle_weights_mode: int
    num_stages: int
    smem_size: int
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    num_bytes_per_pull: int

    def __str__(self) -> str:  # same text as C++ operator<<
        return (
            f"MegaMoEConfig(block_m={self.block_m}, block_n={self.block_n}, block_k={self.block_k}, "
            f"load_block_m={self.load_block_m}, load_block_n={self.load_block_n}, "
            f"store_block_m_l1={self.store_block_m_l1}, store_block_m_l2={self.store_block_m_l2}, "
            f"sf_block_m={self.sf_block_m}, sf_block_n={self.sf_block_n}, "
            f"num_ring_tokens={self.num_ring_tokens}, num_sf_ring_tokens={self.num_sf_ring_tokens}, "
            f"swizzle_acts_mode={self.swizzle_acts_mode}, swizzle_weights_mode={self.swizzle_weights_mode}, "
            f"num_stages={self.num_stages}, smem_size={self.smem_size}, "
            f"num_dispatch_threads={self.num_dispatch_threads}, num_non_epilogue_threads={self.num_non_epilogue_threads}, "
            f"num_epilogue_threads={self.num_epilogue_threads}, num_bytes_per_pull={self.num_bytes_per_pull})"
        )


def get_mega_moe_config(
    num_ranks: int,
    num_experts: int,
    num_experts_per_rank: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_ring_tokens: int,
    num_sf_ring_tokens: int,
    mma_kind: str = MMA_MXFP8FP4,
    force_block_m: int = 0,
) -> MegaMoEConfig:
    """``get_mega_moe_config`` (HEUR:201-267)."""
    _, block_m, store_block_m_l1, store_block_m_l2, block_k, num_epilogue_threads = get_block_config_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, num_tokens, mma_kind, force_block_m
    )  # fmt: skip
    _host_assert(
        num_ring_tokens % block_m == 0, "num_ring_tokens must be a multiple of block_m"
    )
    block_n = 128
    load_block_m = block_m // 2
    load_block_n = block_n
    sf_block_m, sf_block_n = (
        (align(block_m, 128), align(block_n, 128))
        if _is_mma_with_sf(mma_kind)
        else (0, 0)
    )
    swizzle_acts_mode = 128  # FP8 activations, FP8 weights and FP4 weights unpacked to 8 bit in smem all use 128B swizzle
    swizzle_weights_mode = 128
    gran_k = 32
    num_dispatch_threads = 128
    num_non_epilogue_threads = 128

    # Pull: divide token bytes by 2 until <= num_max_pull_bytes
    num_max_pull_bytes = MAX_PULL_BYTES_WITH_SF if _is_mma_with_sf(mma_kind) else 4096
    num_bytes_per_pull = hidden * _num_mma_elem_bytes(mma_kind)
    while num_bytes_per_pull > num_max_pull_bytes:
        _host_assert(num_bytes_per_pull % 2 == 0, "num_bytes_per_pull must stay even")
        num_bytes_per_pull //= 2

    num_stages, smem_size = get_pipeline_config_for_mega_moe(
        SM100_SMEM_CAPACITY, num_experts, hidden, block_m, block_n, block_k, num_bytes_per_pull,
        store_block_m_l1, store_block_m_l2, sf_block_m, sf_block_n, gran_k,
        num_dispatch_threads // 32, num_epilogue_threads // 32, mma_kind,
    )  # fmt: skip
    return MegaMoEConfig(
        block_m, block_n, block_k, load_block_m, load_block_n, store_block_m_l1, store_block_m_l2,
        sf_block_m, sf_block_n, num_ring_tokens, num_sf_ring_tokens if _is_mma_with_sf(mma_kind) else 0,
        swizzle_acts_mode, swizzle_weights_mode, num_stages, smem_size,
        num_dispatch_threads, num_non_epilogue_threads, num_epilogue_threads, num_bytes_per_pull,
    )  # fmt: skip


def get_mega_moe_print_key(
    num_ranks: int,
    num_experts: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
) -> str:
    """The key text of the ``DG_PRINT_CONFIGS`` ``MegaMoEConfig(...)`` line (HEUR:257-259)."""
    return (
        f"MegaMoEConfig(num_ranks={num_ranks}, num_experts={num_experts}, hidden={hidden}, "
        f"intermediate_hidden={intermediate_hidden}, num_max_tokens_per_rank={num_max_tokens_per_rank}, "
        f"num_tokens={num_tokens}, num_topk={num_topk})"
    )


# ===================================================================================================================
# SM107 launcher: shipped stages / dynamic smem (IMPL:100-144)
# ===================================================================================================================
def get_sm107_store_block_m(block_m: int) -> int:
    """Symmetric store block of the SM107 fork (IMPL:100-102): 8 (BM<=16) / 16 (<=64) / 32 (<=192) / 40."""
    return 8 if block_m <= 16 else 16 if block_m <= 64 else 32 if block_m <= 192 else 40


@dataclass(frozen=True)
class Sm107PipelineConfig:
    num_stages: int
    smem_size: int  # requested dynamic shared memory
    smem_fixed: int  # `fixed107` (absorbs n1 * 8192 B of unused unpacked-weight space: the quirk to reproduce)
    smem_per_stage: int
    n1: int  # SM100-sizing stage count at 232448 (internal)
    n2: int  # SM100-sizing stage count at 232448 + 2**18 (internal)
    stages_if_fixed_clean: (
        int  # what a clean derivation would give (for documentation / tests)
    )


def get_sm107_pipeline_config(
    num_experts: int,
    hidden: int,
    block_m: int,
    block_n: int = 128,
    block_k: int = 128,
    num_bytes_per_pull: int = 0,
    sf_block_m: int = 0,
    sf_block_n: int = 128,
    num_dispatch_threads: int = 128,
    num_epilogue_threads: int = 256,
    smem_capacity: int = SM100_SMEM_CAPACITY,
    max_stages: int = 0,
    umma_k: int = UMMA_K_PACKED,
    packed_fp4_weights: bool = True,
    dispatch_pipe: int = 3,
) -> Sm107PipelineConfig:
    """SM107 re-derivation of ``(num_stages, smem_size)`` (IMPL:108-144; env knobs become arguments).

    ``smem_capacity`` == ``DG_MEGA_MOE_SM107_SMEM`` (default 232448), ``max_stages`` == ``DG_MEGA_MOE_SM107_STAGES``.
    The first two evaluations always use the SM100 capacity (and its ``+ 2**18``) to extract ``fixed`` / ``per-stage``.
    """
    store_block_m = get_sm107_store_block_m(block_m)
    if num_bytes_per_pull == 0:
        num_bytes_per_pull = hidden
        while num_bytes_per_pull > MAX_PULL_BYTES_WITH_SF:
            num_bytes_per_pull //= 2
    if sf_block_m == 0:
        sf_block_m = align(block_m, 128)

    def get_pipeline(capacity: int) -> Tuple[int, int]:
        return get_pipeline_config_for_mega_moe(
            capacity, num_experts, hidden, block_m, block_n, block_k, num_bytes_per_pull, store_block_m, store_block_m,
            sf_block_m, sf_block_n, 32, num_dispatch_threads // 32, num_epilogue_threads // 32, MMA_MXFP8FP4,
        )  # fmt: skip

    # Derive the fixed and per-stage sizes from two (host-only) evaluations of the SM100 sizing
    n1, s1 = get_pipeline(SM100_SMEM_CAPACITY)
    n2, s2 = get_pipeline(SM100_SMEM_CAPACITY + (1 << 18))
    _host_assert(n2 != n1, "degenerate SM100 sizing")
    # Packed FP4 weights (K=64 UMMA) halve the weight stage (8 KB instead of 16 KB)
    packed_weights = umma_k == 64 and packed_fp4_weights
    smem_per_stage = _idiv(s2 - s1, n2 - n1) - (
        block_n * block_k // 2 if packed_weights else 0
    )
    # Double-buffered dispatch pull (pipe 2) adds one pull buffer + barrier per dispatch warp
    _host_assert(1 <= dispatch_pipe <= 3, "dispatch_pipe must be in [1, 3]")
    pull_buffer_bytes = (num_dispatch_threads // 32) * num_bytes_per_pull
    smem_fixed = (
        s1 - n1 * smem_per_stage
        + ((pull_buffer_bytes + (num_dispatch_threads // 32) * 8) if dispatch_pipe == 2 else 0)
        - ((pull_buffer_bytes - 1024) if dispatch_pipe == 3 else 0)
    )  # fmt: skip
    num_stages = min(_idiv(smem_capacity - smem_fixed, smem_per_stage), 32)
    if max_stages > 0:
        num_stages = min(num_stages, max_stages)
    _host_assert(num_stages >= 2, "fewer than 2 pipeline stages fit")
    # Documentation value only: stages if the quirk (n1 stages of unpacked weights inside `fixed107`) were removed
    clean_fixed = smem_fixed - (n1 * (block_n * block_k // 2) if packed_weights else 0)
    clean = min(_idiv(smem_capacity - clean_fixed, smem_per_stage), 32)
    return Sm107PipelineConfig(
        num_stages,
        smem_fixed + num_stages * smem_per_stage,
        smem_fixed,
        smem_per_stage,
        n1,
        n2,
        clean,
    )


# ===================================================================================================================
# SharedStorage layout (K:250-275), TMEM (K:283-290), registers (K:385-388), combine chunking (K:1652-1665)
# ===================================================================================================================
SMEM_MEMBER_ORDER: Tuple[str, ...] = (
    "expert_token_count", "dispatch_send_buffer", "smem_d", "smem_a", "smem_b", "smem_sfa", "smem_sfb", "amax_reduction",
    "task_infos", "dispatch_barriers", "full_barriers", "empty_barriers", "tmem_full_barriers", "tmem_empty_barriers",
    "combine_barriers", "task_info_full_barriers", "task_info_empty_barriers", "tmem_ptr_in_smem",
)  # fmt: skip


def get_sm107_smem_layout(
    block_m: int,
    num_stages: int,
    num_experts: int,
    *,
    sf_block_n: int = 128,
    block_k: int = 128,
    num_dispatch_warps: int = 4,
    num_epilogue_warps: int = 8,
    dispatch_pipe: int = 3,
    num_bytes_per_pull: int = 0,
    packed_fp4_weights: bool = True,
) -> Tuple[Dict[str, int], Dict[str, int], int, int]:
    """Byte offsets / sizes of the ``SharedStorage`` members of the SM107 kernel (K:250-275, C++ struct rules).

    Returns ``(offsets, sizes, exact_end, sizeof)``; ``offsets`` additionally carries ``reusable_bytes`` (= offset of
    ``dispatch_barriers``: the combine overlays ``[0, reusable)``), ``end`` and ``sizeof``.
    """
    store_block_m = get_sm107_store_block_m(block_m)
    sf_block_m = align(block_m, 128)
    load_block_m = block_m // 2
    num_epilogue_warpgroups = num_epilogue_warps // 4
    num_pull_slots = 2 if dispatch_pipe == 2 else 1
    if dispatch_pipe == 3:
        dispatch_send_bytes = 1 * 16
    else:
        _host_assert(
            num_bytes_per_pull > 0,
            "num_bytes_per_pull is needed for dispatch modes 1/2",
        )
        dispatch_send_bytes = num_dispatch_warps * num_pull_slots * num_bytes_per_pull
    smem_d_bytes = align(
        max(
            num_epilogue_warpgroups
            * 2
            * store_block_m
            * (128 // 2),  # l1: fp8 [wg][tma stage 2][STORE_BLOCK_M * 64]
            num_epilogue_warpgroups * store_block_m * 128 * 2,
        ),  # l2: bf16 [wg][STORE_BLOCK_M * 128]
        SMEM_ALIGNMENT,
    )  # fmt: skip  (the union is alignas(1024): its size is rounded up)
    members = [
        ("expert_token_count", 4 * num_experts, SMEM_ALIGNMENT),
        ("dispatch_send_buffer", dispatch_send_bytes, SMEM_ALIGNMENT),
        ("smem_d", smem_d_bytes, SMEM_ALIGNMENT),
        ("smem_a", num_stages * load_block_m * block_k, SMEM_ALIGNMENT),
        ("smem_b", num_stages * 128 * block_k // (2 if packed_fp4_weights else 1), SMEM_ALIGNMENT),
        ("smem_sfa", num_stages * sf_block_m * (block_k // 128) * 4, 4),
        ("smem_sfb", num_stages * sf_block_n * (block_k // 128) * 4, 4),
        ("amax_reduction", num_epilogue_warps * (store_block_m // 2) * 8, 8),  # float2[8][STORE_BLOCK_M/2]
        ("task_infos", 2 * 32, 16),  # TaskInfo is alignas(16), 32 B
        ("dispatch_barriers", num_dispatch_warps * num_pull_slots * 8, 8),
        ("full_barriers", num_stages * 8, 8),
        ("empty_barriers", num_stages * 8, 8),
        ("tmem_full_barriers", 2 * 8, 8),
        ("tmem_empty_barriers", 2 * 8, 8),
        ("combine_barriers", num_epilogue_warps * 2 * 8, 8),
        ("task_info_full_barriers", 2 * 8, 8),
        ("task_info_empty_barriers", 2 * 8, 8),
        ("tmem_ptr_in_smem", 4, 4),
    ]  # fmt: skip
    assert tuple(m[0] for m in members) == SMEM_MEMBER_ORDER
    offs, end, sizeof = _struct_layout(members, SMEM_ALIGNMENT)
    sizes = {name: size for name, size, _ in members}
    offs = dict(offs)
    offs["reusable_bytes"] = offs["dispatch_barriers"]
    offs["end"] = end
    offs["sizeof"] = sizeof
    return offs, sizes, end, sizeof


def get_num_aligned_tmem_cols(num_cols: int) -> int:
    """``utils::get_num_aligned_tmem_cols`` (common/utils.cuh:43-51)."""
    _host_assert(num_cols <= 512, "too many tensor memory columns")
    for c in (32, 64, 128, 256):
        if num_cols <= c:
            return c
    return 512


def get_tmem_layout(block_m: int, sf_block_n: int = 128) -> Dict[str, int]:
    """TMEM map (K:283-290): accumulator stage 0/1, token SF ("SFA"), weight SF ("SFB")."""
    sf_block_m = align(block_m, 128)
    accum = block_m * 2  # UMMA_N * kNumEpilogueStages
    sfa = sf_block_m // 32
    sfb = sf_block_n // 32
    used = accum + sfa + sfb
    return {"accum_cols": accum, "sfa_cols": sfa, "sfb_cols": sfb, "used_cols": used,
            "alloc_cols": get_num_aligned_tmem_cols(used), "sfa_col": accum, "sfb_col": accum + sfa}  # fmt: skip


def get_register_split(num_experts_per_rank: int) -> Tuple[int, int, int]:
    """``(dispatch, non-epilogue, epilogue)`` setmaxnreg targets (K:385-388): more epilogue registers iff E/R <= 64."""
    more_epilogue = num_experts_per_rank <= 64
    regs = (48, 40, 208) if more_epilogue else (96, 88, 160)
    assert regs[0] * 128 + regs[1] * 128 + regs[2] * 256 <= 64512
    return regs


def get_num_combine_chunks(
    hidden: int, reusable_bytes: int, num_epilogue_warps: int = 8
) -> int:
    """``kNumChunks`` of the combine stage (K:1652-1662): 1 iff 3 slots x 8 warps x H x 2 B fit the reusable smem
    and H <= 32 * 128, else 2."""
    num_hidden_bytes = hidden * 2
    num_chunk_slots, num_max_registers_for_buffer = 3, 128
    return (
        1
        if (
            num_chunk_slots * num_epilogue_warps * num_hidden_bytes <= reusable_bytes
            and hidden <= 32 * num_max_registers_for_buffer
        )
        else 2
    )


# ===================================================================================================================
# The factory
# ===================================================================================================================
def get_sm107_kernel_config(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    *,
    num_sms: Optional[int] = None,
    activation_clamp: Optional[float] = None,
    activation_alpha: float = 1.0,
    activation_beta: float = 0.0,
    fast_math: bool = True,
    block_m: int = 0,
    per_die: bool = False,
    localized_weights: bool = False,
    localized_shared_weights: bool = False,
    device_num_locality_domains: int = NUM_DEVICE_LOCALITY_DOMAINS,
    smem_capacity: int = SM100_SMEM_CAPACITY,
    max_stages: int = 0,
    trace: bool = False,
    tma_l2_promotion: Optional[int] = None,
) -> Sm107KernelConfig:
    """Derive the compile-time configuration of the SM107 kernel variant used for a call with ``num_tokens`` tokens
    on the calling rank (the port of ``sm107_fp8_fp4_mega_moe``'s host part, ``IMPL:87-144,265-280``).

    ``num_max_tokens_per_rank`` is the user value (aligned up to 1920 here; passing an aligned value is idempotent).
    ``block_m`` > 0 forces the variant (== ``DG_MEGA_MOE_SM107_BLOCK_M``; only {16,32,64,128,192,240} can be
    instantiated).  ``num_sms`` None -> the current CUDA device (212 on Rubin VR200).
    """
    num_sms = resolve_num_sms(num_sms)
    _host_assert(
        num_sms > 0 and num_sms % 2 == 0,
        "num_sms must be positive and even (2-CTA clusters)",
    )
    _host_assert(
        num_ranks >= 1 and num_experts > 0 and num_experts % num_ranks == 0,
        "num_experts must be positive and divisible by num_ranks",
    )
    _host_assert(
        num_topk >= 1 and num_shared_experts >= 0,
        "invalid num_topk / num_shared_experts",
    )
    _host_assert(
        num_topk + (1 if num_shared_experts > 0 else 0) <= 32,
        "top-k + shared must fit in a single warp",
    )
    _host_assert(
        hidden > 0 and intermediate_hidden > 0 and num_max_tokens_per_rank > 0,
        "hidden, intermediate_hidden and num_max_tokens_per_rank must be positive",
    )
    _host_assert(
        hidden % 128 == 0 and intermediate_hidden % 128 == 0,
        "hidden and intermediate_hidden must be multiples of 128",
    )
    clamp = math.inf if activation_clamp is None else float(activation_clamp)
    _host_assert(clamp >= 0, "activation_clamp must be >= 0")
    _host_assert(
        math.isfinite(activation_alpha) and math.isfinite(activation_beta),
        "alpha/beta must be finite",
    )
    _host_assert(num_tokens >= 0, "num_tokens must be >= 0")
    T = align(num_max_tokens_per_rank, TOKEN_ALIGNMENT)
    _host_assert(num_tokens <= T, "num_tokens must be <= num_max_tokens_per_rank")
    if block_m:
        _host_assert(
            block_m in VALID_BLOCK_MS,
            f"block_m={block_m}: only {VALID_BLOCK_MS} can be instantiated",
        )
    epr = num_experts // num_ranks

    # Buffer-level quantities (ring tokens come from the symmetric buffer: all variants share them)
    layout = build_symm_buffer_layout(
        num_ranks,
        num_experts,
        T,
        num_topk,
        hidden,
        intermediate_hidden,
        "fp8xfp4",
        "swiglu",
        num_shared_experts,
        num_sms,
    )

    # SM100-sized config (determines block_m, the pull size, SF block sizes ...)  (IMPL:94-98)
    base = get_mega_moe_config(
        num_ranks, num_experts, epr, T, num_tokens, num_topk, hidden, intermediate_hidden,
        layout.num_ring_tokens, layout.num_sf_ring_tokens, MMA_MXFP8FP4, force_block_m=block_m,
    )  # fmt: skip
    bm = base.block_m
    _host_assert(
        bm in VALID_BLOCK_MS,
        f"block_m={bm} cannot be instantiated (valid: {VALID_BLOCK_MS})",
    )
    store_block_m = get_sm107_store_block_m(bm)
    block_k = base.block_k
    _host_assert(block_k == 128, "only BLOCK_K = 128 is ported")

    # Shipped SM107 pipeline (IMPL:108-144)
    pipe = get_sm107_pipeline_config(
        num_experts, hidden, bm, base.block_n, block_k, base.num_bytes_per_pull, base.sf_block_m, base.sf_block_n,
        base.num_dispatch_threads, base.num_epilogue_threads, smem_capacity, max_stages,
    )  # fmt: skip

    # Per-die (IMPL:274-280): needs both N dims splittable into 2 domains x whole clusters
    clusters_split = (intermediate_hidden * 2) % (
        base.block_n * 2 * 2
    ) == 0 and hidden % (base.block_n * 2 * 2) == 0
    _host_assert(
        not (localized_weights or localized_shared_weights) or clusters_split,
        "localized weights need 2*I and H divisible by 512",
    )
    num_locality_domains = (
        2
        if (
            per_die
            and clusters_split
            and device_num_locality_domains == NUM_DEVICE_LOCALITY_DOMAINS
        )
        else 1
    )

    offs, sizes, end, sizeof = get_sm107_smem_layout(
        bm, pipe.num_stages, num_experts, sf_block_n=base.sf_block_n, block_k=block_k,
        num_dispatch_warps=base.num_dispatch_threads // 32, num_epilogue_warps=base.num_epilogue_threads // 32,
    )  # fmt: skip
    _host_assert(
        sizeof <= pipe.smem_size,
        "internal: SharedStorage larger than the requested smem",
    )
    tmem = get_tmem_layout(bm, base.sf_block_n)
    regs = get_register_split(epr)
    num_threads = (
        base.num_dispatch_threads
        + base.num_non_epilogue_threads
        + base.num_epilogue_threads
    )
    num_chunks = get_num_combine_chunks(
        hidden, offs["reusable_bytes"], base.num_epilogue_threads // 32
    )
    chunk_bytes = hidden * 2 // num_chunks
    # static asserts of the combine stage (K:1668-1675)
    _host_assert(hidden % num_chunks == 0 and chunk_bytes % 16 == 0 and (chunk_bytes // 16) % 32 == 0,
                 f"invalid combine chunking: hidden={hidden} -> {num_chunks} chunk(s) of {chunk_bytes} B (needs a multiple of 32 x 16 B, K:1671)")  # fmt: skip
    _host_assert(3 * (base.num_epilogue_threads // 32) * chunk_bytes <= offs["reusable_bytes"],
                 f"hidden is too large for the combine overlay: needs {3 * (base.num_epilogue_threads // 32) * chunk_bytes} B > "
                 f"reusable smem {offs['reusable_bytes']} B ({pipe.num_stages} stages; a stage cap shrinks the reusable region)")  # fmt: skip
    _host_assert(
        base.num_bytes_per_pull > 0 and hidden % base.num_bytes_per_pull == 0,
        "num_bytes_per_pull must divide hidden",
    )
    _host_assert(
        hidden % 512 == 0,
        f"hidden={hidden} is not a multiple of 512 (the register-pull dispatch moves hidden / (32 lanes x 16 B) uint4 per lane)",
    )
    l1_n = intermediate_hidden * 2
    _host_assert(
        l1_n // base.block_n <= 64,
        "too many L1 N blocks for the 64-bit L2 readiness mask",
    )
    wg_block_m = bm // 2
    _host_assert(
        wg_block_m % store_block_m == 0 and store_block_m % 8 == 0,
        "invalid store block",
    )

    return Sm107KernelConfig(
        num_ranks=num_ranks, num_experts=num_experts, num_experts_per_rank=epr,
        hidden=hidden, intermediate_hidden=intermediate_hidden, num_shared_experts=num_shared_experts,
        num_topk=num_topk, num_sms=num_sms, num_max_tokens_per_rank=T,
        block_m=bm, block_n=base.block_n, block_k=block_k, umma_k=UMMA_K_PACKED,
        load_block_m=base.load_block_m, load_block_n=base.load_block_n, store_block_m=store_block_m,
        sf_block_m=base.sf_block_m, sf_block_n=base.sf_block_n, wg_block_m=wg_block_m,
        umma_m=256, umma_n=bm, umma_block_k=128, atom_m=8,
        num_atoms_per_store=store_block_m // 8, num_store_blocks_per_wg=wg_block_m // store_block_m,
        shared_block_k=block_k // 2, l1_out_block_n=base.block_n // 2,
        l1_shape_n=l1_n, l1_shape_k=hidden, l2_shape_n=hidden, l2_shape_k=intermediate_hidden,
        shared_l2_shape_k=intermediate_hidden * num_shared_experts,
        num_l1_block_ns=l1_n // base.block_n, num_l2_block_ns=hidden // base.block_n,
        num_l1_clusters=l1_n // base.block_n // 2, num_l2_clusters=hidden // base.block_n // 2,
        swizzle_a_mode=128, swizzle_b_mode=64, swizzle_cd_mode=128, swizzle_shared_mode=min(block_k // 2, 128),
        num_stages=pipe.num_stages, smem_bytes=pipe.smem_size, accum_stages=2, tma_store_stages=2, schedule_stages=2,
        smem_sizeof=sizeof, smem_end=end, smem_reusable_bytes=offs["reusable_bytes"], smem_offsets=offs, smem_sizes=sizes,
        num_dispatch_threads=base.num_dispatch_threads, num_non_epilogue_threads=base.num_non_epilogue_threads,
        num_epilogue_threads=base.num_epilogue_threads, num_threads=num_threads,
        num_dispatch_warps=base.num_dispatch_threads // 32, num_mma_non_epilogue_warps=base.num_non_epilogue_threads // 32,
        num_epilogue_warps=base.num_epilogue_threads // 32, num_epilogue_warpgroups=base.num_epilogue_threads // 32 // 4,
        regs_dispatch=regs[0], regs_non_epilogue=regs[1], regs_epilogue=regs[2],
        tmem_accum_cols=tmem["accum_cols"], tmem_sfa_cols=tmem["sfa_cols"], tmem_sfb_cols=tmem["sfb_cols"],
        tmem_cols_used=tmem["used_cols"], tmem_cols_alloc=tmem["alloc_cols"],
        tmem_sfa_col=tmem["sfa_col"], tmem_sfb_col=tmem["sfb_col"],
        num_ring_tokens=layout.num_ring_tokens, num_sf_ring_tokens=layout.num_sf_ring_tokens,
        num_ring_blocks=layout.num_ring_tokens // bm, num_shared_sf_tokens=get_num_max_shared_sf_tokens(T),
        num_max_pool_tokens=layout.num_max_pool_tokens, num_bytes_per_pull=base.num_bytes_per_pull,
        num_pull_chunks=hidden // base.num_bytes_per_pull, symm_buffer_bytes=layout.num_bytes,
        num_tokens_per_warp=32 // num_topk, num_activate_lanes=(32 // num_topk) * num_topk,
        num_l1_clusters_per_domain=l1_n // base.block_n // 2 // num_locality_domains,
        num_l2_clusters_per_domain=hidden // base.block_n // 2 // num_locality_domains,
        num_experts_per_lane=cdiv(epr, 32), num_ranks_per_lane=cdiv(num_ranks, 32),
        num_combine_chunks=num_chunks, combine_chunk_bytes=chunk_bytes, combine_chunk_uint4_per_lane=chunk_bytes // 16 // 32,
        pull_uint4_per_lane=hidden // (32 * 16),
        clamp=clamp, alpha=float(activation_alpha), beta=float(activation_beta), fast_math=bool(fast_math),
        weight_packed_fp4=True, per_die=bool(per_die), num_locality_domains=num_locality_domains,
        localized_weights=bool(localized_weights), localized_shared_weights=bool(localized_shared_weights),
        trace=bool(trace), debug_flags=0, epi_prefetch_stages=0, dispatch_pipe=3, weight_prefetch=0, tma_l2_promotion=tma_l2_promotion,
    )  # fmt: skip


MAX_PORT_RANKS = 32  # the compiled ABI passes the NVLink peer table by value (docs/sm107_deepgemm_megamoe/DESIGN.md)
MAX_PORT_SHARED_EXPERTS = 1  # one merged FP8 shared expert (S = 0 or 1) is what has run; S > 1 changes the shared task counts and tensor maps and was never executed
SUPPORTED_FAST_MATH_VALUES: Tuple[bool, ...] = (
    True,
    False,
)  # kernel/epilogue.py implements both: __expf + rcp.approx (True) and nvcc's expf + div.rn.f32 (False); bit-exact vs DeepGEMM at EP1

#: Geometries that have run on Rubin (SM107) GPUs bit-exact against DeepGEMM.  The support predicate admits ONLY these (a structurally valid geometry
#: that never ran is refused, so that a DeepGEMM-API caller such as vLLM falls through to DeepGEMM).  Key ``(hidden, intermediate_hidden, num_topk)``,
#: value ``(admitted num_experts values, evidence)``.  Extend an entry only with a GPU result (bit-exact kernel test and, for a new shape, the A/B harness).
#: The EP size is a second dimension: a call is admitted only with at least :data:`MIN_VALIDATED_EXPERTS_PER_RANK` experts per rank.
VALIDATED_GEOMETRIES: Dict[Tuple[int, int, int], Tuple[Tuple[int, ...], str]] = {
    (5120, 2304, 6): (
        (32, 96, 192, 384),
        "DeepSeek-V4.1-Flash: E=384 at DEP1-DEP32 (bit-exactness and A/B campaigns, vLLM layer at EP1 / EP4); E=192 (EP1 per-die ring-wrap tests, EP4 harness) and "
        "E=96 / E=32 (EP1 full-kernel tests, vLLM layer at EP1) as test sizes of the same shape",
    ),
    (5120, 2304, 3): (
        (128,),
        "MTP / DSpark draft-model layers of vLLM's DeepSeek-V4.1 deployment (128 experts, top-3): EP4 (32 experts per rank), plain / masked / hot-expert / zipf / mixed per-rank M, "
        "M = 1 .. 4096: 120 / 120 (case, M, rank) cells bit-equal to DeepGEMM through vLLM's FlashInferMoeEp adapter; EP1 (128 experts per rank) full-kernel shape plan: 15 / 15",
    ),
}
#: Smallest experts-per-rank (``num_experts // num_ranks``) that has run on GPUs (E=384 at DEP32).  Smaller values (1-expert ranks, experts-per-rank below the
#: top-k, tiny per-lane expert counts) take code paths that never ran, so the support predicate refuses them for every geometry of the table.
MIN_VALIDATED_EXPERTS_PER_RANK = 12


def get_validated_geometries() -> Dict[
    Tuple[int, int, int], Tuple[Tuple[int, ...], str]
]:
    """A copy of :data:`VALIDATED_GEOMETRIES` (``(hidden, intermediate_hidden, num_topk) -> (num_experts values, evidence)``)."""
    return dict(VALIDATED_GEOMETRIES)


def get_mega_moe_unsupported_reason(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    mma_type: str = "fp8xfp4",
    activation: str = "swiglu",
    *,
    num_sms: Optional[int] = None,
    fast_math: bool = True,
    localized_shared_weights: bool = False,
    allow_unvalidated: bool = False,
) -> Optional[str]:
    """``None`` when the port can serve a MegaMoE call of this geometry, else a short human-readable reason (host only, ~1 ms, no GPU work).

    Mirrors what a DeepGEMM-API caller (vLLM) needs to decide per call whether to use the port or fall through to DeepGEMM: only
    ``mma_type='fp8xfp4'`` and ``activation='swiglu'`` are ported (``fast_math`` True and False both), at most one merged shared expert
    (``num_shared_experts`` in {0, 1}), die-local SHARED weights (vLLM's ``VLLM_DSV41_MEGAMOE_PERDIE=1``) are not (the kernel's weight tensor maps only implement die-local
    routed weights; pass ``localized_shared_weights=True`` when the caller will localize the shared pair), at most 32 ranks, the geometry must be one that has run
    on GPUs (:data:`VALIDATED_GEOMETRIES`; ``allow_unvalidated=True`` skips this check for kernel development), and every BLOCK_M variant a rank can select must
    derive a valid kernel configuration (shape divisibility, combine chunking, ring / smem sizing). The device SM count is that of the current CUDA device unless given.
    The EP size is a second validation dimension: at least :data:`MIN_VALIDATED_EXPERTS_PER_RANK` (12) experts per rank, the smallest value that has run.
    The predicate is total: it returns a reason (never raises) for every input, including non-positive or non-integer sizes."""
    try:
        return _unsupported_reason(
            num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts, mma_type, activation,
            num_sms=num_sms, fast_math=fast_math, localized_shared_weights=localized_shared_weights, allow_unvalidated=allow_unvalidated,
        )  # fmt: skip
    except Exception as exc:  # noqa: BLE001 - a DeepGEMM-API caller falls through on any reason string: nothing may escape
        return f"geometry cannot be derived ({type(exc).__name__}: {exc})"


def _unsupported_reason(
    num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts, mma_type, activation,
    *, num_sms, fast_math, localized_shared_weights, allow_unvalidated,
) -> Optional[str]:  # fmt: skip
    if mma_type != "fp8xfp4":
        return f"mma_type={mma_type!r} is not ported (only 'fp8xfp4')"
    if activation != "swiglu":
        return f"activation={activation!r} is not ported (only 'swiglu')"
    if bool(fast_math) not in SUPPORTED_FAST_MATH_VALUES:
        return f"fast_math={bool(fast_math)} is not ported (supported: {SUPPORTED_FAST_MATH_VALUES})"
    if localized_shared_weights:
        return "die-local shared-expert weights (vLLM VLLM_DSV41_MEGAMOE_PERDIE=1) are not implemented: the kernel's weight tensor maps only support die-local routed weights"
    if not 0 <= num_shared_experts <= MAX_PORT_SHARED_EXPERTS:
        return f"num_shared_experts={num_shared_experts} is outside the ported range [0, {MAX_PORT_SHARED_EXPERTS}] (one merged shared expert)"
    if not 1 <= num_ranks <= MAX_PORT_RANKS:
        return (
            f"num_ranks={num_ranks} is outside the ported range [1, {MAX_PORT_RANKS}]"
        )
    for name, value in (
        ("num_experts", num_experts),
        ("num_topk", num_topk),
        ("hidden", hidden),
        ("intermediate_hidden", intermediate_hidden),
        ("num_max_tokens_per_rank", num_max_tokens_per_rank),
    ):
        if value < 1:
            return f"{name}={value} must be positive"
    if num_sms is not None and num_sms < 1:
        return f"num_sms={num_sms} must be positive"
    if not allow_unvalidated:
        entry = VALIDATED_GEOMETRIES.get((hidden, intermediate_hidden, num_topk))
        if entry is None or num_experts not in entry[0]:
            known = "; ".join(
                f"hidden={k[0]}, intermediate_hidden={k[1]}, top-k={k[2]}, num_experts in {list(v[0])}"
                for k, v in VALIDATED_GEOMETRIES.items()
            )
            return (f"geometry (hidden={hidden}, intermediate_hidden={intermediate_hidden}, num_experts={num_experts}, top-k={num_topk}) has not been validated on GPUs "
                    f"(validated: {known})")  # fmt: skip
        if (
            num_experts % num_ranks == 0
            and num_experts // num_ranks < MIN_VALIDATED_EXPERTS_PER_RANK
        ):
            return (f"geometry (hidden={hidden}, intermediate_hidden={intermediate_hidden}, num_experts={num_experts}, top-k={num_topk}) has not been validated on GPUs with "
                    f"{num_experts // num_ranks} experts per rank (num_experts={num_experts} on {num_ranks} ranks; the smallest experts-per-rank that has run is "
                    f"{MIN_VALIDATED_EXPERTS_PER_RANK})")  # fmt: skip
    t = align(num_max_tokens_per_rank, TOKEN_ALIGNMENT)
    try:
        for bm in get_block_m_variants(num_ranks, num_experts, num_topk, t):
            get_sm107_kernel_config(
                num_ranks, num_experts, t, 0, num_topk, hidden, intermediate_hidden, num_shared_experts, num_sms=num_sms, block_m=bm,
            )  # fmt: skip
    except (AssertionError, RuntimeError) as exc:
        return str(exc)
    return None


def resolve_num_sms(num_sms: Optional[int] = None) -> int:
    """``num_sms`` if given, else the SM count of the current CUDA device (``kNumSMs`` must equal the whole device)."""
    if num_sms is not None:
        return int(num_sms)
    try:  # local import: this module stays torch-free
        import torch

        return int(
            torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).multi_processor_count
        )
    except Exception as exc:  # pragma: no cover - CPU-only host
        raise RuntimeError(
            "num_sms was not given and the SM count of the CUDA device could not be queried"
        ) from exc


# ===================================================================================================================
# Per-die (locality domain) weight addressing (K:934-936, 979-983, 998-1000; IMPL:163-176)
# ===================================================================================================================
def get_perdie_weight_tile_coords(
    local_expert_idx: int,
    n_block_idx: int,
    k_block_idx: int,
    shape_n: int,
    *,
    is_shared: bool = False,
    block_n: int = 128,
    block_k: int = 128,
    shared_block_k: int = 64,
    shared_half_k: bool = False,
) -> Tuple[int, int, int]:
    """TMA-B coordinates ``(k_idx, half_row, half_idx)`` of one weight tile of DIE-LOCAL weights (3D tensor map, dims ``(K, rows, domain)``).

    ``n_block_idx`` is the CTA's N block (``2 * n_cluster + cta_rank``).  The weights of every matrix are split along N into two halves,
    half ``h`` homed in locality domain ``h`` (slice ``h`` of the localized tensor): ``half_idx = (n_block * BLOCK_N >= N / 2)``;
    routed rows inside a slice are ``[expert][N / 2]``, shared rows ``[N / 2]``.  ``shared_half_k``: shared-expert half-K stages
    (packed-FP4 mode) advance K by ``SHARED_BLOCK_K`` instead of ``BLOCK_K``."""
    half_n = shape_n // 2
    n_in_matrix = n_block_idx * block_n
    half_idx = 1 if n_in_matrix >= half_n else 0
    half_row = (
        (0 if is_shared else local_expert_idx * half_n)
        + n_in_matrix
        - half_idx * half_n
    )
    k_idx = k_block_idx * (shared_block_k if (is_shared and shared_half_k) else block_k)
    return k_idx, half_row, half_idx


LOCALITY_ALLOC_GRANULARITY = (
    2 << 20
)  # CU_MEM_ALLOC_GRANULARITY_MINIMUM of locality-domain memory (CUDA 13.4); slices are padded to it


@dataclass(frozen=True)
class PerDieWeightTmaSpec:
    """3D tensor map of a die-local weight tensor (``make_tma_3d_desc``, IMPL:163-176)."""

    kind: str  # 'routed_l1' | 'routed_l2' | 'shared_l1' | 'shared_l2'
    dtype: str  # '16U4_ALIGN8B' (packed FP4, routed) | 'UINT8' (FP8 E4M3 shared)
    global_dims: Tuple[
        int, int, int
    ]  # (K in elements, rows per domain slice, num domains = 2); routed rows = E_l * N / 2
    global_strides_bytes: Tuple[
        int, int
    ]  # (row stride, domain-slice stride) in BYTES (dims 1 and 2)
    box_dims: Tuple[int, int, int]  # (inner box in elements, 128 rows, 1)
    swizzle_bytes: int  # 64 for both routed packed FP4 and half-K shared FP8


def _perdie_weight_geometry(
    cfg: Sm107KernelConfig, kind: str
) -> Tuple[int, int, int, int, int, str]:
    """``(K elements, rows per slice, bytes per row, box K elements, swizzle bytes, dtype)`` of one die-local weight tensor of ``cfg``."""
    E, H, I = cfg.num_experts_per_rank, cfg.hidden, cfg.intermediate_hidden
    S = cfg.num_shared_experts
    if kind == "routed_l1":  # (2, E_l, 2I / 2, H / 2 bytes)
        return H, E * (2 * I) // 2, H // 2, cfg.block_k, 64, "16U4_ALIGN8B"
    if kind == "routed_l2":  # (2, E_l, H / 2, I / 2 bytes)
        return I, E * H // 2, I // 2, cfg.block_k, 64, "16U4_ALIGN8B"
    if kind == "shared_l1":  # (2, 2I * S / 2, H bytes)
        return H, (2 * I * S) // 2, H, cfg.shared_block_k, 64, "UINT8"
    if kind == "shared_l2":  # (2, H / 2, I * S bytes)
        return I * S, H // 2, I * S, cfg.shared_block_k, 64, "UINT8"
    raise ValueError(f"unknown weight kind {kind!r}")


def get_perdie_domain_stride_bytes(
    cfg: Sm107KernelConfig, kind: str, granularity: int = LOCALITY_ALLOC_GRANULARITY
) -> int:
    """Bytes between slice 0 and slice 1 of the die-local weight tensor ``kind`` of ``cfg`` as ``locality.localize`` allocates it: one slice
    (``rows * row_bytes``) rounded up to the allocation granularity.  A compile-time constant of the 3D tensor map (the kernel takes only
    the base pointer; ``frontend`` verifies the real ``w.stride(0) * w.element_size()`` equals it)."""
    _, rows, row_bytes, _, _, _ = _perdie_weight_geometry(cfg, kind)
    return -(-(rows * row_bytes) // granularity) * granularity


def get_perdie_weight_tma_spec(
    cfg: Sm107KernelConfig, kind: str, *, row_stride_bytes: Optional[int] = None, domain_stride_bytes: Optional[int] = None,
    granularity: int = LOCALITY_ALLOC_GRANULARITY,
) -> PerDieWeightTmaSpec:  # fmt: skip
    """Geometry of the 3D weight tensor map of one die-local weight tensor of ``cfg`` (kinds ``routed_l1`` / ``routed_l2`` / ``shared_l1`` /
    ``shared_l2``).

    ``row_stride_bytes`` = ``w.stride(-2) * w.element_size()`` (default: the contiguous K bytes of one row) and ``domain_stride_bytes`` =
    ``w.stride(0) * w.element_size()`` (default: :func:`get_perdie_domain_stride_bytes`) of the localized tensor (routed ``(2, E_l, N/2, K/2)``
    packed FP4 int8, shared ``(2, N/2, K)`` FP8).  The routed expert dimension is folded into the rows (``rows = E_l * N / 2``: experts are
    contiguous inside a slice, which ``locality.is_canonical_localized`` guarantees)."""
    k, rows, row_bytes, box_k, swz, dtype = _perdie_weight_geometry(cfg, kind)
    if row_stride_bytes is None:
        row_stride_bytes = row_bytes
    if domain_stride_bytes is None:
        domain_stride_bytes = get_perdie_domain_stride_bytes(cfg, kind, granularity)
    return PerDieWeightTmaSpec(
        kind,
        dtype,
        (k, rows, 2),
        (row_stride_bytes, domain_stride_bytes),
        (box_k, cfg.load_block_n, 1),
        swz,
    )
