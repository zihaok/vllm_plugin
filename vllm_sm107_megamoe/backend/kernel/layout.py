# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / layout/mega_moe.cuh.
"""Pure-Python byte layouts of the SM107 MegaMoE kernel (no cutlass / torch import: CPU-importable).

Three layouts, all bit-identical to DeepGEMM (the symmetric buffer is an external ABI shared with vLLM's staging kernel):

* :func:`smem_offsets`   -- ``SharedStorage`` of ``sm107_fp8_fp4_mega_moe_impl`` (K:250-275).
* ``SIGNALS_*`` / :func:`workspace_bytes` -- ``layout::MegaMoESignals`` + workspace tail (LAY:54-146).
* :func:`buffer_layout`  -- ``layout::MegaMoEBuffer`` regions after the workspace (LAY:345-457).

``K:<line>`` = ``deep_gemm/include/deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh``, ``LAY:<line>`` = ``.../layout/mega_moe.cuh``
of DeepGEMM branch rubin-megamoe-a7a9277-1016 @ 3ec0eb4.  The package's ``Sm107KernelConfig.smem_offsets`` carries the same numbers
(computed independently by ``heuristics.py``); ``check_cfg_smem_offsets`` asserts that both agree.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple


def align(x: int, a: int) -> int:
    return -(-x // a) * a


def ceil_div(x: int, a: int) -> int:
    return -(-x // a)


# ----------------------------------------------------------------------------------------------------------------------
# MegaMoESignals (LAY:54-91).  kNumDeviceLocalityDomains = 2.
# ----------------------------------------------------------------------------------------------------------------------
NUM_MAX_RANKS = 72
NUM_MAX_EXPERTS = 2048
NUM_MAX_EXPERTS_PER_RANK = 512
NUM_MAX_RING_BLOCKS = 1 << 20
NUM_MAX_SHARED_L2_BLOCKS = 1 << 15
NUM_MAX_GRID_SYNC_COUNTERS = 4
NUM_DEVICE_LOCALITY_DOMAINS = 2

SIGNALS_GRID_SYNC_COUNT = 0  # u32[4]
SIGNALS_NVL_BARRIER_COUNTER = 16  # u32
SIGNALS_NVL_BARRIER_SIGNALS = 20  # s32[2]
SIGNALS_L1_TASK_COUNT = 28  # u32[2]
SIGNALS_L2_TASK_COUNT = 36  # u32[2]
SIGNALS_SHARED_L1_TASK_COUNT = 44  # u32[2]
SIGNALS_SHARED_L2_TASK_COUNT = 52  # u32[2]
SIGNALS_COMBINE_READY_GRID_IDX = 128  # u64[72], alignas(128)
SIGNALS_PEER_GRID_IDX = 768  # u64[72], alignas(128)
SIGNALS_EXPERT_SEND_COUNT = 1408  # u64[2048], alignas(128)
SIGNALS_EXPERT_RECV_COUNT = 17792  # u64[2048]
SIGNALS_EXPERT_RECV_COUNT_SUM = 34176  # u64[512]
SIGNALS_L1_FULL_COUNT = 38272  # u32[2^20]
SIGNALS_L1_EMPTY_COUNT = 4232576  # u32[2^20]
SIGNALS_L2_FULL_MASK = 8426880  # u64[2^20]
SIGNALS_L2_EMPTY_COUNT = 16815488  # u32[2^20]
SIGNALS_SHARED_L2_FULL_COUNT = 21009792  # u32[2^15]
SIGNALS_SIZEOF = 21140864  # sizeof(MegaMoESignals) (multiple of its alignas(128))

TOKEN_SRC_METADATA_BYTES = 12  # {rank_idx, token_idx, topk_idx} u32 each
K_MAX_CANDIDATE_BLOCK_M = 240
K_MIN_CANDIDATE_BLOCK_M = 8
K_LCM_CANDIDATE_BLOCK_M = 1920


def _check_signals_layout() -> None:
    """Re-derive the offsets above from the C++ alignment rules (guards typos)."""
    off = 0
    off += 4 * NUM_MAX_GRID_SYNC_COUNTERS
    assert off == SIGNALS_NVL_BARRIER_COUNTER
    off += 4
    assert off == SIGNALS_NVL_BARRIER_SIGNALS
    off += 8
    assert off == SIGNALS_L1_TASK_COUNT
    for name in ("L2_TASK_COUNT", "SHARED_L1_TASK_COUNT", "SHARED_L2_TASK_COUNT"):
        off += 4 * NUM_DEVICE_LOCALITY_DOMAINS
        assert off == globals()["SIGNALS_" + name], name
    off += 4 * NUM_DEVICE_LOCALITY_DOMAINS
    off = align(off, 128)
    assert off == SIGNALS_COMBINE_READY_GRID_IDX
    off += 8 * NUM_MAX_RANKS
    off = align(off, 128)
    assert off == SIGNALS_PEER_GRID_IDX
    off += 8 * NUM_MAX_RANKS
    off = align(off, 128)
    assert off == SIGNALS_EXPERT_SEND_COUNT
    off += 8 * NUM_MAX_EXPERTS
    assert off == SIGNALS_EXPERT_RECV_COUNT
    off += 8 * NUM_MAX_EXPERTS
    assert off == SIGNALS_EXPERT_RECV_COUNT_SUM
    off += 8 * NUM_MAX_EXPERTS_PER_RANK
    assert off == SIGNALS_L1_FULL_COUNT
    off += 4 * NUM_MAX_RING_BLOCKS
    assert off == SIGNALS_L1_EMPTY_COUNT
    off += 4 * NUM_MAX_RING_BLOCKS
    off = align(off, 8)
    assert off == SIGNALS_L2_FULL_MASK
    off += 8 * NUM_MAX_RING_BLOCKS
    assert off == SIGNALS_L2_EMPTY_COUNT
    off += 4 * NUM_MAX_RING_BLOCKS
    assert off == SIGNALS_SHARED_L2_FULL_COUNT
    off += 4 * NUM_MAX_SHARED_L2_BLOCKS
    assert align(off, 128) == SIGNALS_SIZEOF


_check_signals_layout()


def num_max_pool_tokens(num_ranks: int, num_max_tokens_per_rank: int, num_topk: int, num_experts_per_rank: int) -> int:
    """LAY:23-31 ``get_num_max_pool_tokens``."""
    max_recv = num_ranks * num_max_tokens_per_rank
    max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(max_recv * max_experts_per_token + num_experts_per_rank * (K_MAX_CANDIDATE_BLOCK_M - 1),
                 K_LCM_CANDIDATE_BLOCK_M)


def num_max_shared_sf_tokens(num_max_tokens_per_rank: int) -> int:
    """LAY:38-40."""
    return ceil_div(num_max_tokens_per_rank, K_MIN_CANDIDATE_BLOCK_M) * 128


@dataclass(frozen=True)
class WorkspaceLayout:
    """``layout::Workspace`` tail (after ``MegaMoESignals``)."""

    num_ranks: int
    num_experts: int
    num_experts_per_rank: int
    num_max_tokens_per_rank: int
    num_max_pool_tokens: int
    src_token_topk_idx_offset: int  # = SIGNALS_SIZEOF
    token_src_metadata_offset: int  # = SIGNALS_SIZEOF + E * T * 4
    nbytes: int  # total workspace bytes (aligned to 16)

    @property
    def num_shared_l2_pool_blocks(self) -> int:
        return ceil_div(self.num_max_tokens_per_rank, K_MIN_CANDIDATE_BLOCK_M)


def workspace_layout(num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_topk: int) -> WorkspaceLayout:
    assert num_experts % num_ranks == 0
    epr = num_experts // num_ranks
    pool = num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr)
    nbytes = SIGNALS_SIZEOF
    nbytes += num_experts * num_max_tokens_per_rank * 4  # src_token_topk_idx[EPR][R][T] u32
    nbytes += pool * TOKEN_SRC_METADATA_BYTES
    nbytes = align(nbytes, 16)
    return WorkspaceLayout(
        num_ranks=num_ranks, num_experts=num_experts, num_experts_per_rank=epr,
        num_max_tokens_per_rank=num_max_tokens_per_rank, num_max_pool_tokens=pool,
        src_token_topk_idx_offset=SIGNALS_SIZEOF,
        token_src_metadata_offset=SIGNALS_SIZEOF + num_experts * num_max_tokens_per_rank * 4,
        nbytes=nbytes,
    )  # fmt: skip


@dataclass(frozen=True)
class Region:
    """One ``layout::Buffer``: ``num_slots`` (= "ranks" of the Buffer, top-k slots for combine) x ``num_tokens`` x ``token_bytes``."""

    offset: int  # from the symmetric-buffer base
    token_bytes: int
    num_tokens: int
    num_slots: int = 1

    @property
    def nbytes(self) -> int:
        return self.token_bytes * self.num_tokens * self.num_slots

    @property
    def end(self) -> int:
        return self.offset + self.nbytes

    def slot_offset(self, slot: int) -> int:
        return self.offset + slot * self.token_bytes * self.num_tokens


# Region names in DeepGEMM declaration order (the order of layout, LAY:400-449)
REGION_NAMES = (
    "input_token", "input_sf", "input_topk_idx", "input_topk_weights",
    "shared_l1_sf", "shared_l2_token", "shared_l2_sf",
    "l1_token", "l1_sf", "l1_topk_weights", "l2_token", "l2_sf", "combine_token",
)  # fmt: skip


@dataclass(frozen=True)
class BufferLayout:
    workspace: WorkspaceLayout
    regions: Mapping[str, Region]
    total_bytes: int

    def __getitem__(self, name: str) -> Region:
        return self.regions[name]


def buffer_layout(
    *, hidden: int, intermediate_hidden: int, num_ranks: int, num_experts: int, num_max_tokens_per_rank: int,
    num_topk: int, num_ring_tokens: int, num_sf_ring_tokens: int, num_shared_experts: int = 0, with_sf: bool = True,
) -> BufferLayout:  # fmt: skip
    """``layout::MegaMoEBuffer`` (LAY:345-457) for the FP8 + UE8M0-SF path (``with_sf=True``)."""
    T = num_max_tokens_per_rank
    ws = workspace_layout(num_ranks, num_experts, T, num_topk)
    S = num_shared_experts
    shared_inter = intermediate_hidden * S
    max_shared_sf_tokens = num_max_shared_sf_tokens(T) if with_sf else 0
    elem = 1 if with_sf else 2
    input_token_bytes = hidden * elem
    bf16_token_bytes = hidden * 2
    inter_token_bytes = intermediate_hidden * elem
    shared_inter_token_bytes = shared_inter * elem
    input_sf_bytes = hidden // 32 if with_sf else 0
    inter_sf_bytes = intermediate_hidden // 32 if with_sf else 0
    shared_inter_sf_bytes = shared_inter // 32 if with_sf else 0

    regions: Dict[str, Region] = {}
    cursor = ws.nbytes

    def add(name: str, token_bytes: int, num_tokens: int, num_slots: int = 1) -> Region:
        nonlocal cursor
        r = Region(offset=cursor, token_bytes=token_bytes, num_tokens=num_tokens, num_slots=num_slots)
        regions[name] = r
        cursor = r.end
        return r

    add("input_token", input_token_bytes, T)
    add("input_sf", input_sf_bytes, T)
    add("input_topk_idx", num_topk * 8, T)
    add("input_topk_weights", num_topk * 4, T)
    add("shared_l1_sf", input_sf_bytes, max_shared_sf_tokens if S > 0 else 0)
    add("shared_l2_token", shared_inter_token_bytes, T if S > 0 else 0)
    add("shared_l2_sf", shared_inter_sf_bytes, max_shared_sf_tokens if S > 0 else 0)
    add("l1_token", input_token_bytes, num_ring_tokens)
    add("l1_sf", input_sf_bytes, num_sf_ring_tokens)
    add("l1_topk_weights", 4, num_ring_tokens)
    add("l2_token", inter_token_bytes, num_ring_tokens)
    add("l2_sf", inter_sf_bytes, num_sf_ring_tokens)
    add("combine_token", bf16_token_bytes, T, num_slots=num_topk + (1 if S > 0 else 0))
    return BufferLayout(workspace=ws, regions=regions, total_bytes=cursor)


# ----------------------------------------------------------------------------------------------------------------------
# SharedStorage (K:250-275)
# ----------------------------------------------------------------------------------------------------------------------
SMEM_ALIGN = 1024
TASK_INFO_BYTES = 32  # sched::TaskInfo<kHasShared>: 8 x u32, alignas(16)
BARRIER_BYTES = 8  # cutlass::arch::ClusterTransactionBarrier
NUM_DISPATCH_WARPS = 4
NUM_EPILOGUE_WARPS = 8
NUM_EPILOGUE_WARPGROUPS = 2
NUM_TMA_STORE_STAGES = 2
NUM_SCHEDULE_STAGES = 2
NUM_EPILOGUE_STAGES = 2  # TMEM accumulator stages
NUM_COMBINE_BARRIERS = NUM_EPILOGUE_WARPS * 2


def smem_offsets(
    *, block_m: int, num_stages: int, num_experts: int, store_block_m: int, block_k: int = 128,
    packed_fp4: bool = True, dispatch_pipe: int = 3, num_bytes_per_pull: int = 0,
) -> Dict[str, int]:  # fmt: skip
    """Byte offsets / sizes of ``SharedStorage`` members for dispatch mode 3 + packed FP4 weights.

    Returns ``{name: offset, name+'.size': nbytes, 'reusable_bytes', 'end', 'sizeof'}``.
    """
    assert dispatch_pipe == 3 and packed_fp4, "only the register-pull / packed-FP4 configuration is ported"
    sbm = store_block_m
    load_block_m = block_m // 2
    sf_block_m = align(block_m, 128)
    sf_block_n = 128
    out: Dict[str, int] = {}
    cur = 0

    def member(name: str, nbytes: int, alignment: int) -> int:
        nonlocal cur
        cur = align(cur, alignment)
        out[name] = cur
        out[name + ".size"] = nbytes
        cur += nbytes
        return out[name]

    member("expert_token_count", 4 * num_experts, SMEM_ALIGN)
    member("dispatch_send_buffer", 16, SMEM_ALIGN)  # [1][16]: occupies a whole 1 KiB slot
    # union smem_d: sizeof(union) = max(l1, l2) rounded to alignas(1024) of the union members
    l1 = NUM_EPILOGUE_WARPGROUPS * NUM_TMA_STORE_STAGES * sbm * 64
    l2 = NUM_EPILOGUE_WARPGROUPS * sbm * 128 * 2
    member("smem_d", align(max(l1, l2), SMEM_ALIGN), SMEM_ALIGN)
    member("smem_a", num_stages * load_block_m * block_k, SMEM_ALIGN)
    member("smem_b", num_stages * (128 * block_k // 2), SMEM_ALIGN)
    member("smem_sfa", num_stages * sf_block_m * 4, 4)
    member("smem_sfb", num_stages * sf_block_n * 4, 4)
    member("amax_reduction", NUM_EPILOGUE_WARPS * (sbm // 2) * 8, 8)
    member("task_infos", NUM_SCHEDULE_STAGES * TASK_INFO_BYTES, 16)
    reusable = cur
    member("dispatch_barriers", NUM_DISPATCH_WARPS * BARRIER_BYTES, 8)
    member("full_barriers", num_stages * BARRIER_BYTES, 8)
    member("empty_barriers", num_stages * BARRIER_BYTES, 8)
    member("tmem_full_barriers", NUM_EPILOGUE_STAGES * BARRIER_BYTES, 8)
    member("tmem_empty_barriers", NUM_EPILOGUE_STAGES * BARRIER_BYTES, 8)
    member("combine_barriers", NUM_COMBINE_BARRIERS * BARRIER_BYTES, 8)
    member("task_info_full_barriers", NUM_SCHEDULE_STAGES * BARRIER_BYTES, 8)
    member("task_info_empty_barriers", NUM_SCHEDULE_STAGES * BARRIER_BYTES, 8)
    member("tmem_ptr_in_smem", 4, 4)
    out["reusable_bytes"] = reusable
    out["end"] = cur
    out["sizeof"] = align(cur, SMEM_ALIGN)
    return out


def smem_offsets_for_cfg(cfg) -> Dict[str, int]:
    """:func:`smem_offsets` from an ``Sm107KernelConfig`` (duck-typed)."""
    return smem_offsets(block_m=cfg.block_m, num_stages=cfg.num_stages, num_experts=cfg.num_experts,
                        store_block_m=cfg.store_block_m, block_k=cfg.block_k)  # fmt: skip


def check_cfg_smem_offsets(cfg, ours: Optional[Mapping[str, int]] = None) -> None:
    """Assert that the package's ``cfg.smem_offsets`` (if populated) equals this module's independent computation."""
    ours = dict(ours) if ours is not None else smem_offsets_for_cfg(cfg)
    theirs = getattr(cfg, "smem_offsets", None) or {}
    for name, off in theirs.items():
        if name in ours and not name.endswith(".size"):
            assert ours[name] == off, f"smem offset mismatch for {name}: layout.py={ours[name]} cfg={off}"
    assert theirs.get("dispatch_barriers", ours["dispatch_barriers"]) == ours["reusable_bytes"]


# Shipped (SM107 launcher) stage counts and requested dynamic smem for V4.1 (E=384, H=5120, I=2304).
SHIPPED_V41 = {
    16: dict(num_stages=13, smem_bytes=231156, store_block_m=8),
    32: dict(num_stages=12, smem_bytes=229348, store_block_m=16),
    64: dict(num_stages=11, smem_bytes=232404, store_block_m=16),
    128: dict(num_stages=8, smem_bytes=217508, store_block_m=32),
    192: dict(num_stages=7, smem_bytes=224148, store_block_m=32),
    240: dict(num_stages=6, smem_bytes=216708, store_block_m=40),
}  # fmt: skip


def tmem_cols(block_m: int, num_accum_stages: int = NUM_EPILOGUE_STAGES) -> Tuple[int, int, int, int, int]:
    """``(accum_cols, sfa_cols, sfb_cols, alloc_cols, sfa_col)`` (K:283-290)."""
    sf_block_m = align(block_m, 128)
    accum = block_m * num_accum_stages
    sfa = sf_block_m // 32
    sfb = 128 // 32
    used = accum + sfa + sfb
    alloc = 32
    while alloc < used:
        alloc *= 2
    return accum, sfa, sfb, alloc, accum


def smem_summary_table(num_experts: int = 384) -> str:
    """Human-readable table (used to generate/verify DESIGN.md section 6)."""
    rows = []
    for bm, s in SHIPPED_V41.items():
        o = smem_offsets(block_m=bm, num_stages=s["num_stages"], num_experts=num_experts, store_block_m=s["store_block_m"])
        rows.append((bm, s, o))
    names = ["expert_token_count", "dispatch_send_buffer", "smem_d", "smem_a", "smem_b", "smem_sfa", "smem_sfb",
             "amax_reduction", "task_infos", "dispatch_barriers", "full_barriers", "empty_barriers",
             "tmem_full_barriers", "tmem_empty_barriers", "combine_barriers", "task_info_full_barriers",
             "task_info_empty_barriers", "tmem_ptr_in_smem"]  # fmt: skip
    lines = ["member | " + " | ".join(f"BM={bm}" for bm, _, _ in rows)]
    for n in names:
        lines.append(f"{n} | " + " | ".join(f"{o[n]} ({o[n + '.size']})" for _, _, o in rows))
    lines.append("end | " + " | ".join(str(o["end"]) for _, _, o in rows))
    lines.append("sizeof | " + " | ".join(str(o["sizeof"]) for _, _, o in rows))
    lines.append("requested | " + " | ".join(str(s["smem_bytes"]) for _, s, _ in rows))
    return "\n".join(lines)


if __name__ == "__main__":
    print(smem_summary_table())
