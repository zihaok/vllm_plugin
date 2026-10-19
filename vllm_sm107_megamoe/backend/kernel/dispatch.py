# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / dispatch warps (K:399-826).
"""Dispatch warps (warps 0-3 of every CTA): counting, SM offsets, index push, publish, grid syncs, NVLink barrier #1,
register-pull loop (mode 3), workspace cleanup and NVLink barrier #2.

Faithful to ``sm107_fp8_fp4_mega_moe_impl`` (``K:`` = ``deep_gemm/include/deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh``):

=====  ==========================================  =========================  =====================================
phase  what                                         K lines                    DSL notes
=====  ==========================================  =========================  =====================================
P1     per-expert token count (smem atomics)         403-429                    dynamic ``while`` over token chunks
P2     SM base slot: ``atom.global.add.u64``         431-438                    returns old; 3 sequential atomics
P3     index push (remote ``st`` of token*6+k)       440-447                    ``atomicAdd_block`` slot, generic ST
P4     grid sync A (idx 0)                           449-453                    inline-asm spin with 300 s timeout
P5     SM0 publish (grid idx, recv counts)           455-476                    remote ``st.u64`` + ``atom.sys.add``
P6     NVLink barrier #1 (tag 1)                     478-485                    prologue=False, epilogue=True
P7     rendezvous R1 (``barrier.sync 1, 384``)       490-491                    via ``ctx.bar``
P8     pull loop mode 3                              493-745                    no ``break``: carried predicates
P15    rendezvous R2                                 752
P17    cleanup (overlaps the combine)                754-817
P18    NVLink barrier #2 (tag 2)                     819-826                    prologue=True, epilogue=False
=====  ==========================================  =========================  =====================================

The control flow of the C++ (``while (true) { ... break; }``) is restructured without ``break`` / ``continue``
(CuteDSL 4.8 rejects them in dynamic regions, ``base_dsl/ast_preprocessor.py:715-780``): the expert-advance loop folds the
``break`` into its condition and the min-peeling loop carries a ``decided`` predicate (700 random cases were checked against a Python
transcription of K:538-615).

Only the documented ``ctx`` fields of DESIGN.md are used: ``cfg, sm_idx, thread_idx, warp_idx, lane_idx, num_tokens, stats,
sym (.rank_idx, .map), smem.expert_token_count, bar``.  Byte addresses of the workspace/ring regions are derived from
``ctx.sym.base`` + ``layout.buffer_layout(cfg)`` (identical to DeepGEMM's ``MegaMoEBuffer``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass import Uint64
from cutlass.cutlass_dsl import Int32, Int64

from . import layout as L
from .config import BAR_DISPATCH, BAR_DISPATCH_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS, NUM_DISPATCH_THREADS
from . import ptx as X  # the shared PTX helpers (DESIGN.md section 13)
from . import ptx_d as D  # dispatch-specific forms (generic remote ld/st, spin loops with timeout)
from .trace import ACC_DISPATCH_WAIT_L1_EMPTY, NUM_SLOTS, SLOT_NVL1_DONE, SLOT_PULL_DONE  # DeepGEMM-compatible phase trace (no-ops unless cfg.trace)

SLOT_D_GRIDID = 62  # dispatch debug slot of the trace buffer (unused by DeepGEMM's layout): raw %gridid of the launch
# dispatch debug stage stamps of the FIRST pull token of warp 0 of each CTA (clock64, trace builds only): 50 after R1, 51 after fetch_expert_recv_count + first get_num_tokens, 52 after the rank peeling,
# 53 after the remote pull loads were issued, 54 after the SF copy (remote RT consumed), 55 after the ring stores, 56 after the red.release of l1_full_count, 57 after the pull loop
SLOT_D_R1, SLOT_D_FETCH, SLOT_D_PEEL, SLOT_D_ISSUED, SLOT_D_SF, SLOT_D_STORED, SLOT_D_REL, SLOT_D_LOOPEND = 50, 51, 52, 53, 54, 55, 56, 57
ACC_D_WALK, ACC_D_WALK_N = 58, 59  # dispatch debug accumulators (lane 0 of each dispatch warp, flushed by the trace): cycles spent in the expert-advance walk / number of expert advances


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


@dataclass(frozen=True, eq=False)
class DispatchAddrs:
    """Local byte addresses (``Int64``) of the workspace fields and ring/input regions used by dispatch, combine and cleanup.

    Pure address arithmetic over ``base`` (the local symmetric-buffer base) with compile-time offsets from ``layout.py``;
    remote addresses are obtained with ``sym.map(addr, rank)``.  A FROZEN dataclass on purpose: CuteDSL's preprocessor
    treats ``obj.method(...)`` calls on mutable local objects inside dynamic regions as writes (the object would have to be
    a DSL value), frozen dataclasses are treated as read-only (``base_dsl/ast_preprocessor.py:visit_Call``).
    Build with :func:`make_addrs`."""

    base: Any
    cfg: Any
    lay: Any

    @property
    def R(self):
        return self.cfg.num_ranks

    @property
    def E(self):
        return self.cfg.num_experts

    @property
    def EPR(self):
        return self.cfg.num_experts_per_rank

    @property
    def T(self):
        return self.cfg.num_max_tokens_per_rank

    @property
    def regions(self):
        return self.lay.regions

    @property
    def num_shared_l2_pool_blocks(self):
        return self.lay.workspace.num_shared_l2_pool_blocks

    # ---- workspace (MegaMoESignals + tail) ----
    def _at(self, off: int):
        return self.base + Int64(off)

    def grid_sync_count(self, idx: int):
        return self._at(L.SIGNALS_GRID_SYNC_COUNT + 4 * idx)

    def nvl_barrier_counter(self):
        return self._at(L.SIGNALS_NVL_BARRIER_COUNTER)

    def nvl_barrier_signal(self, phase):
        return self._at(L.SIGNALS_NVL_BARRIER_SIGNALS) + Int64(phase) * Int64(4)

    def task_count(self, which: int, domain: int = 0):
        """Task counter ``which`` (0 l1, 1 l2, 2 shared_l1, 3 shared_l2) of locality domain ``domain`` (K:760-773)."""
        base = (L.SIGNALS_L1_TASK_COUNT, L.SIGNALS_L2_TASK_COUNT, L.SIGNALS_SHARED_L1_TASK_COUNT,
                L.SIGNALS_SHARED_L2_TASK_COUNT)[which]  # fmt: skip
        return self._at(base + 4 * domain)

    def combine_ready_grid_idx(self, rank):
        return self._at(L.SIGNALS_COMBINE_READY_GRID_IDX) + Int64(rank) * Int64(8)

    def peer_grid_idx(self, rank):
        return self._at(L.SIGNALS_PEER_GRID_IDX) + Int64(rank) * Int64(8)

    def expert_send_count(self, e):
        return self._at(L.SIGNALS_EXPERT_SEND_COUNT) + Int64(e) * Int64(8)

    def expert_recv_count(self, src_rank, local_expert):
        return self._at(L.SIGNALS_EXPERT_RECV_COUNT) + (Int64(src_rank) * Int64(self.EPR) + Int64(local_expert)) * Int64(8)

    def expert_recv_count_sum(self, local_expert):
        return self._at(L.SIGNALS_EXPERT_RECV_COUNT_SUM) + Int64(local_expert) * Int64(8)

    def l1_full_count(self, ring_block):
        return self._at(L.SIGNALS_L1_FULL_COUNT) + Int64(ring_block) * Int64(4)

    def l1_empty_count(self, ring_block):
        return self._at(L.SIGNALS_L1_EMPTY_COUNT) + Int64(ring_block) * Int64(4)

    def l2_full_mask(self, ring_block):
        return self._at(L.SIGNALS_L2_FULL_MASK) + Int64(ring_block) * Int64(8)

    def l2_empty_count(self, ring_block):
        return self._at(L.SIGNALS_L2_EMPTY_COUNT) + Int64(ring_block) * Int64(4)

    def shared_l2_full_count(self, i):
        return self._at(L.SIGNALS_SHARED_L2_FULL_COUNT) + Int64(i) * Int64(4)

    def src_token_topk_idx(self, local_expert, src_rank, slot):
        off = L.SIGNALS_SIZEOF
        return self._at(off) + ((Int64(local_expert) * Int64(self.R) + Int64(src_rank)) * Int64(self.T) + Int64(slot)) * Int64(4)

    def token_src_metadata(self, pool_token):
        return self._at(self.lay.workspace.token_src_metadata_offset) + Int64(pool_token) * Int64(12)

    # ---- symmetric-buffer regions ----
    def input_token(self, token):
        return self._at(self.regions["input_token"].offset) + Int64(token) * Int64(self.cfg.hidden)

    def input_sf(self, token):
        return self._at(self.regions["input_sf"].offset) + Int64(token) * Int64(self.cfg.hidden // 32)

    def input_topk_idx(self, flat_idx):
        """int64 ``(T, topk)``: address of element ``token * topk + k`` (``flat_idx``)."""
        return self._at(self.regions["input_topk_idx"].offset) + Int64(flat_idx) * Int64(8)

    def input_topk_weights(self, flat_idx):
        return self._at(self.regions["input_topk_weights"].offset) + Int64(flat_idx) * Int64(4)

    def l1_token(self, ring_token):
        return self._at(self.regions["l1_token"].offset) + Int64(ring_token) * Int64(self.cfg.hidden)

    def l1_sf_base(self):
        return self._at(self.regions["l1_sf"].offset)

    def l1_topk_weights(self, ring_token):
        return self._at(self.regions["l1_topk_weights"].offset) + Int64(ring_token) * Int64(4)

    def combine_token(self, slot, token):
        """``combine_token_buffer.get_rank_buffer(slot).get_data_buffer(token)`` (bf16 row of ``2 * hidden`` bytes)."""
        r = self.regions["combine_token"]
        return self._at(r.offset) + (Int64(slot) * Int64(r.num_tokens) + Int64(token)) * Int64(r.token_bytes)


def make_addrs(base, cfg) -> DispatchAddrs:
    """:class:`DispatchAddrs` for the local symmetric buffer at ``base`` (Int64) of the kernel configuration ``cfg``."""
    lay = L.buffer_layout(
        hidden=cfg.hidden, intermediate_hidden=cfg.intermediate_hidden, num_ranks=cfg.num_ranks,
        num_experts=cfg.num_experts, num_max_tokens_per_rank=cfg.num_max_tokens_per_rank,
        num_topk=cfg.num_topk, num_ring_tokens=cfg.num_ring_tokens, num_sf_ring_tokens=cfg.num_sf_ring_tokens,
        num_shared_experts=cfg.num_shared_experts)  # fmt: skip
    return DispatchAddrs(base, cfg, lay)


def launch_tag_of(ctx):
    """Per-launch tag (Int64) of the combine-readiness handshake: the ``ctx.launch_tag`` if the context has one, else ``%gridid + 1``.
    A plain Python helper on purpose (the static ``if`` must not go through the DSL's staged-``if`` type-join check)."""
    tag = getattr(ctx, "launch_tag", None)
    if tag is None:
        tag = X.gridid() + Int64(1)
    return tag


# ----------------------------------------------------------------------------------------------------------------
# comm helpers: DeepGEMM comm::grid_sync / comm::nvlink_barrier (barrier.cuh:33-92) with named-barrier scopes
# ----------------------------------------------------------------------------------------------------------------
@cute.jit
def grid_sync(counter_ga, num_sms: cutlass.Constexpr[int], sm_idx, thread_idx, bar_id: cutlass.Constexpr[int],
              bar_threads: cutlass.Constexpr[int]):
    """``comm::grid_sync<kNumSMs, idx>``: CTA rendezvous (``bar.sync bar_id, bar_threads``), thread 0 arrives (SM0 adds
    ``0x80000000 - (kNumSMs - 1)``, the others 1) and spins on the sense bit, CTA rendezvous.  Every one of the
    ``num_sms`` CTAs must call it (full co-residency)."""
    X.bar_sync(bar_id, bar_threads)
    if thread_idx == Int32(0):
        delta = Int32(1)
        if sm_idx == Int32(0):
            delta = Int32(0x80000000 - (num_sms - 1))
        D.grid_sync_arrive_wait(counter_ga, delta)
    X.bar_sync(bar_id, bar_threads)


@cute.jit
def nvlink_barrier(A, sym, num_ranks: cutlass.Constexpr[int], num_sms: cutlass.Constexpr[int], grid_idx: cutlass.Constexpr[int],
                   sm_idx, thread_idx, bar_id: cutlass.Constexpr[int], bar_threads: cutlass.Constexpr[int],
                   prologue: cutlass.Constexpr[bool], epilogue: cutlass.Constexpr[bool]):
    """``comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, idx, tag>(prologue, epilogue)`` (barrier.cuh:53-92)."""
    if cutlass.const_expr(prologue):
        grid_sync(A.grid_sync_count(grid_idx), num_sms, sm_idx, thread_idx, bar_id, bar_threads)

    if sm_idx == Int32(0):
        counter_ptr = A.nvl_barrier_counter()
        status = D.ld_b32_ordered(counter_ptr) & Int32(3)
        signal_phase = status & Int32(1)
        signal_sign = status >> Int32(1)
        signal_ptr = A.nvl_barrier_signal(signal_phase)

        # send signals to remote ranks
        if thread_idx < Int32(num_ranks):
            delta = Int32(1)
            if signal_sign != Int32(0):
                delta = Int32(-1)
            X.red_add_release_sys_s32(sym.map(signal_ptr, thread_idx), delta)
        X.bar_sync(bar_id, bar_threads)

        # update status and wait arrival
        if thread_idx == Int32(0):
            X.red_add_u32(counter_ptr, Int32(1))
            target = Int32(num_ranks)
            if signal_sign != Int32(0):
                target = Int32(0)
            D.spin_acquire_sys_s32_eq(signal_ptr, target)

    if cutlass.const_expr(epilogue):
        grid_sync(A.grid_sync_count(grid_idx), num_sms, sm_idx, thread_idx, bar_id, bar_threads)


# ----------------------------------------------------------------------------------------------------------------
# scheduler pieces used by the dispatch warps (SCH:112-150): per-lane cached expert counts
# ----------------------------------------------------------------------------------------------------------------
@cute.jit
def fetch_expert_recv_count(A, lane, epr: cutlass.Constexpr[int], num_sms: cutlass.Constexpr[int],
                            num_ranks: cutlass.Constexpr[int]):
    """``MegaMoESchedulerLD::fetch_expert_recv_count`` (SCH:136-150): lane ``l`` caches the counts of experts
    ``l, l+32, ...``: spin on ``ld.volatile.global.b64`` until ``hi32 == kNumSMs * kNumRanks`` and keep ``lo32``.
    Returns a Python list of ``ceil(EPR/32)`` Int32 values."""
    nepl = _ceil_div(epr, 32)
    stored = []
    for i in cutlass.range_constexpr(nepl):
        e = Int32(i * 32) + lane
        val = Int64(0)
        if e < Int32(epr):
            val = X.ld_volatile_b64(A.expert_recv_count_sum(e))
            while (val >> Int64(32)) != Int64(num_sms * num_ranks):
                val = X.ld_volatile_b64(A.expert_recv_count_sum(e))
        stored.append(Int32(val & Int64(0xFFFFFFFF)))
    cute.arch.sync_warp()
    return stored


@cute.jit
def get_num_tokens(stored, expert_idx, lane, epr: cutlass.Constexpr[int]):
    """``get_num_tokens(expert_idx)`` (SCH:112-119): the lane owning ``expert_idx`` broadcasts its cached count."""
    valid_value = Int32(0)
    for i in cutlass.range_constexpr(_ceil_div(epr, 32)):
        if expert_idx == Int32(i * 32) + lane:
            valid_value = stored[i]
    return cute.arch.shuffle_sync(valid_value, expert_idx % Int32(32))


@cute.jit
def get_pool_block_offset(stored, expert_idx, lane, epr: cutlass.Constexpr[int], block_m: cutlass.Constexpr[int]):
    """``get_pool_block_offset(expert_idx)`` (SCH:122-132): ``sum_{e' < e} ceil(count[e'] / BLOCK_M)`` by warp reduce."""
    num_blocks = Int32(0)
    for i in cutlass.range_constexpr(_ceil_div(epr, 32)):
        if Int32(i * 32) + lane < expert_idx:
            num_blocks = num_blocks + (stored[i] + Int32(block_m - 1)) // Int32(block_m)
    return cute.arch.warp_redux_sync(num_blocks, "add")


# ----------------------------------------------------------------------------------------------------------------
# the role
# ----------------------------------------------------------------------------------------------------------------
@cute.jit
def dispatch_role(ctx):
    """Dispatch warps (K:399-826).  The kernel executed ``setmaxnreg.dec cfg.regs_dispatch`` before calling this."""
    cfg = ctx.cfg
    R = cfg.num_ranks
    E = cfg.num_experts
    EPR = cfg.num_experts_per_rank
    H = cfg.hidden
    TOPK = cfg.num_topk
    NSM = cfg.num_sms
    BM = cfg.block_m
    SF_BM = cfg.sf_block_m
    NRB = cfg.num_ring_blocks
    NRT = cfg.num_ring_tokens
    NSRT = cfg.num_sf_ring_tokens
    NL1BN = cfg.num_l1_block_ns
    NTW = cfg.num_tokens_per_warp
    NACT = cfg.num_activate_lanes
    NT = cfg.num_dispatch_threads
    NW = cfg.num_dispatch_warps
    NRPL = _ceil_div(R, 32)  # kNumRanksPerLane
    _NEPL = _ceil_div(EPR, 32)  # kNumExpertsPerLane (unused)
    NU4 = H // (32 * 16)  # uint4 per lane of the register pull (10 for H=5120)
    NSF = H // 128  # SF words per token (40)
    assert NT == 128 and NW == 4 and TOPK <= 32 and R <= NT and H % (32 * 16) == 0 and NSM > 1, "unsupported dispatch config"

    sm_idx = ctx.sm_idx
    thread_idx = ctx.thread_idx
    warp_idx = ctx.warp_idx
    lane = ctx.lane_idx
    num_tokens = ctx.num_tokens
    stats = ctx.stats  # cumulative_local_expert_recv_stats (gmem Int32 pointer, address 0 = none)
    sym = ctx.sym
    my_rank = sym.rank_idx
    A = make_addrs(sym.base, cfg)
    # per-launch tag of the combine-readiness handshake (K:460: `ptx::get_grid_idx() + 1`): the `ctx.launch_tag` (DESIGN.md section 4.2,
    # tag_mode "gridid" = DeepGEMM's value, "epoch" = unique under CUDA-graph replay); fall back to `%gridid + 1` for contexts without it
    launch_tag = launch_tag_of(ctx)
    tr = getattr(ctx, "tr", None)  # the Trace handle (None in contexts without one); every call compiles to nothing unless cfg.trace
    expert_token_count = ctx.smem.expert_token_count  # smem u32[E]

    # ------------------------------------------------------------------------------------------------------------
    # P1: count experts' tokens (K:426-428).  `read_topk_idx` (K:406-423) is a dynamic loop over token chunks:
    # warp (sm, w) starts at token (sm*4 + w) * kNumTokensPerWarp and strides by kNumSMs * 4 * kNumTokensPerWarp.
    # int64 topk_idx is read as its low 32 bits (nvcc emits a 32-bit LDG.E for `static_cast<int>`).
    # ------------------------------------------------------------------------------------------------------------
    i = (sm_idx * Int32(NW) + warp_idx) * Int32(NTW)
    while i < num_tokens:
        if i + lane // Int32(TOPK) < num_tokens and lane < Int32(NACT):
            expert_idx = X.ld_b32(A.input_topk_idx(i * Int32(TOPK) + lane))
            if expert_idx >= Int32(0):
                cute.arch.atomic_add(expert_token_count + expert_idx, Int32(1), sem="relaxed", scope="cta")
        cute.arch.sync_warp()
        i = i + Int32(NSM * NW * NTW)
    X.bar_sync(BAR_DISPATCH, NUM_DISPATCH_THREADS)

    # ------------------------------------------------------------------------------------------------------------
    # P2: SM offset (~6.5 us): base slot of (this SM, expert) in this rank's send stream = lo32 of the returned old value
    # of atom.global.add.u64(expert_send_count[e], (1 << 32) | count).  DeepGEMM's loop is sequential (ld.shared,
    # atom, st.shared per expert); kept as is.
    # ------------------------------------------------------------------------------------------------------------
    for k in cutlass.range_constexpr(0, E, NT):
        e = thread_idx + Int32(k)
        if e < Int32(E):
            send_value = (Int64(1) << Int64(32)) | Int64((expert_token_count + e).load())
            old = X.atom_add_u64(A.expert_send_count(e), send_value)
            (expert_token_count + e).store(Int32(old & Int64(0xFFFFFFFF)))
    X.bar_sync(BAR_DISPATCH, NUM_DISPATCH_THREADS)

    # ------------------------------------------------------------------------------------------------------------
    # P3: write source indices (~2 us with 512 tokens): slot = atomicAdd_block(count[e], 1); remote generic st.u32 of
    # token*topk + k into src_token_topk_idx[e % EPR][my_rank][slot] of rank e / EPR.
    # ------------------------------------------------------------------------------------------------------------
    i = (sm_idx * Int32(NW) + warp_idx) * Int32(NTW)
    while i < num_tokens:
        if i + lane // Int32(TOPK) < num_tokens and lane < Int32(NACT):
            expert_idx = X.ld_b32(A.input_topk_idx(i * Int32(TOPK) + lane))
            if expert_idx >= Int32(0):
                dst_rank_idx = expert_idx // Int32(EPR)
                dst_slot_idx = cute.arch.atomic_add(expert_token_count + expert_idx, Int32(1), sem="relaxed", scope="cta")
                dst_ptr = A.src_token_topk_idx(expert_idx % Int32(EPR), my_rank, dst_slot_idx)
                D.st_generic_b32(sym.map(dst_ptr, dst_rank_idx), i * Int32(TOPK) + lane)
        cute.arch.sync_warp()
        i = i + Int32(NSM * NW * NTW)

    # ------------------------------------------------------------------------------------------------------------
    # P4: grid sync A (dispatch counter idx 0, CTA scope = bar.sync 0,128)
    # ------------------------------------------------------------------------------------------------------------
    grid_sync(A.grid_sync_count(0), NSM, sm_idx, thread_idx, 0, NT)

    # ------------------------------------------------------------------------------------------------------------
    # P5: SM 0 publishes: this launch's grid index to every peer (tag for combine readiness; +1 differs from a zeroed
    # workspace), then per expert the send count into the destination's expert_recv_count[my_rank][e % EPR] and an
    # atom.sys.add of the (hi32 = #SMs, lo32 = count) status into expert_recv_count_sum[e % EPR].
    # ------------------------------------------------------------------------------------------------------------
    if sm_idx == Int32(0):
        if thread_idx < Int32(R):
            D.st_generic_b64(sym.map(A.peer_grid_idx(my_rank), thread_idx), launch_tag)
        cute.arch.sync_warp()
        for k in cutlass.range_constexpr(0, E, NT):
            e = thread_idx + Int32(k)
            if e < Int32(E):
                dst_rank_idx = e // Int32(EPR)
                dst_local_expert_idx = e % Int32(EPR)
                expert_status = X.ld_b64(A.expert_send_count(e))
                D.st_generic_b64(sym.map(A.expert_recv_count(my_rank, dst_local_expert_idx), dst_rank_idx),
                                 expert_status & Int64(0xFFFFFFFF))
                X.atom_add_sys_u64(sym.map(A.expert_recv_count_sum(dst_local_expert_idx), dst_rank_idx), expert_status)
    X.bar_sync(BAR_DISPATCH, NUM_DISPATCH_THREADS)

    # ------------------------------------------------------------------------------------------------------------
    # P6: NVLink barrier #1 (tag 1): prologue=False (grid sync A just happened, only SM 0 wrote since), epilogue=True
    # ------------------------------------------------------------------------------------------------------------
    nvlink_barrier(A, sym, R, NSM, 0, sm_idx, thread_idx, 0, NT, False, True)

    if tr is not None:  # K:487-488 trace slot 7: after NVLink barrier #1 (dispatch thread 0)
        tr.stamp(SLOT_NVL1_DONE, thread_idx == Int32(0))
        if cutlass.const_expr(tr.enabled):  # dispatch debug (trace builds only, no production effect): raw `%gridid` of this launch -> trace slot 62 (CUDA-graph replay tag evidence)
            if thread_idx == Int32(0):
                cute.arch.store(tr.trace_ptr + (tr.sm_idx * Int32(NUM_SLOTS) + Int32(SLOT_D_GRIDID)), Uint64(X.gridid()))

    # P7: R1 -- the epilogue warps cannot take tasks before the dispatch warps passed the NVLink barrier (K:490-491)
    X.bar_sync_unaligned(BAR_DISPATCH_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS)
    if tr is not None:  # dispatch debug stage stamp (trace builds only)
        tr.stamp_clock(SLOT_D_R1, thread_idx == Int32(0))

    # ------------------------------------------------------------------------------------------------------------
    # P8: pull token data and SF from remote ranks into the local L1 ring (mode 3: copy through registers)
    # ------------------------------------------------------------------------------------------------------------
    stored = fetch_expert_recv_count(A, lane, EPR, NSM, R)  # wait for token data arrival (K:533)

    NG = NSM * NW  # kNumGlobalWarps
    token_idx = sm_idx * Int32(NW) + warp_idx
    cur = Int32(-1)  # current_expert_idx
    e_start = Int32(0)  # expert_start_idx
    e_end = Int32(0)  # expert_end_idx
    pool_off = Int32(0)  # expert_pool_block_offset
    stored_rank_count = cute.make_rmem_tensor((NRPL,), Int32)
    rem = cute.make_rmem_tensor((NRPL,), Int32)
    for i2 in cutlass.range_constexpr(NRPL):
        stored_rank_count[i2] = Int32(0)
        rem[i2] = Int32(0)

    first_it = Int32(1)  # dispatch debug: stage stamps only for the first token iteration of warp 0 (trace builds)
    walk_n = Int32(0)  # dispatch debug: number of expert advances of this warp (trace builds)
    # K:536 `for (token_idx = ...; ; token_idx += kNumGlobalWarps)` with `break` at K:552 -> while on the carried expert idx
    while cur < Int32(EPR):
        old_expert = cur
        t_walk0 = tr.clock() if tr is not None else Int64(0)
        # K:539-549 advance expert until within range; the C++ `break` (K:540-541) is folded into the condition
        while token_idx >= e_end and cur < Int32(EPR):
            cur = cur + Int32(1)
            walk_n = walk_n + Int32(1)
            if cur < Int32(EPR):
                pool_off = pool_off + (e_end - e_start + Int32(BM - 1)) // Int32(BM)
                e_start = e_end
                e_end = e_end + get_num_tokens(stored, cur, lane, EPR)
        if tr is not None:  # dispatch debug (trace builds): cycles of the expert walk, lane 0 of each dispatch warp
            if lane == Int32(0):
                tr.add_since(ACC_D_WALK, t_walk0)

        if tr is not None:
            tr.stamp_clock(SLOT_D_FETCH, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))
        if cur < Int32(EPR):  # K:552 `if (current_expert_idx >= kNumExpertsPerRank) break;`
            # K:556-565 per-rank counts of the new expert (plain 64-bit loads, written remotely before NVLink barrier #1)
            if old_expert != cur:
                for i2 in cutlass.range_constexpr(NRPL):
                    j = Int32(i2 * 32) + lane
                    v = Int32(0)
                    if j < Int32(R):
                        v = Int32(X.ld_b64(A.expert_recv_count(j, cur)) & Int64(0xFFFFFFFF))
                    stored_rank_count[i2] = v

            # ---- K:567-615 round-robin rank selection via iterative min-peeling ----
            for i2 in cutlass.range_constexpr(NRPL):
                rem[i2] = stored_rank_count[i2]
            offset = Int32(0)
            tok_in_e = token_idx - e_start  # token_idx_in_expert
            slot = tok_in_e
            cur_rank = Int32(0)  # current_rank_in_expert_idx
            tok_in_rank = Int32(0)  # token_idx_in_rank
            decided = Int32(0)  # replaces `while (true) { ... break; }`
            while decided == Int32(0):
                n_act_lane = Int32(0)
                min_lane = Int32(0x7FFFFFFF)
                for i2 in cutlass.range_constexpr(NRPL):
                    r = rem[i2]
                    if r > Int32(0):
                        n_act_lane = n_act_lane + Int32(1)
                        min_lane = cutlass.min(min_lane, r)
                n_act = cute.arch.warp_redux_sync(n_act_lane, "add")  # __reduce_add_sync
                length = cute.arch.warp_redux_sync(min_lane, "min")  # __reduce_min_sync
                num_round_tokens = length * n_act
                if n_act == Int32(0):  # robustness guard only (the C++ would spin forever); unreachable on valid input
                    decided = Int32(1)
                elif slot < num_round_tokens:  # hit in the current round
                    slot_in_round = slot % n_act
                    num_seen = Int32(0)
                    for i2 in cutlass.range_constexpr(NRPL):
                        mask = cute.arch.vote_ballot_sync(rem[i2] > Int32(0))
                        n_lanes = cute.arch.popc(mask)
                        if slot_in_round >= num_seen and slot_in_round < num_seen + n_lanes:
                            cur_rank = Int32(i2 * 32) + X.fns_b32(mask, Int32(0), slot_in_round - num_seen + Int32(1))
                        num_seen = num_seen + n_lanes
                    tok_in_rank = offset + slot // n_act
                    decided = Int32(1)
                else:  # move into the next round
                    slot = slot - num_round_tokens
                    offset = offset + length
                    for i2 in cutlass.range_constexpr(NRPL):
                        rem[i2] = rem[i2] - cutlass.min(rem[i2], length)

            if tr is not None:
                tr.stamp_clock(SLOT_D_PEEL, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))
            # K:617-621 read source token-topk index (written remotely by the source rank's dispatch in P3)
            stt = X.ld_b32(A.src_token_topk_idx(cur, cur_rank, tok_in_rank))  # src_token_topk_idx
            src_token = stt // Int32(TOPK)
            src_topk = stt % Int32(TOPK)

            # K:628-629
            pool_token = pool_off * Int32(BM) + tok_in_e
            pool_block = pool_token // Int32(BM)
            ring_block = pool_block % Int32(NRB)

            # K:631-637 wait for the ring slot (previous consumer must have finished all L1 N blocks); all lanes spin
            l1_empty_target = (pool_block // Int32(NRB)) * Int32(NL1BN)
            if l1_empty_target > Int32(0):
                empty_ptr = A.l1_empty_count(ring_block)
                t_wait0 = tr.clock() if tr is not None else Int64(0)
                v = X.ld_acquire_gpu_b32(empty_ptr)
                while v < l1_empty_target:
                    v = X.ld_acquire_gpu_b32(empty_ptr)
                if tr is not None:  # K:636 trace slot 34 (lane 0 of each dispatch warp accumulates its stall cycles)
                    if lane == Int32(0):
                        tr.add_since(ACC_DISPATCH_WAIT_L1_EMPTY, t_wait0)

            # K:639-661 register pull: every lane loads its 16 B slices of the remote token (all in flight)
            src_base = sym.map(A.input_token(src_token), cur_rank)
            dst_base = A.l1_token(pool_token % Int32(NRT))
            pull = []
            for j in cutlass.range_constexpr(NU4):
                pull.append(D.ld_generic_v4_b32(src_base + Int64(j * 32 * 16) + Int64(lane) * Int64(16)))
            cute.arch.sync_warp()
            if tr is not None:
                tr.stamp_clock(SLOT_D_ISSUED, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))

            # K:677-680 load the weight first, so that its remote latency overlaps with the SF copy below
            weight = D.ld_generic_b32(sym.map(A.input_topk_weights(stt), cur_rank))

            # K:682-699 load and store SF (UTCCP 4x32-transposed position inside each 128-token group, K:184-188)
            remote_sf = sym.map(A.input_sf(src_token), cur_rank)
            local_sf = A.l1_sf_base()
            tok_in_block = tok_in_e % Int32(BM)
            sf_ring_token = ring_block * Int32(SF_BM) + (tok_in_block & Int32(~127)) + (tok_in_block & Int32(31)) * Int32(4) \
                + ((tok_in_block >> Int32(5)) & Int32(3))  # fmt: skip
            for i2 in cutlass.range_constexpr(_ceil_div(NSF, 32)):
                j = Int32(i2 * 32) + lane
                if j < Int32(NSF):
                    w = D.ld_generic_b32(remote_sf + Int64(j) * Int64(4))
                    X.st_b32(local_sf + (Int64(j) * Int64(NSRT) + Int64(sf_ring_token)) * Int64(4), w)
            cute.arch.sync_warp()
            if tr is not None:
                tr.stamp_clock(SLOT_D_SF, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))

            # K:701-709 store the token into the local ring (ordered before the release below by the warp barrier)
            for j in cutlass.range_constexpr(NU4):
                X.st_v4_b32(dst_base + Int64(j * 32 * 16) + Int64(lane) * Int64(16), pull[j][0], pull[j][1], pull[j][2], pull[j][3])
            cute.arch.sync_warp()
            if tr is not None:
                tr.stamp_clock(SLOT_D_STORED, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))

            # K:711-729 weight + metadata, then publish with red.release.gpu.add(l1_full_count)
            with cute.arch.elect_one():
                X.st_b32(A.l1_topk_weights(pool_token % Int32(NRT)), weight)
                meta = A.token_src_metadata(pool_token)
                X.st_b32(meta, cur_rank)
                X.st_b32(meta + Int64(4), src_token)
                X.st_b32(meta + Int64(8), src_topk)
                add = Int32(1)
                if token_idx == e_end - Int32(1):  # is_last_token: add BLOCK_M - r so the block total is exactly BLOCK_M
                    add = Int32(BM) - (tok_in_e % Int32(BM))
                X.red_add_release_gpu_u32(A.l1_full_count(ring_block), add)
            cute.arch.sync_warp()
            if tr is not None:
                tr.stamp_clock(SLOT_D_REL, (warp_idx == Int32(0)) & (lane == Int32(0)) & (first_it == Int32(1)))
            first_it = Int32(0)

            token_idx = token_idx + Int32(NG)

    if tr is not None:  # K:747-748 trace slot 1: pull done (dispatch thread 0)
        tr.stamp(SLOT_PULL_DONE, thread_idx == Int32(0))
        tr.stamp_clock(SLOT_D_LOOPEND, thread_idx == Int32(0))
        if lane == Int32(0):
            tr.add(ACC_D_WALK_N, walk_n)

    # ------------------------------------------------------------------------------------------------------------
    # TEST HOOK (never set by the production kernel): in dispatch-only staged tests nothing consumes the pulled tokens, so no
    # grid-wide ordering exists between the pull loops of other SMs and this CTA's cleanup (in the real kernel the epilogue
    # grid sync before R2 provides it: every pulled token has been consumed by an L1 task before the L2 epilogue finishes).
    # ------------------------------------------------------------------------------------------------------------
    if cutlass.const_expr(getattr(ctx, "d_test_pull_done_sync", False)):
        grid_sync(A.grid_sync_count(2), NSM, sm_idx, thread_idx, BAR_DISPATCH, NUM_DISPATCH_THREADS)  # counter idx 2: idx 0/1 belong to the protocol

    # ------------------------------------------------------------------------------------------------------------
    # P15: R2 -- cleanup starts only after the epilogue finished all L2 work and the epilogue grid sync (K:752)
    # ------------------------------------------------------------------------------------------------------------
    X.bar_sync_unaligned(BAR_DISPATCH_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS)

    # ------------------------------------------------------------------------------------------------------------
    # P17: clean the workspace for the next launch and accumulate cumulative stats (overlaps the combine, K:754-817)
    # ------------------------------------------------------------------------------------------------------------
    if sm_idx == Int32(0):
        # SM 0: clear expert send count and schedule task counters
        for k in cutlass.range_constexpr(0, E, NT):
            e = thread_idx + Int32(k)
            if e < Int32(E):
                X.st_b64(A.expert_send_count(e), Int64(0))
        if warp_idx == Int32(0):
            with cute.arch.elect_one():
                for w in cutlass.range_constexpr(4):
                    X.st_b32(A.task_count(w, 0), Int32(0))
                for d in cutlass.range_constexpr(1, cfg.num_locality_domains):
                    for w in cutlass.range_constexpr(4):
                        X.st_b32(A.task_count(w, d), Int32(0))
        cute.arch.sync_warp()
        i = thread_idx
        while i < Int32(A.num_shared_l2_pool_blocks):
            X.st_b32(A.shared_l2_full_count(i), Int32(0))
            i = i + Int32(NT)
        cute.arch.sync_warp()
    else:
        # other SMs: each takes experts sm-1, sm-1 + (kNumSMs-1), ... (K:781)
        ex = sm_idx - Int32(1)
        while ex < Int32(EPR):
            # read expert token count before clearing
            num_recv_tokens = Int32(X.ld_b64(A.expert_recv_count_sum(ex)) & Int64(0xFFFFFFFF))
            num_recv_m_blocks = (num_recv_tokens + Int32(BM - 1)) // Int32(BM)
            # compute expert pool block offset (from the counts cached by fetch_expert_recv_count)
            ex_pool_off = get_pool_block_offset(stored, ex, lane, EPR, BM)
            # wait read count ready
            X.bar_sync(BAR_DISPATCH, NUM_DISPATCH_THREADS)
            # clean expert token count, and add cumulative results
            if warp_idx == Int32(0):
                X.st_b64(A.expert_recv_count_sum(ex), Int64(0))
            elif warp_idx == Int32(1):
                with cute.arch.elect_one():
                    if stats.toint() != Int64(0):
                        X.red_add_s32(stats + ex, num_recv_tokens)
                cute.arch.sync_warp()
            # clean per-rank token count
            j = thread_idx
            while j < Int32(R):
                X.st_b64(A.expert_recv_count(j, ex), Int64(0))
                j = j + Int32(NT)
            cute.arch.sync_warp()
            # clean L1 and L2 full stuffs and ring buffer counts
            j = thread_idx
            while j < num_recv_m_blocks:
                slot_idx = (ex_pool_off + j) % Int32(NRB)
                X.st_b32(A.l1_full_count(slot_idx), Int32(0))
                X.st_b32(A.l1_empty_count(slot_idx), Int32(0))
                X.st_b64(A.l2_full_mask(slot_idx), Int64(0))
                X.st_b32(A.l2_empty_count(slot_idx), Int32(0))
                j = j + Int32(NT)
            cute.arch.sync_warp()
            ex = ex + Int32(NSM - 1)

    # ------------------------------------------------------------------------------------------------------------
    # P18: wait for all ranks to finish cleaning: grid sync before the NVLink barrier (tag 2), none after
    # ------------------------------------------------------------------------------------------------------------
    nvlink_barrier(A, sym, R, NSM, 0, sm_idx, thread_idx, 0, NT, True, False)
