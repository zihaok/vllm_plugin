# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / combine epilogue tail (K:1650-1809).
"""Combine: the tail of the epilogue warps (warps 8-15) after the last GEMM task and the TMEM free (done by the kernel).

Faithful to ``sm107_fp8_fp4_mega_moe_impl`` (``K:`` = ``deep_gemm/include/deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh``):

=====  ============================================================  ===========
phase  what                                                           K lines
=====  ============================================================  ===========
P13    load ``peer_grid_idx`` (SM0), epilogue grid sync idx 1 over the     1690-1700
       256 epilogue threads ("all local L2 writes are done")
P14    SM0 threads ``< kNumRanks``: ``st.release.sys.u64`` of the           1705-1707
       peer's grid tag into ``peer.combine_ready_grid_idx[me]``
P15    rendezvous R2 (``barrier.sync 1, 384``, second use)                  1709-1710
P16    per token chunk (``token_chunk = warp*kNumSMs + sm; += kNumSMs*8``): 1712-1809
       lane-per-slot expert ids, ``ballot``, peer-flag poll
       (``ld.acquire.sys.b64 == launch tag`` (= ``%gridid + 1``), ``__all_sync``, 300 s timeout),
       1D TMA loads (EVICT_FIRST) of the 5120 B chunk of each set bit,
       lowest bit first (ascending top-k slot, shared slot last), double
       buffered, FP32 accumulate with ``add.rn.f32.bf16``, ``cvt.rn.bf16x2.f32``,
       ``st.shared.v4``, explicit-EVICT_NORMAL 1D TMA store of ``y``
=====  ============================================================  ===========

The 3 buffer slots per epilogue warp (2 load stages + 1 store) overlay the reusable shared memory from offset 0 (K:1677-1680); the mbarriers
``combine_barriers[2*warp + stage]`` are 1-arrival barriers armed with ``arrive.expect_tx`` after each 1D load (K:1748-1749).  The phase
bookkeeping (``combine_phase ^= load_stage_idx; load_stage_idx ^= 1``) is carried across token chunks exactly as in the C++.

Only the documented ``ctx`` fields are used: ``cfg, sm_idx, warp_idx, lane_idx, num_tokens, y, sym, smem, mb`` (helper objects are read into
locals / Int32 shared addresses before any dynamic region, see the DSL note in ``dispatch.py``).
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Float32, Int32, Int64

from . import ptx as X
from . import ptx_d as D
from .config import BAR_DISPATCH_EPILOGUE, BAR_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS
from .dispatch import grid_sync, launch_tag_of, make_addrs
from .trace import ACC_COMBINE_PEER_WAIT, SLOT_CLOCK_END, SLOT_COMBINE_END, SLOT_COMBINE_START  # DeepGEMM-compatible phase trace (no-ops unless cfg.trace)


@cute.jit
def _move_mask_and_load(mask, stage_idx, token_idx, chunk_byte_offset, lane, A, base_addr, mbar_base_addr,
                        chunk_bytes: cutlass.Constexpr[int], num_epilogue_warps: cutlass.Constexpr[int], ewarp):
    """``move_mask_and_load`` (K:1736-1755): pop the lowest set bit of ``mask`` (= top-k slot, ascending; the shared slot is
    the last bit), issue the 1D TMA load of that slot's chunk into load stage ``stage_idx`` and arm its mbarrier.
    Returns ``(new_mask, loaded)`` with ``loaded`` = 1 if a load was issued."""
    loaded = Int32(0)
    if mask != Int32(0):
        slot_idx = X.ffs(mask) - Int32(1)  # __ffs(mask) - 1
        mask = mask ^ (Int32(1) << slot_idx)
        with cute.arch.elect_one():
            src_ptr = A.combine_token(slot_idx, token_idx) + Int64(chunk_byte_offset)
            dst = base_addr + (ewarp + stage_idx * Int32(num_epilogue_warps)) * Int32(chunk_bytes)
            bar = mbar_base_addr + stage_idx * Int32(8)
            X.tma_load_1d(dst, src_ptr, bar, Int32(chunk_bytes))
            X.Mbar(bar).arrive_expect_tx(Int32(chunk_bytes))
        cute.arch.sync_warp()
        loaded = Int32(1)
    return mask, loaded


@cute.jit
def combine_role(ctx):
    """Epilogue warps 8-15 after the GEMM phase and the TMEM free (K:1650-1809).  Executes R2 exactly once (K:1710)."""
    cfg = ctx.cfg
    R = cfg.num_ranks
    EPR = cfg.num_experts_per_rank
    H = cfg.hidden
    TOPK = cfg.num_topk
    NSM = cfg.num_sms
    NEW = cfg.num_epilogue_warps  # 8
    NET = cfg.num_epilogue_threads  # 256
    NCH = cfg.num_combine_chunks  # kNumChunks (2)
    CHUNK_BYTES = cfg.combine_chunk_bytes  # kNumChunkBytes (5120)
    NU4 = cfg.combine_chunk_uint4_per_lane  # kNumUint4PerLane (10)
    HAS_SHARED = cfg.num_shared_experts > 0
    assert cfg.num_epilogue_threads == 256 and R <= NET and TOPK + (1 if HAS_SHARED else 0) <= 32
    assert cfg.smem_reusable_bytes >= 3 * NEW * CHUNK_BYTES, "combine overlay does not fit the reusable shared memory"

    sm_idx = ctx.sm_idx
    warp_idx = ctx.warp_idx
    lane = ctx.lane_idx
    num_tokens = ctx.num_tokens
    sym = ctx.sym
    my_rank = sym.rank_idx
    A = make_addrs(sym.base, cfg)
    y_addr = ctx.y.toint()  # BF16 y (num_tokens, H) row-major
    tr = getattr(ctx, "tr", None)  # the Trace handle (None in contexts without one); every call compiles to nothing unless cfg.trace
    ewarp = warp_idx - Int32(NEW)  # epilogue_warp_idx
    ethread = ewarp * Int32(32) + lane  # epilogue_thread_idx

    # K:1677-1685: per-warp buffers (2 load stages + 1 store slot of kNumChunkBytes) in the reusable smem from offset 0, and the
    # two per-warp mbarriers combine_barriers[2*warp + stage] (all as 32-bit shared::cta addresses)
    smem_base_addr = ctx.smem.base.toint()
    store_addr = smem_base_addr + (ewarp + Int32(2 * NEW)) * Int32(CHUNK_BYTES)
    mbar_base_addr = ctx.mb.combine(ewarp * Int32(2)).addr  # &combine_barriers[2*warp]; stage s at + 8*s
    combine_phase = Int32(0)
    load_stage_idx = Int32(0)
    # FP32 accumulators of one chunk (80 values = 40 float2 for H=5120): allocated ONCE here (a per-iteration alloca inside the
    # dynamic loop would not be promoted to registers), zeroed per chunk below
    acc = cute.make_rmem_tensor((NU4 * 8,), Float32)

    # K:1690-1694 peers' grid indices for tagging their combine readiness; the load overlaps with the grid sync
    peer_grid_idx = Int64(0)
    if sm_idx == Int32(0) and ethread < Int32(R):
        peer_grid_idx = X.ld_b64(A.peer_grid_idx(ethread))

    # K:1696-1700 all local L2 writes are done after this grid sync (epilogue counter idx 1, 256 epilogue threads, barrier id 2)
    grid_sync(A.grid_sync_count(1), NSM, sm_idx, ethread, BAR_EPILOGUE, NET)

    if tr is not None:  # K:1702-1703 trace slot 5: combine start (epilogue warp 0 lane 0, after the epilogue grid sync)
        tr.stamp(SLOT_COMBINE_START, (ewarp == Int32(0)) & (lane == Int32(0)))

    # K:1705-1707 notify remote ranks; ordered before the cleanup barrier by the dispatch/epilogue sync below
    if sm_idx == Int32(0) and ethread < Int32(R):
        X.st_release_sys_b64(sym.map(A.combine_ready_grid_idx(my_rank), ethread), peer_grid_idx)

    # K:1709-1710 barrier with dispatch warps (R2), so that they can clean the workspace; the combine reuses their smem
    X.bar_sync_unaligned(BAR_DISPATCH_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS)

    # K:1712-1809 iterate over all token chunks, 1 token 1 topk latency: ~3 us
    grid_idx = launch_tag_of(ctx)  # K:1713 `ptx::get_grid_idx() + 1` (the ctx.launch_tag; see dispatch.py)
    token_chunk_idx = ewarp * Int32(NSM) + sm_idx
    while token_chunk_idx < num_tokens * Int32(NCH):
        token_idx = token_chunk_idx // Int32(NCH)
        chunk_idx = token_chunk_idx % Int32(NCH)

        # K:1718-1722 read top-k slot indices: each lane reads one slot (int64 -> low 32 bits), lane TOPK = shared slot
        stored_slot = Int32(-1)
        if lane < Int32(TOPK):
            stored_slot = X.ld_b32(A.input_topk_idx(token_idx * Int32(TOPK) + lane))
        elif HAS_SHARED and lane == Int32(TOPK):
            stored_slot = Int32(TOPK)
        total_mask = X.ballot(stored_slot >= Int32(0))

        # K:1724-1730 wait for the ranks of the selected experts to finish their L2 writes (300 s timeout)
        is_routed = Int32(0)
        peer_idx = Int32(0)
        if lane < Int32(TOPK) and stored_slot >= Int32(0):
            is_routed = Int32(1)
            peer_idx = stored_slot // Int32(EPR)
        t_peer0 = tr.clock() if tr is not None else Int64(0)
        D.vote_all_acquire_sys_b64_eq(A.combine_ready_grid_idx(peer_idx), grid_idx, is_routed)
        if tr is not None:  # K:1728 trace slot 40: combine peer wait (epilogue warp 0 lane 0)
            if (ewarp == Int32(0)) & (lane == Int32(0)):
                tr.add_since(ACC_COMBINE_PEER_WAIT, t_peer0)

        chunk_byte_offset = chunk_idx * Int32(CHUNK_BYTES)

        # K:1757-1758 load the first selection
        mask = total_mask
        mask, do_reduce = _move_mask_and_load(mask, load_stage_idx, token_idx, chunk_byte_offset, lane, A, smem_base_addr,
                                              mbar_base_addr, CHUNK_BYTES, NEW, ewarp)  # fmt: skip

        # K:1760-1779 accumulate all top-k contributions for this chunk in FP32 registers (80 values for H=5120)
        for q in cutlass.range_constexpr(NU4 * 8):
            acc[q] = Float32(0.0)
        while do_reduce != Int32(0):
            # prefetch next top-k into the other buffer while the current one is being accumulated
            mask, do_reduce = _move_mask_and_load(mask, load_stage_idx ^ Int32(1), token_idx, chunk_byte_offset, lane, A,
                                                  smem_base_addr, mbar_base_addr, CHUNK_BYTES, NEW, ewarp)  # fmt: skip
            # accumulate
            X.Mbar(mbar_base_addr + load_stage_idx * Int32(8)).wait(combine_phase)
            buf_addr = smem_base_addr + (ewarp + load_stage_idx * Int32(NEW)) * Int32(CHUNK_BYTES)
            for j in cutlass.range_constexpr(NU4):
                w = X.ld_shared_v4_u32(buf_addr + (Int32(j * 32) + lane) * Int32(16))  # uint4 = 4 x bf16x2
                for l in cutlass.range_constexpr(4):
                    a0, a1 = X.accumulate_bf16x2(acc[(j * 4 + l) * 2], acc[(j * 4 + l) * 2 + 1], w[l])
                    acc[(j * 4 + l) * 2] = a0
                    acc[(j * 4 + l) * 2 + 1] = a1
            X.fence_proxy_async_shared_cta()  # cutlass::arch::fence_view_async_shared()
            combine_phase = combine_phase ^ load_stage_idx
            load_stage_idx = load_stage_idx ^ Int32(1)

        # K:1781-1798 cast to BF16 (RN) and write the chunk into the store slot
        for j in cutlass.range_constexpr(NU4):
            c0 = X.cvt_rn_bf16x2_f32(acc[(j * 4 + 0) * 2 + 1], acc[(j * 4 + 0) * 2])
            c1 = X.cvt_rn_bf16x2_f32(acc[(j * 4 + 1) * 2 + 1], acc[(j * 4 + 1) * 2])
            c2 = X.cvt_rn_bf16x2_f32(acc[(j * 4 + 2) * 2 + 1], acc[(j * 4 + 2) * 2])
            c3 = X.cvt_rn_bf16x2_f32(acc[(j * 4 + 3) * 2 + 1], acc[(j * 4 + 3) * 2])
            # wait for the previous store to release the shared memory, then write
            if j == 0:
                X.tma_store_wait(0)
                cute.arch.sync_warp()
            X.st_shared_v4_u32(store_addr + (Int32(j * 32) + lane) * Int32(16), c0, c1, c2, c3)
        cute.arch.sync_warp()

        # K:1800-1808 TMA store the token chunk (explicit EVICT_NORMAL hint, no wait: the CTA stays alive through cleanup)
        with cute.arch.elect_one():
            X.fence_proxy_async_shared_cta()  # cute::tma_store_fence
            X.tma_store_1d(y_addr + Int64(token_idx) * Int64(H * 2) + Int64(chunk_byte_offset), store_addr, Int32(CHUNK_BYTES))
            X.tma_store_commit()  # cute::tma_store_arrive
        cute.arch.sync_warp()

        token_chunk_idx = token_chunk_idx + Int32(NSM * NEW)

    if tr is not None:  # K:1810-1811 trace slots 6 (+ 61 clock64): combine end
        tr.stamp(SLOT_COMBINE_END, (ewarp == Int32(0)) & (lane == Int32(0)))
        tr.stamp_clock(SLOT_CLOCK_END, (ewarp == Int32(0)) & (lane == Int32(0)))

    # staged-test hook (never set by the production kernel): combine-only tests have no dispatch cleanup keeping the CTA alive
    if cutlass.const_expr(getattr(ctx, "d_test_combine_tail_wait", False)):
        X.tma_store_wait(0)
