# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / mma/sm100.cuh / ptx/tcgen05.cuh; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION & AFFILIATES).
"""MMA issue warp (warp 6, leader CTA only), K:1019-1202.

Swap-AB 2-CTA ``tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale`` with the Rubin K=64 instruction (idesc bit 31): the packed-FP4
weights (256 rows of the CTA pair) are the UMMA-A operand, the FP8 tokens (``N = align16(valid_m)`` rows, split ``N/2`` + ``N/2``
across the two CTAs' shared memory) are UMMA-B.  Per 128-K stage: UTCCP copies of the token SF (``SF_BLOCK_M/128`` x 128 words ->
TMEM column ``kTmemStartColOfSFA + 4 i``) and the weight SF (1 x 128 words -> ``kTmemStartColOfSFB``), then 2 UMMAs (k = 0, 1) with
SF ids 0 / 2 and smem descriptor low words advanced by ``+4`` (tokens, 64 B >> 4) and ``+2`` (packed FP4 weights, 32 B >> 4).
Then ``tcgen05.commit ... multicast::cluster`` (CTA mask 0b11) to ``empty[stage]`` and, on the last k-block, to ``tmem_full[accum_stage]``.
"""

SUPPORTS_SHARED_EXPERTS = True  # DESIGN.md section 17: this module implements the shared-expert (S > 0) paths

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from . import ptx
from . import ptx_g
from . import scheduler as sched
from .scheduler import PHASE_NONE, PHASE_SHARED_LINEAR1

_BLOCK_K = 128  # UMMA_BLOCK_K: two UMMA_K = 64 per stage
_SHARED_BLOCK_K = 64  # half-K shared stage (kSharedHalfK): one UMMA_K = 64 per stage


@cute.jit
def _mma_k_loop(ctx, num_k: Int32, accum_stage: Int32, idesc_base, stage: Int32, phase: Int32, kblk: Int32):
    """K loop of one task (K:1103-1191).  ``kblk`` = running k-block count of this warp (``trace_kblk_count``, only advanced in trace builds)."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    a_stage_bytes = cfg.load_block_m * _BLOCK_K  # sizeof(smem_a[0])
    b_stage_bytes = 128 * _BLOCK_K // 2  # sizeof(smem_b[0]): packed FP4 (8192 B)
    a_lo0 = ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.a(0)))
    b_lo0 = ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.b(0)))
    tmem_c = accum_stage * cfg.umma_n  # accum stage column: accum_stage_idx * UMMA_N

    for k_block_idx in cutlass.range(num_k, unroll=2):  # DeepGEMM: `#pragma unroll 2` (K:1104)
        tc = ctx.tr.clock()
        ctx.mb.full(stage).wait(phase)  # wait TMA load completion
        sched.trace_since(ctx, 9, tc, rec)
        sched.trace_event(ctx, 1, kblk, rec)
        ptx.tcgen05_fence_after_thread_sync()
        a_base_lo = a_lo0 + stage * (a_stage_bytes // 16)  # tokens descriptor (UMMA-B)
        b_base_lo = b_lo0 + stage * (b_stage_bytes // 16)  # weights descriptor (UMMA-A)
        with cute.arch.elect_one():
            # UTCCP copy SFA (tokens) and SFB (weights) to TMEM (K:1148-1160); SFA first
            for i in cutlass.range_constexpr(cfg.sf_block_m // 128):
                d = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SF_SBO128, ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.sfa(stage)) + i * 512))
                ptx_g.utccp_2cta(ctx.tmem.sfa_col + 4 * i, d)
            for i in cutlass.range_constexpr(cfg.sf_block_n // 128):
                d = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SF_SBO128, ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.sfb(stage)) + i * 512))
                ptx_g.utccp_2cta(ctx.tmem.sfb_col + 4 * i, d)
            # 2 UMMAs per 128-K stage (K:1163-1180); each consumes kNumSFPerUMMA = 2 consecutive UE8M0 bytes of the 4 per column
            for k in cutlass.range_constexpr(_BLOCK_K // cfg.umma_k):
                idesc = idesc_base | ((k * 2) << 29) | ((k * 2) << 4)  # a_sf_id = b_sf_id = k * kNumSFPerUMMA (K:1166-1167)
                tok_desc = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SW128_SBO1024, a_base_lo + 4 * k)
                w_desc = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SW64_SBO512, b_base_lo + 2 * k)
                if cutlass.const_expr(k == 0):
                    scale_c = Int32(k_block_idx > 0)  # accumulate flag: k_block_idx > 0 or umma_k_block_idx > 0 or k > 0
                else:
                    scale_c = Int32(1)
                # fma(b_desc (weights), a_desc (tokens), accum_stage_idx * UMMA_N, ..., kTmemStartColOfSFB, kTmemStartColOfSFA)
                ptx_g.mma_mxf8f6f4_2cta(tmem_c, w_desc, tok_desc, idesc, scale_c, ctx.tmem.sfb_col, ctx.tmem.sfa_col)
        cute.arch.sync_warp()
        # commit to the mbarrier objects (no explicit tcgen05.fence::before_thread_sync: implied by tcgen05.commit)
        with cute.arch.elect_one():
            ptx_g.umma_arrive_multicast_2x1sm(ctx.mb.empty(stage).addr, 0b11)
        if k_block_idx == num_k - 1:
            with cute.arch.elect_one():
                ptx_g.umma_arrive_multicast_2x1sm(ctx.mb.tmem_full(accum_stage).addr, 0b11)  # the accumulator pipeline is not multicast
        cute.arch.sync_warp()
        sched.trace_event(ctx, 2, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


@cute.jit
def _mma_k_loop_shared(ctx, num_k: Int32, accum_stage: Int32, idesc_base, stage: Int32, phase: Int32, kblk: Int32):
    """K loop of one SHARED-expert task (K:1115-1145, ``half_k``): per 64-K stage ONE UMMA on SWIZZLE_64B tiles (tokens and FP8 weights, SBO 512 B), E4M3 x E4M3
    instruction descriptor (``idesc_base = IDESC_BASE_SHARED | N``), SF bytes {0, 1} / {2, 3} of the SF word (``sf_id = (k_block * 64 % 128) / 32``), the SF
    UTCCP copy is redone every stage (the same SF word serves two consecutive half-K stages)."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    a_stage_bytes = cfg.load_block_m * _BLOCK_K  # sizeof(smem_a[0]) (unchanged stage stride)
    b_stage_bytes = 128 * _BLOCK_K // 2  # sizeof(smem_b[0]) = 8192 B
    a_lo0 = ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.a(0)))
    b_lo0 = ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.b(0)))
    tmem_c = accum_stage * cfg.umma_n

    for k_block_idx in cutlass.range(num_k, unroll=2):  # DeepGEMM: the same `#pragma unroll 2` loop covers the half-K shared path (K:1104)
        tc = ctx.tr.clock()
        ctx.mb.full(stage).wait(phase)
        sched.trace_since(ctx, 9, tc, rec)
        sched.trace_event(ctx, 1, kblk, rec)
        ptx.tcgen05_fence_after_thread_sync()
        a_base_lo = a_lo0 + stage * (a_stage_bytes // 16)
        b_base_lo = b_lo0 + stage * (b_stage_bytes // 16)
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(cfg.sf_block_m // 128):
                d = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SF_SBO128, ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.sfa(stage)) + i * 512))
                ptx_g.utccp_2cta(ctx.tmem.sfa_col + 4 * i, d)
            for i in cutlass.range_constexpr(cfg.sf_block_n // 128):
                d = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SF_SBO128, ptx_g.smem_desc_lo(ctx.smem.addr(ctx.smem.sfb(stage)) + i * 512))
                ptx_g.utccp_2cta(ctx.tmem.sfb_col + 4 * i, d)
            sf_id = ((k_block_idx * _SHARED_BLOCK_K) % 128) // 32  # + k * kNumSFPerUMMA with k = 0
            idesc = idesc_base | (sf_id << 29) | (sf_id << 4)
            tok_desc = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SW64_SBO512, a_base_lo)
            w_desc = ptx_g.make_desc64(ptx_g.SMEM_DESC_HI_SW64_SBO512, b_base_lo)
            scale_c = Int32(k_block_idx > 0)  # accumulate flag: k_block_idx > 0 or k > 0
            ptx_g.mma_mxf8f6f4_2cta(tmem_c, w_desc, tok_desc, idesc, scale_c, ctx.tmem.sfb_col, ctx.tmem.sfa_col)
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            ptx_g.umma_arrive_multicast_2x1sm(ctx.mb.empty(stage).addr, 0b11)
        if k_block_idx == num_k - 1:
            with cute.arch.elect_one():
                ptx_g.umma_arrive_multicast_2x1sm(ctx.mb.tmem_full(accum_stage).addr, 0b11)
        cute.arch.sync_warp()
        sched.trace_event(ctx, 2, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


@cute.jit
def mma_role(ctx):
    """MMA warp body (leader CTA only; the kernel guards the call)."""
    cfg = ctx.cfg
    sched_stage = Int32(0)
    sched_phase = Int32(0)
    stage = Int32(0)
    phase = Int32(0)
    current_iter_idx = Int32(0)
    kblk = Int32(0)
    rec = ctx.lane_idx == 0  # trace recorder lane
    tc = ctx.tr.clock()
    t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
    sched.trace_since(ctx, 12, tc, rec)
    while t0 != PHASE_NONE:
        valid_m = t5
        num_k = t7 // _BLOCK_K  # task_info.shape_k / BLOCK_K (K:1070)
        # dynamic update of UMMA N from the effective M: n_dim = align16(valid_m) >> 3
        aligned_n = (valid_m + 15) // 16 * 16
        idesc_base = Int32(ptx_g.IDESC_BASE_ROUTED) | ((aligned_n >> 3) << 17)
        if cutlass.const_expr(cfg.num_shared_experts > 0):
            is_shared = t0 >= PHASE_SHARED_LINEAR1
            if is_shared:
                num_k = t7 // _SHARED_BLOCK_K  # half-K stages
                idesc_base = Int32(ptx_g.IDESC_BASE_SHARED) | ((aligned_n >> 3) << 17)  # E4M3 weights
        # wait tensor memory empty barrier arrival
        accum_stage = current_iter_idx % cfg.accum_stages
        accum_phase = (current_iter_idx // cfg.accum_stages) & 1
        current_iter_idx = current_iter_idx + 1
        ctx.tr.stamp(2, rec & (current_iter_idx == 1))  # first MMA task starts
        tc = ctx.tr.clock()
        ctx.mb.tmem_empty(accum_stage).wait(accum_phase ^ 1)
        sched.trace_since(ctx, 8, tc, rec)
        ptx.tcgen05_fence_after_thread_sync()
        sched.trace_count(ctx, 11, rec)
        if cutlass.const_expr(cfg.num_shared_experts > 0):
            if is_shared:
                stage, phase, kblk = _mma_k_loop_shared(ctx, num_k, accum_stage, idesc_base, stage, phase, kblk)
            else:
                stage, phase, kblk = _mma_k_loop(ctx, num_k, accum_stage, idesc_base, stage, phase, kblk)
        else:
            stage, phase, kblk = _mma_k_loop(ctx, num_k, accum_stage, idesc_base, stage, phase, kblk)
        tc = ctx.tr.clock()
        t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
        sched.trace_since(ctx, 12, tc, rec)

    ctx.tr.stamp(3, rec)  # MMA done

    # to safely deconstruct barriers, we need another round of waits (K:1197-1201)
    if current_iter_idx > 0:
        last = current_iter_idx - 1
        ctx.mb.tmem_empty(last % cfg.accum_stages).wait((last // cfg.accum_stages) & 1)
