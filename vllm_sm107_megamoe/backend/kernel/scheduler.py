# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / scheduler/{sm107_mega_moe,mega_moe}.cuh.
"""Task scheduling: the consumer side of the 2-slot ``task_infos`` ring and the scheduler warp (warp 7 of the leader CTA).

Mirrors ``sched::MegaMoESchedulerLD`` (``SCH`` = ``scheduler/sm107_mega_moe.cuh``, ``SCH0`` = ``scheduler/mega_moe.cuh``).

* **TaskInfo** (32 B, ``alignas(16)``): eight u32 words ``[block_phase, local_expert_idx, m_block_idx, n_cluster_idx, pool_block_idx,
  valid_m, shape_n, shape_k]`` (``SCH0:91-134``).  The scheduler warp publishes every task to BOTH CTAs of the pair
  (``publish_task``, SCH:261-271); TMA-A, TMA-B, the MMA warp (leader only) and the epilogue warps each run their own private
  2-stage copy of ``get_next_task`` (SCH:100-105) over that ring.
* Consumer-side objects are loop-carried as plain Int32 locals (``sched_stage``, ``sched_phase``) by the role functions.
"""

SUPPORTS_SHARED_EXPERTS = True  # DESIGN.md section 17: this module implements the shared-expert (S > 0) paths

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Int64

from . import ptx
from . import scheduler_perdie as pd  # per-die (locality-domain) claim / steal / L2-wait helpers (identical to the single-queue code for one domain)

# BlockPhase (SCH0:83-89)
PHASE_NONE = 0
PHASE_LINEAR1 = 1
PHASE_LINEAR2 = 2
PHASE_SHARED_LINEAR1 = 3
PHASE_SHARED_LINEAR2 = 4

TASK_INFO_BYTES = 32


@cute.jit
def trace_since(ctx, slot: cutlass.Constexpr, t0, rec):
    """``DG_KMOE_TIMED`` epilogue (K:138-139): ``trace_acc[slot] += clock64() - t0`` recorded by the ``rec`` thread only (``t0 = ctx.tr.clock()`` taken before
    the timed region).  Compiles to nothing unless ``cfg.trace``."""
    if cutlass.const_expr(ctx.tr.enabled):
        if rec:
            ctx.tr.add_since(slot, t0)


@cute.jit
def trace_count(ctx, slot: cutlass.Constexpr, rec):
    """``trace_add(slot, 1)`` by the ``rec`` thread (task counters 11 / 28 / 29)."""
    if cutlass.const_expr(ctx.tr.enabled):
        if rec:
            ctx.tr.add(slot, 1)


@cute.jit
def trace_event(ctx, ev: cutlass.Constexpr, kblk, rec):
    """``trace_event(ev, trace_kblk_count)`` (K:132-137): clock64 into the k-block event log window, by the ``rec`` thread."""
    if cutlass.const_expr(ctx.tr.enabled):
        if rec:
            ctx.tr.event(ev, kblk)


@cute.jit
def advance_sched_stage(sched_stage: Int32, sched_phase: Int32):
    """``advance_sched_pipeline`` (SCH:94-98): 2 stages, ``phase ^= (stage == 0)`` after the toggle."""
    sched_stage = sched_stage ^ 1
    if sched_stage == 0:
        sched_phase = sched_phase ^ 1
    return sched_stage, sched_phase


@cute.jit
def get_next_task(ctx, sched_stage: Int32, sched_phase: Int32):
    """Consumer ``get_next_task`` (SCH:100-105): wait ``task_info_full[stage]``, copy the 32 B task out of smem.

    Returns ``(w0..w7, sched_stage, sched_phase)``; ``w0 = block_phase`` (0 = sentinel = no more tasks)."""
    ctx.mb.task_info_full(sched_stage).wait(sched_phase)
    p = ctx.smem.task_info(sched_stage)  # u32 pointer to the slot
    w0, w1, w2, w3 = ptx.ld_shared_v4_u32(p)
    w4, w5, w6, w7 = ptx.ld_shared_v4_u32(p + 4)
    sched_stage, sched_phase = advance_sched_stage(sched_stage, sched_phase)
    return w0, w1, w2, w3, w4, w5, w6, w7, sched_stage, sched_phase


@cute.jit
def release_task_info(ctx, sched_stage: Int32):
    """``release_task_info`` (SCH:107-111), called by every epilogue thread after ``tmem_full`` (K:1308): complete the
    metadata reads (``fence.acq_rel.cta``) then arrive remotely on CTA 0's ``task_info_empty[sched_stage ^ 1]``
    (count 512 = 2 CTAs x 256 epilogue threads).  ``sched_stage`` is the epilogue's CURRENT (already advanced) stage."""
    ptx.fence_acq_rel_cta()
    ctx.mb.task_info_empty(sched_stage ^ 1).arrive_remote(0)


@cute.jit
def publish_task(ctx, w0, w1, w2, w3, w4, w5, w6, w7, sched_stage: Int32, sched_phase: Int32):
    """Scheduler warp ``publish_task`` (SCH:261-271); all 32 lanes execute it.

    Lane ``c`` (c < 2): remote ``mbarrier.arrive.expect_tx(32)`` on CTA c's ``task_info_full[stage]`` then two
    ``st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`` stores (16 B each) of the 32 B task into CTA c's slot."""
    lane = ctx.lane_idx
    if lane < 2:
        ctx.mb.task_info_full(sched_stage).arrive_expect_tx_remote(TASK_INFO_BYTES, lane)
        dst = ctx.smem.task_info(sched_stage)
        full = ctx.mb.task_info_full(sched_stage)
        # st.async v4 x2: dst and mbarrier are both mapped (mapa) to CTA `lane`'s shared memory (ptx::st_async_cluster)
        ptx.st_async_v4(dst, w0, w1, w2, w3, full, lane)
        ptx.st_async_v4(dst + 4, w4, w5, w6, w7, full, lane)
    cute.arch.sync_warp()
    sched_stage, sched_phase = advance_sched_stage(sched_stage, sched_phase)
    return sched_stage, sched_phase


# ----------------------------------------------------------------------------------------------------------------------
# Scheduler warp (warp 7 of the leader CTA): task claiming (SCH:113-259) and the publish loop (SCH:290-328)
# ----------------------------------------------------------------------------------------------------------------------
WAVES_DONE = -1  # kNumSchedL1WavesDone = 0xffffffff (as a signed 32-bit pattern)


@cute.jit
def _claim(ptr):
    """``get_next_local_task_idx`` (SCH:196-201): the elected lane does ``atom.global.add.u32(counter, 1)``, the old value is broadcast."""
    result = Int32(0)
    if ptx.elect_sync():
        result = ptx.atom_add_u32(ptr, 1)
    return ptx.shfl_idx(result, 0)


@cute.jit
def _warp_inclusive_sum(value, lane):
    """``math::warp_inclusive_sum``: Hillis-Steele with ``shfl.up``."""
    for i in cutlass.range_constexpr(5):
        synced = ptx.shfl_up(value, 1 << i)
        value = Int32(cutlass.select_(lane >= (1 << i), value + synced, value))
    return value


@cute.jit
def _create_task(ctx, stored, pool_m_block, block_m: cutlass.Constexpr):
    """``create_task`` (SCH:161-194): map a pool (global M) block to ``(local_expert, m_block_in_expert, valid_m)`` with a warp scan of the
    per-expert block counts held in the per-lane arrays ``stored`` (expert ``i*32 + lane`` in ``stored[i]``)."""
    cfg = ctx.cfg
    lane = ctx.lane_idx
    local_expert = Int32(0)
    m_block = Int32(0)
    valid_m = Int32(0)
    block_offset = Int32(0)
    for i in cutlass.range_constexpr(cfg.num_experts_per_lane):
        expert_idx = lane + i * 32
        num_tokens = stored[i]
        num_m_blocks = (num_tokens + (block_m - 1)) // block_m
        inclusive = _warp_inclusive_sum(num_m_blocks, lane)
        lane_pool_block_offset = block_offset + inclusive - num_m_blocks
        is_owner = (expert_idx < cfg.num_experts_per_rank) & (pool_m_block >= lane_pool_block_offset) & (
            pool_m_block < lane_pool_block_offset + num_m_blocks)
        owner_mask = ptx.ballot(is_owner)
        if owner_mask != 0:
            owner_lane = ptx.ffs(owner_mask) - 1  # __ffs(mask) - 1
            owner_m_block_idx = pool_m_block - lane_pool_block_offset
            rem = num_tokens - owner_m_block_idx * block_m
            owner_valid_m = Int32(cutlass.select_(rem < block_m, rem, Int32(block_m)))  # min(rem, BLOCK_M)
            local_expert = ptx.shfl_idx(expert_idx, owner_lane)
            m_block = ptx.shfl_idx(owner_m_block_idx, owner_lane)
            valid_m = ptx.shfl_idx(owner_valid_m, owner_lane)
        block_offset = block_offset + ptx.shfl_idx(inclusive, 31)
    return local_expert, m_block, valid_m


@cute.jit
def _get_next_task(ctx, stored, waves: Int32, num_total_m_blocks: Int32, dom):
    """``get_next_task`` (SCH:224-259); ``dom`` = this SM's locality domain (``pd.load_domain``, unused for one domain).  Returns ``(valid, w0..w7, waves)``; the C++ ``while(true)`` with ``continue``/``break``
    is a predicate-carried loop.  Claim order: ``waves`` warm-up L1 claims, then strict L2/L1 alternation (one L1 claim after every L2 claim) until the
    L1 queue is exhausted (``waves = DONE``), then L2 only."""
    cfg = ctx.cfg
    l1n, l2n = cfg.num_l1_clusters, cfg.num_l2_clusters
    valid = Boolean(False)
    active = Boolean(True)
    t_phase = Int32(PHASE_NONE)
    t_expert = Int32(0)
    t_m = Int32(0)
    t_n = Int32(0)
    t_pool = Int32(0)
    t_valid_m = Int32(0)
    t_shape_n = Int32(0)
    t_shape_k = Int32(0)
    while active:
        if (waves != WAVES_DONE) & (waves != 0):
            waves = waves - 1  # one local L1 task per scheduler; globally one CTA-pair wave
            ok, m_block_idx, n_cluster_idx = pd.claim(ctx, ctx.ws.l1_task_count(0), dom, l1n, num_total_m_blocks)  # N-cluster fastest, M-block major; per-domain queues + stealing (SCH:203-222)
            if not ok:  # no more L1 tasks
                waves = Int32(WAVES_DONE)
            else:
                t_expert, t_m, t_valid_m = _create_task(ctx, stored, m_block_idx, cfg.block_m)
                t_phase = Int32(PHASE_LINEAR1)
                t_n = n_cluster_idx
                t_pool = m_block_idx
                t_shape_n = Int32(cfg.l1_shape_n)
                t_shape_k = Int32(cfg.l1_shape_k)
                valid = Boolean(True)
                active = Boolean(False)
        else:
            ok, m_block_idx, n_cluster_idx = pd.claim(ctx, ctx.ws.l2_task_count(0), dom, l2n, num_total_m_blocks)
            if not ok:
                active = Boolean(False)  # break: no more tasks
            else:
                if waves != WAVES_DONE:  # the next task should be L1
                    waves = Int32(1)
                t_expert, t_m, t_valid_m = _create_task(ctx, stored, m_block_idx, cfg.block_m)
                t_phase = Int32(PHASE_LINEAR2)
                t_n = n_cluster_idx
                t_pool = m_block_idx
                t_shape_n = Int32(cfg.l2_shape_n)
                t_shape_k = Int32(cfg.l2_shape_k)
                # wait until all required L1 tasks are fetched from all scheduling queues (SCH:250-254)
                pd.wait_l1_claimed(ctx, m_block_idx)  # the L1 counters of ALL domains >= (pool_block + 1) * (l1n / D)
                valid = Boolean(True)
                active = Boolean(False)
    return valid, t_phase, t_expert, t_m, t_n, t_pool, t_valid_m, t_shape_n, t_shape_k, waves


@cute.jit
def _shared_mainloop(ctx, phase: cutlass.Constexpr, shape_n: cutlass.Constexpr, shape_k: cutlass.Constexpr, counter_ptr, num_m_blocks, sched_stage, sched_phase, dom):
    """``shared_mainloop<phase, kShapeN, kShapeK>`` (SCH:274-288): dynamic claims over ``(m_block, n_cluster)`` of the shared GEMM
    (m-block major, N-cluster fastest): ``wait task_info_empty[slot]; claim; publish [phase, 0, m, n, m, valid_m, N, K]``; ``pool_block == m_block`` (shared tasks
    have no expert and no ring), ``valid_m = min(num_tokens - m * BLOCK_M, BLOCK_M)``."""
    cfg = ctx.cfg
    nclusters = shape_n // 256  # kNumNClusters = kShapeN / BLOCK_N / 2
    more = Boolean(True)
    while more:
        ctx.mb.task_info_empty(sched_stage).wait(sched_phase ^ 1)
        ok, m_block_idx, n_cluster_idx = pd.claim(ctx, counter_ptr, dom, nclusters, num_m_blocks)
        if not ok:  # get_next_task_idx fails: queue drained
            more = Boolean(False)
        else:
            rem = ctx.num_tokens - m_block_idx * cfg.block_m
            valid_m = Int32(cutlass.select_(rem < cfg.block_m, rem, Int32(cfg.block_m)))  # cute::min(num_tokens - m * BLOCK_M, BLOCK_M)
            z = Int32(0)
            sched_stage, sched_phase = publish_task(ctx, Int32(phase), z, m_block_idx, n_cluster_idx, m_block_idx, valid_m, Int32(shape_n), Int32(shape_k),
                                                    sched_stage, sched_phase)
    return sched_stage, sched_phase


@cute.jit
def scheduler_role(ctx):
    """Scheduler warp (warp 7, leader CTA only): ``mainloop`` (SCH:290-328) for S = 0, one locality domain.

    Order: wait for every expert's receive count (``hi32 == kNumSMs * kNumRanks``), compute the warm-up waves, then loop
    ``wait task_info_empty[slot]; claim; publish`` until the claim fails, then publish the sentinel (``BlockPhase::None``)."""
    cfg = ctx.cfg
    lane = ctx.lane_idx
    sched_stage = Int32(0)
    sched_phase = Int32(0)
    dom = pd.load_domain(ctx)  # sm_locality_domain_idx = sm_locality_domains[%smid] (SCH:292); 0 for one domain
    pd.trace_domain(ctx, dom)  # cfg.trace only

    # shared-expert phases (S > 0, SCH:296-304): the shared L1 tasks do not depend on the dispatch; the shared L2 tasks go BEFORE the routed tasks when one wave
    # fits the dispatch gap (`ceil(num_tokens / BLOCK_M) * kNumSharedL2Clusters <= kNumSMs / 2`), else at the tail
    if cutlass.const_expr(cfg.num_shared_experts > 0):
        num_m_blocks_shared = (ctx.num_tokens + (cfg.block_m - 1)) // cfg.block_m
        shared_l2_early = num_m_blocks_shared * (cfg.l2_shape_n // 256) <= (cfg.num_sms // 2)
        sched_stage, sched_phase = _shared_mainloop(ctx, PHASE_SHARED_LINEAR1, cfg.l1_shape_n * cfg.num_shared_experts, cfg.l1_shape_k,
                                                    ctx.ws.shared_l1_task_count(0), num_m_blocks_shared, sched_stage, sched_phase, dom)
        if shared_l2_early:
            sched_stage, sched_phase = _shared_mainloop(ctx, PHASE_SHARED_LINEAR2, cfg.l2_shape_n, cfg.shared_l2_shape_k,
                                                        ctx.ws.shared_l2_task_count(0), num_m_blocks_shared, sched_stage, sched_phase, dom)

    # fetch_expert_recv_count (SCH:138-159): lane l caches experts l, l + 32, ...
    stored = []
    for i in cutlass.range_constexpr(cfg.num_experts_per_lane):
        expert_idx = lane + i * 32
        value = Int64(0)
        if expert_idx < cfg.num_experts_per_rank:
            value = ptx.ld_volatile_b64(ctx.ws.expert_recv_count_sum(expert_idx))
            while (value >> 32) != (cfg.num_sms * cfg.num_ranks):  # do { ld.volatile } while (hi32 != kNumSMs * kNumRanks)
                value = ptx.ld_volatile_b64(ctx.ws.expert_recv_count_sum(expert_idx))
        stored.append(Int32(value & 0xFFFFFFFF))
    cute.arch.sync_warp()

    # num_total_m_blocks = get_pool_block_offset(kNumExpertsPerRank) (SCH:124-136): per-lane ceil_div sums + redux.add
    nb = Int32(0)
    for i in cutlass.range_constexpr(cfg.num_experts_per_lane):
        blocks = (stored[i] + (cfg.block_m - 1)) // cfg.block_m
        nb = nb + Int32(cutlass.select_((lane + i * 32) < cfg.num_experts_per_rank, blocks, Int32(0)))
    num_total_m_blocks = ptx.redux_add_u32(nb)

    # warm-up waves (SCH:153-158, SCH0:18-45)
    nc = cfg.num_sms // 2
    l1n, l2n = cfg.num_l1_clusters, cfg.num_l2_clusters
    num_total_l1_waves = (num_total_m_blocks * l1n + (nc - 1)) // nc
    warm_first_l2 = (((nc + l2n - 1) // l2n) * l1n + nc - 1) // nc  # compile-time
    diff = l1n - l2n if l1n > l2n else 0
    warm_interleave = (l1n + (num_total_m_blocks - 1) * diff + (nc - 1)) // nc + 1
    min_warm = Int32(cutlass.select_(warm_interleave > warm_first_l2, warm_interleave, Int32(warm_first_l2)))
    waves = Int32(cutlass.select_(min_warm < num_total_l1_waves, min_warm, num_total_l1_waves))

    # generate routed tasks: keep the original wait -> claim -> publish ordering (the claim advances global counters
    # and must not run before the schedule slot is released by the consumers)
    more = Boolean(True)
    while more:
        ctx.mb.task_info_empty(sched_stage).wait(sched_phase ^ 1)
        valid, w0, w1, w2, w3, w4, w5, w6, w7, waves = _get_next_task(ctx, stored, waves, num_total_m_blocks, dom)
        if valid:
            sched_stage, sched_phase = publish_task(ctx, w0, w1, w2, w3, w4, w5, w6, w7, sched_stage, sched_phase)
        else:
            more = Boolean(False)

    if cutlass.const_expr(cfg.num_shared_experts > 0):
        if not shared_l2_early:
            sched_stage, sched_phase = _shared_mainloop(ctx, PHASE_SHARED_LINEAR2, cfg.l2_shape_n, cfg.shared_l2_shape_k,
                                                        ctx.ws.shared_l2_task_count(0), num_m_blocks_shared, sched_stage, sched_phase, dom)

    # sentinel
    ctx.mb.task_info_empty(sched_stage).wait(sched_phase ^ 1)
    z = Int32(0)
    sched_stage, sched_phase = publish_task(ctx, z, z, z, z, z, z, z, z, sched_stage, sched_phase)
