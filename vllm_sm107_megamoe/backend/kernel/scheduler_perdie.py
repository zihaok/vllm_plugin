# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) scheduler/sm107_mega_moe.cuh (MegaMoESchedulerLD).
"""Per-die (locality-domain) task queues of the scheduler warp: claim / steal / L2-wait helpers with ONE interface for ``kNumLocalityDomains`` = 1 and 2.

``SCH`` = ``scheduler/sm107_mega_moe.cuh``.  With ``cfg.num_locality_domains == 1`` every helper reduces EXACTLY to the single-queue code of ``scheduler.py``
(one counter, ``idx >= num_m_blocks * nclusters`` test, ``m = idx / nclusters``, ``n = idx % nclusters``, the L2 wait on one L1 counter); with 2 domains

* every counter family (L1, L2, shared L1, shared L2) is an array of ``kNumLocalityDomains`` contiguous ``u32`` (``counter[d * kDomainCounterStride]``, stride 1; zeroed
  by dispatch's SM 0 cleanup together with the domain-0 counters, ``K:760-773``);
* domain ``d`` OWNS the N clusters ``[d * C / D, (d + 1) * C / D)`` (``C`` = clusters of the GEMM: L1 ``2I / 256`` = 18, L2 ``H / 256`` = 20 -> 9 / 10 per domain, i.e. the
  N half whose weights are homed in domain ``d``): the claimed index ``i`` of queue ``d`` is valid iff ``i < num_m_blocks * (C / D)`` and means
  ``m_block = i / (C / D)``, ``n_cluster = d * (C / D) + i % (C / D)`` (N-cluster fastest, M-block major; ``get_next_task_idx<C>``, SCH:203-213);
* a scheduler whose own queue is drained STEALS from queue ``d ^ 1`` only if the GEMM has at least ``kNumMinStealWaves`` (= 2) waves of tasks,
  ``num_m_blocks * C >= 2 * (kNumSMs / 2)`` (SCH:216-221; "when there are few tasks, we disable work stealing to avoid cross-die latency");
* an L2 task waits until the L1 counters of BOTH domains are ``>= (pool_block + 1) * (kNumL1Clusters / D)`` (SCH:250-254), i.e. all L1 tasks of its pool block and the earlier
  ones were CLAIMED from both queues (a thief's claims advance the victim's counter).

``dom`` is read once per scheduler warp from ``sm_locality_domains[%smid]`` (:func:`perdie.load_domain`).  A balanced map (equal TPC counts per domain, what
``locality.get_balanced_sm_locality_domains`` returns) is REQUIRED: with an (almost) empty domain its queue is only drained by thieves, and below the steal threshold nobody
(or only a lone pair, which blocks itself in the L2 wait) claims it, so the L2 wait above never ends.  There is NO timeout in this wait (DeepGEMM has none either):
measured with DeepGEMM and with this port, one domain with 0 or 1 TPC hangs below the steal threshold and completes above it; a balanced map never hangs.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, Uint64

from . import ptx
from .perdie import load_domain  # re-export: ``dom = load_domain(ctx)`` at the top of ``scheduler_role``

__all__ = ["load_domain", "trace_domain", "claim", "wait_l1_claimed"]

# Per-die diagnostics in the kTrace buffer (compiled only with ``cfg.trace``; slots unused by DeepGEMM's trace): set slots written by the scheduler warp (lane 0) of the leader CTA,
# accumulate slots counted by lane 0 of the scheduler warp (flushed at kernel end like every accumulator).
TRACE_SLOT_SMID = 46  # physical %smid + 1 of this CTA (the map is indexed by it, NOT by blockIdx)
TRACE_SLOT_DOMAIN = 47  # sm_locality_domain_idx + 1 as read from the map
TRACE_ACC_OWN = 48  # successful claims from the own domain's queue (L1 + L2 + shared)
TRACE_ACC_STOLEN = 49  # successful claims stolen from the other domain's queue


@cute.jit
def trace_domain(ctx, dom):
    """``cfg.trace`` only: record ``%smid`` and the domain this scheduler warp read (end-to-end check of the map lookup: ``dom == map[smid]``)."""
    if cutlass.const_expr(ctx.tr.enabled):
        if ctx.lane_idx == 0:
            cute.arch.store(ctx.tr._slot_ptr(TRACE_SLOT_SMID), Uint64(ptx.smid() + 1))
            cute.arch.store(ctx.tr._slot_ptr(TRACE_SLOT_DOMAIN), Uint64(dom + 1))


@cute.jit
def _claim_idx(counter_ptr):
    """``get_next_local_task_idx`` (SCH:196-201): the elected lane does ``atom.global.add.u32(counter, 1)``, the old value is broadcast from lane 0
    (identical to ``scheduler._claim``)."""
    result = Int32(0)
    if ptx.elect_sync():
        result = ptx.atom_add_u32(counter_ptr, 1)
    return ptx.shfl_idx(result, 0)


@cute.jit
def claim(ctx, counter_ptr, dom, nclusters: cutlass.Constexpr, num_m_blocks):
    """``get_next_task_idx<nclusters>(counter, num_m_blocks, m_block_idx, n_cluster_idx)`` (SCH:203-222): claim the next ``(m_block, n_cluster)`` task of a GEMM with
    ``nclusters`` N clusters whose counters start at ``counter_ptr`` (the domain-0 counter; ``ctx.ws.l1_task_count(0)`` / ``l2_task_count(0)`` / ``shared_l1_task_count(0)`` /
    ``shared_l2_task_count(0)``).  All 32 lanes execute it (warp-uniform result).  Returns ``(ok, m_block_idx, n_cluster_idx)``; ``ok`` is False when the queue(s)
    this scheduler may take from are drained (the counters have still been advanced, like DeepGEMM's)."""
    cfg = ctx.cfg
    D = cfg.num_locality_domains
    ok = Boolean(False)
    m_block_idx = Int32(0)
    n_cluster_idx = Int32(0)
    if cutlass.const_expr(D == 1):
        idx = _claim_idx(counter_ptr)
        ok = idx < num_m_blocks * nclusters
        m_block_idx = idx // nclusters
        n_cluster_idx = idx % nclusters  # N-cluster fastest, M-block major
    else:
        nl = nclusters // D  # kNumLocalClusters
        limit = num_m_blocks * nl
        idx = _claim_idx(counter_ptr + dom * cfg.domain_counter_stride)
        ok = idx < limit
        m_block_idx = idx // nl
        n_cluster_idx = dom * nl + idx % nl
        drained = idx >= limit
        if cutlass.const_expr(ctx.tr.enabled):
            if ctx.lane_idx == 0:
                if ok:
                    ctx.tr.add(TRACE_ACC_OWN, 1)
        # work stealing only with >= kNumMinStealWaves waves of tasks (SCH:220): `kNumLocalityDomains > 1 and num_m_blocks * C >= 2 * (kNumSMs / 2) and claim(dom ^ 1)`
        if drained & (num_m_blocks * nclusters >= cfg.num_min_steal_waves * (cfg.num_sms // 2)):
            other = dom ^ 1
            idx2 = _claim_idx(counter_ptr + other * cfg.domain_counter_stride)
            ok = idx2 < limit
            m_block_idx = idx2 // nl
            n_cluster_idx = other * nl + idx2 % nl
            if cutlass.const_expr(ctx.tr.enabled):
                if ctx.lane_idx == 0:
                    if ok:
                        ctx.tr.add(TRACE_ACC_STOLEN, 1)
    return ok, m_block_idx, n_cluster_idx


@cute.jit
def wait_l1_claimed(ctx, pool_block_idx):
    """The L2 task wait (SCH:250-254): spin until every domain's L1 counter is ``>= (pool_block_idx + 1) * (kNumL1Clusters / D)`` (``ld.volatile.global.b32``)."""
    cfg = ctx.cfg
    D = cfg.num_locality_domains
    num_required = (pool_block_idx + 1) * (cfg.num_l1_clusters // D)
    for i in cutlass.range_constexpr(D):
        while ptx.ld_volatile_b32(ctx.ws.l1_task_count(i)) < num_required:
            pass
