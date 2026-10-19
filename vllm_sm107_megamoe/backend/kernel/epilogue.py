# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / common/math.cuh / ptx/ld_st.cuh.
"""GEMM epilogues (epilogue warps 8-15, K:1210-1640): L1 (SwiGLU + FP8 requantisation + TMA store) and L2 (BF16 push to the source rank).

The 8 epilogue warps form 2 warpgroups; warp ``w`` of a warpgroup owns TMEM lane quadrant ``w`` (weight rows ``32 w .. 32 w + 31`` of the
CTA's 128 rows); the warpgroup ``wg`` owns tokens ``wg * WG_BLOCK_M ..`` (accumulator columns).  Per 8-token atom two
``tcgen05.ld.16x256b.x1`` loads (lane offset 0 / 16) give each thread 8 FP32: ``v0,v1`` (gate row ``lane/4``, tokens ``2(lane%4), +1``),
``v2,v3`` (up row ``lane/4 + 8``), ``v4..v7`` the same for rows +16 (gate/up interleave 8 => adjacent rows of one channel).

Numerics follow DeepGEMM exactly (op order, BF16 rounding before the clamp, packed ``.rn`` FP32 ops, ``ex2.approx`` / ``rcp.approx.ftz``
under ``fast_math``, UE8M0 by the integer-carry method, E4M3 ``satfinite`` RN pack).  ``fast_math=False`` (kFastMath = false) is the IEEE branch: ``expf(-s)`` as nvcc expands it
(``ptx_g.expf_ieee``) and the true division ``gate / denom`` (``div.rn.f32``); ``alpha != 1`` / ``beta != 0`` / a finite ``clamp`` are ``const_expr`` branches (compiled out at the
defaults; the clamp constants are the BF16 round-to-nearest-even encodings of the float32 clamp).
"""

SUPPORTS_SHARED_EXPERTS = True  # DESIGN.md section 17: this module implements the shared-expert (S > 0) paths

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32
from cutlass.cute.nvgpu import cpasync

from . import ptx
from . import ptx_g
from . import scheduler as sched
from .scheduler import PHASE_NONE, PHASE_LINEAR1, PHASE_SHARED_LINEAR1
from .tma_warps import smem_layout_l1_out

_LOG2E = 1.4426950408889634  # the float32 nearest value (0x3FB8AA3B) is what `__expf` multiplies by
_NUM_BANK_GROUP_BYTES = 16
_ATOM_M = 8


def _bf16_bits_pair(value: float) -> int:
    """Two copies of the BF16 round-to-nearest-even encoding of ``value`` packed in 32 bits (the ``{c, c}`` clamp constant)."""
    import struct

    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return bf16 | (bf16 << 16)


def require_default_epilogue_options(cfg) -> None:
    """``Sm107KernelConfig`` fields that are part of the compile key and of the parity manifest but that the CuteDSL epilogue does not
    implement.  A hand-built config with another value would compile an unchanged kernel while the cache key / manifest claim a different DeepGEMM variant, so the
    epilogue refuses them (trace time only: no effect on the generated PTX of the default builds, which are the only ones the host factory produces)."""
    for field, expected, what in (
        ("epi_prefetch_stages", 0, "kEpiPrefetchStages: TMEM-load prefetch of the next store block"),
        ("tma_store_stages", 2, "kNumTMAStoreStages: TMA-store smem stages"),
        ("debug_flags", 0, "kDebugFlags: DeepGEMM's debug-only skips / shortcuts"),
    ):
        value = getattr(cfg, field)
        if value != expected:
            raise NotImplementedError(f"cfg.{field}={value!r} ({what}) is not implemented by the CuteDSL port (the epilogue implements only {field}={expected})")


@cute.jit
def _release_tmem(ctx, accum_stage):
    """``epi_release_tmem`` (K:1251-1254): ``tcgen05.fence::before_thread_sync`` + remote arrive on CTA 0's ``tmem_empty[accum_stage]``."""
    ptx.tcgen05_fence_before_thread_sync()
    ctx.mb.tmem_empty(accum_stage).arrive_remote(0)


@cute.jit
def _num_valid_store_blocks(ctx, valid_m, wg):
    """``get_num_valid_store_blocks`` (K:1246-1250)."""
    cfg = ctx.cfg
    wg_start_m = wg * cfg.wg_block_m
    rows = valid_m - wg_start_m
    num_valid_wg_rows = Int32(cutlass.select_(valid_m <= wg_start_m, Int32(0), Int32(cutlass.select_(rows < cfg.wg_block_m, rows, Int32(cfg.wg_block_m)))))
    return (num_valid_wg_rows + (cfg.store_block_m - 1)) // cfg.store_block_m


@cute.jit
def _load_cached_weight(ctx, token_base, lane, j_atom_m: cutlass.Constexpr, prev):
    """Per-lane register cache of the top-k weight of token ``token_base + lane`` (K:1355-1360): the load is skipped by lanes beyond WG_BLOCK_M when it is not a multiple of 32."""
    cfg = ctx.cfg
    out = prev
    if cutlass.const_expr(cfg.wg_block_m % 32 == 0):
        out = ptx.ld_f32(ctx.buf.region_addr("l1_topk_weights", token=token_base + lane))
    else:
        if (j_atom_m + lane) < cfg.wg_block_m:
            out = ptx.ld_f32(ctx.buf.region_addr("l1_topk_weights", token=token_base + lane))
    return out


@cute.jit
def _wait_l2_empty(ctx, pool_block_idx, ring_block_idx, rec):
    """K:1322-1328: all 256 epilogue threads spin on ``ld.acquire.gpu`` until the generation's L2 consumers of this ring slot are done."""
    cfg = ctx.cfg
    num_expected_blocks = cfg.num_l2_block_ns * (pool_block_idx // cfg.num_ring_blocks)
    tc = ctx.tr.clock()
    while ptx.ld_acquire_gpu_b32(ctx.ws.l2_empty_count(ring_block_idx)) != num_expected_blocks:
        pass
    sched.trace_since(ctx, 27, tc, rec)


@cute.jit
def _epilogue_l1(ctx, pool_block_idx, valid_m, n_cluster_idx, accum_stage, is_shared=None):
    """Unified L1 epilogue (K:1322-1530) for routed and shared-expert tasks.  ``is_shared``: ``None`` in S = 0 builds (static: the shared branches are not
    traced, the code is identical to the routed-only epilogue) and a runtime Boolean (``task_info.is_shared()``) in S > 0 builds."""
    cfg = ctx.cfg
    BM, WGBM, SBM = cfg.block_m, cfg.wg_block_m, cfg.store_block_m
    NA = SBM // _ATOM_M  # atoms per store block
    NSB = WGBM // SBM  # store blocks per warpgroup
    ew = ctx.warp_idx - 8  # epilogue_warp_idx
    wg = ew // 4
    wiw = ew % 4  # warp_idx_in_wg
    lane = ctx.lane_idx
    shared_build = is_shared is not None
    ring_block_idx = pool_block_idx % cfg.num_ring_blocks
    block_idx = ring_block_idx  # routed tasks
    if cutlass.const_expr(shared_build):
        if is_shared:
            block_idx = pool_block_idx  # shared tasks: the token block itself, no ring (K:1315)
    ring_m_idx = ring_block_idx * BM  # ring-buffer offset of the reusable data buffers
    m_idx = block_idx * BM
    n_block_idx = n_cluster_idx * 2 + ctx.cta_rank

    # wait L2 block empty (K:1322-1328), routed tasks only
    if cutlass.const_expr(shared_build):
        if not is_shared:
            _wait_l2_empty(ctx, pool_block_idx, ring_block_idx, (ew == 0) & (lane == 0))
    else:
        _wait_l2_empty(ctx, pool_block_idx, ring_block_idx, (ew == 0) & (lane == 0))

    num_valid = _num_valid_store_blocks(ctx, valid_m, wg)
    if num_valid == 0:  # epi_prologue: nothing to load, release the TMEM stage right away
        _release_tmem(ctx, accum_stage)

    # SF store base of this task (K:1465-1476): `l2_sf` ring (or the shared L2 SF buffer for shared tasks), column word `k_idx / 4`, byte `k_idx % 4`
    sf_k_idx = n_block_idx * 2 + wiw // 2
    sf_mn_stride = Int64(cfg.num_sf_ring_tokens * 4)
    sf_base = ctx.buf.region_addr("l2_sf")
    if cutlass.const_expr(shared_build):
        if is_shared:  # `kNumSharedSFTokens` stride into the shared L2 SF buffer (K:1472-1474)
            sf_mn_stride = Int64(cfg.num_shared_sf_tokens * 4)
            sf_base = ctx.buf.region_addr("shared_l2_sf")
    sf_ptr0 = sf_base + Int64(sf_k_idx // 4) * sf_mn_stride + Int64(sf_k_idx % 4)

    stored_cached_weight = Float32(1.0)
    for s in cutlass.range_constexpr(NSB):
        if s < num_valid:
            # ---- TMEM loads of the store block (one wait); the LAST valid block releases the stage (epi_load_if_valid)
            raw = []
            for i in cutlass.range_constexpr(NA):
                tmem_addr = accum_stage * cfg.umma_n + wg * WGBM + (s * NA + i) * _ATOM_M
                lo = ptx.tmem_ld_16x256b_x1(tmem_addr)
                hi = ptx.tmem_ld_16x256b_x1(tmem_addr | 0x00100000)
                raw.append(list(lo) + list(hi))
            if s + 1 == num_valid:
                ptx.tmem_ld_wait()
                _release_tmem(ctx, accum_stage)
            raw = ptx_g.tmem_wait_ld_dep(raw)  # `tcgen05.wait::ld` that the loaded registers depend on (no hoisting of uses above it)

            # ---- SwiGLU + weight + amax, per 8-token atom
            acts = []
            amaxes = []
            for i in cutlass.range_constexpr(NA):
                j = s * NA + i
                # load the top-k weights of 32 tokens into a per-lane register cache (K:1355-1360)
                if cutlass.const_expr((j * _ATOM_M) % 32 == 0):
                    if cutlass.const_expr(shared_build):
                        if not is_shared:  # shared tasks keep the weight 1.0 (K:1355)
                            stored_cached_weight = _load_cached_weight(ctx, ring_m_idx + wg * WGBM + j * _ATOM_M, lane, j * _ATOM_M, stored_cached_weight)
                    else:
                        stored_cached_weight = _load_cached_weight(ctx, ring_m_idx + wg * WGBM + j * _ATOM_M, lane, j * _ATOM_M, stored_cached_weight)
                w0 = ptx.shfl_idx(stored_cached_weight, (j * _ATOM_M) % 32 + (lane % 4) * 2 + 0)
                w1 = ptx.shfl_idx(stored_cached_weight, (j * _ATOM_M) % 32 + (lane % 4) * 2 + 1)
                v = raw[i]
                atom_acts = []
                for k in cutlass.range_constexpr(2):
                    # BF16 rounding of the FP32 accumulators BEFORE the clamp: float22bfloat162_rn({x, y}) = low:rn(x), high:rn(y)
                    gate = ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[4 * k + 1]), ptx_g.bits_f32(v[4 * k + 0]))
                    up = ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[4 * k + 3]), ptx_g.bits_f32(v[4 * k + 2]))
                    if cutlass.const_expr(cfg.clamp != float("inf")):
                        c = _bf16_bits_pair(cfg.clamp)
                        nc = _bf16_bits_pair(-cfg.clamp)
                        gate = ptx.hmin2_bf16x2(gate, Int32(c))
                        up = ptx.hmax2_bf16x2(up, Int32(nc))
                        up = ptx.hmin2_bf16x2(up, Int32(c))
                    gx, gy = ptx.bf16x2_to_f32x2(gate)
                    sx, sy = gx, gy  # sigmoid_input
                    if cutlass.const_expr(cfg.alpha != 1.0):
                        sx, sy = ptx.fmul2_rn((sx, sy), (Float32(cfg.alpha), Float32(cfg.alpha)))
                    if cutlass.const_expr(cfg.fast_math):
                        ex = ptx.ex2_approx((-sx) * Float32(_LOG2E))  # __expf(-s) (kFastMath): mul.f32 log2e + ex2.approx.f32 (NO .ftz, as in DeepGEMM's PTX)
                        ey = ptx.ex2_approx((-sy) * Float32(_LOG2E))
                    else:
                        ex = ptx_g.expf_ieee(-sx)  # expf(-s) (kFastMath = false): nvcc's expf expansion (fma.rm range reduction + ex2.approx.ftz + exponent scale)
                        ey = ptx_g.expf_ieee(-sy)
                    dx, dy = ptx.fadd2_rn((Float32(1.0), Float32(1.0)), (ex, ey))
                    if cutlass.const_expr(cfg.fast_math):
                        gated = ptx.fmul2_rn((gx, gy), (ptx.rcp_approx_ftz(dx), ptx.rcp_approx_ftz(dy)))
                    else:
                        gated = (ptx_g.div_rn_f32(gx, dx), ptx_g.div_rn_f32(gy, dy))  # true division gate / denom (div.rn.f32)
                    ux, uy = ptx.bf16x2_to_f32x2(up)
                    if cutlass.const_expr(cfg.beta != 0.0):
                        ux, uy = ptx.fadd2_rn((ux, uy), (Float32(cfg.beta), Float32(cfg.beta)))
                    act = ptx.fmul2_rn(ptx.fmul2_rn(gated, (ux, uy)), (w0, w1))  # (gated * up) * weight
                    atom_acts.append(act)
                # thread-level amax over the two channel rows (k), then warp-level: shfl.xor 4, 8, 16 (K:1411-1422)
                ax = ptx_g.fmax_f32(Float32(0.0), ptx_g.fabs_f32(atom_acts[0][0]))
                ay = ptx_g.fmax_f32(Float32(0.0), ptx_g.fabs_f32(atom_acts[0][1]))
                ax = ptx_g.fmax_f32(ax, ptx_g.fabs_f32(atom_acts[1][0]))
                ay = ptx_g.fmax_f32(ay, ptx_g.fabs_f32(atom_acts[1][1]))
                for xm in cutlass.range_constexpr(3):
                    ax = ptx_g.fmax_f32(ax, ptx.shfl_xor(ax, 4 << xm))
                for xm in cutlass.range_constexpr(3):
                    ay = ptx_g.fmax_f32(ay, ptx.shfl_xor(ay, 4 << xm))
                # warp-pair exchange through smem: lanes < 4 hold the warp's amax of 2 tokens each
                if lane < 4:
                    ptx.st_shared_v2_f32(ctx.smem.amax(ew) + 2 * (i * (_ATOM_M // 2) + lane), ax, ay)
                cute.arch.sync_warp()
                acts.append(atom_acts)
                amaxes.append((ax, ay))

            # wait shared memory release from the previous TMA store, and fence `amax_reduction` (K:1430-1434)
            tma_stage = s % 2
            ptx.tma_store_wait(1)
            ctx.bar.epilogue_wg(wg)

            # ---- cast to FP8 E4M3, STSM into smem_d, UE8M0 SF byte stores
            for i in cutlass.range_constexpr(NA):
                wpx, wpy = ptx.ld_shared_v2_f32(ctx.smem.amax(ew ^ 1) + 2 * (i * (_ATOM_M // 2) + lane % 4))
                ax = ptx_g.fmax_f32(amaxes[i][0], wpx)
                ay = ptx_g.fmax_f32(amaxes[i][1], wpy)
                sf_exp_x = ptx_g.get_ue8m0_sf_exp(ax)
                sf_exp_y = ptx_g.get_ue8m0_sf_exp(ay)
                inv_x = ptx_g.get_ue8m0_sf_inv(sf_exp_x)
                inv_y = ptx_g.get_ue8m0_sf_inv(sf_exp_y)
                upper = ptx.fmul2_rn(acts[i][0], (inv_x, inv_y))
                lower = ptx.fmul2_rn(acts[i][1], (inv_x, inv_y))
                fp8x4 = ptx.cvt_e4m3x4_f32(upper[0], upper[1], lower[0], lower[1])
                # STSM (m16n8.x1.trans.b8): row = lane, col = warp_idx_in_wg; 64B swizzle => (col ^ (row / 2)) * 16
                row = lane
                col = wiw
                smem_addr = ctx.smem.addr(ctx.smem.d_l1(wg, tma_stage)) + (i * _ATOM_M * 64 + row * 64 + (col ^ (row // 2)) * _NUM_BANK_GROUP_BYTES)
                ptx.stmatrix_m16n8_x1_trans_b8(smem_addr, fp8x4)
                # SF -> l2_sf ring, MN-major UE8M0 bytes: only one warp per pair writes (both hold the same SF); the address math is hoisted out of the
                # branch (as nvcc does for DeepGEMM) so that only the two byte stores are predicated
                token_base_idx = wg * WGBM + s * SBM + i * _ATOM_M  # < BLOCK_M
                sf_token_idx = block_idx * cfg.sf_block_m + (token_base_idx & ~127) + (token_base_idx & 31) * 4 + ((token_base_idx >> 5) & 3) + (lane * 2) * 4
                sf_addr = sf_ptr0 + Int64(sf_token_idx) * Int64(4)
                if (wiw % 2 == 0) & (lane < 4):
                    ptx_g.st_global_b8(sf_addr, sf_exp_x)
                    ptx_g.st_global_b8(sf_addr + Int64(4 * 4), sf_exp_y)
                cute.arch.sync_warp()
            ctx.bar.epilogue_wg(wg)

            # ---- issue the TMA store after all atoms of the block (one thread per warpgroup)
            if wiw == 0:
                with cute.arch.elect_one():
                    ptx.fence_proxy_async_shared_cta()
                    if cutlass.const_expr(shared_build):
                        if is_shared:
                            _tma_store_l1_block(ctx, wg, tma_stage, (m_idx + wg * WGBM + s * SBM) // SBM, n_block_idx, True)
                        else:
                            _tma_store_l1_block(ctx, wg, tma_stage, (m_idx + wg * WGBM + s * SBM) // SBM, n_block_idx, False)
                    else:
                        _tma_store_l1_block(ctx, wg, tma_stage, (m_idx + wg * WGBM + s * SBM) // SBM, n_block_idx, False)
                    ptx.tma_store_commit()
            cute.arch.sync_warp()

    # notify L2 and increment the L1 empty count (K:1510-1526)
    ptx.tma_store_wait(0)
    ctx.bar.epilogue()
    if ew == 0:
        with cute.arch.elect_one():
            if cutlass.const_expr(shared_build):
                if is_shared:
                    ptx.red_add_release_gpu_u32(ctx.ws.shared_l2_full_count(pool_block_idx), 1)  # ptx::red_add_rel (K:1515-1518)
                else:
                    ptx.red_xor_release_gpu_b64(ctx.ws.l2_full_mask(ring_block_idx), Int64(1) << Int64(n_block_idx))
                    ptx.red_add_u32(ctx.ws.l1_empty_count(ring_block_idx), 1)
            else:
                ptx.red_xor_release_gpu_b64(ctx.ws.l2_full_mask(ring_block_idx), Int64(1) << Int64(n_block_idx))
                ptx.red_add_u32(ctx.ws.l1_empty_count(ring_block_idx), 1)
    cute.arch.sync_warp()


@cute.jit
def _tma_store_l1_block(ctx, wg, stage: cutlass.Constexpr, row_tile, n_block_idx, shared: cutlass.Constexpr = False):
    """``SM90_TMA_STORE_2D`` of one ``STORE_BLOCK_M x 64`` FP8 tile (K:1496-1506): smem_d.l1[wg][stage] -> l2_token ring at
    ``(n_block * 64, m_idx + wg * WG_BLOCK_M + s * STORE_BLOCK_M)``.  No cache-hint operand."""
    cfg = ctx.cfg
    s_tile = cute.make_tensor(ctx.smem.d_l1(wg, stage), smem_layout_l1_out(cfg))
    if cutlass.const_expr(shared):  # tensor_map_shared_l1_output (K:1498)
        g_tile = cute.local_tile(ctx.tma.shared_l1_out_t, (cfg.store_block_m, 64), (row_tile, n_block_idx))
        ts, tg = cpasync.tma_partition(ctx.tma.shared_l1_out_atom, 0, cute.make_layout(1), cute.group_modes(s_tile, 0, 2), cute.group_modes(g_tile, 0, 2))
        cute.copy(ctx.tma.shared_l1_out_atom, ts, tg)
    else:
        g_tile = cute.local_tile(ctx.tma.l1_out_t, (cfg.store_block_m, 64), (row_tile, n_block_idx))
        ts, tg = cpasync.tma_partition(ctx.tma.l1_out_atom, 0, cute.make_layout(1), cute.group_modes(s_tile, 0, 2), cute.group_modes(g_tile, 0, 2))
        cute.copy(ctx.tma.l1_out_atom, ts, tg)


@cute.jit
def _bump_l2_empty(ctx, ring_block_idx, ew):
    """+1 on this ring slot's ``l2_empty_count`` (relaxed ``red``) by one elected lane of epilogue warp 0, then ``__syncwarp`` (K:1533-1539)."""
    if ew == 0:
        with cute.arch.elect_one():
            ptx.red_add_u32(ctx.ws.l2_empty_count(ring_block_idx), 1)
    cute.arch.sync_warp()


@cute.jit
def _epilogue_l2(ctx, pool_block_idx, valid_m, n_cluster_idx, accum_stage, is_shared=None):
    """L2 BF16 epilogue (K:1531-1640): STSM transpose into smem_d (128B swizzle) then 16 B stores of two token rows per warp to the SOURCE rank's combine buffer
    (``combine_token[top-k slot][source token]``).  Shared-expert tasks (``is_shared`` runtime Boolean, S > 0 builds; ``None`` in S = 0 builds) write to the OWN
    rank's extra slot ``kNumTopk`` with the synthesized metadata ``(rank_idx, pool_m_idx + row, kNumTopk)`` and do not touch ``l2_empty_count``."""
    cfg = ctx.cfg
    BM, WGBM, SBM = cfg.block_m, cfg.wg_block_m, cfg.store_block_m
    NA = SBM // _ATOM_M
    NSB = WGBM // SBM
    NROWS = SBM // 8  # kNumRowsPerWarp
    shared_build = is_shared is not None
    ew = ctx.warp_idx - 8
    wg = ew // 4
    wiw = ew % 4
    lane = ctx.lane_idx
    ring_block_idx = pool_block_idx % cfg.num_ring_blocks
    pool_m_idx = pool_block_idx * BM  # full-pool offset for the non-ring metadata
    n_block_idx = n_cluster_idx * 2 + ctx.cta_rank
    n_idx = n_block_idx * cfg.block_n

    # the L2 activations of this ring slot are fully consumed (the MMA finished): +1 on l2_empty (relaxed red), K:1533-1539 (routed tasks only)
    if cutlass.const_expr(shared_build):
        if not is_shared:
            _bump_l2_empty(ctx, ring_block_idx, ew)
    else:
        _bump_l2_empty(ctx, ring_block_idx, ew)

    num_valid = _num_valid_store_blocks(ctx, valid_m, wg)
    if num_valid == 0:
        _release_tmem(ctx, accum_stage)

    for s in cutlass.range_constexpr(NSB):
        if s < num_valid:
            # read the source metadata of this warp's rows before the TMEM loads to overlap the latency (K:1555-1565)
            meta_rank = []
            meta_token = []
            meta_topk = []
            for j in cutlass.range_constexpr(NROWS):
                m_idx_in_block = wg * WGBM + s * SBM + j * 8 + wiw * 2 + lane // 16
                p = ctx.ws.token_src_metadata(pool_m_idx + m_idx_in_block)
                r0 = Int32(0)
                r1 = Int32(0)
                r2 = Int32(0)
                if m_idx_in_block < valid_m:
                    if cutlass.const_expr(shared_build):
                        if is_shared:
                            r0 = ctx.sym.rank_idx
                            r1 = pool_m_idx + m_idx_in_block
                            r2 = Int32(cfg.num_topk)
                        else:
                            r0 = ptx.ld_b32(p)
                            r1 = ptx.ld_b32(p + 1)
                            r2 = ptx.ld_b32(p + 2)
                    else:
                        r0 = ptx.ld_b32(p)
                        r1 = ptx.ld_b32(p + 1)
                        r2 = ptx.ld_b32(p + 2)
                meta_rank.append(r0)
                meta_token.append(r1)
                meta_topk.append(r2)
            cute.arch.sync_warp()

            raw = []
            for i in cutlass.range_constexpr(NA):
                tmem_addr = accum_stage * cfg.umma_n + wg * WGBM + (s * NA + i) * _ATOM_M
                lo = ptx.tmem_ld_16x256b_x1(tmem_addr)
                hi = ptx.tmem_ld_16x256b_x1(tmem_addr | 0x00100000)
                raw.append(list(lo) + list(hi))
            if s + 1 == num_valid:
                ptx.tmem_ld_wait()
                _release_tmem(ctx, accum_stage)
            raw = ptx_g.tmem_wait_ld_dep(raw)

            d_base = ctx.smem.addr(ctx.smem.d_l2(wg))  # bf16 [STORE_BLOCK_M * 128] per warpgroup (swizzle 128B), as bytes
            for i in cutlass.range_constexpr(NA):
                v = raw[i]
                # wait shared memory release from the previous NVLink store: skipped for the first store block (K:1574-1577)
                if cutlass.const_expr(i == 0 and s > 0):
                    ctx.bar.epilogue_wg(wg)
                row = lane % 8
                col = (ew % 2) * 4 + lane // 8
                smem_addr = d_base + ((wiw // 2) * SBM * 128 + i * _ATOM_M * 128 + row * (_NUM_BANK_GROUP_BYTES * 8) + (col ^ row) * _NUM_BANK_GROUP_BYTES)
                ptx.stmatrix_m8n8_x4_trans_b16(
                    smem_addr,
                    ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[1]), ptx_g.bits_f32(v[0])),
                    ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[3]), ptx_g.bits_f32(v[2])),
                    ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[5]), ptx_g.bits_f32(v[4])),
                    ptx.cvt_rn_bf16x2_f32(ptx_g.bits_f32(v[7]), ptx_g.bits_f32(v[6])),
                )
            ctx.bar.epilogue_wg(wg)  # wait shared memory ready

            # write into the remote buffers: each warp writes 2 rows (the half-warps), 16 B per lane (256 B per row)
            row_in_atom = (wiw * 2 + lane // 16) % _ATOM_M
            bank_group_idx = lane % 8
            for j in cutlass.range_constexpr(NROWS):
                row_in_store = j * 8 + wiw * 2 + lane // 16
                m_idx_in_block = wg * WGBM + s * SBM + row_in_store
                if m_idx_in_block < valid_m:  # skip padding rows beyond the actual token count for this expert
                    smem_addr = d_base + ((lane % 16 // 8) * SBM * 128 + row_in_store * 128 + (bank_group_idx ^ row_in_atom) * _NUM_BANK_GROUP_BYTES)
                    x, y, z, w = ptx.ld_shared_v4_u32(smem_addr)
                    dst = ctx.buf.region_addr("combine_token", token=meta_token[j], slot=meta_topk[j]) + Int64(n_idx * 2) + Int64((lane % 16) * 16)
                    if cutlass.const_expr(cfg.num_ranks > 1):
                        # DeepGEMM: the pointer comes out of `sym_buffer.map` (runtime peer offset) -> GENERIC `st.v4.b32` (SASS ST.E.128) at DEP > 1 ...
                        ptx.st_generic_v4_b32(ctx.sym.map(dst, meta_rank[j]), x, y, z, w)
                    else:
                        # ... and a `st.global.v4.b32` (STG.E.128) at EP1, where the map is the identity
                        ptx.st_v4_b32(ctx.sym.map(dst, meta_rank[j]), x, y, z, w)

    # ensure the next epilogue is safe to use shared memory (K:1634-1635)
    ctx.bar.epilogue()


@cute.jit
def epilogue_role(ctx):
    """Epilogue warps 8-15 GEMM part (K:1294-1641): persistent loop over the published tasks (the kernel executes R1 before and the TMEM free after)."""
    cfg = ctx.cfg
    require_default_epilogue_options(cfg)
    sched_stage = Int32(0)
    sched_phase = Int32(0)
    current_iter_idx = Int32(0)
    rec = ((ctx.warp_idx - 8) == 0) & (ctx.lane_idx == 0)  # trace recorder: epilogue warp 0 lane 0
    tc = ctx.tr.clock()
    t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
    sched.trace_since(ctx, 31, tc, rec)
    while t0 != PHASE_NONE:
        # wait UMMA arrival
        accum_stage = current_iter_idx % cfg.accum_stages
        accum_phase = (current_iter_idx // cfg.accum_stages) & 1
        current_iter_idx = current_iter_idx + 1
        tc = ctx.tr.clock()
        ctx.mb.tmem_full(accum_stage).wait(accum_phase)
        sched.trace_since(ctx, 24, tc, rec)
        ptx.tcgen05_fence_after_thread_sync()
        t_epi = ctx.tr.clock()  # trace_epi_t0
        # now we can release the task
        sched.release_task_info(ctx, sched_stage)
        valid_m = ptx.shfl_idx(t5, 0)  # `ptx::exchange(task_info.valid_m, 0)`: tells NVCC there is no warp divergence
        if cutlass.const_expr(cfg.num_shared_experts > 0):
            is_shared = t0 >= PHASE_SHARED_LINEAR1
            if (t0 == PHASE_LINEAR1) | (t0 == PHASE_SHARED_LINEAR1):
                _epilogue_l1(ctx, t4, valid_m, t3, accum_stage, is_shared)
                sched.trace_since(ctx, 25, t_epi, rec)
                sched.trace_count(ctx, 28, rec)
            else:
                _epilogue_l2(ctx, t4, valid_m, t3, accum_stage, is_shared)
                sched.trace_since(ctx, 26, t_epi, rec)
                sched.trace_count(ctx, 29, rec)
        else:
            if t0 == PHASE_LINEAR1:
                _epilogue_l1(ctx, t4, valid_m, t3, accum_stage)
                sched.trace_since(ctx, 25, t_epi, rec)
                sched.trace_count(ctx, 28, rec)
            else:
                _epilogue_l2(ctx, t4, valid_m, t3, accum_stage)
                sched.trace_since(ctx, 26, t_epi, rec)
                sched.trace_count(ctx, 29, rec)
        tc = ctx.tr.clock()
        t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
        sched.trace_since(ctx, 31, tc, rec)
    ctx.tr.stamp(4, rec)  # epilogue GEMM done (K:1643)
    ctx.tr.stamp_clock(61, rec)  # GEMM-only builds have no combine: close the CTA lifetime here (the combine role overwrites slot 61 with its own, later, value)
