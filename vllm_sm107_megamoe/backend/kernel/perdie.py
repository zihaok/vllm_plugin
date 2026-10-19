# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / scheduler/sm107_mega_moe.cuh / csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp.
"""Per-die (locality-domain) pieces of the kernel that are not scheduler logic.

DeepGEMM runs the production recipe with ``DG_MEGA_MOE_SM107_PERDIE`` (``kNumLocalityDomains = 2``) and die-local routed weights
(``kLocalizedWeights``; shared weights plain: ``kLocalizedSharedWeights = false``).  Rubin has two locality domains ("dies"); the die-local weights are
``(2, E_l, N/2, K/2)`` tensors whose slice ``d`` (the N half ``d`` of every expert) is homed in domain ``d``.  This module provides

* :func:`load_domain` -- ``sm_locality_domains[%smid]`` (``SCH:292``, ``ptx::get_sm_idx``): the SM -> domain map is a ``uint8[num_sms]`` indexed by the PHYSICAL
  SM id (NOT ``blockIdx``); read once by the scheduler warp of the leader CTA (both CTAs of a cluster are one TPC, hence one domain);
* :func:`make_weight_pair` -- the 3D TMA tensor map of a die-local weight tensor (``make_tma_3d_desc``, ``IMPL:163-176``): global dims ``(rows_per_slice, K bytes, 2)``,
  strides ``(row bytes, slice stride)``, box ``(128 rows, 64 B, 1)``, SWIZZLE_64B.  Like the 2D routed map of G (``tma_warps.make_tma_params``) it is a ``UINT8`` map over the
  byte view of the packed FP4 weights (same smem image / expect-tx bytes as ``16U4_ALIGN8B``; DESIGN.md section 4.4);
* :func:`b_tile_coords` -- the TMA-B coordinates ``(half_row / 128, half_idx)`` of ``K:934-936,998-1000``; ``tma_warps._b_k_loop`` then loads the tile ``(k, half_row, half_idx)``
  of the 3D map (the SF map, barriers and expect-tx are identical to the plain-weights loop).

The scheduler side lives in ``scheduler_perdie.py``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op

from . import ptx


@dsl_user_op
def ld_global_u8(ga, *, loc=None, ip=None) -> Int32:
    """``ld.global.u8`` (zero-extended into a 32-bit register): the ``uint8_t`` load of ``sm_locality_domains[%smid]`` (SASS ``LDG.E.U8``)."""
    return Int32(ptx._asm(T.i32(), [ptx._gaddr(ga, loc=loc, ip=ip)], "ld.global.u8 $0, [$1];", "=r,l", loc=loc, ip=ip))


@cute.jit
def load_domain(ctx):
    """``sm_locality_domain_idx = kNumLocalityDomains == 1 ? 0u : sm_locality_domains[ptx::get_sm_idx()]`` (SCH:292).  Executed by every lane of the scheduler warp."""
    dom = Int32(0)
    if cutlass.const_expr(ctx.cfg.num_locality_domains > 1):
        dom = ld_global_u8(ctx.sm_locality_domains + ptx.smid())
    return dom


# ----------------------------------------------------------------------------------------------------------------------
# 3D weight tensor maps
# ----------------------------------------------------------------------------------------------------------------------
def smem_layout_b3(cfg):
    """Packed-FP4 weights stage ring as seen by the 3D atom: ``(128 rows, 64 B, 1, stages)`` bytes, SWIZZLE_64B (``Swizzle<2,4,3>``), stage stride 8192 B
    (the 2D layout of ``tma_warps.smem_layout_b`` with the unit "domain" mode the 3D box needs)."""
    return cute.make_composed_layout(
        cute.make_swizzle(2, 4, 3),
        0,
        cute.make_layout((128, 64, 1, cfg.num_stages), stride=(64, 1, 0, 8192)),
    )


def smem_layout_sfb(cfg):
    """Weight SF stage ring ``(128 words, 1, stages)`` int32, no swizzle (identical to ``tma_warps.smem_layout_sfb``; the SF maps are NOT localized)."""
    return cute.make_layout((128, 1, cfg.num_stages), stride=(1, 128, 128))


def weight_geometry(cfg, kind: str):
    """``(rows per domain slice, bytes per row)`` of the die-local weight tensor ``kind`` ('routed_l1' ``(2, E_l, 2I/2, H/2)``, 'routed_l2' ``(2, E_l, H/2, I/2)``);
    the routed expert dimension is folded into the rows (experts are contiguous inside a slice)."""
    El, H, I = cfg.num_experts_per_rank, cfg.hidden, cfg.intermediate_hidden
    if kind == "routed_l1":
        return El * I, H // 2
    if kind == "routed_l2":
        return El * (H // 2), I // 2
    raise ValueError(f"unsupported die-local weight kind {kind!r}")


def domain_stride_bytes(cfg, kind: str) -> int:
    """Compile-time slice stride of the die-local tensor ``kind``: one slice (``rows * row_bytes``) rounded up to the allocation granularity (2 MiB), exactly what
    ``locality.localize`` / DeepGEMM's allocator produce (``heuristics.get_perdie_domain_stride_bytes``; the frontend verifies the real tensor)."""
    from ..heuristics import get_perdie_domain_stride_bytes

    return get_perdie_domain_stride_bytes(cfg, kind)


def make_weight_pair(cfg, w_ptr, kind: str):
    """``(atom, tma_tensor)`` of the 3D map of die-local routed weights; ``w_ptr`` = ``w.data_ptr()`` = slice 0 (slice 1 = ``+ domain_stride_bytes``).
    Runs in the host ``@cute.jit __call__`` (DESIGN.md section 4.4) next to the 2D maps of ``tma_warps.make_tma_params``."""
    rows, row_bytes = weight_geometry(cfg, kind)
    dstride = domain_stride_bytes(cfg, kind)
    g_w = cute.make_tensor(w_ptr, cute.make_layout((rows, row_bytes, 2), stride=(row_bytes, 1, dstride)))
    op_ld = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
    return cpasync.make_tiled_tma_atom(op_ld, g_w, smem_layout_b3(cfg), (128, 64, 1), 1)


# ----------------------------------------------------------------------------------------------------------------------
# TMA-B (warp 5): tile coordinates of a 3D map (the K-block loop itself is ``tma_warps._b_k_loop``)
# ----------------------------------------------------------------------------------------------------------------------
@cute.jit
def b_tile_coords(local_expert_idx, n_block_idx, shape_n):
    """``K:934-936``: ``half_n = shape_n / 2``, ``n_in_matrix = n_block * BLOCK_N``, ``half_idx = n_in_matrix >= half_n``,
    ``half_row = expert * half_n + n_in_matrix - half_idx * half_n`` (routed).  Returns ``(half_row / 128, half_idx)`` (``half_n`` is a multiple of 128: 2304 / 2560, so every
    CTA tile lies inside one half and both CTAs of a cluster are in the same half: ``C / 2`` clusters per half, ``C`` = 18 / 20).  All three arguments are runtime ``Int32``."""
    half_n = shape_n // 2
    n_in_matrix = n_block_idx * 128
    half_idx = Int32(cutlass.select_(n_in_matrix >= half_n, Int32(1), Int32(0)))
    half_row = local_expert_idx * half_n + n_in_matrix - half_idx * half_n
    return half_row // 128, half_idx
