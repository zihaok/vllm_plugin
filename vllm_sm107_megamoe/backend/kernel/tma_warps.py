# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / common/tma_copy.cuh.
"""TMA-A (warp 4: tokens + token SF) and TMA-B (warp 5: packed-FP4 weights + weight SF) roles, K:827-1018.

Both warps run on BOTH CTAs of the pair.  Every load is the 2-SM form ``cp.async.bulk.tensor.2d.cta_group::2`` issued through
a CuteDSL TMA atom (``CopyBulkTensorTileG2SOp(CtaGroup.TWO)``): each CTA loads its own distinct tile into its own shared memory
and all ``complete_tx`` bytes land on the LEADER CTA's ``full[stage]`` barrier (peer bit of the barrier address cleared by the
atom).  Cache hint ``EVICT_NORMAL`` on every tile load (K: ``tma::copy`` default argument).

L2 promotion: the CuTe DSL atoms always encode ``CU_TENSOR_MAP_L2_PROMOTION_L2_128B`` while DeepGEMM uses L2_256B on every map.
For the weight maps (the DRAM-streaming operand) the 128 B promotion costs 2.6-3.9 % of kernel time at BLOCK_M >= 128 (EP1 GEMM-only A/B), so for those maps the host additionally creates
``TensorMap`` objects with an explicit promotion (``cutlass.experimental.cuda.tensor_map``; grid-constant kernel arguments = param-space descriptors like the atoms' own) and
the TMA-B warp passes their address to ``cute.copy(..., tma_desc_ptr=)`` (same instruction, only the descriptor differs).  Policy ``l2_promotion_bytes(cfg)``.

Tensor-map deviation: the packed-FP4 weight map is created as a ``UINT8`` map over the byte view of the
weights (box 64 B x 128 rows, SWIZZLE_64B) instead of ``16U4_ALIGN8B`` over FP4 elements (box 128 fp4 x 128 rows): the shared-memory
image and the transaction bytes are identical.  The token maps are SWIZZLE_128B UINT8 boxes (128 B x LOAD_BLOCK_M rows), SF maps
INT32 boxes (SF_BLOCK_M x 1) / (128 x 1) without swizzle, as in DeepGEMM.
"""

SUPPORTS_SHARED_EXPERTS = True  # DESIGN.md section 17: this module implements the shared-expert (S > 0) paths

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import dsl_user_op, extract_mlir_attributes, extract_mlir_values, new_from_mlir_values
from cutlass._mlir import ir as _ir
import dataclasses as _dc

try:  # experimental DSL API (CUTLASS DSL >= 4.8): host-created tensor maps with an explicit L2 promotion
    from cutlass.experimental.cuda.tensor_map import TensorMapL2Promotion as _TMProm, TensorMapSwizzle as _TMSwz, create_tensor_map_tiled as _create_tm
    from cutlass.tensor_utils import TensorMapManager as _TMM, TensorMapUpdateMode as _TMU

    HAVE_TENSOR_MAP_API = True
except Exception:  # pragma: no cover - DSL without the experimental TensorMap API: the atoms' own descriptors (L2_128B) are used
    HAVE_TENSOR_MAP_API = False

from . import ptx
from . import ptx_g
from . import perdie as pdw  # die-local (3D) weight tensor maps (kLocalizedWeights)
from . import scheduler as sched
from .scheduler import PHASE_LINEAR1, PHASE_LINEAR2, PHASE_NONE, PHASE_SHARED_LINEAR1

SHARED_BLOCK_K = 64  # half-K shared stage (packed-FP4 default: kSharedHalfK): K elements per shared stage

BLOCK_K = 128  # tokens K bytes per stage (FP8) / weights K elements per stage


# ----------------------------------------------------------------------------------------------------------------------
# L2 promotion of the weight tensor maps
# ----------------------------------------------------------------------------------------------------------------------
L2_PROMOTION_AUTO_MIN_BLOCK_M = 128  # measured (EP1 GEMM-only A/B): 256 B pays at BLOCK_M >= 128 (-2.6..-3.9 % kernel time), is neutral (+-0.9 %) below


def l2_promotion_bytes(cfg) -> int:
    """L2 promotion (bytes) of the weight tensor maps: ``0`` = keep the DSL atoms' own descriptors (the DSL encodes L2_128B); 64 / 128 / 256 = additionally create host
    tensor maps with that promotion (DeepGEMM: 256 on every map).  ``cfg.tma_l2_promotion`` (``None`` / absent = auto) forces a value; auto = 256 for
    ``BLOCK_M >= 128`` (where DeepGEMM's streaming of the weights is DRAM-bound) and for die-local weights (per-die, ``cfg.localized_weights``: DEP4 per-die S=1 A/B,
    256 B = -3.1..-3.5 % at BLOCK_M 16..64 as well) and 0 for the small block sizes with plain weights; always 0 when the DSL has no ``TensorMap`` API."""
    forced = getattr(cfg, "tma_l2_promotion", None)
    if forced is not None:
        forced = int(forced)
        if forced not in (0, 64, 128, 256):
            raise ValueError(f"tma_l2_promotion must be one of 0, 64, 128, 256 (got {forced})")
        if forced and not HAVE_TENSOR_MAP_API:
            raise RuntimeError("tma_l2_promotion > 0 needs cutlass.experimental.cuda.tensor_map (CUTLASS DSL >= 4.8)")
        return forced
    if not HAVE_TENSOR_MAP_API:
        return 0
    if getattr(cfg, "localized_weights", False):
        return 256
    return 256 if cfg.block_m >= L2_PROMOTION_AUTO_MIN_BLOCK_M else 0


_TENSORMAP_MGR = _TMM(_TMU.GMEM, 128) if HAVE_TENSOR_MAP_API else None  # used only for its pointer conversion (`get_tensormap_ptr`)


@dsl_user_op
def _tm_ptr(tm, *, loc=None, ip=None):
    """Generic-space ``!cute.ptr<tma_descriptor_tiled>`` of a grid-constant ``TensorMap`` kernel argument (what ``cute.copy(..., tma_desc_ptr=)`` takes)."""
    base = cute.make_ptr(cutlass.Int64, tm.get_ptr(loc=loc, ip=ip), assumed_align=128, loc=loc, ip=ip)
    return _TENSORMAP_MGR.get_tensormap_ptr(base, cute.AddressSpace.generic, loc=loc, ip=ip)


def desc_ptr(tp, name):
    """Descriptor pointer of the host tensor map ``name`` of ``tp`` (a ``TmaParams``), or ``None`` (static): the atom's own descriptor is used."""
    if name in tp.tm_names:
        return _tm_ptr(getattr(tp, name + "_tm"))
    return None


# ----------------------------------------------------------------------------------------------------------------------
# Shared-memory layouts of the TMA destinations (host and device must build the same objects)
# ----------------------------------------------------------------------------------------------------------------------
def smem_layout_a(cfg):
    """Tokens stage ring ``(LOAD_BLOCK_M, 128, stages)`` bytes, SWIZZLE_128B (``Swizzle<3,4,3>``), stage stride ``LOAD_BLOCK_M*128``."""
    return cute.make_composed_layout(
        cute.make_swizzle(3, 4, 3),
        0,
        cute.make_layout((cfg.load_block_m, 128, cfg.num_stages), stride=(128, 1, cfg.load_block_m * 128)),
    )


def smem_layout_a_shared(cfg):
    """Shared-expert (half-K) token stage ring ``(LOAD_BLOCK_M, 64, stages)`` bytes, SWIZZLE_64B (``Swizzle<2,4,3>``), same stage stride as the routed
    ring (``LOAD_BLOCK_M*128``; only the first half of a stage is used)."""
    return cute.make_composed_layout(
        cute.make_swizzle(2, 4, 3),
        0,
        cute.make_layout((cfg.load_block_m, 64, cfg.num_stages), stride=(64, 1, cfg.load_block_m * 128)),
    )


def smem_layout_b(cfg):
    """Packed-FP4 weights stage ring ``(128 rows, 64 B, stages)`` bytes, SWIZZLE_64B (``Swizzle<2,4,3>``), stage stride 8192 B."""
    return cute.make_composed_layout(
        cute.make_swizzle(2, 4, 3),
        0,
        cute.make_layout((128, 64, cfg.num_stages), stride=(64, 1, 8192)),
    )


def smem_layout_sfa(cfg):
    """Token SF stage ring ``(SF_BLOCK_M words, 1, stages)`` int32, no swizzle."""
    return cute.make_layout((cfg.sf_block_m, 1, cfg.num_stages), stride=(1, cfg.sf_block_m, cfg.sf_block_m))


def smem_layout_sfb(cfg):
    """Weight SF stage ring ``(128 words, 1, stages)`` int32, no swizzle."""
    return cute.make_layout((128, 1, cfg.num_stages), stride=(1, 128, 128))


def smem_layout_l1_out(cfg):
    """L1-output staging tile ``smem_d.l1[wg][stage]``: ``(STORE_BLOCK_M rows, 64 B)`` FP8, SWIZZLE_64B (``Swizzle<2,4,3>``)."""
    return cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, cute.make_layout((cfg.store_block_m, 64), stride=(64, 1)))


# ----------------------------------------------------------------------------------------------------------------------
# TmaParams: the (atom, tma tensor) pairs of the routed path (S = 0).  Flattenable kernel argument.
# ----------------------------------------------------------------------------------------------------------------------
class TmaParams:
    """Container of ``(atom, tma_tensor)`` pairs, passed as one opaque kernel argument (contract 4.4).

    Routed fields: ``l1_acts, l1_acts_sf, l1_w, l1_w_sf, l1_out (S2G store), l2_acts, l2_acts_sf, l2_w, l2_w_sf``; shared-expert fields (present only when
    ``cfg.num_shared_experts > 0``, the nine extra maps of K:84-92): ``shared_l1_acts, shared_l1_acts_sf, shared_l1_w, shared_l1_w_sf, shared_l1_out,
    shared_l2_acts, shared_l2_acts_sf, shared_l2_w, shared_l2_w_sf``.  ``self.fields`` = the names present (a static tuple).
    """

    ROUTED_FIELDS = ("l1_acts", "l1_acts_sf", "l1_w", "l1_w_sf", "l1_out", "l2_acts", "l2_acts_sf", "l2_w", "l2_w_sf")
    SHARED_FIELDS = ("shared_l1_acts", "shared_l1_acts_sf", "shared_l1_w", "shared_l1_w_sf", "shared_l1_out", "shared_l2_acts", "shared_l2_acts_sf",
                     "shared_l2_w", "shared_l2_w_sf")
    FIELDS = ROUTED_FIELDS  # kept for compatibility (S = 0 users)

    def __init__(self, _tms=None, **pairs):
        names = list(self.ROUTED_FIELDS)
        if "shared_l1_acts" in pairs:
            names += list(self.SHARED_FIELDS)
        self.fields = tuple(names)
        for name in self.fields:
            atom, tma_tensor = pairs[name]
            setattr(self, name + "_atom", atom)
            setattr(self, name + "_t", tma_tensor)
        # host-created tensor maps with an explicit L2 promotion (``_tms``: field name -> ``TensorMap``), grid-constant kernel arguments next to the atoms
        self.tm_names = tuple(_tms.keys()) if _tms else ()
        for name in self.tm_names:
            setattr(self, name + "_tm", _tms[name])

    def _leaves(self):
        out = []
        for name in self.fields:
            out.append(getattr(self, name + "_atom"))
            out.append(getattr(self, name + "_t"))
        return out

    def __extract_mlir_values__(self):
        values = []
        for leaf in self._leaves():
            values.extend(extract_mlir_values(leaf))
        for name in self.tm_names:
            values.append(getattr(self, name + "_tm").value)
        return values

    def __extract_mlir_attributes__(self):
        """Kernel-argument attributes of the leaves (TMA descriptor atoms are `grid_constant` params): without this the atoms lose their
        param-space attribute and `cute_nvgpu.atom.make_exec_tma` fails to legalize (same protocol as `cpasync.TmaInfo`)."""
        attrs = []
        for leaf in self._leaves():
            attrs.extend(extract_mlir_attributes(leaf))
        for _name in self.tm_names:  # `!cuda.tensor_map` values are `__grid_constant__` params too
            attrs.append(_ir.DictAttr.get({"cuda.grid_constant": _ir.UnitAttr.get()}))
        return attrs

    def __new_from_mlir_values__(self, values):
        index = 0
        pairs = {}
        for name in self.fields:
            new = []
            for leaf in (getattr(self, name + "_atom"), getattr(self, name + "_t")):
                n = len(extract_mlir_values(leaf))
                new.append(new_from_mlir_values(leaf, values[index:index + n]))
                index += n
            pairs[name] = tuple(new)
        tms = {}
        for name in self.tm_names:
            tms[name] = _dc.replace(getattr(self, name + "_tm"), value=values[index])
            index += 1
        assert index == len(values)
        return TmaParams(_tms=tms, **pairs)

    def prefetch(self):
        """``prefetch.tensormap`` of every descriptor (K:142-161; called by warp 0 in the prologue).

        NOTE: do not call this as a METHOD inside a dynamic region (`if warp_idx == 0:`): the DSL treats `tp.method()` as a possible
        mutation of `tp` and carries it through the region, which breaks the atoms; use the module-level `prefetch_tma_params(tp)`."""
        for name in self.fields:
            if name in self.tm_names:
                ptx_g.prefetch_tensormap(_tm_ptr(getattr(self, name + "_tm")))
            else:
                cpasync.prefetch_descriptor(getattr(self, name + "_atom"))


@cute.jit
def prefetch_tma_params(tp):
    """Module-level form of ``TmaParams.prefetch`` (safe inside ``if warp_idx == 0:``); the maps with a host tensor map prefetch THAT descriptor."""
    for name in tp.fields:
        if cutlass.const_expr(name in tp.tm_names):
            ptx_g.prefetch_tensormap(_tm_ptr(getattr(tp, name + "_tm")))
        else:
            cpasync.prefetch_descriptor(getattr(tp, name + "_atom"))


def make_tma_params(cfg, *, sym, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w=None, shared_l1_w_sf=None, shared_l2_w=None, shared_l2_w_sf=None):
    """Create the TMA atoms of the routed path (S = 0).  Runs in the host ``@cute.jit __call__`` (contract 4.4).

    Ring buffers come from the symmetric buffer (``sym.base`` + ``layout.buffer_layout(cfg)`` regions ``l1_token``/``l1_sf``/``l2_token``/
    ``l2_sf``); weights come from the kernel arguments: ``l1_w`` Uint8 packed fp4 ``(E_l*2I, H/2)``, ``l1_w_sf`` Int32 ``(2I, (H/128)*E_l)``
    MN-major, ``l2_w`` ``(E_l*H, I/2)``, ``l2_w_sf`` ``(H, (I/128)*E_l)``.
    """
    from .ctx import BufferMap

    sym_base = sym.base_address if hasattr(sym, "base_address") else sym.base  # comm.SymBufferHost payload or a test stand-in
    buf = BufferMap(sym_base, cfg)

    def region_ptr(name, dtype):
        return cute.make_ptr(dtype, buf.region_addr(name), cute.AddressSpace.gmem, assumed_align=16)

    H, I, El = cfg.hidden, cfg.intermediate_hidden, cfg.num_experts_per_rank
    ring, sfring = cfg.num_ring_tokens, cfg.num_sf_ring_tokens
    lbm, sbm, sfbm = cfg.load_block_m, cfg.store_block_m, cfg.sf_block_m
    op_ld = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.TWO)
    lay_a, lay_b = smem_layout_a(cfg), smem_layout_b(cfg)
    lay_sfa, lay_sfb = smem_layout_sfa(cfg), smem_layout_sfb(cfg)
    T = cute.make_tensor
    L = cute.make_layout

    def ld_pair(pointer, shape, stride, smem_layout, tiler):
        return cpasync.make_tiled_tma_atom(op_ld, T(pointer, L(shape, stride=stride)), smem_layout, tiler, 1)

    l1_token, l1_sf = region_ptr("l1_token", Uint8), region_ptr("l1_sf", Int32)
    l2_token, l2_sf = region_ptr("l2_token", Uint8), region_ptr("l2_sf", Int32)
    pairs = {}
    pairs["l1_acts"] = ld_pair(l1_token, (ring, H), (H, 1), lay_a, (lbm, 128))
    pairs["l1_acts_sf"] = ld_pair(l1_sf, (sfring, H // 128), (1, sfring), lay_sfa, (sfbm, 1))
    if cfg.localized_weights:  # die-local routed weights (2, E_l, N/2, K/2): 3D maps (rows of a slice, K bytes, domain), K:934-936,998-1000 / IMPL:163-176
        pairs["l1_w"] = pdw.make_weight_pair(cfg, l1_w, "routed_l1")
    else:
        pairs["l1_w"] = ld_pair(l1_w, (El * 2 * I, H // 2), (H // 2, 1), lay_b, (128, 64))
    pairs["l1_w_sf"] = ld_pair(l1_w_sf, (2 * I, (H // 128) * El), (1, 2 * I), lay_sfb, (128, 1))
    pairs["l2_acts"] = ld_pair(l2_token, (ring, I), (I, 1), lay_a, (lbm, 128))
    pairs["l2_acts_sf"] = ld_pair(l2_sf, (sfring, I // 128), (1, sfring), lay_sfa, (sfbm, 1))
    if cfg.localized_weights:
        pairs["l2_w"] = pdw.make_weight_pair(cfg, l2_w, "routed_l2")
    else:
        pairs["l2_w"] = ld_pair(l2_w, (El * H, I // 2), (I // 2, 1), lay_b, (128, 64))
    pairs["l2_w_sf"] = ld_pair(l2_w_sf, (H, (I // 128) * El), (1, H), lay_sfb, (128, 1))
    # L1 output store (TMA 2D S2G): box (64 B = post-SwiGLU channels of one N block, STORE_BLOCK_M rows), SWIZZLE_64B, into the
    # L2 activation ring.  The staging smem layout is smem_d.l1[wg][stage] = (STORE_BLOCK_M, 64) bytes with Swizzle<2,4,3>.
    lay_out = smem_layout_l1_out(cfg)
    pairs["l1_out"] = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(), T(l2_token, L((ring, I), stride=(I, 1))), lay_out, (sbm, 64)
    )
    if cfg.num_shared_experts > 0:
        # nine shared-expert maps (host `sm107_fp8_fp4_mega_moe.hpp:208-258`): half-K boxes (64 K elements, SWIZZLE_64B) for the shared tokens and the FP8 weights
        # (the FP8 weight tile has the smem image of the routed packed-FP4 tile: 128 rows x 64 B, SW64 -> `lay_b`); shared L1 input = the input token buffer
        S = cfg.num_shared_experts
        IS = I * S
        Tn = cfg.num_max_tokens_per_rank
        nsf = cfg.num_shared_sf_tokens
        lay_as = smem_layout_a_shared(cfg)
        x_token = region_ptr("input_token", Uint8)
        sh_l1_sf = region_ptr("shared_l1_sf", Int32)
        sh_l2_token, sh_l2_sf = region_ptr("shared_l2_token", Uint8), region_ptr("shared_l2_sf", Int32)
        pairs["shared_l1_acts"] = ld_pair(x_token, (Tn, H), (H, 1), lay_as, (lbm, 64))
        pairs["shared_l1_acts_sf"] = ld_pair(sh_l1_sf, (nsf, H // 128), (1, nsf), lay_sfa, (sfbm, 1))
        pairs["shared_l1_w"] = ld_pair(shared_l1_w, (2 * IS, H), (H, 1), lay_b, (128, 64))
        pairs["shared_l1_w_sf"] = ld_pair(shared_l1_w_sf, (2 * IS, H // 128), (1, 2 * IS), lay_sfb, (128, 1))
        pairs["shared_l1_out"] = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileS2GOp(), T(sh_l2_token, L((Tn, IS), stride=(IS, 1))), lay_out, (sbm, 64)
        )
        pairs["shared_l2_acts"] = ld_pair(sh_l2_token, (Tn, IS), (IS, 1), lay_as, (lbm, 64))
        pairs["shared_l2_acts_sf"] = ld_pair(sh_l2_sf, (nsf, IS // 128), (1, nsf), lay_sfa, (sfbm, 1))
        pairs["shared_l2_w"] = ld_pair(shared_l2_w, (H, IS), (IS, 1), lay_b, (128, 64))
        pairs["shared_l2_w_sf"] = ld_pair(shared_l2_w_sf, (H, IS // 128), (1, H), lay_sfb, (128, 1))
    tms = {}
    promo = l2_promotion_bytes(cfg)
    if promo:
        # host tensor maps (param-space descriptors) with an explicit L2 promotion for the weight maps: UINT8 byte maps, box (64 B, 128 rows), SWIZZLE_64B: the same
        # fields as the atoms' own descriptors except the promotion (the DSL atoms always encode L2_128B)
        prom = {64: _TMProm.l2_64b, 128: _TMProm.l2_128b, 256: _TMProm.l2_256b}[promo]

        def w_map(ptr, row_bytes, rows):
            return _create_tm(ptr.toint(), cutlass.Uint8, [row_bytes, rows], [row_bytes // 16], [64, 128], swizzle=_TMSwz.s64b, l2_promotion=prom)

        if cfg.localized_weights:  # 3D die-local maps (K bytes, rows of a slice, domain), slice stride compile-time
            for name, w_ptr, kind in (("l1_w", l1_w, "routed_l1"), ("l2_w", l2_w, "routed_l2")):
                rows_s, row_bytes = pdw.weight_geometry(cfg, kind)
                tms[name] = _create_tm(w_ptr.toint(), cutlass.Uint8, [row_bytes, rows_s, 2], [row_bytes // 16, pdw.domain_stride_bytes(cfg, kind) // 16], [64, 128, 1],
                                       swizzle=_TMSwz.s64b, l2_promotion=prom)
        else:
            tms["l1_w"] = w_map(l1_w, H // 2, El * 2 * I)
            tms["l2_w"] = w_map(l2_w, I // 2, El * H)
        if cfg.num_shared_experts > 0:  # FP8 shared weights (1 B / element): rows of K bytes
            tms["shared_l1_w"] = w_map(shared_l1_w, H, 2 * IS)
            tms["shared_l2_w"] = w_map(shared_l2_w, IS, H)
    return TmaParams(_tms=tms, **pairs)


# ----------------------------------------------------------------------------------------------------------------------
# Roles
# ----------------------------------------------------------------------------------------------------------------------

@cute.jit
def _spin_until_equal(ptr, expected):
    """``while (ptx::ld_acq(ptr) != expected);`` (K:854): ``ld.acquire.gpu.global.b32`` spin by all lanes."""
    while ptx.ld_acquire_gpu_b32(ptr) != expected:
        pass


@cute.jit
def _a_k_loop(
    ctx,
    atom_a,
    t_a,
    atom_sf,
    t_sf,
    num_k: Int32,
    block_idx: Int32,
    m_idx: Int32,
    stage: Int32,
    phase: Int32,
    is_l2: cutlass.Constexpr,
    mask_ptr,
    generation: Int32,
    kblk: Int32,
):
    """K-block loop of TMA-A for one task (K:862-910).  ``kblk`` = running k-block count (``trace_kblk_count``, advanced in trace builds only)."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    s_a = cute.make_tensor(ctx.smem.a(0), smem_layout_a(cfg))
    s_sf = cute.make_tensor(ctx.smem.sfa(0), smem_layout_sfa(cfg))
    g_a = cute.local_tile(cute.domain_offset((m_idx, 0), t_a), (cfg.load_block_m, 128), (0, None))
    g_sf = cute.local_tile(t_sf, (cfg.sf_block_m, 1), (block_idx, None))
    ta_s, ta_g = cpasync.tma_partition(atom_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2), cute.group_modes(g_a, 0, 2))
    ts_s, ts_g = cpasync.tma_partition(atom_sf, 0, cute.make_layout(1), cute.group_modes(s_sf, 0, 2), cute.group_modes(g_sf, 0, 2))
    num_a_bytes = cfg.load_block_m * BLOCK_K * 2 + cfg.sf_block_m * 4 * 2  # K:903-904 (both CTAs' bytes, 1 B/elem)

    pending = Int64(0)
    expected = Int64(0)
    if cutlass.const_expr(is_l2):
        # L2KBlockDependency (SCH0:140-166): kFullMask = (1 << kNumL1BlockNs) - 1; pending starts full
        full_mask = (1 << cfg.num_l1_block_ns) - 1 if cfg.num_l1_block_ns < 64 else -1  # (~0 as a signed 64-bit pattern)
        pending = Int64(full_mask)
        expected = Int64(full_mask)
        if (generation & 1) != 0:
            expected = Int64(0)

    for k_block_idx in cutlass.range(num_k, unroll=1):
        if cutlass.const_expr(is_l2):
            # while (pending & k_block_mask) pending = ld_acq_gpu(mask_ptr) ^ expected;  (2 L1 N blocks feed one L2 K block)
            k_mask = Int64(3) << Int64(k_block_idx * 2)
            tc = ctx.tr.clock()
            while (pending & k_mask) != Int64(0):
                pending = ptx.ld_acquire_gpu_b64(mask_ptr) ^ expected
            sched.trace_since(ctx, 17, tc, rec)
        full = ctx.mb.full(stage)
        tc = ctx.tr.clock()
        ctx.mb.empty(stage).wait(phase ^ 1)  # wait consumer release
        sched.trace_since(ctx, 16, tc, rec)
        sched.trace_event(ctx, 3, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        cute.copy(atom_a, ta_g[(None, k_block_idx)], ta_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        cute.copy(atom_sf, ts_g[(None, k_block_idx)], ts_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        if ctx.is_leader_cta:
            with cute.arch.elect_one():
                full.arrive_expect_tx(num_a_bytes)
        else:
            with cute.arch.elect_one():
                full.arrive_remote(0)
        cute.arch.sync_warp()
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


@cute.jit
def _a_k_loop_shared(ctx, atom_a, t_a, atom_sf, t_sf, num_k: Int32, block_idx: Int32, m_idx: Int32, stage: Int32, phase: Int32, kblk: Int32):
    """K-block loop of TMA-A for one SHARED-expert task (half-K stages, K:862-910 with ``half_k``): 64-K SWIZZLE_64B token boxes, SF word ``k_block / 2``
    (the same word serves two consecutive half-K stages)."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    s_a = cute.make_tensor(ctx.smem.a(0), smem_layout_a_shared(cfg))
    s_sf = cute.make_tensor(ctx.smem.sfa(0), smem_layout_sfa(cfg))
    g_a = cute.local_tile(cute.domain_offset((m_idx, 0), t_a), (cfg.load_block_m, SHARED_BLOCK_K), (0, None))
    g_sf = cute.local_tile(t_sf, (cfg.sf_block_m, 1), (block_idx, None))
    ta_s, ta_g = cpasync.tma_partition(atom_a, 0, cute.make_layout(1), cute.group_modes(s_a, 0, 2), cute.group_modes(g_a, 0, 2))
    ts_s, ts_g = cpasync.tma_partition(atom_sf, 0, cute.make_layout(1), cute.group_modes(s_sf, 0, 2), cute.group_modes(g_sf, 0, 2))
    num_a_bytes = cfg.load_block_m * SHARED_BLOCK_K * 2 + cfg.sf_block_m * 4 * 2  # (LOAD_BLOCK_M x 64 x 1 B + smem_sfa[0]) x 2 CTAs

    for k_block_idx in cutlass.range(num_k, unroll=1):
        full = ctx.mb.full(stage)
        tc = ctx.tr.clock()
        ctx.mb.empty(stage).wait(phase ^ 1)  # wait consumer release
        sched.trace_since(ctx, 16, tc, rec)
        sched.trace_event(ctx, 3, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        cute.copy(atom_a, ta_g[(None, k_block_idx)], ta_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        cute.copy(atom_sf, ts_g[(None, k_block_idx // 2)], ts_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        if ctx.is_leader_cta:
            with cute.arch.elect_one():
                full.arrive_expect_tx(num_a_bytes)
        else:
            with cute.arch.elect_one():
                full.arrive_remote(0)
        cute.arch.sync_warp()
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


@cute.jit
def tma_a_role(ctx):
    """TMA-A warp (warp 4, both CTAs): tokens + token SF; K:827-911."""
    cfg = ctx.cfg
    tp = ctx.tma
    sched_stage = Int32(0)
    sched_phase = Int32(0)
    stage = Int32(0)
    phase = Int32(0)
    kblk = Int32(0)
    rec = ctx.lane_idx == 0  # trace recorder lane
    tc = ctx.tr.clock()
    t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
    sched.trace_since(ctx, 18, tc, rec)
    while t0 != PHASE_NONE:
        pool_block_idx = t4
        valid_m = t5
        shape_k = t7
        num_k = (shape_k + (BLOCK_K - 1)) // BLOCK_K
        ring_block_idx = pool_block_idx % cfg.num_ring_blocks
        block_idx = ring_block_idx  # routed tasks
        generation = pool_block_idx // cfg.num_ring_blocks
        aligned_valid_m = (valid_m + 15) // 16 * 16
        if cutlass.const_expr(cfg.num_shared_experts > 0):
            is_shared = t0 >= PHASE_SHARED_LINEAR1
            if is_shared:
                block_idx = pool_block_idx  # shared tasks: no ring (K:848)
        m_idx = block_idx * cfg.block_m
        if not ctx.is_leader_cta:
            m_idx = m_idx + aligned_valid_m // 2  # K:879-880
        if t0 == PHASE_LINEAR1:
            # wait the entire token arrival (K:851-855)
            tc = ctx.tr.clock()
            _spin_until_equal(ctx.ws.l1_full_count(block_idx), Int32(cfg.block_m) * (generation + 1))
            sched.trace_since(ctx, 17, tc, rec)
            stage, phase, kblk = _a_k_loop(ctx, tp.l1_acts_atom, tp.l1_acts_t, tp.l1_acts_sf_atom, tp.l1_acts_sf_t, num_k, block_idx,
                                           m_idx, stage, phase, False, ctx.ws.l2_full_mask(block_idx), generation, kblk)
        else:
            if cutlass.const_expr(cfg.num_shared_experts > 0):
                num_k_shared = shape_k // SHARED_BLOCK_K
                if t0 == PHASE_LINEAR2:
                    stage, phase, kblk = _a_k_loop(ctx, tp.l2_acts_atom, tp.l2_acts_t, tp.l2_acts_sf_atom, tp.l2_acts_sf_t, num_k, block_idx,
                                                   m_idx, stage, phase, True, ctx.ws.l2_full_mask(block_idx), generation, kblk)
                elif t0 == PHASE_SHARED_LINEAR1:
                    stage, phase, kblk = _a_k_loop_shared(ctx, tp.shared_l1_acts_atom, tp.shared_l1_acts_t, tp.shared_l1_acts_sf_atom,
                                                          tp.shared_l1_acts_sf_t, num_k_shared, block_idx, m_idx, stage, phase, kblk)
                else:
                    # SharedLinear2: wait until every shared L1 N block of this token block has been stored (K:855-859)
                    tc = ctx.tr.clock()
                    _spin_until_equal(ctx.ws.shared_l2_full_count(block_idx), Int32(cfg.shared_l2_shape_k // 128 * 2))
                    sched.trace_since(ctx, 19, tc, rec)
                    stage, phase, kblk = _a_k_loop_shared(ctx, tp.shared_l2_acts_atom, tp.shared_l2_acts_t, tp.shared_l2_acts_sf_atom,
                                                          tp.shared_l2_acts_sf_t, num_k_shared, block_idx, m_idx, stage, phase, kblk)
            else:
                stage, phase, kblk = _a_k_loop(ctx, tp.l2_acts_atom, tp.l2_acts_t, tp.l2_acts_sf_atom, tp.l2_acts_sf_t, num_k, block_idx,
                                               m_idx, stage, phase, True, ctx.ws.l2_full_mask(block_idx), generation, kblk)
        tc = ctx.tr.clock()
        t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
        sched.trace_since(ctx, 18, tc, rec)


@cute.jit
def _b_k_loop(
    ctx,
    atom_w,
    t_w,
    atom_sf,
    t_sf,
    num_k: Int32,
    w_tile_row: Int32,
    n_block_idx: Int32,
    sfb_k_base: Int32,
    stage: Int32,
    phase: Int32,
    kblk: Int32,
    w_domain=0,
    desc_w=None,
):
    """K-block loop of TMA-B for one task (K:955-1017).  ``kblk`` = running k-block count (``trace_kblk_count``, advanced in trace builds only)."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    s_sf = cute.make_tensor(ctx.smem.sfb(0), smem_layout_sfb(cfg))
    g_sf = cute.local_tile(t_sf, (128, 1), (n_block_idx, None))
    if cutlass.const_expr(cfg.localized_weights):  # die-local weights = 3D map (rows of the slice, K bytes, domain): tile (half_row / 128, k, half_idx) (K:998-1000)
        s_b = cute.make_tensor(ctx.smem.b(0), pdw.smem_layout_b3(cfg))
        g_w = cute.local_tile(t_w, (128, 64, 1), (w_tile_row, None, w_domain))
        tb_s, tb_g = cpasync.tma_partition(atom_w, 0, cute.make_layout(1), cute.group_modes(s_b, 0, 3), cute.group_modes(g_w, 0, 3))
    else:
        s_b = cute.make_tensor(ctx.smem.b(0), smem_layout_b(cfg))
        g_w = cute.local_tile(t_w, (128, 64), (w_tile_row, None))
        tb_s, tb_g = cpasync.tma_partition(atom_w, 0, cute.make_layout(1), cute.group_modes(s_b, 0, 2), cute.group_modes(g_w, 0, 2))
    ts_s, ts_g = cpasync.tma_partition(atom_sf, 0, cute.make_layout(1), cute.group_modes(s_sf, 0, 2), cute.group_modes(g_sf, 0, 2))
    num_b_bytes = 128 * BLOCK_K * 2 // 2 + 128 * 4 * 2  # K:1009-1010: 128 rows x 128 fp4 (1/2 B) x 2 CTAs + SFB words x 2

    for k_block_idx in cutlass.range(num_k, unroll=1):
        full = ctx.mb.full(stage)
        tc = ctx.tr.clock()
        ctx.mb.empty(stage).wait(phase ^ 1)
        sched.trace_since(ctx, 20, tc, rec)
        sched.trace_event(ctx, 0, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        if cutlass.const_expr(desc_w is None):
            cute.copy(atom_w, tb_g[(None, k_block_idx)], tb_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        else:  # host tensor map with an explicit L2 promotion
            cute.copy(atom_w, tb_g[(None, k_block_idx)], tb_s[(None, stage)], tma_bar_ptr=full.ptr, tma_desc_ptr=desc_w, cache_policy=Int64(ptx.EVICT_NORMAL))
        cute.copy(atom_sf, ts_g[(None, sfb_k_base + k_block_idx)], ts_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        if ctx.is_leader_cta:
            with cute.arch.elect_one():
                full.arrive_expect_tx(num_b_bytes)
        else:
            with cute.arch.elect_one():
                full.arrive_remote(0)
        cute.arch.sync_warp()
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


@cute.jit
def _b_k_loop_shared(ctx, atom_w, t_w, atom_sf, t_sf, num_k: Int32, n_block_idx: Int32, stage: Int32, phase: Int32, kblk: Int32, desc_w=None):
    """K-block loop of TMA-B for one SHARED-expert task (half-K stages, K:955-1017 shared branch): FP8 weights as 64-K x 128-row boxes (SW64, the same 8 KB smem
    image as a routed packed-FP4 tile), no expert offset in the weight row / SF coordinates, SF word ``k_block / 2``."""
    cfg = ctx.cfg
    rec = ctx.lane_idx == 0  # trace recorder lane
    s_b = cute.make_tensor(ctx.smem.b(0), smem_layout_b(cfg))
    s_sf = cute.make_tensor(ctx.smem.sfb(0), smem_layout_sfb(cfg))
    g_w = cute.local_tile(t_w, (128, SHARED_BLOCK_K), (n_block_idx, None))
    g_sf = cute.local_tile(t_sf, (128, 1), (n_block_idx, None))
    tb_s, tb_g = cpasync.tma_partition(atom_w, 0, cute.make_layout(1), cute.group_modes(s_b, 0, 2), cute.group_modes(g_w, 0, 2))
    ts_s, ts_g = cpasync.tma_partition(atom_sf, 0, cute.make_layout(1), cute.group_modes(s_sf, 0, 2), cute.group_modes(g_sf, 0, 2))
    num_b_bytes = 128 * SHARED_BLOCK_K * 2 + 128 * 4 * 2  # kNumSharedWeightBytes (LOAD_BLOCK_N x 64 x 1 B x 2 CTAs) + smem_sfb[0] x 2

    for k_block_idx in cutlass.range(num_k, unroll=1):
        full = ctx.mb.full(stage)
        tc = ctx.tr.clock()
        ctx.mb.empty(stage).wait(phase ^ 1)
        sched.trace_since(ctx, 20, tc, rec)
        sched.trace_event(ctx, 0, kblk, rec)
        if cutlass.const_expr(ctx.tr.enabled):
            kblk = kblk + 1
        if cutlass.const_expr(desc_w is None):
            cute.copy(atom_w, tb_g[(None, k_block_idx)], tb_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        else:
            cute.copy(atom_w, tb_g[(None, k_block_idx)], tb_s[(None, stage)], tma_bar_ptr=full.ptr, tma_desc_ptr=desc_w, cache_policy=Int64(ptx.EVICT_NORMAL))
        cute.copy(atom_sf, ts_g[(None, k_block_idx // 2)], ts_s[(None, stage)], tma_bar_ptr=full.ptr, cache_policy=Int64(ptx.EVICT_NORMAL))
        if ctx.is_leader_cta:
            with cute.arch.elect_one():
                full.arrive_expect_tx(num_b_bytes)
        else:
            with cute.arch.elect_one():
                full.arrive_remote(0)
        cute.arch.sync_warp()
        stage, phase = ptx.advance_stage(stage, phase, cfg.num_stages)
    return stage, phase, kblk


def require_default_tma_options(cfg) -> None:
    """``cfg.weight_prefetch`` (kWeightPrefetch: L2 prefetch of the routed weight tiles k-blocks ahead of the smem pipeline) is part of the
    compile key and of the parity manifest but is not implemented by the CuteDSL TMA-B warp; a config with another value would compile an unchanged kernel.  Trace time only."""
    if cfg.weight_prefetch != 0:
        raise NotImplementedError(f"cfg.weight_prefetch={cfg.weight_prefetch!r} (kWeightPrefetch: L2 prefetch of the weight tiles) is not implemented by the CuteDSL port (TMA-B implements only weight_prefetch=0)")


@cute.jit
def tma_b_role(ctx):
    """TMA-B warp (warp 5, both CTAs): packed-FP4 weights + weight SF; K:912-1018."""
    cfg = ctx.cfg
    require_default_tma_options(cfg)
    tp = ctx.tma
    sched_stage = Int32(0)
    sched_phase = Int32(0)
    stage = Int32(0)
    phase = Int32(0)
    kblk = Int32(0)
    rec = ctx.lane_idx == 0  # trace recorder lane
    dw1 = desc_ptr(tp, "l1_w")  # host tensor maps with an explicit L2 promotion (None: the atom's own descriptor)
    dw2 = desc_ptr(tp, "l2_w")
    dsw1 = desc_ptr(tp, "shared_l1_w")
    dsw2 = desc_ptr(tp, "shared_l2_w")
    tc = ctx.tr.clock()
    t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
    sched.trace_since(ctx, 21, tc, rec)
    while t0 != PHASE_NONE:
        local_expert_idx = t1
        n_cluster_idx = t3
        shape_n = t6
        shape_k = t7
        shape_sfb_k = (shape_k + 127) // 128
        num_k = (shape_k + (BLOCK_K - 1)) // BLOCK_K
        n_block_idx = n_cluster_idx * 2
        if not ctx.is_leader_cta:
            n_block_idx = n_block_idx + 1
        # weight rows: n_idx = local_expert * shape_n + n_block * 128 -> tile row index n_idx / 128 (K:965)
        w_tile_row = local_expert_idx * (shape_n // 128) + n_block_idx
        w_domain = Int32(0)
        if cutlass.const_expr(cfg.localized_weights):  # die-local weights: (half_row / 128, half_idx) of the 3D map, half_idx = n_block * 128 >= shape_n / 2 (K:934-936)
            w_tile_row, w_domain = pdw.b_tile_coords(local_expert_idx, n_block_idx, shape_n)
        sfb_k_base = local_expert_idx * shape_sfb_k  # K:968 (routed)
        if t0 == PHASE_LINEAR1:
            stage, phase, kblk = _b_k_loop(ctx, tp.l1_w_atom, tp.l1_w_t, tp.l1_w_sf_atom, tp.l1_w_sf_t, num_k, w_tile_row, n_block_idx,
                                           sfb_k_base, stage, phase, kblk, w_domain, dw1)
        else:
            if cutlass.const_expr(cfg.num_shared_experts > 0):
                num_k_shared = shape_k // SHARED_BLOCK_K
                if t0 == PHASE_LINEAR2:
                    stage, phase, kblk = _b_k_loop(ctx, tp.l2_w_atom, tp.l2_w_t, tp.l2_w_sf_atom, tp.l2_w_sf_t, num_k, w_tile_row, n_block_idx,
                                                   sfb_k_base, stage, phase, kblk, w_domain, dw2)
                elif t0 == PHASE_SHARED_LINEAR1:
                    stage, phase, kblk = _b_k_loop_shared(ctx, tp.shared_l1_w_atom, tp.shared_l1_w_t, tp.shared_l1_w_sf_atom, tp.shared_l1_w_sf_t,
                                                          num_k_shared, n_block_idx, stage, phase, kblk, dsw1)
                else:
                    stage, phase, kblk = _b_k_loop_shared(ctx, tp.shared_l2_w_atom, tp.shared_l2_w_t, tp.shared_l2_w_sf_atom, tp.shared_l2_w_sf_t,
                                                          num_k_shared, n_block_idx, stage, phase, kblk, dsw2)
            else:
                stage, phase, kblk = _b_k_loop(ctx, tp.l2_w_atom, tp.l2_w_t, tp.l2_w_sf_atom, tp.l2_w_sf_t, num_k, w_tile_row, n_block_idx,
                                               sfb_k_base, stage, phase, kblk, w_domain, dw2)
        tc = ctx.tr.clock()
        t0, t1, t2, t3, t4, t5, t6, t7, sched_stage, sched_phase = sched.get_next_task(ctx, sched_stage, sched_phase)
        sched.trace_since(ctx, 21, tc, rec)
