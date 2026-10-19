# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / layout/mega_moe.cuh.
"""Device-side views handed to the role functions: :class:`KernelCtx` and its parts (DESIGN.md section 4).

Everything here is traced inside ``@cute.kernel`` code; the classes are plain Python containers of DSL values (ids, pointers, barriers).
They are READ-ONLY inside roles (state that changes across loop iterations must live in local variables, see DESIGN.md section 4.1).
"""

from __future__ import annotations


import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64
from cutlass.cute.typing import AddressSpace

from . import layout as L
from .dslutil import DslStatic, init_fields as _init, static_dataclass
from . import ptx
from .config import (
    BAR_DISPATCH,
    BAR_DISPATCH_EPILOGUE,
    BAR_EPILOGUE,
    BAR_EPILOGUE_WG0,
    NUM_DISPATCH_EPILOGUE_THREADS,
    NUM_DISPATCH_THREADS,
    NUM_EPILOGUE_THREADS,
    KernelBuild,
)


def _gptr(dtype, addr, align):
    """Typed gmem ``cute.Pointer`` at the (runtime) generic address ``addr``."""
    return cute.make_ptr(dtype, addr, AddressSpace.gmem, assumed_align=align)


# ---------------------------------------------------------------------------------------------------------------------
# shared memory
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class SmemMap(DslStatic):
    """Typed views of ``SharedStorage`` (DESIGN.md section 6). ``base`` is the 1024-aligned dynamic smem start (Uint8 smem pointer)."""

    def __init__(self, base, cfg):
        _init(
            self, base=base, cfg=cfg,
            off=dict(cfg.smem_offsets) if getattr(cfg, "smem_offsets", None) else L.smem_offsets_for_cfg(cfg),
            # stage strides in bytes
            a_stage_bytes=cfg.load_block_m * cfg.block_k,  # fp8
            b_stage_bytes=128 * cfg.block_k // 2,  # packed fp4: 8192
            sfa_stage_bytes=cfg.sf_block_m * 4,
            sfb_stage_bytes=cfg.sf_block_n * 4,
            task_info_bytes=L.TASK_INFO_BYTES,
        )  # fmt: skip

    # -- generic -------------------------------------------------------------------------------------------------------
    def at(self, name: str, byte_off=0, dtype=cutlass.Uint8):
        """Pointer to ``name`` + ``byte_off`` bytes, recast to ``dtype`` (``byte_off`` may be a runtime Int32)."""
        p = self.base + (self.off[name] + byte_off)
        return p if dtype is cutlass.Uint8 else cute.recast_ptr(p, dtype=dtype)

    def addr(self, ptr):
        """32-bit shared::cta address (Int32) of a smem pointer."""
        return ptr.toint()

    # -- named members ---------------------------------------------------------------------------------------------------
    @property
    def expert_token_count(self):
        return self.at("expert_token_count", 0, cutlass.Uint32)

    @property
    def send_buffer(self):
        return self.at("dispatch_send_buffer")

    def d_l1(self, wg, stage):
        """``smem_d.l1[wg][stage]``: fp8, ``STORE_BLOCK_M*64`` bytes each."""
        sbm = self.cfg.store_block_m
        return self.at("smem_d", (wg * 2 + stage) * (sbm * 64))

    def d_l2(self, wg):
        """``smem_d.l2[wg]``: bf16, ``STORE_BLOCK_M*128`` elements (= ``*256`` bytes) each."""
        sbm = self.cfg.store_block_m
        return self.at("smem_d", wg * (sbm * 256), cutlass.BFloat16)

    def a(self, stage):
        return self.at("smem_a", stage * self.a_stage_bytes)

    def b(self, stage):
        return self.at("smem_b", stage * self.b_stage_bytes)

    def sfa(self, stage):
        return self.at("smem_sfa", stage * self.sfa_stage_bytes, cutlass.Uint32)

    def sfb(self, stage):
        return self.at("smem_sfb", stage * self.sfb_stage_bytes, cutlass.Uint32)

    def amax(self, warp):
        """``amax_reduction[warp]``: ``STORE_BLOCK_M/2`` float2 (= 8 B each)."""
        return self.at("amax_reduction", warp * (self.cfg.store_block_m // 2) * 8, cutlass.Float32)

    def task_info(self, slot):
        return self.at("task_infos", slot * self.task_info_bytes, cutlass.Uint32)

    @property
    def tmem_ptr(self):
        return self.at("tmem_ptr_in_smem", 0, cutlass.Uint32)

    def reusable(self, byte_off=0):
        """Combine overlay: any byte offset inside ``[0, reusable_bytes)`` (dead after the GEMM phase)."""
        return self.base + byte_off


# ---------------------------------------------------------------------------------------------------------------------
# mbarriers
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class Barriers(DslStatic):
    """Handles to the mbarriers of ``SharedStorage`` (DESIGN.md section 7). Every accessor returns a :class:`ptx.Mbar`."""

    def __init__(self, smem: SmemMap):
        _init(self, _s=smem)

    def _mb(self, name: str, i):
        # NOTE: index in Int64 units AFTER the recast, so that a runtime stage index keeps the 8-byte alignment
        # information the TMA atom (`tma_bar`) requires.
        return ptx.Mbar(cute.recast_ptr(self._s.base + self._s.off[name], dtype=cutlass.Int64) + i)

    def dispatch(self, i):
        return self._mb("dispatch_barriers", i)

    def full(self, stage):
        return self._mb("full_barriers", stage)

    def empty(self, stage):
        return self._mb("empty_barriers", stage)

    def tmem_full(self, accum_stage):
        return self._mb("tmem_full_barriers", accum_stage)

    def tmem_empty(self, accum_stage):
        return self._mb("tmem_empty_barriers", accum_stage)

    def combine(self, i):
        return self._mb("combine_barriers", i)

    def task_info_full(self, slot):
        return self._mb("task_info_full_barriers", slot)

    def task_info_empty(self, slot):
        return self._mb("task_info_empty_barriers", slot)


# ---------------------------------------------------------------------------------------------------------------------
# tensor memory (plain Python ints: the allocation base column is 0 on both CTAs)
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class TmemMap(DslStatic):
    def __init__(self, cfg):
        _init(self, cfg=cfg, alloc_cols=cfg.tmem_cols_alloc, sfa_col=cfg.tmem_sfa_col, sfb_col=cfg.tmem_sfb_col,
              num_cols_used=cfg.tmem_cols_used)  # fmt: skip

    def accum_col(self, stage):
        """TMEM column of accumulator stage ``stage`` (stage may be a runtime Int32): ``stage * UMMA_N``."""
        return stage * self.cfg.umma_n


# ---------------------------------------------------------------------------------------------------------------------
# named barriers (ids 0..4 reserved)
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class NamedBarriers(DslStatic):
    def dispatch(self):
        """``bar.sync 0, 128`` (dispatch warps)."""
        ptx.bar_sync(BAR_DISPATCH, NUM_DISPATCH_THREADS)

    def dispatch_epilogue_sync(self):
        """``barrier.sync 1, 384`` (UNALIGNED): R1/R2 rendezvous of the dispatch warps with the epilogue warps."""
        ptx.bar_sync_unaligned(BAR_DISPATCH_EPILOGUE, NUM_DISPATCH_EPILOGUE_THREADS)

    def epilogue(self):
        """``bar.sync 2, 256`` (epilogue warps)."""
        ptx.bar_sync(BAR_EPILOGUE, NUM_EPILOGUE_THREADS)

    def epilogue_wg(self, wg):
        """``bar.sync 3+wg, 128`` (one epilogue warpgroup)."""
        ptx.bar_sync(BAR_EPILOGUE_WG0 + wg, 128)


# ---------------------------------------------------------------------------------------------------------------------
# global workspace (MegaMoESignals + tails): typed pointers into the LOCAL symmetric buffer
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class Workspace(DslStatic):
    """Accessors of ``layout::Workspace``. ``base`` = local symmetric-buffer base (Int64 generic address)."""

    def __init__(self, base, cfg):
        _init(
            self, base=base, cfg=cfg, num_ranks=cfg.num_ranks, epr=cfg.num_experts_per_rank, T=cfg.num_max_tokens_per_rank,
            src_token_topk_offset=L.SIGNALS_SIZEOF,
            token_src_metadata_offset=L.SIGNALS_SIZEOF + cfg.num_experts * cfg.num_max_tokens_per_rank * 4,
        )  # fmt: skip

    def addr(self, off, idx=0, stride=0):
        """Int64 generic address of byte ``off + idx*stride``."""
        if isinstance(idx, int) and idx == 0:
            return self.base + Int64(off)
        return self.base + Int64(off) + Int64(idx) * Int64(stride)

    def _p(self, dtype, off, idx=0, stride=0, align=None):
        return _gptr(dtype, self.addr(off, idx, stride), align or (dtype.width // 8))

    # --- grid/NVLink synchronisation
    def grid_sync_count(self, idx):
        return self._p(cutlass.Uint32, L.SIGNALS_GRID_SYNC_COUNT + 4 * idx) if isinstance(idx, int) else self._p(
            cutlass.Uint32, L.SIGNALS_GRID_SYNC_COUNT, idx, 4)

    def nvl_barrier_counter(self):
        return self._p(cutlass.Uint32, L.SIGNALS_NVL_BARRIER_COUNTER)

    def nvl_barrier_signal(self, phase):
        return self._p(cutlass.Int32, L.SIGNALS_NVL_BARRIER_SIGNALS, phase, 4)

    # --- scheduling
    def l1_task_count(self, dom=0):
        return self._p(cutlass.Uint32, L.SIGNALS_L1_TASK_COUNT, dom, 4)

    def l2_task_count(self, dom=0):
        return self._p(cutlass.Uint32, L.SIGNALS_L2_TASK_COUNT, dom, 4)

    def shared_l1_task_count(self, dom=0):
        return self._p(cutlass.Uint32, L.SIGNALS_SHARED_L1_TASK_COUNT, dom, 4)

    def shared_l2_task_count(self, dom=0):
        return self._p(cutlass.Uint32, L.SIGNALS_SHARED_L2_TASK_COUNT, dom, 4)

    # --- combine readiness (peer indexed)
    def combine_ready_grid_idx(self, rank=0):
        return self._p(cutlass.Uint64, L.SIGNALS_COMBINE_READY_GRID_IDX, rank, 8)

    def peer_grid_idx(self, rank=0):
        return self._p(cutlass.Uint64, L.SIGNALS_PEER_GRID_IDX, rank, 8)

    # --- expert counts
    def expert_send_count(self, expert=0):
        return self._p(cutlass.Uint64, L.SIGNALS_EXPERT_SEND_COUNT, expert, 8)

    def expert_recv_count(self, src_rank=0, local_expert=0):
        return self._p(cutlass.Uint64, L.SIGNALS_EXPERT_RECV_COUNT, Int32(src_rank) * self.epr + local_expert, 8)

    def expert_recv_count_sum(self, local_expert=0):
        return self._p(cutlass.Uint64, L.SIGNALS_EXPERT_RECV_COUNT_SUM, local_expert, 8)

    # --- ring signals (index = ring block)
    def l1_full_count(self, ring_block=0):
        return self._p(cutlass.Uint32, L.SIGNALS_L1_FULL_COUNT, ring_block, 4)

    def l1_empty_count(self, ring_block=0):
        return self._p(cutlass.Uint32, L.SIGNALS_L1_EMPTY_COUNT, ring_block, 4)

    def l2_full_mask(self, ring_block=0):
        return self._p(cutlass.Uint64, L.SIGNALS_L2_FULL_MASK, ring_block, 8)

    def l2_empty_count(self, ring_block=0):
        return self._p(cutlass.Uint32, L.SIGNALS_L2_EMPTY_COUNT, ring_block, 4)

    def shared_l2_full_count(self, block=0):
        return self._p(cutlass.Uint32, L.SIGNALS_SHARED_L2_FULL_COUNT, block, 4)

    # --- tails
    def src_token_topk_idx(self, local_expert=0, src_rank=0, slot=0):
        """u32 at ``((local_expert*R + src_rank)*T + slot)*4`` after the signals (LAY get_src_token_topk_idx_ptr)."""
        idx = (Int64(local_expert) * self.num_ranks + Int64(src_rank)) * self.T + Int64(slot)
        return self._p(cutlass.Uint32, self.src_token_topk_offset, idx, 4)

    def token_src_metadata(self, pool_token=0):
        """12-byte record ``{rank_idx, token_idx, topk_idx}``; returns a Uint32 pointer to the first word."""
        return self._p(cutlass.Uint32, self.token_src_metadata_offset, pool_token, L.TOKEN_SRC_METADATA_BYTES)


# ---------------------------------------------------------------------------------------------------------------------
# symmetric-buffer regions (Int64 generic addresses of the LOCAL buffer)
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class BufferMap(DslStatic):
    def __init__(self, base, cfg):
        _init(
            self, base=base, cfg=cfg,
            layout=L.buffer_layout(
                hidden=cfg.hidden, intermediate_hidden=cfg.intermediate_hidden, num_ranks=cfg.num_ranks,
                num_experts=cfg.num_experts, num_max_tokens_per_rank=cfg.num_max_tokens_per_rank, num_topk=cfg.num_topk,
                num_ring_tokens=cfg.num_ring_tokens, num_sf_ring_tokens=cfg.num_sf_ring_tokens,
                num_shared_experts=cfg.num_shared_experts,
            ),
        )  # fmt: skip

    def region_addr(self, name: str, token=0, slot=0):
        """Int64 address of ``name[slot][token]`` (``slot`` only for ``combine_token``).

        EVERY byte-offset term is evaluated in Int64 and every product of Python ints is folded on the host.  A DSL ``Int32 * python_int`` stays Int32 (it wraps modulo
        2**32), so the former ``r.offset + slot * token_bytes * num_tokens`` with a runtime Int32 ``slot`` addressed the wrong combine slot as soon as
        ``offset + slot * token_bytes * num_tokens`` crossed 2 GiB (e.g. DEP32 E=384 with T_max >= 7680, EP1 E=96 with T_max >= 32640; DESIGN.md section 16 "Int32 address
        overflow").  The runtime ``slot`` / ``token`` operands are sign-extended first (``Int64(x)``) and multiplied by a host-folded Python-int stride.
        """
        r = self.layout[name]
        slot_stride = r.token_bytes * r.num_tokens  # python int: bytes of one slot (= one "rank" Buffer of the layout)
        addr = self.base + Int64(r.offset)
        if isinstance(slot, int):
            if slot != 0:
                addr = addr + Int64(slot * slot_stride)
        else:
            addr = addr + Int64(slot) * Int64(slot_stride)
        if isinstance(token, int):
            if token != 0:
                addr = addr + Int64(token * r.token_bytes)
        else:
            addr = addr + Int64(token) * Int64(r.token_bytes)
        return addr

    def token_bytes(self, name: str) -> int:
        return self.layout[name].token_bytes


# ---------------------------------------------------------------------------------------------------------------------
# the context object
# ---------------------------------------------------------------------------------------------------------------------
@static_dataclass
class KernelCtx(DslStatic):
    """Identical, read-only object passed to every role function (see DESIGN.md section 4.2)."""

    def __init__(
        self, *, build: KernelBuild, sm_idx, thread_idx, warp_idx, lane_idx, cta_rank, y, stats, num_tokens, sym,
        l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w, shared_l1_w_sf, shared_l2_w, shared_l2_w_sf, trace_ptr,
        sm_locality_domains, tma, smem: SmemMap, tr, launch_tag=None,
    ):  # fmt: skip
        smem_map = smem
        _init(
            self, build=build, cfg=build.cfg, sm_idx=sm_idx, thread_idx=thread_idx, warp_idx=warp_idx, lane_idx=lane_idx,
            cta_rank=cta_rank, is_leader_cta=cta_rank == 0, y=y, stats=stats, num_tokens=num_tokens, sym=sym,
            l1_w=l1_w, l1_w_sf=l1_w_sf, l2_w=l2_w, l2_w_sf=l2_w_sf, shared_l1_w=shared_l1_w, shared_l1_w_sf=shared_l1_w_sf,
            shared_l2_w=shared_l2_w, shared_l2_w_sf=shared_l2_w_sf, trace_ptr=trace_ptr,
            sm_locality_domains=sm_locality_domains, tma=tma, smem=smem_map, mb=Barriers(smem_map), tmem=TmemMap(build.cfg),
            ws=Workspace(sym.base, build.cfg), buf=BufferMap(sym.base, build.cfg), bar=NamedBarriers(), tr=tr,
            d_test_pull_done_sync=bool(build.dispatch_only_sync),  # tests only (D): see KernelBuild.dispatch_only_sync
            launch_tag=launch_tag,  # Int64: per-launch tag of the combine-readiness handshake (set by the kernel after the PDL wait)
        )  # fmt: skip

    def with_fields(self, **kw):
        """A copy of this (frozen) context with some fields replaced (used by the kernel to attach ``launch_tag`` after the PDL wait)."""
        new = object.__new__(type(self))
        for k, v in self.__dict__.items():
            object.__setattr__(new, k, v)
        for k, v in kw.items():
            object.__setattr__(new, k, v)
        return new
