# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp.
"""``DeepGemmSm107MegaMoeKernel``: launch geometry, prologue, role dispatch and epilogue-end of the SM107 MegaMoE port (DESIGN.md).

Reproduces ``sm107_fp8_fp4_mega_moe_impl`` (K = ``deep_gemm/include/deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh``):
grid = ``cfg.num_sms`` CTAs, block = 512 threads (16 warps), cluster (2,1,1), dynamic smem = ``cfg.smem_bytes``, 1 CTA/SM, PDL only if requested.
The role bodies live in other modules (dispatch/scheduler/tma_warps/mma_warp/epilogue/combine) and are resolved at construction time
(:func:`resolve_roles`): the real module if its file exists, else the stub of ``stubs.py``.
"""

from __future__ import annotations

import importlib
import importlib.util
from typing import Dict

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint64

from . import ptx
from . import stubs
from .config import (
    NUM_DISPATCH_WARPS,
    WARP_EPILOGUE_BEGIN,
    WARP_MMA,
    WARP_SCHEDULER,
    WARP_TMA_A,
    WARP_TMA_B,
    KernelBuild,
)
from .ctx import KernelCtx, SmemMap
from .trace import Trace, SLOT_START, SLOT_CLOCK_START

_ROLE_NAMES = ("dispatch_role", "scheduler_role", "tma_a_role", "tma_b_role", "mma_role", "epilogue_role", "combine_role")


def resolve_roles(mode: str = "auto") -> Dict[str, object]:
    """Role functions: the real module's function (DESIGN.md section 15).

    A module that is missing, fails to import or lacks the function raises: stubs never mask real errors (a missing role file must not become a silent no-op role).
    ``mode="stubs"`` forces the stubs (skeleton tests).
    """
    roles: Dict[str, object] = {}
    for name in _ROLE_NAMES:
        modname = stubs.ROLE_MODULES[name]
        fq = f"{__package__}.{modname}"
        if mode == "stubs":
            roles[name] = getattr(stubs, name)
            continue
        if importlib.util.find_spec(fq) is None:
            raise ImportError(f"role module {fq} ({name}) is missing from the package; the stub roles are used only with roles='stubs'")
        mod = importlib.import_module(fq)
        if not hasattr(mod, name):
            raise ImportError(f"{fq} exists but does not define the role function {name}() (DESIGN.md section 4.3)")
        roles[name] = getattr(mod, name)
    return roles


def shared_experts_supported(mode: str = "auto", mode_mask: int = 7) -> bool:
    """True when the GEMM role modules (scheduler / tma_warps / mma_warp / epilogue) declare ``SUPPORTS_SHARED_EXPERTS = True`` (shared experts, ``num_shared_experts > 0``).

    Until G implements the S>0 GEMM phases, :class:`jit.MegaMoeRunner` refuses to LAUNCH a build with the GEMM bit active (it would silently drop the shared expert and return wrong
    results); compiling it (SASS/resource studies) and dispatch-only / combine-only staged builds (``mode_mask`` without bit 1, staged tests) stay allowed.
    ``mode="stubs"`` always returns True (the stub roles ignore S by design; skeleton tests).
    """
    if mode == "stubs" or not (mode_mask & 2):
        return True
    ok = True
    for name in ("scheduler_role", "tma_a_role", "tma_b_role", "mma_role", "epilogue_role"):
        fq = f"{__package__}.{stubs.ROLE_MODULES[name]}"
        if importlib.util.find_spec(fq) is None:
            continue
        if not getattr(importlib.import_module(fq), "SUPPORTS_SHARED_EXPERTS", False):
            ok = False
    return ok


def resolve_prefetch(mode: str = "auto"):
    """The optional ``tma_warps.prefetch_tma_params(tma_params)`` (the DSL 4.8.0 ``prefetch_descriptor`` does not legalize for 2-SM atoms yet)."""
    fq = f"{__package__}.tma_warps"
    if mode == "stubs" or importlib.util.find_spec(fq) is None:
        return None
    return getattr(importlib.import_module(fq), "prefetch_tma_params", None)


def role_sources() -> Dict[str, str]:
    """Which module provides each role (``stubs`` or the real module) -- for logs/tests."""
    out = {}
    for name in _ROLE_NAMES:
        fq = f"{__package__}.{stubs.ROLE_MODULES[name]}"
        out[name] = fq if importlib.util.find_spec(fq) is not None else f"{__package__}.stubs"
    return out


def _arch_for_dsl(arch: str) -> str:
    """``sm_107a`` -> ``sm_107`` (the form of ``cute.arch.alloc_tmem(arch=...)``)."""
    return arch.rstrip("af") if arch[-1] in "af" else arch


@cute.jit
def _prologue(ctx, num_stages: cutlass.Constexpr, tmem_cols: cutlass.Constexpr, dsl_arch: cutlass.Constexpr):
    """K:292-344: cluster sync, warp-specialised init, full cluster sync, PDL wait (all 512 threads)."""
    cfg = ctx.cfg
    smem = ctx.smem
    warp_idx = ctx.warp_idx
    lane = ctx.lane_idx

    ptx.cluster_sync_relaxed()  # K:293 cluster_sync_with_relaxed_arrive (needed before 2-CTA TMEM allocation)

    if warp_idx == 0:
        # K:296-304: zero `expert_token_count` (align(4E,1024) bytes) with one elected lane (`st.bulk`)
        with cute.arch.elect_one():
            ptx.st_shared_zero(smem.expert_token_count, ((4 * cfg.num_experts + 1023) // 1024) * 1024)
    elif warp_idx == 1:
        # K:305-310: dispatch barriers (mode 3 does not use them, initialised for parity)
        if lane < 4:
            ctx.mb.dispatch(lane).init(1)
        ptx.fence_barrier_init()
    elif warp_idx == 2:
        # K:311-336: GEMM / TMEM / combine / schedule barriers with DeepGEMM's exact arrival counts (DESIGN.md section 7)
        with cute.arch.elect_one():
            for i in cutlass.range_constexpr(num_stages):
                ctx.mb.full(i).init(2 * 2)  # arrive at 2 CTAs, A + B
                ctx.mb.empty(i).init(1)
            for i in cutlass.range_constexpr(cfg.accum_stages):
                ctx.mb.tmem_full(i).init(1)  # arrive at all CTAs
                ctx.mb.tmem_empty(i).init(2 * cfg.num_epilogue_threads)  # arrive only at the leader CTA
            for i in cutlass.range_constexpr(cfg.num_epilogue_warps * 2):
                ctx.mb.combine(i).init(1)
            for i in cutlass.range_constexpr(cfg.schedule_stages):
                ctx.mb.task_info_full(i).init(1)
                ctx.mb.task_info_empty(i).init(2 * cfg.num_epilogue_threads)  # kNumScheduleConsumerThreads
        ptx.fence_barrier_init()
    elif warp_idx == 3:
        # K:337-340: 2-CTA tensor memory allocation (address written to smem)
        ptx.tmem_alloc(smem.tmem_ptr, tmem_cols, 2, True)  # `tcgen05.alloc.exclusive.cta_group::2` (DeepGEMM's form for all six BLOCK_M)

    ptx.cluster_arrive()  # K:341 cute::cluster_sync()
    ptx.cluster_wait()

    # K:343-346: wait for the primary kernel (PDL); executed by every thread BEFORE the role split
    cute.arch.griddepcontrol_wait()
    ctx.tr.stamp(SLOT_START, ctx.thread_idx == 0)
    ctx.tr.stamp_clock(SLOT_CLOCK_START, ctx.thread_idx == 0)


@cute.jit
def _launch_tag(ctx):
    """``ctx.launch_tag`` (Int64): the per-launch tag of the combine-readiness handshake (K:460 push, K:1713 wait; never equal to a zeroed flag).

    ``tag_mode="gridid"``: ``%gridid + 1`` exactly like DeepGEMM. NOTE (measured, DESIGN.md section 16): ``%gridid`` is unique per direct launch but IDENTICAL for
    every replay of the same CUDA-graph kernel node, so under graph replay a flag left by the previous replay equals the new tag.
    ``tag_mode="epoch"``: ``(nvl_barrier_counter >> 1) + 1``, read once by every CTA right after the PDL wait: the counter only changes at NVLink barrier #1/#2
    (+1 each, never reset), barrier #1 needs ALL CTAs to have passed grid sync A (which every CTA reaches after this read), so the value is exactly 2n for all CTAs
    of launch n and the tag increases by one per launch (unique per launch, also under graph replay). Needs the barriers to run (mode_mask with dispatch).
    """
    tag = Int64(0)
    if cutlass.const_expr(ctx.build.tag_mode == "gridid"):
        tag = ptx.gridid() + Int64(1)
    else:
        counter = ptx.ld_volatile_b32(ctx.ws.nvl_barrier_counter())
        tag = Int64(counter >> Int32(1)) + Int64(1)
    return tag


@cute.jit
def _debug_mark(ctx, base_slot: cutlass.Constexpr):
    """``KernelBuild.debug_marks``: one lane per warp bumps ``trace_ptr[base_slot + warp_idx]`` (u64 counters, zeroed by the test)."""
    with cute.arch.elect_one():
        cute.arch.atomic_add(ctx.trace_ptr + (base_slot + ctx.warp_idx), Uint64(1))
    if ctx.thread_idx == 0:
        if ctx.sm_idx == 0:
            if cutlass.const_expr(base_slot == 0):
                cute.arch.store(ctx.trace_ptr + 32, Uint64(ctx.launch_tag if ctx.launch_tag is not None else 0))  # test visibility of ctx.launch_tag


class DeepGemmSm107MegaMoeKernel:
    """Host ``@cute.jit __call__`` + ``@cute.kernel`` body; one instance per ``KernelBuild`` (= one compiled variant)."""

    def __init__(self, build: KernelBuild):
        self.build = build
        self.cfg = build.cfg
        self.roles = resolve_roles(build.roles)
        self.prefetch_fn = resolve_prefetch(build.roles)
        self.dsl_arch = _arch_for_dsl(build.arch)

    # ------------------------------------------------------------------------------------------------------ identity
    def name(self) -> str:
        """Compile-cache key: short readable prefix + 16-hex digest of every constexpr."""
        return self.build.name()

    # ------------------------------------------------------------------------------------------------------ host side
    @cute.jit
    def __call__(
        self,
        y: cute.Pointer,
        stats: cute.Pointer,
        num_tokens: Int32,
        sym,  # comm.SymBufferHost payload (base, offsets, rank)
        l1_w: cute.Pointer,
        l1_w_sf: cute.Pointer,
        l2_w: cute.Pointer,
        l2_w_sf: cute.Pointer,
        shared_l1_w: cute.Pointer,
        shared_l1_w_sf: cute.Pointer,
        shared_l2_w: cute.Pointer,
        shared_l2_w_sf: cute.Pointer,
        trace_ptr: cute.Pointer,
        sm_locality_domains: cute.Pointer,
        stream,
    ):
        cfg = self.cfg
        sym_dev = sym.make_device_object()
        tma_params = self._make_tma_params(sym, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w, shared_l1_w_sf, shared_l2_w, shared_l2_w_sf)
        self.kernel(
            y, stats, num_tokens, sym_dev, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w, shared_l1_w_sf, shared_l2_w, shared_l2_w_sf,
            trace_ptr, sm_locality_domains, tma_params,
        ).launch(
            grid=[cfg.num_sms, 1, 1],
            block=[cfg.num_threads, 1, 1],
            max_number_threads=[cfg.num_threads, 1, 1],  # `.maxntid 512` like DeepGEMM's __launch_bounds__(512, 1) (instead of `.reqntid`)
            cluster=[2, 1, 1],
            smem=cfg.smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=self.build.pdl,
            stream=stream,
        )  # fmt: skip

    def _make_tma_params(self, sym, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w, shared_l1_w_sf, shared_l2_w, shared_l2_w_sf):
        """The factory (``tma_warps.make_tma_params``) if present, else ``None`` (DESIGN.md section 4.4)."""
        mod_name = f"{__package__}.tma_warps"
        if self.build.roles == "stubs" or importlib.util.find_spec(mod_name) is None:
            return None
        mod = importlib.import_module(mod_name)
        factory = getattr(mod, "make_tma_params", None)
        if factory is None:
            return None
        return factory(
            self.cfg, sym=sym, l1_w=l1_w, l1_w_sf=l1_w_sf, l2_w=l2_w, l2_w_sf=l2_w_sf,
            shared_l1_w=shared_l1_w, shared_l1_w_sf=shared_l1_w_sf, shared_l2_w=shared_l2_w, shared_l2_w_sf=shared_l2_w_sf,
        )  # fmt: skip

    # ------------------------------------------------------------------------------------------------------ device side
    @cute.kernel
    def kernel(
        self,
        y: cute.Pointer,
        stats: cute.Pointer,
        num_tokens: Int32,
        sym,  # comm.SymBuffer (device)
        l1_w: cute.Pointer,
        l1_w_sf: cute.Pointer,
        l2_w: cute.Pointer,
        l2_w_sf: cute.Pointer,
        shared_l1_w: cute.Pointer,
        shared_l1_w_sf: cute.Pointer,
        shared_l2_w: cute.Pointer,
        shared_l2_w_sf: cute.Pointer,
        trace_ptr: cute.Pointer,
        sm_locality_domains: cute.Pointer,
        tma_params,
    ):
        build = self.build
        cfg = self.cfg
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()  # warp-uniform (make_warp_uniform)
        lane = cute.arch.lane_idx()
        sm_idx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.block_idx_in_cluster()
        base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)  # 1024-aligned dynamic smem start
        smem = SmemMap(base, cfg)
        tr = Trace(cfg, trace_ptr, sm_idx)

        ctx = KernelCtx(
            build=build, sm_idx=sm_idx, thread_idx=tidx, warp_idx=warp_idx, lane_idx=lane, cta_rank=cta_rank,
            y=y, stats=stats, num_tokens=num_tokens, sym=sym,
            l1_w=l1_w, l1_w_sf=l1_w_sf, l2_w=l2_w, l2_w_sf=l2_w_sf,
            shared_l1_w=shared_l1_w, shared_l1_w_sf=shared_l1_w_sf, shared_l2_w=shared_l2_w, shared_l2_w_sf=shared_l2_w_sf,
            trace_ptr=trace_ptr, sm_locality_domains=sm_locality_domains, tma=tma_params, smem=smem, tr=tr,
        )  # fmt: skip

        if cutlass.const_expr(tma_params is not None and self.prefetch_fn is not None):
            if warp_idx == 0:
                self.prefetch_fn(tma_params)  # K:142-161 prefetch of the tensor maps (G: tma_warps.prefetch_tma_params, optional)

        _prologue(ctx, cfg.num_stages, cfg.tmem_cols_alloc, self.dsl_arch)
        ctx = ctx.with_fields(launch_tag=_launch_tag(ctx))  # after the PDL wait; dead (DCE'd) in every role that does not use it

        if cutlass.const_expr(build.debug_marks):
            _debug_mark(ctx, 0)

        roles = self.roles
        regs_dispatch, regs_non_epilogue, regs_epilogue = build.regs

        # ---------------------------------------------------------------- role split (K:399-1812)
        if warp_idx < NUM_DISPATCH_WARPS:
            cute.arch.setmaxregister_decrease(regs_dispatch)  # K:401
            if cutlass.const_expr(build.has_dispatch):
                roles["dispatch_role"](ctx)  # contains R1 (K:491) and R2 (K:752)
            else:
                ctx.bar.dispatch_epilogue_sync()  # R1
                ctx.bar.dispatch_epilogue_sync()  # R2
        elif warp_idx == WARP_TMA_A:
            cute.arch.setmaxregister_decrease(regs_non_epilogue)  # K:829
            if cutlass.const_expr(build.has_gemm):
                roles["tma_a_role"](ctx)
        elif warp_idx == WARP_TMA_B:
            cute.arch.setmaxregister_decrease(regs_non_epilogue)
            if cutlass.const_expr(build.has_gemm):
                roles["tma_b_role"](ctx)
        elif warp_idx == WARP_MMA:
            cute.arch.setmaxregister_decrease(regs_non_epilogue)
            if cutlass.const_expr(build.has_gemm):
                if ctx.is_leader_cta:  # only the leader CTA issues the 2-CTA UMMAs
                    roles["mma_role"](ctx)
        elif warp_idx == WARP_SCHEDULER:
            cute.arch.setmaxregister_decrease(regs_non_epilogue)  # K:1205
            if cutlass.const_expr(build.has_gemm):
                if ctx.is_leader_cta:  # K:1207-1209
                    roles["scheduler_role"](ctx)
        else:
            cute.arch.setmaxregister_increase(regs_epilogue)  # K:1212
            # K:1217 `DG_TRAP_ONLY_DEVICE_ASSERT(tmem_ptr_in_smem == 0)`: epilogue TMEM addresses carry no warp bits, so the 2-CTA allocation base MUST be 0
            if ptx.ld_shared_u32(ctx.smem.tmem_ptr) != Int32(0):
                ptx.trap()
            ctx.bar.dispatch_epilogue_sync()  # R1 (K:1292): the epilogue cannot take tasks before the dispatch warps passed NVLink barrier #1
            if cutlass.const_expr(build.has_gemm):
                roles["epilogue_role"](ctx)
            # K:1645-1648: TMEM free by epilogue warp 0 (same logical warp on both CTAs), after all accumulators were consumed
            if warp_idx == WARP_EPILOGUE_BEGIN:
                ptx.tmem_dealloc(Int32(0), cfg.tmem_cols_alloc, 2, True)  # `tcgen05.dealloc.exclusive.cta_group::2 0, ncols`
            if cutlass.const_expr(build.has_combine):
                roles["combine_role"](ctx)  # contains R2 (K:1710)
            else:
                ctx.bar.dispatch_epilogue_sync()  # R2

        if cutlass.const_expr(build.debug_marks):
            _debug_mark(ctx, 16)
        tr.flush()  # K:1813-1819 (kTrace only)
