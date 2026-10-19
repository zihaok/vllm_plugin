# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / kernel skeleton stubs.
"""Stub role functions used by ``kernel.py`` while a role module does not exist yet (DESIGN.md section 4.3 / 15).

Each stub has the exact signature of the real role and performs the CTA-level rendezvous the real role must contain, nothing else:
``dispatch_role`` executes R1 and R2, ``combine_role`` executes R2.  With ``KernelBuild.debug_marks`` the stubs bump visit counters
in the trace buffer so that tests can see which roles ran.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Int32

from . import comm
from .config import (
    BAR_DISPATCH,
    GRID_SYNC_DISPATCH,
    NUM_DISPATCH_THREADS,
    NVL_TAG_AFTER_WORKSPACE_CLEAN,
    NVL_TAG_BEFORE_PULL,
)

# stub role names -> (module, function) of the real implementation
ROLE_MODULES = {
    "dispatch_role": "dispatch",
    "scheduler_role": "scheduler",
    "tma_a_role": "tma_warps",
    "tma_b_role": "tma_warps",
    "mma_role": "mma_warp",
    "epilogue_role": "epilogue",
    "combine_role": "combine",
}


@cute.jit
def dispatch_role(ctx):
    if cutlass.const_expr(ctx.build.debug_gridsync_iters > 0):
        # stress: every CTA's dispatch warps take part in N software grid syncs (needs all num_sms CTAs co-resident)
        i = Int32(0)
        while i < Int32(ctx.build.debug_gridsync_iters):
            comm.grid_sync(ctx.ws, 0, ctx.cfg.num_sms, ctx.sm_idx, ctx.thread_idx, (BAR_DISPATCH, NUM_DISPATCH_THREADS))
            i = i + 1
    if cutlass.const_expr(ctx.build.debug_nvlink_iters > 0):
        # stress: the DeepGEMM sequence grid sync A -> NVLink barrier #1 (False, True) -> NVLink barrier #2 (True, False), N times, all ranks x all CTAs
        j = Int32(0)
        while j < Int32(ctx.build.debug_nvlink_iters):
            comm.grid_sync(ctx.ws, GRID_SYNC_DISPATCH, ctx.cfg.num_sms, ctx.sm_idx, ctx.thread_idx, (BAR_DISPATCH, NUM_DISPATCH_THREADS))
            comm.nvlink_barrier(ctx.ws, ctx.sym, ctx.cfg.num_ranks, ctx.cfg.num_sms, NUM_DISPATCH_THREADS, GRID_SYNC_DISPATCH,
                                NVL_TAG_BEFORE_PULL, ctx.sm_idx, ctx.thread_idx, (BAR_DISPATCH, NUM_DISPATCH_THREADS), False, True)
            comm.nvlink_barrier(ctx.ws, ctx.sym, ctx.cfg.num_ranks, ctx.cfg.num_sms, NUM_DISPATCH_THREADS, GRID_SYNC_DISPATCH,
                                NVL_TAG_AFTER_WORKSPACE_CLEAN, ctx.sm_idx, ctx.thread_idx, (BAR_DISPATCH, NUM_DISPATCH_THREADS), True, False)
            j = j + 1
    ctx.bar.dispatch_epilogue_sync()  # R1 (K:491)
    ctx.bar.dispatch_epilogue_sync()  # R2 (K:752)


@cute.jit
def scheduler_role(ctx):
    pass


@cute.jit
def tma_a_role(ctx):
    pass


@cute.jit
def tma_b_role(ctx):
    pass


@cute.jit
def mma_role(ctx):
    pass


@cute.jit
def epilogue_role(ctx):
    pass


@cute.jit
def combine_role(ctx):
    ctx.bar.dispatch_epilogue_sync()  # R2 (K:1710)
