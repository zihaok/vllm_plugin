# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp.
"""Kernel-side constants and the :class:`KernelBuild` wrapper (CPU-importable: no cutlass / torch import).

The compile-time quantities themselves live in the package's ``Sm107KernelConfig`` (package-level ``config.py``); this module only adds what
is specific to the kernel build: ``mode_mask`` (staged testing), ``pdl``, the target arch and compiler options, plus the fixed
constants of the warp/barrier layout (DESIGN.md sections 3, 9, 11).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Optional, Tuple

# ---------------------------------------------------------------------------------------------------------------------
# mode_mask (DESIGN.md section 3.1)
# ---------------------------------------------------------------------------------------------------------------------
MODE_DISPATCH = 1  # dispatch roles do their work (warps 0-3)
MODE_GEMM = 2  # scheduler / TMA-A / TMA-B / MMA / epilogue GEMM part
MODE_COMBINE = 4  # combine / epilogue tail
MODE_ALL = MODE_DISPATCH | MODE_GEMM | MODE_COMBINE

# ---------------------------------------------------------------------------------------------------------------------
# Warp roles (K:399-1212): 16 warps = 512 threads
# ---------------------------------------------------------------------------------------------------------------------
NUM_THREADS = 512
NUM_WARPS = 16
WARP_DISPATCH_BEGIN = 0
NUM_DISPATCH_WARPS = 4
WARP_TMA_A = 4
WARP_TMA_B = 5
WARP_MMA = 6
WARP_SCHEDULER = 7
WARP_EPILOGUE_BEGIN = 8
NUM_EPILOGUE_WARPS = 8

# Named barriers (DESIGN.md section 9); ids 0..4 are reserved
BAR_DISPATCH = 0  # 128 threads (warps 0-3), aligned `bar.sync`
BAR_DISPATCH_EPILOGUE = 1  # 384 threads (warps 0-3 + 8-15), UNALIGNED `barrier.sync`
BAR_EPILOGUE = 2  # 256 threads (warps 8-15), aligned
BAR_EPILOGUE_WG0 = 3  # 128 threads each (warpgroup 0 / 1 of the epilogue), aligned; id = 3 + wg
NUM_RESERVED_BARRIERS = 5
NUM_DISPATCH_THREADS = 128
NUM_EPILOGUE_THREADS = 256
NUM_DISPATCH_EPILOGUE_THREADS = NUM_DISPATCH_THREADS + NUM_EPILOGUE_THREADS
NUM_NON_EPILOGUE_THREADS = 128  # warps 4-7 (TMA-A, TMA-B, MMA, scheduler)
# setmaxnreg pool of one SM as used by DeepGEMM (K:385-398 / heuristics): 128*r_dispatch + 128*r_non_epilogue + 256*r_epilogue == 64512 for both shipped splits
# (96/88/160 and 48/40/208); a split above this budget would deadlock inside setmaxnreg.inc, so KernelBuild rejects it.
REG_POOL_BUDGET = 64512
# PRODUCT DEFAULT split (dispatch, non-epilogue, epilogue) at EVERY EP size: spill-free. DeepGEMM itself uses 48/40/208 when experts-per-rank <= 64 (DEP>=8), where the 48-register dispatch cap
# spills the pull loop (measured: the cliff is between 48 and 56 dispatch registers; 56/40/200 behaves like 96/88/160); `faithful_regs=True` restores DeepGEMM's split for parity runs.
DEFAULT_REGS = (96, 88, 160)

# Grid-sync counters inside MegaMoESignals.grid_sync_count[]
GRID_SYNC_DISPATCH = 0
GRID_SYNC_EPILOGUE = 1
# NVLink barrier tags
NVL_TAG_BEFORE_PULL = 1
NVL_TAG_AFTER_WORKSPACE_CLEAN = 2

CLUSTER_SIZE = 2
DEFAULT_ARCH = "sm_107a"
MAX_SMEM_OPTIN = 232448  # sm_107 per-CTA dynamic smem limit (cudaDevAttrMaxSharedMemoryPerBlockOptin)
# DeepGEMM compiles with ptxas --register-usage-level=10; the DSL path honours it (DESIGN.md section 16)
DEFAULT_PTXAS_OPTIONS = "--register-usage-level=10"


@dataclass(frozen=True)
class KernelBuild:
    """One compiled kernel instance = ``Sm107KernelConfig`` + staged-test mask + launch/compile switches.

    ``cfg`` is the package's ``Sm107KernelConfig``; everything that changes the generated code must be reflected in :meth:`name`
    (FlashInfer rule M8), because the compile cache is keyed by it.
    """

    cfg: Any  # Sm107KernelConfig
    mode_mask: int = MODE_ALL
    pdl: bool = False  # launch with the programmatic-dependent-launch attribute (DeepGEMM default: off)
    arch: str = DEFAULT_ARCH
    ptxas_options: str = ""  # extra ptxas options handed to cute.compile (e.g. DEFAULT_PTXAS_OPTIONS)
    opt_level: int = 3  # cute.compile OptLevel
    debug_marks: bool = False  # tests only: every warp bumps u64 visit counters in `trace_ptr` ([0..15] entry, [16..31] exit)
    roles: str = "auto"  # "auto": real role modules when their files exist, else stubs; "stubs": force the stubs (skeleton tests)
    debug_gridsync_iters: int = 0  # stub dispatch role only: that many software grid syncs over all CTAs (co-residency / deadlock stress)
    debug_nvlink_iters: int = 0  # stub dispatch role only: that many (grid sync A + NVLink barrier #1 + NVLink barrier #2) rounds between all ranks
    tag_mode: str = "epoch"  # per-launch tag of the combine-readiness handshake (`ctx.launch_tag`): "epoch" (DEFAULT) = `(nvl_barrier_counter >> 1) + 1` latched at CTA start
    # (unique per launch, also under CUDA-graph replay), "gridid" = DeepGEMM (`%gridid + 1`; identical for every replay of one graph node: DeepGEMM and the port return STALE results
    # under graph replay with changing inputs + rank skew, measured, DESIGN.md section 16) -- kept for A/B parity runs and mapping comparisons only
    dispatch_only_sync: bool = False  # tests only: exposed as `ctx.d_test_pull_done_sync`; dispatch.py adds one grid sync (idx 1) before R2 so that a
    # dispatch-only launch (mode_mask=1) orders every SM's pulls before any CTA's workspace cleanup (the epilogue grid sync does this in the real kernel)
    timeout_sec: float = 300.0  # spin-wait timeout of every software barrier / signal wait (DeepGEMM comm::kNumTimeoutCycles = 300 s at 2 GHz); baked into the PTX, so it is part of
    # the compile-cache digest; hang tests use a short value (e.g. 10) instead of the SM107_DG_TIMEOUT_SEC environment variable
    faithful_regs: bool = False  # True: DeepGEMM's own split (`cfg.regs_*`: 96/88/160 when experts-per-rank > 64, else 48/40/208) for parity / mapping runs; default (False): DEFAULT_REGS = 96/88/160 at every EP size
    regs_override: Optional[Tuple[int, int, int]] = None  # (dispatch, non-epilogue, epilogue) setmaxnreg targets replacing cfg.regs_* (None = DeepGEMM's split:
    # 96/88/160 when experts-per-rank > 64 else 48/40/208); each a multiple of 8 in [24, 256] with 128*r0 + 128*r1 + 256*r2 <= REG_POOL_BUDGET

    def __post_init__(self) -> None:
        if not 0 <= self.mode_mask <= MODE_ALL:
            raise ValueError(f"mode_mask must be in [0, {MODE_ALL}], got {self.mode_mask}")
        cfg = self.cfg
        if cfg.num_threads != NUM_THREADS:
            raise ValueError("the SM107 kernel needs 512 threads")
        if cfg.num_sms % CLUSTER_SIZE != 0:
            raise ValueError("num_sms must be even")
        if cfg.smem_bytes > MAX_SMEM_OPTIN:
            raise ValueError(f"requested dynamic smem {cfg.smem_bytes} exceeds the sm_107 limit {MAX_SMEM_OPTIN}")
        if cfg.num_ranks > 72:
            raise ValueError("at most 72 ranks")
        if self.roles not in ("auto", "stubs"):
            raise ValueError("roles must be 'auto' or 'stubs'")
        if self.tag_mode not in ("gridid", "epoch"):
            raise ValueError("tag_mode must be 'gridid' or 'epoch'")
        if not float(self.timeout_sec) > 0:
            raise ValueError("timeout_sec must be positive")
        if self.faithful_regs and self.regs_override is not None:
            raise ValueError("faithful_regs=True and regs_override are mutually exclusive")
        if self.regs_override is not None:
            r = tuple(int(v) for v in self.regs_override)
            if len(r) != 3 or any(v % 8 != 0 or not 24 <= v <= 256 for v in r):
                raise ValueError(f"regs_override must be three multiples of 8 in [24, 256] (setmaxnreg), got {self.regs_override}")
            used = NUM_DISPATCH_THREADS * r[0] + NUM_NON_EPILOGUE_THREADS * r[1] + NUM_EPILOGUE_THREADS * r[2]
            if used > REG_POOL_BUDGET:
                raise ValueError(f"regs_override {r}: 128*{r[0]} + 128*{r[1]} + 256*{r[2]} = {used} exceeds the register pool {REG_POOL_BUDGET}")
            object.__setattr__(self, "regs_override", r)

    # -------------------------------------------------------------------------------------------------- derived constants
    @property
    def regs(self) -> Tuple[int, int, int]:
        """setmaxnreg targets (dispatch, non-epilogue, epilogue)."""
        if self.regs_override is not None:
            return self.regs_override
        if self.faithful_regs:
            return self.cfg.regs_dispatch, self.cfg.regs_non_epilogue, self.cfg.regs_epilogue
        return DEFAULT_REGS

    @property
    def has_dispatch(self) -> bool:
        return bool(self.mode_mask & MODE_DISPATCH)

    @property
    def has_gemm(self) -> bool:
        return bool(self.mode_mask & MODE_GEMM)

    @property
    def has_combine(self) -> bool:
        return bool(self.mode_mask & MODE_COMBINE)

    # -------------------------------------------------------------------------------------------------- compile key
    def digest_inputs(self) -> Tuple[Tuple[str, Any], ...]:
        extra = (
            ("mode_mask", self.mode_mask),
            ("pdl", self.pdl),
            ("arch", self.arch),
            ("ptxas_options", self.ptxas_options),
            ("opt_level", self.opt_level),
            ("debug_marks", self.debug_marks),
            ("roles", self.roles),
            ("debug_gridsync_iters", self.debug_gridsync_iters),
            ("dispatch_only_sync", self.dispatch_only_sync),
            ("tag_mode", self.tag_mode),
            ("debug_nvlink_iters", self.debug_nvlink_iters),
            ("regs", self.regs),
            ("faithful_regs", bool(self.faithful_regs)),
            ("timeout_sec", float(self.timeout_sec)),
            ("contract", CONTRACT_VERSION),
        )
        return tuple(self.cfg.kernel_name_digest_inputs) + extra

    def digest(self) -> str:
        text = ";".join(f"{k}={v}" for k, v in self.digest_inputs())
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def name(self) -> str:
        """Short readable prefix + 16-hex digest over EVERY constexpr: ``sm107dg_bm128_dep4_s1_m7_<digest16>``."""
        cfg = self.cfg
        regs = f"_r{'-'.join(str(v) for v in self.regs_override)}" if self.regs_override is not None else ""
        return f"sm107dg_bm{cfg.block_m}_dep{cfg.num_ranks}_s{cfg.num_shared_experts}_m{self.mode_mask}{regs}_{self.digest()}"


# Bumped whenever kernel/*.py changes in a way that must invalidate cached compiles but is not visible in the config
# (jit.py additionally hashes the kernel sources through ``extra_key_files``).
CONTRACT_VERSION = "1.8"
