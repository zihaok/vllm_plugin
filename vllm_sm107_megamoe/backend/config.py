# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / csrc/jit_kernels/{heuristics/mega_moe.hpp,impls/sm107_fp8_fp4_mega_moe.hpp}.
"""Configuration dataclasses of the experimental SM107 (Rubin) DeepGEMM-style MegaMoE package.

* :class:`Sm107KernelConfig`: EVERY compile-time quantity of one compiled kernel variant (one BLOCK_M of one
  model/EP configuration).  It is a plain frozen dataclass: pure Python, no torch / CUDA / cutlass import, so the
  kernel code, the JIT layer and the tests can all consume it.  Instances are produced by
  :func:`flashinfer.experimental.sm107_deepgemm_megamoe.heuristics.get_sm107_kernel_config`, a bit-exact port of
  DeepGEMM's host heuristics (``csrc/jit_kernels/heuristics/mega_moe.hpp`` + the SM107 launcher).
* :class:`Sm107MegaMoeConfig`: the call-independent user/model configuration (E, H, I, top-k, EP size, T_max,
  shared experts, activation numerics).  ``kernel_config(num_tokens)`` derives the per-call kernel variant.

Source citations ``K:<line>`` refer to ``deep_gemm/include/deep_gemm/impls/sm107_fp8_fp4_mega_moe.cuh`` of DeepGEMM
branch ``rubin-megamoe-a7a9277-1016`` @ 3ec0eb4.
"""

from __future__ import annotations

import dataclasses
import hashlib
import math
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple

# The six BLOCK_M values the kernel can be instantiated with (K:216 BLOCK_M % 16, K:1238 WG_BLOCK_M % STORE_BLOCK_M,
# heuristics/mega_moe.hpp:185 smem_a stage % 1024).  8 and 96 pass the host candidate check only.
VALID_BLOCK_MS: Tuple[int, ...] = (16, 32, 64, 128, 192, 240)

_INF = math.inf


def _f32_hex(x: float) -> str:
    """Canonical text of a float32 value (the kernel receives clamp/alpha/beta as hex-float template literals)."""
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return struct.pack("<f", x).hex()


@dataclass(frozen=True, kw_only=True)
class Sm107KernelConfig:
    """All compile-time quantities of one SM107 MegaMoE kernel variant.

    Field names are a contract with the kernel modules: new fields may be added, existing ones are never
    renamed.  Every value is a plain int / bool / float / str / dict.  Derived values are filled by the factory; do not
    use ``dataclasses.replace`` on primary fields (``block_m``, ``num_stages`` ...): derive a new config with
    ``heuristics.get_sm107_kernel_config(..., block_m=...)`` instead (``__post_init__`` rejects stale derived values).
    """

    # ----------------------------------------------------------------------------------------------------- problem
    num_ranks: int  # kNumRanks (EP size)
    num_experts: int  # kNumExperts (total, all ranks)
    num_experts_per_rank: int  # kNumExpertsPerRank = E / R
    hidden: int  # kHidden (H)
    intermediate_hidden: int  # kIntermediateHidden (I)
    num_shared_experts: int  # kNumSharedExperts (S; 0 or 1 for DeepSeek-V4.1)
    num_topk: int  # kNumTopk
    num_sms: int  # kNumSMs (whole device, 212 on Rubin VR200); must be even
    num_max_tokens_per_rank: int  # kNumMaxTokensPerRank, ALREADY aligned to 1920

    # ----------------------------------------------------------------------------------------------------- tiling
    block_m: int  # BLOCK_M in {16, 32, 64, 128, 192, 240} (= UMMA_N, swap-AB)
    block_n: int  # BLOCK_N = 128
    block_k: int  # BLOCK_K = 128 (K per pipeline stage)
    umma_k: int  # UMMA_K = 64 (Rubin K=64 block-scaled UMMA, idesc bit 31)
    load_block_m: int  # LOAD_BLOCK_M = BLOCK_M / 2 (each CTA of the pair loads half of the token tile)
    load_block_n: int  # LOAD_BLOCK_N = BLOCK_N = 128
    store_block_m: int  # STORE_BLOCK_M: 8 (BM<=16) / 16 (<=64) / 32 (<=192) / 40 (240)
    sf_block_m: int  # SF_BLOCK_M = align(BLOCK_M, 128)
    sf_block_n: int  # SF_BLOCK_N = 128
    wg_block_m: int  # WG_BLOCK_M = BLOCK_M / 2 (tokens per epilogue warpgroup)
    umma_m: int = (
        256  # UMMA_M = 2 * LAYOUT_AD_M (2-CTA, swap-AB: weights are the A operand)
    )
    umma_n: int = 0  # UMMA_N = BLOCK_M (accumulator column stride; the instruction N is rewritten per task)
    umma_block_k: int = 128  # UMMA_BLOCK_K (two K=64 UMMAs per stage)
    atom_m: int = 8  # ATOM_M (tokens per TMEM-load atom)
    num_atoms_per_store: int = 0  # kNumAtomsPerStore = STORE_BLOCK_M / 8
    num_store_blocks_per_wg: int = 0  # kNumStoreBlocks = WG_BLOCK_M / STORE_BLOCK_M
    shared_block_k: int = (
        64  # SHARED_BLOCK_K: half-K shared-expert stages (packed FP4 mode)
    )
    l1_out_block_n: int = (
        64  # L1_OUT_BLOCK_N = BLOCK_N / 2 (post-SwiGLU channels per N block)
    )
    l1_shape_n: int = 0  # 2 * I
    l1_shape_k: int = 0  # H
    l2_shape_n: int = 0  # H
    l2_shape_k: int = 0  # I
    shared_l2_shape_k: int = 0  # I * S
    num_l1_block_ns: int = 0  # 2I / 128 (36 for I=2304); L2 readiness mask has one bit per L1 N block (<= 64)
    num_l2_block_ns: int = 0  # H / 128
    num_l1_clusters: int = 0  # L1 N / 256 (CTA-pair tasks per M block, 18 for I=2304)
    num_l2_clusters: int = 0  # L2 N / 256 (20 for H=5120)
    swizzle_a_mode: int = 128  # kSwizzleAMode (activations)
    swizzle_b_mode: int = 64  # kSwizzleBMode (packed FP4 routed weights)
    swizzle_cd_mode: int = 128  # kSwizzleCDMode
    swizzle_shared_mode: int = (
        64  # kSwizzleSharedMode (half-K shared-expert tokens/weights)
    )

    # ----------------------------------------------------------------------------------------------------- pipeline
    num_stages: (
        int  # kNumStages: SHIPPED stage count (SM107 launcher, not the SM100 one)
    )
    smem_bytes: int  # requested dynamic shared memory (launch size), IMPL:141
    accum_stages: int = 2  # kNumEpilogueStages (TMEM accumulator stages)
    tma_store_stages: int = 2  # kNumTMAStoreStages
    schedule_stages: int = 2  # kNumScheduleStages
    smem_sizeof: int = (
        0  # sizeof(SharedStorage) (end rounded up to the 1024 B alignment)
    )
    smem_end: int = 0  # exact end of the last member (tmem_ptr_in_smem + 4)
    smem_reusable_bytes: int = 0  # kNumReusableSmemBytes = offsetof(dispatch_barriers); combine overlays [0, reusable)
    # name -> byte offset of the SharedStorage members (K:250-275) incl. 'reusable_bytes', 'end', 'sizeof'
    smem_offsets: Dict[str, int] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )
    # name -> size in bytes of the SharedStorage members
    smem_sizes: Dict[str, int] = field(
        default_factory=dict, compare=False, hash=False, repr=False
    )

    # ----------------------------------------------------------------------------------------------------- threads
    num_dispatch_threads: int = 128  # warps 0-3
    num_non_epilogue_threads: int = 128  # warps 4-7 (TMA-A, TMA-B, MMA, scheduler)
    num_epilogue_threads: int = 256  # warps 8-15 (2 warpgroups)
    num_threads: int = 512
    num_dispatch_warps: int = 4
    num_mma_non_epilogue_warps: int = 4
    num_epilogue_warps: int = 8
    num_epilogue_warpgroups: int = 2
    regs_dispatch: int = (
        0  # setmaxnreg targets: 96/88/160 if E/R > 64 else 48/40/208 (K:385-388)
    )
    regs_non_epilogue: int = 0
    regs_epilogue: int = 0

    # ----------------------------------------------------------------------------------------------------- TMEM
    tmem_accum_cols: int = (
        0  # 2 * BLOCK_M (two accumulator stages, UMMA_N columns each)
    )
    tmem_sfa_cols: int = 0  # SF_BLOCK_M / 32 (token SF, "SFA" in DeepGEMM naming)
    tmem_sfb_cols: int = 0  # SF_BLOCK_N / 32 (weight SF)
    tmem_cols_used: int = 0
    tmem_cols_alloc: int = 0  # kNumTmemCols: next of {32, 64, 128, 256, 512}
    tmem_sfa_col: int = 0  # kTmemStartColOfSFA
    tmem_sfb_col: int = 0  # kTmemStartColOfSFB

    # ----------------------------------------------------------------------------------------------------- ring / buffers
    num_ring_tokens: int = (
        0  # kNumRingTokens (max over all candidate BLOCK_M, aligned to 1920)
    )
    num_sf_ring_tokens: int = 0  # kNumSFRingTokens (max over candidates)
    num_ring_blocks: int = 0  # kNumRingBlocks = num_ring_tokens / BLOCK_M
    num_shared_sf_tokens: int = 0  # kNumSharedSFTokens = ceil(T/8) * 128
    num_max_pool_tokens: int = 0  # workspace metadata pool span
    num_bytes_per_pull: int = 0  # kNumBytesPerPull (H, halved while > 8192)
    num_pull_chunks: int = 1  # kHidden / kNumBytesPerPull
    symm_buffer_bytes: int = 0  # total bytes of the symmetric buffer of this configuration (all variants share it)

    # ----------------------------------------------------------------------------------------------------- dispatch / scheduler / combine derived
    num_tokens_per_warp: int = (
        0  # kNumTokensPerWarp = 32 / topk (5 for top-6; 30 of 32 lanes active)
    )
    num_activate_lanes: int = 0  # kNumActivateLanes = kNumTokensPerWarp * topk
    num_experts_per_lane: int = 0  # ceil(E/R / 32): scheduler per-lane expert arrays (3/2/1/1 for DEP4/8/16/32 at E=384)
    num_ranks_per_lane: int = 0  # kNumRanksPerLane = ceil(R / 32)
    num_combine_chunks: int = (
        2  # kNumChunks (1 or 2) of the combine stage (K:1661-1662)
    )
    combine_chunk_bytes: int = 0  # kNumChunkBytes (5120 B for H=5120)
    combine_chunk_uint4_per_lane: int = 0  # kNumUint4PerLane (10 for H=5120)
    pull_uint4_per_lane: int = (
        0  # uint4 pull_regs[...] of the dispatch warps = kNumBytesPerPull / 512 (K:656)
    )

    # ----------------------------------------------------------------------------------------------------- numerics
    clamp: float = _INF  # kActivationClamp (inf compiles the clamp out); V4.1: 10.0
    alpha: float = 1.0  # kActivationAlpha
    beta: float = 0.0  # kActivationBeta
    fast_math: bool = True  # kFastMath

    # ----------------------------------------------------------------------------------------------------- variants / flags
    weight_packed_fp4: bool = True  # weight_dtype_t = cutlass::float_e2m1_t (packed FP4, 64B swizzle, 16U4_ALIGN8B)
    per_die: bool = False
    num_locality_domains: int = (
        1  # kNumLocalityDomains (2 only for the per-die variant)
    )
    localized_weights: bool = False  # kLocalizedWeights
    localized_shared_weights: bool = False  # kLocalizedSharedWeights
    num_l1_clusters_per_domain: int = 0  # kNumL1Clusters / kNumLocalityDomains (9 for I=2304 with 2 domains; all of them with 1)
    num_l2_clusters_per_domain: int = (
        0  # kNumL2Clusters / kNumLocalityDomains (10 for H=5120 with 2 domains)
    )
    num_min_steal_waves: int = 2  # kNumMinStealWaves: a drained domain queue steals from the other one only with >= 2 waves of tasks
    domain_counter_stride: int = 1  # kDomainCounterStride: per-domain task counters are contiguous u32 (counter[d] at +d)
    trace: bool = False  # kTrace
    debug_flags: int = 0  # kDebugFlags
    epi_prefetch_stages: int = 0  # kEpiPrefetchStages
    dispatch_pipe: int = (
        3  # kDispatchPipe (3 = register pull; modes 1/2 are not ported)
    )
    weight_prefetch: int = 0  # kWeightPrefetch
    tma_l2_promotion: Optional[int] = (
        None  # kernel/tma_warps.py: L2 promotion (bytes) of the weight TMA descriptors; None = auto (256 for block_m >= 128, 0 = the CuTe DSL atom's own L2_128B below); 0 / 64 / 128 / 256 force it
    )

    def __post_init__(self) -> None:
        # Cheap consistency checks: catch a stale derived field after dataclasses.replace() of a primary field.
        def check(cond: bool, what: str) -> None:
            if not cond:
                raise ValueError(
                    f"inconsistent Sm107KernelConfig ({what}); derive variants with "
                    "heuristics.get_sm107_kernel_config(...) instead of dataclasses.replace on primary fields"
                )

        check(
            self.block_m in VALID_BLOCK_MS,
            f"block_m={self.block_m} not in {VALID_BLOCK_MS}",
        )
        check(
            self.tma_l2_promotion in (None, 0, 64, 128, 256),
            f"tma_l2_promotion={self.tma_l2_promotion} not in (None, 0, 64, 128, 256)",
        )
        check(
            self.num_experts == self.num_experts_per_rank * self.num_ranks,
            "num_experts != E/R * R",
        )
        check(
            self.load_block_m * 2 == self.block_m
            and self.wg_block_m * 2 == self.block_m,
            "load/wg_block_m",
        )
        check(self.umma_n in (0, self.block_m), "umma_n")
        if (
            self.umma_n == 0
        ):  # allow construction without it, fill the (derived) default
            object.__setattr__(self, "umma_n", self.block_m)
        check(self.sf_block_m == -(-self.block_m // 128) * 128, "sf_block_m")
        if self.num_atoms_per_store == 0:
            object.__setattr__(
                self, "num_atoms_per_store", self.store_block_m // self.atom_m
            )
        if self.num_store_blocks_per_wg == 0:
            object.__setattr__(
                self, "num_store_blocks_per_wg", self.wg_block_m // self.store_block_m
            )
        check(
            self.wg_block_m % self.store_block_m == 0
            and self.store_block_m % self.atom_m == 0,
            "store_block_m",
        )
        check(
            self.num_threads
            == self.num_dispatch_threads
            + self.num_non_epilogue_threads
            + self.num_epilogue_threads,
            "num_threads",
        )
        check(
            self.num_ring_tokens == 0
            or self.num_ring_blocks * self.block_m == self.num_ring_tokens,
            "num_ring_blocks",
        )
        check(self.num_sms % 2 == 0, "num_sms must be even (2-CTA clusters)")
        check(
            self.num_max_tokens_per_rank % 1920 == 0,
            "num_max_tokens_per_rank must be aligned to 1920",
        )
        if self.smem_offsets:
            check(
                self.smem_offsets.get("dispatch_barriers") == self.smem_reusable_bytes,
                "smem offsets",
            )
            check(
                self.smem_sizeof <= self.smem_bytes,
                "requested smem < sizeof(SharedStorage)",
            )

    # ------------------------------------------------------------------------------------------------ compile key
    #: Fields that change the generated code (== the 35 template arguments of ``sm107_fp8_fp4_mega_moe_impl``
    #: (``IMPL:294-352``) plus the launch smem size, the register split and the TMEM allocation).  JIT-level extras
    #: that are not part of this config (target arch, ``cute.compile`` options, ``mode_mask``, NCU/trace toggles) must be
    #: appended by ``jit.py`` when it builds the kernel name (FlashInfer rule M8).
    _COMPILE_KEY_FIELDS = (
        "num_max_tokens_per_rank", "hidden", "intermediate_hidden", "num_experts", "num_shared_experts", "num_topk",
        "block_m", "block_n", "block_k", "store_block_m", "sf_block_m", "sf_block_n",
        "num_ring_tokens", "num_sf_ring_tokens", "num_stages", "num_bytes_per_pull",
        "num_dispatch_threads", "num_non_epilogue_threads", "num_epilogue_threads",
        "num_sms", "num_ranks", "clamp", "alpha", "beta", "fast_math",
        "weight_packed_fp4", "umma_k", "trace", "debug_flags", "epi_prefetch_stages", "dispatch_pipe",
        "weight_prefetch", "tma_l2_promotion", "num_locality_domains", "localized_weights", "localized_shared_weights",
        "smem_bytes", "regs_dispatch", "regs_non_epilogue", "regs_epilogue", "tmem_cols_alloc",
    )  # fmt: skip

    @property
    def kernel_name_digest_inputs(self) -> Tuple[Tuple[str, Any], ...]:
        """Tuple of ``(field_name, canonical_value)`` of everything that must key the compile cache."""
        out = []
        for name in self._COMPILE_KEY_FIELDS:
            value = getattr(self, name)
            if name in ("clamp", "alpha", "beta"):
                value = _f32_hex(
                    value
                )  # canonical float32 text (the kernel template takes float32 literals)
            out.append((name, value))
        return tuple(out)

    def cache_key(self) -> str:
        """16-hex-digit digest of :attr:`kernel_name_digest_inputs` (SHA-256 of the canonical text)."""
        text = ";".join(f"{k}={v}" for k, v in self.kernel_name_digest_inputs)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def short_name(self) -> str:
        """Short readable kernel-name prefix + digest, e.g. ``sm107dg_bm128_dep4_s1_<digest16>``."""
        return f"sm107dg_bm{self.block_m}_dep{self.num_ranks}_s{self.num_shared_experts}_{self.cache_key()}"

    # ------------------------------------------------------------------------------------------------ convenience
    @property
    def num_registers(self) -> Tuple[int, int, int]:
        return self.regs_dispatch, self.regs_non_epilogue, self.regs_epilogue

    @property
    def has_shared(self) -> bool:
        return self.num_shared_experts > 0

    def to_dict(self) -> Dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, kw_only=True)
class Sm107MegaMoeConfig:
    """Call-independent configuration of the SM107 DeepGEMM-style MegaMoE (model + EP + numerics).

    The per-call part (the BLOCK_M variant, i.e. the compiled kernel) is derived from the live token count of the
    calling rank with :meth:`kernel_config`.  No environment variables are read (FlashInfer experimental policy):
    every DeepGEMM env knob that matters is an explicit field.
    """

    num_experts: int
    num_max_tokens_per_rank: int  # user value; aligned up to 1920 internally (SymmBuffer.num_max_tokens_per_rank is aligned)
    num_topk: int
    hidden: int
    intermediate_hidden: int
    num_shared_experts: int = 0
    num_ranks: int = 1
    num_sms: Optional[int] = (
        None  # None -> read from the current CUDA device (212 on Rubin VR200)
    )
    mma_type: str = "fp8xfp4"
    activation: str = "swiglu"
    activation_clamp: Optional[float] = None  # None -> no clamp (V4.1: 10.0)
    activation_alpha: float = 1.0
    activation_beta: float = 0.0
    fast_math: bool = True
    per_die: bool = False  # per-die scheduling (DeepGEMM: DG_MEGA_MOE_SM107_PERDIE=1; needs die-local routed weights)
    block_m_override: int = 0  # DG_MEGA_MOE_SM107_BLOCK_M (0: heuristic)
    launch_smem_capacity: int = (
        232448  # DG_MEGA_MOE_SM107_SMEM (capacity used to size the stages)
    )
    stages_cap: int = 0  # DG_MEGA_MOE_SM107_STAGES (0: uncapped)
    tma_l2_promotion: Optional[int] = (
        None  # forwarded to Sm107KernelConfig.tma_l2_promotion (None = auto: 256 B weight-map L2 promotion for block_m >= 128)
    )

    def __post_init__(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        check(
            self.mma_type == "fp8xfp4",
            f"the SM107 kernel supports mma_type='fp8xfp4' only, got {self.mma_type!r}",
        )
        check(
            self.activation == "swiglu",
            f"the SM107 kernel supports activation='swiglu' only, got {self.activation!r}",
        )
        check(
            self.num_ranks >= 1 and self.num_experts % self.num_ranks == 0,
            "num_experts must be divisible by num_ranks",
        )
        check(
            self.num_topk >= 1 and self.num_shared_experts >= 0,
            "invalid num_topk / num_shared_experts",
        )
        check(
            self.hidden % 128 == 0 and self.intermediate_hidden % 128 == 0,
            "hidden and intermediate_hidden must be multiples of 128",
        )
        check(
            self.activation_clamp is None or self.activation_clamp >= 0,
            "activation_clamp must be >= 0",
        )
        check(
            math.isfinite(self.activation_alpha)
            and math.isfinite(self.activation_beta),
            "alpha/beta must be finite",
        )
        check(
            self.num_sms is None or (self.num_sms > 0 and self.num_sms % 2 == 0),
            "num_sms must be even",
        )
        check(
            self.block_m_override in (0,) + VALID_BLOCK_MS,
            f"block_m_override must be 0 or one of {VALID_BLOCK_MS}",
        )
        check(
            self.tma_l2_promotion in (None, 0, 64, 128, 256),
            "tma_l2_promotion must be None, 0, 64, 128 or 256",
        )

    @property
    def num_experts_per_rank(self) -> int:
        return self.num_experts // self.num_ranks

    @property
    def aligned_num_max_tokens_per_rank(self) -> int:
        return -(-self.num_max_tokens_per_rank // 1920) * 1920

    def kernel_config(
        self, num_tokens: int, *, num_sms: Optional[int] = None, trace: bool = False
    ) -> Sm107KernelConfig:
        """Kernel variant for a call with ``num_tokens`` live tokens on the calling rank (== ``y.size(0)``)."""
        from .heuristics import (
            get_sm107_kernel_config,
        )  # local import: config.py has no dependencies

        return get_sm107_kernel_config(
            self.num_ranks, self.num_experts, self.num_max_tokens_per_rank, num_tokens, self.num_topk,
            self.hidden, self.intermediate_hidden, self.num_shared_experts,
            num_sms=num_sms if num_sms is not None else self.num_sms,
            activation_clamp=self.activation_clamp, activation_alpha=self.activation_alpha,
            activation_beta=self.activation_beta, fast_math=self.fast_math,
            block_m=self.block_m_override, per_die=self.per_die,
            smem_capacity=self.launch_smem_capacity, max_stages=self.stages_cap, trace=trace, tma_l2_promotion=self.tma_l2_promotion,
        )  # fmt: skip


#: ``kernel_name`` of the moe_ep (Tier 2) backend, resolved through ``flashinfer.moe_ep.core.kernel.registry``
MOE_EP_KERNEL_NAME = "sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl"


@dataclass
class Sm107_Mxfp8_Mxfp4_Bf16_DeepgemmCutedsl_MegaMoeConfig:
    """User-facing config of the moe_ep (Tier 2) backend ``sm107_mxfp8_mxfp4_bf16_deepgemm_cutedsl``.

    MXFP8 E4M3 activations x packed MXFP4 weights -> BF16, DeepGEMM kernel mapping (see the package README).  Field names that
    overlap with ``Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig`` keep their meaning.  No environment variables: the DeepGEMM
    tuning knobs that matter are explicit fields.
    """

    intermediate_size: int  # width after activation; FC1 N is 2 * intermediate_size
    top_k: int
    kernel_name: str = field(default=MOE_EP_KERNEL_NAME, init=False)
    kind: Literal["mxfp4_mxfp8"] = field(default="mxfp4_mxfp8", init=False)
    gate_up_clamp: Optional[float] = None
    activation_clamp: Optional[float] = None  # deprecated alias of gate_up_clamp
    fast_math: bool = True  # kFastMath: True = __expf + rcp.approx (DeepGEMM's default), False = IEEE expf + true division; both bit-exact vs DeepGEMM at EP1
    apply_topk_in_fc1: bool = (
        True  # the top-k weight is applied in the FC1 epilogue (always, as in DeepGEMM)
    )
    activation: Literal["swiglu"] = "swiglu"
    combine_dtype: Literal["bf16"] = "bf16"
    knobs: Optional[dict] = (
        None  # accepted for API parity; DeepGEMM has no online/offline knob cache
    )
    num_shared_experts: int = 0  # fused shared experts need a weight pack with shared tensors (not in the canonical MoEWeightPack yet)
    per_die: bool = False  # per-die execution: routed weights are localized by `preprocess_weights` (die-local routed weights, the production recipe)
    block_m_override: int = 0  # DG_MEGA_MOE_SM107_BLOCK_M (0: heuristic)
    launch_smem_capacity: int = 232448  # DG_MEGA_MOE_SM107_SMEM
    stages_cap: int = 0  # DG_MEGA_MOE_SM107_STAGES
    tag_mode: Literal["epoch", "gridid"] = (
        "epoch"  # combine-readiness tag: "epoch" is exact under CUDA-graph replay, "gridid" is DeepGEMM's (stale under replay)
    )
    regs_override: Optional[Tuple[int, int, int]] = (
        None  # (dispatch, non_epilogue, epilogue) explicit setmaxnreg split (default: the spill-free 96/88/160 at every EP size)
    )
    faithful_regs: bool = False  # True: DeepGEMM's own split (48/40/208 at experts-per-rank <= 64) for parity runs; mutually exclusive with regs_override
    timeout_sec: float = 300.0  # spin-wait trap timeout baked into the kernel
    allow_unvalidated_geometry: bool = False  # True: skip the validated-geometry check of validate_init (heuristics.VALIDATED_GEOMETRIES; kernel development only)
    precompile_on_warmup: bool = True  # the first launch of a workspace (what MoEEpMegaLayer.warmup() does) builds EVERY BLOCK_M variant of the geometry (host only, ~1-1.5 s each cold); False: variants compile lazily at their first use

    def __post_init__(self) -> None:
        def check(cond: bool, msg: str) -> None:
            if not cond:
                raise ValueError(msg)

        from .heuristics import (
            SUPPORTED_FAST_MATH_VALUES,
        )  # local import: config.py has no module-level dependencies

        check(
            bool(self.fast_math) in SUPPORTED_FAST_MATH_VALUES,
            f"fast_math={self.fast_math!r} is not supported (supported: {SUPPORTED_FAST_MATH_VALUES})",
        )
        check(
            self.tag_mode in ("epoch", "gridid"),
            f"tag_mode must be 'epoch' or 'gridid', got {self.tag_mode!r}",
        )
        check(self.regs_override is None or (len(self.regs_override) == 3 and all(int(v) > 0 and int(v) % 8 == 0 for v in self.regs_override)),
              "regs_override must be three positive multiples of 8")  # fmt: skip
        check(self.timeout_sec > 0, "timeout_sec must be positive")
        if self.tag_mode == "gridid":
            import warnings

            warnings.warn(
                "tag_mode='gridid' reproduces DeepGEMM's %gridid combine tag, which is identical for every replay of one CUDA-graph node and returns stale results when the "
                "inputs change between replays (CUDA graph capture is refused under it); use it for parity runs only, the default is tag_mode='epoch'",
                stacklevel=3,
            )
        check(
            not (self.faithful_regs and self.regs_override is not None),
            "faithful_regs and regs_override are mutually exclusive",
        )
        check(
            self.intermediate_size > 0 and self.intermediate_size % 128 == 0,
            "intermediate_size must be a positive multiple of 128",
        )
        check(self.top_k > 0, "top_k must be positive")
        check(
            self.activation == "swiglu",
            f"only activation='swiglu' is supported, got {self.activation!r}",
        )
        check(
            self.combine_dtype == "bf16",
            f"only combine_dtype='bf16' is supported, got {self.combine_dtype!r}",
        )
        check(
            self.apply_topk_in_fc1,
            "apply_topk_in_fc1=False is not supported: the kernel always applies the top-k weight in the FC1 epilogue (as DeepGEMM does)",
        )
        check(
            self.knobs is None,
            "knobs are not supported (DeepGEMM has no knob cache); use the explicit fields",
        )
        check(
            self.num_shared_experts == 0,
            "fused shared experts are not supported by the moe_ep backend yet (the canonical MoEWeightPack has no shared tensors)",
        )
        check(
            self.block_m_override in (0,) + VALID_BLOCK_MS,
            f"block_m_override must be 0 or one of {VALID_BLOCK_MS}",
        )
        check(self.gate_up_clamp is None or self.activation_clamp is None or self.gate_up_clamp == self.activation_clamp,
              "gate_up_clamp and its deprecated alias activation_clamp disagree")  # fmt: skip
        clamp = self.resolved_clamp
        check(clamp is None or clamp >= 0, "the activation clamp must be >= 0")

    @property
    def resolved_clamp(self) -> Optional[float]:
        return (
            self.gate_up_clamp
            if self.gate_up_clamp is not None
            else self.activation_clamp
        )
