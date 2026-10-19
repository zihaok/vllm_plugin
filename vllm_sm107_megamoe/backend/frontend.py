# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / csrc/apis/mega_moe.hpp (fp8_fp4_mega_moe host checks) + deep_gemm/mega/__init__.py.
"""DeepGEMM-compatible entry point ``fp8_fp4_mega_moe`` of the SM107 MegaMoE port (Tier 1).

Same name, signature and semantics as ``deep_gemm.fp8_fp4_mega_moe`` for the supported subset (SwiGLU, MXFP8 x MXFP4 packed
FP4 routed weights, optional fused FP8 shared experts, ``recipe == (1, 1, 32)``).  The host asserts are those of
``csrc/apis/mega_moe.hpp:293-416``.  The per-call kernel variant (BLOCK_M, one of six compiled kernels) is selected from THIS
call's ``y.size(0)`` (``get_block_m_for_mega_moe``), exactly as DeepGEMM selects it from ``num_tokens = y.size(0)``.

The compiled kernel comes from ``jit.get_or_compile_kernel(cfg)`` (owned by the kernel-foundation role).  Its host-level
calling convention is::

    runner(y, stats, num_tokens, sym_buffer,
           l1_weights, l1_weights_sf, l2_weights, l2_weights_sf,
           shared_l1_weights, shared_l1_weights_sf, shared_l2_weights, shared_l2_weights_sf)

with torch tensors (``None`` for absent optional arguments), ``stats`` the ``cumulative_local_expert_recv_stats`` tensor or ``None``,
``num_tokens = y.size(0)`` and ``sym_buffer`` the :class:`~.workspace.SymmBuffer` (``base_ptr``, ``peer_offsets``,
``group.rank()``, ``layout``).  ``set_kernel_factory`` replaces the factory (tests, experiments).

Per-die (locality-domain) execution: when the routed weights are die-local (4D ``(2, E_l, N/2, K/2)``, made by
``locality.localize_mega_moe_weights``) the call selects the per-die kernel variant (two task queues, 3D weight tensor maps) and passes the
cached SM -> domain map (``sm_locality_domains=`` keyword of the runner).  DeepGEMM switches this with ``DG_MEGA_MOE_SM107_PERDIE``; the port
reads no environment variable: the mode is ``None`` (auto: per-die iff the routed weights are localized, the production recipe), or forced with :func:`set_per_die_mode` (``True`` on unlocalized weights is DeepGEMM's "unlocalized per-die scheduling").

Nothing here synchronises the device or compiles in the call path when the variant is already built, so the function is CUDA-graph
capturable once the variants have been precompiled (:func:`precompile_mega_moe_variants`) and, for per-die, the SM map was probed
(``locality.get_balanced_sm_locality_domains()`` outside capture).
"""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import math
import weakref
from typing import Callable, Dict, Iterator, NamedTuple, Optional, Set, Tuple

import torch

from .config import VALID_BLOCK_MS, Sm107KernelConfig
from .heuristics import (
    MAX_PORT_RANKS,
    MAX_PORT_SHARED_EXPERTS,
    SM100_SMEM_CAPACITY,
    SUPPORTED_FAST_MATH_VALUES,
    align,
    build_symm_buffer_layout,
    get_block_m_variants,
    get_sm107_kernel_config,
)
from .heuristics import get_block_m_for_mega_moe as _heuristic_block_m_for_mega_moe
from .weights import _validate_activation

# C++ guard around the host-side tensor bookkeeping of a call: tensor method / property calls (``size``, ``stride``, ``data_ptr``, ...) then skip the Python-level
# ``TorchFunctionMode`` stack.  ``torch.set_default_device("cuda")`` (DeepGEMM's ``init_dist``, vLLM) installs one, which routes EVERY such call through Python (~1.3 us
# instead of ~0.15 us): the ~100 calls of a validated launch cost ~50 us (S=0) / ~85 us (S=1) with it, ~12 / ~19 us without.  Nothing inside the guard creates a tensor.
try:
    _no_torch_function = torch._C.DisableTorchFunction
except (
    AttributeError
):  # pragma: no cover - a torch without the C++ guard: nothing to skip
    _no_torch_function = contextlib.nullcontext


def _host_assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


@dataclasses.dataclass(frozen=True)
class KernelOptions:
    """Build options of the kernel variants (explicit arguments instead of DeepGEMM's ``DG_MEGA_MOE_SM107_*`` environment variables).

    ``tag_mode``: ``"epoch"`` (default) tags the combine-readiness handshake with a per-launch counter and is exact under CUDA-graph replay;
    ``"gridid"`` is DeepGEMM's ``%gridid + 1`` (identical for every replay of one graph node: stale results under graph replay with changing inputs;
    keep it for A/B parity runs).  Register split (``setmaxnreg``): the product default is the spill-free 96/88/160 at EVERY EP size; ``faithful_regs=True``
    restores DeepGEMM's own split (96/88/160 when experts-per-rank > 64, else 48/40/208: its 48-register dispatch cap spills at DEP >= 8) for parity runs;
    ``regs_override``: explicit ``(dispatch, non_epilogue, epilogue)`` split (mutually exclusive with ``faithful_regs``).  ``timeout_sec``: spin-wait trap timeout baked into the
    kernel.  ``block_m``: force one BLOCK_M variant (``DG_MEGA_MOE_SM107_BLOCK_M``; 0 = the heuristic's choice per call).  ``smem_capacity`` /
    ``max_stages``: ``DG_MEGA_MOE_SM107_SMEM`` / ``_STAGES``."""

    tag_mode: str = "epoch"
    regs_override: Optional[Tuple[int, int, int]] = None
    timeout_sec: float = 300.0
    faithful_regs: bool = False
    block_m: int = 0
    smem_capacity: int = SM100_SMEM_CAPACITY
    max_stages: int = 0

    def __post_init__(self) -> None:
        _host_assert(
            self.tag_mode in ("epoch", "gridid"),
            f"tag_mode must be 'epoch' or 'gridid', got {self.tag_mode!r}",
        )
        if self.regs_override is not None:
            regs = tuple(int(v) for v in self.regs_override)
            _host_assert(
                len(regs) == 3 and all(v > 0 and v % 8 == 0 for v in regs),
                f"regs_override must be three positive multiples of 8, got {self.regs_override!r}",
            )
            object.__setattr__(self, "regs_override", regs)
        _host_assert(float(self.timeout_sec) > 0, "timeout_sec must be positive")
        _host_assert(
            not (self.faithful_regs and self.regs_override is not None),
            "faithful_regs and regs_override are mutually exclusive",
        )
        _host_assert(
            self.block_m == 0 or self.block_m in VALID_BLOCK_MS,
            f"block_m must be 0 or one of {VALID_BLOCK_MS}",
        )
        _host_assert(
            self.smem_capacity > 0 and self.max_stages >= 0,
            "invalid smem_capacity / max_stages",
        )


_DEFAULT_KERNEL_OPTIONS = (
    KernelOptions()
)  # immutable (frozen): safe as an argument default
_interned_options: Dict[KernelOptions, KernelOptions] = {}


def _intern_options(options: KernelOptions) -> KernelOptions:
    """One object per distinct option set: the validated-call records of the launch path are keyed on the identity of the options object, so equal options built by
    different callers (e.g. a fresh ``KernelOptions(...)`` around every launch) must share it."""
    got = _interned_options.get(options)
    if got is None:
        if len(_interned_options) >= 64:
            _interned_options.clear()
        got = _interned_options[options] = options
    return got


_kernel_factory: Optional[Callable[[Sm107KernelConfig], Callable]] = None
_per_die_mode: Optional[bool] = None
_kernel_options = _DEFAULT_KERNEL_OPTIONS


def set_kernel_options(
    options: Optional[KernelOptions] = None, **kwargs
) -> KernelOptions:
    """Set the process-wide kernel build options (:class:`KernelOptions`; fields as keywords, e.g. ``set_kernel_options(regs_override=(56, 40, 200))``;
    no argument restores the defaults).  Variants are keyed by their options, so changing them builds new variants; call it before the first launch / precompile."""
    global _kernel_options
    _kernel_options = _intern_options(
        options if options is not None else KernelOptions(**kwargs)
    )
    return _kernel_options


def get_kernel_options() -> KernelOptions:
    return _kernel_options


def get_block_m_for_mega_moe(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
    mma_type: str = "fp8xfp4",
) -> int:
    """BLOCK_M of a launch with ``num_tokens`` tokens on the calling rank (``deep_gemm.get_block_m_for_mega_moe``).  A forced BLOCK_M (``KernelOptions.block_m``,
    DeepGEMM's ``DG_MEGA_MOE_SM107_BLOCK_M``) wins, as in DeepGEMM where this helper and the launch share one override: the shared-expert scale rows that a caller
    stages (``shared_l1_acts_sf``) are laid out per BLOCK_M, so the helper must name the BLOCK_M the kernel will run."""
    bm = _heuristic_block_m_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, mma_type
    )  # validates the arguments as before
    return _kernel_options.block_m or bm


@contextlib.contextmanager
def kernel_options_scope(options: KernelOptions) -> Iterator[None]:
    """Temporarily switch the kernel build options (the moe_ep backend wraps its launches in the options of its config)."""
    global _kernel_options
    saved, _kernel_options = _kernel_options, _intern_options(options)
    try:
        yield
    finally:
        _kernel_options = saved


# variant group (the runner key without block_m) -> BLOCK_Ms built in this process; read by the backend's validate_capture_ready
_built_variants: Dict[tuple, Set[int]] = {}


def set_per_die_mode(mode: Optional[bool]) -> None:
    """``None``: per-die execution iff the routed weights are die-local (default); ``True`` / ``False`` force the per-die scheduling on / off
    (``DG_MEGA_MOE_SM107_PERDIE`` of DeepGEMM; localized weights with ``False`` run the plain scheduler on 3D weight maps)."""
    global _per_die_mode
    _per_die_mode = None if mode is None else bool(mode)


def get_per_die_mode() -> Optional[bool]:
    return _per_die_mode


def set_kernel_factory(
    factory: Optional[Callable[[Sm107KernelConfig], Callable]],
) -> None:
    """Install ``factory(cfg) -> runner`` instead of ``jit.get_or_compile_kernel`` (``None`` restores the default)."""
    global _kernel_factory
    _kernel_factory = factory
    _cached_runner.cache_clear()  # runners are memoised per variant key: forget the ones built by the previous factory
    _built_variants.clear()
    _call_records.clear()  # ... and so are the bound launchers of validated calls


def _get_kernel(
    cfg: Sm107KernelConfig, opts: KernelOptions = _DEFAULT_KERNEL_OPTIONS
) -> Callable:
    if _kernel_factory is not None:
        return _kernel_factory(cfg)
    try:
        from .jit import get_or_compile_kernel  # owned by the kernel-foundation role
    except ImportError as exc:  # the JIT layer does not exist (yet)
        raise NotImplementedError(
            "the CuteDSL kernel JIT layer (sm107_deepgemm_megamoe.jit.get_or_compile_kernel) is not available"
        ) from exc
    return get_or_compile_kernel(
        cfg,
        tag_mode=opts.tag_mode,
        regs_override=opts.regs_override,
        timeout_sec=opts.timeout_sec,
        faithful_regs=opts.faithful_regs,
    )


@functools.lru_cache(maxsize=256)
def _cached_kernel_config(
    num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, block_m: int, num_topk: int, hidden: int,
    intermediate_hidden: int, num_shared_experts: int, num_sms: int, clamp: float, alpha: float, beta: float, fast_math: bool,
    per_die: bool = False, localized_weights: bool = False, localized_shared_weights: bool = False,
    smem_capacity: int = SM100_SMEM_CAPACITY, max_stages: int = 0,
) -> Sm107KernelConfig:  # fmt: skip
    # `block_m` was already selected from this call's token count: force it so the config does not depend on num_tokens
    device_domains = 2
    if (
        per_die or localized_weights
    ):  # the per-die kernel needs a device that reports exactly 2 locality domains
        from . import locality
        from .heuristics import LOCALITY_ALLOC_GRANULARITY

        device_domains = locality.get_num_locality_domains()
        # the 3D weight tensor maps bake in the padded slice stride (heuristics.get_perdie_domain_stride_bytes)
        _host_assert(
            locality.get_granularity() == LOCALITY_ALLOC_GRANULARITY,
            f"the kernel variants assume a locality allocation granularity of {LOCALITY_ALLOC_GRANULARITY} bytes, the driver reports {locality.get_granularity()}",
        )  # fmt: skip
    return get_sm107_kernel_config(
        num_ranks, num_experts, num_max_tokens_per_rank, 0, num_topk, hidden, intermediate_hidden, num_shared_experts,
        num_sms=num_sms, activation_clamp=None if math.isinf(clamp) else clamp, activation_alpha=alpha, activation_beta=beta,
        fast_math=fast_math, block_m=block_m, per_die=per_die, localized_weights=localized_weights,
        localized_shared_weights=localized_shared_weights, device_num_locality_domains=device_domains,
        smem_capacity=smem_capacity, max_stages=max_stages,
    )  # fmt: skip


def require_sm107() -> None:
    """Arch gate: the kernel targets compute capability 10.7 (Rubin, ``sm_107a``) exactly.  A CPU-only host passes (objects can
    be built for tests; no kernel can run there)."""
    if not torch.cuda.is_available():
        return
    cc = torch.cuda.get_device_capability()
    if tuple(cc) != (10, 7):
        raise RuntimeError(
            f"the SM107 DeepGEMM-style MegaMoE requires a Rubin GPU (compute capability 10.7); this device is sm_{cc[0]}{cc[1]}"
        )


@functools.lru_cache(maxsize=64)
def _required_buffer_bytes(
    num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int,
    num_shared_experts: int, num_sms: int,
) -> int:  # fmt: skip
    return build_symm_buffer_layout(
        num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, "fp8xfp4", "swiglu",
        num_shared_experts, num_sms,
    ).num_bytes  # fmt: skip


@functools.lru_cache(maxsize=4096)
def _block_m_for_call(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    num_topk: int,
) -> int:
    return _heuristic_block_m_for_mega_moe(
        num_ranks, num_experts, num_max_tokens_per_rank, num_tokens, num_topk, "fp8xfp4"
    )  # the heuristic only: callers apply opts.block_m


@functools.lru_cache(maxsize=256)
def _cached_runner(
    num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, block_m: int, num_topk: int, hidden: int,
    intermediate_hidden: int, num_shared_experts: int, num_sms: int, clamp: float, alpha: float, beta: float, fast_math: bool,
    per_die: bool = False, localized_weights: bool = False, localized_shared_weights: bool = False, opts: KernelOptions = _DEFAULT_KERNEL_OPTIONS,
) -> Callable:  # fmt: skip
    """Runner of one kernel variant, memoised on the primitive variant key (the JIT layer's own memo costs a config digest per call).
    Exceptions (e.g. a compile attempted inside a CUDA graph capture, an option the kernel does not implement) are not cached."""
    _check_supported_variant(fast_math, num_shared_experts, localized_shared_weights)
    cfg = _cached_kernel_config(
        num_ranks, num_experts, num_max_tokens_per_rank, block_m, num_topk, hidden, intermediate_hidden, num_shared_experts,
        num_sms, clamp, alpha, beta, fast_math, per_die, localized_weights, localized_shared_weights, opts.smem_capacity, opts.max_stages,
    )  # fmt: skip
    runner = _get_kernel(cfg, opts)
    _built_variants.setdefault((num_ranks, num_experts, num_max_tokens_per_rank, num_topk, hidden, intermediate_hidden, num_shared_experts, num_sms,
                                clamp, alpha, beta, fast_math, per_die, localized_weights, localized_shared_weights, opts), set()).add(block_m)  # fmt: skip
    return runner


def built_block_ms(
    num_ranks: int, num_experts: int, num_max_tokens_per_rank: int, num_topk: int, hidden: int, intermediate_hidden: int,
    num_shared_experts: int = 0, *, activation_clamp: Optional[float] = None, activation_alpha: float = 1.0, activation_beta: float = 0.0,
    fast_math: bool = True, num_sms: Optional[int] = None, per_die: bool = False, localized_weights: bool = False,
    localized_shared_weights: bool = False, kernel_options: Optional[KernelOptions] = None,
) -> frozenset:  # fmt: skip
    """BLOCK_M variants of this geometry already built (compiled / loaded) in this process: what a CUDA graph capture can launch without compiling."""
    from .heuristics import resolve_num_sms

    key = (num_ranks, num_experts, align(num_max_tokens_per_rank, 1920), num_topk, hidden, intermediate_hidden, num_shared_experts,
           resolve_num_sms(num_sms), math.inf if activation_clamp is None else float(activation_clamp), float(activation_alpha),
           float(activation_beta), bool(fast_math), bool(per_die), bool(localized_weights), bool(localized_shared_weights),
           kernel_options if kernel_options is not None else _kernel_options)  # fmt: skip
    return frozenset(_built_variants.get(key, ()))


def _sm_locality_domains(num_sms: int) -> torch.Tensor:
    """The cached SM -> locality-domain map (``uint8[num_sms]``, indexed by ``%smid``) for the per-die kernel argument."""
    from . import locality

    if (
        not locality.has_balanced_sm_locality_domains()
        and torch.cuda.is_current_stream_capturing()
    ):
        raise RuntimeError(
            "per-die MegaMoE needs the SM -> locality-domain map: call locality.get_balanced_sm_locality_domains() before CUDA graph capture"
        )
    domains = locality.get_balanced_sm_locality_domains()
    _host_assert(
        domains.numel() == num_sms,
        f"SM locality map has {domains.numel()} entries, device has {num_sms} SMs",
    )
    return domains


# ----------------------------------------------------------------------------------------------------- host checks
def _check_sf_layout(
    sf: torch.Tensor, mn: int, k: int, num_groups: Optional[int], name: str
) -> None:
    """``check_sf_layout(sf, mn, k, 1, 32, num_groups, tma_stride_check=true, sm90_sfb_check=false, type=Int)`` (layout.hpp:101-148)."""
    _host_assert(
        sf.dtype == torch.int32,
        f"{name}: scale factors must be int32 (packed UE8M0), got {sf.dtype}",
    )
    _host_assert(
        sf.dim() == (3 if num_groups is not None else 2),
        f"{name}: expected {3 if num_groups is not None else 2} dims, got {sf.dim()}",
    )
    if num_groups is not None:
        _host_assert(
            sf.size(-3) == num_groups,
            f"{name}: expected {num_groups} groups, got {sf.size(-3)}",
        )
    _host_assert(
        sf.size(-2) == mn, f"{name}: expected size(-2) == {mn}, got {sf.size(-2)}"
    )
    _host_assert(
        sf.size(-1) == -(-k // 128),
        f"{name}: expected size(-1) == ceil({k}/128), got {sf.size(-1)}",
    )
    if num_groups is not None:
        _host_assert(
            sf.stride(-3) == sf.stride(-1) * sf.size(-1),
            f"{name}: bad group stride {sf.stride()}",
        )
    _host_assert(
        sf.stride(-2) == 1 or mn == 1,
        f"{name}: scale factors must be MN-major (stride(-2) == 1), got strides {sf.stride()}",
    )
    _host_assert(
        sf.stride(-1) >= align(mn, 4),
        f"{name}: stride(-1) must be >= align(mn, 4) = {align(mn, 4)} (TMA)",
    )
    _host_assert(
        sf.stride(-1) % 4 == 0,
        f"{name}: stride(-1) must be a multiple of 4 (TMA), got {sf.stride(-1)}",
    )
    if (
        sf.stride(-1) != mn
    ):  # port limitation: DeepGEMM accepts a padded outer stride, the kernel's tensor map bakes in the dense one
        raise NotImplementedError(
            f"{name}: the port's scale-factor tensor maps assume the dense TMA layout stride(-1) == mn == {mn}, got stride(-1) == {sf.stride(-1)} (padded); "
            "produce the scales with deep_gemm.transform_sf_into_required_layout / transform_weights_for_mega_moe for a multiple-of-4 mn"
        )


def _check_supported_variant(
    fast_math: bool, num_shared_experts: int, localized_shared_weights: bool
) -> None:
    """Options that DeepGEMM accepts and the port's kernel does not implement (each would silently compute wrong rows or fail at trace time): refused with a clear error."""
    if bool(fast_math) not in SUPPORTED_FAST_MATH_VALUES:
        raise NotImplementedError(
            f"fast_math={bool(fast_math)} is not ported (supported: {SUPPORTED_FAST_MATH_VALUES})"
        )
    if num_shared_experts > MAX_PORT_SHARED_EXPERTS:
        raise NotImplementedError(
            f"num_shared_experts={num_shared_experts} is not supported: the port runs at most one merged shared expert (num_shared_experts in {{0, {MAX_PORT_SHARED_EXPERTS}}})"
        )
    if localized_shared_weights:
        raise NotImplementedError(
            "die-local shared-expert weights (vLLM VLLM_DSV41_MEGAMOE_PERDIE=1; DeepGEMM accepts them) are not implemented: the kernel's weight tensor maps only support "
            "die-local ROUTED weights; localize the routed pair only and keep the shared pair plain"
        )


def _check_k_major_contiguous(w: torch.Tensor, name: str) -> None:
    _host_assert(w.is_contiguous(), f"{name}: weights must be contiguous (K-major)")


class MegaMoeCall(NamedTuple):
    """Validated geometry of one ``fp8_fp4_mega_moe`` call."""

    num_experts_per_rank: int
    hidden: int
    intermediate_hidden: int
    num_shared_experts: int
    clamp: float
    localized_weights: bool
    localized_shared_weights: bool


def check_mega_moe_arguments(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]],
    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]],
    cumulative_local_expert_recv_stats: Optional[torch.Tensor],
    recipe: Tuple[int, int, int],
    activation: str,
    activation_clamp: Optional[float],
    activation_alpha: float,
    activation_beta: float,
    fast_math: bool = True,
):
    """Host asserts of ``csrc/apis/mega_moe.hpp:293-416`` for the SM107 FP8 x packed-FP4 path (incl. die-local routed weights), plus the options the port does not
    implement (``NotImplementedError``: more than one shared expert, die-local shared weights, padded scale-factor strides).

    Returns a :class:`MegaMoeCall`."""
    num_tokens = y.size(0)
    rm, rn, rk = recipe
    _host_assert(
        (rm, rn, rk) == (1, 1, 32), f"recipe must be (1, 1, 32), got {tuple(recipe)}"
    )
    _validate_activation(activation)
    if activation != "swiglu":
        raise NotImplementedError(
            f"the SM107 MegaMoE port supports activation='swiglu' only, got {activation!r}"
        )
    _check_supported_variant(fast_math, 0, False)
    _host_assert(
        (shared_l1_weights is None) == (shared_l2_weights is None),
        "shared_l1_weights and shared_l2_weights must be given together",
    )

    # Activation checks
    clamp = math.inf if activation_clamp is None else float(activation_clamp)
    _host_assert(clamp >= 0, "activation_clamp must be >= 0")
    _host_assert(math.isfinite(activation_alpha), "activation_alpha must be finite")
    _host_assert(math.isfinite(activation_beta), "activation_beta must be finite")

    # Routed weights: packed FP4 (int8, two values per byte): contiguous K-major 3D, or die-local 4D (2, E_l, N/2, K/2) (per-die variant)
    (l1_w, l1_sf), (l2_w, l2_sf) = l1_weights, l2_weights
    localized_weights = l1_w.dim() == 4
    _host_assert(
        (l2_w.dim() == 4) == localized_weights,
        "l1_weights and l2_weights must both be (or both not be) die-local",
    )
    for name, w in (("l1_weights", l1_w), ("l2_weights", l2_w)):
        _host_assert(
            w.dtype == torch.int8,
            f"{name} must be packed FP4 stored as int8, got {w.dtype}",
        )
        if localized_weights:
            from . import locality

            _host_assert(
                w.dim() == 4 and w.size(0) == 2 and locality.is_canonical_localized(w),
                f"{name}: 4D weights must be die-local (2, E_l, N/2, K/2) tensors made by locality.localize",
            )
        else:
            _host_assert(
                w.dim() == 3,
                f"{name} must be 3D (E_l, N, K/2) (or die-local 4D), got {w.dim()}D",
            )
            _check_k_major_contiguous(w, name)

    def routed_shape(w: torch.Tensor) -> Tuple[int, int, int]:  # logical (E_l, N, K)
        return (
            (w.size(1), w.size(0) * w.size(2), w.size(3) * 2)
            if localized_weights
            else (w.size(0), w.size(1), w.size(2) * 2)
        )

    num_experts_per_rank, intermediate_hidden_2, hidden = routed_shape(l1_w)
    num_experts_per_rank_, hidden_, intermediate_hidden = routed_shape(l2_w)
    _host_assert(
        num_tokens <= sym_buffer.num_max_tokens_per_rank,
        f"num_tokens={num_tokens} exceeds num_max_tokens_per_rank={sym_buffer.num_max_tokens_per_rank}",
    )
    _host_assert(
        num_experts_per_rank == num_experts_per_rank_, "L1 / L2 expert counts differ"
    )
    _host_assert(hidden == hidden_, "L1 K != L2 N (hidden)")
    _host_assert(
        intermediate_hidden_2 == 2 * intermediate_hidden, "L1 N must be 2 * L2 K"
    )

    # Weight SF layout: UE8M0 packed int32, MN-major, TMA aligned
    _check_sf_layout(
        l1_sf, intermediate_hidden * 2, hidden, num_experts_per_rank, "l1_weights_sf"
    )
    _check_sf_layout(
        l2_sf, hidden, intermediate_hidden, num_experts_per_rank, "l2_weights_sf"
    )

    # Shared experts (FP8 E4M3), plain 2D (N, K): die-local 3D shared weights (DeepGEMM accepts them) are refused (the kernel's TMA-B only implements die-local routed weights)
    num_shared_experts = 0
    if shared_l1_weights is not None:
        (s1_w, s1_sf), (s2_w, s2_sf) = shared_l1_weights, shared_l2_weights
        _check_supported_variant(True, 0, s1_w.dim() == 3 or s2_w.dim() == 3)
        for name, w in (("shared_l1_weights", s1_w), ("shared_l2_weights", s2_w)):
            _host_assert(
                w.dtype == torch.float8_e4m3fn,
                f"{name} must be float8_e4m3fn, got {w.dtype}",
            )
            _host_assert(w.dim() == 2, f"{name} must be 2D (N, K), got {w.dim()}D")
            _check_k_major_contiguous(w, name)
        shared_l1_n, shared_l1_k = tuple(s1_w.shape)
        shared_l2_n, shared_l2_k = tuple(s2_w.shape)
        shared_intermediate_hidden = shared_l2_k
        _host_assert(
            shared_intermediate_hidden % intermediate_hidden == 0,
            "shared intermediate hidden must be a multiple of the routed one",
        )
        num_shared_experts = shared_intermediate_hidden // intermediate_hidden
        _check_supported_variant(True, num_shared_experts, False)
        _host_assert(
            shared_l1_n == shared_intermediate_hidden * 2,
            "shared L1 N must be 2 * shared intermediate hidden",
        )
        _host_assert(shared_l1_k == hidden, "shared L1 K must equal hidden")
        _host_assert(shared_l2_n == hidden, "shared L2 N must equal hidden")
        _check_sf_layout(
            s1_sf, shared_intermediate_hidden * 2, hidden, None, "shared_l1_weights_sf"
        )
        _check_sf_layout(
            s2_sf, hidden, shared_intermediate_hidden, None, "shared_l2_weights_sf"
        )

    # Stats counter
    if cumulative_local_expert_recv_stats is not None:
        s = cumulative_local_expert_recv_stats
        _host_assert(
            s.dtype == torch.int32, "cumulative_local_expert_recv_stats must be int32"
        )
        _host_assert(
            s.numel() == num_experts_per_rank,
            f"cumulative_local_expert_recv_stats must have {num_experts_per_rank} elements",
        )
        _host_assert(
            s.is_contiguous(), "cumulative_local_expert_recv_stats must be contiguous"
        )

    # Output (not asserted by DeepGEMM, required by the kernel: contiguous BF16 (num_tokens, H), 16 B aligned for the 1D bulk store)
    _host_assert(
        y.dtype == torch.bfloat16 and y.dim() == 2 and y.size(1) == hidden,
        f"y must be bfloat16 (num_tokens, {hidden}), got {tuple(y.shape)} {y.dtype}",
    )
    _host_assert(
        y.is_contiguous() and y.data_ptr() % 16 == 0,
        "y must be contiguous and 16 B aligned",
    )

    # Buffer bytes (evaluated with the call-time number of shared experts) and expert count
    num_ranks = sym_buffer.group.size()
    _host_assert(
        num_ranks <= MAX_PORT_RANKS,
        f"the port serves at most {MAX_PORT_RANKS} ranks (the NVLink peer table is passed to the kernel by value), got {num_ranks}",
    )
    num_experts = num_experts_per_rank * num_ranks
    _host_assert(
        num_experts == sym_buffer.num_experts,
        f"experts mismatch: weights imply {num_experts}, symmetric buffer was created for {sym_buffer.num_experts}",
    )
    _host_assert(
        sym_buffer.hidden == hidden
        and sym_buffer.intermediate_hidden == intermediate_hidden,
        "symmetric buffer / weights shape mismatch",
    )
    need = _required_buffer_bytes(
        num_ranks, num_experts, sym_buffer.num_max_tokens_per_rank, sym_buffer.num_topk, hidden, intermediate_hidden,
        num_shared_experts, sym_buffer.num_sms,
    )  # fmt: skip
    _host_assert(
        sym_buffer.buffer.nbytes >= need,
        f"the symmetric buffer has {sym_buffer.buffer.nbytes} bytes but this call needs {need}",
    )
    return MegaMoeCall(
        num_experts_per_rank,
        hidden,
        intermediate_hidden,
        num_shared_experts,
        clamp,
        localized_weights,
        False,
    )


# ----------------------------------------------------------------------------------------------------- entry points
def fp8_fp4_mega_moe(
    y: torch.Tensor,
    l1_weights: Tuple[torch.Tensor, torch.Tensor],
    l2_weights: Tuple[torch.Tensor, torch.Tensor],
    sym_buffer,
    shared_l1_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    shared_l2_weights: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    cumulative_local_expert_recv_stats: Optional[torch.Tensor] = None,
    recipe: Tuple[int, int, int] = (1, 1, 32),
    activation: str = "swiglu",
    activation_clamp: Optional[float] = None,
    fast_math: bool = True,
    activation_alpha: float = 1.0,
    activation_beta: float = 0.0,
):
    """Fused MegaMoE (dispatch + L1 + SwiGLU + L2 + combine) for one rank; DeepGEMM-compatible (``deep_gemm.fp8_fp4_mega_moe``)."""
    with _no_torch_function():
        return _fp8_fp4_mega_moe(
            y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
            recipe, activation, activation_clamp, fast_math, activation_alpha, activation_beta,
        )  # fmt: skip


# ------------------------------------------------------------------------------------------------- validated-call records
# A launch validates ~100 properties of ~10 tensors and builds the variant key; none of that changes between two calls with the same operands.  The first call with a
# given set of operands validates (``check_mega_moe_arguments``) and stores a record: the validated geometry, the symmetric buffer it was validated against (weak
# references: a destroyed or re-created buffer never matches) and one bound launcher per BLOCK_M (``MegaMoeRunner.bind``: every pointer / the peer table
# precomputed).  Later calls compare the operands' (data_ptr, shape, stride, dtype) with the record (a changed pointer, shape, layout or dtype is a miss and is validated
# again), check the per-call output ``y`` (the only operand that normally changes), and launch through the bound launcher.  Records hold no tensor, only integers and weak references.
_call_records: Dict[tuple, "_CallRecord"] = {}
_CALL_RECORDS_MAX = 256


class _CallRecord:
    __slots__ = (
        "sig",
        "buf_ref",
        "buf_tensor_ref",
        "opts",
        "per_die_mode",
        "hidden",
        "max_tokens",
        "num_ranks",
        "num_experts",
        "num_topk",
        "launchers",
        "domains",
    )


def _tensor_sig(t):
    return None if t is None else (t.data_ptr(), t.shape, t.stride(), t.dtype)


def _fp8_fp4_mega_moe(
    y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
    recipe, activation, activation_clamp, fast_math, activation_alpha, activation_beta,
):  # fmt: skip
    opts = _kernel_options
    l1_w, l1_sf = l1_weights
    l2_w, l2_sf = l2_weights
    s1_w, s1_sf = shared_l1_weights if shared_l1_weights is not None else (None, None)
    s2_w, s2_sf = shared_l2_weights if shared_l2_weights is not None else (None, None)
    stats = cumulative_local_expert_recv_stats
    try:
        rec_key = (
            id(sym_buffer),
            id(l1_w),
            id(l1_sf),
            id(l2_w),
            id(l2_sf),
            id(opts),
            activation_clamp,
            activation_alpha,
            activation_beta,
            fast_math,
            activation,
            tuple(recipe),
        )
        rec = _call_records.get(rec_key)
    except TypeError:  # unhashable argument (e.g. a list as ``recipe``): the validating path reports it
        rec_key = rec = None
    if rec is not None:
        sig = (
            _tensor_sig(l1_w),
            _tensor_sig(l1_sf),
            _tensor_sig(l2_w),
            _tensor_sig(l2_sf),
            _tensor_sig(s1_w),
            _tensor_sig(s1_sf),
            _tensor_sig(s2_w),
            _tensor_sig(s2_sf),
            _tensor_sig(stats),
        )
        buf_tensor = sym_buffer.buffer
        if (
            sig == rec.sig
            and rec.buf_ref() is sym_buffer
            and buf_tensor is not None
            and rec.buf_tensor_ref() is buf_tensor
            and rec.opts is opts
            and rec.per_die_mode is _per_die_mode
        ):
            shape = y.shape
            if (
                len(shape) == 2
                and shape[1] == rec.hidden
                and shape[0] <= rec.max_tokens
                and y.dtype is torch.bfloat16
                and y.is_contiguous()
                and not (y.data_ptr() & 15)
            ):
                num_tokens = shape[0]
                block_m = opts.block_m or _block_m_for_call(
                    rec.num_ranks,
                    rec.num_experts,
                    rec.max_tokens,
                    num_tokens,
                    rec.num_topk,
                )
                launch = rec.launchers.get(block_m)
                if launch is not None:
                    launch(y, num_tokens)
                    return None
    return _validate_and_launch(
        rec_key, y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, stats, recipe, activation, activation_clamp, fast_math,
        activation_alpha, activation_beta,
    )  # fmt: skip


def _validate_and_launch(
    rec_key, y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
    recipe, activation, activation_clamp, fast_math, activation_alpha, activation_beta,
):  # fmt: skip
    call = check_mega_moe_arguments(
        y, l1_weights, l2_weights, sym_buffer, shared_l1_weights, shared_l2_weights, cumulative_local_expert_recv_stats,
        recipe, activation, activation_clamp, activation_alpha, activation_beta, fast_math,
    )  # fmt: skip
    num_tokens = y.size(0)
    num_ranks = sym_buffer.group.size()
    # BLOCK_M of THIS call (per-rank, from the calling rank's own token count; nothing that crosses ranks depends on it)
    opts = _kernel_options
    block_m = opts.block_m or _block_m_for_call(
        num_ranks,
        sym_buffer.num_experts,
        sym_buffer.num_max_tokens_per_rank,
        num_tokens,
        sym_buffer.num_topk,
    )
    per_die = call.localized_weights if _per_die_mode is None else _per_die_mode
    key = (
        num_ranks, sym_buffer.num_experts, sym_buffer.num_max_tokens_per_rank, block_m, sym_buffer.num_topk, call.hidden,
        call.intermediate_hidden, call.num_shared_experts, sym_buffer.num_sms, call.clamp, float(activation_alpha),
        float(activation_beta), bool(fast_math), bool(per_die), call.localized_weights, call.localized_shared_weights,
    )  # fmt: skip
    runner = _cached_runner(*key, opts)
    s1 = shared_l1_weights if shared_l1_weights is not None else (None, None)
    s2 = shared_l2_weights if shared_l2_weights is not None else (None, None)
    domains = None
    if (
        per_die
        and _cached_kernel_config(
            *key, opts.smem_capacity, opts.max_stages
        ).num_locality_domains
        > 1
    ):
        domains = _sm_locality_domains(sym_buffer.num_sms)
    bind = getattr(runner, "bind", None)
    if (
        bind is None or rec_key is None
    ):  # a custom kernel factory / test double without the bound-launcher fast path: call it directly
        args = (y, cumulative_local_expert_recv_stats, num_tokens, sym_buffer,
                l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1], s1[0], s1[1], s2[0], s2[1])  # fmt: skip
        return (
            runner(*args, sm_locality_domains=domains)
            if domains is not None
            else runner(*args)
        )
    launch = bind(sym_buffer, l1_weights[0], l1_weights[1], l2_weights[0], l2_weights[1], s1[0], s1[1], s2[0], s2[1],
                  stats=cumulative_local_expert_recv_stats, sm_locality_domains=domains)  # fmt: skip
    rec = _call_records.get(rec_key)
    new_sig = (_tensor_sig(l1_weights[0]), _tensor_sig(l1_weights[1]), _tensor_sig(l2_weights[0]), _tensor_sig(l2_weights[1]), _tensor_sig(s1[0]), _tensor_sig(s1[1]),
               _tensor_sig(s2[0]), _tensor_sig(s2[1]), _tensor_sig(cumulative_local_expert_recv_stats))  # fmt: skip
    try:
        buf_ref, buf_tensor_ref = (
            weakref.ref(sym_buffer),
            weakref.ref(sym_buffer.buffer),
        )
    except TypeError:  # operands that cannot be weakly referenced (test doubles): no record, every call validates
        buf_ref = buf_tensor_ref = None
    if buf_ref is not None:
        if (
            rec is None
            or rec.sig != new_sig
            or rec.buf_ref() is not sym_buffer
            or rec.buf_tensor_ref() is not sym_buffer.buffer
            or rec.opts is not opts
            or rec.per_die_mode is not _per_die_mode
        ):
            if len(_call_records) >= _CALL_RECORDS_MAX:
                _call_records.clear()
            rec = _CallRecord()
            rec.sig, rec.buf_ref, rec.buf_tensor_ref, rec.opts, rec.per_die_mode = (
                new_sig,
                buf_ref,
                buf_tensor_ref,
                opts,
                _per_die_mode,
            )
            rec.hidden, rec.max_tokens, rec.num_ranks, rec.num_experts, rec.num_topk = (
                call.hidden,
                sym_buffer.num_max_tokens_per_rank,
                num_ranks,
                sym_buffer.num_experts,
                sym_buffer.num_topk,
            )
            rec.launchers = {}
            rec.domains = domains  # the process-wide SM -> locality-domain map (tiny, owned by ``locality``): pinned because the bound launchers hold its address
            _call_records[rec_key] = rec
        rec.launchers[block_m] = launch
    launch(y, num_tokens)
    return None


def precompile_mega_moe_variants(
    num_ranks: int,
    num_experts: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_shared_experts: int = 0,
    *,
    activation_clamp: Optional[float] = None,
    activation_alpha: float = 1.0,
    activation_beta: float = 0.0,
    fast_math: bool = True,
    num_sms: Optional[int] = None,
    block_ms: Optional[Tuple[int, ...]] = None,
    per_die: bool = False,
    localized_weights: bool = False,
    localized_shared_weights: bool = False,
    kernel_options: Optional[KernelOptions] = None,
) -> Tuple[int, ...]:
    """Build every kernel variant a rank can select for ``M in [0, T_max]`` (call at engine init / warmup, never in capture).

    Returns the BLOCK_M values compiled.  ``block_ms`` restricts the set (default: all distinct ``get_block_m_for_mega_moe``
    results for this configuration; DeepGEMM's vLLM warmup enumerates exactly this set)."""
    from .heuristics import resolve_num_sms

    _check_supported_variant(fast_math, num_shared_experts, localized_shared_weights)
    sms = resolve_num_sms(num_sms)
    T = align(num_max_tokens_per_rank, 1920)
    opts = kernel_options if kernel_options is not None else _kernel_options
    todo = (
        tuple(block_ms)
        if block_ms is not None
        else (
            (opts.block_m,)
            if opts.block_m
            else get_block_m_variants(num_ranks, num_experts, num_topk, T)
        )
    )
    for bm in todo:
        assert bm in VALID_BLOCK_MS, bm
        _cached_runner(
            num_ranks, num_experts, T, bm, num_topk, hidden, intermediate_hidden, num_shared_experts, sms,
            math.inf if activation_clamp is None else float(activation_clamp), float(activation_alpha),
            float(activation_beta), bool(fast_math), bool(per_die), bool(localized_weights), bool(localized_shared_weights), opts,
        )  # fmt: skip
    return todo
