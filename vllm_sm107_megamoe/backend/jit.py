# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) csrc/jit_kernels/impls/sm107_fp8_fp4_mega_moe.hpp (launcher / JIT).
"""Compile + cache of the SM107 DeepGEMM-style MegaMoE kernel variants (DESIGN.md section 15).

``get_or_compile_kernel(cfg)`` returns a :class:`MegaMoeRunner` for ONE ``Sm107KernelConfig`` (= one BLOCK_M variant):

* the kernel is compiled with ``cute.compile[EnableTVMFFI]`` through ``flashinfer.jit.cute_dsl_core.build_and_load_cute_dsl_kernel``
  (disk cache keyed by ``KernelBuild.name()`` = short readable prefix + 16-hex digest of EVERY constexpr, invalidated by the DSL stack
  fingerprint and a SHA-256 over all kernel sources; ``FLASHINFER_CUTE_DSL_DISABLE_CACHE=1`` disables the disk layer),
* an in-process memo avoids repeated cache lookups; compiling inside a CUDA-graph capture raises a clear error,
* the runner adapts torch tensors / the package's ``SymmBuffer`` to the compiled ABI (DESIGN.md section 2): pointers as ints, the peer table as the nested tuple
  ``(base, (offsets...), rank, max_ranks)``, the CUDA stream as an explicit handle (``EnvStream`` cannot be used: it needs a GPU-tensor parameter).

TVM-FFI facts measured on Rubin: ``DataPointer`` parameters accept Python ints (NOT torch tensors); the stream parameter is an int handle
(``torch.cuda.current_stream().cuda_stream``); a dataclass argument is flattened to a nested tuple in field order including the constexpr
``max_ranks`` (which must be passed back as the same int).
"""

from __future__ import annotations

import glob
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

# No environment mutation at import (FlashInfer experimental policy): the DSL auto-detects the device arch ("sm_107a" on SM107,
# cutlass.base_dsl.env_manager.detect_gpu_arch) and the target chip is passed explicitly with cute.GPUArch(build.arch) in compile_options();
# compile-only hosts without an SM107 GPU must export CUTE_DSL_ARCH=sm_107a themselves.

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_NAME = "sm107_deepgemm_megamoe"
DEFAULT_PTXAS_OPTIONS = (
    "--register-usage-level=10"  # DeepGEMM: `--ptxas-options=--register-usage-level=10`
)

_LOCK = threading.RLock()
_MEMO: Dict[str, "MegaMoeRunner"] = {}
# first-level memo keyed by the call arguments (cfg object identity + knobs): skips make_build()/name()/digest hashing (~100-200 us) on repeated calls with the same cfg object
_FAST: Dict[tuple, Tuple[Any, "MegaMoeRunner"]] = {}
_STATS: Dict[str, Any] = {"compiles": 0, "memo_hits": 0, "last_compile_seconds": {}}


# ---------------------------------------------------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------------------------------------------------
def kernel_source_files() -> List[str]:
    """Every source file whose content must invalidate the compile cache (kernel code + config/heuristics + this file)."""
    files = sorted(glob.glob(os.path.join(_PKG_DIR, "kernel", "*.py")))
    files += [
        os.path.join(_PKG_DIR, f) for f in ("jit.py", "config.py", "heuristics.py")
    ]
    return [f for f in files if os.path.isfile(f)]


def make_build(cfg, *, mode_mask: int = 7, pdl: bool = False, roles: str = "auto", arch: str = "sm_107a",
               ptxas_options: str = DEFAULT_PTXAS_OPTIONS, opt_level: int = 3, debug_marks: bool = False,
               debug_gridsync_iters: int = 0, dispatch_only_sync: bool = False, tag_mode: str = "epoch",
               regs_override=None, timeout_sec: float = 300.0, faithful_regs: bool = False):  # fmt: skip
    from .kernel.config import KernelBuild

    return KernelBuild(cfg, mode_mask=mode_mask, pdl=pdl, arch=arch, ptxas_options=ptxas_options, opt_level=opt_level,
                       debug_marks=debug_marks, roles=roles, debug_gridsync_iters=debug_gridsync_iters,
                       dispatch_only_sync=dispatch_only_sync, tag_mode=tag_mode, regs_override=regs_override, timeout_sec=timeout_sec, faithful_regs=faithful_regs)  # fmt: skip


def _fake_args(cfg):
    """Fake (metadata-only) arguments defining the compiled ABI (DESIGN.md section 2); values are runtime scalars/pointers, never baked in."""
    import cutlass
    from cutlass import Int32, Int64
    from cutlass.cute.runtime import make_fake_stream, make_ptr
    from cutlass.cute.typing import AddressSpace

    from .kernel import comm

    gp = lambda dt, al=16: make_ptr(dt, 0, AddressSpace.gmem, assumed_align=al)  # noqa: E731
    R = cfg.num_ranks
    sym = comm.SymBufferHost(Int64(0), tuple(Int64(0) for _ in range(R)), Int32(0), R)
    return (
        gp(cutlass.BFloat16), gp(cutlass.Int32, 4), Int32(0), sym,
        gp(cutlass.Uint8), gp(cutlass.Int32), gp(cutlass.Uint8), gp(cutlass.Int32),
        gp(cutlass.Uint8), gp(cutlass.Int32), gp(cutlass.Uint8), gp(cutlass.Int32),
        gp(cutlass.Uint64, 8), gp(cutlass.Uint8, 1),
        make_fake_stream(),
    )  # fmt: skip


def compile_options(build) -> tuple:
    import cutlass.cute as cute

    opts: list = [
        cute.EnableTVMFFI(True),
        cute.OptLevel(build.opt_level),
        cute.GPUArch(build.arch),
    ]
    if build.ptxas_options:
        opts.append(cute.PtxasOptions(build.ptxas_options))
    return tuple(opts)


def compile_kernel_object(build):
    """``cute.compile`` of one variant (in-memory, TVM-FFI enabled). Returns the compiled kernel object (has ``export_to_c``)."""
    import cutlass.cute as cute

    from .kernel import comm
    from .kernel.kernel import DeepGemmSm107MegaMoeKernel

    # The spin-wait timeout is a trace-time constant of the PTX emitters (comm.py / ptx_d.py read these module globals when the kernel is traced); compiles are
    # serialised by `_LOCK`, so set them for the duration of this compile only.
    cycles = int(float(build.timeout_sec) * 2_000_000_000)
    saved = [(comm, "TIMEOUT_CYCLES", comm.TIMEOUT_CYCLES)]
    comm.TIMEOUT_CYCLES = cycles
    try:
        from .kernel import ptx_d

        saved.append((ptx_d, "NUM_TIMEOUT_CYCLES", ptx_d.NUM_TIMEOUT_CYCLES))
        ptx_d.NUM_TIMEOUT_CYCLES = cycles
    except ImportError:  # ptx_d absent (stub-only trees)
        pass
    try:
        kern = DeepGemmSm107MegaMoeKernel(build)
        return cute.compile[compile_options(build)](kern, *_fake_args(build.cfg))
    finally:
        for mod, name, val in saved:
            setattr(mod, name, val)


# ---------------------------------------------------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------------------------------------------------
class MegaMoeRunner:
    """Launch adapter of one compiled variant (host-level ABI: torch tensors + ``SymmBuffer``).

    ``runner(y, stats, num_tokens, sym_buffer, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w=None, shared_l1_w_sf=None, shared_l2_w=None,
    shared_l2_w_sf=None, *, trace=None, sm_locality_domains=None, stream=None)``; ``sym_buffer`` needs ``base_ptr`` (int), ``peer_offsets``
    (list of R ints, ``[rank] == 0``) and ``group.rank()`` (or ``rank_idx``).  Launches on the current torch stream (or ``stream``), never
    synchronizes, and is CUDA-graph capturable (all Python-side conversions are plain integer reads).
    """

    def __init__(self, build, fn: Callable, compile_seconds: float, source: str):
        self.build = build
        self.cfg = build.cfg
        self.fn = fn
        self.compile_seconds = compile_seconds
        self.source = source  # "compiled" | "disk-cache" | "memo"
        self._sym_cache: Dict[
            int, tuple
        ] = {}  # id(sym_buffer) -> (sym_buffer (pinned), base, peer offsets, payload)
        # Shared experts (S > 0): the real roles do not implement them yet -> launching would silently return wrong results, so refuse (compiling stays allowed).
        # Per-die variants (num_locality_domains > 1) read the physical-SM -> locality-domain map through `sm_locality_domains`: a null map would be an illegal memory access in the scheduler warp.
        self.needs_domain_map = int(getattr(build.cfg, "num_locality_domains", 1)) > 1
        self.unsupported: Optional[str] = None
        if build.cfg.num_shared_experts > 0:
            from .kernel.kernel import shared_experts_supported

            if not shared_experts_supported(build.roles, build.mode_mask):
                self.unsupported = (
                    f"variant {build.name()} has num_shared_experts={build.cfg.num_shared_experts}, but the CuteDSL GEMM roles do not implement shared experts yet "
                    "(each GEMM role module must declare SUPPORTS_SHARED_EXPERTS = True); refusing to launch (the shared-expert contribution would be silently dropped)"
                )
        # Die-local SHARED weights (kLocalizedSharedWeights): the weight tensor maps of the kernel only implement die-local ROUTED weights, a launch would read wrong rows.
        if self.unsupported is None and getattr(
            build.cfg, "localized_shared_weights", False
        ):
            self.unsupported = (
                f"variant {build.name()} has localized_shared_weights=True, but the kernel's weight tensor maps only implement die-local ROUTED weights "
                "(DeepGEMM PERDIE=routed); refusing to launch (the shared-expert rows would be read from the wrong addresses)"
            )

    @property
    def name(self) -> str:
        return self.build.name()

    def sym_payload(self, sym_buffer) -> tuple:
        """The ``(base, (offsets...), rank, R)`` tuple of the compiled ABI.

        Cached per buffer OBJECT: the entry holds a reference to the buffer (so its ``id`` cannot be reused by another buffer while the entry lives) and is re-validated
        against the buffer's current base address and peer offsets on every call (the payload is a function of exactly these and of the buffer's rank)."""
        base = int(sym_buffer.base_ptr)
        offsets = tuple(sym_buffer.peer_offsets)
        hit = self._sym_cache.get(id(sym_buffer))
        if (
            hit is not None
            and hit[0] is sym_buffer
            and hit[1] == base
            and hit[2] == offsets
        ):
            return hit[3]
        offsets = tuple(int(o) for o in offsets)
        rank = (
            int(sym_buffer.group.rank())
            if hasattr(sym_buffer, "group") and sym_buffer.group is not None
            else int(getattr(sym_buffer, "rank_idx", 0))
        )
        if len(offsets) != self.cfg.num_ranks:
            raise ValueError(
                f"symmetric buffer has {len(offsets)} peers, kernel variant is built for {self.cfg.num_ranks} ranks"
            )
        payload = (base, offsets, rank, self.cfg.num_ranks)
        if len(self._sym_cache) > 64:
            self._sym_cache.clear()
        self._sym_cache[id(sym_buffer)] = (sym_buffer, base, offsets, payload)
        return payload

    def bind(self, sym_buffer, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w=None, shared_l1_w_sf=None, shared_l2_w=None, shared_l2_w_sf=None,
             *, stats=None, trace=None, sm_locality_domains=None) -> Callable:  # fmt: skip
        """Fast path for static operands (one MoE layer): returns ``launch(y, num_tokens, stream=None)`` with every pointer/peer table precomputed
        (per call: ``y.data_ptr()`` + the stream handle). The tensors must stay alive and un-reallocated while the callable is used."""
        import torch

        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported)
        if self.needs_domain_map and sm_locality_domains is None:
            raise ValueError(
                f"{self.name}: per-die variant (num_locality_domains={self.cfg.num_locality_domains}) needs the `sm_locality_domains` map (uint8 per physical SM)"
            )
        ptr = lambda t: 0 if t is None else int(t.data_ptr())  # noqa: E731
        sym = self.sym_payload(sym_buffer)
        fixed = (
            ptr(l1_w),
            ptr(l1_w_sf),
            ptr(l2_w),
            ptr(l2_w_sf),
            ptr(shared_l1_w),
            ptr(shared_l1_w_sf),
            ptr(shared_l2_w),
            ptr(shared_l2_w_sf),
        )
        tail = (ptr(trace), ptr(sm_locality_domains))
        stats_ptr = ptr(stats)
        fn = self.fn
        current_stream = torch.cuda.current_stream

        def launch(y, num_tokens: int, stream=None) -> None:
            st = (
                current_stream().cuda_stream
                if stream is None
                else (stream.cuda_stream if hasattr(stream, "cuda_stream") else stream)
            )
            fn(int(y.data_ptr()), stats_ptr, int(num_tokens), sym, *fixed, *tail, st)

        return launch

    def call_raw(self, y_ptr: int, stats_ptr: int, num_tokens: int, sym: tuple, l1_w: int, l1_w_sf: int, l2_w: int, l2_w_sf: int,
                 sh_l1_w: int = 0, sh_l1_w_sf: int = 0, sh_l2_w: int = 0, sh_l2_w_sf: int = 0, trace_ptr: int = 0,
                 domains_ptr: int = 0, stream: Optional[int] = None) -> None:  # fmt: skip
        """Pointer-level call in the compiled ABI order (all pointers as Python ints; ``sym = (base, offsets, rank, R)``)."""
        if self.unsupported is not None:
            raise NotImplementedError(self.unsupported)
        if self.needs_domain_map and not domains_ptr:
            raise ValueError(
                f"{self.name}: per-die variant (num_locality_domains={self.cfg.num_locality_domains}) needs the `sm_locality_domains` map (uint8 per physical SM)"
            )
        if stream is None:
            import torch

            stream = torch.cuda.current_stream().cuda_stream
        self.fn(y_ptr, stats_ptr, num_tokens, sym, l1_w, l1_w_sf, l2_w, l2_w_sf, sh_l1_w, sh_l1_w_sf, sh_l2_w, sh_l2_w_sf,
                trace_ptr, domains_ptr, stream)  # fmt: skip

    def __call__(self, y, stats, num_tokens, sym_buffer, l1_w, l1_w_sf, l2_w, l2_w_sf, shared_l1_w=None, shared_l1_w_sf=None,
                 shared_l2_w=None, shared_l2_w_sf=None, *, trace=None, sm_locality_domains=None, stream=None) -> None:  # fmt: skip
        ptr = lambda t: 0 if t is None else int(t.data_ptr())  # noqa: E731
        if stream is not None and hasattr(stream, "cuda_stream"):
            stream = stream.cuda_stream
        self.call_raw(
            ptr(y), ptr(stats), int(num_tokens), self.sym_payload(sym_buffer), ptr(l1_w), ptr(l1_w_sf), ptr(l2_w), ptr(l2_w_sf),
            ptr(shared_l1_w), ptr(shared_l1_w_sf), ptr(shared_l2_w), ptr(shared_l2_w_sf), ptr(trace), ptr(sm_locality_domains), stream,
        )  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------------------------------------------------
def _ensure_not_capturing(what: str) -> None:
    import torch

    if torch.cuda.is_available() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            f"{what} requires a host-side compile and cannot run during CUDA graph capture; call get_or_compile_kernel()/precompile "
            "for every BLOCK_M variant on ALL ranks before capturing."
        )


_RESIDENCY_CHECKED: Dict[Tuple[int, int, int], int] = {}


def check_residency(cfg) -> int:
    """Plan-time guards of the persistent kernel: the device must have exactly ``cfg.num_sms`` SMs and room for ``num_sms/2``
    co-resident 2-CTA clusters (every software grid sync / NVLink barrier deadlocks otherwise). Cached per (device, num_sms). Returns max active clusters."""
    import torch

    dev = torch.cuda.current_device()
    key = (dev, cfg.num_sms, cfg.smem_bytes)
    hit = _RESIDENCY_CHECKED.get(key)
    if hit is not None:
        return hit
    sms = torch.cuda.get_device_properties(dev).multi_processor_count
    if sms != cfg.num_sms:
        raise RuntimeError(
            f"kernel variant built for {cfg.num_sms} SMs but device {dev} has {sms} (DeepGEMM requires kNumSMs == the whole device)"
        )
    from cutlass.utils import HardwareInfo

    clusters = int(HardwareInfo(dev).get_max_active_clusters(2))
    if clusters * 2 < cfg.num_sms:
        raise RuntimeError(
            f"only {clusters} 2-CTA clusters can be co-resident, {cfg.num_sms // 2} are needed: the grid syncs would deadlock"
        )
    _RESIDENCY_CHECKED[key] = clusters
    return clusters


def get_or_compile_build(build, *, use_disk_cache: bool = True) -> MegaMoeRunner:
    """Runner for ``build`` (a ``KernelBuild``): in-process memo -> disk cache -> compile (+ export)."""
    name = build.name()
    with _LOCK:
        hit = _MEMO.get(name)
        if hit is not None:
            _STATS["memo_hits"] += 1
            return hit
        _ensure_not_capturing(f"compiling kernel {name}")
        check_residency(build.cfg)
        t0 = time.time()
        state = {"compiled_now": False}

        def compile_fn():
            state["compiled_now"] = True
            return compile_kernel_object(build)

        if use_disk_cache:
            from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

            fn = build_and_load_cute_dsl_kernel(
                MODULE_NAME, name, compile_fn, extra_key_files=kernel_source_files()
            )
        else:
            fn = compile_fn()
        dt = time.time() - t0
        if state["compiled_now"]:
            _STATS["compiles"] += 1
        _STATS["last_compile_seconds"][name] = dt
        runner = MegaMoeRunner(
            build, fn, dt, "compiled" if state["compiled_now"] else "disk-cache"
        )
        _MEMO[name] = runner
        return runner


def get_or_compile_kernel(cfg, *, mode_mask: int = 7, pdl: bool = False, roles: str = "auto", arch: str = "sm_107a",
                          ptxas_options: str = DEFAULT_PTXAS_OPTIONS, opt_level: int = 3, debug_marks: bool = False,
                          debug_gridsync_iters: int = 0, dispatch_only_sync: bool = False, tag_mode: str = "epoch",
                          regs_override=None, timeout_sec: float = 300.0, faithful_regs: bool = False, use_disk_cache: bool = True) -> MegaMoeRunner:  # fmt: skip
    """The host-level runner for one ``Sm107KernelConfig`` (see :class:`MegaMoeRunner`). Cheap after the first call (memo).

    ``tag_mode``: "epoch" (default: launch-unique combine tag, safe under CUDA-graph replay) or "gridid" (DeepGEMM's `%gridid + 1`: stale results under graph replay with changing inputs; parity runs only).
    ``regs_override``: optional (dispatch, non-epilogue, epilogue) setmaxnreg split replacing the config's (DESIGN.md section 8.4 / KernelBuild).
    ``timeout_sec``: spin-wait timeout baked into the kernel (DeepGEMM: 300 s); part of the cache key (replaces the SM107_DG_TIMEOUT_SEC environment variable for tests).
    ``faithful_regs``: False (product default) = the spill-free ``setmaxnreg`` split 96/88/160 at every EP size; True = DeepGEMM's own split (96/88/160 when experts-per-rank > 64, else 48/40/208: its 48-register
    dispatch cap spills at DEP>=8) for parity / mapping runs. ``regs_override`` (explicit split) is mutually exclusive with ``faithful_regs=True``.
    """
    regs_key = None if regs_override is None else tuple(int(v) for v in regs_override)
    key = (
        id(cfg),
        mode_mask,
        pdl,
        roles,
        arch,
        ptxas_options,
        opt_level,
        debug_marks,
        debug_gridsync_iters,
        dispatch_only_sync,
        tag_mode,
        regs_key,
        float(timeout_sec),
        bool(faithful_regs),
        use_disk_cache,
    )
    hit = _FAST.get(key)
    if hit is not None and hit[0] is cfg:
        _STATS["memo_hits"] += 1
        return hit[1]
    build = make_build(cfg, mode_mask=mode_mask, pdl=pdl, roles=roles, arch=arch, ptxas_options=ptxas_options, opt_level=opt_level,
                       debug_marks=debug_marks, debug_gridsync_iters=debug_gridsync_iters,
                       dispatch_only_sync=dispatch_only_sync, tag_mode=tag_mode, regs_override=regs_key, timeout_sec=timeout_sec, faithful_regs=faithful_regs)  # fmt: skip
    runner = get_or_compile_build(build, use_disk_cache=use_disk_cache)
    with _LOCK:
        if len(_FAST) > 256:
            _FAST.clear()
        _FAST[key] = (
            cfg,
            runner,
        )  # keeps cfg alive, so id(cfg) cannot be recycled while the entry exists
    return runner


def is_compiled(cfg, *, mode_mask: int = 7, pdl: bool = False, roles: str = "auto", arch: str = "sm_107a", ptxas_options: str = DEFAULT_PTXAS_OPTIONS,
                opt_level: int = 3, debug_marks: bool = False, debug_gridsync_iters: int = 0, dispatch_only_sync: bool = False, tag_mode: str = "epoch",
                regs_override=None, timeout_sec: float = 300.0, faithful_regs: bool = False) -> bool:  # fmt: skip
    """True when this variant is already held by the in-process memo, i.e. ``get_or_compile_kernel`` with the same arguments cannot compile (safe inside a CUDA-graph capture).
    A variant that exists only in the disk cache is NOT "compiled" here (loading it is still host work): call ``get_or_compile_kernel``/``precompile_variants`` before capturing."""
    build = make_build(cfg, mode_mask=mode_mask, pdl=pdl, roles=roles, arch=arch, ptxas_options=ptxas_options, opt_level=opt_level, debug_marks=debug_marks,
                       debug_gridsync_iters=debug_gridsync_iters, dispatch_only_sync=dispatch_only_sync, tag_mode=tag_mode, regs_override=regs_override, timeout_sec=timeout_sec, faithful_regs=faithful_regs)  # fmt: skip
    with _LOCK:
        return build.name() in _MEMO


def compiled_variant_names() -> List[str]:
    """Names of every variant held by the in-process memo (logs / capture-readiness reports)."""
    with _LOCK:
        return sorted(_MEMO)


compiled_names = compiled_variant_names  # short alias


def precompile_variants(cfgs: Sequence[Any], **kw) -> List[MegaMoeRunner]:
    """Compile (or load) every config of ``cfgs`` (e.g. one per BLOCK_M a rank can select); call before any graph capture."""
    return [get_or_compile_kernel(c, **kw) for c in cfgs]


def clear_memo() -> None:
    with _LOCK:
        _MEMO.clear()
        _FAST.clear()


def stats() -> Dict[str, Any]:
    return dict(_STATS)
