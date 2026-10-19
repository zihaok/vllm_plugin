# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) deep_gemm/mega/sm107_locality.py + csrc/jit_kernels/impls/sm107_locality.hpp + deep_gemm/include/deep_gemm/impls/sm107_locality_probe.cuh (SM107 / Rubin locality domains).
"""Per-die (locality-domain) host helpers of the SM107 MegaMoE, in pure Python on top of ``cuda-python`` 13.4 (``cuda.bindings.driver``).

Rubin (VR200) has two locality domains ("dies") per GPU.  The per-die MegaMoE variant needs three host-side pieces, all
provided here WITHOUT importing ``deep_gemm`` and without compiling any C++ at run time:

1. **die-local weight allocation** -- :func:`localize` turns ``(*B, N, K)`` into ``(2, *B, N/2, K)`` with slice ``d`` homed in
   locality domain ``d`` (CUDA 13.4 native: ``cuMemCreate`` on ``{CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN, {device, domain}}``,
   ``cuMemAddressReserve`` / ``cuMemMap`` / ``cuMemSetAccess``; slices are padded to the allocation granularity, 2 MiB on Rubin);
   :func:`localize_mega_moe_weights` applies it to the transformed ``(weights, sf)`` tuples (scale factors stay where they are);
2. **SM -> domain map** -- :func:`get_balanced_sm_locality_domains`: a ``uint8[num_sms]`` device array indexed by the PHYSICAL ``%smid``
   (NOT ``blockIdx``), produced by DeepGEMM's pointer-chasing latency probe and balanced to equal TPC counts (an imbalanced map hangs
   per-die scheduling, DeepGEMM commit f428f2d).  The probe kernel is DeepGEMM's ``sm107_locality_probe_chase<32, 8, 4096>`` compiled
   once to PTX (``nvcc -ptx -arch=sm_90``, ``.version`` lowered to 8.0) and JIT-loaded by the driver (``cuModuleLoadData``);
   it must be called OUTSIDE CUDA graph capture (it allocates, launches and synchronizes) -- the kernel launch only reuses the cached map;
3. **introspection** -- :func:`is_localized`, :func:`weights_localized`, :func:`get_locality_domain_of`.

Memory ownership: the allocation is exported to torch through ``__cuda_array_interface__``; every tensor / view keeps the reservation
alive and the last one to die unmaps and releases it (the same lifetime DeepGEMM gets from the stable ``from_blob`` deleter).

Assumptions inherited from DeepGEMM: one CUDA device per process (the map and the probe module are cached per device ordinal), a device
that reports exactly 2 locality domains (``CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT``), compute capability (10, 7).
"""

from __future__ import annotations

import ctypes
import threading
import warnings
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

NUM_SMS_PER_TPC = 2  # the two SMs of a TPC share a locality domain
_PROBE_NUM_HOPS = 32  # kProbeHops = chunk bytes / line bytes
_PROBE_CHUNKS_PER_SM = 8  # kProbeChunksPerSM
_PROBE_CHUNK_BYTES = 4096  # kProbeChunkBytes
_PROBE_LINE_BYTES = 128  # kProbeLineBytes
_PROBE_CHAIN_LINE_STRIDE = (
    7  # kChainLineStride: coprime with the line count, so the chain visits every line
)
_PROBE_MIN_FAR_NEAR_RATIO = 1.25  # kMinFarNearRatio
_PROBE_MAX_ATTEMPTS = 5  # kNumMaxProbeAttempts
_PROBE_SMEM_BYTES = 232448  # SM100ArchSpec::smem_capacity: the maximum shared memory keeps one block per SM
_PROBE_KERNEL_NAME = b"sm107_locality_probe_chase"

# DeepGEMM's `sm107_locality_probe_chase<32, 8, 4096>` (deep_gemm/impls/sm107_locality_probe.cuh, MIT License, DeepSeek) with an
# extern "C" entry point: `nvcc -ptx -arch=sm_90` (CUDA 13.4.92), `.version` 9.4 lowered to 8.0 (only basic instructions are used:
# `ld.global.cg.u32`, `%smid`, `%clock64`).  One thread per block chases pointer chains through 8 chunks of 4 KiB per SM, one hop per
# 128-byte line, and stores the cycles per hop of each chunk as uint16 at `out[smid * 8 + chunk]` (0 if the chain was corrupted).
_PROBE_PTX = """\
.version 8.0
.target sm_90
.address_size 64

    // .globl    sm107_locality_probe_chase

.visible .entry sm107_locality_probe_chase(
    .param .u64 sm107_locality_probe_chase_param_0,
    .param .u64 sm107_locality_probe_chase_param_1
)
.maxntid 32, 1, 1
.minnctapersm 1
{
    .reg .pred     %p<4>;
    .reg .b32     %r<47>;
    .reg .b64     %rd<86>;


    ld.param.u64     %rd8, [sm107_locality_probe_chase_param_0];
    ld.param.u64     %rd9, [sm107_locality_probe_chase_param_1];
    mov.u32     %r5, %tid.x;
    setp.ne.s32     %p1, %r5, 0;
    @%p1 bra     $L__BB0_5;

    // begin inline asm
    mov.u32 %r6, %smid;
    // end inline asm
    shl.b32     %r8, %r6, 3;
    cvta.to.global.u64     %rd10, %rd9;
    mul.wide.u32     %rd11, %r8, 2;
    add.s64     %rd85, %rd10, %rd11;
    mul.wide.u32     %rd12, %r8, 1024;
    shl.b64     %rd13, %rd12, 2;
    add.s64     %rd84, %rd8, %rd13;
    mov.u32     %r7, 0;
    mov.u32     %r45, %r7;

$L__BB0_2:
    // begin inline asm
    ld.global.cg.u32 %r9, [%rd84];
    // end inline asm
    // begin inline asm
    mov.u64     %rd15, %clock64;
    // end inline asm
    mul.wide.u32     %rd49, %r9, 4;
    add.s64     %rd16, %rd84, %rd49;
    // begin inline asm
    ld.global.cg.u32 %r10, [%rd16];
    // end inline asm
    mul.wide.u32     %rd50, %r10, 4;
    add.s64     %rd17, %rd84, %rd50;
    // begin inline asm
    ld.global.cg.u32 %r11, [%rd17];
    // end inline asm
    mul.wide.u32     %rd51, %r11, 4;
    add.s64     %rd18, %rd84, %rd51;
    // begin inline asm
    ld.global.cg.u32 %r12, [%rd18];
    // end inline asm
    mul.wide.u32     %rd52, %r12, 4;
    add.s64     %rd19, %rd84, %rd52;
    // begin inline asm
    ld.global.cg.u32 %r13, [%rd19];
    // end inline asm
    mul.wide.u32     %rd53, %r13, 4;
    add.s64     %rd20, %rd84, %rd53;
    // begin inline asm
    ld.global.cg.u32 %r14, [%rd20];
    // end inline asm
    mul.wide.u32     %rd54, %r14, 4;
    add.s64     %rd21, %rd84, %rd54;
    // begin inline asm
    ld.global.cg.u32 %r15, [%rd21];
    // end inline asm
    mul.wide.u32     %rd55, %r15, 4;
    add.s64     %rd22, %rd84, %rd55;
    // begin inline asm
    ld.global.cg.u32 %r16, [%rd22];
    // end inline asm
    mul.wide.u32     %rd56, %r16, 4;
    add.s64     %rd23, %rd84, %rd56;
    // begin inline asm
    ld.global.cg.u32 %r17, [%rd23];
    // end inline asm
    mul.wide.u32     %rd57, %r17, 4;
    add.s64     %rd24, %rd84, %rd57;
    // begin inline asm
    ld.global.cg.u32 %r18, [%rd24];
    // end inline asm
    mul.wide.u32     %rd58, %r18, 4;
    add.s64     %rd25, %rd84, %rd58;
    // begin inline asm
    ld.global.cg.u32 %r19, [%rd25];
    // end inline asm
    mul.wide.u32     %rd59, %r19, 4;
    add.s64     %rd26, %rd84, %rd59;
    // begin inline asm
    ld.global.cg.u32 %r20, [%rd26];
    // end inline asm
    mul.wide.u32     %rd60, %r20, 4;
    add.s64     %rd27, %rd84, %rd60;
    // begin inline asm
    ld.global.cg.u32 %r21, [%rd27];
    // end inline asm
    mul.wide.u32     %rd61, %r21, 4;
    add.s64     %rd28, %rd84, %rd61;
    // begin inline asm
    ld.global.cg.u32 %r22, [%rd28];
    // end inline asm
    mul.wide.u32     %rd62, %r22, 4;
    add.s64     %rd29, %rd84, %rd62;
    // begin inline asm
    ld.global.cg.u32 %r23, [%rd29];
    // end inline asm
    mul.wide.u32     %rd63, %r23, 4;
    add.s64     %rd30, %rd84, %rd63;
    // begin inline asm
    ld.global.cg.u32 %r24, [%rd30];
    // end inline asm
    mul.wide.u32     %rd64, %r24, 4;
    add.s64     %rd31, %rd84, %rd64;
    // begin inline asm
    ld.global.cg.u32 %r25, [%rd31];
    // end inline asm
    mul.wide.u32     %rd65, %r25, 4;
    add.s64     %rd32, %rd84, %rd65;
    // begin inline asm
    ld.global.cg.u32 %r26, [%rd32];
    // end inline asm
    mul.wide.u32     %rd66, %r26, 4;
    add.s64     %rd33, %rd84, %rd66;
    // begin inline asm
    ld.global.cg.u32 %r27, [%rd33];
    // end inline asm
    mul.wide.u32     %rd67, %r27, 4;
    add.s64     %rd34, %rd84, %rd67;
    // begin inline asm
    ld.global.cg.u32 %r28, [%rd34];
    // end inline asm
    mul.wide.u32     %rd68, %r28, 4;
    add.s64     %rd35, %rd84, %rd68;
    // begin inline asm
    ld.global.cg.u32 %r29, [%rd35];
    // end inline asm
    mul.wide.u32     %rd69, %r29, 4;
    add.s64     %rd36, %rd84, %rd69;
    // begin inline asm
    ld.global.cg.u32 %r30, [%rd36];
    // end inline asm
    mul.wide.u32     %rd70, %r30, 4;
    add.s64     %rd37, %rd84, %rd70;
    // begin inline asm
    ld.global.cg.u32 %r31, [%rd37];
    // end inline asm
    mul.wide.u32     %rd71, %r31, 4;
    add.s64     %rd38, %rd84, %rd71;
    // begin inline asm
    ld.global.cg.u32 %r32, [%rd38];
    // end inline asm
    mul.wide.u32     %rd72, %r32, 4;
    add.s64     %rd39, %rd84, %rd72;
    // begin inline asm
    ld.global.cg.u32 %r33, [%rd39];
    // end inline asm
    mul.wide.u32     %rd73, %r33, 4;
    add.s64     %rd40, %rd84, %rd73;
    // begin inline asm
    ld.global.cg.u32 %r34, [%rd40];
    // end inline asm
    mul.wide.u32     %rd74, %r34, 4;
    add.s64     %rd41, %rd84, %rd74;
    // begin inline asm
    ld.global.cg.u32 %r35, [%rd41];
    // end inline asm
    mul.wide.u32     %rd75, %r35, 4;
    add.s64     %rd42, %rd84, %rd75;
    // begin inline asm
    ld.global.cg.u32 %r36, [%rd42];
    // end inline asm
    mul.wide.u32     %rd76, %r36, 4;
    add.s64     %rd43, %rd84, %rd76;
    // begin inline asm
    ld.global.cg.u32 %r37, [%rd43];
    // end inline asm
    mul.wide.u32     %rd77, %r37, 4;
    add.s64     %rd44, %rd84, %rd77;
    // begin inline asm
    ld.global.cg.u32 %r38, [%rd44];
    // end inline asm
    mul.wide.u32     %rd78, %r38, 4;
    add.s64     %rd45, %rd84, %rd78;
    // begin inline asm
    ld.global.cg.u32 %r39, [%rd45];
    // end inline asm
    mul.wide.u32     %rd79, %r39, 4;
    add.s64     %rd46, %rd84, %rd79;
    // begin inline asm
    ld.global.cg.u32 %r40, [%rd46];
    // end inline asm
    mul.wide.u32     %rd80, %r40, 4;
    add.s64     %rd47, %rd84, %rd80;
    // begin inline asm
    ld.global.cg.u32 %r41, [%rd47];
    // end inline asm
    // begin inline asm
    mov.u64     %rd48, %clock64;
    // end inline asm
    sub.s64     %rd5, %rd48, %rd15;
    setp.eq.s32     %p2, %r41, -1;
    mov.u32     %r46, %r7;
    @%p2 bra     $L__BB0_4;

    shr.s64     %rd81, %rd5, 63;
    shr.u64     %rd82, %rd81, 59;
    add.s64     %rd83, %rd5, %rd82;
    cvt.u32.u64     %r43, %rd83;
    shr.u32     %r44, %r43, 5;
    and.b32      %r46, %r44, 65535;

$L__BB0_4:
    st.global.u16     [%rd85], %r46;
    add.s64     %rd85, %rd85, 2;
    add.s64     %rd84, %rd84, 4096;
    add.s32     %r45, %r45, 1;
    setp.ne.s32     %p3, %r45, 8;
    @%p3 bra     $L__BB0_2;

$L__BB0_5:
    ret;

}"""


class LocalityDomainError(RuntimeError):
    """Locality domains are unavailable or an allocation / driver call failed."""


# --------------------------------------------------------------------------------------------------------------------
# cuda-python helpers
# --------------------------------------------------------------------------------------------------------------------
def _drv():
    try:
        from cuda.bindings import driver as drv
    except ImportError as exc:  # pragma: no cover - cuda-python missing
        raise LocalityDomainError(
            "locality domains need cuda-python >= 13.4 (cuda.bindings.driver)"
        ) from exc
    return drv


def _ck(result: Tuple[Any, ...], what: str):
    """Unwrap a cuda-python ``(err, *values)`` result."""
    drv = _drv()
    err = result[0]
    if err != drv.CUresult.CUDA_SUCCESS:
        raise LocalityDomainError(f"{what} failed: {err}")
    if len(result) == 1:
        return None
    return result[1] if len(result) == 2 else result[1:]


def _device_ordinal() -> int:
    return torch.cuda.current_device()


def _cu_device():
    return _ck(_drv().cuDeviceGet(_device_ordinal()), "cuDeviceGet")


def _in_graph_capture() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


# --------------------------------------------------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------------------------------------------------
_num_domains_cache: Dict[int, int] = {}
_granularity_cache: Dict[int, int] = {}


def get_num_locality_domains() -> int:
    """``CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT`` of the current device (1 when unsupported); cached per device ordinal."""
    ordinal = _device_ordinal()
    cached = _num_domains_cache.get(ordinal)
    if cached is not None:
        return cached
    try:
        drv = _drv()
        attr = drv.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT
        err, value = drv.cuDeviceGetAttribute(attr, _cu_device())
    except (LocalityDomainError, AttributeError):
        return 1
    count = max(int(value), 1) if err == drv.CUresult.CUDA_SUCCESS else 1
    _num_domains_cache[ordinal] = count
    return count


def is_localization_available() -> bool:
    """True on a compute capability (10, 7) device that reports exactly 2 locality domains (DeepGEMM's condition)."""
    return (
        torch.cuda.is_available()
        and tuple(torch.cuda.get_device_capability()) == (10, 7)
        and get_num_locality_domains() == 2
    )


def get_granularity() -> int:
    """Allocation granularity of locality-domain memory (slices are padded to it; 2 MiB on Rubin); cached per device ordinal."""
    ordinal = _device_ordinal()
    cached = _granularity_cache.get(ordinal)
    if cached is not None:
        return cached
    drv = _drv()
    prop = _alloc_prop(0)
    value = int(_ck(drv.cuMemGetAllocationGranularity(prop, drv.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM),
                    "cuMemGetAllocationGranularity"))  # fmt: skip
    _granularity_cache[ordinal] = value
    return value


_libcuda: Optional[ctypes.CDLL] = None


def get_locality_domain_of(ptr: int) -> int:
    """Locality domain ordinal of the allocation holding the device address ``ptr`` (-1: not localized / not a driver mapping).

    ``cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)`` is called straight through ``libcuda`` (ctypes, like
    DeepGEMM's lazy dlsym): cuda-python 13.4.1's high-level wrapper does not know this attribute's result type."""
    global _libcuda
    if _libcuda is None:
        lib = ctypes.CDLL("libcuda.so.1")
        lib.cuPointerGetAttribute.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_uint64,
        ]
        lib.cuPointerGetAttribute.restype = ctypes.c_int
        _libcuda = lib
    attr = int(_drv().CUpointer_attribute.CU_POINTER_ATTRIBUTE_LOCALITY_DOMAIN_ORDINAL)
    value = ctypes.c_int(-1)
    rc = _libcuda.cuPointerGetAttribute(ctypes.byref(value), attr, int(ptr))
    return int(value.value) if rc == 0 else -1


def _alloc_prop(domain: int):
    drv = _drv()
    prop = drv.CUmemAllocationProp()
    prop.type = drv.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = (
        drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN
    )
    prop.location.localized.deviceId = _device_ordinal()
    prop.location.localized.localityDomainId = domain
    return prop


# --------------------------------------------------------------------------------------------------------------------
# localized allocation
# --------------------------------------------------------------------------------------------------------------------
class _Reservation:
    """One virtual-address reservation with ``num_domains`` mappings, slice ``d`` backed by memory of locality domain ``d``.

    Exports the whole reservation as a flat uint8 CUDA array (``__cuda_array_interface__``); torch tensors built from it keep this
    object alive, and ``__del__`` (the last tensor / view going away) unmaps and frees everything."""

    def __init__(self, num_slice_bytes: int, num_domains: int) -> None:
        drv = _drv()
        self.num_slice_bytes = num_slice_bytes
        self.num_domains = num_domains
        self.num_bytes = num_slice_bytes * num_domains
        self.base = 0
        self.mappings: list = []  # (address, handle)
        self.device = _device_ordinal()
        granularity = get_granularity()
        try:
            self.base = int(
                _ck(
                    drv.cuMemAddressReserve(self.num_bytes, granularity, 0, 0),
                    "cuMemAddressReserve",
                )
            )
            for d in range(num_domains):
                handle = _ck(
                    drv.cuMemCreate(num_slice_bytes, _alloc_prop(d), 0),
                    f"cuMemCreate(domain {d})",
                )
                address = self.base + d * num_slice_bytes
                try:
                    _ck(
                        drv.cuMemMap(address, num_slice_bytes, 0, handle, 0),
                        f"cuMemMap(domain {d})",
                    )
                except Exception:
                    drv.cuMemRelease(handle)
                    raise
                self.mappings.append((address, handle))
            access = drv.CUmemAccessDesc()
            access.location.type = drv.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
            access.location.id = self.device
            access.flags = drv.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            _ck(
                drv.cuMemSetAccess(self.base, self.num_bytes, [access], 1),
                "cuMemSetAccess",
            )
        except Exception:
            self.release()
            raise

    @property
    def __cuda_array_interface__(self) -> Dict[str, Any]:
        return {
            "shape": (self.num_bytes,),
            "typestr": "|u1",
            "data": (self.base, False),
            "version": 3,
            "strides": None,
        }

    def release(self) -> None:
        if not self.base:
            return
        base, self.base = self.base, 0
        drv = _drv()
        try:  # noqa: SIM105 (in-flight kernels may still read the memory)
            torch.cuda.synchronize(self.device)
        except Exception:  # interpreter shutdown / CUDA already torn down: still try to release below  # noqa: S110
            pass
        for address, handle in self.mappings:
            drv.cuMemUnmap(address, self.num_slice_bytes)
            drv.cuMemRelease(handle)
        self.mappings = []
        drv.cuMemAddressFree(base, self.num_bytes)

    def __del__(self) -> None:
        try:  # noqa: SIM105 (no helper objects in a finalizer)
            self.release()
        except Exception:  # never raise from a finalizer  # noqa: S110
            pass


def _contiguous_strides(sizes: Sequence[int]) -> list:
    strides, acc = [], 1
    for s in reversed(list(sizes)):
        strides.append(acc)
        acc *= max(int(s), 1)
    return list(reversed(strides))


def get_slice_stride_bytes(sizes: Sequence[int], element_size: int) -> int:
    """Bytes between slice 0 and slice 1 of ``empty_per_domain(sizes, dtype)``: one slice, padded up to the allocation granularity."""
    numel = 1
    for s in sizes:
        numel *= int(s)
    granularity = get_granularity()
    return -(-max(numel * element_size, 1) // granularity) * granularity


def empty_per_domain(sizes: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
    """``torch.empty((num_domains, *sizes), dtype=dtype)`` with slice ``d`` homed in locality domain ``d``.

    Slices are padded to the allocation granularity, so ``stride(0)`` may exceed ``prod(sizes)`` (each slice is contiguous)."""
    num_domains = get_num_locality_domains()
    if num_domains < 2:
        raise LocalityDomainError("this GPU has a single locality domain")
    sizes = [int(s) for s in sizes]
    elem = torch.empty((), dtype=dtype).element_size()
    numel = 1
    for s in sizes:
        numel *= s
    num_slice_bytes = get_slice_stride_bytes(sizes, elem)
    assert num_slice_bytes % elem == 0
    reservation = _Reservation(num_slice_bytes, num_domains)
    flat = torch.as_tensor(
        reservation, device="cuda"
    )  # uint8, keeps `reservation` alive
    assert (
        flat.data_ptr() == reservation.__cuda_array_interface__["data"][0]
        and flat.numel() == reservation.num_bytes
    )
    typed = flat if dtype == torch.uint8 else flat.view(dtype)
    return torch.as_strided(
        typed,
        [num_domains] + sizes,
        [num_slice_bytes // elem] + _contiguous_strides(sizes),
    )


def is_localized(t: torch.Tensor) -> bool:
    """Whether ``t`` is ``(num_domains, ...)`` with contiguous slices and slice ``d`` homed in locality domain ``d``."""
    num_domains = get_num_locality_domains()
    if num_domains < 2 or t.dim() < 2 or t.size(0) != num_domains:
        return False
    strides = _contiguous_strides([t.size(i) for i in range(1, t.dim())])
    if any(t.stride(i) != strides[i - 1] for i in range(1, t.dim())):
        return False
    base = t.data_ptr()
    elem = t.element_size()
    return all(
        get_locality_domain_of(base + d * t.stride(0) * elem) == d
        for d in range(num_domains)
    )


def is_canonical_localized(t: torch.Tensor) -> bool:
    """:func:`is_localized` AND the slice stride is the one :func:`empty_per_domain` / :func:`localize` produce for this shape
    (slices padded to the granularity).  The kernel bakes that stride into its 3D weight tensor map, so the full-size tensor made by
    :func:`localize` (or DeepGEMM's native allocator, which pads identically) is the only accepted layout; sub-views are not."""
    return is_localized(t) and t.stride(0) * t.element_size() == get_slice_stride_bytes(
        t.shape[1:], t.element_size()
    )


def weights_localized(weights: Any) -> bool:
    """Introspection for the MegaMoE kernel: are these weights die-local?  Accepts a weight tensor or a ``(weights, sf)`` tuple.

    Routed weights are 4D ``(2, E_l, N/2, K)``, shared weights 3D ``(2, N/2, K)``; scale factors are never localized."""
    w = weights[0] if isinstance(weights, (tuple, list)) else weights
    return isinstance(w, torch.Tensor) and w.dim() in (3, 4) and is_localized(w)


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """``(*B, N, K)`` -> ``(2, *B, N / 2, K)``, slice ``d`` homed in locality domain ``d`` (a new allocation, one copy)."""
    num_domains = get_num_locality_domains()
    dim = dim % t.dim()
    if num_domains != 2:
        raise LocalityDomainError(f"{num_domains} locality domains (need exactly 2)")
    if t.size(dim) % num_domains:
        raise ValueError(
            f"`{tuple(t.shape)}` must split into {num_domains} parts along dim {dim}"
        )
    sizes = [*t.shape[:dim], t.size(dim) // num_domains, *t.shape[dim + 1 :]]
    localized = empty_per_domain(sizes, t.dtype)
    localized.copy_(t.unflatten(dim, (num_domains, -1)).movedim(dim, 0))
    assert is_localized(localized)
    return localized


def localize_mega_moe_weights(
    l1: Tuple[torch.Tensor, torch.Tensor],
    l2: Tuple[torch.Tensor, torch.Tensor],
    shared_l1: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    shared_l2: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):
    """Localize transformed MegaMoE ``(weights, sf)`` tuples; the (small) SF tensors stay where they are.

    Localizing the routed pair only is the production recipe (DeepGEMM: ``DG_MEGA_MOE_SM107_PERDIE=1`` with die-local routed weights).  The shared pair can be localized as well (DeepGEMM accepts that), but the port's launch REFUSES die-local shared weights with ``NotImplementedError`` (the kernel's weight tensor maps only implement
    die-local routed weights): keep the shared pair plain.  Call after ``transform_weights_for_mega_moe`` and before the first launch / CUDA graph capture."""

    def loc(pair):
        return None if pair is None else (localize(pair[0]), pair[1])

    return loc(l1), loc(l2), loc(shared_l1), loc(shared_l2)


# --------------------------------------------------------------------------------------------------------------------
# SM -> locality domain map (latency probe)
# --------------------------------------------------------------------------------------------------------------------
def balance_sm_locality_domains(sm_domain: torch.Tensor) -> torch.Tensor:
    """Move only SURPLUS TPCs of the larger domain so that both domains own the same number of TPCs (pairs of SMs)
    (DeepGEMM ``_balance_sm_locality_domains``; the previous prefix-slice version turned a nearly balanced map into 2/206 and hung)."""
    tpcs = sm_domain.view(-1, NUM_SMS_PER_TPC)
    tpc_domain = tpcs.select(1, 0)
    num_tpcs = tpcs.size(0)
    assert num_tpcs % 2 == 0
    larger = int(tpc_domain.sum().item() * 2 > num_tpcs)
    surplus = (tpc_domain == larger).nonzero().flatten()[num_tpcs // 2 :]
    balanced = sm_domain.clone()
    balanced.view(-1, NUM_SMS_PER_TPC).index_fill_(0, surplus, larger ^ 1)
    return balanced


_probe_modules: Dict[
    int, Tuple[Any, Any]
] = {}  # device ordinal -> (CUmodule, CUfunction)
_raw_domains: Dict[int, torch.Tensor] = {}
_balanced_domains: Dict[int, torch.Tensor] = {}
_probe_lock = threading.RLock()


def _probe_function():
    drv = _drv()
    ordinal = _device_ordinal()
    cached = _probe_modules.get(ordinal)
    if cached is not None:
        return cached[1]
    module = _ck(
        drv.cuModuleLoadData(_PROBE_PTX.encode() + b"\0"),
        "cuModuleLoadData(locality probe PTX)",
    )
    func = _ck(
        drv.cuModuleGetFunction(module, _PROBE_KERNEL_NAME), "cuModuleGetFunction"
    )
    _ck(drv.cuFuncSetAttribute(func, drv.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, _PROBE_SMEM_BYTES),
        "cuFuncSetAttribute(max dynamic smem)")  # fmt: skip
    _probe_modules[ordinal] = (module, func)
    return func


def _launch_probe(buf: torch.Tensor, latency: torch.Tensor, num_sms: int) -> None:
    """One block of 32 threads per SM (1 block/SM through the maximal dynamic shared memory); thread 0 chases the chains."""
    drv = _drv()
    func = _probe_function()
    stream = drv.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (
        (ctypes.c_void_p(buf.data_ptr()), ctypes.c_void_p(latency.data_ptr())),
        (ctypes.c_void_p, ctypes.c_void_p),
    )
    _ck(
        drv.cuLaunchKernel(
            func, num_sms, 1, 1, 32, 1, 1, _PROBE_SMEM_BYTES, stream, args, 0
        ),
        "cuLaunchKernel(locality probe)",
    )


def get_sm_locality_domains() -> torch.Tensor:
    """Raw per-SM locality domain (``uint8[num_sms]``, indexed by ``%smid``) from the latency probe, before balancing (cached)."""
    ordinal = _device_ordinal()
    with _probe_lock:
        cached = _raw_domains.get(ordinal)
        if cached is not None:
            return cached
        if _in_graph_capture():
            raise LocalityDomainError(
                "get_sm_locality_domains() allocates and synchronizes: call it before CUDA graph capture"
            )
        if not is_localization_available():
            raise LocalityDomainError(
                "SM locality probing needs a compute capability (10, 7) device with 2 locality domains"
            )
        num_sms = torch.cuda.get_device_properties(ordinal).multi_processor_count
        buffer = empty_per_domain(
            [num_sms, _PROBE_CHUNKS_PER_SM, _PROBE_CHUNK_BYTES], torch.uint8
        )
        # chain[line][0] = element index (in 32-bit words) of the next line: stride 7 lines, a full cycle over the 32 lines of a chunk
        lines = _PROBE_CHUNK_BYTES // _PROBE_LINE_BYTES
        chain = torch.zeros(
            (lines, _PROBE_LINE_BYTES // 4), dtype=torch.int32, device="cuda"
        )
        chain[:, 0] = (
            (
                torch.arange(lines, dtype=torch.int32, device="cuda")
                + _PROBE_CHAIN_LINE_STRIDE
            )
            % lines
            * (_PROBE_LINE_BYTES // 4)
        )
        domains = None
        for _ in range(_PROBE_MAX_ATTEMPTS):
            for d in range(2):
                buffer[d].view(torch.int32).view(-1, _PROBE_CHUNK_BYTES // 4).copy_(
                    chain.flatten()
                )
            latency = torch.zeros(
                (2, num_sms, _PROBE_CHUNKS_PER_SM), dtype=torch.int16, device="cuda"
            )
            torch.zeros(256 << 20, dtype=torch.uint8, device="cuda")  # flush L2
            for d in range(2):
                _launch_probe(buffer[d], latency[d], num_sms)
            torch.cuda.current_stream().synchronize()
            if not bool(
                latency.gt(0).all().item()
            ):  # some SM did not run its block (e.g. another kernel holds SMs): retry
                continue
            cycles, order = latency.float().median(-1).values.sort(0)
            candidate = order[0].to(
                torch.uint8
            )  # the domain with the LOWER latency is the SM's own
            tpcs = candidate.view(-1, NUM_SMS_PER_TPC)
            if bool(
                (cycles[1] >= cycles[0] * _PROBE_MIN_FAR_NEAR_RATIO).all().item()
            ) and torch.equal(tpcs, tpcs[:, :1].expand_as(tpcs)):
                domains = candidate
                break
        if domains is None:
            warnings.warn(
                "SM107 locality probe found no stable mapping; using an even mapping",
                RuntimeWarning,
                stacklevel=2,
            )
            domains = (
                (torch.arange(num_sms // NUM_SMS_PER_TPC, device="cuda") % 2)
                .to(torch.uint8)
                .repeat_interleave(NUM_SMS_PER_TPC)
            )
        _raw_domains[ordinal] = domains
        return domains


def get_balanced_sm_locality_domains() -> torch.Tensor:
    """Per-SM locality domain (``uint8[num_sms]`` on the device, indexed by ``%smid``, balanced to equal TPC counts), probed once and
    cached.  Call it OUTSIDE CUDA graph capture (it allocates, launches and synchronizes); the per-die kernel launch reuses the cached
    tensor (its pointer is the ``sm_locality_domains`` kernel argument)."""
    ordinal = _device_ordinal()
    with _probe_lock:
        cached = _balanced_domains.get(ordinal)
        if cached is None:
            cached = balance_sm_locality_domains(get_sm_locality_domains())
            _balanced_domains[ordinal] = cached
        return cached


def has_balanced_sm_locality_domains() -> bool:
    """Whether the map was already probed (i.e. :func:`get_balanced_sm_locality_domains` is safe to call during capture)."""
    return _device_ordinal() in _balanced_domains


def destroy() -> None:
    """Forget the cached SM map and unload the probe module of the current device.  Localized tensors are independent: each keeps its own
    reservation alive and releases it when the last view is garbage collected."""
    ordinal = _device_ordinal()
    with _probe_lock:
        _raw_domains.pop(ordinal, None)
        _balanced_domains.pop(ordinal, None)
        cached = _probe_modules.pop(ordinal, None)
        if cached is not None:
            _drv().cuModuleUnload(cached[0])
