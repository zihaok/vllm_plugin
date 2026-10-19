# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / comm/barrier.cuh + layout/sym_buffer.cuh; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION & AFFILIATES).
"""Symmetric-memory peer table and the software grid / NVLink barriers of the SM107 DeepGEMM-style MegaMoE kernel.

* :class:`SymBufferHost` (host payload, JIT-arg adapter) -> ``make_device_object()`` -> :class:`SymBuffer` (ONE kernel
  argument carrying ``base`` (Int64), ``rank_idx`` (Int32) and the by-value peer-offset table (``llvm.byval`` +
  ``cute_nvgpu.grid_constant`` struct of ``max_ranks`` Int64; DSL >= 4.7 allows any width, so DEP32 / 72 ranks work).
  Adapted from the vendored ``communication/nvlink_domain/symmetric_buffer.py`` (BSD-3-Clause).
* :func:`grid_sync` / :func:`nvlink_barrier`: DeepGEMM's exact counter protocol (``comm/barrier.cuh``).

DSL notes (DESIGN.md section 4.1): the helpers below contain no ``break/continue/return``; spin loops are predicate-carried
``while`` loops.  ``sync_scope`` should be a STATIC descriptor ``(barrier_id, num_threads)`` (CTA-local aligned named
barrier, e.g. ``(0, 128)`` for the dispatch warps, ``(2, 256)`` for the epilogue warps): closures that capture runtime
values raise ``SCOPE_CLOSURE_CAPTURE`` inside staged ``if`` regions.  A zero-argument callable is also accepted.
"""

from dataclasses import dataclass
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm
from cutlass.base_dsl.dsl import extract_mlir_values, get_mlir_types, new_from_mlir_values
from cutlass.base_dsl.runtime.jit_arg_adapters import JitArgAdapterRegistry
from cutlass.base_dsl.typing import get_c_pointers
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, dsl_user_op

from . import ptx

try:  # pragma: no cover - symbol location differs between DSL builds
    from cutlass.base_dsl.typing import MLIR_DYNAMIC_INDEX
except ImportError:  # pragma: no cover
    MLIR_DYNAMIC_INDEX = -(2**31)

#: DeepGEMM ``layout::kNumMaxRanks`` (size of the peer arrays of the signals; the by-value table may be narrower).
MAX_RANKS = 72
#: ``kFinishSumTag`` of the software grid sync.
GRID_SYNC_FINISH_TAG = 0x80000000
#: 300 s at 2 GHz (``comm::kNumTimeoutCycles``).
TIMEOUT_CYCLES = 300 * 2_000_000_000


def _byval_struct_type(rank_count: int):
    return ir.Type.parse(f"!llvm.struct<(array<{rank_count} x i64>)>")


# ---------------------------------------------------------------------------------------------------------------------
# device object (kernel argument)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SymBuffer:
    """Device-side symmetric-buffer view: ``base``, ``rank_idx`` and the by-value peer-offset table.

    Mirrors DeepGEMM ``layout::SymBuffer<kNumRanks>`` (``base``, ``offsets[]``, ``rank_idx``); ``max_ranks`` plays the role of
    ``kNumRanks`` (compile-time).  ``map(x, dst_rank)`` adds ``offsets[dst_rank]`` (identity when ``max_ranks == 1``).
    """

    base: Any  # Int64: local base address of the symmetric buffer
    rank_idx: Any  # Int32: own rank
    table: Any  # ir.Value: !llvm.ptr to the by-value struct (array<max_ranks x i64>)
    max_ranks: cutlass.Constexpr[int]

    # ---- DSL dynamic-value protocol (one kernel argument, three MLIR values)
    def __extract_mlir_values__(self) -> list:
        return [Int64(self.base).ir_value(), Int32(self.rank_idx).ir_value(), self.table]

    def __new_from_mlir_values__(self, values: list) -> "SymBuffer":
        return SymBuffer(Int64(values[0]), Int32(values[1]), values[2], self.max_ranks)

    def __get_mlir_types__(self) -> list:
        return [ir.Type.parse("i64"), ir.Type.parse("i32"), ir.Type.parse("!llvm.ptr")]

    def __extract_mlir_attributes__(self) -> list:
        return [
            ir.DictAttr.get({}),
            ir.DictAttr.get({}),
            ir.DictAttr.get(
                {
                    "cute_nvgpu.grid_constant": ir.UnitAttr.get(),
                    "llvm.byval": ir.TypeAttr.get(_byval_struct_type(self.max_ranks)),
                }
            ),
        ]

    # ---- peer mapping
    @dsl_user_op
    def offset_of(self, dst_rank, *, loc=None, ip=None) -> Int64:
        """``offsets[dst_rank]`` (Int64) loaded from the by-value table."""
        i64_type = ir.Type.parse("i64")
        offset_pointer = llvm.getelementptr(
            ir.Type.parse("!llvm.ptr"),
            self.table,
            [Int32(dst_rank).ir_value(loc=loc, ip=ip)],
            [MLIR_DYNAMIC_INDEX],
            i64_type,
            no_wrap_flags="None",
            loc=loc,
            ip=ip,
        )
        return Int64(llvm.load(i64_type, offset_pointer, loc=loc, ip=ip))

    @dsl_user_op
    def map(self, x, dst_rank, *, loc=None, ip=None):
        """Map a LOCAL symmetric address to the same location in rank ``dst_rank``'s buffer (DeepGEMM ``SymBuffer::map``).

        ``x``: gmem/generic ``cute.Pointer`` or ``Int64``/int address.  Returns the same kind (a ``cute.Pointer`` of the same
        dtype/memspace/alignment, or ``Int64``).  Identity (no table load) when ``max_ranks == 1``.
        """
        if self.max_ranks == 1:
            return x
        if isinstance(x, cute.Pointer):
            if x.memspace not in (AddressSpace.gmem, AddressSpace.generic):
                raise ValueError("only gmem pointers can be mapped to a symmetric peer")
            mapped = x.toint(loc=loc, ip=ip) + self.offset_of(dst_rank, loc=loc, ip=ip)
            return cute.make_ptr(x.dtype, mapped, x.memspace, assumed_align=x.max_alignment, loc=loc, ip=ip)
        return Int64(x) + self.offset_of(dst_rank, loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------------------------------
# host payload (JIT argument)
# ---------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class SymBufferHost:
    """Host launch payload: ``base_address`` of the local buffer, ``offsets[i] = peer_base[i] - local_base`` (``max_ranks`` entries,
    ``offsets[rank] == 0``) and the own ``rank``.  Plain Python ints are accepted for every field."""

    base_address: Any  # Int64 / int
    offsets: tuple  # tuple of Int64 / int, length == max_ranks
    rank: Any  # Int32 / int
    max_ranks: cutlass.Constexpr[int]

    @staticmethod
    def _as_int64(value) -> Int64:
        return value if isinstance(value, Int64) else Int64(int(value))

    @staticmethod
    def _as_int32(value) -> Int32:
        return value if isinstance(value, Int32) else Int32(int(value))

    @dsl_user_op
    def make_device_object(self, *, loc=None, ip=None) -> SymBuffer:
        """Build the device object INSIDE the host ``@cute.jit`` function (alloca'd by-value table filled with the offsets)."""
        offsets = tuple(self.offsets)
        if len(offsets) != self.max_ranks:
            raise ValueError(f"Expected {self.max_ranks} peer offsets, got {len(offsets)}.")
        pointer_type = ir.Type.parse("!llvm.ptr")
        struct_type = _byval_struct_type(self.max_ranks)
        i64_type = ir.Type.parse("i64")
        one = arith.constant(value=ir.IntegerAttr.get(i64_type, 1), result=i64_type, loc=loc, ip=ip)
        buffer = llvm.alloca(res=pointer_type, elem_type=struct_type, array_size=one, alignment=64, loc=loc, ip=ip)
        for index, offset in enumerate(offsets):
            slot = llvm.getelementptr(pointer_type, buffer, [], [index], i64_type, no_wrap_flags="None", loc=loc, ip=ip)
            llvm.store(self._as_int64(offset).ir_value(loc=loc, ip=ip), slot, loc=loc, ip=ip)
        return SymBuffer(self._as_int64(self.base_address), self._as_int32(self.rank), buffer, self.max_ranks)


@JitArgAdapterRegistry.register_jit_arg_adapter(SymBufferHost)
class _SymBufferHostAdapter:
    """Expands a :class:`SymBufferHost` into ``[base: i64, offsets: i64 x R, rank: i32]`` scalar host arguments."""

    def __init__(self, argument: SymBufferHost) -> None:
        self._argument = argument
        offsets = tuple(argument.offsets)
        if len(offsets) != int(argument.max_ranks):
            raise ValueError(f"Expected {int(argument.max_ranks)} peer offsets, got {len(offsets)}.")
        self._fields = (
            Int64(argument.base_address),
            *(Int64(offset) for offset in offsets),
            Int32(argument.rank),
        )

    def __c_pointers__(self) -> list:
        pointers: list = []
        for field in self._fields:
            pointers.extend(get_c_pointers(field))
        return pointers

    def __get_mlir_types__(self) -> list:
        types: list = []
        for field in self._fields:
            types.extend(get_mlir_types(field))
        return types

    def __extract_mlir_values__(self) -> list:
        values: list = []
        for field in self._fields:
            values.extend(extract_mlir_values(field))
        return values

    def __new_from_mlir_values__(self, values: list) -> SymBufferHost:
        value_index = 0
        rebuilt = []
        for field in self._fields:
            n = len(get_mlir_types(field))
            rebuilt.append(new_from_mlir_values(field, values[value_index : value_index + n]))
            value_index += n
        if value_index != len(values):
            raise ValueError(f"Consumed {value_index} MLIR values, got {len(values)}.")
        result = object.__new__(SymBufferHost)
        object.__setattr__(result, "base_address", rebuilt[0])
        object.__setattr__(result, "offsets", tuple(rebuilt[1:-1]))
        object.__setattr__(result, "rank", rebuilt[-1])
        object.__setattr__(result, "max_ranks", self._argument.max_ranks)
        return result


def make_sym_buffer_host(base_address: int, peer_bases, rank: int) -> SymBufferHost:
    """Host convenience: ``peer_bases[i]`` = base address of rank i's buffer mapped into THIS process (``handle.buffer_ptrs``);
    offsets are relative to the local base (``peer_bases[rank]``)."""
    peer_bases = [int(p) for p in peer_bases]
    local = peer_bases[rank]
    assert int(base_address) == local, "base_address must be the local entry of peer_bases"
    return SymBufferHost(int(base_address), tuple(p - local for p in peer_bases), int(rank), len(peer_bases))


# ---------------------------------------------------------------------------------------------------------------------
# software grid sync / NVLink barrier (DeepGEMM comm/barrier.cuh)
# ---------------------------------------------------------------------------------------------------------------------
def _scope_sync(sync_scope):
    """CTA-level rendezvous: ``(barrier_id, num_threads)`` -> ``bar.sync`` (preferred) or a zero-argument callable."""
    if isinstance(sync_scope, tuple):
        ptx.bar_sync(sync_scope[0], sync_scope[1])
    else:
        sync_scope()


@cute.jit
def _spin_grid_sync(count_addr: Int64, old: Int32):
    """DeepGEMM ``comm::wait_until(pred)``: ``while (!pred()) { if (clock64() - start >= timeout) trap; }`` with
    ``pred = ((ld.acquire.gpu(count) ^ old) & 0x80000000) != 0`` (300 s timeout -> trap, no printf).  The predicate is evaluated first, the
    clock only on the not-done path (as in DeepGEMM's SASS: LDG.E.STRONG.GPU ; CCTL.IVALL ; YIELD ; LOP3 ; ISETP ; branch ; clock check)."""
    start = ptx.clock64()
    cur = ptx.ld_acquire_gpu_b32(count_addr)
    while (cur ^ old) >= Int32(0):  # sign bit of (cur ^ old) == kFinishSumTag not yet set
        if (ptx.clock64() - start) >= Int64(TIMEOUT_CYCLES):
            ptx.trap()
        cur = ptx.ld_acquire_gpu_b32(count_addr)


@cute.jit
def _spin_nvl_signal(signal_addr: Int64, target: Int32):
    """``wait_until([&]() { return ptx::ld_acq_sys(signal_ptr) == target; })`` (300 s timeout -> trap)."""
    start = ptx.clock64()
    cur = ptx.ld_acquire_sys_s32(signal_addr)
    while cur != target:
        if (ptx.clock64() - start) >= Int64(TIMEOUT_CYCLES):
            ptx.trap()
        cur = ptx.ld_acquire_sys_s32(signal_addr)


@cute.jit
def grid_sync(ws, idx: int, num_sms: int, sm_idx, thread_idx_in_group, sync_scope):
    """DeepGEMM ``comm::grid_sync<kNumSMs, idx>``: all ``num_sms`` CTAs must call it (full residency).

    ``ws.grid_sync_count(idx)`` -> gmem address of the counter. ``sync_scope``: CTA rendezvous (see module doc).
    Protocol: ``sync_scope(); if thread==0: old = atom.release.gpu.add(counter, sm==0 ? 0x80000000-(num_sms-1) : 1);
    spin ((ld.acquire.gpu(counter) ^ old) & 0x80000000) != 0; sync_scope()``.
    """
    _scope_sync(sync_scope)
    if thread_idx_in_group == Int32(0):
        count_addr = ws.grid_sync_count(idx)
        delta = Int32(cutlass.select_(sm_idx == Int32(0), Int32(GRID_SYNC_FINISH_TAG - (num_sms - 1)), Int32(1)))
        old = ptx.atom_add_release_gpu_u32(count_addr, delta)
        _spin_grid_sync(ptx_addr(count_addr), old)
    _scope_sync(sync_scope)


def ptx_addr(x) -> Int64:
    """Generic address (Int64) of a gmem pointer / address value."""
    if isinstance(x, cute.Pointer):
        return x.toint()
    return Int64(x)


@cute.jit
def nvlink_barrier(
    ws,
    sym: SymBuffer,
    num_ranks: int,
    num_sms: int,
    num_threads: int,
    grid_idx: int,
    tag: int,
    sm_idx,
    thread_idx,
    sync_scope,
    prologue: bool = True,
    epilogue: bool = True,
):
    """DeepGEMM ``comm::nvlink_barrier<kNumRanks, kNumSMs, kNumThreads, grid_idx, tag>``.

    ``ws.nvl_barrier_counter()`` / ``ws.nvl_barrier_signal(phase)`` -> gmem addresses (local); remote signals are reached
    with ``sym.map``.  Only SM 0 talks to the peers; ``prologue``/``epilogue`` select the surrounding grid syncs
    (barrier #1 uses ``(False, True)``, barrier #2 ``(True, False)``).  ``tag`` is diagnostic only (DeepGEMM prints it on timeout).
    """
    assert num_ranks <= num_threads, "Insufficient threads"
    if cutlass.const_expr(prologue):
        grid_sync(ws, grid_idx, num_sms, sm_idx, thread_idx, sync_scope)
    if sm_idx == Int32(0):
        counter_addr = ptx_addr(ws.nvl_barrier_counter())
        status = ptx.ld_b32(counter_addr) & Int32(3)
        signal_phase = status & Int32(1)
        signal_sign = status >> Int32(1)
        signal_addr = ptx_addr(ws.nvl_barrier_signal(signal_phase))
        # Send signals to remote ranks
        if thread_idx < Int32(num_ranks):
            delta = Int32(cutlass.select_(signal_sign != Int32(0), Int32(-1), Int32(1)))
            ptx.red_add_release_sys_s32(sym.map(signal_addr, thread_idx), delta)
        _scope_sync(sync_scope)
        # Update status and wait arrival
        if thread_idx == Int32(0):
            ptx.red_add_u32(counter_addr, Int32(1))
            target = Int32(cutlass.select_(signal_sign != Int32(0), Int32(0), Int32(num_ranks)))
            _spin_nvl_signal(signal_addr, target)
    if cutlass.const_expr(epilogue):
        grid_sync(ws, grid_idx, num_sms, sm_idx, thread_idx, sync_scope)
