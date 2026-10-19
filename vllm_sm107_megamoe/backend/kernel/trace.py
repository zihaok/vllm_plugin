# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / kTrace instrumentation.
"""DeepGEMM-compatible in-kernel phase trace (``kTrace``), compile-time optional; never part of a performance build (DESIGN.md section 14).

Buffer: ``num_sms * (64 + 4*64)`` u64 (zeroed by the caller). Per CTA ``c``: summary slots at ``trace[c*64 + slot]``; k-block event log (clock64) at
``trace[num_sms*64 + c*4*64 + ev*64 + (g - EVENT_START)]``.  Slot semantics are identical to ``sm107_fp8_fp4_mega_moe.cuh`` (K:116-139):

=== set slots: globaltimer ns (``stamp``) ===
 0 start after the PDL wait (thread 0)      7 after NVLink barrier #1 (dispatch thread 0)   1 pull done (dispatch thread 0)
 2 first MMA task starts (MMA lane 0)       3 MMA done (MMA lane 0)                          4 epilogue GEMM done (epilogue warp 0 lane 0)
 5 combine start (after the epilogue grid sync)                                                6 combine end (+ 61 clock64)         60 clock64 at start
=== accumulate slots: clock64 cycles summed in registers, flushed by ``atom.add`` once at kernel end (``add``) ===
 8 MMA wait tmem_empty   9 MMA wait full   11 MMA tasks (count)   12 MMA get_next_task   16 TMA-A wait empty   17 TMA-A wait tokens / L2 dependency
 18 TMA-A get_next_task  19 TMA-A wait shared L2 count   20 TMA-B wait empty   21 TMA-B get_next_task   24 epilogue wait tmem_full
 25 epilogue L1 task time  26 epilogue L2 task time  27 epilogue wait l2_empty  28 L1 task count  29 L2 task count  31 epilogue get_next_task
 34 dispatch wait l1_empty   40 combine peer wait
=== events (clock64; window starts at k-block EVENT_START of each role's running k-block count) ===
 0 TMA-B issue   1 MMA full seen   2 MMA commit   3 TMA-A issue
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass import Uint64

from . import ptx
from .dslutil import DslStatic, init_fields, static_dataclass

NUM_SLOTS = 64
NUM_EVENTS = 64
EVENT_START = 400

# set slots
SLOT_START = 0
SLOT_PULL_DONE = 1
SLOT_MMA_FIRST = 2
SLOT_MMA_DONE = 3
SLOT_EPI_DONE = 4
SLOT_COMBINE_START = 5
SLOT_COMBINE_END = 6
SLOT_NVL1_DONE = 7
SLOT_CLOCK_START = 60
SLOT_CLOCK_END = 61
# accumulate slots
ACC_MMA_WAIT_TMEM_EMPTY = 8
ACC_MMA_WAIT_FULL = 9
ACC_MMA_TASKS = 11
ACC_MMA_NEXT_TASK = 12
ACC_TMAA_WAIT_EMPTY = 16
ACC_TMAA_WAIT_TOKENS = 17
ACC_TMAA_NEXT_TASK = 18
ACC_TMAA_WAIT_SHARED_L2 = 19
ACC_TMAB_WAIT_EMPTY = 20
ACC_TMAB_NEXT_TASK = 21
ACC_EPI_WAIT_TMEM_FULL = 24
ACC_EPI_L1_TIME = 25
ACC_EPI_L2_TIME = 26
ACC_EPI_WAIT_L2_EMPTY = 27
ACC_EPI_L1_COUNT = 28
ACC_EPI_L2_COUNT = 29
ACC_EPI_NEXT_TASK = 31
ACC_DISPATCH_WAIT_L1_EMPTY = 34
ACC_COMBINE_PEER_WAIT = 40
# events
EV_TMAB_ISSUE = 0
EV_MMA_FULL_SEEN = 1
EV_MMA_COMMIT = 2
EV_TMAA_ISSUE = 3


def trace_buffer_u64s(num_sms: int) -> int:
    """Number of u64 words the caller must provide (zeroed)."""
    return num_sms * (NUM_SLOTS + 4 * NUM_EVENTS)


@static_dataclass
class Trace(DslStatic):
    """Per-thread trace handle (frozen dataclass: usable inside staged regions; the accumulator TENSOR is mutated in place).

    All methods compile to nothing when ``enabled`` is False (``cfg.trace``)."""

    def __init__(self, cfg, trace_ptr, sm_idx):
        acc = None
        if bool(cfg.trace):
            acc = cute.make_rmem_tensor((NUM_SLOTS,), Uint64)
            acc.fill(0)
        init_fields(self, enabled=bool(cfg.trace), num_sms=cfg.num_sms, trace_ptr=trace_ptr, sm_idx=sm_idx, acc=acc)

    # -- internal address helpers
    def _slot_ptr(self, slot):
        return self.trace_ptr + (self.sm_idx * NUM_SLOTS + slot)

    def _event_ptr(self, ev, g):
        return self.trace_ptr + (self.num_sms * NUM_SLOTS + self.sm_idx * 4 * NUM_EVENTS + ev * NUM_EVENTS + (g - EVENT_START))

    # -- API (DESIGN.md section 14)
    def now(self):
        return ptx.globaltimer()

    @cute.jit
    def stamp(self, slot, pred=True):
        """``trace[cta][slot] = globaltimer`` (call from ONE thread of the role)."""
        if cutlass.const_expr(self.enabled):
            if pred:
                cute.arch.store(self._slot_ptr(slot), self.now())

    @cute.jit
    def stamp_clock(self, slot, pred=True):
        if cutlass.const_expr(self.enabled):
            if pred:
                cute.arch.store(self._slot_ptr(slot), Uint64(ptx.clock64()))

    def clock(self):
        """clock64 value to pass to :meth:`add` later (0 when disabled so that it folds away)."""
        if cutlass.const_expr(self.enabled):
            return Uint64(ptx.clock64())
        return Uint64(0)

    def add(self, slot, value):
        """``acc[slot] += value`` (register accumulation; ``slot`` is a Python int)."""
        if cutlass.const_expr(self.enabled):
            self.acc[slot] = self.acc[slot] + Uint64(value)

    def add_since(self, slot, t0):
        """``acc[slot] += clock64 - t0`` where ``t0 = tr.clock()`` taken before the timed region."""
        if cutlass.const_expr(self.enabled):
            self.acc[slot] = self.acc[slot] + (Uint64(ptx.clock64()) - t0)

    @cute.jit
    def event(self, ev, g):
        """Log ``clock64`` into the event window when ``EVENT_START <= g < EVENT_START + NUM_EVENTS`` (``g`` runtime Int32)."""
        if cutlass.const_expr(self.enabled):
            if g >= EVENT_START:
                if g < EVENT_START + NUM_EVENTS:
                    cute.arch.store(self._event_ptr(ev, g), Uint64(ptx.clock64()))

    @cute.jit
    def flush(self):
        """End of kernel (all threads): ``atom.add`` of every non-zero accumulator into the CTA's slots."""
        if cutlass.const_expr(self.enabled):
            for i in cutlass.range_constexpr(NUM_SLOTS):
                v = self.acc[i]
                if v != Uint64(0):
                    cute.arch.atomic_add(self._slot_ptr(i), v)
