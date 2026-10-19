# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / ptx/{ld_st,utils,tma}.cuh + comm/barrier.cuh; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION & AFFILIATES).
"""Inline-PTX helpers of the dispatch / combine roles (dispatch / NVLink protocol / combine / cleanup).

Every wrapper is ONE PTX instruction (or one tight spin loop) emitted with ``llvm.inline_asm(has_side_effects=True)``,
spelled exactly like DeepGEMM's ``asm volatile`` helpers (``ptx/ld_st.cuh``, ``ptx/utils.cuh``, ``ptx/tma.cuh``,
``comm/barrier.cuh``) so that the SASS qualifiers (``.STRONG.GPU``, ``.STRONG.SYS``, ``ATOMG.E.ADD.64``, ``REDG``,
generic ``LD.E.128`` for remote pulls ...) can be compared 1:1 with DeepGEMM's cubin.

Names follow DESIGN.md section 13 (``ptx.py``) wherever a function exists there; this module carries
(a) the forms ``ptx.py`` does not provide (generic-address remote loads/stores: DeepGEMM's ``*sym_buffer.map(p, r)``
dereferences a GENERIC pointer and compiles to ``LD.E.128`` / ``ST.E``, not ``LDG`` / ``STG``), (b) timeout spin loops
with DeepGEMM's 300 s ``clock64`` budget, and (c) a self-contained ``grid_sync`` / ``nvlink_barrier`` used until
``comm.py`` is available (same counter protocol, same PTX).  ``ga`` operands are ``cute.Pointer`` (gmem) or ``Int64``.
"""

from __future__ import annotations


from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T, dsl_user_op

# DeepGEMM comm::kNumTimeoutCycles = 300 s at 2 GHz (comm/barrier.cuh:14). A MODULE GLOBAL read at trace time by the asm builders below: `jit.compile_kernel_object` sets it from
# `KernelBuild.timeout_sec` for the duration of each compile (part of the compile-cache key); never capture it in default arguments / module-level constants.
NUM_TIMEOUT_CYCLES = 300 * 2_000_000_000
EVICT_NORMAL = 0x1000000000000000  # cute::TMA::CacheHintSm100::EVICT_NORMAL
EVICT_FIRST = 0x12F0000000000000  # cute::TMA::CacheHintSm100::EVICT_FIRST


def _asm(res_ty, operands, text, constraints, loc=None, ip=None, side=True):
    return llvm.inline_asm(
        res_ty, operands, text, constraints,
        has_side_effects=side, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)  # fmt: skip


def _a(x, loc=None, ip=None):
    """gmem address operand (cute.Pointer | Int64 | ir.Value) -> i64 ir.Value."""
    if isinstance(x, Int64):
        return x.ir_value(loc=loc, ip=ip)
    if isinstance(x, ir.Value):
        return x
    return x.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


def _v(x, loc=None, ip=None):
    return x if isinstance(x, ir.Value) else x.ir_value(loc=loc, ip=ip)


def _sm(x, loc=None, ip=None):
    """smem pointer (cute.Pointer in smem) -> i32 shared-window address ir.Value."""
    return x.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)


# ----------------------------------------------------------------------------------------------------------------
# identity registers
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def gridid(*, loc=None, ip=None) -> Int64:
    """``mov.u64 %gridid`` (DeepGEMM ``ptx::get_grid_idx``, volatile)."""
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %gridid;", "=l", loc, ip))


@dsl_user_op
def clock64(*, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc, ip))


# ----------------------------------------------------------------------------------------------------------------
# LOCAL global-memory loads / stores / atomics (workspace, rings)
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def ld_b32(ga, *, loc=None, ip=None) -> Int32:
    return Int32(_asm(T.i32(), [_a(ga, loc, ip)], "ld.global.u32 $0, [$1];", "=r,l", loc, ip, side=False))


@dsl_user_op
def ld_b32_ordered(ga, *, loc=None, ip=None) -> Int32:
    """Plain ``ld.global.u32`` that must not be CSE'd / hoisted by the compiler (barrier phase counter read by SM 0 on
    every NVLink barrier: ``(*counter_ptr) & 3``); volatile asm flag, PTX instruction unchanged."""
    return Int32(_asm(T.i32(), [_a(ga, loc, ip)], "ld.global.u32 $0, [$1];", "=r,l", loc, ip, side=True))


@dsl_user_op
def ld_b64(ga, *, loc=None, ip=None) -> Int64:
    return Int64(_asm(T.i64(), [_a(ga, loc, ip)], "ld.global.u64 $0, [$1];", "=l,l", loc, ip, side=False))


@dsl_user_op
def ld_volatile_b64(ga, *, loc=None, ip=None) -> Int64:
    """``ptx::ld_volatile(const uint64_t*)``: ``ld.volatile.global.b64``."""
    return Int64(_asm(T.i64(), [_a(ga, loc, ip)], "ld.volatile.global.b64 $0, [$1];", "=l,l", loc, ip))


@dsl_user_op
def ld_acquire_gpu_b32(ga, *, loc=None, ip=None) -> Int32:
    """``ptx::ld_acq``: ``ld.acquire.gpu.global.b32`` (+ memory clobber)."""
    return Int32(_asm(T.i32(), [_a(ga, loc, ip)], "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", loc, ip))


@dsl_user_op
def ld_acquire_sys_b64(ga, *, loc=None, ip=None) -> Int64:
    """``ptx::ld_acq_sys(const uint64_t*)``: ``ld.acquire.sys.global.b64``."""
    return Int64(_asm(T.i64(), [_a(ga, loc, ip)], "ld.acquire.sys.global.b64 $0, [$1];", "=l,l,~{memory}", loc, ip))


@dsl_user_op
def ld_acquire_sys_s32(ga, *, loc=None, ip=None) -> Int32:
    """``ptx::ld_acq_sys(const int*)``: ``ld.acquire.sys.global.s32`` (no memory clobber in DeepGEMM)."""
    return Int32(_asm(T.i32(), [_a(ga, loc, ip)], "ld.acquire.sys.global.s32 $0, [$1];", "=r,l", loc, ip))


@dsl_user_op
def st_b32(ga, v, *, loc=None, ip=None) -> None:
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "st.global.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def st_b64(ga, v, *, loc=None, ip=None) -> None:
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "st.global.u64 [$0], $1;", "l,l", loc, ip)


@dsl_user_op
def st_v4_b32(ga, x, y, z, w, *, loc=None, ip=None) -> None:
    _asm(None, [_a(ga, loc, ip)] + [_v(t, loc, ip) for t in (x, y, z, w)],
         "st.global.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)  # fmt: skip


@dsl_user_op
def st_release_sys_b64(ga, v, *, loc=None, ip=None) -> None:
    """``ptx::st_rel_sys``: ``st.release.sys.global.u64``."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "st.release.sys.global.u64 [$0], $1;", "l,l,~{memory}", loc, ip)


@dsl_user_op
def atom_add_u64(ga, v, *, loc=None, ip=None) -> Int64:
    """``ptx::atomic_add(uint64_t*)``: ``atom.global.add.u64`` (relaxed), returns old (SASS ``ATOMG.E.ADD.64.STRONG.GPU``)."""
    return Int64(_asm(T.i64(), [_a(ga, loc, ip), _v(v, loc, ip)], "atom.global.add.u64 $0, [$1], $2;", "=l,l,l", loc, ip))


@dsl_user_op
def atom_add_sys_u64(ga, v, *, loc=None, ip=None) -> Int64:
    """``ptx::atomic_add_sys``: ``atom.sys.global.add.u64`` (remote ``expert_recv_count_sum`` publish)."""
    return Int64(_asm(T.i64(), [_a(ga, loc, ip), _v(v, loc, ip)], "atom.sys.global.add.u64 $0, [$1], $2;", "=l,l,l", loc, ip))


@dsl_user_op
def atom_add_release_gpu_u32(ga, v, *, loc=None, ip=None) -> Int32:
    """``ptx::atomic_add_rel``: ``atom.release.gpu.global.add.u32`` (grid sync arrive)."""
    return Int32(_asm(T.i32(), [_a(ga, loc, ip), _v(v, loc, ip)],
                      "atom.release.gpu.global.add.u32 $0, [$1], $2;", "=r,l,r,~{memory}", loc, ip))  # fmt: skip


@dsl_user_op
def red_add_u32(ga, v, *, loc=None, ip=None) -> None:
    """``ptx::red_add(uint32_t*)``: ``red.gpu.global.add.u32`` (relaxed)."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "red.gpu.global.add.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def red_add_s32(ga, v, *, loc=None, ip=None) -> None:
    """``ptx::red_add(int*)``: ``red.gpu.global.add.s32`` (cumulative expert stats)."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "red.gpu.global.add.s32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def red_add_release_gpu_u32(ga, v, *, loc=None, ip=None) -> None:
    """``ptx::red_add_rel``: ``red.release.gpu.global.add.u32`` (NO memory clobber in DeepGEMM)."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "red.release.gpu.global.add.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def red_add_release_sys_s32(ga, v, *, loc=None, ip=None) -> None:
    """``ptx::red_add_rel_sys``: ``red.release.sys.global.add.s32`` (NVLink barrier signal)."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "red.release.sys.global.add.s32 [$0], $1;", "l,r", loc, ip)


# ----------------------------------------------------------------------------------------------------------------
# REMOTE (peer-mapped) accesses: generic-address forms (DeepGEMM dereferences the generic pointer returned by
# SymBuffer::map: SASS LD.E.128 / LD.E / ST.E / ST.E.64 / ST.E.128).
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def ld_generic_v4_b32(ga, *, loc=None, ip=None):
    """16 B load through a generic address (remote token pull, K:660-661): 4 x Int32."""
    res = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_a(ga, loc, ip)],
               "ld.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,l", loc, ip, side=False)  # fmt: skip
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def ld_generic_b32(ga, *, loc=None, ip=None) -> Int32:
    """4 B load through a generic address (remote SF word / top-k weight bits, K:678-698)."""
    return Int32(_asm(T.i32(), [_a(ga, loc, ip)], "ld.u32 $0, [$1];", "=r,l", loc, ip, side=False))


@dsl_user_op
def st_generic_b32(ga, v, *, loc=None, ip=None) -> None:
    """Remote index push (K:446): ``st.u32`` through a generic address."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "st.u32 [$0], $1;", "l,r", loc, ip)


@dsl_user_op
def st_generic_b64(ga, v, *, loc=None, ip=None) -> None:
    """Remote per-rank count / grid-id push (K:460, K:468): ``st.u64`` through a generic address."""
    _asm(None, [_a(ga, loc, ip), _v(v, loc, ip)], "st.u64 [$0], $1;", "l,l", loc, ip)


@dsl_user_op
def st_generic_v4_b32(ga, x, y, z, w, *, loc=None, ip=None) -> None:
    """Remote combine write-back (K:1630, the L2 epilogue): ``st.v4.u32`` through a generic address."""
    _asm(None, [_a(ga, loc, ip)] + [_v(t, loc, ip) for t in (x, y, z, w)],
         "st.v4.u32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc, ip)  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# spin loops with DeepGEMM's timeout (comm::wait_until: clock64 budget, then trap instead of printf + assert)
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def grid_sync_arrive_wait(counter_ga, delta, *, loc=None, ip=None) -> None:
    """Thread 0 of ``comm::grid_sync`` (barrier.cuh:41-49): ``atom.release.gpu.add`` then spin on ``ld.acquire.gpu`` until
    the sense bit (0x80000000) of the counter flips relative to the value returned by the atomic; 300 s timeout."""
    _asm(
        None, [_a(counter_ga, loc, ip), _v(delta, loc, ip)],
        "{\n\t"
        ".reg .b32 old, cur; .reg .b64 t0, t1, dt; .reg .pred pdone, pto;\n\t"
        "mov.u64 t0, %clock64;\n\t"
        "atom.release.gpu.global.add.u32 old, [$0], $1;\n\t"
        "GS_SPIN:\n\t"
        "ld.acquire.gpu.global.b32 cur, [$0];\n\t"
        "xor.b32 cur, cur, old;\n\t"
        "and.b32 cur, cur, 0x80000000;\n\t"
        "setp.ne.u32 pdone, cur, 0;\n\t"
        "@pdone bra GS_DONE;\n\t"
        "mov.u64 t1, %clock64;\n\t"
        "sub.u64 dt, t1, t0;\n\t"
        f"setp.ge.u64 pto, dt, {NUM_TIMEOUT_CYCLES};\n\t"
        "@pto trap;\n\t"
        "bra GS_SPIN;\n\t"
        "GS_DONE:\n\t"
        "}",
        "l,r,~{memory}", loc, ip)  # fmt: skip


@dsl_user_op
def spin_acquire_sys_s32_eq(ga, target, *, loc=None, ip=None) -> None:
    """NVLink-barrier wait (barrier.cuh:82-87): spin on ``ld.acquire.sys.global.s32`` until it equals ``target``; 300 s timeout."""
    _asm(
        None, [_a(ga, loc, ip), _v(target, loc, ip)],
        "{\n\t"
        ".reg .s32 cur; .reg .b64 t0, t1, dt; .reg .pred pdone, pto;\n\t"
        "mov.u64 t0, %clock64;\n\t"
        "NB_SPIN:\n\t"
        "ld.acquire.sys.global.s32 cur, [$0];\n\t"
        "setp.eq.s32 pdone, cur, $1;\n\t"
        "@pdone bra NB_DONE;\n\t"
        "mov.u64 t1, %clock64;\n\t"
        "sub.u64 dt, t1, t0;\n\t"
        f"setp.ge.u64 pto, dt, {NUM_TIMEOUT_CYCLES};\n\t"
        "@pto trap;\n\t"
        "bra NB_SPIN;\n\t"
        "NB_DONE:\n\t"
        "}",
        "l,r", loc, ip)  # fmt: skip


@dsl_user_op
def vote_all_acquire_sys_b64_eq(ga, tag, active_pred, *, loc=None, ip=None) -> None:
    """Combine peer-flag wait (K:1729): every lane spins until ``__all_sync(not routed or ld.acquire.sys.b64(flag) == tag)``;
    ``active_pred`` (Int32 0/1) = this lane is a routed lane; 300 s timeout (each lane polls its own peer flag)."""
    _asm(
        None, [_a(ga, loc, ip), _v(tag, loc, ip), _v(active_pred, loc, ip)],
        "{\n\t"
        ".reg .b64 cur, t0, t1, dt; .reg .pred pact, pok, pall, pto;\n\t"
        "setp.ne.b32 pact, $2, 0;\n\t"
        "mov.b64 cur, 0;\n\t"
        "mov.u64 t0, %clock64;\n\t"
        "PF_SPIN:\n\t"
        "@pact ld.acquire.sys.global.b64 cur, [$0];\n\t"
        "setp.eq.b64 pok, cur, $1;\n\t"
        "not.pred pall, pact;\n\t"
        "or.pred pok, pok, pall;\n\t"
        "vote.sync.all.pred pall, pok, 0xffffffff;\n\t"
        "@pall bra PF_DONE;\n\t"
        "mov.u64 t1, %clock64;\n\t"
        "sub.u64 dt, t1, t0;\n\t"
        f"setp.ge.u64 pto, dt, {NUM_TIMEOUT_CYCLES};\n\t"
        "@pto trap;\n\t"
        "bra PF_SPIN;\n\t"
        "PF_DONE:\n\t"
        "}",
        "l,l,r,~{memory}", loc, ip)  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# warp helpers
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def fns_b32(mask, base, offset, *, loc=None, ip=None) -> Int32:
    """``__fns``: position of the ``offset``-th set bit of ``mask`` counting from bit ``base`` (non-volatile)."""
    return Int32(_asm(T.i32(), [_v(mask, loc, ip), _v(base, loc, ip), _v(offset, loc, ip)],
                      "fns.b32 $0, $1, $2, $3;", "=r,r,r,r", loc, ip, side=False))  # fmt: skip


# ----------------------------------------------------------------------------------------------------------------
# bulk copies / fences / numerics used by the combine
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def tma_load_1d(dst_sp, src_ga, mbar_sp, nbytes, hint=EVICT_FIRST, *, loc=None, ip=None) -> None:
    """``ptx::tma_load_1d``: ``cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint``."""
    _asm(
        None, [_sm(dst_sp, loc, ip), _a(src_ga, loc, ip), _v(nbytes, loc, ip), _sm(mbar_sp, loc, ip),
               _v(Int64(hint), loc, ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l,~{memory}", loc, ip)  # fmt: skip


@dsl_user_op
def tma_store_1d(dst_ga, src_sp, nbytes, hint=EVICT_NORMAL, *, loc=None, ip=None) -> None:
    """``ptx::tma_store_1d``: ``cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint`` (explicit EVICT_NORMAL)."""
    _asm(
        None, [_a(dst_ga, loc, ip), _sm(src_sp, loc, ip), _v(nbytes, loc, ip), _v(Int64(hint), loc, ip)],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], $2, $3;",
        "l,r,r,l,~{memory}", loc, ip)  # fmt: skip


@dsl_user_op
def tma_store_commit(*, loc=None, ip=None) -> None:
    """``cute::tma_store_arrive``: ``cp.async.bulk.commit_group``."""
    _asm(None, [], "cp.async.bulk.commit_group;", "~{memory}", loc, ip)


@dsl_user_op
def tma_store_wait0(*, loc=None, ip=None) -> None:
    """``ptx::tma_store_wait<0>``: ``cp.async.bulk.wait_group 0`` (NOT ``.read``)."""
    _asm(None, [], "cp.async.bulk.wait_group 0;", "~{memory}", loc, ip)


@dsl_user_op
def fence_proxy_async_shared_cta(*, loc=None, ip=None) -> None:
    """``cute::tma_store_fence`` / ``cutlass::arch::fence_view_async_shared``: ``fence.proxy.async.shared::cta``."""
    _asm(None, [], "fence.proxy.async.shared::cta;", "~{memory}", loc, ip)


@dsl_user_op
def mbarrier_arrive_expect_tx(mbar_sp, nbytes, *, loc=None, ip=None) -> None:
    """``ptx::mbarrier_arrive_and_set_tx``: ``mbarrier.arrive.expect_tx.shared::cta.b64 _, [bar], bytes``."""
    _asm(None, [_sm(mbar_sp, loc, ip), _v(nbytes, loc, ip)],
         "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r", loc, ip)  # fmt: skip


@dsl_user_op
def mbarrier_wait_parity(mbar_sp, phase, *, loc=None, ip=None) -> None:
    """``ClusterBarrier::wait(phase)``: ``mbarrier.try_wait.parity.shared::cta.b64`` spin loop (no suspend-time hint)."""
    _asm(
        None, [_sm(mbar_sp, loc, ip), _v(phase, loc, ip)],
        "{\n\t"
        ".reg .pred P1;\n\t"
        "MB_WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1;\n\t"
        "@P1 bra MB_DONE;\n\t"
        "bra MB_WAIT;\n\t"
        "MB_DONE:\n\t"
        "}",
        "r,r,~{memory}", loc, ip)  # fmt: skip


@dsl_user_op
def ld_shared_v4_b32(sp, *, loc=None, ip=None):
    """``ptx::ld_shared(uint4*)``: ``ld.shared.v4.u32`` -> 4 x Int32."""
    res = _asm(llvm.StructType.get_literal([T.i32()] * 4), [_sm(sp, loc, ip)],
               "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc, ip)  # fmt: skip
    return tuple(Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(4))


@dsl_user_op
def st_shared_v4_b32(sp, x, y, z, w, *, loc=None, ip=None) -> None:
    """``ptx::st_shared(ptr, x, y, z, w)``: ``st.shared.v4.u32``."""
    _asm(None, [_sm(sp, loc, ip)] + [_v(t, loc, ip) for t in (x, y, z, w)],
         "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r", loc, ip)  # fmt: skip


@dsl_user_op
def accumulate_bf16x2(acc0: Float32, acc1: Float32, word, *, loc=None, ip=None):
    """``ptx::accumulate(float2& a, nv_bfloat162 b)``: two ``add.rn.f32.bf16`` (fused cast + add, RN); returns (acc0', acc1').
    ``word`` is the packed bf16x2 register: low half -> acc0 (``b.x``), high half -> acc1 (``b.y``)."""
    res = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]), [_v(acc0, loc, ip), _v(acc1, loc, ip), _v(word, loc, ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $4;\n\tadd.rn.f32.bf16 $0, lo, $2;\n\tadd.rn.f32.bf16 $1, hi, $3;\n\t}",
        "=f,=f,0,1,r", loc, ip, side=False)  # fmt: skip
    return (Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
            Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)))  # fmt: skip


@dsl_user_op
def cvt_rn_bf16x2_f32(hi: Float32, lo: Float32, *, loc=None, ip=None) -> Int32:
    """``__float22bfloat162_rn(float2{lo, hi})``: ``cvt.rn.bf16x2.f32 d, hi, lo`` (``lo`` -> low half)."""
    return Int32(_asm(T.i32(), [_v(hi, loc, ip), _v(lo, loc, ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f", loc, ip, side=False))


# ----------------------------------------------------------------------------------------------------------------
# named barriers (ids reserved by DESIGN.md section 9: 0 = dispatch 128, 1 = dispatch+epilogue 384 unaligned, 2 = epilogue 256,
# 3/4 = epilogue warpgroups 128) -- the ctx.bar.* wraps the same two instructions
# ----------------------------------------------------------------------------------------------------------------
@dsl_user_op
def bar_sync_aligned(barrier_id, num_threads, *, loc=None, ip=None) -> None:
    """``ptx::sync_aligned``: ``bar.sync id, n``."""
    _asm(None, [_v(Int32(barrier_id), loc, ip), _v(Int32(num_threads), loc, ip)], "bar.sync $0, $1;", "r,r", loc, ip)


@dsl_user_op
def bar_sync_unaligned(barrier_id, num_threads, *, loc=None, ip=None) -> None:
    """``ptx::sync_unaligned``: ``barrier.sync id, n``."""
    _asm(None, [_v(Int32(barrier_id), loc, ip), _v(Int32(num_threads), loc, ip)], "barrier.sync $0, $1;", "r,r", loc, ip)
