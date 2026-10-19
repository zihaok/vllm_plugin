# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) ptx/{utils,ld_st,tma,tcgen05}.cuh, common/math.cuh and sm107_fp8_fp4_mega_moe.cuh; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION & AFFILIATES).
"""Inline-PTX helper library of the SM107 DeepGEMM-style MegaMoE kernel (DESIGN.md section 13).

Conventions
-----------
* Every helper is a ``@dsl_user_op`` function (or a method of :class:`Mbar`); asm is emitted through ``llvm.inline_asm``
  with the EXACT instruction text of DeepGEMM / CUTLASS (the only deviation is that shared-memory addresses are passed as
  32-bit registers instead of 64-bit ones).  No helper contains ``break/continue/return`` and none opens an
  ``elect_one`` region, so they can be called from inside dynamic regions.
* Address operands: ``ga`` = global address (``cute.Pointer`` in gmem/generic space OR ``Int64``/int generic address, e.g. the
  result of ``SymBuffer.map``); ``sp`` = shared-memory ``cute.Pointer`` (or an ``Int32`` shared::cta address).
* Memory/sync helpers are side-effecting asm (``has_side_effects=True``; ``~{memory}`` clobber where DeepGEMM has one or
  where the instruction orders memory).  Pure math helpers (``cvt``, ``ex2``, packed f32x2 ...) are side-effect free.

Which memory-space form where (verified against DeepGEMM's kernel.ptx via its ``.loc`` line info, BLOCK_M=128, sm_107a)
------------------------------------------------------------------------------------------------------------------
DeepGEMM dereferences the GENERIC pointer returned by ``SymBuffer::map`` (so every access THROUGH ``map`` is a generic ``ld``/``st``, SASS ``LD.E*``/``ST.E*``),
while local workspace / ring / counter accesses compile to ``.global`` forms (SASS ``LDG.E*``/``STG.E*``):
  K:661  remote token pull ``src[j*32+lane]``                   generic ``ld.v4.b32``   -> ``ld_generic_v4_b32``   (x10 per token, D)
  K:678/697 remote top-k weight / SF word loads                 generic ``ld.b32``      -> ``ld_generic_b32``      (D)
  K:707  LOCAL ring store ``dst[j*32+lane] = pull_regs[j]``     ``st.global.v4.b32``    -> ``st_v4_b32``           (D; NOT generic)
  K:446  remote ``src_token_topk_idx`` index push               generic ``st.b32``      -> ``st_generic_b32``      (D)
  K:460/468 remote ``peer_grid_idx`` / ``expert_recv_count``    generic ``st.b64``      -> ``st_generic_b64``      (D)
  K:1630 L2-epilogue remote BF16 write-back (16 B / lane)       generic ``st.v4.b32``   -> ``st_generic_v4_b32``   (G)
  remote atomics / reductions / acquire loads / release stores (``atom.sys``, ``red.release.sys``, ``ld.acquire.sys``, ``st.release.sys``) are written with
  explicit ``.global`` in DeepGEMM's asm helpers -> the ``ga`` helpers below (``atom_add_sys_u64`` ...) are right for remote pointers too.
Spin loops: ``comm._spin_*`` follow DeepGEMM's ``wait_until`` (predicate first, clock64 timeout check only on the not-done path, trap instead of printf).
The ptxas-inserted ``MEMBAR.ALL.CTA`` / ``@!PT LDS`` fillers before release atomics in DeepGEMM's SASS are context-dependent (PTX is identical).

API (name -> PTX emitted)
-------------------------
identity / misc
  smid() -> %smid | gridid() -> %gridid (Int64) | lane_id() -> %laneid | clock64() -> %clock64 | globaltimer() -> %globaltimer
  elect_sync() -> Boolean (``elect.sync``) | nanosleep(ns) -> ``nanosleep.u32`` | trap() -> ``trap;``
  advance_stage(stage, phase, n) -> (stage', phase')  [DeepGEMM advance_pipeline: phase flips when stage wraps to 0]
barriers / fences / cluster
  bar_sync(id, n) -> ``bar.sync id, n`` (aligned) | bar_sync_unaligned(id, n) -> ``barrier.sync id, n`` | bar_arrive(id, n) -> ``bar.arrive``
  fence_acq_rel_cta() -> ``fence.acq_rel.cta`` | fence_proxy_async_shared_cta() -> ``fence.proxy.async.shared::cta``
  fence_barrier_init() -> ``fence.mbarrier_init.release.cluster``
  tcgen05_fence_before_thread_sync() / tcgen05_fence_after_thread_sync() -> ``tcgen05.fence::{before,after}_thread_sync``
  cluster_arrive_relaxed()/cluster_arrive()/cluster_wait() -> ``barrier.cluster.{arrive.relaxed.aligned,arrive.aligned,wait.aligned}``;
  cluster_sync_relaxed() = arrive.relaxed + wait, cluster_sync() = arrive + wait
mbarrier (class Mbar over an smem pointer)
  init(count) -> ``mbarrier.init.shared::cta.b64`` | wait(phase) -> try_wait.parity spin loop with hint 0x989680 (CUTLASS ClusterBarrier::wait)
  try_wait(phase) -> Boolean | arrive() | arrive_count(n) | arrive_expect_tx(bytes) | expect_tx(bytes)
  arrive_remote(cta) -> ``mapa.shared::cluster`` + ``mbarrier.arrive.shared::cluster.b64`` | arrive_expect_tx_remote(bytes, cta)
  cluster_addr(cta) -> Int32 (``mapa.shared::cluster.u32``) | addr -> Int32 shared::cta address | addr_leader -> addr & 0xFEFFFFFF
shared memory
  ld_shared_u32/f32/v2_f32/v4_f32/v4_u32/b16(sp) | st_shared_u32/f32/v2_f32/v2_u32/v4_u32(sp, ...) | st_shared_zero(sp, nbytes) (``st.bulk.weak.shared::cta``)
  mapa_shared(addr, cta) -> Int32 | st_async_v4(dst_sp, x, y, z, w, mbar, cta) -> ``st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4``
global loads: ld_b32/ld_b64/ld_f32/ld_v4_b32(ga) [plain ``ld.global``] | ld_volatile_b32/b64 [``ld.volatile.global``]
  GENERIC-space variants (what DeepGEMM's PTX uses for accesses through ``SymBuffer::map`` pointers, SASS ``LD.E``/``ST.E`` instead of ``LDG``/``STG``):
  ld_generic_b32/b64/v4_b32(ga) [``ld.b32``/``ld.b64``/``ld.v4.b32``] | st_generic_b32/b64/v4_b32(ga, ...) [``st.b32``/``st.b64``/``st.v4.b32``]
  ld_acquire_gpu_b32/b64 [``ld.acquire.gpu.global``] | ld_acquire_sys_b32/s32/b64 [``ld.acquire.sys.global``]
global stores: st_b32/st_b64/st_f32/st_v4_b32(ga, ...) [plain ``st.global``] | st_b8(ga, v) [``st.global.b8``: L1-epilogue SF byte stores, SASS ``STG.E.U8``] | st_release_sys_b64 / st_release_gpu_b64 [``st.release.{sys,gpu}.global.u64``]
atomics: atom_add_u32/u64(ga, v) -> old [``atom.global.add``] | atom_add_sys_u64 [``atom.sys.global.add.u64``] | atom_add_release_gpu_u32 [``atom.release.gpu.global.add.u32``]
  red_add_u32/s32 [``red.gpu.global.add``] | red_add_release_gpu_u32 | red_add_release_sys_s32 | red_xor_release_gpu_b64 | red_or_release_gpu_b64 | red_or_release_sys_b64
bulk copy: tma_load_1d(dst_sp, src_ga, mbar, nbytes, hint=EVICT_FIRST) [``cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint``]
  tma_store_1d(dst_ga, src_sp, nbytes, hint=EVICT_NORMAL) [``cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint``]
  tma_store_commit() [``cp.async.bulk.commit_group``] | tma_store_wait(n) [``cp.async.bulk.wait_group n``, NOT ``.read``]
warp-level: ballot/all_sync/any_sync(pred) | shfl_idx/shfl_xor/shfl_up(v, x) | redux_add_s32/u32, redux_min_u32, redux_max_f32 [``redux.sync``]
  fns_b32(mask, base, offset) | popc(x) | ffs(x) (1-based, 0 if none) | clz(x)
numerics (DeepGEMM modifiers exactly): cvt_rn_bf16x2_f32(hi, lo) | add_rn_f32_bf16(acc, bf16_bits) | accumulate_bf16x2(ax, ay, packed)
  fadd2_rn/fmul2_rn/ffma2_rn(pair, pair[, pair]) [``{add,mul,fma}.rn.f32x2``] | rcp_approx_ftz(x) [``rcp.approx.ftz.f32``, = DeepGEMM math::fast_rcp]
  ex2_approx(x) [``ex2.approx.f32``: what DeepGEMM's ``__expf`` compiles to (NO .ftz!)] | ex2_approx_ftz(x) [``ex2.approx.ftz.f32``]
  hmin2_bf16x2(a, b) | hmax2_bf16x2(a, b) [``min/max.bf16x2``] | bf16x2_to_f32x2(r) | cvt_e4m3x4_f32(x, y, z, w) [2x ``cvt.rn.satfinite.e4m3x2.f32``]
tensor memory / stmatrix: tmem_ld_16x256b_x1(taddr) -> 4xInt32 | tmem_ld_wait() [``tcgen05.wait::ld.sync.aligned``]
  tmem_st_16x256b_x1(taddr, r0..r3) | tmem_st_wait() | tmem_alloc(slot_sp, ncols, cta_group=2, exclusive=True) / tmem_dealloc(taddr, ncols, cta_group=2, exclusive=True)
  [``tcgen05.alloc/dealloc[.exclusive].cta_group::N.sync.aligned``: DeepGEMM's cubin uses the ``.exclusive.cta_group::2`` forms for ALL six BLOCK_M]
  stmatrix_m16n8_x1_trans_b8 / stmatrix_m16n8_x2_trans_b8 / stmatrix_m8n8_x4_trans_b16 / stmatrix_m8n8_x2_trans_b16 / stmatrix_m8n8_x2_b16
shared atomics / warp sync: atom_add_shared_u32(sp, v) -> old [``atom.shared.cta.add.u32``, = ``atomicAdd_block``] | syncwarp() [``bar.warp.sync 0xffffffff``]
"""


import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import arith, llvm, nvvm
from cutlass.base_dsl.dsl import extract_mlir_values, get_mlir_types, new_from_mlir_values
from cutlass.cutlass_dsl import Boolean, Float32, Int32, Int64, T, dsl_user_op

#: L2 cache-hint policies used by the kernel (``cute::TMA::CacheHintSm100``).
EVICT_NORMAL = 0x1000000000000000
EVICT_FIRST = 0x12F0000000000000
#: ``mbarrier.try_wait`` suspend-time hint of CUTLASS' ``ClusterBarrier::wait``.
_TRY_WAIT_TICKS = 0x989680


# ---------------------------------------------------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------------------------------------------------
def _asm(res, operands, asm, constraints, *, side_effects=True, loc=None, ip=None):
    return llvm.inline_asm(
        res,
        operands,
        asm,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


def _i32(x, *, loc=None, ip=None):
    return Int32(x).ir_value(loc=loc, ip=ip)


def _i64(x, *, loc=None, ip=None):
    return Int64(x).ir_value(loc=loc, ip=ip)


def _f32(x, *, loc=None, ip=None):
    return Float32(x).ir_value(loc=loc, ip=ip)


def _gaddr(x, *, loc=None, ip=None):
    """Global/generic address operand -> ``i64`` value (accepts gmem/generic ``cute.Pointer``, ``Int64``, int)."""
    if isinstance(x, cute.Pointer):
        return x.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int64(x).ir_value(loc=loc, ip=ip)


def _saddr(x, *, loc=None, ip=None):
    """Shared::cta address operand -> ``i32`` value (accepts smem ``cute.Pointer``, :class:`Mbar`, ``Int32``, int)."""
    if isinstance(x, Mbar):
        x = x.ptr
    if isinstance(x, cute.Pointer):
        return x.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    return Int32(x).ir_value(loc=loc, ip=ip)


def _struct(types):
    return llvm.StructType.get_literal(types)


def _unpack(res, n, ty, wrap, *, loc=None, ip=None):
    return tuple(wrap(llvm.extractvalue(ty, res, [i], loc=loc, ip=ip)) for i in range(n))


# ---------------------------------------------------------------------------------------------------------------------
# identity / misc (group A)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def smid(*, loc=None, ip=None) -> Int32:
    """``%smid`` (physical SM id)."""
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %smid;", "=r", loc=loc, ip=ip))


@dsl_user_op
def gridid(*, loc=None, ip=None) -> Int64:
    """``%gridid`` (unique per launch; DeepGEMM ``ptx::get_grid_idx``)."""
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %gridid;", "=l", loc=loc, ip=ip))


@dsl_user_op
def lane_id(*, loc=None, ip=None) -> Int32:
    """``%laneid``."""
    return Int32(_asm(T.i32(), [], "mov.u32 $0, %laneid;", "=r", loc=loc, ip=ip))


@dsl_user_op
def clock64(*, loc=None, ip=None) -> Int64:
    """``%clock64`` (per-SM cycle counter)."""
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %clock64;", "=l", loc=loc, ip=ip))


@dsl_user_op
def globaltimer(*, loc=None, ip=None) -> Int64:
    """``%globaltimer`` (ns)."""
    return Int64(_asm(T.i64(), [], "mov.u64 $0, %globaltimer;", "=l", loc=loc, ip=ip))


@dsl_user_op
def elect_sync(*, loc=None, ip=None) -> Boolean:
    """``elect.sync`` over the full warp: True on exactly one (the elected) lane. DeepGEMM ``cute::elect_one_sync()``."""
    return Boolean(nvvm.elect_sync(loc=loc, ip=ip))


@dsl_user_op
def nanosleep(ns, *, loc=None, ip=None) -> None:
    """``nanosleep.u32 ns``."""
    _asm(None, [_i32(ns, loc=loc, ip=ip)], "nanosleep.u32 $0;", "r", loc=loc, ip=ip)


@dsl_user_op
def trap(*, loc=None, ip=None) -> None:
    """``trap;`` (timeout path of the spin loops; aborts the kernel with an error)."""
    _asm(None, [], "trap;", "", loc=loc, ip=ip)


def advance_stage(stage, phase, num_stages: int):
    """DeepGEMM ``advance_pipeline``: ``stage = stage == n-1 ? 0 : stage+1; phase ^= (stage == 0)``.

    Plain (non-jit) arithmetic helper: returns ``(stage', phase')`` as ``Int32``; usable as
    ``stage, phase = ptx.advance_stage(stage, phase, N)`` for loop-carried locals.
    """
    stage = Int32(stage)
    phase = Int32(phase)
    last = stage == Int32(num_stages - 1)
    new_stage = Int32(cutlass.select_(last, Int32(0), stage + Int32(1)))
    new_phase = phase ^ Int32(last)
    return new_stage, new_phase


# ---------------------------------------------------------------------------------------------------------------------
# named barriers / fences / cluster (group A)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def bar_sync(barrier_id, num_threads, *, loc=None, ip=None) -> None:
    """Aligned named barrier ``bar.sync id, n`` (DeepGEMM ``ptx::sync_aligned(n, id)``)."""
    if isinstance(barrier_id, int) and isinstance(num_threads, int):
        _asm(None, [], f"bar.sync {barrier_id}, {num_threads};", "~{memory}", loc=loc, ip=ip)
    else:
        _asm(None, [_i32(barrier_id, loc=loc, ip=ip), _i32(num_threads, loc=loc, ip=ip)],
             "bar.sync $0, $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def bar_sync_unaligned(barrier_id, num_threads, *, loc=None, ip=None) -> None:
    """Unaligned named barrier ``barrier.sync id, n`` (DeepGEMM ``ptx::sync_unaligned(n, id)``; used for R1/R2 id 1, 384 thr)."""
    if isinstance(barrier_id, int) and isinstance(num_threads, int):
        _asm(None, [], f"barrier.sync {barrier_id}, {num_threads};", "~{memory}", loc=loc, ip=ip)
    else:
        _asm(None, [_i32(barrier_id, loc=loc, ip=ip), _i32(num_threads, loc=loc, ip=ip)],
             "barrier.sync $0, $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def bar_arrive(barrier_id, num_threads, *, loc=None, ip=None) -> None:
    """``bar.arrive id, n`` (non-blocking arrive on a named barrier)."""
    _asm(None, [_i32(barrier_id, loc=loc, ip=ip), _i32(num_threads, loc=loc, ip=ip)],
         "bar.arrive $0, $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def fence_acq_rel_cta(*, loc=None, ip=None) -> None:
    """``fence.acq_rel.cta;`` (DeepGEMM ``ptx::fence_acq_rel_cta``)."""
    _asm(None, [], "fence.acq_rel.cta;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def fence_proxy_async_shared_cta(*, loc=None, ip=None) -> None:
    """``fence.proxy.async.shared::cta;`` (cute ``tma_store_fence``)."""
    _asm(None, [], "fence.proxy.async.shared::cta;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def fence_barrier_init(*, loc=None, ip=None) -> None:
    """``fence.mbarrier_init.release.cluster;`` (``cutlass::arch::fence_barrier_init``)."""
    _asm(None, [], "fence.mbarrier_init.release.cluster;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_before_thread_sync(*, loc=None, ip=None) -> None:
    """``tcgen05.fence::before_thread_sync;``"""
    _asm(None, [], "tcgen05.fence::before_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def tcgen05_fence_after_thread_sync(*, loc=None, ip=None) -> None:
    """``tcgen05.fence::after_thread_sync;``"""
    _asm(None, [], "tcgen05.fence::after_thread_sync;", "", loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive_relaxed(*, loc=None, ip=None) -> None:
    """``barrier.cluster.arrive.relaxed.aligned;``"""
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def cluster_arrive(*, loc=None, ip=None) -> None:
    """``barrier.cluster.arrive.aligned;`` (release semantics)."""
    _asm(None, [], "barrier.cluster.arrive.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def cluster_wait(*, loc=None, ip=None) -> None:
    """``barrier.cluster.wait.aligned;`` (acquire semantics)."""
    _asm(None, [], "barrier.cluster.wait.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def cluster_sync_relaxed(*, loc=None, ip=None) -> None:
    """DeepGEMM ``comm::cluster_sync_with_relaxed_arrive``: arrive.relaxed + wait."""
    _asm(None, [], "barrier.cluster.arrive.relaxed.aligned;\nbarrier.cluster.wait.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def cluster_sync(*, loc=None, ip=None) -> None:
    """``cute::cluster_sync``: arrive (release) + wait (acquire)."""
    _asm(None, [], "barrier.cluster.arrive.aligned;\nbarrier.cluster.wait.aligned;", "~{memory}", loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------------------------------
# mbarrier handle (group A)
# ---------------------------------------------------------------------------------------------------------------------
class Mbar:
    """Thin handle over an 8-byte mbarrier in shared memory (``ptr``: smem ``cute.Pointer`` or ``Int32`` shared address).

    Mirrors ``cutlass::arch::ClusterTransactionBarrier`` as used by DeepGEMM.  The handle is a dynamic-value object (it
    carries the smem pointer through the DSL's ``__extract_mlir_values__`` protocol) so it may be passed to/returned from
    ``@cute.jit`` functions and loop-carried.
    """

    def __init__(self, ptr):
        self.ptr = ptr

    # --- DSL dynamic-value protocol
    def __extract_mlir_values__(self):
        return extract_mlir_values(self.ptr)

    def __new_from_mlir_values__(self, values):
        return Mbar(new_from_mlir_values(self.ptr, values))

    def __get_mlir_types__(self):
        return get_mlir_types(self.ptr)

    # --- addresses
    @property
    def addr(self) -> Int32:
        """Shared::cta address (Int32)."""
        if isinstance(self.ptr, cute.Pointer):
            return self.ptr.toint()
        return Int32(self.ptr)

    @property
    def addr_leader(self) -> Int32:
        """``addr & 0xFEFFFFFF``: leader-CTA form of the barrier address used by 2-SM TMA loads (peer bit cleared)."""
        return self.addr & Int32(0xFEFFFFFF)

    @dsl_user_op
    def cluster_addr(self, cta_rank, *, loc=None, ip=None) -> Int32:
        """``mapa.shared::cluster.u32``: address of this barrier inside CTA ``cta_rank`` of the cluster."""
        return mapa_shared(self.addr, cta_rank, loc=loc, ip=ip)

    # --- operations
    @dsl_user_op
    def init(self, count, *, loc=None, ip=None) -> None:
        """``mbarrier.init.shared::cta.b64 [addr], count;``"""
        _asm(None, [_saddr(self.ptr, loc=loc, ip=ip), _i32(count, loc=loc, ip=ip)],
             "mbarrier.init.shared::cta.b64 [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip

    @dsl_user_op
    def wait(self, phase, *, loc=None, ip=None) -> None:
        """Spin on ``mbarrier.try_wait.parity`` with the CUTLASS suspend hint until phase ``phase`` (parity) completes."""
        _asm(
            None,
            [_saddr(self.ptr, loc=loc, ip=ip), _i32(phase, loc=loc, ip=ip), _i32(_TRY_WAIT_TICKS, loc=loc, ip=ip)],
            "{\n\t"
            ".reg .pred       P1; \n\t"
            "LAB_WAIT: \n\t"
            "mbarrier.try_wait.parity.shared::cta.b64 P1, [$0], $1, $2; \n\t"
            "@P1 bra DONE; \n\t"
            "bra     LAB_WAIT; \n\t"
            "DONE: \n\t"
            "}",
            "r,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def try_wait(self, phase, *, loc=None, ip=None) -> Boolean:
        """Single non-blocking-ish ``mbarrier.try_wait.parity`` (may suspend up to the hardware time limit); True when the phase completed."""
        r = _asm(
            T.i32(),
            [_saddr(self.ptr, loc=loc, ip=ip), _i32(phase, loc=loc, ip=ip)],
            "{\n\t.reg .pred p;\n\tmbarrier.try_wait.parity.shared::cta.b64 p, [$1], $2;\n\tselp.u32 $0, 1, 0, p;\n\t}",
            "=r,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )
        return Int32(r) != Int32(0)

    @dsl_user_op
    def arrive(self, *, loc=None, ip=None) -> None:
        """``mbarrier.arrive.shared::cta.b64 _, [addr];``"""
        _asm(None, [_saddr(self.ptr, loc=loc, ip=ip)], "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", loc=loc, ip=ip)

    @dsl_user_op
    def arrive_count(self, count, *, loc=None, ip=None) -> None:
        """``mbarrier.arrive.shared::cta.b64 _, [addr], count;``"""
        _asm(None, [_saddr(self.ptr, loc=loc, ip=ip), _i32(count, loc=loc, ip=ip)],
             "mbarrier.arrive.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip

    @dsl_user_op
    def arrive_expect_tx(self, nbytes, *, loc=None, ip=None) -> None:
        """``mbarrier.arrive.expect_tx.shared::cta.b64 _, [addr], bytes;``"""
        _asm(None, [_saddr(self.ptr, loc=loc, ip=ip), _i32(nbytes, loc=loc, ip=ip)],
             "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip

    @dsl_user_op
    def expect_tx(self, nbytes, *, loc=None, ip=None) -> None:
        """``mbarrier.expect_tx.shared::cta.b64 [addr], bytes;``"""
        _asm(None, [_saddr(self.ptr, loc=loc, ip=ip), _i32(nbytes, loc=loc, ip=ip)],
             "mbarrier.expect_tx.shared::cta.b64 [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip

    @dsl_user_op
    def arrive_remote(self, cta_rank, *, loc=None, ip=None) -> None:
        """Arrive on this barrier's counterpart in CTA ``cta_rank`` (CUTLASS ``ClusterBarrier::arrive(cta_id)``):
        ``mapa.shared::cluster.u32`` + ``mbarrier.arrive.shared::cluster.b64 _, [remote];``"""
        _asm(
            None,
            [_saddr(self.ptr, loc=loc, ip=ip), _i32(cta_rank, loc=loc, ip=ip)],
            "{\n\t.reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32  remAddr32, $0, $1;\n\t"
            "mbarrier.arrive.shared::cluster.b64  _, [remAddr32];\n\t}",
            "r,r,~{memory}",
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def arrive_expect_tx_remote(self, nbytes, cta_rank, *, loc=None, ip=None) -> None:
        """``mapa`` + ``mbarrier.arrive.expect_tx.shared::cluster.b64 _, [remote], bytes;``"""
        _asm(
            None,
            [_saddr(self.ptr, loc=loc, ip=ip), _i32(cta_rank, loc=loc, ip=ip), _i32(nbytes, loc=loc, ip=ip)],
            "{\n\t.reg .b32 remAddr32;\n\t"
            "mapa.shared::cluster.u32  remAddr32, $0, $1;\n\t"
            "mbarrier.arrive.expect_tx.shared::cluster.b64  _, [remAddr32], $2;\n\t}",
            "r,r,r,~{memory}",
            loc=loc,
            ip=ip,
        )


@dsl_user_op
def mapa_shared(addr, cta_rank, *, loc=None, ip=None) -> Int32:
    """``mapa.shared::cluster.u32``: map a shared::cta address (Int32/smem pointer) into CTA ``cta_rank`` of the cluster."""
    return Int32(
        _asm(
            T.i32(),
            [_saddr(addr, loc=loc, ip=ip), _i32(cta_rank, loc=loc, ip=ip)],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# shared memory ops (groups A/B)
# ---------------------------------------------------------------------------------------------------------------------
USE_ST_BULK = True  # set False to use the vector-store fallback of st_shared_zero


@dsl_user_op
def st_shared_zero(sp, nbytes: int, *, loc=None, ip=None) -> None:
    """Zero ``nbytes`` (multiple of 8, compile-time int) of shared memory: ``st.bulk.weak.shared::cta [a], size, 0;`` (DeepGEMM
    ``ptx::st_shared_bulk``; called by ONE elected lane).  Falls back to unrolled ``st.shared.v4.u32`` stores when
    ``USE_ST_BULK`` is False (nbytes must then be a multiple of 16)."""
    assert isinstance(nbytes, int) and nbytes % 8 == 0
    if USE_ST_BULK:
        _asm(
            None,
            [_saddr(sp, loc=loc, ip=ip), _i64(nbytes, loc=loc, ip=ip)],
            "{\n\t.reg .u64 a64;\n\tcvt.u64.u32 a64, $0;\n\tst.bulk.weak.shared::cta [a64], $1, 0;\n\t}",
            "r,l,~{memory}",
            loc=loc,
            ip=ip,
        )
    else:
        assert nbytes % 16 == 0
        base = _saddr(sp, loc=loc, ip=ip)
        z = _i32(0, loc=loc, ip=ip)
        for i in range(nbytes // 16):
            _asm(None, [base, z, z, z, z],
                 f"st.shared.v4.u32 [$0 + {16 * i}], {{$1, $2, $3, $4}};", "r,r,r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def ld_shared_u32(sp, *, loc=None, ip=None) -> Int32:
    """``ld.shared.u32``"""
    return Int32(_asm(T.i32(), [_saddr(sp, loc=loc, ip=ip)], "ld.shared.u32 $0, [$1];", "=r,r", loc=loc, ip=ip))


@dsl_user_op
def ld_shared_f32(sp, *, loc=None, ip=None) -> Float32:
    """``ld.shared.f32``"""
    return Float32(_asm(T.f32(), [_saddr(sp, loc=loc, ip=ip)], "ld.shared.f32 $0, [$1];", "=f,r", loc=loc, ip=ip))


@dsl_user_op
def ld_shared_v2_f32(sp, *, loc=None, ip=None):
    """``ld.shared.v2.f32`` -> (Float32, Float32)"""
    res = _asm(_struct([T.f32()] * 2), [_saddr(sp, loc=loc, ip=ip)], "ld.shared.v2.f32 {$0, $1}, [$2];", "=f,=f,r", loc=loc, ip=ip)
    return _unpack(res, 2, T.f32(), Float32, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v4_f32(sp, *, loc=None, ip=None):
    """``ld.shared.v4.f32`` -> 4 x Float32"""
    res = _asm(_struct([T.f32()] * 4), [_saddr(sp, loc=loc, ip=ip)], "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
               "=f,=f,=f,=f,r", loc=loc, ip=ip)  # fmt: skip
    return _unpack(res, 4, T.f32(), Float32, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v4_u32(sp, *, loc=None, ip=None):
    """``ld.shared.v4.u32`` -> 4 x Int32"""
    res = _asm(_struct([T.i32()] * 4), [_saddr(sp, loc=loc, ip=ip)], "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,r", loc=loc, ip=ip)  # fmt: skip
    return _unpack(res, 4, T.i32(), Int32, loc=loc, ip=ip)


@dsl_user_op
def ld_shared_b16(sp, *, loc=None, ip=None) -> Int32:
    """``ld.shared.b16`` zero-extended to Int32 (SASS ``LDS.U16``)."""
    return Int32(_asm(T.i32(), [_saddr(sp, loc=loc, ip=ip)],
                      "{\n\t.reg .b16 h;\n\tld.shared.b16 h, [$1];\n\tcvt.u32.u16 $0, h;\n\t}", "=r,r", loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def st_shared_u32(sp, v, *, loc=None, ip=None) -> None:
    """``st.shared.u32``"""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "st.shared.u32 [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def st_shared_f32(sp, v, *, loc=None, ip=None) -> None:
    """``st.shared.f32``"""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _f32(v, loc=loc, ip=ip)], "st.shared.f32 [$0], $1;", "r,f,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def st_shared_v2_f32(sp, a, b, *, loc=None, ip=None) -> None:
    """``st.shared.v2.f32``"""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _f32(a, loc=loc, ip=ip), _f32(b, loc=loc, ip=ip)],
         "st.shared.v2.f32 [$0], {$1, $2};", "r,f,f,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_shared_v2_u32(sp, x, y, *, loc=None, ip=None) -> None:
    """``st.shared.v2.u32``"""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(x, loc=loc, ip=ip), _i32(y, loc=loc, ip=ip)],
         "st.shared.v2.u32 [$0], {$1, $2};", "r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_shared_v4_u32(sp, x, y, z, w, *, loc=None, ip=None) -> None:
    """``st.shared.v4.u32``"""
    _asm(None, [_saddr(sp, loc=loc, ip=ip)] + [_i32(v, loc=loc, ip=ip) for v in (x, y, z, w)],
         "st.shared.v4.u32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_async_v4(dst_sp, x, y, z, w, mbar, cta_rank, *, loc=None, ip=None) -> None:
    """``st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4`` of 16 B into CTA ``cta_rank``'s copy of ``dst_sp``,
    completing 16 bytes of transaction on CTA ``cta_rank``'s copy of ``mbar`` (DeepGEMM ``ptx::st_async_cluster``; both the
    destination and the barrier are ``mapa``-mapped)."""
    _asm(
        None,
        [_saddr(dst_sp, loc=loc, ip=ip), _saddr(mbar, loc=loc, ip=ip), _i32(cta_rank, loc=loc, ip=ip)]
        + [_i32(v, loc=loc, ip=ip) for v in (x, y, z, w)],
        "{\n\t.reg .b32 rdst;\n\t.reg .b32 rbar;\n\t"
        "mapa.shared::cluster.u32 rdst, $0, $2;\n\t"
        "mapa.shared::cluster.u32 rbar, $1, $2;\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4 [rdst], {$3, $4, $5, $6}, [rbar];\n\t}",
        "r,r,r,r,r,r,r,~{memory}",
        loc=loc,
        ip=ip,
    )


# ---------------------------------------------------------------------------------------------------------------------
# global loads / stores / atomics (group B)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def ld_b32(ga, *, loc=None, ip=None) -> Int32:
    """plain ``ld.global.b32``"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_b64(ga, *, loc=None, ip=None) -> Int64:
    """plain ``ld.global.b64``"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip)], "ld.global.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_f32(ga, *, loc=None, ip=None) -> Float32:
    """plain ``ld.global.f32``"""
    return Float32(_asm(T.f32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.global.f32 $0, [$1];", "=f,l", loc=loc, ip=ip))


@dsl_user_op
def ld_v4_b32(ga, *, loc=None, ip=None):
    """plain ``ld.global.v4.b32`` (SASS ``LDG.E.128``; remote token pulls) -> 4 x Int32"""
    res = _asm(_struct([T.i32()] * 4), [_gaddr(ga, loc=loc, ip=ip)], "ld.global.v4.b32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,l", loc=loc, ip=ip)  # fmt: skip
    return _unpack(res, 4, T.i32(), Int32, loc=loc, ip=ip)


@dsl_user_op
def ld_volatile_b32(ga, *, loc=None, ip=None) -> Int32:
    """``ld.volatile.global.b32`` (DeepGEMM ``ptx::ld_volatile(uint32_t*)``)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.volatile.global.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_volatile_b64(ga, *, loc=None, ip=None) -> Int64:
    """``ld.volatile.global.b64`` (DeepGEMM ``ptx::ld_volatile(uint64_t*)``)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip)], "ld.volatile.global.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_gpu_b32(ga, *, loc=None, ip=None) -> Int32:
    """``ld.acquire.gpu.global.b32`` (DeepGEMM ``ptx::ld_acq``)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.acquire.gpu.global.b32 $0, [$1];", "=r,l,~{memory}", loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_gpu_b64(ga, *, loc=None, ip=None) -> Int64:
    """``ld.acquire.gpu.global.u64`` (DeepGEMM ``ptx::ld_acq_gpu``)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip)], "ld.acquire.gpu.global.u64 $0, [$1];", "=l,l,~{memory}", loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_sys_b32(ga, *, loc=None, ip=None) -> Int32:
    """``ld.acquire.sys.global.u32`` (DeepGEMM ``ptx::ld_acq_sys(uint32_t*)``)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.acquire.sys.global.u32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_sys_s32(ga, *, loc=None, ip=None) -> Int32:
    """``ld.acquire.sys.global.s32`` (DeepGEMM ``ptx::ld_acq_sys(int*)``; NVLink barrier signals)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.acquire.sys.global.s32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_acquire_sys_b64(ga, *, loc=None, ip=None) -> Int64:
    """``ld.acquire.sys.global.b64`` (DeepGEMM ``ptx::ld_acq_sys(uint64_t*)``; combine ready flags)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip)], "ld.acquire.sys.global.b64 $0, [$1];", "=l,l,~{memory}", loc=loc, ip=ip))


@dsl_user_op
def st_b32(ga, v, *, loc=None, ip=None) -> None:
    """plain ``st.global.b32``"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "st.global.b32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def st_b64(ga, v, *, loc=None, ip=None) -> None:
    """plain ``st.global.b64``"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "st.global.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


@dsl_user_op
def st_f32(ga, v, *, loc=None, ip=None) -> None:
    """plain ``st.global.f32``"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _f32(v, loc=loc, ip=ip)], "st.global.f32 [$0], $1;", "l,f", loc=loc, ip=ip)


@dsl_user_op
def st_v4_b32(ga, x, y, z, w, *, loc=None, ip=None) -> None:
    """plain ``st.global.v4.b32`` (SASS ``STG.E.128``; L2 write-back to remote combine buffers)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip)] + [_i32(v, loc=loc, ip=ip) for v in (x, y, z, w)],
         "st.global.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_b8(ga, v, *, loc=None, ip=None) -> None:
    """``st.global.b8`` of the low 8 bits of ``v`` (SASS ``STG.E.U8``; the L1 epilogue's per-token UE8M0 SF byte stores, K:1470ff)."""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)],
         "{\n\t.reg .b16 b;\n\tcvt.u16.u32 b, $1;\n\tst.global.b8 [$0], b;\n\t}", "l,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def st_release_sys_b64(ga, v, *, loc=None, ip=None) -> None:
    """``st.release.sys.global.u64`` (DeepGEMM ``ptx::st_rel_sys``)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "st.release.sys.global.u64 [$0], $1;", "l,l,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def st_release_gpu_b64(ga, v, *, loc=None, ip=None) -> None:
    """``st.release.gpu.global.u64`` (DeepGEMM ``ptx::st_rel``)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "st.release.gpu.global.u64 [$0], $1;", "l,l,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def atom_add_u32(ga, v, *, loc=None, ip=None) -> Int32:
    """``atom.global.add.u32`` -> old value (DeepGEMM ``ptx::atomic_add``; task claims)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "atom.global.add.u32 $0, [$1], $2;", "=r,l,r", loc=loc, ip=ip))


@dsl_user_op
def atom_add_u64(ga, v, *, loc=None, ip=None) -> Int64:
    """``atom.global.add.u64`` -> old value (DeepGEMM ``ptx::atomic_add``; per-expert send counts)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "atom.global.add.u64 $0, [$1], $2;", "=l,l,l", loc=loc, ip=ip))


@dsl_user_op
def atom_add_sys_u64(ga, v, *, loc=None, ip=None) -> Int64:
    """``atom.sys.global.add.u64`` -> old value (DeepGEMM ``ptx::atomic_add_sys``; remote expert_recv_count_sum)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "atom.sys.global.add.u64 $0, [$1], $2;", "=l,l,l", loc=loc, ip=ip))


@dsl_user_op
def atom_add_release_gpu_u32(ga, v, *, loc=None, ip=None) -> Int32:
    """``atom.release.gpu.global.add.u32`` -> old value (DeepGEMM ``ptx::atomic_add_rel``; grid sync)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "atom.release.gpu.global.add.u32 $0, [$1], $2;",
                      "=r,l,r,~{memory}", loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def red_add_u32(ga, v, *, loc=None, ip=None) -> None:
    """``red.gpu.global.add.u32`` (relaxed; DeepGEMM ``ptx::red_add(uint32_t*)``)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "red.gpu.global.add.u32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def red_add_s32(ga, v, *, loc=None, ip=None) -> None:
    """``red.gpu.global.add.s32`` (relaxed; DeepGEMM ``ptx::red_add(int*)``; cumulative stats)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "red.gpu.global.add.s32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def red_add_release_gpu_u32(ga, v, *, loc=None, ip=None) -> None:
    """``red.release.gpu.global.add.u32`` (DeepGEMM ``ptx::red_add_rel``; ``l1_full_count`` publish)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "red.release.gpu.global.add.u32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def red_add_release_sys_s32(ga, v, *, loc=None, ip=None) -> None:
    """``red.release.sys.global.add.s32`` (DeepGEMM ``ptx::red_add_rel_sys``; NVLink barrier signals)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "red.release.sys.global.add.s32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def red_xor_release_gpu_b64(ga, v, *, loc=None, ip=None) -> None:
    """``red.release.gpu.global.xor.b64`` (DeepGEMM ``ptx::red_xor_rel``; ``l2_full_mask`` publish)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "red.release.gpu.global.xor.b64 [$0], $1;", "l,l,~{memory}", loc=loc, ip=ip)


@dsl_user_op
def red_or_release_gpu_b64(ga, v, *, loc=None, ip=None) -> None:
    """``red.release.gpu.global.or.b64`` (DeepGEMM ``ptx::red_or_rel_gpu``)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "red.release.gpu.global.or.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


@dsl_user_op
def red_or_release_sys_b64(ga, v, *, loc=None, ip=None) -> None:
    """``red.release.sys.global.or.b64`` (DeepGEMM ``ptx::red_or_rel_sys``)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "red.release.sys.global.or.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


# generic-space accesses: DeepGEMM's PTX uses ``ld.v4.b32`` / ``st.v4.b32`` / ``st.b32`` / ``st.b64`` (generic, SASS ``LD.E.128`` / ``ST.E.128`` /
# ``ST.E``) for accesses through pointers produced by ``SymBuffer::map`` (remote token pulls, remote index/count stores) and for the local ring stores
# of the pulled registers, and ``.global`` forms only where the compiler can prove the address space.
@dsl_user_op
def ld_generic_b32(ga, *, loc=None, ip=None) -> Int32:
    """generic ``ld.b32`` (SASS ``LD.E``)"""
    return Int32(_asm(T.i32(), [_gaddr(ga, loc=loc, ip=ip)], "ld.b32 $0, [$1];", "=r,l", loc=loc, ip=ip))


@dsl_user_op
def ld_generic_b64(ga, *, loc=None, ip=None) -> Int64:
    """generic ``ld.b64`` (SASS ``LD.E.64``)"""
    return Int64(_asm(T.i64(), [_gaddr(ga, loc=loc, ip=ip)], "ld.b64 $0, [$1];", "=l,l", loc=loc, ip=ip))


@dsl_user_op
def ld_generic_v4_b32(ga, *, loc=None, ip=None):
    """generic ``ld.v4.b32`` (SASS ``LD.E.128``; DeepGEMM's 10 x 16 B register pull of each dispatched token) -> 4 x Int32"""
    res = _asm(_struct([T.i32()] * 4), [_gaddr(ga, loc=loc, ip=ip)], "ld.v4.b32 {$0, $1, $2, $3}, [$4];",
               "=r,=r,=r,=r,l", loc=loc, ip=ip)  # fmt: skip
    return _unpack(res, 4, T.i32(), Int32, loc=loc, ip=ip)


@dsl_user_op
def st_generic_b32(ga, v, *, loc=None, ip=None) -> None:
    """generic ``st.b32`` (SASS ``ST.E``; remote ``src_token_topk_idx`` stores)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "st.b32 [$0], $1;", "l,r", loc=loc, ip=ip)


@dsl_user_op
def st_generic_b64(ga, v, *, loc=None, ip=None) -> None:
    """generic ``st.b64`` (SASS ``ST.E.64``; remote ``expert_recv_count`` / ``peer_grid_idx`` stores)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip), _i64(v, loc=loc, ip=ip)], "st.b64 [$0], $1;", "l,l", loc=loc, ip=ip)


@dsl_user_op
def st_generic_v4_b32(ga, x, y, z, w, *, loc=None, ip=None) -> None:
    """generic ``st.v4.b32`` (SASS ``ST.E.128``; local ring stores of the pulled token registers)"""
    _asm(None, [_gaddr(ga, loc=loc, ip=ip)] + [_i32(v, loc=loc, ip=ip) for v in (x, y, z, w)],
         "st.v4.b32 [$0], {$1, $2, $3, $4};", "l,r,r,r,r", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def atom_add_shared_u32(sp, v, *, loc=None, ip=None) -> Int32:
    """``atom.shared.cta.add.u32`` -> old value (CUDA ``atomicAdd_block`` on shared memory; SASS ``ATOMS.ADD``;
    dispatch token counting K:427,443)."""
    return Int32(_asm(T.i32(), [_saddr(sp, loc=loc, ip=ip), _i32(v, loc=loc, ip=ip)], "atom.shared.cta.add.u32 $0, [$1], $2;", "=r,r,r", loc=loc, ip=ip))


@dsl_user_op
def syncwarp(*, loc=None, ip=None) -> None:
    """``bar.warp.sync 0xffffffff;`` (``__syncwarp()``)."""
    _asm(None, [], "bar.warp.sync 0xffffffff;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def tmem_alloc(slot_sp, ncols: int, cta_group: int = 2, exclusive: bool = True, *, loc=None, ip=None) -> None:
    """``tcgen05.alloc[.exclusive].cta_group::N.sync.aligned.shared::cta.b32 [slot], ncols;`` (full warp; the TMEM address is written
    to shared memory at ``slot_sp``).  DeepGEMM's cubin uses ``.exclusive.cta_group::2`` for all six BLOCK_M (``cute.arch.alloc_tmem`` only
    emits ``.exclusive`` for > 512 columns on sm_107, so use this helper for parity)."""
    assert cta_group in (1, 2)
    ex = ".exclusive" if exclusive else ""
    _asm(None, [_saddr(slot_sp, loc=loc, ip=ip), _i32(ncols, loc=loc, ip=ip)],
         f"tcgen05.alloc{ex}.cta_group::{cta_group}.sync.aligned.shared::cta.b32 [$0], $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def tmem_dealloc(taddr, ncols: int, cta_group: int = 2, exclusive: bool = True, *, loc=None, ip=None) -> None:
    """``tcgen05.dealloc[.exclusive].cta_group::N.sync.aligned.b32 taddr, ncols;`` (full warp; same logical warp on both CTAs for cta_group::2)."""
    assert cta_group in (1, 2)
    ex = ".exclusive" if exclusive else ""
    _asm(None, [_i32(taddr, loc=loc, ip=ip), _i32(ncols, loc=loc, ip=ip)],
         f"tcgen05.dealloc{ex}.cta_group::{cta_group}.sync.aligned.b32 $0, $1;", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


# ---------------------------------------------------------------------------------------------------------------------
# bulk copy (group B)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def tma_load_1d(dst_sp, src_ga, mbar, nbytes, hint=EVICT_FIRST, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint`` (DeepGEMM ``ptx::tma_load_1d``;
    1D combine loads, EVICT_FIRST). The caller arms the barrier with ``arrive_expect_tx(nbytes)`` afterwards, as DeepGEMM does."""
    _asm(
        None,
        [_saddr(dst_sp, loc=loc, ip=ip), _gaddr(src_ga, loc=loc, ip=ip), _i32(nbytes, loc=loc, ip=ip),
         _saddr(mbar, loc=loc, ip=ip), _i64(hint, loc=loc, ip=ip)],
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l,~{memory}",
        loc=loc,
        ip=ip,
    )  # fmt: skip


@dsl_user_op
def tma_store_1d(dst_ga, src_sp, nbytes, hint=EVICT_NORMAL, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint`` (DeepGEMM ``ptx::tma_store_1d``; the caller issues
    ``tma_store_commit()`` afterwards, as ``cute::tma_store_arrive``)."""
    _asm(
        None,
        [_gaddr(dst_ga, loc=loc, ip=ip), _saddr(src_sp, loc=loc, ip=ip), _i32(nbytes, loc=loc, ip=ip), _i64(hint, loc=loc, ip=ip)],
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint [$0], [$1], $2, $3;",
        "l,r,r,l,~{memory}",
        loc=loc,
        ip=ip,
    )  # fmt: skip


@dsl_user_op
def tma_store_commit(*, loc=None, ip=None) -> None:
    """``cp.async.bulk.commit_group;`` (cute ``tma_store_arrive``)"""
    _asm(None, [], "cp.async.bulk.commit_group;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def tma_store_wait(n: int = 0, *, loc=None, ip=None) -> None:
    """``cp.async.bulk.wait_group n;`` WITHOUT ``.read`` (waits for completion of the writes; DeepGEMM ``tma_store_wait<n>``)."""
    assert isinstance(n, int)
    _asm(None, [], f"cp.async.bulk.wait_group {n};", "~{memory}", loc=loc, ip=ip)


# ---------------------------------------------------------------------------------------------------------------------
# warp-level (group C)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def ballot(pred, *, loc=None, ip=None) -> Int32:
    """``vote.sync.ballot.b32`` over the full warp (``__ballot_sync(0xffffffff, pred)``)."""
    return Int32(_asm(T.i32(), [_i32(Int32(pred), loc=loc, ip=ip)],
                      "{\n\t.reg .pred p;\n\tsetp.ne.b32 p, $1, 0;\n\tvote.sync.ballot.b32 $0, p, 0xffffffff;\n\t}",
                      "=r,r", loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def all_sync(pred, *, loc=None, ip=None) -> Boolean:
    """``vote.sync.all.pred`` (``__all_sync(0xffffffff, pred)``)."""
    r = _asm(T.i32(), [_i32(Int32(pred), loc=loc, ip=ip)],
             "{\n\t.reg .pred p, q;\n\tsetp.ne.b32 p, $1, 0;\n\tvote.sync.all.pred q, p, 0xffffffff;\n\tselp.u32 $0, 1, 0, q;\n\t}",
             "=r,r", loc=loc, ip=ip)  # fmt: skip
    return Int32(r) != Int32(0)


@dsl_user_op
def any_sync(pred, *, loc=None, ip=None) -> Boolean:
    """``vote.sync.any.pred`` (``__any_sync(0xffffffff, pred)``)."""
    r = _asm(T.i32(), [_i32(Int32(pred), loc=loc, ip=ip)],
             "{\n\t.reg .pred p, q;\n\tsetp.ne.b32 p, $1, 0;\n\tvote.sync.any.pred q, p, 0xffffffff;\n\tselp.u32 $0, 1, 0, q;\n\t}",
             "=r,r", loc=loc, ip=ip)  # fmt: skip
    return Int32(r) != Int32(0)


def _shfl(kind: str, v, x, *, loc=None, ip=None):
    """32-bit shuffle of an Int32/Float32 value (result has the type of ``v``)."""
    is_f32 = isinstance(v, Float32)
    src = _f32(v, loc=loc, ip=ip) if is_f32 else _i32(v, loc=loc, ip=ip)
    if is_f32:
        src = llvm.bitcast(T.i32(), src, loc=loc, ip=ip)
    clamp = "0x0" if kind == "up" else "0x1f"
    r = _asm(T.i32(), [src, _i32(x, loc=loc, ip=ip)], f"shfl.sync.{kind}.b32 $0, $1, $2, {clamp}, 0xffffffff;", "=r,r,r",
             side_effects=True, loc=loc, ip=ip)  # fmt: skip
    if is_f32:
        return Float32(llvm.bitcast(T.f32(), r, loc=loc, ip=ip))
    return Int32(r)


@dsl_user_op
def shfl_idx(v, lane, *, loc=None, ip=None):
    """``shfl.sync.idx.b32`` (``__shfl_sync(0xffffffff, v, lane)``; DeepGEMM ``ptx::exchange``)."""
    return _shfl("idx", v, lane, loc=loc, ip=ip)


@dsl_user_op
def shfl_xor(v, mask, *, loc=None, ip=None):
    """``shfl.sync.bfly.b32`` (``__shfl_xor_sync(0xffffffff, v, mask)``)."""
    return _shfl("bfly", v, mask, loc=loc, ip=ip)


@dsl_user_op
def shfl_up(v, delta, *, loc=None, ip=None):
    """``shfl.sync.up.b32`` (``__shfl_up_sync(0xffffffff, v, delta)``)."""
    return _shfl("up", v, delta, loc=loc, ip=ip)


@dsl_user_op
def redux_add_u32(v, *, loc=None, ip=None) -> Int32:
    """``redux.sync.add.u32`` (``__reduce_add_sync(0xffffffff, v)``)."""
    return Int32(_asm(T.i32(), [_i32(v, loc=loc, ip=ip)], "redux.sync.add.u32 $0, $1, 0xffffffff;", "=r,r", loc=loc, ip=ip))


@dsl_user_op
def redux_add_s32(v, *, loc=None, ip=None) -> Int32:
    """``redux.sync.add.s32``"""
    return Int32(_asm(T.i32(), [_i32(v, loc=loc, ip=ip)], "redux.sync.add.s32 $0, $1, 0xffffffff;", "=r,r", loc=loc, ip=ip))


@dsl_user_op
def redux_min_u32(v, *, loc=None, ip=None) -> Int32:
    """``redux.sync.min.u32`` (``__reduce_min_sync(0xffffffff, v)``)."""
    return Int32(_asm(T.i32(), [_i32(v, loc=loc, ip=ip)], "redux.sync.min.u32 $0, $1, 0xffffffff;", "=r,r", loc=loc, ip=ip))


@dsl_user_op
def redux_max_f32(v, *, loc=None, ip=None) -> Float32:
    """``redux.sync.max.f32`` (DeepGEMM ``ptx::reduce_max_sync``)."""
    return Float32(_asm(T.f32(), [_f32(v, loc=loc, ip=ip)], "redux.sync.max.f32 $0, $1, 0xffffffff;", "=f,f", loc=loc, ip=ip))


@dsl_user_op
def fns_b32(mask, base, offset, *, loc=None, ip=None) -> Int32:
    """``fns.b32 d, mask, base, offset`` (``__fns``: position of the ``offset``-th set bit of ``mask`` counted from ``base``)."""
    return Int32(_asm(T.i32(), [_i32(mask, loc=loc, ip=ip), _i32(base, loc=loc, ip=ip), _i32(offset, loc=loc, ip=ip)],
                      "fns.b32 $0, $1, $2, $3;", "=r,r,r,r", side_effects=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def popc(x, *, loc=None, ip=None) -> Int32:
    """``popc.b32`` (``__popc``)."""
    return Int32(_asm(T.i32(), [_i32(x, loc=loc, ip=ip)], "popc.b32 $0, $1;", "=r,r", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def clz(x, *, loc=None, ip=None) -> Int32:
    """``clz.b32`` (``__clz``)."""
    return Int32(_asm(T.i32(), [_i32(x, loc=loc, ip=ip)], "clz.b32 $0, $1;", "=r,r", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def ffs(x, *, loc=None, ip=None) -> Int32:
    """``__ffs``: 1-based index of the least significant set bit, 0 if ``x == 0`` (= ``32 - clz(x & -x)``)."""
    return Int32(
        _asm(
            T.i32(),
            [_i32(x, loc=loc, ip=ip)],
            "{\n\t.reg .b32 neg, iso, c;\n\tsub.s32 neg, 0, $1;\n\tand.b32 iso, $1, neg;\n\tclz.b32 c, iso;\n\tsub.s32 $0, 32, c;\n\t}",
            "=r,r",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# numerics (group C)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def cvt_rn_bf16x2_f32(hi, lo, *, loc=None, ip=None) -> Int32:
    """``cvt.rn.bf16x2.f32 d, hi, lo``: packed BF16 pair (``hi`` -> bits 31:16, ``lo`` -> bits 15:0); RN, as ``__float22bfloat162_rn({lo, hi})``."""
    return Int32(_asm(T.i32(), [_f32(hi, loc=loc, ip=ip), _f32(lo, loc=loc, ip=ip)], "cvt.rn.bf16x2.f32 $0, $1, $2;", "=r,f,f",
                      side_effects=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def add_rn_f32_bf16(acc, bf16_bits, *, loc=None, ip=None) -> Float32:
    """``add.rn.f32.bf16 acc, bf16, acc`` (DeepGEMM ``ptx::accumulate``): fused (bf16 -> f32 cast + add), RN. ``bf16_bits``: Int32 whose low 16 bits are the BF16."""
    b16 = arith.trunci(T.i16(), _i32(bf16_bits, loc=loc, ip=ip), loc=loc, ip=ip)
    return Float32(_asm(T.f32(), [b16, _f32(acc, loc=loc, ip=ip)], "add.rn.f32.bf16 $0, $1, $0;", "=f,h,0",
                        side_effects=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def accumulate_bf16x2(acc_x, acc_y, packed, *, loc=None, ip=None):
    """Two ``add.rn.f32.bf16`` steps: ``acc_x += lo16(packed)``, ``acc_y += hi16(packed)`` (DeepGEMM ``accumulate(float2&, nv_bfloat162)``)."""
    res = _asm(
        _struct([T.f32(), T.f32()]),
        [_i32(packed, loc=loc, ip=ip), _f32(acc_x, loc=loc, ip=ip), _f32(acc_y, loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $2;\n\t"
        "add.rn.f32.bf16 $0, lo, $3;\n\tadd.rn.f32.bf16 $1, hi, $4;\n\t}",
        "=f,=f,r,f,f",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, T.f32(), Float32, loc=loc, ip=ip)


def _packed_f32x2(op: str, nin: int, args, *, loc=None, ip=None):
    """Packed f32x2 arithmetic: ``op`` in {add, mul, fma}; ``args``: ``nin`` pairs; returns (Float32, Float32)."""
    flat = [_f32(v, loc=loc, ip=ip) for pair in args for v in pair]
    regs = ["va", "vb", "vc"][:nin]
    body = "{\n\t.reg .b64 va, vb, vc, vd;\n\t"
    for k in range(nin):
        body += f"mov.b64 {regs[k]}, {{${2 + 2 * k}, ${3 + 2 * k}}};\n\t"
    body += f"{op}.rn.f32x2 vd, {', '.join(regs)};\n\tmov.b64 {{$0, $1}}, vd;\n\t}}"
    res = _asm(_struct([T.f32(), T.f32()]), flat, body, "=f,=f" + ",f" * (2 * nin), side_effects=False, loc=loc, ip=ip)
    return _unpack(res, 2, T.f32(), Float32, loc=loc, ip=ip)


@dsl_user_op
def fadd2_rn(a, b, *, loc=None, ip=None):
    """``add.rn.f32x2`` (``__fadd2_rn``): ``a``, ``b`` are (x, y) pairs -> (x, y)."""
    return _packed_f32x2("add", 2, (a, b), loc=loc, ip=ip)


@dsl_user_op
def fmul2_rn(a, b, *, loc=None, ip=None):
    """``mul.rn.f32x2`` (``__fmul2_rn``)."""
    return _packed_f32x2("mul", 2, (a, b), loc=loc, ip=ip)


@dsl_user_op
def ffma2_rn(a, b, c, *, loc=None, ip=None):
    """``fma.rn.f32x2`` (``__ffma2_rn``): ``a * b + c``."""
    return _packed_f32x2("fma", 3, (a, b, c), loc=loc, ip=ip)


@dsl_user_op
def ex2_approx(x, *, loc=None, ip=None) -> Float32:
    """``ex2.approx.f32`` (NO ftz): the instruction DeepGEMM's ``__expf(x)`` compiles to (``mul.f32 x, 0f3FB8AA3B`` (log2e) then this),
    see its kernel.ptx. Use this one for bit parity with DeepGEMM; subnormal results are produced, not flushed."""
    return Float32(_asm(T.f32(), [_f32(x, loc=loc, ip=ip)], "ex2.approx.f32 $0, $1;", "=f,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def ex2_approx_ftz(x, *, loc=None, ip=None) -> Float32:
    """``ex2.approx.ftz.f32`` (flushes subnormal results; NOT what DeepGEMM emits, see :func:`ex2_approx`)."""
    return Float32(_asm(T.f32(), [_f32(x, loc=loc, ip=ip)], "ex2.approx.ftz.f32 $0, $1;", "=f,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def rcp_approx_ftz(x, *, loc=None, ip=None) -> Float32:
    """``rcp.approx.ftz.f32`` (DeepGEMM ``math::fast_rcp``)."""
    return Float32(_asm(T.f32(), [_f32(x, loc=loc, ip=ip)], "rcp.approx.ftz.f32 $0, $1;", "=f,f", side_effects=False, loc=loc, ip=ip))


@dsl_user_op
def hmin2_bf16x2(a, b, *, loc=None, ip=None) -> Int32:
    """``min.bf16x2`` (``__hmin2`` on ``nv_bfloat162``) on packed Int32 operands."""
    return Int32(_asm(T.i32(), [_i32(a, loc=loc, ip=ip), _i32(b, loc=loc, ip=ip)], "min.bf16x2 $0, $1, $2;", "=r,r,r",
                      side_effects=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def hmax2_bf16x2(a, b, *, loc=None, ip=None) -> Int32:
    """``max.bf16x2`` (``__hmax2`` on ``nv_bfloat162``) on packed Int32 operands."""
    return Int32(_asm(T.i32(), [_i32(a, loc=loc, ip=ip), _i32(b, loc=loc, ip=ip)], "max.bf16x2 $0, $1, $2;", "=r,r,r",
                      side_effects=False, loc=loc, ip=ip))  # fmt: skip


@dsl_user_op
def bf16x2_to_f32x2(r, *, loc=None, ip=None):
    """Exact widening of a packed BF16 pair -> (lo, hi) as (Float32, Float32) (``__bfloat1622float2``: 2 x ``cvt.f32.bf16``)."""
    res = _asm(
        _struct([T.f32(), T.f32()]),
        [_i32(r, loc=loc, ip=ip)],
        "{\n\t.reg .b16 lo, hi;\n\tmov.b32 {lo, hi}, $2;\n\tcvt.f32.bf16 $0, lo;\n\tcvt.f32.bf16 $1, hi;\n\t}",
        "=f,=f,r",
        side_effects=False,
        loc=loc,
        ip=ip,
    )
    return _unpack(res, 2, T.f32(), Float32, loc=loc, ip=ip)


@dsl_user_op
def cvt_e4m3x4_f32(x, y, z, w, *, loc=None, ip=None) -> Int32:
    """``__nv_fp8x4_e4m3(float4{x, y, z, w})``: 4 x E4M3 (RN, satfinite) packed little-endian (x -> byte 0 ... w -> byte 3);
    2 x ``cvt.rn.satfinite.e4m3x2.f32``."""
    return Int32(
        _asm(
            T.i32(),
            [_f32(v, loc=loc, ip=ip) for v in (x, y, z, w)],
            "{\n\t.reg .b16 lo, hi;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;\n\t"
            "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;\n\t"
            "mov.b32 $0, {lo, hi};\n\t}",
            "=r,f,f,f,f",
            side_effects=False,
            loc=loc,
            ip=ip,
        )
    )


# ---------------------------------------------------------------------------------------------------------------------
# tensor memory loads / stmatrix (group C)
# ---------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def tmem_ld_16x256b_x1(taddr, *, loc=None, ip=None):
    """``tcgen05.ld.sync.aligned.16x256b.x1.b32`` (``SM100_TMEM_LOAD_16dp256b1x``) -> 4 x Int32. ``taddr`` = lane << 16 | column."""
    res = _asm(_struct([T.i32()] * 4), [_i32(taddr, loc=loc, ip=ip)],
               "tcgen05.ld.sync.aligned.16x256b.x1.b32 {$0, $1, $2, $3}, [$4];", "=r,=r,=r,=r,r", loc=loc, ip=ip)  # fmt: skip
    return _unpack(res, 4, T.i32(), Int32, loc=loc, ip=ip)


@dsl_user_op
def tmem_ld_wait(*, loc=None, ip=None) -> None:
    """``tcgen05.wait::ld.sync.aligned;`` (``cutlass::arch::fence_view_async_tmem_load``)."""
    _asm(None, [], "tcgen05.wait::ld.sync.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def tmem_st_16x256b_x1(taddr, r0, r1, r2, r3, *, loc=None, ip=None) -> None:
    """``tcgen05.st.sync.aligned.16x256b.x1.b32`` (extra helper: populate TMEM in unit tests / replay utilities)."""
    _asm(None, [_i32(taddr, loc=loc, ip=ip)] + [_i32(v, loc=loc, ip=ip) for v in (r0, r1, r2, r3)],
         "tcgen05.st.sync.aligned.16x256b.x1.b32 [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def tmem_st_wait(*, loc=None, ip=None) -> None:
    """``tcgen05.wait::st.sync.aligned;``"""
    _asm(None, [], "tcgen05.wait::st.sync.aligned;", "~{memory}", loc=loc, ip=ip)


@dsl_user_op
def stmatrix_m16n8_x1_trans_b8(sp, r, *, loc=None, ip=None) -> None:
    """``stmatrix.sync.aligned.m16n8.x1.trans.shared.b8`` (DeepGEMM ``SM100_U8x4_STSM_T``)."""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(r, loc=loc, ip=ip)],
         "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [$0], {$1};", "r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def stmatrix_m16n8_x2_trans_b8(sp, r0, r1, *, loc=None, ip=None) -> None:
    """``stmatrix.sync.aligned.m16n8.x2.trans.shared.b8`` (DeepGEMM ``SM100_U8x8_STSM_T``)."""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(r0, loc=loc, ip=ip), _i32(r1, loc=loc, ip=ip)],
         "stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [$0], {$1, $2};", "r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def stmatrix_m8n8_x4_trans_b16(sp, r0, r1, r2, r3, *, loc=None, ip=None) -> None:
    """``stmatrix.sync.aligned.x4.m8n8.shared.b16.trans`` (DeepGEMM ``SM90_U32x4_STSM_T``)."""
    _asm(None, [_saddr(sp, loc=loc, ip=ip)] + [_i32(v, loc=loc, ip=ip) for v in (r0, r1, r2, r3)],
         "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans [$0], {$1, $2, $3, $4};", "r,r,r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def stmatrix_m8n8_x2_trans_b16(sp, r0, r1, *, loc=None, ip=None) -> None:
    """``stmatrix.sync.aligned.x2.m8n8.shared.b16.trans`` (DeepGEMM ``SM90_U32x2_STSM_T``)."""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(r0, loc=loc, ip=ip), _i32(r1, loc=loc, ip=ip)],
         "stmatrix.sync.aligned.x2.m8n8.shared.b16.trans [$0], {$1, $2};", "r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip


@dsl_user_op
def stmatrix_m8n8_x2_b16(sp, r0, r1, *, loc=None, ip=None) -> None:
    """``stmatrix.sync.aligned.x2.m8n8.shared.b16`` (DeepGEMM ``SM90_U32x2_STSM_N``)."""
    _asm(None, [_saddr(sp, loc=loc, ip=ip), _i32(r0, loc=loc, ip=ip), _i32(r1, loc=loc, ip=ip)],
         "stmatrix.sync.aligned.x2.m8n8.shared.b16 [$0], {$1, $2};", "r,r,r,~{memory}", loc=loc, ip=ip)  # fmt: skip
