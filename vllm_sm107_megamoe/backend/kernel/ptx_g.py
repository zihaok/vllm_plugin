# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / ptx/tcgen05.cuh + mma/sm100.cuh; PTX helper patterns derived from FlashInfer's vendored NVIDIA CuTe-DSL MegaMoE (BSD-3-Clause, Copyright (c) NVIDIA CORPORATION & AFFILIATES).
"""PTX forms of the GEMM roles: tcgen05 MMA / UTCCP / commit, descriptor builders, TMEM loads (GEMM core).

Everything here is a thin ``llvm.inline_asm`` wrapper with ``has_side_effects=True``: one PTX statement per call, so
that ptxas sees the same instruction sequence as DeepGEMM's ``asm volatile`` blocks (``K`` = DeepGEMM
``impls/sm107_fp8_fp4_mega_moe.cuh``, ``TCG`` = ``ptx/tcgen05.cuh``, ``MMA`` = ``mma/sm100.cuh``).

Descriptor/idesc helpers are plain integer math (no CuTe fragments): the constants are DeepGEMM's.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Int16, Int32, Int64, Uint32, Uint64
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

# ----------------------------------------------------------------------------------------------------------------------
# Constants (verified against `analysis/desc_calc.py` test vectors in test_sm107_dg_gemm_core_desc.py)
# ----------------------------------------------------------------------------------------------------------------------
EVICT_NORMAL = 0x1000000000000000  # CacheHintSm100::EVICT_NORMAL (tile TMA loads, K: tma::copy default hint)
EVICT_FIRST = 0x12F0000000000000

# SmemDescriptor high 32 bits (K: make_umma_desc; lo = start_address >> 4 (14 bits), LBO = 0):
#   [32,46) SBO >> 4 | [46,48) version = 1 | [61,64) layout type (2 = SW128, 4 = SW64, 0 = none)
SMEM_DESC_HI_SW128_SBO1024 = 0x40004040  # tokens (FP8), SBO = 8 rows * 128 B = 1024 B
SMEM_DESC_HI_SW64_SBO512 = 0x80004020  # packed-FP4 routed weights, SBO = 8 rows * 64 B = 512 B
SMEM_DESC_HI_SF_SBO128 = 0x00004008  # SF (UTCCP source), no swizzle, SBO = 8 * 16 B = 128 B

# InstrDescriptorBlockScaled (K:1030-1044), routed weights (A = E2M1 -> format code 5 under kind::mxf8f6f4) x tokens (B = E4M3):
#   [4,6) b_sf_id | [7,10) a_format | [10,13) b_format | [15] a_major | [16] b_major | [17,23) n_dim = N >> 3 |
#   [23] scale_format (1 = UE8M0) | [24,29) m_dim = M >> 4 | [29,31) a_sf_id | [31] k_size (1 = K=64)
IDESC_A_FORMAT_E2M1 = 5
IDESC_A_FORMAT_E4M3 = 0
IDESC_B_FORMAT_E4M3 = 0


def make_idesc_base(a_format: int = IDESC_A_FORMAT_E2M1, b_format: int = IDESC_B_FORMAT_E4M3, umma_m: int = 256) -> int:
    """Static part of the 32-bit instruction descriptor (everything except N and the SF ids)."""
    desc = 0
    desc |= (a_format & 0x7) << 7
    desc |= (b_format & 0x7) << 10
    desc |= 1 << 23  # scale_format = UE8M0
    desc |= (umma_m >> 4) << 24  # m_dim
    desc |= 1 << 31  # k_size: K = 64 per UMMA (bit 31)
    return desc & 0xFFFFFFFF


IDESC_BASE_ROUTED = make_idesc_base()  # 0x90800280
IDESC_BASE_SHARED = make_idesc_base(IDESC_A_FORMAT_E4M3)  # 0x90800000


def idesc_with_n_and_sf_id(base, aligned_n, sf_id: int):
    """``make_runtime_instr_desc_with_sf_id`` after ``update_instr_desc_with_umma_n`` (MMA:144-158 / K:1077,1167):
    ``n_dim_ = N >> 3`` (runtime), ``a_sf_id_ = b_sf_id_ = sf_id``.  ``aligned_n`` may be a runtime Int32."""
    return base | ((aligned_n >> 3) << 17) | (sf_id << 29) | (sf_id << 4)


# ----------------------------------------------------------------------------------------------------------------------
# Descriptors
# ----------------------------------------------------------------------------------------------------------------------
def smem_desc_lo(smem_addr):
    """Low 32 bits of a SmemDescriptor: ``start_address >> 4`` (14 bits), LBO = 0.  ``smem_addr``: Int32 shared address."""
    return (smem_addr >> 4) & 0x3FFF


@dsl_user_op
def make_desc64(hi: int, lo, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Uint64:
    """64-bit smem descriptor from a compile-time ``hi`` word and a runtime ``lo`` word (Int32/Uint32)."""
    lo64 = Uint64(Uint32(lo))
    return Uint64(hi << 32) | lo64


# ----------------------------------------------------------------------------------------------------------------------
# tcgen05 instructions
# ----------------------------------------------------------------------------------------------------------------------
@dsl_user_op
def utccp_2cta(tmem_col, smem_desc, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> None:
    """``tcgen05.cp.cta_group::2.32x128b.warpx4`` (``cute::SM100_UTCCP_4x32dp128bit_2cta::copy``): SF smem -> TMEM column."""
    llvm.inline_asm(
        None,
        [Int32(tmem_col).ir_value(loc=loc, ip=ip), Uint64(smem_desc).ir_value(loc=loc, ip=ip)],
        "tcgen05.cp.cta_group::2.32x128b.warpx4 [$0], $1;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def mma_mxf8f6f4_2cta(
    tmem_c,
    desc_a,
    desc_b,
    idesc,
    scale_c,
    tmem_sfa,
    tmem_sfb,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> None:
    """``SM100_MMA_MXF8F6F4_2x1SM_SS::fma`` (TCG:67-86): ``tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale``.

    Operand order is the PTX one: ``[tmem_c], desc_a, desc_b, idesc, [tmem_sfa], [tmem_sfb], p`` with ``p = (scale_c != 0)``.
    Under swap-AB DeepGEMM passes the WEIGHT descriptor as ``desc_a`` (+ weight SF column as ``tmem_sfa``) and the TOKEN
    descriptor as ``desc_b`` (+ token SF column as ``tmem_sfb``) (K:1176-1179)."""
    llvm.inline_asm(
        None,
        [
            Int32(tmem_c).ir_value(loc=loc, ip=ip),
            Uint64(desc_a).ir_value(loc=loc, ip=ip),
            Uint64(desc_b).ir_value(loc=loc, ip=ip),
            Uint32(idesc).ir_value(loc=loc, ip=ip),
            Int32(scale_c).ir_value(loc=loc, ip=ip),
            Int32(tmem_sfa).ir_value(loc=loc, ip=ip),
            Int32(tmem_sfb).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t"
        ".reg .pred p;\n\t"
        "setp.ne.b32 p, $4, 0;\n\t"
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale [$0], $1, $2, $3, [$5], [$6], p;\n\t"
        "}\n",
        "r,l,l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def umma_arrive_multicast_2x1sm(
    mbar_addr, cta_mask: int = 0b11, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None
) -> None:
    """``tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [mbar], mask``
    (``cutlass::arch::umma_arrive_multicast_2x1SM``; the caller wraps it in ``elect.sync`` like CUTLASS does).
    ``mbar_addr``: Int32 shared::cta address of the mbarrier (the same offset is signalled in every CTA of the mask)."""
    llvm.inline_asm(
        None,
        [Int32(mbar_addr).ir_value(loc=loc, ip=ip), Int16(cta_mask).ir_value(loc=loc, ip=ip)],
        "{\n\t"
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64 [$0], $1; \n\t"
        "}",
        "r,h",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# ----------------------------------------------------------------------------------------------------------------------
# Epilogue helpers (numerics-critical forms; DeepGEMM `math::` / `ptx::` equivalents)
# ----------------------------------------------------------------------------------------------------------------------
from cutlass import Float32  # noqa: E402


@dsl_user_op
def f32_bits(x, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Int32:
    """``__float_as_uint`` (bit pattern of a Float32 as Int32)."""
    return Int32(llvm.bitcast(T.i32(), Float32(x).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def bits_f32(i, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Float32:
    """``__uint_as_float``."""
    return Float32(llvm.bitcast(T.f32(), Int32(i).ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@dsl_user_op
def fabs_f32(x, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Float32:
    """``fabsf`` (``abs.f32``)."""
    return Float32(llvm.inline_asm(T.f32(), [Float32(x).ir_value(loc=loc, ip=ip)], "abs.f32 $0, $1;", "=f,f", has_side_effects=False,
                                   is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def fmax_f32(a, b, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Float32:
    """``fmaxf`` (``max.f32``: returns the non-NaN operand if one is NaN, like CUDA ``fmaxf``)."""
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], "max.f32 $0, $1, $2;",
                                   "=f,f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def expf_ieee(a, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Float32:
    """CUDA ``expf(a)`` (the ``kFastMath = false`` branch of DeepGEMM's L1 epilogue computes ``expf(-sigmoid_input)``) exactly as nvcc 13.4 expands it with DeepGEMM's JIT flags
    (``--gpu-architecture=sm_100f -O3``, no fast math, no ftz; reproduce with ``run/G_nvcc_ptx.sh``): a fixed 10-instruction PTX sequence - range reduction by one ``fma.rn`` +
    ``cvt.sat`` + ``fma.rm`` + two ``fma.rn`` (hi / lo parts of log2 e), ``ex2.approx.ftz.f32`` of the reduced argument, then the scale 2^k as an integer shift of the ``fma.rm``
    bits and a plain ``mul.f32`` (which also produces the subnormal results ``ex2.approx.ftz`` would flush).  NOT ``ex2.approx(a * log2e)`` (that is ``__expf``, the
    ``kFastMath`` path)."""
    return Float32(llvm.inline_asm(
        T.f32(), [Float32(a).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b32 kb, sb;\n\t.reg .f32 c1, c2, t, ts, kf, kr, ng, r1, r2, sc, e;\n\t"
        "fma.rn.f32 t, $1, 0f3BBB989D, 0f3F000000;\n\t"
        "cvt.sat.f32.f32 ts, t;\n\t"
        "mov.f32 c1, 0f4B400001;\n\t"
        "mov.f32 c2, 0f437C0000;\n\t"
        "fma.rm.f32 kf, ts, c2, c1;\n\t"
        "add.f32 kr, kf, 0fCB40007F;\n\t"
        "neg.f32 ng, kr;\n\t"
        "fma.rn.f32 r1, $1, 0f3FB8AA3B, ng;\n\t"
        "fma.rn.f32 r2, $1, 0f32A57060, r1;\n\t"
        "mov.b32 kb, kf;\n\t"
        "shl.b32 sb, kb, 23;\n\t"
        "mov.b32 sc, sb;\n\t"
        "ex2.approx.ftz.f32 e, r2;\n\t"
        "mul.f32 $0, e, sc;\n\t}",
        "=f,f", has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def div_rn_f32(a, b, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> Float32:
    """IEEE round-to-nearest-even true division ``a / b`` (``div.rn.f32``, what nvcc emits for ``float / float`` without fast math; the ``kFastMath = false`` ``gate / denom``)."""
    return Float32(llvm.inline_asm(T.f32(), [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)], "div.rn.f32 $0, $1, $2;", "=f,f,f",
                                   has_side_effects=False, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip))


@dsl_user_op
def st_global_b8(ga, v, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> None:
    """plain ``st.global.b8`` (byte store of the low 8 bits of ``v``; the UE8M0 SF byte stores of the L1 epilogue, K:1488-1489)."""
    addr = ga.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip) if hasattr(ga, "toint") else Int64(ga).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(None, [addr, Int32(v).ir_value(loc=loc, ip=ip)], "{\n\t.reg .b16 t;\n\tcvt.u16.u32 t, $1;\n\tst.global.b8 [$0], t;\n\t}",
                    "l,r,~{memory}", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)


@cute.jit
def get_ue8m0_sf_exp(amax):
    """``math::get_ue8m0_sf_exp<e4m3>(float amax)`` (common/math.cuh:115-147): UE8M0 exponent by the integer-carry method
    (do NOT use ``cvt.rp.satfinite.ue8m0x2``): ``rounded = (bits + 0x7FFFFF - (0x60 << 16)) >> 23; max(rounded, 105 + 8) - 8``."""
    bits = f32_bits(amax)
    rounded = (bits + (0x7FFFFF - (0x60 << 16))) >> 23  # amax >= 0: logical == arithmetic shift
    return Int32(cutlass.select_(rounded > (105 + 8), rounded, Int32(105 + 8))) - 8


@cute.jit
def get_ue8m0_sf_inv(sf_exp):
    """``math::get_ue8m0_sf_inv<float>``: ``__uint_as_float((254 - sf_exp) << 23)`` (exact power of two)."""
    return bits_f32((254 - sf_exp) << 23)


@dsl_user_op
def tmem_wait_ld_dep(groups, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None):
    """``tcgen05.wait::ld.sync.aligned`` through which every TMEM-loaded register is tied (``"+r"``-style matching constraints):
    uses of the loaded values cannot be scheduled above the wait by the compiler (DeepGEMM relies on NVCC for this ordering).
    ``groups``: list of lists of Int32; returns the same structure."""
    flat = [Int32(v) for g in groups for v in g]
    n = len(flat)
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * n),
        [v.ir_value(loc=loc, ip=ip) for v in flat],
        "tcgen05.wait::ld.sync.aligned;",
        ",".join(["=r"] * n) + "," + ",".join(str(i) for i in range(n)),
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    out = [Int32(llvm.extractvalue(T.i32(), res, [i], loc=loc, ip=ip)) for i in range(n)]
    regrouped = []
    k = 0
    for g in groups:
        regrouped.append(out[k:k + len(g)])
        k += len(g)
    return regrouped


@dsl_user_op
def prefetch_tensormap(desc_ptr, *, loc: Optional[ir.Location] = None, ip: Optional[ir.InsertionPoint] = None) -> None:
    """``prefetch.tensormap [addr]`` of a tensor-map descriptor given as a generic-space pointer (a grid-constant kernel argument; DeepGEMM ``cute::prefetch_tma_descriptor``)."""
    addr = desc_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(None, [addr], "prefetch.tensormap [$0];", "l", has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip)
