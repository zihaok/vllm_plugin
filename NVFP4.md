# W4A4 NVFP4 MegaMoE (`vllm_nvfp4_megamoe`)

An opt-in vLLM plugin that runs the NVFP4 target MoE layers of
`nvidia/DeepSeek-V4.1-Flash-NVFP4` on Rubin (SM107) with NVIDIA's CuTe DSL
`BlockScaledSwapAbMegaMoeKernel`. Each layer runs:

- routed experts as W4A4 NVFP4 (E2M1 values, E4M3 scale per 16 values, FP32 global scales);
- the shared expert as MXFP8 x MXFP8, fused into the same persistent kernel;
- the shared-expert output added in the kernel's top-k reduce.

Every other MegaMoE layer stays on the SM107 plugin of this repository
(`vllm_sm107_megamoe`), which this one extends. That includes MXFP4 checkpoints and
the MXFP4 DSpark draft bundled in the NVFP4 checkpoint.

## What runs per NVFP4 MoE layer

Two launches on the current stream, both capturable in CUDA graphs:

1. **Staging (Triton, `staging.py`)** reads each BF16 token once and writes:
   - the NVFP4 routed input: packed E2M1 data and plain E4M3 scales, quantized against the layer's static input global scale;
   - the MXFP8 shared-expert input, with UE8M0 scales in the kernel's 32x4x4 swizzled layout;
   - the routing (int32 ids, fp32 weights). Rows past the live tokens get id `-1`.

   The tokens are read as 32-bit words and FP4 pairs are packed with `cvt.rn.satfinite.e2m1x2.f32`. It runs at about 78% of the device copy bandwidth.
2. **NVIDIA's kernel** runs three device kernels:
   - the router;
   - the persistent fused kernel: pull dispatch, NVFP4 FC1, SwiGLU with clamp, the top-k weight, NVFP4 requantization, FC2 and the MXFP8 shared FFN;
   - the top-k reduce, which adds the shared-expert output.

The kernel's `max_tokens_per_rank` is compile-time, and its router and shared FFN
always cover all of those rows. The plugin therefore compiles one variant per token
bucket. The default buckets are 256, 1024, 2048, 3072, 4096, 4608, 5632, 6144 and
`round_up(max_num_batched_tokens, 128)`, overridable with `VLLM_NVFP4_MEGAMOE_BUCKETS`.
All variants compile before CUDA graph capture, during the first eager forward
(vLLM's profile run). Each call uses the smallest bucket that holds the most tokens
any DP rank runs, so every EP rank launches the same variant.

Memory:
- One torch symmetric-memory slab per process holds the staged tokens, their scales, the top-k scores and every bucket's symmetric workspace. Peers address it through one `SymmetricBufferHost` peer-offset table. NVSHMEM is not needed.
- Workspaces are zeroed once. Each launch resets its own counters behind an all-rank NVLink barrier, so consecutive layers reuse them.

## Weights and scales

The DeepSeek-V4.1 loader indexes eight NVFP4 tensors per routed expert: `weight`,
`weight_scale`, `weight_scale_2` and `input_scale` of `w1`/`w3` and `w2`. The plugin
registers exactly those and writes each checkpoint shard straight into the kernel
layout (`layout.py`):

- FC1 rows go in 16-row gate/up stripes;
- block scales go in the flat 32x4x4 atom swizzle;
- the shared expert's 32x32 UE8M0 block scales are expanded to 1x32.

Loading needs no post-load transform, and reloads keep the parameter addresses. The
GPU check compares every byte against FlashInfer's
`preprocess_prequantized_block_scaled_weights`.

Global scales, per layer:
- `A1` is the max `w1`/`w3` `input_scale` over all experts. Tokens are quantized once, before dispatch, so every expert must accept them.
- `A2` is the max `w2` `input_scale`, the same choice as vLLM's native NVFP4 MegaMoE.

From those:
- tokens are quantized with `1 / A1`;
- `fc1_alpha = A1 * weight_scale_2` (`w1` and `w3` must match, and they do in this checkpoint);
- the FC1 output, with the top-k weight applied, is requantized with `1 / A2`;
- `fc2_alpha = A2 * w2 weight_scale_2`.

`VLLM_NVFP4_MEGAMOE_L2_MARGIN` (default 1.0) adds headroom to `A2`.

## Use

```bash
export VLLM_PLUGINS=nvfp4_megamoe                       # not together with sm107_megamoe
export VLLM_NVFP4_MEGAMOE_KERNEL_DIR=/path/to/cutedsl_megamoe   # shared-expert fusion branch
vllm serve nvidia/DeepSeek-V4.1-Flash-NVFP4 --enable-expert-parallel \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}' ...
```

Requirements:
- the vLLM MegaMoE plugin hooks (see README.md), CuTe DSL 4.8, a torch with `torch.distributed._symmetric_memory`, and EP over the GPUs of one NVLink domain;
- the fused shared expert must stay enabled (unsharded shared MLP, `VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION` unset);
- EPLB is not supported on NVFP4 layers.

## The kernel is not in this repository

`BlockScaledSwapAbMegaMoeKernel` and its helpers come from NVIDIA's
`cutedsl_megamoe`, a version with the fused shared expert
(`shared_expert`/`reduce_shared_output` problem fields). They are imported at run
time from `VLLM_NVFP4_MEGAMOE_KERNEL_DIR`, and nothing from that repository is
vendored here. Ask the kernel's owners for access. If the checkout ships the EP4 tuning
table (`ci/gb200_perf_guard/best_params.py`), per-bucket kernel settings are read
from it. Otherwise generic defaults apply. `VLLM_NVFP4_MEGAMOE_IMPL` overrides them
with a JSON object.

## Validation

`tests/nvfp4_gpu_check.py` runs under `torchrun --nproc_per_node=4` on one Rubin
node. It checks:

- the weight layout against FlashInfer's converter;
- the staged bytes against a torch quantizer;
- the layer output against FP32 math at several token counts, for two layers;
- that a two-layer CUDA graph replay is bit-identical to eager.

It also times the layer.
