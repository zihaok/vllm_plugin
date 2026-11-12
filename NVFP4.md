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
- with redundant (EPLB) experts, only validated `(physical experts, EP size)` pairs are admitted (see [Geometries](#geometries));
- EPLB is supported with an explicit communicator, see [EPLB](#eplb).

## EPLB

`--enable-eplb` works on NVFP4 layers. Set the communicator explicitly: vLLM picks
`nixl` whenever NIXL is installed (as in the disagg image) and no communicator is
set, and the plugin refuses `nixl`. The DEP16 decode setting:

```bash
--enable-eplb --eplb-config '{"communicator":"symm_mem","num_redundant_experts":16,"window_size":1000,"step_interval":3000,"use_async":true}'
```

With 16 redundant experts, 384 logical experts become 400 physical experts, 25 per
rank at EP16. That pair is validated on GPUs; see [Geometries](#geometries).

### How it works

- **Mapping and load recording in the plugin's staging.** On the DeepSeek-V4/V4.1
  MegaMoE path, no router maps expert ids. The gate hands `forward` logical
  top-k ids. vLLM maps them to physical replicas and records the expert load only
  in its own MegaMoE staging (`prepare_megamoe*_inputs(..., eplb_state=...)`),
  which this plugin replaces. The plugin's staging launch therefore calls vLLM's
  `eplb_map_and_record` (`fused_moe/eplb_triton.py`) in its routing block, exactly
  as vLLM's staging does:
  - the same Knuth hash of the row index picks the replica;
  - the same mask (`live & k < top_k`), padding rows `-1`;
  - load is counted only for rows below `num_unpadded_tokens` while
    `should_record` is set;
  - each route is counted once, because the routing block runs in the first
    column program only.

  The arguments are vLLM's `_eplb_kernel_args`, copied to `vllm_nvfp4_megamoe/eplb.py`
  so a rename of that private helper cannot break the import. They are rebuilt
  from the layer's `eplb_state` on every call. EplbState updates all of those tensors
  in place, so captured CUDA graphs follow every commit. Without EPLB, the
  arguments are inert and the kernel specializes (`HAS_EPLB`) to exactly the
  previous staging.
- **Per-slot alphas move with their experts.** `fc1_alpha = A1 * w13 weight_scale_2`
  and `fc2_alpha = A2 * w2 weight_scale_2` are per local slot. They are
  non-persistent fp32 buffers, allocated when the layer is built and filled in
  place. `get_expert_weights()` returns them after the eight checkpoint tensors (10
  views per layer), so EPLB moves them row by row with the weights. EPLB has no
  post-move hook. An expert that moved without its alphas would run with
  `ws2[old] / ws2[new]` scaling, and the FC1 output would also be clamped and
  requantized wrongly. `A1` and `A2` are maxima over all logical experts, so they do
  not depend on the placement. With EPLB, the plugin checks that every EP rank
  loaded the same values.
- **View checks.** Each view must cover its tensor in place, with the same
  pointer, element count and contiguity. Every layer of a model must return the
  same list of views (count, shapes, dtypes). EPLB sizes its buffers from layer 0
  and zips the other layers against it, which silently truncates a mismatch. For
  the same reason, a checkpoint whose target MoE layers mix NVFP4 and MXFP4
  (through the quantization ignore list) is refused at construction when EPLB is
  on.
- **The kernel is unchanged.** It maps id `i` to rank `i // (E / EP)` over
  contiguous per-rank blocks, drops ids `>= E`, and rebuilds its routing state on
  every launch. It reads weights, scales and alphas through base pointer plus
  local expert stride, so in-place row moves are transparent, even under CUDA
  graphs. Its own DLB helper-slot mechanism stays off (`helper_expert_count = 0`,
  which the runtime asserts). That mechanism assumes a fixed home/helper slot
  layout that vLLM's EPLB does not keep.
- **The DSpark draft** (MXFP4, 128 experts, top-3) stays on the SM107 path. The
  V4.1 draft is not registered with EPLB, but it gets the same number of
  redundant slots: 144 at DEP16, which the SM107 gate admits.

### Communicators

| Communicator | NVFP4 layers |
|---|---|
| `symm_mem` | Recommended, with `use_async: true`. Copies rows into its own symmetric receive buffer and waits on host barriers only. |
| `torch_gloo` | Accepted, sync or async. Stages rows through host memory, so it is slower. |
| `torch_nccl`, `pynccl` | Sync EPLB only. Refused with `use_async: true`, because NCCL kernels on the async stream would run next to the persistent kernel that occupies every SM. |
| `nixl` | Refused. It has not been validated with NVFP4 layers: it registers the live weights and alphas for RDMA reads by peers. |
| `torch_xccl` | Refused (XPU backend). |

Elastic EP is refused. The policy is the SM107 plugin's (`_check_eplb`), copied into
`vllm_nvfp4_megamoe/eplb.py` with NVFP4's own reasons so that this package runs next to
the SM107 package a serving image already has; a test keeps the communicator sets equal.

### Kernel settings by the logical expert count

The tuned tables (the checkout's GB200 table and the Rubin table in `config.py`)
match DeepSeek-V4.1-Flash as `(H, I, E, top-k) = (5120, 2304, 384, 6)`. Matched on
the physical count (400), every bucket would silently fall back to the
`KernelSettings()` defaults. The settings do not depend on E, so the runtime looks
them up by the **logical** count. The kernel's `ProblemDesc` gets the **physical**
count. The logical count is part of the runtime cache key. The runtime logs the
settings it chose for each bucket, and asserts that every compiled kernel has
`expert_count == E_physical`, `local_expert_count == E_physical / EP` and no helper
slots. `VLLM_NVFP4_MEGAMOE_IMPL` overrides still apply on top. The tables were
measured at EP4 with 96 local experts. Retuning the DEP16 decode buckets at 25 per
rank is an optimization, not a correctness item.

**Bit-identity caveat.** Restoring the Rubin table also restores its
`combine_dedup` entries for buckets of 5760 and above. The dedup combine sums each
token's terms per destination rank and rounds them to BF16 again, so outputs change
at BF16 level after every rebalance. That is expected, not a bug. Tests that need
bit-identity across moves force non-dedup settings with
`VLLM_NVFP4_MEGAMOE_RUBIN_TABLE=bitwise` (or `0`, or `VLLM_NVFP4_MEGAMOE_IMPL`), and
refuse to run when a bucket they use has `combine_dedup`. The dedup buckets are
checked to a tolerance at the EPLB geometry by `nvfp4_eplb_gpu_check.py dedup`, a
separate process with the production table: FP32 reference, mapping, load,
determinism and graph replay. At E=384 without EPLB they are checked by
`nvfp4_gpu_check.py`. The dedup local
workspace is larger, about 4.0 instead of 1.2 GiB at 12288, and it is allocated
before KV sizing. Compare KV capacity with the final settings.

### Restrictions

- **Reload under CUDA graphs.** A reload writes each expert into the slots of its
  current placement, including the new `weight_scale_2` rows. The alphas derived
  from them are refilled in place only by the next *eager* forward (the lazy
  re-finalize is Python that graph replays skip). On a decode server with
  `FULL_DECODE_ONLY` graphs, every forward can be a replay, so replays would keep
  the old alphas next to the new scales. After any reload under CUDA graphs, run
  one eager forward on every rank, or re-capture, before serving. `1 / A1` is
  passed to staging by value and baked into captured graphs, so a reload that
  changes a `w1`/`w3` `input_scale` always needs re-capture. A reload while an
  async EPLB cycle is in flight is not supported.
- **Memory.** The `symm_mem` receive buffer holds one layer's 10 views: about 475
  MiB per rank at 25 local experts. Each redundant expert adds about 19.9 MB per
  MoE layer. Both are allocated before KV sizing.
- The replica choice hashes the per-rank row index (vLLM's rule), so small decode
  batches spread poorly over replicas.

## Geometries

NVFP4 layers with redundant (EPLB) experts run only at `(physical experts, EP
size)` pairs that ran on GPUs (`config.NVFP4_GEOMETRIES`, for H=5120, I=2304,
top-6). Other such pairs are refused before allocation unless
`VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY=1`. The validation runs themselves
set that. Layouts without redundant experts (EPLB off, or 0 redundant experts)
are admitted at any EP size, as before EPLB support. An unlisted pair logs one
warning per process.

| Experts | EP | Per rank | Status |
|---:|---:|---:|---|
| 384 | 4 | 96 | validated (`nvfp4_gpu_check.py`, DEP4 campaign) |
| 384 | 16 | 24 | validated (deployed DEP16 decode without EPLB) |
| 388 | 4 | 97 | pending: EPLB with 4 redundant experts at DEP4 |
| 400 | 4 | 100 | validated (`nvfp4_eplb_gpu_check.py` `all` and `dedup` at EP4, job 1044262) |
| 400 | 8 | 50 | pending: EPLB with 16 redundant experts at DEP8 |
| 400 | 16 | 25 | validated (`nvfp4_eplb_gpu_check.py` `all` and `dedup` at EP16 on 4 nodes, job 1044263) |

Set a pending pair to `True` in `NVFP4_GEOMETRIES` once `tests/nvfp4_eplb_gpu_check.py`
passes `geometry`, `dedup` and `move` at that pair, and record the job there. For
`num_redundant_experts` other than 16, add the pair and validate it the same way.

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

EPLB:

- `tests/test_nvfp4_eplb.py` runs on CPU with pytest; tests that need torch or vLLM skip without them. It covers:
  - the communicator policy, the mixed-format refusal and the geometry allowlist;
  - the 10 views, the in-place and signature checks, and refused mutants;
  - the alpha buffers, filled in place and moved by vLLM's real rearrangement code on gloo ranks with every communicator, including the async `transfer_layer` and `move_from_buffer` path. In the negative control, alphas left out of the moved list go stale;
  - the table lookup by logical count, and the inert EPLB arguments.
- `tests/nvfp4_eplb_gpu_check.py` runs under `torchrun` with one process per GPU and prints a PASS/FAIL summary:
  - `staging`: the plugin's staging against vLLM's own staging and a torch model of the mapping (int32 and int64 ids). Without EPLB, it is checked against vLLM's staging and the torch quantizers;
  - `geometry --experts E --logical L` (default 400/384):
    - the runtime is built for E physical experts with the logical tables' settings;
    - the kernel output is checked against the FP32 reference, with heavy load on both replica and home slots;
    - eager is deterministic, graph replay matches eager, and the output is bit-identical to E = L (bitwise table);
  - `dedup`: the same at the production Rubin table, so the `combine_dedup` buckets run at E physical experts (tolerance checks only, in its own process);
  - `move`: sync and async EPLB under CUDA graphs with `symm_mem`, `torch_gloo` and `torch_nccl`, with the negative controls;
  - `all`: staging, then geometry 100/96 at EP4 (25 per rank), then geometry and move 400/384 at EP4, EP8 or EP16.
- `tests/nvfp4_gpu_check.py` stays the EPLB-off gate: E=384, staging exact, dedup buckets. Re-run it with every EPLB validation.

  ```bash
  torchrun --nproc_per_node=4 tests/nvfp4_gpu_check.py --out nvfp4_ep4.json
  torchrun --nproc_per_node=4 tests/nvfp4_eplb_gpu_check.py all --out eplb_ep4.json
  torchrun --nproc_per_node=4 tests/nvfp4_eplb_gpu_check.py dedup --out eplb_dedup_ep4.json
  # 4 nodes x 4 GPUs (DEP16)
  srun -N4 --ntasks-per-node=1 torchrun --nnodes=4 --nproc_per_node=4 --rdzv-backend=c10d \
       --rdzv-endpoint=$MASTER_ADDR:29500 tests/nvfp4_eplb_gpu_check.py all --out eplb_ep16.json
  ```
