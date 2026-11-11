# SM107 MegaMoE plugin for vLLM

> **NVFP4 W4A4 variant:** the `vllm_nvfp4_megamoe` package in this repository runs
> NVFP4 checkpoints (`nvidia/DeepSeek-V4.1-Flash-NVFP4`) on NVIDIA's Rubin MegaMoE
> kernel with the shared expert fused, and keeps MXFP4 layers (such as the DSpark
> draft) on this plugin. Enable it with `VLLM_PLUGINS=nvfp4_megamoe`; see
> [NVFP4.md](NVFP4.md).

Opt-in Rubin SM107 MegaMoE: MXFP8 activations, MXFP4 routed weights, BF16
output, fused FP8 shared expert, and die-local routed weights. The kernel
and its host support are included in this wheel; no FlashInfer fork is fetched.

## Installation

Use an existing compatible vLLM ARM64 CUDA 13.4 environment. The validated
base is vLLM `1b2a666391f2a24228a1436b64d547c34cc6b6ce`
with the small generic MegaMoE plugin hooks, CuTe DSL 4.8, cuda-python 13.4,
TVM-FFI 0.1.11, and the nightly's stock FlashInfer 0.7.2. This package deliberately
does not resolve or upgrade the serving stack.

```bash
uv pip install --no-deps .
VLLM_PLUGINS=sm107_megamoe vllm serve MODEL \
  --kernel-config '{"moe_backend":"deep_gemm_mega_moe"}'
```

Add `sm107_megamoe` to the comma-separated `VLLM_PLUGINS` allowlist if other
plugins are needed. Installation alone does not activate this plugin. Exclude
the name from that variable to use native DeepGEMM. Restart the server when
switching: weight ownership and CUDA graphs are established during initialization.

The vLLM hook makes `DeepGemmMegaMoEExperts` a `PluggableLayer` and routes its
backend calls through `_import_backend()`. It adds no SM107-specific backend
name, kernel source, or installer to vLLM. Both the target and DSpark draft
model use the registered expert subclass.

The exact one-file hook is included in
[`patches/vllm-megamoe-plugin.patch`](patches/vllm-megamoe-plugin.patch).
Apply it to the matching vLLM checkout if your version does not already have
the hook. The validated fork branch is
[`zihaok_rubin_flashinfer_mega_moe_rebase`](https://github.com/zihaok/vllm/tree/zihaok_rubin_flashinfer_mega_moe_rebase);
the hook is included in
[`e616282`](https://github.com/zihaok/vllm/commit/e61628259b61b17459d65c9db67bdc50ff58c62c).

## Runtime behavior

The plugin owns routed-weight localization, reload handling, symmetric buffers,
and precompilation of all supported BLOCK_M variants before CUDA graph capture.
Routing, input staging, shared-expert integration, and model execution stay in
vLLM. Supported model geometries are checked before allocation; see
[EPLB (Rubin disagg campaign)](#eplb-rubin-disagg-campaign) for EPLB and the
geometries admitted for it. The validated model is DeepSeek-V4.1-Flash with its
DSpark draft.

The stock FlashInfer installation provides only the generic CuTe DSL disk-cache
helper used by this plugin. The SM107 kernel implementation is in this package.
The existing `FLASHINFER_WORKSPACE_BASE` cache location is honored. Quack 0.6.5's
incompatible argument converter is adapted in memory before kernel compilation;
neither Quack nor FlashInfer installation files are rewritten.

## Provenance and validation

`UPSTREAM.json` records the imported FlashInfer revision and per-file hashes.
GPU kernel sources are retained verbatim. Attribution and licenses are included.
The full DeepSeek-V4.1-Flash run measured 100,368 output tokens/s versus 100,507
for the previous FlashInfer integration, a 0.14% difference. All 489,672 turns
completed successfully. See [`VALIDATION.md`](VALIDATION.md) for the settings,
accuracy, operator checks, and measurement details.

## EPLB (Rubin disagg campaign)

Branch `eplb-wip` (fast-forwarded into `rubin-disagg`) adds EPLB support and
admits the expert-parallel geometries that the Rubin disaggregated-serving
campaign validated on GPUs. Deviations from the published plugin:

- `backend/heuristics.py`, a file recorded in `UPSTREAM.json`: the admission
  gate also admits the exact `(num_experts, EP size)` pairs of the new
  `VALIDATED_EP_GEOMETRIES` table. Nothing else in the file changed. The file
  is part of the kernel compile-cache key, so kernels cached by an earlier
  build are compiled once more (about 1.2 s per BLOCK_M variant). The kernel
  sources under `backend/kernel/` still match `UPSTREAM.json`.
- `vllm_sm107_megamoe/experts.py`: EPLB support, described below.
- `VALIDATION.md` describes a plugin that refused EPLB and these geometries;
  its results do not cover them.

### Admitted geometries

As published, the gate admits H=5120, I=2304 with E in {32, 96, 192, 384}
(top-6) or E=128 (top-3), at any EP size with at least 12 experts per rank.
It now also admits exactly these pairs:

| Top-k | Physical experts | EP | Experts per rank | Campaign use |
|---:|---:|---:|---:|---|
| 6 | 388 | 4 | 97 | target with EPLB, 4 redundant experts |
| 6 | 392 | 8 | 49 | target with EPLB, 8 redundant experts |
| 6 | 400 | 8, 16 | 50, 25 | target with EPLB, 16 redundant experts |
| 6 | 416 | 32 | 13 | target with EPLB, 32 redundant experts |
| 3 | 128 | 16, 32 | 8, 4 | DSpark draft at DEP16 and DEP32 |
| 3 | 132 | 4 | 33 | draft with EPLB, 4 redundant experts |
| 3 | 136 | 8 | 17 | draft with EPLB, 8 redundant experts |
| 3 | 144 | 8, 16 | 18, 9 | draft with EPLB, 16 redundant experts |
| 3 | 160 | 32 | 5 | draft with EPLB, 32 redundant experts |

Every pair ran bit-exact against the independent DeepGEMM 3ec0eb4 C++ Rubin
MegaMoE: max_abs 0 and equal expert receive statistics at every point, all six
BLOCK_M variants, M from 1 to T plus a random M per rank, Gaussian routing plus
Zipf-with-masked-routes and hot-expert extras, one fused shared expert, SwiGLU
clamp 10, die-local routed weights. Evidence: Slurm jobs 854696 and 854741,
report `Rubin_disagg_campaign/validation/plugin_eplb/kernel_validation.md` (raw
rows in `all_checks.jsonl`). Other EP sizes for these expert counts did not
run and stay refused, for example E=400 at EP4, which DEP4 with 16 redundant
experts would need.

As in the published gate, the token budget is not checked. The pairs ran with
`max-num-batched-tokens` 2560 (aligned T=3840); E=128 at EP16 and EP32, and
E=416 and E=160 at EP32, also ran with 5120 (T=5760).

For EPLB this means `num_redundant_experts` 16 at DEP8 or DEP16, 32 at DEP32,
8 at DEP8, or 4 at DEP4; the target and the draft get the same number. Without
EPLB the draft now also runs at DEP16 and DEP32.

### How EPLB works with the plugin

- vLLM keeps routing, replica selection, and expert-load recording. Its input
  staging maps logical expert ids to physical replicas before the kernel runs,
  and the kernel addresses experts by physical id. `set_eplb_state` and
  `update_expert_map` are vLLM's native methods.
- `get_expert_weights()` returns `(num_local_experts, -1)` views of the
  kernel's buffers. Each die-local `w13_weight` and `w2_weight` gives one view
  per die slice; the slices are contiguous and padded apart. The MN-major
  scales use vLLM's transpose view. Shared-expert parameters are excluded.
  Each view is checked to cover exactly its die slice or scale, in place.
- EPLB applies the same row moves to every view of a layer, so both halves of
  an expert move together and stay on their dies. Rows are copied in place, so
  weight addresses, CUDA graphs, and bound kernel launches stay valid.
- A weight reload after EPLB moved experts writes each expert into the slots of
  its current placement; before EPLB is set up, the loaders use vLLM's initial
  placement. A reload while an async EPLB cycle is in flight is not supported.
- Elastic EP is refused: the kernel variants and symmetric buffer are built for
  a fixed EP size and physical expert count.

### EPLB communicators

| Communicator | Status | How it moves rows |
|---|---|---|
| `symm_mem` | Recommended, with `use_async: true` | Copies weight rows into its own symmetric receive buffer; waits with host barriers only |
| `torch_gloo` | Accepted, synchronous or async | Stages rows through host memory; slower |
| `torch_nccl`, `pynccl` | Synchronous EPLB only, not recommended; refused with `use_async: true` | NCCL send and receive of the rows. Each rearrangement blocks decoding while every layer moves, and vLLM itself avoids `torch_nccl` by default because batched isend/irecv can hang under load. |
| `nixl` | Refused | Registers the live weights for RDMA reads by peers. Die-local weights are driver VMM allocations without an exportable handle type, outside PyTorch's allocator. |
| `torch_xccl` | Refused | XPU backend; on CUDA it would run NCCL transfers on the async EPLB stream |

vLLM selects `nixl` whenever NIXL is installed, as in the disagg image, and no
communicator is set. Set the communicator explicitly, for example at DEP8 or
DEP16 (use 32 redundant experts at DEP32):

```bash
--enable-eplb --eplb-config '{"communicator": "symm_mem", "num_redundant_experts": 16, "use_async": true}'
```

### Validation status

- Kernel: the admitted pairs above, bit-exact against DeepGEMM 3ec0eb4.
- GPU EPLB tests: `tests/test_eplb_gpu.py`, opt-in with
  `SM107_MEGAMOE_GPU_TESTS=1` on one 4-GPU Rubin node. The die-local memory of
  `backend.localize`, the plugin kernel through the layer's `forward`, and
  vLLM's EPLB code are real; only vLLM's group coordinators are stand-ins.
  - Sync (E=132, top-3, EP4): EplbState's profile pass, then
    `rearrange_expert_weights_inplace` to a random placement with `symm_mem`,
    `torch_gloo`, `torch_nccl` and `pynccl` in turn. After each move every row
    is byte-equal to the expected expert, die slices are still in their
    locality domain, padding is untouched, parameter addresses and kernel
    launch records are unchanged, and the kernel output is bit-identical to the
    output before EPLB. A reload afterwards follows the moved placement.
  - Async (E=388, top-6, EP4, `symm_mem` and `torch_gloo`): EplbState's worker
    thread and side stream compute placements from the recorded load and move
    one layer at a time while CUDA graphs of the layers replay. Every replay
    stays bit-identical through two full rearrangement cycles.
  - In the serving image
    `inferencex-cu134-dev-20261019-dynamo-20261010-vllmimg-31e27f4-plugin-92f4b8b.sqsh`
    with this branch on `PYTHONPATH` (Slurm job 855478, logs in
    `Rubin_disagg_campaign/validation/plugin_eplb/integ/logs`), all three tests
    passed. The sync rounds changed 258 to 263 of 264 physical slots; the async
    runs needed 27 steps (`symm_mem`) and 1427 steps (`torch_gloo`) for two
    cycles and changed 686 of 776 slots. A negative control that commits a new
    placement without moving the weights changed 99.9 % of the output elements
    and failed the byte check on both layers.
- CPU tests: `tests/test_eplb.py` runs vLLM's EPLB code on gloo ranks with host
  tensors in the plugin's padded per-die layout:
  `rearrange_expert_weights_inplace` with the profile pass, and
  `transfer_layer` followed by `move_from_buffer`, called one after the other
  on one thread, with and without migration batching. The `symm_mem` case uses
  host shared memory in place of the CUDA rendezvous. The file also covers the
  refusals, the view check, the reload, and canaries for vLLM's rule against
  async NCCL and for `symm_mem`'s host-only barriers. `tests/test_admission.py`
  checks the admission table. In job 855478, 99 tests passed and the 3 GPU
  tests were skipped without the opt-in.
- Not yet validated: end-to-end EPLB serving (a disagg run with
  `--enable-eplb`), `symm_mem` pushes across nodes (MNNVL), the DSpark draft
  as a second EPLB model in the same worker, and long runs under load.
