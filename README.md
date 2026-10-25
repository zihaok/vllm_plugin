# SM107 MegaMoE plugin for vLLM

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
vLLM. Supported model geometries are checked before allocation. EPLB support
is new and not yet validated on GPUs; see
[EPLB (Rubin disagg campaign)](#eplb-rubin-disagg-campaign). The validated
model is DeepSeek-V4.1-Flash with its DSpark draft.

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

The `eplb-wip` branch adds EPLB support for the Rubin disaggregated-serving
campaign. It deviates from the published plugin as follows:

- `UPSTREAM.json`: no recorded file changed. The vendored kernel, its host
  files, and the admission table in `backend/heuristics.py` are byte-identical
  to the imported revision. The only runtime change is in
  `vllm_sm107_megamoe/experts.py`; tests and this section were added.
- `VALIDATION.md`: those runs used a plugin that refused EPLB. None of its GPU,
  kernel, or end-to-end results cover EPLB.

How EPLB works with the plugin:

- vLLM keeps routing, replica selection, and expert-load recording. Its input
  staging maps logical expert ids to physical replicas before the kernel runs,
  and the kernel addresses experts by physical id. `set_eplb_state` and
  `update_expert_map` are vLLM's native methods.
- `get_expert_weights()` returns `(num_local_experts, -1)` views that alias the
  kernel's buffers. Each die-local `w13_weight` and `w2_weight` gives one view
  per die slice; the slices are contiguous and padded apart. The MN-major
  scales use vLLM's transpose view. Shared-expert parameters are excluded.
- EPLB applies the same row moves to every view of a layer, so both halves of
  an expert move together and stay on their dies. Rows are copied in place, so
  weight addresses, CUDA graphs, and bound kernel launches stay valid.

EPLB communicators:

| Communicator | Status | How it moves rows |
|---|---|---|
| `symm_mem` | Accepted; recommended, including with `use_async: true` | Copies weight rows into its own symmetric receive buffer; CPU barriers only |
| `torch_gloo` | Accepted | Stages rows through host memory |
| `torch_nccl`, `pynccl` | Accepted; vLLM allows them only with `use_async: false` | NCCL send and receive from the rows |
| `nixl` | Refused | Registers the live weights for RDMA reads by peers. Die-local weights are driver VMM allocations without an exportable handle type, outside PyTorch's allocator. |
| `torch_xccl` | Refused | XPU backend; on CUDA it would run NCCL transfers on the async EPLB stream |

vLLM selects `nixl` whenever NIXL is installed, as in the disagg image, and no
communicator is set. Set the communicator explicitly, for example:

```bash
--enable-eplb --eplb-config '{"communicator": "symm_mem", "num_redundant_experts": 16, "use_async": true}'
```

Elastic EP is refused: the kernel variants and symmetric buffer are built for a
fixed EP size and physical expert count.

The admission table is unchanged and checks the physical expert count.
Counts that include redundant experts, such as 384 + 16 or 128 + 16, are
refused at engine start until they are validated on GPUs and added to
`backend/heuristics.py`. Until then, EPLB runs only with
`num_redundant_experts: 0` at admitted geometries.

Validation status:

- CPU tests (`tests/test_eplb.py`) run vLLM's own EPLB code from the serving
  image (vLLM `0.32.0rc1.dev281+g49cb2bbf7` with the plugin hook) on gloo
  ranks. Die-local weights are simulated with host tensors in the plugin's
  padded per-die layout. The tests cover `rearrange_expert_weights_inplace`
  (including the profile pass) and the async worker's `transfer_layer`, with
  and without migration batching, followed by `move_from_buffer`. They use the
  `torch_nccl` (isend/irecv over gloo), `torch_gloo`, and `symm_mem`
  communicator classes; `symm_mem` uses host shared memory in place of the
  CUDA rendezvous. The layer maps include local moves, swaps, remote receives,
  and redundant replicas. Every die slice and scale of every expert row is
  compared byte for byte with the native layout of the logical expert that
  belongs there. The tests also check that views alias the kernel buffers,
  that die-slice padding is untouched, and that refused setups fail before
  allocation. In the serving image
  `inferencex-cu134-dev-20261019-dynamo-20261010-vllmimg-31e27f4-plugin-92f4b8b.sqsh`
  on one Rubin node (Slurm job 854750, commit `3345bba`), all 19 tests passed:
  16 EPLB tests and the 3 existing tests. A negative control in the same job
  gave EPLB clones instead of views (all of them, only the die-1 slices, or
  only the scales); each variant failed the byte comparison.
- Not yet validated: kernel correctness at physical expert counts with
  redundant experts, and end-to-end EPLB serving on GPUs.
