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
vLLM. Supported model geometries are checked before allocation; EPLB is not
supported. The validated model is DeepSeek-V4.1-Flash with its DSpark draft.

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
