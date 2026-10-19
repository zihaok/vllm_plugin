# Validation

Validated on 2026-10-19 with DeepSeek-V4.1-Flash and its DSpark draft on four
Rubin SM107 GPUs, DP4/TP1/EP4.

## Full E2E comparison

| Implementation | Job | Output tokens/s | Accuracy |
|---|---:|---:|---:|
| Previous FlashInfer integration, vLLM `3049448` | 799658 | 100,507.0001 | 0.5317 |
| This plugin, vLLM `e616282` | 802216 | 100,367.5662 | 0.5314 |

The measured throughput difference is **-0.1387%**. Both runs completed all
489,672 turns with zero failed, dropped, empty, errored, undecodable, or missing
results. These measurements establish parity for this workload, not a speedup.

Both used the original full 24-repeat workload, early stopping disabled,
adaptive verification disabled, concurrency 3072, max-num-seqs 768,
max-num-batched-tokens 8190, speculative K=5, decoder replay enabled, and
decoder-replay trim threshold 768. Client image `ghcr.io/mlcommons/endpoints:d3c57f2`,
random seeds, load pattern, and the other serving arguments match. No scheduling
or tuning parameters were changed to recover performance.

The plugin's metric window covers 387,568 requests over 3640.7561 seconds.
The report also verifies the full 489,672-turn workload independently of that
measurement window.

Throughput uses the unchanged endpoint client's metric-window accounting.
Both runs log the existing tool-call tokenizer fallback caused by the model's
missing Hugging Face chat template. The comparison preserves that behavior;
it does not change the client's absolute token-counting method.

Final report SHA-256:

```text
Previous: 3c02538e21b51f44a76c2ee92914831705e044c73d43030b555d2e3482346b69
Plugin:   47f11ff7a36754b041f00390c880d54a20ceab966d9ac7507689f81a5986036d
```

## Operator validation

On four Rubin SM107 GPUs, all eight comparisons against the previous
FlashInfer kernel sources produced bit-identical outputs and expert statistics:

- Target: H=5120, I=2304, E=384, top-k=6, one shared expert, EP=4;
  M=64, 1024, 3072, 5376, 8190.
- Draft: H=5120, I=2304, E=128, top-k=3, one shared expert, EP=4;
  M=64, 1024, 3072.

Target CUDA graph latency, median of per-iteration maximum across four GPUs,
pooled over two passes and both allocation orders:

| M | Previous kernel (us) | Plugin (us) | Change |
|---:|---:|---:|---:|
| 64 | 148.69 | 148.48 | -0.14% |
| 1024 | 200.71 | 200.71 | 0.00% |
| 3072 | 331.01 | 330.64 | -0.11% |
| 5376 | 536.73 | 536.46 | -0.05% |
| 8190 | 784.25 | 782.41 | -0.23% |

These differences are small; this validates equivalent latency, not a kernel
speedup. The GPU kernel files match the imported source hashes in `UPSTREAM.json`.

The image build also passed 40 checkpoint-layout, reload, and input-staging
tests. All six BLOCK_M variants (16, 32, 64, 128, 192, 240) compiled for both
target and draft shapes, including a Quack-first import regression check.
The final image's three package tests passed after a logging-only adjustment.

## Environment preservation

The image starts from the stock ARM64 CUDA 13.4 vLLM nightly. All 13 native
vLLM libraries retain their original hashes, including DeepGEMM and FlashMLA.
The installed FlashInfer 0.7.2 and Quack files remain unchanged. The previous
experimental FlashInfer MegaMoE API is absent. The plugin uses FlashInfer's
generic CuTe DSL cache helper and owns its kernel sources and compatibility
adapter.
