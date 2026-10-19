# SPDX-License-Identifier: Apache-2.0
# Port of DeepGEMM (MIT License, Copyright (c) 2025 DeepSeek; see licenses/LICENSE.deepgemm.txt) sm107_fp8_fp4_mega_moe.cuh / mapping manifest.
"""Mapping manifest of one compiled variant (schema ``mmx-manifest-1``).

``build_manifest(build)`` returns the launch geometry, the 35 DeepGEMM template arguments (by DeepGEMM's names), the register split and the TMEM columns of
a ``KernelBuild``; ``write_manifest(build, dest_dir)`` writes ``manifest.json`` next to the kept ``kernel.cubin`` / ``kernel.ptx`` so that a
SASS / PTX mapping checker can compare the port with DeepGEMM's dump of the same variant.  CPU-importable (no cutlass).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

from .config import CLUSTER_SIZE, KernelBuild

WEIGHT_DTYPE_NAME = "cutlass::float_e2m1_t"  # packed FP4 routed weights (cfg.weight_packed_fp4)


def build_manifest(build: KernelBuild) -> Dict[str, Any]:
    cfg = build.cfg
    config = {
        "kNumMaxTokensPerRank": cfg.num_max_tokens_per_rank, "kHidden": cfg.hidden, "kIntermediateHidden": cfg.intermediate_hidden,
        "kNumExperts": cfg.num_experts, "kNumSharedExperts": cfg.num_shared_experts, "kNumTopk": cfg.num_topk,
        "BLOCK_M": cfg.block_m, "BLOCK_N": cfg.block_n, "BLOCK_K": cfg.block_k, "STORE_BLOCK_M": cfg.store_block_m,
        "SF_BLOCK_M": cfg.sf_block_m, "SF_BLOCK_N": cfg.sf_block_n, "kNumRingTokens": cfg.num_ring_tokens,
        "kNumSFRingTokens": cfg.num_sf_ring_tokens, "kNumStages": cfg.num_stages, "kNumBytesPerPull": cfg.num_bytes_per_pull,
        "kNumDispatchThreads": cfg.num_dispatch_threads, "kNumNonEpilogueThreads": cfg.num_non_epilogue_threads,
        "kNumEpilogueThreads": cfg.num_epilogue_threads, "kNumSMs": cfg.num_sms, "kNumRanks": cfg.num_ranks,
        "clamp": cfg.clamp, "alpha": cfg.alpha, "beta": cfg.beta, "kFastMath": cfg.fast_math,
        "weight_dtype": WEIGHT_DTYPE_NAME if cfg.weight_packed_fp4 else "cutlass::float_e4m3_t", "kUmmaK": cfg.umma_k,
        "kTrace": cfg.trace, "kDebugFlags": cfg.debug_flags, "kEpiPrefetchStages": cfg.epi_prefetch_stages,
        "kDispatchPipe": cfg.dispatch_pipe, "kWeightPrefetch": cfg.weight_prefetch,
        "kNumLocalityDomains": cfg.num_locality_domains, "kLocalizedWeights": cfg.localized_weights,
        "kLocalizedSharedWeights": cfg.localized_shared_weights,
    }  # fmt: skip
    return {
        "schema": "mmx-manifest-1",
        "kernel": "cutedsl",
        "arch": build.arch,
        "name": build.name(),
        "mode_mask": build.mode_mask,
        "roles": build.roles,
        "launch": {
            "grid": [cfg.num_sms, 1, 1],
            "block": [cfg.num_threads, 1, 1],
            "cluster": [CLUSTER_SIZE, 1, 1],
            "dynamic_smem_bytes": cfg.smem_bytes,
            "launch_reg_cap": 65536 // cfg.num_threads,
        },
        "config": config,
        "regs": dict(zip(("dispatch", "non_epilogue", "epilogue"), build.regs, strict=False)),
        "tmem_cols": cfg.tmem_cols_alloc,
    }


def write_manifest(build: KernelBuild, dest_dir: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, "manifest.json")
    with open(path, "w") as f:
        json.dump(build_manifest(build), f, indent=1)
    return path
