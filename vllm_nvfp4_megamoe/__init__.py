# SPDX-License-Identifier: Apache-2.0
"""Explicitly enabled W4A4 NVFP4 MegaMoE plugin for vLLM on Rubin (SM107).

NVFP4 target MoE layers (NVFP4 routed experts, fused MXFP8 shared expert) run on
NVIDIA's CuTe DSL ``BlockScaledSwapAbMegaMoeKernel`` (cutedsl_megamoe with the
shared-expert fusion). The kernel is not part of this package: it is imported
from the checkout named by ``VLLM_NVFP4_MEGAMOE_KERNEL_DIR``. Every other MegaMoE
layer (MXFP4 checkpoints, the MXFP4 DSpark draft of the NVFP4 checkpoint) runs on
the ``vllm_sm107_megamoe`` plugin, which this one extends. Importing this module
does not initialize CUDA.
"""

import os

PLUGIN_NAME = "nvfp4_megamoe"
_registered = False


def register() -> None:
    """Register the expert class only when explicitly named in VLLM_PLUGINS."""
    global _registered
    enabled = [name.strip() for name in os.environ.get("VLLM_PLUGINS", "").split(",")]
    if _registered or PLUGIN_NAME not in enabled:
        return
    if "sm107_megamoe" in enabled:
        raise RuntimeError(
            "VLLM_PLUGINS lists both nvfp4_megamoe and sm107_megamoe. Both replace "
            "DeepGemmMegaMoEExperts; nvfp4_megamoe already runs every non-NVFP4 "
            "MegaMoE layer on the SM107 plugin, so list only nvfp4_megamoe."
        )

    from vllm.model_executor.custom_op import PluggableLayer
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    if not issubclass(DeepGemmMegaMoEExperts, PluggableLayer) or not hasattr(
        DeepGemmMegaMoEExperts, "_import_backend"
    ):
        raise RuntimeError(
            "NVFP4 MegaMoE needs the generic PluggableLayer and _import_backend "
            "hooks in DeepGemmMegaMoEExperts (see the vllm_sm107_megamoe README)."
        )
    from .experts import VllmNvfp4MegaMoEExperts

    PluggableLayer.register_oot(name="DeepGemmMegaMoEExperts")(VllmNvfp4MegaMoEExperts)
    _registered = True
