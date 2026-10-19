# SPDX-License-Identifier: Apache-2.0
"""Explicitly enabled SM107 MegaMoE plugin; importing does not initialize CUDA."""

import os

_registered = False


def register() -> None:
    """Register experts only when explicitly named in VLLM_PLUGINS."""
    global _registered
    enabled = os.environ.get("VLLM_PLUGINS", "").split(",")
    if _registered or "sm107_megamoe" not in enabled:
        return

    from vllm.model_executor.custom_op import PluggableLayer
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    if not issubclass(DeepGemmMegaMoEExperts, PluggableLayer) or not hasattr(
        DeepGemmMegaMoEExperts, "_import_backend"
    ):
        raise RuntimeError(
            "SM107 MegaMoE needs the generic PluggableLayer and _import_backend "
            "hooks in DeepGemmMegaMoEExperts. See this plugin's README."
        )
    from .experts import SM107MegaMoEExperts

    PluggableLayer.register_oot(name="DeepGemmMegaMoEExperts")(SM107MegaMoEExperts)
    _registered = True
