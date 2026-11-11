# SPDX-License-Identifier: Apache-2.0
"""NVFP4 routed experts with a fused MXFP8 shared expert on NVIDIA's Rubin MegaMoE kernel.

``VllmNvfp4MegaMoEExperts`` replaces ``DeepGemmMegaMoEExperts``. A layer is NVFP4
when the model's quantization config says its routed experts are NVFP4
(``_is_nvfp4_expert_layer``, e.g. ``moe_quant_algo: NVFP4`` in
nvidia/DeepSeek-V4.1-Flash-NVFP4) and it is a target layer (index below
``num_hidden_layers``). Every other layer, such as the MXFP4 DSpark draft of that
checkpoint, is a plain ``SM107MegaMoEExperts``.

NVFP4 layers:

* register the eight NVFP4 checkpoint tensors the DeepSeek-V4.1 loader indexes
  (``w13/w2`` ``weight``, ``weight_scale``, ``weight_scale_2``, ``input_scale``)
  and write every shard straight into the kernel layout (``layout.py``); the
  shared expert is stored as MXFP8 with UE8M0 1x32 scales in the same layouts;
* quantize tokens with the layer's static input global scale ``1 / A1``
  (``A1`` = max ``w1``/``w3`` ``input_scale`` over all experts, the same on every
  rank), and fold it into the FC1 alpha: ``fc1_alpha = A1 * weight_scale_2``;
* requantize the FC1 output (top-k weight applied) with ``1 / A2`` (``A2`` = max
  ``w2`` ``input_scale``), folded into ``fc2_alpha = A2 * w2 weight_scale_2``;
* run one staging launch plus the kernel (router, fused routed + shared FFN,
  top-k reduce adding the shared-expert output) per forward.
"""

import logging
import math
import os
import re
import types
from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_NVFP4,
    DeepGemmMegaMoEExperts,
    MegaMoEWeightSpec,
)
from vllm_sm107_megamoe.experts import SM107MegaMoEExperts

from . import layout
from .config import bucket_for, l2_margin, token_buckets
from .staging import scale_multiplier

logger = logging.getLogger("vllm.plugins.nvfp4_megamoe")

_TARGET_EXPERTS_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.ffn\.experts$")
_ROUTED_NAMES = (
    "w13_weight", "w13_weight_scale", "w13_weight_scale_2", "w13_input_scale",
    "w2_weight", "w2_weight_scale", "w2_weight_scale_2", "w2_input_scale",
)


def is_nvfp4_routed_layer(vllm_config: Any, prefix: str) -> bool:
    """Whether the experts at ``prefix`` are NVFP4 target-model experts."""
    quant_config = getattr(vllm_config, "quant_config", None)
    predicate = getattr(quant_config, "_is_nvfp4_expert_layer", None)
    if predicate is None or not predicate(prefix):
        return False
    match = _TARGET_EXPERTS_RE.search(prefix)
    if match is None:
        return False
    num_layers = vllm_config.model_config.hf_text_config.num_hidden_layers
    return int(match.group(1)) < num_layers


class VllmNvfp4MegaMoEExperts(SM107MegaMoEExperts):
    """NVFP4 MegaMoE experts on NVIDIA's kernel; any other layer is SM107MegaMoEExperts."""

    _nvfp4: bool = False
    _nvfp4_runtimes: ClassVar[dict[tuple, Any]] = {}

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any):
        if not is_nvfp4_routed_layer(vllm_config, kwargs.get("prefix", "")):
            super().__init__(vllm_config, **kwargs)
            return
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb or getattr(parallel_config, "enable_elastic_ep", False):
            raise NotImplementedError("NVFP4 MegaMoE plugin: EPLB and elastic EP are not supported")
        activation = kwargs["activation"]
        if activation.name != "swiglu" or activation.alpha != 1.0 or activation.beta != 0.0:
            raise NotImplementedError(f"NVFP4 MegaMoE plugin supports plain or clamped SwiGLU, not {activation}")
        object.__setattr__(self, "_nvfp4", True)  # read by _create_weights during the base __init__
        kwargs["weight_format"] = MEGA_MOE_NVFP4
        # Skips SM107MegaMoEExperts.__init__ (MXFP4 only); the base class checks the SM10x family and shapes.
        DeepGemmMegaMoEExperts.__init__(self, vllm_config, **kwargs)
        if not self.has_fused_shared_experts:
            raise NotImplementedError(
                f"{self.prefix}: the NVFP4 MegaMoE plugin needs the fused MXFP8 shared expert "
                "(unsharded shared MLP, VLLM_DISABLE_DSV4_MEGAMOE_SHARED_EXPERT_FUSION unset)"
            )
        # input_scale of every logical expert (all ranks see all of them): the kernel takes one global
        # FC1-input scale per layer (tokens of any rank reach any expert) and one FC2-input scale per layer.
        self._a1_in = torch.zeros(self.num_logical_experts, 2, dtype=torch.float32)
        self._a2_in = torch.zeros(self.num_logical_experts, dtype=torch.float32)
        self._ws2_13 = torch.zeros(self.num_logical_experts, 2, dtype=torch.float32)
        self._nvfp4_weights = None
        self._nvfp4_kwargs = None
        self._nvfp4_buckets = token_buckets(self.max_num_tokens)
        logger.info(
            "%s: NVFP4 MegaMoE (NVIDIA Rubin kernel, W4A4 routed + fused MXFP8 shared expert), buckets %s",
            self.prefix, self._nvfp4_buckets,
        )

    # ---------------------------------------------------------------------------------------------- weights
    def _kernel_fuses_shared_experts(self, spec: MegaMoEWeightSpec) -> bool:
        if not self._nvfp4:
            return super()._kernel_fuses_shared_experts(spec)
        return spec.dtype == torch.float8_e4m3fn and spec.scale_group_k == 32

    def _create_weights(self) -> None:
        if not self._nvfp4:
            return super()._create_weights()
        e, h, i = self.num_local_experts, self.hidden_size, self.intermediate_size

        def register(name: str, data: torch.Tensor) -> None:
            self.register_parameter(name, nn.Parameter(data, requires_grad=False))

        # Kernel storage: FC1 rows in 16-row gate/up stripes, scales flat 32x4x4-swizzled.
        register("w13_weight", torch.zeros(e, 2 * i, h // 2, dtype=torch.uint8))
        register("w13_weight_scale", torch.zeros(e, 2 * i * (h // 16), dtype=torch.uint8))
        register("w13_weight_scale_2", torch.zeros(e, 2, dtype=torch.float32))
        register("w13_input_scale", torch.zeros(e, 2, dtype=torch.float32))
        register("w2_weight", torch.zeros(e, h, i // 2, dtype=torch.uint8))
        register("w2_weight_scale", torch.zeros(e, h * (i // 16), dtype=torch.uint8))
        register("w2_weight_scale_2", torch.zeros(e, dtype=torch.float32))
        register("w2_input_scale", torch.zeros(e, dtype=torch.float32))
        if self.shared_spec is not None:
            s = i * self.num_shared_experts
            register("shared_l1_weight", torch.zeros(2 * s, h, dtype=torch.uint8))
            register("shared_l1_weight_scale", torch.zeros(2 * s * (h // 32), dtype=torch.uint8))
            register("shared_l2_weight", torch.zeros(h, s, dtype=torch.uint8))
            register("shared_l2_weight_scale", torch.zeros(h * (s // 32), dtype=torch.uint8))

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if not self._nvfp4:
            return super().weight_loader(param, loaded_weight, weight_name, shard_id, expert_id, return_success)
        if shard_id not in ("w1", "w2", "w3"):
            raise ValueError(f"Unsupported expert shard id: {shard_id}")
        is_global = weight_name.endswith(("weight_scale_2", "input_scale"))
        if is_global and not loaded_weight.is_meta:
            value = float(loaded_weight.float().reshape(-1)[0])
            if weight_name.endswith("input_scale"):
                if shard_id == "w2":
                    self._a2_in[expert_id] = value
                else:
                    self._a1_in[expert_id, layout.gate_up_half(shard_id)] = value
            elif shard_id != "w2":
                self._ws2_13[expert_id, layout.gate_up_half(shard_id)] = value
            self._nvfp4_kwargs = None
        local_ids = self._map_global_expert_id(expert_id)
        prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_ids or prefix not in weight_name:
            return False if return_success else None
        h, i = self.hidden_size, self.intermediate_size
        for local_id in local_ids:
            dst = param.data[local_id]
            if is_global:
                value = loaded_weight.reshape(()).to(torch.float32)
                (dst if shard_id == "w2" else dst[layout.gate_up_half(shard_id)]).copy_(value)
            elif weight_name.endswith("weight_scale"):
                scale = loaded_weight.view(torch.uint8)
                if shard_id == "w2":
                    if tuple(scale.shape) != (h, i // 16):
                        raise ValueError(f"w2 weight_scale {tuple(scale.shape)} != {(h, i // 16)}")
                    layout.write_plain(dst, scale)
                else:
                    if tuple(scale.shape) != (i, h // 16):
                        raise ValueError(f"{shard_id} weight_scale {tuple(scale.shape)} != {(i, h // 16)}")
                    layout.write_gate_up_half(dst, scale, layout.gate_up_half(shard_id))
            elif weight_name.endswith("weight"):
                data = loaded_weight.view(torch.uint8)
                if shard_id == "w2":
                    if tuple(data.shape) != (h, i // 2):
                        raise ValueError(f"w2 weight {tuple(data.shape)} != {(h, i // 2)}")
                    dst.copy_(data)
                else:
                    if tuple(data.shape) != (i, h // 2):
                        raise ValueError(f"{shard_id} weight {tuple(data.shape)} != {(i, h // 2)}")
                    layout.write_gate_up_data(dst, data, layout.gate_up_half(shard_id))
            else:
                raise ValueError(f"Unexpected NVFP4 expert tensor {weight_name}")
        self._nvfp4_kwargs = None
        return True if return_success else None

    def shared_weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, weight_name: str) -> None:
        if not self._nvfp4:
            return super().shared_weight_loader(param, loaded_weight, weight_name)
        shard_id = "w1" if ".w1." in weight_name else "w3" if ".w3." in weight_name else "w2"
        h, s = self.hidden_size, self.intermediate_size * self.num_shared_experts
        rows, cols = (h, s) if shard_id == "w2" else (s, h)
        if "weight_scale" in weight_name:
            scale = layout.expand_block_scales(loaded_weight, rows, cols, 32)
            if shard_id == "w2":
                layout.write_plain(param.data, scale)
            else:
                layout.write_gate_up_half(param.data, scale, layout.gate_up_half(shard_id))
        else:
            data = loaded_weight.view(torch.uint8)
            if tuple(data.shape) != (rows, cols):
                raise ValueError(f"shared {shard_id} weight {tuple(data.shape)} != {(rows, cols)}")
            if shard_id == "w2":
                param.data.copy_(data)
            else:
                layout.write_gate_up_data(param.data, data, layout.gate_up_half(shard_id))
        self._nvfp4_kwargs = None

    def _finalize_nvfp4(self) -> None:
        """Global scales and kernel views after loading (eager; before CUDA graph capture)."""
        if self._nvfp4_kwargs is not None:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(f"{self.prefix}: NVFP4 MegaMoE weights must be finalized before graph capture")
        if not torch.equal(self._ws2_13[:, 0], self._ws2_13[:, 1]):
            worst = int((self._ws2_13[:, 0] - self._ws2_13[:, 1]).abs().argmax())
            raise ValueError(
                f"{self.prefix}: expert {worst} has w1 weight_scale_2 {float(self._ws2_13[worst, 0])} != w3 "
                f"{float(self._ws2_13[worst, 1])}; the kernel takes one FC1 scale per expert"
            )
        a1 = float(self._a1_in.max())
        a2 = float(self._a2_in.max()) * l2_margin()
        if not (a1 > 0 and a2 > 0 and math.isfinite(a1) and math.isfinite(a2)):
            raise ValueError(f"{self.prefix}: NVFP4 input scales not loaded (A1={a1}, A2={a2})")
        device = self.w13_weight.device
        ws2_13 = self.w13_weight_scale_2.data[:, 0].to(torch.float32)
        weights = types.SimpleNamespace(
            w13=self.w13_weight.data, w13_sf=self.w13_weight_scale.data,
            w2=self.w2_weight.data, w2_sf=self.w2_weight_scale.data,
            fc1_alpha=(ws2_13 * a1).contiguous(),
            fc2_alpha=(self.w2_weight_scale_2.data.to(torch.float32) * a2).contiguous(),
            fc1_norm_const=torch.full((self.num_local_experts,), 1.0 / a2, dtype=torch.float32, device=device),
            shared_w13=self.shared_l1_weight.data, shared_w13_sf=self.shared_l1_weight_scale.data,
            shared_w2=self.shared_l2_weight.data, shared_w2_sf=self.shared_l2_weight_scale.data,
        )
        self._nvfp4_weights = weights
        self._input_norm_const = 1.0 / a1
        self._scale_mul = scale_multiplier(self._input_norm_const)
        from .runtime import Nvfp4MegaMoERuntime

        self._nvfp4_kwargs = Nvfp4MegaMoERuntime.layer_kwargs(weights)
        logger.debug("%s: NVFP4 MegaMoE A1=%.4e A2=%.4e", self.prefix, a1, a2)

    # ------------------------------------------------------------------------------------------------ runtime
    def get_symm_buffer(self):
        if not self._nvfp4:
            return super().get_symm_buffer()
        runtime = self.__dict__.get("_nvfp4_runtime")
        if runtime is not None and self._nvfp4_kwargs is not None:
            return runtime
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up the NVFP4 MegaMoE plugin before CUDA graph capture")
        self._finalize_nvfp4()
        ep_group = get_ep_group()
        key = (
            id(ep_group.device_group), torch.cuda.current_device(), self.num_experts, self.max_num_tokens,
            self.top_k, self.hidden_size, self.intermediate_size, self.num_shared_experts, self.activation.clamp,
        )
        runtime = self._nvfp4_runtimes.get(key)
        if runtime is None:
            # The SM107 plugin's backend import adapts Quack's CuTe DSL argument converter in memory.
            SM107MegaMoEExperts._import_backend()
            from .runtime import Nvfp4MegaMoERuntime

            runtime = Nvfp4MegaMoERuntime(
                ep_group=ep_group, num_experts=self.num_experts, hidden=self.hidden_size,
                intermediate=self.intermediate_size,
                shared_intermediate=self.intermediate_size * self.num_shared_experts, topk=self.top_k,
                clamp=self.activation.clamp, max_num_tokens=self.max_num_tokens,
            )
            runtime.compile_all(self._nvfp4_kwargs)
            self._nvfp4_runtimes[key] = runtime
        object.__setattr__(self, "_nvfp4_runtime", runtime)
        return runtime

    def get_block_m(self, num_tokens: int) -> int:
        if not self._nvfp4:
            return super().get_block_m(num_tokens)
        return bucket_for(self._nvfp4_buckets, num_tokens)

    def get_expert_weights(self) -> list[torch.Tensor]:
        if self._nvfp4:
            raise NotImplementedError("NVFP4 MegaMoE plugin: EPLB is not supported")
        return super().get_expert_weights()

    def _bucket_tokens(self, num_tokens: int) -> int:
        """Tokens that size the kernel variant: the most any DP rank runs, so every EP rank agrees."""
        if is_forward_context_available():
            dp_metadata = getattr(get_forward_context(), "dp_metadata", None)
            across = getattr(dp_metadata, "num_tokens_across_dp_cpu", None)
            if across is not None and across.numel():
                return max(num_tokens, int(across.max()))
        return num_tokens

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: nn.Module | None = None,
    ) -> torch.Tensor:
        if not self._nvfp4:
            return super().forward(hidden_states, topk_weights, topk_ids, shared_experts)
        if shared_experts is not None:
            raise ValueError(f"{self.prefix} fuses the shared expert into MegaMoE; pass shared_experts=None.")
        num_tokens = hidden_states.shape[0]
        if num_tokens > self.max_num_tokens:
            raise ValueError(f"MegaMoE got {num_tokens} tokens, but was sized for {self.max_num_tokens}.")
        self._synchronize_first_launch()
        runtime = self.get_symm_buffer()
        bucket = runtime.bucket(self._bucket_tokens(num_tokens))
        is_padding = None
        if is_forward_context_available():
            import vllm.envs as envs

            if envs.VLLM_MOE_SKIP_PADDING:
                is_padding = get_forward_context().is_padding
                if is_padding is not None:
                    is_padding = is_padding[:num_tokens]
        if self.capture_fn is not None:
            self.capture_fn(topk_ids)
        if hidden_states.stride(-1) != 1 or hidden_states.stride(0) != self.hidden_size:
            hidden_states = hidden_states.contiguous()
        y = runtime.run(
            hidden_states, topk_ids, topk_weights, is_padding, bucket=bucket, layer_kwargs=self._nvfp4_kwargs,
            norm=self._input_norm_const, scale_mul=self._scale_mul,
        )
        return y[:num_tokens]


VllmNvfp4MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]


def _routed_names() -> tuple[str, ...]:
    return _ROUTED_NAMES


if os.environ.get("VLLM_NVFP4_MEGAMOE_DEBUG") == "1":
    logger.setLevel(logging.DEBUG)
