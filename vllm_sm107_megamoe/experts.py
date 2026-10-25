# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone CuTe DSL SM107 MegaMoE through the native expert interface."""

from typing import Any, ClassVar

import torch
from torch import nn
from vllm.config import VllmConfig
from vllm.distributed import get_ep_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
    MEGA_MOE_MXFP4,
    DeepGemmMegaMoEExperts,
)

logger = init_logger("vllm.plugins.sm107_megamoe")

# Routed parameters that ``backend.localize`` moves into die-local memory.
LOCALIZED_WEIGHTS = ("w13_weight", "w2_weight")

# EPLB communicators that only read and write expert rows through device copies
# or NCCL / gloo staging, so die-local weights work like any device tensor
# (vllm/distributed/eplb/eplb_communicator.py):
#   symm_mem    copies weight rows into its own symmetric receive buffer and
#               synchronizes with CPU barriers only; use it with async EPLB.
#   torch_gloo  stages rows through host memory.
#   torch_nccl, pynccl  send and receive rows with NCCL; synchronous EPLB only.
SUPPORTED_EPLB_COMMUNICATORS = frozenset(
    {"symm_mem", "torch_gloo", "torch_nccl", "pynccl"}
)
# NCCL kernels on the async EPLB stream would run next to the MegaMoE kernel,
# which keeps every SM resident and synchronizes them in software. vLLM's
# EPLBConfig rejects these with use_async as well; the plugin does not rely on it.
_SYNC_ONLY_EPLB_COMMUNICATORS = frozenset({"torch_nccl", "pynccl"})
_UNSUPPORTED_EPLB_COMMUNICATORS = {
    "nixl": (
        "it registers the live expert weights for RDMA reads by peers, and "
        "die-local routed weights are driver VMM allocations without an "
        "exportable handle type, outside PyTorch's allocator"
    ),
    "torch_xccl": (
        "it is the XPU backend; on CUDA it would run NCCL transfers on the "
        "async EPLB stream next to the MegaMoE kernel"
    ),
}


def _check_eplb(parallel_config: Any) -> None:
    """Refuse EPLB setups that cannot move die-local routed weights."""
    if getattr(parallel_config, "enable_elastic_ep", False):
        raise NotImplementedError(
            "Plugin SM107 MegaMoE does not support elastic EP: its kernel "
            "variants and symmetric buffer are built for a fixed EP size and "
            "physical expert count"
        )
    eplb_config = parallel_config.eplb_config
    communicator = eplb_config.communicator
    if communicator in SUPPORTED_EPLB_COMMUNICATORS:
        if communicator in _SYNC_ONLY_EPLB_COMMUNICATORS and getattr(
            eplb_config, "use_async", False
        ):
            raise NotImplementedError(
                f"Plugin SM107 MegaMoE cannot use EPLB communicator "
                f"{communicator!r} with use_async=true: its NCCL kernels would "
                "run on the async EPLB stream next to the MegaMoE kernel, which "
                "needs every SM. Set --eplb-config.communicator=symm_mem."
            )
        return
    reason = _UNSUPPORTED_EPLB_COMMUNICATORS.get(
        communicator, "it has not been checked with die-local routed weights"
    )
    raise NotImplementedError(
        f"Plugin SM107 MegaMoE cannot use EPLB communicator {communicator!r}: "
        f"{reason}. Set --eplb-config.communicator=symm_mem, or torch_gloo; "
        "torch_nccl and pynccl need --eplb-config.use_async=false. vLLM "
        "selects nixl when NIXL is installed and no communicator is set."
    )


class SM107MegaMoEExperts(DeepGemmMegaMoEExperts):
    """MXFP4 routed experts with fused MXFP8 shared experts on Rubin.

    The registered routed parameters own die-local allocations from construction,
    so checkpoint reloads preserve their addresses and graph-capture lifetime.
    Scales and shared weights retain the native MegaMoE layout.

    EPLB uses vLLM's native expert-load recording, replica mapping and
    ``set_eplb_state``. The kernel addresses experts by physical id, and EPLB
    rearranges the routed weights in place through :meth:`get_expert_weights`.
    Once EPLB is set up, a reload writes each expert into the slots of its
    current placement.
    """

    _symm_buffers: ClassVar[dict[tuple, Any]] = {}
    _prepared_variants: ClassVar[set[tuple]] = set()
    _synchronized_ep_groups: ClassVar[set[tuple[int, int]]] = set()

    def __init__(self, vllm_config: VllmConfig, **kwargs: Any):
        parallel_config = vllm_config.parallel_config
        if parallel_config.enable_eplb:
            _check_eplb(parallel_config)
        if kwargs["weight_format"] != MEGA_MOE_MXFP4:
            raise NotImplementedError("Plugin SM107 MegaMoE requires MXFP4 weights")
        if kwargs["activation"].name != "swiglu":
            raise NotImplementedError("Plugin SM107 MegaMoE requires SwiGLU")
        backend = self._import_backend()
        if not backend.is_localization_available():
            raise NotImplementedError(
                "Plugin SM107 MegaMoE requires a two-domain SM107 GPU "
                "and cuda-python 13.4 or later"
            )
        super().__init__(vllm_config, **kwargs)
        logger.info_once(
            "Using plugin CuTe DSL SM107 MegaMoE with die-local routed weights "
            "and %d fused shared expert(s).",
            self.num_shared_experts,
        )
        if parallel_config.enable_eplb:
            logger.info_once(
                "Plugin SM107 MegaMoE EPLB: %d physical experts, %d logical, "
                "communicator %s.",
                self.num_experts,
                self.num_logical_experts,
                parallel_config.eplb_config.communicator,
            )

    @staticmethod
    def _import_backend():
        from . import api as backend
        from .compatibility import ensure_quack_compatibility

        ensure_quack_compatibility()

        return backend

    def _create_weights(self) -> None:
        backend = self._import_backend()
        reason = backend.get_unsupported_reason_for_mega_moe(
            get_ep_group().world_size,
            self.num_experts,
            self.max_num_tokens,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
            fast_math=self.fast_math,
        )
        if reason is not None:
            raise NotImplementedError(f"Plugin SM107 MegaMoE: {reason}")
        super()._create_weights()
        for name in LOCALIZED_WEIGHTS:
            param = getattr(self, name)
            param.data = backend.localize(param.data)

    def _map_global_expert_id(self, expert_id: int) -> list[int]:
        """Local slots of logical expert ``expert_id``, for every weight loader.

        Before EPLB is set up this is vLLM's initial placement (physical ``p``
        holds logical ``p % num_logical_experts``). Afterwards EPLB may have
        moved the expert, so the current replica map decides; reloads are the
        only callers then, and each pays one small device-to-host copy.
        """
        mapping = self.eplb_state.logical_to_physical_map
        if mapping is None:
            return super()._map_global_expert_id(expert_id)
        return [
            p - self.experts_start_idx
            for p in mapping[expert_id].tolist()
            if self.experts_start_idx <= p < self.experts_end_idx
        ]

    def weight_loader(
        self,
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        weight_name: str,
        shard_id: str,
        expert_id: int,
        return_success: bool = False,
    ) -> bool | None:
        if param is not self.w13_weight and param is not self.w2_weight:
            return super().weight_loader(
                param, loaded_weight, weight_name, shard_id, expert_id, return_success
            )
        local_ids = self._map_global_expert_id(expert_id)
        param_prefix = "w2_" if shard_id == "w2" else "w13_"
        if not local_ids or param_prefix not in weight_name:
            return False if return_success else None
        for local_id in local_ids:
            dst = param.data[:, local_id]
            logical = dst.flatten(0, 1).clone()
            self._load_weight(logical, loaded_weight, shard_id, False)
            dst.copy_(logical.view_as(dst))
        return True if return_success else None

    def get_symm_buffer(self):
        buffer = getattr(self, "_runtime_buffer", None)
        if buffer is not None:
            return buffer
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Warm up Plugin SM107 MegaMoE before capture")
        buffer = super().get_symm_buffer()
        key = (
            id(get_ep_group().device_group),
            torch.accelerator.current_device_index(),
            self.num_experts,
            buffer.num_max_tokens_per_rank,
            self.top_k,
            self.hidden_size,
            self.intermediate_size,
            self.num_shared_experts,
            self.activation,
            self.fast_math,
        )
        if key not in self._prepared_variants:
            backend = self._import_backend()
            backend.get_balanced_sm_locality_domains()
            backend.precompile_mega_moe_variants(
                get_ep_group().world_size,
                self.num_experts,
                buffer.num_max_tokens_per_rank,
                self.top_k,
                self.hidden_size,
                self.intermediate_size,
                self.num_shared_experts,
                **self._kernel_kwargs,
                fast_math=self.fast_math,
                per_die=True,
                localized_weights=True,
            )
            torch.distributed.barrier(group=get_ep_group().cpu_group)
            self._prepared_variants.add(key)
        self._runtime_buffer = buffer
        return buffer

    def get_expert_weights(self) -> list[torch.Tensor]:
        """Routed parameters as ``(num_local_experts, -1)`` views for EPLB.

        EPLB moves an expert by copying whole rows of these views in place, so
        each view aliases its parameter with one contiguous row per local
        expert. A die-local ``(2, E_l, N/2, K/2)`` weight gives one view per die
        slice: each slice is contiguous and the slices are padded apart. EPLB
        applies the same row moves to every view of a layer, so both halves of
        an expert move together and stay on their dies. Scales keep the native
        layout and transpose trick. Shared-expert parameters are not routed.
        """
        views = []
        for name, param in self.named_parameters(recurse=False):
            if name.startswith("shared_"):
                continue
            tensor = param.data
            if name in LOCALIZED_WEIGHTS:
                if tensor.dim() != 4 or tensor.shape[1] != self.num_local_experts:
                    raise AssertionError(
                        f"Plugin SM107 MegaMoE EPLB {name}: expected die-local "
                        f"(domains, {self.num_local_experts}, N/2, K/2), got "
                        f"shape={tuple(tensor.shape)}"
                    )
                parts = tensor.unbind(0)
            else:
                parts = (tensor,)
            for die, part in enumerate(parts):
                view = self._eplb_view(name, part)
                # The view must cover exactly this die slice (or the whole
                # scale), in place: EPLB writes the kernel's memory through it.
                offset = die * tensor.stride(0) * tensor.element_size()
                if (
                    view.data_ptr() != tensor.data_ptr() + offset
                    or view.numel() != part.numel()
                    or not view.is_contiguous()
                ):
                    raise AssertionError(
                        f"Plugin SM107 MegaMoE EPLB {name}: the view of part "
                        f"{die} does not cover its memory in place"
                    )
                views.append(view)
        return views

    def _eplb_view(self, name: str, t: torch.Tensor) -> torch.Tensor:
        """A ``(num_local_experts, -1)`` view over contiguous memory of ``t``."""
        e = self.num_local_experts
        if t.shape[0] == e:
            if t.is_contiguous():
                return t.view(e, -1)
            if t.dim() == 3 and t.stride(1) == 1 and t.stride(2) == t.shape[1]:
                # MN-major scales have shape (E, M, N) and memory layout (E, N, M).
                back = t.transpose(1, 2)
                if back.is_contiguous():
                    return back.view(e, -1)
        raise AssertionError(
            f"Plugin SM107 MegaMoE EPLB {name}: expected contiguous memory with "
            f"{e} local experts on dim 0, got shape={tuple(t.shape)} "
            f"stride={tuple(t.stride())} dtype={t.dtype}"
        )


SM107MegaMoEExperts.weight_loader.supports_moe_loading = True  # type: ignore[attr-defined]
