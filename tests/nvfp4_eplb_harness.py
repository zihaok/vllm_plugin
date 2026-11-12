# SPDX-License-Identifier: Apache-2.0
"""Helpers for ``test_nvfp4_eplb.py``; spawned ranks import this module by name.

NVFP4 MegaMoE layers (``VllmNvfp4MegaMoEExperts``) are built on the host at a
small shape, loaded through their weight loaders, and moved by vLLM's EPLB code
and communicator classes as they are in the serving image. No GPU kernel runs:
the plugin's staging and NVIDIA's kernel are not involved, only the weights, the
per-slot alpha buffers and the views EPLB moves.
"""

import os
from datetime import timedelta
from types import SimpleNamespace

import torch
from vllm.model_executor.models.interfaces import MixtureOfExperts

# Smallest shape the NVFP4 layouts take: FC1 halves of 64-row multiples, 128x4 scale atoms.
HIDDEN, INTERMEDIATE = 512, 256
TOP_K = 2
NUM_HIDDEN_LAYERS = 4
ALLOW_ENV = "VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY"
# get_expert_weights() order of an NVFP4 layer.
VIEW_NAMES = (
    "w13_weight", "w13_weight_scale", "w13_weight_scale_2", "w13_input_scale",
    "w2_weight", "w2_weight_scale", "w2_weight_scale_2", "w2_input_scale",
    "_fc1_alpha", "_fc2_alpha",
)


class Coordinator:
    """What the plugin, the communicators and EplbState use of vLLM's GroupCoordinator."""

    def __init__(self, world_size: int, rank: int = 0, cpu_group=None, device_group=None) -> None:
        self.world_size = world_size
        self.rank_in_group = rank
        self.cpu_group = cpu_group
        self.device_group = device_group if device_group is not None else cpu_group
        self.device_communicator = None

    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist

        dist.all_reduce(tensor, group=self.cpu_group)
        return tensor


def install_fakes(patch, ep: Coordinator) -> None:
    """Skip the device check and use ``ep`` as the EP group; ``patch`` is ``setattr``-like."""
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_nvfp4_megamoe import experts

    patch(native, "check_deep_gemm_mega_moe_supported", lambda *a: None)
    patch(experts, "get_ep_group", lambda: ep)


def make_config(
    enable_eplb: bool = False,
    communicator: str | None = "symm_mem",
    elastic_ep: bool = False,
    use_async: bool = False,
    not_nvfp4: tuple = (),
) -> SimpleNamespace:
    """A vLLM config stub; target layers whose index is in ``not_nvfp4`` are MXFP4 (ignore list)."""

    def is_nvfp4(prefix: str) -> bool:
        import re

        match = re.search(r"layers\.(\d+)\.ffn\.experts$", prefix)
        return match is None or int(match.group(1)) not in not_nvfp4

    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=elastic_ep,
            eplb_config=SimpleNamespace(communicator=communicator, use_async=use_async),
            num_ubatches=0,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=256),
        compilation_config=SimpleNamespace(static_forward_context={}),
        quant_config=SimpleNamespace(_is_nvfp4_expert_layer=is_nvfp4),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=NUM_HIDDEN_LAYERS)),
    )


def shared_mlp() -> SimpleNamespace:
    """An unsharded MXFP8 shared MLP (checkpoint 32x32 UE8M0 block scales), which NVFP4 layers fuse."""

    def linear(rows: int, cols: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=torch.zeros(rows, cols, dtype=torch.uint8).view(torch.float8_e4m3fn),
            weight_scale=torch.zeros(rows // 32, cols // 32, dtype=torch.uint8),
        )

    return SimpleNamespace(gate_up_proj=linear(2 * INTERMEDIATE, HIDDEN), down_proj=linear(HIDDEN, INTERMEDIATE))


def make_layer(prefix: str, *, num_experts: int, num_local_experts: int, start: int, num_logical: int,
               config: SimpleNamespace | None = None):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

    # The test shape is not a validated geometry: callers set ALLOW_ENV (the tests through monkeypatch).
    layer = VllmNvfp4MegaMoEExperts(
        config or make_config(),
        prefix=prefix,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=start,
        num_logical_experts=num_logical,
        top_k=TOP_K,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        weight_format=native.MEGA_MOE_MXFP4,  # replaced by NVFP4 for NVFP4 layers
        activation=native.MegaMoEActivation(clamp=10.0),
        shared_experts=shared_mlp(),
    )
    assert layer._nvfp4 and layer.has_fused_shared_experts
    return layer


def scalars(seed: int, expert: int) -> dict:
    """Global scales of one logical expert; weight_scale_2 differs between experts by up to 8x, so a slot that
    keeps its previous occupant's alphas is visible."""
    g = torch.Generator().manual_seed(seed * 100_003 + expert)
    u = torch.rand(3, generator=g).tolist()
    return {
        "w1_input_scale": 2e-3 + 1e-3 * u[0],
        "w3_input_scale": 2e-3 + 1e-3 * u[1],
        "w2_input_scale": 5e-3 + 6e-3 * u[2],
        "ws2_13": 2.0 ** -9 * (1 + expert % 8),
        "ws2_2": 2.0 ** -9 * (1 + (3 * expert) % 8),
    }


def checkpoint(seed: int, num_logical: int) -> list:
    """NVFP4 checkpoint shards ``(expert, shard, name, tensor)`` of every logical expert."""
    g = torch.Generator().manual_seed(seed)
    shards = []
    for expert in range(num_logical):
        s = scalars(seed, expert)
        for shard in ("w1", "w2", "w3"):
            rows, cols = (HIDDEN, INTERMEDIATE) if shard == "w2" else (INTERMEDIATE, HIDDEN)
            prefix = "w2" if shard == "w2" else "w13"
            shards.append((expert, shard, f"{prefix}_weight",
                           torch.randint(0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8)))
            shards.append((expert, shard, f"{prefix}_weight_scale",
                           torch.randint(0, 127, (rows, cols // 16), generator=g, dtype=torch.uint8)
                           .view(torch.float8_e4m3fn)))
            ws2 = s["ws2_2" if shard == "w2" else "ws2_13"]
            shards.append((expert, shard, f"{prefix}_weight_scale_2", torch.tensor(ws2, dtype=torch.float32)))
            shards.append((expert, shard, f"{prefix}_input_scale",
                           torch.tensor(s[f"{shard}_input_scale"], dtype=torch.float32)))
    return shards


def load(layer, shards: list) -> None:
    for expert, shard, name, value in shards:
        layer.weight_loader(getattr(layer, name), value, f"experts.{name}", shard, expert)


def reference_layer(num_logical: int, shards: list):
    """Every logical expert in its own slot, in the kernel layout, alphas filled."""
    layer = make_layer("reference.layers.0.ffn.experts", num_experts=num_logical, num_local_experts=num_logical,
                       start=0, num_logical=num_logical)
    load(layer, shards)
    layer._fill_scale_buffers()
    return layer


def assert_holds(layer, row: int, reference, expert: int) -> None:
    """Every EPLB view row of local slot ``row`` holds logical ``expert`` (alphas included)."""
    for name, got, want in zip(VIEW_NAMES, layer.get_expert_weights(), reference.get_expert_weights(),
                               strict=True):
        assert torch.equal(got[row], want[expert]), (layer.prefix, name, row, expert)


def alphas_consistent(layer) -> bool:
    """Every slot's alphas equal its current weight_scale_2 times the layer's global scales."""
    a1 = float(layer._a1_in.max())
    from vllm_nvfp4_megamoe.config import l2_margin

    a2 = float(layer._a2_in.max()) * l2_margin()
    fc1 = layer.w13_weight_scale_2.data[:, 0].float() * a1
    fc2 = layer.w2_weight_scale_2.data.float() * a2
    return torch.equal(layer._fc1_alpha, fc1) and torch.equal(layer._fc2_alpha, fc2)


def addresses(layer) -> list[int]:
    return [p.data_ptr() for p in layer.parameters()] + [b.data_ptr() for b in layer.buffers()]


class MoEModel(MixtureOfExperts):
    """Uses the protocol's own ``set_eplb_state``, as EplbState.add_model does."""

    def __init__(self, layers: list) -> None:
        self.moe_layers = layers
        self.num_moe_layers = len(layers)


def symm_mem_offsets(layer_views: list) -> list[int]:
    """Byte offsets of SymmMemEplbCommunicator's per-tensor receive buffers."""
    import itertools

    from vllm.utils.math_utils import round_up

    return [0, *itertools.accumulate(round_up(w.nbytes, 256) for w in layer_views)]


def cpu_symm_mem_communicator(layer_views: list, coordinator: Coordinator, rank: int, buffers: list):
    """vLLM's SymmMemEplbCommunicator with host shared memory as the symmetric buffers: only the CUDA
    rendezvous in ``__init__`` is replaced (same carving); row choice, pushes and CPU barriers are vLLM's.
    Both the older (``_cpu_group``) and newer (``_group_coordinator``) barrier attributes are set."""
    from vllm.distributed.eplb.eplb_communicator import SymmMemEplbCommunicator

    offsets = symm_mem_offsets(layer_views)

    def as_layer(buffer: torch.Tensor) -> list[torch.Tensor]:
        return [buffer[o: o + w.nbytes].view(w.dtype).view(w.shape) for o, w in zip(offsets, layer_views)]

    communicator = object.__new__(SymmMemEplbCommunicator)
    communicator._handle = None
    communicator._peer_buffers = [as_layer(b) for b in buffers]
    communicator._buffer = as_layer(buffers[rank])
    communicator._stream = None
    communicator._cpu_group = coordinator.cpu_group
    communicator._group_coordinator = coordinator
    communicator._recv_rows = []
    communicator._sends = []
    return communicator


def rank_main(rank: int, world_size: int, init_file: str, cases: list, num_logical: int, old: list[int],
              new: list[list[int]], symm_buffers: list | None) -> None:
    """One EP rank: load NVFP4 layers, rearrange with vLLM's EPLB code, check every view row.

    ``cases`` are ``(communicator, mode)``. Modes: ``sync`` (rearrange_expert_weights_inplace), ``async`` /
    ``async_batched`` (the async worker's transfer_layer + move_from_buffer), and the negative control
    ``sync_without_alphas`` (the alpha views left out of the moved list), which must leave stale alphas.
    """
    import torch.distributed as dist
    from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
    from vllm.distributed.eplb.eplb_state import compute_logical_maps
    from vllm.distributed.eplb.migration_scheduler import schedule_migration_batches_for_layers
    from vllm.distributed.eplb.rebalance_execute import (
        move_from_buffer,
        rearrange_expert_weights_inplace,
        transfer_layer,
    )

    from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

    os.environ[ALLOW_ENV] = "1"  # this spawned process only
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=2))
    try:
        group = dist.group.WORLD
        ep = Coordinator(world_size, rank, cpu_group=group)
        install_fakes(setattr, ep)
        num_layers = len(new)
        num_physical = len(old)
        num_local = num_physical // world_size
        first = rank * num_local
        old_maps = torch.tensor([old] * num_layers)
        new_maps = torch.tensor(new)
        shards = [checkpoint(i, num_logical) for i in range(num_layers)]
        references = [reference_layer(num_logical, shards[i]) for i in range(num_layers)]

        for case, (communicator_name, mode) in enumerate(cases):
            VllmNvfp4MegaMoEExperts._eplb_view_signatures = {}
            layers = []
            case_config = make_config(True, communicator_name)  # one engine: its layers share the VllmConfig
            for i in range(num_layers):
                layer = make_layer(f"case{case}.layers.{i}.ffn.experts", num_experts=num_physical,
                                   num_local_experts=num_local, start=first, num_logical=num_logical,
                                   config=case_config)
                load(layer, shards[i])
                layer._fill_scale_buffers()  # EPLB on: also checks A1/A2 agree over the gloo group
                for row in range(num_local):  # initial placement: physical p holds logical old[p]
                    assert_holds(layer, row, references[i], old[first + row])
                layers.append(layer)
            pointers = [addresses(layer) for layer in layers]

            model = MoEModel(layers)
            logical_to_physical, replica_count = compute_logical_maps(old_maps, num_logical)
            model.set_eplb_state(torch.zeros(num_layers, num_physical, dtype=torch.int32), logical_to_physical,
                                 replica_count)
            assert all(len(w) == len(VIEW_NAMES) for w in model.expert_weights)
            moved_weights = model.expert_weights
            if mode == "sync_without_alphas":
                moved_weights = [views[:-2] for views in model.expert_weights]

            if communicator_name == "symm_mem":
                communicator = cpu_symm_mem_communicator(moved_weights[0], ep, rank, symm_buffers)
                buffer = communicator.owned_expert_buffer
            else:
                buffer = [torch.empty_like(w) for w in moved_weights[0]]
                communicator = create_eplb_communicator(ep, communicator_name, moved_weights, buffer)
                buffer = communicator.owned_expert_buffer or buffer

            if mode in ("sync", "sync_without_alphas"):
                if communicator.needs_profile_buffer_reservation:
                    rearrange_expert_weights_inplace(old_maps, new_maps, moved_weights, buffer, group, communicator,
                                                     is_profile=True)
                    for i, layer in enumerate(layers):
                        for row in range(num_local):
                            assert_holds(layer, row, references[i], old[first + row])
                rearrange_expert_weights_inplace(old_maps, new_maps, moved_weights, buffer, group, communicator)
            else:
                batches = None
                if mode == "async_batched":
                    batches = schedule_migration_batches_for_layers(num_local, old_maps.numpy(), new_maps.numpy())
                for i in range(num_layers):
                    metadata = transfer_layer(
                        old_layer_indices=old_maps[i], new_layer_indices=new_maps[i],
                        expert_weights=moved_weights[i], expert_weights_buffer=buffer, ep_group=group,
                        communicator=communicator, layer_idx=i, enable_migration_batching=batches is not None,
                        migration_batches=None if batches is None else batches[i],
                    )
                    move_from_buffer(expert_weights=moved_weights[i], expert_weights_buffers=buffer,
                                     transfer_metadata=metadata, new_indices=new_maps[i].numpy(), ep_rank=rank)

            if mode == "sync_without_alphas":
                # Negative control: the checkpoint tensors moved, the alphas did not. Some slot of some rank
                # must now hold an expert whose weight_scale_2 disagrees with its alphas.
                stale = torch.tensor([int(not all(alphas_consistent(layer) for layer in layers))])
                dist.all_reduce(stale, op=dist.ReduceOp.MAX, group=group)
                assert int(stale) == 1, "negative control: stale alphas were not detected"
                # A re-finalize recomputes them from the moved weight_scale_2 rows, in place.
                for layer in layers:
                    layer._fill_scale_buffers()
            for i, layer in enumerate(layers):
                for row in range(num_local):
                    assert_holds(layer, row, references[i], new[i][first + row])
                assert alphas_consistent(layer), (communicator_name, mode, i)
                assert addresses(layer) == pointers[i]
            dist.barrier()
    finally:
        dist.destroy_process_group()
