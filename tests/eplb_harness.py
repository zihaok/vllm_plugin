# SPDX-License-Identifier: Apache-2.0
"""Helpers for ``test_eplb.py``; spawned ranks import this module by name.

``localize`` is replaced by a host tensor with the plugin's per-die layout, so
no GPU kernel runs. vLLM's EPLB code and communicator classes are used as they
are in the serving image.
"""

import itertools
import math
from datetime import timedelta
from types import SimpleNamespace

import torch
from vllm.model_executor.models.interfaces import MixtureOfExperts

HIDDEN, INTERMEDIATE = 256, 384
# Locality-domain allocation granularity on Rubin; die slices are padded to it.
GRANULARITY = 2 << 20
CANARY = 0xA5
LOCALIZED_WEIGHTS = ("w13_weight", "w2_weight")


def localize(t: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Host stand-in for ``backend.localize``.

    ``(*B, N, K)`` -> ``(2, *B, N/2, K)`` with the strides of the driver
    allocation: each die slice is contiguous and padded to the granularity.
    The padding is filled with ``CANARY``.
    """
    dim %= t.dim()
    sizes = [*t.shape[:dim], t.shape[dim] // 2, *t.shape[dim + 1 :]]
    elem = t.element_size()
    slice_bytes = -(-max(math.prod(sizes) * elem, 1) // GRANULARITY) * GRANULARITY
    flat = torch.full((2 * slice_bytes,), CANARY, dtype=torch.uint8)
    strides = [math.prod(sizes[i + 1 :]) for i in range(len(sizes))]
    out = torch.as_strided(
        flat.view(t.dtype), [2, *sizes], [slice_bytes // elem, *strides]
    )
    out.copy_(t.unflatten(dim, (2, -1)).movedim(dim, 0))
    return out


def padding_intact(t: torch.Tensor) -> bool:
    """Whether the bytes after each die slice still hold ``CANARY``."""
    assert t.element_size() == 1
    raw = torch.as_strided(t.view(torch.uint8), (t.untyped_storage().nbytes(),), (1,), 0)
    used = math.prod(t.shape[1:])
    step = t.stride(0)
    return all(
        bool((raw[d * step + used : (d + 1) * step] == CANARY).all())
        for d in range(t.shape[0])
    )


def shared_mlp() -> SimpleNamespace:
    """An unsharded MXFP8 shared MLP that the native class fuses (S=1)."""

    def linear(rows: int, cols: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=torch.zeros(rows, cols, dtype=torch.uint8).view(
                torch.float8_e4m3fn
            ),
            weight_scale=torch.zeros(rows, cols // 32, dtype=torch.uint8).view(
                torch.float8_e8m0fnu
            ),
        )

    return SimpleNamespace(
        gate_up_proj=linear(2 * INTERMEDIATE, HIDDEN),
        down_proj=linear(HIDDEN, INTERMEDIATE),
    )


def fake_backend() -> SimpleNamespace:
    from vllm_sm107_megamoe import api

    return SimpleNamespace(
        is_localization_available=lambda: True,
        get_unsupported_reason_for_mega_moe=lambda *a, **k: None,
        localize=localize,
        # Only inspected: the native class fuses the shared expert when the
        # backend API takes shared-expert arguments.
        get_symm_buffer_for_mega_moe=api.get_symm_buffer_for_mega_moe,
        fp8_fp4_mega_moe=api.fp8_fp4_mega_moe,
        get_block_m_for_mega_moe=api.get_block_m_for_mega_moe,
    )


def install_fakes(patch, ep_size: int) -> None:
    """Swap the GPU backend and EP group; ``patch`` is ``setattr``-like."""
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_sm107_megamoe import experts

    backend = fake_backend()
    patch(native, "check_deep_gemm_mega_moe_supported", lambda *a: None)
    patch(experts.SM107MegaMoEExperts, "_import_backend", staticmethod(lambda: backend))
    patch(experts, "get_ep_group", lambda: SimpleNamespace(world_size=ep_size))


def make_config(
    enable_eplb: bool = False,
    communicator: str | None = None,
    elastic_ep: bool = False,
    use_async: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=enable_eplb,
            enable_elastic_ep=elastic_ep,
            eplb_config=SimpleNamespace(
                communicator=communicator, use_async=use_async
            ),
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


def make_experts(
    cls,
    prefix: str,
    *,
    num_experts: int,
    num_local_experts: int,
    start: int,
    num_logical: int,
    shared: bool,
    config: SimpleNamespace | None = None,
):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    return cls(
        config or make_config(),
        prefix=prefix,
        num_experts=num_experts,
        num_local_experts=num_local_experts,
        experts_start_idx=start,
        num_logical_experts=num_logical,
        top_k=1,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        weight_format=native.MEGA_MOE_MXFP4,
        activation=native.MegaMoEActivation(),
        shared_experts=shared_mlp() if shared else None,
    )


def checkpoint(seed: int, num_logical: int) -> list:
    """Random MXFP4 checkpoint shards ``(expert, shard, name, tensor)``."""
    generator = torch.Generator().manual_seed(seed)
    shards = []
    for expert in range(num_logical):
        for shard in ("w1", "w2", "w3"):
            rows, cols = (
                (HIDDEN, INTERMEDIATE) if shard == "w2" else (INTERMEDIATE, HIDDEN)
            )
            prefix = "w2" if shard == "w2" else "w13"
            for suffix, width, dtype in (
                ("weight", cols // 2, torch.uint8),
                ("weight_scale", cols // 32, torch.float8_e8m0fnu),
            ):
                value = torch.randint(
                    0, 256, (rows, width), generator=generator, dtype=torch.uint8
                ).view(dtype)
                shards.append((expert, shard, f"{prefix}_{suffix}", value))
    return shards


def load(layer, shards: list) -> None:
    for expert, shard, name, value in shards:
        layer.weight_loader(getattr(layer, name), value, f"experts.{name}", shard, expert)


def assert_holds(layer, row: int, reference, expert: int) -> None:
    """Local row ``row`` of ``layer`` holds logical ``expert`` of ``reference``.

    ``reference`` is the native class with every logical expert in the native
    kernel layout. A die slice ``d`` holds rows ``[d*N/2, (d+1)*N/2)``.
    """
    for name, ref in reference.named_parameters(recurse=False):
        want = ref.data[expert]
        got = getattr(layer, name).data
        if name in LOCALIZED_WEIGHTS:
            for die, half in enumerate(want.chunk(2, dim=0)):
                assert torch.equal(got[die, row], half), (name, die, row, expert)
        else:
            assert torch.equal(got[row], want), (name, row, expert)


class MoEModel(MixtureOfExperts):
    """Uses the protocol's own ``set_eplb_state``, as EplbState.add_model does."""

    def __init__(self, layers: list) -> None:
        self.moe_layers = layers
        self.num_moe_layers = len(layers)


def symm_mem_offsets(layer_views: list) -> list[int]:
    """Byte offsets of SymmMemEplbCommunicator's per-tensor receive buffers."""
    from vllm.utils.math_utils import round_up

    return [0, *itertools.accumulate(round_up(w.nbytes, 256) for w in layer_views)]


def cpu_symm_mem_communicator(layer_views: list, cpu_group, rank: int, buffers: list):
    """vLLM's SymmMemEplbCommunicator with host shared memory as the symmetric
    buffers: only the CUDA rendezvous in ``__init__`` is replaced (same buffer
    carving); row choice, pushes and CPU barriers are vLLM's code."""
    from vllm.distributed.eplb.eplb_communicator import SymmMemEplbCommunicator

    offsets = symm_mem_offsets(layer_views)

    def as_layer(buffer: torch.Tensor) -> list[torch.Tensor]:
        return [
            buffer[o : o + w.nbytes].view(w.dtype).view(w.shape)
            for o, w in zip(offsets, layer_views)
        ]

    communicator = object.__new__(SymmMemEplbCommunicator)
    communicator._handle = None
    communicator._peer_buffers = [as_layer(b) for b in buffers]
    communicator._buffer = as_layer(buffers[rank])
    communicator._stream = None
    communicator._cpu_group = cpu_group
    communicator._recv_rows = []
    communicator._sends = []
    return communicator


def rank_main(
    rank: int,
    world_size: int,
    init_file: str,
    cases: list,
    num_logical: int,
    old: list[int],
    new: list[list[int]],
    symm_buffers: list | None,
) -> None:
    """One EP rank: load experts, rearrange with vLLM's EPLB code, check bytes.

    ``cases`` are ``(communicator, mode)``; mode ``sync`` is
    ``rearrange_expert_weights_inplace``; ``async`` and ``async_batched`` are
    the async worker's ``transfer_layer`` (without / with migration batching)
    followed by the main thread's ``move_from_buffer`` for every layer.
    """
    import torch.distributed as dist
    from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
    from vllm.distributed.eplb.eplb_state import compute_logical_maps
    from vllm.distributed.eplb.migration_scheduler import (
        schedule_migration_batches_for_layers,
    )
    from vllm.distributed.eplb.rebalance_execute import (
        move_from_buffer,
        rearrange_expert_weights_inplace,
        transfer_layer,
    )
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_sm107_megamoe.experts import SM107MegaMoEExperts

    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=2),
    )
    try:
        install_fakes(setattr, world_size)
        group = dist.group.WORLD
        num_layers = len(new)
        num_physical = len(old)
        num_local = num_physical // world_size
        first = rank * num_local
        old_maps = torch.tensor([old] * num_layers)
        new_maps = torch.tensor(new)
        references = []
        for i in range(num_layers):
            reference = make_experts(
                native.DeepGemmMegaMoEExperts,
                f"reference.{i}",
                num_experts=num_logical,
                num_local_experts=num_logical,
                start=0,
                num_logical=num_logical,
                shared=False,
            )
            load(reference, checkpoint(i, num_logical))
            references.append(reference)

        for communicator_name, mode in cases:
            layers = []
            for i in range(num_layers):
                layer = make_experts(
                    SM107MegaMoEExperts,
                    f"model.layers.{i}.experts",
                    num_experts=num_physical,
                    num_local_experts=num_local,
                    start=first,
                    num_logical=num_logical,
                    shared=True,
                    config=make_config(True, communicator_name),
                )
                assert layer.has_fused_shared_experts
                load(layer, checkpoint(i, num_logical))
                # Initial placement: physical p holds logical old[p], including
                # the redundant replicas.
                for row in range(num_local):
                    assert_holds(layer, row, references[i], old[first + row])
                layers.append(layer)
            pointers = [[p.data_ptr() for p in layer.parameters()] for layer in layers]

            model = MoEModel(layers)
            logical_to_physical, replica_count = compute_logical_maps(
                old_maps, num_logical
            )
            model.set_eplb_state(
                torch.zeros(num_layers, num_physical, dtype=torch.int32),
                logical_to_physical,
                replica_count,
            )
            for i, layer in enumerate(layers):
                assert layer.eplb_state.logical_to_physical_map.data_ptr() == (
                    logical_to_physical[i].data_ptr()
                )
            assert all(len(w) == 6 for w in model.expert_weights)

            if communicator_name == "symm_mem":
                communicator = cpu_symm_mem_communicator(
                    model.expert_weights[0], group, rank, symm_buffers
                )
                buffer = communicator.owned_expert_buffer
            else:
                # EplbState.add_model for every communicator but symm_mem.
                buffer = [torch.empty_like(w) for w in model.expert_weights[0]]
                communicator = create_eplb_communicator(
                    SimpleNamespace(cpu_group=group, device_group=group),
                    communicator_name,
                    model.expert_weights,
                    buffer,
                )
                buffer = communicator.owned_expert_buffer or buffer

            if mode == "sync":
                if communicator.needs_profile_buffer_reservation:
                    # The profile run's dummy all_gather reads the die views.
                    rearrange_expert_weights_inplace(
                        old_maps,
                        new_maps,
                        model.expert_weights,
                        buffer,
                        group,
                        communicator,
                        is_profile=True,
                    )
                    for i, layer in enumerate(layers):
                        for row in range(num_local):
                            assert_holds(layer, row, references[i], old[first + row])
                rearrange_expert_weights_inplace(
                    old_maps,
                    new_maps,
                    model.expert_weights,
                    buffer,
                    group,
                    communicator,
                )
            else:
                batches = None
                if mode == "async_batched":
                    batches = schedule_migration_batches_for_layers(
                        num_local, old_maps.numpy(), new_maps.numpy()
                    )
                for i in range(num_layers):
                    metadata = transfer_layer(
                        old_layer_indices=old_maps[i],
                        new_layer_indices=new_maps[i],
                        expert_weights=model.expert_weights[i],
                        expert_weights_buffer=buffer,
                        ep_group=group,
                        communicator=communicator,
                        layer_idx=i,
                        enable_migration_batching=batches is not None,
                        migration_batches=None if batches is None else batches[i],
                    )
                    move_from_buffer(
                        expert_weights=model.expert_weights[i],
                        expert_weights_buffers=buffer,
                        transfer_metadata=metadata,
                        new_indices=new_maps[i].numpy(),
                        ep_rank=rank,
                    )

            for i, layer in enumerate(layers):
                for row in range(num_local):
                    assert_holds(layer, row, references[i], new[i][first + row])
                for name in LOCALIZED_WEIGHTS:
                    assert padding_intact(getattr(layer, name).data), name
                assert [p.data_ptr() for p in layer.parameters()] == pointers[i]
            dist.barrier()
    finally:
        dist.destroy_process_group()
