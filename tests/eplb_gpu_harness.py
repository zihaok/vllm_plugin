# SPDX-License-Identifier: Apache-2.0
"""GPU EPLB checks for ``test_eplb_gpu.py``; spawned ranks import this module by name.

One process per GPU of one Rubin node. Only vLLM's group coordinators are
stand-ins (over plain torch.distributed groups: NCCL device groups and gloo CPU
groups, separate ones for EP and EPLB as in vLLM). Everything else is real:

* ``SM107MegaMoEExperts`` layers at admitted geometries, with the die-local
  routed weights of ``backend.localize`` (driver VMM memory homed in each
  locality domain) and one fused shared expert;
* the plugin kernel through the layer's ``forward``, where vLLM's staging maps
  logical expert ids to physical replicas and records the expert load;
* vLLM's EPLB: ``EplbState`` with its policy, async worker thread and side
  stream, ``rearrange_expert_weights_inplace`` and the communicator classes.

A token's output does not depend on where its experts live (each expert row is
computed alone and the combine sums the top-k slots in a fixed order), so every
forward after a rearrangement must be bit-identical to the one before it.
"""

import math
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.distributed as dist
from vllm.model_executor.models.interfaces import MixtureOfExperts

# DeepSeek-V4.1-Flash expert shape, the only one the admission gate knows.
HIDDEN, INTERMEDIATE = 5120, 2304
# max_num_batched_tokens: the symmetric buffer is aligned to T=3840, the T at
# which the validated geometries ran.
MAX_TOKENS = 2560
SWIGLU_CLAMP = 10.0
NUM_LAYERS = 2
CANARY = 0xA5
LOCALIZED_WEIGHTS = ("w13_weight", "w2_weight")
# get_expert_weights() order: w13 die 0, w13 die 1, w13 scale, w2 die 0, w2 die 1,
# w2 scale. Die of each view (None: scales are not die-local).
VIEW_DIES = (0, 1, None, 0, 1, None)
MODEL_NAME = "sm107-megamoe-eplb-gpu-test"


class Coordinator:
    """What the MoE layer, the communicators and EplbState use of vLLM's
    GroupCoordinator."""

    def __init__(self, device_group, cpu_group) -> None:
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.world_size = device_group.size()
        self.rank_in_group = device_group.rank()
        self.device_communicator = None

    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.cpu_group)
        return tensor


def init_distributed(rank: int, world_size: int, port: int):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    ranks = list(range(world_size))
    ep = Coordinator(
        dist.new_group(ranks, backend="nccl"), dist.new_group(ranks, backend="gloo")
    )
    eplb = Coordinator(
        dist.new_group(ranks, backend="nccl"), dist.new_group(ranks, backend="gloo")
    )
    install_groups(ep, eplb)
    return ep, eplb


def install_groups(ep: Coordinator, eplb: Coordinator) -> None:
    from vllm.distributed.eplb import async_worker, eplb_state
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_sm107_megamoe import experts

    for module in (native, experts, eplb_state):
        module.get_ep_group = lambda: ep
    eplb_state.get_eplb_group = lambda: eplb
    async_worker.get_eplb_group = lambda: eplb
    eplb_state.get_node_count = lambda: 1


def eplb_config(communicator: str, num_redundant: int, use_async: bool):
    from vllm.config.parallel import EPLBConfig

    return EPLBConfig(
        window_size=4,
        step_interval=8,
        num_redundant_experts=num_redundant,
        use_async=use_async,
        communicator=communicator,
    )


def vllm_config(eplb) -> SimpleNamespace:
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            enable_eplb=True,
            enable_elastic_ep=False,
            eplb_config=eplb,
            num_ubatches=0,
        ),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=MAX_TOKENS),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )


class MoEModel(MixtureOfExperts):
    """Uses the protocol's own ``set_eplb_state``, as EplbState.add_model does."""

    def __init__(self, layers: list, num_logical: int, num_physical: int) -> None:
        self.moe_layers = layers
        self.num_moe_layers = len(layers)
        self.num_expert_groups = 1
        self.num_logical_experts = self.num_routed_experts = num_logical
        self.num_physical_experts = num_physical
        self.num_local_physical_experts = layers[0].num_local_experts
        self.num_redundant_experts = num_physical - num_logical
        self.num_shared_experts = 1
        self.expert_weights = []


def model_config() -> SimpleNamespace:
    return SimpleNamespace(model=MODEL_NAME, compute_hash=lambda: MODEL_NAME)


def shared_mlp() -> SimpleNamespace:
    """Shapes of an unsharded MXFP8 shared MLP, which the native class fuses."""

    def linear(rows: int, cols: int) -> SimpleNamespace:
        return SimpleNamespace(
            weight=torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device="meta"),
            weight_scale=torch.empty(rows, cols // 32, dtype=torch.uint8, device="meta"),
        )

    return SimpleNamespace(
        gate_up_proj=linear(2 * INTERMEDIATE, HIDDEN),
        down_proj=linear(HIDDEN, INTERMEDIATE),
    )


def make_layers(config, ep: Coordinator, top_k: int, num_physical: int, num_logical: int):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_sm107_megamoe.experts import SM107MegaMoEExperts

    num_local = num_physical // ep.world_size
    layers = []
    with torch.device("cuda", torch.cuda.current_device()):
        for index in range(NUM_LAYERS):
            layer = SM107MegaMoEExperts(
                config,
                prefix=f"model.layers.{index}.ffn.experts",
                num_experts=num_physical,
                num_local_experts=num_local,
                experts_start_idx=ep.rank_in_group * num_local,
                num_logical_experts=num_logical,
                top_k=top_k,
                hidden_size=HIDDEN,
                intermediate_size=INTERMEDIATE,
                weight_format=native.MEGA_MOE_MXFP4,
                activation=native.MegaMoEActivation(clamp=SWIGLU_CLAMP),
                shared_experts=shared_mlp(),
            )
            assert layer.has_fused_shared_experts
            layers.append(layer)
    return layers


def _generator(*key: int) -> torch.Generator:
    seed = 0
    for part in key:
        seed = seed * 1_000_003 + part
    return torch.Generator(device="cuda").manual_seed(seed % (1 << 62))


def expert_shards(layer: int, expert: int, version: int = 0) -> list:
    """Checkpoint shards ``(shard, name, tensor)`` of one logical expert: MXFP4
    weights with UE8M0 scales of 2^-9 .. 2^0, the same on every rank."""
    shards = []
    for code, shard in enumerate(("w1", "w2", "w3")):
        rows, cols = (
            (HIDDEN, INTERMEDIATE) if shard == "w2" else (INTERMEDIATE, HIDDEN)
        )
        prefix = "w2" if shard == "w2" else "w13"
        generator = _generator(version, layer, expert, code)
        weight = torch.randint(
            0, 256, (rows, cols // 2), generator=generator, device="cuda", dtype=torch.uint8
        )
        scale = torch.randint(
            118, 128, (rows, cols // 32), generator=generator, device="cuda", dtype=torch.uint8
        )
        shards.append((shard, f"{prefix}_weight", weight))
        shards.append((shard, f"{prefix}_weight_scale", scale))
    return shards


def load_experts(layer, index: int, experts, version: int = 0) -> int:
    """Load logical ``experts`` through the layer's weight loaders; returns how
    many shard loads landed on this rank."""
    loaded = 0
    for expert in experts:
        for shard, name, value in expert_shards(index, expert, version):
            loaded += bool(
                layer.weight_loader(
                    getattr(layer, name),
                    value,
                    f"experts.{name}",
                    shard,
                    expert,
                    return_success=True,
                )
            )
    return loaded


def load_shared(layer, index: int) -> None:
    generator = _generator(7, index)
    prefix = f"model.layers.{index}.ffn.shared_experts"
    for shard, (rows, cols) in (
        ("w1", (INTERMEDIATE, HIDDEN)),
        ("w3", (INTERMEDIATE, HIDDEN)),
        ("w2", (HIDDEN, INTERMEDIATE)),
    ):
        weight = torch.randn(rows, cols, generator=generator, device="cuda") * 0.25
        layer.load_shared_expert_weight(
            f"{prefix}.{shard}.weight", weight.to(torch.float8_e4m3fn)
        )
        scale = torch.randint(
            118, 128, (rows, cols // 32), generator=generator, device="cuda", dtype=torch.uint8
        )
        layer.load_shared_expert_weight(f"{prefix}.{shard}.weight_scale", scale)


def initial_placement(num_physical: int, num_logical: int) -> torch.Tensor:
    """vLLM's initial physical -> logical map: physical p holds p % num_logical."""
    return torch.arange(num_physical) % num_logical


def shuffled_placement(num_physical: int, num_logical: int, seed: int) -> torch.Tensor:
    """Every logical expert once plus replicas of some, at random physical slots:
    local moves, swaps, remote receives and replicas in one map."""
    generator = torch.Generator().manual_seed(seed)
    replicas = torch.randperm(num_logical, generator=generator)
    slots = torch.cat([torch.arange(num_logical), replicas[: num_physical - num_logical]])
    return slots[torch.randperm(num_physical, generator=generator)]


def build(rank: int, world_size: int, port: int, top_k: int, num_logical: int,
          num_physical: int, communicators: tuple, use_async: bool):  # fmt: skip
    """Groups, two loaded layers in the initial placement, the MoE model."""
    ep, eplb = init_distributed(rank, world_size, port)
    if "pynccl" in communicators:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        eplb.device_communicator = SimpleNamespace(
            pynccl_comm=PyNcclCommunicator(eplb.cpu_group, torch.device("cuda", rank))
        )
    config = vllm_config(
        eplb_config(communicators[0], num_physical - num_logical, use_async)
    )
    layers = make_layers(config, ep, top_k, num_physical, num_logical)
    placement = initial_placement(num_physical, num_logical)
    for index, layer in enumerate(layers):
        start, end = layer.experts_start_idx, layer.experts_end_idx
        experts = sorted(set(placement[start:end].tolist()))
        # w1, w3 and w2, weight and scale: six loads per local logical expert.
        assert load_experts(layer, index, experts) == 6 * len(experts)
        load_shared(layer, index)
        for name in LOCALIZED_WEIGHTS:
            fill_padding(getattr(layer, name).data)
    model = MoEModel(layers, num_logical, num_physical)
    return ep, eplb, config, layers, model, placement.repeat(NUM_LAYERS, 1)


def _raw_bytes(t: torch.Tensor) -> torch.Tensor:
    return torch.as_strided(t.view(torch.uint8), (t.untyped_storage().nbytes(),), (1,), 0)


def fill_padding(t: torch.Tensor) -> None:
    """Write ``CANARY`` into the padding after each die slice."""
    raw, used, step = _raw_bytes(t), math.prod(t.shape[1:]), t.stride(0)
    for die in range(t.shape[0]):
        raw[die * step + used : (die + 1) * step].fill_(CANARY)


def padding_intact(t: torch.Tensor) -> bool:
    raw, used, step = _raw_bytes(t), math.prod(t.shape[1:]), t.stride(0)
    return all(
        bool((raw[die * step + used : (die + 1) * step] == CANARY).all())
        for die in range(t.shape[0])
    )


def expected_rows(layer, index: int, expert: int, version: int = 0) -> list:
    """The six EPLB view rows of logical ``expert`` in the kernel layout, made
    by vLLM's own shard writer from the checkpoint shards."""
    n13, n2, k13, k2 = 2 * INTERMEDIATE, HIDDEN, HIDDEN, INTERMEDIATE
    dst = {
        "w13_weight": torch.zeros(n13, k13 // 2, dtype=torch.int8, device="cuda"),
        "w13_weight_scale": torch.zeros(k13 // 128, n13, dtype=torch.int32, device="cuda").T,
        "w2_weight": torch.zeros(n2, k2 // 2, dtype=torch.int8, device="cuda"),
        "w2_weight_scale": torch.zeros(k2 // 128, n2, dtype=torch.int32, device="cuda").T,
    }
    for shard, name, value in expert_shards(index, expert, version):
        layer._load_weight(dst[name], value, shard, name.endswith("scale"))
    w13, s13, w2, s2 = dst.values()
    return [
        w13[: n13 // 2].flatten(),
        w13[n13 // 2 :].flatten(),
        s13.T.flatten(),
        w2[: n2 // 2].flatten(),
        w2[n2 // 2 :].flatten(),
        s2.T.flatten(),
    ]


def check_layer(layer, index: int, placement: torch.Tensor, version: int = 0) -> int:
    """Every local row holds, byte for byte, the logical expert ``placement``
    puts there; die slices stay in their locality domain; padding untouched.
    Returns the number of rows checked."""
    from vllm_sm107_megamoe.backend import locality

    views = layer.get_expert_weights()
    assert len(views) == len(VIEW_DIES)
    cache: dict = {}
    for row in range(layer.num_local_experts):
        expert = int(placement[layer.experts_start_idx + row])
        if expert not in cache:
            cache[expert] = expected_rows(layer, index, expert, version)
        for k, (view, want) in enumerate(zip(views, cache[expert])):
            assert torch.equal(view[row], want), (index, row, expert, k)
            die = VIEW_DIES[k]
            if die is not None:
                assert locality.get_locality_domain_of(view[row].data_ptr()) == die, (
                    index, row, k,
                )  # fmt: skip
    for name in LOCALIZED_WEIGHTS:
        weight = getattr(layer, name).data
        assert locality.is_canonical_localized(weight), name
        assert padding_intact(weight), name
    return layer.num_local_experts


def make_inputs(num_tokens: int, num_logical: int, top_k: int, seed: int):
    """BF16 activations and top-k routing with Zipf-skewed expert popularity
    (over a fixed shuffle of the experts, the same on every rank)."""
    generator = _generator(11, seed)
    order = torch.randperm(num_logical, generator=torch.Generator().manual_seed(1234))
    popularity = torch.empty(num_logical)
    popularity[order] = 1.0 / torch.arange(1, num_logical + 1, dtype=torch.float) ** 1.2
    popularity = popularity.cuda().expand(num_tokens, -1).contiguous()
    ids = torch.multinomial(popularity, top_k, replacement=False, generator=generator)
    weights = torch.rand(num_tokens, top_k, generator=generator, device="cuda") + 0.05
    hidden = torch.randn(num_tokens, HIDDEN, generator=generator, device="cuda")
    return (
        hidden.to(torch.bfloat16),
        (weights / weights.sum(-1, keepdim=True)).float(),
        ids.to(torch.int32),
    )


def bits(y: torch.Tensor) -> torch.Tensor:
    return y.view(torch.int16)


def check_reference(y: torch.Tensor) -> None:
    assert y.dtype == torch.bfloat16 and bool(torch.isfinite(y).all())
    assert float((y != 0).float().mean()) > 0.5


def snapshot(layers) -> tuple:
    """Parameter addresses and the kernel's validated-launch records."""
    from vllm_sm107_megamoe.backend import frontend

    return (
        [[p.data_ptr() for p in layer.parameters()] for layer in layers],
        {key: record.sig for key, record in frontend._call_records.items()},
    )




def sync_main(rank: int, world_size: int, port: int, top_k: int, num_logical: int,
              num_physical: int, communicators: tuple) -> None:  # fmt: skip
    """Synchronous EPLB with each communicator in turn, through vLLM's
    EplbState (profile pass, current placement) and
    ``rearrange_expert_weights_inplace`` to a new placement, then the map
    commit; afterwards bytes, locality, padding, addresses and kernel output.
    Finally a reload after the moves must follow the current placement."""
    from vllm.distributed.eplb.eplb_state import EplbState, _commit_eplb_maps
    from vllm.distributed.eplb.rebalance_execute import (
        rearrange_expert_weights_inplace,
    )

    ep, eplb, config, layers, model, placement = build(
        rank, world_size, port, top_k, num_logical, num_physical, communicators, False
    )
    try:
        model_cfg = model_config()
        inputs = {m: make_inputs(m, num_logical, top_k, 100 * rank + m) for m in (64, 2560)}
        states, reference, before = [], None, None
        for round_, communicator in enumerate(communicators):
            config.parallel_config.eplb_config = eplb_config(
                communicator, num_physical - num_logical, False
            )
            state = EplbState(config.parallel_config, torch.device("cuda", rank))
            state.add_model(model, model_cfg)
            state.update_mapping(model_cfg, placement)
            states.append(state)  # keep every communicator's buffers alive
            model_state = state.model_states[MODEL_NAME]
            state.step(is_profile=True)  # the engine's profile run
            if reference is None:
                reference = [[layer(*inputs[m]) for m in sorted(inputs)] for layer in layers]
                for y in sum(reference, []):
                    check_reference(y)
                for index, layer in enumerate(layers):
                    check_layer(layer, index, placement[index])
                before = snapshot(layers)
            new = torch.stack(
                [
                    shuffled_placement(num_physical, num_logical, 1000 * round_ + i)
                    for i in range(NUM_LAYERS)
                ]
            )
            rearrange_expert_weights_inplace(
                model_state.physical_to_logical_map,
                new.cuda(),
                model.expert_weights,
                model_state.expert_buffer,
                ep.device_group,
                model_state.communicator,
            )
            _commit_eplb_maps(model_state, new_physical_to_logical_map=new)
            torch.cuda.synchronize()
            moved = int((new != placement).sum())
            placement = new
            assert torch.equal(model_state.physical_to_logical_map.cpu(), new)
            rows = sum(check_layer(layer, i, new[i]) for i, layer in enumerate(layers))
            outputs = [[layer(*inputs[m]) for m in sorted(inputs)] for layer in layers]
            for want, got in zip(sum(reference, []), sum(outputs, [])):
                assert torch.equal(bits(got), bits(want)), communicator
            assert snapshot(layers) == before, communicator
            dist.barrier(group=ep.cpu_group)
            if rank == 0:
                print(
                    f"SYNC_EPLB_OK communicator={communicator} "
                    f"physical_slots_changed={moved} rows_checked={rows}",
                    flush=True,
                )

        # Reload a new checkpoint version after the moves: every replica that
        # the current placement assigns must receive it.
        for index, layer in enumerate(layers):
            load_experts(layer, index, range(num_logical), version=1)
            check_layer(layer, index, placement[index], version=1)
        assert snapshot(layers)[0] == before[0]
        dist.barrier(group=ep.cpu_group)
        if rank == 0:
            print("RELOAD_AFTER_EPLB_OK", flush=True)
    finally:
        dist.destroy_process_group()


def async_main(rank: int, world_size: int, port: int, top_k: int, num_logical: int,
               num_physical: int, communicator: str) -> None:  # fmt: skip
    """Async EPLB as vLLM serves it: EplbState's worker thread computes the
    placement from the recorded load and moves one layer at a time on its own
    stream while the main thread replays CUDA graphs of the MoE layers; each
    step commits at most one moved layer. Every replay must stay bit-identical
    to the output before EPLB, through two full rearrangement cycles."""
    from vllm.distributed.eplb import eplb_state as eplb_state_module
    from vllm.distributed.eplb.eplb_state import EplbState

    ep, eplb, config, layers, model, placement = build(
        rank, world_size, port, top_k, num_logical, num_physical, (communicator,), True
    )
    try:
        commits: list[int] = []
        move_to_workspace = eplb_state_module._move_to_workspace

        def counting_move_to_workspace(model_state, ep_rank):
            commits.append(model_state.pending_result.layer_idx)
            move_to_workspace(model_state=model_state, ep_rank=ep_rank)

        eplb_state_module._move_to_workspace = counting_move_to_workspace

        model_cfg = model_config()
        state = EplbState(config.parallel_config, torch.device("cuda", rank))
        state.add_model(model, model_cfg)
        model_state = state.model_states[MODEL_NAME]
        state.step(is_profile=True)
        num_tokens = 512
        inputs = make_inputs(num_tokens, num_logical, top_k, 100 * rank + 7)
        state.prepare_forward(model_cfg, num_tokens)
        reference = [layer(*inputs) for layer in layers]
        for y in reference:
            check_reference(y)
        graphs, outputs = [], []
        for layer in layers:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs.append(layer(*inputs))
            graphs.append(graph)
        torch.cuda.synchronize()
        dist.barrier(group=ep.cpu_group)
        before = snapshot(layers)
        start = model_state.physical_to_logical_map.cpu().clone()

        state.start_async_loop()
        steps, cycles, deadline = 0, 0, time.monotonic() + 600
        while True:
            state.prepare_forward(model_cfg, num_tokens)
            for graph in graphs:
                graph.replay()
            for want, got in zip(reference, outputs):
                assert torch.equal(bits(got), bits(want)), (steps, list(commits))
            state.step()
            steps += 1
            cycles = sum(layer == NUM_LAYERS - 1 for layer in commits)
            done = torch.tensor([int(cycles >= 2 or time.monotonic() > deadline)])
            dist.all_reduce(done, op=dist.ReduceOp.MIN, group=ep.cpu_group)
            if int(done):
                break
        # Let a cycle that started in the last step finish without applying
        # it (vLLM's own drain), so no transfer is in flight at teardown.
        state.drain_async()
        assert cycles >= 2, (steps, commits)
        assert state.async_worker_error is None
        final = model_state.physical_to_logical_map.cpu()
        moved = torch.tensor([int((final != start).sum())])
        dist.all_reduce(moved, op=dist.ReduceOp.MAX, group=ep.cpu_group)
        assert int(moved) > 0, "the EPLB policy never changed the placement"
        rows = sum(check_layer(layer, i, final[i]) for i, layer in enumerate(layers))
        for graph in graphs:
            graph.replay()
        for layer, want, got in zip(layers, reference, outputs):
            assert torch.equal(bits(got), bits(want))
            assert torch.equal(bits(layer(*inputs)), bits(want))  # eager, too
        assert snapshot(layers) == before
        dist.barrier(group=ep.cpu_group)
        if rank == 0:
            print(
                f"ASYNC_EPLB_OK communicator={communicator} steps={steps} "
                f"commits={commits} physical_slots_changed={int(moved)} "
                f"rows_checked={rows}",
                flush=True,
            )
    finally:
        dist.destroy_process_group()
