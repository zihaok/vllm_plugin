# SPDX-License-Identifier: Apache-2.0
"""EPLB over die-local routed weights, through vLLM's own EPLB code.

CPU only. ``localize`` is simulated by a host tensor in the plugin's
``(2, E_l, N/2, K/2)`` layout with padded die slices. The rearrangement cases
spawn gloo ranks that run vLLM's ``rearrange_expert_weights_inplace`` (or the
async worker's ``transfer_layer`` + ``move_from_buffer``) with vLLM's
communicator classes, then compare every die slice and scale of every expert
row byte for byte with the native layout of the logical expert that belongs
there.
"""

import sys
import typing
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # spawned ranks import the harness by name
    sys.path.insert(0, _HERE)

import eplb_harness as harness  # noqa: E402

# Two ranks, 6 logical + 2 redundant experts, 4 slots per rank. Initially
# physical p holds logical p % 6. Each new layer map mixes unchanged rows,
# local moves, swaps, remote receives and replicas of a remote receive.
TWO_RANKS = dict(
    num_logical=6,
    old=[0, 1, 2, 3, 4, 5, 0, 1],
    new=[[5, 5, 2, 0, 3, 1, 4, 4], [1, 4, 4, 3, 2, 0, 5, 1]],
)
ONE_RANK = dict(num_logical=3, old=[0, 1, 2, 0], new=[[2, 2, 1, 0], [0, 2, 1, 1]])


def _plugin_experts():
    from vllm_sm107_megamoe.experts import SM107MegaMoEExperts

    return SM107MegaMoEExperts


def _run_ranks(tmp_path, monkeypatch, world_size, cases, scenario):
    """Spawn ``world_size`` gloo ranks running ``harness.rank_main``."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    symm_buffers = None
    if any(name == "symm_mem" for name, _ in cases):
        harness.install_fakes(monkeypatch.setattr, world_size)
        num_physical = len(scenario["old"])
        layer = harness.make_experts(
            _plugin_experts(),
            "sizing.experts",
            num_experts=num_physical,
            num_local_experts=num_physical // world_size,
            start=0,
            num_logical=scenario["num_logical"],
            shared=True,
        )
        nbytes = harness.symm_mem_offsets(layer.get_expert_weights())[-1]
        symm_buffers = [
            torch.zeros(nbytes, dtype=torch.uint8).share_memory_()
            for _ in range(world_size)
        ]
    torch.multiprocessing.spawn(
        harness.rank_main,
        args=(
            world_size,
            str(tmp_path / "rendezvous"),
            cases,
            scenario["num_logical"],
            scenario["old"],
            scenario["new"],
            symm_buffers,
        ),
        nprocs=world_size,
        join=True,
    )


def _needs_accelerator(cases):
    # gloo staging and symm_mem synchronize the accelerator stream; direct
    # isend/irecv (torch_nccl's code path, run over gloo here) does not.
    if any(name != "torch_nccl" for name, _ in cases):
        if not torch.accelerator.is_available():
            pytest.skip("the staged and symm_mem communicators sync a device stream")


@pytest.mark.parametrize(
    "cases",
    [
        [("torch_nccl", "sync")],
        [("torch_gloo", "sync"), ("torch_gloo", "async"), ("torch_gloo", "async_batched")],
        [("symm_mem", "sync"), ("symm_mem", "async"), ("symm_mem", "async_batched")],
    ],
    ids=["torch_nccl", "torch_gloo", "symm_mem"],
)
def test_vllm_eplb_moves_die_slices_and_scales_between_two_ranks(
    tmp_path, monkeypatch, cases
):
    _needs_accelerator(cases)
    _run_ranks(tmp_path, monkeypatch, 2, cases, TWO_RANKS)


def test_vllm_eplb_moves_die_slices_and_scales_within_one_rank(
    tmp_path, monkeypatch
):
    cases = [("torch_nccl", "sync"), ("torch_nccl", "async"), ("symm_mem", "sync")]
    _needs_accelerator(cases)
    _run_ranks(tmp_path, monkeypatch, 1, cases, ONE_RANK)


def test_expert_weights_are_contiguous_views_of_the_kernel_buffers(monkeypatch):
    harness.install_fakes(monkeypatch.setattr, 2)
    num_local = 4
    layer = harness.make_experts(
        _plugin_experts(),
        "model.layers.0.experts",
        num_experts=8,
        num_local_experts=num_local,
        start=4,
        num_logical=6,
        shared=True,
        config=harness.make_config(True, "symm_mem"),
    )
    assert layer.has_fused_shared_experts
    params = dict(layer.named_parameters(recurse=False))
    w13, w2 = params["w13_weight"].data, params["w2_weight"].data
    s13, s2 = params["w13_weight_scale"].data, params["w2_weight_scale"].data
    for w in (w13, w2):  # the simulated driver layout: padded die slices
        assert w.dim() == 4 and w.shape[:2] == (2, num_local)
        assert w.stride(0) > w[0].numel()
    for s in (s13, s2):  # native MN-major scales
        assert not s.is_contiguous() and s.transpose(1, 2).is_contiguous()

    views = layer.get_expert_weights()
    sources = [(w13, 0), (w13, 1), (s13, None), (w2, 0), (w2, 1), (s2, None)]
    assert len(views) == len(sources)
    for view, (param, die) in zip(views, sources):
        assert view.dim() == 2 and view.shape[0] == num_local
        assert view.is_contiguous() and view.dtype == param.dtype
        assert view.untyped_storage().data_ptr() == param.untyped_storage().data_ptr()
        offset = 0 if die is None else die * param.stride(0) * param.element_size()
        assert view.data_ptr() == param.data_ptr() + offset
        assert view.numel() == (param.numel() if die is None else param[die].numel())
    view_storages = {v.untyped_storage().data_ptr() for v in views}
    for name, param in params.items():
        if name.startswith("shared_"):
            assert param.untyped_storage().data_ptr() not in view_storages, name

    # A row copy through a view is a write into that expert's die slice / scale.
    for k, (view, (param, die)) in enumerate(zip(views, sources)):
        for row in range(num_local):
            n = view[row].numel()
            pattern = (torch.arange(n, dtype=torch.int64) * 7 + 31 * k + row) % 127
            view[row].copy_(pattern.to(view.dtype))
            target = param[die, row] if die is not None else param[row].transpose(0, 1)
            assert torch.equal(target.flatten(), pattern.to(view.dtype)), (k, row)
    for k, (view, _) in enumerate(zip(views, sources)):
        for row in range(num_local):
            n = view[row].numel()
            pattern = (torch.arange(n, dtype=torch.int64) * 7 + 31 * k + row) % 127
            assert torch.equal(view[row], pattern.to(view.dtype)), (k, row)
    assert harness.padding_intact(w13) and harness.padding_intact(w2)


@pytest.mark.parametrize("communicator", ["nixl", "torch_xccl", None, "unknown"])
def test_eplb_communicators_that_cannot_move_die_local_weights_are_refused(
    monkeypatch, communicator
):
    harness.install_fakes(monkeypatch.setattr, 2)
    config = harness.make_config(True, communicator)
    with pytest.raises(NotImplementedError) as error:
        harness.make_experts(
            _plugin_experts(),
            "model.layers.0.experts",
            num_experts=8,
            num_local_experts=4,
            start=0,
            num_logical=6,
            shared=True,
            config=config,
        )
    message = str(error.value)
    assert f"EPLB communicator {communicator!r}" in message
    assert "symm_mem" in message
    if communicator == "nixl":
        assert "RDMA" in message and "VMM" in message
    # Refused before any weight is allocated or the layer is registered.
    assert config.compilation_config.static_forward_context == {}


def test_elastic_ep_is_refused(monkeypatch):
    harness.install_fakes(monkeypatch.setattr, 2)
    config = harness.make_config(True, "symm_mem", elastic_ep=True)
    with pytest.raises(NotImplementedError, match="elastic EP"):
        harness.make_experts(
            _plugin_experts(),
            "model.layers.0.experts",
            num_experts=8,
            num_local_experts=4,
            start=0,
            num_logical=6,
            shared=True,
            config=config,
        )


@pytest.mark.parametrize("communicator", ["symm_mem", "torch_gloo", "torch_nccl", "pynccl"])
def test_copying_eplb_communicators_are_accepted(monkeypatch, communicator):
    harness.install_fakes(monkeypatch.setattr, 2)
    layer = harness.make_experts(
        _plugin_experts(),
        "model.layers.0.experts",
        num_experts=8,
        num_local_experts=4,
        start=4,
        num_logical=6,
        shared=True,
        config=harness.make_config(True, communicator),
    )
    assert len(layer.get_expert_weights()) == 6


def test_every_vllm_eplb_communicator_is_classified():
    from vllm.config.parallel import EPLBCommunicatorBackend

    from vllm_sm107_megamoe import experts

    known = set(typing.get_args(EPLBCommunicatorBackend))
    classified = experts.SUPPORTED_EPLB_COMMUNICATORS | set(
        experts._UNSUPPORTED_EPLB_COMMUNICATORS
    )
    assert known == classified


def test_eplb_layer_state_is_native(monkeypatch):
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import (
        DeepGemmMegaMoEExperts,
    )

    experts_cls = _plugin_experts()
    assert experts_cls.set_eplb_state is DeepGemmMegaMoEExperts.set_eplb_state
    assert experts_cls.update_expert_map is DeepGemmMegaMoEExperts.update_expert_map
    harness.install_fakes(monkeypatch.setattr, 2)
    layer = harness.make_experts(
        experts_cls,
        "model.layers.0.experts",
        num_experts=8,
        num_local_experts=4,
        start=4,
        num_logical=6,
        shared=True,
        config=harness.make_config(True, "symm_mem"),
    )
    load_view = torch.zeros(3, 8, dtype=torch.int32)
    mapping = torch.full((3, 6, 3), -1, dtype=torch.int64)
    count = torch.ones(3, 6, dtype=torch.int64)
    layer.set_eplb_state(2, load_view, mapping, count)
    state = layer.eplb_state
    assert state.expert_load_view.data_ptr() == load_view[2].data_ptr()
    assert state.logical_to_physical_map.data_ptr() == mapping[2].data_ptr()
    assert state.logical_replica_count.data_ptr() == count[2].data_ptr()


@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_nccl_communicators_are_refused_with_async_eplb(monkeypatch, communicator):
    harness.install_fakes(monkeypatch.setattr, 2)
    config = harness.make_config(True, communicator, use_async=True)
    with pytest.raises(NotImplementedError, match="use_async") as error:
        harness.make_experts(
            _plugin_experts(),
            "model.layers.0.experts",
            num_experts=8,
            num_local_experts=4,
            start=0,
            num_logical=6,
            shared=True,
            config=config,
        )
    assert "symm_mem" in str(error.value)
    assert config.compilation_config.static_forward_context == {}


@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_vllm_also_rejects_nccl_communicators_with_async_eplb(communicator):
    # The plugin refuses this combination itself; this records that the
    # serving image's vLLM agrees.
    from vllm.config.parallel import EPLBConfig

    with pytest.raises(ValueError, match="async"):
        EPLBConfig(use_async=True, communicator=communicator)
    EPLBConfig(use_async=False, communicator=communicator)


def test_symm_mem_eplb_communicator_synchronizes_on_the_cpu_only():
    """The MegaMoE kernel keeps every SM resident and synchronizes them in
    software. Async EPLB next to it is safe while symm_mem moves rows with
    copies and waits with a host barrier; this fails if vLLM changes that."""
    import ast
    import inspect
    import textwrap

    from vllm.distributed.eplb.eplb_communicator import SymmMemEplbCommunicator

    calls: dict = {}
    for method in (SymmMemEplbCommunicator.execute, SymmMemEplbCommunicator._barrier):
        tree = ast.parse(textwrap.dedent(inspect.getsource(method)))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else func.id
                calls.setdefault(name, []).append(node)
    assert set(calls) <= {
        "device_stream", "_barrier", "zip", "copy_", "clear",
        "gpu_sync_allowed", "current_stream", "synchronize", "all_reduce", "zeros",
    }, sorted(calls)  # fmt: skip
    (all_reduce,) = calls["all_reduce"]
    group = next(k.value for k in all_reduce.keywords if k.arg == "group")
    assert isinstance(group, ast.Attribute) and group.attr == "_cpu_group"
    (zeros,) = calls["zeros"]
    assert not any(k.arg == "device" for k in zeros.keywords)  # a host tensor


def _shifted(view: "torch.Tensor") -> "torch.Tensor":
    # One element further into the same allocation; die slices are padded, so
    # this stays inside it.
    return torch.as_strided(view, view.shape, view.stride(), view.storage_offset() + 1)


@pytest.mark.parametrize(
    "mutate", [lambda view: view.clone(), _shifted], ids=["copy", "shifted"]
)
def test_expert_weights_refuse_views_that_miss_the_kernel_memory(monkeypatch, mutate):
    """Mutants of the die-slice views (the first views EPLB gets) must trip the
    in-place check of get_expert_weights."""
    harness.install_fakes(monkeypatch.setattr, 2)
    experts_cls = _plugin_experts()
    layer = harness.make_experts(
        experts_cls,
        "model.layers.0.experts",
        num_experts=8,
        num_local_experts=4,
        start=4,
        num_logical=6,
        shared=True,
        config=harness.make_config(True, "symm_mem"),
    )
    view = experts_cls._eplb_view

    def mutated_view(self, name, t):
        good = view(self, name, t)
        return mutate(good) if name in harness.LOCALIZED_WEIGHTS else good

    monkeypatch.setattr(experts_cls, "_eplb_view", mutated_view)
    with pytest.raises(AssertionError, match="in place"):
        layer.get_expert_weights()


def test_reload_after_eplb_moves_follows_the_current_placement(monkeypatch):
    from vllm.distributed.eplb.eplb_state import compute_logical_maps
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    harness.install_fakes(monkeypatch.setattr, 2)
    # Rank 1 of 2: physical experts 4..7 of 8 for 6 logical experts; the
    # initial placement puts logical 4, 5, 0, 1 there.
    layer = harness.make_experts(
        _plugin_experts(),
        "model.layers.0.experts",
        num_experts=8,
        num_local_experts=4,
        start=4,
        num_logical=6,
        shared=True,
        config=harness.make_config(True, "symm_mem"),
    )
    harness.load(layer, harness.checkpoint(0, 6))
    pointers = [p.data_ptr() for p in layer.parameters()]
    # EPLB moved the experts: physical -> logical is now [5, 5, 2, 0, 3, 1, 4, 4],
    # so this rank holds logical 3, 1, 4, 4.
    placement = torch.tensor([[5, 5, 2, 0, 3, 1, 4, 4]])
    logical_to_physical, replica_count = compute_logical_maps(placement, 6)
    layer.set_eplb_state(
        0, torch.zeros(1, 8, dtype=torch.int32), logical_to_physical, replica_count
    )
    assert [layer._map_global_expert_id(e) for e in range(6)] == [
        [], [1], [], [0], [2, 3], [],
    ]  # fmt: skip
    reference = harness.make_experts(
        native.DeepGemmMegaMoEExperts,
        "reference",
        num_experts=6,
        num_local_experts=6,
        start=0,
        num_logical=6,
        shared=False,
    )
    shards = harness.checkpoint(1, 6)
    harness.load(reference, shards)
    harness.load(layer, shards)  # a reload: every shard of every expert
    for row, expert in enumerate([3, 1, 4, 4]):
        harness.assert_holds(layer, row, reference, expert)
    for name in harness.LOCALIZED_WEIGHTS:
        assert harness.padding_intact(getattr(layer, name).data), name
    assert [p.data_ptr() for p in layer.parameters()] == pointers
