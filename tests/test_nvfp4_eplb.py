# SPDX-License-Identifier: Apache-2.0
"""EPLB on NVFP4 MegaMoE layers, without a GPU.

* the communicator policy, the mixed-format refusal and the geometry allowlist;
* the EPLB views (8 checkpoint tensors + 2 alpha buffers): order, in-place aliasing,
  signature consistency, and mutants that must be refused;
* the alpha buffers: filled in place, consistent with weight_scale_2, moved by vLLM's
  real rearrangement code on gloo ranks with every communicator (negative control:
  left out of the moved list, they go stale);
* tuning tables looked up by the logical expert count, the physical count in the kernel;
* the EPLB staging arguments (inert without EPLB, vLLM's layout and indexing).

Tests that need torch / vLLM skip cleanly without them; the GPU checks are
``nvfp4_eplb_gpu_check.py``.
"""

import sys
import typing
from pathlib import Path
from types import SimpleNamespace

import pytest

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # spawned ranks import the harness by name
    sys.path.insert(0, _HERE)

try:
    import torch
    import vllm  # noqa: F401

    import nvfp4_eplb_harness as harness

    _MISSING = None
except Exception as error:  # noqa: BLE001 - torch / vLLM absent or unusable here: skip those tests
    torch = harness = None
    _MISSING = f"needs torch and vLLM ({error})"
needs_vllm = pytest.mark.skipif(_MISSING is not None, reason=_MISSING or "")


# ------------------------------------------------------------------------------------------- pure Python
def test_geometry_allowlist(monkeypatch):
    from vllm_nvfp4_megamoe import config

    monkeypatch.delenv(config.ALLOW_UNVALIDATED_GEOMETRY_ENV, raising=False)
    shape = (5120, 2304, 6)
    for experts, ep in ((384, 4), (384, 16), (400, 4), (400, 16)):  # validated on GPUs
        assert config.geometry_unsupported_reason(experts, ep, *shape, logical_experts=384) is None
    for experts, ep in ((400, 8), (388, 4)):  # pending GPU validation
        reason = config.geometry_unsupported_reason(experts, ep, *shape, logical_experts=384)
        assert reason is not None and "pending" in reason and config.ALLOW_UNVALIDATED_GEOMETRY_ENV in reason
    for experts, ep, logical in ((392, 8, 384), (100, 4, 96)):
        reason = config.geometry_unsupported_reason(experts, ep, *shape, logical_experts=logical)
        assert reason is not None and "never ran" in reason
    # another model shape with redundant experts
    assert config.geometry_unsupported_reason(388, 4, 4096, 2304, 6, logical_experts=384) is not None
    # Without redundant experts (EPLB off, or 0 redundant) every pair is admitted, as before EPLB support.
    for experts, ep in ((384, 8), (384, 2), (384, 32)):
        assert config.geometry_unsupported_reason(experts, ep, *shape, logical_experts=experts) is None
    assert config.geometry_unsupported_reason(384, 4, 4096, 2304, 6, logical_experts=384) is None
    monkeypatch.setenv(config.ALLOW_UNVALIDATED_GEOMETRY_ENV, "1")
    for experts, ep, logical in ((400, 8, 384), (388, 4, 384), (100, 4, 96)):
        assert config.geometry_unsupported_reason(experts, ep, *shape, logical_experts=logical) is None


@pytest.mark.parametrize("table", ["1", "bitwise", "0"])
def test_tuning_tables_are_looked_up_by_the_logical_count(monkeypatch, table):
    from vllm_nvfp4_megamoe import config

    monkeypatch.setenv("VLLM_NVFP4_MEGAMOE_RUBIN_TABLE", table)
    monkeypatch.delenv("VLLM_NVFP4_MEGAMOE_IMPL", raising=False)
    monkeypatch.delenv(config.KERNEL_DIR_ENV, raising=False)  # no GB200 table: only the Rubin one
    buckets = config.token_buckets(12288)
    for bucket in buckets:
        logical = config.kernel_settings(bucket, 5120, 2304, 384, 6)
        physical = config.kernel_settings(bucket, 5120, 2304, 400, 6)
        assert physical == config.KernelSettings()  # why the physical count must not be used
        rubin = config._rubin_settings(bucket)
        assert logical == (rubin or config.KernelSettings())
    if table != "0":
        assert any(config.kernel_settings(b, 5120, 2304, 384, 6) != config.KernelSettings() for b in buckets)
    if table == "bitwise":
        assert not any(config.kernel_settings(b, 5120, 2304, 384, 6).combine_dedup for b in buckets)
    # The override still applies on top of the logical-count tables.
    monkeypatch.setenv("VLLM_NVFP4_MEGAMOE_IMPL", '{"combine_dedup": false, "token_back_mode": "epi_warps"}')
    assert not any(config.kernel_settings(b, 5120, 2304, 384, 6).combine_dedup for b in buckets)


# ---------------------------------------------------------------------------------- torch only (no vLLM)
def test_inert_and_live_eplb_args():
    torch = pytest.importorskip("torch")  # noqa: F811 - only torch is needed here
    from vllm_nvfp4_megamoe import eplb

    assert eplb.eplb_kernel_args(None) == eplb.INERT_EPLB_ARGS
    empty = SimpleNamespace(logical_to_physical_map=None)
    assert eplb.eplb_kernel_args(empty) == eplb.INERT_EPLB_ARGS
    eplb.check_eplb_args(eplb.INERT_EPLB_ARGS, 384, 400, "x")  # inert: nothing to check

    mapping = torch.full((3, 6, 3), -1, dtype=torch.int64)
    load = torch.zeros(3, 8, dtype=torch.int32)
    state = SimpleNamespace(
        logical_to_physical_map=mapping[1], logical_replica_count=torch.ones(3, 6, dtype=torch.int64)[1],
        expert_load_view=load[1], should_record_tensor=torch.ones((), dtype=torch.bool),
        num_unpadded_tokens_tensors=[torch.tensor(5, dtype=torch.int32)],
    )
    args = eplb.eplb_kernel_args(state)
    assert len(args) == len(eplb.INERT_EPLB_ARGS) == 8
    assert args[0].data_ptr() == mapping[1].data_ptr() and args[2].data_ptr() == load[1].data_ptr()
    assert args[4] is state.num_unpadded_tokens_tensors[0]
    assert args[5:] == (6, 3, 8)
    eplb.check_eplb_args(args, 6, 8, "x")
    with pytest.raises(RuntimeError, match="logical"):
        eplb.check_eplb_args(args, 6, 12, "x")


# ----------------------------------------------------------------------------------------- torch + vLLM
# Two ranks, 6 logical + 2 redundant experts, 4 slots per rank. Initially physical p holds logical p % 6. Each
# new layer map mixes unchanged rows, local moves, swaps, remote receives and replicas of a remote receive.
TWO_RANKS = dict(num_logical=6, old=[0, 1, 2, 3, 4, 5, 0, 1], new=[[5, 5, 2, 0, 3, 1, 4, 4], [1, 4, 4, 3, 2, 0, 5, 1]])
ONE_RANK = dict(num_logical=3, old=[0, 1, 2, 0], new=[[2, 2, 1, 0], [0, 2, 1, 1]])


@pytest.fixture
def fakes(monkeypatch):
    """Two-rank EP group stand-in, fresh signature registry, unvalidated test geometry admitted."""
    from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

    monkeypatch.setattr(VllmNvfp4MegaMoEExperts, "_eplb_view_signatures", {})
    monkeypatch.setenv(harness.ALLOW_ENV, "1")
    harness.install_fakes(monkeypatch.setattr, harness.Coordinator(2))


def _layer(config=None, index=0, start=4):
    return harness.make_layer(f"model.layers.{index}.ffn.experts", num_experts=8, num_local_experts=4,
                              start=start, num_logical=6, config=config)


@needs_vllm
@pytest.mark.parametrize("communicator", ["nixl", "torch_xccl", None, "unknown"])
def test_refused_eplb_communicators(fakes, communicator):
    config = harness.make_config(True, communicator)
    with pytest.raises(NotImplementedError) as error:
        _layer(config)
    message = str(error.value)
    assert "NVFP4 MegaMoE plugin" in message and f"EPLB communicator {communicator!r}" in message
    assert "symm_mem" in message
    if communicator == "nixl":
        assert "not been validated" in message
    assert config.compilation_config.static_forward_context == {}  # refused before any allocation


@needs_vllm
@pytest.mark.parametrize("communicator", ["torch_nccl", "pynccl"])
def test_nccl_communicators_are_refused_with_async_eplb(fakes, communicator):
    config = harness.make_config(True, communicator, use_async=True)
    with pytest.raises(NotImplementedError, match="use_async"):
        _layer(config)
    assert config.compilation_config.static_forward_context == {}


@needs_vllm
@pytest.mark.parametrize("enable_eplb", [True, False])
def test_elastic_ep_is_refused(fakes, enable_eplb):
    with pytest.raises(NotImplementedError, match="elastic EP"):
        _layer(harness.make_config(enable_eplb, "symm_mem", elastic_ep=True))


@needs_vllm
@pytest.mark.parametrize("communicator,use_async", [("symm_mem", True), ("symm_mem", False), ("torch_gloo", True),
                                                    ("torch_nccl", False), ("pynccl", False)])
def test_accepted_eplb_communicators(fakes, communicator, use_async):
    layer = _layer(harness.make_config(True, communicator, use_async=use_async))
    assert layer._eplb_enabled
    assert len(layer.get_expert_weights()) == 10


@needs_vllm
def test_every_vllm_eplb_communicator_is_classified():
    from vllm.config.parallel import EPLBCommunicatorBackend

    from vllm_nvfp4_megamoe import eplb, experts
    from vllm_sm107_megamoe import experts as sm107_experts

    known = set(typing.get_args(EPLBCommunicatorBackend))
    assert known == eplb.SUPPORTED_EPLB_COMMUNICATORS | set(experts._NVFP4_UNSUPPORTED_EPLB)
    # The NVFP4 copy of the policy keeps the SM107 plugin's sets (it is a copy so the package runs next to the
    # SM107 package a serving image already has).
    assert eplb.SUPPORTED_EPLB_COMMUNICATORS == sm107_experts.SUPPORTED_EPLB_COMMUNICATORS
    assert eplb.SYNC_ONLY_EPLB_COMMUNICATORS == sm107_experts._SYNC_ONLY_EPLB_COMMUNICATORS
    assert set(eplb.UNSUPPORTED_EPLB_COMMUNICATORS) == set(sm107_experts._UNSUPPORTED_EPLB_COMMUNICATORS)


@needs_vllm
def test_mixed_nvfp4_mxfp4_target_layers_are_refused_with_eplb(fakes):
    from vllm_nvfp4_megamoe.experts import check_uniform_target_formats

    config = harness.make_config(True, "symm_mem", not_nvfp4=(2,))
    with pytest.raises(NotImplementedError, match=r"layers \[2\] are not NVFP4"):
        _layer(config, index=0)
    assert config.compilation_config.static_forward_context == {}
    # The MXFP4 layer itself is refused too (checked before the SM107 branch allocates anything).
    with pytest.raises(NotImplementedError, match="mixed"):
        check_uniform_target_formats(config, "model.layers.2.ffn.experts")
    check_uniform_target_formats(config, "model.layers.7.ffn.experts")  # draft layers are not target layers
    # Without EPLB a mixed checkpoint keeps working as before.
    _layer(harness.make_config(False, not_nvfp4=(2,)), index=0)
    # Uniform checkpoints are fine.
    _layer(harness.make_config(True, "symm_mem"), index=1)


@needs_vllm
def test_unvalidated_geometry_is_refused_before_allocation(fakes, monkeypatch):
    monkeypatch.delenv(harness.ALLOW_ENV)
    config = harness.make_config(True, "symm_mem")
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

    with pytest.raises(NotImplementedError, match="has not been validated"):
        VllmNvfp4MegaMoEExperts(
            config, prefix="model.layers.0.ffn.experts", num_experts=8, num_local_experts=4, experts_start_idx=4,
            num_logical_experts=6, top_k=harness.TOP_K, hidden_size=harness.HIDDEN,
            intermediate_size=harness.INTERMEDIATE, weight_format=native.MEGA_MOE_MXFP4,
            activation=native.MegaMoEActivation(clamp=10.0), shared_experts=harness.shared_mlp(),
        )
    assert config.compilation_config.static_forward_context == {}


@needs_vllm
def test_expert_weights_are_in_place_views_with_alphas_last(fakes):
    num_local = 4
    layer = _layer(harness.make_config(True, "symm_mem"))
    views = layer.get_expert_weights()
    sources = [getattr(layer, name) for name in harness.VIEW_NAMES]
    assert len(views) == len(sources) == 10
    for name, view, source in zip(harness.VIEW_NAMES, views, sources):
        assert view.dim() == 2 and view.shape[0] == num_local and view.is_contiguous(), name
        assert view.dtype == source.dtype and view.data_ptr() == source.data_ptr(), name
        assert view.numel() == source.numel(), name
    assert {"_fc1_alpha", "_fc2_alpha", "_fc1_norm_const"} <= dict(layer.named_buffers()).keys()
    assert not any(name.startswith("_fc") for name, _ in layer.named_parameters())  # not checkpoint tensors
    assert "_fc1_alpha" not in layer.state_dict()  # non-persistent
    view_storages = {v.untyped_storage().data_ptr() for v in views}
    for name, param in layer.named_parameters(recurse=False):
        if name.startswith("shared_"):
            assert param.untyped_storage().data_ptr() not in view_storages, name
    # A row write through a view is a write into that slot of the kernel tensor.
    for k, (view, source) in enumerate(zip(views, sources)):
        for row in range(num_local):
            pattern = ((torch.arange(view[row].numel()) * 7 + 31 * k + row) % 97).to(view.dtype)
            view[row].copy_(pattern)
            assert torch.equal(source[row].reshape(-1), pattern), (k, row)


@needs_vllm
@pytest.mark.parametrize("mutate", [lambda v: v.clone(), lambda v: v[:, :1]], ids=["copy", "narrow"])
def test_expert_weights_refuse_views_that_miss_the_kernel_memory(fakes, monkeypatch, mutate):
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts

    layer = _layer(harness.make_config(True, "symm_mem"))
    native = DeepGemmMegaMoEExperts.get_expert_weights
    monkeypatch.setattr(DeepGemmMegaMoEExperts, "get_expert_weights", lambda self: [mutate(v) for v in native(self)])
    with pytest.raises(AssertionError, match="in place"):
        layer.get_expert_weights()


@needs_vllm
def test_layers_of_one_model_must_expose_the_same_view_signature(fakes):
    config = harness.make_config(True, "symm_mem")  # one engine: all layers share its VllmConfig
    a = _layer(config, index=0)
    b = _layer(config, index=1)
    views = a.get_expert_weights()
    assert len(b.get_expert_weights()) == len(views)
    with pytest.raises(RuntimeError, match="differ from those of model.layers.0"):
        b._check_eplb_signature(views[:-2])  # e.g. an MXFP4 layer's 6 views next to NVFP4's 10
    # Another model (another module root, e.g. a separately registered drafter) keeps its own signature.
    other = harness.make_layer("mtp.layers.0.ffn.experts", num_experts=4, num_local_experts=2, start=0,
                               num_logical=4, config=config)
    assert len(other.get_expert_weights()) == 10
    # Another engine in the same process (same module path, other geometry) is not held to this one's signature.
    second = harness.make_layer("model.layers.0.ffn.experts", num_experts=4, num_local_experts=2, start=0,
                                num_logical=4, config=harness.make_config(True, "symm_mem"))
    second._check_eplb_signature(views[:-2])


@needs_vllm
def test_eplb_layer_state_is_native(fakes):
    from vllm.model_executor.layers.fused_moe.deep_gemm_mega_moe import DeepGemmMegaMoEExperts

    from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

    assert VllmNvfp4MegaMoEExperts.set_eplb_state is DeepGemmMegaMoEExperts.set_eplb_state
    assert VllmNvfp4MegaMoEExperts.update_expert_map is DeepGemmMegaMoEExperts.update_expert_map


@needs_vllm
def test_staging_args_follow_the_layer_eplb_state(fakes):
    from vllm.distributed.eplb.eplb_state import compute_logical_maps

    from vllm_nvfp4_megamoe import eplb

    layer = _layer(harness.make_config(True, "symm_mem"))
    assert eplb.eplb_kernel_args(layer.eplb_state) == eplb.INERT_EPLB_ARGS  # before EplbState.add_model
    placement = torch.tensor([[5, 5, 2, 0, 3, 1, 4, 4]])
    l2p, count = compute_logical_maps(placement, 6)
    load = torch.zeros(1, 8, dtype=torch.int32)
    layer.set_eplb_state(0, load, l2p, count)
    layer.eplb_state.should_record_tensor = torch.ones((), dtype=torch.bool)
    layer.eplb_state.num_unpadded_tokens_tensors = [torch.tensor(0, dtype=torch.int32)]
    args = eplb.eplb_kernel_args(layer.eplb_state)
    assert args[0].data_ptr() == l2p[0].data_ptr() and args[2].data_ptr() == load[0].data_ptr()
    assert args[5:] == (6, l2p.shape[-1], 8)
    eplb.check_eplb_args(args, layer.num_logical_experts, layer.num_experts, layer.prefix)


@needs_vllm
def test_local_eplb_args_copy_matches_vllm():
    """Canary: the plugin's copy of vLLM's private _eplb_kernel_args must give the same arguments."""
    pytest.importorskip("triton")
    try:
        from vllm.model_executor.layers.fused_moe.prepare_megamoe import _eplb_kernel_args
    except ImportError as error:
        pytest.skip(f"vLLM has no prepare_megamoe._eplb_kernel_args ({error}); compare against its successor")
    from vllm_nvfp4_megamoe import eplb

    assert _eplb_kernel_args(None) == eplb.eplb_kernel_args(None)
    mapping = torch.full((2, 6, 3), -1, dtype=torch.int64)
    state = SimpleNamespace(
        logical_to_physical_map=mapping[1], logical_replica_count=torch.ones(6, dtype=torch.int64),
        expert_load_view=torch.zeros(8, dtype=torch.int32), should_record_tensor=torch.ones((), dtype=torch.bool),
        num_unpadded_tokens_tensors=[torch.tensor(3, dtype=torch.int32)],
    )
    ours, theirs = eplb.eplb_kernel_args(state), _eplb_kernel_args(state)
    assert len(ours) == len(theirs)
    for a, b in zip(ours, theirs):
        assert a is b or a == b


@needs_vllm
def test_vllm_eplb_map_and_record_signature():
    """Canary: the staging kernel calls vLLM's eplb_map_and_record positionally in this order."""
    pytest.importorskip("triton")
    import inspect

    from vllm.model_executor.layers.fused_moe.eplb_triton import eplb_map_and_record

    fn = getattr(eplb_map_and_record, "fn", eplb_map_and_record)
    assert list(inspect.signature(fn).parameters) == [
        "logical_ids", "token_idx", "mask", "logical_to_physical", "replica_count", "num_logical", "map_stride",
        "expert_load", "num_physical", "record_enabled", "num_unpadded_tokens",
    ]


@needs_vllm
def test_alpha_buffers_are_filled_in_place_from_the_current_slots(fakes):
    from vllm.distributed.eplb.eplb_state import compute_logical_maps

    layer = _layer(harness.make_config(False))  # EPLB off: no cross-rank scale check
    harness.load(layer, harness.checkpoint(0, 6))
    pointers = harness.addresses(layer)
    a1, a2 = layer._fill_scale_buffers()
    assert harness.alphas_consistent(layer)
    assert torch.equal(layer._fc1_norm_const, torch.full((4,), 1.0 / a2, dtype=torch.float32))
    assert a1 == float(layer._a1_in.max())
    # Bit-identical to the arithmetic before EPLB support (fresh tensors from the same expressions).
    ws2_13 = layer.w13_weight_scale_2.data[:, 0].to(torch.float32)
    assert torch.equal(layer._fc1_alpha, (ws2_13 * a1).contiguous())
    assert torch.equal(layer._fc2_alpha, (layer.w2_weight_scale_2.data.to(torch.float32) * a2).contiguous())
    # After EPLB moved the experts, a reload writes each expert into its current slots and a re-fill keeps the
    # buffer addresses (graphs and kernels bound to them stay valid).
    l2p, count = compute_logical_maps(torch.tensor([[5, 5, 2, 0, 3, 1, 4, 4]]), 6)
    layer.set_eplb_state(0, torch.zeros(1, 8, dtype=torch.int32), l2p, count)
    shards = harness.checkpoint(1, 6)
    harness.load(layer, shards)
    layer._fill_scale_buffers()
    reference = harness.reference_layer(6, shards)
    for row, expert in enumerate([3, 1, 4, 4]):
        harness.assert_holds(layer, row, reference, expert)
    assert harness.addresses(layer) == pointers


def _run_ranks(tmp_path, monkeypatch, world_size, cases, scenario):
    """Spawn ``world_size`` gloo ranks running ``harness.rank_main``."""
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    monkeypatch.setenv(harness.ALLOW_ENV, "1")
    symm_buffers = None
    if any(name == "symm_mem" for name, _ in cases):
        from vllm_nvfp4_megamoe.experts import VllmNvfp4MegaMoEExperts

        monkeypatch.setattr(VllmNvfp4MegaMoEExperts, "_eplb_view_signatures", {})
        harness.install_fakes(monkeypatch.setattr, harness.Coordinator(world_size))
        num_physical = len(scenario["old"])
        layer = harness.make_layer("sizing.layers.0.ffn.experts", num_experts=num_physical,
                                   num_local_experts=num_physical // world_size, start=0,
                                   num_logical=scenario["num_logical"])
        nbytes = harness.symm_mem_offsets(layer.get_expert_weights())[-1]
        symm_buffers = [torch.zeros(nbytes, dtype=torch.uint8).share_memory_() for _ in range(world_size)]
    torch.multiprocessing.spawn(
        harness.rank_main,
        args=(world_size, str(tmp_path / "rendezvous"), cases, scenario["num_logical"], scenario["old"],
              scenario["new"], symm_buffers),
        nprocs=world_size, join=True,
    )


def _needs_accelerator(cases):
    # gloo staging and symm_mem synchronize the accelerator stream; torch_nccl's isend/irecv path over gloo
    # with host tensors does not.
    if any(name != "torch_nccl" for name, _ in cases) and not torch.accelerator.is_available():
        pytest.skip("the staged and symm_mem communicators sync a device stream")


@pytest.mark.parametrize(
    "cases",
    [
        [("torch_nccl", "sync"), ("torch_nccl", "sync_without_alphas")],
        [("torch_gloo", "sync"), ("torch_gloo", "async"), ("torch_gloo", "async_batched")],
        [("symm_mem", "sync"), ("symm_mem", "async"), ("symm_mem", "async_batched")],
    ],
    ids=["torch_nccl", "torch_gloo", "symm_mem"],
)
@needs_vllm
def test_vllm_eplb_moves_nvfp4_experts_and_alphas_between_two_ranks(tmp_path, monkeypatch, cases):
    _needs_accelerator(cases)
    _run_ranks(tmp_path, monkeypatch, 2, cases, TWO_RANKS)


@needs_vllm
def test_vllm_eplb_moves_nvfp4_experts_within_one_rank(tmp_path, monkeypatch):
    cases = [("torch_nccl", "sync"), ("torch_nccl", "async"), ("torch_nccl", "sync_without_alphas")]
    _run_ranks(tmp_path, monkeypatch, 1, cases, ONE_RANK)
