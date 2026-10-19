# SPDX-License-Identifier: Apache-2.0
"""Reload behavior against vLLM's native checkpoint-layout implementation."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")


def test_plugin_reload_preserves_localized_parameters_and_matches_native_bytes(
    monkeypatch,
):
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    from vllm_sm107_megamoe.experts import SM107MegaMoEExperts

    monkeypatch.setattr(native, "check_deep_gemm_mega_moe_supported", lambda *a: None)
    backend = SimpleNamespace(
        is_localization_available=lambda: True,
        get_unsupported_reason_for_mega_moe=lambda *a, **k: None,
        localize=lambda t: t.unflatten(-2, (2, -1)).movedim(-3, 0).contiguous(),
    )
    monkeypatch.setattr(
        SM107MegaMoEExperts, "_import_backend", staticmethod(lambda: backend)
    )
    monkeypatch.setattr(
        "vllm_sm107_megamoe.experts.get_ep_group",
        lambda: SimpleNamespace(world_size=1),
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=4),
        compilation_config=SimpleNamespace(static_forward_context={}),
    )
    kwargs = {
        "num_experts": 2,
        "num_local_experts": 2,
        "experts_start_idx": 0,
        "top_k": 1,
        "hidden_size": 256,
        "intermediate_size": 384,
        "weight_format": native.MEGA_MOE_MXFP4,
        "activation": native.MegaMoEActivation(),
    }
    reference = native.DeepGemmMegaMoEExperts(
        config, prefix="model.layers.0.experts", **kwargs
    )
    actual = SM107MegaMoEExperts(config, prefix="model.layers.1.experts", **kwargs)
    parameters = list(actual.parameters())
    pointers = [p.data_ptr() for p in parameters]
    for seed in (0, 1):
        generator = torch.Generator().manual_seed(seed)
        shards = []
        for expert in range(2):
            for shard in ("w1", "w2", "w3"):
                rows, cols = (256, 384) if shard == "w2" else (384, 256)
                prefix = "w2" if shard == "w2" else "w13"
                for suffix, width, dtype in (
                    ("weight", cols // 2, torch.uint8),
                    ("weight_scale", cols // 32, torch.float8_e8m0fnu),
                ):
                    value = torch.randint(
                        0, 256, (rows, width), generator=generator, dtype=torch.uint8
                    ).view(dtype)
                    shards.append((expert, shard, f"{prefix}_{suffix}", value))
        for index in torch.randperm(len(shards), generator=generator).tolist():
            expert, shard, name, value = shards[index]
            for layer in (reference, actual):
                assert layer.weight_loader(
                    getattr(layer, name),
                    value,
                    f"experts.{name}",
                    shard,
                    expert,
                    return_success=True,
                )
        for name, expected in reference.named_parameters():
            value = getattr(actual, name)
            if value.dim() == 4:
                value = value.movedim(0, 1).flatten(1, 2)
            assert torch.equal(value, expected), name
        assert all(a is b for a, b in zip(parameters, actual.parameters()))
        assert [p.data_ptr() for p in actual.parameters()] == pointers
