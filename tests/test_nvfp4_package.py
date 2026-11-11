"""CPU checks of the NVFP4 MegaMoE plugin: opt-in activation, buckets and weight layouts."""

import os
import subprocess
import sys

import pytest


def test_nvfp4_installation_without_opt_in_does_not_import_vllm_or_cuda():
    code = """
import sys
import vllm_nvfp4_megamoe
vllm_nvfp4_megamoe.register()
assert not {'vllm', 'torch', 'cutlass', 'flashinfer', 'triton'} & sys.modules.keys()
"""
    env = {k: v for k, v in os.environ.items() if k != "VLLM_PLUGINS"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_nvfp4_refuses_both_megamoe_plugins():
    code = """
import vllm_nvfp4_megamoe
try:
    vllm_nvfp4_megamoe.register()
except RuntimeError as error:
    assert "sm107_megamoe" in str(error)
else:
    raise AssertionError("listing both plugins must fail")
"""
    env = dict(os.environ, VLLM_PLUGINS="sm107_megamoe,nvfp4_megamoe")
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_buckets(monkeypatch):
    from vllm_nvfp4_megamoe.config import bucket_for, token_buckets

    monkeypatch.delenv("VLLM_NVFP4_MEGAMOE_BUCKETS", raising=False)
    buckets = token_buckets(8190)
    assert buckets[-1] == 8192 and all(b % 128 == 0 for b in buckets)
    assert bucket_for(buckets, 4608) == 4608 and bucket_for(buckets, 4609) == 5632
    assert bucket_for(buckets, 1) == buckets[0]
    monkeypatch.setenv("VLLM_NVFP4_MEGAMOE_BUCKETS", "512,4096")
    assert token_buckets(5000) == (512, 4096, 5120)
    with pytest.raises(ValueError):
        bucket_for(token_buckets(5000), 5121)


def test_layout_writers_match_the_reference_transforms():
    torch = pytest.importorskip("torch")
    from vllm_nvfp4_megamoe import layout

    inter, hidden = 256, 512
    g = torch.Generator().manual_seed(0)
    w1, w3 = (torch.randint(0, 256, (inter, hidden // 2), generator=g, dtype=torch.uint8) for _ in range(2))
    s1, s3 = (torch.randint(0, 256, (inter, hidden // 16), generator=g, dtype=torch.uint8) for _ in range(2))
    w2_scale = torch.randint(0, 256, (hidden, inter // 16), generator=g, dtype=torch.uint8)

    data = torch.zeros(2 * inter, hidden // 2, dtype=torch.uint8)
    layout.write_gate_up_data(data, w1, 0)
    layout.write_gate_up_data(data, w3, 1)
    assert torch.equal(data, layout.interleave_reference(torch.cat([w1, w3]), inter))

    scales = torch.zeros(2 * inter * (hidden // 16), dtype=torch.uint8)
    layout.write_gate_up_half(scales, s1, 0)
    layout.write_gate_up_half(scales, s3, 1)
    expected = layout.to_blocked_reference(layout.interleave_reference(torch.cat([s1, s3]), inter))
    assert torch.equal(scales, expected)

    flat = torch.zeros(hidden * (inter // 16), dtype=torch.uint8)
    layout.write_plain(flat, w2_scale)
    assert torch.equal(flat, layout.to_blocked_reference(w2_scale))

    block = torch.randint(110, 130, (inter // 32, hidden // 32), generator=g, dtype=torch.uint8)
    expanded = layout.expand_block_scales(block, inter, hidden, 32)
    assert expanded.shape == (inter, hidden // 32)
    assert torch.equal(expanded[::32], block) and torch.equal(expanded[31::32], block)
