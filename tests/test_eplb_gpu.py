# SPDX-License-Identifier: Apache-2.0
"""EPLB over real die-local weights on one Rubin node, with the plugin kernel.

Opt-in (``SM107_MEGAMOE_GPU_TESTS=1``): each test spawns one process per GPU
of a 4-GPU Rubin node and compiles kernel variants on first use. See
``eplb_gpu_harness.py`` for what is real and what is checked. The geometries
are EPLB physical expert counts the admission gate admits at EP4: 128 + 4
(top-3, the DSpark draft) and 384 + 4 (top-6, the target).
"""

import os
import socket
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:  # spawned ranks import the harness by name
    sys.path.insert(0, _HERE)

import eplb_gpu_harness as harness  # noqa: E402

WORLD_SIZE = 4

pytestmark = pytest.mark.skipif(
    os.environ.get("SM107_MEGAMOE_GPU_TESTS") != "1",
    reason="GPU EPLB tests are opt-in: set SM107_MEGAMOE_GPU_TESTS=1 on a Rubin node",
)


@pytest.fixture
def rubin_node(monkeypatch):
    if torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip(f"needs {WORLD_SIZE} GPUs")
    from vllm_sm107_megamoe import api

    if not api.is_localization_available():
        pytest.skip("needs SM107 GPUs with two locality domains")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")


def _spawn(fn, *args) -> None:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    torch.multiprocessing.spawn(
        fn, args=(WORLD_SIZE, port, *args), nprocs=WORLD_SIZE, join=True
    )


def test_sync_eplb_moves_die_local_experts_with_every_communicator(rubin_node):
    _spawn(
        harness.sync_main,
        3,
        128,
        132,
        ("symm_mem", "torch_gloo", "torch_nccl", "pynccl"),
    )


@pytest.mark.parametrize("communicator", ["symm_mem", "torch_gloo"])
def test_async_eplb_under_cuda_graphs_keeps_outputs_bit_identical(
    rubin_node, communicator
):
    _spawn(harness.async_main, 6, 384, 388, communicator)
