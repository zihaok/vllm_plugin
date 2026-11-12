# SPDX-License-Identifier: Apache-2.0
"""GPU checks of EPLB on NVFP4 MegaMoE layers (Rubin), one process per GPU.

Subcommands (E/L default to 400/384, the serving ``num_redundant_experts: 16``):

``staging``
    The plugin's staging with EPLB arguments against vLLM's own MegaMoE staging
    (``prepare_megamoe_inputs``, whose routing block calls the same
    ``eplb_map_and_record``) and against a torch model of the mapping, for random
    placements with 1-3 replicas per expert, random padding, ``num_unpadded`` and
    ``should_record``, int32 and int64 ids: staged ids and ``expert_load_view``
    bit-equal; recording off leaves the load untouched; padding and rows past the
    tokens are -1 and never counted. Without EPLB (no args, or inert args) the
    staged ids/weights equal vLLM's staging without EPLB and every activation byte
    equals nvfp4_gpu_check's torch quantizers. Needs no kernel.
``geometry``
    Two real ``VllmNvfp4MegaMoEExperts`` layers at E physical experts over the EP
    group, every logical expert loaded once plus vLLM's initial replicas, an EPLB
    layer state with the initial placement, and routing biased onto the replicated
    experts on rows 1 and 2 mod 4 (vLLM's hash sends row parity 1 to the replica
    slot, 0 to the home slot): the runtime is built for E physical experts with
    the settings of the L logical experts' tables; staged ids equal the torch
    mapping; the recorded load equals the staged routes; replica slots carry >= 30%
    of the routes to replicated experts; the output matches the FP32 reference of
    ``nvfp4_gpu_check.py`` (rel L2 < 3e-2); eager is deterministic and a two-layer
    CUDA graph replays bit-identically; and (``--baseline``, default when E != L)
    the output is bit-identical to layers at E = L without EPLB. Refuses to run its
    bit-identity checks on combine_dedup buckets.
``dedup``
    ``geometry`` with the production Rubin table (``RUBIN_TABLE=1``, tokens
    8192,6144,4608,96), so the combine_dedup buckets (>= 5760) are compiled and run
    at E physical experts: FP32 reference, mapping, load, determinism and graph
    replay (at 8192) only; no bitwise baseline. Must run as its own process.
``move``
    vLLM's EplbState on two layers whose weight_scale_2 differs between experts;
    CUDA graphs of both layers captured once. Sync rounds with each communicator
    (``rearrange_expert_weights_inplace`` to a random placement, then the map
    commit): every row of all 10 views equals its expert's bytes (alphas
    included), addresses and compiled kernels unchanged, graph replay and eager
    bit-identical to the output before EPLB. Negative controls, each of which must
    be detected: (a) commit a placement without moving the weights; (b) move with
    the alpha views left out of the list; (c) the staging mapping disabled after a
    move. A reload after the moves follows the placement. Then async EPLB through
    EplbState's worker thread and side stream (``--async-communicators``) while
    the graphs replay, through two full rearrangement cycles.
``all``
    staging, then at EP4 geometry 100/96 (25 experts per rank, the DEP16 per-rank
    layout on one node), and at EP4, EP8 or EP16 geometry and move at 400/384.
    Does not include ``dedup`` (another table mode).

Tables: every subcommand but ``dedup`` defaults ``VLLM_NVFP4_MEGAMOE_RUBIN_TABLE``
to ``bitwise`` (the dedup combine rounds per destination rank, so its outputs
legitimately change when experts move); set it yourself to override. Unvalidated
geometries are admitted for these runs (``VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY=1``).
The EPLB-off gate stays ``tests/nvfp4_gpu_check.py`` (E=384, staging_exact, dedup buckets).

usage (one Rubin node):
    torchrun --nproc_per_node=4 tests/nvfp4_eplb_gpu_check.py all [--out FILE]
    torchrun --nproc_per_node=4 tests/nvfp4_eplb_gpu_check.py dedup --experts 400 --logical 384
four nodes x 4 GPUs (DEP16; MASTER_ADDR = first node, one torchrun per node):
    srun -N4 --ntasks-per-node=1 torchrun --nnodes=4 --nproc_per_node=4 --rdzv-backend=c10d \\
        --rdzv-endpoint=$MASTER_ADDR:29500 tests/nvfp4_eplb_gpu_check.py all
    Plain ``srun --ntasks-per-node=4`` works too: RANK, WORLD_SIZE and LOCAL_RANK fall back to
    SLURM_PROCID / SLURM_NTASKS / SLURM_LOCALID (set MASTER_ADDR, MASTER_PORT).
Needs VLLM_NVFP4_MEGAMOE_KERNEL_DIR for ``geometry``, ``dedup`` and ``move``. Exit status 0 only if every check
passed.
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import nvfp4_gpu_check as base  # noqa: E402 - synthetic checkpoint and FP32 reference

H, I, K, CLAMP = base.H, base.I, base.K, base.CLAMP
ALLOW_ENV = "VLLM_NVFP4_MEGAMOE_ALLOW_UNVALIDATED_GEOMETRY"
MODEL_NAME = "nvfp4-megamoe-eplb-gpu-check"
KNUTH = 2654435769
NUM_LAYERS = 2
RESULTS: list[dict] = []


# ------------------------------------------------------------------------------------------------ plumbing
def env_int(*names: str) -> int:
    for name in names:
        if name in os.environ:
            return int(os.environ[name])
    raise RuntimeError(f"none of {names} is set; launch with torchrun or srun")


def log(*args) -> None:
    if dist.get_rank() == 0:
        print(*args, flush=True)


class Coordinator:
    """What the layers, the communicators and EplbState use of vLLM's GroupCoordinator."""

    def __init__(self, device_group, cpu_group) -> None:
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.world_size = device_group.size()
        self.rank_in_group = device_group.rank()
        self.device_communicator = None

    def cpu_all_reduce_(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.cpu_group)
        return tensor


class ModelConfig:
    """Hashable stand-in for vLLM's ModelConfig (EplbState keys its model states by it)."""

    model = MODEL_NAME

    def compute_hash(self) -> str:
        return MODEL_NAME


def init_distributed():
    rank = env_int("RANK", "SLURM_PROCID")
    world = env_int("WORLD_SIZE", "SLURM_NTASKS")
    local = env_int("LOCAL_RANK", "SLURM_LOCALID")
    per_node = os.environ.get("LOCAL_WORLD_SIZE") or os.environ.get("SLURM_NTASKS_PER_NODE") or ""
    match = re.match(r"\d+", per_node)  # SLURM may write e.g. "4(x4)"
    local_world = int(match.group(0)) if match else torch.cuda.device_count()
    os.environ.setdefault("RANK", str(rank))
    os.environ.setdefault("WORLD_SIZE", str(world))
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    ranks = list(range(world))
    ep = Coordinator(dist.new_group(ranks, backend="nccl"), dist.new_group(ranks, backend="gloo"))
    eplb = Coordinator(dist.new_group(ranks, backend="nccl"), dist.new_group(ranks, backend="gloo"))
    nodes = max(1, world // max(1, local_world))

    from vllm.distributed.eplb import async_worker, eplb_state
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    import vllm_nvfp4_megamoe.experts as nv_experts
    import vllm_sm107_megamoe.experts as sm_experts

    for module in (native, sm_experts, nv_experts, eplb_state):
        module.get_ep_group = lambda: ep
    eplb_state.get_eplb_group = lambda: eplb
    async_worker.get_eplb_group = lambda: eplb
    eplb_state.get_node_count = lambda: nodes
    return ep, eplb


def agree(ok: bool, ep: Coordinator) -> bool:
    """True only if ``ok`` on every rank."""
    flag = torch.tensor([int(bool(ok))])
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=ep.cpu_group)
    return bool(flag.item())


def anyone(flag: bool, ep: Coordinator) -> bool:
    value = torch.tensor([int(bool(flag))])
    dist.all_reduce(value, op=dist.ReduceOp.MAX, group=ep.cpu_group)
    return bool(value.item())


def check(name: str, ok: bool, ep: Coordinator, detail: str = "") -> bool:
    ok = agree(ok, ep)
    RESULTS.append({"check": name, "pass": ok, "detail": detail})
    log(f"[{'PASS' if ok else 'FAIL'}] {name}{(': ' + detail) if detail else ''}")
    return ok


def bits(y: torch.Tensor) -> torch.Tensor:
    return y.view(torch.int16)


def round_up(v: int, m: int) -> int:
    return (v + m - 1) // m * m


# ------------------------------------------------------------------------------------- mapping reference
def map_reference(ids, valid, l2p, replicas):
    """vLLM's replica choice in torch: Knuth hash of the row index modulo the replica count; -1 where invalid."""
    rows = torch.arange(ids.shape[0], device=ids.device, dtype=torch.int64)
    hashed = (rows * KNUTH) & 0xFFFFFFFF
    safe = torch.where(valid, ids.long(), torch.zeros_like(ids, dtype=torch.int64))
    replica = hashed[:, None] % replicas[safe]
    physical = l2p[safe, replica]
    return torch.where(valid, physical, torch.full_like(physical, -1))


def load_reference(physical, valid, num_unpadded: int, record: bool, num_physical: int):
    rows = torch.arange(physical.shape[0], device=physical.device)[:, None].expand_as(physical)
    counted = valid & (rows < num_unpadded) & (physical >= 0)
    if not record:
        counted = torch.zeros_like(counted)
    return torch.bincount(physical[counted], minlength=num_physical).to(torch.int32)


def random_placement(num_physical: int, num_logical: int, generator, max_replicas: int = 3) -> torch.Tensor:
    """Every logical expert once plus replicas (at most ``max_replicas`` per expert), shuffled over the slots."""
    extra = num_physical - num_logical
    assert extra <= (max_replicas - 1) * num_logical
    counts = torch.ones(num_logical, dtype=torch.int64)
    while int(counts.sum()) < num_physical:
        e = int(torch.randint(0, num_logical, (1,), generator=generator))
        if counts[e] < max_replicas:
            counts[e] += 1
    slots = torch.repeat_interleave(torch.arange(num_logical), counts)
    return slots[torch.randperm(num_physical, generator=generator)]


def shuffled_placement(num_physical: int, num_logical: int, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    replicas = torch.randperm(num_logical, generator=generator)
    slots = torch.cat([torch.arange(num_logical), replicas[: num_physical - num_logical]])
    return slots[torch.randperm(num_physical, generator=generator)]


# ------------------------------------------------------------------------------------------- 1. staging
def run_staging(cli, ep) -> None:
    from vllm.distributed.eplb.eplb_state import EplbLayerState, compute_logical_maps
    from vllm.model_executor.layers.fused_moe.prepare_megamoe import prepare_megamoe_inputs

    from vllm_nvfp4_megamoe import eplb as nv_eplb
    from vllm_nvfp4_megamoe import staging

    E, L = cli.experts, cli.logical
    rank = dist.get_rank()
    g = torch.Generator().manual_seed(4321 + rank)
    dev = torch.device("cuda", torch.cuda.current_device())
    norm = 1.0 / 2.5e-3
    scale_mul = staging.scale_multiplier(norm)
    ok_all = {"ids": True, "load": True, "weights": True, "live_range": True, "tail": True, "record_off": True,
              "count": True}
    failures = []
    for trial in range(cli.trials):
        n = int(torch.randint(1, 600, (1,), generator=g)) if trial % 3 else int(torch.randint(600, 5000, (1,), generator=g))
        rows = round_up(n, 128) + 128 * int(torch.randint(0, 3, (1,), generator=g))
        placement = random_placement(E, L, g)
        l2p, replicas = compute_logical_maps(placement[None], L)
        l2p, replicas = l2p.to(dev), replicas.to(dev)
        record = bool(trial % 4 != 3)
        num_unpadded = n if trial % 2 else int(torch.randint(0, n + 1, (1,), generator=g))
        pad = None
        if trial % 3 != 0:
            pad = (torch.rand(n, generator=g) < 0.1).to(dev)
        x = torch.randn(n, H, generator=g).to(dev, torch.bfloat16)
        # int64 in half the trials: the native DSV4 gate hands int64 ids (fused_topk_bias with
        # hash_indices_dtype=int64, e.g. decode batches of <= 16 tokens), a separate Triton specialization.
        id_dtype = torch.int64 if (trial // 2) % 2 else torch.int32
        ids = torch.rand(n, L, generator=g).topk(K, dim=1).indices.to(dev, id_dtype)
        w = torch.rand(n, K, generator=g).to(dev)

        def layer_state(load):
            state = EplbLayerState()
            state.set_layer_state(0, load, l2p, replicas)
            state.should_record_tensor = torch.tensor(record, dtype=torch.bool, device=dev)
            state.num_unpadded_tokens_tensors = [torch.tensor(num_unpadded, dtype=torch.int32, device=dev)]
            return state

        load_v = torch.zeros(1, E, dtype=torch.int32, device=dev)
        load_p = torch.zeros(1, E, dtype=torch.int32, device=dev)
        ids_v = torch.full((n, K), -7, dtype=torch.int64, device=dev)
        w_v = torch.empty((n, K), dtype=torch.float32, device=dev)
        prepare_megamoe_inputs(x, w, ids, torch.empty_like(x), None, ids_v, w_v, is_padding=pad, hidden_quant=None,
                               eplb_state=layer_state(load_v))
        out = staging_buffers(rows, dev)
        staging.stage_nvfp4_mxfp8(x, ids, w, pad, norm=norm, scale_mul=scale_mul, rows=rows, **out,
                                  eplb=nv_eplb.eplb_kernel_args(layer_state(load_p)))
        torch.cuda.synchronize()
        valid = torch.ones(n, K, dtype=torch.bool, device=dev) if pad is None else (~pad)[:, None].expand(n, K)
        physical = map_reference(ids, valid, l2p[0], replicas[0])
        load_ref = load_reference(physical, valid, num_unpadded, record, E)
        staged = out["ids_out"]
        checks = {
            "ids": torch.equal(staged[:n].long(), ids_v) and torch.equal(staged[:n].long(), physical),
            "load": torch.equal(load_p, load_v) and torch.equal(load_p[0], load_ref),
            "weights": torch.equal(out["weights_out"][:n], w_v),
            "live_range": bool(((staged[:n][valid] >= 0) & (staged[:n][valid] < E)).all()),
            "tail": bool((staged[n:rows] == -1).all()) and bool((out["weights_out"][n:rows] == 0).all()),
            "record_off": record or int(load_p.sum()) == 0,
            "count": int(load_p.sum()) == (int((valid[:num_unpadded]).sum()) if record else 0),
        }
        for key, value in checks.items():
            if not value:
                ok_all[key] = False
                failures.append(f"trial {trial} n={n} rows={rows} unpadded={num_unpadded} record={record} "
                                f"pad={pad is not None}: {key}")
    if failures and rank == 0:
        print("\n".join(failures[:20]), flush=True)
    check("staging: ids == vLLM staging == torch mapping", ok_all["ids"], ep, f"{cli.trials} trials, E={E} L={L}")
    check("staging: expert_load_view == vLLM == torch", ok_all["load"], ep)
    check("staging: weights == vLLM", ok_all["weights"], ep)
    check("staging: live ids in [0, E)", ok_all["live_range"], ep)
    check("staging: rows past the tokens -1 / 0", ok_all["tail"], ep)
    check("staging: should_record=0 leaves the load untouched", ok_all["record_off"], ep)
    check("staging: one count per live unpadded (row, k)", ok_all["count"], ep)

    # Without EPLB (eplb=None, and the inert args of a layer state EplbState never filled), against independent
    # references rather than against each other (both take the same HAS_EPLB=False specialization): vLLM's own
    # staging without EPLB for ids/weights, and nvfp4_gpu_check's torch quantizers for every activation byte.
    # tests/nvfp4_gpu_check.py (staging_exact, E=384, incl. dedup buckets) remains the full EPLB-off gate.
    n, rows = 1000, 1024
    x = torch.randn(n, H, generator=g).to(dev, torch.bfloat16)
    ids = torch.rand(n, L, generator=g).topk(K, dim=1).indices.to(dev, torch.int32)
    w = torch.rand(n, K, generator=g).to(dev)
    pad = (torch.rand(n, generator=g) < 0.1).to(dev)
    plain, inert = staging_buffers(rows, dev), staging_buffers(rows, dev)
    staging.stage_nvfp4_mxfp8(x, ids, w, pad, norm=norm, scale_mul=scale_mul, rows=rows, **plain)
    staging.stage_nvfp4_mxfp8(x, ids, w, pad, norm=norm, scale_mul=scale_mul, rows=rows, **inert,
                              eplb=nv_eplb.eplb_kernel_args(EplbLayerState()))
    ids_v = torch.full((n, K), -7, dtype=torch.int64, device=dev)
    w_v = torch.empty((n, K), dtype=torch.float32, device=dev)
    prepare_megamoe_inputs(x, w, ids, torch.empty_like(x), None, ids_v, w_v, is_padding=pad, hidden_quant=None,
                           eplb_state=None)
    torch.cuda.synchronize()
    keep = ~pad
    q, qsf = base.quant_nvfp4_codes(x, norm)
    sq, ssf = base.quant_mxfp8(x)
    sf_rows = round_up(n, 128)

    def matches_references(out) -> bool:
        return (
            torch.equal(out["ids_out"][:n].long(), ids_v) and bool((out["ids_out"][n:] == -1).all())
            and torch.equal(out["weights_out"][:n], w_v) and bool((out["weights_out"][n:] == 0).all())
            and torch.equal(base.nibble_canon(out["act"][:n][keep]), base.nibble_canon(q[keep]))
            and torch.equal(raw(out["act_sf"])[:n][keep], qsf.view(torch.uint8)[keep])
            and torch.equal(raw(out["shared_act"])[:n][keep], sq.view(torch.uint8)[keep])
            and torch.equal(base.unswizzle(out["shared_act_sf"][: sf_rows * (H // 32)], sf_rows, H // 32)[:n][keep],
                            ssf[keep])
        )

    no_eplb_ok = matches_references(plain) and matches_references(inert)
    # The same tokens with the mapping on: the activation bytes do not depend on EPLB.
    placement = random_placement(E, L, g)
    l2p, replicas = (t.to(dev) for t in compute_logical_maps(placement[None], L))
    state = EplbLayerState()
    state.set_layer_state(0, torch.zeros(1, E, dtype=torch.int32, device=dev), l2p, replicas)
    state.should_record_tensor = torch.ones((), dtype=torch.bool, device=dev)
    state.num_unpadded_tokens_tensors = [torch.tensor(n, dtype=torch.int32, device=dev)]
    mapped = staging_buffers(rows, dev)
    staging.stage_nvfp4_mxfp8(x, ids, w, pad, norm=norm, scale_mul=scale_mul, rows=rows, **mapped,
                              eplb=nv_eplb.eplb_kernel_args(state))
    torch.cuda.synchronize()
    passthrough = torch.equal(plain["ids_out"][:n], torch.where(pad[:, None], -1, ids)) and bool(
        (plain["ids_out"][n:] == -1).all())
    acts = all(torch.equal(raw(plain[k]), raw(mapped[k])) for k in ("act", "act_sf", "shared_act", "shared_act_sf"))
    check("staging: without EPLB (None and inert args) == vLLM staging + torch quantizers", no_eplb_ok, ep)
    check("staging: without EPLB the ids pass through", passthrough, ep)
    check("staging: activations independent of the mapping", acts, ep)


def raw(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.uint8)


def staging_buffers(rows: int, dev) -> dict:
    return {
        "act": torch.full((rows, H // 2), 0x5A, dtype=torch.uint8, device=dev),
        "act_sf": torch.full((rows, H // 16), 0x5A, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn),
        "shared_act": torch.full((rows, H), 0x5A, dtype=torch.uint8, device=dev).view(torch.float8_e4m3fn),
        "shared_act_sf": torch.full((round_up(rows, 128) * (H // 32),), 0x5A, dtype=torch.uint8, device=dev),
        "ids_out": torch.full((rows, K), -7, dtype=torch.int32, device=dev),
        "weights_out": torch.full((rows, K), 7.0, dtype=torch.float32, device=dev),
    }


# ---------------------------------------------------------------------------------------- layer building
def make_config(max_tokens: int, eplb=None) -> SimpleNamespace:
    from vllm.config.parallel import EPLBConfig

    return SimpleNamespace(
        # enable_fault_tolerance: read by vLLM's async EPLB worker when a transfer fails (else an AttributeError
        # would hide the real error).
        parallel_config=SimpleNamespace(enable_eplb=eplb is not None, enable_elastic_ep=False,
                                        eplb_config=eplb or EPLBConfig(), num_ubatches=0,
                                        enable_fault_tolerance=False),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=max_tokens),
        compilation_config=SimpleNamespace(static_forward_context={}),
        quant_config=SimpleNamespace(_is_nvfp4_expert_layer=lambda prefix: True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=40)),
    )


def eplb_config(communicator: str, num_redundant: int, use_async: bool):
    from vllm.config.parallel import EPLBConfig

    return EPLBConfig(window_size=4, step_interval=8, num_redundant_experts=num_redundant, use_async=use_async,
                      communicator=communicator)


def scalars(index: int, expert: int, version: int, spread: bool) -> dict:
    """nvfp4_gpu_check's scalars; ``spread`` widens weight_scale_2 between experts (0.5x .. 2.25x more) so stale
    alphas cannot hide, and ``version`` gives a different checkpoint for reload checks."""
    s = dict(base.expert_scalars(index + 1000 * version, expert))
    if spread:
        s["ws2_13"] *= 0.5 + 0.25 * ((expert * 7) % 8)
        s["ws2_2"] *= 0.5 + 0.25 * ((expert * 3) % 8)
    return s


def tensors(index: int, expert: int, shard: str, version: int):
    return base.expert_tensors(index + 1000 * version, expert, shard)


def make_layers(config, ep, num_physical: int, num_logical: int, tag: str) -> list:
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    import vllm_nvfp4_megamoe.experts as nv_experts

    num_local = num_physical // ep.world_size
    layers = []
    with torch.device("cuda", torch.cuda.current_device()):
        for index in range(NUM_LAYERS):
            layer = nv_experts.VllmNvfp4MegaMoEExperts(
                config, prefix=f"{tag}.layers.{index}.ffn.experts", num_experts=num_physical,
                num_local_experts=num_local, experts_start_idx=ep.rank_in_group * num_local,
                num_logical_experts=num_logical, top_k=K, hidden_size=H, intermediate_size=I,
                weight_format=native.MEGA_MOE_MXFP4, activation=native.MegaMoEActivation(clamp=CLAMP),
                shared_experts=base.shared_mlp_stub(),
            )
            assert layer._nvfp4 and layer.has_fused_shared_experts
            layers.append(layer)
    return layers


def load_layer(layer, index: int, num_logical: int, version: int = 0, spread: bool = False) -> None:
    """Every logical expert through the weight loaders (globals on every rank, data where it is placed)."""
    for expert in range(num_logical):
        s = scalars(index, expert, version, spread)
        placed = bool(layer._map_global_expert_id(expert))
        for shard in ("w1", "w2", "w3"):
            prefix = "w2" if shard == "w2" else "w13"
            values = {
                "input_scale": torch.tensor(s[f"{shard}_input_scale"], dtype=torch.float32),
                "weight_scale_2": torch.tensor(s["ws2_2" if shard == "w2" else "ws2_13"], dtype=torch.float32),
            }
            if placed:
                values["weight"], values["weight_scale"] = tensors(index, expert, shard, version)
            for suffix, value in values.items():
                name = f"{prefix}_{suffix}"
                layer.weight_loader(getattr(layer, name), value, f"model.layers.{index}.ffn.experts.{name}", shard,
                                    expert)
    for shard in ("w1", "w3", "w2"):
        data, scale = base.shared_tensors(index, shard)
        prefix = f"model.layers.{index}.ffn.shared_experts.{shard}"
        layer.load_shared_expert_weight(f"{prefix}.weight", data)
        layer.load_shared_expert_weight(f"{prefix}.weight_scale_inv", scale)


def global_scales(layer) -> tuple[float, float]:
    from vllm_nvfp4_megamoe.config import l2_margin

    return float(layer._a1_in.max()), float(layer._a2_in.max()) * l2_margin()


def expected_rows(index: int, expert: int, version: int, spread: bool, a1: float, a2: float) -> list:
    """The 10 EPLB view rows of logical ``expert``, written with the plugin's layout functions."""
    from vllm_nvfp4_megamoe import layout

    dev = torch.device("cuda", torch.cuda.current_device())
    s = scalars(index, expert, version, spread)
    (d1, s1), (d3, s3), (d2, s2) = (tensors(index, expert, sh, version) for sh in ("w1", "w3", "w2"))
    w13 = torch.zeros(2 * I, H // 2, dtype=torch.uint8, device=dev)
    layout.write_gate_up_data(w13, d1, 0)
    layout.write_gate_up_data(w13, d3, 1)
    w13_sf = torch.zeros(2 * I * (H // 16), dtype=torch.uint8, device=dev)
    layout.write_gate_up_half(w13_sf, s1.view(torch.uint8), 0)
    layout.write_gate_up_half(w13_sf, s3.view(torch.uint8), 1)
    w2_sf = torch.zeros(H * (I // 16), dtype=torch.uint8, device=dev)
    layout.write_plain(w2_sf, s2.view(torch.uint8))

    def f32(*values):
        return torch.tensor(values, dtype=torch.float32, device=dev)

    return [
        w13.reshape(-1), w13_sf, f32(s["ws2_13"], s["ws2_13"]), f32(s["w1_input_scale"], s["w3_input_scale"]),
        d2.reshape(-1), w2_sf, f32(s["ws2_2"]), f32(s["w2_input_scale"]),
        f32(s["ws2_13"]) * a1, f32(s["ws2_2"]) * a2,
    ]


def layer_holds(layer, index: int, placement, version: int = 0, spread: bool = False) -> tuple[bool, str]:
    """Every local row of all 10 views holds the logical expert ``placement`` puts there."""
    a1, a2 = global_scales(layer)
    views = layer.get_expert_weights()
    if len(views) != 10:
        return False, f"{len(views)} views"
    cache: dict = {}
    for row in range(layer.num_local_experts):
        expert = int(placement[layer.experts_start_idx + row])
        if expert not in cache:
            cache[expert] = expected_rows(index, expert, version, spread, a1, a2)
        for k, (view, want) in enumerate(zip(views, cache[expert])):
            if not torch.equal(view[row].reshape(-1), want.reshape(-1)):
                return False, f"layer {index} row {row} expert {expert} view {k}"
    return True, ""


def alphas_consistent(layer) -> bool:
    a1, a2 = global_scales(layer)
    return torch.equal(layer._fc1_alpha, layer.w13_weight_scale_2.data[:, 0].float() * a1) and torch.equal(
        layer._fc2_alpha, layer.w2_weight_scale_2.data.float() * a2)


def snapshot(layers) -> tuple:
    """Addresses of every parameter and buffer, the bound kernel views and the compiled kernels."""
    runtime = layers[0].get_symm_buffer()
    return (
        [[p.data_ptr() for p in layer.parameters()] + [b.data_ptr() for b in layer.buffers()] for layer in layers],
        [id(layer._nvfp4_kwargs) for layer in layers],
        {b: id(k) for b, k in runtime.compiled.items()},
    )


def attach_state(layer, placement, num_logical: int, unpadded: torch.Tensor, record: torch.Tensor) -> None:
    """A one-layer EPLB state (what EplbState.add_model sets) for ``placement`` (physical -> logical)."""
    from vllm.distributed.eplb.eplb_state import compute_logical_maps

    dev = unpadded.device
    l2p, replicas = compute_logical_maps(placement[None].cpu(), num_logical)
    layer.set_eplb_state(0, torch.zeros(1, placement.numel(), dtype=torch.int32, device=dev), l2p.to(dev),
                         replicas.to(dev))
    layer.eplb_state.should_record_tensor = record
    layer.eplb_state.num_unpadded_tokens_tensors = [unpadded]


def routing(n: int, num_logical: int, num_replicated: int, seed: int):
    """BF16 tokens and logical top-k; rows 1 and 2 mod 4 prefer the replicated experts (0 .. num_replicated-1).

    vLLM picks the replica as Knuth(row) % replicas, and with 2 replicas (the initial placement) that is row % 2
    (the multiplier is odd): row 1 mod 4 lands on the replica slot, row 2 mod 4 on the home slot, so both carry
    heavy load. (Biasing only even rows would send every biased route to the home slot.)
    """
    g = base.generator(seed, dist.get_rank(), n)
    x = torch.randn(n, H, generator=g, device="cuda").to(torch.bfloat16)
    scores = torch.rand(n, num_logical, generator=g, device="cuda")
    if num_replicated:
        rows = torch.arange(n, device="cuda") % 4
        biased = (rows == 1) | (rows == 2)
        scores[biased, :num_replicated] += 2.0
    # int64 for small batches, as the native gate hands them (fused_topk_bias, hash_indices_dtype=int64, <= 16).
    ids = scores.topk(K, dim=1).indices.to(torch.int64 if n <= 16 else torch.int32)
    w = torch.rand(n, K, generator=g, device="cuda") + 0.05
    return x, (w / w.sum(1, keepdim=True) * 1.5).float(), ids


def zipf_routing(n: int, num_logical: int, seed: int):
    """Zipf-skewed expert popularity (the same expert order on every rank), so EPLB has load to balance."""
    g = base.generator(11, seed)
    order = torch.randperm(num_logical, generator=torch.Generator().manual_seed(1234))
    popularity = torch.empty(num_logical)
    popularity[order] = 1.0 / torch.arange(1, num_logical + 1, dtype=torch.float) ** 1.2
    ids = torch.multinomial(popularity.cuda().expand(n, -1).contiguous(), K, replacement=False, generator=g)
    w = torch.rand(n, K, generator=g, device="cuda") + 0.05
    x = torch.randn(n, H, generator=g, device="cuda").to(torch.bfloat16)
    return x, (w / w.sum(-1, keepdim=True)).float(), ids.to(torch.int32)


def capture(layers, inputs) -> tuple[list, list]:
    """One CUDA graph per layer (as vLLM's piecewise capture would hold them)."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graphs, outputs = [], []
    for layer in layers:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs.append(layer.forward(*inputs))
        graphs.append(graph)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    return graphs, outputs


def replay_equal(graphs, outputs, reference) -> bool:
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    return all(torch.equal(bits(got), bits(want)) for got, want in zip(outputs, reference))


# ------------------------------------------------------------------------------------------ 2. geometry
def table_settings(buckets, num_experts: int) -> dict:
    """What config.kernel_settings returns per bucket for an expert count (the runtime must use the LOGICAL one)."""
    from vllm_nvfp4_megamoe import config

    return {b: config.kernel_settings(b, H, I, num_experts, K) for b in buckets}


def require_bitwise(max_tokens: int, token_counts, num_logical: int, what: str) -> None:
    """Bit-identity checks need non-dedup settings in every bucket they run (combine_dedup re-rounds per rank)."""
    from vllm_nvfp4_megamoe import config

    buckets = config.token_buckets(max_tokens)
    used = sorted({config.bucket_for(buckets, n) for n in token_counts})
    dedup = [b for b in used if table_settings([b], num_logical)[b].combine_dedup]
    if dedup:
        raise SystemExit(
            f"{what}: buckets {dedup} use combine_dedup (VLLM_NVFP4_MEGAMOE_RUBIN_TABLE="
            f"{os.environ.get('VLLM_NVFP4_MEGAMOE_RUBIN_TABLE')}, VLLM_NVFP4_MEGAMOE_IMPL="
            f"{os.environ.get('VLLM_NVFP4_MEGAMOE_IMPL')}), so its bit-identity checks would fail spuriously; use "
            "RUBIN_TABLE=bitwise for this subcommand and the 'dedup' subcommand for the dedup buckets")


def run_geometry(cli, ep, tolerance_only: bool = False) -> None:
    """``tolerance_only`` (the ``dedup`` subcommand): the production table, checked only against the FP32 reference
    and graph == eager, since the dedup combine is not bit-identical to E = L."""
    E, L = cli.experts, cli.logical
    if E % ep.world_size:
        raise SystemExit(f"{E} experts do not split over {ep.world_size} ranks")
    tokens = [int(v) for v in cli.tokens.split(",") if v]
    max_tokens = max(tokens)
    graph_tokens = max_tokens if tolerance_only else min(4608, max_tokens)
    run_counts = tokens + [min(4608, max_tokens), graph_tokens]
    use_baseline = cli.baseline and E != L and not tolerance_only
    if use_baseline:
        require_bitwise(max_tokens, run_counts, L, "geometry --baseline")
    dev = torch.device("cuda", torch.cuda.current_device())
    started = time.time()
    with_eplb = E != L
    mode = "dedup/tolerance" if tolerance_only else "bitwise"
    config = make_config(max_tokens, eplb_config("symm_mem", E - L, False) if with_eplb else None)
    layers = make_layers(config, ep, E, L, tag=f"geo{E}_{mode.split('/')[0]}")
    placement = torch.arange(E) % L  # vLLM's initial placement: physical p holds logical p % L
    unpadded = torch.zeros((), dtype=torch.int32, device=dev)
    record = torch.ones((), dtype=torch.bool, device=dev)
    for index, layer in enumerate(layers):
        load_layer(layer, index, L)
        if with_eplb:
            attach_state(layer, placement, L, unpadded, record)
        layer._fill_scale_buffers()  # the alphas (else filled by the first forward's finalize)
    torch.cuda.synchronize()
    log(f"[geometry] E={E} L={L} EP{ep.world_size} ({mode}): layers loaded ({time.time() - started:.0f} s)")
    held = all(layer_holds(layer, i, placement)[0] for i, layer in enumerate(layers))
    check(f"geometry {E}/{L}: loaded rows hold the initial placement (10 views)", held, ep)

    baseline = None
    if use_baseline:
        baseline = make_layers(make_config(max_tokens), ep, L, L, tag=f"base{L}")
        for index, layer in enumerate(baseline):
            load_layer(layer, index, L)

    # The runtime must be built for E physical experts but take its settings from the tables of the L logical
    # experts (matched on E = 400 every bucket would silently fall back to KernelSettings()).
    rt = layers[0].get_symm_buffer()
    want = table_settings(rt.buckets, L)
    by_physical = table_settings(rt.buckets, E)
    distinguishing = [b for b in rt.buckets if want[b] != by_physical[b]]
    check(f"geometry {E}/{L}: runtime built for {E} physical / {L} logical experts, settings by the logical count",
          (rt.num_experts, rt.num_logical_experts, rt.num_local_experts) == (E, L, E // ep.world_size)
          and rt.settings == want, ep,
          f"{len(distinguishing)} of {len(rt.buckets)} buckets differ from a lookup by the physical count")
    rubin_on = os.environ.get("VLLM_NVFP4_MEGAMOE_RUBIN_TABLE", "1").strip().lower() not in ("0", "off", "false", "no")
    if with_eplb and L == 384 and rubin_on and rt.buckets[-1] >= 3072:  # the Rubin table covers buckets >= 3072
        check(f"geometry {E}/{L}: the logical-count lookup is observable (some bucket differs from E={E})",
              bool(distinguishing), ep)
    dedup_buckets = [b for b in rt.buckets if rt.settings[b].combine_dedup]
    if tolerance_only:
        used_dedup = sorted({rt.bucket(n) for n in run_counts} & set(dedup_buckets))
        check(f"geometry {E}/{L}: dedup mode runs combine_dedup buckets", bool(used_dedup), ep,
              f"dedup buckets used {used_dedup} (RUBIN_TABLE={os.environ.get('VLLM_NVFP4_MEGAMOE_RUBIN_TABLE')})")

    for idx, n in [(0, n) for n in tokens] + [(1, min(4608, max_tokens))]:
        layer = layers[idx]
        x, w, ids = routing(n, L, E - L, 11 + idx)
        unpadded.fill_(n)
        if with_eplb:
            layer.eplb_state.expert_load_view.zero_()
        y = layer.forward(x, w, ids)
        rt = layer.get_symm_buffer()
        bucket = rt.bucket(n)
        torch.cuda.synchronize()
        sf_rows = round_up(n, 128)
        act, act_sf = rt.act[:n].clone(), rt.act_sf[:n].view(torch.uint8).clone()
        sh_act = rt.sh_act[:n].clone()
        sh_sf = base.unswizzle(rt.sh_sf[: sf_rows * (H // 32)], sf_rows, H // 32)[:n].clone()
        staged = rt.topk_idx[:bucket].clone()
        valid = torch.ones(n, K, dtype=torch.bool, device=dev)
        replicated = ids.long() < (E - L)  # routes that chose an expert with two slots (initial placement)
        if with_eplb:
            state = layer.eplb_state
            physical = map_reference(ids, valid, state.logical_to_physical_map, state.logical_replica_count)
            load_ok = torch.equal(state.expert_load_view, load_reference(physical, valid, n, True, E)) and int(
                state.expert_load_view.sum()) == n * K
        else:
            physical, load_ok = ids.long(), True
        ids_ok = torch.equal(staged[:n].long(), physical) and bool((staged[n:] == -1).all())
        in_range = bool(((staged[:n] >= 0) & (staged[:n] < E)).all())
        norm = layer._input_norm_const
        x_deq = base.dequant_fp4(act, act_sf.view(torch.float8_e4m3fn), H) / norm
        xs_deq = base.dequant_mxfp8(sh_act, sh_sf)
        ref, _ = base.reference(idx, x_deq, xs_deq, ids.long(), w, float(layer._a2_in.max()))
        yf = y.float()
        rel = float((yf - ref).norm() / ref.norm())
        rel_all = torch.tensor([rel], dtype=torch.float64)
        dist.all_reduce(rel_all, op=dist.ReduceOp.MAX, group=ep.cpu_group)
        hits = torch.tensor([int((staged[:n] >= L).sum()), int(replicated.sum())])
        dist.all_reduce(hits, group=ep.cpu_group)
        replica_hits, replicated_routes = int(hits[0]), int(hits[1])
        finite = bool(torch.isfinite(yf).all())
        tag = (f"geometry {E}/{L} layer{idx} M={n} bucket={bucket}"
               f"{' dedup' if rt.settings[bucket].combine_dedup else ''} ids={str(ids.dtype)[6:]}")
        check(f"{tag}: staged ids == torch mapping, tail -1, ids in [0, E)", ids_ok and in_range, ep)
        check(f"{tag}: recorded load == staged routes ({n * K} per rank)", load_ok, ep)
        check(f"{tag}: output vs FP32 reference", finite and float(rel_all) < 3e-2, ep, f"rel_l2 {float(rel_all):.3e}")
        if with_eplb and n >= 64:
            # About half the routes to a replicated expert hash to its replica slot; >= 30% leaves noise headroom.
            fraction = replica_hits / max(1, replicated_routes)
            check(f"{tag}: replica slots carry >= 30% of the routes to replicated experts", fraction >= 0.3, ep,
                  f"{replica_hits} of {replicated_routes} routes ({fraction:.0%}), all ranks")
        if baseline is not None:
            yb = baseline[idx].forward(x, w, ids)
            torch.cuda.synchronize()
            brt = baseline[idx].get_symm_buffer()
            plain = not rt.settings[bucket].combine_dedup and not brt.settings[brt.bucket(n)].combine_dedup
            diff = float((yb.float() - yf).norm() / max(float(yb.float().norm()), 1e-30))
            check(f"{tag}: bit-identical to E={L} without EPLB", plain and torch.equal(bits(yb), bits(y)), ep,
                  f"rel diff {diff:.2e}{'' if plain else '; a bucket uses combine_dedup'}")

    # CUDA graph of both layers back to back, EPLB mapping inside the graph (a dedup bucket in dedup mode).
    n = graph_tokens
    x0, w0, ids0 = routing(n, L, E - L, 99)
    unpadded.fill_(n)
    eager = layers[1].forward(layers[0].forward(x0, w0, ids0), w0, ids0).clone()
    again = layers[1].forward(layers[0].forward(x0, w0, ids0), w0, ids0).clone()
    torch.cuda.synchronize()
    bucket = layers[0].get_symm_buffer().bucket(n)
    check(f"geometry {E}/{L}: two-layer eager forward deterministic (M={n}, bucket {bucket})",
          torch.equal(bits(again), bits(eager)), ep)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = layers[1].forward(layers[0].forward(x0, w0, ids0), w0, ids0)
    torch.cuda.current_stream().wait_stream(stream)
    same = True
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        same &= torch.equal(bits(out), bits(eager))
    check(f"geometry {E}/{L}: two-layer CUDA graph replay == eager (M={n}, bucket {bucket})", same, ep)
    del graph


# --------------------------------------------------------------------------------------------- 3. moves
def run_moves(cli, ep, eplb) -> None:
    from vllm.distributed.eplb import eplb_state as eplb_state_module
    from vllm.distributed.eplb.eplb_communicator import create_eplb_communicator
    from vllm.distributed.eplb.eplb_state import EplbState, _commit_eplb_maps
    from vllm.distributed.eplb.rebalance_execute import rearrange_expert_weights_inplace
    from vllm.model_executor.models.interfaces import MixtureOfExperts

    import vllm_nvfp4_megamoe.experts as nv_experts
    from vllm_nvfp4_megamoe.eplb import INERT_EPLB_ARGS

    E, L = cli.experts, cli.logical
    if E == L or E % ep.world_size:
        raise SystemExit("move needs redundant experts (E > L) that split over the EP ranks")
    rank = dist.get_rank()
    dev = torch.device("cuda", torch.cuda.current_device())
    communicators = [c for c in cli.communicators.split(",") if c]
    max_tokens = max(int(v) for v in cli.tokens.split(",") if v)
    require_bitwise(max_tokens, [cli.move_tokens], L, "move")  # outputs must be bit-identical across moves
    model_cfg = ModelConfig()

    class MoEModel(MixtureOfExperts):
        def __init__(self, layers) -> None:
            self.moe_layers = layers
            self.num_moe_layers = len(layers)
            self.num_expert_groups = 1
            self.num_logical_experts = self.num_routed_experts = L
            self.num_physical_experts = E
            self.num_local_physical_experts = layers[0].num_local_experts
            self.num_redundant_experts = E - L
            self.num_shared_experts = 1
            self.expert_weights = []

    if "pynccl" in communicators:
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        eplb.device_communicator = SimpleNamespace(pynccl_comm=PyNcclCommunicator(eplb.cpu_group, dev))

    # ---- sync rounds, one EplbState (its map tensors are what the graphs captured)
    config = make_config(max_tokens, eplb_config(communicators[0], E - L, False))
    layers = make_layers(config, ep, E, L, tag=f"sync{E}")
    for index, layer in enumerate(layers):
        load_layer(layer, index, L, spread=True)
    model = MoEModel(layers)
    state = EplbState(config.parallel_config, dev)
    state.add_model(model, model_cfg)
    model_state = state.model_states[MODEL_NAME]
    state.step(is_profile=True)
    check(f"move {E}/{L}: EplbState collected 10 views per layer", all(len(v) == 10 for v in model.expert_weights),
          ep)
    num_tokens = cli.move_tokens
    inputs = zipf_routing(num_tokens, L, 100 * rank + 7)
    state.prepare_forward(model_cfg, num_tokens)
    model_state.expert_load_pass.zero_()
    reference = [layer.forward(*inputs).clone() for layer in layers]
    torch.cuda.synchronize()
    sums = [int(layer.eplb_state.expert_load_view.sum()) for layer in layers]
    check(f"move {E}/{L}: forward records one count per route", sums == [num_tokens * K] * NUM_LAYERS, ep,
          f"per-layer load sums {sums}")
    check(f"move {E}/{L}: reference outputs finite", all(bool(torch.isfinite(y.float()).all()) for y in reference),
          ep)
    graphs, outputs = capture(layers, inputs)
    check(f"move {E}/{L}: graph replay == eager before EPLB", replay_equal(graphs, outputs, reference), ep)
    placement = model_state.physical_to_logical_map.cpu().clone()
    held = all(layer_holds(layer, i, placement[i], spread=True)[0] for i, layer in enumerate(layers))
    check(f"move {E}/{L}: initial placement bytes (10 views)", held, ep)
    before = snapshot(layers)
    keep = []  # communicators and buffers stay alive until the end (symm_mem rendezvous)

    def communicator_for(name, weights):
        if weights is model.expert_weights and name == communicators[0]:
            return model_state.communicator, model_state.expert_buffer
        buffer = None if name == "symm_mem" else [torch.empty_like(w) for w in weights[0]]
        communicator = create_eplb_communicator(eplb, name, weights, buffer)
        buffer = communicator.owned_expert_buffer or buffer
        keep.append((communicator, buffer))
        return communicator, buffer

    def move(new, name, weights):
        communicator, buffer = communicator_for(name, weights)
        rearrange_expert_weights_inplace(model_state.physical_to_logical_map, new.to(dev), weights, buffer,
                                         ep.device_group, communicator)
        _commit_eplb_maps(model_state, new_physical_to_logical_map=new)
        torch.cuda.synchronize()

    for round_, name in enumerate(communicators):
        new = torch.stack([shuffled_placement(E, L, 1000 * round_ + i) for i in range(NUM_LAYERS)])
        move(new, name, model.expert_weights)
        moved = int((new != placement).sum())
        placement = new
        held, where = True, ""
        for i, layer in enumerate(layers):
            ok, info = layer_holds(layer, i, new[i], spread=True)
            held &= ok and alphas_consistent(layer)
            where = where or info
        check(f"move {E}/{L} sync {name}: every row of all 10 views follows the placement", held, ep,
              f"{moved} slots changed{(', first mismatch ' + where) if where else ''}")
        check(f"move {E}/{L} sync {name}: graph replay bit-identical", replay_equal(graphs, outputs, reference), ep)
        eager = all(torch.equal(bits(layer.forward(*inputs)), bits(want)) for layer, want in zip(layers, reference))
        check(f"move {E}/{L} sync {name}: eager bit-identical", eager, ep)
        check(f"move {E}/{L} sync {name}: addresses, bound views, compiled kernels unchanged",
              snapshot(layers) == before, ep)

    # ---- negative controls (each must be detected)
    # (a) commit a new placement without moving any weight
    current = placement.clone()
    _commit_eplb_maps(model_state, new_physical_to_logical_map=torch.stack(
        [shuffled_placement(E, L, 7000 + i) for i in range(NUM_LAYERS)]))
    torch.cuda.synchronize()
    detected = anyone(not replay_equal(graphs, outputs, reference), ep)
    check(f"move {E}/{L} negative (a): placement committed without moving weights is detected", detected, ep)
    _commit_eplb_maps(model_state, new_physical_to_logical_map=current)
    torch.cuda.synchronize()
    check(f"move {E}/{L} negative (a): restored", replay_equal(graphs, outputs, reference), ep)

    # (b) move with the alpha views left out of the list
    without_alphas = [views[:-2] for views in model.expert_weights]
    new = torch.stack([shuffled_placement(E, L, 8000 + i) for i in range(NUM_LAYERS)])
    move(new, "torch_gloo", without_alphas)
    placement = new
    stale = anyone(not all(alphas_consistent(layer) for layer in layers), ep)
    detected = anyone(not replay_equal(graphs, outputs, reference), ep)
    check(f"move {E}/{L} negative (b): alphas left behind are stale and change the output", stale and detected, ep)
    for layer in layers:  # the in-place refill (what a re-finalize does) repairs them under the same graphs
        layer._fill_scale_buffers()
    check(f"move {E}/{L} negative (b): in-place alpha refill restores the output",
          replay_equal(graphs, outputs, reference), ep)

    # (c) the staging mapping disabled after a move (eager: the graphs hold the mapping)
    real_args = nv_experts.eplb_kernel_args
    nv_experts.eplb_kernel_args = lambda eplb_state: INERT_EPLB_ARGS
    try:
        unmapped = [layer.forward(*inputs) for layer in layers]
        torch.cuda.synchronize()
    finally:
        nv_experts.eplb_kernel_args = real_args
    detected = anyone(not all(torch.equal(bits(y), bits(want)) for y, want in zip(unmapped, reference)), ep)
    check(f"move {E}/{L} negative (c): staging without the mapping after a move is detected", detected, ep)

    # ---- reload after the moves: every slot the placement assigns receives the new version
    for index, layer in enumerate(layers):
        load_layer(layer, index, L, version=1, spread=True)
        layer._finalize_nvfp4()
    torch.cuda.synchronize()
    held = all(layer_holds(layer, i, placement[i], version=1, spread=True)[0] for i, layer in enumerate(layers))
    check(f"move {E}/{L}: reload after EPLB follows the placement (alphas refilled in place)",
          held and snapshot(layers)[0] == before[0], ep)
    del graphs, outputs

    # ---- async EPLB under CUDA graphs, through EplbState's worker thread
    for name in [c for c in cli.async_communicators.split(",") if c]:
        run_async(cli, ep, name, MoEModel, model_cfg, EplbState, eplb_state_module)


def run_async(cli, ep, name, MoEModel, model_cfg, EplbState, eplb_state_module) -> None:
    E, L = cli.experts, cli.logical
    rank = dist.get_rank()
    dev = torch.device("cuda", torch.cuda.current_device())
    max_tokens = max(int(v) for v in cli.tokens.split(",") if v)
    config = make_config(max_tokens, eplb_config(name, E - L, True))
    layers = make_layers(config, ep, E, L, tag=f"async_{name}_{E}")
    for index, layer in enumerate(layers):
        load_layer(layer, index, L, spread=True)
    model = MoEModel(layers)
    commits: list[int] = []
    move_to_workspace = eplb_state_module._move_to_workspace

    def counting_move_to_workspace(model_state, ep_rank):
        commits.append(model_state.pending_result.layer_idx)
        move_to_workspace(model_state=model_state, ep_rank=ep_rank)

    eplb_state_module._move_to_workspace = counting_move_to_workspace
    try:
        state = EplbState(config.parallel_config, dev)
        state.add_model(model, model_cfg)
        model_state = state.model_states[MODEL_NAME]
        state.step(is_profile=True)
        num_tokens = cli.move_tokens
        inputs = zipf_routing(num_tokens, L, 100 * rank + 9)
        state.prepare_forward(model_cfg, num_tokens)
        reference = [layer.forward(*inputs).clone() for layer in layers]
        graphs, outputs = capture(layers, inputs)
        dist.barrier(group=ep.cpu_group)
        before = snapshot(layers)
        start = model_state.physical_to_logical_map.cpu().clone()
        state.start_async_loop()
        steps, cycles, all_equal, deadline = 0, 0, True, time.monotonic() + cli.async_timeout
        while True:
            state.prepare_forward(model_cfg, num_tokens)
            all_equal &= replay_equal(graphs, outputs, reference)
            state.step()
            steps += 1
            cycles = sum(layer == NUM_LAYERS - 1 for layer in commits)
            done = torch.tensor([int(cycles >= 2 or time.monotonic() > deadline)])
            dist.all_reduce(done, op=dist.ReduceOp.MIN, group=ep.cpu_group)
            if int(done):
                break
        state.drain_async()  # vLLM's own drain: let an in-flight cycle finish without applying it
        tag = f"async {name} {E}/{L}"
        check(f"{tag}: two full rearrangement cycles", cycles >= 2 and state.async_worker_error is None, ep,
              f"steps {steps}, commits {commits[:12]}{'...' if len(commits) > 12 else ''}")
        check(f"{tag}: every replay bit-identical", all_equal, ep)
        final = model_state.physical_to_logical_map.cpu()
        moved = torch.tensor([int((final != start).sum())])
        dist.all_reduce(moved, op=dist.ReduceOp.MAX, group=ep.cpu_group)
        check(f"{tag}: the policy changed the placement", int(moved) > 0, ep, f"{int(moved)} slots")
        held = all(layer_holds(layer, i, final[i], spread=True)[0] and alphas_consistent(layer)
                   for i, layer in enumerate(layers))
        check(f"{tag}: final rows follow the placement (10 views)", held, ep)
        eager = all(torch.equal(bits(layer.forward(*inputs)), bits(want)) for layer, want in zip(layers, reference))
        check(f"{tag}: final replay and eager bit-identical", replay_equal(graphs, outputs, reference) and eager, ep)
        check(f"{tag}: addresses, bound views, compiled kernels unchanged", snapshot(layers) == before, ep)
        del graphs, outputs
    finally:
        eplb_state_module._move_to_workspace = move_to_workspace


# ------------------------------------------------------------------------------------------------- main
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("staging", "geometry", "dedup", "move", "all"))
    parser.add_argument("--experts", type=int, default=None, help="physical experts E (default 400)")
    parser.add_argument("--logical", type=int, default=None, help="logical experts L (default 384)")
    parser.add_argument("--tokens", default=None,
                        help="token counts, the largest sizes the layers (default 4608,1000,96,1; dedup: "
                             "8192,6144,4608,96)")
    parser.add_argument("--trials", type=int, default=24, help="staging trials per rank")
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                        help="geometry: compare bitwise against E=L layers without EPLB")
    parser.add_argument("--communicators", default="symm_mem,torch_gloo,torch_nccl", help="move: sync rounds")
    parser.add_argument("--async-communicators", default="symm_mem,torch_gloo", help="move: async runs")
    parser.add_argument("--move-tokens", type=int, default=512)
    parser.add_argument("--async-timeout", type=float, default=900.0)
    parser.add_argument("--out", default="")
    cli = parser.parse_args()

    os.environ.setdefault(ALLOW_ENV, "1")
    # The runtime cache does not key on the table mode, so one process runs one mode: 'dedup' the production
    # Rubin table (combine_dedup buckets, tolerance checks), everything else the bit-identical entries.
    os.environ.setdefault("VLLM_NVFP4_MEGAMOE_RUBIN_TABLE", "1" if cli.command == "dedup" else "bitwise")
    if cli.tokens is None:
        cli.tokens = "8192,6144,4608,96" if cli.command == "dedup" else "4608,1000,96,1"
    ep, eplb = init_distributed()
    world = ep.world_size
    started = time.time()
    log(f"[nvfp4-eplb] {cli.command} on {world} ranks, RUBIN_TABLE={os.environ['VLLM_NVFP4_MEGAMOE_RUBIN_TABLE']}")
    E, L = cli.experts or 400, cli.logical or 384  # 16 redundant experts: the serving eplb-config

    def geometry(e, l, tolerance_only=False):
        run_geometry(SimpleNamespace(**{**vars(cli), "experts": e, "logical": l}), ep, tolerance_only)

    def moves(e, l):
        run_moves(SimpleNamespace(**{**vars(cli), "experts": e, "logical": l}), ep, eplb)

    if cli.command == "staging":
        run_staging(SimpleNamespace(**{**vars(cli), "experts": E, "logical": L}), ep)
    elif cli.command == "geometry":
        geometry(E, L)
    elif cli.command == "dedup":
        geometry(E, L, tolerance_only=True)
    elif cli.command == "move":
        moves(E, L)
    else:
        run_staging(SimpleNamespace(**{**vars(cli), "experts": 400, "logical": 384}), ep)
        if world == 4:
            geometry(100, 96)  # 25 experts per rank: the DEP16 per-rank layout on one node
        if world in (4, 8, 16):
            geometry(400, 384)
            moves(400, 384)
        else:
            log(f"[nvfp4-eplb] 'all' knows EP4, EP8 and EP16; ran staging only on {world} ranks")
        log("[nvfp4-eplb] reminder: 'all' runs the bit-identical table only; run 'dedup' as a separate process "
            "for the production combine_dedup buckets")

    passed = bool(RESULTS) and all(r["pass"] for r in RESULTS)
    failed = [r["check"] for r in RESULTS if not r["pass"]]
    log(f"\n[nvfp4-eplb] SUMMARY: {sum(r['pass'] for r in RESULTS)}/{len(RESULTS)} checks passed "
        f"in {time.time() - started:.0f} s -> {'ALL PASS' if passed else 'FAIL'}")
    for name in failed:
        log(f"  FAIL {name}")
    if dist.get_rank() == 0 and cli.out:
        with open(cli.out, "w") as handle:
            json.dump({"command": cli.command, "world": world, "pass": passed, "checks": RESULTS}, handle, indent=2)
    dist.barrier(group=ep.cpu_group)
    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
