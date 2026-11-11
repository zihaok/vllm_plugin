# SPDX-License-Identifier: Apache-2.0
"""GPU check of the NVFP4 MegaMoE plugin layer on one Rubin node (EP over all local GPUs).

Real ``VllmNvfp4MegaMoEExperts`` layers at DeepSeek-V4.1-Flash geometry are loaded
through their weight loaders from synthetic NVFP4 checkpoint tensors (the same on
every rank) and checked against an FP32 reference:

1. weight layout: the kernel tensors equal FlashInfer's
   ``preprocess_prequantized_block_scaled_weights`` (or this package's independent
   reference transforms when FlashInfer's helper is unavailable);
2. staging: the staged NVFP4 / MXFP8 tokens, scales and routing equal a torch
   quantizer byte for byte (FP4 negative zero folded);
3. output: ``forward`` against FP32 math on the staged inputs (FC2-input NVFP4
   requantization emulated), for several token counts and both layers;
4. CUDA graphs: a captured forward (two layers back to back) replays bit-identically;
5. timing (L2 flushed): eager and graph forward per token count.

usage: torchrun --nproc_per_node=4 tests/nvfp4_gpu_check.py [--tokens 8190,4608,...] [--out FILE]
Needs VLLM_NVFP4_MEGAMOE_KERNEL_DIR (cutedsl_megamoe with the shared-expert fusion).
"""

import argparse
import json
import os
import statistics
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist

H, I, E, K, CLAMP = 5120, 2304, 384, 6, 10.0
A1_RANGE = (2.0e-3, 3.0e-3)  # w1/w3 input_scale per expert: x ~ N(0, 1) fits 6 * 448 * A1
A2_RANGE = (5.0e-3, 1.1e-2)  # w2 input_scale per expert
E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)


def log(*args):
    if dist.get_rank() == 0:
        print(*args, flush=True)


class Coordinator:
    def __init__(self, device_group, cpu_group) -> None:
        self.device_group = device_group
        self.cpu_group = cpu_group
        self.world_size = device_group.size()
        self.rank_in_group = device_group.rank()
        self.device_communicator = None


def generator(*key: int) -> torch.Generator:
    seed = 0
    for part in key:
        seed = (seed * 1_000_003 + part) % (1 << 62)
    return torch.Generator(device="cuda").manual_seed(seed)


def expert_scalars(layer: int, expert: int) -> dict:
    g = torch.Generator().manual_seed(layer * 100_003 + expert)
    u = torch.rand(4, generator=g).tolist()
    ws2_13 = 2.0 ** -7 * (0.5 + 0.25 * (expert % 4))
    return {
        "w1_input_scale": A1_RANGE[0] + (A1_RANGE[1] - A1_RANGE[0]) * u[0],
        "w3_input_scale": A1_RANGE[0] + (A1_RANGE[1] - A1_RANGE[0]) * u[1],
        "w2_input_scale": A2_RANGE[0] + (A2_RANGE[1] - A2_RANGE[0]) * u[2],
        "ws2_13": ws2_13,
        "ws2_2": 2.0 ** -7 * (0.5 + 0.25 * ((expert // 4) % 4)),
    }


def expert_tensors(layer: int, expert: int, shard: str):
    """Packed E2M1 data and E4M3 block scales of one checkpoint shard."""
    rows, cols = (H, I) if shard == "w2" else (I, H)
    g = generator(layer, expert, {"w1": 1, "w2": 2, "w3": 3}[shard])
    data = torch.randint(0, 256, (rows, cols // 2), generator=g, device="cuda", dtype=torch.uint8)
    scale = (torch.rand(rows, cols // 16, generator=g, device="cuda") * 1.75 + 0.25).to(torch.float8_e4m3fn)
    return data, scale


def shared_tensors(layer: int, shard: str):
    """MXFP8 shared expert shard: E4M3 data and UE8M0 32x32 block scales (as the checkpoint)."""
    rows, cols = (H, I) if shard == "w2" else (I, H)
    g = generator(7, layer, {"w1": 1, "w2": 2, "w3": 3}[shard])
    data = (torch.randn(rows, cols, generator=g, device="cuda") * 0.25).to(torch.float8_e4m3fn)
    scale = torch.randint(121, 125, (rows // 32, cols // 32), generator=g, device="cuda", dtype=torch.uint8)
    return data, scale.view(torch.float8_e8m0fnu)


def shared_mlp_stub():
    def linear(rows, cols):
        return SimpleNamespace(
            weight=torch.empty(rows, cols, dtype=torch.float8_e4m3fn, device="meta"),
            weight_scale=torch.empty(rows // 32, cols // 32, dtype=torch.uint8, device="meta"),
        )

    return SimpleNamespace(gate_up_proj=linear(2 * I, H), down_proj=linear(H, I))


# ------------------------------------------------------------------------------------------------ references
def e2m1_table(device):
    return torch.tensor(E2M1_VALUES, dtype=torch.float32, device=device)


def dequant_fp4(packed: torch.Tensor, scale: torch.Tensor, cols: int) -> torch.Tensor:
    """Packed E2M1 [r, cols/2] x E4M3 per-16 scales -> fp32 [r, cols] (no global scale)."""
    table = e2m1_table(packed.device)
    codes = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(packed.shape[0], cols).long()
    return table[codes] * scale.float().repeat_interleave(16, dim=1)


def e2m1_round(v: torch.Tensor) -> torch.Tensor:
    """Round to nearest-even E2M1 value, saturating at 6 (values only)."""
    a = v.abs()
    code = sum(((a > t) if strict else (a >= t)).int() for t, strict in
               ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)))
    return torch.tensor(E2M1_VALUES[:8], device=v.device)[code] * torch.sign(v)


def quant_nvfp4_codes(x: torch.Tensor, norm: float):
    """FlashInfer quantize_nvfp4_block16 arithmetic -> (packed codes [r, c/2] uint8, e4m3 scales [r, c/16])."""
    r, c = x.shape
    g = x.float().view(r, c // 16, 16)
    scale_mul = float(torch.tensor(norm / 6.0, dtype=torch.float32))
    sf = (g.abs().amax(-1) * scale_mul).clamp(max=448.0).to(torch.float8_e4m3fn)
    sf_f = sf.float()
    v = g * torch.where(sf_f > 0, norm / sf_f, torch.zeros_like(sf_f))[..., None]
    a = v.abs()
    code = sum(((a > t) if strict else (a >= t)).int() for t, strict in
               ((0.25, True), (0.75, False), (1.25, True), (1.75, False), (2.5, True), (3.5, False), (5.0, True)))
    code = code | (((v < 0) & (code != 0)).int() << 3)
    code = code.view(r, c // 2, 2)
    return (code[..., 0] | (code[..., 1] << 4)).to(torch.uint8), sf


def quant_mxfp8(x: torch.Tensor):
    """Power-of-two scale at or above amax/448 per 32 (vLLM MegaMoE staging) -> (e4m3 [r, c], e8m0 bytes [r, c/32])."""
    r, c = x.shape
    g = x.float().view(r, c // 32, 32)
    y = g.abs().amax(-1).clamp_min(1e-30) / 448.0
    bits = y.view(torch.int32)
    exp = (((bits >> 23) & 0xFF) - 127 + ((bits & 0x7FFFFF) != 0).int()).clamp(-127, 127)
    inv = ((127 - exp) << 23).view(torch.float32)
    q = (g * inv[..., None]).clamp(-448.0, 448.0).to(torch.float8_e4m3fn).view(r, c)
    return q, (exp + 127).to(torch.uint8)


def dequant_mxfp8(q: torch.Tensor, sf: torch.Tensor) -> torch.Tensor:
    scale = torch.pow(2.0, sf.float() - 127.0)
    return q.float() * scale.repeat_interleave(32, dim=1)


def fake_nvfp4(x: torch.Tensor, norm: float) -> torch.Tensor:
    """NVFP4 quantize-dequantize against a static global scale (FC2-input requantization)."""
    r, c = x.shape
    g = x.float().view(r, c // 16, 16)
    sf = (g.abs().amax(-1) * (norm / 6.0)).clamp(max=448.0).to(torch.float8_e4m3fn).float()
    q = e2m1_round(g * torch.where(sf > 0, norm / sf, torch.zeros_like(sf))[..., None])
    return (q * sf[..., None] / norm).view(r, c)


def fake_mxfp8(x: torch.Tensor) -> torch.Tensor:
    q, sf = quant_mxfp8(x)
    return dequant_mxfp8(q, sf)


def nibble_canon(b: torch.Tensor) -> torch.Tensor:
    """Fold the FP4 negative zero (nibble 0x8) to +0 in packed bytes."""
    lo, hi = b & 0x0F, b >> 4
    lo = torch.where(lo == 8, torch.zeros_like(lo), lo)
    hi = torch.where(hi == 8, torch.zeros_like(hi), hi)
    return lo | (hi << 4)


def unswizzle(flat: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    rb, cb = rows // 128, cols // 4
    raw = flat.view(torch.uint8).reshape(rb, cb, 32, 4, 4).permute(0, 1, 3, 2, 4).reshape(rb, cb, 128, 4)
    return raw.permute(0, 2, 1, 3).reshape(rows, cols)


def swiglu(h: torch.Tensor) -> torch.Tensor:
    gate, up = h[:, :I], h[:, I:]
    return torch.nn.functional.silu(gate.clamp(max=CLAMP)) * up.clamp(-CLAMP, CLAMP)


def reference(layer_idx, x_deq, xs_deq, ids, weights, a2):
    """FP32 routed + shared expert output for one rank's tokens (inputs already NVFP4 / MXFP8 dequantized)."""
    n = x_deq.shape[0]
    y = torch.zeros(n, H, dtype=torch.float32, device=x_deq.device)
    y_noq = torch.zeros_like(y)
    flat_ids = ids.reshape(-1)
    for expert in torch.unique(flat_ids[flat_ids >= 0]).tolist():
        tok, slot = (ids == expert).nonzero(as_tuple=True)
        s = expert_scalars(layer_idx, expert)
        w13 = torch.cat([dequant_fp4(*expert_tensors(layer_idx, expert, sh), H) for sh in ("w1", "w3")])
        w13 *= s["ws2_13"]
        w2 = dequant_fp4(*expert_tensors(layer_idx, expert, "w2"), I) * s["ws2_2"]
        act = swiglu(x_deq[tok] @ w13.t()) * weights[tok, slot][:, None]
        y.index_add_(0, tok, fake_nvfp4(act, 1.0 / a2) @ w2.t())
        y_noq.index_add_(0, tok, act @ w2.t())
    shared = {}
    for sh in ("w1", "w3", "w2"):
        data, scale = shared_tensors(layer_idx, sh)
        rows, cols = data.shape
        shared[sh] = data.float() * torch.pow(2.0, scale.view(torch.uint8).float() - 127.0).repeat_interleave(
            32, 0).repeat_interleave(32, 1)
    hs = swiglu(xs_deq @ torch.cat([shared["w1"], shared["w3"]]).t())
    y_shared = fake_mxfp8(hs) @ shared["w2"].t()
    return y + y_shared, y_noq + hs @ shared["w2"].t()


# ------------------------------------------------------------------------------------------------ timing
def timed(fn, warmup=10, iters=50):
    flush = torch.empty(300 * 2**20, dtype=torch.uint8, device="cuda")
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(iters)]
    for start, end in pairs:
        flush.zero_()
        torch.cuda._sleep(2_000_000)
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    dist.barrier()
    local = [s.elapsed_time(e) * 1000 for s, e in pairs][1:]
    everyone = [None] * dist.get_world_size()
    dist.all_gather_object(everyone, local)
    # slowest rank per launch, median over launches
    return statistics.median(max(r[i] for r in everyone) for i in range(len(local)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", default="8190,4608,5124,3072,1000,96")
    parser.add_argument("--time-tokens", default="8190,4608")
    parser.add_argument("--out", default="")
    cli = parser.parse_args()
    rank, local, world = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local))
    ranks = list(range(world))
    ep = Coordinator(dist.new_group(ranks, backend="nccl"), dist.new_group(ranks, backend="gloo"))

    from vllm.config.parallel import EPLBConfig
    from vllm.model_executor.layers.fused_moe import deep_gemm_mega_moe as native

    import vllm_nvfp4_megamoe.experts as nv_experts
    import vllm_sm107_megamoe.experts as sm_experts

    for module in (native, sm_experts, nv_experts):
        module.get_ep_group = lambda: ep
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(enable_eplb=False, enable_elastic_ep=False, eplb_config=EPLBConfig(),
                                        num_ubatches=0),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=8190),
        compilation_config=SimpleNamespace(static_forward_context={}),
        quant_config=SimpleNamespace(_is_nvfp4_expert_layer=lambda prefix: True),
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(num_hidden_layers=40)),
    )
    local_experts = E // world
    report = {"world": world, "checks": {}, "timing_us": {}}
    started = time.time()

    def make_layer(index):
        with torch.device("cuda", local):
            layer = nv_experts.VllmNvfp4MegaMoEExperts(
                config, prefix=f"model.layers.{index}.ffn.experts", num_experts=E, num_local_experts=local_experts,
                experts_start_idx=rank * local_experts, num_logical_experts=E, top_k=K, hidden_size=H,
                intermediate_size=I, weight_format=native.MEGA_MOE_MXFP4,
                activation=native.MegaMoEActivation(clamp=CLAMP), shared_experts=shared_mlp_stub(),
            )
        assert layer._nvfp4 and layer.has_fused_shared_experts
        loaded = 0
        for expert in range(E):
            s = expert_scalars(index, expert)
            is_local = rank * local_experts <= expert < (rank + 1) * local_experts
            for shard in ("w1", "w2", "w3"):
                prefix = "w2" if shard == "w2" else "w13"
                tensors = {
                    "input_scale": torch.tensor(s[f"{shard}_input_scale"], dtype=torch.float32),
                    "weight_scale_2": torch.tensor(s["ws2_2" if shard == "w2" else "ws2_13"], dtype=torch.float32),
                }
                if is_local:
                    tensors["weight"], tensors["weight_scale"] = expert_tensors(index, expert, shard)
                for suffix, value in tensors.items():
                    name = f"{prefix}_{suffix}"
                    loaded += bool(layer.weight_loader(getattr(layer, name), value,
                                                       f"model.layers.{index}.ffn.experts.{name}", shard, expert,
                                                       return_success=True))
        assert loaded == local_experts * 12, loaded
        for shard in ("w1", "w3", "w2"):
            data, scale = shared_tensors(index, shard)
            base = f"model.layers.{index}.ffn.shared_experts.{shard}"
            layer.load_shared_expert_weight(f"{base}.weight", data)
            layer.load_shared_expert_weight(f"{base}.weight_scale_inv", scale)
        return layer

    layers = [make_layer(0), make_layer(1)]
    torch.cuda.synchronize()
    log(f"[check] layers built and loaded ({time.time() - started:.1f} s)")

    # ---- 1. layout vs FlashInfer (or the independent reference transforms)
    from vllm_nvfp4_megamoe import layout

    helper = None
    try:
        import importlib.util

        import flashinfer

        path = os.path.join(os.path.dirname(flashinfer.__file__),
                            "moe_ep/kernel_src/sm107/next_cutedsl_megamoe/shim/kernel_helpers.py")
        spec = importlib.util.spec_from_file_location("fi_kernel_helpers", path)
        helper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(helper)
    except Exception as error:  # noqa: BLE001
        log(f"[check] FlashInfer helper unavailable ({error}); using the package's reference transforms")
    layer = layers[0]
    ok = True
    for le in (0, local_experts - 1):
        expert = rank * local_experts + le
        (d1, s1), (d3, s3), (d2, s2) = (expert_tensors(0, expert, sh) for sh in ("w1", "w3", "w2"))
        if helper is not None:
            (f1, f1sf), (f2, f2sf) = helper.preprocess_prequantized_block_scaled_weights(
                torch.cat([d1, d3])[None], d2[None], torch.cat([s1, s3])[None], s2[None], quant_kind="nvfp4",
                hidden_size=H, intermediate_size=I)
            ref = (f1.permute(0, 2, 1).contiguous().view(torch.uint8)[0], f1sf.view(torch.uint8)[0],
                   f2.permute(0, 2, 1).contiguous().view(torch.uint8)[0], f2sf.view(torch.uint8)[0])
        else:
            ref = (layout.interleave_reference(torch.cat([d1, d3]), I),
                   layout.to_blocked_reference(layout.interleave_reference(torch.cat([s1, s3]).view(torch.uint8), I)),
                   d2, layout.to_blocked_reference(s2.view(torch.uint8)))
        mine = (layer.w13_weight[le], layer.w13_weight_scale[le], layer.w2_weight[le], layer.w2_weight_scale[le])
        ok &= all(torch.equal(a.reshape(-1), b.reshape(-1)) for a, b in zip(mine, ref))
    sh = {s: shared_tensors(0, s) for s in ("w1", "w3", "w2")}
    exp = {s: layout.expand_block_scales(sh[s][1], *sh[s][0].shape, 32) for s in sh}
    if helper is not None:
        (g1, g1sf), (g2, g2sf) = helper.preprocess_prequantized_block_scaled_weights(
            torch.cat([sh["w1"][0], sh["w3"][0]])[None], sh["w2"][0][None],
            torch.cat([exp["w1"], exp["w3"]]).view(torch.float8_e8m0fnu)[None], exp["w2"].view(torch.float8_e8m0fnu)[None],
            quant_kind="mxfp8_e4m3", hidden_size=H, intermediate_size=I)
        sref = (g1.permute(0, 2, 1).contiguous().view(torch.uint8)[0], g1sf.view(torch.uint8)[0],
                g2.permute(0, 2, 1).contiguous().view(torch.uint8)[0], g2sf.view(torch.uint8)[0])
    else:
        sref = (layout.interleave_reference(torch.cat([sh["w1"][0], sh["w3"][0]]).view(torch.uint8), I),
                layout.to_blocked_reference(layout.interleave_reference(torch.cat([exp["w1"], exp["w3"]]), I)),
                sh["w2"][0].view(torch.uint8), layout.to_blocked_reference(exp["w2"]))
    smine = (layer.shared_l1_weight, layer.shared_l1_weight_scale, layer.shared_l2_weight, layer.shared_l2_weight_scale)
    ok &= all(torch.equal(a.reshape(-1), b.reshape(-1)) for a, b in zip(smine, sref))
    report["checks"]["layout"] = bool(ok)
    log(f"[check] layout {'PASS' if ok else 'FAIL'} ({'FlashInfer helper' if helper else 'reference transforms'})")

    # ---- 2/3. staging and forward vs reference
    def inputs(n, seed):
        g = generator(seed, rank, n)
        x = torch.randn(n, H, generator=g, device="cuda").to(torch.bfloat16)
        ids = torch.rand(n, E, generator=g, device="cuda").topk(K, dim=1).indices
        w = torch.rand(n, K, generator=g, device="cuda") + 0.05
        w = (w / w.sum(1, keepdim=True) * 1.5).float()
        return x, w, ids

    worst = 0.0
    for idx, n in [(0, int(v)) for v in cli.tokens.split(",")] + [(1, 4608)]:
        layer = layers[idx]
        x, w, ids = inputs(n, 11 + idx)
        y = layer.forward(x, w, ids)
        rt = layer.get_symm_buffer()
        bucket = rt.bucket(n)
        torch.cuda.synchronize()
        sf_rows = (n + 127) // 128 * 128
        staged = (rt.act[:n].clone(), rt.act_sf[:n].view(torch.uint8).clone(), rt.sh_act[:n].clone(),
                  unswizzle(rt.sh_sf[: sf_rows * (H // 32)], sf_rows, H // 32)[:n].clone(),
                  rt.topk_idx[:bucket].clone(), rt.topk_w[:bucket].clone())
        norm = layer._input_norm_const
        q, qsf = quant_nvfp4_codes(x, norm)
        sq, ssf = quant_mxfp8(x)
        canon = nibble_canon
        stage_ok = (torch.equal(canon(staged[0]), canon(q)) and torch.equal(staged[1], qsf.view(torch.uint8))
                    and torch.equal(staged[2].view(torch.uint8), sq.view(torch.uint8)) and torch.equal(staged[3], ssf)
                    and torch.equal(staged[4][:n], ids.int()) and bool((staged[4][n:] == -1).all())
                    and torch.equal(staged[5][:n], w) and bool((staged[5][n:] == 0).all()))
        a2 = float(layer._a2_in.max())
        x_deq = dequant_fp4(staged[0], staged[1].view(torch.float8_e4m3fn), H) / norm
        xs_deq = dequant_mxfp8(staged[2], staged[3])
        ref, ref_noq = reference(idx, x_deq, xs_deq, ids, w, a2)
        yf = y.float()
        rel = float((yf - ref).norm() / ref.norm())
        rel_noq = float((yf - ref_noq).norm() / ref_noq.norm())
        cos = float(torch.nn.functional.cosine_similarity(yf.flatten(), ref.flatten(), dim=0))
        finite = bool(torch.isfinite(yf).all())
        stats = torch.tensor([rel, rel_noq, float(stage_ok), float(finite)], device="cuda")
        everyone = [torch.zeros_like(stats) for _ in range(world)]
        dist.all_gather(everyone, stats)
        rel_max = max(float(t[0]) for t in everyone)
        good = all(bool(t[2]) and bool(t[3]) for t in everyone) and rel_max < 3e-2
        worst = max(worst, rel_max)
        report["checks"][f"layer{idx}_M{n}"] = {
            "bucket": bucket, "rel_l2_vs_emulated": rel_max,
            "rel_l2_vs_unquantized_fc2": max(float(t[1]) for t in everyone),
            "cosine_rank0": cos, "staging_exact": all(bool(t[2]) for t in everyone), "pass": good,
        }
        log(f"[check] layer{idx} M={n} bucket={bucket}: rel_l2 {rel_max:.3e} (vs unquantized FC2 input "
            f"{max(float(t[1]) for t in everyone):.3e}), cos {cos:.6f}, staging "
            f"{'exact' if all(bool(t[2]) for t in everyone) else 'MISMATCH'} -> {'PASS' if good else 'FAIL'}")

    # ---- 4. CUDA graph: two layers back to back, replay == eager
    n = 4608
    x0, w0, ids0 = inputs(n, 99)
    eager = layers[1].forward(layers[0].forward(x0, w0, ids0), w0, ids0).clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = layers[1].forward(layers[0].forward(x0, w0, ids0), w0, ids0)
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    same = torch.equal(out, eager)
    graph.replay()
    torch.cuda.synchronize()
    same &= torch.equal(out, eager)
    flag = torch.tensor([float(same)], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    report["checks"]["graph_two_layers_bitwise"] = bool(flag.item())
    log(f"[check] CUDA graph (2 layers, M={n}) replay == eager: {'PASS' if flag.item() else 'FAIL'}")
    del graph

    # ---- 5. timing
    for n in [int(v) for v in cli.time_tokens.split(",") if v]:
        x, w, ids = inputs(n, 5)
        layer = layers[0]
        report["timing_us"][f"eager_M{n}"] = timed(lambda: layer.forward(x, w, ids))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            layer.forward(x, w, ids)
        torch.cuda.current_stream().wait_stream(stream)
        report["timing_us"][f"graph_M{n}"] = timed(graph.replay)
        del graph
        log(f"[time] M={n}: eager {report['timing_us'][f'eager_M{n}']:.1f} us, graph "
            f"{report['timing_us'][f'graph_M{n}']:.1f} us per layer (staging + kernel, slowest rank, L2 flushed)")

    passed = report["checks"]["layout"] and report["checks"]["graph_two_layers_bitwise"] and all(
        v["pass"] for k, v in report["checks"].items() if k.startswith("layer"))
    report["pass"] = bool(passed)
    report["worst_rel_l2"] = worst
    log(f"[check] {'ALL PASS' if passed else 'FAILED'} in {time.time() - started:.0f} s")
    if rank == 0 and cli.out:
        with open(cli.out, "w") as handle:
            json.dump(report, handle, indent=2)
    dist.barrier()
    dist.destroy_process_group()
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
