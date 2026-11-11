# SPDX-License-Identifier: Apache-2.0
"""Token buckets, kernel settings and environment knobs of the NVFP4 MegaMoE plugin.

The kernel's ``max_tokens_per_rank`` is compile-time, and its shared expert and
router always cover all of them, so the plugin compiles one variant per token
bucket and pads every call to the smallest bucket that holds it. Every EP rank
must pick the same bucket (the variants differ in their symmetric layouts).

Environment:

``VLLM_NVFP4_MEGAMOE_KERNEL_DIR``
    Checkout of NVIDIA cutedsl_megamoe with the shared-expert fusion (the
    ``sources`` package). Required for NVFP4 layers.
``VLLM_NVFP4_MEGAMOE_BUCKETS``
    Comma-separated token buckets (multiples of 128). The largest bucket is
    always ``round_up(max_num_batched_tokens, 128)``.
``VLLM_NVFP4_MEGAMOE_IMPL``
    JSON object of kernel-setting overrides applied to every bucket (keys of
    ``KernelSettings``).
``VLLM_NVFP4_MEGAMOE_L2_MARGIN``
    Headroom factor on the FC2 input global scale (default 1.0).
"""

import dataclasses
import importlib
import json
import logging
import os
import sys

logger = logging.getLogger("vllm.plugins.nvfp4_megamoe")

KERNEL_DIR_ENV = "VLLM_NVFP4_MEGAMOE_KERNEL_DIR"
DEFAULT_BUCKETS = (256, 1024, 2048, 3072, 4096, 4608, 5632, 6144, 8192)
TOKEN_ROW_ALIGN = 128  # shared-expert scale atom rows


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def token_buckets(max_num_tokens: int) -> tuple[int, ...]:
    top = round_up(max_num_tokens, TOKEN_ROW_ALIGN)
    text = os.environ.get("VLLM_NVFP4_MEGAMOE_BUCKETS", "")
    chosen = [int(v) for v in text.split(",") if v.strip()] if text else list(DEFAULT_BUCKETS)
    for value in chosen:
        if value <= 0 or value % TOKEN_ROW_ALIGN:
            raise ValueError(f"NVFP4 MegaMoE bucket {value} is not a positive multiple of {TOKEN_ROW_ALIGN}")
    return tuple(sorted({v for v in chosen if v < top} | {top}))


def bucket_for(buckets: tuple[int, ...], num_tokens: int) -> int:
    for bucket in buckets:
        if num_tokens <= bucket:
            return bucket
    raise ValueError(f"{num_tokens} tokens exceed the largest NVFP4 MegaMoE bucket {buckets[-1]}")


@dataclasses.dataclass(frozen=True)
class KernelSettings:
    """Per-bucket implementation settings of BlockScaledSwapAbMegaMoeKernel."""

    mma_tiler_mnk: tuple[int, int, int] = (256, 256, 256)
    cluster_shape_mn: tuple[int, int] = (4, 1)
    fallback_cluster_shape_mn: tuple[int, int] | None = (2, 1)
    schedule_policy: tuple[str, int | None] = ("phase_interleave", 2)
    fc2_tma_stages: int = 1
    epi_flag_batches: tuple[int, int] = (2, 4)
    token_in_flag_batch: int = 1
    fc2_use_bulk: bool = True
    token_back_mode: str = "epi_warps"
    work_id_mode: str = "atomic_counter"
    token_padding_block: int = 64
    sf_padding_block: int = 128


def _ints(text) -> tuple[int, ...]:
    if isinstance(text, (tuple, list)):
        return tuple(int(v) for v in text)
    return tuple(int(v) for v in str(text).split(","))


def _schedule(text) -> tuple[str, int | None]:
    if isinstance(text, (tuple, list)):
        return (str(text[0]), None if len(text) < 2 or text[1] is None else int(text[1]))
    mode, _, hint = str(text).partition(",")
    return mode.strip(), (int(hint) if hint.strip() not in ("", "none", "None") else None)


def _from_published(params: dict) -> KernelSettings:
    """KernelSettings from one entry of the checkout's EP4 parameter tables."""
    fallback = params.get("fallback_cluster_shape_mnk")
    return KernelSettings(
        mma_tiler_mnk=_ints(params["mma_tiler_mnk"]),
        cluster_shape_mn=_ints(params["cluster_shape_mnk"])[:2],
        fallback_cluster_shape_mn=None if fallback is None else _ints(fallback)[:2],
        schedule_policy=_schedule(params["schedule_policy"]),
        fc2_tma_stages=int(params.get("fc2_tma_stages") or 1),
        epi_flag_batches=_ints(params["epi_flag_batches"]),
        token_in_flag_batch=int(params.get("token_in_flag_batch") or 1),
        fc2_use_bulk=bool(params.get("fc2_use_bulk", True)),
        token_back_mode=str(params.get("token_back_mode", "epi_warps")),
        work_id_mode=str(params.get("work_id_mode", "atomic_counter")),
    )


def _published_table():
    """``get_params(model, quant, tokens)`` of the kernel checkout, if it ships one."""
    kernel_dir = os.environ.get(KERNEL_DIR_ENV)
    if not kernel_dir:
        return None
    if kernel_dir not in sys.path:
        sys.path.append(kernel_dir)
    try:
        return importlib.import_module("ci.gb200_perf_guard.best_params").get_params
    except Exception as error:  # noqa: BLE001 - optional table; defaults apply
        logger.warning("NVFP4 MegaMoE: no tuned parameter table in %s (%s); using defaults", kernel_dir, error)
        return None


def kernel_settings(bucket: int, hidden: int, intermediate: int, experts: int, topk: int) -> KernelSettings:
    settings = KernelSettings()
    get_params = _published_table()
    if get_params is not None and (hidden, intermediate, experts, topk) == (5120, 2304, 384, 6):
        table_tokens = next((t for t in (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768) if bucket <= t), None)
        if table_tokens is not None:
            try:
                settings = _from_published(get_params("deepseek_v4_1_flash", "nvfp4", table_tokens))
            except Exception as error:  # noqa: BLE001 - fall back to defaults
                logger.warning("NVFP4 MegaMoE: tuned parameters for %d tokens unusable (%s)", table_tokens, error)
    overrides = os.environ.get("VLLM_NVFP4_MEGAMOE_IMPL")
    if overrides:
        values = json.loads(overrides)
        for key in ("mma_tiler_mnk", "cluster_shape_mn", "epi_flag_batches"):
            if key in values:
                values[key] = _ints(values[key])
        if "fallback_cluster_shape_mn" in values and values["fallback_cluster_shape_mn"] is not None:
            values["fallback_cluster_shape_mn"] = _ints(values["fallback_cluster_shape_mn"])
        if "schedule_policy" in values:
            values["schedule_policy"] = _schedule(values["schedule_policy"])
        settings = dataclasses.replace(settings, **values)
    return settings


def l2_margin() -> float:
    value = float(os.environ.get("VLLM_NVFP4_MEGAMOE_L2_MARGIN", "1.0"))
    if not value >= 1.0:
        raise ValueError("VLLM_NVFP4_MEGAMOE_L2_MARGIN must be >= 1")
    return value
