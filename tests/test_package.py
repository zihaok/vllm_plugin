"""Package activation and provenance checks that need no GPU stack."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def test_installation_without_opt_in_does_not_import_vllm_or_cuda():
    code = """
import sys
import vllm_sm107_megamoe
vllm_sm107_megamoe.register()
assert not {'vllm', 'torch', 'cutlass', 'flashinfer'} & sys.modules.keys()
"""
    env = {k: v for k, v in os.environ.items() if k != "VLLM_PLUGINS"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_gpu_kernel_sources_match_the_qualified_flashinfer_revision():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "UPSTREAM.json").read_text())
    for name, expected in manifest["original_sha256"].items():
        if name.startswith("kernel/"):
            actual = root / "vllm_sm107_megamoe/backend" / name
            assert hashlib.sha256(actual.read_bytes()).hexdigest() == expected, name
