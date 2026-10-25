# SPDX-License-Identifier: Apache-2.0
"""Admission gate: geometries admitted only where GPU validation ran.

Pure Python (no GPU): the SM count of Rubin is given explicitly.
"""

import pytest

from vllm_sm107_megamoe.backend import heuristics

HIDDEN, INTERMEDIATE, SHARED = 5120, 2304, 1
NUM_SMS = 212  # Rubin VR200

# (top-k, num_experts, EP size) that ran bit-exact against DeepGEMM 3ec0eb4 in
# the Rubin disagg campaign kernel validation (Slurm jobs 854696 and 854741)
# and that the published gate refused.
VALIDATED_EP_PAIRS = [
    (6, 388, 4),
    (6, 392, 8),
    (6, 400, 8),
    (6, 400, 16),
    (6, 416, 32),
    (3, 128, 16),
    (3, 128, 32),
    (3, 132, 4),
    (3, 136, 8),
    (3, 144, 8),
    (3, 144, 16),
    (3, 160, 32),
]
# Admitted by the published gate; must stay admitted.
PUBLISHED = [(6, 384, ep) for ep in (1, 2, 4, 8, 16, 32)] + [
    (3, 128, ep) for ep in (1, 2, 4, 8)
]
# Structurally valid, but these expert counts never ran at these EP sizes.
NOT_VALIDATED = [
    (6, 400, 4),  # the gen-only matrix's DEP4 with 16 redundant experts
    (3, 144, 4),
    (6, 400, 2),
    (6, 392, 4),
    (6, 388, 2),
    (6, 416, 8),
    (6, 416, 16),
    (3, 160, 8),
    (3, 160, 16),
    (3, 136, 4),
    (3, 132, 2),
    (6, 128, 4),
    (6, 96, 16),
]


def reason(top_k, num_experts, num_ranks, num_tokens=2560, **kwargs):
    return heuristics.get_mega_moe_unsupported_reason(
        num_ranks, num_experts, num_tokens, top_k, HIDDEN, INTERMEDIATE, SHARED,
        num_sms=NUM_SMS, **kwargs,
    )  # fmt: skip


def test_exact_pairs_are_the_validated_ones():
    table = heuristics.get_validated_ep_geometries()
    assert set(table) == {(HIDDEN, INTERMEDIATE, 6), (HIDDEN, INTERMEDIATE, 3)}
    pairs = {(key[2], e, ep) for key, (entries, _) in table.items() for e, ep in entries}
    assert pairs == set(VALIDATED_EP_PAIRS)
    assert all(evidence for _, evidence in table.values())


@pytest.mark.parametrize("num_tokens", [320, 2530, 2560, 5120])
@pytest.mark.parametrize("top_k,num_experts,num_ranks", VALIDATED_EP_PAIRS)
def test_validated_ep_pairs_are_admitted(top_k, num_experts, num_ranks, num_tokens):
    assert reason(top_k, num_experts, num_ranks, num_tokens) is None


@pytest.mark.parametrize("top_k,num_experts,num_ranks", PUBLISHED)
def test_published_geometries_stay_admitted(top_k, num_experts, num_ranks):
    assert reason(top_k, num_experts, num_ranks) is None


@pytest.mark.parametrize("top_k,num_experts,num_ranks", NOT_VALIDATED)
def test_other_ep_sizes_stay_refused(top_k, num_experts, num_ranks):
    refusal = reason(top_k, num_experts, num_ranks)
    assert refusal is not None and "has not been validated" in refusal
    # Refused for lack of GPU evidence, not because the kernel cannot derive it.
    assert reason(top_k, num_experts, num_ranks, allow_unvalidated=True) is None
