from __future__ import annotations

from copy import deepcopy

from lewm.benchmarks.go2_physical_claim_canonical import (
    canonical_content_sha256,
    canonical_content_sha256_valid,
    canonical_json_equal,
)


def test_canonical_equality_is_type_exact_for_json_scalars() -> None:
    assert canonical_json_equal({"value": True}, {"value": True})
    assert not canonical_json_equal({"value": True}, {"value": 1})
    assert not canonical_json_equal({"value": False}, {"value": 0})
    assert not canonical_json_equal({"value": 1}, {"value": 1.0})
    assert not canonical_json_equal({"value": 0.0}, {"value": -0.0})


def test_content_hash_validation_rejects_type_change_with_stale_hash() -> None:
    value = {"accepted": True, "count": 1}
    value["content_sha256"] = canonical_content_sha256(
        value, hash_field="content_sha256"
    )
    assert canonical_content_sha256_valid(value, hash_field="content_sha256")
    for field, replacement in (("accepted", 1), ("count", 1.0)):
        changed = deepcopy(value)
        changed[field] = replacement
        assert not canonical_content_sha256_valid(
            changed, hash_field="content_sha256"
        )
