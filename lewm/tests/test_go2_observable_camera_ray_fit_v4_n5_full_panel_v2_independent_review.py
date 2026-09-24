"""Independent hostile review probes for the full-panel V2 executor.

Failing tests are review findings.  They exercise the explicit requirement that
no ordinary importable object graph expose mutable authority lifecycle state.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import FunctionType
from typing import Any

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v2 as policy,
)
from lewm.tests.n5_full_panel_v2_test_support import active_test_authority


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_SOURCES = {
    policy.POLICY_RELATIVE_PATH: (
        "096b597b0e84a6822fd8fcdd8221da27e95757aaa2c05ca148afad6e23ad60d2"
    ),
    policy.LAUNCHER_RELATIVE_PATH: (
        "03311bb48da80b912c2576844adf5cd488c1b9a0818268d2252902d860436591"
    ),
    policy.TRAINER_RELATIVE_PATH: (
        "357369b652c489ab99937c06afaed0ec4cf66aa1f46017f74f5dac46da93d3aa"
    ),
    policy.VERIFIER_RELATIVE_PATH: (
        "cab757839c3d784cb5760f30c2bde6163311bfbf87df1620c9c0f77ff69b624b"
    ),
    policy.FINALIZER_RELATIVE_PATH: (
        "a5dc625b8b270913df56d8b5044c263ba3fdbd1ef6cb3e6f62e084a5335ee323"
    ),
}


def _closure_value(function: FunctionType, name: str) -> Any:
    cells = dict(zip(function.__code__.co_freevars, function.__closure__ or ()))
    assert name in cells, f"expected {function.__name__} to close over {name}"
    return cells[name].cell_contents


def _test_records(
    capability: policy.TestAuthorityCapabilityV2,
) -> dict[int, policy._AuthorityRecord]:
    test_scope = _closure_value(policy._test_transition_authority, "test_scope")
    scopes = _closure_value(test_scope, "test_scopes")
    return scopes[id(capability)][2]


def test_frozen_v2_source_and_v1_block_parents_match_handoff() -> None:
    for relative, expected in EXPECTED_SOURCES.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert hashlib.sha256(
        (ROOT / policy.V1_REVIEW_RELATIVE_PATH).read_bytes()
    ).hexdigest() == policy.V1_REVIEW_FILE_SHA256
    assert hashlib.sha256(
        (ROOT / policy.V1_BLOCK_RELATIVE_PATH).read_bytes()
    ).hexdigest() == policy.V1_BLOCK_FILE_SHA256
    assert hashlib.sha256(
        (ROOT / policy.V1_EXPLOIT_TEST_RELATIVE_PATH).read_bytes()
    ).hexdigest() == policy.V1_EXPLOIT_TEST_FILE_SHA256


def test_importable_authority_api_exposes_no_mutable_lifecycle_registry() -> None:
    exposed = {
        name: value
        for function in (
            policy.verify_authority,
            policy.require_verified_authority,
            policy.transition_authority,
            policy.create_test_authority_capability,
        )
        for name, cell in zip(
            function.__code__.co_freevars,
            function.__closure__ or (),
        )
        if isinstance((value := cell.cell_contents), dict)
    }
    assert exposed == {}, (
        "importable function closures expose mutable authority lifecycle "
        f"registries: {sorted(exposed)}"
    )


def test_consumed_authority_cannot_be_reset_through_importable_closure(
    tmp_path: Path,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    capability.transition(
        authority,
        target_path=attempt,
        from_states=("active",),
        to_state="terminal",
    )
    records = _test_records(capability)
    record = records[id(authority)]
    records[id(authority)] = policy._AuthorityRecord(
        authority=authority,
        issuance_digest=record.issuance_digest,
        state="active",
    )

    with pytest.raises(PermissionError, match="consumed|replayed|one use"):
        capability.transition(
            authority,
            target_path=attempt,
            from_states=("active",),
            to_state="terminal",
        )


def test_reconstructed_authority_cannot_be_registered_through_importable_closure(
    tmp_path: Path,
) -> None:
    capability, authority, attempt = active_test_authority(tmp_path)
    forged = object.__new__(policy.VerifiedAuthorityV2)
    for slot in policy.VerifiedAuthorityV2.__slots__:
        object.__setattr__(forged, slot, object.__getattribute__(authority, slot))

    records = _test_records(capability)
    records[id(forged)] = policy._AuthorityRecord(
        authority=forged,
        issuance_digest=policy._authority_digest(forged),
        state="active",
    )

    with pytest.raises(PermissionError, match="forged|cloned|reconstructed"):
        capability.validate(
            forged,
            target_path=attempt,
            allowed_states=("active",),
        )
