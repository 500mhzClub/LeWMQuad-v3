from __future__ import annotations

import hashlib
from pathlib import Path

from lewm.models import shared_observable_camera_ray_jepa_v5_authority as authority
from scripts import finalize_go2_shared_jepa_v5_gate as finalizer_wrapper
from scripts import go2_shared_jepa_v5_launcher as launcher
from scripts import publish_go2_shared_jepa_v5_checkpoint as publisher_wrapper
from scripts import run_go2_shared_jepa_v5_gate as runner_wrapper
from scripts.check_go2_shared_v5_source_closure import (
    PENDING_GENERATED_AUTHORITIES,
    ROOT,
    build_manifest,
    verify_manifest,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_recursive_manifest_matches_every_local_source_byte() -> None:
    verify_manifest(require_tracked=False)
    manifest = build_manifest()

    assert manifest["source_count"] == len(manifest["source_paths"])
    assert len(manifest["source_paths"]) == len(set(manifest["source_paths"]))
    assert not any(path.startswith("config/") for path in manifest["source_paths"])
    assert manifest["pending_generated_authorities"] == list(
        PENDING_GENERATED_AUTHORITIES
    )


def test_captured_core_and_launcher_hash_chain_is_closed() -> None:
    core_path = ROOT / "scripts/go2_shared_jepa_v5_one_shot.py"
    launcher_path = ROOT / "scripts/go2_shared_jepa_v5_launcher.py"

    assert launcher.EXPECTED_CORE_FILE_SHA256 == _sha256(core_path)
    for wrapper in (runner_wrapper, finalizer_wrapper, publisher_wrapper):
        assert wrapper.EXPECTED_LAUNCHER_FILE_SHA256 == _sha256(launcher_path)


def test_every_production_authority_binding_remains_fail_closed() -> None:
    wrapper_bindings = (
        runner_wrapper.CANONICAL_G2_RUNNER_AUTHORITY_FILE_SHA256,
        runner_wrapper.CANONICAL_G3_RUNNER_AUTHORITY_FILE_SHA256,
        finalizer_wrapper.CANONICAL_G2_FINALIZER_AUTHORITY_FILE_SHA256,
        finalizer_wrapper.CANONICAL_G3_FINALIZER_AUTHORITY_FILE_SHA256,
        publisher_wrapper.CANONICAL_G2_CANDIDATE_PUBLISHER_AUTHORITY_FILE_SHA256,
        publisher_wrapper.CANONICAL_FULL_PROMOTION_PUBLISHER_AUTHORITY_FILE_SHA256,
    )
    model_bindings = (
        authority.CANONICAL_AUTHORITY_FILE_SHA256,
        authority.CANONICAL_G2_FINAL_REPORT_FILE_SHA256,
        authority.CANONICAL_G3_FINAL_REPORT_FILE_SHA256,
        authority.CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256,
        authority.CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256,
        authority.CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256,
    )

    assert all(value is None for value in wrapper_bindings)
    assert all(value is None for value in model_bindings)
