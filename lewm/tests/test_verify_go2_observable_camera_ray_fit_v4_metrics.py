from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import verify_go2_observable_camera_ray_fit_v4_metrics as verifier


def test_metric_verifier_is_confined_to_v2_successor_root() -> None:
    assert verifier.CANONICAL_DEVELOPMENT_ROOT.name == "development_fit_v2"
    assert verifier.CANONICAL_ATTEMPT_ROOT.parent == (
        verifier.CANONICAL_DEVELOPMENT_ROOT
    )
    assert "development_fit_v1" not in str(verifier.CANONICAL_RECEIPT_ROOT)


def _write_authorization(path: Path, *, authorized: bool) -> tuple[str, str]:
    review = {
        "independent_reviewer": "synthetic-independent-reviewer" if authorized else None,
        "review_completed": authorized,
        "source_closure_approved": authorized,
        "target_partition_constants_approved": authorized,
    }
    licenses = {
        "authorizes_verification_only_checkpoint_use": authorized,
        "authorizes_selected_train_target_access": authorized,
        "authorizes_selected_train_rgb_access": authorized,
        "authorizes_model_inference": authorized,
        "authorizes_metric_receipt_creation": authorized,
        "authorizes_holdout": False,
        "authorizes_g2": False,
        "authorizes_runtime": False,
        "authorizes_promotion": False,
    }
    core = {
        "schema": verifier.AUTHORIZATION_SCHEMA,
        "status": (
            "authorized_after_independent_review"
            if authorized
            else "pending_independent_review"
        ),
        "authoritative": False,
        "scope": "exact_train_only_checkpoint_metric_reverification",
        "target_partition_boundary": verifier.TARGET_PARTITION_BOUNDARY,
        "review": review,
        "licenses": licenses,
    }
    content_sha = hashlib.sha256(verifier._canonical_json_bytes(core)).hexdigest()
    value = {**core, "content_sha256": content_sha}
    payload = verifier._canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest(), content_sha


def test_bound_metric_authorization_is_narrow_and_library_still_cannot_compute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    called = False

    def forbidden(**_kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        raise AssertionError("protected metric computation ran")

    monkeypatch.setattr(verifier, "_compute_exact_receipt", forbidden)
    authorization = verifier.preflight_metric_verifier_authorization(
        verifier.CANONICAL_AUTHORIZATION_PATH,
        verifier.AUTHORIZATION_FILE_SHA256,
    )
    assert authorization["licenses"] == {
        "authorizes_verification_only_checkpoint_use": True,
        "authorizes_selected_train_target_access": True,
        "authorizes_selected_train_rgb_access": True,
        "authorizes_model_inference": True,
        "authorizes_metric_receipt_creation": True,
        "authorizes_holdout": False,
        "authorizes_g2": False,
        "authorizes_runtime": False,
        "authorizes_promotion": False,
    }
    with pytest.raises(PermissionError, match="library computation is unsupported"):
        verifier.recompute_exact_metric_verification(
            metric_authorization_path=verifier.CANONICAL_AUTHORIZATION_PATH,
            metric_authorization_file_sha256=verifier.AUTHORIZATION_FILE_SHA256,
        )
    assert called is False


def test_authorized_library_preflight_still_cannot_compute(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "metric_authorization.json"
    file_sha, content_sha = _write_authorization(path, authorized=True)
    monkeypatch.setattr(verifier, "CANONICAL_AUTHORIZATION_PATH", path.resolve())
    monkeypatch.setattr(verifier, "AUTHORIZATION_FILE_SHA256", file_sha)
    monkeypatch.setattr(verifier, "AUTHORIZATION_CONTENT_SHA256", content_sha)
    events: list[str] = []

    def compute(*, authorization: dict[str, Any], marker: str) -> dict[str, Any]:
        events.append("compute")
        assert authorization["status"] == "authorized_after_independent_review"
        assert marker == "after-preflight"
        return {"verified": True}

    monkeypatch.setattr(verifier, "_compute_exact_receipt", compute)
    with pytest.raises(PermissionError, match="library computation is unsupported"):
        verifier.recompute_exact_metric_verification(
            metric_authorization_path=path,
            metric_authorization_file_sha256=file_sha,
            marker="after-preflight",
        )
    assert events == []


def test_metric_authorization_path_is_canonical_before_read(tmp_path: Path) -> None:
    copy = tmp_path / "copy.json"
    copy.write_bytes(verifier.CANONICAL_AUTHORIZATION_PATH.read_bytes())
    with pytest.raises(PermissionError, match="path is not canonical"):
        verifier.preflight_metric_verifier_authorization(
            copy,
            hashlib.sha256(copy.read_bytes()).hexdigest(),
        )


def test_imported_metric_verifier_exposes_no_loader_or_runtime_capability() -> None:
    forbidden = {
        "_ContentAddressedLoader",
        "_ContentAddressedFinder",
        "_ContentAddressedRuntime",
        "_load_content_addressed_runtime",
        "_load_captured_launcher",
        "_capture_canonical_runtime_sources",
        "_reverify_loaded_runtime_sources",
        "_ACTIVE_RUNTIME_MODULES",
        "_ACTIVE_CONTENT_RUNTIME",
        "__verified_runtime__",
        "RUNTIME_MODULE_PATHS",
        "ALLOWED_UNTRACKED_IMPORT_ROOTS",
    }
    assert forbidden.isdisjoint(vars(verifier))


def test_target_partition_constants_are_exactly_frozen() -> None:
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate

    assert set(gate.EXPECTED_TARGET_PARTITION_SIGNATURES) == {5, 16, 32, 320}
    binding = gate.target_partition_binding_v4(5)
    assert binding["freeze_file_sha256"] == verifier.TARGET_PARTITION_FREEZE_FILE_SHA256
    assert binding["verifier_file_sha256"] == verifier.TARGET_PARTITION_VERIFIER_FILE_SHA256
    assert gate.validate_target_partition_binding_v4(binding, fit_size=5) == binding


def test_metric_authorization_file_is_canonical_and_self_hashed() -> None:
    raw = verifier.CANONICAL_AUTHORIZATION_PATH.read_bytes()
    value = json.loads(raw)
    assert raw == verifier._canonical_json_bytes(value) + b"\n"
    core = dict(value)
    declared = core.pop("content_sha256")
    assert hashlib.sha256(verifier._canonical_json_bytes(core)).hexdigest() == declared
    assert hashlib.sha256(raw).hexdigest() == verifier.AUTHORIZATION_FILE_SHA256
    assert value["status"] == "authorized_after_independent_review"
    assert value["review"] == {
        "independent_reviewer": "/root/v4_final_independent_review",
        "review_completed": True,
        "source_closure_approved": True,
        "target_partition_constants_approved": True,
    }
    assert value["licenses"] == {
        "authorizes_verification_only_checkpoint_use": True,
        "authorizes_selected_train_target_access": True,
        "authorizes_selected_train_rgb_access": True,
        "authorizes_model_inference": True,
        "authorizes_metric_receipt_creation": True,
        "authorizes_holdout": False,
        "authorizes_g2": False,
        "authorizes_runtime": False,
        "authorizes_promotion": False,
    }


def test_invented_metric_receipt_fails_exact_recomputation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth_path = tmp_path / "metric_authorization.json"
    auth_sha, auth_content = _write_authorization(auth_path, authorized=True)
    receipt_root = tmp_path / "metric_verifications"
    receipt_root.mkdir()
    receipt_path = receipt_root / "seed_20260710_n5.json"
    invented_core = {"schema": "synthetic_invented_perfect_metrics", "score": 1.0}
    invented = {
        **invented_core,
        "content_sha256": hashlib.sha256(
            verifier._canonical_json_bytes(invented_core)
        ).hexdigest(),
    }
    invented_raw = verifier._canonical_json_bytes(invented) + b"\n"
    receipt_path.write_bytes(invented_raw)
    monkeypatch.setattr(verifier, "CANONICAL_AUTHORIZATION_PATH", auth_path.resolve())
    monkeypatch.setattr(verifier, "AUTHORIZATION_FILE_SHA256", auth_sha)
    monkeypatch.setattr(verifier, "AUTHORIZATION_CONTENT_SHA256", auth_content)
    monkeypatch.setattr(verifier, "CANONICAL_RECEIPT_ROOT", receipt_root)
    monkeypatch.setattr(
        verifier,
        "_compute_exact_receipt",
        lambda **_kwargs: {
            "schema": "synthetic_recomputed_metrics",
            "content_sha256": "f" * 64,
        },
    )
    with pytest.raises(PermissionError, match="library computation is unsupported"):
        verifier.reverify_canonical_metric_receipt(
            receipt_path=receipt_path,
            receipt_file_sha256=hashlib.sha256(invented_raw).hexdigest(),
            metric_authorization_path=auth_path,
            metric_authorization_file_sha256=auth_sha,
            seed=20260710,
            fit_size=5,
        )


def test_metric_receipt_writer_is_unavailable_to_library_callers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "metric_verifications"
    monkeypatch.setattr(verifier, "CANONICAL_RECEIPT_ROOT", root)
    value = {
        "seed": 20260710,
        "fit_size": 5,
        "content_sha256": "a" * 64,
    }
    path = root / "seed_20260710_n5.json"
    with pytest.raises(PermissionError, match="library publication is unsupported"):
        verifier._write_exclusive(path, value)
    assert not path.exists()
