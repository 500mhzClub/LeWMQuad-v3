from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts import finalize_go2_observable_camera_ray_fit_v4_ladder as runner


def test_finalizer_is_confined_to_v2_successor_root() -> None:
    assert runner.CANONICAL_DEVELOPMENT_ROOT.name == "development_fit_v2"
    assert runner.CANONICAL_ATTEMPT_ROOT.parent == runner.CANONICAL_DEVELOPMENT_ROOT
    assert runner.CANONICAL_GATE_ROOT.parent == runner.CANONICAL_DEVELOPMENT_ROOT
    assert "development_fit_v1" not in str(runner.CANONICAL_METRIC_RECEIPT_ROOT)


@pytest.fixture(autouse=True)
def _synthetic_unit_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
    from scripts import verify_go2_observable_camera_ray_fit_v4_metrics as verifier

    monkeypatch.setattr(runner, "gate", gate)
    monkeypatch.setattr(runner, "metric_verifier", verifier)
    monkeypatch.setattr(
        runner,
        "__name__",
        "_lewm_v4_ca_test.scripts.finalize_go2_observable_camera_ray_fit_v4_ladder",
    )
    monkeypatch.setattr(
        runner,
        "__verified_logical_name__",
        "scripts.finalize_go2_observable_camera_ray_fit_v4_ladder",
        raising=False,
    )


def _canonical_file(path: Path, value: dict[str, object]) -> str:
    payload = runner._canonical_json_bytes(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_caller_hashed_loader_requires_exact_canonical_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    value = {"content_sha256": "a" * 64, "schema": "synthetic"}
    digest = _canonical_file(path, value)
    assert runner.load_caller_hashed_json(
        path, digest, name="synthetic result"
    ) == value
    with pytest.raises(ValueError, match="caller SHA"):
        runner.load_caller_hashed_json(path, "0" * 64, name="synthetic result")

    noncanonical = json.dumps(value, indent=2).encode() + b"\n"
    path.write_bytes(noncanonical)
    with pytest.raises(ValueError, match="canonical"):
        runner.load_caller_hashed_json(
            path,
            hashlib.sha256(noncanonical).hexdigest(),
            name="synthetic result",
        )


def test_loader_rejects_duplicate_json_keys(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.json"
    payload = b'{"schema":"first","schema":"second"}\n'
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        runner.load_caller_hashed_json(
            path,
            hashlib.sha256(payload).hexdigest(),
            name="synthetic result",
        )


def test_imported_finalizer_exposes_no_loader_or_runtime_capability() -> None:
    forbidden = {
        "_ContentAddressedLoader",
        "_ContentAddressedFinder",
        "_ContentAddressedRuntime",
        "_load_content_addressed_runtime",
        "_load_captured_launcher",
        "_capture_canonical_runtime_sources",
        "_reverify_loaded_runtime_sources",
        "_RuntimeModuleProxy",
        "_ACTIVE_RUNTIME_MODULES",
        "_ACTIVE_CONTENT_RUNTIME",
        "RUNTIME_MODULE_PATHS",
        "ALLOWED_UNTRACKED_IMPORT_ROOTS",
    }
    assert forbidden.isdisjoint(vars(runner))


def test_finalizer_import_does_not_preload_repository_dependencies() -> None:
    code = (
        "import sys; "
        "from scripts import finalize_go2_observable_camera_ray_fit_v4_ladder; "
        "blocked=[n for n in sys.modules if n=='lewm' or n.startswith('lewm.') "
        "or n.endswith('launch_go2_observable_camera_ray_fit_v4') "
        "or n.endswith('verify_go2_observable_camera_ray_fit_v4_metrics')]; "
        "assert not blocked, blocked"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.ROOT,
        env={**dict(os.environ), "PYTHONPATH": str(runner.ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_canonical_finalizer_library_run_is_unsupported() -> None:
    code = (
        "from types import SimpleNamespace; "
        "from scripts import finalize_go2_observable_camera_ray_fit_v4_ladder as f; "
        "\ntry: f.run(SimpleNamespace())\n"
        "except PermissionError as e: assert 'library computation is unsupported' in str(e)\n"
        "else: raise AssertionError('canonical finalizer run unexpectedly executed')"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code],
        cwd=runner.ROOT,
        env={**dict(os.environ), "PYTHONPATH": str(runner.ROOT)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_gate_writer_is_immutable_and_no_replace(tmp_path: Path) -> None:
    gate = {"content_sha256": "b" * 64, "schema": "synthetic_gate"}
    output = tmp_path / "gate.json"
    receipt = runner.write_gate_exclusive(
        output, gate, enforce_canonical_root=False
    )
    assert receipt["file_sha256"] == hashlib.sha256(
        runner._canonical_json_bytes(gate) + b"\n"
    ).hexdigest()
    assert json.loads(output.read_text()) == gate
    with pytest.raises(FileExistsError):
        runner.write_gate_exclusive(
            output, gate, enforce_canonical_root=False
        )


def test_stage_mode_plumbs_only_bound_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    args = runner.parse_args(
        [
            "stage",
            "--reservation",
            f"reservation.json:{'0' * 64}",
            "--result",
            f"result.json:{'1' * 64}",
            "--checkpoint",
            f"checkpoint.pt:{'2' * 64}",
            "--completion",
            f"completed.json:{'3' * 64}",
            "--metric-verification",
            f"metric.json:{'6' * 64}",
            "--trainer-authorization",
            f"authorization.json:{'4' * 64}",
            "--trainer-review-record",
            f"review.json:{'5' * 64}",
            "--seed",
            "20260710",
            "--fit-size",
            "5",
        ]
    )
    loaded = {"fit_size": 5}
    artifacts = {"synthetic": "artifacts"}
    metric_receipt = {"synthetic": "metric"}
    finalized = {"schema": "synthetic_gate", "content_sha256": "2" * 64}
    monkeypatch.setattr(
        runner,
        "_validate_stage_artifact_bundle",
        lambda **_kwargs: (loaded, artifacts, metric_receipt),
    )
    monkeypatch.setattr(
        runner.gate,
        "finalize_development_fit_stage_v4",
        lambda result, **kwargs: (
            finalized
            if (
                result is loaded
                and kwargs["artifact_binding"] is artifacts
                and kwargs["metric_verification_receipt"] is metric_receipt
            )
            else (_ for _ in ()).throw(AssertionError("wrong stage plumbing"))
        ),
    )
    monkeypatch.setattr(
        runner,
        "write_gate_exclusive",
        lambda path, gate_value, **kwargs: {
            "path": str(path),
            "content": gate_value,
        },
    )
    receipt = runner.run(args)
    assert receipt["content"] is finalized
    assert receipt["path"].endswith("stage_seed_20260710_n5.json")


def test_gate_writer_removes_output_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "gate.json"
    monkeypatch.setattr(
        runner,
        "_fsync_directory",
        lambda _path: (_ for _ in ()).throw(OSError("synthetic fsync failure")),
    )
    with pytest.raises(OSError, match="synthetic fsync"):
        runner.write_gate_exclusive(
            output,
            {"schema": "synthetic", "content_sha256": "a" * 64},
            enforce_canonical_root=False,
        )
    assert not output.exists()


def test_bound_path_requires_full_sha256() -> None:
    assert runner._parse_bound_path(f"some:path.json:{'a' * 64}") == (
        Path("some:path.json"),
        "a" * 64,
    )
    with pytest.raises(ValueError):
        runner._parse_bound_path("result.json:not-a-hash")


def test_stage_artifact_validation_preflights_both_authorities_before_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempt = tmp_path / "attempts" / "seed_20260710" / "n5"
    attempt.mkdir(parents=True)
    for filename in ("reservation.json", "result.json", "checkpoint.pt", "completed.json"):
        (attempt / filename).write_bytes(b"")
    metric_root = tmp_path / "metric_verifications"
    metric_root.mkdir()
    metric_path = metric_root / "seed_20260710_n5.json"
    metric_path.write_bytes(b"")
    monkeypatch.setattr(runner, "CANONICAL_ATTEMPT_ROOT", tmp_path / "attempts")
    monkeypatch.setattr(runner, "CANONICAL_METRIC_RECEIPT_ROOT", metric_root)
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_preflight_stage_authorization",
        lambda **_kwargs: events.append("trainer_authority") or {},
    )
    monkeypatch.setattr(
        runner.metric_verifier,
        "preflight_metric_verifier_authorization",
        lambda *_args: events.append("metric_authority") or {},
    )
    monkeypatch.setattr(
        runner,
        "_load_bound",
        lambda *_args, **_kwargs: (
            events.append("artifact_bytes"),
            (_ for _ in ()).throw(RuntimeError("stop after order observation")),
        )[1],
    )
    bound = lambda path: f"{path}:{'a' * 64}"
    with pytest.raises(RuntimeError, match="order observation"):
        runner._validate_stage_artifact_bundle(
            seed=20260710,
            fit_size=5,
            reservation_bound=bound(attempt / "reservation.json"),
            result_bound=bound(attempt / "result.json"),
            checkpoint_bound=bound(attempt / "checkpoint.pt"),
            completion_bound=bound(attempt / "completed.json"),
            metric_verification_bound=bound(metric_path),
            authorization_bound=f"authorization.json:{'b' * 64}",
            review_bound=f"review.json:{'c' * 64}",
        )
    assert events == ["trainer_authority", "metric_authority", "artifact_bytes"]


def test_noncanonical_attempt_path_is_rejected_before_authority_or_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "attempts" / "seed_20260710" / "n5"
    canonical.mkdir(parents=True)
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    for directory in (canonical, wrong):
        for filename in ("reservation.json", "result.json", "checkpoint.pt", "completed.json"):
            (directory / filename).write_bytes(b"")
    metric_root = tmp_path / "metric_verifications"
    metric_root.mkdir()
    metric = metric_root / "seed_20260710_n5.json"
    metric.write_bytes(b"")
    monkeypatch.setattr(runner, "CANONICAL_ATTEMPT_ROOT", tmp_path / "attempts")
    monkeypatch.setattr(runner, "CANONICAL_METRIC_RECEIPT_ROOT", metric_root)
    events: list[str] = []
    monkeypatch.setattr(
        runner,
        "_preflight_stage_authorization",
        lambda **_kwargs: events.append("authority"),
    )
    monkeypatch.setattr(
        runner,
        "_load_bound",
        lambda *_args, **_kwargs: events.append("read"),
    )
    bound = lambda path: f"{path}:{'a' * 64}"
    with pytest.raises(PermissionError, match="path is not canonical"):
        runner._validate_stage_artifact_bundle(
            seed=20260710,
            fit_size=5,
            reservation_bound=bound(wrong / "reservation.json"),
            result_bound=bound(canonical / "result.json"),
            checkpoint_bound=bound(canonical / "checkpoint.pt"),
            completion_bound=bound(canonical / "completed.json"),
            metric_verification_bound=bound(metric),
            authorization_bound=f"authorization.json:{'b' * 64}",
            review_bound=f"review.json:{'c' * 64}",
        )
    assert events == []


def test_execution_validator_rejects_dummy_33_to_44_mapping_without_open(
    tmp_path: Path,
) -> None:
    dummy = tmp_path / "stage_seed_20260710_n33.json"
    dummy.write_text("{}\n")
    with pytest.raises(ValueError):
        runner.validate_canonical_stage_gate_for_execution(
            dummy,
            hashlib.sha256(dummy.read_bytes()).hexdigest(),
            expected_seed=20260710,
            expected_next_fit_size=44,
        )


def test_self_consistent_mapping_cannot_bypass_canonical_artifact_reopen(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_root = tmp_path / "gates"
    gate_root.mkdir()
    stage_path = gate_root / "stage_seed_20260710_n5.json"
    stage_core = {
        "schema": runner.gate.STAGE_GATE_SCHEMA,
        "seed": 20260710,
        "fit_size": 5,
        "reviewed_inputs": {
            "trainer_authorization_file_sha256": "a" * 64,
            "trainer_review_record_file_sha256": "b" * 64,
        },
        "artifacts": {
            "reservation": {"file_sha256": "1" * 64},
            "result": {"file_sha256": "2" * 64},
            "checkpoint": {"file_sha256": "3" * 64},
            "completion": {"file_sha256": "4" * 64},
            "metric_verification": {"file_sha256": "5" * 64},
        },
    }
    stage = {
        **stage_core,
        "content_sha256": runner.gate.canonical_json_sha256(stage_core),
    }
    stage_sha = _canonical_file(stage_path, stage)
    monkeypatch.setattr(runner, "CANONICAL_GATE_ROOT", gate_root)
    monkeypatch.setattr(runner.gate, "_validate_stage_gate_schema", lambda _value: None)
    monkeypatch.setattr(runner, "_preflight_reviewed_inputs", lambda _value: None)
    monkeypatch.setattr(
        runner,
        "_validate_stage_artifact_bundle",
        lambda **_kwargs: (_ for _ in ()).throw(
            PermissionError("dummy canonical artifacts do not exist")
        ),
    )
    with pytest.raises(PermissionError, match="dummy canonical artifacts"):
        runner.verify_canonical_stage_gate(
            stage_path,
            stage_sha,
            expected_seed=20260710,
            expected_fit_size=5,
        )


def test_seed_reverification_visits_all_four_canonical_stage_chains(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gate_root = tmp_path / "gates"
    gate_root.mkdir()
    stage_hashes = [f"{index:064x}" for index in range(1, 5)]
    seed_core = {
        "schema": runner.gate.SEED_GATE_SCHEMA,
        "seed": 20260710,
        "reviewed_inputs": {},
        "stage_gate_file_sha256": stage_hashes,
        "seed_20260710_gate": None,
    }
    seed_gate = {
        **seed_core,
        "content_sha256": runner.gate.canonical_json_sha256(seed_core),
    }
    seed_path = gate_root / "seed_20260710.json"
    seed_sha = _canonical_file(seed_path, seed_gate)
    monkeypatch.setattr(runner, "CANONICAL_GATE_ROOT", gate_root)
    monkeypatch.setattr(runner.gate, "_validate_seed_gate_schema", lambda _value: None)
    monkeypatch.setattr(runner, "_preflight_reviewed_inputs", lambda _value: None)
    visited: list[int] = []

    def stage_verify(
        _path: Path,
        _sha: str,
        *,
        expected_seed: int,
        expected_fit_size: int,
        **_kwargs: object,
    ) -> dict[str, object]:
        assert expected_seed == 20260710
        visited.append(expected_fit_size)
        return {"fit_size": expected_fit_size}

    monkeypatch.setattr(runner, "verify_canonical_stage_gate", stage_verify)
    monkeypatch.setattr(
        runner.gate,
        "finalize_development_fit_seed_v4",
        lambda stages, **_kwargs: (
            seed_gate
            if [stage["fit_size"] for stage in stages]
            == list(runner.gate.LADDER_FIT_SIZES)
            else (_ for _ in ()).throw(AssertionError("incomplete stage chain"))
        ),
    )
    verified = runner.verify_canonical_seed_gate(
        seed_path,
        seed_sha,
        expected_seed=20260710,
    )
    assert verified == seed_gate
    assert visited == list(runner.gate.LADDER_FIT_SIZES)
