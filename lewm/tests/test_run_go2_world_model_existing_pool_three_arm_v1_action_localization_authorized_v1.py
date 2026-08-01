from __future__ import annotations

from pathlib import Path

import pytest

from scripts import (
    extract_go2_world_model_existing_pool_three_arm_v1_action_localization_v1 as worker,
)
from scripts import (
    run_go2_world_model_existing_pool_three_arm_v1_action_localization_authorized_v1 as supervisor,
)


def _authority() -> dict[str, object]:
    return {
        "source_commit": "1" * 40,
        "review_commit": "2" * 40,
        "execution_head": "3" * 40,
        "caps": {
            "maximum_wall_seconds": worker.MAXIMUM_WALL_SECONDS,
            "maximum_gpu_seconds": 0,
            "maximum_training_updates": 0,
            "maximum_optimizer_steps": 0,
        },
        "runtime": {
            "python_invocation_path": "/exact/python",
        },
        "execution": {
            "worker_path": "/exact/worker.py",
            "checker_path": "/exact/checker.py",
            "supervisor_path": str(worker.SUPERVISOR_PATH),
        },
    }


def test_recursive_source_closure_and_reservation_templates() -> None:
    assert set(worker.REQUIRED_SOURCE_PATHS) == {
        "lewm_package",
        "benchmarks_package",
        "counterfactual_metrics",
        "h6_main_pool_census",
        "localization_metrics",
        "three_arm_metrics",
        "datasets_package",
        "h6_dataset",
        "h6_sequence_contract_v1",
        "h6_sequence_contract_v2",
        "worker",
        "checker",
        "external_supervisor",
    }
    authority_binding = {
        "path": str(worker.AUTHORITY_PATH),
        "file_sha256": "a" * 64,
        "byte_count": 123,
    }
    reservation = worker.expected_reservation(
        _authority(), authority_binding, supervisor_nonce="b" * 64
    )
    assert reservation["status"] == "RESERVED_ATTEMPT_CONSUMED"
    assert reservation["maximum_attempts"] == 1
    assert "<SUPERVISOR_BOUND_RESERVATION_SHA256>" in reservation[
        "worker_command_template"
    ]
    assert "<WORKER_RESULT_SHA256>" in reservation["checker_command_template"]


def test_supervisor_refuses_polluted_campaign_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "campaign" / "attempt_v1"
    monkeypatch.setattr(worker, "ATTEMPT_ROOT", attempt)
    attempt.parent.mkdir()
    (attempt.parent / "unexpected.txt").write_text("occupied", encoding="utf-8")
    with pytest.raises(
        supervisor.LocalizationSupervisionError, match="already occupied"
    ):
        supervisor._reserve(
            _authority(),
            {
                "path": str(worker.AUTHORITY_PATH),
                "file_sha256": "a" * 64,
                "byte_count": 123,
            },
        )


def test_reservation_atomically_materializes_final_namespace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    development_root = tmp_path / "dev"
    development_root.mkdir()
    attempt = development_root / "campaign" / "attempt_v1"
    monkeypatch.setattr(worker, "ATTEMPT_ROOT", attempt)
    authority_binding = {
        "path": str(worker.AUTHORITY_PATH),
        "file_sha256": "a" * 64,
        "byte_count": 123,
    }
    reservation, binding = supervisor._reserve(_authority(), authority_binding)
    assert reservation["status"] == "RESERVED_ATTEMPT_CONSUMED"
    assert binding == worker.file_binding(attempt / "reservation.json")
    assert worker.exact_root_inventory({"reservation.json"}) == [
        "reservation.json"
    ]
    assert not any(
        path.name.startswith(".world_model_action_localization_reservation_")
        for path in development_root.iterdir()
    )


def test_strict_commit_order_rejects_equal_commits() -> None:
    with pytest.raises(worker.LocalizationWorkerError, match="distinct"):
        worker._require_strict_commit_ancestor("a" * 40, "a" * 40, label="test")


def test_supervisor_requires_explicit_empty_gpu_visibility_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="explicitly empty"):
        supervisor._require_explicit_empty_gpu_visibility_before_worker_import()
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "")
    supervisor._require_explicit_empty_gpu_visibility_before_worker_import()


def test_post_root_reservation_failure_is_closed_by_terminal_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempt = tmp_path / "campaign" / "attempt_v1"
    monkeypatch.setattr(worker, "ATTEMPT_ROOT", attempt)
    authority = _authority()
    authority["execution"]["supervisor_path"] = str(Path(supervisor.__file__).resolve())
    authority_binding = {
        "path": str(worker.AUTHORITY_PATH),
        "file_sha256": "a" * 64,
        "byte_count": 123,
    }
    monkeypatch.setattr(
        worker,
        "load_and_validate_authority",
        lambda *_args, **_kwargs: (authority, authority_binding),
    )

    def fail_after_root(
        _authority_value: object, _binding_value: object
    ) -> tuple[dict[str, object], dict[str, object]]:
        attempt.parent.mkdir()
        attempt.mkdir()
        raise supervisor.LocalizationReservationError(
            "synthetic post-root failure", attempt_root_created=True
        )

    monkeypatch.setattr(supervisor, "_reserve", fail_after_root)
    terminal, terminal_binding = supervisor.supervise(
        worker.AUTHORITY_PATH,
        expected_authority_sha256="a" * 64,
        expected_authority_byte_count=123,
    )
    assert terminal["status"] == supervisor.FAILURE_STATUS
    assert terminal["attempt_consumed"] is True
    assert terminal["reservation_binding"] is None
    assert terminal_binding == worker.file_binding(
        attempt / "terminal_supervision.json"
    )
