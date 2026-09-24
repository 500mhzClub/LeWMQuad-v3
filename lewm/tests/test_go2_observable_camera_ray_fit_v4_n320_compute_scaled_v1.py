from __future__ import annotations

from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
from scripts import run_go2_observable_camera_ray_fit_v4_n320_compute_scaled_v1 as subject


def _numeric(*, passes: bool) -> dict:
    thresholds = asdict(gate.FIT_THRESHOLDS[320])
    contract = subject.frozen.expected_numeric_check_contract(thresholds)
    checks = []
    for index, (name, (comparison, threshold)) in enumerate(contract.items()):
        value = threshold
        if index == 0 and not passes:
            value = threshold - 0.01 if comparison == "greater_than_or_equal" else threshold + 0.01
        checks.append(
            {
                "name": name,
                "comparison": comparison,
                "value": value,
                "threshold": threshold,
                "passes": index != 0 or passes,
            }
        )
    failed = [check for check in checks if not check["passes"]]
    return {
        "fit_size": 320,
        "thresholds": thresholds,
        "wrong_rgb_dependence_assessable": True,
        "check_count": 26,
        "checks": checks,
        "failure_count": len(failed),
        "failed_checks": failed,
        "passes": not failed,
    }


def _review() -> dict:
    core = subject.review_core()
    core["reviewer"] = "/root/different_reviewer"
    return subject.frozen._self_hashed(core)


def _bind(name: str, value: dict) -> tuple[dict, bytes]:
    raw = subject.frozen._json_payload(value)
    return subject._json_binding(name, value, raw), raw


def test_exact_compute_only_contract_and_distinct_namespace() -> None:
    row = subject.row_contract()
    assert row == {
        "seed": 20260710,
        "fit_size": 320,
        "updates": 40_000,
        "batch_size": 5,
        "frame_exposures": 200_000,
        "schedule_sha256": "54cf287353be8942706c6904ef5d39bf227c4eeb37c1f5065a21eea8da1a7117",
        "first_4000_schedule_sha256": "4084f8d5c14989cb76df4f01e4de46b0b6a88537ba607ccc4152795304bc3bd6",
        "key": "seed_20260710_n320_compute_scaled",
    }
    assert subject.OUTPUT_ROOT_RELATIVE_PATH.endswith("/n320_compute_scaled_v1")
    assert subject.RESERVATION_SCHEMA != subject.frozen.ROW_RESERVATION_SCHEMA
    assert subject.RESULT_SCHEMA != subject.frozen.ROW_RESULT_SCHEMA
    assert subject.EXPECTED_INITIAL_STATE_SHA256 == "a03f76eb539480ecb19ed4331ca4dc70eb1b3cba9f1453add4dcdc586a5ae1d2"
    science = subject.science_contract()
    assert science["predecessor_checkpoint_opens"] == 0
    assert science["maximum_attempts"] == 1
    assert science["retry_authorized"] is False


def test_schedule_and_prefix_are_exact_cpu_contract() -> None:
    torch = pytest.importorskip("torch")
    del torch
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    schedule = base._deterministic_training_batches(
        frame_count=320, batch_size=5, steps=40_000, seed=20260710
    )
    assert base.canonical_json_sha256(schedule) == subject.ROW.schedule_sha256
    assert base.canonical_json_sha256(schedule[:4000]) == subject.ROW.first_4000_schedule_sha256


def test_review_binds_current_two_sources_and_reused_committed_runner(tmp_path: Path) -> None:
    review = _review()
    raw = subject.frozen._json_payload(review)
    path = tmp_path / "review.json"
    path.write_bytes(raw)
    loaded, loaded_raw = subject.validate_review(
        hashlib.sha256(raw).hexdigest(), path=path
    )
    assert loaded == review
    assert loaded_raw == raw
    assert loaded["terminal_predecessor"] == subject.predecessor_contract()
    assert loaded["reused_ladder_source"]["file_sha256"] == subject.OLD_RUNNER_FILE_SHA256
    assert set(loaded["successor_sources"]) == {
        subject.RUNNER_RELATIVE_PATH,
        subject.TEST_RELATIVE_PATH,
    }


def test_review_rejects_implementation_author_as_reviewer(tmp_path: Path) -> None:
    core = subject.review_core()
    core["reviewer"] = subject.IMPLEMENTATION_AUTHOR
    review = subject.frozen._self_hashed(core)
    raw = subject.frozen._json_payload(review)
    path = tmp_path / "review.json"
    path.write_bytes(raw)
    with pytest.raises(PermissionError):
        subject.validate_review(hashlib.sha256(raw).hexdigest(), path=path)


def test_terminal_predecessor_pins_full_gate_path_and_never_authorizes_load() -> None:
    predecessor = subject.predecessor_contract()
    assert predecessor["gate"]["path"] == "rows/row_03_seed_20260710_n320/gate.json"
    assert predecessor["gate"]["file_sha256"] == "2e26b0081d51dcc19f671962b9ab00cd57f825800b7dc0b3ab65c0401b25d003"
    assert predecessor["reservation"]["file_sha256"] == "0241955da9257792ca5ffc7dceb1d45bd712ea27157c169e041dbe572b1cf347"
    assert predecessor["checkpoint_bound_not_loaded"]["file_sha256"] == "b0f5cc9105cc945bb9d3a6e68a8cf129f8467c661fee21cb3b4a0f3c8f431ab3"
    assert subject.REVIEW_LICENSES["predecessor_checkpoint_model_use_authorized"] is False


@pytest.mark.parametrize("passes", [True, False])
def test_gate_is_five_file_terminal_and_shared_v5_is_pass_conditional(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, passes: bool
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    monkeypatch.setattr(subject, "OUTPUT_ROOT", root)
    reservation = subject.frozen._self_hashed({"schema": subject.RESERVATION_SCHEMA})
    reservation_binding, reservation_raw = _bind("reservation.json", reservation)
    (root / "reservation.json").write_bytes(reservation_raw)
    checkpoint_raw = b"development-checkpoint"
    (root / "checkpoint.pt").write_bytes(checkpoint_raw)
    checkpoint_binding = subject.frozen.artifact_binding(
        "checkpoint.pt", checkpoint_raw, content_sha256="0" * 64
    )
    result = subject.frozen._self_hashed({"schema": subject.RESULT_SCHEMA})
    result_binding, result_raw = _bind("result.json", result)
    (root / "result.json").write_bytes(result_raw)
    artifacts = {
        "reservation": reservation_binding,
        "checkpoint": checkpoint_binding,
        "result": result_binding,
    }
    metric = subject.frozen._self_hashed(
        {
            "schema": subject.METRIC_SCHEMA,
            "row": subject.row_contract(),
            "artifacts": artifacts,
            "numeric_gate": _numeric(passes=passes),
            "retry_authorized": False,
        }
    )
    gate_value, _binding = subject.publish_metric_and_gate(
        review={}, artifacts=artifacts, metric=metric
    )
    assert sorted(path.name for path in root.iterdir()) == subject.SUCCESS_INVENTORY
    assert gate_value["status"] == ("passed" if passes else "failed_numeric_gate")
    assert gate_value["licenses"]["shared_v5_development_use_authorized"] is passes
    assert all(
        gate_value["licenses"][name] is False
        for name in (
            "g2_authorized", "navigation_authorized", "heldout_authorized",
            "production_authorized", "promotion_authorized",
        )
    )
    with pytest.raises(PermissionError, match="reservation fields changed"):
        subject.validate_terminal_bundle({})


def test_infrastructure_failure_removes_owned_partials_and_is_no_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    root.mkdir()
    monkeypatch.setattr(subject, "OUTPUT_ROOT", root)
    reservation = subject.frozen._self_hashed({"schema": subject.RESERVATION_SCHEMA})
    _binding_value, reservation_raw = _bind("reservation.json", reservation)
    (root / "reservation.json").write_bytes(reservation_raw)
    for name in ("checkpoint.pt", "result.json", "metric_verification.json"):
        (root / name).write_bytes(b"partial")
    failed = subject.terminate_failure(
        review={}, reservation=reservation, reservation_raw=reservation_raw,
        stage="test", error=RuntimeError("boom"),
    )
    assert sorted(path.name for path in root.iterdir()) == subject.FAILURE_INVENTORY
    assert failed["retry_authorized"] is False
    assert failed["licenses"]["shared_v5_development_use_authorized"] is False


def test_post_reservation_commit_failure_is_terminal_two_file_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "attempt"
    monkeypatch.setattr(subject, "OUTPUT_ROOT", root)
    calls = 0

    def fail_first_fsync(_path: Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("synthetic reservation fsync failure")

    monkeypatch.setattr(subject.frozen, "_fsync_directory", fail_first_fsync)
    inputs = type("Inputs", (), {"subset_receipt": {"content_sha256": "1" * 64}})()
    with pytest.raises(OSError):
        subject.reserve(
            review={}, predecessor={}, inputs=inputs,
            target_partition={"content_sha256": "2" * 64}, initialization={"attempt_identity": "3" * 64},
            resource={}, determinism={},
        )
    assert sorted(path.name for path in root.iterdir()) == subject.FAILURE_INVENTORY


def test_internal_verifier_has_no_caller_artifact_protocol() -> None:
    args = subject.parse_args(["--internal-verify"])
    assert args.internal_verify is True
    source = Path(subject.__file__).read_text(encoding="utf-8")
    assert "stdin=subprocess.DEVNULL" in source
    with pytest.raises(ValueError):
        subject.parse_args(["--internal-verify", "--review-sha256", "0" * 64])
