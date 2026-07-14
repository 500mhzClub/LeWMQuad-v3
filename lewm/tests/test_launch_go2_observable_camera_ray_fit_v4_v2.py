from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import launch_go2_observable_camera_ray_fit_v4_v2 as launcher


ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _write_review(
    path: Path,
    *,
    reviewer: str = "/root/different_agent",
    corrupt_source: str | None = None,
) -> str:
    sources = {
        relative: {
            "path": relative,
            "file_sha256": (
                "0" * 64
                if relative == corrupt_source
                else hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            ),
        }
        for relative in launcher.SUCCESSOR_SOURCE_PATHS
    }
    core = {
        "schema": launcher.SUCCESSOR_REVIEW_SCHEMA,
        "status": "different_agent_review_passed_frozen_ladder",
        "implementation_author": "/root/g5_perf_closure",
        "reviewer": reviewer,
        "review_completed": True,
        "source_closure_approved": True,
        "n5_reopen_approved": True,
        "successor_sources": sources,
        "predecessor_verifier": {
            "path": launcher.PREDECESSOR_VERIFIER_RELATIVE_PATH,
            "file_sha256": launcher.PREDECESSOR_VERIFIER_FILE_SHA256,
        },
        "failed_invocation": {
            "path": launcher.FAILURE_RECORD_RELATIVE_PATH,
            "file_sha256": launcher.FAILURE_RECORD_FILE_SHA256,
            "exception": (
                "PermissionError: V4 spawned RGB terminal differs from captured source"
            ),
            "phase": "captured_trainer_decode_selected_rgb_before_receipt",
        },
        "n5_artifacts": launcher.N5_ARTIFACT_BINDINGS,
        "execution_policy": launcher.SUCCESSOR_EXECUTION_POLICY,
        "licenses": launcher.SUCCESSOR_LICENSES,
    }
    value = {**core, "content_sha256": hashlib.sha256(_canonical(core)).hexdigest()}
    raw = _canonical(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_successor_review_binds_all_four_sources_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review.json"
    digest = _write_review(path)
    monkeypatch.setattr(launcher, "CANONICAL_SUCCESSOR_REVIEW_PATH", path.resolve())
    review = launcher.preflight_successor_review(path.resolve(), digest)
    assert set(review["successor_sources"]) == set(launcher.SUCCESSOR_SOURCE_PATHS)

    corrupt = tmp_path / "corrupt.json"
    corrupt_digest = _write_review(
        corrupt,
        corrupt_source=launcher.SUCCESSOR_TRAINER_RELATIVE_PATH,
    )
    monkeypatch.setattr(launcher, "CANONICAL_SUCCESSOR_REVIEW_PATH", corrupt.resolve())
    with pytest.raises(PermissionError, match="successor source changed"):
        launcher.preflight_successor_review(corrupt.resolve(), corrupt_digest)


def _base(seed: int, fit_size: int) -> list[str]:
    values = [
        "--successor-review",
        str(launcher.CANONICAL_SUCCESSOR_REVIEW_PATH),
        "--successor-review-sha256",
        "a" * 64,
        "--dataset-manifest",
        "manifest.json",
        "--dataset-manifest-sha256",
        "b" * 64,
        "--audit-receipt",
        "audit.json",
        "--audit-receipt-sha256",
        "c" * 64,
        "--trainer-authorization",
        "authorization.json",
        "--trainer-authorization-sha256",
        "d" * 64,
        "--trainer-review-record",
        "trainer-review.json",
        "--trainer-review-record-sha256",
        "e" * 64,
        "--fit-size",
        str(fit_size),
        "--steps",
        str(launcher.DEFAULT_STEPS[fit_size]),
        "--seed",
        str(seed),
    ]
    if fit_size != 5:
        previous = {16: 5, 32: 16, 320: 32}[fit_size]
        values += [
            "--previous-stage-gate",
            f"stage_seed_{seed}_n{previous}.json",
            "--previous-stage-gate-sha256",
            "f" * 64,
        ]
    if seed == 20260711:
        values += [
            "--seed-20260710-gate",
            "seed_20260710.json",
            "--seed-20260710-gate-sha256",
            "1" * 64,
        ]
    return values


@pytest.mark.parametrize("fit_size", [16, 32, 320])
def test_cli_requires_predecessor_for_every_later_rung(fit_size: int) -> None:
    parsed = launcher.parse_args(_base(20260710, fit_size))
    assert parsed.previous_stage_gate is not None
    missing = _base(20260710, fit_size)
    index = missing.index("--previous-stage-gate")
    del missing[index : index + 4]
    with pytest.raises(PermissionError, match="require.*predecessor gate"):
        launcher.parse_args(missing)


@pytest.mark.parametrize("fit_size", [5, 16, 32, 320])
def test_cli_requires_first_seed_gate_for_every_second_seed_rung(fit_size: int) -> None:
    parsed = launcher.parse_args(_base(20260711, fit_size))
    assert parsed.seed_20260710_gate is not None
    missing = _base(20260711, fit_size)
    index = missing.index("--seed-20260710-gate")
    del missing[index : index + 4]
    with pytest.raises(PermissionError, match="second V4 seed requires"):
        launcher.parse_args(missing)


def test_both_execution_terminals_capture_successor_sources_and_review_first() -> None:
    source = (ROOT / launcher.SUCCESSOR_LAUNCHER_RELATIVE_PATH).read_text()
    for relative in launcher.SUCCESSOR_SOURCE_PATHS:
        assert relative in source
    worker_start = source.index("def _rgb_worker_terminal")
    execute_start = source.index("def _execute_captured_trainer")
    worker = source[worker_start:execute_start]
    execute = source[execute_start:source.index("def parse_args", execute_start)]
    assert worker.index("successor_review = preflight_successor_review") < worker.index(
        "receipt = preflight_exact_authorization"
    )
    assert execute.index("successor_review = preflight_successor_review") < execute.index(
        "receipt = preflight_exact_authorization"
    )
    assert 'load("scripts.train_go2_observable_camera_ray_fit_v4_v2")' in worker
    assert 'load("scripts.train_go2_observable_camera_ray_fit_v4_v2")' in execute
    assert "successor_review_file_sha256" in worker


def test_predecessor_trainer_and_launcher_bytes_remain_frozen() -> None:
    expected = {
        "scripts/train_go2_observable_camera_ray_fit_v4.py": (
            "299980cdcb5ef561102f325bbb3db3dfd7aa8217b8a45446b0437badb8f27cfa"
        ),
        "scripts/launch_go2_observable_camera_ray_fit_v4.py": (
            "71d95ae79cd90c64bee8b06f2787b336d72e2fca1e23fcb0cc52f921350a2ff4"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
