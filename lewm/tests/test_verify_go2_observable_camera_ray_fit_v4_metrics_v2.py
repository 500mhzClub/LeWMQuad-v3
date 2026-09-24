from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import launch_go2_observable_camera_ray_fit_v4_v2 as launcher_v2
from scripts import verify_go2_observable_camera_ray_fit_v4_metrics_v2 as verifier


ROOT = Path(__file__).resolve().parents[2]


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _review(path: Path, *, reviewer: str = "/root/different_agent") -> str:
    sources = {
        relative: {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
        for relative in verifier.SUCCESSOR_SOURCE_PATHS
    }
    core = {
        "schema": verifier.SUCCESSOR_REVIEW_SCHEMA,
        "status": "different_agent_review_passed_frozen_ladder",
        "implementation_author": "/root/g5_perf_closure",
        "reviewer": reviewer,
        "review_completed": True,
        "source_closure_approved": True,
        "n5_reopen_approved": True,
        "successor_sources": sources,
        "predecessor_verifier": {
            "path": verifier.PREDECESSOR_VERIFIER_RELATIVE_PATH,
            "file_sha256": verifier.PREDECESSOR_VERIFIER_FILE_SHA256,
        },
        "failed_invocation": {
            "path": verifier.FAILURE_RECORD_RELATIVE_PATH,
            "file_sha256": verifier.FAILURE_RECORD_FILE_SHA256,
            "exception": (
                "PermissionError: V4 spawned RGB terminal differs from captured source"
            ),
            "phase": "captured_trainer_decode_selected_rgb_before_receipt",
        },
        "n5_artifacts": verifier.N5_ARTIFACT_BINDINGS,
        "execution_policy": launcher_v2.SUCCESSOR_EXECUTION_POLICY,
        "licenses": launcher_v2.SUCCESSOR_LICENSES,
    }
    value = {**core, "content_sha256": hashlib.sha256(_canonical(core)).hexdigest()}
    raw = _canonical(value) + b"\n"
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def test_successor_covers_frozen_ladder_and_uses_frozen_inline_decoder() -> None:
    for seed in (20260710, 20260711):
        for fit_size in (5, 16, 32, 320):
            assert verifier.canonical_metric_receipt_path(seed, fit_size).name == (
                f"seed_{seed}_n{fit_size}.json"
            )
    for seed, fit_size in ((20260709, 5), (20260710, 4), (20260711, 321)):
        with pytest.raises(ValueError, match="outside the frozen ladder"):
            verifier.canonical_metric_receipt_path(seed, fit_size)

    tree = ast.parse((ROOT / verifier.SUCCESSOR_RELATIVE_PATH).read_text())
    decode_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "decode_selected_rgb"
    ]
    assert len(decode_calls) == 1
    keywords = {keyword.arg: keyword.value for keyword in decode_calls[0].keywords}
    assert isinstance(keywords["maximum_workers"], ast.Constant)
    assert keywords["maximum_workers"].value == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == "ProcessPoolExecutor"
        for node in ast.walk(tree)
    )


def test_successor_review_must_be_different_agent_and_bind_exact_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review.json"
    digest = _review(path)
    monkeypatch.setattr(verifier, "CANONICAL_SUCCESSOR_REVIEW_PATH", path.resolve())
    review = verifier.preflight_successor_review(path.resolve(), digest)
    assert review["reviewer"] == "/root/different_agent"

    self_review = tmp_path / "self_review.json"
    self_digest = _review(self_review, reviewer="/root/g5_perf_closure")
    monkeypatch.setattr(
        verifier,
        "CANONICAL_SUCCESSOR_REVIEW_PATH",
        self_review.resolve(),
    )
    with pytest.raises(PermissionError, match="different-agent source review"):
        verifier.preflight_successor_review(self_review.resolve(), self_digest)


def test_successor_cannot_compute_or_publish_as_an_imported_library() -> None:
    assert not verifier.__name__.startswith("_lewm_v4_ca_")
    with pytest.raises(PermissionError, match="library computation is unsupported"):
        verifier._compute_exact_receipt(
            authorization={},
            seed=20260710,
            fit_size=5,
            reservation_path=Path("never-open"),
            reservation_file_sha256="0" * 64,
            result_path=Path("never-open"),
            result_file_sha256="0" * 64,
            checkpoint_path=Path("never-open"),
            checkpoint_file_sha256="0" * 64,
            completion_path=Path("never-open"),
            completion_file_sha256="0" * 64,
            trainer_authorization_path=Path("never-open"),
            trainer_authorization_file_sha256="0" * 64,
            trainer_review_path=Path("never-open"),
            trainer_review_file_sha256="0" * 64,
        )
    with pytest.raises(PermissionError, match="library publication is unsupported"):
        verifier._write_exclusive(
            verifier.canonical_metric_receipt_path(20260710, 5),
            {"seed": 20260710, "fit_size": 5, "content_sha256": "0" * 64},
        )


def test_immutable_n5_and_failure_lineage_hashes_match() -> None:
    attempt = verifier.CANONICAL_N5_ATTEMPT
    assert {path.name for path in attempt.iterdir()} == set(
        verifier.N5_ARTIFACT_BINDINGS
    )
    for name, binding in verifier.N5_ARTIFACT_BINDINGS.items():
        path = attempt / name
        assert path.is_file() and not path.is_symlink()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == binding["file_sha256"]
    for relative, expected in (
        (
            verifier.PREDECESSOR_VERIFIER_RELATIVE_PATH,
            verifier.PREDECESSOR_VERIFIER_FILE_SHA256,
        ),
        (verifier.FAILURE_RECORD_RELATIVE_PATH, verifier.FAILURE_RECORD_FILE_SHA256),
    ):
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert not (verifier.CANONICAL_DEVELOPMENT_ROOT / "metric_verifications").exists()


def test_cli_parser_accepts_only_frozen_seeds_and_rungs() -> None:
    base = [
        "--successor-review", "review.json", "--successor-review-sha256", "0" * 64,
        "--metric-authorization", "metric.json", "--metric-authorization-sha256", "0" * 64,
        "--trainer-authorization", "trainer.json", "--trainer-authorization-sha256", "0" * 64,
        "--trainer-review-record", "trainer-review.json", "--trainer-review-record-sha256", "0" * 64,
        "--reservation", "reservation.json", "--reservation-sha256", "0" * 64,
        "--result", "result.json", "--result-sha256", "0" * 64,
        "--checkpoint", "checkpoint.pt", "--checkpoint-sha256", "0" * 64,
        "--completion", "completed.json", "--completion-sha256", "0" * 64,
    ]
    for seed in (20260710, 20260711):
        for fit_size in (5, 16, 32, 320):
            parsed = verifier.parse_args(
                [*base, "--seed", str(seed), "--fit-size", str(fit_size)]
            )
            assert (parsed.seed, parsed.fit_size) == (seed, fit_size)
    with pytest.raises(SystemExit):
        verifier.parse_args([*base, "--seed", "20260709", "--fit-size", "5"])
    with pytest.raises(SystemExit):
        verifier.parse_args([*base, "--seed", "20260710", "--fit-size", "4"])
