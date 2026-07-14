from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from scripts import finalize_go2_observable_camera_ray_fit_v4_ladder_v2 as finalizer
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


def _write_review(path: Path) -> str:
    sources = {}
    for relative in verifier.SUCCESSOR_SOURCE_PATHS:
        sources[relative] = {
            "path": relative,
            "file_sha256": hashlib.sha256((ROOT / relative).read_bytes()).hexdigest(),
        }
    core = {
        "schema": verifier.SUCCESSOR_REVIEW_SCHEMA,
        "status": "different_agent_review_passed_frozen_ladder",
        "implementation_author": "/root/g5_perf_closure",
        "reviewer": "/root/different_agent",
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
    payload = _canonical(value) + b"\n"
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def test_finalizer_v2_reexecutes_verifier_v2_with_completion_binding() -> None:
    source = (ROOT / verifier.SUCCESSOR_FINALIZER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    imports = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module == "scripts"
        for alias in node.names
    ]
    assert "verify_go2_observable_camera_ray_fit_v4_metrics_v2" in imports
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "reverify_canonical_metric_receipt"
    ]
    assert len(calls) == 1
    keywords = {keyword.arg for keyword in calls[0].keywords}
    assert {"completion_path", "completion_file_sha256"}.issubset(keywords)
    assert verifier.SUCCESSOR_RELATIVE_PATH in source
    assert verifier.SUCCESSOR_FINALIZER_RELATIVE_PATH in source


def test_finalizer_successor_review_preflight_binds_full_execution_closure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "review.json"
    digest = _write_review(path)
    monkeypatch.setattr(finalizer, "CANONICAL_SUCCESSOR_REVIEW_PATH", path.resolve())
    reviewed_path, reviewed_sha, review = finalizer._preflight_successor_sources(
        f"{path.resolve()}:{digest}"
    )
    assert reviewed_path == path.resolve()
    assert reviewed_sha == digest
    assert set(review["successor_sources"]) == set(verifier.SUCCESSOR_SOURCE_PATHS)


def test_finalizer_v2_is_not_available_as_an_imported_library() -> None:
    with pytest.raises(PermissionError, match="library computation is unsupported"):
        finalizer._require_captured_private_finalizer()


def test_finalizer_cli_requires_review_and_preserves_all_modes() -> None:
    review = f"review.json:{'0' * 64}"
    common = ["--successor-review", review]
    stage = finalizer.parse_args(
        [
            *common,
            "stage",
            "--reservation", f"reservation.json:{'0' * 64}",
            "--result", f"result.json:{'0' * 64}",
            "--checkpoint", f"checkpoint.pt:{'0' * 64}",
            "--completion", f"completed.json:{'0' * 64}",
            "--metric-verification", f"metric.json:{'0' * 64}",
            "--trainer-authorization", f"trainer.json:{'0' * 64}",
            "--trainer-review-record", f"trainer-review.json:{'0' * 64}",
            "--seed", "20260711",
            "--fit-size", "320",
        ]
    )
    assert (stage.mode, stage.seed, stage.fit_size) == ("stage", 20260711, 320)
    seed = finalizer.parse_args(
        [*common, "seed", "--stage-gate", f"stage.json:{'0' * 64}", "--seed", "20260710"]
    )
    assert seed.mode == "seed"
    combined = finalizer.parse_args(
        [
            *common,
            "two-seed",
            "--seed-20260710-gate", f"first.json:{'0' * 64}",
            "--seed-20260711-gate", f"second.json:{'0' * 64}",
        ]
    )
    assert combined.mode == "two-seed"
    with pytest.raises(SystemExit):
        finalizer.parse_args(["stage"])


def test_predecessor_finalizer_and_verifier_bytes_remain_frozen() -> None:
    expected = {
        "scripts/verify_go2_observable_camera_ray_fit_v4_metrics.py": (
            "235f7a6e2cabeaa2ff68c09c82894f69c9bfd47af0bea687dbaec5b06f27f67f"
        ),
        "scripts/finalize_go2_observable_camera_ray_fit_v4_ladder.py": (
            "375b1dcd3a548cf7b130fb67291ef5116effcc0197a28be42643bfc59e710ec6"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == digest
