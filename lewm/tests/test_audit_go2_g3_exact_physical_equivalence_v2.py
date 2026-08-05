from __future__ import annotations

import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import threading

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/audit_go2_g3_exact_physical_equivalence_v2.py"
RUNNER = ROOT / "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v2.py"
V2_OUTPUT = ROOT / ".generated/go2_g3_exact_physical_equivalence/v2/candidate.json"
V1_OUTPUT = ROOT / ".generated/go2_g3_exact_physical_equivalence/v1/candidate.json"
RUNNER_SHA256 = "d759cb7fa395646d435bdd0af220a098d7d1e908970a30c4f17fc9e391c296e8"
PROFILE_SHA256 = "2b00cbe295ef4d0ef9f66e42b1aa7188751045240cba923392d83fd1bc709314"
DESIGN_SHA256 = "a82de141575efe9e12f0deea05477f558439d87bcb1af3bc36e0d377a36c95b1"


def _environment() -> dict[str, str]:
    return {
        **dict(os.environ),
        "PYTHONPATH": f"{ROOT}:{ROOT / 'lewm_worlds'}",
        "HIP_VISIBLE_DEVICES": "",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def test_v2_cli_probe_executes_frozen_runner_and_worker_subprocess_only() -> None:
    assert not V2_OUTPUT.exists()
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--probe", "--workers", "1"],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout)
    assert probe["captured_runner_executed"] is True
    assert probe["one_worker_crossed_process_boundary"] is True
    assert probe["runner_module_name"] == probe["evaluate_job_module_name"]
    assert probe["runner_module_name"] == probe["one_worker_module_name"]
    assert probe["runner_module_name"] == probe["one_worker_evaluate_job_module_name"]
    assert len(probe["captured_module_sha256s"]) == 8
    assert probe["runner_source_sha256"] == RUNNER_SHA256
    assert probe["profile_sha256"] == PROFILE_SHA256
    assert probe["governing_design_sha256"] == DESIGN_SHA256
    assert probe["production_promotion_authorized"] is False
    assert not V2_OUTPUT.exists()


def test_v2_launcher_import_exposes_no_authority_or_substitution_surface() -> None:
    namespace = runpy.run_path(str(LAUNCHER), run_name="review_import_only")
    forbidden = {
        "run",
        "main",
        "sealed_bootstrap_probe",
        "_invoke_sealed_child",
        "_read_fixed_runner_source",
        "EXPECTED_RUNNER_SOURCE_SHA256",
        "RUNNER_RELATIVE_PATH",
        "DEFAULT_OUTPUT",
        "_CHILD_BOOTSTRAP",
        "subprocess",
        "Path",
    }
    assert forbidden.isdisjoint(namespace)
    tree = ast.parse(LAUNCHER.read_text(encoding="utf-8"))
    assert not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)) for node in ast.walk(tree))


def test_v2_cli_rejects_caller_output_and_noncanonical_copied_launcher(
    tmp_path: Path,
) -> None:
    caller_output = tmp_path / "caller.json"
    completed = subprocess.run(
        [sys.executable, str(LAUNCHER), "--output", str(caller_output)],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 2
    assert "unrecognized arguments" in completed.stderr
    assert not caller_output.exists()

    copied = tmp_path / LAUNCHER.name
    copied.write_bytes(LAUNCHER.read_bytes())
    copied_run = subprocess.run(
        [sys.executable, str(copied), "--probe", "--workers", "1"],
        cwd=ROOT,
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert copied_run.returncode != 0
    assert "canonical fixed path" in copied_run.stderr
    assert not V2_OUTPUT.exists()


def test_v2_runner_design_binding_and_v1_sources_outputs_are_unchanged() -> None:
    assert V2_OUTPUT != V1_OUTPUT
    assert not V2_OUTPUT.exists()
    assert hashlib.sha256(RUNNER.read_bytes()).hexdigest() == RUNNER_SHA256
    runner_text = RUNNER.read_text(encoding="utf-8")
    assert "docs/lewm_go2_g3_two_resolution_v2_design_contract_2026-07-13.md" in runner_text
    assert DESIGN_SHA256 in runner_text
    assert PROFILE_SHA256 in runner_text
    assert hashlib.sha256(
        (ROOT / "lewm/benchmarks/go2_g3_exact_physical_equivalence.py").read_bytes()
    ).hexdigest() == "b0155968a267afb08817987c3779e61e2e59b32e60281b1116a3757ac4fa461d"
    assert hashlib.sha256(
        (
            ROOT
            / "lewm/benchmarks/go2_g3_exact_physical_equivalence_runner_v1.py"
        ).read_bytes()
    ).hexdigest() == "4fbceaa49519d811de3f1508c99099c8b1ddda8cb7dacefcd8aa153a05f4a3b3"
    assert hashlib.sha256(
        (ROOT / "scripts/audit_go2_g3_exact_physical_equivalence.py").read_bytes()
    ).hexdigest() == "c22091aed4a554d87f912d4aa98c92ef3c529e61a39f8b2b06e568e36a56af3b"
    assert hashlib.sha256(V1_OUTPUT.read_bytes()).hexdigest() == (
        "b7176cca80306768c6c851c61c2ba31636093b15bae777b1966cb2d56edc3d4c"
    )


def test_atomic_publication_never_replaces_existing_or_concurrent_candidate(
    tmp_path: Path,
) -> None:
    from lewm.benchmarks import go2_g3_exact_physical_equivalence_runner_v2 as runner

    existing = tmp_path / "existing.json"
    existing.write_bytes(b"sentinel\n")
    with pytest.raises(FileExistsError, match="concurrently"):
        runner._write_atomic_no_replace(existing, {"writer": "replacement"})
    assert existing.read_bytes() == b"sentinel\n"

    destination = tmp_path / "candidate.json"
    barrier = threading.Barrier(2)

    def publish(writer: int) -> tuple[str, int]:
        barrier.wait()
        try:
            runner._write_atomic_no_replace(destination, {"writer": writer})
        except FileExistsError:
            return "lost", writer
        return "won", writer

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))
    assert sorted(status for status, _writer in outcomes) == ["lost", "won"]
    winning_writer = next(writer for status, writer in outcomes if status == "won")
    assert destination.read_bytes() == (
        json.dumps(
            {"writer": winning_writer},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        + b"\n"
    )
    assert not list(tmp_path.glob(".g3-v2-*.tmp"))


def test_publication_rejects_noncanonical_audit_path_and_symlink_parent(
    tmp_path: Path,
) -> None:
    from lewm.benchmarks import go2_g3_exact_physical_equivalence_runner_v2 as runner

    with pytest.raises(PermissionError, match="not canonical"):
        runner._assert_canonical_output_path(tmp_path / "candidate.json")
    with pytest.raises(PermissionError, match="not canonical"):
        runner._sealed_run(
            output=tmp_path / "candidate.json",
            workers=1,
            expected_runner_source_sha256=hashlib.sha256(RUNNER.read_bytes()).hexdigest(),
        )
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    linked_parent = tmp_path / "linked"
    linked_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(PermissionError, match="real directories"):
        runner._write_atomic_no_replace(
            linked_parent / "candidate.json",
            {"writer": "symlink"},
        )
    assert not (real_parent / "candidate.json").exists()
    assert not V2_OUTPUT.exists()
