from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def _run(code: str) -> subprocess.CompletedProcess[str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": f"{ROOT}:{ROOT / 'lewm_worlds'}",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def test_sealed_bootstrap_executes_frozen_runner_and_one_worker_child() -> None:
    completed = _run(
        "from scripts import audit_go2_g3_exact_physical_equivalence as r; "
        "import json; print(json.dumps(r.sealed_bootstrap_probe(),sort_keys=True))"
    )
    assert completed.returncode == 0, completed.stderr
    probe = json.loads(completed.stdout)
    assert probe["captured_runner_executed"] is True
    assert probe["one_worker_crossed_process_boundary"] is True
    assert probe["runner_module_name"] == probe["evaluate_job_module_name"]
    assert probe["runner_module_name"] == probe["one_worker_module_name"]
    assert probe["runner_module_name"] == probe["one_worker_evaluate_job_module_name"]
    assert len(probe["captured_module_sha256s"]) == 5
    assert all(len(value) == 64 for value in probe["captured_module_sha256s"].values())
    binding = probe["synthetic_job_binding"]
    job_core = {
        key: binding[key]
        for key in (
            "index",
            "scene_id",
            "family",
            "manifest_sha256",
            "runner_source_sha256",
            "source_graph_sha256",
        )
    }
    encoded = json.dumps(
        job_core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    assert binding["job_sha256"] == hashlib.sha256(encoded).hexdigest()


def test_imported_evaluate_job_monkeypatch_cannot_forge_sealed_execution() -> None:
    completed = _run(
        "from scripts import audit_go2_g3_exact_physical_equivalence as r; "
        "r._evaluate_job=lambda job:(0,{'forged':True},{}); "
        "p=r.sealed_bootstrap_probe(); "
        "print(p['captured_runner_executed'],p['one_worker_crossed_process_boundary'],"
        "p['runner_module_name'])"
    )
    assert completed.returncode == 0, completed.stderr
    captured, crossed, module_name = completed.stdout.strip().split()
    assert captured == "True"
    assert crossed == "True"
    assert module_name != "scripts.audit_go2_g3_exact_physical_equivalence"


def test_runner_source_hash_and_canonical_launcher_location_are_frozen() -> None:
    from scripts import audit_go2_g3_exact_physical_equivalence as launcher

    runner = ROOT / launcher.RUNNER_RELATIVE_PATH
    assert hashlib.sha256(runner.read_bytes()).hexdigest() == (
        launcher.EXPECTED_RUNNER_SOURCE_SHA256
    )
    source = Path(launcher.__file__).resolve()
    assert source == ROOT / "scripts/audit_go2_g3_exact_physical_equivalence.py"
