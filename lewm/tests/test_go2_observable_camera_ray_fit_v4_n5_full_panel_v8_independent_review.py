"""Independent CPU-only review checks for the frozen N5 full-panel V8 closure."""
from __future__ import annotations

import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_full_panel_v8 as policy,
)


ROOT = Path(__file__).resolve().parents[2]
EXECUTOR = ROOT / policy.EXECUTOR_RELATIVE_PATH
FROZEN_V8 = {
    policy.POLICY_RELATIVE_PATH: (
        "99a2777d3ba2ad8baf62b98944f05aa1affb2e74834f337a2ba0644e9c03c84c"
    ),
    policy.EXECUTOR_RELATIVE_PATH: (
        "f163aaf04722bb118796912bcfcdf1f4e24b7e54990e41a9d164acc08b233500"
    ),
    "lewm/tests/n5_full_panel_v8_synthetic_execution.py": (
        "4d11b499d4cc2ffe4a31d0ed5df73a84649947bfd8a78522556719f8af21316c"
    ),
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_full_panel_v8.py": (
        "700092f5ea2885e23dba03b65c5a24737060c20e934413af1886ff454ec3e5b4"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_full_panel_v8_"
    "implementation_handoff_2026-07-13.md": (
        "536f31de0d8fe0cec26417b73e29ff3ef396086b05d5b2f104e7202f98df25b1"
    ),
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _nested_functions(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return {
        node.name: ast.unparse(node)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef)
    }


def test_v8_frozen_candidate_and_v7_terminal_inventory_rehash() -> None:
    assert {
        relative: _sha256(ROOT / relative) for relative in FROZEN_V8
    } == FROZEN_V8
    static = policy.preflight_static_authority()
    terminal = static["v7_terminal"]
    assert terminal["source_review"]["content_sha256"] == (
        policy.V7_REVIEW_RECORD_CONTENT_SHA256
    )
    assert terminal["reservation"]["content_sha256"] == (
        policy.V7_RESERVATION_CONTENT_SHA256
    )
    assert terminal["failure"]["content_sha256"] == (
        policy.V7_FAILURE_CONTENT_SHA256
    )
    assert terminal["failure"]["failure_stage"] == "verification"
    assert terminal["failure"]["failure"] == {
        "class": "runtime",
        "code": "execution_failure",
    }
    assert terminal["journal_integrity"] == "intact"
    assert terminal["metric_receipt_published"] is False
    assert terminal["gate_published"] is False
    assert terminal["numeric_payload_survived"] is False
    assert terminal["numeric_payload_inspected"] is False
    assert terminal["retry_authorized"] is False
    attempt = (ROOT / policy.V7_RESERVATION_RELATIVE_PATH).parent
    assert sorted(path.name for path in attempt.iterdir()) == [
        "failed.json",
        "reservation.json",
    ]
    output = ROOT / policy.V7_OUTPUT_ROOT_RELATIVE_PATH
    assert list((output / "metric_verifications").iterdir()) == []
    assert list((output / "gates").iterdir()) == []
    assert not policy.CANONICAL_OUTPUT_ROOT.exists()


def test_v8_fresh_process_allows_one_frozen_interop_setter_only() -> None:
    code = f"""
import io
import json
import sys
sys.path.insert(0, {str(ROOT)!r})
import torch
from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base
buffer = io.BytesIO()
torch.save({{'tensor': torch.ones((64, 64))}}, buffer)
loaded = torch.load(io.BytesIO(buffer.getvalue()), map_location='cpu', weights_only=False)
loaded['tensor'].detach().to(device='cpu').contiguous().numpy().tobytes(order='C')
first = base.configure_determinism(20260710)
try:
    base.configure_determinism(20260710)
except Exception as error:
    second = {{'type': type(error).__name__, 'message': str(error)}}
else:
    second = None
print(json.dumps({{'first': first, 'second': second}}, sort_keys=True))
"""
    environment = dict(os.environ)
    for name in (
        "HIP_VISIBLE_DEVICES",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
    ):
        environment[name] = ""
    environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
    for name in policy.THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8", "replace")
    assert completed.stderr == b""
    result = json.loads(completed.stdout)
    assert result["first"]["torch_num_interop_threads"] == 1
    assert result["second"]["type"] == "RuntimeError"
    assert "cannot set number of interop threads" in result["second"]["message"]


def test_v8_child_is_compute_only_and_parent_publishes_after_validation() -> None:
    functions = _nested_functions(EXECUTOR)
    child = functions["_compute_verification_receipt_child"]
    parent = functions["_run_independent_verification"]
    assert child.count("verifier._validate_attempt_bundle(token, args)") == 1
    assert child.count("verifier._compute_receipt(token, bundle)") == 1
    assert "compatibility.write_exclusive = forbid_publication" in child
    assert "_write_canonical_json" not in child
    assert "_compute_verification_receipt_child" not in parent
    assert parent.index("_validate_verification_response") < parent.index(
        "_write_canonical_json"
    )
    assert "no fallback" in parent


def test_v8_exact_verifier_command_and_selector_sanitization_are_literal() -> None:
    functions = _nested_functions(EXECUTOR)
    parent = functions["_run_independent_verification"]
    child_environment = functions["_verification_child_environment"]
    outer = functions["_isolated_child"]
    assert (
        "[sys.executable, '-I', '-B', str(Path(__file__).resolve()), "
        "'--verification-child']" in parent
    )
    assert "close_fds=True" in parent
    assert "timeout=policy.VERIFICATION_TIMEOUT_SECONDS" in parent
    assert "stdout=subprocess.PIPE" in parent
    assert "stderr=subprocess.PIPE" in parent
    for name in (
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
    ):
        assert repr(name) in child_environment
        assert repr(name) in outer
    assert "environment['HIP_VISIBLE_DEVICES'] = '0'" in child_environment
    assert "environment['HIP_VISIBLE_DEVICES'] = '0'" in outer
