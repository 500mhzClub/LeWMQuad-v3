from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]
CONTRACT_PATH = (
    ROOT
    / "lewm/benchmarks/go2_post_action_projective_support_corridor_contract_v1.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("_projective_support_contract", CONTRACT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_import_is_source_only() -> None:
    program = f"""
import importlib.util, sys
spec=importlib.util.spec_from_file_location('_contract',{str(CONTRACT_PATH)!r})
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert 'numpy' not in sys.modules
print(module.EXPERIMENT_ID)
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "rgb_post_action_projective_support_corridor_joint_jepa_v1\n"


def test_counts_schedule_and_hashes_are_frozen() -> None:
    contract = _load()
    assert sum(row["states"] for row in contract.ROLE_COUNTS.values()) == 5_172
    assert sum(row["action_rows"] for row in contract.ROLE_COUNTS.values()) == 46_548
    assert 46_548 * contract.STATION_COUNT == 512_028
    assert contract.MAXIMUM_UPDATES * contract.EFFECTIVE_BATCH_SIZE == 16_000
    assert contract.ACTION_VOCABULARY[contract.HOLD_ACTION_INDEX] == "hold"
    assert contract.REMOTE_POSE_SHA256 == (
        "df96a4d23e9f2a297467c7384e54e9d7f8eac64609e937392f0db51e3c87abc3"
    )
    assert contract.LABEL_ROOT_RELATIVE_PATH.endswith("labels_v3")
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith("attempt_v1")
    assert contract.integrity_adapter_amendment_binding() == {
        "path": contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
        "file_sha256": (
            "40e07c1daa388ed56a0473577af758d9085dfac26133cbbf83eaa849f9726d45"
        ),
        "byte_count": 3_645,
    }
    assert contract.schedule_schema_adapter_amendment_binding() == {
        "path": contract.SCHEDULE_SCHEMA_ADAPTER_AMENDMENT_RELATIVE_PATH,
        "file_sha256": (
            "276f2dfc7cdb7355904858cdbd9f58fd5991051296414dc52e3f02a468516e1d"
        ),
        "byte_count": 4_445,
    }
    assert set(contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS) == {
        "reservation",
        "failure",
    }
    assert contract.LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS == {
        "reservation": {
            "path": (
                ".generated/go2_post_action_projective_support_labels_v2/"
                "reservation.json"
            ),
            "file_sha256": (
                "48eaec32a56bf0f872c0141ed359f2b673653c71bc76b5db96f4cf040b4bb165"
            ),
            "content_sha256": (
                "2cce455b5bf302cd4b43a263caf9b427b8b9512f1388b2eeb00dbf655939e803"
            ),
            "byte_count": 2_362,
        },
        "failure": {
            "path": (
                ".generated/go2_post_action_projective_support_labels_v2/"
                "failure.json"
            ),
            "file_sha256": (
                "4fd4e3ec067564a423e8dba41a75862df5b3c5051d4ae2a3ca8b015936a18ecd"
            ),
            "content_sha256": (
                "7b3cd79f76924ad12907303ca1d214bf260ace9d64c63bed5fa5814a71e74528"
            ),
            "byte_count": 2_417,
        },
    }
    science = contract.science_contract()
    assert science["schedule_schema_adapter_amendment"] == (
        contract.schedule_schema_adapter_amendment_binding()
    )
    assert science["label_v1_terminal_predecessor_bindings"] == (
        contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    )
    assert science["label_v2_terminal_predecessor_bindings"] == (
        contract.LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS
    )
    assert science["label_preflight_attempt"] == (
        "v3_science_identical_schedule_schema_adapter"
    )


def test_v2_terminal_predecessor_validator_is_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _load()
    reservation = contract.with_content_sha256(
        {"status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT"}
    )
    ledger = {
        name: 1
        for name in (
            "raw_manifest_opens",
            "raw_pairs_opens",
            "raw_endpoints_opens",
            "raw_audit_opens",
            "schedule_opens",
        )
    }
    ledger.update(
        {
            name: 0
            for name in (
                "geometry_contract_opens",
                "geometry_contract_validation_calls",
                "directional_policy_opens",
                "primitive_registry_opens",
                "scene_join_calls_started",
                "render_summary_opens",
                "source_frames_jsonl_opens",
                "scene_manifest_opens",
                "rgb_opens",
                "checkpoint_opens",
                "runtime_output_opens",
                "g2_opens",
                "navigation_opens",
                "heldout_opens",
                "sealed_opens",
                "production_opens",
            )
        }
    )
    failure = contract.with_content_sha256(
        {
            "access_ledger": ledger,
            "error": {"message": "frozen presentation schedule identity changed"},
            "phase": "prepare_execution_binding",
            "reservation_content_sha256": reservation["content_sha256"],
            "status": "FAILED_TERMINAL_NO_RETRY",
        }
    )
    bindings = {}
    for name, value in (("reservation", reservation), ("failure", failure)):
        relative = Path("receipts") / f"{name}.json"
        raw = contract.canonical_json_bytes(value) + b"\n"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        bindings[name] = {
            "path": str(relative),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": value["content_sha256"],
            "byte_count": len(raw),
        }
    monkeypatch.setattr(contract, "LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS", bindings)
    assert contract.validate_label_v2_terminal_predecessor(root=tmp_path) == {
        "reservation": reservation,
        "failure": failure,
    }

    bad_ledger = dict(ledger)
    bad_ledger["rgb_opens"] = 1
    bad_failure = contract.with_content_sha256(
        {
            "access_ledger": bad_ledger,
            "error": {"message": "frozen presentation schedule identity changed"},
            "phase": "prepare_execution_binding",
            "reservation_content_sha256": reservation["content_sha256"],
            "status": "FAILED_TERMINAL_NO_RETRY",
        }
    )
    raw = contract.canonical_json_bytes(bad_failure) + b"\n"
    path = tmp_path / bindings["failure"]["path"]
    path.write_bytes(raw)
    bindings["failure"].update(
        file_sha256=hashlib.sha256(raw).hexdigest(),
        content_sha256=bad_failure["content_sha256"],
        byte_count=len(raw),
    )
    with pytest.raises(PermissionError, match="failure semantics changed"):
        contract.validate_label_v2_terminal_predecessor(root=tmp_path)


def test_canonical_receipt_rejects_duplicate_or_noncanonical_json() -> None:
    contract = _load()
    receipt = contract.with_content_sha256({"schema": "synthetic", "value": 1})
    raw = contract.canonical_json_bytes(receipt) + b"\n"
    assert contract.parse_canonical_json(raw, name="synthetic") == receipt
    duplicate = b'{"content_sha256":"0","schema":"a","schema":"b"}\n'
    try:
        contract.parse_canonical_json(duplicate, name="duplicate")
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate JSON keys were accepted")
    try:
        contract.parse_canonical_json(raw[:-1], name="unterminated")
    except ValueError:
        pass
    else:
        raise AssertionError("noncanonical JSON was accepted")
