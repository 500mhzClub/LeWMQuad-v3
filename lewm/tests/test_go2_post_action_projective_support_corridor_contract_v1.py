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
    assert contract.LABEL_ROOT_RELATIVE_PATH.endswith("labels_v4")
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
    assert contract.source_episode_id_adapter_amendment_binding() == {
        "path": contract.SOURCE_EPISODE_ID_ADAPTER_AMENDMENT_RELATIVE_PATH,
        "file_sha256": (
            "5b848d5c5163c4f12b7d5071264a545f491bdbbea47c7e1116464813e4c37509"
        ),
        "byte_count": 5_501,
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
    assert contract.LABEL_V3_TERMINAL_PREDECESSOR_BINDINGS == {
        "reservation": {
            "path": (
                ".generated/go2_post_action_projective_support_labels_v3/"
                "reservation.json"
            ),
            "file_sha256": (
                "387c7dc37fa3f34fc048e3bab64a82196811689ddd5fbd8648ad017f182bb28e"
            ),
            "content_sha256": (
                "22fa973b1ac0afb6b8f1ef8a0d3fe7f2da75e275fbd338f0e91385b592ed4627"
            ),
            "byte_count": 2_362,
        },
        "builder_claim": {
            "path": (
                ".generated/go2_post_action_projective_support_labels_v3/"
                "builder_claim.json"
            ),
            "file_sha256": (
                "f451a9105cb3cf9baf8035fda7d04530d6044ecc0ae8a898adf4de447732fea9"
            ),
            "content_sha256": (
                "b96c94c1aebbe862f04361414338e6fa38a58fe17db71b1c5cb64e16ef680e92"
            ),
            "byte_count": 504,
        },
        "builder_failure": {
            "path": (
                ".generated/go2_post_action_projective_support_labels_v3/"
                "failure.json"
            ),
            "file_sha256": (
                "998a5bca429ba2db13dc2996aadd57ff64d3cedef3f3c00420786040f3aa73d8"
            ),
            "content_sha256": (
                "86a57a2ec562e9395b967778fa9133e11e3b1711acae4846b855130745a6271e"
            ),
            "byte_count": 2_551,
        },
        "preflight_failure": {
            "path": (
                ".generated/"
                "go2_post_action_projective_support_labels_v3_preflight_failure.json"
            ),
            "file_sha256": (
                "6eb23a50388a4a10f755dee494848cbfb7750045e84beb900f091adbc26465d7"
            ),
            "content_sha256": (
                "ad0536d7aba6544c797913b7e993a3e900c2ae443b9da6f7ba2771bfff21164e"
            ),
            "byte_count": 2_585,
        },
        "builder_execution_binding": {
            "path": (
                "docs/lewm_go2_post_action_projective_support_labels_v3_"
                "execution_binding_2026-07-28.json"
            ),
            "file_sha256": (
                "ada9f377db4f3adf6fe6e796bc5f8410f01a69c4a6ecb271ee353435fe2944d7"
            ),
            "content_sha256": (
                "12a5c9ccc2c001f9116e8bfafb31c4029e62cd91fc999e580eda16124a6534bb"
            ),
            "byte_count": 111_848,
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
    assert science["source_episode_id_adapter_amendment"] == (
        contract.source_episode_id_adapter_amendment_binding()
    )
    assert science["label_v3_terminal_predecessor_bindings"] == (
        contract.LABEL_V3_TERMINAL_PREDECESSOR_BINDINGS
    )
    assert science["label_preflight_attempt"] == (
        "v4_science_identical_source_episode_id_adapter"
    )


def test_v3_terminal_predecessor_validator_accepts_bound_receipt_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = _load()
    execution_binding = contract.with_content_sha256({"schema": "synthetic_binding"})
    execution_raw = contract.canonical_json_bytes(execution_binding) + b"\n"
    execution_relative = Path("receipts/builder_execution_binding.json")
    execution_record = {
        "path": str(execution_relative),
        "file_sha256": hashlib.sha256(execution_raw).hexdigest(),
        "content_sha256": execution_binding["content_sha256"],
        "byte_count": len(execution_raw),
    }
    reservation = contract.with_content_sha256(
        {"status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT"}
    )
    claim = contract.with_content_sha256(
        {
            "execution_binding_content_sha256": execution_binding["content_sha256"],
            "reservation_content_sha256": reservation["content_sha256"],
            "resume_authorized": False,
            "retry_authorized": False,
            "second_invocation_authorized": False,
            "status": "CLAIMED_ONE_EXACT_LABEL_BUILDER_INVOCATION",
        }
    )
    protected = {
        key: 0
        for key in (
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
    builder_failure = contract.with_content_sha256(
        {
            "access_ledger": protected,
            "error": {"message": "source episode_id must be a nonempty string"},
            "execution_binding_content_sha256": execution_binding["content_sha256"],
            "phase": "materialize_and_publish_manifest_last",
            "reservation_content_sha256": reservation["content_sha256"],
            "schedule_prefix_sha256": contract.SCHEDULE_PREFIX_SHA256,
            "status": "FAILED_TERMINAL_NO_RETRY",
        }
    )
    preflight_failure = contract.with_content_sha256(
        {
            "error_message": "source episode_id must be a nonempty string",
            "label_builder_execution_binding": {
                key: execution_record[key]
                for key in ("path", "file_sha256", "byte_count")
            },
            "phase": "materialize_label_bundle",
            "protected_access_counts": protected,
            "resume": False,
            "retry": False,
            "status": "TERMINAL_LABEL_PREFLIGHT_STOP",
            "training_authorized": False,
        }
    )
    values = {
        "reservation": reservation,
        "builder_claim": claim,
        "builder_failure": builder_failure,
        "preflight_failure": preflight_failure,
        "builder_execution_binding": execution_binding,
    }
    relative_paths = {
        name: Path("receipts") / f"{name}.json" for name in values
    }
    relative_paths["builder_execution_binding"] = execution_relative
    bindings = {}
    for name, value in values.items():
        raw = contract.canonical_json_bytes(value) + b"\n"
        relative = relative_paths[name]
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        bindings[name] = {
            "path": str(relative),
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "content_sha256": value["content_sha256"],
            "byte_count": len(raw),
        }
    monkeypatch.setattr(contract, "LABEL_V3_TERMINAL_PREDECESSOR_BINDINGS", bindings)
    assert contract.validate_label_v3_terminal_predecessor(root=tmp_path) == values


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
