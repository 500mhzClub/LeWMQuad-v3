from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys


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
    assert contract.LABEL_ROOT_RELATIVE_PATH.endswith("labels_v2")
    assert contract.integrity_adapter_amendment_binding() == {
        "path": contract.INTEGRITY_ADAPTER_AMENDMENT_RELATIVE_PATH,
        "file_sha256": (
            "40e07c1daa388ed56a0473577af758d9085dfac26133cbbf83eaa849f9726d45"
        ),
        "byte_count": 3_645,
    }
    assert set(contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS) == {
        "reservation",
        "failure",
    }


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
