#!/usr/bin/env python3
"""Externally supervise one exact existing-pool three-arm experiment.

This source grants no authority.  It accepts only a separately committed,
caller-bound authority, verifies its plan/review/source/runtime/input closure,
exclusively reserves the one fixed development attempt, and then launches the
bound worker once under a hard subprocess wall ceiling.  A receipt-only JSON
checker runs under the same ceiling.  Reservation consumes the attempt even
when the worker or checker fails; retry, resume, refill, and overwrite remain
false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import signal
import shutil
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_ROOT = REPO_ROOT / ".generated" / "dev"
ATTEMPT_ROOT = (
    DEVELOPMENT_ROOT
    / "world_model_existing_pool_three_arm_v1_integrity_replacement_v3"
    / "attempt_v1"
)
ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v3/attempt_v1"
)
PREDECESSOR_ATTEMPT_ROOT = (
    DEVELOPMENT_ROOT
    / "world_model_existing_pool_three_arm_v1_integrity_replacement_v2"
    / "attempt_v1"
)
PREDECESSOR_ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v2/attempt_v1"
)
WORKER_RELATIVE = Path(
    "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py"
)
CHECKER_RELATIVE = Path(
    "scripts/check_go2_world_model_existing_pool_three_arm_v1.py"
)
SUPERVISOR_RELATIVE = Path(
    "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
)
PREREGISTRATION_RELATIVE = Path(
    "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v3_preregistration_2026-08-01.md"
)
PLAN_RELATIVE = Path(
    "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v3_plan_2026-08-01.json"
)
REVIEW_RELATIVE = Path(
    "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v3_independent_source_review_2026-08-01.json"
)
AUTHORITY_RELATIVE = Path(
    "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
    "replacement_v3_execution_authority_2026-08-01.json"
)

AUTHORITY_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_EXACT_EXISTING_POOL_THREE_ARM_V1_INTEGRITY_REPLACEMENT_V3_"
    "ATTEMPT"
)
PLAN_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "plan_v1"
)
PLAN_PURPOSE = (
    "existing_pool_three_arm_v1_factual_learning_integrity_replacement_v3"
)
REVIEW_SCHEMA = (
    "lewm_go2_world_model_follow_on_v3_integrity_replacement_independent_"
    "source_review_v1"
)
REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"
EXPECTED_REVIEW_SCOPE = {
    "source_only": True,
    "scientific_plan_normalized_against_integrity_replacement_v2": True,
    "normalized_scientific_difference_count": 0,
    "production_source_paths_changed_from_integrity_replacement_v2": [
        "worker",
        "checker",
        "external_supervisor",
    ],
    "retained_checkpoint_loader_correction": (
        "accept exact strided scalar-long ema_update_count=1000 while retaining "
        "finite strided float32 for every other state entry"
    ),
    "arm_trainability_change": (
        "restore requires_grad=true after all independent ArmCore copies without "
        "changing parameter values or inventory"
    ),
    "parity_probe_mode_alignment": (
        "run the unchanged payload-free exact-zero probe with substrate and copied "
        "arms recursively in evaluation mode, then restore every arm recursively "
        "to training mode before return"
    ),
    "lifecycle_change": (
        "fresh V3 namespace, exact direct-predecessor evidence, exact plan and "
        "checker validation, and source-review-authority commit ordering"
    ),
}
EXPECTED_REVIEW_VERIFICATION = {
    "focused_test_count": 74,
    "focused_tests_passed": 74,
    "strict_json_passed": True,
    "python_compilation_passed": True,
    "git_diff_check_passed": True,
    "required_source_path_count": 32,
    "complete_spatial_state_tensor_count": 187,
    "migrated_state_tensor_count": 108,
    "rejected_state_tensor_count": 79,
    "arm_parameter_tensor_count": 36,
    "predictor_parameter_tensor_count": 30,
    "memory_parameter_tensor_count": 6,
    "normalized_plan_diff_empty": True,
    "predecessor_identity_evidence_live_verified": True,
    "checker_v3_attempt_envelope_exact": True,
    "exact_runtime_synthetic_parity_probe_passed": True,
    "parity_probe_state_optimizer_and_rng_invariance_passed": True,
    "source_review_authority_commit_order_enforced": True,
    "real_idle_device_preflight_passed": True,
    "replacement_root_absent_at_review": True,
}
EXPECTED_REVIEW_CUSTODY = {
    "pack_payloads_opened": False,
    "real_predecessor_or_runtime_checkpoint_or_snapshot_payloads_opened": False,
    "synthetic_checkpoint_fixtures_used": True,
    "predecessor_terminal_json_identity_evidence_opened": True,
    "rgb_payloads_opened": False,
    "heldout_or_sealed_opened": False,
    "network_access_used": False,
    "protected_runtime_payloads_opened": False,
    "execution_authority_granted": False,
    "attempt_reserved": False,
}
PREDECESSOR_FAILURE_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v2_"
    "terminal_pretraining_source_failure_result_v1"
)
PREDECESSOR_FAILURE_STATUS = (
    "PASS_COMPLETE_TERMINAL_PRETRAINING_SOURCE_FAILURE_AUDIT"
)
PREDECESSOR_FAILURE_BINDING = {
    "path": str(
        (
            REPO_ROOT
            / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
            "replacement_v2_terminal_pretraining_source_failure_result_"
            "2026-08-01.json"
        ).resolve()
    ),
    "file_sha256": (
        "8cb652fb8e88d725b187f45d3b2988de440b9feabe253d372de90fb6134a1902"
    ),
    "byte_count": 8487,
}
PREDECESSOR_SOURCE_COMMIT = "0468ac755851b6ed86206ab3825a04a03bd22567"
PREDECESSOR_REVIEW_COMMIT = "23b2112a70a72cc7f301f288e519d4e47b9b8d92"
PREDECESSOR_AUTHORITY_COMMIT = "0a4522430ddccde7d55dd1fb46dcc3483c501833"
PREDECESSOR_ORIGINAL_FAILURE_BINDING = {
    "path": str(
        (
            REPO_ROOT
            / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_integrity_"
            "replacement_v1_terminal_pretraining_source_failure_result_"
            "2026-08-01.json"
        ).resolve()
    ),
    "file_sha256": (
        "a96f63aeb119163cd24e17272bfbf5228206c498d706578162c310841423ac1b"
    ),
    "byte_count": 7008,
}
PREDECESSOR_TERMINAL_ARTIFACTS = {
    "reservation": {
        "path": str((PREDECESSOR_ATTEMPT_ROOT / "reservation.json").resolve()),
        "file_sha256": (
            "0f36a2e96dd7943a218c61a918e21eca58072ee29e6eb3e15b880b26031e7adc"
        ),
        "byte_count": 17048,
    },
    "failure": {
        "path": str((PREDECESSOR_ATTEMPT_ROOT / "failure.json").resolve()),
        "file_sha256": (
            "6d9009febd7c307d0c0c9f453b8ffdd5d272ef9efee7eb5ae1b502a41ad0a603"
        ),
        "byte_count": 1094,
    },
    "terminal_supervision": {
        "path": str(
            (PREDECESSOR_ATTEMPT_ROOT / "terminal_supervision.json").resolve()
        ),
        "file_sha256": (
            "598d1451289f41d924cd8e9242bdc3ba8550cc092ebec6c96565af4a86144a60"
        ),
        "byte_count": 2797,
    },
}
PREDECESSOR_AUTHORIZED_IDENTITY = {
    "authority_commit": PREDECESSOR_AUTHORITY_COMMIT,
    "source_commit": PREDECESSOR_SOURCE_COMMIT,
    "review_commit": PREDECESSOR_REVIEW_COMMIT,
    "authority_binding": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
                "integrity_replacement_v2_execution_authority_2026-08-01.json"
            ).resolve()
        ),
        "file_sha256": (
            "40c0a95885ec9f7ee67abbf2ebe2672f98d7874c03dd68e3a916fe8eaf101204"
        ),
        "byte_count": 16075,
    },
    "plan_binding": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
                "integrity_replacement_v2_plan_2026-08-01.json"
            ).resolve()
        ),
        "file_sha256": (
            "4c645887cb2cb96f8acff3ccc5b0c123ae9e8549b4866a40d3036c403fec9c31"
        ),
        "byte_count": 6259,
    },
    "independent_review_binding": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
                "integrity_replacement_v2_independent_source_review_"
                "2026-08-01.json"
            ).resolve()
        ),
        "file_sha256": (
            "cdd740ab0cd04a293b263a8ecd4d0bc1a71b2c3137b7390ffefac127f661fd75"
        ),
        "byte_count": 13999,
    },
    "preregistration_binding": {
        "path": str(
            (
                REPO_ROOT
                / "docs/lewm_go2_world_model_existing_pool_three_arm_v1_"
                "integrity_replacement_v2_preregistration_2026-08-01.md"
            ).resolve()
        ),
        "file_sha256": (
            "ac8e734302bd0f025e7ee91fcd425549390e7c9064e0a87446d4ade493368216"
        ),
        "byte_count": 6852,
    },
}
RESERVATION_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "reservation_v1"
)
RESULT_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "result_v1"
)
RESULT_STATUS = "COMPLETE_PENDING_TERMINAL_REVIEW"
CHECK_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "receipt_check_v1"
)
TERMINAL_SCHEMA = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v3_"
    "supervision_terminal_v1"
)
ARM_ORDER = ["conditioned", "blind", "shuffled"]
WORKER_OUTPUT_PATHS = frozenset(
    {
        "pack/manifest.json",
        "pack/train_frames.u8",
        "pack/train_actions.npy",
        "pack/train_meta.json",
        "pack/val_frames.u8",
        "pack/val_actions.npy",
        "pack/val_meta.json",
        "overlap_audit.json",
        "shuffle_audit.json",
    }
    | {
        f"arms/{arm}/measurements/update_{update:06d}.json"
        for arm in ARM_ORDER
        for update in range(0, 701, 100)
    }
    | {
        f"arms/{arm}/snapshots/update_{update:06d}.pt"
        for arm in ARM_ORDER
        for update in range(0, 701, 100)
    }
)
EXACT_CHILD_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "HIP_VISIBLE_DEVICES": "0",
    "ROCR_VISIBLE_DEVICES": "0",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONHASHSEED": "0",
    "PYTHONSAFEPATH": "1",
    "OMP_NUM_THREADS": "1",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_OPTIONAL_LOCKS": "0",
}
EXPECTED_RUNTIME = {
    "python_invocation_path": str(
        REPO_ROOT / ".generated/venvs/world_model_rocm_7_2_1_v1/bin/python"
    ),
    "environment": dict(EXACT_CHILD_ENVIRONMENT),
    "bindings": {
        "python_executable_target": {
            "path": "/usr/bin/python3.12",
            "file_sha256": (
                "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
            ),
            "byte_count": 8020928,
        },
        "python_environment_config": {
            "path": str(
                (
                    REPO_ROOT
                    / ".generated/venvs/world_model_rocm_7_2_1_v1/pyvenv.cfg"
                ).resolve()
            ),
            "file_sha256": (
                "49222cc65a628e83d00d99da60f1dea8d59bc01a3ea9616227f330e2ecd50577"
            ),
            "byte_count": 223,
        },
        "git_executable": {
            "path": "/usr/bin/git",
            "file_sha256": (
                "2a8c18fbf43da9f692d75474c72bea9dfd796c260b0f3dfe456376abc3bbd668"
            ),
            "byte_count": 4066232,
        },
    },
}
EXPECTED_INPUT_BINDINGS = {
    "predecessor_checkpoint": {
        "path": str(
            (
                REPO_ROOT
                / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_"
                "jepa_v1/attempt_v1/snapshots/update_1000.pt"
            ).resolve()
        ),
        "file_sha256": (
            "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
        ),
        "byte_count": 52282877,
    },
    "train_index": {
        "path": str(
            (
                REPO_ROOT
                / ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_"
                "integrity/train.jsonl"
            ).resolve()
        ),
        "file_sha256": (
            "aee2a54cddd849162648f9b8cfd54a0a28a25bd0705b6482e6af7435c85f4d77"
        ),
        "byte_count": 10328000,
    },
    "validation_index": {
        "path": str(
            (
                REPO_ROOT
                / ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_"
                "integrity/val.jsonl"
            ).resolve()
        ),
        "file_sha256": (
            "83592e2fea5927802881f076a58a9710100bea017d658c1b978ba651369beac6"
        ),
        "byte_count": 1317888,
    },
    "index_manifest": {
        "path": str(
            (
                REPO_ROOT
                / ".generated/go2_recurrent_h4_rgb_sequence_index_v2_schedule_"
                "integrity/manifest.json"
            ).resolve()
        ),
        "file_sha256": (
            "d19fd672d9878e064b20e40a12ce84849f0a13af05a73d2281505ea8d331a36e"
        ),
        "byte_count": 26926,
    },
}
AMD_SMI_BINDING = {
    "path": "/opt/rocm-7.1.1/libexec/amdsmi_cli/amdsmi_cli.py",
    "file_sha256": (
        "4f231c2ed6b7e66a2829fa82265f53ffc33b2d6a0c746a86a5f240f0be09bcf8"
    ),
    "byte_count": 9693,
}
AMD_SMI_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "LANG": "C.UTF-8",
    "LC_ALL": "C.UTF-8",
    "PYTHONNOUSERSITE": "1",
    "PYTHONDONTWRITEBYTECODE": "1",
}
GIT_EXECUTABLE = "/usr/bin/git"
GIT_ENVIRONMENT = {
    key: EXACT_CHILD_ENVIRONMENT[key]
    for key in (
        "PATH",
        "LANG",
        "LC_ALL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_TERMINAL_PROMPT",
        "GIT_OPTIONAL_LOCKS",
    )
}
MINIMUM_FREE_OUTPUT_BYTES = 16 * 1024**3
REQUIRED_SOURCE_PATHS = {
    "lewm_package": "lewm/__init__.py",
    "benchmarks_package": "lewm/benchmarks/__init__.py",
    "counterfactual_metrics": "lewm/benchmarks/counterfactual.py",
    "datasets_package": "lewm/datasets/__init__.py",
    "models_package": "lewm/models/__init__.py",
    "base_world_model": "lewm/models/lewm.py",
    "phase2d_spatial_model": "lewm/models/phase2d_spatial_lewm.py",
    "base_predictor": "lewm/models/predictor.py",
    "primitive_affordance": "lewm/models/primitive_affordance.py",
    "sigreg": "lewm/models/sigreg.py",
    "source_action_utility": "lewm/models/source_action_utility.py",
    "spatial_lewm": "lewm/models/spatial_lewm.py",
    "spatial_predictor": "lewm/models/spatial_predictor.py",
    "worker": "scripts/execute_go2_world_model_existing_pool_three_arm_v1.py",
    "checker": "scripts/check_go2_world_model_existing_pool_three_arm_v1.py",
    "external_supervisor": (
        "scripts/run_go2_world_model_existing_pool_three_arm_authorized_v1.py"
    ),
    "experiment_metrics": (
        "lewm/benchmarks/go2_world_model_existing_pool_three_arm_v1.py"
    ),
    "temporal_metrics": (
        "lewm/benchmarks/go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "h6_dataset": (
        "lewm/datasets/go2_explicit_plan_discounted_successor_state_v27.py"
    ),
    "h6_main_pool_census": (
        "lewm/benchmarks/go2_recurrent_jepa_main_pool_census.py"
    ),
    "h6_sequence_contract_v2": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences_v2.py"
    ),
    "h6_sequence_contract_v1": (
        "lewm/datasets/go2_recurrent_h4_rgb_sequences.py"
    ),
    "temporal_model": (
        "lewm/models/rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_model": (
        "lewm/models/rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "encoders": "lewm/models/encoders.py",
    "temporal_training_core": (
        "scripts/run_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "temporal_evaluator": (
        "scripts/evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1.py"
    ),
    "spatial_evaluator": (
        "scripts/evaluate_go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "spatial_metrics": (
        "lewm/benchmarks/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1.py"
    ),
    "place_data": "lewm/datasets/go2_memory_role_place_triplets_v1.py",
    "packer": "scripts/dev_pack_h6_temporal_frames.py",
    "scaled_runtime": "scripts/dev_train_temporal_jepa_scaled.py",
}

_SHA256 = frozenset("0123456789abcdef")
_AUTHORITY_KEYS = frozenset(
    {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "authorizer",
        "issued_at",
        "scientific_claim_authorized",
        "network_access",
        "source_commit",
        "review_commit",
        "preregistration_binding",
        "plan_binding",
        "review_binding",
        "source_bindings",
        "runtime",
        "input_bindings",
        "predecessor_terminal_failure_binding",
        "output_root",
        "attempt",
        "caps",
        "authorized_command",
        "execution",
        "external_supervisor",
    }
)
_PLAN_KEYS = frozenset(
    {
        "schema",
        "purpose",
        "citable_as_scientific_evidence",
        "authorizes_retry_or_resume",
        "development_only",
        "claim_scope",
        "arm_order",
        "output_root",
        "attempt",
        "caps",
        "runtime",
        "input_bindings",
        "input_binding_interpretation",
        "execution",
        "network_access",
        "minimum_free_output_bytes_before_reservation",
        "result_chain",
        "replacement_of",
        "predecessor_terminal_failure_binding",
        "prior_attempt_runtime_payloads_authorized_as_inputs",
        "pack_rebuilt_fresh",
        "integrity_corrections",
    }
)
_ATTEMPT_KEYS = frozenset(
    {
        "id",
        "root",
        "maximum_attempts",
        "must_be_absent",
        "reservation_consumes_attempt",
        "retry",
        "resume",
        "overwrite",
        "refill",
    }
)
class ThreeArmSupervisionError(RuntimeError):
    """Raised when authority or one-shot execution fails closed."""


def _fail(message: str) -> None:
    raise ThreeArmSupervisionError(message)


def _reject_protected_path(path: Path, *, label: str) -> None:
    lowered = tuple(part.lower() for part in Path(path).parts)
    if any(
        part == "sealed_test.json"
        or part == "sealed"
        or part.startswith("sealed_")
        or part in {"heldout", "held_out", "held-out"}
        or part.startswith("heldout_")
        or part.startswith("held_out_")
        or part.startswith("held-out-")
        for part in lowered
    ):
        _fail(f"{label} path is custody-protected")


def _strict_object(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            _fail(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ThreeArmSupervisionError(
                    f"non-finite JSON value in {label}: {value}"
                )
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ThreeArmSupervisionError(f"invalid JSON in {label}") from exc


def _is_sha256(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in _SHA256 for character in value)
    )


def _is_commit(value: Any) -> bool:
    return (
        type(value) is str
        and len(value) == 40
        and all(character in _SHA256 for character in value)
    )


def _plain_dict(value: Any, *, label: str) -> dict[str, Any]:
    if type(value) is not dict:
        _fail(f"{label} must be a plain JSON object")
    return value


def _exact_keys(value: Mapping[str, Any], expected: Iterable[str], *, label: str) -> None:
    observed = set(value)
    required = set(expected)
    if observed != required:
        _fail(
            f"{label} keys changed: missing={sorted(required - observed)}, "
            f"unexpected={sorted(observed - required)}"
        )


def file_binding(path: Path) -> dict[str, Any]:
    """Hash one regular non-symlink file under a stable inode/size check."""

    selected = Path(path)
    _reject_protected_path(selected, label="bound file")
    if selected.is_symlink() or not selected.is_file():
        _fail(f"bound file is absent, non-regular, or a symlink: {selected}")
    before = selected.stat()
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        _fail(f"bound file changed while being read: {selected}")
    return {
        "path": str(selected.resolve()),
        "file_sha256": digest.hexdigest(),
        "byte_count": int(after.st_size),
    }


def binding_shape(value: Any, *, label: str) -> dict[str, Any]:
    binding = _plain_dict(value, label=label)
    _exact_keys(
        binding,
        ("path", "file_sha256", "byte_count"),
        label=label,
    )
    if type(binding["path"]) is not str or not binding["path"]:
        _fail(f"{label}.path is invalid")
    _reject_protected_path(Path(binding["path"]), label=label)
    if not _is_sha256(binding["file_sha256"]):
        _fail(f"{label}.file_sha256 is invalid")
    if type(binding["byte_count"]) is not int or binding["byte_count"] < 1:
        _fail(f"{label}.byte_count is invalid")
    return dict(binding)


def _resolve_bound_path(path_text: str) -> Path:
    value = Path(path_text)
    return value if value.is_absolute() else REPO_ROOT / value


def verify_binding(value: Any, *, label: str) -> dict[str, Any]:
    expected = binding_shape(value, label=label)
    actual = file_binding(_resolve_bound_path(expected["path"]))
    if actual["byte_count"] != expected["byte_count"]:
        _fail(f"{label} byte count changed")
    if actual["file_sha256"] != expected["file_sha256"]:
        _fail(f"{label} SHA-256 changed")
    return actual


def _read_bound_json(value: Any, *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    expected = binding_shape(value, label=label)
    path = _resolve_bound_path(str(expected["path"]))
    actual = verify_binding(expected, label=label)
    raw = path.read_bytes()
    if file_binding(path) != actual:
        _fail(f"{label} changed while being parsed")
    document = strict_json_bytes(raw, label=label)
    return _plain_dict(document, label=label), actual


def _git_output(*args: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        [GIT_EXECUTABLE, *args],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return result.stdout if binary else result.stdout.strip()


def _git_head() -> str:
    value = _git_output("rev-parse", "HEAD")
    assert isinstance(value, str)
    return value


def _require_commit_ancestor(commit: Any, *, label: str) -> str:
    if not _is_commit(commit):
        _fail(f"{label} commit must be full lowercase Git hex")
    result = subprocess.run(
        [GIT_EXECUTABLE, "merge-base", "--is-ancestor", str(commit), "HEAD"],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        _fail(f"{label} commit is not an ancestor of HEAD")
    return str(commit)


def _require_strict_commit_ancestor(
    ancestor: str, descendant: str, *, label: str
) -> None:
    if ancestor == descendant:
        _fail(f"{label} commits must be distinct")
    result = subprocess.run(
        [GIT_EXECUTABLE, "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=REPO_ROOT,
        env=GIT_ENVIRONMENT,
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        _fail(f"{label} commit ordering is invalid")


def _require_binding_at_commit(
    binding: Mapping[str, Any], *, commit: str, label: str
) -> None:
    try:
        path = _resolve_bound_path(str(binding["path"])).resolve(strict=True)
        relative = path.relative_to(REPO_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise ThreeArmSupervisionError(
            f"{label} must be a tracked repository file"
        ) from exc
    try:
        raw = _git_output("show", f"{commit}:{relative.as_posix()}", binary=True)
    except subprocess.CalledProcessError as exc:
        raise ThreeArmSupervisionError(
            f"{label} is absent from commit {commit}"
        ) from exc
    assert isinstance(raw, bytes)
    if len(raw) != int(binding["byte_count"]):
        _fail(f"committed {label} byte count changed")
    if hashlib.sha256(raw).hexdigest() != str(binding["file_sha256"]):
        _fail(f"committed {label} SHA-256 changed")


def _validate_binding_map(
    value: Any, *, label: str, verify_files: bool
) -> dict[str, dict[str, Any]]:
    mapping = _plain_dict(value, label=label)
    if not mapping:
        _fail(f"{label} must not be empty")
    result: dict[str, dict[str, Any]] = {}
    for name, item in mapping.items():
        if type(name) is not str or not name:
            _fail(f"{label} has an invalid binding name")
        binding = binding_shape(item, label=f"{label}.{name}")
        if verify_files:
            verify_binding(binding, label=f"{label}.{name}")
        result[name] = binding
    return result


def _validate_attempt(value: Any, *, output_root: str) -> dict[str, Any]:
    attempt = _plain_dict(value, label="authority.attempt")
    _exact_keys(attempt, _ATTEMPT_KEYS, label="authority.attempt")
    expected_root = str(ATTEMPT_ROOT.resolve(strict=False))
    if (
        attempt.get("id") != ATTEMPT_ID
        or attempt.get("root") != expected_root
        or output_root != expected_root
        or attempt.get("maximum_attempts") != 1
        or attempt.get("must_be_absent") is not True
        or attempt.get("reservation_consumes_attempt") is not True
        or attempt.get("retry") is not False
        or attempt.get("resume") is not False
        or attempt.get("overwrite") is not False
        or attempt.get("refill") is not False
    ):
        _fail("authority attempt is not the exact fresh one-shot attempt")
    return dict(attempt)


def _validate_caps(value: Any) -> dict[str, Any]:
    caps = _plain_dict(value, label="authority.caps")
    _exact_keys(
        caps,
        ("maximum_wall_seconds", "maximum_gpu_seconds", "maximum_training_updates"),
        label="authority.caps",
    )
    wall = caps.get("maximum_wall_seconds")
    gpu = caps.get("maximum_gpu_seconds")
    updates = caps.get("maximum_training_updates")
    if (
        type(wall) not in (int, float)
        or not math.isfinite(float(wall))
        or float(wall) != 43_200.0
        or type(gpu) not in (int, float)
        or not math.isfinite(float(gpu))
        or float(gpu) != 36_000.0
        or type(updates) is not int
        or updates != 700
    ):
        _fail(
            "authority caps must be exactly 43,200 wall seconds, 36,000 GPU "
            "seconds, and 700 updates"
        )
    return dict(caps)


def _validate_runtime(value: Any, *, verify_files: bool) -> dict[str, Any]:
    runtime = _plain_dict(value, label="authority.runtime")
    if set(runtime) != {"python_invocation_path", "environment", "bindings"}:
        _fail("authority runtime keys changed")
    invocation = runtime["python_invocation_path"]
    environment = runtime["environment"]
    if type(invocation) is not str or not Path(invocation).is_absolute():
        _fail("runtime Python invocation must be absolute")
    if type(environment) is not dict or any(
        type(key) is not str or type(item) is not str
        for key, item in environment.items()
    ):
        _fail("runtime environment must be a string map")
    if environment != EXACT_CHILD_ENVIRONMENT:
        _fail("runtime environment is not the exact allowlisted child environment")
    bindings = _validate_binding_map(
        runtime["bindings"], label="authority.runtime.bindings", verify_files=verify_files
    )
    required_bindings = (
        "python_executable_target",
        "python_environment_config",
        "git_executable",
    )
    if set(bindings) != set(required_bindings):
        _fail("runtime executable binding inventory changed")
    if verify_files:
        invocation_path = Path(invocation)
        target = _resolve_bound_path(bindings["python_executable_target"]["path"])
        config = _resolve_bound_path(bindings["python_environment_config"]["path"])
        git = _resolve_bound_path(bindings["git_executable"]["path"])
        if (
            invocation_path.resolve(strict=True) != target.resolve(strict=True)
            or config.name != "pyvenv.cfg"
            or invocation_path.parent.parent != config.parent
            or git.resolve(strict=True) != Path(GIT_EXECUTABLE).resolve(strict=True)
        ):
            _fail("runtime Python/Git executables differ from their bindings")
    return {
        "python_invocation_path": invocation,
        "environment": dict(environment),
        "bindings": bindings,
    }


def _validate_exact_runtime(value: Any, *, verify_files: bool) -> dict[str, Any]:
    runtime = _validate_runtime(value, verify_files=verify_files)
    if runtime != EXPECTED_RUNTIME:
        _fail("authority runtime differs from the exact preregistered runtime")
    return runtime


def _validate_exact_inputs(
    value: Any, *, verify_files: bool
) -> dict[str, dict[str, Any]]:
    inputs = _validate_binding_map(
        value,
        label="authority.input_bindings",
        verify_files=verify_files,
    )
    if inputs != EXPECTED_INPUT_BINDINGS:
        _fail("authority input bindings differ from the exact preregistered inputs")
    return inputs


def _validate_review(
    review: Mapping[str, Any],
    *,
    source_commit: str,
    source_bindings: list[dict[str, Any]],
    plan_binding: Mapping[str, Any],
    preregistration_binding: Mapping[str, Any],
    predecessor_failure_binding: Mapping[str, Any],
) -> None:
    _exact_keys(
        review,
        (
            "schema",
            "status",
            "authority_granted_by_this_document",
            "reviewer",
            "reviewed_source_commit",
            "reviewed_source_bindings",
            "reviewed_plan_binding",
            "reviewed_predecessor_terminal_failure_binding",
            "reviewed_preregistration_binding",
            "review_scope",
            "verification",
            "custody",
            "resolved_findings",
            "remaining_findings",
        ),
        label="independent source review",
    )
    reviewer = review.get("reviewer")
    if (
        review.get("schema") != REVIEW_SCHEMA
        or review.get("status") != REVIEW_STATUS
        or review.get("authority_granted_by_this_document") is not False
        or type(reviewer) is not dict
        or set(reviewer) != {"identity", "materialization"}
        or type(reviewer.get("identity")) is not str
        or not reviewer["identity"].strip()
        or type(reviewer.get("materialization")) is not str
        or not reviewer["materialization"].strip()
        or review.get("reviewed_source_commit") != source_commit
        or review.get("reviewed_source_bindings") != source_bindings
        or review.get("reviewed_plan_binding") != plan_binding
        or review.get("reviewed_preregistration_binding")
        != preregistration_binding
        or review.get("reviewed_predecessor_terminal_failure_binding")
        != predecessor_failure_binding
        or review.get("review_scope") != EXPECTED_REVIEW_SCOPE
        or review.get("verification") != EXPECTED_REVIEW_VERIFICATION
        or review.get("custody") != EXPECTED_REVIEW_CUSTODY
        or type(review.get("resolved_findings")) is not list
        or review.get("remaining_findings") != []
    ):
        _fail("independent source review is not an exact non-authorizing PASS")


def _validate_plan_registered_fields(plan: Mapping[str, Any]) -> None:
    _exact_keys(plan, _PLAN_KEYS, label="plan")
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("purpose") != PLAN_PURPOSE
        or plan.get("citable_as_scientific_evidence") is not False
        or plan.get("authorizes_retry_or_resume") is not False
        or plan.get("development_only") is not True
        or plan.get("claim_scope")
        != "requested_action_factual_learnability_only"
        or plan.get("arm_order") != ARM_ORDER
        or plan.get("input_binding_interpretation")
        != {
            "h6_indices_bind_rgb_leaf_paths_transitively": True,
            "individual_rgb_leaves_prehashed_in_authority": False,
            "opened_rgb_leaf_bytes_identity_bound_during_fresh_packing": True,
            "permitted_temporal_positions": [0, 1, 2, 3],
        }
        or plan.get("network_access") is not False
        or plan.get("minimum_free_output_bytes_before_reservation")
        != MINIMUM_FREE_OUTPUT_BYTES
        or plan.get("result_chain")
        != [
            "integrity_replacement_v3_external_supervisor_reservation",
            "integrity_replacement_v3_worker_result",
            "integrity_replacement_v3_receipt_only_checker",
            "integrity_replacement_v3_external_supervisor_terminal",
        ]
        or plan.get("prior_attempt_runtime_payloads_authorized_as_inputs")
        is not False
        or plan.get("pack_rebuilt_fresh") is not True
    ):
        _fail("bound plan differs from the exact authorized experiment")


def _validate_authorizer(value: Any, *, issued_at: Any) -> dict[str, Any]:
    authorizer = _plain_dict(value, label="authority.authorizer")
    _exact_keys(authorizer, ("identity",), label="authority.authorizer")
    if (
        type(authorizer.get("identity")) is not str
        or not authorizer["identity"].strip()
        or type(issued_at) is not str
        or not issued_at.strip()
    ):
        _fail("durable authority authorizer/issue evidence is absent")
    return dict(authorizer)


def _validate_predecessor_failure(
    document: Mapping[str, Any],
) -> None:
    # The direct predecessor audit is a source document with an exact byte/hash
    # binding.  Compare the supplied mapping with those exact reviewed bytes
    # first, then assert the scientific boundary explicitly so no semantically
    # different pretraining failure can be substituted.
    expected_binding = verify_binding(
        PREDECESSOR_FAILURE_BINDING,
        label="predecessor terminal failure audit",
    )
    expected_document = strict_json_bytes(
        Path(expected_binding["path"]).read_bytes(),
        label="predecessor terminal failure audit",
    )
    if type(expected_document) is not dict or document != expected_document:
        _fail("predecessor terminal failure audit is not replacement-safe")

    attempt = document.get("attempt")
    terminal_artifacts = document.get("terminal_artifacts")
    authorized_identity = document.get("authorized_identity")
    accounting = document.get("execution_accounting")
    terminal_evidence = document.get("terminal_evidence")
    failure = document.get("failure")
    root_cause = document.get("root_cause")
    diagnostic = document.get("source_only_synthetic_diagnostic")
    correction = document.get("narrow_integrity_correction")
    scientific_conclusion = document.get("scientific_conclusion")
    successor = document.get("successor_boundary")
    custody = document.get("custody")
    if any(
        type(section) is not dict
        for section in (
            attempt,
            terminal_artifacts,
            authorized_identity,
            accounting,
            terminal_evidence,
            failure,
            root_cause,
            diagnostic,
            correction,
            scientific_conclusion,
            successor,
            custody,
        )
    ):
        _fail("predecessor terminal failure audit is not replacement-safe")
    _exact_keys(
        document,
        (
            "schema",
            "status",
            "date",
            "attempt",
            "terminal_artifacts",
            "authorized_identity",
            "predecessor_failure_audit",
            "execution_accounting",
            "terminal_evidence",
            "failure",
            "root_cause",
            "source_only_synthetic_diagnostic",
            "narrow_integrity_correction",
            "scientific_conclusion",
            "successor_boundary",
            "custody",
        ),
        label="predecessor terminal failure audit",
    )
    if (
        document.get("schema") != PREDECESSOR_FAILURE_SCHEMA
        or document.get("status") != PREDECESSOR_FAILURE_STATUS
        or document.get("date") != "2026-08-01"
        or terminal_artifacts != PREDECESSOR_TERMINAL_ARTIFACTS
        or authorized_identity != PREDECESSOR_AUTHORIZED_IDENTITY
        or document.get("predecessor_failure_audit")
        != PREDECESSOR_ORIGINAL_FAILURE_BINDING
        or attempt
        != {
            "id": PREDECESSOR_ATTEMPT_ID,
            "root": str(PREDECESSOR_ATTEMPT_ROOT.resolve()),
            "consumed": True,
            "retry_authorized": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
            "refill_authorized": False,
        }
        or accounting
        != {
            "supervisor_wall_elapsed_seconds_at_terminal": 15.994941210956313,
            "training_updates_completed": 0,
            "optimizer_steps_completed": 0,
            "scientific_verdict_emitted": False,
            "worker_result_published": False,
            "receipt_checker_run": False,
        }
        or terminal_evidence
        != {
            "terminal_status": "CONSUMED_TERMINAL_FAILURE",
            "worker_failure_status": "ATTEMPT_CONSUMED_WORKER_FAILURE",
            "phase_receipts_empty": True,
            "result_binding_absent": True,
            "receipt_check_binding_absent": True,
            "automatic_checkpoint_selection_performed": False,
            "citable_as_scientific_evidence": False,
        }
        or failure.get("type") != "ThreeArmWorkerError"
        or failure.get("message")
        != "head, blind-treatment, or online/target parity probe failed"
        or failure.get("classification")
        != "SOURCE_PARITY_PROBE_COMPARED_DIFFERENT_ATTENTION_MODES"
        or failure.get("checkpoint_loader_integrity_correction_passed") is not True
        or failure.get("arm_clone_trainability_integrity_correction_passed")
        is not True
        or failure.get("packing_completed_before_failure") is not True
        or failure.get("all_arm_optimizers_constructed_before_failure") is not True
        or failure.get("runtime_failure_receipt_identifies_individual_subprobe")
        is not False
        or root_cause.get("frozen_reference_mode") != "evaluation"
        or root_cause.get("copied_arm_mode_at_runtime_probe") != "training"
        or root_cause.get("exact_zero_conditioned_helper_predicate") is not True
        or root_cause.get("runtime_receipt_scope") != "aggregate guard failure only"
        or diagnostic.get("uses_real_training_or_validation_payload") is not False
        or diagnostic.get("uses_real_checkpoint_or_snapshot_payload") is not False
        or diagnostic.get("interpretation_boundary")
        != (
            "this independently reproduces and isolates the source defect; the "
            "consumed V2 terminal receipt itself records only the aggregate guard "
            "failure"
        )
        or correction.get("science_identical") is not True
        or correction.get("scope")
        != "payload-free pretraining parity probe mode alignment only"
        or correction.get("parameter_values_changed") is not False
        or correction.get("parameter_names_shapes_dtypes_or_count_changed")
        is not False
        or correction.get("optimizer_groups_hyperparameters_or_state_changed")
        is not False
        or correction.get("random_number_generation_changed") is not False
        or correction.get("loss_schedule_data_metrics_or_thresholds_changed")
        is not False
        or correction.get("actual_update_zero_or_training_mode_changed")
        is not False
        or scientific_conclusion.get("data_learnability_tested") is not False
        or scientific_conclusion.get("objective_learnability_tested") is not False
        or scientific_conclusion.get("architecture_learnability_tested")
        is not False
        or scientific_conclusion.get("only_valid_conclusion")
        != (
            "integrity replacement V2 is a consumed pre-training source "
            "parity-guard failure"
        )
        or successor.get("retry_or_resume_attempt_v1") is not False
        or successor.get("reuse_attempt_v1_pack_or_runtime_payloads") is not False
        or successor.get("fresh_one_shot_authority_required") is not True
        or successor.get("fresh_absent_output_root_required") is not True
        or successor.get("fresh_integrity_replacement_v3_preregistration_required")
        is not True
        or successor.get("this_document_authorizes_v3") is not False
        or custody.get("sealed_or_heldout_material_opened") is not False
        or custody.get("protected_evaluation_authorized") is not False
        or custody.get("network_access_used") is not False
        or custody.get(
            "replacement_v2_runtime_tensor_or_pack_payload_reopened_after_terminal"
        )
        is not False
    ):
        _fail("predecessor terminal failure audit is not replacement-safe")


def _reverify_predecessor_failure_evidence(
    document: Mapping[str, Any],
) -> None:
    for label, binding in document["terminal_artifacts"].items():
        verify_binding(binding, label=f"predecessor terminal {label}")
    for label in (
        "authority_binding",
        "plan_binding",
        "independent_review_binding",
        "preregistration_binding",
    ):
        verify_binding(
            document["authorized_identity"][label],
            label=f"predecessor {label}",
        )
    verify_binding(
        document["predecessor_failure_audit"],
        label="original predecessor terminal failure audit",
    )


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, dict[str, Any]],
]:
    """Validate the complete launch closure before consuming the attempt."""

    if authority_path.resolve(strict=True) != (
        REPO_ROOT / AUTHORITY_RELATIVE
    ).resolve(strict=True):
        _fail("caller selected a different V3 authority path")
    authority_binding = file_binding(authority_path)
    if authority_binding["byte_count"] != expected_byte_count:
        _fail("authority byte count disagrees with caller")
    if authority_binding["file_sha256"] != expected_sha256:
        _fail("authority SHA-256 disagrees with caller")
    _require_binding_at_commit(
        authority_binding, commit="HEAD", label="execution authority"
    )
    authority_raw = authority_path.read_bytes()
    if file_binding(authority_path) != authority_binding:
        _fail("authority changed while being parsed")
    authority = _plain_dict(
        strict_json_bytes(authority_raw, label="authority"),
        label="authority",
    )
    _exact_keys(authority, _AUTHORITY_KEYS, label="authority")
    if (
        authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("authority_granted_by_this_document") is not True
        or authority.get("scientific_claim_authorized") is not False
        or authority.get("network_access") is not False
    ):
        _fail("authority semantic grant is invalid")
    authority["authorizer"] = _validate_authorizer(
        authority.get("authorizer"), issued_at=authority.get("issued_at")
    )
    source_commit = _require_commit_ancestor(
        authority.get("source_commit"), label="authorized source"
    )
    review_commit = _require_commit_ancestor(
        authority.get("review_commit"), label="independent review"
    )
    execution_head = _git_head()
    _require_strict_commit_ancestor(
        source_commit,
        review_commit,
        label="source-before-review",
    )
    _require_strict_commit_ancestor(
        review_commit,
        execution_head,
        label="review-before-authority",
    )
    preregistration_binding = binding_shape(
        authority["preregistration_binding"],
        label="preregistration",
    )
    verify_binding(preregistration_binding, label="preregistration")
    preregistration_path = _resolve_bound_path(
        preregistration_binding["path"]
    ).resolve(strict=True)
    if preregistration_path != (REPO_ROOT / PREREGISTRATION_RELATIVE).resolve(
        strict=True
    ):
        _fail("authority binds a different V3 preregistration")
    declared_plan_binding = binding_shape(authority["plan_binding"], label="plan")
    declared_review_binding = binding_shape(
        authority["review_binding"], label="independent source review"
    )
    if _resolve_bound_path(declared_plan_binding["path"]).resolve(strict=True) != (
        REPO_ROOT / PLAN_RELATIVE
    ).resolve(strict=True):
        _fail("authority binds a different V3 plan")
    if _resolve_bound_path(declared_review_binding["path"]).resolve(strict=True) != (
        REPO_ROOT / REVIEW_RELATIVE
    ).resolve(strict=True):
        _fail("authority binds a different V3 independent review")
    plan, plan_binding = _read_bound_json(declared_plan_binding, label="plan")
    review, review_binding = _read_bound_json(
        declared_review_binding, label="independent source review"
    )
    _require_binding_at_commit(
        preregistration_binding,
        commit=source_commit,
        label="preregistration",
    )
    _require_binding_at_commit(plan_binding, commit=source_commit, label="plan")
    _require_binding_at_commit(plan_binding, commit="HEAD", label="plan")
    _require_binding_at_commit(
        review_binding,
        commit=review_commit,
        label="independent source review",
    )
    _require_binding_at_commit(
        review_binding,
        commit="HEAD",
        label="independent source review",
    )

    raw_sources = authority["source_bindings"]
    if type(raw_sources) is not list or not raw_sources:
        _fail("authority source closure is absent")
    source_bindings: list[dict[str, Any]] = []
    by_name: dict[str, dict[str, Any]] = {}
    for row_value in raw_sources:
        row = _plain_dict(row_value, label="source binding row")
        _exact_keys(row, ("name", "binding"), label="source binding row")
        name = row["name"]
        if type(name) is not str or not name or name in by_name:
            _fail("source binding names are invalid or duplicated")
        binding = binding_shape(row["binding"], label=f"source {name}")
        verify_binding(binding, label=f"source {name}")
        _require_binding_at_commit(
            binding, commit=source_commit, label=f"source {name}"
        )
        row_copy = {"name": name, "binding": binding}
        source_bindings.append(row_copy)
        by_name[name] = binding
    if set(by_name) != set(REQUIRED_SOURCE_PATHS):
        _fail("source closure inventory changed")
    for name, relative_path in REQUIRED_SOURCE_PATHS.items():
        expected = REPO_ROOT / relative_path
        observed = _resolve_bound_path(by_name[name]["path"]).resolve(strict=True)
        if observed != expected.resolve(strict=True):
            _fail(f"authority binds a different {name} source")

    output_root = authority["output_root"]
    if type(output_root) is not str:
        _fail("authority output_root is invalid")
    attempt = _validate_attempt(authority["attempt"], output_root=output_root)
    caps = _validate_caps(authority["caps"])
    runtime = _validate_exact_runtime(authority["runtime"], verify_files=True)
    inputs = _validate_exact_inputs(authority["input_bindings"], verify_files=True)
    if binding_shape(
        authority["predecessor_terminal_failure_binding"],
        label="predecessor terminal failure audit",
    ) != PREDECESSOR_FAILURE_BINDING:
        _fail("authority binds a different predecessor terminal failure audit")
    predecessor_failure, predecessor_failure_binding = _read_bound_json(
        authority["predecessor_terminal_failure_binding"],
        label="predecessor terminal failure audit",
    )
    _require_binding_at_commit(
        predecessor_failure_binding,
        commit=source_commit,
        label="predecessor terminal failure audit",
    )
    _validate_predecessor_failure(predecessor_failure)
    _reverify_predecessor_failure_evidence(predecessor_failure)

    execution = _plain_dict(authority["execution"], label="authority.execution")
    _exact_keys(execution, ("worker_path", "checker_path"), label="authority.execution")
    if execution != {
        "worker_path": str((REPO_ROOT / WORKER_RELATIVE).resolve()),
        "checker_path": str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
    }:
        _fail("authority execution paths changed")

    command = _plain_dict(
        authority["authorized_command"], label="authority.authorized_command"
    )
    _exact_keys(command, ("argv_template",), label="authority.authorized_command")
    expected_command = [
        runtime["python_invocation_path"],
        str((REPO_ROOT / SUPERVISOR_RELATIVE).resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        "<CALLER_BOUND_AUTHORITY_BYTE_COUNT>",
        "--expected-authority-sha256",
        "<CALLER_BOUND_AUTHORITY_SHA256>",
    ]
    if command["argv_template"] != expected_command:
        _fail("authority does not bind the exact external supervisor invocation")

    external = _plain_dict(
        authority["external_supervisor"], label="authority.external_supervisor"
    )
    _exact_keys(
        external,
        ("source_binding", "terminal_reviewer"),
        label="authority.external_supervisor",
    )
    if (
        binding_shape(external["source_binding"], label="external supervisor")
        != by_name["external_supervisor"]
        or type(external["terminal_reviewer"]) is not str
        or not external["terminal_reviewer"].strip()
    ):
        _fail("external supervisor contract is invalid")
    if Path(verify_binding(external["source_binding"], label="external supervisor")["path"]) != Path(__file__).resolve():
        _fail("authority external supervisor source does not identify this file")

    _validate_plan_registered_fields(plan)
    if (
        plan.get("schema") != PLAN_SCHEMA
        or plan.get("purpose") != PLAN_PURPOSE
        or plan.get("citable_as_scientific_evidence") is not False
        or plan.get("authorizes_retry_or_resume") is not False
        or plan.get("development_only") is not True
        or plan.get("claim_scope")
        != "requested_action_factual_learnability_only"
        or plan.get("arm_order") != ARM_ORDER
        or plan.get("output_root") != output_root
        or plan.get("attempt") != attempt
        or plan.get("caps") != caps
        or plan.get("runtime") != runtime
        or plan.get("input_bindings") != inputs
        or plan.get("input_binding_interpretation")
        != {
            "h6_indices_bind_rgb_leaf_paths_transitively": True,
            "individual_rgb_leaves_prehashed_in_authority": False,
            "opened_rgb_leaf_bytes_identity_bound_during_fresh_packing": True,
            "permitted_temporal_positions": [0, 1, 2, 3],
        }
        or plan.get("predecessor_terminal_failure_binding")
        != predecessor_failure_binding
        or plan.get("replacement_of")
        != {
            "attempt_id": PREDECESSOR_ATTEMPT_ID,
            "output_root": str(PREDECESSOR_ATTEMPT_ROOT.resolve()),
            "terminal_status": "CONSUMED_TERMINAL_FAILURE",
            "retry_or_resume_authorized": False,
            "runtime_payload_reuse_authorized": False,
        }
        or plan.get("integrity_corrections")
        != {
            "checkpoint_loader": {
                "scope": "checkpoint_loader_validation_only",
                "ema_update_count": {
                    "dtype": "torch.int64",
                    "layout": "torch.strided",
                    "shape": [],
                    "exact_value": 1000,
                    "migrated": False,
                },
                "all_other_state_tensors": "finite_strided_torch.float32",
                "scientific_contract_changed": False,
            },
            "arm_clone_trainability": {
                "scope": "post_clone_requires_grad_restoration_only",
                "operation": "ArmCore.requires_grad_(true)",
                "parameter_values_changed": False,
                "parameter_inventory_changed": False,
                "expected_parameter_tensor_count": 36,
                "expected_predictor_tensor_count": 30,
                "expected_memory_tensor_count": 6,
                "scientific_contract_changed": False,
            },
            "payload_free_parity_probe_mode_alignment": {
                "scope": "pretraining_payload_free_contract_probe_mode_only",
                "operation": (
                    "all copied ArmCore module trees eval for the unchanged "
                    "no_grad zero-tensor probe, then all recursively train before "
                    "return"
                ),
                "probe_inputs_changed": False,
                "probe_assertions_or_thresholds_changed": False,
                "parameter_values_inventory_identities_or_requires_grad_changed": False,
                "optimizer_groups_hyperparameters_or_state_changed": False,
                "rng_state_changed": False,
                "actual_training_forward_mode_changed": False,
                "actual_evaluation_forward_mode_changed": False,
                "data_loss_metrics_thresholds_or_schedule_changed": False,
                "scientific_contract_changed": False,
            },
        }
        or plan.get("prior_attempt_runtime_payloads_authorized_as_inputs")
        is not False
        or plan.get("pack_rebuilt_fresh") is not True
        or plan.get("execution") != execution
        or plan.get("network_access") is not False
        or plan.get("minimum_free_output_bytes_before_reservation")
        != MINIMUM_FREE_OUTPUT_BYTES
        or plan.get("result_chain")
        != [
            "integrity_replacement_v3_external_supervisor_reservation",
            "integrity_replacement_v3_worker_result",
            "integrity_replacement_v3_receipt_only_checker",
            "integrity_replacement_v3_external_supervisor_terminal",
        ]
    ):
        _fail("bound plan differs from the exact authorized experiment")
    _validate_review(
        review,
        source_commit=source_commit,
        source_bindings=source_bindings,
        plan_binding=plan_binding,
        preregistration_binding=preregistration_binding,
        predecessor_failure_binding=predecessor_failure_binding,
    )
    # Normalize values whose paths may have been repository-relative.
    authority = dict(authority)
    authority["attempt"] = attempt
    authority["caps"] = caps
    authority["runtime"] = runtime
    authority["input_bindings"] = inputs
    authority["predecessor_terminal_failure_binding"] = (
        predecessor_failure_binding
    )
    authority["source_bindings"] = source_bindings
    authority["review_commit"] = review_commit
    authority["preregistration_binding"] = preregistration_binding
    authority["review_binding"] = binding_shape(
        authority["review_binding"], label="authority.review_binding"
    )
    authority["plan_binding"] = binding_shape(
        authority["plan_binding"], label="authority.plan_binding"
    )
    return authority, authority_binding, plan, plan_binding, by_name


def _require_fresh_attempt_root(path_text: str) -> Path:
    expected = ATTEMPT_ROOT.resolve(strict=False)
    candidate = Path(path_text)
    if not candidate.is_absolute() or candidate.resolve(strict=False) != expected:
        _fail("attempt root is not the one exact authorized development root")
    development = DEVELOPMENT_ROOT.resolve(strict=True)
    try:
        relative = expected.relative_to(development)
    except ValueError as exc:
        raise ThreeArmSupervisionError("attempt root escapes .generated/dev") from exc
    cursor = development
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            _fail(f"attempt path contains a symlink: {cursor}")
        if not cursor.exists():
            break
    if candidate.exists() or candidate.is_symlink():
        _fail(f"attempt root is not fresh: {candidate}")
    free_bytes = int(shutil.disk_usage(DEVELOPMENT_ROOT).free)
    if free_bytes < MINIMUM_FREE_OUTPUT_BYTES:
        _fail(
            "development output volume lacks the 16 GiB preregistered free-space floor"
        )
    return candidate


def _require_idle_authorized_device() -> None:
    verify_binding(AMD_SMI_BINDING, label="authorized-device process inspector")
    try:
        completed = subprocess.run(
            [AMD_SMI_BINDING["path"], "process", "-g", "0", "--json"],
            cwd=REPO_ROOT,
            env=AMD_SMI_ENVIRONMENT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ThreeArmSupervisionError(
            "authorized-device idle preflight could not complete"
        ) from exc
    if completed.returncode != 0:
        _fail("authorized-device idle preflight failed")
    observed = strict_json_bytes(
        completed.stdout,
        label="authorized-device process inventory",
    )
    if observed != [
        {
            "gpu": 0,
            "process_list": [
                {"process_info": "No running processes detected"}
            ],
        }
    ]:
        _fail("authorized device 0 is not idle")


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    raw = json.dumps(
        dict(value), indent=2, sort_keys=True, allow_nan=False
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags, 0o644)
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return file_binding(path)


def _reserve_attempt(
    attempt_root: Path,
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    worker_binding: Mapping[str, Any],
    checker_binding: Mapping[str, Any],
    worker_command: Sequence[str],
    checker_command_template: Sequence[str],
    supervisor_nonce: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Exclusively create the attempt and its attempt-consuming reservation."""

    parent = attempt_root.parent
    parent.mkdir(mode=0o755, parents=False, exist_ok=True)
    if parent.is_symlink() or not parent.is_dir():
        _fail("attempt parent is not a regular directory")
    os.mkdir(attempt_root, mode=0o755)
    parent_descriptor = os.open(parent, os.O_RDONLY)
    try:
        os.fsync(parent_descriptor)
    finally:
        os.close(parent_descriptor)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "supervisor_nonce": supervisor_nonce,
        "authority_binding": dict(authority_binding),
        "plan_binding": dict(plan_binding),
        "review_binding": dict(authority["review_binding"]),
        "source_commit": authority["source_commit"],
        "review_commit": authority["review_commit"],
        "preregistration_binding": dict(authority["preregistration_binding"]),
        "source_bindings": authority["source_bindings"],
        "runtime": authority["runtime"],
        "input_bindings": authority["input_bindings"],
        "predecessor_terminal_failure_binding": authority[
            "predecessor_terminal_failure_binding"
        ],
        "attempt": authority["attempt"],
        "caps": authority["caps"],
        "worker_binding": dict(worker_binding),
        "checker_binding": dict(checker_binding),
        "output_root": authority["output_root"],
        "execution": authority["execution"],
        "worker_command": list(worker_command),
        "checker_command_template": list(checker_command_template),
        "authorized_device_idle_preflight_passed": True,
        "maximum_attempts": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    reservation_binding = _write_json_exclusive(
        attempt_root / "reservation.json", reservation
    )
    directory_descriptor = os.open(attempt_root, os.O_RDONLY)
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)
    return reservation, reservation_binding


def _child_environment(runtime: Mapping[str, Any]) -> dict[str, str]:
    environment = runtime.get("environment")
    if environment != EXACT_CHILD_ENVIRONMENT:
        _fail("refusing to construct an unbound child environment")
    return dict(EXACT_CHILD_ENVIRONMENT)


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _run_once(
    argv: Sequence[str], *, timeout: float, env: Mapping[str, str]
) -> dict[str, Any]:
    if timeout <= 0.0:
        _fail("hard wall ceiling exhausted")
    started = time.monotonic()
    process = subprocess.Popen(
        list(argv),
        cwd=REPO_ROOT,
        env=dict(env),
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise ThreeArmSupervisionError(
            "supervised command exceeded hard wall ceiling"
        ) from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if returncode != 0:
        _terminate_process_group(process)
        _fail(f"supervised command exited with status {returncode}")
    return {"argv": list(argv), "elapsed_seconds": elapsed, "exit_code": 0}


def _remaining_wall(*, wall_started: float, wall_cap: float) -> float:
    remaining = wall_cap - (time.monotonic() - wall_started)
    if remaining <= 0.0:
        _fail("hard wall ceiling exhausted")
    return remaining


def _load_result_if_present(
    attempt_root: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    path = attempt_root / "result.json"
    if not path.exists():
        return None, None
    binding = file_binding(path)
    document = strict_json_bytes(path.read_bytes(), label="worker result")
    if type(document) is not dict:
        _fail("worker result must be a JSON object")
    if file_binding(path) != binding:
        _fail("worker result changed while being loaded")
    return document, binding


def _expected_result_attempt(
    authority_attempt: Mapping[str, Any],
    *,
    reservation_binding: Mapping[str, Any],
    supervisor_nonce: str,
) -> dict[str, Any]:
    return {
        **dict(authority_attempt),
        "reservation": {
            "binding": dict(reservation_binding),
            "supervisor_nonce": supervisor_nonce,
            "status": "RESERVED_ATTEMPT_CONSUMED",
            "maximum_attempts": 1,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
        },
    }


def _validate_worker_result(
    result: Mapping[str, Any],
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    supervisor_nonce: str,
) -> None:
    runtime = result.get("runtime")
    if (
        type(runtime) is not dict
        or set(runtime) != {"authorized", "observed"}
        or runtime.get("authorized") != authority["runtime"]
        or type(runtime.get("observed")) is not dict
        or not runtime["observed"]
    ):
        _fail("worker result runtime evidence is not exactly linked")
    observed = runtime["observed"]
    expected_runtime = {
        "device_name": "AMD Radeon AI PRO R9700",
        "device_arch": "gfx1201",
        "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
        "torch_hip": "7.2.53211-e1a6bc5663",
        "numpy_version": "1.26.4",
        "pillow_version": "11.3.0",
    }
    if any(observed.get(key) != value for key, value in expected_runtime.items()):
        _fail("worker result observed runtime identity changed")
    gpu_elapsed = observed.get("gpu_phase_elapsed_seconds")
    wall_elapsed = observed.get("wall_elapsed_seconds")
    inventory = observed.get("output_inventory")
    if (
        type(gpu_elapsed) not in (int, float)
        or not math.isfinite(float(gpu_elapsed))
        or float(gpu_elapsed) < 0.0
        or float(gpu_elapsed) > float(authority["caps"]["maximum_gpu_seconds"])
        or type(wall_elapsed) not in (int, float)
        or not math.isfinite(float(wall_elapsed))
        or float(wall_elapsed) < 0.0
        or float(wall_elapsed) > float(authority["caps"]["maximum_wall_seconds"])
        or type(inventory) is not list
        or len(inventory) != len(WORKER_OUTPUT_PATHS)
        or any(type(item) is not str or not item for item in inventory)
        or len(set(inventory)) != len(inventory)
        or set(inventory) != WORKER_OUTPUT_PATHS
    ):
        _fail("worker result observed runtime/cap/output evidence is invalid")
    if (
        result.get("schema") != RESULT_SCHEMA
        or result.get("status") != RESULT_STATUS
        or result.get("authority_binding") != authority_binding
        or result.get("plan_binding") != plan_binding
        or result.get("review_binding") != authority["review_binding"]
        or result.get("source_commit") != authority["source_commit"]
        or result.get("attempt")
        != _expected_result_attempt(
            authority["attempt"],
            reservation_binding=reservation_binding,
            supervisor_nonce=supervisor_nonce,
        )
        or result.get("caps") != authority["caps"]
        or result.get("input_bindings") != authority["input_bindings"]
        or result.get("predecessor_terminal_failure_binding")
        != authority["predecessor_terminal_failure_binding"]
    ):
        _fail("worker result is not an exact linked success")


def _reservation_unchanged(
    path: Path,
    *,
    reservation: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
) -> bool:
    try:
        if file_binding(path) != reservation_binding:
            return False
        observed = strict_json_bytes(path.read_bytes(), label="reservation")
        return observed == reservation and file_binding(path) == reservation_binding
    except BaseException:
        return False


def _reverify_contract(authority: Mapping[str, Any]) -> None:
    verify_binding(authority["preregistration_binding"], label="preregistration")
    verify_binding(authority["plan_binding"], label="plan")
    verify_binding(authority["review_binding"], label="independent source review")
    for row in authority["source_bindings"]:
        verify_binding(row["binding"], label=f"source {row['name']}")
    _validate_runtime(authority["runtime"], verify_files=True)
    predecessor_failure, _binding = _read_bound_json(
        authority["predecessor_terminal_failure_binding"],
        label="predecessor terminal failure audit",
    )
    _validate_predecessor_failure(predecessor_failure)
    _reverify_predecessor_failure_evidence(predecessor_failure)
    _validate_binding_map(
        authority["input_bindings"],
        label="authority.input_bindings",
        verify_files=True,
    )


def _write_terminal(
    attempt_root: Path, value: Mapping[str, Any]
) -> dict[str, Any] | None:
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        return None
    return _write_json_exclusive(attempt_root / "terminal_supervision.json", value)


def supervise(
    authority_path: Path,
    *,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authority, authority_binding, _plan, plan_binding, sources = (
        load_and_validate_authority(
            authority_path,
            expected_byte_count=expected_authority_byte_count,
            expected_sha256=expected_authority_sha256,
        )
    )
    attempt_root = _require_fresh_attempt_root(authority["output_root"])
    invocation = str(authority["runtime"]["python_invocation_path"])
    child_env = _child_environment(authority["runtime"])
    wall_cap = float(authority["caps"]["maximum_wall_seconds"])
    gpu_cap = float(authority["caps"]["maximum_gpu_seconds"])
    wall_started = time.monotonic()
    supervisor_nonce = secrets.token_hex(32)
    worker_argv = [
        invocation,
        str((REPO_ROOT / WORKER_RELATIVE).resolve()),
        "--authority",
        str(authority_path.resolve()),
        "--expected-authority-byte-count",
        str(authority_binding["byte_count"]),
        "--expected-authority-sha256",
        str(authority_binding["file_sha256"]),
    ]
    checker_command_template = [
        invocation,
        str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
        "--manifest",
        str((attempt_root / "result.json").resolve()),
        "--expected-file-sha256",
        "<WORKER_RESULT_SHA256>",
        "--expected-byte-count",
        "<WORKER_RESULT_BYTE_COUNT>",
        "--output",
        str((attempt_root / "receipt_check.json").resolve()),
    ]
    _require_idle_authorized_device()
    reservation, reservation_binding = _reserve_attempt(
        attempt_root,
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        worker_binding=sources["worker"],
        checker_binding=sources["checker"],
        worker_command=worker_argv,
        checker_command_template=checker_command_template,
        supervisor_nonce=supervisor_nonce,
    )
    phases: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    result_binding: dict[str, Any] | None = None
    check_binding: dict[str, Any] | None = None
    failure: str | None = None
    try:
        phases.append(
            _run_once(
                worker_argv,
                timeout=_remaining_wall(
                    wall_started=wall_started,
                    wall_cap=min(wall_cap, gpu_cap),
                ),
                env=child_env,
            )
        )
        result, result_binding = _load_result_if_present(attempt_root)
        if result is None or result_binding is None:
            _fail("worker completed without result.json")
        _validate_worker_result(
            result,
            authority=authority,
            authority_binding=authority_binding,
            plan_binding=plan_binding,
            reservation_binding=reservation_binding,
            supervisor_nonce=supervisor_nonce,
        )
        if file_binding(authority_path) != authority_binding:
            _fail("authority changed during worker execution")
        _reverify_contract(authority)
        remaining = _remaining_wall(wall_started=wall_started, wall_cap=wall_cap)
        checker_argv = [
            invocation,
            str((REPO_ROOT / CHECKER_RELATIVE).resolve()),
            "--manifest",
            str((attempt_root / "result.json").resolve()),
            "--expected-file-sha256",
            str(result_binding["file_sha256"]),
            "--expected-byte-count",
            str(result_binding["byte_count"]),
            "--output",
            str((attempt_root / "receipt_check.json").resolve()),
        ]
        phases.append(_run_once(checker_argv, timeout=remaining, env=child_env))
        check_path = attempt_root / "receipt_check.json"
        check_binding = file_binding(check_path)
        check_raw = check_path.read_bytes()
        if file_binding(check_path) != check_binding:
            _fail("receipt check changed while being parsed")
        check = strict_json_bytes(check_raw, label="receipt check")
        if (
            type(check) is not dict
            or check.get("schema") != CHECK_SCHEMA
            or check.get("status") != "PASS"
            or check.get("manifest_binding") != result_binding
            or check.get("predecessor_terminal_failure_binding")
            != authority["predecessor_terminal_failure_binding"]
            or check.get("pack_payloads_opened") is not False
            or check.get("input_data_opened") is not False
            or check.get("runtime_payloads_opened") is not False
            or check.get("rgb_bytes_opened") is not False
            or check.get("checkpoints_opened") is not False
            or check.get("sealed_material_opened") is not False
        ):
            _fail("receipt-only checker did not exactly pass")
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"
        if result_binding is None:
            try:
                result, result_binding = _load_result_if_present(attempt_root)
            except BaseException as result_exc:
                failure += (
                    "; result receipt load failed: "
                    f"{type(result_exc).__name__}: {result_exc}"
                )

    if not _reservation_unchanged(
        attempt_root / "reservation.json",
        reservation=reservation,
        reservation_binding=reservation_binding,
    ):
        changed = "ThreeArmSupervisionError: reservation changed after consumption"
        failure = changed if failure is None else f"{failure}; {changed}"

    blocked = {signal.SIGINT, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        wall_elapsed = time.monotonic() - wall_started
        if failure is None and wall_elapsed > wall_cap:
            failure = (
                "ThreeArmSupervisionError: terminal validation exceeded hard wall "
                f"ceiling ({wall_elapsed:.6f} > {wall_cap:.6f} seconds)"
            )
        terminal = {
            "schema": TERMINAL_SCHEMA,
            "status": (
                RESULT_STATUS if failure is None else "CONSUMED_TERMINAL_FAILURE"
            ),
            "citable_as_scientific_evidence": False,
            "scientific_verdict_emitted": False,
            "authorizes_retry_or_resume": False,
            "authority_binding": authority_binding,
            "plan_binding": plan_binding,
            "review_binding": authority["review_binding"],
            "source_commit": authority["source_commit"],
            "predecessor_terminal_failure_binding": authority[
                "predecessor_terminal_failure_binding"
            ],
            "execution_head": _git_head(),
            "attempt_root": str(attempt_root.resolve()),
            "reservation_binding": reservation_binding,
            "result_binding": result_binding,
            "receipt_check_binding": check_binding,
            "phase_receipts": phases,
            "wall_elapsed_seconds": wall_elapsed,
            "wall_ceiling_seconds": wall_cap,
            "gpu_ceiling_seconds": float(
                authority["caps"]["maximum_gpu_seconds"]
            ),
            "failure": failure,
            "terminal_reviewer": authority["external_supervisor"]["terminal_reviewer"],
            "automatic_checkpoint_selection_performed": False,
            "retry_authorized": False,
            "resume_authorized": False,
            "overwrite_authorized": False,
            "refill_authorized": False,
            "supervisor_nonce": supervisor_nonce,
        }
        terminal_binding = _write_terminal(attempt_root, terminal)
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    return terminal, terminal_binding


def _raise_on_termination_signal(signum: int, _frame: Any) -> None:
    raise ThreeArmSupervisionError(f"supervisor received signal {signum}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.expected_authority_byte_count < 1:
        parser.error("authority byte count must be positive")
    if not _is_sha256(args.expected_authority_sha256):
        parser.error("authority SHA-256 must be lowercase hexadecimal")
    signal.signal(signal.SIGINT, _raise_on_termination_signal)
    signal.signal(signal.SIGTERM, _raise_on_termination_signal)
    terminal, terminal_binding = supervise(
        args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
    )
    if terminal_binding is None:
        print("pre-reservation failure; no attempt consumed", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "status": terminal["status"],
                "terminal_supervision": terminal_binding,
            },
            sort_keys=True,
        )
    )
    return 0 if terminal["status"] == RESULT_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
