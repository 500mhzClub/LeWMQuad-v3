#!/usr/bin/env python3
"""Collect one-shot synchronized physical counterfactuals for Go2 pilot V1.

The CLI has no scientific knobs.  All scene identities, actions, histories,
runtime files, thresholds, devices, and output paths come from an exact
authority-bound plan.  Genesis and Torch remain lazy runtime dependencies.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import copy
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _package_root in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402


POLICY_STEP_COUNT = pilot.BLOCK_SIZE * 5
LIVE_RENDER_RECEIPT_SCHEMA = (
    "lewm_go2_world_model_counterfactual_live_render_receipt_v1"
)
COUNTERFACTUAL_QUALITY_SCHEMA = (
    "lewm_go2_world_model_counterfactual_frame_quality_v2"
)
COUNTERFACTUAL_LOW_INFO_REASON_NAMES = frozenset({
    "low_rgb_texture",
    "near_wall_depth",
    "near_forward_geometry",
})

STAGE_WALL_TIME_KEYS = (
    "native_render_wall_seconds",
    "camera_quality_resize_wall_seconds",
    "png_encode_write_hash_wall_seconds",
)

_SANITIZED_SELECTOR_KEYS = {
    "AMD_VULKAN_ICD",
    "CUDA_VISIBLE_DEVICES",
    "DISPLAY",
    "DRI_PRIME",
    "EGL_DEVICE_ID",
    "GS_BACKEND",
    "GS_PARA_LEVEL",
    "HIP_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "MESA_VK_DEVICE_SELECT",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PYTHONHOME",
    "PYTHONINSPECT",
    "PYTHONOPTIMIZE",
    "PYTHONPATH",
    "PYTHONSTARTUP",
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONHASHSEED",
    "PYTHONNOUSERSITE",
    "PYTHONSAFEPATH",
    "PYOPENGL_PLATFORM",
    "ROCR_VISIBLE_DEVICES",
    "VK_DRIVER_FILES",
    "VK_ICD_FILENAMES",
    "WAYLAND_DISPLAY",
}

EXPECTED_SOURCE_PATHS = {
    "lewm_package_init": "lewm/__init__.py",
    "benchmarks_package_init": "lewm/benchmarks/__init__.py",
    "counterfactual_benchmark_support": "lewm/benchmarks/counterfactual.py",
    "collector": "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "contract": "lewm/benchmarks/go2_world_model_counterfactual_pilot_v1.py",
    "datasets_package_init": "lewm/datasets/__init__.py",
    "pilot_consumer": "lewm/datasets/go2_world_model_counterfactual_pilot_v1.py",
    "genesis_package_init": "lewm_genesis/lewm_genesis/__init__.py",
    "genesis_batch_renderer": "lewm_genesis/lewm_genesis/batch_renderer.py",
    "genesis_parity_checks": "lewm_genesis/lewm_genesis/parity_checks.py",
    "genesis_render_replay": "lewm_genesis/lewm_genesis/render_replay.py",
    "rollout": "lewm_genesis/lewm_genesis/rollout.py",
    "scene_builder": "lewm_genesis/lewm_genesis/scene_builder.py",
    "scene_loader": "lewm_genesis/lewm_genesis/scene_loader.py",
    "go2_adapter": "lewm_genesis/lewm_genesis/go2_adapter.py",
    "genesis_contract": "lewm_genesis/lewm_genesis/lewm_contract.py",
    "camera_safety": "lewm_genesis/lewm_genesis/camera_safety.py",
    "vision_quality": "lewm_genesis/lewm_genesis/vision_quality.py",
    "textures": "lewm_genesis/lewm_genesis/textures.py",
    "collectors_package_init": "lewm_genesis/lewm_genesis/collectors/__init__.py",
    "collector_base": "lewm_genesis/lewm_genesis/collectors/base.py",
    "collector_frontier": "lewm_genesis/lewm_genesis/collectors/frontier.py",
    "collector_ou_noise": "lewm_genesis/lewm_genesis/collectors/ou_noise.py",
    "collector_primitive_curriculum": (
        "lewm_genesis/lewm_genesis/collectors/primitive_curriculum.py"
    ),
    "collector_recovery": "lewm_genesis/lewm_genesis/collectors/recovery.py",
    "collector_route_teacher": (
        "lewm_genesis/lewm_genesis/collectors/route_teacher.py"
    ),
    "worlds_package_init": "lewm_worlds/lewm_worlds/__init__.py",
    "world_corpus": "lewm_worlds/lewm_worlds/corpus.py",
    "world_exporters_package_init": (
        "lewm_worlds/lewm_worlds/exporters/__init__.py"
    ),
    "world_gazebo_exporter": (
        "lewm_worlds/lewm_worlds/exporters/to_gazebo_sdf.py"
    ),
    "world_splits": "lewm_worlds/lewm_worlds/splits.py",
    "world_families": "lewm_worlds/lewm_worlds/families.py",
    "world_randomization": "lewm_worlds/lewm_worlds/randomization.py",
    "world_manifest": "lewm_worlds/lewm_worlds/manifest.py",
    "world_scene_validation": "lewm_worlds/lewm_worlds/scene_validation.py",
    "world_planning_grid": "lewm_worlds/lewm_worlds/planning_grid.py",
    "world_scene_graph": "lewm_worlds/lewm_worlds/scene_graph.py",
    "world_genesis_exporter": "lewm_worlds/lewm_worlds/exporters/to_genesis.py",
    "world_labels_package_init": "lewm_worlds/lewm_worlds/labels/__init__.py",
    "world_labels_derived": "lewm_worlds/lewm_worlds/labels/derived.py",
    "world_labels_topology": "lewm_worlds/lewm_worlds/labels/topology.py",
    "scene_generator_materializer": "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "smoke_rgb_writer": "scripts/collect_go2_world_model_counterfactual_pilot_v1.py",
    "checker": "scripts/check_go2_world_model_counterfactual_pilot_v1.py",
    "external_supervisor": "scripts/run_go2_world_model_counterfactual_smoke_authorized_v1.py",
}
if EXPECTED_SOURCE_PATHS != dict(pilot.AUTHORITY_SOURCE_PATHS):
    raise RuntimeError("collector and source-only authority path closures disagree")

NON_SMOKE_AUTHORITY_CONTRACTS = {
    "sizing_calibration_only": (
        "lewm_go2_world_model_counterfactual_calibration_execution_authority_v2",
        "AUTHORIZED_ONE_EXACT_160_BRANCH_CALIBRATION_V2_SUCCESSOR",
    ),
}
NON_SMOKE_REQUIRED_SOURCES = frozenset({
    "collector",
    "contract",
    "pilot_consumer",
    "checker",
    "calibration_analyzer",
    "pilot_joiner",
    "external_supervisor",
})
NON_SMOKE_SOURCE_PATHS = {
    "calibration_analyzer": (
        "scripts/analyze_go2_world_model_counterfactual_calibration_v1.py"
    ),
    "external_supervisor": (
        "scripts/run_go2_world_model_counterfactual_calibration_authorized_v1.py"
    ),
    "pilot_joiner": "scripts/join_go2_world_model_counterfactual_pilot_v1.py",
}
CALIBRATION_PREDECESSOR_FAILURE_RELATIVE = Path(
    "docs/lewm_go2_world_model_counterfactual_calibration_v1_terminal_failure_result_2026-08-02.json"
)
CALIBRATION_PREDECESSOR_ATTEMPT_ID = "lewm-go2-wm-counterfactual-calibration-v1"
CALIBRATION_PREDECESSOR_ROOT = str(
    (
        ROOT
        / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v1"
    ).resolve()
)
CALIBRATION_PREDECESSOR_TERMINAL_SHA256 = (
    "c5509f97c1d1cca27b7f283187ce7bf644579c4caa03eb1ccfcfda9c18e58315"
)
CALIBRATION_PREDECESSOR_PHYSICS_SHA256 = (
    "34ba69825322e34ebec0ccbab5f1a21fdd4ac60f99cc4fe5f70b158a7aaaaaa3"
)
CALIBRATION_SUCCESSOR_ATTEMPT_ID = "lewm-go2-wm-counterfactual-calibration-v2"
CALIBRATION_SUCCESSOR_ROOT = str(
    (
        ROOT
        / ".generated/dev/lewm-go2-wm-counterfactual-calibration-v2"
    ).resolve()
)


def _is_within_calibration_predecessor_root(path: str | Path) -> bool:
    candidate = Path(path).resolve(strict=False)
    predecessor_root = Path(CALIBRATION_PREDECESSOR_ROOT).resolve(strict=False)
    try:
        candidate.relative_to(predecessor_root)
    except ValueError:
        return False
    return True


def _as_numpy(value: Any) -> np.ndarray:
    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_method = getattr(value, "numpy", None)
    if callable(numpy_method):
        value = numpy_method()
    return np.asarray(value)


def _counterfactual_quality_disposition(
    raw_quality: Mapping[str, Any],
) -> dict[str, Any]:
    """Separate retained low-information observations from hard corruption.

    Low-information reasons describe useful, stratifiable navigation states and
    are not technical corruption.  Every other assessor reason, including an
    unresolved camera-safety condition or malformed/missing/non-finite arrays,
    remains a hard failure.
    """

    if not isinstance(raw_quality, Mapping) or set(raw_quality) != {
        "valid",
        "invalid_reasons",
        "rgb_stats",
        "depth_stats",
    }:
        raise pilot.PilotContractError("raw rendered-frame quality shape changed")
    if (
        type(raw_quality["valid"]) is not bool
        or not isinstance(raw_quality["rgb_stats"], Mapping)
        or not isinstance(raw_quality["depth_stats"], Mapping)
    ):
        raise pilot.PilotContractError("raw rendered-frame quality is malformed")
    reasons = raw_quality["invalid_reasons"]
    if (
        not isinstance(reasons, list)
        or any(not isinstance(reason, str) or not reason for reason in reasons)
        or len(reasons) != len(set(reasons))
    ):
        raise pilot.PilotContractError("raw rendered-frame reasons are invalid")
    low_info = [
        reason for reason in reasons if reason in COUNTERFACTUAL_LOW_INFO_REASON_NAMES
    ]
    hard_failures = [
        reason for reason in reasons if reason not in COUNTERFACTUAL_LOW_INFO_REASON_NAMES
    ]
    if bool(raw_quality["valid"]) != (not reasons):
        raise pilot.PilotContractError("raw rendered-frame validity disagrees with reasons")
    retained = not hard_failures
    return {
        "schema": COUNTERFACTUAL_QUALITY_SCHEMA,
        "retained": retained,
        "hard_valid": retained,
        "raw_assessment_valid": bool(raw_quality["valid"]),
        "observed_reasons": list(reasons),
        "low_information": bool(low_info),
        "low_info_reasons": low_info,
        "hard_failure_reasons": hard_failures,
        "rgb_stats": copy.deepcopy(raw_quality["rgb_stats"]),
        "depth_stats": copy.deepcopy(raw_quality["depth_stats"]),
    }


def _expected_authority_caps(plan: Mapping[str, Any]) -> dict[str, Any]:
    counts = plan["expected_counts"]
    total_lanes = int(counts["total_branches"])
    return {
        "scenes": int(counts["scenes"]),
        "states": int(counts["states"]),
        "candidate_branches": int(counts["candidate_branches"]),
        "sentinel_branches": int(counts["sentinel_branches"]),
        "total_branches": total_lanes,
        "candidate_branch_simulated_seconds": total_lanes * 0.5,
        "total_lane_simulated_seconds_including_common_prefix": total_lanes * 1.5,
        "policy_steps_per_lane": 75,
        "total_lane_policy_steps": total_lanes * 75,
        "total_lane_physics_steps": total_lanes * 750,
        "native_render_calls": int(counts["context_frames"] + counts["target_frames"]),
        "stored_rgb_frames": int(counts["context_frames"] + counts["target_frames"]),
    }


def _validate_calibration_predecessor_failure(
    binding: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    reviewed_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the consumed V1 failure before accepting a fresh V2 successor."""

    normalized_binding = pilot._validate_binding_shape(  # noqa: SLF001
        binding, label="calibration predecessor terminal-failure result"
    )
    expected_path = (ROOT / CALIBRATION_PREDECESSOR_FAILURE_RELATIVE).resolve()
    if Path(str(normalized_binding["path"])).resolve() != expected_path:
        raise pilot.PilotContractError(
            "calibration predecessor terminal-failure result path changed"
        )
    normalized_reviewed_binding = pilot._validate_binding_shape(  # noqa: SLF001
        reviewed_binding, label="reviewed calibration predecessor failure"
    )
    if normalized_binding != normalized_reviewed_binding:
        raise pilot.PilotContractError(
            "calibration predecessor failure is not in the reviewed closure"
        )
    raw, actual_binding = pilot.read_bound_json(
        expected_path,
        expected_sha256=str(normalized_binding["file_sha256"]),
        expected_byte_count=int(normalized_binding["byte_count"]),
        label="calibration predecessor terminal-failure result",
    )
    if actual_binding != normalized_binding or not isinstance(raw, Mapping):
        raise pilot.PilotContractError(
            "calibration predecessor terminal-failure result binding changed"
        )
    if set(raw) != {
        "schema",
        "status",
        "date",
        "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document",
        "attempt",
        "bindings",
        "terminal_evidence",
        "root_cause",
        "successor_boundary",
    }:
        raise pilot.PilotContractError(
            "calibration predecessor terminal-failure result fields changed"
        )
    attempt = raw["attempt"]
    if (
        raw["schema"]
        != "lewm_go2_world_model_counterfactual_calibration_terminal_failure_result_v1"
        or raw["status"]
        != "PASS_CONSUMED_TERMINAL_FAILURE_AUDIT_NO_RETRY_NO_RESUME"
        or raw["authority_granted_by_this_document"] is not False
        or raw["scientific_claim_granted_by_this_document"] is not False
        or not isinstance(attempt, Mapping)
        or set(attempt)
        != {
            "attempt_id",
            "output_root",
            "attempt_consumed",
            "retry_authorized",
            "resume_authorized",
            "refill_authorized",
            "overwrite_authorized",
            "artifact_reuse_for_successor_authorized",
        }
        or attempt["attempt_id"] != CALIBRATION_PREDECESSOR_ATTEMPT_ID
        or attempt["output_root"] != CALIBRATION_PREDECESSOR_ROOT
        or attempt["attempt_consumed"] is not True
        or any(
            attempt[name] is not False
            for name in (
                "retry_authorized",
                "resume_authorized",
                "refill_authorized",
                "overwrite_authorized",
                "artifact_reuse_for_successor_authorized",
            )
        )
    ):
        raise pilot.PilotContractError(
            "calibration predecessor attempt was not an exact consumed failure"
        )
    bindings = raw["bindings"]
    terminal_evidence = raw["terminal_evidence"]
    successor = raw["successor_boundary"]
    if (
        not isinstance(bindings, Mapping)
        or not isinstance(bindings.get("terminal_supervision"), Mapping)
        or not isinstance(bindings.get("physics_result"), Mapping)
        or bindings["terminal_supervision"].get("file_sha256")
        != CALIBRATION_PREDECESSOR_TERMINAL_SHA256
        or bindings["terminal_supervision"].get("byte_count") != 3217
        or bindings["physics_result"].get("file_sha256")
        != CALIBRATION_PREDECESSOR_PHYSICS_SHA256
        or bindings["physics_result"].get("byte_count") != 25773
        or not isinstance(terminal_evidence, Mapping)
        or terminal_evidence.get("terminal_status") != "CONSUMED_TERMINAL_FAILURE"
        or terminal_evidence.get("terminal_physics_result_binding_was_null") is not True
        or terminal_evidence.get("physics_result_status") != "FAILED"
        or not isinstance(successor, Mapping)
        or successor.get("v1_retry_authorized") is not False
        or successor.get("v1_resume_authorized") is not False
        or successor.get("v1_refill_authorized") is not False
        or successor.get("v1_root_or_artifact_reuse_authorized") is not False
        or successor.get("fresh_v2_successor_authorized_by_this_result") is not False
    ):
        raise pilot.PilotContractError(
            "calibration predecessor failure evidence or successor boundary changed"
        )
    if (
        plan["attempt_id"] != CALIBRATION_SUCCESSOR_ATTEMPT_ID
        or Path(str(plan["output_root"])).resolve(strict=False)
        != Path(CALIBRATION_SUCCESSOR_ROOT).resolve(strict=False)
    ):
        raise pilot.PilotContractError(
            "calibration authority does not bind the exact V2 successor identity"
        )
    if (
        plan["attempt_id"] == CALIBRATION_PREDECESSOR_ATTEMPT_ID
        or _is_within_calibration_predecessor_root(plan["output_root"])
    ):
        raise pilot.PilotContractError(
            "calibration V2 successor cannot retry, resume, refill, or reuse V1"
        )
    successor_input_bindings: list[tuple[str, Mapping[str, Any]]] = [
        (f"runtime {name}", runtime_binding)
        for name, runtime_binding in plan["runtime_bindings"].items()
    ]
    for state in plan["states"]:
        if state["scene_generation"] is not None:
            raise pilot.PilotContractError(
                "calibration V2 successor cannot reuse generated V1 scene artifacts"
            )
        successor_input_bindings.extend((
            (
                f"state {state['state_id']} scene manifest",
                state["scene_manifest_binding"],
            ),
            (
                f"state {state['state_id']} Genesis scene",
                state["scene_genesis_binding"],
            ),
        ))
    for label, input_binding in successor_input_bindings:
        normalized_input = pilot._validate_binding_shape(  # noqa: SLF001
            input_binding, label=label
        )
        if _is_within_calibration_predecessor_root(normalized_input["path"]):
            raise pilot.PilotContractError(
                "calibration V2 successor cannot reuse V1 root artifacts"
            )
    return normalized_binding


def _validate_non_smoke_authority(
    authority: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate exact calibration authority without weakening smoke V1."""

    purpose = str(plan["purpose"])
    if purpose not in NON_SMOKE_AUTHORITY_CONTRACTS:
        raise pilot.PilotContractError(
            "this source authorizes calibration only; bounded pilot needs a "
            "separate reviewed supervisor and authority contract"
        )
    expected_schema, expected_status = NON_SMOKE_AUTHORITY_CONTRACTS[purpose]
    required = {
        "schema",
        "status",
        "authority_granted_by_this_document",
        "scientific_claim_authorized",
        "authorizer",
        "issued_at",
        "source_commit",
        "review_binding",
        "plan_binding",
        "predecessor_failure_binding",
        "source_bindings",
        "attempt",
        "caps",
        "runtime_bindings",
        "execution",
        "network_access",
        "external_supervisor",
        "platform_gate_disposition",
    }
    if not isinstance(authority, Mapping) or set(authority) != required:
        raise pilot.PilotContractError("non-smoke execution authority fields changed")
    if (
        authority["schema"] != expected_schema
        or authority["status"] != expected_status
        or authority["authority_granted_by_this_document"] is not True
        or authority["scientific_claim_authorized"] is not False
    ):
        raise pilot.PilotContractError("authority does not grant this exact plan purpose")
    authorizer = authority["authorizer"]
    if (
        not isinstance(authorizer, Mapping)
        or set(authorizer) != {"identity", "basis"}
        or not isinstance(authorizer["identity"], str)
        or not authorizer["identity"].strip()
        or not isinstance(authorizer["basis"], str)
        or not authorizer["basis"].strip()
    ):
        raise pilot.PilotContractError("authority authorizer is invalid")
    try:
        datetime.fromisoformat(str(authority["issued_at"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise pilot.PilotContractError("authority issued_at is not ISO-8601") from error
    source_commit = authority["source_commit"]
    if not isinstance(source_commit, str) or re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        raise pilot.PilotContractError("authority source_commit is invalid")
    normalized_plan_binding = pilot._validate_binding_shape(  # noqa: SLF001
        authority["plan_binding"], label="authority plan binding"
    )
    if _is_within_calibration_predecessor_root(normalized_plan_binding["path"]):
        raise pilot.PilotContractError(
            "calibration V2 successor plan cannot reuse the V1 root"
        )
    if normalized_plan_binding != dict(plan_binding):
        raise pilot.PilotContractError("authority does not bind the selected plan")
    review_binding = pilot._validate_binding_shape(  # noqa: SLF001
        authority["review_binding"], label="authority review binding"
    )
    raw_sources = authority["source_bindings"]
    if not isinstance(raw_sources, list) or not raw_sources:
        raise pilot.PilotContractError("authority source bindings are absent")
    sources: list[dict[str, Any]] = []
    source_names: set[str] = set()
    for source in raw_sources:
        if not isinstance(source, Mapping) or set(source) != {"name", "binding"}:
            raise pilot.PilotContractError("authority source binding changed")
        name = source["name"]
        if not isinstance(name, str) or not name or name in source_names:
            raise pilot.PilotContractError("authority source name repeats")
        source_names.add(name)
        binding = pilot._validate_binding_shape(  # noqa: SLF001
            source["binding"], label=f"authority source {name}"
        )
        expected_relative = {
            **EXPECTED_SOURCE_PATHS,
            **NON_SMOKE_SOURCE_PATHS,
        }.get(name)
        if expected_relative is not None and Path(binding["path"]).resolve() != (
            ROOT / expected_relative
        ).resolve():
            raise pilot.PilotContractError(f"authority source {name} path changed")
        sources.append({"name": name, "binding": binding})
    if not NON_SMOKE_REQUIRED_SOURCES.issubset(source_names):
        raise pilot.PilotContractError("non-smoke authority source closure is incomplete")
    predecessor_rows = [
        source["binding"]
        for source in sources
        if source["name"] == "predecessor_terminal_failure_result"
    ]
    if len(predecessor_rows) != 1:
        raise pilot.PilotContractError(
            "reviewed calibration predecessor failure binding is absent"
        )
    predecessor_failure_binding = _validate_calibration_predecessor_failure(
        authority["predecessor_failure_binding"],
        plan=plan,
        reviewed_binding=predecessor_rows[0],
    )
    attempt = authority["attempt"]
    expected_attempt = {
        "id": plan["attempt_id"],
        "root": plan["output_root"],
        "maximum_attempts": 1,
        "must_be_absent": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "refill": False,
    }
    if pilot.canonical_json_bytes(attempt) != pilot.canonical_json_bytes(expected_attempt):
        raise pilot.PilotContractError("authority attempt boundary changed")
    caps = authority["caps"]
    expected_caps = _expected_authority_caps(plan)
    if (
        not isinstance(caps, Mapping)
        or set(caps) != {*expected_caps, "wall_seconds"}
        or any(caps.get(name) != expected for name, expected in expected_caps.items())
        or isinstance(caps.get("wall_seconds"), bool)
        or not isinstance(caps.get("wall_seconds"), (int, float))
        or not math.isfinite(float(caps["wall_seconds"]))
        or float(caps["wall_seconds"]) <= 0.0
    ):
        raise pilot.PilotContractError("authority work caps changed from the exact plan")
    if pilot.canonical_json_bytes(authority["runtime_bindings"]) != pilot.canonical_json_bytes(
        plan["runtime_bindings"]
    ) or pilot.canonical_json_bytes(authority["execution"]) != pilot.canonical_json_bytes(
        plan["execution_contract"]
    ):
        raise pilot.PilotContractError("authority runtime or execution binding changed")
    if authority["network_access"] is not False:
        raise pilot.PilotContractError("counterfactual collection must forbid network access")
    disposition = authority["platform_gate_disposition"]
    if (
        not isinstance(disposition, Mapping)
        or set(disposition) != {
            "platform_hard_gates_resolved",
            "scope",
            "outputs_eligible_for_training_after_receipt_join",
            "outputs_eligible_for_scientific_claim",
            "authorizes_this_exact_generation",
            "authorizes_promotion",
            "basis",
        }
        or disposition["platform_hard_gates_resolved"] is not True
        or disposition["scope"] != purpose
        or disposition["outputs_eligible_for_training_after_receipt_join"]
        is not (purpose == "bounded_wm_a_pilot")
        or disposition["outputs_eligible_for_scientific_claim"] is not False
        or disposition["authorizes_this_exact_generation"] is not True
        or disposition["authorizes_promotion"] is not False
        or not isinstance(disposition["basis"], str)
        or not disposition["basis"].strip()
    ):
        raise pilot.PilotContractError("non-smoke platform-gate disposition changed")
    supervisor = authority["external_supervisor"]
    if (
        not isinstance(supervisor, Mapping)
        or set(supervisor) != {"source_binding", "terminal_reviewer"}
        or not isinstance(supervisor["terminal_reviewer"], str)
        or not supervisor["terminal_reviewer"].strip()
    ):
        raise pilot.PilotContractError("authority external supervisor changed")
    supervisor_binding = pilot._validate_binding_shape(  # noqa: SLF001
        supervisor["source_binding"], label="external supervisor"
    )
    reviewed_supervisor = next(
        source["binding"] for source in sources if source["name"] == "external_supervisor"
    )
    if supervisor_binding != reviewed_supervisor:
        raise pilot.PilotContractError("external supervisor is not in the reviewed closure")
    normalized = dict(authority)
    normalized["plan_binding"] = normalized_plan_binding
    normalized["review_binding"] = review_binding
    normalized["predecessor_failure_binding"] = predecessor_failure_binding
    normalized["source_bindings"] = sources
    normalized["external_supervisor"] = {
        "source_binding": supervisor_binding,
        "terminal_reviewer": supervisor["terminal_reviewer"],
    }
    return normalized


def _validate_authority_for_plan(
    authority: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if plan["purpose"] == "source_integration_smoke":
        return pilot.validate_authority(
            authority, plan=plan, plan_binding=plan_binding
        )
    return _validate_non_smoke_authority(
        authority, plan=plan, plan_binding=plan_binding
    )


def _git_output(*args: str, binary: bool = False) -> bytes | str:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    return completed.stdout if binary else completed.stdout.strip()


def _repo_relative(path: str | Path, *, label: str) -> str:
    resolved = Path(path).resolve(strict=True)
    try:
        relative = resolved.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise pilot.PilotContractError(f"{label} must be inside repository") from exc
    return relative.as_posix()


def _binding_at_commit(
    binding: Mapping[str, Any], *, commit: str, label: str
) -> None:
    relative = _repo_relative(binding["path"], label=label)
    try:
        payload = _git_output("show", f"{commit}:{relative}", binary=True)
    except subprocess.CalledProcessError as exc:
        raise pilot.PilotContractError(
            f"{label} is absent from commit {commit}"
        ) from exc
    assert isinstance(payload, bytes)
    if (
        len(payload) != int(binding["byte_count"])
        or hashlib.sha256(payload).hexdigest() != binding["file_sha256"]
    ):
        raise pilot.PilotContractError(f"{label} differs from commit {commit}")


def _validate_git_authority_boundary(
    *,
    plan_binding: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> None:
    head = str(_git_output("rev-parse", "HEAD"))
    source_commit = str(authority["source_commit"])
    try:
        _git_output("merge-base", "--is-ancestor", source_commit, head)
    except subprocess.CalledProcessError as exc:
        raise pilot.PilotContractError(
            "authority source_commit is not an ancestor of HEAD"
        ) from exc
    _binding_at_commit(plan_binding, commit=head, label="pilot plan")
    _binding_at_commit(authority_binding, commit=head, label="execution authority")
    _binding_at_commit(
        authority["review_binding"], commit=head, label="source review"
    )
    sources = authority["source_bindings"]
    if authority["schema"] == pilot.SMOKE_AUTHORITY_SCHEMA:
        expected_names: Sequence[str] = pilot.AUTHORITY_SOURCE_NAMES
    else:
        expected_names = tuple(str(source["name"]) for source in sources)
    for expected_name, source in zip(expected_names, sources, strict=True):
        if source["name"] != expected_name:
            raise pilot.PilotContractError("authority source order changed")
        relative = _repo_relative(
            source["binding"]["path"], label=f"source {expected_name}"
        )
        exact_source_paths = {**EXPECTED_SOURCE_PATHS, **NON_SMOKE_SOURCE_PATHS}
        if expected_name in exact_source_paths and relative != exact_source_paths[expected_name]:
            raise pilot.PilotContractError(
                f"source {expected_name} path changed: {relative}"
            )
        pilot.require_binding(
            source["binding"], label=f"authority source {expected_name}"
        )
        _binding_at_commit(
            source["binding"],
            commit=source_commit,
            label=f"source {expected_name}",
        )
    supervisor_binding = authority["external_supervisor"]["source_binding"]
    pilot.require_binding(supervisor_binding, label="external supervisor source")
    _binding_at_commit(
        supervisor_binding, commit=head, label="external supervisor source"
    )


def _validate_python_runtime(plan: Mapping[str, Any]) -> None:
    invocation = Path(plan["execution_contract"]["python_invocation_path"])
    target = Path(plan["runtime_bindings"]["python_executable_target"]["path"])
    environment_config = Path(
        plan["runtime_bindings"]["python_environment_config"]["path"]
    )
    if invocation.absolute() != Path(sys.executable).absolute():
        raise pilot.PilotContractError("collector Python invocation path changed")
    if invocation.resolve(strict=True) != target.resolve(strict=True):
        raise pilot.PilotContractError("Python invocation does not resolve to bound target")
    if environment_config.name != "pyvenv.cfg" or invocation.parent.parent != environment_config.parent:
        raise pilot.PilotContractError("Python invocation and pyvenv.cfg disagree")


def _validate_execution_environment(plan: Mapping[str, Any]) -> None:
    expected = dict(plan["execution_contract"]["environment"])
    if expected != pilot.EXECUTION_ENVIRONMENT:
        raise pilot.PilotContractError("execution environment contract changed")
    for key in sorted(_SANITIZED_SELECTOR_KEYS):
        if key in expected:
            if os.environ.get(key) != expected[key]:
                raise pilot.PilotContractError(
                    f"execution environment selector {key} changed"
                )
        elif key in os.environ:
            raise pilot.PilotContractError(
                f"forbidden inherited execution selector remains set: {key}"
            )


def _copy_exact_plan_receipt(
    plan_binding: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    """Persist the external authorized plan as a receipt-local exact witness."""

    source = Path(str(plan_binding["path"]))
    actual = pilot.require_binding(plan_binding, label="authorized external plan")
    raw = source.read_bytes()
    if (
        len(raw) != int(actual["byte_count"])
        or hashlib.sha256(raw).hexdigest() != str(actual["file_sha256"])
    ):
        raise pilot.PilotContractError("authorized plan changed while copied")
    destination = output_root / "authorized_plan.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
    copied = pilot.file_binding(destination)
    if (
        copied["byte_count"] != actual["byte_count"]
        or copied["file_sha256"] != actual["file_sha256"]
    ):
        raise pilot.PilotContractError("local plan receipt differs from authority")
    return _relative_output_binding(copied, output_root=output_root)


def _capture_components(runner: Any) -> dict[str, np.ndarray]:
    robot = runner.build.robot
    leg_indices = np.asarray(runner._leg_dof_idx, dtype=np.int64)  # noqa: SLF001
    policy_actions = getattr(runner.policy, "_last_actions", None)
    if policy_actions is None:
        raise pilot.PilotContractError("policy action-history state is uninitialized")
    return {
        "qpos": _as_numpy(robot.get_qpos()),
        "dofs_velocity": _as_numpy(robot.get_dofs_velocity()),
        "base_pos_world": _as_numpy(robot.get_pos()),
        "base_quat_wxyz": _as_numpy(robot.get_quat()),
        "base_lin_vel_world": _as_numpy(robot.get_vel()),
        "base_ang_vel_world": _as_numpy(robot.get_ang()),
        "leg_joint_pos": _as_numpy(robot.get_dofs_position(leg_indices.tolist())),
        "leg_joint_vel": _as_numpy(robot.get_dofs_velocity(leg_indices.tolist())),
        "runner_last_executed": np.asarray(
            runner._last_executed, dtype=np.float32  # noqa: SLF001
        ),
        "policy_last_actions": np.asarray(policy_actions, dtype=np.float32),
    }


def _initialize_exact_clones(runner: Any) -> None:
    """Put every env at the one manifest spawn without randomization or settle."""

    n_envs = int(runner.n_envs)
    envs = list(range(n_envs))
    pack = runner.pack
    robot = runner.build.robot
    positions = np.tile(
        np.asarray(pack.robot.spawn_xyz_m, dtype=np.float32), (n_envs, 1)
    )
    quaternions = np.tile(
        np.asarray(pack.robot.spawn_quat_wxyz, dtype=np.float32), (n_envs, 1)
    )
    stance = np.asarray(
        getattr(runner.policy, "reset_stance_rad", runner._stance),  # noqa: SLF001
        dtype=np.float32,
    )
    stances = np.tile(stance, (n_envs, 1))
    leg_indices = np.asarray(runner._leg_dof_idx, dtype=np.int64)  # noqa: SLF001
    robot.set_pos(positions, envs_idx=envs, zero_velocity=True)
    robot.set_quat(quaternions, envs_idx=envs, zero_velocity=False)
    robot.set_dofs_position(stances, leg_indices.tolist(), envs_idx=envs)
    robot.set_dofs_velocity(
        np.zeros_like(stances), leg_indices.tolist(), envs_idx=envs
    )
    action_width = len(runner.policy.policy_joint_names)
    runner.policy._last_actions = np.zeros(  # noqa: SLF001
        (n_envs, action_width), dtype=np.float32
    )
    runner._last_executed.fill(0.0)  # noqa: SLF001
    runner._sim_time_ns = 0  # noqa: SLF001


def _extract_render_arrays(rendered: Any) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(rendered, tuple) or len(rendered) < 2:
        raise pilot.PilotContractError("native render did not return RGB and depth")
    rgb = _as_numpy(rendered[0])
    depth = _as_numpy(rendered[1])
    if rgb.ndim == 3:
        rgb = rgb[None, ...]
    if depth.ndim == 2:
        depth = depth[None, ...]
    return rgb, depth


def _capture_replayed_frame(
    render_build: Any,
    *,
    components: Mapping[str, np.ndarray],
    env_index: int,
    safe_camera_pose_from_base: Any,
    camera_safety_config_from_pack: Any,
    effective_camera_mount_xyz_rpy: Any,
    assess_rendered_frame: Any,
    stage_wall_times: dict[str, float],
) -> dict[str, Any]:
    """Render one captured physical pose in the isolated one-env render scene."""

    from PIL import Image

    processing_started = time.perf_counter()
    positions = np.asarray(components["base_pos_world"], dtype=np.float32)
    quats_wxyz = np.asarray(components["base_quat_wxyz"], dtype=np.float32)
    if (
        positions.ndim != 2
        or positions.shape[1:] != (3,)
        or quats_wxyz.ndim != 2
        or quats_wxyz.shape != (positions.shape[0], 4)
        or isinstance(env_index, bool)
        or not isinstance(env_index, int)
        or env_index < 0
        or env_index >= positions.shape[0]
    ):
        raise pilot.PilotContractError("captured physical replay pose shape changed")
    if int(getattr(render_build, "n_envs", -1)) != 1:
        raise pilot.PilotContractError("render replay scene must contain exactly one env")
    if bool(getattr(render_build.camera, "_is_batched", False)):
        raise pilot.PilotContractError("render replay camera must be non-batched")
    if not np.all(np.isfinite(positions[env_index])) or not np.all(
        np.isfinite(quats_wxyz[env_index])
    ):
        raise pilot.PilotContractError("captured physical replay pose is nonfinite")
    quat_wxyz = quats_wxyz[env_index]
    quat_xyzw = quat_wxyz[[1, 2, 3, 0]]
    mount_xyz, mount_rpy = effective_camera_mount_xyz_rpy(render_build.pack)
    camera_config = camera_safety_config_from_pack(render_build.pack)
    pose, safety = safe_camera_pose_from_base(
        positions[env_index],
        quat_xyzw,
        mount_xyz_body=mount_xyz,
        mount_rpy_body=mount_rpy,
        objects=render_build.pack.static_objects,
        config=camera_config,
    )
    render_build.camera.set_pose(
        pos=np.asarray(pose.position, dtype=np.float32),
        lookat=np.asarray(pose.lookat, dtype=np.float32),
        up=np.asarray(pose.up, dtype=np.float32),
    )
    processing_before_render = time.perf_counter() - processing_started
    render_started = time.perf_counter()
    rendered = render_build.camera.render(
        rgb=True,
        depth=True,
        force_render=True,
    )
    native_render_elapsed = time.perf_counter() - render_started
    processing_started = time.perf_counter()
    rgb_native, depth_native = _extract_render_arrays(rendered)
    native_width, native_height = render_build.pack.camera.native_resolution
    stored_width, stored_height = render_build.pack.camera.training_resolution
    expected_rgb = (1, native_height, native_width, 3)
    expected_depth = (1, native_height, native_width)
    if rgb_native.shape != expected_rgb or depth_native.shape != expected_depth:
        raise pilot.PilotContractError(
            "native RGB/depth shape changed: "
            f"rgb={rgb_native.shape} depth={depth_native.shape}"
        )
    rgb = np.asarray(rgb_native[0], dtype=np.uint8)
    depth = np.asarray(depth_native[0])
    quality = assess_rendered_frame(
        rgb,
        depth,
        require_depth=True,
        camera_safety=dict(safety),
    )
    stored_rgb = np.asarray(
        Image.fromarray(rgb).resize(
            (stored_width, stored_height), Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    processing_after_render = time.perf_counter() - processing_started
    stage_wall_times["native_render_wall_seconds"] += native_render_elapsed
    stage_wall_times["camera_quality_resize_wall_seconds"] += (
        processing_before_render + processing_after_render
    )
    return {
        "stored_rgb": stored_rgb,
        "quality": quality,
        "native_resolution": [native_width, native_height],
        "stored_resolution": [stored_width, stored_height],
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": "solid_materials_box_physics_preserved",
        "source_base_position_xyz_m": positions[env_index].copy(),
        "source_base_quaternion_wxyz": quat_wxyz.copy(),
        "camera_pose_world": {
            "position_xyz_m": np.asarray(pose.position, dtype=np.float32).copy(),
            "lookat_xyz_m": np.asarray(pose.lookat, dtype=np.float32).copy(),
            "up_xyz": np.asarray(pose.up, dtype=np.float32).copy(),
        },
    }


def _capture_sequential_render_replay(
    render_build: Any,
    *,
    states: Sequence[Mapping[str, Any]],
    trial: Mapping[str, Any],
    safe_camera_pose_from_base: Any,
    camera_safety_config_from_pack: Any,
    effective_camera_mount_xyz_rpy: Any,
    assess_rendered_frame: Any,
    stage_wall_times: dict[str, float],
) -> dict[str, Any]:
    """Replay three representative contexts and every physical endpoint."""

    if len(trial["history_snapshots"]) != pilot.CONTEXT_FRAME_COUNT:
        raise pilot.PilotContractError("captured context snapshot count changed")
    lane_counts = [
        pilot.lane_count_for_role(str(state["role"])) for state in states
    ]
    lane_starts = np.cumsum([0, *lane_counts[:-1]]).tolist()
    expected_envs = sum(lane_counts)
    endpoint = trial["branch_endpoint"]
    endpoint_positions = np.asarray(endpoint["base_pos_world"])
    if endpoint_positions.ndim != 2 or endpoint_positions.shape[0] != expected_envs:
        raise pilot.PilotContractError("captured branch endpoint lane count changed")
    context_frames: dict[str, list[dict[str, Any]]] = {}
    for group_index, state in enumerate(states):
        state_id = str(state["state_id"])
        representative_lane = int(lane_starts[group_index])
        context_frames[state_id] = [
            _capture_replayed_frame(
                render_build,
                components=snapshot,
                env_index=representative_lane,
                safe_camera_pose_from_base=safe_camera_pose_from_base,
                camera_safety_config_from_pack=camera_safety_config_from_pack,
                effective_camera_mount_xyz_rpy=effective_camera_mount_xyz_rpy,
                assess_rendered_frame=assess_rendered_frame,
                stage_wall_times=stage_wall_times,
            )
            for snapshot in trial["history_snapshots"]
        ]
    branch_frames = [
        _capture_replayed_frame(
            render_build,
            components=endpoint,
            env_index=env_index,
            safe_camera_pose_from_base=safe_camera_pose_from_base,
            camera_safety_config_from_pack=camera_safety_config_from_pack,
            effective_camera_mount_xyz_rpy=effective_camera_mount_xyz_rpy,
            assess_rendered_frame=assess_rendered_frame,
            stage_wall_times=stage_wall_times,
        )
        for env_index in range(expected_envs)
    ]
    native_render_calls = sum(len(rows) for rows in context_frames.values()) + len(
        branch_frames
    )
    return {
        "context_frames": context_frames,
        "branch_frames": branch_frames,
        "native_render_calls": native_render_calls,
    }


def _write_png_exclusive(
    path: Path,
    rgb: np.ndarray,
    *,
    stage_wall_times: dict[str, float],
) -> dict[str, Any]:
    from PIL import Image

    started = time.perf_counter()
    selected = Path(path)
    selected.parent.mkdir(parents=True, exist_ok=True)
    with selected.open("xb") as stream:
        Image.fromarray(np.asarray(rgb, dtype=np.uint8)).save(
            stream,
            format="PNG",
        )
    binding = pilot.file_binding(selected)
    stage_wall_times["png_encode_write_hash_wall_seconds"] += (
        time.perf_counter() - started
    )
    return binding


def _new_stage_wall_times() -> dict[str, float]:
    return {name: 0.0 for name in STAGE_WALL_TIME_KEYS}


def _relative_output_binding(
    binding: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    root = output_root.resolve(strict=True)
    selected = Path(binding["path"]).resolve(strict=True)
    try:
        relative = selected.relative_to(root)
    except ValueError as exc:
        raise pilot.PilotContractError(
            f"output binding escapes receipt root: {selected}"
        ) from exc
    return {
        "path": relative.as_posix(),
        "file_sha256": str(binding["file_sha256"]),
        "byte_count": int(binding["byte_count"]),
    }


def _scene_receipt_binding(
    state: Mapping[str, Any],
    *,
    binding_name: str,
    output_root: Path,
) -> dict[str, Any]:
    """Keep plan inputs absolute; relativize only smoke-generated outputs."""

    if binding_name not in {"scene_manifest_binding", "scene_genesis_binding"}:
        raise pilot.PilotContractError("unsupported scene binding receipt field")
    binding = state[binding_name]
    if state["scene_generation"] is not None:
        return _relative_output_binding(binding, output_root=output_root)
    result = dict(binding)
    if not Path(result["path"]).is_absolute():
        raise pilot.PilotContractError(
            f"plan-supplied {binding_name} must remain an absolute input binding"
        )
    return result


def _capture_runtime_versions() -> dict[str, str]:
    """Capture dependency versions from the authority-selected Python env."""

    import genesis
    import PIL
    import torch

    versions = {
        "python": platform.python_version(),
        "genesis": getattr(genesis, "__version__", None),
        "torch": getattr(torch, "__version__", None),
        "numpy": getattr(np, "__version__", None),
        "pillow": getattr(PIL, "__version__", None),
    }
    expected = {"python", "genesis", "torch", "numpy", "pillow"}
    if set(versions) != expected or any(
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or not value.isprintable()
        for value in versions.values()
    ):
        raise pilot.PilotContractError("runtime version receipt is invalid")
    return versions


def _failure_receipt(exc: Exception) -> dict[str, Any]:
    """Preserve bounded JSON diagnostics without weakening failure semantics."""

    failure: dict[str, Any] = {"type": type(exc).__name__, "message": str(exc)}
    diagnostics = getattr(exc, "diagnostics", None)
    if diagnostics is not None:
        failure["diagnostics"] = json.loads(
            pilot.canonical_json_bytes(diagnostics).decode("utf-8")
        )
    return failure


def _quat_tip_rad(quat_wxyz: Sequence[float]) -> float:
    qw, qx, qy, qz = (float(value) for value in quat_wxyz)
    roll = math.atan2(
        2.0 * (qw * qx + qy * qz),
        1.0 - 2.0 * (qx * qx + qy * qy),
    )
    pitch_arg = max(-1.0, min(1.0, 2.0 * (qw * qy - qz * qx)))
    pitch = math.asin(pitch_arg)
    return max(abs(roll), abs(pitch))


def _json_lane(array: np.ndarray, lane_index: int) -> list[Any]:
    return np.asarray(array[lane_index]).tolist()


def _endpoint_state(
    components: Mapping[str, np.ndarray], lane_index: int
) -> dict[str, Any]:
    return {
        name: _json_lane(np.asarray(components[name]), lane_index)
        for name in pilot.SYNC_COMPONENTS
    }


def _trajectory_rows(
    samples: Sequence[Mapping[str, np.ndarray]],
    timestamps_ns: Sequence[int],
    lane_index: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy_step_index, (sample, timestamp_ns) in enumerate(
        zip(samples, timestamps_ns, strict=True)
    ):
        rows.append(
            {
                "policy_step_index": policy_step_index,
                "timestamp_ns": int(timestamp_ns),
                "base_pos_world": _json_lane(sample["base_pos_world"], lane_index),
                "base_quat_wxyz": _json_lane(
                    sample["base_quat_wxyz"], lane_index
                ),
                "base_lin_vel_world": _json_lane(
                    sample["base_lin_vel_world"], lane_index
                ),
                "base_ang_vel_world": _json_lane(
                    sample["base_ang_vel_world"], lane_index
                ),
                "leg_joint_pos": _json_lane(sample["leg_joint_pos"], lane_index),
                "leg_joint_vel": _json_lane(sample["leg_joint_vel"], lane_index),
            }
        )
    return rows


def _path_length_m(
    prebranch: Mapping[str, np.ndarray],
    samples: Sequence[Mapping[str, np.ndarray]],
    lane_index: int,
) -> float:
    points = [np.asarray(prebranch["base_pos_world"])[lane_index, :2]]
    points.extend(np.asarray(sample["base_pos_world"])[lane_index, :2] for sample in samples)
    return float(
        sum(
            np.linalg.norm(np.asarray(right) - np.asarray(left))
            for left, right in zip(points, points[1:])
        )
    )


def _target_progress_m(
    *,
    state: Mapping[str, Any],
    prebranch: Mapping[str, np.ndarray],
    endpoint: Mapping[str, np.ndarray],
    lane_index: int,
) -> float:
    target_xy = np.asarray(state["target_xy_m"], dtype=np.float64)
    start_xy = np.asarray(prebranch["base_pos_world"])[lane_index, :2]
    end_xy = np.asarray(endpoint["base_pos_world"])[lane_index, :2]
    progress = float(
        np.linalg.norm(start_xy - target_xy) - np.linalg.norm(end_xy - target_xy)
    )
    if not math.isfinite(progress):
        raise pilot.PilotContractError("physical target progress is nonfinite")
    return progress


def _render_frame(
    *,
    frame_index: int,
    frame_identity: str,
    timestamp_ns: int,
    env_index: int,
    components: Mapping[str, np.ndarray],
    state_id: str,
    frame_kind: str,
    action_id: int | None,
) -> dict[str, Any]:
    position = np.asarray(components["base_pos_world"])[env_index]
    qw, qx, qy, qz = (
        float(value)
        for value in np.asarray(components["base_quat_wxyz"])[env_index]
    )
    return {
        "frame_index": int(frame_index),
        "frame_identity": frame_identity,
        "timestamp_ns": int(timestamp_ns),
        "env_index": int(env_index),
        "base_pose_world": {
            "position": {
                "x": float(position[0]),
                "y": float(position[1]),
                "z": float(position[2]),
            }
        },
        "base_quat_world_xyzw": [qx, qy, qz, qw],
        "joint_state": {
            "position": _json_lane(components["leg_joint_pos"], env_index),
            "velocity": _json_lane(components["leg_joint_vel"], env_index),
        },
        "command_context": {
            "command_source": "counterfactual_pilot_v1",
            "state_id": state_id,
            "frame_kind": frame_kind,
            "action_id": action_id,
        },
    }


def _replay_pose_audit(
    replay_frame: Mapping[str, Any],
    *,
    expected_components: Mapping[str, np.ndarray],
    env_index: int,
) -> dict[str, Any]:
    """Bind one rendered frame to the exact captured physical and camera pose."""

    source_position = np.asarray(
        replay_frame["source_base_position_xyz_m"], dtype=np.float32
    )
    source_quaternion = np.asarray(
        replay_frame["source_base_quaternion_wxyz"], dtype=np.float32
    )
    expected_position = np.asarray(
        expected_components["base_pos_world"], dtype=np.float32
    )[env_index]
    expected_quaternion = np.asarray(
        expected_components["base_quat_wxyz"], dtype=np.float32
    )[env_index]
    if not np.array_equal(source_position, expected_position) or not np.array_equal(
        source_quaternion, expected_quaternion
    ):
        raise pilot.PilotContractError(
            "render replay source pose changed from captured physical state"
        )
    camera_pose = replay_frame.get("camera_pose_world")
    if not isinstance(camera_pose, Mapping) or set(camera_pose) != {
        "position_xyz_m",
        "lookat_xyz_m",
        "up_xyz",
    }:
        raise pilot.PilotContractError("render replay camera pose receipt changed")
    normalized_camera: dict[str, list[float]] = {}
    for name in ("position_xyz_m", "lookat_xyz_m", "up_xyz"):
        vector = np.asarray(camera_pose[name], dtype=np.float32)
        if vector.shape != (3,) or not np.all(np.isfinite(vector)):
            raise pilot.PilotContractError(
                f"render replay camera pose {name} is invalid"
            )
        normalized_camera[name] = vector.tolist()
    return {
        "source_base_pose_world": {
            "position_xyz_m": source_position.tolist(),
            "quaternion_wxyz": source_quaternion.tolist(),
        },
        "camera_pose_world": normalized_camera,
    }


def _group_trial_receipts(
    *,
    plan: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    trial: Mapping[str, Any],
    rgb_root: Path,
    stage_wall_times: dict[str, float],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    action_catalog = plan["action_catalog"]
    execution = plan["execution_contract"]
    history_by_state: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for history in trial["history_blocks"]:
        for row in history["states"]:
            history_by_state[str(row["state_id"])].append(row)
    frame_receipts: list[dict[str, Any]] = []
    quality_audits: list[dict[str, Any]] = []
    render_sentinel_audits: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    endpoint_components = trial["branch_endpoint"]
    lane_counts = [pilot.lane_count_for_role(str(state["role"])) for state in states]
    local_lane_starts = np.cumsum([0, *lane_counts[:-1]]).tolist()
    all_lane_counts = [
        pilot.lane_count_for_role(str(state["role"])) for state in plan["states"]
    ]
    global_lane_starts = np.cumsum([0, *all_lane_counts[:-1]]).tolist()
    sentinel_audits = {
        str(audit["state_id"]): audit for audit in trial["sentinel_audits"]
    }
    render_replay = trial.get("render_replay")
    if not isinstance(render_replay, Mapping):
        raise pilot.PilotContractError("trial is missing isolated render replay")
    context_replays = render_replay.get("context_frames")
    branch_replays = render_replay.get("branch_frames")
    if not isinstance(context_replays, Mapping) or not isinstance(
        branch_replays, list
    ):
        raise pilot.PilotContractError("isolated render replay shape changed")
    if len(branch_replays) != sum(lane_counts):
        raise pilot.PilotContractError("isolated branch replay count changed")
    for group_index, state in enumerate(states):
        state_id = str(state["state_id"])
        local_lane_start = int(local_lane_starts[group_index])
        global_lane_start = int(global_lane_starts[int(state["group_index"])])
        lane_delta = global_lane_start - local_lane_start
        synchronization_audit = dict(
            trial["synchronization_audits"][group_index]
        )
        synchronization_audit["group_index"] = int(state["group_index"])
        synchronization_audit["lane_start"] = global_lane_start
        sentinel_audit = sentinel_audits.get(state_id)
        if sentinel_audit is not None:
            sentinel_audit = dict(sentinel_audit)
            sentinel_audit["group_index"] = int(state["group_index"])
            sentinel_audit["candidate_lane"] = int(
                sentinel_audit["candidate_lane"]
            ) + lane_delta
            sentinel_audit["sentinel_lane"] = int(
                sentinel_audit["sentinel_lane"]
            ) + lane_delta
        context_identities: list[str] = []
        context_artifact_ids: list[str] = []
        state_context_replays = context_replays.get(state_id)
        if (
            not isinstance(state_context_replays, list)
            or len(state_context_replays) != pilot.CONTEXT_FRAME_COUNT
        ):
            raise pilot.PilotContractError(
                f"isolated context replay count changed for {state_id}"
            )
        for context_index, replay_frame in enumerate(state_context_replays):
            identity = pilot.render_frame_identity(
                state_id=state_id, frame_kind="context", index=context_index
            )
            context_identities.append(identity)
            artifact_id = identity
            context_artifact_ids.append(artifact_id)
            rgb = np.asarray(replay_frame["stored_rgb"])
            quality = _counterfactual_quality_disposition(
                replay_frame["quality"]
            )
            if quality["retained"] is not True:
                raise pilot.PilotContractError(
                    f"camera-invalid context frame {identity}: "
                    f"{quality['hard_failure_reasons']}"
                )
            binding = _write_png_exclusive(
                rgb_root / f"{state_id}.context.{context_index}.png",
                rgb,
                stage_wall_times=stage_wall_times,
            )
            binding = _relative_output_binding(
                binding, output_root=Path(plan["output_root"])
            )
            frame_receipts.append(
                {
                    "artifact_id": artifact_id,
                    "frame_identity": identity,
                    **binding,
                    "width": 224,
                    "height": 224,
                    "mode": "RGB",
                    "format": "PNG",
                    "camera_valid": True,
                    "low_information": quality["low_information"],
                    "low_info_reasons": list(quality["low_info_reasons"]),
                }
            )
            quality_audits.append(
                {
                    "frame_identity": identity,
                    "native_resolution": list(replay_frame["native_resolution"]),
                    "camera_valid": True,
                    "quality": quality,
                    "replay_pose": _replay_pose_audit(
                        replay_frame,
                        expected_components=trial["history_snapshots"][context_index],
                        env_index=local_lane_start,
                    ),
                }
            )
        state_histories = history_by_state[state_id]
        history_executed = [row["executed"].tolist() for row in state_histories]
        prebranch_snapshot = trial["history_snapshots"][-1]
        prebranch_position = _json_lane(
            prebranch_snapshot["base_pos_world"], local_lane_start
        )
        prebranch_quaternion = _json_lane(
            prebranch_snapshot["base_quat_wxyz"], local_lane_start
        )
        prebranch_pose = {
            "position_xyz_m": prebranch_position,
            "quaternion_wxyz": prebranch_quaternion,
        }
        context_pose_sequence = [
            {
                "position_xyz_m": _json_lane(
                    snapshot["base_pos_world"], local_lane_start
                ),
                "quaternion_wxyz": _json_lane(
                    snapshot["base_quat_wxyz"], local_lane_start
                ),
            }
            for snapshot in trial["history_snapshots"]
        ]
        context = {
            "rgb_artifact_ids": context_artifact_ids,
            "frame_identities": context_identities,
            "history_action_ids": list(state["history_action_ids"]),
            "history_executed_blocks": history_executed,
            "executed_block_sha256s": [
                pilot.canonical_block_sha256(block) for block in history_executed
            ],
            "endpoint_command_ticks": [0, 5, 10],
            "prebranch_state_sha256": synchronization_audit[
                "prebranch_state_sha256"
            ],
            "prebranch_base_pose_world": prebranch_pose,
            "context_base_pose_world_sequence": context_pose_sequence,
            "target_relative_body_xy_m": pilot.target_world_to_body_xy(
                target_xy_m=state["target_xy_m"],
                base_position_xyz_m=prebranch_position,
                base_quaternion_wxyz=prebranch_quaternion,
            ),
        }
        branches: list[dict[str, Any]] = []
        branch_rgb_by_offset: dict[int, np.ndarray] = {}
        for lane in pilot.lane_layout(
            state_id,
            role=str(state["role"]),
            state_index_in_scene=int(state["state_index_in_scene"]),
            sentinel_duplicate_action_id=state.get(
                "sentinel_duplicate_action_id"
            ),
        ):
            lane_offset = int(lane["lane_offset"])
            env_index = local_lane_start + lane_offset
            lane_index = global_lane_start + lane_offset
            kind = str(lane["kind"])
            action_id = int(lane["action_id"])
            identity = pilot.render_frame_identity(
                state_id=state_id,
                frame_kind=kind,
                index=lane_offset,
            )
            trajectory = _trajectory_rows(
                trial["trajectory_samples"],
                trial["trajectory_times_ns"],
                env_index,
            )
            heights = [float(row["base_pos_world"][2]) for row in trajectory]
            tips = [_quat_tip_rad(row["base_quat_wxyz"]) for row in trajectory]
            executed = np.asarray(trial["branch_executed"])[env_index].tolist()
            replay_frame = branch_replays[env_index]
            rgb = np.asarray(replay_frame["stored_rgb"])
            quality = _counterfactual_quality_disposition(
                replay_frame["quality"]
            )
            if quality["retained"] is not True:
                raise pilot.PilotContractError(
                    f"camera-invalid branch frame {identity}: "
                    f"{quality['hard_failure_reasons']}"
                )
            binding = _write_png_exclusive(
                rgb_root / f"{state_id}.{kind}.{lane_offset}.png",
                rgb,
                stage_wall_times=stage_wall_times,
            )
            binding = _relative_output_binding(
                binding, output_root=Path(plan["output_root"])
            )
            frame_receipt = {
                "artifact_id": identity,
                "frame_identity": identity,
                **binding,
                "width": 224,
                "height": 224,
                "mode": "RGB",
                "format": "PNG",
                "camera_valid": True,
                "low_information": quality["low_information"],
                "low_info_reasons": list(quality["low_info_reasons"]),
            }
            frame_receipts.append(frame_receipt)
            quality_audits.append(
                {
                    "frame_identity": identity,
                    "native_resolution": list(replay_frame["native_resolution"]),
                    "camera_valid": True,
                    "quality": quality,
                    "replay_pose": _replay_pose_audit(
                        replay_frame,
                        expected_components=endpoint_components,
                        env_index=env_index,
                    ),
                }
            )
            branch_rgb_by_offset[lane_offset] = rgb.copy()
            branch = {
                "lane_index": lane_index,
                "lane_offset": lane_offset,
                "kind": kind,
                "action_id": action_id,
                "action_name": str(action_catalog[action_id]["name"]),
                "duplicates_candidate_action_id": lane.get(
                    "duplicates_candidate_action_id"
                ),
                "requested_block": copy.deepcopy(
                    action_catalog[action_id]["requested_block"]
                ),
                "executed_block": executed,
                "executed_block_sha256": pilot.canonical_block_sha256(executed),
                "clipped": bool(np.asarray(trial["branch_clipped"])[env_index]),
                "trajectory_policy_step_samples": trajectory,
                "endpoint_state": _endpoint_state(endpoint_components, env_index),
                "physical_fell": min(heights) < float(
                    execution["fall_z_threshold_m"]
                ),
                "physical_tipped": max(tips) > float(
                    execution["tip_threshold_rad"]
                ),
                "physical_path_length_m": _path_length_m(
                    prebranch_snapshot,
                    trial["trajectory_samples"],
                    env_index,
                ),
                "physical_target_progress_m": _target_progress_m(
                    state=state,
                    prebranch=prebranch_snapshot,
                    endpoint=endpoint_components,
                    lane_index=env_index,
                ),
                "render_frame_identity": identity,
                "frame_receipt": frame_receipt,
            }
            branches.append(branch)
        candidate_branches = branches[: pilot.ACTION_COUNT]
        base_trajectory_identities = {
            pilot.canonical_json_sha256(
                [
                    {
                        "base_pos_world": row["base_pos_world"],
                        "base_quat_wxyz": row["base_quat_wxyz"],
                    }
                    for row in branch["trajectory_policy_step_samples"]
                ]
            )
            for branch in candidate_branches
        }
        endpoint_pose_identities = {
            pilot.canonical_json_sha256(
                {
                    "base_pos_world": branch["endpoint_state"]["base_pos_world"],
                    "base_quat_wxyz": branch["endpoint_state"]["base_quat_wxyz"],
                }
            )
            for branch in candidate_branches
        }
        candidate_png_hashes = {
            str(branch["frame_receipt"]["file_sha256"])
            for branch in candidate_branches
        }
        if len(base_trajectory_identities) < 2 or len(endpoint_pose_identities) < 2:
            raise pilot.PilotContractError(
                f"candidate physical responses collapsed for {state_id}"
            )
        if len(candidate_png_hashes) < 2:
            raise pilot.PilotContractError(
                f"candidate sequential render responses collapsed for {state_id}"
            )
        if sentinel_audit is not None:
            action_id = int(sentinel_audit["action_id"])
            candidate = branch_rgb_by_offset[action_id]
            sentinel = branch_rgb_by_offset[pilot.ACTION_COUNT]
            candidate_sha = hashlib.sha256(candidate.tobytes()).hexdigest()
            sentinel_sha = hashlib.sha256(sentinel.tobytes()).hexdigest()
            render_equal = np.array_equal(candidate, sentinel)
            png_equal = (
                branches[action_id]["frame_receipt"]["file_sha256"]
                == branches[pilot.ACTION_COUNT]["frame_receipt"]["file_sha256"]
            )
            render_audit = {
                "state_id": state_id,
                "group_index": int(state["group_index"]),
                "action_id": action_id,
                "candidate_lane": global_lane_start + action_id,
                "sentinel_lane": global_lane_start + pilot.ACTION_COUNT,
                "exact_equality_required": True,
                "stored_rgb_equal": bool(render_equal),
                "candidate_stored_rgb_sha256": candidate_sha,
                "sentinel_stored_rgb_sha256": sentinel_sha,
                "passed": bool(
                    render_equal and png_equal and candidate_sha == sentinel_sha
                ),
            }
            if not render_audit["passed"]:
                raise pilot.PilotContractError(
                    f"render duplicate-sentinel audit failed for {state_id}"
                )
            render_sentinel_audits.append(render_audit)
        receipts.append(
            {
                "schema": pilot.STATE_RECEIPT_SCHEMA,
                "attempt_id": str(plan["attempt_id"]),
                "status": "PHYSICS_COMPLETE",
                "physics_validated": False,
                "citable_as_scientific_evidence": False,
                "authorizes_retry_or_resume": False,
                "state": {
                    "state_id": state_id,
                    "role": str(state["role"]),
                    "family": str(state["family"]),
                    "scene_id": str(state["scene_id"]),
                    "group_index": int(state["group_index"]),
                    "state_index_in_scene": int(state["state_index_in_scene"]),
                    "lane_start": global_lane_start,
                    "lane_count": pilot.lane_count_for_role(str(state["role"])),
                    "scene_manifest_binding": _scene_receipt_binding(
                        state,
                        binding_name="scene_manifest_binding",
                        output_root=Path(plan["output_root"]),
                    ),
                    "scene_genesis_binding": _scene_receipt_binding(
                        state,
                        binding_name="scene_genesis_binding",
                        output_root=Path(plan["output_root"]),
                    ),
                    "target_xy_m": list(state["target_xy_m"]),
                },
                "context": context,
                "synchronization_audit": synchronization_audit,
                "branches": branches,
                "sentinel_audit": sentinel_audit,
                "render_sentinel_audit": (
                    render_sentinel_audits[-1]
                    if sentinel_audit is not None
                    else None
                ),
            }
        )
    return receipts, frame_receipts, quality_audits, render_sentinel_audits


def _runtime_imports() -> dict[str, Any]:
    from lewm_genesis.camera_safety import (
        camera_safety_config_from_pack,
        safe_camera_pose_from_base,
    )
    from lewm_genesis.go2_adapter import resolve_go2_urdf
    from lewm_genesis.lewm_contract import (
        PrimitiveRegistry,
        SafetyLimits,
        expand_primitive_to_block,
    )
    from lewm_genesis.rollout import (
        GenesisGo2PPOPolicy,
        RolloutConfig,
        RolloutRunner,
    )
    from lewm_genesis.scene_builder import build_scene_from_pack
    from lewm_genesis.scene_loader import (
        effective_camera_mount_xyz_rpy,
        load_platform_manifest,
        load_scene_pack,
    )
    from lewm_genesis.vision_quality import (
        LOW_INFO_REASON_NAMES,
        assess_rendered_frame,
    )

    if LOW_INFO_REASON_NAMES != COUNTERFACTUAL_LOW_INFO_REASON_NAMES:
        raise pilot.PilotContractError(
            "counterfactual low-information reason registry changed"
        )

    return {
        "camera_safety_config_from_pack": camera_safety_config_from_pack,
        "safe_camera_pose_from_base": safe_camera_pose_from_base,
        "resolve_go2_urdf": resolve_go2_urdf,
        "PrimitiveRegistry": PrimitiveRegistry,
        "SafetyLimits": SafetyLimits,
        "expand_primitive_to_block": expand_primitive_to_block,
        "GenesisGo2PPOPolicy": GenesisGo2PPOPolicy,
        "RolloutConfig": RolloutConfig,
        "RolloutRunner": RolloutRunner,
        "build_scene_from_pack": build_scene_from_pack,
        "effective_camera_mount_xyz_rpy": effective_camera_mount_xyz_rpy,
        "load_platform_manifest": load_platform_manifest,
        "load_scene_pack": load_scene_pack,
        "assess_rendered_frame": assess_rendered_frame,
    }


def _load_action_blocks(
    *, plan: Mapping[str, Any], registry: Any, expand: Any
) -> list[list[list[float]]]:
    blocks: list[list[list[float]]] = []
    for action in plan["action_catalog"]:
        actual = np.asarray(
            expand(registry, str(action["name"])), dtype=np.float32
        )
        expected = np.asarray(action["requested_block"], dtype=np.float32)
        if not np.array_equal(actual, expected):
            raise pilot.PilotContractError(
                f"registry expansion disagrees for action {action['name']}"
            )
        blocks.append(actual.tolist())
    return blocks


def _materialize_smoke_scene(plan: dict[str, Any]) -> dict[str, Any]:
    """Create the one source-declared scene after attempt reservation."""

    from lewm_worlds.exporters.to_genesis import export_genesis_scene
    from lewm_worlds.families import build_family_manifest
    from lewm_worlds.manifest import manifest_sha256
    from lewm_worlds.splits import plan_corpus

    if plan["purpose"] != "source_integration_smoke" or len(plan["states"]) != 1:
        raise pilot.PilotContractError("scene materialization is smoke-only")
    state = plan["states"][0]
    declaration = state["scene_generation"]
    if declaration["scene_generator_binding"] != pilot.file_binding(Path(__file__)):
        raise pilot.PilotContractError("smoke scene generator source binding changed")
    corpus_plan = plan_corpus(
        plan_seed=int(declaration["plan_seed"]),
        totals={
            str(declaration["split"]): {
                str(declaration["family"]): 1,
            }
        },
        validate=True,
    )
    if len(corpus_plan.assignments) != 1:
        raise pilot.PilotContractError("smoke scene declaration produced !=1 scene")
    assignment = corpus_plan.assignments[0]
    if (
        assignment.scene_index != declaration["scene_index"]
        or assignment.scene_id != state["scene_id"]
        or assignment.family != state["family"]
        or assignment.split != declaration["split"]
    ):
        raise pilot.PilotContractError("materialized smoke scene identity drifted")
    manifest = build_family_manifest(
        scene_seed=assignment.scene_seed,
        family=assignment.family,
        split=assignment.split,
        difficulty_tier=None,
    )
    if not manifest.landmarks:
        raise pilot.PilotContractError("generated smoke scene has no target landmark")
    canonical_target = min(manifest.landmarks, key=lambda item: item.object_id)
    canonical_target_xy = np.asarray(
        canonical_target.center_xyz_m[:2], dtype=np.float64
    )
    target_xy = np.asarray(state["target_xy_m"], dtype=np.float64)
    if not np.array_equal(target_xy, canonical_target_xy):
        raise pilot.PilotContractError(
            "smoke target_xy_m is not the exact canonical generated landmark center"
        )
    scene_dir = (
        Path(plan["output_root"])
        / "generated_scene"
        / assignment.split
        / assignment.family
        / assignment.scene_id
    )
    manifest_payload = manifest.to_dict()
    manifest_payload["manifest_sha256"] = manifest_sha256(manifest)
    manifest_binding = pilot.write_json_exclusive(
        scene_dir / "manifest.json", manifest_payload
    )
    genesis_binding = pilot.write_json_exclusive(
        scene_dir / "genesis_scene.json", export_genesis_scene(manifest)
    )
    state["scene_manifest_binding"] = manifest_binding
    state["scene_genesis_binding"] = genesis_binding
    return {
        "declaration": dict(declaration),
        "scene_manifest_binding": _relative_output_binding(
            manifest_binding, output_root=Path(plan["output_root"])
        ),
        "scene_genesis_binding": _relative_output_binding(
            genesis_binding, output_root=Path(plan["output_root"])
        ),
        "scene_seed": int(assignment.scene_seed),
        "scene_seed_salt": int(assignment.scene_seed_salt),
        "target_landmark_id": str(canonical_target.object_id),
    }


def _prepare_plan_scenes(plan: dict[str, Any]) -> dict[str, Any] | None:
    """Materialize the smoke or retain exact pre-bound calibration/pilot scenes."""

    if plan["purpose"] == "source_integration_smoke":
        return _materialize_smoke_scene(plan)
    if plan["purpose"] not in NON_SMOKE_AUTHORITY_CONTRACTS:
        raise pilot.PilotContractError("unsupported collection purpose")
    for state in plan["states"]:
        if state["scene_generation"] is not None:
            raise pilot.PilotContractError(
                "calibration/pilot state unexpectedly requests scene materialization"
            )
        for name in ("scene_manifest_binding", "scene_genesis_binding"):
            binding = state[name]
            if not isinstance(binding, Mapping) or not Path(binding["path"]).is_absolute():
                raise pilot.PilotContractError(
                    f"calibration/pilot {name} is not an absolute bound input"
                )
    return None


def derive_source_integration_smoke_scene(
    *, family: str, plan_seed: int
) -> dict[str, Any]:
    """Derive plan metadata in memory without opening an existing corpus.

    The returned values are the exact ``scene_id`` and ``target_xy_m`` a plan
    author must bind.  Runtime materialization independently recomputes them
    after reservation and fails on any drift.
    """

    from lewm_worlds.families import build_family_manifest
    from lewm_worlds.splits import plan_corpus

    corpus_plan = plan_corpus(
        plan_seed=int(plan_seed),
        totals={"calibration_smoke": {str(family): 1}},
        validate=True,
    )
    if len(corpus_plan.assignments) != 1:
        raise pilot.PilotContractError("smoke metadata derivation produced !=1 scene")
    assignment = corpus_plan.assignments[0]
    manifest = build_family_manifest(
        scene_seed=assignment.scene_seed,
        family=assignment.family,
        split=assignment.split,
        difficulty_tier=None,
    )
    if not manifest.landmarks:
        raise pilot.PilotContractError("generated smoke scene has no target landmark")
    target = min(manifest.landmarks, key=lambda item: item.object_id)
    return {
        "family": assignment.family,
        "split": assignment.split,
        "plan_seed": int(plan_seed),
        "scene_index": 0,
        "scene_id": assignment.scene_id,
        "scene_seed": int(assignment.scene_seed),
        "scene_seed_salt": int(assignment.scene_seed_salt),
        "target_landmark_id": str(target.object_id),
        "target_xy_m": [
            float(target.center_xyz_m[0]),
            float(target.center_xyz_m[1]),
        ],
    }


def _build_rollout_runner(
    *,
    plan: Mapping[str, Any],
    runtime: Mapping[str, Any],
    platform: Mapping[str, Any],
    build: Any,
    registry: Any,
) -> Any:
    """Construct the production runner through the exact bound runtime API."""

    execution = plan["execution_contract"]
    policy = runtime["GenesisGo2PPOPolicy"](
        checkpoint_path=plan["runtime_bindings"]["policy_checkpoint"]["path"],
        cfg_path=plan["runtime_bindings"]["policy_config"]["path"],
        device=str(execution["policy_device"]),
        deduplicate_exact_observation_rows=True,
    )
    config = runtime["RolloutConfig"](
        n_blocks=0,
        fall_z_threshold_m=float(execution["fall_z_threshold_m"]),
        tip_threshold_rad=float(execution["tip_threshold_rad"]),
        rgb_capture_per_block=False,
        seed=int(execution["seed"]),
        randomize_spawn_pose=False,
    )
    safety_factory = getattr(runtime["SafetyLimits"], "from_manifest", None)
    if not callable(safety_factory):
        raise pilot.PilotContractError(
            "bound SafetyLimits runtime lacks the from_manifest factory"
        )
    runner = runtime["RolloutRunner"](
        build,
        policy,
        registry,
        safety_factory(dict(platform)),
        config=config,
    )
    if runner._policy_steps_per_command_tick != 5:  # noqa: SLF001
        raise pilot.PilotContractError("runtime controller cadence disagrees with plan")
    if runner._physics_steps_per_policy != 10:  # noqa: SLF001
        raise pilot.PilotContractError("runtime physics decimation disagrees with smoke cap")
    runner.policy_steps_per_command_tick = 5
    return runner


def _require_scene_parallelization(
    *, build: Any, execution: Mapping[str, Any]
) -> None:
    """Require Genesis to honor the plan-bound solver parallelization mode."""

    observed = int(getattr(build.scene, "_para_level", -1))  # noqa: SLF001
    expected = int(execution["environment"]["GS_PARA_LEVEL"])
    if observed != expected:
        raise pilot.PilotContractError(
            "Genesis scene parallelization level disagrees with plan: "
            f"observed {observed}, expected {expected}"
        )


def _collect_scene(
    *,
    plan: Mapping[str, Any],
    states: Sequence[Mapping[str, Any]],
    runtime: Mapping[str, Any],
    platform: Mapping[str, Any],
    registry: Any,
    action_blocks: Sequence[Sequence[Sequence[float]]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
]:
    scene_started = time.perf_counter()
    execution = plan["execution_contract"]
    state0 = states[0]
    scene_dir = Path(state0["scene_manifest_binding"]["path"]).parent
    pack = runtime["load_scene_pack"](
        scene_dir,
        platform_manifest=platform,
        workspace_root=ROOT,
    )
    if pack.scene_id != state0["scene_id"] or pack.family != state0["family"]:
        raise pilot.PilotContractError("loaded scene identity disagrees with plan")
    if Path(pack.robot.urdf_path).resolve() != Path(
        plan["runtime_bindings"]["go2_urdf"]["path"]
    ).resolve():
        raise pilot.PilotContractError("loaded scene resolved an unbound Go2 URDF")
    build_started = time.perf_counter()
    build = runtime["build_scene_from_pack"](
        pack,
        n_envs=sum(
            pilot.lane_count_for_role(str(state["role"])) for state in states
        ),
        backend=str(execution["backend"]),
        show_viewer=False,
        render_robot=False,
        apply_textures=False,
        batched_camera=False,
    )
    build_wall_seconds = time.perf_counter() - build_started
    _require_scene_parallelization(build=build, execution=execution)
    if bool(getattr(build.camera, "_is_batched", False)):
        raise pilot.PilotContractError("physical lockstep camera must be non-batched")
    runner = _build_rollout_runner(
        plan=plan,
        runtime=runtime,
        platform=platform,
        build=build,
        registry=registry,
    )
    _initialize_exact_clones(runner)
    stage_wall_times = _new_stage_wall_times()
    pipeline_started = time.perf_counter()
    lockstep_started = time.perf_counter()
    trial = pilot.execute_lockstep_trial(
        runner=runner,
        states=states,
        action_blocks=action_blocks,
        capture_components=lambda: _capture_components(runner),
        capture_sim_time_ns=lambda: int(runner._sim_time_ns),  # noqa: SLF001
    )
    if (
        trial["history_times_ns"] != [0, 500_000_000, 1_000_000_000]
        or len(trial["trajectory_times_ns"]) != POLICY_STEP_COUNT
        or trial["trajectory_times_ns"][-1] != 1_500_000_000
        or int(runner._sim_time_ns) != 1_500_000_000  # noqa: SLF001
    ):
        raise pilot.PilotContractError(
            "observed simulation duration/policy-step count disagrees with smoke caps"
        )
    lockstep_wall_seconds = time.perf_counter() - lockstep_started
    render_build_started = time.perf_counter()
    render_build = runtime["build_scene_from_pack"](
        pack,
        n_envs=1,
        backend=str(execution["backend"]),
        show_viewer=False,
        render_robot=False,
        apply_textures=False,
        batched_camera=False,
    )
    render_scene_build_wall_seconds = time.perf_counter() - render_build_started
    if render_build is build or getattr(render_build, "scene", None) is getattr(
        build, "scene", None
    ):
        raise pilot.PilotContractError(
            "render replay must use a scene separate from physical lockstep"
        )
    trial["render_replay"] = _capture_sequential_render_replay(
        render_build,
        states=states,
        trial=trial,
        safe_camera_pose_from_base=runtime["safe_camera_pose_from_base"],
        camera_safety_config_from_pack=runtime[
            "camera_safety_config_from_pack"
        ],
        effective_camera_mount_xyz_rpy=runtime[
            "effective_camera_mount_xyz_rpy"
        ],
        assess_rendered_frame=runtime["assess_rendered_frame"],
        stage_wall_times=stage_wall_times,
    )
    receipt_started = time.perf_counter()
    receipts, frame_receipts, quality_audits, render_sentinel_audits = (
        _group_trial_receipts(
        plan=plan,
        states=states,
        trial=trial,
        rgb_root=Path(plan["output_root"])
        / "scenes"
        / str(state0["role"])
        / str(state0["scene_id"])
        / "rgb",
        stage_wall_times=stage_wall_times,
    )
    )
    post_lockstep_receipt_wall_seconds = time.perf_counter() - receipt_started
    scene_pipeline_wall_seconds = time.perf_counter() - pipeline_started
    scene_total_wall_seconds = time.perf_counter() - scene_started
    wall_values = {
        **stage_wall_times,
        "lockstep_execution_wall_seconds": lockstep_wall_seconds,
        "render_scene_build_wall_seconds": render_scene_build_wall_seconds,
        "post_lockstep_receipt_wall_seconds": post_lockstep_receipt_wall_seconds,
        "scene_pipeline_wall_seconds": scene_pipeline_wall_seconds,
        "scene_total_wall_seconds": scene_total_wall_seconds,
    }
    if any(
        not math.isfinite(value) or value < 0.0 for value in wall_values.values()
    ):
        raise pilot.PilotContractError("scene stage wall timing is invalid")
    if (
        stage_wall_times["native_render_wall_seconds"]
        + stage_wall_times["camera_quality_resize_wall_seconds"]
        + lockstep_wall_seconds
        + render_scene_build_wall_seconds
        + post_lockstep_receipt_wall_seconds
        > scene_pipeline_wall_seconds
        or stage_wall_times["png_encode_write_hash_wall_seconds"]
        > post_lockstep_receipt_wall_seconds
        or build_wall_seconds + scene_pipeline_wall_seconds
        > scene_total_wall_seconds
    ):
        raise pilot.PilotContractError("scene stage wall timing is inconsistent")
    return receipts, frame_receipts, quality_audits, render_sentinel_audits, {
        "scene_id": str(state0["scene_id"]),
        "family": str(state0["family"]),
        "role": str(state0["role"]),
        "states": len(states),
        "envs": sum(
            pilot.lane_count_for_role(str(state["role"])) for state in states
        ),
        "physics_build_wall_seconds": build_wall_seconds,
        "physics_simulation_wall_seconds": lockstep_wall_seconds,
        "common_prefix_step_wall_seconds": float(
            trial["common_prefix_step_wall_seconds"]
        ),
        "branch_step_wall_seconds": float(trial["branch_step_wall_seconds"]),
        **wall_values,
        "native_render_calls": int(trial["render_replay"]["native_render_calls"]),
        "stored_rgb_frames": len(frame_receipts),
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": "solid_materials_box_physics_preserved",
    }


def collect(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
    supervisor_nonce: str,
) -> tuple[dict[str, Any], Path]:
    if re.fullmatch(r"[0-9a-f]{64}", supervisor_nonce) is None:
        raise pilot.PilotContractError("supervisor ownership nonce is invalid")
    raw_plan, plan_binding = pilot.read_bound_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="counterfactual pilot plan",
    )
    plan = copy.deepcopy(pilot.validate_plan(raw_plan))
    raw_authority, authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="counterfactual execution authority",
    )
    authority = _validate_authority_for_plan(
        raw_authority,
        plan=plan,
        plan_binding=plan_binding,
    )
    review_binding = authority["review_binding"]
    raw_review, actual_review_binding = pilot.read_bound_json(
        Path(review_binding["path"]),
        expected_sha256=str(review_binding["file_sha256"]),
        expected_byte_count=int(review_binding["byte_count"]),
        label="independent source review",
    )
    if actual_review_binding != review_binding:
        raise pilot.PilotContractError("source review binding changed")
    pilot.validate_source_review(raw_review, authority=authority)
    _validate_git_authority_boundary(
        plan_binding=plan_binding,
        authority_binding=authority_binding,
        authority=authority,
    )
    source_by_name = {
        str(row["name"]): row["binding"] for row in authority["source_bindings"]
    }
    if plan["purpose"] == "source_integration_smoke":
        declaration_binding = plan["states"][0]["scene_generation"][
            "scene_generator_binding"
        ]
        if declaration_binding != source_by_name["scene_generator_materializer"]:
            raise pilot.PilotContractError(
                "plan scene generator binding disagrees with reviewed authority source"
            )
    output_root = pilot.fresh_development_output_root(
        Path(plan["output_root"]),
        development_root=ROOT / ".generated" / "dev",
    )
    reservation = {
        "schema": (
            "lewm_go2_world_model_counterfactual_smoke_reservation_v1"
            if plan["purpose"] == "source_integration_smoke"
            else "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
        ),
        "status": "RESERVED_ATTEMPT_CONSUMED",
        "attempt": dict(authority["attempt"]),
        "plan_binding": plan_binding,
        "authority_binding": authority_binding,
        "supervisor_nonce": supervisor_nonce,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    reservation_binding = pilot.write_json_exclusive(
        output_root / "reservation.json", reservation
    )
    reservation_binding = _relative_output_binding(
        reservation_binding, output_root=output_root
    )
    plan_receipt_binding = _copy_exact_plan_receipt(
        plan_binding, output_root=output_root
    )
    state_receipt_bindings: list[dict[str, Any]] = []
    written_state_receipts: list[dict[str, Any]] = []
    render_receipt_bindings: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    scene_materialization: dict[str, Any] | None = None
    runtime_versions: dict[str, str] | None = None
    failure: dict[str, Any] | None = None
    collection_started = time.perf_counter()
    try:
        pilot.require_plan_bindings(plan)
        _validate_python_runtime(plan)
        _validate_execution_environment(plan)
        runtime_versions = _capture_runtime_versions()
        scene_materialization = _prepare_plan_scenes(plan)
        runtime = _runtime_imports()
        platform = runtime["load_platform_manifest"](
            plan["runtime_bindings"]["platform_manifest"]["path"]
        )
        resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), ROOT)
        if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
            raise pilot.PilotContractError("platform resolves a different Go2 URDF")
        registry = runtime["PrimitiveRegistry"].from_yaml(
            plan["runtime_bindings"]["primitive_registry"]["path"]
        )
        action_blocks = _load_action_blocks(
            plan=plan,
            registry=registry,
            expand=runtime["expand_primitive_to_block"],
        )
        states_by_scene: dict[
            tuple[str, str], list[Mapping[str, Any]]
        ] = defaultdict(list)
        for state in plan["states"]:
            states_by_scene[(str(state["role"]), str(state["scene_id"]))].append(
                state
            )
        for (role, scene_id), states in states_by_scene.items():
            scene_dir = output_root / "scenes" / role / scene_id
            (
                receipts,
                frame_receipts,
                quality_audits,
                render_sentinel_audits,
                metrics,
            ) = _collect_scene(
                plan=plan,
                states=states,
                runtime=runtime,
                platform=platform,
                registry=registry,
                action_blocks=action_blocks,
            )
            render_receipt = {
                "schema": LIVE_RENDER_RECEIPT_SCHEMA,
                "attempt_id": str(plan["attempt_id"]),
                "status": "RENDER_COMPLETE",
                "physics_validated": False,
                "citable_as_scientific_evidence": False,
                "scene": {
                    "role": role,
                    "scene_id": scene_id,
                    "family": str(states[0]["family"]),
                    "scene_manifest_binding": _scene_receipt_binding(
                        states[0],
                        binding_name="scene_manifest_binding",
                        output_root=output_root,
                    ),
                    "scene_genesis_binding": _scene_receipt_binding(
                        states[0],
                        binding_name="scene_genesis_binding",
                        output_root=output_root,
                    ),
                },
                "render_contract": dict(pilot.RENDER_CONTRACT),
                "native_render_calls": int(metrics["native_render_calls"]),
                "stored_rgb_frames": int(metrics["stored_rgb_frames"]),
                "depth_rendered": True,
                "depth_persisted": False,
                "visual_mode": "solid_materials_box_physics_preserved",
                "visual_domain_fidelity_claimed": False,
                "frame_receipts": frame_receipts,
                "quality_audits": quality_audits,
                "render_sentinel_audits": render_sentinel_audits,
            }
            render_receipt_binding = pilot.write_json_exclusive(
                scene_dir / "live_render_receipt.json", render_receipt
            )
            render_receipt_binding = _relative_output_binding(
                render_receipt_binding, output_root=output_root
            )
            render_receipt_bindings.append(render_receipt_binding)
            for receipt in receipts:
                receipt["render_receipt_binding"] = render_receipt_binding
                receipt_binding = pilot.write_json_exclusive(
                    scene_dir / "state_receipts" / f"{receipt['state']['state_id']}.json",
                    receipt,
                )
                receipt_binding = _relative_output_binding(
                    receipt_binding, output_root=output_root
                )
                state_receipt_bindings.append(receipt_binding)
                written_state_receipts.append(receipt)
            scene_metrics.append(metrics)
        elapsed = time.perf_counter() - collection_started
        if elapsed > float(authority["caps"]["wall_seconds"]):
            raise pilot.PilotContractError("authority wall_seconds cap exceeded")
        if (
            sum(int(row["native_render_calls"]) for row in scene_metrics)
            != int(authority["caps"]["native_render_calls"])
            or sum(int(row["stored_rgb_frames"]) for row in scene_metrics)
            != int(authority["caps"]["stored_rgb_frames"])
        ):
            raise pilot.PilotContractError("observed render work disagrees with caps")
    except Exception as exc:  # fail closed, preserve partial one-shot evidence
        failure = _failure_receipt(exc)
    roles: dict[str, int] = defaultdict(int)
    scene_keys: set[tuple[str, str]] = set()
    candidate_branches = 0
    sentinel_branches = 0
    context_frames = 0
    for receipt in written_state_receipts:
        role = str(receipt["state"]["role"])
        roles[role] += 1
        scene_keys.add((role, str(receipt["state"]["scene_id"])))
        candidate_branches += sum(
            branch["kind"] == "candidate" for branch in receipt["branches"]
        )
        sentinel_branches += sum(
            branch["kind"] == "sentinel" for branch in receipt["branches"]
        )
        context_frames += len(receipt["context"]["frame_identities"])
    observed_counts = {
        "scenes": len(scene_keys),
        "states": len(written_state_receipts),
        "roles": dict(sorted(roles.items())),
        "actions": len(plan["action_catalog"]),
        "candidate_branches": candidate_branches,
        "sentinel_branches": sentinel_branches,
        "total_branches": candidate_branches + sentinel_branches,
        "context_frames": context_frames,
        "target_frames": candidate_branches + sentinel_branches,
    }
    result = {
        "schema": pilot.PHYSICS_RESULT_SCHEMA,
        "attempt_id": str(plan["attempt_id"]),
        "purpose": str(plan["purpose"]),
        "status": "PHYSICS_COMPLETE" if failure is None else "FAILED",
        "physics_validated": False,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "allows_refill": False,
        "allows_overwrite": False,
        "branch_mechanism": pilot.BRANCH_MECHANISM,
        "plan_binding": plan_binding,
        "plan_receipt_binding": plan_receipt_binding,
        "authority_binding": authority_binding,
        "review_binding": review_binding,
        "reservation_binding": reservation_binding,
        "caps": dict(authority["caps"]),
        "execution_contract": dict(plan["execution_contract"]),
        "runtime_versions": runtime_versions,
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": list(authority["source_bindings"]),
        "expected_counts": dict(plan["expected_counts"]),
        "observed_counts": observed_counts,
        "scene_materialization": scene_materialization,
        "state_receipt_bindings": state_receipt_bindings,
        "render_receipt_bindings": render_receipt_bindings,
        "scene_metrics": scene_metrics,
        "visual_domain_limitation": (
            "solid materials and box primitives preserve the physics geometry; "
            "this collection does not establish final visual-domain fidelity"
        ),
        "collection_wall_seconds": time.perf_counter() - collection_started,
        "failure": failure,
    }
    result_path = output_root / "physics_result.json"
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-byte-count", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--supervisor-nonce", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result, result_path = collect(
        plan_path=args.plan,
        expected_plan_byte_count=args.expected_plan_byte_count,
        expected_plan_sha256=args.expected_plan_sha256,
        authority_path=args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
        supervisor_nonce=args.supervisor_nonce,
    )
    print(
        json.dumps(
            {"status": result["status"], "physics_result": str(result_path)},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PHYSICS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
