#!/usr/bin/env python3
"""Collect one separately reviewed bounded WM-A pilot.

This entrypoint exists because the frozen calibration collector deliberately
does not accept pilot authority.  It reuses that collector's reviewed runtime
primitives but validates a distinct pilot authority and never weakens or edits
the calibration authority boundary.
"""
from __future__ import annotations

import argparse
import copy
from collections import defaultdict
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as kernel  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_experiment_authority_v1 as authority_contract  # noqa: E402
from scripts import build_go2_world_model_bounded_branch_experiment_plan_v1 as plan_contract  # noqa: E402


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ).stdout.strip()
    except subprocess.CalledProcessError as exc:
        raise pilot.PilotContractError(
            f"git boundary check failed: {' '.join(args)}"
        ) from exc


def load_and_validate_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    try:
        plan_contract._reject_protected_path(plan_path, label="bounded branch plan")  # noqa: SLF001
        plan_contract._reject_protected_path(  # noqa: SLF001
            authority_path, label="bounded branch authority"
        )
    except plan_contract.BoundedBranchPlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc
    raw_plan, plan_binding = pilot.read_bound_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="bounded branch plan",
    )
    plan = copy.deepcopy(pilot.validate_plan(raw_plan))
    parity_result_binding = kernel._validate_visual_domain_parity_result(  # noqa: SLF001
        plan
    )
    if parity_result_binding != plan["visual_domain_parity_result_binding"]:
        raise pilot.PilotContractError(
            "bounded plan visual-domain parity binding changed"
        )
    raw_authority, authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="bounded branch authority",
    )
    gate_binding = raw_authority.get("calibration_gate_binding")
    if not isinstance(gate_binding, Mapping):
        raise pilot.PilotContractError("authority calibration gate binding is absent")
    try:
        plan_contract._reject_protected_path(  # noqa: SLF001
            Path(str(gate_binding["path"])), label="bounded branch calibration gate"
        )
    except plan_contract.BoundedBranchPlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc
    raw_gate, actual_gate_binding = pilot.read_bound_json(
        Path(str(gate_binding["path"])),
        expected_sha256=str(gate_binding["file_sha256"]),
        expected_byte_count=int(gate_binding["byte_count"]),
        label="bounded branch calibration gate",
    )
    if actual_gate_binding != dict(gate_binding):
        raise pilot.PilotContractError("calibration gate binding changed")
    try:
        authority = authority_contract.validate_authority_v1(
            raw_authority,
            plan=plan,
            plan_binding=plan_binding,
            gate=raw_gate,
            gate_binding=actual_gate_binding,
        )
    except authority_contract.BoundedBranchAuthorityError as exc:
        raise pilot.PilotContractError(str(exc)) from exc
    review_binding = authority["review_binding"]
    try:
        plan_contract._reject_protected_path(  # noqa: SLF001
            Path(str(review_binding["path"])), label="bounded branch source review"
        )
    except plan_contract.BoundedBranchPlanError as exc:
        raise pilot.PilotContractError(str(exc)) from exc
    raw_review, actual_review = pilot.read_bound_json(
        Path(str(review_binding["path"])),
        expected_sha256=str(review_binding["file_sha256"]),
        expected_byte_count=int(review_binding["byte_count"]),
        label="bounded branch independent source review",
    )
    if actual_review != review_binding:
        raise pilot.PilotContractError("source review binding changed")
    pilot.validate_source_review(raw_review, authority=authority)

    head = _git("rev-parse", "HEAD")
    source_commit = str(authority["source_commit"])
    _git("merge-base", "--is-ancestor", source_commit, head)
    for binding, label in (
        (plan_binding, "bounded branch plan"),
        (authority_binding, "bounded branch authority"),
        (actual_gate_binding, "bounded branch calibration gate"),
        (actual_review, "bounded branch source review"),
    ):
        kernel._binding_at_commit(binding, commit=head, label=label)  # noqa: SLF001
    expected_paths = authority_contract.canonical_source_paths_v1()
    for row in authority["source_bindings"]:
        name = str(row["name"])
        binding = row["binding"]
        if pilot.file_binding(REPO_ROOT / expected_paths[name]) != binding:
            raise pilot.PilotContractError(f"working source changed for {name}")
        kernel._binding_at_commit(  # noqa: SLF001
            binding, commit=source_commit, label=f"bounded branch source {name}"
        )
    return authority, authority_binding, plan, plan_binding


def _validate_bound_scenes(plan: Mapping[str, Any]) -> None:
    for state in plan["states"]:
        if state["scene_generation"] is not None:
            raise pilot.PilotContractError("bounded pilot may not materialize scenes")
        for name in ("scene_manifest_binding", "scene_genesis_binding"):
            binding = state[name]
            if not Path(str(binding["path"])).is_absolute():
                raise pilot.PilotContractError("bounded pilot scene input is not absolute")


def _validated_render_receipt_identity_v1(
    *, plan: Mapping[str, Any], metrics: Mapping[str, Any]
) -> dict[str, Any]:
    """Derive the receipt identity from the exact plan and observed render path."""

    if pilot.canonical_json_bytes(plan.get("render_contract")) != pilot.canonical_json_bytes(
        pilot.TEXTURED_V03_RENDER_CONTRACT
    ):
        raise pilot.PilotContractError(
            "bounded receipt requires the versioned textured_v03 render contract"
        )
    if (
        metrics.get("depth_rendered") is not True
        or metrics.get("depth_persisted") is not False
        or metrics.get("visual_mode") != pilot.TEXTURED_V03_VISUAL_MODE
    ):
        raise pilot.PilotContractError(
            "bounded render metrics disagree with the textured_v03 sensor path"
        )
    native_render_calls = metrics.get("native_render_calls")
    rgb_render_calls = metrics.get("rgb_render_calls")
    auxiliary_depth_render_calls = metrics.get("auxiliary_depth_render_calls")
    stored_rgb_frames = metrics.get("stored_rgb_frames")
    if (
        isinstance(native_render_calls, bool)
        or not isinstance(native_render_calls, int)
        or native_render_calls < 1
        or isinstance(rgb_render_calls, bool)
        or not isinstance(rgb_render_calls, int)
        or rgb_render_calls != native_render_calls
        or isinstance(auxiliary_depth_render_calls, bool)
        or not isinstance(auxiliary_depth_render_calls, int)
        or auxiliary_depth_render_calls != native_render_calls
        or isinstance(stored_rgb_frames, bool)
        or not isinstance(stored_rgb_frames, int)
        or stored_rgb_frames != native_render_calls
    ):
        raise pilot.PilotContractError(
            "bounded render-call and stored-frame accounting changed"
        )
    parity_binding = pilot._validate_binding_shape(  # noqa: SLF001
        plan.get("visual_domain_parity_result_binding"),
        label="bounded visual-domain parity result",
    )
    parity_terminal_binding = pilot._validate_binding_shape(  # noqa: SLF001
        plan.get("visual_domain_parity_terminal_binding"),
        label="bounded visual-domain parity terminal",
    )
    parity_review_binding = pilot._validate_binding_shape(  # noqa: SLF001
        plan.get("visual_domain_parity_review_binding"),
        label="bounded visual-domain parity review",
    )
    mesh_values = metrics.get("derived_mesh_bindings")
    if not isinstance(mesh_values, list) or not mesh_values:
        raise pilot.PilotContractError(
            "bounded textured_v03 derived mesh closure is absent"
        )
    mesh_bindings = [
        pilot._validate_binding_shape(  # noqa: SLF001
            value,
            label=f"bounded derived textured mesh {index}",
        )
        for index, value in enumerate(mesh_values)
    ]
    mesh_paths = [str(value["path"]) for value in mesh_bindings]
    if (
        mesh_paths != sorted(mesh_paths)
        or len(mesh_paths) != len(set(mesh_paths))
        or any(not Path(path).is_absolute() or Path(path).suffix != ".obj" for path in mesh_paths)
    ):
        raise pilot.PilotContractError(
            "bounded textured_v03 derived mesh closure changed"
        )
    return {
        "render_contract": dict(plan["render_contract"]),
        "native_render_calls": native_render_calls,
        "rgb_render_calls": rgb_render_calls,
        "auxiliary_depth_render_calls": auxiliary_depth_render_calls,
        "stored_rgb_frames": stored_rgb_frames,
        # One historical RGB-only render and one separate transient depth-only
        # quality render occur for every logical render call.  Depth is never
        # encoded or persisted as experiment data.
        "depth_rendered": True,
        "depth_persisted": False,
        "visual_mode": pilot.TEXTURED_V03_VISUAL_MODE,
        "visual_domain_fidelity_claimed": True,
        "visual_domain_parity_result_binding": parity_binding,
        "visual_domain_parity_terminal_binding": parity_terminal_binding,
        "visual_domain_parity_review_binding": parity_review_binding,
        "derived_mesh_bindings": mesh_bindings,
    }


def _load_supervisor_owned_reservation_v1(
    *,
    output_root_text: object,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    supervisor_nonce: str,
    supervisor_pid: int,
) -> tuple[Path, dict[str, Any]]:
    """Require the exact reservation created by this collector's live parent.

    The collector is deliberately unable to create or consume an attempt on its
    own.  The reviewed supervisor must first create the fresh attempt root and
    its exclusive reservation, then exec this process directly.  Binding both
    the random nonce and the live parent PID prevents an arbitrary standalone
    collector invocation from crossing that boundary.
    """

    if re.fullmatch(r"[0-9a-f]{64}", supervisor_nonce) is None:
        raise pilot.PilotContractError("supervisor ownership nonce is invalid")
    if isinstance(supervisor_pid, bool) or not isinstance(supervisor_pid, int):
        raise pilot.PilotContractError("supervisor PID is invalid")
    if supervisor_pid <= 1 or os.getppid() != supervisor_pid:
        raise pilot.PilotContractError(
            "collector is not a direct child of the reserved supervisor"
        )
    if not isinstance(output_root_text, str) or not Path(output_root_text).is_absolute():
        raise pilot.PilotContractError("bounded pilot output root is invalid")
    development_root = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    selected = Path(os.path.abspath(output_root_text))
    try:
        relative = selected.relative_to(development_root)
    except ValueError as exc:
        raise pilot.PilotContractError(
            "supervisor-owned output root escapes .generated/dev"
        ) from exc
    if not relative.parts:
        raise pilot.PilotContractError(
            "supervisor-owned output root cannot equal .generated/dev"
        )
    cursor = development_root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise pilot.PilotContractError(
                "supervisor-owned output root traverses a symlink"
            )
        if not cursor.exists() or not cursor.is_dir():
            raise pilot.PilotContractError(
                "supervisor-owned output root is absent or not a regular directory"
            )
    output_root = selected.resolve(strict=True)
    # Before this boundary the supervisor has authority to create exactly one
    # file.  Any other pre-existing entry is stale or foreign attempt state.
    if {entry.name for entry in output_root.iterdir()} != {"reservation.json"}:
        raise pilot.PilotContractError(
            "supervisor-owned output root contains pre-existing attempt state"
        )
    reservation_path = output_root / "reservation.json"
    if reservation_path.is_symlink() or not reservation_path.is_file():
        raise pilot.PilotContractError("supervisor reservation is not a regular file")
    absolute_binding = pilot.file_binding(reservation_path)
    reservation, actual_binding = pilot.read_bound_json(
        reservation_path,
        expected_sha256=str(absolute_binding["file_sha256"]),
        expected_byte_count=int(absolute_binding["byte_count"]),
        label="bounded branch supervisor reservation",
    )
    expected_keys = {
        "schema",
        "status",
        "attempt",
        "plan_binding",
        "authority_binding",
        "supervisor_nonce",
        "supervisor_pid",
        "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt",
        "retry_authorized",
        "resume_authorized",
        "overwrite_authorized",
        "refill_authorized",
    }
    if (
        actual_binding != absolute_binding
        or not isinstance(reservation, Mapping)
        or set(reservation) != expected_keys
        or reservation.get("schema")
        != "lewm_go2_world_model_counterfactual_attempt_reservation_v1"
        or reservation.get("status") != "RESERVED_ATTEMPT_CONSUMED"
        or reservation.get("attempt") != authority["attempt"]
        or reservation.get("plan_binding") != dict(plan_binding)
        or reservation.get("authority_binding") != dict(authority_binding)
        or reservation.get("supervisor_nonce") != supervisor_nonce
        or reservation.get("supervisor_pid") != supervisor_pid
        or reservation.get("root_creation_consumes_attempt") is not True
        or reservation.get("reservation_records_consumed_attempt") is not True
        or reservation.get("retry_authorized") is not False
        or reservation.get("resume_authorized") is not False
        or reservation.get("overwrite_authorized") is not False
        or reservation.get("refill_authorized") is not False
    ):
        raise pilot.PilotContractError("supervisor reservation identity changed")
    return output_root, kernel._relative_output_binding(  # noqa: SLF001
        absolute_binding, output_root=output_root
    )


def collect_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
    supervisor_nonce: str,
    supervisor_pid: int,
) -> tuple[dict[str, Any], Path]:
    authority, authority_binding, plan, plan_binding = load_and_validate_v1(
        plan_path=plan_path,
        expected_plan_byte_count=expected_plan_byte_count,
        expected_plan_sha256=expected_plan_sha256,
        authority_path=authority_path,
        expected_authority_byte_count=expected_authority_byte_count,
        expected_authority_sha256=expected_authority_sha256,
    )
    output_root, reservation_binding = _load_supervisor_owned_reservation_v1(
        output_root_text=plan["output_root"],
        authority=authority,
        authority_binding=authority_binding,
        plan_binding=plan_binding,
        supervisor_nonce=supervisor_nonce,
        supervisor_pid=supervisor_pid,
    )
    plan_receipt_binding = kernel._copy_exact_plan_receipt(  # noqa: SLF001
        plan_binding, output_root=output_root
    )
    state_receipt_bindings: list[dict[str, Any]] = []
    written_state_receipts: list[dict[str, Any]] = []
    render_receipt_bindings: list[dict[str, Any]] = []
    scene_metrics: list[dict[str, Any]] = []
    runtime_versions: dict[str, str] | None = None
    failure: dict[str, Any] | None = None
    stored_rgb_bytes = 0
    started = time.perf_counter()
    try:
        pilot.require_plan_bindings(plan)
        kernel._validate_python_runtime(plan)  # noqa: SLF001
        kernel._validate_execution_environment(plan)  # noqa: SLF001
        _validate_bound_scenes(plan)
        runtime_versions = kernel._capture_runtime_versions()  # noqa: SLF001
        runtime = kernel._runtime_imports()  # noqa: SLF001
        platform = runtime["load_platform_manifest"](
            plan["runtime_bindings"]["platform_manifest"]["path"]
        )
        resolved_urdf = runtime["resolve_go2_urdf"](dict(platform), REPO_ROOT)
        if pilot.file_binding(resolved_urdf) != plan["runtime_bindings"]["go2_urdf"]:
            raise pilot.PilotContractError("platform resolves a different Go2 URDF")
        registry = runtime["PrimitiveRegistry"].from_yaml(
            plan["runtime_bindings"]["primitive_registry"]["path"]
        )
        action_blocks = kernel._load_action_blocks(  # noqa: SLF001
            plan=plan,
            registry=registry,
            expand=runtime["expand_primitive_to_block"],
        )
        states_by_scene: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for state in plan["states"]:
            states_by_scene[(str(state["role"]), str(state["scene_id"]))].append(state)
        for (role, scene_id), states in states_by_scene.items():
            scene_dir = output_root / "scenes" / role / scene_id
            receipts, frames, quality, sentinels, metrics = kernel._collect_scene(  # noqa: SLF001
                plan=plan,
                states=states,
                runtime=runtime,
                platform=platform,
                registry=registry,
                action_blocks=action_blocks,
            )
            stored_rgb_bytes += sum(int(frame["byte_count"]) for frame in frames)
            if stored_rgb_bytes > int(
                authority["caps"]["stored_rgb_byte_ceiling"]
            ):
                raise pilot.PilotContractError(
                    "stored RGB byte ceiling exceeded during collection"
                )
            render_receipt = {
                "schema": pilot.TEXTURED_V03_LIVE_RENDER_RECEIPT_V3_SCHEMA,
                "attempt_id": str(plan["attempt_id"]),
                "status": "RENDER_COMPLETE",
                "physics_validated": False,
                "citable_as_scientific_evidence": False,
                "scene": {
                    "role": role,
                    "scene_id": scene_id,
                    "family": str(states[0]["family"]),
                    "scene_manifest_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                        states[0], binding_name="scene_manifest_binding", output_root=output_root
                    ),
                    "scene_genesis_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                        states[0], binding_name="scene_genesis_binding", output_root=output_root
                    ),
                },
                **_validated_render_receipt_identity_v1(
                    plan=plan,
                    metrics=metrics,
                ),
                "frame_receipts": frames,
                "quality_audits": quality,
                "render_sentinel_audits": sentinels,
            }
            render_binding = pilot.write_json_exclusive(
                scene_dir / "live_render_receipt.json", render_receipt
            )
            render_binding = kernel._relative_output_binding(  # noqa: SLF001
                render_binding, output_root=output_root
            )
            render_receipt_bindings.append(render_binding)
            for receipt in receipts:
                receipt["render_receipt_binding"] = render_binding
                state_binding = pilot.write_json_exclusive(
                    scene_dir / "state_receipts" / f"{receipt['state']['state_id']}.json",
                    receipt,
                )
                state_binding = kernel._relative_output_binding(  # noqa: SLF001
                    state_binding, output_root=output_root
                )
                state_receipt_bindings.append(state_binding)
                written_state_receipts.append(receipt)
            scene_metrics.append(metrics)
        elapsed = time.perf_counter() - started
        if elapsed > float(authority["caps"]["wall_seconds"]):
            raise pilot.PilotContractError("authority wall_seconds cap exceeded")
        if (
            sum(int(row["native_render_calls"]) for row in scene_metrics)
            != int(authority["caps"]["native_render_calls"])
            or sum(int(row["rgb_render_calls"]) for row in scene_metrics)
            != int(authority["caps"]["rgb_render_calls"])
            or sum(
                int(row["auxiliary_depth_render_calls"])
                for row in scene_metrics
            )
            != int(authority["caps"]["auxiliary_depth_render_calls"])
            or sum(int(row["stored_rgb_frames"]) for row in scene_metrics)
            != int(authority["caps"]["stored_rgb_frames"])
        ):
            raise pilot.PilotContractError("observed render work disagrees with caps")
        if stored_rgb_bytes > int(authority["caps"]["stored_rgb_byte_ceiling"]):
            raise pilot.PilotContractError("stored RGB byte ceiling exceeded")
    except Exception as exc:  # fail closed while preserving one-shot evidence
        failure = kernel._failure_receipt(exc)  # noqa: SLF001

    roles: dict[str, int] = defaultdict(int)
    scene_keys: set[tuple[str, str]] = set()
    candidate_branches = 0
    sentinel_branches = 0
    context_frames = 0
    for receipt in written_state_receipts:
        role = str(receipt["state"]["role"])
        roles[role] += 1
        scene_keys.add((role, str(receipt["state"]["scene_id"])))
        candidate_branches += sum(branch["kind"] == "candidate" for branch in receipt["branches"])
        sentinel_branches += sum(branch["kind"] == "sentinel" for branch in receipt["branches"])
        context_frames += len(receipt["context"]["frame_identities"])
    observed = {
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
        "review_binding": authority["review_binding"],
        "reservation_binding": reservation_binding,
        "caps": dict(authority["caps"]),
        "execution_contract": dict(plan["execution_contract"]),
        "runtime_versions": runtime_versions,
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": list(authority["source_bindings"]),
        "expected_counts": dict(plan["expected_counts"]),
        "observed_counts": observed,
        "scene_materialization": None,
        "state_receipt_bindings": state_receipt_bindings,
        "render_receipt_bindings": render_receipt_bindings,
        "scene_metrics": scene_metrics,
        "visual_domain_limitation": (
            "textured-v03 exact historical RGB call plus a separate transient depth-only "
            "quality render; hard near-wall or low-texture navigation observations are "
            "not silently screened or refilled"
        ),
        "collection_wall_seconds": time.perf_counter() - started,
        "failure": failure,
    }
    result_path = output_root / "physics_result.json"
    pilot.write_json_exclusive(result_path, result)
    return result, result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--expected-plan-byte-count", required=True, type=int)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--authority", required=True, type=Path)
    parser.add_argument("--expected-authority-byte-count", required=True, type=int)
    parser.add_argument("--expected-authority-sha256", required=True)
    parser.add_argument("--supervisor-nonce", required=True)
    parser.add_argument("--supervisor-pid", required=True, type=int)
    args = parser.parse_args(argv)
    result, path = collect_v1(
        plan_path=args.plan,
        expected_plan_byte_count=args.expected_plan_byte_count,
        expected_plan_sha256=args.expected_plan_sha256,
        authority_path=args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
        supervisor_nonce=args.supervisor_nonce,
        supervisor_pid=args.supervisor_pid,
    )
    print(json.dumps({"status": result["status"], "physics_result": str(path)}, sort_keys=True))
    return 0 if result["status"] == "PHYSICS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_load_supervisor_owned_reservation_v1",
    "collect_v1",
    "load_and_validate_v1",
]
