#!/usr/bin/env python3
"""Collect the one-shot 64-scene recurrent-dynamics replication panel.

This collector deliberately leaves the historical bounded-branch collector
unchanged.  It reuses that collector's reviewed validation and receipt helpers
and the counterfactual pilot's reviewed runtime primitives, while changing the
scene unit to exactly four states.  Collection is development-only and cannot
join, refill, resume, or access a successor or protected role.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_world_model_bounded_branch_experiment_authorized_v1 as bounded  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as kernel  # noqa: E402


AUTHORITY_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_execution_authority_v1"
)
AUTHORITY_STATUS = (
    "AUTHORIZED_ONE_SCENE_DIVERSITY_RECURRENT_REPLICATION_V1"
)
RESERVATION_SCHEMA = (
    "lewm_go2_scene_diversity_recurrent_replication_v1_collection_reservation_v1"
)
RESERVATION_STATUS = "RESERVED_ONE_SHOT_COLLECTION_CONSUMED"

EXPECTED_HISTORY_PANEL = (
    (3, 3),
    (5, 5),
    (6, 6),
    (7, 7),
    (8, 8),
    (0, 1),
    (2, 3),
    (4, 5),
)
EXPECTED_COUNTS = {
    "scenes": 64,
    "states": 256,
    "roles": {"eval": 128, "train": 128},
    "actions": 9,
    "candidate_branches": 2304,
    "sentinel_branches": 0,
    "total_branches": 2304,
    "context_frames": 768,
    "target_frames": 2304,
}
EXPECTED_CAPS = {
    "auxiliary_depth_render_calls": 3072,
    "candidate_branch_simulated_seconds": 1152.0,
    "candidate_branches": 2304,
    "native_render_calls": 3072,
    "policy_steps_per_lane": 75,
    "rgb_render_calls": 3072,
    "scenes": 64,
    "selected_device_vram_byte_ceiling": 16_977_405_952,
    "sentinel_branches": 0,
    "states": 256,
    "stored_rgb_byte_ceiling": 512 * 1024 * 1024,
    "stored_rgb_frames": 3072,
    "total_branches": 2304,
    "total_lane_physics_steps": 1_728_000,
    "total_lane_policy_steps": 172_800,
    "total_lane_simulated_seconds_including_common_prefix": 3456.0,
    "wall_seconds": 7200.0,
}
EXPECTED_PERMISSIONS = {
    "train_receipt_access": True,
    "train_context_rgb_access": True,
    "eval_receipt_access_after_checkpoint": True,
    "eval_context_rgb_access_after_checkpoint": True,
    "successor_rgb_access": False,
    "data_generation": True,
    "sealed_or_protected_access": False,
    "retry_resume_overwrite": False,
}
AUTHORITY_FIELDS = {
    "schema",
    "status",
    "attempt_id",
    "attempt_root",
    "collection_root",
    "plan_binding",
    "preregistration_binding",
    "source_review_binding",
    "source_bindings",
    "dino",
    "config",
    "caps",
    "permissions",
}


def _standard_binding(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Translate the pilot's historical binding spelling at one boundary."""

    return {
        "path": str(binding["path"]),
        "sha256": str(binding["file_sha256"]),
        "byte_count": int(binding["byte_count"]),
    }


def _reject_protected_path(path: Path, *, label: str) -> None:
    lowered = [part.lower() for part in Path(path).parts]
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
        raise pilot.PilotContractError(f"{label} path is custody-protected")


def _validate_scene_diversity_plan_v1(plan: Mapping[str, Any]) -> dict[str, Any]:
    """Require the exact prospective data intervention, not merely its size."""

    if (
        plan.get("purpose") != "bounded_wm_a_pilot"
        or plan.get("states_per_scene") != 4
        or plan.get("history_blocks") != 2
        or plan.get("expected_counts") != EXPECTED_COUNTS
    ):
        raise pilot.PilotContractError("scene-diversity plan identity or counts changed")
    states = plan.get("states")
    if not isinstance(states, list) or len(states) != EXPECTED_COUNTS["states"]:
        raise pilot.PilotContractError("scene-diversity state panel changed")

    scenes: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    scene_counts: Counter[tuple[str, str]] = Counter()
    histories: Counter[tuple[str, str, tuple[int, int]]] = Counter()
    seen_scene_keys: set[tuple[str, str, str]] = set()
    for state in states:
        role = str(state["role"])
        family = str(state["family"])
        scene_id = str(state["scene_id"])
        scene_key = (role, family, scene_id)
        scenes[scene_key].append(state)
        if scene_key not in seen_scene_keys:
            seen_scene_keys.add(scene_key)
            scene_counts[(role, family)] += 1
        history = tuple(state["history_action_ids"])
        if history not in EXPECTED_HISTORY_PANEL:
            raise pilot.PilotContractError("scene-diversity history tape changed")
        histories[(role, family, history)] += 1
        if state["candidate_action_ids"] != list(range(pilot.ACTION_COUNT)):
            raise pilot.PilotContractError("scene-diversity candidate grid changed")
        for binding_name in ("scene_manifest_binding", "scene_genesis_binding"):
            binding = state[binding_name]
            _reject_protected_path(Path(str(binding["path"])), label=binding_name)

    expected_role_families = {
        (role, family) for role in ("train", "eval") for family in pilot.FAMILIES
    }
    if set(scene_counts) != expected_role_families or any(
        scene_counts[key] != 4 for key in expected_role_families
    ):
        raise pilot.PilotContractError(
            "scene-diversity panel must have four scenes per family and role"
        )
    if len(scenes) != 64 or any(
        len(scene_states) != 4
        or [int(state["state_index_in_scene"]) for state in scene_states]
        != [0, 1, 2, 3]
        for scene_states in scenes.values()
    ):
        raise pilot.PilotContractError(
            "scene-diversity panel must have four ordered states per scene"
        )
    if any(
        histories[(role, family, history)] != 2
        for role, family in expected_role_families
        for history in EXPECTED_HISTORY_PANEL
    ):
        raise pilot.PilotContractError(
            "scene-diversity history panel is not exactly balanced"
        )
    return dict(plan)


def _validate_output_roots_v1(
    *, authority: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[Path, Path]:
    attempt_text = authority.get("attempt_root")
    collection_text = authority.get("collection_root")
    if not isinstance(attempt_text, str) or not isinstance(collection_text, str):
        raise pilot.PilotContractError("authority output roots are malformed")
    attempt_root = Path(os.path.abspath(attempt_text))
    collection_root = Path(os.path.abspath(collection_text))
    development_root = (REPO_ROOT / ".generated/dev").resolve(strict=True)
    _reject_protected_path(attempt_root, label="attempt root")
    _reject_protected_path(collection_root, label="collection root")
    try:
        attempt_root.relative_to(development_root)
        collection_root.relative_to(attempt_root)
    except ValueError as exc:
        raise pilot.PilotContractError(
            "authority output roots escape the development attempt"
        ) from exc
    if collection_root == attempt_root:
        raise pilot.PilotContractError("collection root must be inside the attempt root")
    if str(collection_root) != str(plan.get("output_root")):
        raise pilot.PilotContractError("plan and authority collection roots disagree")
    if (
        not attempt_root.is_dir()
        or attempt_root.is_symlink()
        or collection_root.exists()
        or collection_root.is_symlink()
        or collection_root.parent != attempt_root
    ):
        raise pilot.PilotContractError(
            "collection root is not a fresh direct child of the attempt root"
        )
    return attempt_root.resolve(strict=True), collection_root


def _validate_authority_v1(
    authority: Mapping[str, Any],
    *,
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        not isinstance(authority, Mapping)
        or set(authority) != AUTHORITY_FIELDS
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("status") != AUTHORITY_STATUS
        or authority.get("attempt_id") != plan.get("attempt_id")
        or authority.get("plan_binding") != _standard_binding(plan_binding)
        or authority.get("caps") != EXPECTED_CAPS
        or authority.get("permissions") != EXPECTED_PERMISSIONS
    ):
        raise pilot.PilotContractError("scene-diversity execution authority changed")
    # The runner owns review, source, DINO, and model-config validation.  The
    # collector still requires their exact top-level presence so an authority
    # cannot silently shed that closure at the process boundary.
    for name in ("preregistration_binding", "source_review_binding"):
        if not isinstance(authority.get(name), Mapping):
            raise pilot.PilotContractError(f"authority {name} is absent")
    if not isinstance(authority.get("source_bindings"), Mapping):
        raise pilot.PilotContractError("authority source binding map is absent")
    if not isinstance(authority_binding, Mapping):
        raise pilot.PilotContractError("authority caller binding is absent")
    _validate_output_roots_v1(authority=authority, plan=plan)
    return dict(authority)


def load_and_validate_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    _reject_protected_path(plan_path, label="scene-diversity plan")
    _reject_protected_path(authority_path, label="scene-diversity authority")
    raw_plan, plan_binding = pilot.read_bound_json(
        plan_path,
        expected_sha256=expected_plan_sha256,
        expected_byte_count=expected_plan_byte_count,
        label="scene-diversity plan",
    )
    plan = _validate_scene_diversity_plan_v1(
        copy.deepcopy(pilot.validate_plan(raw_plan))
    )
    bounded._validate_plan_parity_prerequisites_v1(plan)  # noqa: SLF001
    raw_authority, historical_authority_binding = pilot.read_bound_json(
        authority_path,
        expected_sha256=expected_authority_sha256,
        expected_byte_count=expected_authority_byte_count,
        label="scene-diversity authority",
    )
    authority_binding = _standard_binding(historical_authority_binding)
    authority = _validate_authority_v1(
        raw_authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
    )
    return authority, authority_binding, plan, plan_binding


def _create_collection_root_v1(
    *,
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    _attempt_root, output_root = _validate_output_roots_v1(
        authority=authority, plan=plan
    )
    try:
        os.mkdir(output_root, mode=0o700)
    except OSError as exc:
        raise pilot.PilotContractError(
            "could not exclusively reserve fresh collection root"
        ) from exc
    output_root = output_root.resolve(strict=True)
    reservation = {
        "schema": RESERVATION_SCHEMA,
        "status": RESERVATION_STATUS,
        "attempt_id": str(authority["attempt_id"]),
        "attempt_root": str(authority["attempt_root"]),
        "collection_root": str(output_root),
        "plan_binding": _standard_binding(plan_binding),
        "authority_binding": dict(authority_binding),
        "root_creation_consumes_attempt": True,
        "retry_authorized": False,
        "resume_authorized": False,
        "overwrite_authorized": False,
        "refill_authorized": False,
    }
    binding = pilot.write_json_exclusive(output_root / "reservation.json", reservation)
    return output_root, kernel._relative_output_binding(  # noqa: SLF001
        binding, output_root=output_root
    )


def collect_v1(
    *,
    plan_path: Path,
    expected_plan_byte_count: int,
    expected_plan_sha256: str,
    authority_path: Path,
    expected_authority_byte_count: int,
    expected_authority_sha256: str,
) -> tuple[dict[str, Any], Path]:
    authority, authority_binding, plan, plan_binding = load_and_validate_v1(
        plan_path=plan_path,
        expected_plan_byte_count=expected_plan_byte_count,
        expected_plan_sha256=expected_plan_sha256,
        authority_path=authority_path,
        expected_authority_byte_count=expected_authority_byte_count,
        expected_authority_sha256=expected_authority_sha256,
    )
    output_root, reservation_binding = _create_collection_root_v1(
        authority=authority,
        authority_binding=authority_binding,
        plan=plan,
        plan_binding=plan_binding,
    )
    plan_receipt_binding = kernel._copy_exact_plan_receipt(  # noqa: SLF001
        plan_binding, output_root=output_root
    )
    state_bindings_by_id: dict[str, dict[str, Any]] = {}
    receipts_by_id: dict[str, dict[str, Any]] = {}
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
        bounded._validate_bound_scenes(plan)  # noqa: SLF001
        runtime_versions = kernel._capture_runtime_versions()  # noqa: SLF001
        runtime = kernel._runtime_imports(textured_v03=True)  # noqa: SLF001
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
            if len(states) != 4:
                raise pilot.PilotContractError(
                    "scene-diversity runtime scene does not contain four states"
                )
            scene_dir = output_root / "scenes" / role / scene_id
            receipts, frames, quality, sentinels, metrics = kernel._collect_scene(  # noqa: SLF001
                plan=plan,
                states=states,
                runtime=runtime,
                platform=platform,
                registry=registry,
                action_blocks=action_blocks,
            )
            if [str(receipt["state"]["state_id"]) for receipt in receipts] != [
                str(state["state_id"]) for state in states
            ]:
                raise pilot.PilotContractError(
                    "scene-diversity collection changed planned state identity or order"
                )
            stored_rgb_bytes += sum(int(frame["byte_count"]) for frame in frames)
            if stored_rgb_bytes > EXPECTED_CAPS["stored_rgb_byte_ceiling"]:
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
                        states[0],
                        binding_name="scene_manifest_binding",
                        output_root=output_root,
                    ),
                    "scene_genesis_binding": kernel._scene_receipt_binding(  # noqa: SLF001
                        states[0],
                        binding_name="scene_genesis_binding",
                        output_root=output_root,
                    ),
                },
                **bounded._validated_render_receipt_identity_v1(  # noqa: SLF001
                    plan=plan, metrics=metrics
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
                state_id = str(receipt["state"]["state_id"])
                receipt["render_receipt_binding"] = render_binding
                state_binding = pilot.write_json_exclusive(
                    scene_dir / "state_receipts" / f"{state_id}.json", receipt
                )
                state_bindings_by_id[state_id] = kernel._relative_output_binding(  # noqa: SLF001
                    state_binding, output_root=output_root
                )
                receipts_by_id[state_id] = receipt
            scene_metrics.append(metrics)
            if time.perf_counter() - started > EXPECTED_CAPS["wall_seconds"]:
                raise pilot.PilotContractError("authority wall_seconds cap exceeded")

        ordered_state_ids = [str(state["state_id"]) for state in plan["states"]]
        if set(state_bindings_by_id) != set(ordered_state_ids):
            raise pilot.PilotContractError("collection did not produce the exact state panel")
        if (
            sum(int(row["native_render_calls"]) for row in scene_metrics)
            != EXPECTED_CAPS["native_render_calls"]
            or sum(int(row["rgb_render_calls"]) for row in scene_metrics)
            != EXPECTED_CAPS["rgb_render_calls"]
            or sum(int(row["auxiliary_depth_render_calls"]) for row in scene_metrics)
            != EXPECTED_CAPS["auxiliary_depth_render_calls"]
            or sum(int(row["stored_rgb_frames"]) for row in scene_metrics)
            != EXPECTED_CAPS["stored_rgb_frames"]
        ):
            raise pilot.PilotContractError("observed render work disagrees with caps")
    except Exception as exc:  # one-shot failure remains terminal evidence
        failure = kernel._failure_receipt(exc)  # noqa: SLF001

    ordered_state_ids = [str(state["state_id"]) for state in plan["states"]]
    written_state_receipts = [
        receipts_by_id[state_id]
        for state_id in ordered_state_ids
        if state_id in receipts_by_id
    ]
    state_receipt_bindings = [
        state_bindings_by_id[state_id]
        for state_id in ordered_state_ids
        if state_id in state_bindings_by_id
    ]
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
    if failure is None and observed != EXPECTED_COUNTS:
        failure = kernel._failure_receipt(  # noqa: SLF001
            pilot.PilotContractError("observed collection counts changed")
        )
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
        "reservation_binding": reservation_binding,
        "caps": dict(authority["caps"]),
        "execution_contract": dict(plan["execution_contract"]),
        "runtime_versions": runtime_versions,
        "runtime_bindings": dict(plan["runtime_bindings"]),
        "source_bindings": dict(authority["source_bindings"]),
        "expected_counts": dict(plan["expected_counts"]),
        "observed_counts": observed,
        "scene_materialization": None,
        "state_receipt_bindings": state_receipt_bindings,
        "render_receipt_bindings": render_receipt_bindings,
        "scene_metrics": scene_metrics,
        "visual_domain_limitation": (
            "textured-v03 exact historical RGB call plus a separate transient "
            "depth-only quality render; observations are not screened or refilled"
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
    args = parser.parse_args(argv)
    result, path = collect_v1(
        plan_path=args.plan,
        expected_plan_byte_count=args.expected_plan_byte_count,
        expected_plan_sha256=args.expected_plan_sha256,
        authority_path=args.authority,
        expected_authority_byte_count=args.expected_authority_byte_count,
        expected_authority_sha256=args.expected_authority_sha256,
    )
    print(
        json.dumps(
            {"status": result["status"], "physics_result": str(path)},
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "PHYSICS_COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY_SCHEMA",
    "AUTHORITY_STATUS",
    "EXPECTED_CAPS",
    "EXPECTED_COUNTS",
    "EXPECTED_HISTORY_PANEL",
    "EXPECTED_PERMISSIONS",
    "_validate_authority_v1",
    "_validate_scene_diversity_plan_v1",
    "collect_v1",
    "load_and_validate_v1",
]
