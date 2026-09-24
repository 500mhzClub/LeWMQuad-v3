#!/usr/bin/env python3
"""Read-only task-relevance check for the consumed visual-parity attempt.

This does not revise the preregistered exact-parity FAIL.  It asks the narrower
development question whether the bound candidate images are close enough to
the historical inputs for the frozen downstream spatial encoder to preserve
pair identity.  No rendering, training, selection, or execution is authorized.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import evaluate_go2_world_model_visual_domain_parity_v1 as parity  # noqa: E402
from scripts import run_go2_world_model_visual_domain_parity_authorized_v1 as supervisor  # noqa: E402


SCHEMA = "lewm_go2_world_model_visual_domain_parity_task_relevant_input_adequacy_v1"
PASS_STATUS = "PASS_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY"
FAIL_STATUS = "FAIL_TASK_RELEVANT_INPUT_ADEQUACY_DEVELOPMENT_ONLY"
PROGRESSION_SCHEMA = "go2_world_model_progression_v1_analysis_v1"
PROGRESSION_STATUS = "PASS_COMPLETE_FIXED_COMPARISON_ANALYSIS"
ARMS = ("masked_plain", "masked_delta", "full_plain", "full_delta")
SEEDS = (2026080201, 2026080202, 2026080203)
EXPECTED_INVENTORY_FILES = 141
DESCRIPTOR_DIMENSION = 1536
THRESHOLDS = {
    "required_candidate_duplicate_exact_match_count": 32,
    "maximum_reference_candidate_normalized_l1": 1.0 / 255.0,
    "minimum_reference_candidate_rgb_ssim": 0.99,
    "required_paired_nearest_neighbour_retrieval_count": 32,
    "maximum_worst_paired_to_nearest_nonself_descriptor_distance_ratio": 0.1,
    "required_progression_snapshot_rehash_count": 12,
    "required_consumed_inventory_file_count": EXPECTED_INVENTORY_FILES,
}


class TaskRelevanceEvaluationError(RuntimeError):
    """Raised when immutable input custody cannot be established."""


@contextmanager
def _authority_sources_at_frozen_commit(authority: Mapping[str, Any]):
    """Let the original authority validator re-open source bytes at its commit.

    A consumed attempt must not become unreadable merely because a source file
    was subsequently edited.  All non-source bindings continue to be checked
    against the live filesystem by the original validator.
    """

    from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector

    rows = authority.get("source_bindings")
    commit = authority.get("source_commit")
    if not isinstance(rows, list) or not isinstance(commit, str):
        raise TaskRelevanceEvaluationError("authority source closure is malformed")
    frozen = {}
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != {"name", "binding"}:
            raise TaskRelevanceEvaluationError("authority source row is malformed")
        binding = row["binding"]
        if not isinstance(binding, Mapping):
            raise TaskRelevanceEvaluationError("authority source binding is malformed")
        key = (
            str(binding.get("path")),
            str(binding.get("file_sha256")),
            binding.get("byte_count"),
        )
        collector._binding_at_commit(  # noqa: SLF001
            binding, commit=commit, label=f"frozen authority source {row['name']}"
        )
        frozen[key] = dict(binding)

    original_require = pilot.require_binding
    original_file_binding = pilot.file_binding
    frozen_by_path = {str(Path(row[0]).resolve(strict=False)): value for row, value in frozen.items()}

    def require_historical(value: object, *, label: str) -> dict[str, Any]:
        if isinstance(value, Mapping):
            key = (
                str(value.get("path")),
                str(value.get("file_sha256")),
                value.get("byte_count"),
            )
            if key in frozen:
                return dict(frozen[key])
        return original_require(value, label=label)

    def file_binding_historical(path: Path) -> dict[str, Any]:
        selected = str(Path(path).resolve(strict=False))
        if selected in frozen_by_path:
            return dict(frozen_by_path[selected])
        return original_file_binding(path)

    pilot.require_binding = require_historical
    pilot.file_binding = file_binding_historical
    try:
        yield
    finally:
        pilot.require_binding = original_require
        pilot.file_binding = original_file_binding


def _binding(path: Path, sha256: str, byte_count: int) -> dict[str, Any]:
    return {
        "path": str(Path(path).resolve(strict=False)),
        "file_sha256": sha256,
        "byte_count": byte_count,
    }


def _read_json(binding: Mapping[str, Any], *, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        value, actual = pilot.read_bound_json(
            Path(str(binding["path"])),
            expected_sha256=str(binding["file_sha256"]),
            expected_byte_count=int(binding["byte_count"]),
            label=label,
        )
    except (KeyError, TypeError, ValueError, OSError, pilot.PilotContractError) as exc:
        raise TaskRelevanceEvaluationError(f"{label} binding failed") from exc
    return dict(value), actual


def _rehash(binding: Mapping[str, Any], *, label: str) -> dict[str, Any]:
    try:
        _raw, actual = pilot.read_bound_bytes(
            Path(str(binding["path"])),
            expected_sha256=str(binding["file_sha256"]),
            expected_byte_count=int(binding["byte_count"]),
            label=label,
        )
    except (KeyError, TypeError, ValueError, OSError, pilot.PilotContractError) as exc:
        raise TaskRelevanceEvaluationError(f"{label} binding failed") from exc
    return actual


def progression_snapshot_bindings_v1(
    analysis: Mapping[str, Any],
) -> list[dict[str, Any]]:
    required = {
        "schema", "status", "development_only",
        "citable_as_world_model_usefulness_evidence", "input_result",
        "configuration", "decoder_anchor_by_seed", "contrasts",
        "proxy_routing", "terminal_snapshot_bindings", "uncertainty_limit",
    }
    panel = analysis.get("terminal_snapshot_bindings")
    if (
        set(analysis) != required
        or analysis.get("schema") != PROGRESSION_SCHEMA
        or analysis.get("status") != PROGRESSION_STATUS
        or analysis.get("development_only") is not True
        or analysis.get("citable_as_world_model_usefulness_evidence") is not False
        or not isinstance(panel, Mapping)
        or set(panel) != {str(seed) for seed in SEEDS}
    ):
        raise TaskRelevanceEvaluationError("progression analysis identity changed")
    bindings: list[dict[str, Any]] = []
    for seed in SEEDS:
        seed_panel = panel[str(seed)]
        if not isinstance(seed_panel, Mapping) or set(seed_panel) != set(ARMS):
            raise TaskRelevanceEvaluationError("progression snapshot panel changed")
        for arm in ARMS:
            declared = seed_panel[arm]
            if not isinstance(declared, Mapping) or set(declared) != {
                "path", "byte_count", "sha256"
            }:
                raise TaskRelevanceEvaluationError("progression snapshot binding changed")
            actual = _rehash(
                {
                    "path": declared["path"],
                    "byte_count": declared["byte_count"],
                    "file_sha256": declared["sha256"],
                },
                label=f"progression snapshot {arm}/{seed}",
            )
            bindings.append({"arm": arm, "seed": seed, **actual})
    return bindings


def _validate_terminal_failure(
    terminal: Mapping[str, Any],
    *,
    terminal_binding: Mapping[str, Any],
    plan: Mapping[str, Any],
    plan_binding: Mapping[str, Any],
    authority: Mapping[str, Any],
    authority_binding: Mapping[str, Any],
) -> None:
    required = {
        "schema", "status", "authority_granted_by_this_document",
        "scientific_claim_granted_by_this_document", "authorizes_retry_or_resume",
        "plan_binding", "authority_binding", "reservation_binding",
        "reservation_path", "root_creation_consumes_attempt",
        "reservation_records_consumed_attempt", "wall_seconds", "failed_at", "failure",
    }
    root = Path(str(plan["output_root"]))
    wall = terminal.get("wall_seconds")
    if (
        set(terminal) != required
        or terminal.get("schema") != supervisor.TERMINAL_SCHEMA
        or terminal.get("status") != supervisor.TERMINAL_FAILURE_STATUS
        or terminal.get("authority_granted_by_this_document") is not False
        or terminal.get("scientific_claim_granted_by_this_document") is not False
        or terminal.get("authorizes_retry_or_resume") is not False
        or terminal.get("root_creation_consumes_attempt") is not True
        or terminal.get("reservation_records_consumed_attempt") is not True
        or terminal.get("plan_binding") != plan_binding
        or terminal.get("authority_binding") != authority_binding
        or terminal.get("reservation_path") != str(root / "reservation.json")
        or terminal.get("failure") != {
            "type": "VisualDomainParitySupervisionError",
            "message": "visual-domain parity evaluator did not pass exactly",
        }
        or isinstance(wall, bool)
        or not isinstance(wall, (int, float))
        or not math.isfinite(float(wall))
        or not 0.0 <= float(wall) <= float(authority["caps"]["wall_seconds"])
        or Path(str(terminal_binding["path"])) != root / "terminal_failure.json"
    ):
        raise TaskRelevanceEvaluationError("terminal failure identity changed")
    supervisor._require_aware_iso8601(  # noqa: SLF001
        terminal.get("failed_at"), label="parity terminal failure time"
    )


def consumed_inventory_v1(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = supervisor._canonical_directory(  # noqa: SLF001
        Path(str(plan["output_root"])), label="consumed parity root"
    )
    expected_top = {
        "reservation.json", "generation_receipt.json", "candidate_panel.json",
        "parity_result.json", "terminal_failure.json", "scenes",
    }
    if {path.name for path in root.iterdir()} != expected_top:
        raise TaskRelevanceEvaluationError("consumed parity top-level inventory changed")
    files = [
        supervisor._nofollow_regular(root / name, label=f"consumed parity {name}")  # noqa: SLF001
        for name in expected_top - {"scenes"}
    ]
    scenes_root = supervisor._canonical_directory(  # noqa: SLF001
        root / "scenes", label="consumed parity scenes"
    )
    expected_scenes = {
        supervisor._expected_scene_dir(plan, index).name  # noqa: SLF001
        for index in range(len(plan["scenes"]))
    }
    if {path.name for path in scenes_root.iterdir()} != expected_scenes:
        raise TaskRelevanceEvaluationError("consumed parity scene inventory changed")
    for scene_index, scene in enumerate(plan["scenes"]):
        scene_root = supervisor._canonical_directory(  # noqa: SLF001
            supervisor._expected_scene_dir(plan, scene_index),  # noqa: SLF001
            label=f"consumed parity scene {scene_index}",
        )
        if {path.name for path in scene_root.iterdir()} != {"rows", "scene_result.json"}:
            raise TaskRelevanceEvaluationError("consumed parity scene files changed")
        files.append(supervisor._nofollow_regular(  # noqa: SLF001
            scene_root / "scene_result.json", label="consumed parity scene result"
        ))
        rows_root = supervisor._canonical_directory(  # noqa: SLF001
            scene_root / "rows", label="consumed parity pose rows"
        )
        expected_poses = {f"pose_{int(pose['pose_index']):02d}" for pose in scene["poses"]}
        if {path.name for path in rows_root.iterdir()} != expected_poses:
            raise TaskRelevanceEvaluationError("consumed parity pose inventory changed")
        for pose in scene["poses"]:
            pose_root = supervisor._canonical_directory(  # noqa: SLF001
                rows_root / f"pose_{int(pose['pose_index']):02d}",
                label="consumed parity pose",
            )
            expected = {
                "candidate.png", "duplicate.png",
                "candidate_receipt.json", "duplicate_receipt.json",
            }
            if {path.name for path in pose_root.iterdir()} != expected:
                raise TaskRelevanceEvaluationError("consumed parity render inventory changed")
            files.extend(
                supervisor._nofollow_regular(  # noqa: SLF001
                    pose_root / name, label="consumed parity render leaf"
                )
                for name in expected
            )
    if len(files) != EXPECTED_INVENTORY_FILES:
        raise TaskRelevanceEvaluationError("consumed parity file count changed")
    result = []
    for path in sorted(files, key=lambda value: value.relative_to(root).as_posix()):
        binding = pilot.file_binding(path)
        result.append({"relative_path": path.relative_to(root).as_posix(), **binding})
    return result


def descriptor_retrieval_metrics_v1(
    reference_descriptors: object,
    candidate_descriptors: object,
) -> dict[str, Any]:
    reference = np.asarray(reference_descriptors, dtype=np.float64)
    candidate = np.asarray(candidate_descriptors, dtype=np.float64)
    if (
        reference.shape != (32, DESCRIPTOR_DIMENSION)
        or candidate.shape != reference.shape
        or not np.isfinite(reference).all()
        or not np.isfinite(candidate).all()
    ):
        raise TaskRelevanceEvaluationError("latent descriptor panel changed")
    distances = np.sqrt(np.mean(
        (reference[:, None, :] - candidate[None, :, :]) ** 2, axis=2
    ))
    nearest = np.argmin(distances, axis=1)
    own = np.diag(distances)
    masked = distances.copy()
    masked[np.arange(32), np.arange(32)] = np.inf
    nonself = masked.min(axis=1)
    if not np.isfinite(nonself).all() or bool((nonself <= 0.0).any()):
        return {
            "descriptor_dimension": DESCRIPTOR_DIMENSION,
            "paired_nearest_neighbour_retrieval_count": int(
                np.count_nonzero(nearest == np.arange(32))
            ),
            "maximum_paired_descriptor_distance": float(own.max()),
            "minimum_nearest_nonself_descriptor_distance": float(nonself.min()),
            "worst_paired_to_nearest_nonself_descriptor_distance_ratio": None,
        }
    return {
        "descriptor_dimension": DESCRIPTOR_DIMENSION,
        "paired_nearest_neighbour_retrieval_count": int(
            np.count_nonzero(nearest == np.arange(32))
        ),
        "maximum_paired_descriptor_distance": float(own.max()),
        "minimum_nearest_nonself_descriptor_distance": float(nonself.min()),
        "worst_paired_to_nearest_nonself_descriptor_distance_ratio": float(
            np.max(own / nonself)
        ),
    }


def _same_parity_recomputation(
    stored: Mapping[str, Any], recomputed: Mapping[str, Any]
) -> bool:
    """Compare exact structure while allowing one-ULP diagnostic float drift."""

    left = dict(stored)
    right = dict(recomputed)
    left_measurements = left.get("measurements")
    right_measurements = right.get("measurements")
    if not isinstance(left_measurements, Mapping) or not isinstance(
        right_measurements, Mapping
    ) or set(left_measurements) != set(right_measurements):
        return False
    for name in (
        "maximum_reference_candidate_normalized_l1",
        "minimum_reference_candidate_rgb_ssim",
    ):
        if not math.isclose(
            float(left_measurements[name]),
            float(right_measurements[name]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            return False
    exact_left = dict(left_measurements)
    exact_right = dict(right_measurements)
    for name in (
        "maximum_reference_candidate_normalized_l1",
        "minimum_reference_candidate_rgb_ssim",
    ):
        exact_right[name] = exact_left[name]
    if pilot.canonical_json_bytes(exact_left) != pilot.canonical_json_bytes(exact_right):
        return False
    right["measurements"] = left["measurements"]
    return pilot.canonical_json_bytes(left) == pilot.canonical_json_bytes(right)


def _descriptors(
    source_panel: Mapping[str, Any], candidate_panel: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    import torch
    from lewm.benchmarks import go2_rgb_recurrent_patch_memory_temporal_jepa_v1 as masks
    from lewm.datasets import go2_explicit_plan_discounted_successor_state_v27 as h6
    from scripts import dev_probe_counterfactual_action_fidelity as probe
    from scripts import evaluate_go2_world_model_counterfactual_action_regret_v1 as downstream

    source_rows = source_panel.get("rows")
    candidate_rows = candidate_panel.get("rows")
    if (
        not isinstance(source_rows, list) or not isinstance(candidate_rows, list)
        or len(source_rows) != 32 or len(candidate_rows) != 32
        or [row.get("pair_id") for row in source_rows]
        != [row.get("pair_id") for row in candidate_rows]
    ):
        raise TaskRelevanceEvaluationError("descriptor pair panel changed")

    tensors = []
    for domain, rows in (("reference", source_rows), ("candidate", candidate_rows)):
        for row in rows:
            binding = row["rgb_binding"]
            raw, actual = pilot.read_bound_bytes(
                Path(str(binding["path"])),
                expected_sha256=str(binding["file_sha256"]),
                expected_byte_count=int(binding["byte_count"]),
                label=f"{domain} descriptor RGB {row['pair_id']}",
            )
            if actual != binding:
                raise TaskRelevanceEvaluationError("descriptor RGB binding changed")
            tensors.append(h6.rectify_h6_rgb_bytes(raw))

    device = torch.device("cpu")
    model, label, identity = probe.build_model(None, device)
    if label != "predecessor_init" or identity.get("scaled_snapshot") is not None:
        raise TaskRelevanceEvaluationError("frozen predecessor encoder identity changed")
    target_masks = [
        masks.batched_mask_indices("val", [row], device=device)[0][0]
        for row in downstream.MASK_ROW_INDICES
    ]
    descriptors: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(tensors), 8):
            batch = torch.stack(tensors[start : start + 8]).to(device)
            full = model.encode_target_full_frame(batch)
            for row in full:
                masked = torch.stack([row[index] for index in target_masks])
                descriptor = downstream.masked_token_descriptor_v1(
                    masked.detach().cpu().numpy()
                )
                if descriptor.shape != (DESCRIPTOR_DIMENSION,):
                    raise TaskRelevanceEvaluationError("descriptor dimension changed")
                descriptors.append(descriptor)
    predecessor = identity["predecessor"]
    predecessor_binding = {
        "path": predecessor["path"],
        "byte_count": predecessor["byte_count"],
        "file_sha256": predecessor["sha256"],
    }
    return (
        np.stack(descriptors[:32]),
        np.stack(descriptors[32:]),
        predecessor_binding,
    )


def evaluate_task_relevance_v1(
    *,
    parity_result_binding: Mapping[str, Any],
    terminal_failure_binding: Mapping[str, Any],
    progression_analysis_binding: Mapping[str, Any],
) -> dict[str, Any]:
    stored_result, actual_result = _read_json(
        parity_result_binding, label="immutable exact parity result"
    )
    terminal, actual_terminal = _read_json(
        terminal_failure_binding, label="parity terminal failure"
    )
    plan_binding = terminal.get("plan_binding")
    authority_binding = terminal.get("authority_binding")
    if not isinstance(plan_binding, Mapping) or not isinstance(authority_binding, Mapping):
        raise TaskRelevanceEvaluationError("terminal chain bindings are absent")
    frozen_authority, _frozen_authority_binding = _read_json(
        authority_binding, label="frozen parity authority"
    )
    try:
        with _authority_sources_at_frozen_commit(frozen_authority):
            plan, actual_plan, authority, actual_authority, review, review_binding = (
                supervisor.load_and_validate_chain_v1(
                    plan_path=Path(str(plan_binding["path"])),
                    expected_plan_sha256=str(plan_binding["file_sha256"]),
                    expected_plan_byte_count=int(plan_binding["byte_count"]),
                    authority_path=Path(str(authority_binding["path"])),
                    expected_authority_sha256=str(authority_binding["file_sha256"]),
                    expected_authority_byte_count=int(authority_binding["byte_count"]),
                    require_fresh_output=False,
                )
            )
    except (KeyError, supervisor.VisualDomainParitySupervisionError) as exc:
        raise TaskRelevanceEvaluationError("parity plan/authority chain changed") from exc
    _validate_terminal_failure(
        terminal, terminal_binding=actual_terminal, plan=plan,
        plan_binding=actual_plan, authority=authority,
        authority_binding=actual_authority,
    )
    root = Path(str(plan["output_root"]))
    if Path(str(actual_result["path"])) != root / "parity_result.json":
        raise TaskRelevanceEvaluationError("parity result escaped consumed root")

    reservation_binding = terminal.get("reservation_binding")
    if not isinstance(reservation_binding, Mapping):
        raise TaskRelevanceEvaluationError("terminal reservation binding is absent")
    reservation = supervisor._validate_reservation(  # noqa: SLF001
        binding=reservation_binding, plan=plan, plan_binding=actual_plan,
        authority=authority, authority_binding=actual_authority, capability=None,
    )

    source_binding = stored_result.get("source_rgb_reference_binding")
    candidate_binding = stored_result.get("candidate_rgb_panel_binding")
    if not isinstance(source_binding, Mapping) or not isinstance(candidate_binding, Mapping):
        raise TaskRelevanceEvaluationError("parity panel bindings are absent")
    source_panel, actual_source = parity._read_bound_json(  # noqa: SLF001
        source_binding, label="task-relevance source panel"
    )
    candidate_panel, actual_candidate = parity._read_bound_json(  # noqa: SLF001
        candidate_binding, label="task-relevance candidate panel"
    )
    with _authority_sources_at_frozen_commit(authority):
        recomputed = parity.evaluate_v1(
            source_panel=source_panel, source_panel_binding=actual_source,
            candidate_panel=candidate_panel, candidate_panel_binding=actual_candidate,
        )
    if (
        not _same_parity_recomputation(stored_result, recomputed)
        or stored_result.get("status") != parity.FAIL_STATUS
        or stored_result.get("scientific_claim_granted_by_this_document") is not False
    ):
        raise TaskRelevanceEvaluationError("immutable exact parity FAIL changed")

    generation_binding = recomputed["candidate_producer_lineage"]["generation_receipt_binding"]
    generation, actual_generation = _read_json(
        generation_binding, label="parity generation receipt"
    )
    scene_bindings = []
    combined_rows = []
    for scene_index in range(len(plan["scenes"])):
        scene_result, scene_binding = supervisor._validate_scene_result(  # noqa: SLF001
            plan=plan, plan_binding=actual_plan,
            authority_binding=actual_authority, scene_index=scene_index,
        )
        scene_bindings.append(scene_binding)
        combined_rows.extend(scene_result["render_rows"])
    combined_rows.sort(key=lambda row: row["pair_id"])
    if generation.get("render_rows") != combined_rows:
        raise TaskRelevanceEvaluationError("scene/generation render rows changed")

    inventory = consumed_inventory_v1(plan)
    inventory_by_path = {row["path"]: row for row in inventory}
    for binding in (
        actual_result, actual_terminal, reservation, actual_generation,
        actual_candidate, *scene_bindings,
    ):
        row = inventory_by_path.get(binding["path"])
        if row is None or any(row[key] != binding[key] for key in (
            "path", "file_sha256", "byte_count"
        )):
            raise TaskRelevanceEvaluationError("consumed inventory binding changed")

    analysis, actual_analysis = _read_json(
        progression_analysis_binding, label="progression analysis"
    )
    snapshots = progression_snapshot_bindings_v1(analysis)
    reference_descriptors, candidate_descriptors, predecessor = _descriptors(
        source_panel, candidate_panel
    )
    latent = descriptor_retrieval_metrics_v1(
        reference_descriptors, candidate_descriptors
    )
    pixel = dict(recomputed["measurements"])
    ratio = latent["worst_paired_to_nearest_nonself_descriptor_distance_ratio"]
    passed = (
        pixel["candidate_duplicate_exact_match_count"]
        == THRESHOLDS["required_candidate_duplicate_exact_match_count"]
        and pixel["maximum_reference_candidate_normalized_l1"]
        <= THRESHOLDS["maximum_reference_candidate_normalized_l1"]
        and pixel["minimum_reference_candidate_rgb_ssim"]
        >= THRESHOLDS["minimum_reference_candidate_rgb_ssim"]
        and latent["paired_nearest_neighbour_retrieval_count"]
        == THRESHOLDS["required_paired_nearest_neighbour_retrieval_count"]
        and ratio is not None
        and ratio <= THRESHOLDS[
            "maximum_worst_paired_to_nearest_nonself_descriptor_distance_ratio"
        ]
        and len(snapshots) == THRESHOLDS["required_progression_snapshot_rehash_count"]
        and len(inventory) == THRESHOLDS["required_consumed_inventory_file_count"]
    )

    render_receipts = [
        {"pair_index": index, **binding}
        for index, pair in enumerate(
            recomputed["candidate_producer_lineage"]["render_receipt_bindings"]
        )
        for binding in (pair["candidate"], pair["duplicate"])
    ]
    result = {
        "schema": SCHEMA,
        "status": PASS_STATUS if passed else FAIL_STATUS,
        "authority_granted_by_this_document": False,
        "scientific_claim_granted_by_this_document": False,
        "development_only": True,
        "protected_material_opened": False,
        "scope_limit": (
            "Bound task-input adequacy only; the exact parity FAIL remains immutable. "
            "No generalization, planning, promotion, retry, resume, render, or execution claim."
        ),
        "threshold_origin": (
            "Post-hoc engineering thresholds selected after the exact parity failure "
            "and fixed before calibration or bounded-branch use."
        ),
        "immutable_exact_parity_failure": {
            "status": parity.FAIL_STATUS,
            "preserved": True,
            "reference_candidate_exact_match_count": pixel[
                "reference_candidate_exact_match_count"
            ],
            "required_reference_candidate_exact_match_count": recomputed["thresholds"][
                "required_reference_candidate_exact_match_count"
            ],
        },
        "thresholds": dict(THRESHOLDS),
        "measurements": {
            "pixels": pixel,
            "frozen_predecessor_descriptor_retrieval": latent,
            "progression_snapshot_rehash_count": len(snapshots),
            "consumed_inventory_file_count": len(inventory),
            "consumed_inventory_total_byte_count": sum(
                row["byte_count"] for row in inventory
            ),
            "consumed_inventory_sha256": hashlib.sha256(
                pilot.canonical_json_bytes(inventory)
            ).hexdigest(),
        },
        "bindings": {
            "parity_result": actual_result,
            "terminal_failure": actual_terminal,
            "reservation": reservation,
            "generation_receipt": actual_generation,
            "source_panel": actual_source,
            "candidate_panel": actual_candidate,
            "scene_results": scene_bindings,
            "render_receipts": render_receipts,
            "consumed_inventory": inventory,
            "progression_analysis": actual_analysis,
            "progression_snapshots": snapshots,
            "frozen_predecessor": predecessor,
            "evaluator_source": pilot.file_binding(Path(__file__)),
        },
    }
    # Terminal rehash after model inference.
    for row in inventory:
        _rehash(row, label=f"terminal consumed inventory {row['relative_path']}")
    for row in snapshots:
        _rehash(row, label=f"terminal progression snapshot {row['arm']}/{row['seed']}")
    _rehash(predecessor, label="terminal frozen predecessor")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("parity-result", "terminal-failure", "progression-analysis"):
        parser.add_argument(f"--{name}", required=True, type=Path)
        parser.add_argument(f"--expected-{name}-sha256", required=True)
        parser.add_argument(f"--expected-{name}-byte-count", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = evaluate_task_relevance_v1(
        parity_result_binding=_binding(
            args.parity_result, args.expected_parity_result_sha256,
            args.expected_parity_result_byte_count,
        ),
        terminal_failure_binding=_binding(
            args.terminal_failure, args.expected_terminal_failure_sha256,
            args.expected_terminal_failure_byte_count,
        ),
        progression_analysis_binding=_binding(
            args.progression_analysis, args.expected_progression_analysis_sha256,
            args.expected_progression_analysis_byte_count,
        ),
    )
    output = Path(args.output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite task-relevance result: {output}")
    pilot.write_json_exclusive(output, result)
    return 0 if result["status"] == PASS_STATUS else 1


if __name__ == "__main__":
    raise SystemExit(main())
