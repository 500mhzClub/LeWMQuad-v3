from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.benchmarks import go2_dynamic_cartesian_n32 as contract


PRIMARY_FILE_SHA256 = "9" * 64
IMPLEMENTATION_FILE_SHA256 = "1" * 64
PRIMARY_ATTEMPT_FILE_SHA256 = "8" * 64
REPLICATION_ATTEMPT_FILE_SHA256 = "7" * 64


def _rehash(value: dict) -> None:
    value.pop("content_sha256", None)
    value["content_sha256"] = contract.canonical_json_sha256(value)


def _manifest() -> dict:
    entries = [
        {"role": role, "path": path, "sha256": format(index + 1, "064x")}
        for index, (role, path) in enumerate(sorted(contract.IMPLEMENTATION_SOURCE_PATHS.items()))
    ]
    state_contract = _state_contract()
    core = {
        "schema": contract.IMPLEMENTATION_MANIFEST_SCHEMA,
        "binding": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["binding"],
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS[
                "attempt_control_amendment"
            ],
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "sources": {
            "entries": entries,
            "entry_count": len(entries),
            "source_map_sha256": contract.canonical_json_sha256(entries),
        },
        "tests": {
            "command": contract.IMPLEMENTATION_TEST_COMMAND,
            "passed": contract.IMPLEMENTATION_TEST_PASSED,
            "all_passed": True,
        },
        "inputs": contract.INPUT_BINDINGS,
        "resource_policy": contract.RESOURCE_POLICY,
        "model_config": contract.MODEL_CONFIG,
        "objective": contract.OBJECTIVE_CONTRACT,
        "preprocessing": contract.PREPROCESSING_CONTRACT,
        "controls": contract.CONTROL_CONTRACT,
        "projective_query_support": contract.PROJECTIVE_QUERY_SUPPORT,
        "model_initial_state_sha256": {
            "20260710": "a" * 64,
            "20260711": "b" * 64,
        },
        "model_state_contract_sha256": {
            str(seed): contract.canonical_json_sha256(state_contract)
            for seed in contract.EXPECTED_SEEDS
        },
        "schedules": contract.SCHEDULE_CONTRACT,
        "commands": contract.COMMAND_CONTRACT,
    }
    return {**core, "content_sha256": contract.canonical_json_sha256(core)}


def _attempt_file_sha256(seed: int) -> str:
    return (
        PRIMARY_ATTEMPT_FILE_SHA256
        if seed == contract.EXPECTED_SEEDS[0]
        else REPLICATION_ATTEMPT_FILE_SHA256
    )


def _attempt_marker(
    seed: int,
    *,
    primary: dict | None = None,
    primary_attempt_marker: dict | None = None,
) -> dict:
    manifest = _manifest()
    invocation = contract.authoritative_runner_invocation(
        seed=seed,
        implementation_manifest_file_sha256=IMPLEMENTATION_FILE_SHA256,
        primary_file_sha256=(None if primary is None else PRIMARY_FILE_SHA256),
        primary_attempt_marker_file_sha256=(
            None
            if primary_attempt_marker is None
            else PRIMARY_ATTEMPT_FILE_SHA256
        ),
    )
    core = {
        "schema": contract.ATTEMPT_MARKER_SCHEMA,
        "authoritative": True,
        "seed": seed,
        "created_at_utc": "2026-07-11T00:00:00+00:00",
        "invocation": invocation,
        "invocation_sha256": contract.canonical_json_sha256(invocation),
        "canonical_result_path": contract.COMMAND_CONTRACT["canonical_outputs"][
            str(seed)
        ],
        "canonical_attempt_marker_path": contract.ATTEMPT_MARKER_PATHS[seed],
        "contract": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["binding"],
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS[
                "attempt_control_amendment"
            ],
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "implementation_manifest": {
            "path": (
                f"{contract.REPOSITORY_ROOT}/docs/"
                "lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json"
            ),
            "sha256": IMPLEMENTATION_FILE_SHA256,
            "content_sha256": manifest["content_sha256"],
        },
        "seed_20260710_result": (
            None
            if primary is None
            else {
                "path": contract.COMMAND_CONTRACT["canonical_outputs"]["20260710"],
                "sha256": PRIMARY_FILE_SHA256,
                "content_sha256": primary["content_sha256"],
            }
        ),
        "seed_20260710_attempt_marker": (
            None
            if primary_attempt_marker is None
            else {
                "path": contract.ATTEMPT_MARKER_PATHS[20260710],
                "sha256": PRIMARY_ATTEMPT_FILE_SHA256,
                "content_sha256": primary_attempt_marker["content_sha256"],
            }
        ),
        "attempt_consumed": True,
        "retry_permitted": False,
        "payload_access_started": False,
    }
    return {**core, "content_sha256": contract.canonical_json_sha256(core)}


def _metrics(
    *,
    passing: bool = True,
    holdout_favorable: bool = True,
    scale: int = 1,
    nll_override: float | None = None,
) -> dict:
    strong = passing and holdout_favorable
    nll = (0.01 if strong else 0.5) if nll_override is None else nll_override
    joint_base = (
        [[10, 0, 0], [0, 20, 0], [0, 0, 10]]
        if strong
        else [[5, 3, 2], [5, 10, 5], [2, 3, 5]]
    )
    uk_base = [[10, 0], [0, 30]] if strong else [[5, 5], [15, 15]]
    fo_base = [[20, 0], [0, 10]] if strong else [[10, 10], [5, 5]]
    joint = [[item * scale for item in row] for row in joint_base]
    unknown_known = [[item * scale for item in row] for row in uk_base]
    free_occupied = [[item * scale for item in row] for row in fo_base]
    truth_support = [sum(row) for row in joint]
    cell_count = sum(truth_support)
    known_count = sum(truth_support[1:])

    def ratio(numerator: int, denominator: int) -> float | None:
        return None if denominator == 0 else numerator / denominator

    def balanced(matrix: list[list[int]]) -> float:
        return sum(matrix[index][index] / sum(row) for index, row in enumerate(matrix)) / len(matrix)

    class_recall = {
        name: ratio(joint[index][index], truth_support[index])
        for index, name in enumerate(contract.CLASS_NAMES)
    }
    class_precision = {
        name: ratio(joint[index][index], sum(row[index] for row in joint))
        for index, name in enumerate(contract.CLASS_NAMES)
    }
    distance_support_values = [2, 2, 4, 4, 8]
    distance_support = {
        name: support * scale
        for name, support in zip(
            contract.DISTANCE_BIN_NAMES, distance_support_values, strict=True
        )
    }
    distance_recall = dict.fromkeys(
        contract.DISTANCE_BIN_NAMES, 1.0 if strong else 0.5
    )
    posterior = {
        truth_name: {
            predicted_name: (
                {"p05": 0.8, "p50": 0.9, "p95": 0.99}
                if strong and truth_name == predicted_name
                else {"p05": 0.001, "p50": 0.05, "p95": 0.1}
                if strong
                else {"p05": 0.2, "p50": 0.33, "p95": 0.5}
            )
            for predicted_name in contract.CLASS_NAMES
        }
        for truth_name in contract.CLASS_NAMES
    }
    return {
        "raw_joint_nll": nll,
        "raw_joint_accuracy": ratio(
            sum(joint[index][index] for index in range(3)), cell_count
        ),
        "raw_hierarchical_balanced_nll": nll,
        "raw_unknown_known_weighted_nll": nll,
        "raw_known_free_occupied_weighted_nll": nll,
        "raw_known_free_occupied_nll": nll,
        "raw_known_free_occupied_accuracy": ratio(
            free_occupied[0][0] + free_occupied[1][1], known_count
        ),
        "cell_count": cell_count,
        "known_cell_count": known_count,
        "joint_confusion": joint,
        "unknown_known_confusion": unknown_known,
        "free_occupied_confusion": free_occupied,
        "unknown_known_balanced_accuracy": balanced(unknown_known),
        "free_occupied_balanced_accuracy": balanced(free_occupied),
        "class_recall": class_recall,
        "class_precision": class_precision,
        "free_average_precision": 1.0 if strong else 0.5,
        "occupied_average_precision": 1.0 if strong else 0.5,
        "posterior_quantiles_by_truth_class": posterior,
        "distance_free_recall": distance_recall,
        "distance_free_support": distance_support,
    }


def _conditions(
    *,
    passing: bool = True,
    holdout_favorable: bool = True,
    scale: int = 1,
) -> dict:
    correct = _metrics(
        passing=passing, holdout_favorable=holdout_favorable, scale=scale
    )
    return {
        "correct_rgb": correct,
        "role_global_shuffled_rgb": _metrics(
            passing=passing,
            holdout_favorable=holdout_favorable,
            scale=scale,
            nll_override=correct["raw_hierarchical_balanced_nll"] + 0.5,
        ),
        "same_scene_wrong_view_rgb": _metrics(
            passing=passing,
            holdout_favorable=holdout_favorable,
            scale=scale,
            nll_override=correct["raw_hierarchical_balanced_nll"] + 0.5,
        ),
    }


def _controls(seed: int, panel: str) -> dict:
    identities = contract.CONTROL_PERMUTATION_SHA256[seed][panel]
    return {
        "role_global_shuffle": {
            "schema": "lewm_go2_micro_overfit_shuffle_v1",
            "seed": seed,
            "namespace": panel,
            "record_count": 320,
            "permutation_sha256": identities["role_global_shuffle"],
            "same_image_pairs": 0,
            "same_scene_pairs": 0,
            "same_transition_pairs": 0,
        },
        "same_scene_wrong_view": {
            "schema": "lewm_go2_micro_overfit_same_scene_wrong_view_v1",
            "seed": seed,
            "namespace": panel,
            "record_count": 320,
            "permutation_sha256": identities["same_scene_wrong_view"],
            "same_image_pairs": 0,
            "same_transition_pairs": 0,
            "different_scene_pairs": 0,
            "scenes": {
                "scene": {"frame_count": 320, "transition_count": 160, "rotation": 2}
            },
        },
        "wrong_rgb_uses_target_attitude": True,
    }


def _panel(
    seed: int,
    panel: str,
    *,
    fit_passing: bool = True,
    holdout_favorable: bool = True,
) -> dict:
    report = {
        "schema": contract.PANEL_REPORT_SCHEMA,
        "panel": panel,
        "frame_count": 320,
        "target_batch_size": 4,
        "combined_model_batch_size": 12,
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "wrong_rgb_uses_target_attitude": True,
        "conditions": _conditions(
            passing=fit_passing, holdout_favorable=holdout_favorable, scale=5
        ),
        "families": {
            family: {
                "conditions": _conditions(
                    passing=fit_passing,
                    holdout_favorable=holdout_favorable,
                    scale=1,
                )
            }
            for family in contract.FAMILIES
        },
        "controls": _controls(seed, panel),
    }
    if panel == "fit":
        report["fit_gate"] = contract.fit_panel_gate_report(report)
    return report


def _event_record(**overrides: int) -> dict[str, int]:
    result = dict.fromkeys(contract.EVENT_FIELDS, 0)
    result.update(overrides)
    return result


def _stage(seed: int, branch: str, *, passes: bool) -> dict:
    config = contract.BRANCH_CONFIGS[branch]
    updates = config["updates"]
    schedule = contract.deterministic_minibatch_schedule(seed=seed, branch=branch)
    report = _panel(seed, "fit", fit_passing=passes)
    curve = [
        {
            "step": step,
            "batch_loss": 0.01,
            "gradient_norm_before_clip": 0.2,
            "fit_panel": report,
        }
        for step in range(contract.EVALUATION_INTERVAL, updates + 1, contract.EVALUATION_INTERVAL)
    ]
    training, evaluation = contract._expected_stage_access(updates, len(curve))
    return {
        "schema": contract.STAGE_SCHEMA,
        "stage": branch,
        "config": config,
        "maximum_steps": updates,
        "completed_steps": updates,
        "batch_size": 4,
        "evaluation_interval": 100,
        "optimizer": contract.OPTIMIZER_CONFIGS[branch],
        "objective": contract.OBJECTIVE_CONTRACT,
        "fixed_update_budget_consumed": True,
        "one_direct_forward_backward_per_update": True,
        "gradient_accumulation_or_microbatching": False,
        "initial_state_sha256": _manifest()["model_initial_state_sha256"][str(seed)],
        "final_state_sha256": "e" * 64,
        "exact_initial_state_restart_verified": True,
        "minibatch_indices": schedule,
        "minibatch_indices_sha256": contract.canonical_json_sha256(schedule),
        "learning_curve": curve,
        "terminal_fit_gate": contract.terminal_fit_gate_summary(curve, updates, 100),
        "training_access": training,
        "fit_evaluation_access": evaluation,
        "holdouts_evaluated": passes,
    }


_REFERENCE: dict | None = None


def _reference() -> dict:
    global _REFERENCE
    if _REFERENCE is None:
        path = Path(contract.INPUT_BINDINGS["static_patch7_comparator"]["path"])
        with path.open("r", encoding="utf-8") as stream:
            _REFERENCE = contract.extract_faithful_patch7_family_reference(json.load(stream))
    return _REFERENCE


def _state_contract() -> dict:
    entries = [
        {"name": "online", "dtype": "torch.float32", "shape": [2], "requires_grad": True},
        {"name": "target", "dtype": "torch.float32", "shape": [3], "requires_grad": False},
        {"name": "buffer", "dtype": "torch.float32", "shape": [1], "requires_grad": None},
    ]
    return {"entry_count": len(entries), "entries": entries}


def _access(stages: dict, qualifying: str | None, seed: int) -> dict:
    invoked = [stage for stage in stages.values() if stage is not None]
    summed = dict.fromkeys(contract.EVENT_FIELDS, 0)
    for stage in invoked:
        for source in (stage["training_access"], stage["fit_evaluation_access"]):
            for field in contract.EVENT_FIELDS:
                summed[field] += source[field]
    fit = dict(summed)
    fit.update(
        image_decode_events=320,
        label_shard_npz_open_events=20,
        model_calls=0,
        model_output_frames=0,
        model_attitude_frames=0,
    )
    panels = {
        "fit": {
            "authorized": True,
            "artifact_hash_passes": 2,
            "image_hash_byte_open_events": 640,
            "shard_hash_byte_open_events": 40,
            "dataset_access": fit,
        }
    }
    for panel, shards in (("same_scene_holdout", 20), ("cross_scene_holdout", 25)):
        authorized = qualifying is not None
        panels[panel] = {
            "authorized": authorized,
            "authorized_by_branch": qualifying,
            "artifact_hash_passes": 2 if authorized else 0,
            "image_hash_byte_open_events": 640 if authorized else 0,
            "shard_hash_byte_open_events": 2 * shards if authorized else 0,
            "dataset_access": _event_record(
                **(
                    {
                        "image_requests": 960,
                        "target_requests": 320,
                        "attitude_requests": 320,
                        "image_decode_events": 320,
                        "label_shard_npz_open_events": shards,
                        "model_calls": 80,
                        "model_output_frames": 960,
                        "model_attitude_frames": 960,
                    }
                    if authorized
                    else {}
                )
            ),
            "one_shot_evaluation": authorized,
        }
    zero_role = {"image_byte_opens": 0, "label_shard_byte_opens": 0, "model_outputs": 0}
    return {
        "schema": contract.ACCESS_LEDGER_SCHEMA,
        "panels": panels,
        "fit_dataset_totals": fit,
        "sidecar": {
            "manifest_byte_opens": 2,
            "train_role_byte_opens": 2,
            "checkpoint_selection_role_byte_opens": 0,
            "probability_calibration_role_byte_opens": 0,
            "g2_evaluation_role_byte_opens": 0,
        },
        "dataset_roles": {
            "train": {
                "panel_transition_rows_joined": 480,
                "model_outputs": summed["model_output_frames"] + (1920 if qualifying else 0),
            },
            "checkpoint_selection": zero_role,
            "probability_calibration": zero_role,
            "g2_evaluation": zero_role,
        },
        "holdout_payloads_opened_only_after_terminal_fit_pass": True,
        "wrong_rgb_target_attitude_frames": sum(
            stage["fit_evaluation_access"]["model_attitude_frames"] * 2 // 3
            for stage in invoked
        )
        + (1280 if qualifying else 0),
        "non_train_image_opens": 0,
        "non_train_label_shard_opens": 0,
        "non_train_model_outputs": 0,
        "controlled_metadata_reads": {
            "implementation_manifest_byte_opens": 2,
            "source_byte_opens": {
                entry["role"]: 2 for entry in _manifest()["sources"]["entries"]
            },
            "input_byte_opens": {name: 2 for name in contract.INPUT_BINDINGS},
            "seed_20260710_result_byte_opens": 0 if seed == 20260710 else 2,
            "authoritative_attempt_marker_byte_opens": 2,
            "seed_20260710_attempt_marker_byte_opens": (
                0 if seed == 20260710 else 2
            ),
        },
    }


def _result(
    seed: int,
    *,
    faithful_pass: bool = True,
    ceiling_pass: bool = True,
    holdout_favorable: bool = True,
    primary: dict | None = None,
) -> dict:
    manifest = _manifest()
    primary_attempt_marker = (
        None if primary is None else _attempt_marker(contract.EXPECTED_SEEDS[0])
    )
    attempt_marker = _attempt_marker(
        seed,
        primary=primary,
        primary_attempt_marker=primary_attempt_marker,
    )
    faithful = _stage(seed, "production_faithful", passes=faithful_pass)
    ceiling = None if faithful_pass else _stage(seed, "ceiling_optimizer", passes=ceiling_pass)
    qualifying = (
        "production_faithful"
        if faithful_pass
        else "ceiling_optimizer"
        if ceiling_pass
        else None
    )
    faithful["holdouts_evaluated"] = qualifying == "production_faithful"
    if ceiling is not None:
        ceiling["holdouts_evaluated"] = qualifying == "ceiling_optimizer"
    stages = {"production_faithful": faithful, "ceiling_optimizer": ceiling}
    reference = _reference()
    holdouts = (
        None
        if qualifying is None
        else {
            panel: _panel(seed, panel, holdout_favorable=holdout_favorable)
            for panel in contract.HOLDOUT_PANELS
        }
    )
    checks = (
        None
        if holdouts is None
        else {
            panel: contract.strict_patch7_holdout_checks(
                holdouts[panel], reference["panels"][panel]
            )
            for panel in contract.HOLDOUT_PANELS
        }
    )
    decision = contract.per_seed_decision(faithful, ceiling, checks)
    sources = {entry["role"]: entry["sha256"] for entry in manifest["sources"]["entries"]}
    state_contract = _state_contract()
    inputs = {
        **contract.INPUT_BINDINGS,
        "seed_20260710_result": (
            None
            if primary is None
            else {
                "path": contract.COMMAND_CONTRACT["canonical_outputs"]["20260710"],
                "sha256": PRIMARY_FILE_SHA256,
                "content_sha256": primary["content_sha256"],
            }
        ),
    }
    snapshot = {
        "head": "head",
        "status_short": "",
        "tracked_dirty_diff_sha256": "f" * 64,
        "tracked_dirty_diff_bytes": 0,
    }
    core = {
        "schema": contract.RESULT_SCHEMA,
        "authoritative": True,
        "aggregation_eligible": True,
        "promotion_eligible": False,
        "seed": seed,
        "created_at_utc": "2026-07-11T00:00:00+00:00",
        "completed_at_utc": "2026-07-11T01:00:00+00:00",
        "invocation": list(attempt_marker["invocation"]),
        "execution": {
            "device": {
                "device": "cuda:0",
                "device_name": "AMD Radeon PRO R9700",
                "total_memory_bytes": 20 * 1024**3,
                "hip_visible_devices": "0",
                "hsa_override_gfx_version_unset": True,
                "raphael_rejected": True,
            },
            "determinism": {
                "seed": seed,
                "torch_deterministic_algorithms": True,
                "warn_only": False,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "batch_size_frames": 4,
            "evaluation_combined_model_batch_size": 12,
            "evaluation_interval": 100,
            "branches": contract.BRANCH_CONFIGS,
            "source_workers": 6,
            "native_threads_per_worker": 1,
            "fp32_no_autocast_amp_compile_quantization_or_query_chunking": True,
        },
        "contract": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["binding"],
            "sha256": contract.EXECUTION_BINDING_SHA256,
        },
        "preoutput_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS["preoutput_amendment"],
            "sha256": contract.PREOUTPUT_AMENDMENT_SHA256,
        },
        "attempt_control_amendment": {
            "path": contract.IMPLEMENTATION_SOURCE_PATHS[
                "attempt_control_amendment"
            ],
            "sha256": contract.ATTEMPT_CONTROL_AMENDMENT_SHA256,
        },
        "attempt_marker": {
            "path": contract.ATTEMPT_MARKER_PATHS[seed],
            "sha256": _attempt_file_sha256(seed),
            "content_sha256": attempt_marker["content_sha256"],
        },
        "implementation_manifest": {
            "path": f"{contract.REPOSITORY_ROOT}/docs/lewm_go2_dynamic_cartesian_n32_v1_implementation_manifest_2026-07-11.json",
            "sha256": "1" * 64,
            "content_sha256": manifest["content_sha256"],
        },
        "implementation_manifest_content_sha256": manifest["content_sha256"],
        "inputs": inputs,
        "source_hashes": sources,
        "git": {"start": snapshot, "end": snapshot},
        "model_config": copy.deepcopy(contract.MODEL_CONFIG),
        "model": {
            "class": "EgomotionBevJepa",
            "entrypoint": "occupancy_logits",
            "initialization": {
                "initial_state_sha256": manifest["model_initial_state_sha256"][str(seed)],
                "state_contract": state_contract,
                "state_contract_sha256": contract.canonical_json_sha256(state_contract),
                "parameter_count": 5,
                "trainable_parameter_count": 2,
            },
            "all_invoked_branches_restart_same_initial_state": True,
            "n32_weights_are_not_checkpointed_or_promotable": True,
        },
        "preprocessing": copy.deepcopy(contract.PREPROCESSING_CONTRACT),
        "objective": copy.deepcopy(contract.OBJECTIVE_CONTRACT),
        "projective_query_support": copy.deepcopy(contract.PROJECTIVE_QUERY_SUPPORT),
        "panel_join": copy.deepcopy(contract.PANEL_JOIN_CONTRACT),
        "projection_parity": {
            "content_sha256": contract.INPUT_BINDINGS["fit_projection_parity"]["content_sha256"],
            "frame_count": 320,
            "mismatched_cells": 0,
        },
        "stages": stages,
        "qualifying_branch": qualifying,
        "patch7_reference": reference,
        "holdouts": holdouts,
        "holdout_checks": checks,
        "decision": decision,
        "artifact_verification": {
            "fit_verified_before_first_payload_access": True,
            "fit_verified_after_last_model_access": True,
            "holdouts_verified_only_after_terminal_fit_pass": True,
            "holdouts_evaluated_once": qualifying is not None,
        },
        "access_ledger": _access(stages, qualifying, seed),
        "publication": {
            "mode": "private_staging_hardlink_noreplace",
            "canonical_output": contract.COMMAND_CONTRACT["canonical_outputs"][str(seed)],
        },
        "shared_jepa_construction_licensed": False,
        "g2_licensed": False,
        "runtime_licensed": False,
    }
    return {**core, "content_sha256": contract.canonical_json_sha256(core)}


def test_canonical_json_and_registered_identity() -> None:
    assert contract.canonical_json_bytes({"b": 1, "a": 2}) == b'{"a":2,"b":1}'
    assert contract.EXPECTED_SEEDS == (20260710, 20260711)
    assert contract.MODEL_CONFIG["occupancy_weight"] == 2.0
    with pytest.raises(ValueError):
        contract.canonical_json_sha256({"x": float("nan")})


def test_pure_schedules_are_deterministic_distinct_and_prefix_matched() -> None:
    faithful = contract.deterministic_minibatch_schedule(seed=20260710, branch="production_faithful")
    ceiling = contract.deterministic_minibatch_schedule(seed=20260710, branch="ceiling_optimizer")
    assert ceiling[: len(faithful)] == faithful
    assert faithful != contract.deterministic_minibatch_schedule(seed=20260711, branch="production_faithful")
    assert sorted(index for batch in faithful[:80] for index in batch) == list(range(320))


@pytest.mark.parametrize("seed", contract.EXPECTED_SEEDS)
@pytest.mark.parametrize("branch", tuple(contract.BRANCH_CONFIGS))
def test_schedule_validator_accepts_every_registered_schedule(seed: int, branch: str) -> None:
    schedule = contract.deterministic_minibatch_schedule(seed=seed, branch=branch)
    assert contract.validate_minibatch_schedule(schedule, seed=seed, branch=branch) == contract.SCHEDULE_SHA256[(seed, branch)]


def test_schedule_validator_rejects_bool_duplicate_and_reordering() -> None:
    schedule = contract.deterministic_minibatch_schedule(seed=20260710, branch="production_faithful")
    for mutation in (True, schedule[0][1]):
        changed = copy.deepcopy(schedule)
        changed[0][0] = mutation
        with pytest.raises(ValueError):
            contract.validate_minibatch_schedule(changed, seed=20260710, branch="production_faithful")
    changed = copy.deepcopy(schedule)
    changed[0], changed[1] = changed[1], changed[0]
    with pytest.raises(ValueError, match="drift"):
        contract.validate_minibatch_schedule(changed, seed=20260710, branch="production_faithful")


def test_manifest_exact_source_map_and_projective_support() -> None:
    manifest = _manifest()
    assert contract.validate_implementation_manifest(manifest) == manifest
    changed = copy.deepcopy(manifest)
    changed["sources"]["entries"][0]["path"] += ".alias"
    changed["sources"]["source_map_sha256"] = contract.canonical_json_sha256(changed["sources"]["entries"])
    _rehash(changed)
    with pytest.raises(ValueError, match="role/path/hash"):
        contract.validate_implementation_manifest(changed)
    changed = copy.deepcopy(manifest)
    changed["projective_query_support"]["support_point_count"] = 1
    _rehash(changed)
    with pytest.raises(ValueError, match="projective"):
        contract.validate_implementation_manifest(changed)


def test_full_authoritative_result_and_seed_pair_validate() -> None:
    manifest = _manifest()
    first = _result(20260710)
    first_marker = _attempt_marker(20260710)
    assert contract.validate_authoritative_result(
        first,
        20260710,
        manifest,
        IMPLEMENTATION_FILE_SHA256,
        first_marker,
        PRIMARY_ATTEMPT_FILE_SHA256,
    ) == first
    second = _result(20260711, primary=first)
    second_marker = _attempt_marker(
        20260711, primary=first, primary_attempt_marker=first_marker
    )
    assert contract.validate_authoritative_result(
        second,
        20260711,
        manifest,
        IMPLEMENTATION_FILE_SHA256,
        second_marker,
        REPLICATION_ATTEMPT_FILE_SHA256,
        primary_result=first,
        primary_file_sha256=PRIMARY_FILE_SHA256,
        primary_attempt_marker=first_marker,
        primary_attempt_marker_file_sha256=PRIMARY_ATTEMPT_FILE_SHA256,
    ) == second
    pair = contract.validate_seed_pair(
        first,
        second,
        manifest,
        IMPLEMENTATION_FILE_SHA256,
        PRIMARY_FILE_SHA256,
        first_marker,
        PRIMARY_ATTEMPT_FILE_SHA256,
        second_marker,
        REPLICATION_ATTEMPT_FILE_SHA256,
    )
    assert pair["both_favorable"] is True
    assert pair["shared_jepa_construction_licensed"] is True
    both_fit_branches_fail = _result(
        20260710, faithful_pass=False, ceiling_pass=False
    )
    validated_failure = contract.validate_authoritative_result(
        both_fit_branches_fail,
        20260710,
        manifest,
        IMPLEMENTATION_FILE_SHA256,
        first_marker,
        PRIMARY_ATTEMPT_FILE_SHA256,
    )
    assert validated_failure["decision"]["classification"] == "fit_gate_failed"
    assert validated_failure["holdouts"] is None


def test_attempt_marker_and_external_file_identity_fail_closed() -> None:
    manifest = _manifest()
    marker = _attempt_marker(20260710)
    assert contract.validate_attempt_marker(
        marker,
        20260710,
        manifest,
        IMPLEMENTATION_FILE_SHA256,
    ) == marker
    changed = copy.deepcopy(marker)
    changed["retry_permitted"] = True
    _rehash(changed)
    with pytest.raises(ValueError, match="attempt-marker identity"):
        contract.validate_attempt_marker(
            changed,
            20260710,
            manifest,
            IMPLEMENTATION_FILE_SHA256,
        )
    with pytest.raises(ValueError, match="attempt-marker binding"):
        contract.validate_authoritative_result(
            _result(20260710),
            20260710,
            manifest,
            IMPLEMENTATION_FILE_SHA256,
            marker,
            "6" * 64,
        )


def test_seed_pair_rejects_different_external_manifest_identity() -> None:
    first = _result(20260710)
    first_marker = _attempt_marker(20260710)
    second = _result(20260711, primary=first)
    second_marker = _attempt_marker(
        20260711, primary=first, primary_attempt_marker=first_marker
    )
    with pytest.raises(ValueError, match="implementation|invocation"):
        contract.validate_seed_pair(
            first,
            second,
            _manifest(),
            "2" * 64,
            PRIMARY_FILE_SHA256,
            first_marker,
            PRIMARY_ATTEMPT_FILE_SHA256,
            second_marker,
            REPLICATION_ATTEMPT_FILE_SHA256,
        )


@pytest.mark.parametrize(
    ("mutator", "message"),
    (
        (lambda value: value["stages"]["production_faithful"]["learning_curve"].pop(), "curve"),
        (lambda value: value["stages"]["production_faithful"]["terminal_fit_gate"].update(passes=False), "terminal"),
        (lambda value: value["stages"]["production_faithful"].update(initial_state_sha256="0" * 64), "state"),
        (lambda value: value["stages"]["production_faithful"].update(exact_initial_state_restart_verified=False), "stage"),
        (lambda value: value["preprocessing"].update(model_dtype="float16"), "preprocessing"),
        (lambda value: value["objective"].update(jepa_weight=1.0), "objective"),
        (lambda value: value["projective_query_support"].update(support_point_count=1), "support"),
        (lambda value: value["implementation_manifest"].update(sha256="7" * 64), "implementation manifest"),
        (lambda value: value["holdouts"]["same_scene_holdout"].update(wrong_rgb_uses_target_attitude=False), "panel report"),
        (lambda value: value["access_ledger"].update(non_train_model_outputs=1), "forbidden"),
        (lambda value: value["access_ledger"]["panels"]["cross_scene_holdout"]["dataset_access"].update(model_calls=79), "access"),
        (lambda value: value["model"]["initialization"].update(trainable_parameter_count=5), "parameter"),
        (lambda value: value["panel_join"].update(global_rows_sha256="0" * 64), "join"),
        (lambda value: value["holdouts"]["same_scene_holdout"]["controls"]["role_global_shuffle"].update(permutation_sha256="0" * 64), "permutation"),
    ),
)
def test_result_tampering_fails_closed(mutator, message: str) -> None:
    value = _result(20260710)
    mutator(value)
    _rehash(value)
    with pytest.raises(ValueError, match=message):
        contract.validate_authoritative_result(
            value,
            20260710,
            _manifest(),
            IMPLEMENTATION_FILE_SHA256,
            _attempt_marker(20260710),
            PRIMARY_ATTEMPT_FILE_SHA256,
        )


def test_holdout_check_is_recomputed_from_raw_report() -> None:
    value = _result(20260710)
    value["holdout_checks"]["cross_scene_holdout"]["passes"] = False
    value["decision"] = contract.per_seed_decision(
        value["stages"]["production_faithful"], None, value["holdout_checks"]
    )
    _rehash(value)
    with pytest.raises(ValueError, match="raw reports"):
        contract.validate_authoritative_result(
            value,
            20260710,
            _manifest(),
            IMPLEMENTATION_FILE_SHA256,
            _attempt_marker(20260710),
            PRIMARY_ATTEMPT_FILE_SHA256,
        )


@pytest.mark.parametrize(
    "mutator",
    (
        lambda report: report["conditions"]["correct_rgb"].update(
            raw_joint_nll=-1.0
        ),
        lambda report: report["conditions"]["correct_rgb"].update(
            raw_joint_accuracy=100.0
        ),
        lambda report: report["conditions"]["correct_rgb"][
            "class_recall"
        ].update(unknown=100.0),
        lambda report: report["conditions"]["correct_rgb"].update(
            cell_count=201
        ),
        lambda report: report["conditions"]["correct_rgb"]["joint_confusion"][
            0
        ].__setitem__(0, -1),
        lambda report: report["conditions"]["correct_rgb"][
            "unknown_known_confusion"
        ][0].__setitem__(0, 49),
        lambda report: report["conditions"]["correct_rgb"][
            "posterior_quantiles_by_truth_class"
        ]["unknown"]["unknown"].update(p05=0.99, p95=0.1),
        lambda report: report["conditions"]["correct_rgb"][
            "distance_free_support"
        ].update({"0.0_to_0.5": 11}),
        lambda report: report["conditions"]["correct_rgb"].update(
            raw_hierarchical_balanced_nll=0.2
        ),
        lambda report: report["conditions"]["correct_rgb"].update(
            free_average_precision=1.1
        ),
        lambda report: report["families"][contract.FAMILIES[0]]["conditions"][
            "correct_rgb"
        ].update(
            raw_joint_nll=0.02,
            raw_hierarchical_balanced_nll=0.02,
            raw_unknown_known_weighted_nll=0.02,
            raw_known_free_occupied_weighted_nll=0.02,
            raw_known_free_occupied_nll=0.02,
        ),
    ),
)
def test_metric_tampering_fails_closed(mutator) -> None:
    report = _panel(20260710, "fit")
    mutator(report)
    with pytest.raises(ValueError):
        contract.validate_panel_report(
            report,
            seed=20260710,
            panel="fit",
            require_fit_gate=True,
        )


def test_metric_validator_accepts_all_frozen_patch7_reports() -> None:
    path = Path(contract.INPUT_BINDINGS["static_patch7_comparator"]["path"])
    payload = json.loads(path.read_text())
    panels: list[tuple[tuple[str, ...], dict]] = []

    def walk(value: object, location: tuple[str, ...] = ()) -> None:
        if isinstance(value, dict):
            if (
                isinstance(value.get("conditions"), dict)
                and set(value["conditions"]) == set(contract.CONDITIONS)
                and isinstance(value.get("families"), dict)
                and set(value["families"]) == set(contract.FAMILIES)
            ):
                panels.append((location, value))
            for key, child in value.items():
                walk(child, (*location, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*location, str(index)))

    walk(payload)
    assert len(panels) == 112
    for location, report in panels:
        name = "/".join(location)
        contract._conditions(report, name=name)
        for family in contract.FAMILIES:
            contract._conditions(report["families"][family], name=f"{name}/{family}")
        contract._validate_family_aggregation(report, name=name)


def test_seed11_requires_exact_primary_content_and_external_file_hash() -> None:
    first = _result(20260710)
    second = _result(20260711, primary=first)
    first_marker = _attempt_marker(20260710)
    second_marker = _attempt_marker(
        20260711, primary=first, primary_attempt_marker=first_marker
    )
    with pytest.raises(ValueError, match="external"):
        contract.validate_authoritative_result(
            second,
            20260711,
            _manifest(),
            IMPLEMENTATION_FILE_SHA256,
            second_marker,
            REPLICATION_ATTEMPT_FILE_SHA256,
            primary_result=first,
        )
    with pytest.raises(ValueError, match="binding|invocation"):
        contract.validate_authoritative_result(
            second,
            20260711,
            _manifest(),
            IMPLEMENTATION_FILE_SHA256,
            second_marker,
            REPLICATION_ATTEMPT_FILE_SHA256,
            primary_result=first,
            primary_file_sha256="8" * 64,
            primary_attempt_marker=first_marker,
            primary_attempt_marker_file_sha256=PRIMARY_ATTEMPT_FILE_SHA256,
        )


def test_static_patch7_reference_retains_frozen_identity_without_numpy_math() -> None:
    assert contract.canonical_json_sha256(_reference()) == contract.PATCH7_REFERENCE_SHA256


def test_contract_import_does_not_import_torch() -> None:
    import os
    import subprocess
    import sys

    environment = dict(os.environ)
    environment["PYTHONPATH"] = contract.REPOSITORY_ROOT
    subprocess.run(
        (
            sys.executable,
            "-c",
            "import sys; import lewm.benchmarks.go2_dynamic_cartesian_n32; "
            "assert 'torch' not in sys.modules",
        ),
        check=True,
        env=environment,
    )
