from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/check_go2_world_model_existing_pool_three_arm_v1.py"
REPLACEMENT_SCHEMA_PREFIX = (
    "lewm_go2_world_model_existing_pool_three_arm_v1_integrity_replacement_v2_"
)
REPLACEMENT_ATTEMPT_ID = (
    "world_model_existing_pool_three_arm_v1_integrity_replacement_v2/attempt_v1"
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("existing_pool_three_arm_checker", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _raw(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8") + b"\n"


def _inert(path: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(path.encode("utf-8")).hexdigest(),
        "byte_count": 1,
    }


def _write(root: Path, relative: str, value: object) -> dict[str, Any]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = _raw(value)
    path.write_bytes(raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _fixture(
    root: Path,
    *,
    mutate_result: Callable[[dict[str, Any]], None] | None = None,
    mutate_measurement: Callable[[dict[str, Any]], None] | None = None,
    overlap_audit: dict[str, Any] | None = None,
    action_receipts: dict[str, dict[str, Any]] | None = None,
) -> tuple[Path, dict[str, Any]]:
    checker = _load_checker()
    root.mkdir(parents=True, exist_ok=True)
    reservation_path = root / "reservation.json"
    reservation_path.write_bytes(b"x")
    reservation_path.chmod(0)
    pack_path = root / "pack/manifest.json"
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_bytes(b"x")
    pack_path.chmod(0)
    pack_artifact_bindings: dict[str, dict[str, dict[str, Any]]] = {}
    for role, artifacts in checker.PACK_ARTIFACT_RELATIVE_PATHS.items():
        pack_artifact_bindings[role] = {}
        for name, relative in artifacts.items():
            artifact_path = root / relative
            artifact_path.write_bytes(b"x")
            artifact_path.chmod(0)
            pack_artifact_bindings[role][name] = _inert(relative)
    authority_binding = _inert("/synthetic/authority.json")
    plan_binding = _inert("/synthetic/plan.json")
    predecessor_terminal_failure_binding = _inert(
        "/synthetic/predecessor_terminal_failure.json"
    )
    audit_base = {
        "status": "PASS",
        "passed": True,
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
    }
    fixture_overlap_audit = {
        "schema": checker.OVERLAP_AUDIT_SCHEMA,
        "status": "PASS",
        "passed": True,
        "row_count": 18_048,
        "role_row_counts": {"train": 16_000, "val": 2_048},
        "role_scene_counts": {"train": 1_000, "val": 150},
        "checks": {
            "role_scene_disjointness": True,
            "train_all_actions_supported": True,
            "train_all_ordered_pairs_supported": True,
        },
        "failed_checks": [],
        "diagnostic_checks": {
            "train_all_ordered_triples_supported": False,
        },
        "failed_diagnostic_checks": [
            "train_all_ordered_triples_supported",
        ],
        "role_scene_overlap_count": 0,
        "role_scene_overlap": [],
        "train_support": {
            "visible_action_positions": [0, 1, 2],
            "action_count": 9,
            "action_count_by_position": {"a0": 9, "a1": 9, "a2": 9},
            "ordered_pair_count": 81,
            "ordered_pair_count_by_position": {
                "a0_a1": 81,
                "a1_a2": 81,
            },
            "ordered_triple_count": 722,
            "missing_action_ids_by_position": {
                "a0": [],
                "a1": [],
                "a2": [],
            },
            "missing_action_ids": [],
            "missing_ordered_pairs_by_position": {
                "a0_a1": [],
                "a1_a2": [],
            },
            "missing_ordered_pairs": [],
            "missing_ordered_triples": [
                list(values)
                for values in checker.EXPECTED_MISSING_ORDERED_TRAIN_TRIPLES
            ],
        },
        "entropy": {"train": {"a2": 3.0}, "val": {"a2": 3.0}},
        "mutual_information_bits": {
            "train": {"candidate_family": 0.1},
            "val": {"candidate_family": 0.1},
        },
        "scene_diagnostics": {
            "train": {"scene_count": 1_000},
            "val": {"scene_count": 150},
        },
        "gate_scope": checker.OVERLAP_GATE_SCOPE,
    }
    overlap_binding = _write(
        root,
        "overlap_audit.json",
        {
            **audit_base,
            "schema": "lewm_go2_world_model_existing_pool_three_arm_overlap_audit_v1",
            "audit": (
                copy.deepcopy(overlap_audit)
                if overlap_audit is not None
                else fixture_overlap_audit
            ),
        },
    )
    families = [f"family-{index}" for index in range(8)]
    mapping_rows = []
    for index in range(16_000):
        donor = (index + 8) % 16_000
        mapping_rows.append(
            {
                "row_position": index,
                "row_index": index,
                "role": "train",
                "family": families[index % 8],
                "scene_id": f"scene-{index}",
                "factual_candidate_action_id": index % 9,
                "donor_position": donor,
                "donor_index": donor,
                "donor_scene_id": f"scene-{donor}",
                "deranged_candidate_action_id": donor % 9,
            }
        )
    mapping_sha256 = hashlib.sha256(
        json.dumps(
            mapping_rows,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    shuffle_binding = _write(
        root,
        "shuffle_audit.json",
        {
            **audit_base,
            "schema": (
                "lewm_go2_world_model_existing_pool_three_arm_"
                "candidate_action_derangement_v1"
            ),
            "audit": {
                "schema": checker.SHUFFLE_AUDIT_SCHEMA,
                "status": "PASS",
                "passed": True,
                "algorithm": (
                    "role_family_local_cyclic_then_exact_bipartite_derangement_v1"
                ),
                "candidate_action_position": 2,
                "changed_action_positions": [2],
                "row_count": 16_000,
                "role_family_group_count": 8,
                "group_selected_offsets": {
                    f"train:{family}": 8 for family in families
                },
                "group_methods": {
                    f"train:{family}": "dual_hash_ranked_cyclic_search"
                    for family in families
                },
                "mapping_sha256": mapping_sha256,
                "checks": {
                    "donor_map_is_global_bijection": True,
                    "donor_identity_zero_fixed_points": True,
                    "different_scene_donors": True,
                    "candidate_a2_zero_fixed_points": True,
                    "role_family_action_marginals_exact": True,
                },
                "fixed_donor_identity_count": 0,
                "same_scene_donor_count": 0,
                "fixed_candidate_action_count": 0,
                "mapping_rows": mapping_rows,
            },
        },
    )
    arms: dict[str, Any] = {}
    checkpoints: dict[str, Any] = {}
    for arm in checker.ARM_NAMES:
        action_receipt = (
            copy.deepcopy(action_receipts[arm])
            if action_receipts is not None
            else {
                "bootstrap_algorithm": (
                    checker.ACTION_IDENTIFICATION_BOOTSTRAP_ALGORITHM
                ),
                "bootstrap_interpretation": (
                    checker.ACTION_IDENTIFICATION_BOOTSTRAP_INTERPRETATION
                ),
                "bootstrap_seed": checker.ACTION_IDENTIFICATION_BOOTSTRAP_SEED,
                "bootstrap_replicates": (
                    checker.ACTION_IDENTIFICATION_BOOTSTRAP_REPLICATES
                ),
                "bootstrap_lower_index": (
                    checker.ACTION_IDENTIFICATION_BOOTSTRAP_LOWER_INDEX
                ),
                "family_action_supporting_scene_counts": {
                    family: [2] * 9 for family in checker.REGISTERED_FAMILIES
                },
                "minimum_family_action_supporting_scene_count": 2,
                "balanced_accuracy": 0.5,
                "balanced_accuracy_one_sided_95_lower_bound": 0.3,
                "balanced_chance": 1.0 / 9.0,
                "exact_tie_count": 2_048 if arm == "blind" else 0,
                "exact_tie_rate": 1.0 if arm == "blind" else 0.0,
                "unique_winner_count": 0 if arm == "blind" else 2_048,
                "unique_winner_accuracy": 0.0 if arm == "blind" else 0.5,
                "hardest_wrong_action_margin": 0.2,
                "hardest_wrong_action_margin_one_sided_95_lower_bound": 0.1,
            }
        )
        measurement_bindings: list[dict[str, Any]] = []
        snapshot_bindings: list[dict[str, Any]] = []
        for update in checker.MEASUREMENT_UPDATES:
            measurement = {
                "schema": checker.METRICS_SCHEMA,
                "status": "COMPLETE",
                "arm": arm,
                "update": update,
                "authority_binding": authority_binding,
                "plan_binding": plan_binding,
                "encoder_sha256": "e" * 64,
                "target_sha256": "f" * 64,
                "panel": {
                    "kind": "scene_disjoint_factual_validation",
                    "row_count": 2_048,
                    "row_indices_sha256": "5" * 64,
                },
                "validation": {
                    "row_count": 2_048,
                    "factual_energy": {
                        "mean": 1.0 + update / 1000.0,
                        "family_equal_mean": 1.0,
                        "action_equal_mean": 1.0,
                        "family_count": 8,
                        "action_count": 9,
                        "scene_count": 150,
                    },
                    "cross_arm": {
                        "conditioned_vs_blind_log_energy_advantage": 0.2,
                        "conditioned_vs_blind_one_sided_95_lower_bound": 0.1,
                        "conditioned_vs_shuffled_log_energy_advantage": 0.2,
                        "conditioned_vs_shuffled_one_sided_95_lower_bound": 0.1,
                        "scene_cluster_count": 150,
                    },
                    "controls": {
                        "persistence_log_energy_advantage": 0.3,
                        "persistence_one_sided_95_lower_bound": 0.2,
                        "wrong_history_log_energy_advantage": 0.3,
                        "wrong_history_one_sided_95_lower_bound": 0.2,
                    },
                    "action_identification": copy.deepcopy(action_receipt),
                    "representation": {
                        "prediction_effective_rank": 8.0,
                        "target_effective_rank": 16.0,
                        "prediction_to_target_rank_ratio": 0.5,
                    },
                },
                "training": (
                    None
                    if update < 700
                    else {
                        "row_count": 16_000,
                        "family_count": 8,
                        "factual_mean_energy": 0.8,
                        "conditioned_vs_blind_family_equal_log_energy_advantage": 0.2,
                        "conditioned_vs_shuffled_family_equal_log_energy_advantage": 0.2,
                        "backward_calls": 0,
                        "optimizer_steps": 0,
                    }
                ),
                "optimization": {
                    "completed_updates": update,
                    "optimizer_steps": update,
                    "loss": None if update == 0 else 0.5,
                    "learning_rate_fraction": 0.5,
                    "predictor_learning_rate": 0.0002,
                    "memory_learning_rate": 0.0006,
                    "warmup_updates": 150,
                    "schedule_horizon_updates": 3_000,
                },
                "integrity": {
                    "candidate_blind_treatment_exact": True,
                    "shuffled_derangement_exact": True,
                    "factual_evaluation_exact": True,
                    "frozen_substrate_exact": True,
                    "no_gradient_during_evaluation": True,
                    "finite": True,
                },
            }
            if mutate_measurement is not None and arm == "conditioned" and update == 0:
                mutate_measurement(measurement)
            measurement_bindings.append(
                _write(
                    root,
                    f"arms/{arm}/measurements/update_{update:06d}.json",
                    measurement,
                )
            )
            snapshot_path = root / f"arms/{arm}/snapshots/update_{update:06d}.pt"
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            snapshot_path.write_bytes(b"x")
            snapshot_path.chmod(0)
            snapshot_bindings.append(
                _inert(f"arms/{arm}/snapshots/update_{update:06d}.pt")
            )
        arms[arm] = {
            "status": "COMPLETE",
            "measurement_bindings": measurement_bindings,
        }
        checkpoints[arm] = snapshot_bindings
    conditioned_action = (
        action_receipts["conditioned"]
        if action_receipts is not None
        else {
            "balanced_accuracy_one_sided_95_lower_bound": 0.3,
            "hardest_wrong_action_margin_one_sided_95_lower_bound": 0.1,
        }
    )
    result = {
        "schema": checker.RESULT_SCHEMA,
        "status": checker.RESULT_STATUS,
        "authority_binding": authority_binding,
        "plan_binding": plan_binding,
        "review_binding": _inert("/synthetic/review.json"),
        "source_commit": "a" * 40,
        "attempt": {
            "id": REPLACEMENT_ATTEMPT_ID,
            "root": str(root.resolve()),
            "maximum_attempts": 1,
            "must_be_absent": True,
            "reservation_consumes_attempt": True,
            "retry": False,
            "resume": False,
            "overwrite": False,
            "refill": False,
            "reservation": {
                "binding": _inert("reservation.json"),
                "supervisor_nonce": "b" * 64,
                "status": "RESERVED_ATTEMPT_CONSUMED",
                "maximum_attempts": 1,
                "retry": False,
                "resume": False,
                "overwrite": False,
                "refill": False,
            },
        },
        "caps": {
            "maximum_wall_seconds": 43_200,
            "maximum_gpu_seconds": 36_000,
            "maximum_training_updates": 700,
        },
        "runtime": {
            "authorized": {"device": "synthetic-authorized"},
            "observed": {
                "device_name": "AMD Radeon AI PRO R9700",
                "device_arch": "gfx1201",
                "torch_version": "2.9.1+rocm7.2.1.gitff65f5bc",
                "torch_hip": "7.2.53211-e1a6bc5663",
                "numpy_version": "1.26.4",
                "pillow_version": "11.3.0",
                "gpu_phase_elapsed_seconds": 1.0,
                "wall_elapsed_seconds": 1.25,
                "output_inventory": sorted(checker.CORE_OUTPUT_PATHS),
            },
        },
        "input_bindings": {"development_manifest": _inert("/synthetic/dev.json")},
        "predecessor_terminal_failure_binding": (
            predecessor_terminal_failure_binding
        ),
        "pack_binding": _inert("pack/manifest.json"),
        "pack_artifact_bindings": pack_artifact_bindings,
        "overlap_audit_binding": overlap_binding,
        "shuffle_audit_binding": shuffle_binding,
        "arms": arms,
        "joint_decision": {
            "status": "PASS_EXISTING_POOL_FACTUAL_ACTION_LEARNABILITY",
            "citable_as_scientific_evidence": False,
            "scientific_claim_authorized": False,
            "treatment": {
                "conditioned_action_gains": [1, 1, 1],
                "blind_action_gains": [1, 1, 0],
                "shuffled_action_gains": [1, 1, 1],
                "blind_preserves_factual_history": True,
                "shuffled_changes_only_training_candidate": True,
                "shuffled_validation_uses_factual_candidate": True,
                "requested_executed_equivalence_claimed": False,
            },
            "schedule": {
                "seed": 20_260_731,
                "updates": 700,
                "sequence_batch": 256,
                "microbatch": 32,
                "train_rows": 16_000,
                "validation_rows": 2_048,
                "warmup_updates": 150,
                "schedule_horizon_updates": 3_000,
                "observation_updates": list(checker.MEASUREMENT_UPDATES),
                "early_stopping": False,
                "checkpoint_selection": False,
            },
            "frozen_substrate": {
                "encoder_initial_sha256": "e" * 64,
                "encoder_final_sha256": "e" * 64,
                "target_initial_sha256": "f" * 64,
                "target_final_sha256": "f" * 64,
                "requires_grad": False,
                "evaluation_mode": True,
                "gradient_tensor_count": 0,
                "ema_update_count": 0,
            },
            "evidence": {
                "train_fit_update_700": {
                    "conditioned_vs_blind_family_equal_log_energy_advantage": 0.2,
                    "conditioned_vs_shuffled_family_equal_log_energy_advantage": 0.2,
                },
                "validation_tail": [
                    {
                        "update": update,
                        "conditioned_vs_blind_log_energy_advantage": 0.2,
                        "conditioned_vs_blind_one_sided_95_lower_bound": 0.1,
                        "conditioned_vs_shuffled_log_energy_advantage": 0.2,
                        "conditioned_vs_shuffled_one_sided_95_lower_bound": 0.1,
                        "prediction_to_target_rank_ratio": 0.5,
                    }
                    for update in (500, 600, 700)
                ],
                "conditioned_update_700": {
                    "balanced_accuracy_one_sided_95_lower_bound": (
                        conditioned_action[
                            "balanced_accuracy_one_sided_95_lower_bound"
                        ]
                    ),
                    "hardest_wrong_action_margin_one_sided_95_lower_bound": (
                        conditioned_action[
                            "hardest_wrong_action_margin_one_sided_95_lower_bound"
                        ]
                    ),
                    "persistence_one_sided_95_lower_bound": 0.2,
                    "wrong_history_one_sided_95_lower_bound": 0.2,
                },
            },
            "gate_precedence": list(checker._DECISION_PRECEDENCE),
        },
        "accounting": {
            "bound_h6_rows": 18_048,
            "initial_rgb_leaf_opens": 72_192,
            "verification_rgb_leaf_reopens": 192,
            "total_rgb_leaf_opens": 72_384,
            "forbidden_future_rgb_leaf_opens": 0,
            "packed_frame_bytes": 2_716_729_344,
            "training_schedule_row_presentations": 179_200,
            "sequence_presentations_per_arm": 179_200,
            "total_arm_head_sequence_presentations": 537_600,
            "shared_online_context_frame_encodings": 537_600,
            "shared_future_target_frame_encodings": 179_200,
            "actual_training_frame_encodings": 716_800,
            "optimizer_steps_per_arm": 700,
            "total_optimizer_steps": 2_100,
            "target_ema_steps": 0,
            "validation_row_panels_per_arm": 16_384,
            "shared_validation_frame_encodings": 65_536,
            "nine_way_arm_candidate_row_queries": 442_368,
            "validation_backward_calls": 0,
            "validation_optimizer_steps": 0,
            "train_fit_rows": 16_000,
            "train_fit_shared_frame_encodings": 64_000,
            "train_fit_arm_factual_row_queries": 48_000,
            "train_fit_backward_calls": 0,
            "train_fit_optimizer_steps": 0,
            "total_shared_frame_encodings": 846_336,
            "measurement_receipts": 24,
            "snapshot_bindings": 24,
            "sealed_open_count": 0,
            "heldout_open_count": 0,
            "network_access_count": 0,
            "training_consumed_pack_only": True,
        },
        "forbidden_access": {
            "sealed_material_opened": False,
            "heldout_material_opened": False,
            "network_access_used": False,
            "validation_used_for_gradient_updates": False,
            "existing_pool_modified": False,
        },
        "checkpoint_bindings": checkpoints,
    }
    if mutate_result is not None:
        mutate_result(result)
    path = root / "result.json"
    path.write_bytes(_raw(result))
    return path, result


def _producer_action_receipts() -> dict[str, dict[str, Any]]:
    from lewm.benchmarks import go2_world_model_existing_pool_three_arm_v1 as metrics
    from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as worker

    scene_actions = [
        (scene, action) for scene in range(150) for action in range(9)
    ]
    panel = [scene_actions[index % len(scene_actions)] for index in range(2_048)]
    factual = [action for _scene, action in panel]
    scenes = [f"validation-scene-{scene:03d}" for scene, _action in panel]
    families = [
        metrics.REGISTERED_FAMILIES[scene % 8] for scene, _action in panel
    ]
    identified_energies = []
    for action in factual:
        energy = [1.0] * 9
        energy[action] = 0.0
        identified_energies.append(energy)
    blind_energies = [[1.0] * 9 for _ in panel]
    identified = metrics.summarize_nine_way_action_identification(
        identified_energies,
        factual,
        scenes,
        families,
    )
    blind = metrics.summarize_nine_way_action_identification(
        blind_energies,
        factual,
        scenes,
        families,
    )
    return {
        "conditioned": worker._action_identification_receipt(identified),
        "blind": worker._action_identification_receipt(blind),
        "shuffled": worker._action_identification_receipt(identified),
    }


def _producer_overlap_audit(checker: Any) -> dict[str, Any]:
    from scripts import execute_go2_world_model_existing_pool_three_arm_v1 as worker

    missing = set(checker.EXPECTED_MISSING_ORDERED_TRAIN_TRIPLES)
    triples = [
        (first, second, third)
        for first in range(9)
        for second in range(9)
        for third in range(9)
        if (first, second, third) not in missing
    ]
    assert len(triples) == 722
    train = []
    for index in range(16_000):
        scene_index = index % 1_000
        triple = triples[index % len(triples)]
        train.append(
            {
                "index": index,
                "role": "train",
                "family": checker.REGISTERED_FAMILIES[scene_index % 8],
                "scene_id": f"train-scene-{scene_index:04d}",
                "actions": (*triple, triple[0], triple[1], triple[2]),
            }
        )
    val = []
    for index in range(2_048):
        scene_index = index % 150
        val.append(
            {
                "index": index,
                "role": "val",
                "family": checker.REGISTERED_FAMILIES[scene_index % 8],
                "scene_id": f"val-scene-{scene_index:03d}",
                "actions": tuple((index + position) % 9 for position in range(6)),
            }
        )
    return worker.build_overlap_audit(train, val)


def test_strict_json_rejects_duplicate_keys_and_nonfinite_values() -> None:
    checker = _load_checker()
    with pytest.raises(checker.ThreeArmReceiptError, match="duplicate JSON key"):
        checker.strict_json_bytes(b'{"x": 1, "x": 2}', label="fixture")
    with pytest.raises(checker.ThreeArmReceiptError, match="non-finite"):
        checker.strict_json_bytes(b'{"x": NaN}', label="fixture")


def test_replacement_result_and_report_schemas_are_exact() -> None:
    checker = _load_checker()
    assert checker.RESULT_SCHEMA == REPLACEMENT_SCHEMA_PREFIX + "result_v1"
    assert checker.REPORT_SCHEMA == REPLACEMENT_SCHEMA_PREFIX + "receipt_check_v1"
    assert checker.ATTEMPT_ID == REPLACEMENT_ATTEMPT_ID


def test_checker_rejects_stale_attempt_identity_root_and_reservation(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    cases = (
        (
            "replacement_v1_id",
            lambda result: result["attempt"].__setitem__(
                "id",
                "world_model_existing_pool_three_arm_v1_integrity_replacement_"
                "v1/attempt_v1",
            ),
        ),
        (
            "original_id",
            lambda result: result["attempt"].__setitem__(
                "id", "world_model_existing_pool_three_arm_v1/attempt_v1"
            ),
        ),
        (
            "replacement_v1_root",
            lambda result: result["attempt"].__setitem__(
                "root", "/synthetic/integrity_replacement_v1/attempt_v1"
            ),
        ),
        (
            "reservation_retry",
            lambda result: result["attempt"]["reservation"].__setitem__(
                "retry", True
            ),
        ),
    )
    for case_name, mutate in cases:
        case_root = tmp_path / case_name
        manifest, _ = _fixture(case_root, mutate_result=mutate)
        binding = checker.file_binding(manifest)
        with pytest.raises(checker.ThreeArmReceiptError, match="exact fresh V2"):
            checker.check_manifest(
                manifest,
                expected_file_sha256=binding["file_sha256"],
                expected_byte_count=binding["byte_count"],
                output_path=case_root / "receipt_check.json",
            )


def test_file_binding_rejects_protected_paths_before_open(tmp_path: Path) -> None:
    checker = _load_checker()
    protected = tmp_path / "sealed_test.json"
    with pytest.raises(checker.ThreeArmReceiptError, match="protected"):
        checker.file_binding(protected)


def test_checker_stats_but_does_not_open_inert_snapshots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(tmp_path)
    binding = checker.file_binding(manifest)
    original_read = checker._read_regular_file

    def reject_pack_open(path: Path, *, expected_bytes: int, label: str) -> bytes:
        if "pack" in Path(path).parts:
            raise AssertionError(f"checker attempted to open inert pack payload: {path}")
        return original_read(path, expected_bytes=expected_bytes, label=label)

    monkeypatch.setattr(checker, "_read_regular_file", reject_pack_open)
    report, report_binding = checker.check_manifest(
        manifest,
        expected_file_sha256=binding["file_sha256"],
        expected_byte_count=binding["byte_count"],
        output_path=tmp_path / "receipt_check.json",
    )
    assert report["status"] == "PASS"
    assert report["schema"] == REPLACEMENT_SCHEMA_PREFIX + "receipt_check_v1"
    assert report["phase"] == (
        "existing_pool_three_arm_v1_integrity_replacement_v2"
    )
    assert report["predecessor_terminal_failure_binding"] == _inert(
        "/synthetic/predecessor_terminal_failure.json"
    )
    assert report["opened_json_receipt_count"] == 26
    assert report["pack_payloads_opened"] is False
    assert report["checkpoints_opened"] is False
    assert report["rgb_bytes_opened"] is False
    assert report_binding == checker.file_binding(tmp_path / "receipt_check.json")
    assert (tmp_path / "arms/conditioned/snapshots/update_000000.pt").stat().st_size == 1
    assert (tmp_path / "pack/train_frames.u8").stat().st_size == 1
    assert len(checker.CORE_OUTPUT_PATHS) == 57
    assert "reservation.json" not in checker.CORE_OUTPUT_PATHS


def test_actual_worker_audit_and_action_receipts_pass_real_checker(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    overlap_audit = _producer_overlap_audit(checker)
    action_receipts = _producer_action_receipts()
    assert set(overlap_audit) == set(checker.OVERLAP_AUDIT_KEYS)
    assert overlap_audit["diagnostic_checks"] == {
        "train_all_ordered_triples_supported": False,
    }
    assert overlap_audit["train_support"]["missing_ordered_triples"] == [
        list(values) for values in checker.EXPECTED_MISSING_ORDERED_TRAIN_TRIPLES
    ]
    assert all(
        set(receipt) == set(checker.ACTION_IDENTIFICATION_RECEIPT_KEYS)
        for receipt in action_receipts.values()
    )

    manifest, _ = _fixture(
        tmp_path,
        overlap_audit=overlap_audit,
        action_receipts=action_receipts,
    )
    binding = checker.file_binding(manifest)
    report, _report_binding = checker.check_manifest(
        manifest,
        expected_file_sha256=binding["file_sha256"],
        expected_byte_count=binding["byte_count"],
        output_path=tmp_path / "receipt_check.json",
    )
    assert report["status"] == "PASS"


@pytest.mark.parametrize(
    ("field", "invalid"),
    (
        ("bootstrap_algorithm", "unregistered"),
        ("bootstrap_interpretation", "frequentist"),
        ("bootstrap_seed", 20_260_804),
        ("bootstrap_replicates", 9_999),
        ("bootstrap_lower_index", 499),
    ),
)
def test_checker_rejects_changed_action_bootstrap_provenance(
    tmp_path: Path,
    field: str,
    invalid: object,
) -> None:
    checker = _load_checker()

    def mutate(value: dict[str, Any]) -> None:
        value["validation"]["action_identification"][field] = invalid

    manifest, _ = _fixture(tmp_path, mutate_measurement=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="bootstrap"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("missing_family", "keys changed"),
        ("short_vector", "exactly nine actions"),
        ("non_integer", "non-negative JSON integer"),
        ("zero", "not positive"),
        ("inconsistent_minimum", "inconsistent"),
        ("below_two", "below two"),
    ),
)
def test_checker_rejects_invalid_family_action_support_audit(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    checker = _load_checker()
    first_family = checker.REGISTERED_FAMILIES[0]

    def mutate(value: dict[str, Any]) -> None:
        action = value["validation"]["action_identification"]
        support = action["family_action_supporting_scene_counts"]
        if case == "missing_family":
            support.pop(first_family)
        elif case == "short_vector":
            support[first_family] = [2] * 8
        elif case == "non_integer":
            support[first_family][0] = True
        elif case == "zero":
            support[first_family][0] = 0
            action["minimum_family_action_supporting_scene_count"] = 0
        elif case == "inconsistent_minimum":
            action["minimum_family_action_supporting_scene_count"] = 3
        elif case == "below_two":
            support[first_family][0] = 1
            action["minimum_family_action_supporting_scene_count"] = 1
        else:
            raise AssertionError(case)

    manifest, _ = _fixture(tmp_path, mutate_measurement=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match=message):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_absent_or_size_changed_inert_pack_artifact(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    manifest, result = _fixture(tmp_path)
    train_frames = tmp_path / "pack/train_frames.u8"
    train_frames.unlink()
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="inert payload is absent"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )

    train_frames.write_bytes(b"xx")
    manifest.write_bytes(_raw(result))
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="byte count changed"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_absent_inert_snapshot(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(tmp_path)
    (tmp_path / "arms/conditioned/snapshots/update_000000.pt").unlink()
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="inert payload is absent"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_wrong_measurement_arm(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(
        tmp_path,
        mutate_measurement=lambda value: value.__setitem__("arm", "blind"),
    )
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="identity"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_requires_frozen_encoder_identity_across_all_arms(
    tmp_path: Path,
) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(
        tmp_path,
        mutate_measurement=lambda value: value.__setitem__(
            "encoder_sha256", "1" * 64
        ),
    )
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="encoder/target"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_missing_registered_measurement_panel(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(
        tmp_path,
        mutate_measurement=lambda value: value["validation"].pop("controls"),
    )
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="keys changed"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_recomputes_decision_precedence(tmp_path: Path) -> None:
    checker = _load_checker()

    def mutate(value: dict[str, Any]) -> None:
        value["joint_decision"]["status"] = "LOCALIZE_TRAIN_FIT_FAILURE"

    manifest, _ = _fixture(tmp_path, mutate_result=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="precedence"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_requires_exact_registered_accounting(tmp_path: Path) -> None:
    checker = _load_checker()

    def mutate(value: dict[str, Any]) -> None:
        value["accounting"].pop("actual_training_frame_encodings")

    manifest, _ = _fixture(tmp_path, mutate_result=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="omits registered"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_overlap_audit_with_scene_leakage(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, result = _fixture(tmp_path)
    overlap_path = tmp_path / "overlap_audit.json"
    overlap = json.loads(overlap_path.read_text(encoding="utf-8"))
    overlap["audit"]["role_scene_overlap_count"] = 1
    overlap["audit"]["role_scene_overlap"] = ["leaked-scene"]
    overlap["audit"]["checks"]["role_scene_disjointness"] = False
    result["overlap_audit_binding"] = _write(
        tmp_path, "overlap_audit.json", overlap
    )
    manifest.write_bytes(_raw(result))
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="support/split"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_measurement_binding_into_snapshot_tree(tmp_path: Path) -> None:
    checker = _load_checker()

    def mutate(value: dict[str, Any]) -> None:
        value["arms"]["conditioned"]["measurement_bindings"][0]["path"] = (
            "arms/conditioned/snapshots/update_000000.json"
        )

    manifest, _ = _fixture(tmp_path, mutate_result=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="forbidden"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_requires_safe_forbidden_access_assertions(tmp_path: Path) -> None:
    checker = _load_checker()

    def mutate(value: dict[str, Any]) -> None:
        value["forbidden_access"]["validation_used_for_gradient_updates"] = True

    manifest, _ = _fixture(tmp_path, mutate_result=mutate)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="unsafe"):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_rejects_result_binding_mismatch(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(tmp_path)
    binding = checker.file_binding(manifest)
    with pytest.raises(checker.ThreeArmReceiptError, match="SHA-256"):
        checker.check_manifest(
            manifest,
            expected_file_sha256="0" * 64,
            expected_byte_count=binding["byte_count"],
            output_path=tmp_path / "receipt_check.json",
        )


def test_checker_output_is_exclusive(tmp_path: Path) -> None:
    checker = _load_checker()
    manifest, _ = _fixture(tmp_path)
    binding = checker.file_binding(manifest)
    output = tmp_path / "receipt_check.json"
    checker.check_manifest(
        manifest,
        expected_file_sha256=binding["file_sha256"],
        expected_byte_count=binding["byte_count"],
        output_path=output,
    )
    with pytest.raises(FileExistsError):
        checker.check_manifest(
            manifest,
            expected_file_sha256=binding["file_sha256"],
            expected_byte_count=binding["byte_count"],
            output_path=output,
        )


def test_help_is_source_only() -> None:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--expected-file-sha256" in completed.stdout
