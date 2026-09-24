from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from lewm.benchmarks.go2_categorical_radial_n32 import (
    CONDITIONS,
    FAMILIES,
    HOLDOUT_PANELS,
    REFERENCE_MACRO_ASSERTIONS,
    SMOKE_RESULT_SCHEMA,
    categorical_holdout_checks,
    extract_faithful_patch7_family_reference,
    fit_panel_gate_report,
    per_seed_decision,
    terminal_fit_gate_summary,
)
from scripts import finalize_go2_categorical_radial_n32 as finalizer


REGISTERED_SCHEDULE_SHA256 = dict(finalizer.SCHEDULE_SHA256)


def _metrics(*, nll: float, score: float, far: float) -> dict:
    return {
        "raw_hierarchical_balanced_nll": nll,
        "unknown_known_balanced_accuracy": score,
        "free_occupied_balanced_accuracy": score,
        "class_recall": {
            "unknown": score,
            "free": score,
            "occupied": score,
        },
        "distance_free_recall": {
            "0.0_to_0.5": None,
            "0.5_to_1.0": None,
            "1.0_to_2.0": far,
            "2.0_to_3.0": far,
            "3.0_plus": far,
        },
        "distance_free_support": {
            "0.0_to_0.5": 0,
            "0.5_to_1.0": 0,
            "1.0_to_2.0": 100,
            "2.0_to_3.0": 100,
            "3.0_plus": 100,
        },
    }


def _conditions(*, nll: float, score: float, far: float) -> dict:
    return {
        "correct_rgb": _metrics(nll=nll, score=score, far=far),
        "role_global_shuffled_rgb": {"raw_hierarchical_balanced_nll": 0.5},
        "same_scene_wrong_view_rgb": {"raw_hierarchical_balanced_nll": 0.5},
    }


def _controls(*, seed: int, panel: str) -> dict:
    return {
        "role_global_shuffle": {
            "seed": seed,
            "namespace": panel,
            "record_count": 320,
            "permutation_sha256": "1" * 64,
            "same_image_pairs": 0,
            "same_scene_pairs": 0,
            "same_transition_pairs": 0,
        },
        "same_scene_wrong_view": {
            "seed": seed,
            "namespace": panel,
            "record_count": 320,
            "permutation_sha256": "2" * 64,
            "same_image_pairs": 0,
            "same_transition_pairs": 0,
            "different_scene_pairs": 0,
            "scenes": {},
        },
    }


def _panel_report(
    *,
    panel: str,
    seed: int,
    nll: float,
    score: float,
    far: float,
) -> dict:
    conditions = _conditions(nll=nll, score=score, far=far)
    report = {
        "schema": finalizer.PANEL_REPORT_SCHEMA,
        "panel": panel,
        "frame_count": 320,
        "target_batch_size": 4,
        "combined_model_batch_size": 12,
        "model_call_dtype": "float32",
        "metric_accumulator_dtype": "float64",
        "conditions": copy.deepcopy(conditions),
        "families": {
            family: {"conditions": copy.deepcopy(conditions)} for family in FAMILIES
        },
        "controls": _controls(seed=seed, panel=panel),
    }
    if panel == "fit":
        report["fit_gate"] = fit_panel_gate_report(report)
    return report


def _patch7_reference() -> dict:
    panels = {}
    for panel in HOLDOUT_PANELS:
        assertion = REFERENCE_MACRO_ASSERTIONS[panel]
        metrics = _metrics(
            nll=assertion["hierarchical_nll"],
            score=0.8,
            far=assertion["far_free_recall"],
        )
        conditions = {
            "correct_rgb": copy.deepcopy(metrics),
            "role_global_shuffled_rgb": {
                "raw_hierarchical_balanced_nll": 0.5
            },
            "same_scene_wrong_view_rgb": {
                "raw_hierarchical_balanced_nll": 0.5
            },
        }
        controls = _controls(seed=20260710, panel=panel)
        panels[panel] = {
            "panel": panel,
            "frame_count": 320,
            "conditions": copy.deepcopy(conditions),
            "families": {
                family: {"conditions": copy.deepcopy(conditions)}
                for family in FAMILIES
            },
            "role_global_shuffle": controls["role_global_shuffle"],
            "same_scene_wrong_view": controls["same_scene_wrong_view"],
            "access": {
                "image_decode_events": 960,
                "label_access_events": 320,
                "label_shard_npz_open_events": 20,
                "distinct_image_paths_opened": 320,
                "distinct_label_shards_opened": 20,
                "non_train_image_opens": 0,
                "non_train_label_shard_opens": 0,
            },
        }
    patch7 = {
        "stages": {
            "production_faithful": {
                "patch7_16x16": {
                    "final_state_sha256": finalizer.PATCH7_FINAL_STATE_SHA256,
                    "final_panels": panels,
                }
            }
        },
        "post_selection_support_audit": {
            "fit": {},
            "same_scene_holdout": {},
            "cross_scene_holdout": {},
        },
    }
    return extract_faithful_patch7_family_reference(patch7)


def _candidate_holdouts(*, seed: int, reference: dict) -> dict:
    return {
        panel: _panel_report(
            panel=panel,
            seed=seed,
            nll=REFERENCE_MACRO_ASSERTIONS[panel]["hierarchical_nll"] * 0.5,
            score=0.85,
            far=REFERENCE_MACRO_ASSERTIONS[panel]["far_free_recall"] + 0.15,
        )
        for panel in HOLDOUT_PANELS
    }


def _minibatches(updates: int, seed: int) -> list[list[int]]:
    offset = seed % 320
    return [
        [((step * 4) + index + offset) % 320 for index in range(4)]
        for step in range(updates)
    ]


@pytest.fixture(autouse=True)
def _synthetic_schedule_commitments(monkeypatch: pytest.MonkeyPatch) -> None:
    for seed in finalizer.EXPECTED_SEEDS:
        for config in finalizer.BRANCH_CONFIGS.values():
            updates = int(config["updates"])
            monkeypatch.setitem(
                finalizer.SCHEDULE_SHA256,
                (seed, updates),
                finalizer._canonical_json_sha256(_minibatches(updates, seed)),
            )


def _stage(
    *,
    stage_name: str,
    seed: int,
    passes: bool,
    holdouts_evaluated: bool,
) -> dict:
    config = finalizer.BRANCH_CONFIGS[stage_name]
    updates = int(config["updates"])
    fit_report = _panel_report(
        panel="fit",
        seed=seed,
        nll=0.01 if passes else 0.2,
        score=0.995 if passes else 0.5,
        far=0.995 if passes else 0.5,
    )
    curve = [
        {
            "step": step,
            "batch_loss": 0.01,
            "gradient_norm_before_clip": 0.5,
            "fit_panel": copy.deepcopy(fit_report),
        }
        for step in range(100, updates + 1, 100)
    ]
    batches = _minibatches(updates, seed)
    initial_state = (
        finalizer.EXPECTED_SEED10_INITIAL_STATE_SHA256
        if seed == 20260710
        else finalizer.EXPECTED_SEED11_INITIAL_STATE_SHA256
    )
    return {
        "schema": finalizer.STAGE_SCHEMA,
        "stage": stage_name,
        "maximum_steps": updates,
        "completed_steps": updates,
        "batch_size": 4,
        "evaluation_interval": 100,
        "optimizer": finalizer._expected_optimizer(stage_name),
        "fixed_update_budget_consumed": True,
        "initial_state_sha256": initial_state,
        "final_state_sha256": "c" * 64,
        "minibatch_indices": batches,
        "minibatch_indices_sha256": finalizer._canonical_json_sha256(batches),
        "learning_curve": curve,
        "terminal_fit_gate": terminal_fit_gate_summary(curve, updates, 100),
        "training_access": {
            "image_requests": updates * 4,
            "target_requests": updates * 4,
            "image_decode_events": 320 if stage_name == "production_faithful" else 0,
            "label_shard_npz_open_events": (
                20 if stage_name == "production_faithful" else 0
            ),
            "model_calls": updates,
            "model_output_frames": updates * 4,
        },
        "fit_evaluation_access": {
            "image_requests": (updates // 100) * 960,
            "target_requests": (updates // 100) * 320,
            "image_decode_events": 0,
            "label_shard_npz_open_events": 0,
            "model_calls": (updates // 100) * 80,
            "model_output_frames": (updates // 100) * 80 * 12,
        },
        "holdouts_evaluated": holdouts_evaluated,
    }


def _source_hashes() -> dict:
    return finalizer._runner_source_hashes()


def _access_ledger(*, branch: str, holdouts_authorized: bool) -> dict:
    panels = {
        "fit": {
            "authorized": True,
            "artifact_hash_passes": 2,
            "image_hash_byte_open_events": 640,
            "shard_hash_byte_open_events": 40,
        }
    }
    for panel in HOLDOUT_PANELS:
        counts = finalizer.EXPECTED_PANEL_ARTIFACT_COUNTS[panel]
        if holdouts_authorized:
            panels[panel] = {
                "authorized": True,
                "artifact_hash_passes": 2,
                "image_hash_byte_open_events": 640,
                "shard_hash_byte_open_events": 2 * counts["shards"],
                "dataset_access": {
                    "image_decode_events": 320,
                    "label_shard_npz_open_events": counts["shards"],
                    "image_requests": 960,
                    "target_requests": 320,
                    "model_calls": 80,
                    "model_output_frames": 960,
                },
            }
        else:
            panels[panel] = {
                "authorized": False,
                "artifact_hash_passes": 0,
                "image_hash_byte_open_events": 0,
                "shard_hash_byte_open_events": 0,
                "model_output_frames": 0,
            }
    zero = {
        "image_byte_opens": 0,
        "label_shard_byte_opens": 0,
        "model_outputs": 0,
    }
    invoked = [finalizer.BRANCH_CONFIGS["production_faithful"]]
    if branch == "ceiling_optimizer":
        invoked.append(finalizer.BRANCH_CONFIGS["ceiling_optimizer"])
    fit_image_requests = sum(
        int(config["updates"]) * 4
        + (int(config["updates"]) // 100) * 960
        for config in invoked
    )
    fit_target_requests = sum(
        int(config["updates"]) * 4
        + (int(config["updates"]) // 100) * 320
        for config in invoked
    )
    return {
        "panels": panels,
        "fit_dataset_totals": {
            "image_decode_events": 320,
            "label_shard_npz_open_events": 20,
            "image_requests": fit_image_requests,
            "target_requests": fit_target_requests,
        },
        "checkpoint_selection": copy.deepcopy(zero),
        "probability_calibration": copy.deepcopy(zero),
        "g2_evaluation": copy.deepcopy(zero),
        "non_train_image_opens": 0,
        "non_train_label_shard_opens": 0,
        "non_train_model_outputs": 0,
    }


def _artifact(*, seed: int, branch: str = "production_faithful") -> dict:
    reference = _patch7_reference()
    faithful_passes = branch == "production_faithful"
    faithful = _stage(
        stage_name="production_faithful",
        seed=seed,
        passes=faithful_passes,
        holdouts_evaluated=faithful_passes,
    )
    ceiling = None
    if not faithful_passes:
        ceiling = _stage(
            stage_name="ceiling_optimizer",
            seed=seed,
            passes=True,
            holdouts_evaluated=True,
        )
    holdouts = _candidate_holdouts(seed=seed, reference=reference)
    checks = {
        panel: categorical_holdout_checks(
            holdouts[panel],
            reference["panels"][panel],
        )
        for panel in HOLDOUT_PANELS
    }
    decision = per_seed_decision(faithful, ceiling, checks)
    initial_state = (
        finalizer.EXPECTED_SEED10_INITIAL_STATE_SHA256
        if seed == 20260710
        else finalizer.EXPECTED_SEED11_INITIAL_STATE_SHA256
    )
    evidence = {
        str(path.resolve()): digest
        for path, digest in finalizer.BOUND_EVIDENCE.items()
    }
    core = {
        "schema": finalizer.RESULT_SCHEMA,
        "authoritative": True,
        "aggregation_eligible": True,
        "promotion_eligible": False,
        "seed": seed,
        "created_at_utc": "2026-07-10T22:00:00+00:00",
        "completed_at_utc": "2026-07-10T22:01:00+00:00",
        "invocation": ["runner", "--seed", str(seed)],
        "execution": {
            "device": "cpu",
            "device_name": "cpu",
            "determinism": {
                "seed": seed,
                "requested": "strict_deterministic_algorithms",
                "effective": "strict_where_supported_warn_on_unsupported",
                "warn_only": True,
                "torch_deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
            },
            "batch_size_frames": 4,
            "evaluation_interval": 100,
            "branches": copy.deepcopy(finalizer.BRANCH_CONFIGS),
            "fp32_no_autocast_amp_compile_or_quantization": True,
        },
        "contract": {
            "path": str(
                finalizer.RUNNER_SOURCE_BINDINGS["n32_contract"].resolve()
            ),
            "sha256": finalizer.EXECUTION_BINDING_SHA256,
        },
        "inputs": {
            name: {
                "path": str(
                    {
                        "panel": finalizer.PANEL_PATH,
                        "ladder_manifest": finalizer.LADDER_PATH,
                        "v3_result": finalizer.V3_RESULT_PATH,
                        "patch7_reference_result": finalizer.PATCH7_RESULT_PATH,
                    }[name].resolve()
                ),
                "sha256": file_hash,
                "content_sha256": content_hash,
            }
            for name, (file_hash, content_hash) in finalizer.EXPECTED_INPUTS.items()
        },
        "source_hashes": _source_hashes(),
        "git": {"start": {}, "end": {}},
        "model": {
            "class": "CategoricalRadialPerceptionFullRay",
            "parameter_count": finalizer.REGISTERED_PARAMETER_COUNT,
            "initial_state_sha256": initial_state,
            "all_invoked_branches_restart_same_initial_state": True,
        },
        "stages": {
            "production_faithful": faithful,
            "ceiling_optimizer": ceiling,
        },
        "patch7_reference": reference,
        "holdouts": holdouts,
        "holdout_checks": checks,
        "decision": decision,
        "artifact_verification": {
            "fit_verified_before_access": True,
            "holdouts_verified_only_after_terminal_fit_pass": True,
            "evidence_hashes": evidence,
        },
        "access_ledger": _access_ledger(
            branch=branch,
            holdouts_authorized=True,
        ),
        "categorical_radial_full_train_candidate_licensed": False,
    }
    core["inputs"]["seed_20260710_authorization"] = (
        None
        if seed == 20260710
        else {
            "path": str(finalizer.CANONICAL_RESULT_PATHS[20260710]),
            "sha256": "a" * 64,
        }
    )
    return {**core, "content_sha256": finalizer._canonical_json_sha256(core)}


def _rehash(artifact: dict) -> None:
    core = dict(artifact)
    core.pop("content_sha256", None)
    artifact["content_sha256"] = finalizer._canonical_json_sha256(core)


def _write(path: Path, artifact: dict) -> str:
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    return finalizer._sha256_file(path)


def _write_pair(
    tmp_path: Path,
    *,
    primary_branch: str = "production_faithful",
    replication_branch: str = "production_faithful",
) -> tuple[Path, str, Path, str]:
    primary_path = tmp_path / "seed10.json"
    replication_path = tmp_path / "seed11.json"
    primary = _artifact(seed=20260710, branch=primary_branch)
    primary_hash = _write(primary_path, primary)
    replication = _artifact(seed=20260711, branch=replication_branch)
    replication["inputs"]["seed_20260710_authorization"] = {
        "path": str(primary_path),
        "sha256": primary_hash,
    }
    _rehash(replication)
    replication_hash = _write(replication_path, replication)
    return primary_path, primary_hash, replication_path, replication_hash


def _bind_test_result_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(
        finalizer.CANONICAL_RESULT_PATHS,
        20260710,
        tmp_path / "seed10.json",
    )
    monkeypatch.setitem(
        finalizer.CANONICAL_RESULT_PATHS,
        20260711,
        tmp_path / "seed11.json",
    )


def _bind_synthetic_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    hashes = {
        str(path.resolve()): finalizer._sha256_file(path)
        for path in finalizer.BOUND_EVIDENCE
    }
    reference = _patch7_reference()
    monkeypatch.setattr(
        finalizer,
        "_load_bound_evidence",
        lambda: {
            "pre_hashes": hashes,
            "post_parse_hashes": hashes,
            "panels": {},
            "patch7_reference": reference,
        },
    )
    monkeypatch.setattr(
        finalizer,
        "_expected_controls",
        lambda _panels, seed: {
            panel: _controls(seed=seed, panel=panel)
            for panel in ("fit", *HOLDOUT_PANELS)
        },
    )


def _main_args(
    primary: Path,
    primary_hash: str,
    replication: Path,
    replication_hash: str,
    output: Path,
) -> list[str]:
    return [
        "--seed-20260710-result",
        str(primary),
        "--expected-seed-20260710-result-sha256",
        primary_hash,
        "--seed-20260711-result",
        str(replication),
        "--expected-seed-20260711-result-sha256",
        replication_hash,
        "--output",
        str(output),
    ]


def test_favorable_same_branch_licenses_full_training_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_result_paths(tmp_path, monkeypatch)
    _bind_synthetic_evidence(monkeypatch)
    primary, primary_hash, replication, replication_hash = _write_pair(tmp_path)
    output = tmp_path / "final.json"
    assert finalizer.main(
        _main_args(primary, primary_hash, replication, replication_hash, output)
    ) == 0
    result = json.loads(output.read_text())
    assert result["schema"] == finalizer.TWO_SEED_RESULT_SCHEMA
    assert result["aggregation"]["classification"] == (
        "two_seed_favorable_same_branch"
    )
    assert result["categorical_radial_full_train_candidate_licensed"] is True


def test_branch_disagreement_is_valid_two_seed_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _bind_test_result_paths(tmp_path, monkeypatch)
    _bind_synthetic_evidence(monkeypatch)
    primary, primary_hash, replication, replication_hash = _write_pair(
        tmp_path,
        replication_branch="ceiling_optimizer",
    )
    output = tmp_path / "final.json"
    assert finalizer.main(
        _main_args(primary, primary_hash, replication, replication_hash, output)
    ) == 0
    result = json.loads(output.read_text())
    assert result["aggregation"]["classification"] == "two_seed_inconclusive"
    assert result["categorical_radial_full_train_candidate_licensed"] is False


def test_metric_tamper_is_rejected_after_content_rehash() -> None:
    artifact = _artifact(seed=20260710)
    artifact["holdouts"]["same_scene_holdout"]["families"][FAMILIES[0]][
        "conditions"
    ]["correct_rgb"]["raw_hierarchical_balanced_nll"] = 0.9
    _rehash(artifact)
    with pytest.raises(ValueError, match="holdout comparison"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_curve_cadence_tamper_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    artifact["stages"]["production_faithful"]["learning_curve"].pop(0)
    _rehash(artifact)
    with pytest.raises(ValueError, match="cadence|curve"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_forbidden_access_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    artifact["access_ledger"]["non_train_image_opens"] = 1
    _rehash(artifact)
    with pytest.raises(ValueError, match="forbidden non-train access"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_source_drift_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    transitive_name = next(
        name for name in artifact["source_hashes"] if name not in {"runner", "n32_pure"}
    )
    artifact["source_hashes"][transitive_name]["sha256"] = "d" * 64
    _rehash(artifact)
    with pytest.raises(ValueError, match="transitive frozen source drift"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_exact_seeded_schedule_tamper_is_rejected_even_if_epoch_is_preserved() -> None:
    artifact = _artifact(seed=20260710)
    stage = artifact["stages"]["production_faithful"]
    stage["minibatch_indices"][0], stage["minibatch_indices"][1] = (
        stage["minibatch_indices"][1],
        stage["minibatch_indices"][0],
    )
    stage["minibatch_indices_sha256"] = finalizer._canonical_json_sha256(
        stage["minibatch_indices"]
    )
    _rehash(artifact)
    with pytest.raises(ValueError, match="exact seeded minibatch"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_transitive_stage_request_access_tamper_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    artifact["stages"]["production_faithful"]["training_access"][
        "image_requests"
    ] += 1
    _rehash(artifact)
    with pytest.raises(ValueError, match="model-output access"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_exact_control_sha_tamper_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    expected = {
        panel: copy.deepcopy(artifact["holdouts"][panel]["controls"])
        for panel in HOLDOUT_PANELS
    }
    expected["fit"] = copy.deepcopy(
        artifact["stages"]["production_faithful"]["learning_curve"][0][
            "fit_panel"
        ]["controls"]
    )
    artifact["holdouts"]["same_scene_holdout"]["controls"][
        "role_global_shuffle"
    ]["permutation_sha256"] = "3" * 64
    _rehash(artifact)
    with pytest.raises(ValueError, match="exact deterministic control"):
        finalizer._validate_authoritative_result(
            artifact,
            expected_seed=20260710,
            expected_controls=expected,
        )


def test_bound_evidence_path_hash_tamper_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    evidence = artifact["artifact_verification"]["evidence_hashes"]
    path = next(iter(evidence))
    evidence[path] = "4" * 64
    _rehash(artifact)
    with pytest.raises(ValueError, match="evidence path/hash"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_positive_nested_unauthorized_holdout_access_is_rejected() -> None:
    ledger = _access_ledger(
        branch="production_faithful",
        holdouts_authorized=False,
    )
    ledger["panels"]["same_scene_holdout"]["dataset_access"] = {
        "image_requests": 1
    }
    with pytest.raises(ValueError, match="fields drift|unauthorized dataset access"):
        finalizer._validate_panel_access(ledger, holdouts_authorized=False)


def test_replication_failure_has_distinct_classification() -> None:
    primary = {"decision": {"favorable": True, "qualifying_optimizer_stage": "production_faithful"}}
    replication = {
        "decision": {
            "favorable": False,
            "qualifying_optimizer_stage": "production_faithful",
        }
    }
    result = finalizer._aggregate_validated_results(primary, replication)
    assert result["classification"] == "two_seed_replication_failed"
    assert result["categorical_radial_full_train_candidate_licensed"] is False


def test_registered_schedule_commitments_match_execution_binding() -> None:
    assert REGISTERED_SCHEDULE_SHA256 == {
        (20260710, 2000): "3de32de003991942d8e08f0d12296b6b3018831225394c12ba2da438cc94ab02",
        (20260710, 5000): "0bc06fd8bef9bbf49da8459104ccb1dbb7994aa0a7e99b560b244e91a1690b8d",
        (20260711, 2000): "34a5e5256c939be00c40e9594b05d2087416d5c1275d44e5904bc3dcb29d6e4b",
        (20260711, 5000): "304c1d87a5719900d12b0fc6caedc3ef6abb6f3d88e035b07ff4421bfb060cc7",
    }


def test_seed11_initial_state_hash_is_pinned() -> None:
    artifact = _artifact(seed=20260711)
    artifact["model"]["initial_state_sha256"] = "b" * 64
    artifact["stages"]["production_faithful"]["initial_state_sha256"] = "b" * 64
    _rehash(artifact)
    with pytest.raises(ValueError, match="initialization drift"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260711)


def test_execution_binding_path_requires_resolved_equality() -> None:
    artifact = _artifact(seed=20260710)
    artifact["contract"]["path"] = (
        "/tmp/docs/lewm_go2_categorical_radial_n32_execution_binding_2026-07-10.md"
    )
    _rehash(artifact)
    with pytest.raises(ValueError, match="execution binding drift"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_string_coercible_minibatch_index_is_rejected_with_matching_raw_hash() -> None:
    artifact = _artifact(seed=20260710)
    stage = artifact["stages"]["production_faithful"]
    stage["minibatch_indices"][0][0] = str(stage["minibatch_indices"][0][0])
    stage["minibatch_indices_sha256"] = finalizer._canonical_json_sha256(
        stage["minibatch_indices"]
    )
    _rehash(artifact)
    with pytest.raises(ValueError, match="JSON integers"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_access_ledger_rejects_unknown_top_level_fields() -> None:
    artifact = _artifact(seed=20260710)
    artifact["access_ledger"]["unvalidated_extra"] = 0
    _rehash(artifact)
    with pytest.raises(ValueError, match="top-level fields drift"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("seed", 0.9),
        ("execution_batch", 4.9),
        ("panel_frame_count", 320.9),
        ("stage_completed_steps", 2000.9),
        ("access_count", 0.9),
    ),
)
def test_fractional_structural_counts_are_rejected(field: str, value: float) -> None:
    artifact = _artifact(seed=20260710)
    if field == "seed":
        artifact["seed"] = value
    elif field == "execution_batch":
        artifact["execution"]["batch_size_frames"] = value
    elif field == "panel_frame_count":
        artifact["holdouts"]["same_scene_holdout"]["frame_count"] = value
    elif field == "stage_completed_steps":
        artifact["stages"]["production_faithful"]["completed_steps"] = value
    else:
        artifact["access_ledger"]["non_train_image_opens"] = value
    _rehash(artifact)
    with pytest.raises(ValueError, match="JSON integer"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


@pytest.mark.parametrize(
    ("metric_path", "value"),
    (
        ("nll", -0.1),
        ("recall", 1.5),
        ("support", 1.5),
        ("confusion", [[0.5]]),
    ),
)
def test_favorable_artifact_rejects_invalid_physical_metric_domains(
    metric_path: str,
    value: object,
) -> None:
    artifact = _artifact(seed=20260710)
    metrics = artifact["holdouts"]["same_scene_holdout"]["families"][FAMILIES[0]][
        "conditions"
    ]["correct_rgb"]
    if metric_path == "nll":
        metrics["raw_hierarchical_balanced_nll"] = value
    elif metric_path == "recall":
        metrics["class_recall"]["free"] = value
    elif metric_path == "support":
        metrics["distance_free_support"]["3.0_plus"] = value
    else:
        metrics["joint_confusion"] = value
    _rehash(artifact)
    with pytest.raises(ValueError, match=r"nonnegative|\[0, 1\]|JSON integer"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


@pytest.mark.parametrize("value", ("0.01", True))
def test_curve_scalars_reject_string_and_bool_aliases(value: object) -> None:
    artifact = _artifact(seed=20260710)
    artifact["stages"]["production_faithful"]["learning_curve"][0][
        "batch_loss"
    ] = value
    _rehash(artifact)
    with pytest.raises(ValueError, match="finite JSON number"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


@pytest.mark.parametrize(
    ("name", "value"),
    (("amsgrad", 0), ("constant_learning_rate", 1), ("gradient_clip", True)),
)
def test_optimizer_rejects_bool_integer_aliases(name: str, value: object) -> None:
    artifact = _artifact(seed=20260710)
    artifact["stages"]["production_faithful"]["optimizer"][name] = value
    _rehash(artifact)
    with pytest.raises(ValueError, match="JSON boolean|finite JSON number"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_determinism_rejects_integer_boolean_alias() -> None:
    artifact = _artifact(seed=20260710)
    artifact["execution"]["determinism"]["warn_only"] = 1
    _rehash(artifact)
    with pytest.raises(ValueError, match="JSON boolean"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_sha256_fields_require_json_strings() -> None:
    artifact = _artifact(seed=20260710)
    artifact["stages"]["production_faithful"]["final_state_sha256"] = int("1" * 64)
    _rehash(artifact)
    with pytest.raises(ValueError, match="contract drift"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_precommitted_file_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "seed10.json"
    _write(path, _artifact(seed=20260710))
    with pytest.raises(ValueError, match="precommitted SHA-256"):
        finalizer._load_expected_json(path, expected_sha256="0" * 64)


def test_smoke_artifact_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    artifact["schema"] = SMOKE_RESULT_SCHEMA
    artifact["authoritative"] = False
    artifact["aggregation_eligible"] = False
    _rehash(artifact)
    with pytest.raises(ValueError, match="smoke"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_self_license_is_rejected() -> None:
    artifact = _artifact(seed=20260710)
    artifact["categorical_radial_full_train_candidate_licensed"] = True
    _rehash(artifact)
    with pytest.raises(ValueError, match="authoritative and aggregation eligible"):
        finalizer._validate_authoritative_result(artifact, expected_seed=20260710)


def test_atomic_exclusive_output_refuses_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    finalizer._atomic_write_json_exclusive(output, {"version": 1})
    with pytest.raises(FileExistsError, match="already exists"):
        finalizer._atomic_write_json_exclusive(output, {"version": 2})
