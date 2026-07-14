from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from lewm.benchmarks import go2_shared_jepa_v5_matched_training_v1 as contract
from lewm.models import (
    shared_observable_camera_ray_jepa_v5_full_training_v4_loss as corrected_loss,
)
from scripts import run_go2_shared_jepa_v5_matched_training_v1 as runner


ROOT = Path(__file__).resolve().parents[2]


def _review(sources: dict[str, str]) -> dict[str, object]:
    return contract.with_content_sha256(
        {
            "schema": contract.REVIEW_SCHEMA,
            "status": "PASS",
            "implementation_author": contract.IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/lean_shared_v5_independent_review",
            "reviewed_sources": sources,
            "science_contract": contract.science_contract(),
            "source_only": True,
            "findings": [],
            "authority": contract.REVIEW_AUTHORITY,
        }
    )


def _binding(path: str, file_hash: str, content_hash: str) -> dict[str, object]:
    return {
        "path": path,
        "file_sha256": file_hash,
        "content_sha256": content_hash,
        "byte_count": 123,
    }


def _authorization(review_binding: dict[str, object]) -> dict[str, object]:
    return contract.with_content_sha256(
        {
            "schema": contract.AUTHORIZATION_SCHEMA,
            "status": "authorized_one_exact_development_attempt",
            "authorizer": "/root",
            "independent_review": review_binding,
            "raw": {
                "root": contract.RAW_ROOT_RELATIVE_PATH,
                "manifest": _binding(
                    contract.RAW_MANIFEST_RELATIVE_PATH,
                    contract.RAW_MANIFEST_FILE_SHA256,
                    contract.RAW_MANIFEST_CONTENT_SHA256,
                ),
                "audit": _binding(
                    contract.RAW_AUDIT_RELATIVE_PATH,
                    contract.RAW_AUDIT_FILE_SHA256,
                    contract.RAW_AUDIT_CONTENT_SHA256,
                ),
                "role_counts": contract.ROLE_COUNTS,
                "grant": contract._expected_raw_authority(),
            },
            "camera": {
                "root": contract.CAMERA_ROOT_RELATIVE_PATH,
                "gate": _binding(
                    contract.CAMERA_GATE_RELATIVE_PATH, "1" * 64, "2" * 64
                ),
                "checkpoint": _binding(
                    contract.CAMERA_CHECKPOINT_RELATIVE_PATH, "3" * 64, "4" * 64
                ),
                "seed": 20260710,
                "fit_size": 320,
                "updates": 40_000,
                "gate_must_pass_all_checks": 26,
            },
            "experiment": contract.science_contract(),
            "authority": contract.EXECUTION_AUTHORITY,
        }
    )


def _passing_scope() -> dict[str, object]:
    return {
        "physical": {
            **{name: 0.99 for name in contract.PHYSICAL_LOWER_THRESHOLDS},
            "depth_median_error_m": 0.01,
            "depth_p95_error_m": 0.02,
            "derived_raster_nll": 0.01,
            "distance_group_balanced_accuracy": [0.99, 0.99],
            "present_class_recall": {
                "UNKNOWN": 0.99,
                "FREE": 0.99,
                "OCCUPIED": 0.99,
            },
        },
        "jepa": {
            "prediction_valid_cell_count": 100,
            "target_cross_sample_std_mean": 0.10,
            "target_cross_sample_effective_rank": 8.0,
            "warped_persistence_target_change": 0.10,
            "prediction_to_warped_persistence_ratio": 0.5,
            "wrong_action_advantage_over_target_change": 0.20,
            "wrong_commanded_delta_advantage_over_target_change": 0.20,
            "wrong_action_prediction_sensitivity": 0.20,
            "wrong_commanded_delta_prediction_sensitivity": 0.20,
        },
    }


def _candidate(update: int, *, v4_loss: float = 0.1) -> dict[str, object]:
    return {
        "update": update,
        "scopes": {scope: _passing_scope() for scope in contract.SCOPES},
        "aggregate_complete_v4_loss": v4_loss,
        "aggregate_prediction_to_persistence_ratio": 0.5,
    }


def test_frozen_science_is_the_small_matched_experiment() -> None:
    science = contract.science_contract()
    assert science["arms"]["order"] == ["promoted_jepa", "matched_no_jepa"]
    assert science["arms"]["sole_backward_difference"] == {
        "promoted_jepa": "established_jepa_plus_camera",
        "matched_no_jepa": "camera_only",
    }
    assert science["presentation_count"] == 128_000
    assert science["update_count"] == 8_000
    assert science["optimizer"] == contract.OPTIMIZER_CONTRACT
    assert science["corrected_loss"]["camera_terms"] == list(contract.CAMERA_TERMS)
    assert science["corrected_loss"]["real_microbatch_size"] == 4
    assert science["corrected_loss"]["synthetic_b16_pooling"] is False
    assert science["maximum_attempts"] == 1
    assert science["retry_authorized"] is False


def test_review_and_authorization_are_strict_and_separate() -> None:
    sources = {path: hashlib.sha256(path.encode()).hexdigest() for path in contract.SOURCE_PATHS}
    review = _review(sources)
    assert contract.validate_review(review, expected_sources=sources) == review
    review_raw = contract.canonical_json_bytes(review) + b"\n"
    parsed_review = contract.parse_canonical_json(review_raw, name="review")
    assert contract.validate_review(parsed_review, expected_sources=sources) == parsed_review
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=review["content_sha256"],
    )
    authorization = _authorization(review_binding)
    assert contract.validate_authorization(
        authorization, review_binding=review_binding
    ) == authorization
    assert review["authority"]["training_authorized"] is False
    assert authorization["authority"]["matched_training_authorized"] is True
    assert authorization["authority"][
        "matched_selected_update_diagnostic_authorized"
    ] is True
    assert authorization["authority"]["g2_authorized"] is False

    extra = dict(review)
    extra["undeclared"] = False
    with pytest.raises(PermissionError):
        contract.validate_review(extra, expected_sources=sources)
    swapped = dict(authorization)
    swapped["raw"] = dict(swapped["raw"])
    swapped["raw"]["role_counts"] = {
        **contract.ROLE_COUNTS,
        "train": {**contract.ROLE_COUNTS["train"], "pairs": 4261},
    }
    swapped_core = dict(swapped)
    swapped_core.pop("content_sha256")
    swapped["content_sha256"] = contract.canonical_json_sha256(swapped_core)
    with pytest.raises(PermissionError):
        contract.validate_authorization(swapped, review_binding=review_binding)


def test_bound_sources_pin_the_retained_model_and_corrected_loss() -> None:
    observed = contract.current_source_bindings(ROOT)
    assert tuple(observed) == contract.SOURCE_PATHS
    assert observed[contract.MODEL_RELATIVE_PATH] == contract.MODEL_FILE_SHA256
    assert observed[contract.LOSS_RELATIVE_PATH] == contract.LOSS_FILE_SHA256
    assert "go2_shared_jepa_v5_full_training_v4_policy" not in (
        ROOT / contract.RUNNER_RELATIVE_PATH
    ).read_text()


def test_schedule_is_train_only_exact_and_shared() -> None:
    complete, remainder = divmod(
        contract.PRESENTATION_COUNT, contract.TRAIN_PAIR_COUNT
    )
    indices = list(range(contract.TRAIN_PAIR_COUNT)) * complete + list(range(remainder))
    assert len(contract.validate_schedule_indices(indices)) == 128_000
    pair_ids = [hashlib.sha256(f"pair:{index}".encode()).hexdigest() for index in range(4262)]
    first = contract.schedule_core(indices, pair_ids)
    second = contract.schedule_core(indices, pair_ids)
    assert first == second
    assert first["seed"] == 20260713
    escaped = list(indices)
    escaped[-1] = contract.TRAIN_PAIR_COUNT
    with pytest.raises(ValueError, match="train role"):
        contract.validate_schedule_indices(escaped)
    repeated = list(indices)
    repeated[-1] = repeated[-2]
    with pytest.raises(ValueError, match="repeats"):
        contract.validate_schedule_indices(repeated)


def test_promoted_only_selection_is_exact_and_all_scope_gated() -> None:
    candidates = [_candidate(update) for update in contract.CHECKPOINT_UPDATES]
    selection = contract.select_promoted_checkpoint(candidates)
    assert selection["selected_arm"] == "promoted_jepa"
    assert selection["selected_update"] == 1000
    assert selection["matched_no_jepa_influenced_selection"] is False
    damaged = [_candidate(update) for update in contract.CHECKPOINT_UPDATES]
    for candidate in damaged:
        candidate["scopes"]["visual_sensor_stress"]["physical"][
            "derived_raster_nll"
        ] = 0.2
    with pytest.raises(ValueError, match="no promoted checkpoint"):
        contract.select_promoted_checkpoint(damaged)


def test_fixed_calibration_grid_requires_precision_recall_and_obstacles() -> None:
    reports = {
        contract.canonical_json_sha256(list(values)): {
            "admitted_free_count": 100,
            "admitted_free_true_free_count": 100,
            "useful_free_count": 100,
            "useful_free_admitted_count": 100,
            "obstacle_within_2m_count": 100,
            "obstacle_within_2m_excluded_count": 100,
            "obstacle_within_2m_detected_count": 100,
        }
        for values in contract.threshold_grid()
    }
    selected = contract.select_calibration_threshold(reports)
    assert selected["admitted_free_precision"] == 1.0
    assert selected["useful_free_recall"] == 1.0
    broken = dict(reports)
    broken.pop(next(iter(broken)))
    with pytest.raises(ValueError, match="grid"):
        contract.select_calibration_threshold(broken)


def test_pre_g2_metadata_requires_predictor_ema_and_deployment_state() -> None:
    evaluation = [
        {"name": "encoder.weight"},
        {"name": "bev_decoder.weight"},
        {"name": "evidence_head.weight"},
        {"name": "target_encoder.weight"},
        {"name": "target_bev_decoder.weight"},
        {"name": "predictor.weight"},
    ]
    deployment = evaluation[:3]
    metadata = contract.pre_g2_candidate_metadata(
        model_config={"schema": "production"},
        evaluation_state_manifest=evaluation,
        evaluation_state_sha256="a" * 64,
        deployment_state_manifest=deployment,
        deployment_state_sha256="b" * 64,
        selection={"selected_arm": "promoted_jepa"},
        calibration={"arm": "promoted_jepa"},
        primitive_vocabulary=[f"p{index}" for index in range(9)],
        commanded_delta_table=[[0.0, 0.0, 0.0] for _ in range(9)],
        training_snapshot={"path": "snapshot.pt"},
    )
    assert metadata["required_evaluation_state_prefixes"] == [
        "target_encoder.",
        "target_bev_decoder.",
        "predictor.",
    ]
    for name, value in contract.PRE_G2_DENIALS.items():
        assert metadata[name] == value
    missing = [item for item in evaluation if not item["name"].startswith("predictor.")]
    with pytest.raises(ValueError, match="incomplete"):
        contract.pre_g2_candidate_metadata(
            model_config={"schema": "production"},
            evaluation_state_manifest=missing,
            evaluation_state_sha256="a" * 64,
            deployment_state_manifest=deployment,
            deployment_state_sha256="b" * 64,
            selection={"selected_arm": "promoted_jepa"},
            calibration={"arm": "promoted_jepa"},
            primitive_vocabulary=[f"p{index}" for index in range(9)],
            commanded_delta_table=[[0.0, 0.0, 0.0] for _ in range(9)],
            training_snapshot={"path": "snapshot.pt"},
        )


def test_five_camera_terms_current_next_and_four_b4_scalars_all_backpropagate() -> None:
    microbatch_scalars = []
    all_terms = []
    for _microbatch in range(4):
        current = [torch.tensor(1.0, requires_grad=True) for _ in contract.CAMERA_TERMS]
        next_ = [torch.tensor(1.0, requires_grad=True) for _ in contract.CAMERA_TERMS]
        all_terms.extend(current + next_)
        current_total = 0.25 * sum(current)
        next_total = 0.25 * sum(next_)
        microbatch_scalars.append(0.5 * current_total + 0.5 * next_total)
    update = corrected_loss.average_four_microbatch_tensor_scalars_v4(
        microbatch_scalars
    )
    update.backward()
    assert all(term.grad is not None and float(term.grad) == 0.03125 for term in all_terms)


def test_only_backward_membership_differs_between_arms() -> None:
    camera = object()
    joint_total = object()
    joint = SimpleNamespace(
        total=joint_total,
        observable_camera_ray_v4=SimpleNamespace(total=camera),
    )
    assert runner.Trainer.backward_for_arm(joint, "promoted_jepa") is joint_total
    assert runner.Trainer.backward_for_arm(joint, "matched_no_jepa") is camera
    with pytest.raises(ValueError):
        runner.Trainer.backward_for_arm(joint, "third_arm")


def test_parent_reserves_once_before_runtime_or_payload(tmp_path: Path) -> None:
    review = {"content_sha256": "a" * 64}
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        b"review",
        content_sha256=review["content_sha256"],
    )
    authorization = _authorization(review_binding)
    output = tmp_path / "attempt"
    reservation, raw = runner._reserve_output(
        output,
        review=review,
        review_raw=b"review",
        authorization=authorization,
        authorization_raw=b"authorization",
        sources={path: "c" * 64 for path in contract.SOURCE_PATHS},
        environment={"test": True},
    )
    assert reservation["status"] == "reserved_before_torch_camera_raw_or_rgb"
    assert reservation["torch_imported_before_reservation"] is False
    assert reservation["camera_or_raw_opened_before_reservation"] is False
    assert (output / "reservation.json").read_bytes() == raw
    with pytest.raises(RuntimeError, match="already"):
        runner._reserve_output(
            output,
            review=review,
            review_raw=b"review",
            authorization=authorization,
            authorization_raw=b"authorization",
            sources={path: "c" * 64 for path in contract.SOURCE_PATHS},
            environment={"test": True},
        )


def test_reservation_commit_failure_terminalizes_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    review = {"content_sha256": "a" * 64}
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        b"review",
        content_sha256=review["content_sha256"],
    )
    authorization = _authorization(review_binding)
    original = runner._publish_json

    def fail_reservation(path: Path, core: dict[str, object]):
        if path.name == "reservation.json":
            raise OSError("injected reservation failure")
        return original(path, core)

    monkeypatch.setattr(runner, "_publish_json", fail_reservation)
    output = tmp_path / "failed_attempt"
    with pytest.raises(OSError, match="injected"):
        runner._reserve_output(
            output,
            review=review,
            review_raw=b"review",
            authorization=authorization,
            authorization_raw=b"authorization",
            sources={path: "c" * 64 for path in contract.SOURCE_PATHS},
            environment={"test": True},
        )
    failure = contract.parse_canonical_json(
        (output / "reservation_failed.json").read_bytes(), name="reservation failure"
    )
    assert failure["status"] == "failed_reservation_commit"
    assert failure["torch_imported"] is False
    assert failure["camera_raw_or_rgb_opened"] is False
    assert failure["retry_authorized"] is False


def test_runner_is_stdlib_until_reservation_and_child_is_isolated() -> None:
    path = ROOT / contract.RUNNER_RELATIVE_PATH
    source = path.read_text()
    tree = ast.parse(source)
    forbidden = {"torch", "numpy", "PIL"}
    top_imports = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top_imports.extend(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_imports.append(node.module.split(".")[0])
    assert forbidden.isdisjoint(top_imports)
    parent_source = inspect.getsource(runner.run_parent)
    assert parent_source.index("_reserve_output(") < parent_source.index("_load_runtime()")
    assert parent_source.index("_reserve_output(") < parent_source.index(
        "_camera_model_after_reservation"
    )
    command = runner._child_command()
    assert command[1:] == (
        "-I",
        "-B",
        str(ROOT / contract.RUNNER_RELATIVE_PATH),
        "--internal-verify",
    )
    assert "-I" in command and "-B" in command
    internal_source = inspect.getsource(runner.run_internal_verifier)
    assert '"checkpoint_open_count": 1' in internal_source
    assert "model.load_state_dict(evaluation_state, strict=True)" in internal_source
    assert "qualified_checkpoint.pt" not in source


def test_isolated_child_import_does_not_load_numpy_torch_or_pillow() -> None:
    runner_path = str(ROOT / contract.RUNNER_RELATIVE_PATH)
    probe = (
        "import importlib.util,pathlib,sys;"
        f"p=pathlib.Path({runner_path!r});"
        "s=importlib.util.spec_from_file_location('_matched_runner_probe',p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "forbidden={'numpy','torch','PIL'};"
        "loaded={name.split('.')[0] for name in sys.modules};"
        "assert forbidden.isdisjoint(loaded),(forbidden & loaded);"
        "print('stdlib-only')"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", probe],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "stdlib-only\n"
    assert completed.stderr == ""


def test_every_torch_deserialization_is_weights_only_without_fallback() -> None:
    source = (ROOT / contract.RUNNER_RELATIVE_PATH).read_text()
    tree = ast.parse(source)

    def dotted(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return f"{dotted(node.value)}.{node.attr}"
        return ""

    loads = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "load"
        and "torch" in dotted(node.func)
    ]
    assert len(loads) == 3
    for call in loads:
        keywords = {item.arg: item.value for item in call.keywords}
        assert isinstance(keywords.get("weights_only"), ast.Constant)
        assert keywords["weights_only"].value is True
    assert "weights_only=False" not in source
    assert "except TypeError" not in source


def test_cli_keeps_parent_credentials_out_of_internal_verifier() -> None:
    internal = runner.parse_args(["--internal-verify"])
    assert internal.internal_verify is True
    with pytest.raises(SystemExit):
        runner.parse_args(["--internal-verify", "--review-sha256", "a" * 64])
    with pytest.raises(SystemExit):
        runner.parse_args(["--run", "--review-sha256", "a" * 64])
