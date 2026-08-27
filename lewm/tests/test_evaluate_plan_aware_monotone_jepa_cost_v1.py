from __future__ import annotations

import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from lewm.safety import plan_aware_monotone_jepa_cost_metrics_v1 as metrics
from scripts import evaluate_plan_aware_monotone_jepa_cost_v1 as evaluator


def _training_row(
    candidate: int,
    *,
    progress: float,
    heading: float,
    completed: bool = False,
) -> dict[str, object]:
    return {
        "state_id": "synthetic-000",
        "candidate_index": candidate,
        "p_d": progress,
        "p_theta": heading,
        "completed": completed,
        "oracle_viability_admissible": True,
        "base_features": np.zeros(138, dtype=np.float32),
        "query_features": np.zeros(71, dtype=np.float32),
        "kinematic_anchor": -float(candidate),
    }


def test_state_training_pair_targets_are_conditioned_margin_borda_differences() -> None:
    distance_margin = evaluator.OLD_METRICS.DISTANCE_PREFERENCE_MARGIN_M
    heading_margin = evaluator.OLD_METRICS.HEADING_PREFERENCE_MARGIN_RAD
    rows = [
        _training_row(0, progress=0.0, heading=0.0),
        # Both differences are exactly on their inclusive deadbands.
        _training_row(
            1,
            progress=distance_margin,
            heading=heading_margin,
        ),
        # This is beyond the distance deadband relative to candidate zero.
        _training_row(
            2,
            progress=distance_margin + 1e-4,
            heading=0.0,
        ),
        # Completion is the first route-ordering key.
        _training_row(3, progress=-1.0, heading=-1.0, completed=True),
    ]

    payload = evaluator.state_training_payload(rows)
    targets = payload["pairwise_targets"]
    utility = payload["utility"]

    assert targets.dtype == np.float32
    assert np.array_equal(targets, -targets.T)
    assert evaluator.OLD_METRICS.route_preference(rows[0], rows[1]) == 0
    for left in range(len(rows)):
        for right in range(len(rows)):
            expected = evaluator.CONTRACT.margin_borda_pairwise_target(
                float(utility[left]), float(utility[right])
            )
            assert targets[left, right] == expected
    assert set(np.unique(targets)) <= {-1.0, 0.0, 1.0}


def test_zero_and_singleton_admissible_fit_states_are_retained_but_not_optimized() -> None:
    zero = [
        {
            **_training_row(candidate, progress=float(candidate), heading=0.0),
            "state_id": "synthetic-000",
            "family": metrics.FAMILY_IDS[0],
            "route_intent_role": "translational",
            "waypoint_features": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "oracle_viability_admissible": False,
        }
        for candidate in range(evaluator.CANDIDATE_COUNT)
    ]
    singleton = [
        {
            **_training_row(candidate, progress=float(candidate), heading=0.0),
            "state_id": "synthetic-001",
            "family": metrics.FAMILY_IDS[0],
            "route_intent_role": "translational",
            "waypoint_features": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
            "oracle_viability_admissible": candidate == 3,
        }
        for candidate in range(evaluator.CANDIDATE_COUNT)
    ]
    zero_payload = evaluator.state_training_payload(zero)
    singleton_payload = evaluator.state_training_payload(singleton)
    assert zero_payload["optimization_status"] == "SKIPPED_ZERO_ADMISSIBLE"
    assert singleton_payload["optimization_status"] == "SKIPPED_SINGLETON_ADMISSIBLE"
    assert zero_payload["contributes_optimizer_step"] is False
    assert singleton_payload["contributes_optimizer_step"] is False
    assert "base_features" not in zero_payload
    assert "base_features" not in singleton_payload

    datasets = {
        evaluator.FIT: {
            "synthetic-000": zero,
            "synthetic-001": singleton,
        },
        evaluator.CALIBRATION: {},
        evaluator.HELDOUT: {},
    }
    summary = evaluator._fit_optimization_summary(datasets[evaluator.FIT])
    assert summary["fit_states_total"] == 2
    assert summary["fit_states_contributing"] == 0
    assert summary["fit_state_ids_skipped_zero_admissible"] == ["synthetic-000"]
    assert summary["fit_state_ids_skipped_singleton_admissible"] == [
        "synthetic-001"
    ]
    target_rows = evaluator._training_target_rows(datasets)
    assert len(target_rows) == 2 * evaluator.CANDIDATE_COUNT
    assert not any(row["used_for_fit"] for row in target_rows)


class _LifecycleStop(RuntimeError):
    pass


def _synthetic_source_closure(
    rows: list[dict[str, object]],
) -> dict[str, object]:
    return evaluator.CONTRACT.attach_self_digest(
        {
            "schema": evaluator.CONTRACT.SOURCE_CLOSURE_SCHEMA_VERSION,
            "experiment_id": evaluator.CONTRACT.EXPERIMENT_ID,
            "source_commit": evaluator.SOURCE_COMMIT,
            "declared_paths": [str(row["path"]) for row in rows],
            "rows": copy.deepcopy(rows),
            "row_count": len(rows),
            "missing_paths": [],
            "complete": True,
            "outcome_or_result_payloads_parsed": [],
            "custody_only_predecessor_results_hashed_without_parsing": [],
            "generated_cache_paths_traversed": [],
            "accidental_exposures": [],
        }
    )


def test_runtime_freeze_uses_exact_single_parent_contract_custody(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = "f" * 40
    monkeypatch.setattr(
        evaluator,
        "_runtime_execution_correction_custody",
        lambda: {"source_freeze_commit": freeze},
    )
    assert evaluator._runtime_source_freeze() == freeze

    def reject() -> dict[str, object]:
        raise evaluator.QualificationError("arbitrary descendant rejected")

    monkeypatch.setattr(
        evaluator, "_runtime_execution_correction_custody", reject
    )
    with pytest.raises(evaluator.QualificationError, match="arbitrary descendant"):
        evaluator._runtime_source_freeze()


def test_evaluator_conditional_helper_boundary_is_exact_and_scrubbed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    helper = evaluator._conditional_helper()
    monkeypatch.setenv("PYTHONPATH", "/usr/lib/python3/dist-packages")
    monkeypatch.setenv("PYTHONHOME", "/synthetic/wrong-prefix")
    observed: dict[str, object] = {}

    def run(command: list[str], **kwargs: object) -> object:
        observed["command"] = command
        observed["environment"] = kwargs["env"]
        return evaluator.subprocess.CompletedProcess(command, 0, stdout="pass\n")

    monkeypatch.setattr(evaluator.subprocess, "run", run)
    receipt = evaluator._run_conditional_helper_cli(
        attempt=tmp_path,
        arguments=("run-stage-b",),
        log_name="synthetic.log",
    )
    assert observed["command"][:4] == [
        str(helper.GPU_INTERPRETER),
        "-E",
        "-s",
        str(helper.SELF),
    ]
    environment = observed["environment"]
    assert isinstance(environment, dict)
    for key in helper.SCRUBBED_PYTHON_ENVIRONMENT_KEYS:
        assert key not in environment
    assert receipt["child_environment_contract"] == (
        helper.child_environment_contract(helper.GPU_INTERPRETER)
    )


def test_correction_refreeze_preflight_preserves_scientific_authorities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior_freeze = "b" * 40
    head = "c" * 40
    mutable_paths = [
        str(path)
        for path in evaluator.CONTRACT.SOURCE_CLOSURE_DEFAULT_PATHS
        if str(path).endswith(".py")
    ]
    mutable_path = "scripts/evaluate_plan_aware_monotone_jepa_cost_v1.py"
    authority_payloads = {
        relative: f"authority:{relative}\n".encode()
        for relative in evaluator.CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
    }
    rows: list[dict[str, object]] = []
    for relative, payload in authority_payloads.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        rows.append(
            {
                "path": relative,
                "sha256": evaluator.hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
        )
    for relative in mutable_paths:
        old_implementation = f"old:{relative}\n".encode()
        current_implementation = (
            b"corrected implementation\n"
            if relative == mutable_path
            else old_implementation
        )
        mutable = tmp_path / relative
        mutable.parent.mkdir(parents=True, exist_ok=True)
        mutable.write_bytes(current_implementation)
        rows.append(
            {
                "path": relative,
                "sha256": evaluator.hashlib.sha256(old_implementation).hexdigest(),
                "bytes": len(old_implementation),
            }
        )
    archive = tmp_path / ".qualification.failed-smoke"
    closure_path = archive / "receipts/source_closure.json"
    evaluator.atomic_json(closure_path, _synthetic_source_closure(rows))
    custody = [
        {
            "archive_path": str(archive),
            "failure_receipt": {
                "path": str(archive / "receipts/failure.json"),
                "sha256": "a" * 64,
                "bytes": 1,
                "content_digest": "d" * 64,
            },
            "source_freeze_commit": prior_freeze,
            "source_closure": {
                **evaluator.binding(closure_path),
                "content_digest": evaluator.load_json(closure_path)[
                    "content_digest"
                ],
            },
            "inventory": {
                "files": 2,
                "bytes": 2,
                "manifest_sha256": "e" * 64,
            },
            "files_reused": 0,
        }
    ]
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)

    def git_output(*args: str) -> str:
        if args[:2] == ("rev-list", "--parents"):
            return f"{head} {prior_freeze}\n{prior_freeze} {evaluator.SOURCE_COMMIT}"
        if args == (
            "diff",
            "--name-only",
            evaluator.SOURCE_COMMIT,
            head,
        ):
            return mutable_path
        return ""

    monkeypatch.setattr(evaluator, "git_output", git_output)
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        lambda *_args, **_kwargs: evaluator.subprocess.CompletedProcess([], 0),
    )
    receipt = evaluator._correction_refreeze_preflight(
        head=head,
        prior_smoke_failure_custody=custody,
        authority_payloads=authority_payloads,
    )
    assert receipt["scientific_authorities_unchanged"] is True
    assert receipt["files_reused"] == 0
    assert receipt["required_enclosing_commit_subject"] == (
        "Freeze plan-aware monotone JEPA route cost"
    )

    changed_authority = copy.deepcopy(authority_payloads)
    changed_authority[
        evaluator.CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS[1]
    ] += b"scientific drift"
    with pytest.raises(
        evaluator.QualificationError,
        match="scientific authority changed after fit outcomes opened",
    ):
        evaluator._correction_refreeze_preflight(
            head=head,
            prior_smoke_failure_custody=custody,
            authority_payloads=changed_authority,
        )


def test_freeze_contract_has_smoke_authorised_corrective_refresh_branch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    head = "c" * 40
    custody = [{"synthetic": "validated upstream"}]
    authority_payloads = {
        relative: f"refreshed:{relative}\n".encode()
        for relative in evaluator.CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
    }
    closure = _synthetic_source_closure([])
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "OUTPUT_ROOT",
        tmp_path / "qualification",
    )
    monkeypatch.setattr(
        evaluator,
        "git_output",
        lambda *args: head if args == ("rev-parse", "HEAD") else "",
    )
    monkeypatch.setattr(
        evaluator,
        "_validated_prior_smoke_failure_custody",
        lambda *_args, **_kwargs: custody,
    )
    monkeypatch.setattr(
        evaluator, "_scientific_authority_payloads", lambda: authority_payloads
    )
    monkeypatch.setattr(
        evaluator,
        "_correction_refreeze_preflight",
        lambda **_kwargs: {
            "mode": "VALIDATED_TRAINING_SMOKE_CORRECTION_REFREEZE"
        },
    )
    for relative in authority_payloads:
        stale = tmp_path / relative
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_bytes(b"stale authority\n")
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "build_source_closure",
        lambda *_args, **_kwargs: closure,
    )
    monkeypatch.setattr(evaluator, "_validate_frozen_authorities", lambda: {})
    result = evaluator.freeze_contract()
    assert result["freeze_mode"] == (
        "VALIDATED_TRAINING_SMOKE_CORRECTION_REFREEZE"
    )
    assert result["base_head"] == head
    assert result["files_reused"] == 0
    assert result["required_enclosing_commit_subject"] == (
        "Freeze plan-aware monotone JEPA route cost"
    )
    for relative, payload in authority_payloads.items():
        assert (tmp_path / relative).read_bytes() == payload


def test_runtime_corrected_freeze_rejects_empty_or_closure_only_descendant(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    prior_freeze = "b" * 40
    head = "c" * 40
    python_paths = [
        str(path)
        for path in evaluator.CONTRACT.SOURCE_CLOSURE_DEFAULT_PATHS
        if str(path).endswith(".py")
    ]
    immutable_paths = list(
        evaluator.CONTRACT.CORRECTION_REFREEZE_IMMUTABLE_AUTHORITY_PATHS
    )
    for relative in (*python_paths, *immutable_paths):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"frozen:{relative}\n".encode())
    snapshot = evaluator.CONTRACT.build_source_closure(
        tmp_path,
        paths=(*python_paths, *immutable_paths),
        require_complete=True,
    )
    archive = tmp_path / ".qualification.failed-smoke"
    snapshot_path = archive / "receipts/source_closure.json"
    evaluator.atomic_json(snapshot_path, snapshot)
    custody = [
        {
            "archive_path": str(archive),
            "failure_receipt": {
                "path": str(archive / "receipts/failure.json"),
                "sha256": "a" * 64,
                "bytes": 1,
                "content_digest": "d" * 64,
            },
            "source_freeze_commit": prior_freeze,
            "source_closure": {
                **evaluator.binding(snapshot_path),
                "content_digest": snapshot["content_digest"],
            },
            "inventory": {
                "files": 2,
                "bytes": 2,
                "manifest_sha256": "e" * 64,
            },
            "files_reused": 0,
        }
    ]
    corrected_path = tmp_path / python_paths[0]

    def fake_git(args: list[str], **_kwargs: object) -> object:
        command = tuple(args[1:])
        if command == ("rev-parse", "HEAD"):
            stdout = head
        elif command == ("rev-list", "--parents", "-n", "1", head):
            stdout = f"{head} {prior_freeze}"
        elif command == ("status", "--porcelain=v1"):
            stdout = ""
        elif command == ("show", "-s", "--format=%s", head):
            stdout = evaluator.CONTRACT.CONTRACT_FREEZE_COMMIT_SUBJECT
        elif command == (
            "rev-list",
            "--parents",
            f"{evaluator.SOURCE_COMMIT}..{head}",
        ):
            stdout = (
                f"{head} {prior_freeze}\n"
                f"{prior_freeze} {evaluator.SOURCE_COMMIT}"
            )
        elif command == (
            "diff",
            "--name-only",
            evaluator.SOURCE_COMMIT,
            head,
        ):
            changed = [str(evaluator.CONTRACT.TRACKED_SOURCE_CLOSURE_PATH)]
            if corrected_path.read_bytes().startswith(b"corrected:"):
                changed.insert(0, python_paths[0])
            stdout = "\n".join(changed)
        else:
            stdout = ""
        return evaluator.subprocess.CompletedProcess(args, 0, stdout=stdout)

    monkeypatch.setattr(evaluator.CONTRACT.subprocess, "run", fake_git)
    with pytest.raises(
        evaluator.CONTRACT.ContractError,
        match="no implementation/test Python byte change",
    ):
        evaluator.CONTRACT.validate_execution_freeze_custody(
            tmp_path,
            prior_smoke_failure_custody=custody,
        )

    corrected_path.write_bytes(b"corrected: implementation fault\n")
    receipt = evaluator.CONTRACT.validate_execution_freeze_custody(
        tmp_path,
        prior_smoke_failure_custody=custody,
    )
    assert receipt["custody_mode"] == (
        "VALIDATED_TRAINING_SMOKE_CORRECTION_FREEZE"
    )
    assert receipt["mutable_correction_observed_by_archive"] == {
        str(archive): True
    }


@pytest.mark.parametrize(
    ("stop_at", "expected"),
    [
        ("contract", ["preflight", "fit", "smoke", "train", "contract"]),
        (
            "heldout",
            [
                "preflight",
                "fit",
                "smoke",
                "train",
                "contract",
                "calibration",
                "heldout",
            ],
        ),
    ],
)
def test_execute_opens_splits_only_after_durable_lifecycle_barriers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    stop_at: str,
    expected: list[str],
) -> None:
    """Exercise orchestration without opening any panel or scientific value."""

    events: list[str] = []
    attempt = tmp_path / "attempt"

    monkeypatch.setattr(
        evaluator,
        "_runtime_execution_correction_custody",
        lambda: {
            "source_freeze_commit": "a" * 40,
            "conditional_child_environment_preflight": None,
            "execution_correction_replay": None,
        },
    )
    monkeypatch.setattr(
        evaluator,
        "_validate_frozen_authorities",
        lambda **_kwargs: {
            "route_role_authority": {
                "records": [
                    {"state_id": f"synthetic-{role}", "route_role": "translational"}
                    for role in evaluator.SPLIT_ROLES
                ]
            }
        },
    )

    def new_attempt(
        _output_root: Path, _source_freeze: str, **_kwargs: object
    ) -> Path:
        attempt.mkdir()
        return attempt

    monkeypatch.setattr(evaluator, "_new_attempt", new_attempt)
    def preflight(_attempt: Path) -> dict[str, object]:
        events.append("preflight")
        value = evaluator.attach_digest({"schema": "synthetic-preflight"})
        evaluator.atomic_json(
            attempt / "receipts/conditional_child_environment_preflight.json",
            value,
        )
        return value

    monkeypatch.setattr(
        evaluator, "_conditional_child_environment_preflight", preflight
    )
    monkeypatch.setattr(evaluator, "_preexecution_receipt", lambda **_kwargs: {})
    monkeypatch.setattr(
        evaluator,
        "split_ids",
        lambda: {
            evaluator.FIT: ["synthetic-fit"],
            evaluator.CALIBRATION: ["synthetic-calibration"],
            evaluator.HELDOUT: ["synthetic-heldout"],
        },
    )
    monkeypatch.setattr(evaluator, "route_line_index", lambda: {})
    monkeypatch.setattr(evaluator, "_index_by_state", lambda _path: {})
    monkeypatch.setattr(evaluator, "state_manifest_map", lambda: {})
    monkeypatch.setattr(evaluator, "tensor_index", lambda: {})
    monkeypatch.setattr(evaluator, "_active_experiment_processes", lambda: [])

    def authorised(role: str, **_kwargs: object) -> dict[str, list[object]]:
        events.append(role)
        if role != evaluator.FIT:
            assert (attempt / "receipts/evaluation_contract.json").is_file()
        if role == evaluator.HELDOUT and stop_at == "heldout":
            raise _LifecycleStop("synthetic heldout stop")
        return {f"synthetic-{role}": []}

    def smoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        events.append("smoke")
        return {}

    def train(*_args: object, **_kwargs: object) -> tuple[dict, dict, dict]:
        events.append("train")
        checkpoints = attempt / "checkpoints"
        checkpoints.mkdir(parents=True)
        (checkpoints / "no_latent_final_epoch_060.pt").write_bytes(b"final-only")
        (checkpoints / "latent_true_future_final_epoch_060.pt").write_bytes(
            b"final-only"
        )
        return {}, {}, {}

    def evaluation_contract(**_kwargs: object) -> dict[str, object]:
        events.append("contract")
        if stop_at == "contract":
            raise _LifecycleStop("synthetic contract-publication stop")
        assert (attempt / "checkpoints/no_latent_final_epoch_060.pt").is_file()
        assert (attempt / "checkpoints/latent_true_future_final_epoch_060.pt").is_file()
        evaluator.atomic_json(
            attempt / "receipts/evaluation_contract.json",
            {"schema": "synthetic-durable-evaluation-contract"},
        )
        return {"future_latent_derangement_by_state": {}}

    monkeypatch.setattr(evaluator, "authorised_dataset", authorised)
    monkeypatch.setattr(evaluator, "run_training_smoke", smoke)
    monkeypatch.setattr(evaluator, "train_rankers", train)
    monkeypatch.setattr(evaluator, "write_evaluation_contract", evaluation_contract)

    with pytest.raises(_LifecycleStop):
        evaluator.execute()

    assert events == expected
    if stop_at == "contract":
        assert evaluator.CALIBRATION not in events
        assert evaluator.HELDOUT not in events
    else:
        assert events.index("contract") < events.index(evaluator.CALIBRATION)
        assert events.index("contract") < events.index(evaluator.HELDOUT)
    assert not attempt.exists()
    failures = list(tmp_path.glob(".*.failed-*"))
    assert len(failures) == 1
    failure = evaluator.load_json(failures[0] / "receipts/failure.json")
    assert failure["partial_artifacts_reusable"] is False
    assert failure["nothing_running"] is True
    assert failure["source_freeze_commit"] == "a" * 40
    assert failure["error_type"] == "_LifecycleStop"
    assert "synthetic" in failure["error_message"]
    assert failure["phase"] == (
        "EVALUATION_CONTRACT_PUBLICATION"
        if stop_at == "contract"
        else "HELDOUT_OPEN"
    )
    assert failure["full_training_epochs_completed"] == (
        evaluator.CONTRACT.TRAINING["epochs"]
    )
    assert failure["final_checkpoint_published"] is True
    assert failure["calibration_rows_opened"] == (
        0
        if stop_at == "contract"
        else evaluator.CONTRACT.ROLE_ROW_COUNTS[evaluator.CALIBRATION]
    )
    assert failure["heldout_rows_opened"] == 0
    assert failure["prohibition_counters"] == evaluator._prohibition_counters()


class _DummyRanker:
    def __init__(self) -> None:
        self._state = {"weight": torch.tensor([1.0], dtype=torch.float32)}

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {name: value.clone() for name, value in self._state.items()}

    def load_state_dict(self, state: dict[str, torch.Tensor], strict: bool = True) -> None:
        assert strict is True
        self._state = {name: value.clone() for name, value in state.items()}

    def to(self, *args: object, **kwargs: object) -> "_DummyRanker":
        return self

    def eval(self) -> "_DummyRanker":
        return self

    def requires_grad_(self, _enabled: bool) -> "_DummyRanker":
        return self


def _synthetic_checkpoint(
    path: Path,
    *,
    condition: str,
    epoch: int,
) -> dict[str, object]:
    model = _DummyRanker()
    parameter_digest = evaluator._parameter_digest(model)
    parameter_count = sum(value.numel() for value in model.state_dict().values())
    fit_optimization = {
        "fit_states_total": 1,
        "fit_state_ids_total": ["synthetic-000"],
        "fit_states_contributing": 1,
        "fit_state_ids_contributing": ["synthetic-000"],
        "fit_states_skipped_zero_admissible": 0,
        "fit_state_ids_skipped_zero_admissible": [],
        "fit_states_skipped_singleton_admissible": 0,
        "fit_state_ids_skipped_singleton_admissible": [],
        "epoch_average_denominator": 1,
    }
    fit_optimization_digest = evaluator.hashlib.sha256(
        evaluator.canonical_bytes(fit_optimization)[:-1]
    ).hexdigest()
    training_history = [
        {
            "epoch": epoch_index,
            "optimizer_steps": 1,
            "fit_states_total": 1,
            "fit_states_contributing": 1,
            "fit_states_skipped_zero_admissible": 0,
            "fit_states_skipped_singleton_admissible": 0,
            "epoch_average_denominator": 1,
            "loss": 1.0,
            "pair": 1.0,
            "list": 1.0,
            "residual": 1.0,
        }
        for epoch_index in range(1, evaluator.CONTRACT.TRAINING["epochs"] + 1)
    ]
    evaluator._atomic_torch_save(
        path,
        {
            "schema": "plan_aware_ranker_checkpoint_receipt_v1",
            "condition": condition,
            "seed_family": evaluator.CONTRACT.ROUTE_COST_SEED,
            "seed_metadata": copy.deepcopy(
                evaluator.CONTRACT.CHECKPOINT_SEED_METADATA[condition]
            ),
            "epoch": epoch,
            "final_epoch_only": True,
            "optimizer_state_persisted": False,
            "state_dict": model.state_dict(),
            "parameter_digest": parameter_digest,
            "parameter_count": parameter_count,
            "model_contract": evaluator.MODEL.model_contract(condition),
            "fit_optimization": fit_optimization,
            "training_history": training_history,
        },
    )
    return {
        **evaluator.binding(path, relative_to=path.parents[1]),
        "seed_metadata": copy.deepcopy(
            evaluator.CONTRACT.CHECKPOINT_SEED_METADATA[condition]
        ),
        "parameter_digest": parameter_digest,
        "parameter_count": parameter_count,
        "epoch": epoch,
        "fit_optimization_digest": fit_optimization_digest,
    }


def test_checkpoint_loader_accepts_only_bound_final_epoch_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    conditions = (
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL",
    )
    monkeypatch.setattr(
        evaluator.MODEL,
        "build_matched_rankers",
        lambda _seed: (_DummyRanker(), _DummyRanker()),
    )
    bindings = {}
    for condition in conditions:
        path = tmp_path / "checkpoints" / f"{condition}.pt"
        bindings[condition] = _synthetic_checkpoint(
            path,
            condition=condition,
            epoch=evaluator.CONTRACT.TRAINING["epochs"],
        )

    loaded = evaluator._load_checkpoint_models(bindings, output_root=tmp_path)
    assert set(loaded) == set(conditions)
    for condition in conditions:
        checkpoint = torch.load(
            tmp_path / str(bindings[condition]["path"]),
            map_location="cpu",
            weights_only=False,
        )
        assert checkpoint["condition"] == condition
        assert checkpoint["seed_metadata"] == evaluator.CONTRACT.CHECKPOINT_SEED_METADATA[
            condition
        ]
        assert checkpoint["model_contract"] == evaluator.MODEL.model_contract(condition)

    stale_path = tmp_path / str(bindings[conditions[0]]["path"])
    bindings[conditions[0]] = _synthetic_checkpoint(
        stale_path,
        condition=conditions[0],
        epoch=evaluator.CONTRACT.TRAINING["epochs"] - 1,
    )
    with pytest.raises(evaluator.QualificationError, match="metadata drift"):
        evaluator._load_checkpoint_models(bindings, output_root=tmp_path)

    bindings[conditions[0]] = _synthetic_checkpoint(
        stale_path,
        condition=conditions[0],
        epoch=evaluator.CONTRACT.TRAINING["epochs"],
    )
    stale_path.write_bytes(stale_path.read_bytes() + b"tamper")
    with pytest.raises(evaluator.QualificationError, match="checkpoint drift"):
        evaluator._load_checkpoint_models(bindings, output_root=tmp_path)

    bindings[conditions[0]] = _synthetic_checkpoint(
        stale_path,
        condition=conditions[0],
        epoch=evaluator.CONTRACT.TRAINING["epochs"],
    )
    checkpoint = torch.load(stale_path, map_location="cpu", weights_only=False)
    checkpoint["seed_metadata"] = copy.deepcopy(checkpoint["seed_metadata"])
    checkpoint["seed_metadata"]["condition_torch_seed"] += 1
    evaluator._atomic_torch_save(stale_path, checkpoint)
    file_record = evaluator.binding(stale_path, relative_to=tmp_path)
    bindings[conditions[0]] = {
        **bindings[conditions[0]],
        **file_record,
    }
    with pytest.raises(evaluator.QualificationError, match="metadata drift"):
        evaluator._load_checkpoint_models(bindings, output_root=tmp_path)


def _aggregate_summary(value: float = 0.5) -> dict[str, object]:
    aggregate = {
        "pairwise_accuracy": value,
        "spearman_rho": value,
        "normalized_regret": value,
        "best_route_top3_rate": value,
        "selected_progress_ratio": value,
        "selected_route_progress_m_mean": value,
        "selected_heading_progress_rad_mean": value,
        "selected_combined_route_utility_mean": value,
        "selected_combined_route_utility_count": 1,
        "selected_immediate_contacts_h1": 0,
        "selected_nonviable_successors": 0,
        "selected_stuck": 0,
    }
    return {
        "populations": {
            population: {"aggregate": copy.deepcopy(aggregate)}
            for population in metrics.POPULATION_IDS
        },
        "descriptive_adverse_downranking": {
            "outcomes": {
                outcome: {
                    "overall": {
                        "pair_count": 0,
                        "correct_credit": 0.0,
                        "pairwise_accuracy": None,
                    }
                }
                for outcome in metrics.ADVERSE_DOWNRANKING_OUTCOMES
            }
        },
    }


def _storage_result_core() -> dict[str, object]:
    heldout = {
        source: _aggregate_summary()
        for source in (
            "KINEMATIC_ROUTE_BASELINE",
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL",
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE",
            "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE",
            *evaluator.CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS,
        )
    }
    return {
        "primary_classification": "PLAN_AWARE_JEPA_COST_NO_SIGNAL",
        "secondary_classifications": ["ENCODER_ROUTE_INFORMATION_INSUFFICIENT"],
        "next_experiment": "NON_GREEDY_LOCAL_SUBGOAL_JEPA_PLANNING_V1",
        "next_experiment_specification": (
            evaluator.CONTRACT.next_experiment_specification_for_primary(
                "PLAN_AWARE_JEPA_COST_NO_SIGNAL"
            )
        ),
        "stage_execution": {
            "stage_a": "PASS_COMPLETE",
            "stage_b": "NOT_RUN_TRUE_GATE_FAILED",
            "stage_c": "NOT_RUN_STAGE_B_NOT_AUTHORISED",
            "execution_correction_custody": {
                "amendment": {"sha256": "a" * 64},
                "files_reused": 0,
                "execution_correction_replay": {"pass": True},
            },
        },
        "metrics": {
            "stage_a_decisions": {
                "derangement": {
                    "pass": False,
                    "damage": {
                        "pairwise_accuracy_loss": 0.0,
                        "selected_progress_m_loss": 0.0,
                        "selected_progress_ratio_loss": 0.0,
                        "normalized_regret_worsening": 0.0,
                        "best_route_top3_loss": 0.0,
                    },
                },
                "true_future_gate": {"pass": False},
                "true_incremental_value": {"pass": False},
            },
            "stage_a": {evaluator.HELDOUT: heldout},
            "fit_optimization": {
                "fit_states_total": 32,
                "fit_state_ids_total": [f"synthetic-{index:03d}" for index in range(32)],
                "fit_states_contributing": 30,
                "fit_state_ids_contributing": [
                    f"synthetic-{index:03d}" for index in range(30)
                ],
                "fit_states_skipped_zero_admissible": 1,
                "fit_state_ids_skipped_zero_admissible": ["synthetic-030"],
                "fit_states_skipped_singleton_admissible": 1,
                "fit_state_ids_skipped_singleton_admissible": ["synthetic-031"],
                "epoch_average_denominator": 30,
            },
            "stage_a_raw_cost_rereduced": {
                "schema": "plan_aware_raw_goal_cosine_successor_metric_reduction_v1",
                "frozen_source_binding": copy.deepcopy(
                    evaluator.CONTRACT.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING
                ),
                "source_rows": 1_728,
            },
            "stage_a_raw_cost_matched_comparisons": {
                "schema": "plan_aware_matched_raw_cost_comparisons_v1",
                "comparison_mapping": {
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": (
                        "RAW_TRUE_FUTURE_GOAL_COSINE"
                    ),
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": (
                        "RAW_R1_GOAL_COSINE"
                    ),
                    "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": (
                        "RAW_RR_GOAL_COSINE"
                    ),
                },
                "metric_ids": [],
                "classification_gate": False,
                "comparisons": {
                    candidate: {
                        "comparator_id": comparator,
                        "status": "NOT_RUN_STAGE_B_NOT_AUTHORISED",
                        "classification_gate": False,
                    }
                    for candidate, comparator in {
                        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": (
                            "RAW_TRUE_FUTURE_GOAL_COSINE"
                        ),
                        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": (
                            "RAW_R1_GOAL_COSINE"
                        ),
                        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": (
                            "RAW_RR_GOAL_COSINE"
                        ),
                    }.items()
                },
            },
            "historical_raw_latent_goal_cosine_comparators": {
                "policy": "HISTORICAL_BYTE_BOUND_COMPARATOR_COPY_NO_COSINE_RECOMPUTATION",
                "source_binding": {
                    "path": "synthetic-predecessor.json",
                    "sha256": "f" * 64,
                    "bytes": 1,
                },
                "comparators": {
                    source: _aggregate_summary()
                    for source in (
                        "RAW_TRUE_FUTURE_GOAL_COSINE",
                        "RAW_R1_GOAL_COSINE",
                        "RAW_RR_GOAL_COSINE",
                    )
                },
                "metric_definition_comparable_to_successor_borda_metrics": False,
                "use": "historical_context_only",
            },
        },
        "predecessor_narrative_authority": list(
            evaluator.CONTRACT.PRESERVED_PREDECESSOR_NARRATIVE
        ),
        "predecessor_fact_authority": list(
            evaluator.CONTRACT.PRESERVED_PREDECESSOR_FACTS
        ),
        "runtime_and_storage": {"output_files": 0, "output_bytes": 0},
    }


def test_result_storage_solver_reaches_an_exact_byte_fixed_point(tmp_path: Path) -> None:
    payload = b"synthetic-existing-artifact\n"
    existing = tmp_path / "ledgers" / "fixture.bin"
    existing.parent.mkdir(parents=True)
    existing.write_bytes(payload)

    result, report = evaluator._build_result_with_exact_storage(
        _storage_result_core(), attempt=tmp_path
    )
    result_bytes = evaluator.canonical_bytes(result)

    assert result["content_digest"] == evaluator.content_digest(result)
    assert result["runtime_and_storage"]["output_files"] == 3
    assert result["runtime_and_storage"]["output_bytes"] == (
        len(payload) + len(result_bytes) + len(report)
    )
    assert report == evaluator._report_markdown(result).encode("utf-8")


def _synthetic_stage_a_rows() -> tuple[
    list[dict[str, object]],
    dict[str, dict[str, list[dict[str, object]]]],
    dict[str, dict[str, dict[str, np.ndarray]]],
]:
    ledger: list[dict[str, object]] = []
    candidates: dict[str, dict[str, list[dict[str, object]]]] = {
        role: {} for role in evaluator.SPLIT_ROLES
    }
    scores: dict[str, dict[str, dict[str, np.ndarray]]] = {
        role: {"SYNTHETIC_SCORE": {}} for role in evaluator.SPLIT_ROLES
    }
    cursor = 0
    for role in evaluator.SPLIT_ROLES:
        states = evaluator.CONTRACT.ROLE_STATE_COUNTS[role]
        for state_offset in range(states):
            state_id = f"synthetic-{cursor:03d}"
            family = metrics.FAMILY_IDS[state_offset % len(metrics.FAMILY_IDS)]
            state_candidates: list[dict[str, object]] = []
            state_scores = np.empty(evaluator.CANDIDATE_COUNT, dtype=np.float64)
            for candidate in range(evaluator.CANDIDATE_COUNT):
                row = {
                    "state_id": state_id,
                    "family": family,
                    "split": role,
                    "candidate_index": candidate,
                    "route_intent_role": "translational",
                    "waypoint_features": [1.0, 0.0, 1.0, 0.0, 0.0, 1.0],
                    "predictor_candidate_action_plan_3x10": [
                        [float(candidate)] * 10 for _ in range(3)
                    ],
                    "p_d": float(candidate) / 10.0,
                    "p_theta": math.radians(float(candidate)),
                    "completed": candidate == evaluator.CANDIDATE_COUNT - 1,
                    "stuck": candidate == 2,
                    "immediate_contact_h1": candidate == 0,
                    "descriptive_contact_h2": False,
                    "descriptive_contact_h3": False,
                    "successor_safe_action_count": 0 if candidate == 1 else 1,
                    "successor_viable": candidate != 1,
                    "oracle_viability_admissible": candidate not in (0, 1),
                }
                metric_row = {
                    **row,
                    "role": role,
                }
                metric_row.pop("split")
                state_candidates.append(metric_row)
                score = float(candidate)
                state_scores[candidate] = score
                ledger.append(
                    {
                        **row,
                        "population_membership": evaluator._population_membership(row),
                        "scores": {"SYNTHETIC_SCORE": score},
                    }
                )
            candidates[role][state_id] = state_candidates
            scores[role]["SYNTHETIC_SCORE"][state_id] = state_scores
            cursor += 1
    return ledger, candidates, scores


def test_stage_a_ledger_replays_all_roles_without_model_inference() -> None:
    ledger, candidates, scores = _synthetic_stage_a_rows()
    reproduced = evaluator._replay_stage_a_metrics(ledger)

    assert len(ledger) == evaluator.STATE_COUNT * evaluator.CANDIDATE_COUNT
    for role in evaluator.SPLIT_ROLES:
        expected = metrics.summarize_scores(
            candidates[role],
            scores[role]["SYNTHETIC_SCORE"],
            source_id="SYNTHETIC_SCORE",
        )
        assert reproduced[role]["SYNTHETIC_SCORE"] == expected


def test_predecessor_raw_cost_reduction_is_barriered_bound_and_replayable(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage_rows, _candidates, _scores = _synthetic_stage_a_rows()
    datasets = evaluator._datasets_from_stage_rows(stage_rows)
    path = tmp_path / "candidate_evidence.jsonl.gz"
    successor_map = {
        "RAW_TRUE_FUTURE_GOAL_COSINE": "TRUE_FUTURE",
        "RAW_R1_GOAL_COSINE": "ONE_STEP_PREDICTED",
        "RAW_RR_GOAL_COSINE": "TWO_STEP_PREDICTED",
    }

    def predecessor_rows() -> list[dict[str, object]]:
        output: list[dict[str, object]] = []
        for successor_id, predecessor_source in successor_map.items():
            del successor_id
            for role in evaluator.SPLIT_ROLES:
                for state_id in sorted(
                    datasets[role], key=evaluator.numeric_state_key
                ):
                    for row in datasets[role][state_id]:
                        candidate = int(row["candidate_index"])
                        output.append(
                            {
                                "schema": "jepa_local_waypoint_candidate_evidence_v1",
                                "state_id": state_id,
                                "family": row["family"],
                                "role": role,
                                "candidate_index": candidate,
                                "source": predecessor_source,
                                "population_membership": copy.deepcopy(
                                    row["population_membership"]
                                ),
                                "cost_h3": float(candidate) / 7.0,
                                "oracle_route_fields_h3_primary": {
                                    "p_d": row["p_d"],
                                    "p_theta": row["p_theta"],
                                    "completed": row["completed"],
                                    "stuck": row["stuck"],
                                },
                                "immediate_contact_h1": row[
                                    "immediate_contact_h1"
                                ],
                                "successor_safe_action_count": row[
                                    "successor_safe_action_count"
                                ],
                                "successor_viable": row["successor_viable"],
                                "oracle_viability_admissible": row[
                                    "oracle_viability_admissible"
                                ],
                                "successor_nonviable": not bool(
                                    row["successor_viable"]
                                ),
                                "stuck": row["stuck"],
                                "completed": row["completed"],
                            }
                        )
        return output

    def publish(rows: list[dict[str, object]]) -> dict[str, object]:
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(evaluator.canonical_bytes(row).decode("utf-8"))
        return {
            "path": str(path),
            "sha256": evaluator.sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": 1_728,
            "schema": "jepa_local_waypoint_candidate_evidence_v1",
            "predecessor_sources": list(successor_map.values()),
            "successor_source_mapping": successor_map,
            "rows_per_source": 576,
            "score_field": "cost_h3",
            "score_transform": "score = -cost_h3 (higher is better)",
            "required_reduction_fields": list(
                evaluator.CONTRACT.PREDECESSOR_CANDIDATE_EVIDENCE_BINDING[
                    "required_reduction_fields"
                ]
            ),
            "open_barrier": "synthetic post-heldout barrier",
            "reduction_policy": "synthetic successor Borda reduction",
            "historical_aggregate_policy": "synthetic historical separation",
        }

    source_rows = predecessor_rows()
    record = publish(source_rows)
    monkeypatch.setattr(
        evaluator.CONTRACT, "PREDECESSOR_CANDIDATE_EVIDENCE_BINDING", record
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "RAW_COST_REREDUCED_SOURCE_IDS",
        tuple(successor_map),
    )
    with pytest.raises(evaluator.QualificationError, match="heldout barrier"):
        evaluator._predecessor_raw_goal_score_maps(
            datasets, heldout_barrier_open=False
        )

    maps, projected, receipt = evaluator._predecessor_raw_goal_score_maps(
        datasets, heldout_barrier_open=True
    )
    assert len(projected) == 1_728
    assert receipt["raw_cosine_inference_executions"] == 0
    for role in evaluator.SPLIT_ROLES:
        for source_id in successor_map:
            for state_id, values in maps[role][source_id].items():
                assert np.array_equal(
                    values,
                    np.asarray(
                        [-float(candidate) / 7.0 for candidate in range(12)]
                    ),
                ), state_id

    tampered = copy.deepcopy(source_rows)
    tampered[0]["population_membership"]["ORACLE_CONTACT_FREE"] = True
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "PREDECESSOR_CANDIDATE_EVIDENCE_BINDING",
        publish(tampered),
    )
    with pytest.raises(evaluator.QualificationError, match="panel alignment"):
        evaluator._predecessor_raw_goal_score_maps(
            datasets, heldout_barrier_open=True
        )


def test_matched_raw_cost_comparisons_persist_per_state_and_bootstrap() -> None:
    stage_rows, _candidates, _scores = _synthetic_stage_a_rows()
    for row in stage_rows:
        candidate = float(row["candidate_index"])
        row["scores"] = {
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": candidate,
            "RAW_TRUE_FUTURE_GOAL_COSINE": candidate - 0.1,
            "RAW_R1_GOAL_COSINE": candidate - 0.2,
            "RAW_RR_GOAL_COSINE": candidate - 0.3,
        }
    stage_a = evaluator._replay_stage_a_metrics(stage_rows)
    without_stage_b = evaluator._matched_raw_cost_comparisons(stage_a, None)
    true_comparison = without_stage_b["comparisons"][
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE"
    ]
    assert true_comparison["status"] == "COMPLETE"
    assert len(
        true_comparison["by_role"][evaluator.HELDOUT]["per_state_deltas"]
    ) == evaluator.CONTRACT.ROLE_STATE_COUNTS[evaluator.HELDOUT]
    assert (
        true_comparison["by_role"][evaluator.HELDOUT][
            "descriptive_paired_bootstrap"
        ]["selected_progress_gain_m"]["draws"]
        == metrics.BOOTSTRAP_DRAWS
    )
    assert without_stage_b["comparisons"][
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1"
    ]["status"] == "NOT_RUN_STAGE_B_NOT_AUTHORISED"

    stage_b = {
        "summaries": {
            role: {
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": copy.deepcopy(
                    stage_a[role]["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE"]
                ),
                "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR": copy.deepcopy(
                    stage_a[role]["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE"]
                ),
            }
            for role in evaluator.SPLIT_ROLES
        }
    }
    with_stage_b = evaluator._matched_raw_cost_comparisons(stage_a, stage_b)
    assert all(
        comparison["status"] == "COMPLETE"
        for comparison in with_stage_b["comparisons"].values()
    )


def test_route_only_target_ledger_reproduces_conditioned_training_payload() -> None:
    ledger, _candidates, _scores = _synthetic_stage_a_rows()
    datasets = evaluator._datasets_from_stage_rows(ledger)
    target_rows = evaluator._training_target_rows(datasets)

    assert len(target_rows) == evaluator.STATE_COUNT * evaluator.CANDIDATE_COUNT
    by_identity = {
        (str(row["state_id"]), int(row["candidate_index"])): row
        for row in target_rows
    }
    assert len(by_identity) == len(target_rows)
    for role, states in datasets.items():
        for state_id, rows in states.items():
            positions = evaluator.admissible_positions(rows)
            utilities = evaluator.route_utility([rows[position] for position in positions])
            expected = {
                int(rows[position]["candidate_index"]): float(utilities[index])
                for index, position in enumerate(positions)
            }
            for candidate in range(evaluator.CANDIDATE_COUNT):
                target = by_identity[(state_id, candidate)]
                conditioned = candidate in expected
                assert target["oracle_viability_admissible_conditioning"] is conditioned
                assert target["fit_state_optimization_status"] == (
                    evaluator.CONTRACT.fit_state_optimization_status(len(positions))
                    if role == evaluator.FIT
                    else None
                )
                assert target["used_for_fit"] is (
                    role == evaluator.FIT and conditioned
                )
                assert target["conditioned_margin_borda_utility"] == expected.get(
                    candidate
                )
                assert "all_candidate_margin_borda_utility" not in target


def test_stage_a_ledger_replay_fails_closed_on_identity_and_score_tamper() -> None:
    ledger, _candidates, _scores = _synthetic_stage_a_rows()

    missing = copy.deepcopy(ledger)
    missing.pop()
    with pytest.raises(evaluator.QualificationError, match="row count"):
        evaluator._replay_stage_a_metrics(missing)

    duplicate = copy.deepcopy(ledger)
    duplicate[1]["candidate_index"] = duplicate[0]["candidate_index"]
    with pytest.raises(evaluator.QualificationError, match="candidate set drift"):
        evaluator._replay_stage_a_metrics(duplicate)

    nonfinite = copy.deepcopy(ledger)
    nonfinite[0]["scores"]["SYNTHETIC_SCORE"] = float("nan")
    with pytest.raises(evaluator.QualificationError, match="score holes"):
        evaluator._replay_stage_a_metrics(nonfinite)

    population_tamper = copy.deepcopy(ledger)
    population_tamper[0]["population_membership"]["ORACLE_CONTACT_FREE"] = True
    with pytest.raises(evaluator.QualificationError, match="population-membership"):
        evaluator._replay_stage_a_metrics(population_tamper)


def test_score_row_alias_and_arithmetic_validation_fails_closed() -> None:
    stage_a = {
        "state_id": "synthetic-000",
        "candidate_index": 0,
        "latent_source": "TRUE",
        "deterministic_kinematic_score": 1.0,
        "no_latent_base_residual": 0.2,
        "latent_model_base_residual": 0.3,
        "latent_residual": 0.4,
        "final_no_latent_score": 1.2,
        "latent_branch_zero_score": 1.3,
        "final_latent_score": 1.7,
        "deranged_latent_score": 1.6,
        "scores": {
            "KINEMATIC_ROUTE_BASELINE": 1.0,
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": 1.2,
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": 1.7,
            "LATENT_BRANCH_ZERO_TRUE": 1.3,
            "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE": 1.6,
            "DETERMINISTIC_RANDOM": 0.0,
            **{
                source: -float(index)
                for index, source in enumerate(
                    evaluator.CONTRACT.RAW_COST_REREDUCED_SOURCE_IDS
                )
            },
        },
    }
    evaluator._validate_score_row_arithmetic([stage_a], stage="STAGE_A")

    arithmetic_tamper = copy.deepcopy(stage_a)
    arithmetic_tamper["latent_residual"] = 0.5
    with pytest.raises(evaluator.QualificationError, match="arithmetic/alias"):
        evaluator._validate_score_row_arithmetic(
            [arithmetic_tamper], stage="STAGE_A"
        )

    key_tamper = copy.deepcopy(stage_a)
    key_tamper["scores"].pop("RAW_RR_GOAL_COSINE")
    with pytest.raises(evaluator.QualificationError, match="condition-key"):
        evaluator._validate_score_row_arithmetic([key_tamper], stage="STAGE_A")

    conditional = copy.deepcopy(stage_a)
    conditional["latent_source"] = "R1"
    conditional["deranged_latent_score"] = None
    conditional["route_scores"] = {
        "KINEMATIC_ROUTE_BASELINE": 1.0,
        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": 1.2,
        "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1": 1.7,
        "LATENT_BRANCH_ZERO_R1": 1.3,
        "DETERMINISTIC_RANDOM": 0.0,
    }
    conditional.pop("scores")
    evaluator._validate_score_row_arithmetic(
        [conditional], stage="CONDITIONAL"
    )


def _synthetic_conditional_rows(sources: tuple[str, ...]) -> list[dict[str, object]]:
    stage_a, _candidates, _scores = _synthetic_stage_a_rows()
    rows: list[dict[str, object]] = []
    for source_offset, source in enumerate(sources):
        condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
        zero = f"LATENT_BRANCH_ZERO_{source}"
        for base in stage_a:
            candidate = int(base["candidate_index"])
            rows.append(
                {
                    **copy.deepcopy(base),
                    "latent_source": source,
                    "route_scores": {
                        "KINEMATIC_ROUTE_BASELINE": float(candidate),
                        "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": float(candidate) + 0.1,
                        condition: float(candidate) + float(source_offset) / 100.0,
                        zero: float(candidate) + 0.1,
                        "DETERMINISTIC_RANDOM": -float(candidate),
                    },
                }
            )
    return rows


@pytest.mark.parametrize(
    "sources",
    [
        ("R1", "RR", "P1", "PR"),
        evaluator.STAGE_C_SOURCE_IDS,
    ],
)
def test_conditional_ledgers_replay_without_inference_and_fail_on_tamper(
    sources: tuple[str, ...],
) -> None:
    rows = _synthetic_conditional_rows(sources)
    reproduced = evaluator._replay_conditional_metrics(
        rows, expected_sources=sources
    )
    assert len(rows) == (
        evaluator.STATE_COUNT * evaluator.CANDIDATE_COUNT * len(sources)
    )
    for role in evaluator.SPLIT_ROLES:
        for source in sources:
            assert (
                f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
                in reproduced[role]
            )

    tampered = copy.deepcopy(rows)
    tampered[1]["candidate_index"] = tampered[0]["candidate_index"]
    with pytest.raises(evaluator.QualificationError, match="candidate identity drift"):
        evaluator._replay_conditional_metrics(tampered, expected_sources=sources)

    nonfinite = copy.deepcopy(rows)
    condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{sources[0]}"
    nonfinite[0]["route_scores"][condition] = float("nan")
    with pytest.raises(evaluator.QualificationError, match="score holes"):
        evaluator._replay_conditional_metrics(nonfinite, expected_sources=sources)

    population_tamper = copy.deepcopy(rows)
    population_tamper[0]["population_membership"][
        "ORACLE_VIABILITY_ADMISSIBLE"
    ] = not population_tamper[0]["population_membership"][
        "ORACLE_VIABILITY_ADMISSIBLE"
    ]
    with pytest.raises(evaluator.QualificationError, match="population-membership"):
        evaluator._replay_conditional_metrics(
            population_tamper, expected_sources=sources
        )


def test_stage_c_candidate_matched_route_score_changes_replay_and_tamper() -> None:
    stage_b_rows = _synthetic_conditional_rows(("R1", "RR", "P1", "PR"))
    pr_scores = {
        (str(row["split"]), str(row["state_id"]), int(row["candidate_index"])): float(
            row["route_scores"]["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_PR"]
        )
        for row in stage_b_rows
        if row["latent_source"] == "PR"
    }
    rows = _synthetic_conditional_rows(evaluator.STAGE_C_SOURCE_IDS)
    for row in rows:
        source = str(row["latent_source"])
        condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
        matched = pr_scores[
            (str(row["split"]), str(row["state_id"]), int(row["candidate_index"]))
        ]
        row["matched_pr_score"] = matched
        row["deranged_minus_matched_pr_score"] = (
            float(row["route_scores"][condition]) - matched
        )
    evaluator._validate_stage_c_matched_pr_scores(rows, stage_b_rows)
    summary = evaluator._aggregate_stage_c_route_score_changes(rows)
    for source in evaluator.STAGE_C_SOURCE_IDS:
        assert summary["by_source"][source]["all_roles"]["rows"] == 576
        assert summary["by_source"][source]["by_role"][evaluator.HELDOUT][
            "rows"
        ] == 96

    tampered = copy.deepcopy(rows)
    tampered[0]["deranged_minus_matched_pr_score"] += 1.0
    with pytest.raises(evaluator.QualificationError, match="delta drift"):
        evaluator._aggregate_stage_c_route_score_changes(tampered)

    cross_join_tamper = copy.deepcopy(rows)
    cross_join_tamper[0]["matched_pr_score"] += 1.0
    source = str(cross_join_tamper[0]["latent_source"])
    condition = f"KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_{source}"
    cross_join_tamper[0]["deranged_minus_matched_pr_score"] = (
        float(cross_join_tamper[0]["route_scores"][condition])
        - float(cross_join_tamper[0]["matched_pr_score"])
    )
    with pytest.raises(evaluator.QualificationError, match="matched PR score drift"):
        evaluator._validate_stage_c_matched_pr_scores(
            cross_join_tamper, stage_b_rows
        )


def test_selected_candidate_change_evidence_is_complete_per_population_and_state() -> None:
    rows = _synthetic_conditional_rows(("R1", "RR", "P1", "PR"))
    summaries = evaluator._replay_conditional_metrics(
        rows, expected_sources=("R1", "RR", "P1", "PR")
    )[evaluator.HELDOUT]
    evidence = evaluator._selected_candidate_changes(
        summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_RR"],
        summaries["KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_R1"],
        contrast_id="RR_VS_R1",
    )
    for population in metrics.POPULATION_IDS:
        value = evidence["populations"][population]
        assert value["state_count"] == evaluator.CONTRACT.ROLE_STATE_COUNTS[
            evaluator.HELDOUT
        ]
        assert len(value["states"]) == value["state_count"]
        assert all(
            "left_selected_candidate_index" in row
            and "right_selected_candidate_index" in row
            and "selected_candidate_changed" in row
            for row in value["states"]
        )


def test_stage_b_collapsed_sources_fail_absolute_screens_without_exception() -> None:
    conditional_rows = _synthetic_conditional_rows(("R1", "RR", "P1", "PR"))
    for row in conditional_rows:
        row["route_scores"] = {
            key: 0.0 for key in row["route_scores"]
        }
    stage_b_heldout = evaluator._replay_conditional_metrics(
        conditional_rows, expected_sources=("R1", "RR", "P1", "PR")
    )[evaluator.HELDOUT]

    stage_a_rows, _candidates, _scores = _synthetic_stage_a_rows()
    for row in stage_a_rows:
        row["scores"] = {
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": 0.0,
        }
    stage_a_heldout = evaluator._replay_stage_a_metrics(stage_a_rows)[
        evaluator.HELDOUT
    ]
    decisions = evaluator._stage_b_decisions(
        stage_a_decisions={"true_future_gate": {"pass": True}},
        stage_a_heldout=stage_a_heldout,
        summaries=stage_b_heldout,
    )
    absolute = decisions["predicted_source_absolute_preservation"]
    assert absolute["per_source_pass"] == {
        "R1_RGB_ONE_STEP": False,
        "RR_RGB_ROLLOUT": False,
        "P1_PROPRIO_ONE_STEP": False,
        "PR_PROPRIO_ROLLOUT": False,
    }
    assert decisions["all_predicted_substitutions_fail_materially"] is True
    assert decisions["predicted_gate"]["pass"] is False
    assert decisions["proprioception_gate"]["pass"] is False


def test_terminal_primary_fails_closed_on_unresolved_predicted_substitution_edge() -> None:
    stage_a = {
        "true_future_gate": {"pass": True},
        "true_incremental_value": {"pass": True},
        "derangement": {"pass": True},
    }

    def stage_b(*, predicted: bool, all_fail: bool) -> dict[str, object]:
        return {
            "predicted_gate": {"pass": predicted},
            "all_predicted_substitutions_fail_materially": all_fail,
            "incremental_over_kinematics": {"pass": False},
            "proprioception_gate": {"pass": False},
        }

    primary, _secondaries, _next = evaluator._primary_and_secondaries(
        stage_a, stage_b(predicted=False, all_fail=True), None
    )
    assert primary == "TRUE_FUTURE_COST_SIGNAL_PREDICTOR_ROUTE_NO_GO"

    primary, _secondaries, _next = evaluator._primary_and_secondaries(
        stage_a, stage_b(predicted=True, all_fail=False), None
    )
    assert primary == "TWO_STEP_PLAN_AWARE_JEPA_COST_SIGNAL"

    with pytest.raises(evaluator.QualificationError, match="unresolved"):
        evaluator._primary_and_secondaries(
            stage_a, stage_b(predicted=False, all_fail=False), None
        )


def test_tied_stage_a_short_circuits_incremental_and_stops_at_no_signal() -> None:
    rows, _candidates, _scores = _synthetic_stage_a_rows()
    for row in rows:
        row["scores"] = {
            "KINEMATIC_ROUTE_BASELINE": 0.0,
            "KINEMATIC_PLUS_NO_LATENT_RESIDUAL": 0.0,
            "KINEMATIC_PLUS_JEPA_LATENT_RESIDUAL_TRUE": 0.0,
            "WITHIN_STATE_FUTURE_LATENT_DERANGEMENT_TRUE": 0.0,
        }
    heldout = evaluator._replay_stage_a_metrics(rows)[evaluator.HELDOUT]
    decisions = evaluator._stage_a_decisions(heldout)
    assert decisions["true_future_gate"]["pass"] is False
    assert decisions["true_incremental_value"] == {
        "schema": "plan_aware_incremental_route_value_gate_v1",
        "thresholds": dict(metrics.INCREMENTAL_ROUTE_VALUE_THRESHOLDS),
        "comparisons": {},
        "pass": False,
        "classification": "TRUE_FUTURE_JEPA_INCREMENTAL_ROUTE_VALUE_NOT_EVALUATED",
        "status": "NOT_EVALUATED_TRUE_GATE_FAILED",
        "evaluated": False,
        "reason": "TRUE_FUTURE_GATE_FAILED",
    }
    primary, _secondary, _next = evaluator._primary_and_secondaries(
        decisions, None, None
    )
    assert primary == "PLAN_AWARE_JEPA_COST_NO_SIGNAL"


def test_report_renders_none_metrics_as_na() -> None:
    result = _storage_result_core()
    heldout = result["metrics"]["stage_a"][evaluator.HELDOUT]
    for summary in heldout.values():
        aggregate = summary["populations"][
            metrics.ORACLE_VIABILITY_ADMISSIBLE
        ]["aggregate"]
        for key in (
            "pairwise_accuracy",
            "spearman_rho",
            "normalized_regret",
            "best_route_top3_rate",
            "selected_progress_ratio",
            "selected_route_progress_m_mean",
            "selected_heading_progress_rad_mean",
            "selected_combined_route_utility_mean",
        ):
            aggregate[key] = None
        aggregate["selected_combined_route_utility_count"] = 0
    report = evaluator._report_markdown(result)
    assert "n/a" in report


def test_row_level_latent_bindings_replay_and_fail_closed_on_tamper() -> None:
    state_id = "synthetic-000"
    candidate = 2
    tensor_rows: dict[tuple[str, str, int | None, int | None], dict] = {
        ("CURRENT", state_id, None, None): {
            "path": "current.npy",
            "sha256": "a" * 64,
            "bytes": 10,
        }
    }
    for horizon in evaluator.HORIZONS:
        tensor_rows[("P1_PROPRIO_ONE_STEP", state_id, candidate, horizon)] = {
            "path": "logical-state.npy",
            "sha256": chr(97 + horizon) * 64,
            "bytes": 20,
            "array_index": [candidate, horizon - 1],
        }
    row = {
        "latent_source": "P1",
        "state_id": state_id,
        "candidate_index": candidate,
        "future_derangement_donor_candidate": None,
        "latent_artifact_bindings": evaluator._candidate_latent_bindings(
            tensor_rows,
            state_id=state_id,
            candidate=candidate,
            source="P1",
        ),
    }
    evaluator._validate_persisted_latent_bindings([row], tensor_rows=tensor_rows)
    assert row["latent_artifact_bindings"]["candidate_future"]["H2"][
        "array_index"
    ] == [candidate, 1]

    tampered = copy.deepcopy(row)
    tampered["latent_artifact_bindings"]["candidate_future"]["H2"][
        "sha256"
    ] = "f" * 64
    with pytest.raises(evaluator.QualificationError, match="binding drift"):
        evaluator._validate_persisted_latent_bindings(
            [tampered], tensor_rows=tensor_rows
        )


@pytest.mark.parametrize(
    ("source_id", "stage_c", "donor_component"),
    [
        ("P1_PROPRIO_ONE_STEP", False, None),
        ("PR_VISUAL_CONTEXT_DERANGED", True, "visual"),
    ],
)
def test_prediction_index_custody_and_donor_inputs_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source_id: str,
    stage_c: bool,
    donor_component: str | None,
) -> None:
    monkeypatch.setattr(evaluator, "STATE_COUNT", 1)
    monkeypatch.setattr(evaluator, "CANDIDATE_COUNT", 1)
    recipient = "synthetic-000"
    donor = "synthetic-001"
    shard_recipient = {"path": "recipient.npz", "sha256": "a" * 64, "bytes": 1}
    shard_donor = {"path": "donor.npz", "sha256": "b" * 64, "bytes": 2}
    visual_recipient = [
        {"slot": slot, "path": f"r{slot}.npy", "sha256": "c" * 64, "bytes": 3}
        for slot in (-2, -1, 0)
    ]
    visual_donor = [
        {"slot": slot, "path": f"d{slot}.npy", "sha256": "d" * 64, "bytes": 4}
        for slot in (-2, -1, 0)
    ]
    authorities = {
        "state_order": [recipient],
        "role_by_state": {recipient: evaluator.HELDOUT},
        "family_by_state": {recipient: metrics.FAMILY_IDS[0]},
        "shard_by_state": {recipient: shard_recipient, donor: shard_donor},
        "visual_by_state": {recipient: visual_recipient, donor: visual_donor},
        "action_sha_by_state": {recipient: "e" * 64},
        "stage_c_donors": {
            condition: {recipient: donor}
            for condition in evaluator.STAGE_C_SOURCE_IDS
        },
    }
    monkeypatch.setattr(
        evaluator, "_conditional_prediction_authorities", lambda _attempt: authorities
    )
    prediction = tmp_path / "stage_b" / "predictions" / source_id / "state.npy"
    prediction.parent.mkdir(parents=True)
    prediction.write_bytes(b"synthetic-prediction-state")
    state_binding = evaluator.binding(prediction, relative_to=tmp_path)
    predictor_source = "PR_PROPRIO_ROLLOUT" if stage_c else source_id
    donors = {component: recipient for component in ("visual", "proprio", "control")}
    if donor_component is not None:
        donors[donor_component] = donor
    custody = {
        "source_id": predictor_source,
        "checkpoint": copy.deepcopy(
            evaluator.CONTRACT.CHECKPOINT_BINDINGS[predictor_source]
        ),
        "model_config": {
            "cell": (
                "proprio_one_step"
                if predictor_source == "P1_PROPRIO_ONE_STEP"
                else "proprio_rollout"
            ),
            "use_proprio": True,
            "rollout": predictor_source == "PR_PROPRIO_ROLLOUT",
            "width": 384,
        },
        "strict_state_dict_load": True,
        "eval_mode": True,
        "requires_grad_all_false": True,
        "parameter_digest_before": "f" * 64,
        "parameter_digest_after": "f" * 64,
        "parameter_state_unchanged": True,
        "checkpoint_optimizer_state_deserialized_but_ignored": True,
        "optimizer_state_loaded_into_an_optimizer": False,
        "optimizer_steps": 0,
        "state_batch_calls": 1,
        "model_forward_calls": 3,
        "future_proprioception_read_count": 0,
        "observed_slot_validity_by_horizon": {
            "1": [True, True, True],
            "2": [True, True, False],
            "3": [True, False, False],
        },
    }
    state_row = {
        "state_id": recipient,
        "family": metrics.FAMILY_IDS[0],
        "split_role": evaluator.HELDOUT,
        "source_id": source_id,
        "predictor_source_id": predictor_source,
        "ablation_or_null": source_id if stage_c else None,
        "input_state_ids": donors,
        "visual_context_bindings": (
            visual_donor if donor_component == "visual" else visual_recipient
        ),
        "proprio_shard_binding": (
            shard_donor if donor_component == "proprio" else shard_recipient
        ),
        "control_shard_binding": (
            shard_donor if donor_component == "control" else shard_recipient
        ),
        "candidate_action_sha256": "e" * 64,
        "prediction_state_tensor": state_binding,
    }
    logical = [
        {
            "kind": source_id,
            "state_id": recipient,
            "candidate_index_or_null": 0,
            "horizon_or_null": horizon,
            **state_binding,
            "array_index": [0, horizon - 1],
            "logical_shape": list(evaluator.TENSOR_SHAPE),
            "logical_dtype": "float16",
            "input_state_ids": donors,
        }
        for horizon in evaluator.HORIZONS
    ]
    gate_digest = "1" * 64
    stage_c_digest = "2" * 64 if stage_c else None
    value = evaluator.attach_digest(
        {
            "schema": evaluator._conditional_helper().PREDICTION_INDEX_SCHEMA,
            "status": "PASS",
            "experiment_id": evaluator.CONTRACT.EXPERIMENT_ID,
            "stage_b_gate_digest": gate_digest,
            "stage_c_gate_digest_or_null": stage_c_digest,
            "source_id": source_id,
            "predictor_source_id": predictor_source,
            "ablation_or_null": source_id if stage_c else None,
            "complete": True,
            "states": 1,
            "candidates_per_state": 1,
            "horizons": list(evaluator.HORIZONS),
            "logical_records_count": 3,
            "logical_records": logical,
            "state_records": [state_row],
            "prediction_shape_per_state": list(
                evaluator._conditional_helper().PREDICTION_STATE_SHAPE
            ),
            "dtype": "float16",
            "predictor_custody": custody,
            "future_proprioception": {
                "values_available_to_predictor": False,
                "masked_by_frozen_absence_mechanism": True,
                "read_count": 0,
            },
            "route_outcomes_opened": False,
            "training_executed": False,
        }
    )
    index_path = prediction.parent / "index.json"
    evaluator.atomic_json(index_path, value)
    rows, loaded = evaluator._load_prediction_index(
        attempt=tmp_path,
        source_id=source_id,
        gate_digest=gate_digest,
        stage_c_gate_digest=stage_c_digest,
    )
    assert len(rows) == 3
    assert loaded == value

    tampered = copy.deepcopy(value)
    tampered["predictor_custody"]["optimizer_steps"] = 1
    evaluator.atomic_json(index_path, evaluator.attach_digest(tampered))
    with pytest.raises(evaluator.QualificationError, match="contract drift"):
        evaluator._load_prediction_index(
            attempt=tmp_path,
            source_id=source_id,
            gate_digest=gate_digest,
            stage_c_gate_digest=stage_c_digest,
        )

    donor_tamper = copy.deepcopy(value)
    donor_tamper["state_records"][0]["input_state_ids"] = {
        component: recipient for component in ("visual", "proprio", "control")
    }
    evaluator.atomic_json(index_path, evaluator.attach_digest(donor_tamper))
    if stage_c:
        with pytest.raises(evaluator.QualificationError, match="input-custody"):
            evaluator._load_prediction_index(
                attempt=tmp_path,
                source_id=source_id,
                gate_digest=gate_digest,
                stage_c_gate_digest=stage_c_digest,
            )


def test_stage_c_top_receipt_binds_pr_checkpoint_gate_and_indexes(
    tmp_path: Path,
) -> None:
    gate = tmp_path / "receipts/stage_c_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"stage-c-gate\n")
    replay = tmp_path / "receipts/execution_correction_replay.json"
    replay.write_bytes(b"execution-correction-replay\n")
    for source in evaluator.STAGE_C_SOURCE_IDS:
        path = tmp_path / "stage_b" / "predictions" / source / "index.json"
        path.parent.mkdir(parents=True)
        path.write_bytes(source.encode("utf-8"))
    value = evaluator.attach_digest(
        {
            "schema": "plan_aware_monotone_jepa_cost_v1.stage_c_materialisation.v1",
            "status": "PASS",
            "experiment_id": evaluator.CONTRACT.EXPERIMENT_ID,
            "stage_b_gate_digest": "a" * 64,
            "execution_correction_replay": evaluator.binding(
                replay, relative_to=tmp_path
            ),
            "stage_c_gate": evaluator.binding(gate, relative_to=tmp_path),
            "prediction_indexes": {
                source: evaluator.binding(
                    tmp_path / "stage_b" / "predictions" / source / "index.json",
                    relative_to=tmp_path,
                )
                for source in evaluator.STAGE_C_SOURCE_IDS
            },
            "checkpoint": copy.deepcopy(
                evaluator.CONTRACT.CHECKPOINT_BINDINGS["PR_PROPRIO_ROLLOUT"]
            ),
            "training_executed": False,
            "outcome_informed_donor_selection": False,
            "nothing_running_at_receipt_write": True,
        }
    )
    evaluator._validate_stage_c_top_receipt(
        attempt=tmp_path,
        value=value,
        stage_b_gate_digest="a" * 64,
        stage_c_gate_path=gate,
    )
    tampered = copy.deepcopy(value)
    tampered["checkpoint"] = {**tampered["checkpoint"], "sha256": "0" * 64}
    tampered = evaluator.attach_digest(tampered)
    with pytest.raises(evaluator.QualificationError, match="Stage-C helper"):
        evaluator._validate_stage_c_top_receipt(
            attempt=tmp_path,
            value=tampered,
            stage_b_gate_digest="a" * 64,
            stage_c_gate_path=gate,
        )


def test_true_future_index_cross_binding_reconciles_and_detects_tamper() -> None:
    expected_index_sha = "d" * 64
    target = {
        "entries": [
            {
                "state_id": "synthetic-000",
                "candidate_index": 3,
                "horizon": 2,
                "shape": [768, 1024],
                "dtype": "float16",
                "sha256": "a" * 64,
                "latent_path": "future.npy",
            }
        ]
    }
    tensor = {
        "records": [
            {
                "kind": "TRUE_FUTURE",
                "state_id": "synthetic-000",
                "candidate_index_or_null": 3,
                "horizon_or_null": 2,
                "shape": [768, 1024],
                "dtype": "float16",
                "sha256": "a" * 64,
                "external_existing_artifact": {
                    "path": "future.npy",
                    "sha256": "a" * 64,
                    "index_sha256": expected_index_sha,
                    "array_equality": True,
                },
            }
        ]
    }
    evaluator._validate_true_future_index_reconciliation(
        target,
        tensor,
        expected_index_sha256=expected_index_sha,
        expected_count=1,
    )
    tampered = copy.deepcopy(tensor)
    tampered["records"][0]["external_existing_artifact"]["index_sha256"] = (
        "e" * 64
    )
    with pytest.raises(evaluator.QualificationError, match="reconciliation drift"):
        evaluator._validate_true_future_index_reconciliation(
            target,
            tampered,
            expected_index_sha256=expected_index_sha,
            expected_count=1,
        )


def test_target_encoder_exact_file_record_detects_byte_and_hash_tamper(
    tmp_path: Path,
) -> None:
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"synthetic encoder bytes")
    record = evaluator.binding(encoder)
    evaluator.verify_exact_file_record(encoder, record, label="target_encoder")
    encoder.write_bytes(b"synthetic encoder drift")
    with pytest.raises(evaluator.QualificationError, match="target_encoder SHA drift"):
        evaluator.verify_exact_file_record(encoder, record, label="target_encoder")


def test_stage_c_donor_derangements_are_exact_contract_mappings() -> None:
    ids: dict[str, list[str]] = {role: [] for role in evaluator.SPLIT_ROLES}
    manifests: dict[str, dict[str, str]] = {}
    cursor = 0
    for role in evaluator.SPLIT_ROLES:
        per_family = evaluator.CONTRACT.ROLE_STATE_COUNTS[role] // len(
            metrics.FAMILY_IDS
        )
        for family in metrics.FAMILY_IDS:
            for _ in range(per_family):
                state_id = f"synthetic-{cursor:03d}"
                cursor += 1
                ids[role].append(state_id)
                manifests[state_id] = {"family": family}
    digest = "c" * 64
    observed = evaluator.donor_derangements(ids, manifests, digest)
    for ablation in evaluator.CONTRACT.STAGE_C_ABLATION_INPUTS:
        expected: dict[str, str] = {}
        for role in evaluator.SPLIT_ROLES:
            for family in metrics.FAMILY_IDS:
                cohort = [
                    state_id
                    for state_id in ids[role]
                    if manifests[state_id]["family"] == family
                ]
                authority = evaluator.CONTRACT.build_stage_c_donor_mapping(
                    cohort,
                    split=role,
                    family=family,
                    ablation=ablation,
                    contract_digest=digest,
                )
                expected.update(
                    {
                        row["recipient_state_id"]: row["donor_state_id"]
                        for row in authority["rows"]
                    }
                )
        assert observed[ablation] == expected


def test_byte_and_hash_tampering_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"bound-bytes")
    manifest = [evaluator.binding(artifact, relative_to=tmp_path)]
    evaluator._validate_manifest(tmp_path, manifest)
    artifact.write_bytes(b"tampered!!")
    with pytest.raises(evaluator.QualificationError, match="manifest drift"):
        evaluator._validate_manifest(tmp_path, manifest)

    route_path = tmp_path / "route.jsonl"
    route_path.write_bytes(
        b"".join(
            evaluator.canonical_bytes(
                {"state_id": "synthetic-000", "candidate_index": index, "value": 1}
            )
            for index in range(2)
        )
    )
    monkeypatch.setattr(evaluator, "ROUTE_LABELS", route_path)
    monkeypatch.setattr(evaluator, "STATE_COUNT", 1)
    monkeypatch.setattr(evaluator, "CANDIDATE_COUNT", 2)
    index = evaluator.route_line_index()
    route_path.write_bytes(route_path.read_bytes().replace(b'"value":1', b'"value":2'))
    with pytest.raises(evaluator.QualificationError, match="route row bytes drift"):
        evaluator.read_route_state("synthetic-000", index)


def test_new_attempt_fails_closed_on_existing_canonical_or_live_attempt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "qualification"
    tracked_result = tmp_path / "tracked-result.json"
    tracked_report = tmp_path / "tracked-report.md"
    monkeypatch.setattr(evaluator.CONTRACT, "OUTPUT_ROOT", canonical)
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_RESULT_PATH", tracked_result)
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_REPORT_PATH", tracked_report)

    canonical.mkdir()
    with pytest.raises(evaluator.QualificationError, match="already exists"):
        evaluator._new_attempt(canonical, "a" * 40)
    canonical.rmdir()

    live = tmp_path / ".qualification.attempt-synthetic"
    live.mkdir()
    with pytest.raises(evaluator.QualificationError, match="live/abandoned attempt"):
        evaluator._new_attempt(canonical, "a" * 40)

    live.rmdir()
    tracked_result.write_bytes(b"pre-existing")
    with pytest.raises(evaluator.QualificationError, match="publication destination"):
        evaluator._new_attempt(canonical, "a" * 40)
    assert tracked_result.read_bytes() == b"pre-existing"


def test_new_attempt_rejects_unvalidated_prior_failed_attempt_without_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "qualification"
    tracked_result = tmp_path / "tracked-result.json"
    tracked_report = tmp_path / "tracked-report.md"
    monkeypatch.setattr(evaluator.CONTRACT, "OUTPUT_ROOT", canonical)
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_RESULT_PATH", tracked_result)
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_REPORT_PATH", tracked_report)
    failed = tmp_path / ".qualification.failed-synthetic"
    failed.mkdir()
    with pytest.raises(evaluator.QualificationError, match="lacks its failure receipt"):
        evaluator._new_attempt(canonical, "a" * 40)
    assert failed.is_dir()
    assert not canonical.exists()


def test_new_attempt_allows_only_fresh_corrected_smoke_retry_with_bound_archive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "qualification"
    monkeypatch.setattr(evaluator.CONTRACT, "OUTPUT_ROOT", canonical)
    monkeypatch.setattr(
        evaluator.CONTRACT, "TRACKED_RESULT_PATH", tmp_path / "tracked-result.json"
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "TRACKED_REPORT_PATH", tmp_path / "tracked-report.md"
    )
    monkeypatch.setattr(evaluator, "_active_experiment_processes", lambda: [])
    monkeypatch.setattr(
        evaluator.subprocess,
        "run",
        lambda *_args, **_kwargs: evaluator.subprocess.CompletedProcess([], 0),
    )
    archive = tmp_path / ".qualification.failed-smoke"
    failure_path = archive / "receipts/failure.json"
    evaluator.atomic_json(
        archive / "receipts/source_closure.json",
        evaluator.CONTRACT.attach_self_digest(
            {
                "schema": evaluator.CONTRACT.SOURCE_CLOSURE_SCHEMA_VERSION,
                "experiment_id": evaluator.CONTRACT.EXPERIMENT_ID,
                "source_commit": evaluator.SOURCE_COMMIT,
                "declared_paths": [],
                "rows": [],
                "row_count": 0,
                "missing_paths": [],
                "complete": True,
                "outcome_or_result_payloads_parsed": [],
                "custody_only_predecessor_results_hashed_without_parsing": [],
                "generated_cache_paths_traversed": [],
                "accidental_exposures": [],
            }
        ),
    )
    evaluator.atomic_json(
        failure_path,
        evaluator.attach_digest(
            {
                "schema": "plan_aware_monotone_jepa_failure_v1",
                "source_freeze_commit": "b" * 40,
                "phase": "TRAINING_SMOKE",
                "error_type": "QualificationError",
                "error_message": "ordinary synthetic implementation fault",
                "partial_artifacts_reusable": False,
                "full_training_epochs_completed": 0,
                "calibration_rows_opened": 0,
                "heldout_rows_opened": 0,
                "final_checkpoint_published": False,
                "nothing_running": True,
                "prohibition_counters": evaluator._prohibition_counters(),
            }
        ),
    )
    attempt = evaluator._new_attempt(canonical, "a" * 40)
    assert attempt.is_dir()
    assert list(attempt.iterdir()) == []
    custody = evaluator._validated_prior_smoke_failure_custody(
        canonical, source_freeze="a" * 40
    )
    assert len(custody) == 1
    assert set(custody[0]) == set(
        evaluator.CONTRACT.PRIOR_SMOKE_FAILURE_CUSTODY_RECORD_FIELDS
    )
    assert custody[0]["files_reused"] == 0
    assert custody[0]["source_closure"] is not None
    assert custody[0]["inventory"]["files"] == 2

    attempt.rmdir()
    closure_path = archive / "receipts/source_closure.json"
    valid_closure = evaluator.load_json(closure_path)
    tampered_closure = copy.deepcopy(valid_closure)
    tampered_closure["complete"] = False
    evaluator.atomic_json(closure_path, tampered_closure)
    with pytest.raises(evaluator.QualificationError, match="self-digest drift"):
        evaluator._new_attempt(canonical, "c" * 40)
    evaluator.atomic_json(closure_path, valid_closure)

    invalid = evaluator.load_json(failure_path)
    invalid["phase"] = "ROUTE_RANKER_TRAINING"
    evaluator.atomic_json(failure_path, evaluator.attach_digest(invalid))
    with pytest.raises(evaluator.QualificationError, match="not an authorised"):
        evaluator._new_attempt(canonical, "c" * 40)


def test_tracked_result_and_report_publication_is_all_or_absent(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    output = tmp_path / "output"
    output.mkdir()
    (output / "result.json").write_bytes(b'{"result":true}\n')
    (output / "report.md").write_bytes(b"# synthetic report\n")
    tracked_result = tmp_path / "docs/result.json"
    tracked_report = tmp_path / "docs/report.md"
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_RESULT_PATH", tracked_result)
    monkeypatch.setattr(evaluator.CONTRACT, "TRACKED_REPORT_PATH", tracked_report)

    real_atomic_bytes = evaluator.atomic_bytes

    def fail_after_report_replace(path: Path, payload: bytes) -> None:
        real_atomic_bytes(path, payload)
        if path == tracked_report:
            raise OSError("synthetic post-replace publication failure")

    monkeypatch.setattr(evaluator, "atomic_bytes", fail_after_report_replace)
    with pytest.raises(OSError, match="post-replace"):
        evaluator._publish_tracked_result_and_report(output)
    assert not tracked_result.exists()
    assert not tracked_report.exists()

    monkeypatch.setattr(evaluator, "atomic_bytes", real_atomic_bytes)
    evaluator._publish_tracked_result_and_report(output)
    assert tracked_result.read_bytes() == (output / "result.json").read_bytes()
    assert tracked_report.read_bytes() == (output / "report.md").read_bytes()

    with pytest.raises(evaluator.QualificationError, match="publication destination"):
        evaluator._publish_tracked_result_and_report(output)


def test_execution_correction_runtime_custody_is_mandatory_and_normalized(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    freeze = "e" * 40
    archive = {
        "archive_path": str(tmp_path / ".failed-bound"),
        "inventory": {"files": 2, "bytes": 7, "manifest_sha256": "1" * 64},
        "failure_receipt": {
            "path": "receipts/failure.json",
            "sha256": "2" * 64,
            "bytes": 3,
            "content_digest": "3" * 64,
        },
        "files_reused": 0,
        "pass": True,
    }
    amendment_closure_path = tmp_path / "amendment_source_closure.json"
    amendment_closure_path.write_bytes(b"{}\n")
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "TRACKED_EXECUTION_CORRECTION_SOURCE_CLOSURE_PATH",
        amendment_closure_path,
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "EXECUTION_CORRECTION_FAILED_ARCHIVE",
        Path(archive["archive_path"]),
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_execution_correction_source_closure",
        lambda _path: {"content_digest": "4" * 64, "row_count": 6},
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "validate_execution_correction_freeze_custody",
        lambda _root: {
            "source_freeze_commit": freeze,
            "failed_archive_custody": archive,
            "files_reused": 0,
            "fresh_attempts_authorised": 1,
            "fresh_attempts_already_consumed": 0,
            "pass": True,
        },
    )
    custody = evaluator._runtime_execution_correction_custody()
    assert custody["source_freeze_commit"] == freeze
    assert custody["archive_inventory"] == archive["inventory"]
    assert custody["amendment_source_closure"]["rows"] == 6
    assert custody["files_reused"] == 0
    assert custody["conditional_child_environment_preflight"] is None
    assert custody["execution_correction_replay"] is None

    monkeypatch.setattr(
        evaluator.CONTRACT,
        "validate_execution_correction_freeze_custody",
        lambda _root: {"pass": False},
    )
    with pytest.raises(evaluator.QualificationError, match="runtime custody"):
        evaluator._runtime_execution_correction_custody()


def test_execution_correction_uses_active_closure_and_rejects_live_drift(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    live_source = tmp_path / "changed.py"
    live_source.write_bytes(b"corrected = True\n")
    live_row = {
        "path": live_source.name,
        "sha256": evaluator.sha256_file(live_source),
        "bytes": live_source.stat().st_size,
    }
    historical_closure = {
        "rows": [{**live_row, "sha256": "0" * 64}],
        "row_count": 1,
        "complete": True,
        "content_digest": "1" * 64,
    }
    amendment_closure = {
        "rows": [live_row],
        "row_count": 1,
        "complete": True,
        "content_digest": "2" * 64,
    }

    stub = tmp_path / "stub.bin"
    stub.write_bytes(b"stub")
    predecessor = {
        label: {"path": stub.name, "sha256": "3" * 64, "bytes": 4}
        for label in (
            "tensor_index",
            "batch_manifest",
            "goal_view_index",
            "context_reconstruction_index",
            "persistence_receipt",
        )
    }
    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "PREDECESSOR_ROOT", tmp_path)
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "PANEL_BINDINGS",
        {
            "true_future_index": {
                "path": stub.name,
                "sha256": "3" * 64,
                "bytes": 4,
            }
        },
    )
    monkeypatch.setattr(evaluator.CONTRACT, "CHECKPOINT_BINDINGS", {})
    monkeypatch.setattr(
        evaluator.CONTRACT, "PREDECESSOR_TENSOR_PACKAGE", predecessor
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "ENCODER_BINDING", {"path": str(stub)}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_contract", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_output_schema", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_evaluator_fixture", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_source_closure",
        lambda _path: historical_closure,
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_execution_correction_source_closure",
        lambda _path: amendment_closure,
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_route_role_authority",
        lambda _path: {"records": []},
    )
    monkeypatch.setattr(evaluator, "verify_binding", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        evaluator, "verify_exact_file_record", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(evaluator, "load_json", lambda _path: {})
    monkeypatch.setattr(
        evaluator,
        "_validate_true_future_index_reconciliation",
        lambda *_args, **_kwargs: None,
    )

    frozen = evaluator._validate_frozen_authorities(
        execution_correction_custody={"pass": True}
    )
    assert frozen["source_closure"] is historical_closure
    assert frozen["execution_correction_source_closure"] is amendment_closure

    live_source.write_bytes(b"corrected = False\n")
    with pytest.raises(evaluator.QualificationError, match="source-closure row drift"):
        evaluator._validate_frozen_authorities(
            execution_correction_custody={"pass": True}
        )


def test_execution_correction_attempt_is_fresh_and_reuses_no_archive_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    canonical = tmp_path / "plan_aware"
    archive = tmp_path / ".plan_aware.failed-bound"
    archive.mkdir()
    archive_inventory = {
        "files": 1,
        "bytes": 1,
        "manifest_sha256": "a" * 64,
        "rows": [{"path": "receipt", "sha256": "b" * 64, "bytes": 1}],
    }
    monkeypatch.setattr(evaluator.CONTRACT, "OUTPUT_ROOT", canonical)
    monkeypatch.setattr(
        evaluator.CONTRACT, "EXECUTION_CORRECTION_FAILED_ARCHIVE", archive
    )
    monkeypatch.setattr(evaluator, "_assert_publication_destinations_absent", lambda: None)
    monkeypatch.setattr(evaluator, "_active_experiment_processes", lambda: [])
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "validate_execution_correction_archive",
        lambda: {
            "archive_path": str(archive),
            "inventory": archive_inventory,
            "files_reused": 0,
            "pass": True,
        },
    )
    custody = {
        "source_freeze_commit": "c" * 40,
        "archive_path": str(archive),
        "archive_inventory": archive_inventory,
        "files_reused": 0,
    }
    attempt = evaluator._new_attempt(
        canonical,
        "c" * 40,
        execution_correction_custody=custody,
    )
    assert attempt.is_dir()
    assert list(attempt.iterdir()) == []
    assert list(archive.iterdir()) == []
    attempt.rmdir()

    second_failure = tmp_path / ".plan_aware.failed-second"
    second_failure.mkdir()
    with pytest.raises(evaluator.QualificationError, match="namespace drift"):
        evaluator._new_attempt(
            canonical,
            "c" * 40,
            execution_correction_custody=custody,
        )


def test_replay_gate_is_durable_before_conditional_child_and_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    receipt = evaluator.attach_digest(
        {
            "schema": evaluator.CONTRACT.EXECUTION_CORRECTION_REPLAY_SCHEMA_VERSION,
            "failed_archive": "/bound/archive",
            "fresh_attempt": str(tmp_path),
            "files_reused": 0,
            "pass": True,
        }
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "validate_execution_correction_replay",
        lambda _attempt: copy.deepcopy(receipt),
    )
    observed = evaluator._publish_execution_correction_replay_gate(
        attempt=tmp_path,
        execution_correction_custody={"archive_path": "/bound/archive"},
    )
    path = tmp_path / "receipts/execution_correction_replay.json"
    assert observed == receipt
    assert evaluator.load_json(path) == receipt

    path.unlink()
    (tmp_path / "stage_b").mkdir()
    with pytest.raises(evaluator.QualificationError, match="before execution-correction"):
        evaluator._publish_execution_correction_replay_gate(
            attempt=tmp_path,
            execution_correction_custody={"archive_path": "/bound/archive"},
        )


def test_stage_b_never_starts_helper_without_correction_custody(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gate = tmp_path / "receipts/stage_b_gate.json"
    gate.parent.mkdir(parents=True)
    gate.write_bytes(b"gate\n")
    monkeypatch.setattr(
        evaluator,
        "_publish_stage_b_gate",
        lambda **_kwargs: (gate, {"content_digest": "a" * 64}),
    )
    helper_started = False

    def forbidden_helper(**_kwargs: object) -> dict[str, object]:
        nonlocal helper_started
        helper_started = True
        return {}

    monkeypatch.setattr(evaluator, "_run_conditional_helper_cli", forbidden_helper)
    with pytest.raises(evaluator.QualificationError, match="without correction custody"):
        evaluator._execute_conditional_stage_b(
            attempt=tmp_path,
            source_freeze="c" * 40,
            datasets={},
            models={},
            tensor_rows={},
            evaluation_contract={},
            stage_a_metrics={},
            stage_a_decisions={"true_future_gate": {"pass": True}},
            execution_correction_custody={},
        )
    assert helper_started is False


def test_correction_authority_validation_uses_amendment_closure_for_live_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Historical closure stays frozen while the overlay binds changed Python."""

    live = tmp_path / "changed.py"
    live.write_bytes(b"corrected implementation\n")
    live_row = evaluator.binding(live, relative_to=tmp_path)
    historical_row = {
        "path": "changed.py",
        "sha256": "0" * 64,
        "bytes": 1,
    }
    historical = {"rows": [historical_row], "content_digest": "1" * 64}
    amendment = {
        "rows": [live_row],
        "row_count": 1,
        "content_digest": "2" * 64,
    }
    encoder = tmp_path / "encoder.pt"
    encoder.write_bytes(b"encoder")
    index = tmp_path / "index.json"
    index.write_text("{}\n", encoding="utf-8")
    predecessor = tmp_path / "predecessor"
    predecessor.mkdir()
    predecessor_file = predecessor / "bound.json"
    predecessor_file.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(evaluator, "ROOT", tmp_path)
    monkeypatch.setattr(evaluator, "PREDECESSOR_ROOT", predecessor)
    monkeypatch.setattr(evaluator, "TARGET_LATENT_INDEX", index)
    monkeypatch.setattr(evaluator, "LATENT_INDEX", index)
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_contract", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_output_schema", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT, "load_and_validate_evaluator_fixture", lambda _path: {}
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_source_closure",
        lambda _path: copy.deepcopy(historical),
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_execution_correction_source_closure",
        lambda _path: copy.deepcopy(amendment),
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "load_and_validate_route_role_authority",
        lambda _path: {"records": []},
    )
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "PANEL_BINDINGS",
        {
            "true_future_index": {
                "path": str(index),
                "sha256": evaluator.sha256_file(index),
                "bytes": index.stat().st_size,
            }
        },
    )
    monkeypatch.setattr(evaluator.CONTRACT, "CHECKPOINT_BINDINGS", {})
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "ENCODER_BINDING",
        {
            "path": str(encoder),
            "sha256": evaluator.sha256_file(encoder),
            "bytes": encoder.stat().st_size,
        },
    )
    predecessor_record = {
        "path": predecessor_file.name,
        "sha256": evaluator.sha256_file(predecessor_file),
        "bytes": predecessor_file.stat().st_size,
    }
    monkeypatch.setattr(
        evaluator.CONTRACT,
        "PREDECESSOR_TENSOR_PACKAGE",
        {
            key: copy.deepcopy(predecessor_record)
            for key in (
                "tensor_index",
                "batch_manifest",
                "goal_view_index",
                "context_reconstruction_index",
                "persistence_receipt",
            )
        },
    )
    monkeypatch.setattr(evaluator, "load_json", lambda _path: {})
    monkeypatch.setattr(
        evaluator, "_validate_true_future_index_reconciliation", lambda *_a, **_k: None
    )

    frozen = evaluator._validate_frozen_authorities(
        execution_correction_custody={"pass": True}
    )
    assert frozen["source_closure"] == historical
    assert frozen["execution_correction_source_closure"] == amendment

    live.write_bytes(b"unbound drift\n")
    with pytest.raises(evaluator.QualificationError, match="source-closure row drift"):
        evaluator._validate_frozen_authorities(
            execution_correction_custody={"pass": True}
        )
