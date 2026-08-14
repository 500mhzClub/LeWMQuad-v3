from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pytest

from scripts import run_go2_scorer_fit_full_bank_v2 as runner


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
TARGET = {
    "path": (".generated/go2_branch_corpus_v1_2/scorer_fit/"
             "latents_v2/horizon/smoke-candidate-0.f16"),
    "sha256": HEX_A,
    "byte_count": 4 * 768 * 1024 * 2,
    "shape": [4, 768, 1024],
}


def _closed(kind: str, **values: Any) -> dict[str, Any]:
    return {
        "validation_kind": kind,
        "pass": True,
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
        **values,
    }


def _encoding(kind: str, *, new_context: int, new_horizon: int,
              zero_new: bool, regenerated: bool) -> dict[str, Any]:
    return _closed(
        kind,
        state_count=1,
        horizon_latent_count=12,
        horizon_shape=[4, 768, 1024],
        registered_smoke_shard_inventory_digest=HEX_A,
        registered_smoke_stable_artifact_inventory_digest=HEX_B,
        invocation_new_context_shards=new_context,
        invocation_new_horizon_shards=new_horizon,
        zero_new_resume_verified=zero_new,
        single_registered_shard_regenerated=regenerated,
        only_registered_missing_shard_changed=regenerated,
        single_shard_regeneration_target=dict(TARGET),
    )


class FakePipeline:
    def __init__(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
            training_kind: str = "QUALIFICATION_PASS",
            retained_training_kind: str | None = None,
            retained_development: bool = False,
            retained_smoke: str = "absent",
            ) -> None:
        self.events: list[str] = []
        self.commands_run: list[str] = []
        self.deleted: list[Mapping[str, Any]] = []
        self.validation_counts: dict[str, int] = {}
        self.training_kind = training_kind
        self.retained_training_kind = retained_training_kind
        self.retained_development = retained_development
        self.retained_smoke = retained_smoke
        self.root = tmp_path
        self.commands = {
            stage: [stage] for stage in runner._V2_RUNTIME_STAGE_ROLES
        }

        monkeypatch.setattr(
            runner.SCORER_CONTRACT, "load_contract_for_consumption",
            lambda **_kwargs: {})
        monkeypatch.setattr(
            runner.BUILDER,
            "load_and_validate_full_bank_v2_manifests_for_consumption",
            lambda **_kwargs: {})
        monkeypatch.setattr(
            runner.DESIGN, "audit_v2_runtime_outputs_absent",
            lambda **_kwargs: self.events.append("runtime-absence") or {})
        monkeypatch.setattr(
            runner, "_ensure_final_eval_absent",
            lambda **_kwargs: self.events.append("final-eval-absence"))
        monkeypatch.setattr(
            runner, "_bound_interpreters",
            lambda **_kwargs: {
                "genesis": tmp_path / "genesis-python",
                "rocm": tmp_path / "rocm-python",
            })
        monkeypatch.setattr(
            runner, "validate_runtime_probe_receipt",
            lambda _value, *, runtime_role, authority: {
                runner.RUNTIME_PROBE_SELF_KEY:
                    (HEX_A if runtime_role == "genesis" else HEX_B)})
        monkeypatch.setattr(
            runner, "downstream_command_sequence",
            lambda **_kwargs: self.commands)

    def runtime_probe(self, role: str, _root: Path, _interpreter: Path,
                      _authority: Any) -> Mapping[str, Any]:
        self.events.append(f"probe:{role}")
        return {}

    def command(self, command: list[str], _root: Path) -> int:
        stage = command[0]
        self.events.append(f"command:{stage}")
        self.commands_run.append(stage)
        if (stage == "scorer_training_and_qualification"
                and self.training_kind != "QUALIFICATION_PASS"):
            return 1
        return 0

    def delete(self, target: Mapping[str, Any], _root: Path) -> None:
        self.events.append("delete:registered-smoke-shard")
        self.deleted.append(dict(target))

    def validate(self, kind: str, _root: Path,
                 _interpreter: Path) -> Mapping[str, Any]:
        self.events.append(f"validate:{kind}")
        count = self.validation_counts.get(kind, 0)
        self.validation_counts[kind] = count + 1
        if kind == "branch-smoke":
            return _closed(
                kind, branch_count=12, candidate_indices=list(range(12)),
                rendered_horizon_frame_count=48,
                registered_smoke_artifact_inventory_digest=HEX_C)
        if kind == "encoding-smoke":
            if self.retained_smoke == "deletion-window":
                return _encoding(
                    kind, new_context=0, new_horizon=1,
                    zero_new=True, regenerated=True)
            sequence = (
                (1, 12, False, False),
                (0, 0, True, False),
                (0, 1, True, True),
            )
            new_context, new_horizon, zero_new, regenerated = sequence[
                min(count, len(sequence) - 1)]
            return _encoding(
                kind, new_context=new_context, new_horizon=new_horizon,
                zero_new=zero_new, regenerated=regenerated)
        if kind == "encoding-smoke-optional":
            if self.retained_smoke == "absent":
                return _closed(kind, terminal_present=False)
            complete = self.retained_smoke == "complete"
            return _closed(
                kind, terminal_present=True,
                smoke_protocol_complete=complete,
                zero_new_resume_verified=True,
                single_registered_shard_regenerated=complete,
                only_registered_missing_shard_changed=complete)
        if kind == "branch-corpus":
            return _closed(kind, state_count=120, branch_count=1_440)
        if kind == "encoded-corpus":
            return _closed(
                kind, state_count=120, horizon_latent_count=1_440)
        if kind == "training-terminal-optional":
            if self.retained_training_kind is None:
                return _closed(kind, terminal_present=False)
            return _closed(
                kind, terminal_present=True,
                terminal_kind=self.retained_training_kind,
                qualified=self.retained_training_kind == "QUALIFICATION_PASS",
                terminal_digest=HEX_A)
        if kind == "training-terminal":
            return _closed(
                kind, terminal_kind=self.training_kind,
                qualified=self.training_kind == "QUALIFICATION_PASS",
                terminal_digest=HEX_A)
        if kind == "development-terminal-optional":
            if not self.retained_development:
                return _closed(kind, terminal_present=False)
            return _closed(
                kind, terminal_present=True, terminal_digest=HEX_B,
                qualified_scorer_bound=True, development_state_count=20,
                development_branch_count=240)
        if kind == "development-terminal":
            return _closed(
                kind, terminal_digest=HEX_B, qualified_scorer_bound=True,
                development_state_count=20, development_branch_count=240)
        raise AssertionError(f"unexpected validation kind: {kind}")

    def run(self, *, resume: bool = False) -> tuple[int, dict[str, Any]]:
        return runner.run_pipeline(
            root=self.root,
            command_runner=self.command,
            runtime_probe_invoker=self.runtime_probe,
            validation_invoker=self.validate,
            delete_registered_shard=self.delete,
            authority=object(),
            resume=resume,
        )


def test_pass_pipeline_has_exact_order_and_development_is_pass_gated(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path)
    code, report = fake.run()
    assert code == 0
    assert fake.commands_run == [
        "branch_smoke", "branch_smoke_zero_new", "smoke_encoding",
        "smoke_encoding_zero_new", "smoke_single_shard_regeneration",
        "full_branch_corpus", "full_latent_encoding",
        "scorer_training_and_qualification", "development_transfer",
    ]
    assert fake.deleted == [TARGET]
    assert fake.events.index("validate:training-terminal") < fake.events.index(
        "command:development_transfer")
    assert report["qualified"] is True
    assert report["predictor_access_before_qualification"] is False
    assert report["final_200_state_corpus_generated"] is False


@pytest.mark.parametrize(
    "terminal_kind",
    ["COMPLETION_DEGENERACY_FAILURE", "QUALIFICATION_FAILURE"],
)
def test_frozen_training_failure_stops_before_development(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        terminal_kind: str) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, training_kind=terminal_kind)
    code, report = fake.run()
    assert code == 2
    assert fake.commands_run[-1] == "scorer_training_and_qualification"
    assert "development_transfer" not in fake.commands_run
    assert not any("development-terminal" in event for event in fake.events)
    assert report["development_transfer_started"] is False
    assert report["nothing_running"] is True


def test_resume_existing_pass_and_development_terminal_short_circuits_all_work(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_training_kind="QUALIFICATION_PASS",
        retained_development=True)
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run == []
    assert fake.deleted == []
    assert report["completed_stages"] == [
        "retained_existing_scorer_training_terminal",
        "retained_existing_development_transfer_terminal",
    ]


def test_resume_completed_smoke_never_repeats_deliberate_deletion(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path, retained_smoke="complete")
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run[:2] == [
        "full_branch_corpus", "full_latent_encoding"]
    assert not any(stage.startswith("smoke") for stage in fake.commands_run)
    assert fake.deleted == []
    assert "retained_completed_smoke_protocol" in report["completed_stages"]


def test_optional_smoke_accepts_only_complete_branch_to_encoder_transition(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    smoke_path = tmp_path / runner.SCORER_FIT_RELATIVE_PATH / (
        runner.BUILDER.SCORER_FIT_V2_ENCODING_SMOKE_RECEIPT_NAME)
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text("{}")
    state_digest = "1" * 64
    assignment_digest = "2" * 64
    contract_digest = "3" * 64
    artifact_digest = "4" * 64
    identities = [f"{index:064x}" for index in range(12)]
    rows = [f"{index + 12:064x}" for index in range(12)]
    branch_smoke = {
        "schema": runner.BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_SCHEMA,
        "status": runner.STATUS,
        "pass": True,
        "state_id": "smoke-state",
        "branch_identity_digests": identities,
        "branch_row_digests": rows,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "scorer_fit_corpus_v2_scorer_contract_digest": contract_digest,
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            artifact_digest,
    }
    branch_smoke["smoke_branch_receipt_digest"] = runner.canonical_digest(
        branch_smoke)
    smoke = {
        "schema": "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1",
        "status": runner.STATUS,
        "base_end_to_end_pass": True,
        "pass": True,
        "candidate_indices": list(range(12)),
        "branch_count": 12,
        "rendered_horizon_frame_count": 48,
        "true_latent_trajectory_count": 12,
        "true_latent_trajectory_shape": [4, 768, 1024],
        "state_id": "smoke-state",
        "branch_identity_digests": identities,
        "branch_row_digests": rows,
        "branch_smoke_receipt_digest": "5" * 64,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "scorer_fit_corpus_v2_scorer_contract_digest": contract_digest,
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            artifact_digest,
        "latent_index_digest": "6" * 64,
        "zero_new_resume_verified": True,
        "single_shard_deletion_regeneration_verified": True,
        "smoke_protocol_complete": True,
    }
    smoke["smoke_receipt_digest"] = runner.canonical_digest(smoke)
    corpus_payload = {
        "state_count": 120,
        "attempted_branch_count": 1_440,
        "valid_branch_count": 1_440,
        "invalid_branch_count": 0,
        "complete": True,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
    }
    corpus = {
        "status": runner.STATUS,
        "complete": True,
        "states": 120,
        "state_count": 120,
        "completed_states": 120,
        "expected_branches": 1_440,
        "attempted_branches": 1_440,
        "attempted_count": 1_440,
        "rows": 1_440,
        "valid_branches": 1_440,
        "valid_count": 1_440,
        "invalid_branches": 0,
        "invalid_count": 0,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "corpus_digest_payload": corpus_payload,
        "corpus_digest": runner.canonical_digest(corpus_payload),
    }

    def fake_load(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
        del label
        if path == smoke_path:
            return dict(smoke), b""
        if path.name == runner.BUILDER.SCORER_FIT_V2_CORPUS_RECEIPT_NAME:
            return dict(corpus), b""
        return dict(branch_smoke), b""

    monkeypatch.setattr(runner, "_load_json", fake_load)
    monkeypatch.setattr(
        runner.BUILDER,
        "load_and_validate_full_bank_v2_manifests_for_consumption",
        lambda **_kwargs: {
            "state_manifest": {"state_manifest_digest": state_digest},
            "assignment_manifest": {
                "full_bank_assignment_manifest_digest": assignment_digest},
        })
    monkeypatch.setattr(
        runner.SCORER_CONTRACT, "load_contract_for_consumption",
        lambda **_kwargs: {
            runner.SCORER_CONTRACT.CONTRACT_SELF_KEY: contract_digest,
            runner.SCORER_CONTRACT.ARTIFACT_SELF_KEY: artifact_digest,
        })
    monkeypatch.setattr(
        runner.BUILDER,
        "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
            "optional resume must tolerate an unrelated missing row")))
    projection = runner._optional_encoding_smoke_projection(root=tmp_path)
    assert projection["terminal_present"] is True
    assert projection["smoke_protocol_complete"] is True
    assert projection["requires_full_encoder_refresh"] is True


def test_resume_deletion_window_repairs_once_without_second_deletion(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke="deletion-window")
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run[0] == "smoke_single_shard_regeneration"
    assert fake.commands_run.count("smoke_single_shard_regeneration") == 1
    assert "branch_smoke" not in fake.commands_run
    assert fake.deleted == []
    assert "resumed_interrupted_smoke_shard_regeneration" in report[
        "completed_stages"]


def test_command_surface_has_no_solver_or_final_eval_and_exact_apply_flag(
        tmp_path: Path) -> None:
    commands = runner.downstream_command_sequence(
        root=tmp_path,
        interpreters={
            "genesis": tmp_path / "genesis-python",
            "rocm": tmp_path / "rocm-python",
        },
        authority=object(),
    )
    assert list(commands) == list(runner._V2_RUNTIME_STAGE_ROLES)
    flattened = " ".join(part for command in commands.values()
                         for part in command).lower()
    for forbidden in ("final_eval", "final-eval", "milp", "cp-sat"):
        assert forbidden not in flattened
    assert commands["development_transfer"][-2:] == [
        "--scorer-corpus-design", "full-bank-v2"]


def test_issue_design_calls_classification_before_amendment(tmp_path: Path) -> None:
    events: list[str] = []

    class FakeDesign:
        MASK_CLASSIFICATION_SELF_KEY = "classification_digest"
        DESIGN_SELF_KEY = "design_digest"

        def issue_rotation_mask_classification(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("classification")
            return {
                "classification_digest": HEX_A,
                "counts": {
                    "old_rotation_related_condition_count": 18,
                    "true_branch_execution_requirement_count": 0,
                },
            }

        def issue_design_amendment(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("design")
            return {
                "design_digest": HEX_B,
                "rotation_mask_classification": {"self_digest": HEX_A},
            }

    report = runner.issue_design(root=tmp_path, design=FakeDesign())
    assert events == ["classification", "design"]
    assert report["rotation_mask_classification_digest"] == HEX_A
    assert report["scorer_fit_corpus_v2_design_digest"] == HEX_B
    assert report["solver_or_optimisation_used"] is False
