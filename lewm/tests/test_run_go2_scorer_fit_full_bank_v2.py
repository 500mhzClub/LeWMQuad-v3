from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

from scripts import run_go2_scorer_fit_full_bank_v2 as runner


HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
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
            runner.DESIGN,
            "ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY",
            "encoder_compute_dtype_correction_digest", raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST", HEX_D,
            raising=False)
        self.dtype_correction = {
            runner.DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY: HEX_C,
            "immutable_encoder_import_correction_digest": HEX_D,
            "immutable_encoder_import_correction": {
                "payload": {
                    runner.DESIGN.ENCODER_IMPORT_CORRECTION_SELF_KEY: HEX_D,
                },
            },
        }
        monkeypatch.setattr(
            runner.DESIGN,
            "load_encoder_compute_dtype_correction_for_consumption",
            lambda **_kwargs: self.events.append(
                "validate:encoder-compute-dtype-correction")
            or self.dtype_correction, raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "load_encoder_import_correction_for_consumption",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
                "old import correction must not be live-loaded")))
        def load_contract(**kwargs: Any) -> dict[str, Any]:
            assert kwargs["encoder_compute_dtype_correction"] \
                is self.dtype_correction
            self.events.append("validate:immutable-scorer-contract")
            return {}

        monkeypatch.setattr(
            runner.SCORER_CONTRACT, "load_contract_for_consumption",
            load_contract)
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
    assert report["encoder_import_correction_digest"] == HEX_D
    assert report[
        runner.DESIGN.ENCODER_IMPORT_CORRECTION_SELF_KEY] == HEX_D
    assert report["encoder_compute_dtype_correction_digest"] == HEX_C
    assert report[
        runner.DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY] == HEX_C
    assert report["predictor_access_before_qualification"] is False
    assert report["final_200_state_corpus_generated"] is False
    assert fake.events.index(
        "validate:encoder-compute-dtype-correction") < fake.events.index(
            "validate:immutable-scorer-contract")
    assert fake.events.index(
        "validate:immutable-scorer-contract") < fake.events.index(
            "command:smoke_encoding")


def test_missing_encoder_compute_dtype_correction_blocks_before_any_command(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner.DESIGN,
        "load_encoder_compute_dtype_correction_for_consumption",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.FullBankV2RunnerError("fixture correction missing")))
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="fixture correction missing"):
        fake.run(resume=True)
    assert fake.commands_run == []
    assert not any(event.startswith("probe:") for event in fake.events)


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
        "encoder_compute_dtype": "float32",
        "encoder_compute_dtype_correction_digest": HEX_C,
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
            "design_authority": {
                "encoder_compute_dtype_correction_digest": HEX_C,
            },
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
    smoke["encoder_compute_dtype_correction_digest"] = HEX_D
    smoke["smoke_receipt_digest"] = runner.canonical_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="encoding smoke receipt changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)
    smoke["encoder_compute_dtype_correction_digest"] = HEX_C
    smoke["encoder_compute_dtype"] = "bfloat16"
    smoke["smoke_receipt_digest"] = runner.canonical_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="encoding smoke receipt changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)


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


def test_issue_manifest_replay_correction_preserves_scientific_lineage(
        tmp_path: Path) -> None:
    events: list[str] = []

    class FakeDesign:
        DESIGN_SELF_KEY = "design_digest"
        SOURCE_CORRECTION_SCHEMA = "fixture_structural_correction_v1"
        SOURCE_CORRECTION_SELF_KEY = "correction_digest"
        IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST = HEX_A
        IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST = HEX_B
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST = HEX_C
        MANIFEST_REPLAY_CORRECTION_SCHEMA = "fixture_manifest_replay_v1"
        MANIFEST_REPLAY_CORRECTION_SELF_KEY = "replay_digest"

        def correction(self) -> dict[str, Any]:
            return {
                "schema": self.SOURCE_CORRECTION_SCHEMA,
                "structural_validation_correction_version": 1,
                "immutable_preselection_source_correction_v2_digest": HEX_B,
                "transitive_immutable_preselection_source_correction_v1_digest":
                    HEX_A,
                "correction_digest": HEX_C,
            }

        def replay_correction(self) -> dict[str, Any]:
            return {
                "schema": self.MANIFEST_REPLAY_CORRECTION_SCHEMA,
                "manifest_replay_correction_version": 1,
                "immutable_active_preselection_source_correction_digest":
                    HEX_C,
                "preserved_scientific_manifest_lineage_digest": HEX_C,
                "replay_digest": HEX_D,
            }

        def issue_manifest_replay_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue-replay-correction")
            return self.replay_correction()

        def load_active_design_authority(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("active-replay")
            return {
                "design_amendment": {"design_digest": HEX_B},
                "source_correction": self.correction(),
                "source_correction_digest": HEX_C,
                "manifest_replay_correction": self.replay_correction(),
                "manifest_replay_correction_digest": HEX_D,
                "candidate_outcomes_consumed": False,
            }

    report = runner.issue_source_correction(
        root=tmp_path, design=FakeDesign())
    assert events == ["issue-replay-correction", "active-replay"]
    assert report["scorer_fit_corpus_v2_design_digest"] == HEX_B
    assert report["immutable_preselection_source_correction_v2_digest"] \
        == HEX_B
    assert report[
        "transitive_immutable_preselection_source_correction_v1_digest"] \
        == HEX_A
    assert report["scorer_fit_corpus_v2_source_correction_digest"] == HEX_C
    assert report["scorer_fit_corpus_v2_manifest_lineage_digest"] == HEX_C
    assert report[
        "scorer_fit_corpus_v2_manifest_replay_correction_digest"] == HEX_D
    assert report["selection_started"] is True
    assert report["selection_already_completed_preoutcome"] is True
    assert report["all_five_preoutcome_manifests_already_installed"] is True
    assert report["manifest_written_or_rewritten"] is False
    assert report["solver_or_optimisation_used"] is False


@pytest.mark.parametrize(
    ("schema", "version_key", "version"),
    [
        ("fixture_preselection_source_correction_v1",
         "source_correction_version", 1),
        ("fixture_preselection_source_correction_v2",
         "source_correction_version", 2),
    ],
)
def test_freeze_manifests_refuses_historical_source_corrections_before_builder(
        tmp_path: Path, schema: str, version_key: str, version: int) -> None:
    class HistoricalDesign:
        SOURCE_CORRECTION_SCHEMA = "fixture_structural_correction_v1"
        SOURCE_CORRECTION_SELF_KEY = "correction_digest"
        IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST = HEX_A
        IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST = HEX_B
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST = HEX_C
        MANIFEST_REPLAY_CORRECTION_SCHEMA = "fixture_manifest_replay_v1"
        MANIFEST_REPLAY_CORRECTION_SELF_KEY = "replay_digest"

        def load_active_design_authority(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            return {
                "source_correction": {
                    "schema": schema,
                    version_key: version,
                    "correction_digest": HEX_C,
                },
                "source_correction_digest": HEX_C,
                "candidate_outcomes_consumed": False,
            }

    class BuilderMustNotRun:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"builder accessed before correction gate: {name}")

    with pytest.raises(
            runner.FullBankV2RunnerError,
            match="final preselection structural-validation correction"):
        runner.freeze_manifests(
            root=tmp_path, builder=BuilderMustNotRun(),
            design_authority=HistoricalDesign())


def test_freeze_manifests_requires_operational_replay_wrapper_before_builder(
        tmp_path: Path) -> None:
    class FinalSourceOnlyDesign:
        SOURCE_CORRECTION_SCHEMA = "fixture_structural_correction_v1"
        SOURCE_CORRECTION_SELF_KEY = "correction_digest"
        IMMUTABLE_SOURCE_CORRECTION_V1_DIGEST = HEX_A
        IMMUTABLE_SOURCE_CORRECTION_V2_DIGEST = HEX_B
        IMMUTABLE_ACTIVE_PRESELECTION_SOURCE_CORRECTION_DIGEST = HEX_C
        MANIFEST_REPLAY_CORRECTION_SCHEMA = "fixture_manifest_replay_v1"
        MANIFEST_REPLAY_CORRECTION_SELF_KEY = "replay_digest"

        def load_active_design_authority(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            return {
                "source_correction": {
                    "schema": self.SOURCE_CORRECTION_SCHEMA,
                    "structural_validation_correction_version": 1,
                    "immutable_preselection_source_correction_v2_digest":
                        HEX_B,
                    "transitive_immutable_preselection_source_correction_v1_digest":
                        HEX_A,
                    "correction_digest": HEX_C,
                },
                "source_correction_digest": HEX_C,
                "candidate_outcomes_consumed": False,
            }

    class BuilderMustNotRun:
        def __getattr__(self, name: str) -> Any:
            raise AssertionError(f"builder accessed before replay gate: {name}")

    with pytest.raises(
            runner.FullBankV2RunnerError,
            match="post-install manifest-replay correction"):
        runner.freeze_manifests(
            root=tmp_path, builder=BuilderMustNotRun(),
            design_authority=FinalSourceOnlyDesign())


def test_exact_existing_five_manifest_payloads_are_never_rewritten(
        tmp_path: Path) -> None:
    paths = [tmp_path / f"manifest-{index}.json" for index in range(5)]
    payloads = [{"artifact_index": index} for index in range(5)]
    before: list[tuple[int, int, int, int, bytes]] = []
    for path, payload in zip(paths, payloads, strict=True):
        path.write_bytes(runner._json_bytes(payload, pretty=True))
        path.chmod(0o444)
        observed = path.stat()
        before.append((
            observed.st_ino, observed.st_mtime_ns, observed.st_ctime_ns,
            observed.st_mode, path.read_bytes()))

    reopened = [
        runner._install_or_require_exact_json(
            path, payload, label=f"synthetic manifest {index}")
        for index, (path, payload) in enumerate(
            zip(paths, payloads, strict=True))
    ]

    assert reopened == payloads
    after = []
    for path in paths:
        observed = path.stat()
        after.append((
            observed.st_ino, observed.st_mtime_ns, observed.st_ctime_ns,
            observed.st_mode, path.read_bytes()))
    assert after == before


def test_parser_exposes_source_correction_before_manifest_stage() -> None:
    correction = runner._parser().parse_args(
        ["--stage", "issue-source-correction"])
    manifests = runner._parser().parse_args(
        ["--stage", "freeze-manifests"])
    assert correction.stage == "issue-source-correction"
    assert manifests.stage == "freeze-manifests"


def test_issue_scorer_contract_does_not_require_later_runtime_correction(
        tmp_path: Path) -> None:
    events: list[str] = []
    artifact = {
        "contract_digest": HEX_C,
        "artifact_digest": HEX_D,
    }

    class FakeContract:
        CONTRACT_SELF_KEY = "contract_digest"
        ARTIFACT_SELF_KEY = "artifact_digest"

        def issue_contract(self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue")
            return artifact

        def validate_contract_artifact(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("validate-issued")
            return dict(value)

        def contract_artifact_binding(
                self, value: Mapping[str, Any], *, root: Path,
                ) -> Mapping[str, Any]:
            assert root == tmp_path and value == artifact
            events.append("reopen-issued-bytes")
            return {"self_digest": HEX_D}

        def load_contract_for_consumption(self, **_kwargs: Any) -> None:
            raise AssertionError(
                "historical issuance must not require dtype correction")

    report = runner.issue_scorer_contract(
        root=tmp_path, contract_authority=FakeContract())
    assert events == ["issue", "validate-issued", "reopen-issued-bytes"]
    assert report["scorer_fit_corpus_v2_scorer_contract_digest"] == HEX_C
    assert report["contract_artifact_digest"] == HEX_D


def test_issue_encoder_import_correction_preserves_issued_contract(
        tmp_path: Path) -> None:
    events: list[str] = []
    immutable = {
        "self_digest":
            runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
        "embedded_contract_self_digest":
            runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
    }
    payload = {
        "immutable_successor_scorer_contract_binding": immutable,
        "encoder_import_correction_digest": HEX_D,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_IMPORT_CORRECTION_STATUS = "ISSUED_FIXTURE_CORRECTION"

        def issue_encoder_import_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue")
            return payload

        def load_encoder_import_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("reopen")
            return payload

    report = runner.issue_encoder_import_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == ["issue", "reopen"]
    assert report["scorer_fit_corpus_v2_encoder_import_correction_digest"] \
        == HEX_D
    assert report["scorer_contract_reissued_or_rewritten"] is False
    assert report["preoutcome_manifests_reissued_or_rewritten"] is False
    assert report["replayed_from_immutable_dtype_correction_lineage"] is False


def test_old_import_issue_stage_replays_nested_immutable_at_current_source(
        tmp_path: Path) -> None:
    events: list[str] = []
    old = {
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
        "encoder_import_correction_digest": HEX_D,
    }
    immutable = {"payload": old, "binding": {"self_digest": HEX_D}}
    dtype = {
        "immutable_encoder_import_correction": immutable,
        "immutable_encoder_import_correction_digest": HEX_D,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_IMPORT_CORRECTION_STATUS = "ISSUED_FIXTURE_CORRECTION"
        ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH = (
            runner.DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH)

        def load_encoder_compute_dtype_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("load-dtype")
            return dtype

        def validate_immutable_encoder_import_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert value == immutable
            events.append("validate-immutable-import")
            return immutable

        def issue_encoder_import_correction(self, **_kwargs: Any) -> None:
            raise AssertionError("immutable import correction was reissued")

        def load_encoder_import_correction_for_consumption(
                self, **_kwargs: Any) -> None:
            raise AssertionError(
                "historical import source was live-reinterpreted")

    dtype_path = tmp_path / (
        FakeDesign.ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH)
    dtype_path.parent.mkdir(parents=True)
    dtype_path.write_text("installed fixture marker")
    report = runner.issue_encoder_import_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == ["load-dtype", "validate-immutable-import"]
    assert report["scorer_fit_corpus_v2_encoder_import_correction_digest"] \
        == HEX_D
    assert report["replayed_from_immutable_dtype_correction_lineage"] is True
    assert report["scorer_contract_reissued_or_rewritten"] is False


def test_parser_exposes_encoder_import_correction_before_resume_run() -> None:
    correction = runner._parser().parse_args(
        ["--stage", "issue-encoder-import-correction"])
    resumed = runner._parser().parse_args(
        ["--stage", "run", "--resume"])
    assert correction.stage == "issue-encoder-import-correction"
    assert resumed.stage == "run" and resumed.resume is True


def test_issue_encoder_compute_dtype_correction_preserves_predecessors(
        tmp_path: Path) -> None:
    events: list[str] = []
    payload = {
        "immutable_encoder_import_correction": {
            "payload": {"encoder_import_correction_digest": HEX_D},
            "binding": {"self_digest": HEX_D},
        },
        "immutable_encoder_import_correction_digest": HEX_D,
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
        "encoder_compute_dtype_correction_digest": HEX_C,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS = "ISSUED_FIXTURE_DTYPE"

        def issue_encoder_compute_dtype_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue-dtype")
            return payload

        def load_encoder_compute_dtype_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("reopen-dtype")
            return payload

    report = runner.issue_encoder_compute_dtype_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == ["issue-dtype", "reopen-dtype"]
    assert report["encoder_compute_dtype_correction_digest"] == HEX_C
    assert report["encoder_import_correction_digest"] == HEX_D
    assert report["encoder_import_correction_reissued_or_rewritten"] is False
    assert report["scorer_contract_reissued_or_rewritten"] is False
    assert report["preoutcome_manifests_reissued_or_rewritten"] is False
    assert report["branch_latent_or_scorer_runtime_started_by_issue_stage"] \
        is False


def test_parser_exposes_dtype_correction_before_resume_run() -> None:
    correction = runner._parser().parse_args(
        ["--stage", "issue-encoder-compute-dtype-correction"])
    resumed = runner._parser().parse_args(
        ["--stage", "run", "--resume"])
    assert correction.stage == "issue-encoder-compute-dtype-correction"
    assert resumed.stage == "run" and resumed.resume is True


def test_preoutcome_failure_binds_corrected_source_and_immutable_design() -> None:
    authority = {
        "design_amendment": {
            runner.DESIGN.DESIGN_SELF_KEY: HEX_A,
            "source_repository_commit": "1" * 40,
        },
        "rotation_mask_classification": {
            runner.DESIGN.MASK_CLASSIFICATION_SELF_KEY: HEX_B,
        },
        "source_correction_digest": HEX_C,
        "active_source_repository_commit": "2" * 40,
    }
    failure = SimpleNamespace(
        ordered_scene_ids=[f"scene-{index:02d}" for index in range(17)],
        fit_count=3,
        calibration_count=1,
        reason="fixture full-bank insufficiency",
    )
    receipt = runner._build_feasibility_failure(
        failure, authority=authority)
    assert receipt["source_repository_commit"] == "2" * 40
    assert receipt["scorer_fit_corpus_v2_design_digest"] == HEX_A
    assert receipt["scorer_fit_corpus_v2_source_correction_digest"] == HEX_C
    assert runner._validate_feasibility_failure(
        receipt, authority=authority) == receipt
