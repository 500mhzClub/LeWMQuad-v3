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
HEX_E = "e" * 64
TARGET = {
    "path": (".generated/go2_branch_corpus_v1_2/scorer_fit/"
             "latents_v2/horizon/smoke-candidate-0.f16"),
    "sha256": HEX_A,
    "byte_count": 4 * 768 * 1024 * 2,
    "shape": [4, 768, 1024],
}


def _transaction(
        state: str, *, unstarted_target: str = "NOT_APPLICABLE",
        ) -> dict[str, Any]:
    rows = {
        "UNSTARTED": (
            False, None, "NOT_APPLICABLE", "ABSENT", False, None,
            "ABSENT_OR_PRETRANSACTION",
            "RUN_OR_RESUME_BASE_AND_ZERO_NEW_BEFORE_PREPARED"),
        "PREPARED_MOVE_PENDING": (
            True, HEX_A, "EXACT", "ABSENT", False, None,
            "ABSENT_OR_PRETRANSACTION", "ATOMIC_MOVE_ONCE"),
        "MOVED_REGENERATION_PENDING": (
            True, HEX_A, "ABSENT", "EXACT", False, None,
            "ABSENT_OR_PRETRANSACTION", "RUN_REGENERATION_ENCODER_ONCE"),
        "RESTORED_COMPLETE_PENDING": (
            True, HEX_A, "EXACT", "EXACT", False, None,
            "ABSENT_OR_PRETRANSACTION",
            "CREATE_COMPLETE_WITHOUT_SECOND_MOVE_OR_REGENERATION"),
        "COMPLETE_SMOKE_PUBLICATION_PENDING": (
            True, HEX_A, "EXACT", "EXACT", True, HEX_B,
            "ABSENT_OR_PRETRANSACTION",
            "PUBLISH_COMPLETE_BOUND_PASS_SMOKE_ONLY"),
        "COMPLETE": (
            True, HEX_A, "EXACT", "EXACT", True, HEX_B,
            "EXACT_BOUND_PROTOCOL_PASS", "NO_TRANSACTION_MUTATION"),
    }
    (prepared, prepared_digest, target, backup, complete, complete_digest,
     pass_smoke, next_action) = rows[state]
    value = {
        "transaction_state": state,
        "prepared_present": prepared,
        "prepared_receipt_digest": prepared_digest,
        "target_state": target,
        "backup_state": backup,
        "complete_present": complete,
        "complete_receipt_digest": complete_digest,
        "pass_smoke_state": pass_smoke,
        "next_action": next_action,
        "encoder_path_projection_correction_digest": HEX_E,
        "single_shard_regeneration_transaction_contract_digest": HEX_C,
        "prepared_staged_state": "ABSENT",
        "complete_staged_state": "ABSENT",
        "target_exact": target == "EXACT",
        "backup_exact": backup == "EXACT",
        "target_backup_custody_exact": backup == "EXACT",
        "regenerated_target_custody_exact": state in {
            "RESTORED_COMPLETE_PENDING",
            "COMPLETE_SMOKE_PUBLICATION_PENDING", "COMPLETE"},
        "candidate_outcomes_used_for_selection": False,
        "final_200_state_corpus_generated": False,
    }
    if state == "UNSTARTED":
        value["target_state"] = unstarted_target
        value["target_exact"] = unstarted_target == "EXACT"
    return value


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
        encoder_path_projection_correction_digest=HEX_E,
        registered_smoke_shard_inventory_digest=HEX_A,
        registered_smoke_stable_artifact_inventory_digest=HEX_B,
        invocation_new_context_shards=new_context,
        invocation_new_horizon_shards=new_horizon,
        zero_new_resume_verified=zero_new,
        single_registered_shard_regenerated=regenerated,
        only_registered_missing_shard_changed=regenerated,
        single_shard_regeneration_target=dict(TARGET),
        single_shard_regeneration_transaction_state=(
            "COMPLETE" if regenerated else "NONE"),
        single_shard_regeneration_transaction_complete=regenerated,
        single_shard_regeneration_prepared_digest=(
            HEX_A if regenerated else None),
        single_shard_regeneration_complete_digest=(
            HEX_B if regenerated else None),
        single_shard_regeneration_target_exact=True,
        single_shard_regeneration_backup_exact=regenerated,
    )


class FakePipeline:
    def __init__(
            self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *,
            training_kind: str = "QUALIFICATION_PASS",
            retained_training_kind: str | None = None,
            retained_development: bool = False,
            retained_smoke: str = "absent",
            retained_transaction_state: str | None = None,
            ) -> None:
        self.events: list[str] = []
        self.commands_run: list[str] = []
        self.validation_counts: dict[str, int] = {}
        self.training_kind = training_kind
        self.retained_training_kind = retained_training_kind
        self.retained_development = retained_development
        self.retained_smoke = retained_smoke
        self.retained_transaction_state = (
            retained_transaction_state
            if retained_transaction_state is not None else (
                "COMPLETE" if retained_smoke in {
                    "complete", "partial-corpus-receipt-lag",
                    "complete-corpus-receipt-lag"}
                else "UNSTARTED"))
        self.transaction_resume_origin = self.retained_transaction_state
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
            "ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY",
            "scorer_fit_corpus_v2_encoder_path_projection_correction_digest",
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY",
            "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest",
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY",
            "scorer_fit_corpus_v2_optional_smoke_partial_corpus_resume_"
            "correction_digest",
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST", HEX_C,
            raising=False)
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
        self.path_correction = {
            runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY: HEX_E,
            "single_shard_regeneration_transaction_contract_digest": HEX_C,
            "immutable_encoder_compute_dtype_correction_digest": HEX_C,
            "immutable_encoder_compute_dtype_correction": {
                "payload": self.dtype_correction,
                "binding": {"self_digest": HEX_C},
            },
        }
        self.branch_redrive_correction = {
            runner.DESIGN.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY: HEX_B,
            "immutable_encoder_path_projection_correction_digest": HEX_E,
            "immutable_encoder_path_projection_correction": {
                "payload": self.path_correction,
                "binding": {"self_digest": HEX_E},
            },
        }
        self.partial_resume_correction = {
            runner.DESIGN.
            OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY: HEX_A,
            "immutable_branch_redrive_projection_correction_digest": HEX_B,
            "immutable_branch_redrive_projection_correction": {
                "payload": self.branch_redrive_correction,
                "binding": {
                    "self_digest_key": runner.DESIGN.
                        BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY,
                    "self_digest": HEX_B,
                },
            },
        }
        monkeypatch.setattr(
            runner.DESIGN,
            "load_optional_smoke_partial_corpus_resume_correction_for_"
            "consumption",
            lambda **_kwargs: self.events.append(
                "validate:partial-corpus-resume-correction")
            or self.partial_resume_correction, raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "validate_immutable_branch_redrive_projection_correction",
            lambda value, **_kwargs: dict(value), raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "load_branch_redrive_projection_correction_for_consumption",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
                "historical redrive correction must not be live-loaded")),
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "load_encoder_path_projection_correction_for_consumption",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
                "historical path correction must not be live-loaded")),
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "validate_immutable_encoder_path_projection_correction",
            lambda value, **_kwargs: dict(value), raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "validate_immutable_encoder_compute_dtype_correction",
            lambda value: dict(value), raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "load_encoder_compute_dtype_correction_for_consumption",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
                "old dtype correction must not be live-loaded")),
            raising=False)
        monkeypatch.setattr(
            runner.DESIGN,
            "load_encoder_import_correction_for_consumption",
            lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
                "old import correction must not be live-loaded")))
        def load_contract(**kwargs: Any) -> dict[str, Any]:
            assert kwargs["encoder_path_projection_correction"] \
                == self.path_correction
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
        if stage == "smoke_single_shard_regeneration":
            self.transaction_resume_origin = self.retained_transaction_state
            self.retained_transaction_state = "COMPLETE"
        if (stage == "scorer_training_and_qualification"
                and self.training_kind != "QUALIFICATION_PASS"):
            return 1
        return 0

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
            if (self.commands_run
                    and self.commands_run[-1]
                    == "smoke_single_shard_regeneration"):
                return _encoding(
                    kind, new_context=0,
                    # The strict consumer returns immutable transaction
                    # evidence, not the current recovery invocation count.
                    new_horizon=1,
                    zero_new=True, regenerated=True)
            if (self.retained_smoke in {
                    "complete", "complete-corpus-receipt-lag"}
                    and self.retained_transaction_state == "COMPLETE"):
                return _encoding(
                    kind, new_context=0, new_horizon=1,
                    zero_new=True, regenerated=True)
            if (self.retained_smoke != "absent"
                    and self.retained_transaction_state == "UNSTARTED"):
                return _encoding(
                    kind, new_context=0, new_horizon=0,
                    zero_new=True, regenerated=False)
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
                return _closed(
                    kind, terminal_present=False,
                    **_transaction(self.retained_transaction_state))
            complete = self.retained_smoke in {
                "complete", "partial-corpus-receipt-lag",
                "complete-corpus-receipt-lag"}
            zero_new = self.retained_smoke != "base-only"
            return _closed(
                kind, terminal_present=True,
                smoke_protocol_complete=complete,
                zero_new_resume_verified=zero_new,
                single_registered_shard_regenerated=complete,
                only_registered_missing_shard_changed=complete,
                requires_full_encoder_refresh=
                    self.retained_smoke in {
                        "partial-corpus-receipt-lag",
                        "complete-corpus-receipt-lag"},
                **_transaction(
                    self.retained_transaction_state,
                    unstarted_target="EXACT"))
        if kind == "branch-corpus":
            return _closed(kind, state_count=120, branch_count=1_440)
        if kind == "encoded-corpus":
            projection = _encoding(
                kind, new_context=0, new_horizon=0,
                zero_new=True, regenerated=True)
            projection.update({
                "state_count": 120,
                "horizon_latent_count": 1_440,
            })
            return projection
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
    assert fake.events.index("validate:training-terminal") < fake.events.index(
        "command:development_transfer")
    assert report["qualified"] is True
    assert report["encoder_import_correction_digest"] == HEX_D
    assert report[
        runner.DESIGN.ENCODER_IMPORT_CORRECTION_SELF_KEY] == HEX_D
    assert report["encoder_compute_dtype_correction_digest"] == HEX_C
    assert report[
        runner.DESIGN.ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY] == HEX_C
    assert report["encoder_path_projection_correction_digest"] == HEX_E
    assert report[
        runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY] == HEX_E
    assert report["branch_redrive_projection_correction_digest"] == HEX_B
    assert report[
        runner.DESIGN.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY] == HEX_B
    assert report[
        "optional_smoke_partial_corpus_resume_correction_digest"] == HEX_A
    assert report[
        runner.DESIGN.
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY] == HEX_A
    assert report["predictor_access_before_qualification"] is False
    assert report["final_200_state_corpus_generated"] is False
    assert fake.events.index(
        "validate:partial-corpus-resume-correction") < fake.events.index(
            "validate:immutable-scorer-contract")
    assert fake.events.index(
        "validate:immutable-scorer-contract") < fake.events.index(
            "command:smoke_encoding")


def test_missing_partial_corpus_resume_correction_blocks_before_any_command(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner.DESIGN,
        "load_optional_smoke_partial_corpus_resume_correction_for_consumption",
        lambda **_kwargs: (_ for _ in ()).throw(
            runner.FullBankV2RunnerError("fixture correction missing")))
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="fixture correction missing"):
        fake.run(resume=True)
    assert fake.commands_run == []
    assert not any(event.startswith("probe:") for event in fake.events)


def test_strict_encoder_projection_must_bind_the_loaded_path_correction(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path)
    original = fake.validate

    def changed(kind: str, root: Path,
                interpreter: Path) -> Mapping[str, Any]:
        value = dict(original(kind, root, interpreter))
        if kind == "encoding-smoke":
            value["encoder_path_projection_correction_digest"] = HEX_D
        return value

    with pytest.raises(runner.FullBankV2RunnerError,
                       match="another path-projection correction"):
        runner.run_pipeline(
            root=fake.root,
            command_runner=fake.command,
            runtime_probe_invoker=fake.runtime_probe,
            validation_invoker=changed,
            authority=object(),
        )
    assert fake.commands_run == [
        "branch_smoke", "branch_smoke_zero_new", "smoke_encoding"]


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
    assert fake.validation_counts["encoding-smoke"] == 1
    assert "retained_strict_completed_smoke_protocol" in report[
        "completed_stages"]
    assert "retained_completed_smoke_protocol" in report["completed_stages"]


def test_complete_smoke_with_complete_corpus_receipt_lag_skips_smoke_recovery(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path,
        retained_smoke="complete-corpus-receipt-lag")
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run[:2] == [
        "full_branch_corpus", "full_latent_encoding"]
    assert not any(stage.startswith("smoke") for stage in fake.commands_run)
    # The exact full-index-ahead/old-smoke crash window cannot pass the strict
    # active-index consumer until full encoding publishes its refreshed smoke.
    assert fake.validation_counts.get("encoding-smoke", 0) == 0
    assert "retained_strict_completed_smoke_protocol" not in report[
        "completed_stages"]
    assert "retained_completed_smoke_protocol" in report["completed_stages"]


def test_complete_smoke_with_partial_corpus_lag_resumes_missing_branches_first(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path,
        retained_smoke="partial-corpus-receipt-lag")
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run[:2] == [
        "full_branch_corpus", "full_latent_encoding"]
    assert not any(stage.startswith("smoke") for stage in fake.commands_run)
    assert fake.validation_counts.get("encoding-smoke", 0) == 0
    assert "retained_strict_completed_smoke_protocol" not in report[
        "completed_stages"]
    assert "retained_completed_smoke_protocol" in report["completed_stages"]


def test_resume_normal_complete_requires_strict_live_smoke_lineage(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(monkeypatch, tmp_path, retained_smoke="complete")
    original = fake.validate

    def stale(kind: str, root: Path,
              interpreter: Path) -> Mapping[str, Any]:
        projection = dict(original(kind, root, interpreter))
        if kind == "encoding-smoke":
            projection["single_shard_regeneration_prepared_digest"] = None
        return projection

    with pytest.raises(runner.FullBankV2RunnerError,
                       match="exact COMPLETE transaction proof"):
        runner.run_pipeline(
            root=fake.root, command_runner=fake.command,
            runtime_probe_invoker=fake.runtime_probe,
            validation_invoker=stale, authority=object(), resume=True)
    assert "full_branch_corpus" not in fake.commands_run


def test_optional_smoke_accepts_complete_corpus_lag_and_index_ahead(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    transaction = _transaction("COMPLETE")
    monkeypatch.setattr(
        runner, "_optional_smoke_transaction_status",
        lambda **_kwargs: dict(transaction))
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
    branch_smoke["smoke_branch_receipt_digest"] = (
        runner._builder_default_json_digest(branch_smoke))
    current_index = {
        "schema": "go2_scorer_fit_corpus_v2_latents_index_v1",
        "encoder_path_projection_correction_digest": HEX_E,
    }
    current_index["latents_index_digest"] = (
        runner._encoder_default_json_digest(current_index))
    (smoke_path.parent / "latents_index_v2.json").write_bytes(
        runner._encoder_pretty_json_bytes(current_index))
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
        "encoder_path_projection_correction_digest": HEX_E,
        "single_shard_regeneration_transaction_contract_digest": HEX_C,
        "single_shard_regeneration_prepared_digest": HEX_A,
        "single_shard_regeneration_transaction_complete": True,
        "state_id": "smoke-state",
        "branch_identity_digests": identities,
        "branch_row_digests": rows,
        "branch_smoke_receipt_digest": "5" * 64,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "scorer_fit_corpus_v2_scorer_contract_digest": contract_digest,
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            artifact_digest,
        "latent_index_digest": current_index["latents_index_digest"],
        "zero_new_resume_verified": True,
        "single_shard_deletion_regeneration_verified": True,
        "smoke_protocol_complete": True,
    }
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest(smoke)
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
        "corpus_digest": runner._builder_default_json_digest(corpus_payload),
        "branch_rows_sha256": "6" * 64,
    }

    def fake_load(path: Path, *, label: str) -> tuple[dict[str, Any], bytes]:
        del label
        if path == smoke_path:
            return dict(smoke), runner._encoder_pretty_json_bytes(smoke)
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
                "encoder_path_projection_correction_digest": HEX_E,
                "encoder_path_projection_correction": {
                    runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY:
                        HEX_E,
                    "single_shard_regeneration_transaction_contract_digest":
                        HEX_C,
                },
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

    # The immutable protocol PASS is created before COMPLETE and therefore
    # binds only the transaction contract plus PREPARED.  A later refreshed
    # PASS is the distinct state that must also bind COMPLETE.
    transaction["pass_smoke_state"] = (
        "VALID_REFRESHED_PASS_WITH_EXACT_PROTOCOL_PASS_ARCHIVE")
    smoke["single_shard_regeneration_complete_digest"] = HEX_B
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    refreshed_projection = runner._optional_encoding_smoke_projection(
        root=tmp_path)
    assert refreshed_projection["smoke_protocol_complete"] is True
    smoke.pop("single_shard_regeneration_complete_digest")
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="encoding smoke receipt changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)
    transaction["pass_smoke_state"] = "EXACT_BOUND_PROTOCOL_PASS"

    # A normal full-encoding invocation writes the complete index before its
    # refreshed smoke receipt.  A crash between those two atomic replaces is
    # recoverable only when the index is the exact current, complete corpus
    # successor and the still-complete smoke retains its prior index binding.
    complete_index = {
        "schema": "go2_scorer_fit_corpus_v2_latents_index_v1",
        "status": runner.STATUS,
        "pool": "scorer_fit_v2",
        "corpus_design": "full-bank-v2",
        "complete": True,
        "encoder_compute_dtype": "float32",
        "encoder_compute_dtype_correction_digest": HEX_C,
        "encoder_path_projection_correction_digest": HEX_E,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "scorer_fit_corpus_v2_scorer_contract_digest": contract_digest,
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            artifact_digest,
        "corpus_digest": corpus["corpus_digest"],
        "branch_rows_sha256": corpus["branch_rows_sha256"],
        "context_shape": [120, 3, 768, 1024],
        "horizon_shape": [1_440, 4, 768, 1024],
        "context_records": [{} for _ in range(120)],
        "horizon_records": [{} for _ in range(1_440)],
    }
    complete_index["latents_index_digest"] = (
        runner._encoder_default_json_digest(complete_index))
    assert (complete_index["latents_index_digest"]
            != smoke["latent_index_digest"])
    (smoke_path.parent / "latents_index_v2.json").write_bytes(
        runner._encoder_pretty_json_bytes(complete_index))
    projection = runner._optional_encoding_smoke_projection(root=tmp_path)
    assert projection["terminal_present"] is True
    assert projection["smoke_protocol_complete"] is True
    assert projection["requires_full_encoder_refresh"] is True

    # A strict builder-validated, state-aligned partial corpus may also have
    # advanced beyond the immutable one-state smoke receipt.  It is retained
    # for missing-only branch resume and requests the later full-encoder
    # refresh; it does not replay the completed smoke protocol.
    (smoke_path.parent / "latents_index_v2.json").write_bytes(
        runner._encoder_pretty_json_bytes(current_index))
    complete_corpus = dict(corpus)
    complete_branch_smoke = dict(branch_smoke)
    complete_smoke_branch_digest = smoke["branch_smoke_receipt_digest"]
    partial_rows = [
        {
            "state_id": f"state-{state_index}",
            "candidate_index": candidate_index,
            "valid": True,
        }
        for state_index in range(10)
        for candidate_index in range(12)
    ]
    partial_payload = {
        "state_count": 120,
        "attempted_branch_count": 120,
        "valid_branch_count": 120,
        "invalid_branch_count": 0,
        "complete": False,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
    }
    corpus.clear()
    corpus.update({
        "status": runner.STATUS,
        "complete": False,
        "states": 120,
        "state_count": 120,
        "completed_states": 10,
        "expected_branches": 1_440,
        "attempted_branches": 120,
        "attempted_count": 120,
        "rows": 120,
        "valid_branches": 120,
        "valid_count": 120,
        "invalid_branches": 0,
        "invalid_count": 0,
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "corpus_digest_payload": partial_payload,
        "corpus_digest": runner._builder_default_json_digest(partial_payload),
        "branch_rows_sha256": "7" * 64,
    })
    smoke["branch_smoke_receipt_digest"] = branch_smoke[
        "smoke_branch_receipt_digest"]
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    monkeypatch.setattr(
        runner.BUILDER,
        "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
        lambda **_kwargs: {
            "receipt": dict(corpus),
            "rows": [dict(row) for row in partial_rows],
            "branch_smoke": dict(branch_smoke),
        })
    partial_projection = runner._optional_encoding_smoke_projection(
        root=tmp_path)
    assert partial_projection["terminal_present"] is True
    assert partial_projection["smoke_protocol_complete"] is True
    assert partial_projection["requires_full_encoder_refresh"] is True

    corpus["completed_states"] = 9
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="partial-corpus resume proof changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)

    corpus.clear()
    corpus.update(complete_corpus)
    branch_smoke.clear()
    branch_smoke.update(complete_branch_smoke)
    smoke["branch_smoke_receipt_digest"] = complete_smoke_branch_digest
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    monkeypatch.setattr(
        runner.BUILDER,
        "load_and_validate_full_bank_v2_branch_outputs_for_consumption",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError(
            "complete-corpus recovery must not require all rows")))

    # Restore the ordinary current/current pair for the lineage mutations
    # below so each rejection continues to isolate the field under test.
    (smoke_path.parent / "latents_index_v2.json").write_bytes(
        runner._encoder_pretty_json_bytes(current_index))
    smoke["encoder_compute_dtype_correction_digest"] = HEX_D
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="encoding smoke receipt changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)
    smoke["encoder_compute_dtype_correction_digest"] = HEX_C
    smoke["encoder_path_projection_correction_digest"] = HEX_D
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="smoke authority projection changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)
    smoke["encoder_path_projection_correction_digest"] = HEX_E
    smoke["encoder_compute_dtype"] = "bfloat16"
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest({
        key: value for key, value in smoke.items()
        if key != "smoke_receipt_digest"
    })
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="encoding smoke receipt changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)


def test_exact_pre_refresh_base_metadata_is_the_only_digestless_migration(
    tmp_path: Path) -> None:
    metadata = tmp_path / runner.SCORER_FIT_RELATIVE_PATH
    metadata.mkdir(parents=True)
    index = {"schema": "fixture-index", "fixture": "index"}
    index["latents_index_digest"] = runner._encoder_default_json_digest(index)
    smoke = {
        "schema": "fixture-smoke",
        "base_end_to_end_pass": True,
        "latent_index_digest": index["latents_index_digest"],
    }
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest(smoke)
    summary = {"schema": "fixture-summary", "fixture": "summary"}
    smoke_raw = runner._encoder_pretty_json_bytes(smoke)
    index_raw = runner._encoder_pretty_json_bytes(index)
    summary_raw = runner._encoder_pretty_json_bytes(summary)
    smoke_path = metadata / (
        runner.BUILDER.SCORER_FIT_V2_ENCODING_SMOKE_RECEIPT_NAME)
    index_path = metadata / "latents_index_v2.json"
    summary_path = metadata / "encoding_invocation_summary_v2.json"
    smoke_path.write_bytes(smoke_raw)
    index_path.write_bytes(index_raw)
    summary_path.write_bytes(summary_raw)

    def binding(
            path: Path, raw: bytes, *, schema: str,
            self_key: str | None = None, self_digest: str | None = None,
            ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": str(path.relative_to(tmp_path)),
            "schema": schema,
            "raw_sha256": runner.hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
        if self_key is not None:
            value.update({
                "self_digest_key": self_key,
                "self_digest": self_digest,
            })
        return value

    bundle = {
        "base_smoke_receipt_binding": binding(
            smoke_path, smoke_raw, schema=smoke["schema"],
            self_key="smoke_receipt_digest",
            self_digest=smoke["smoke_receipt_digest"]),
        "latent_index_binding": binding(
            index_path, index_raw, schema=index["schema"],
            self_key="latents_index_digest",
            self_digest=index["latents_index_digest"]),
        "encoding_invocation_summary_binding": binding(
            summary_path, summary_raw, schema=summary["schema"]),
    }
    correction = {
        runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY: HEX_E,
        "immutable_base_smoke_artifact_bundle": bundle,
        "base_smoke_artifact_bundle_digest":
            runner.DESIGN.canonical_digest(bundle),
    }
    assert runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)

    changed = dict(smoke)
    changed["encoder_path_projection_correction_digest"] = HEX_E
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=changed, smoke_raw=smoke_raw,
        correction=correction)

    migrated = {
        **{key: value for key, value in index.items()
           if key != "latents_index_digest"},
        "encoder_path_projection_correction_digest": HEX_E,
    }
    migrated["latents_index_digest"] = (
        runner._encoder_default_json_digest(migrated))
    index_path.write_bytes(runner._encoder_pretty_json_bytes(migrated))
    assert runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)

    compact_migrated_raw = runner.json.dumps(
        migrated, sort_keys=True).encode("utf-8")
    index_path.write_bytes(compact_migrated_raw)
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)
    pretty_migrated_raw = runner._encoder_pretty_json_bytes(migrated)
    duplicate_key_raw = (
        ('{\n  "encoder_path_projection_correction_digest": "'
         + HEX_E + '",').encode("utf-8") + pretty_migrated_raw[1:])
    index_path.write_bytes(duplicate_key_raw)
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)

    index_path.write_bytes(pretty_migrated_raw)
    migrated["unregistered"] = True
    migrated["latents_index_digest"] = (
        runner._encoder_default_json_digest({
            key: value for key, value in migrated.items()
            if key != "latents_index_digest"}))
    index_path.write_bytes(runner._encoder_pretty_json_bytes(migrated))
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)

    index_path.write_bytes(index_raw)
    extra_binding = runner.json.loads(runner.json.dumps(correction))
    extra_bundle = extra_binding["immutable_base_smoke_artifact_bundle"]
    extra_bundle["latent_index_binding"]["unregistered"] = True
    extra_binding["base_smoke_artifact_bundle_digest"] = (
        runner.DESIGN.canonical_digest(extra_bundle))
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=extra_binding)

    summary_path.write_bytes(b'{"fixture":"changed"}\n')
    assert not runner._matches_immutable_pre_path_projection_base_smoke(
        root=tmp_path, smoke=smoke, smoke_raw=smoke_raw,
        correction=correction)


def test_optional_metadata_digest_dialects_are_not_interchangeable() -> None:
    payload = {"unicode": "λ", "nested": {"value": 1}}
    digests = {
        runner.canonical_digest(payload),
        runner._encoder_default_json_digest(payload),
        runner._builder_default_json_digest(payload),
    }
    assert len(digests) == 3


@pytest.mark.parametrize(
    "state",
    [
        "UNSTARTED", "PREPARED_MOVE_PENDING",
        "MOVED_REGENERATION_PENDING", "RESTORED_COMPLETE_PENDING",
        "COMPLETE_SMOKE_PUBLICATION_PENDING", "COMPLETE",
    ],
)
def test_optional_transaction_projection_accepts_only_authorised_states(
        state: str) -> None:
    projection = _transaction(state)
    assert runner._normalise_optional_smoke_transaction_status(
        projection) == projection
    changed = dict(projection)
    changed["target_backup_custody_exact"] = not changed[
        "target_backup_custody_exact"]
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="not an authorised crash state"):
        runner._normalise_optional_smoke_transaction_status(changed)


@pytest.mark.parametrize("staged_state", ["EXACT", "PARTIAL_REGULAR"])
def test_optional_transaction_projection_routes_staged_receipt_recovery(
        staged_state: str) -> None:
    prepared = _transaction("UNSTARTED")
    prepared["prepared_staged_state"] = staged_state
    assert runner._normalise_optional_smoke_transaction_status(
        prepared)["transaction_state"] == "UNSTARTED"

    complete = _transaction("RESTORED_COMPLETE_PENDING")
    complete["complete_staged_state"] = staged_state
    assert runner._normalise_optional_smoke_transaction_status(
        complete)["transaction_state"] == "RESTORED_COMPLETE_PENDING"


def test_optional_transaction_accepts_only_exact_staged_publication_residue(
        ) -> None:
    prepared = _transaction("PREPARED_MOVE_PENDING")
    prepared["prepared_staged_state"] = "EXACT"
    assert runner._normalise_optional_smoke_transaction_status(
        prepared)["transaction_state"] == "PREPARED_MOVE_PENDING"
    prepared["prepared_staged_state"] = "PARTIAL_REGULAR"
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="not an authorised crash state"):
        runner._normalise_optional_smoke_transaction_status(prepared)

    complete = _transaction("COMPLETE_SMOKE_PUBLICATION_PENDING")
    complete["complete_staged_state"] = "EXACT"
    assert (runner._normalise_optional_smoke_transaction_status(
        complete)["transaction_state"]
        == "COMPLETE_SMOKE_PUBLICATION_PENDING")
    complete["complete_staged_state"] = "PARTIAL_REGULAR"
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="not an authorised crash state"):
        runner._normalise_optional_smoke_transaction_status(complete)


@pytest.mark.parametrize(
    ("state", "field"),
    [
        ("UNSTARTED", "complete_staged_state"),
        ("PREPARED_MOVE_PENDING", "prepared_staged_state"),
        ("PREPARED_MOVE_PENDING", "complete_staged_state"),
        ("MOVED_REGENERATION_PENDING", "prepared_staged_state"),
        ("MOVED_REGENERATION_PENDING", "complete_staged_state"),
        ("RESTORED_COMPLETE_PENDING", "prepared_staged_state"),
        ("COMPLETE_SMOKE_PUBLICATION_PENDING", "prepared_staged_state"),
        ("COMPLETE_SMOKE_PUBLICATION_PENDING", "complete_staged_state"),
        ("COMPLETE", "prepared_staged_state"),
        ("COMPLETE", "complete_staged_state"),
    ],
)
def test_optional_transaction_rejects_partial_stage_outside_safe_phases(
        state: str, field: str) -> None:
    projection = _transaction(state)
    projection[field] = "PARTIAL_REGULAR"
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="not an authorised crash state"):
        runner._normalise_optional_smoke_transaction_status(projection)


def test_optional_transaction_is_inspected_before_absent_smoke(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        runner, "_optional_smoke_transaction_status",
        lambda **_kwargs: calls.append("transaction")
        or _transaction("MOVED_REGENERATION_PENDING"))
    monkeypatch.setattr(
        runner, "_load_json",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError(
            "an absent smoke must not be opened")))
    projection = runner._optional_encoding_smoke_projection(root=tmp_path)
    assert calls == ["transaction"]
    assert projection["terminal_present"] is True
    assert projection["smoke_protocol_complete"] is False
    assert projection["transaction_state"] == "MOVED_REGENERATION_PENDING"

    staged = _transaction("UNSTARTED")
    staged["prepared_staged_state"] = "EXACT"
    monkeypatch.setattr(
        runner, "_optional_smoke_transaction_status",
        lambda **_kwargs: dict(staged))
    staged_projection = runner._optional_encoding_smoke_projection(
        root=tmp_path)
    assert staged_projection["terminal_present"] is True
    assert staged_projection["zero_new_resume_verified"] is True


def test_optional_smoke_accepts_exact_index_first_transition_for_recovery(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        runner, "_optional_smoke_transaction_status",
        lambda **_kwargs: _transaction("UNSTARTED"))
    out = tmp_path / runner.SCORER_FIT_RELATIVE_PATH
    out.mkdir(parents=True)
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
    branch_smoke["smoke_branch_receipt_digest"] = (
        runner._builder_default_json_digest(branch_smoke))

    historical_index = {
        "schema": "go2_scorer_fit_corpus_v2_latents_index_v1",
        "fixture": "authority-bound-index",
    }
    historical_index["latents_index_digest"] = (
        runner._encoder_default_json_digest(historical_index))
    smoke = {
        "schema": "go2_scorer_fit_corpus_v2_end_to_end_smoke_receipt_v1",
        "status": runner.STATUS,
        "base_end_to_end_pass": True,
        "pass": False,
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
        "branch_smoke_receipt_digest": branch_smoke[
            "smoke_branch_receipt_digest"],
        "state_manifest_digest": state_digest,
        "full_bank_assignment_manifest_digest": assignment_digest,
        "scorer_fit_corpus_v2_scorer_contract_digest": contract_digest,
        "scorer_fit_corpus_v2_scorer_contract_artifact_digest":
            artifact_digest,
        "latent_index_digest": historical_index["latents_index_digest"],
        "zero_new_resume_verified": False,
        "single_shard_deletion_regeneration_verified": False,
        "smoke_protocol_complete": False,
    }
    smoke["smoke_receipt_digest"] = runner._encoder_default_json_digest(smoke)
    summary = {
        "schema": "go2_scorer_fit_corpus_v2_encoding_invocation_summary_v1",
        "fixture": "authority-bound-summary",
    }
    historical_index_raw = runner._encoder_pretty_json_bytes(historical_index)
    smoke_raw = runner._encoder_pretty_json_bytes(smoke)
    summary_raw = runner._encoder_pretty_json_bytes(summary)
    index_path = out / "latents_index_v2.json"
    smoke_path = out / runner.BUILDER.SCORER_FIT_V2_ENCODING_SMOKE_RECEIPT_NAME
    summary_path = out / "encoding_invocation_summary_v2.json"
    branch_path = out / runner.BUILDER.SCORER_FIT_V2_BRANCH_SMOKE_RECEIPT_NAME
    smoke_path.write_bytes(smoke_raw)
    summary_path.write_bytes(summary_raw)
    branch_path.write_bytes(runner._encoder_pretty_json_bytes(branch_smoke))

    def binding(
            path: Path, raw: bytes, *, schema: str,
            self_key: str | None = None, self_digest: str | None = None,
            ) -> dict[str, Any]:
        value: dict[str, Any] = {
            "path": str(path.relative_to(tmp_path)),
            "schema": schema,
            "raw_sha256": runner.hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
        }
        if self_key is not None:
            value.update({
                "self_digest_key": self_key,
                "self_digest": self_digest,
            })
        return value

    bundle = {
        "latent_index_binding": binding(
            index_path, historical_index_raw,
            schema=historical_index["schema"],
            self_key="latents_index_digest",
            self_digest=historical_index["latents_index_digest"]),
        "base_smoke_receipt_binding": binding(
            smoke_path, smoke_raw, schema=smoke["schema"],
            self_key="smoke_receipt_digest",
            self_digest=smoke["smoke_receipt_digest"]),
        "encoding_invocation_summary_binding": binding(
            summary_path, summary_raw, schema=summary["schema"]),
    }
    correction = {
        runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY: HEX_E,
        "single_shard_regeneration_transaction_contract_digest": HEX_C,
        "immutable_base_smoke_artifact_bundle": bundle,
        "base_smoke_artifact_bundle_digest":
            runner.DESIGN.canonical_digest(bundle),
    }
    migrated_index = {
        **{key: value for key, value in historical_index.items()
           if key != "latents_index_digest"},
        "encoder_path_projection_correction_digest": HEX_E,
    }
    migrated_index["latents_index_digest"] = (
        runner._encoder_default_json_digest(migrated_index))
    index_path.write_bytes(runner._encoder_pretty_json_bytes(migrated_index))

    monkeypatch.setattr(
        runner.BUILDER,
        "load_and_validate_full_bank_v2_manifests_for_consumption",
        lambda **_kwargs: {
            "design_authority": {
                "encoder_compute_dtype_correction_digest": HEX_C,
                "encoder_path_projection_correction_digest": HEX_E,
                "encoder_path_projection_correction": correction,
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
    projection = runner._optional_encoding_smoke_projection(root=tmp_path)
    assert projection["terminal_present"] is True
    assert projection["smoke_protocol_complete"] is False
    assert projection["requires_full_encoder_refresh"] is True

    current_smoke = {
        **{key: value for key, value in smoke.items()
           if key != "smoke_receipt_digest"},
        "encoder_path_projection_correction_digest": HEX_E,
        "latent_index_digest": migrated_index["latents_index_digest"],
    }
    current_smoke["smoke_receipt_digest"] = (
        runner._encoder_default_json_digest(current_smoke))
    smoke_path.write_bytes(runner._encoder_pretty_json_bytes(current_smoke))
    current = runner._optional_encoding_smoke_projection(root=tmp_path)
    assert current["terminal_present"] is True
    assert current["requires_full_encoder_refresh"] is False

    # The atomic producer replaces index first.  The reverse state can never
    # be a valid recovery boundary, even when both documents are self-signed.
    index_path.write_bytes(historical_index_raw)
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="smoke authority projection changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)

    smoke_path.write_bytes(smoke_raw)
    index_path.write_bytes(runner._encoder_pretty_json_bytes(migrated_index))

    migrated_index["unregistered"] = True
    migrated_index["latents_index_digest"] = (
        runner._encoder_default_json_digest({
            key: value for key, value in migrated_index.items()
            if key != "latents_index_digest"}))
    index_path.write_bytes(runner._encoder_pretty_json_bytes(migrated_index))
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="smoke authority projection changed"):
        runner._optional_encoding_smoke_projection(root=tmp_path)


@pytest.mark.parametrize(
    ("transaction_state", "retained_smoke", "first_command"),
    [
        ("UNSTARTED", "base-only", "smoke_encoding_zero_new"),
        ("PREPARED_MOVE_PENDING", "pretransaction",
         "smoke_single_shard_regeneration"),
        ("MOVED_REGENERATION_PENDING", "pretransaction",
         "smoke_single_shard_regeneration"),
        ("RESTORED_COMPLETE_PENDING", "pretransaction",
         "smoke_single_shard_regeneration"),
        ("COMPLETE_SMOKE_PUBLICATION_PENDING", "pretransaction",
         "smoke_single_shard_regeneration"),
    ],
)
def test_resume_each_incomplete_transaction_finishes_exactly_once(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        transaction_state: str, retained_smoke: str,
        first_command: str) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke=retained_smoke,
        retained_transaction_state=transaction_state)
    code, report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run[0] == first_command
    assert fake.commands_run.count("smoke_single_shard_regeneration") == 1
    assert "branch_smoke" not in fake.commands_run
    assert "resumed_smoke_single_shard_regeneration" in report[
        "completed_stages"]


@pytest.mark.parametrize(
    ("transaction_state", "retained_smoke", "staged_field"),
    [
        ("UNSTARTED", "absent", "prepared_staged_state"),
        ("RESTORED_COMPLETE_PENDING", "pretransaction",
         "complete_staged_state"),
    ],
)
def test_resume_partial_receipt_publication_uses_flagged_recovery_only(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
        transaction_state: str, retained_smoke: str,
        staged_field: str) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke=retained_smoke,
        retained_transaction_state=transaction_state)
    original = fake.validate

    def staged(kind: str, root: Path,
               interpreter: Path) -> Mapping[str, Any]:
        projection = dict(original(kind, root, interpreter))
        if kind == "encoding-smoke-optional":
            projection[staged_field] = "PARTIAL_REGULAR"
            projection["terminal_present"] = True
            projection["smoke_protocol_complete"] = False
            projection["zero_new_resume_verified"] = True
            projection["single_registered_shard_regenerated"] = False
        return projection

    code, _report = runner.run_pipeline(
        root=fake.root, command_runner=fake.command,
        runtime_probe_invoker=fake.runtime_probe,
        validation_invoker=staged, authority=object(), resume=True)
    assert code == 0
    assert fake.commands_run.count("smoke_single_shard_regeneration") == 1
    assert "smoke_encoding_zero_new" not in fake.commands_run


def test_complete_transaction_without_bound_pass_never_opens_full_corpus(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke="pretransaction",
        retained_transaction_state="COMPLETE")
    with pytest.raises(runner.FullBankV2RunnerError,
                       match="lacks its exact bound PASS smoke"):
        fake.run(resume=True)
    assert "smoke_single_shard_regeneration" not in fake.commands_run
    assert "full_branch_corpus" not in fake.commands_run


def test_late_resume_accepts_immutable_original_regeneration_evidence(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke="pretransaction",
        retained_transaction_state="RESTORED_COMPLETE_PENDING")
    code, _report = fake.run(resume=True)
    assert code == 0
    assert fake.commands_run.count("smoke_single_shard_regeneration") == 1


def test_resume_rejects_impossible_transaction_regeneration_count(
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    fake = FakePipeline(
        monkeypatch, tmp_path, retained_smoke="pretransaction",
        retained_transaction_state="RESTORED_COMPLETE_PENDING")
    original = fake.validate

    def impossible(kind: str, root: Path,
                   interpreter: Path) -> Mapping[str, Any]:
        projection = dict(original(kind, root, interpreter))
        if (kind == "encoding-smoke" and fake.commands_run
                and fake.commands_run[-1]
                == "smoke_single_shard_regeneration"):
            projection["invocation_new_horizon_shards"] = 2
        return projection

    with pytest.raises(runner.FullBankV2RunnerError,
                       match="transaction evidence changed"):
        runner.run_pipeline(
            root=fake.root, command_runner=fake.command,
            runtime_probe_invoker=fake.runtime_probe,
            validation_invoker=impossible, authority=object(), resume=True)
    assert fake.commands_run.count("smoke_single_shard_regeneration") == 1
    assert "full_branch_corpus" not in fake.commands_run


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
    assert commands["smoke_single_shard_regeneration"][-1] == (
        "--single-shard-regeneration-transaction")


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


def test_old_import_issue_stage_replays_nested_path_lineage_at_current_source(
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
        "encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_encoder_import_correction": immutable,
        "immutable_encoder_import_correction_digest": HEX_D,
    }
    immutable_dtype = {
        "payload": dtype, "binding": {"self_digest": HEX_C}}
    path_correction = {
        "immutable_encoder_compute_dtype_correction": immutable_dtype,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_IMPORT_CORRECTION_STATUS = "ISSUED_FIXTURE_CORRECTION"
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH = (
            runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH)

        def load_encoder_path_projection_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("load-path")
            return path_correction

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert value == immutable_dtype
            events.append("validate-immutable-dtype")
            return immutable_dtype

        def validate_immutable_encoder_import_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert value == immutable
            events.append("validate-immutable-import")
            return immutable

        def issue_encoder_import_correction(self, **_kwargs: Any) -> None:
            raise AssertionError("immutable import correction was reissued")

        def load_encoder_compute_dtype_correction_for_consumption(
                self, **_kwargs: Any) -> None:
            raise AssertionError(
                "historical dtype source was live-reinterpreted")

        def load_encoder_import_correction_for_consumption(
                self, **_kwargs: Any) -> None:
            raise AssertionError(
                "historical import source was live-reinterpreted")

    path = tmp_path / (
        FakeDesign.ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH)
    path.parent.mkdir(parents=True)
    path.write_text("installed fixture marker")
    report = runner.issue_encoder_import_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == [
        "load-path", "validate-immutable-dtype",
        "validate-immutable-import"]
    assert report["scorer_fit_corpus_v2_encoder_import_correction_digest"] \
        == HEX_D
    assert report["replayed_from_immutable_dtype_correction_lineage"] is True
    assert report[
        "replayed_from_immutable_path_projection_correction_lineage"] is True
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
            "payload": {
                "encoder_import_correction_digest": HEX_D,
                "immutable_successor_scorer_contract_binding": {
                    "self_digest": runner.SCORER_CONTRACT.
                        IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
                    "embedded_contract_self_digest": runner.SCORER_CONTRACT.
                        IMMUTABLE_ISSUED_CONTRACT_DIGEST,
                },
            },
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


def test_old_dtype_issue_stage_replays_nested_path_lineage_without_reissue(
        tmp_path: Path) -> None:
    events: list[str] = []
    dtype = {
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
    immutable_dtype = {
        "payload": dtype, "binding": {"self_digest": HEX_C}}
    path_correction = {
        "immutable_encoder_compute_dtype_correction": immutable_dtype,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS = "ISSUED_FIXTURE_DTYPE"
        ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH = (
            runner.DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH)

        def load_encoder_path_projection_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("load-path")
            return path_correction

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert value == immutable_dtype
            events.append("validate-immutable-dtype")
            return immutable_dtype

        def issue_encoder_compute_dtype_correction(
                self, **_kwargs: Any) -> None:
            raise AssertionError("immutable dtype correction was reissued")

        def load_encoder_compute_dtype_correction_for_consumption(
                self, **_kwargs: Any) -> None:
            raise AssertionError(
                "historical dtype source was live-reinterpreted")

    path = tmp_path / (
        FakeDesign.ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH)
    path.parent.mkdir(parents=True)
    path.write_text("installed fixture marker")
    report = runner.issue_encoder_compute_dtype_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == ["load-path", "validate-immutable-dtype"]
    assert report["encoder_compute_dtype_correction_digest"] == HEX_C
    assert report[
        "replayed_from_immutable_path_projection_correction_lineage"] is True
    assert report["encoder_import_correction_reissued_or_rewritten"] is False


def test_issue_encoder_path_projection_correction_preserves_predecessors(
        tmp_path: Path) -> None:
    events: list[str] = []
    dtype = {
        "immutable_encoder_import_correction": {
            "payload": {"encoder_import_correction_digest": HEX_D},
            "binding": {"self_digest": HEX_D},
        },
        "immutable_encoder_import_correction_digest": HEX_D,
        "encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
    }
    immutable_dtype = {
        "payload": dtype, "binding": {"self_digest": HEX_C}}
    payload = {
        "immutable_encoder_compute_dtype_correction": immutable_dtype,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
        "scorer_fit_corpus_v2_encoder_path_projection_correction_digest":
            HEX_E,
    }

    class FakeDesign:
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_encoder_path_projection_correction_digest")
        ENCODER_PATH_PROJECTION_CORRECTION_STATUS = "ISSUED_FIXTURE_PATH"

        def issue_encoder_path_projection_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue-path")
            return payload

        def load_encoder_path_projection_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("reopen-path")
            return payload

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            assert value == immutable_dtype
            events.append("validate-immutable-dtype")
            return immutable_dtype

    report = runner.issue_encoder_path_projection_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == [
        "issue-path", "reopen-path", "validate-immutable-dtype"]
    assert report["encoder_path_projection_correction_digest"] == HEX_E
    assert report["encoder_compute_dtype_correction_digest"] == HEX_C
    assert report["encoder_import_correction_digest"] == HEX_D
    assert report[
        "encoder_compute_dtype_correction_reissued_or_rewritten"] is False
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


def test_parser_exposes_path_projection_correction_before_resume_run() -> None:
    correction = runner._parser().parse_args(
        ["--stage", "issue-encoder-path-projection-correction"])
    resumed = runner._parser().parse_args(
        ["--stage", "run", "--resume"])
    assert correction.stage == "issue-encoder-path-projection-correction"
    assert resumed.stage == "run" and resumed.resume is True


def test_issue_branch_redrive_projection_correction_is_source_only(
        tmp_path: Path) -> None:
    events: list[str] = []
    dtype = {
        "immutable_encoder_import_correction": {
            "payload": {"encoder_import_correction_digest": HEX_D},
        },
        "immutable_encoder_import_correction_digest": HEX_D,
        "encoder_compute_dtype_correction_digest": HEX_C,
    }
    path = {
        "scorer_fit_corpus_v2_encoder_path_projection_correction_digest":
            HEX_E,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_encoder_compute_dtype_correction": {
            "payload": dtype, "binding": {"self_digest": HEX_C}},
    }
    correction = {
        "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest":
            HEX_B,
        "immutable_encoder_path_projection_correction_digest": HEX_E,
        "immutable_encoder_path_projection_correction": {
            "payload": path, "binding": {"self_digest": HEX_E}},
    }

    class FakeDesign:
        BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest")
        BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS = "ISSUED_FIXTURE_REDRIVE"
        ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_encoder_path_projection_correction_digest")
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D

        def issue_branch_redrive_projection_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue-redrive")
            return correction

        def load_branch_redrive_projection_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("reopen-redrive")
            return correction

        def validate_immutable_encoder_path_projection_correction(
                self, value: Mapping[str, Any], **_kwargs: Any
                ) -> Mapping[str, Any]:
            events.append("validate-immutable-path")
            return dict(value)

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("validate-immutable-dtype")
            return dict(value)

    report = runner.issue_branch_redrive_projection_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == [
        "issue-redrive", "reopen-redrive", "validate-immutable-path",
        "validate-immutable-dtype"]
    assert report["branch_redrive_projection_correction_digest"] == HEX_B
    assert report["encoder_path_projection_correction_digest"] == HEX_E
    assert report["retained_valid_branch_count"] == 120
    assert report["retained_invalid_attempt_receipt_count"] == 12
    assert report["manifest_or_identity_replaced"] is False
    assert report["completed_branch_reissued_or_rewritten"] is False
    assert report["candidate_outcome_or_label_value_read_for_correction"] \
        is False
    assert report["branch_latent_or_scorer_runtime_started_by_issue_stage"] \
        is False


def test_parser_exposes_redrive_projection_correction_before_resume_run() -> None:
    correction = runner._parser().parse_args(
        ["--stage", "issue-branch-redrive-projection-correction"])
    resumed = runner._parser().parse_args(
        ["--stage", "run", "--resume"])
    assert correction.stage == "issue-branch-redrive-projection-correction"
    assert resumed.stage == "run" and resumed.resume is True


def test_issue_optional_smoke_partial_corpus_resume_correction_is_source_only(
        tmp_path: Path) -> None:
    events: list[str] = []
    dtype = {
        "immutable_encoder_import_correction": {
            "payload": {"encoder_import_correction_digest": HEX_D},
        },
        "immutable_encoder_import_correction_digest": HEX_D,
        "encoder_compute_dtype_correction_digest": HEX_C,
    }
    path = {
        "scorer_fit_corpus_v2_encoder_path_projection_correction_digest":
            HEX_E,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_encoder_compute_dtype_correction": {
            "payload": dtype, "binding": {"self_digest": HEX_C}},
    }
    redrive = {
        "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest":
            HEX_B,
        "immutable_encoder_path_projection_correction_digest": HEX_E,
        "immutable_encoder_path_projection_correction": {
            "payload": path, "binding": {"self_digest": HEX_E}},
    }
    correction = {
        "scorer_fit_corpus_v2_optional_smoke_partial_corpus_resume_"
        "correction_digest": HEX_A,
        "immutable_branch_redrive_projection_correction_digest": HEX_B,
        "immutable_branch_redrive_projection_correction": {
            "payload": redrive,
            "binding": {
                "self_digest_key": (
                    "scorer_fit_corpus_v2_branch_redrive_projection_"
                    "correction_digest"),
                "self_digest": HEX_B,
            },
        },
    }

    class FakeDesign:
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_optional_smoke_partial_corpus_resume_"
            "correction_digest")
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_STATUS = (
            "ISSUED_FIXTURE_PARTIAL_RESUME")
        BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_branch_redrive_projection_correction_"
            "digest")
        ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_encoder_path_projection_correction_digest")
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D

        def issue_optional_smoke_partial_corpus_resume_correction(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("issue-partial-resume")
            return correction

        def load_optional_smoke_partial_corpus_resume_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("reopen-partial-resume")
            return correction

        def validate_immutable_branch_redrive_projection_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("validate-immutable-redrive")
            return dict(value)

        def validate_immutable_encoder_path_projection_correction(
                self, value: Mapping[str, Any], **_kwargs: Any
                ) -> Mapping[str, Any]:
            events.append("validate-immutable-path")
            return dict(value)

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("validate-immutable-dtype")
            return dict(value)

    report = runner.issue_optional_smoke_partial_corpus_resume_correction(
        root=tmp_path, design_authority=FakeDesign())
    assert events == [
        "issue-partial-resume", "reopen-partial-resume",
        "validate-immutable-redrive", "validate-immutable-path",
        "validate-immutable-dtype",
    ]
    assert report[
        "optional_smoke_partial_corpus_resume_correction_digest"] == HEX_A
    assert report["branch_redrive_projection_correction_digest"] == HEX_B
    assert report["retained_valid_branch_count"] == 120
    assert report["retained_invalid_attempt_receipt_count"] == 12
    assert report["retained_smoke_transaction_complete"] is True
    assert report["manifest_or_identity_replaced"] is False
    assert report["completed_branch_reissued_or_rewritten"] is False
    assert report["branch_or_latent_runtime_started_by_issue_stage"] is False
    assert report["candidate_outcome_or_label_value_read_for_correction"] \
        is False


def test_parser_exposes_partial_corpus_resume_correction_before_resume() -> None:
    correction = runner._parser().parse_args([
        "--stage", "issue-optional-smoke-partial-corpus-resume-correction"])
    resumed = runner._parser().parse_args(
        ["--stage", "run", "--resume"])
    assert correction.stage == (
        "issue-optional-smoke-partial-corpus-resume-correction")
    assert resumed.stage == "run" and resumed.resume is True


def test_historical_correction_issue_stages_replay_newest_immutable_chain(
        tmp_path: Path) -> None:
    dtype = {
        "immutable_encoder_import_correction": {
            "payload": {
                "encoder_import_correction_digest": HEX_D,
                "immutable_successor_scorer_contract_binding": {
                    "self_digest": runner.SCORER_CONTRACT.
                        IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
                    "embedded_contract_self_digest": runner.SCORER_CONTRACT.
                        IMMUTABLE_ISSUED_CONTRACT_DIGEST,
                },
            },
            "binding": {"self_digest": HEX_D},
        },
        "immutable_encoder_import_correction_digest": HEX_D,
        "encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
    }
    path = {
        "scorer_fit_corpus_v2_encoder_path_projection_correction_digest":
            HEX_E,
        "immutable_encoder_compute_dtype_correction_digest": HEX_C,
        "immutable_encoder_compute_dtype_correction": {
            "payload": dtype, "binding": {"self_digest": HEX_C}},
        "immutable_successor_scorer_contract_binding": {
            "self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
            "embedded_contract_self_digest":
                runner.SCORER_CONTRACT.IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        },
    }
    redrive = {
        "scorer_fit_corpus_v2_branch_redrive_projection_correction_digest":
            HEX_B,
        "immutable_encoder_path_projection_correction_digest": HEX_E,
        "immutable_encoder_path_projection_correction": {
            "payload": path,
            "binding": {
                "self_digest_key": (
                    "scorer_fit_corpus_v2_encoder_path_projection_"
                    "correction_digest"),
                "self_digest": HEX_E,
            },
        },
    }
    newest = {
        "scorer_fit_corpus_v2_optional_smoke_partial_corpus_resume_"
        "correction_digest": HEX_A,
        "immutable_branch_redrive_projection_correction_digest": HEX_B,
        "immutable_branch_redrive_projection_correction": {
            "payload": redrive,
            "binding": {
                "self_digest_key": (
                    "scorer_fit_corpus_v2_branch_redrive_projection_"
                    "correction_digest"),
                "self_digest": HEX_B,
            },
        },
    }
    events: list[str] = []

    class FakeDesign:
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_RELATIVE_PATH = (
            Path("scorer_fit/newest.json"))
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_optional_smoke_partial_corpus_resume_"
            "correction_digest")
        BRANCH_REDRIVE_PROJECTION_CORRECTION_RELATIVE_PATH = Path(
            "scorer_fit/redrive.json")
        BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_branch_redrive_projection_correction_"
            "digest")
        BRANCH_REDRIVE_PROJECTION_CORRECTION_STATUS = "ISSUED_REDRIVE"
        ENCODER_PATH_PROJECTION_CORRECTION_RELATIVE_PATH = Path(
            "scorer_fit/path.json")
        ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY = (
            "scorer_fit_corpus_v2_encoder_path_projection_correction_digest")
        ENCODER_PATH_PROJECTION_CORRECTION_STATUS = "ISSUED_PATH"
        ENCODER_COMPUTE_DTYPE_CORRECTION_RELATIVE_PATH = Path(
            "scorer_fit/dtype.json")
        ENCODER_COMPUTE_DTYPE_CORRECTION_SELF_KEY = (
            "encoder_compute_dtype_correction_digest")
        ENCODER_COMPUTE_DTYPE_CORRECTION_STATUS = "ISSUED_DTYPE"
        ENCODER_IMPORT_CORRECTION_RELATIVE_PATH = Path(
            "scorer_fit/import.json")
        ENCODER_IMPORT_CORRECTION_SELF_KEY = (
            "encoder_import_correction_digest")
        ENCODER_IMPORT_CORRECTION_STATUS = "ISSUED_IMPORT"
        IMMUTABLE_ENCODER_COMPUTE_DTYPE_CORRECTION_DIGEST = HEX_C
        IMMUTABLE_ENCODER_IMPORT_CORRECTION_DIGEST = HEX_D

        def load_optional_smoke_partial_corpus_resume_correction_for_consumption(
                self, *, root: Path) -> Mapping[str, Any]:
            assert root == tmp_path
            events.append("load-newest")
            return newest

        def validate_immutable_branch_redrive_projection_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("redrive")
            return dict(value)

        def validate_immutable_encoder_path_projection_correction(
                self, value: Mapping[str, Any], **_kwargs: Any
                ) -> Mapping[str, Any]:
            events.append("path")
            return dict(value)

        def validate_immutable_encoder_compute_dtype_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("dtype")
            return dict(value)

        def validate_immutable_encoder_import_correction(
                self, value: Mapping[str, Any]) -> Mapping[str, Any]:
            events.append("import")
            return dict(value)

        def __getattr__(self, name: str) -> Any:
            if name.startswith(("issue_", "load_branch_redrive_",
                                "load_encoder_")):
                raise AssertionError(
                    f"historical correction stage was live-reissued: {name}")
            raise AttributeError(name)

    marker = tmp_path / (
        FakeDesign.
        OPTIONAL_SMOKE_PARTIAL_CORPUS_RESUME_CORRECTION_RELATIVE_PATH)
    marker.parent.mkdir(parents=True)
    marker.write_text("installed newest fixture")
    design = FakeDesign()
    redrive_report = runner.issue_branch_redrive_projection_correction(
        root=tmp_path, design_authority=design)
    path_report = runner.issue_encoder_path_projection_correction(
        root=tmp_path, design_authority=design)
    dtype_report = runner.issue_encoder_compute_dtype_correction(
        root=tmp_path, design_authority=design)
    import_report = runner.issue_encoder_import_correction(
        root=tmp_path, design_authority=design)
    assert redrive_report[
        "replayed_from_immutable_optional_smoke_partial_corpus_resume_"
        "correction_lineage"] is True
    assert path_report[
        "replayed_from_immutable_optional_smoke_partial_corpus_resume_"
        "correction_lineage"] is True
    assert dtype_report[
        "replayed_from_immutable_optional_smoke_partial_corpus_resume_"
        "correction_lineage"] is True
    assert import_report[
        "replayed_from_immutable_optional_smoke_partial_corpus_resume_"
        "correction_lineage"] is True
    assert events.count("load-newest") == 4


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
