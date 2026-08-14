"""Prospective scorer contract for the full-bank scorer-fit corpus V2.

DEVELOPMENT_ONLY_NOT_CLAIM_BEARING.

This successor changes only the scorer-fit corpus identity/assignment binding:
the same 120 states are each paired with all twelve frozen candidates.  The
oracle, target encoder, rendering/preprocessing, scorer architecture, training
policy, final-epoch rule, no-latent baseline, and qualification criteria are
copied exactly from the corrected oracle-v1.2 scorer contract.

The module deliberately separates a pure payload builder/validator from the
issuer.  Production issuance first replays the design-authority and manifest
producer validators, requires a clean committed source tree, and then creates
the sole contract artifact with ``O_EXCL`` plus file and directory ``fsync``.
It never overwrites or reinterprets a predecessor contract.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any, Mapping

from lewm.oracle import go2_scorer_contract_v1_2 as PREDECESSOR
from lewm.oracle import go2_scorer_fit_corpus_v2_design as DESIGN


STATUS = "DEVELOPMENT_ONLY_NOT_CLAIM_BEARING"
CONTRACT_SCHEMA = "go2_scorer_fit_corpus_v2_scorer_contract_v1"
ARTIFACT_SCHEMA = "go2_scorer_fit_corpus_v2_scorer_contract_artifact_v1"
CONTRACT_STATUS = "ISSUED_PROSPECTIVE_PRE_BRANCH_FULL_BANK_SCORER_CONTRACT"
CONTRACT_SELF_KEY = "scorer_fit_corpus_v2_scorer_contract_digest"
ARTIFACT_SELF_KEY = "contract_artifact_digest"
ROOT = Path(__file__).resolve().parents[2]
PACKAGE_ROOT_RELATIVE = Path(".generated/go2_utility_scorer_fit_corpus_v2")
ARTIFACT_NAME = "scorer_fit_corpus_v2_scorer_contract.json"
ARTIFACT_RELATIVE_PATH = PACKAGE_ROOT_RELATIVE / ARTIFACT_NAME

# The successor contract was issued once, before the first branch, from this
# exact clean source.  A later post-smoke encoder-import compatibility
# correction may validate that historical source binding, but must never
# rebuild, resign, or reinterpret the scientific contract under a newer HEAD.
IMMUTABLE_ISSUED_SOURCE_COMMIT = (
    "72b0d771b748e777a9da47fca88a9d6cfb62d0ef"
)
IMMUTABLE_ISSUED_CONTRACT_DIGEST = (
    "8fc0edae875cba6487ff1a1a771f96b0157da1474ac00de4186ecdb41b66d5df"
)
IMMUTABLE_ISSUED_ARTIFACT_DIGEST = (
    "4455fd397ce7665f02725924a64ab87b1e0e9a3506d9ba64edbcc9b4daa1e121"
)

TERMINAL_INFEASIBILITY_SOURCE_COMMIT = (
    "e1bdbe7adc15d0aa85f69ffb9e97fa198eb152c5"
)
GLOBAL_EXECUTION_AMENDMENT_DIGEST = (
    "f4cbc2e5e7baa1c4cebda3dfde0e5f9744aa50a7d1d1c5c53704338a1bb6f822"
)
GLOBAL_EXACT_MODEL_DIGEST = (
    "57770fe998cb7d7cb952b122077fe7dd8daeadce5f70fa3150c6f8641e5162b4"
)
EXACT_INFEASIBILITY_DIGEST = (
    "eb9d347fee6d4b498cf02a2c0af51483304d9489aa98050a8212e99c737135dc"
)
TERMINAL_INFEASIBILITY_RECEIPT_DIGEST = (
    "d1a17c289e3993f6cad30cc3ff1725246075881c292e930e776d8bcd7fb215d4"
)
V2_SOURCE_COMMIT = "112579b680a83df35b72100e5ecc528b5b34e18f"
V2_BENCHMARK_CONTRACT_DIGEST = (
    "05e91f432b82ad6d30ab6d8a1d4431b4e9c4ccd2470439b17535486d394667f2"
)
V2_BENCHMARK_RECEIPT_DIGEST = (
    "77f9e8d44aab6ca3d7b60c2c46a94cb4afd1be4dc5bc47a67521d70220cf69b9"
)
V2_TERMINAL_FAILURE_RECEIPT_DIGEST = (
    "ae8f23d1127e24a52208d0c48f1635fc8f28b157ed0790ac3f2695a15672d67e"
)
V1_FAILURE_STATUS = "IMMUTABLE_FAIL_COLD_START_INCLUDED_IN_FIRST_TIMED_WAVE"
V1_FAILURE_RECEIPT_DIGEST = (
    "afb4c190cf7d2e93b678a546fc233340102c6f5260110b1471752bc54a0e88d6"
)
V1_FAILURE_RAW_SHA256 = (
    "cc3b07b3ed470058dc395d0eb34d5d6cd83e8edc0140e4c18f249d4d4747fe5b"
)
V1_FAILURE_BYTE_COUNT = 2_688
V1_FAILURE_SOURCE_COMMIT = "d9d129e2bbea8519f7ed3186f3cfb3c661baba04"

STATE_COUNT = 120
FIT_STATE_COUNT = 96
CALIBRATION_STATE_COUNT = 24
CANDIDATE_COUNT = 12
ASSIGNMENT_COUNT = 1_440
FIT_ROW_COUNT = 1_152
CALIBRATION_ROW_COUNT = 288
EPOCHS = 60
BATCH_SIZE = 64
OPTIMISER_UPDATES_PER_EPOCH = 18
OPTIMISER_UPDATES_PER_MODEL = 1_080
EXAMPLE_PRESENTATIONS_PER_MODEL = 69_120
MODEL_COUNT = 2

_HEX40 = re.compile(r"^[0-9a-f]{40}$")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_ARTIFACT_BINDING_BASE_KEYS = frozenset({
    "path", "self_digest_key", "self_digest", "raw_sha256", "byte_count",
})
_AUTHORITY_ARTIFACT_BINDING_KEYS = _ARTIFACT_BINDING_BASE_KEYS | {
    "schema", "source_repository_commit",
}
_MANIFEST_BUNDLE_KEYS = frozenset({
    "design_authority", "source_correction", "source_correction_binding",
    "source_correction_digest", "selection", "selection_binding", "revalidation",
    "revalidation_binding", "small_shard", "small_shard_binding",
    "state_manifest", "state_manifest_binding", "assignment_manifest",
    "assignment_manifest_binding",
})
_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_fit_corpus_v2_design.py",
    "lewm/oracle/go2_scorer_fit_corpus_v2_scorer_contract.py",
    "scripts/build_go2_branch_corpus_v1_2.py",
    "scripts/encode_go2_branch_corpus_v1_2.py",
    "scripts/train_go2_utility_scorer_v1_2.py",
    "scripts/apply_go2_utility_scorer_to_counterfactual_development_v1_2.py",
    "scripts/run_go2_scorer_fit_full_bank_v2.py",
)


class ScorerFitCorpusV2ContractError(RuntimeError):
    """A prospective contract or one of its pre-outcome bindings is invalid."""


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ScorerFitCorpusV2ContractError(message)


def _require_digest(value: Any, label: str) -> str:
    _require(isinstance(value, str) and _HEX64.fullmatch(value) is not None,
             f"{label} is not a lowercase SHA-256 digest")
    return str(value)


def _artifact_binding(value: Mapping[str, Any], *, label: str,
                      self_key: str | None = None,
                      authority_binding: bool = False) -> dict[str, Any]:
    expected_keys = (_AUTHORITY_ARTIFACT_BINDING_KEYS
                     if authority_binding else _ARTIFACT_BINDING_BASE_KEYS)
    _require(isinstance(value, Mapping) and set(value) == expected_keys,
             f"{label} artifact binding schema is not closed")
    binding = dict(value)
    _require(isinstance(binding.get("path"), str) and binding["path"],
             f"{label} artifact path is absent")
    _require(isinstance(binding.get("self_digest_key"), str)
             and binding["self_digest_key"],
             f"{label} self-digest key is absent")
    if self_key is not None:
        _require(binding["self_digest_key"] == self_key,
                 f"{label} self-digest key changed")
    _require_digest(binding.get("self_digest"), f"{label} self digest")
    _require_digest(binding.get("raw_sha256"), f"{label} raw SHA-256")
    _require(type(binding.get("byte_count")) is int
             and binding["byte_count"] > 0,
             f"{label} byte count is invalid")
    if authority_binding:
        _require(isinstance(binding.get("schema"), str) and binding["schema"],
                 f"{label} schema is absent")
        _require(_HEX40.fullmatch(str(
            binding.get("source_repository_commit", ""))) is not None,
                 f"{label} source commit is malformed")
    return binding


def _source_file_binding(relative_path: str, *, root: Path) -> dict[str, Any]:
    path = root / relative_path
    _require(path.is_file() and not path.is_symlink(),
             f"bound implementation is missing or symlinked: {relative_path}")
    return {
        "path": relative_path,
        "byte_count": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _validated_repository_state(*, root: Path, commit: str, status: str,
                                top_level: str) -> dict[str, Any]:
    try:
        resolved_root = root.resolve(strict=True)
        resolved_top = Path(top_level).resolve(strict=True)
    except OSError as exc:
        raise ScorerFitCorpusV2ContractError(
            "cannot resolve scorer-fit V2 source repository") from exc
    _require(resolved_top == resolved_root,
             "scorer-fit V2 source is not issued from the custody root")
    _require(_HEX40.fullmatch(commit) is not None,
             "source repository HEAD is not a full commit digest")
    _require(status == "",
             "source repository is not clean; commit V2 source before issuance")
    implementations = {
        Path(path).stem: _source_file_binding(path, root=resolved_root)
        for path in _SOURCE_PATHS
    }
    return {
        "schema": "go2_scorer_fit_corpus_v2_clean_source_binding_v1",
        "source_repository_root": str(resolved_root),
        "source_repository_commit": commit,
        "source_repository_clean": True,
        "git_status_porcelain_v1": "",
        "git_status_untracked_files": "all",
        "git_ignored_generated_artifacts_permitted": True,
        "nonignored_tracked_or_untracked_changes_permitted": False,
        "bound_implementations": implementations,
        "bound_implementations_digest": canonical_digest(implementations),
    }


def clean_source_binding(*, root: Path = ROOT) -> dict[str, Any]:
    def git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", *arguments], cwd=root, check=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        return completed.stdout.strip("\n")

    try:
        return _validated_repository_state(
            root=root,
            commit=git("rev-parse", "HEAD"),
            status=git("status", "--porcelain=v1", "--untracked-files=all"),
            top_level=git("rev-parse", "--show-toplevel"),
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ScorerFitCorpusV2ContractError(
            "cannot verify clean scorer-fit V2 source") from exc


def training_budget_interpretation() -> dict[str, Any]:
    training = PREDECESSOR.SCORER["training"]
    _require(training.get("epochs") == EPOCHS
             and training.get("batch") == BATCH_SIZE,
             "predecessor scorer epoch/batch budget changed")
    return {
        "frozen_budget_unit": "epochs",
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "fit_examples": FIT_ROW_COUNT,
        "calibration_examples": CALIBRATION_ROW_COUNT,
        "optimizer_updates_per_epoch": OPTIMISER_UPDATES_PER_EPOCH,
        "optimizer_updates_per_model": OPTIMISER_UPDATES_PER_MODEL,
        "example_presentations_per_model": EXAMPLE_PRESENTATIONS_PER_MODEL,
        "models": ["shared_true_latent_scorer", "no_latent_baseline"],
        "aggregate_optimizer_updates":
            MODEL_COUNT * OPTIMISER_UPDATES_PER_MODEL,
        "aggregate_example_presentations":
            MODEL_COUNT * EXAMPLE_PRESENTATIONS_PER_MODEL,
        "step_budget_also_retained": False,
        "final_epoch_only": True,
        "best_epoch_selection_permitted": False,
    }


def _protected_predecessor_contract() -> dict[str, Any]:
    """Return only the scientific fields that V2 is forbidden to change."""

    frozen = PREDECESSOR.contract()
    return {
        "status": frozen["status"],
        "baseline_contract_digest": frozen["baseline_contract_digest"],
        "candidate_bank_digest": frozen["candidate_bank_digest"],
        "oracle_v1_2_digest": frozen["oracle_v1_2_digest"],
        "progress_target_digest": frozen["progress_target_digest"],
        "safety_target_digest": frozen["safety_target_digest"],
        "target_encoder": copy.deepcopy(frozen["target_encoder"]),
        "render_contract": copy.deepcopy(frozen["render_contract"]),
        "preprocess_contract": copy.deepcopy(frozen["preprocess_contract"]),
        "predictor_input_contract": copy.deepcopy(
            frozen["predictor_input_contract"]),
        "scorer": copy.deepcopy(frozen["scorer"]),
    }


def build_contract(
        *, source_binding: Mapping[str, Any],
        design_binding: Mapping[str, Any],
        source_correction_binding: Mapping[str, Any],
        manifest_replay_correction_binding: Mapping[str, Any],
        mask_classification_binding: Mapping[str, Any],
        selection_binding: Mapping[str, Any],
        revalidation_binding: Mapping[str, Any],
        state_manifest_binding: Mapping[str, Any],
        assignment_manifest_binding: Mapping[str, Any],
        ) -> dict[str, Any]:
    """Purely build the closed successor contract from validated bindings."""

    source = dict(source_binding)
    _require(source.get("source_repository_clean") is True
             and _HEX40.fullmatch(str(
                 source.get("source_repository_commit", ""))) is not None,
             "successor contract requires a clean committed source binding")
    design = _artifact_binding(
        design_binding, label="corpus V2 design",
        self_key=DESIGN.DESIGN_SELF_KEY, authority_binding=True)
    source_correction = _artifact_binding(
        source_correction_binding, label="corpus V2 preselection source correction",
        self_key=DESIGN.SOURCE_CORRECTION_SELF_KEY, authority_binding=True)
    replay_correction = _artifact_binding(
        manifest_replay_correction_binding,
        label="corpus V2 post-install manifest replay correction",
        self_key=DESIGN.MANIFEST_REPLAY_CORRECTION_SELF_KEY,
        authority_binding=True)
    masks = _artifact_binding(
        mask_classification_binding, label="rotation-mask classification",
        self_key=DESIGN.MASK_CLASSIFICATION_SELF_KEY, authority_binding=True)
    selection = _artifact_binding(
        selection_binding, label="small-completion selection",
        self_key="full_bank_small_completion_selection_digest")
    revalidation = _artifact_binding(
        revalidation_binding, label="full-bank state revalidation",
        self_key="full_bank_preoutcome_state_revalidation_digest")
    state_manifest = _artifact_binding(
        state_manifest_binding, label="state manifest",
        self_key="state_manifest_digest")
    assignment_manifest = _artifact_binding(
        assignment_manifest_binding, label="assignment manifest",
        self_key="full_bank_assignment_manifest_digest")
    payload = {
        "schema": CONTRACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "complete": True,
        "corpus_design_version": "scorer_fit_corpus_v2_full_bank_12",
        "source_binding": source,
        "source_binding_digest": canonical_digest(source),
        "preoutcome_authority_bindings": {
            "design_amendment": design,
            "preselection_source_correction": source_correction,
            "post_install_manifest_replay_correction": replay_correction,
            "rotation_mask_classification": masks,
            "small_completion_selection": selection,
            "full_bank_state_revalidation": revalidation,
            "state_identity_manifest": state_manifest,
            "expanded_assignment_manifest": assignment_manifest,
        },
        "state_selector_binding": {
            "state_manifest_digest": state_manifest["self_digest"],
            "assignment_manifest_digest": assignment_manifest["self_digest"],
            "revalidation_receipt_digest": revalidation["self_digest"],
        },
        "corpus_counts": {
            "states": STATE_COUNT,
            "fit_states": FIT_STATE_COUNT,
            "calibration_states": CALIBRATION_STATE_COUNT,
            "candidates_per_state": CANDIDATE_COUNT,
            "branches": ASSIGNMENT_COUNT,
            "fit_branches": FIT_ROW_COUNT,
            "calibration_branches": CALIBRATION_ROW_COUNT,
            "candidate_indices": list(range(CANDIDATE_COUNT)),
            "candidate_appearances_each": STATE_COUNT,
            "candidate_fit_appearances_each": FIT_STATE_COUNT,
            "candidate_calibration_appearances_each":
                CALIBRATION_STATE_COUNT,
            "candidate_stratum_appearances_each": 40,
            "candidate_family_appearances_each": 15,
            "candidate_family_fit_appearances_each": 12,
            "candidate_family_calibration_appearances_each": 3,
            "candidate_family_stratum_appearances_each": 5,
            "unordered_pair_cooccurrences_each": STATE_COUNT,
        },
        "training_budget_interpretation": training_budget_interpretation(),
        "protected_predecessor_scientific_contract":
            _protected_predecessor_contract(),
        "preoutcome_lineage": {
            "scorer_fit_corpus_v2_source_correction_digest":
                source_correction["self_digest"],
            "scorer_fit_corpus_v2_manifest_replay_correction_digest":
                replay_correction["self_digest"],
            "terminal_infeasibility_source_commit":
                TERMINAL_INFEASIBILITY_SOURCE_COMMIT,
            "global_execution_amendment_digest":
                GLOBAL_EXECUTION_AMENDMENT_DIGEST,
            "global_exact_model_digest": GLOBAL_EXACT_MODEL_DIGEST,
            "exact_infeasibility_digest": EXACT_INFEASIBILITY_DIGEST,
            "terminal_infeasibility_receipt_digest":
                TERMINAL_INFEASIBILITY_RECEIPT_DIGEST,
            "v1_parallel_failure_status": V1_FAILURE_STATUS,
            "v1_parallel_failure_receipt_digest":
                V1_FAILURE_RECEIPT_DIGEST,
            "v1_parallel_failure_raw_sha256": V1_FAILURE_RAW_SHA256,
            "v1_parallel_failure_byte_count": V1_FAILURE_BYTE_COUNT,
            "v1_parallel_failure_source_commit": V1_FAILURE_SOURCE_COMMIT,
            "v2_parallel_source_commit": V2_SOURCE_COMMIT,
            "v2_parallel_benchmark_contract_digest":
                V2_BENCHMARK_CONTRACT_DIGEST,
            "v2_parallel_benchmark_receipt_digest":
                V2_BENCHMARK_RECEIPT_DIGEST,
            "v2_parallel_terminal_failure_receipt_digest":
                V2_TERMINAL_FAILURE_RECEIPT_DIGEST,
            "v1_and_v2_infrastructure_failures_preserved": True,
            "six_of_twelve_disposition":
                "SUPERSEDED_PRE_OUTCOME_SIX_OF_TWELVE_ALLOCATION_EXACTLY_INFEASIBLE",
            "candidate_outcomes_existed_at_redesign": False,
            "branches_frames_latents_or_scorer_metrics_existed_at_redesign":
                False,
        },
        "single_permitted_change": {
            "predecessor_candidates_per_state": 6,
            "active_candidates_per_state": 12,
            "candidate_indices": list(range(12)),
            "rotation_or_subset_decision": None,
            "scientific_constraints_other_than_exposure_changed": False,
        },
        "candidate_outcome_or_downstream_metric_used_for_selection": False,
        "final_200_state_evaluation_corpus_authorised": False,
    }
    payload[CONTRACT_SELF_KEY] = canonical_digest(payload)
    return payload


def _validate_contract(
        value: Mapping[str, Any], *,
        validate_live_predecessor: bool,
        ) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "scorer-fit V2 contract is not an object")
    contract = dict(value)
    _require(contract.get("schema") == CONTRACT_SCHEMA
             and contract.get("status") == CONTRACT_STATUS
             and contract.get("complete") is True,
             "scorer-fit V2 contract schema/status changed")
    recorded = _require_digest(
        contract.get(CONTRACT_SELF_KEY), CONTRACT_SELF_KEY)
    unsigned = {key: item for key, item in contract.items()
                if key != CONTRACT_SELF_KEY}
    _require(recorded == canonical_digest(unsigned),
             "scorer-fit V2 contract self digest does not verify")
    if validate_live_predecessor:
        _require(contract.get("protected_predecessor_scientific_contract")
                 == _protected_predecessor_contract(),
                 "a protected predecessor scorer field changed")
    _require(contract.get("training_budget_interpretation")
             == training_budget_interpretation(),
             "larger-corpus training budget interpretation changed")
    counts = contract.get("corpus_counts")
    _require(isinstance(counts, Mapping)
             and counts.get("states") == STATE_COUNT
             and counts.get("branches") == ASSIGNMENT_COUNT
             and counts.get("fit_branches") == FIT_ROW_COUNT
             and counts.get("calibration_branches") == CALIBRATION_ROW_COUNT
             and counts.get("candidate_indices") == list(range(12)),
             "full-bank scorer corpus counts changed")
    authorities = contract.get("preoutcome_authority_bindings")
    _require(isinstance(authorities, Mapping) and set(authorities) == {
        "design_amendment", "preselection_source_correction",
        "post_install_manifest_replay_correction",
        "rotation_mask_classification",
        "small_completion_selection", "full_bank_state_revalidation",
        "state_identity_manifest", "expanded_assignment_manifest",
    }, "scorer-fit V2 authority binding schema is not closed")
    _artifact_binding(authorities["design_amendment"],
                      label="corpus V2 design",
                      self_key=DESIGN.DESIGN_SELF_KEY,
                      authority_binding=True)
    _artifact_binding(authorities["preselection_source_correction"],
                      label="corpus V2 preselection source correction",
                      self_key=DESIGN.SOURCE_CORRECTION_SELF_KEY,
                      authority_binding=True)
    _artifact_binding(authorities["post_install_manifest_replay_correction"],
                      label="corpus V2 post-install manifest replay correction",
                      self_key=DESIGN.MANIFEST_REPLAY_CORRECTION_SELF_KEY,
                      authority_binding=True)
    _artifact_binding(authorities["rotation_mask_classification"],
                      label="rotation-mask classification",
                      self_key=DESIGN.MASK_CLASSIFICATION_SELF_KEY,
                      authority_binding=True)
    _artifact_binding(authorities["small_completion_selection"],
                      label="small-completion selection",
                      self_key="full_bank_small_completion_selection_digest")
    _artifact_binding(authorities["full_bank_state_revalidation"],
                      label="full-bank state revalidation",
                      self_key="full_bank_preoutcome_state_revalidation_digest")
    _artifact_binding(authorities["state_identity_manifest"],
                      label="state manifest", self_key="state_manifest_digest")
    _artifact_binding(authorities["expanded_assignment_manifest"],
                      label="assignment manifest",
                      self_key="full_bank_assignment_manifest_digest")
    _require(contract.get(
        "candidate_outcome_or_downstream_metric_used_for_selection") is False,
        "successor contract records outcome-bearing selection")
    _require(contract.get("final_200_state_evaluation_corpus_authorised") is False,
             "successor contract unexpectedly authorises final evaluation")
    return contract


def validate_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a newly built contract against the live predecessor source."""

    return _validate_contract(value, validate_live_predecessor=True)


def build_contract_artifact(
        *, source_binding: Mapping[str, Any],
        design_binding: Mapping[str, Any],
        source_correction_binding: Mapping[str, Any],
        manifest_replay_correction_binding: Mapping[str, Any],
        mask_classification_binding: Mapping[str, Any],
        selection_binding: Mapping[str, Any],
        revalidation_binding: Mapping[str, Any],
        state_manifest_binding: Mapping[str, Any],
        assignment_manifest_binding: Mapping[str, Any],
        ) -> dict[str, Any]:
    contract = build_contract(
        source_binding=source_binding,
        design_binding=design_binding,
        source_correction_binding=source_correction_binding,
        manifest_replay_correction_binding=manifest_replay_correction_binding,
        mask_classification_binding=mask_classification_binding,
        selection_binding=selection_binding,
        revalidation_binding=revalidation_binding,
        state_manifest_binding=state_manifest_binding,
        assignment_manifest_binding=assignment_manifest_binding,
    )
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "status": CONTRACT_STATUS,
        "complete": True,
        CONTRACT_SELF_KEY: contract[CONTRACT_SELF_KEY],
        "source_repository_commit":
            source_binding["source_repository_commit"],
        "source_repository_clean": True,
        "contract": contract,
        "branch_execution_started": False,
        "candidate_outcomes_consumed": False,
        "scorer_training_started": False,
        "predictor_checkpoints_opened": False,
        "final_200_state_evaluation_corpus_authorised": False,
    }
    payload[ARTIFACT_SELF_KEY] = canonical_digest(payload)
    return payload


def _validate_contract_artifact(
        value: Mapping[str, Any], *,
        expected_contract: Mapping[str, Any] | None,
        validate_live_predecessor: bool,
        ) -> dict[str, Any]:
    _require(isinstance(value, Mapping), "scorer-fit V2 contract artifact is not an object")
    artifact = dict(value)
    _require(artifact.get("schema") == ARTIFACT_SCHEMA
             and artifact.get("status") == CONTRACT_STATUS
             and artifact.get("complete") is True,
             "scorer-fit V2 contract artifact schema/status changed")
    recorded = _require_digest(
        artifact.get(ARTIFACT_SELF_KEY), ARTIFACT_SELF_KEY)
    _require(recorded == canonical_digest({
        key: item for key, item in artifact.items() if key != ARTIFACT_SELF_KEY
    }), "scorer-fit V2 contract artifact self digest does not verify")
    contract = _validate_contract(
        artifact.get("contract", {}),
        validate_live_predecessor=validate_live_predecessor)
    _require(artifact.get(CONTRACT_SELF_KEY) == contract[CONTRACT_SELF_KEY],
             "contract artifact embeds a different contract digest")
    if expected_contract is not None:
        _require(contract == dict(expected_contract),
                 "contract artifact differs from validated producer inputs")
    for key in ("branch_execution_started", "candidate_outcomes_consumed",
                "scorer_training_started", "predictor_checkpoints_opened",
                "final_200_state_evaluation_corpus_authorised"):
        _require(artifact.get(key) is False,
                 f"pre-branch contract artifact changed at {key}")
    return artifact


def validate_contract_artifact(value: Mapping[str, Any], *,
                               expected_contract: Mapping[str, Any] | None = None,
                               ) -> dict[str, Any]:
    """Validate either the one issued artifact or a live-source build.

    Downstream consumers historically perform this pure validation after the
    correction-gated loader returns.  Route only the exact immutable artifact
    identity through historical validation so those redundant checks cannot
    reinterpret its predecessor payload under the import-only source change.
    Every other artifact remains bound to the live predecessor source.
    """

    if (isinstance(value, Mapping)
            and value.get(ARTIFACT_SELF_KEY)
            == IMMUTABLE_ISSUED_ARTIFACT_DIGEST):
        return validate_immutable_issued_contract_artifact(
            value, expected_contract=expected_contract)

    return _validate_contract_artifact(
        value, expected_contract=expected_contract,
        validate_live_predecessor=True)


def validate_immutable_issued_contract_artifact(
        value: Mapping[str, Any],
        *, expected_contract: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Validate the one historical artifact without rebinding it to live HEAD.

    This is deliberately stricter than the generic pure validator: the outer
    artifact, embedded contract, and original clean-source commit are immutable
    identities.  A compatibility correction can authorize a later import-only
    source transition, but cannot create a second scorer contract.
    """

    artifact = _validate_contract_artifact(
        value, expected_contract=expected_contract,
        validate_live_predecessor=False)
    contract = artifact["contract"]
    source = contract.get("source_binding")
    _require(
        artifact.get(ARTIFACT_SELF_KEY) == IMMUTABLE_ISSUED_ARTIFACT_DIGEST,
        "issued scorer-fit V2 contract artifact identity changed",
    )
    _require(
        contract.get(CONTRACT_SELF_KEY) == IMMUTABLE_ISSUED_CONTRACT_DIGEST
        and artifact.get(CONTRACT_SELF_KEY)
        == IMMUTABLE_ISSUED_CONTRACT_DIGEST,
        "issued scorer-fit V2 scientific contract identity changed",
    )
    _require(
        artifact.get("source_repository_commit")
        == IMMUTABLE_ISSUED_SOURCE_COMMIT
        and isinstance(source, Mapping)
        and source.get("source_repository_commit")
        == IMMUTABLE_ISSUED_SOURCE_COMMIT
        and source.get("source_repository_clean") is True
        and source.get("git_status_porcelain_v1") == ""
        and contract.get("source_binding_digest") == canonical_digest(source),
        "issued scorer-fit V2 historical clean-source binding changed",
    )
    return artifact


def _exact_output_path(path: Path, *, root: Path) -> Path:
    expected = root / ARTIFACT_RELATIVE_PATH
    requested = path if path.is_absolute() else root / path
    _require(requested == expected,
             "scorer-fit V2 contract must target its exact versioned artifact")
    _require(not any(part == "sealed" or part.startswith("sealed_")
                     or part == "sealed_test.json" for part in requested.parts),
             "scorer-fit V2 contract path crosses sealed custody")
    cursor = Path(requested.anchor)
    for part in requested.parts[1:-1]:
        cursor /= part
        _require(not cursor.is_symlink(),
                 "scorer-fit V2 contract path contains a symlink")
    return requested


def _write_exclusive_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    encoded = (json.dumps(payload, indent=2, sort_keys=True,
                          allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, flags, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as sink:
            descriptor = -1
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
        directory = os.open(path.parent, os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _active_inputs(*, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay exact producers and return their authority/manifest bundles."""

    from scripts import build_go2_branch_corpus_v1_2 as BUILDER

    authority = DESIGN.load_active_design_authority(root=root)
    manifests = BUILDER.load_and_validate_full_bank_v2_manifests_for_consumption(
        out=root / ".generated/go2_branch_corpus_v1_2/scorer_fit")
    _require(isinstance(authority, Mapping),
             "active corpus V2 design authority is not an object")
    _require(isinstance(manifests, Mapping)
             and set(manifests) == _MANIFEST_BUNDLE_KEYS,
             "full-bank manifest producer returned an unexpected schema")
    _require(manifests.get("design_authority") == authority,
             "manifest producer binds a different active design authority")
    return dict(authority), dict(manifests)


def _bindings_from_active_inputs(
        authority: Mapping[str, Any], manifests: Mapping[str, Any],
        ) -> dict[str, dict[str, Any]]:
    design = authority.get("design_amendment")
    masks = authority.get("rotation_mask_classification")
    _require(isinstance(design, Mapping) and isinstance(masks, Mapping),
             "active design authority payload is incomplete")
    return {
        "design_binding": dict(authority["design_amendment_binding"]),
        "source_correction_binding": dict(
            authority["source_correction_binding"]),
        "manifest_replay_correction_binding": dict(
            authority["manifest_replay_correction_binding"]),
        "mask_classification_binding": dict(
            authority["rotation_mask_classification_binding"]),
        "selection_binding": dict(manifests["selection_binding"]),
        "revalidation_binding": dict(manifests["revalidation_binding"]),
        "state_manifest_binding": dict(manifests["state_manifest_binding"]),
        "assignment_manifest_binding":
            dict(manifests["assignment_manifest_binding"]),
    }


def issue_contract(path: Path | None = None, *, root: Path = ROOT) -> dict[str, Any]:
    """Issue the sole prospective contract after producer replay and clean HEAD."""

    output = _exact_output_path(
        root / ARTIFACT_RELATIVE_PATH if path is None else Path(path), root=root)
    _require(not output.exists() and not output.is_symlink(),
             "refusing to overwrite or reinterpret a scorer-fit V2 contract")
    absence_before = DESIGN.audit_v2_runtime_outputs_absent(
        root=root, phase="successor_contract")
    authority, manifests = _active_inputs(root=root)
    source = clean_source_binding(root=root)
    payload = build_contract_artifact(
        source_binding=source,
        **_bindings_from_active_inputs(authority, manifests),
    )
    absence_immediate = DESIGN.audit_v2_runtime_outputs_absent(
        root=root, phase="successor_contract")
    _require(absence_before == absence_immediate,
             "V2 runtime-output absence changed before contract install")
    _write_exclusive_json(output, payload)
    _require(file_sha256(output) == hashlib.sha256(output.read_bytes()).hexdigest(),
             "written scorer-fit V2 contract bytes could not be reverified")
    return payload


def immutable_contract_artifact_binding(
        value: Mapping[str, Any], *, root: Path = ROOT,
        ) -> dict[str, Any]:
    """Return the correction authority's closed historical-contract binding."""

    artifact = validate_immutable_issued_contract_artifact(value)
    path = _exact_output_path(root / ARTIFACT_RELATIVE_PATH, root=root)
    _require(path.is_file() and not path.is_symlink(),
             "scorer-fit V2 contract artifact is missing")
    raw = path.read_bytes()
    _require(json.loads(raw) == artifact,
             "scorer-fit V2 contract artifact bytes changed")
    contract = artifact["contract"]
    return {
        "path": str(ARTIFACT_RELATIVE_PATH),
        "schema": ARTIFACT_SCHEMA,
        "self_digest_key": ARTIFACT_SELF_KEY,
        "self_digest": artifact[ARTIFACT_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "source_repository_commit": IMMUTABLE_ISSUED_SOURCE_COMMIT,
        "embedded_contract_schema": CONTRACT_SCHEMA,
        "embedded_contract_self_digest_key": CONTRACT_SELF_KEY,
        "embedded_contract_self_digest": contract[CONTRACT_SELF_KEY],
    }


def load_contract_for_consumption(
        *, root: Path = ROOT,
        encoder_path_projection_correction: Mapping[str, Any] | None = None,
        ) -> dict[str, Any]:
    """Validate the immutable contract through the active resume correction.

    The contract remains byte-for-byte the artifact issued from historical
    clean commit ``72b0d77``.  The encoder import, FP32-compute, and logical
    path-projection and branch-redrive corrections are now immutable history.
    The optional-smoke partial-corpus resume correction is the sole live-source
    gate.  It binds the complete historical correction chain and this contract
    exactly; no predecessor or manifest authority is replayed under the
    current repository commit.

    ``encoder_path_projection_correction`` remains in the public signature for
    historical callers.  A supplied value is compared byte-semantically with
    the immutable nested payload, never live-revalidated as current source.
    """

    path = _exact_output_path(root / ARTIFACT_RELATIVE_PATH, root=root)
    _require(path.is_file() and not path.is_symlink(),
             "scorer-fit V2 successor contract artifact is missing")
    artifact = validate_immutable_issued_contract_artifact(
        json.loads(path.read_text()))

    correction = (
        DESIGN.load_optional_smoke_partial_corpus_resume_correction_for_consumption(
            root=root))
    _require(isinstance(correction, Mapping),
             "partial-corpus resume correction authority is not an object")

    immutable_redrive = correction.get(
        "immutable_branch_redrive_projection_correction")
    _require(isinstance(immutable_redrive, Mapping),
             "partial-corpus resume correction has no immutable redrive "
             "correction")
    validated_immutable_redrive = (
        DESIGN.validate_immutable_branch_redrive_projection_correction(
            immutable_redrive))
    historical_redrive = validated_immutable_redrive["payload"]
    historical_redrive_binding = validated_immutable_redrive["binding"]
    _require(
        correction.get("immutable_branch_redrive_projection_correction_digest")
        == historical_redrive.get(
            DESIGN.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY)
        and historical_redrive_binding.get("self_digest_key")
        == DESIGN.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY
        and historical_redrive_binding.get("self_digest")
        == historical_redrive.get(
            DESIGN.BRANCH_REDRIVE_PROJECTION_CORRECTION_SELF_KEY),
        "immutable branch-redrive projection correction binding changed",
    )

    immutable_path = historical_redrive.get(
        "immutable_encoder_path_projection_correction")
    _require(isinstance(immutable_path, Mapping),
             "immutable branch-redrive correction has no path correction")
    validated_immutable_path = (
        DESIGN.validate_immutable_encoder_path_projection_correction(
            immutable_path))
    historical_path = validated_immutable_path["payload"]
    historical_binding = validated_immutable_path["binding"]
    _require(
        historical_redrive.get(
            "immutable_encoder_path_projection_correction_digest")
        == historical_path.get(
            DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY)
        and historical_binding.get("self_digest_key")
        == DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY
        and historical_binding.get("self_digest")
        == historical_path.get(
            DESIGN.ENCODER_PATH_PROJECTION_CORRECTION_SELF_KEY),
        "immutable encoder-path-projection correction binding changed",
    )
    if encoder_path_projection_correction is not None:
        _require(isinstance(encoder_path_projection_correction, Mapping)
                 and dict(encoder_path_projection_correction)
                 == historical_path,
                 "caller-supplied historical path correction differs from "
                 "the immutable branch-redrive lineage")

    contract_binding = immutable_contract_artifact_binding(
        artifact, root=root)
    _require(
        historical_path.get(
            "immutable_successor_scorer_contract_binding")
        == contract_binding,
        "partial-corpus resume correction binds a different immutable scorer "
        "contract",
    )
    return validate_immutable_issued_contract_artifact(artifact)


def contract_artifact_binding(value: Mapping[str, Any], *,
                              root: Path = ROOT) -> dict[str, Any]:
    artifact = validate_contract_artifact(value)
    path = _exact_output_path(root / ARTIFACT_RELATIVE_PATH, root=root)
    _require(path.is_file() and not path.is_symlink(),
             "scorer-fit V2 contract artifact is missing")
    raw = path.read_bytes()
    _require(json.loads(raw) == artifact,
             "scorer-fit V2 contract artifact bytes changed")
    return {
        "path": str(ARTIFACT_RELATIVE_PATH),
        "self_digest_key": ARTIFACT_SELF_KEY,
        "self_digest": artifact[ARTIFACT_SELF_KEY],
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


__all__ = [
    "ARTIFACT_NAME", "ARTIFACT_RELATIVE_PATH", "ARTIFACT_SCHEMA",
    "ARTIFACT_SELF_KEY", "ASSIGNMENT_COUNT", "BATCH_SIZE",
    "CALIBRATION_ROW_COUNT", "CALIBRATION_STATE_COUNT", "CANDIDATE_COUNT",
    "CONTRACT_SCHEMA", "CONTRACT_SELF_KEY", "CONTRACT_STATUS", "EPOCHS",
    "EXAMPLE_PRESENTATIONS_PER_MODEL", "FIT_ROW_COUNT", "FIT_STATE_COUNT",
    "IMMUTABLE_ISSUED_ARTIFACT_DIGEST",
    "IMMUTABLE_ISSUED_CONTRACT_DIGEST", "IMMUTABLE_ISSUED_SOURCE_COMMIT",
    "OPTIMISER_UPDATES_PER_MODEL", "PACKAGE_ROOT_RELATIVE", "STATE_COUNT",
    "ScorerFitCorpusV2ContractError", "build_contract",
    "build_contract_artifact", "canonical_digest", "clean_source_binding",
    "contract_artifact_binding", "immutable_contract_artifact_binding",
    "issue_contract",
    "load_contract_for_consumption", "training_budget_interpretation",
    "validate_contract", "validate_contract_artifact",
    "validate_immutable_issued_contract_artifact",
]
