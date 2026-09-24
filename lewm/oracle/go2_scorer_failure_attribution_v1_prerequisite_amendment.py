"""Additive fail-closed prerequisite amendment for the attentive readout.

The original c63f source closure and its interrupted diagnostic namespaces are
immutable lineage.  This module closes those two technical interruptions and
supersedes only their role as prerequisites for the already frozen, sole
prospective attentive-readout attempt.  It changes no model, data, target,
training, metric, gate, or primary threshold.  It also freezes the user's
replacement interpretation routing, including report-only family diagnostics.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping

from lewm.oracle import go2_scorer_failure_attribution_v1_contract as CONTRACT


ROOT = Path(__file__).resolve().parents[2]
STATUS = "ISSUED_FAIL_CLOSED_DIAGNOSTIC_PREREQUISITE_AMENDMENT"
SCHEMA = "go2_scorer_failure_attribution_v1_prerequisite_amendment_v1"
SELF_KEY = "prerequisite_amendment_digest"
BASE_SOURCE_COMMIT = "c63f1d53f18eda1923fc0e768a14822780fbbf5a"
BASE_SOURCE_CLOSURE_DIGEST = (
    "1a879348295212d6096125dc8247713f51d01347f22f4232c128cef32d5224d9")
DIAGNOSTIC_CONTRACT_DIGEST = (
    "152ed758162bdc0d37052c600e43a17b7558dbd9b830f106b4e4602f423c7dc7")
DIAGNOSTIC_CONTRACT_FILE_SHA256 = (
    "110ca94d267e3ec683efdcc2601172da2f85437bff415da4cbbbdc41e0f3e96f")
DIAGNOSTIC_CONTRACT_BYTE_COUNT = 26_815

AMENDMENT_NUMBER = 1
MAXIMUM_AMENDMENTS = 1
NEW_SOURCE_PATHS = (
    "lewm/oracle/go2_scorer_failure_attribution_v1_prerequisite_amendment.py",
    "lewm/tests/test_go2_scorer_failure_attribution_v1_prerequisite_amendment.py",
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
    "lewm/tests/test_train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
)
FROZEN_SOURCE_FILES = {
    "lewm/oracle/go2_scorer_failure_attribution_v1_contract.py":
        ("675de988d1f10e1a46676d7a8b89f0502e95fe680cbe51126aec40ba49361ef6", 39_358),
    "lewm/tests/test_diagnose_go2_scorer_v1_3_latent_dependence_v1.py":
        ("415174411e8bcf53a5f09024da303687434b81d8c5f8d4ea07c503b181f55b6d", 17_253),
    "lewm/tests/test_go2_scorer_failure_attribution_v1_contract.py":
        ("4f7b698b8302f743e070cc299e85362aec999413a787708193f0ed8f3b9161f5", 17_268),
    "lewm/tests/test_run_go2_safety_observability_diagnostic_v1.py":
        ("7704afe38922f8377ed496efbe50ee24018d6562a5ea42c13244a2883f0461b0", 25_852),
    "lewm/tests/test_train_go2_utility_scorer_v1_3_attentive_readout_v1.py":
        ("7d333d94d64684341ea99abf213c1e5d77c42b6213b4136a9b6d7504a8569ea7", 6_564),
    "scripts/diagnose_go2_scorer_v1_3_latent_dependence_v1.py":
        ("6e5a025f224c0efca7d51c0af5271d5a73c51b4e22c34a0c8170ed5f17bd8af3", 54_768),
    "scripts/run_go2_safety_observability_diagnostic_v1.py":
        ("940375a4019d76b41d888312677b07dd81be5ccf3836aa6a86246c93c91a7347", 92_243),
    "scripts/train_go2_utility_scorer_v1_3_attentive_readout_v1.py":
        ("c7f2bd4945a0d39264ac369469a0102caa09d3dc3d5b8fa32021bda040fcb597", 62_036),
}

SAFETY_PLAN_DIGEST = (
    "583025e4ce7c86d82456c83d979da37f3ba7dd468070e24b58942e4562453495")
SAFETY_PLAN_FILE_SHA256 = (
    "88d7858801b3effd0d359af628dfa3419dd4610c2859203d693126a99a46775c")
SAFETY_PLAN_BYTE_COUNT = 7_687_570
SAFETY_MARKER_IDENTITY_SET_DIGEST = (
    "8c562191cb11c0511774dedb1bed42ac9f9ba7d6355a2caffd72c920de14eddc")
SAFETY_MARKER_RECEIPT_SET_DIGEST = (
    "cb118e21e4c20b55cc4970d76c79fb3c7b6b748378e9c15d1e9f6e35e04893e3")
SAFETY_ROW_IDENTITY_SET_DIGEST = (
    "d264aa49d090ed2ef209cefe37f6b209510698c647b45e1b349a4877405b607e")
SAFETY_ROW_RECEIPT_SET_DIGEST = (
    "2f1f2d1cc1ec0c8f858624c0012dcfe8a48bb7c2992ffd0ccd40369082695a2d")
SAFETY_ORPHAN_IDENTITY = (
    "6021ecc564f728d49420c28bcf18d2d64eaea535f9b488593b36f20679913bdb")
SAFETY_ORPHAN_MARKER_DIGEST = (
    "eb808bad198a054ccbc04d34439de4176211a74a7fa6889f7d874c0bd3e636c1")
SAFETY_ORPHAN_MARKER_FILE_SHA256 = (
    "7a3f2a98f558cd2760c2284aa6385aa35d41cb076a8392c393f2f2cf516f54b2")
SAFETY_PRIOR_OVERLAY_DIGEST = (
    "6a6961b7852f0859d543566549c77d7b01f5bf0d7b2fee840265b6d6312460bd")
SAFETY_PRIOR_OVERLAY_FILE_SHA256 = (
    "a609905e7ff7e733f18599a8db90032429920245e41d927600a942a5c4beb59d")
SAFETY_PRIOR_TRACE_DIGEST = (
    "864984ac844d616cd5323c72dcecea4c058351e204db2e9882dbf7dcec8bcfa8")
SAFETY_TARGET_PROJECTION_DIGEST = (
    "53ae2e6135014358ed1a4b29c533103f22742224e342db7dff1b21a7ea650f4b")

LATENT_AUTHORISATION_DIGEST = (
    "b751024c4d029a0c59cf988b732ed865accd377541b1f70cb874d30c331adf88")
LATENT_AUTHORISATION_FILE_SHA256 = (
    "42d698315951653782aa0d7e4da04cc0559c0954191875987d0bbb6a60ac4285")
LATENT_AUTHORISATION_BYTE_COUNT = 188_824
LATENT_TRANSFORM_FREEZE_DIGEST = (
    "5eaaba9b034ec826d74e69e8779a66c9158bf08b4ab4d67ace3f6a03b4f05060")
LATENT_FAILURE_DIGEST = (
    "4ded2316a43ae0c2c66eaf4dcdbb19c8e0592be18c0dfe1658832e16167d283c")
LATENT_FAILURE_FILE_SHA256 = (
    "514398b3e7c03b4dc1d7221f23cb4e72daa11348e40cd1d7c7c88e94aed5acb8")
LATENT_FAILURE_BYTE_COUNT = 1_377
LATENT_TRACEBACK_SHA256 = (
    "1bdc430cfc9b93cec53b676bacc204a85bb27830ed001f65f5dbf98f4f380600")
LATENT_EXCEPTION_BINDING_SHA256 = (
    "5585db8d8894bb1298f53bced468bd25be75ad684f70dcec04651803c96d6b14")

HEX64 = re.compile(r"[0-9a-f]{64}")


class PrerequisiteAmendmentError(RuntimeError):
    """The frozen source, interruptions, or narrow authority changed."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PrerequisiteAmendmentError(message)


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_sha256(path: Path, block_size: int = 8 << 20) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(block_size), b""):
            result.update(block)
    return result.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(),
            f"{label} is absent or not a regular file")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PrerequisiteAmendmentError(f"{label} is invalid JSON") from exc
    require(isinstance(value, dict), f"{label} is not an object")
    return value


def validate_signed(value: Mapping[str, Any], self_key: str,
                    label: str) -> dict[str, Any]:
    result = dict(value)
    recorded = result.pop(self_key, None)
    require(isinstance(recorded, str) and HEX64.fullmatch(recorded) is not None
            and recorded == digest(result), f"{label} self digest changed")
    result[self_key] = recorded
    return result


def signed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    require(SELF_KEY not in result, f"{SELF_KEY} already exists")
    result[SELF_KEY] = digest(result)
    return result


def generated_root(root: Path = ROOT) -> Path:
    return root / CONTRACT.GENERATED_ROOT


def amendment_root(root: Path = ROOT) -> Path:
    return generated_root(root) / "attentive_readout_amendment_v1"


def amendment_path(root: Path = ROOT) -> Path:
    return amendment_root(root) / "prerequisite_amendment.json"


def original_attentive_root(root: Path = ROOT) -> Path:
    return generated_root(root) / "attentive_readout"


def _require_generated_root(root: Path) -> Path:
    logical = generated_root(root)
    if root.resolve() != ROOT.resolve():
        logical.mkdir(parents=True, exist_ok=True)
        return logical
    require(logical.is_symlink(), "registered output alias is absent")
    target = logical.resolve()
    require(target == CONTRACT.REGISTERED_GENERATED_TARGET_ROOT
            and target.is_dir() and not target.is_symlink(),
            "registered output target changed")
    return logical


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True,
            stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PrerequisiteAmendmentError(
            f"cannot bind clean amendment source: {exc}") from exc


def source_closure(root: Path = ROOT) -> dict[str, Any]:
    """Bind the untouched c63f closure plus the four additive source files."""

    require(_git(root, "status", "--porcelain=v1") == "",
            "amendment source must be clean and committed")
    head = _git(root, "rev-parse", "HEAD")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SOURCE_COMMIT, head],
        cwd=root, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    require(ancestor.returncode == 0,
            "amendment source does not descend from c63f")
    changed_paths = tuple(sorted(filter(None, _git(
        root, "diff", "--name-only", f"{BASE_SOURCE_COMMIT}..{head}"
    ).splitlines())))
    require(changed_paths == tuple(sorted(NEW_SOURCE_PATHS)),
            "committed amendment diff is not exactly the four additive paths")
    frozen = {}
    for relative, (expected_sha, expected_bytes) in FROZEN_SOURCE_FILES.items():
        path = root / relative
        require(path.is_file() and not path.is_symlink()
                and path.stat().st_size == expected_bytes
                and file_sha256(path) == expected_sha,
                f"frozen c63f source changed at {relative}")
        frozen[relative] = {
            "path": relative, "sha256": expected_sha,
            "byte_count": expected_bytes,
        }
    additive = {}
    for relative in NEW_SOURCE_PATHS:
        path = root / relative
        require(path.is_file() and not path.is_symlink(),
                f"additive amendment source is absent: {relative}")
        additive[relative] = {
            "path": relative, "sha256": file_sha256(path),
            "byte_count": path.stat().st_size,
        }
    payload = {
        "source_repository_commit": head,
        "source_repository_clean": True,
        "base_source_commit": BASE_SOURCE_COMMIT,
        "base_source_closure_digest": BASE_SOURCE_CLOSURE_DIGEST,
        "exact_committed_additive_path_diff": list(changed_paths),
        "frozen_base_files": frozen,
        "additive_files": additive,
    }
    return {**payload, "amendment_source_closure_digest": digest(payload)}


def _artifact(path: Path, *, relative: str, schema: str,
              self_key: str, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    value = validate_signed(read_json(path, label), self_key, label)
    require(value.get("schema") == schema, f"{label} schema changed")
    return value, {
        "path": relative,
        "schema": schema,
        "self_digest_key": self_key,
        "self_digest": value[self_key],
        "sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
    }


def _safety_interruption(root: Path) -> dict[str, Any]:
    base = generated_root(root) / "safety_observability"
    plan_path = base / "plan.json"
    plan, plan_binding = _artifact(
        plan_path, relative="safety_observability/plan.json",
        schema="go2_safety_observability_diagnostic_v1_plan_v1",
        self_key="historical_calibration_trace_plan_digest",
        label="safety-observability plan")
    require(plan_binding["self_digest"] == SAFETY_PLAN_DIGEST
            and plan_binding["sha256"] == SAFETY_PLAN_FILE_SHA256
            and plan_binding["byte_count"] == SAFETY_PLAN_BYTE_COUNT
            and plan.get("contract_digest") == DIAGNOSTIC_CONTRACT_DIGEST
            and plan.get("branch_count") == 288,
            "safety-observability plan changed")
    identities = sorted(str(row["branch_identity_digest"])
                        for row in plan["entries"])
    require(len(identities) == len(set(identities)) == 288
            and digest(identities) == SAFETY_MARKER_IDENTITY_SET_DIGEST,
            "safety marker identity set changed")
    markers = []
    rows = []
    for identity in identities:
        marker_path = base / "attempts" / f"{identity}.json"
        marker, binding = _artifact(
            marker_path, relative=f"safety_observability/attempts/{identity}.json",
            schema="go2_safety_observability_diagnostic_v1_attempt_v1",
            self_key="attempt_digest", label="safety attempt marker")
        require(marker.get("branch_identity_digest") == identity
                and marker.get("attempt_number") == 1
                and marker.get("maximum_attempts_for_identity") == 1
                and marker.get("retry_or_replacement") is False,
                "safety attempt marker changed")
        markers.append({
            "branch_identity_digest": identity,
            "attempt_digest": marker["attempt_digest"],
            "file_sha256": binding["sha256"],
        })
        row_path = base / "trace_rows" / f"{identity}.json"
        if row_path.exists() or row_path.is_symlink():
            row, row_binding = _artifact(
                row_path,
                relative=f"safety_observability/trace_rows/{identity}.json",
                schema="go2_safety_observability_diagnostic_v1_trace_row_v1",
                self_key="diagnostic_trace_row_digest",
                label="safety trace row")
            require(row.get("branch_identity_digest") == identity
                    and row.get("attempt_digest") == marker["attempt_digest"],
                    "safety trace row identity changed")
            rows.append({
                "branch_identity_digest": identity,
                "trace_row_digest": row["diagnostic_trace_row_digest"],
                "file_sha256": row_binding["sha256"],
                "target_source_kind": row[
                    "replay_aggregate_equality"]["source_kind"],
            })
    require(len(markers) == 288
            and digest(markers) == SAFETY_MARKER_RECEIPT_SET_DIGEST,
            "safety marker receipt set changed")
    row_identities = [row["branch_identity_digest"] for row in rows]
    require(len(rows) == 287
            and digest(row_identities) == SAFETY_ROW_IDENTITY_SET_DIGEST
            and digest(rows) == SAFETY_ROW_RECEIPT_SET_DIGEST,
            "safety trace-row receipt set changed")
    missing = sorted(set(identities) - set(row_identities))
    require(missing == [SAFETY_ORPHAN_IDENTITY],
            "safety interrupted identity changed")
    orphan = markers[identities.index(SAFETY_ORPHAN_IDENTITY)]
    require(orphan["attempt_digest"] == SAFETY_ORPHAN_MARKER_DIGEST
            and orphan["file_sha256"] == SAFETY_ORPHAN_MARKER_FILE_SHA256,
            "safety orphan marker changed")
    require(not (base / "terminal.json").exists()
            and not (base / "terminal.json").is_symlink()
            and not (base / "audit.json").exists()
            and not (base / "audit.json").is_symlink(),
            "interrupted safety namespace unexpectedly gained a terminal")
    overlay_path = (generated_root(root).parent
                    / "go2_scorer_fit_oracle_v1_3/replay_overlays"
                    / f"{SAFETY_ORPHAN_IDENTITY}.json")
    overlay = validate_signed(
        read_json(overlay_path, "prior orphan overlay"),
        "replay_overlay_digest", "prior orphan overlay")
    entry = next(row for row in plan["entries"]
                 if row["branch_identity_digest"] == SAFETY_ORPHAN_IDENTITY)
    require(overlay["replay_overlay_digest"] == SAFETY_PRIOR_OVERLAY_DIGEST
            and file_sha256(overlay_path) == SAFETY_PRIOR_OVERLAY_FILE_SHA256
            and digest(overlay["trace"]) == SAFETY_PRIOR_TRACE_DIGEST
            and entry["frozen_replay_target_projection_digest"]
            == SAFETY_TARGET_PROJECTION_DIGEST,
            "prior orphan overlay binding changed")
    return {
        "terminal_kind": "CLOSED_TECHNICAL_INTERRUPTION_NO_RESULT",
        "plan": plan_binding,
        "marker_count": len(markers),
        "marker_identity_set_digest": digest(identities),
        "marker_receipt_set_digest": digest(markers),
        "trace_row_count": len(rows),
        "trace_row_identity_set_digest": digest(row_identities),
        "trace_row_receipt_set_digest": digest(rows),
        "orphan": {
            "branch_identity_digest": SAFETY_ORPHAN_IDENTITY,
            "attempt_digest": orphan["attempt_digest"],
            "attempt_file_sha256": orphan["file_sha256"],
            "trace_row_absent": True,
            "prior_overlay_digest": overlay["replay_overlay_digest"],
            "prior_overlay_file_sha256": file_sha256(overlay_path),
            "prior_trace_digest": digest(overlay["trace"]),
            "frozen_target_projection_digest":
                entry["frozen_replay_target_projection_digest"],
        },
        "session_exception_evidence": {
            "evidence_status":
                "OPERATOR_OBSERVED_SESSION_EVIDENCE_NOT_ARTIFACT_BACKED",
            "exception_type": "DiagnosticError",
            "exception_message": (
                "new diagnostic replay differs from prior discrete trace evidence"),
            "source_function": "_prior_shared_field_agreement",
            "source_line": 1005,
            "candidate_index": 11,
            "canonical_field_digest": digest({
                "exception_type": "DiagnosticError",
                "exception_message": (
                    "new diagnostic replay differs from prior discrete trace evidence"),
                "source_function": "_prior_shared_field_agreement",
                "source_line": 1005,
                "candidate_index": 11,
            }),
        },
        "terminal_path_absent": True,
        "audit_path_absent": True,
        "retry_or_resume_authorised": False,
        "all_existing_bytes_preserved": True,
        "full_marker_receipts": markers,
        "full_trace_row_receipts": rows,
    }


def _latent_interruption(root: Path) -> dict[str, Any]:
    base = generated_root(root) / "latent_dependence"
    authorisation, auth_binding = _artifact(
        base / "evaluation_authorisation.json",
        relative="latent_dependence/evaluation_authorisation.json",
        schema="go2_scorer_v1_3_latent_dependence_evaluation_authorisation_v1",
        self_key="evaluation_authorisation_digest",
        label="latent evaluation authorisation")
    failure, failure_binding = _artifact(
        base / "technical_failure.json",
        relative="latent_dependence/technical_failure.json",
        schema="go2_scorer_v1_3_latent_dependence_technical_failure_v1",
        self_key="technical_failure_digest", label="latent technical failure")
    exception_bytes = (
        str(failure.get("exception_type", "")) + "\0"
        + str(failure.get("exception_message", "")) + "\0"
        + str(failure.get("traceback", ""))).encode("utf-8")
    require(auth_binding["self_digest"] == LATENT_AUTHORISATION_DIGEST
            and auth_binding["sha256"] == LATENT_AUTHORISATION_FILE_SHA256
            and auth_binding["byte_count"] == LATENT_AUTHORISATION_BYTE_COUNT
            and authorisation["transformation_freeze"]["freeze_digest"]
            == LATENT_TRANSFORM_FREEZE_DIGEST
            and failure_binding["self_digest"] == LATENT_FAILURE_DIGEST
            and failure_binding["sha256"] == LATENT_FAILURE_FILE_SHA256
            and failure_binding["byte_count"] == LATENT_FAILURE_BYTE_COUNT
            and failure.get("status")
            == "INVALID_TECHNICAL_LATENT_DEPENDENCE_SESSION"
            and failure.get("stage") == "calibration_diagnostic_session"
            and failure.get("exception_type") == "LatentDependenceError"
            and failure.get("exception_message")
            == "architecture-mandated invariance check failed"
            and hashlib.sha256(str(failure["traceback"]).encode("utf-8")).hexdigest()
            == LATENT_TRACEBACK_SHA256
            and hashlib.sha256(exception_bytes).hexdigest()
            == LATENT_EXCEPTION_BINDING_SHA256,
            "latent technical interruption changed")
    require(not (base / "result.json").exists()
            and not (base / "result.json").is_symlink(),
            "interrupted latent diagnostic unexpectedly gained a result")
    return {
        "terminal_kind": "CLOSED_TECHNICAL_FAILURE_NO_RESULT",
        "evaluation_authorisation": auth_binding,
        "transformation_freeze_digest": LATENT_TRANSFORM_FREEZE_DIGEST,
        "technical_failure": failure_binding,
        "traceback_sha256": LATENT_TRACEBACK_SHA256,
        "exception_binding": {
            "formula": "sha256(utf8(type + NUL + message + NUL + traceback))",
            "sha256": LATENT_EXCEPTION_BINDING_SHA256,
        },
        "result_path_absent": True,
        "training_executions": failure["training_executions"],
        "predictor_checkpoints_opened": failure["predictor_checkpoints_opened"],
        "predictor_utility_shards_opened":
            failure["predictor_utility_shards_opened"],
        "retry_or_resume_authorised": False,
        "all_existing_bytes_preserved": True,
    }


def _contract_binding(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    path = generated_root(root) / "diagnostic_contract.json"
    raw = read_json(path, "frozen diagnostic contract")
    value = CONTRACT.validate_contract(raw)
    require(value[CONTRACT.CONTRACT_SELF_KEY] == DIAGNOSTIC_CONTRACT_DIGEST
            and value["source_closure"][CONTRACT.SOURCE_CLOSURE_SELF_KEY]
            == BASE_SOURCE_CLOSURE_DIGEST
            and value["source_closure"]["source_repository_commit"]
            == BASE_SOURCE_COMMIT
            and file_sha256(path) == DIAGNOSTIC_CONTRACT_FILE_SHA256
            and path.stat().st_size == DIAGNOSTIC_CONTRACT_BYTE_COUNT,
            "frozen c63f diagnostic contract changed")
    sections = {}
    for key in (
            "frozen_lineage", "identity_sets", "frozen_scorers",
            "current_scorer_architecture", "safety_observability",
            "transformation_suite", "diagnostic_metrics",
            "official_attentive_pooler", "attentive_readout_architecture",
            "attentive_training", "diagnostic_prerequisites",
            "interpretation_rules", "original_gate_replay", "stopping_rules"):
        sections[key] = digest(value[key])
    return value, {
        "path": "diagnostic_contract.json",
        "schema": value["schema"],
        "digest": value[CONTRACT.CONTRACT_SELF_KEY],
        "file_sha256": file_sha256(path),
        "byte_count": path.stat().st_size,
        "source_commit": BASE_SOURCE_COMMIT,
        "source_closure_digest": BASE_SOURCE_CLOSURE_DIGEST,
        "section_digests": sections,
    }


def build_amendment(root: Path = ROOT) -> dict[str, Any]:
    _require_generated_root(root)
    require(not original_attentive_root(root).exists()
            and not original_attentive_root(root).is_symlink(),
            "original attentive_readout namespace is no longer absent")
    closure = source_closure(root)
    _contract, contract_binding = _contract_binding(root)
    safety = _safety_interruption(root)
    latent = _latent_interruption(root)
    return signed({
        "schema": SCHEMA,
        "status": STATUS,
        "complete": True,
        "amendment_number": AMENDMENT_NUMBER,
        "maximum_authorised_amendments": MAXIMUM_AMENDMENTS,
        "source_closure": closure,
        "frozen_c63f_contract": contract_binding,
        "closed_interruptions": {
            "safety_observability": safety,
            "latent_dependence": latent,
        },
        "superseded_scope": {
            "only": [
                "completed safety-observability diagnostic prerequisite",
                "completed latent-dependence diagnostic prerequisite",
                "strong/mixed/no-signal interpretation routing",
                "per-family primary consistency changed from gating to report-only",
            ],
            "scientific_contract_section_digests_unchanged": {
                key: value for key, value in contract_binding["section_digests"].items()
                if key not in {"diagnostic_prerequisites", "interpretation_rules"}
            },
            "original_interpretation_rules_digest":
                contract_binding["section_digests"]["interpretation_rules"],
            "metric_and_original_gate_thresholds_unchanged": True,
            "diagnostic_results_reinterpreted": False,
            "diagnostic_retry_resume_or_replay_authorised": False,
        },
        "prospective_authority": {
            "production_smoke": (
                "one fit-only technical smoke using the actual frozen scorer, "
                "one real microbatch update, checkpoint receipt and reload"),
            "attentive_scorer_attempt_number": 1,
            "maximum_attentive_scorer_attempts": 1,
            "training_executions": 1,
            "calibration_evaluations_after_training": 1,
            "fresh_initialisation_required": True,
            "resume_source": None,
            "original_attentive_attempt_was_consumed": False,
            "this_is_a_retry_or_replacement_scorer": False,
            "no_automatic_further_amendment_or_attempt": True,
        },
        "device_contract": {
            "selected_device": "cuda:0",
            "selected_name": "AMD Radeon AI PRO R9700",
            "selected_architecture": "gfx1201",
            "visible_hip_device_count": 2,
            "cpu_full_corpus_encoding_permitted": False,
        },
        "interpretation_amendment": {
            "strong": (
                "all original scorer criteria pass and both primary quantities "
                "strictly improve over existing ViT-L"),
            "mixed": (
                "exactly one primary threshold passes, or another original "
                "criterion fails, unless the no-signal rule applies"),
            "no_signal": (
                "neither primary threshold passes or both primary quantities "
                "degrade versus existing ViT-L"),
            "per_family_consistency_is_reported_not_gating": True,
        },
        "forbidden": {
            "predictor_checkpoint_or_utility_access": True,
            "predictor_retraining": True,
            "new_simulator_or_final_200_state_corpus": True,
            "oracle_or_label_change": True,
            "new_latent_encoding": True,
            "new_scorer_architecture_or_seed": True,
            "no_latent_baseline_retraining_or_reevaluation": True,
            "qualified_scorer_package_publication": True,
        },
        "original_attentive_namespace_absent": True,
        "new_runtime_namespace": "attentive_readout_amendment_v1",
    })


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True,
                       ensure_ascii=True, allow_nan=False) + "\n").encode("utf-8")


def _publish_once(path: Path, value: Mapping[str, Any]) -> None:
    raw = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        require(path.is_file() and not path.is_symlink()
                and path.read_bytes() == raw,
                "existing prerequisite amendment differs")
        return
    try:
        offset = 0
        while offset < len(raw):
            offset += os.write(descriptor, raw[offset:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def issue_amendment(root: Path = ROOT) -> dict[str, Any]:
    expected = build_amendment(root)
    path = amendment_path(root)
    if path.exists() or path.is_symlink():
        return validate_amendment(root)
    _publish_once(path, expected)
    return validate_amendment(root)


def validate_amendment(root: Path = ROOT) -> dict[str, Any]:
    installed = validate_signed(
        read_json(amendment_path(root), "prerequisite amendment"),
        SELF_KEY, "prerequisite amendment")
    require(installed == build_amendment(root),
            "installed prerequisite amendment does not replay")
    return installed


__all__ = [
    "AMENDMENT_NUMBER", "BASE_SOURCE_COMMIT", "DIAGNOSTIC_CONTRACT_DIGEST",
    "MAXIMUM_AMENDMENTS", "NEW_SOURCE_PATHS", "PrerequisiteAmendmentError",
    "SCHEMA", "SELF_KEY", "STATUS", "amendment_path", "amendment_root",
    "build_amendment", "canonical_bytes", "digest", "file_sha256",
    "issue_amendment", "source_closure", "validate_amendment",
]
