"""Focused tests for the additive diagnostic-prerequisite amendment."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from lewm.oracle import (
    go2_scorer_failure_attribution_v1_prerequisite_amendment as AMENDMENT,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / (
    "lewm/oracle/go2_scorer_failure_attribution_v1_prerequisite_amendment.py")


def _function_source(name: str) -> str:
    source = SOURCE.read_text()
    node = next(value for value in ast.parse(source).body
                if isinstance(value, ast.FunctionDef) and value.name == name)
    return "\n".join(source.splitlines()[node.lineno - 1:node.end_lineno])


def test_exact_c63f_and_four_additive_paths_are_frozen() -> None:
    assert AMENDMENT.BASE_SOURCE_COMMIT == (
        "c63f1d53f18eda1923fc0e768a14822780fbbf5a")
    assert len(AMENDMENT.FROZEN_SOURCE_FILES) == 8
    assert AMENDMENT.NEW_SOURCE_PATHS == (
        "lewm/oracle/go2_scorer_failure_attribution_v1_prerequisite_amendment.py",
        "lewm/tests/test_go2_scorer_failure_attribution_v1_prerequisite_amendment.py",
        "scripts/train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
        "lewm/tests/test_train_go2_utility_scorer_v1_3_attentive_readout_amendment_v1.py",
    )
    source = _function_source("source_closure")
    assert '"diff", "--name-only"' in source
    assert "changed_paths == tuple(sorted(NEW_SOURCE_PATHS))" in source
    assert '"status", "--porcelain=v1"' in source


def test_full_interrupted_namespace_receipt_digests_are_unabbreviated() -> None:
    assert AMENDMENT.SAFETY_MARKER_IDENTITY_SET_DIGEST == (
        "8c562191cb11c0511774dedb1bed42ac9f9ba7d6355a2caffd72c920de14eddc")
    assert AMENDMENT.SAFETY_MARKER_RECEIPT_SET_DIGEST == (
        "cb118e21e4c20b55cc4970d76c79fb3c7b6b748378e9c15d1e9f6e35e04893e3")
    assert AMENDMENT.SAFETY_ROW_IDENTITY_SET_DIGEST == (
        "d264aa49d090ed2ef209cefe37f6b209510698c647b45e1b349a4877405b607e")
    assert AMENDMENT.SAFETY_ROW_RECEIPT_SET_DIGEST == (
        "2f1f2d1cc1ec0c8f858624c0012dcfe8a48bb7c2992ffd0ccd40369082695a2d")
    assert AMENDMENT.LATENT_TRANSFORM_FREEZE_DIGEST == (
        "5eaaba9b034ec826d74e69e8779a66c9158bf08b4ab4d67ace3f6a03b4f05060")
    assert AMENDMENT.LATENT_EXCEPTION_BINDING_SHA256 == (
        "5585db8d8894bb1298f53bced468bd25be75ad684f70dcec04651803c96d6b14")


def test_marker_and_row_receipt_formulas_are_closed() -> None:
    marker = [{
        "branch_identity_digest": "a" * 64,
        "attempt_digest": "b" * 64,
        "file_sha256": "c" * 64,
    }]
    row = [{
        "branch_identity_digest": "a" * 64,
        "trace_row_digest": "d" * 64,
        "file_sha256": "e" * 64,
        "target_source_kind": "FROZEN",
    }]
    expected_marker = hashlib.sha256(json.dumps(
        marker, sort_keys=True, separators=(",", ":"),
        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()
    assert AMENDMENT.digest(marker) == expected_marker
    assert AMENDMENT.digest(row) != AMENDMENT.digest(marker)
    safety = _function_source("_safety_interruption")
    assert "digest(markers) == SAFETY_MARKER_RECEIPT_SET_DIGEST" in safety
    assert "digest(rows) == SAFETY_ROW_RECEIPT_SET_DIGEST" in safety


def test_exception_binding_formula_uses_nul_delimiters() -> None:
    failure = {
        "exception_type": "LatentDependenceError",
        "exception_message": "architecture-mandated invariance check failed",
        "traceback": "trace",
    }
    raw = (failure["exception_type"] + "\0" + failure["exception_message"]
           + "\0" + failure["traceback"]).encode("utf-8")
    assert hashlib.sha256(raw).hexdigest() == hashlib.sha256(
        b"LatentDependenceError\0architecture-mandated invariance check failed"
        b"\0trace").hexdigest()
    source = _function_source("_latent_interruption")
    assert 'failure.get("exception_type", "")) + "\\0"' in source
    assert "LATENT_EXCEPTION_BINDING_SHA256" in source


def test_only_prerequisites_and_interpretation_routing_are_superseded() -> None:
    source = _function_source("build_amendment")
    for required in (
        "completed safety-observability diagnostic prerequisite",
        "completed latent-dependence diagnostic prerequisite",
        "strong/mixed/no-signal interpretation routing",
        "per-family primary consistency changed from gating to report-only",
        "metric_and_original_gate_thresholds_unchanged",
    ):
        assert required in source
    assert 'if key not in {"diagnostic_prerequisites", "interpretation_rules"}' \
        in source
    assert "diagnostic_retry_resume_or_replay_authorised" in source


def test_amendment_is_one_shot_and_preserves_original_namespace_absence() -> None:
    assert AMENDMENT.AMENDMENT_NUMBER == 1
    assert AMENDMENT.MAXIMUM_AMENDMENTS == 1
    source = _function_source("build_amendment")
    assert "not original_attentive_root(root).exists()" in source
    assert '"original_attentive_attempt_was_consumed": False' in source
    assert '"maximum_attentive_scorer_attempts": 1' in source
    assert '"no_automatic_further_amendment_or_attempt": True' in source


def test_signed_amendment_validation_fails_closed() -> None:
    value = AMENDMENT.signed({"schema": "synthetic", "complete": True})
    assert AMENDMENT.validate_signed(
        value, AMENDMENT.SELF_KEY, "synthetic") == value
    tampered = dict(value)
    tampered["complete"] = False
    with pytest.raises(AMENDMENT.PrerequisiteAmendmentError):
        AMENDMENT.validate_signed(tampered, AMENDMENT.SELF_KEY, "synthetic")


def test_no_predictor_or_sealed_route_exists() -> None:
    source = SOURCE.read_text().lower()
    imports = "\n".join(line for line in source.splitlines()
                        if line.lstrip().startswith(("from ", "import ")))
    assert "predictor" not in imports
    assert "sealed" not in source
    assert '"predictor_checkpoint_or_utility_access": true' in source
