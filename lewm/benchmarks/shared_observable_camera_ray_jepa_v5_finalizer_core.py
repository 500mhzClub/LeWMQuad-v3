"""Independent finalization of bound per-instance Shared JEPA V5 outcomes."""
from __future__ import annotations

import hashlib
import json
from typing import Mapping, Sequence

from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_runner_policy import (
    RAW_SCENE_OUTCOME_SCHEMA,
    RUNNER_LEDGER_SCHEMA,
    SyntheticRunnerBatchV6,
    _validated_synthetic_runner_batch_payload,
)


ROLE_MANIFEST_SCHEMA = "lewm_go2_shared_jepa_dataset_roles_v6"


def canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def require_sha256(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def validate_content(value: object, *, name: str) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    normalized = json.loads(canonical_bytes(dict(value)).decode("utf-8"))
    claimed = require_sha256(normalized.get("content_sha256"), f"{name} content")
    core = dict(normalized)
    del core["content_sha256"]
    if canonical_sha256(core) != claimed:
        raise ValueError(f"{name} content hash changed")
    return normalized


def _string_list(value: object, name: str) -> tuple[str, ...]:
    if (
        not isinstance(value, list)
        or not value
        or any(type(item) is not str or not item for item in value)
        or value != sorted(value)
        or len(set(value)) != len(value)
    ):
        raise ValueError(f"{name} must be a sorted unique nonempty string list")
    return tuple(value)


def _derive_gate_record_core(
    *,
    gate: str,
    metric_names: Sequence[str],
    runner_batch: SyntheticRunnerBatchV6,
    runner_payload: tuple[dict, list[dict], dict, dict],
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
    finalizer_source_sha256: str,
) -> dict[str, object]:
    """Derive counts only from immutable instance outcomes issued by the runner."""

    if gate not in {"g2", "g3"}:
        raise ValueError("gate must be g2 or g3")
    metric_tuple = tuple(metric_names)
    if not metric_tuple or len(set(metric_tuple)) != len(metric_tuple):
        raise ValueError("metric names must be nonempty and unique")
    for value, name in (
        (expected_model_state_sha256, "model state"),
        (expected_checkpoint_file_sha256, "evaluated checkpoint"),
        (expected_runner_source_sha256, "runner source"),
        (finalizer_source_sha256, "finalizer source"),
    ):
        require_sha256(value, name)
    if expected_runner_source_sha256 == finalizer_source_sha256:
        raise PermissionError("runner and finalizer source identities must be distinct")
    roles_value, outcomes, ledger_value, file_bindings = runner_payload
    if (
        runner_batch.gate != gate
        or runner_batch.expected_model_state_sha256 != expected_model_state_sha256
        or runner_batch.expected_checkpoint_file_sha256
        != expected_checkpoint_file_sha256
        or runner_batch.expected_runner_source_sha256 != expected_runner_source_sha256
    ):
        raise PermissionError("issued runner batch authority changed")

    roles = validate_content(roles_value, name="dataset role manifest")
    if set(roles) != {
        "schema",
        "protocol_generation",
        "roles",
        "scene_families",
        "content_sha256",
    } or roles.get("schema") != ROLE_MANIFEST_SCHEMA:
        raise ValueError("dataset role manifest fields changed")
    raw_roles = roles.get("roles")
    if not isinstance(raw_roles, Mapping) or set(raw_roles) != {"train", "g2", "g3"}:
        raise ValueError("dataset role inventory changed")
    normalized_roles = {
        name: _string_list(raw_roles[name], f"{name} scenes")
        for name in ("train", "g2", "g3")
    }
    all_scenes = [scene for values in normalized_roles.values() for scene in values]
    if len(all_scenes) != len(set(all_scenes)):
        raise PermissionError("dataset roles are not scene-disjoint")
    scene_families = roles.get("scene_families")
    if (
        not isinstance(scene_families, Mapping)
        or set(scene_families) != set(all_scenes)
        or any(type(value) is not str or not value for value in scene_families.values())
    ):
        raise ValueError("dataset scene-family mapping changed")
    expected_scenes = normalized_roles[gate]
    if set(file_bindings) != set(expected_scenes):
        raise PermissionError("issued scene-result inventory is incomplete")

    normalized_outcomes = [
        validate_content(outcome, name="issued raw scene outcome")
        for outcome in outcomes
    ]
    if [outcome.get("scene_id") for outcome in normalized_outcomes] != list(
        expected_scenes
    ):
        raise PermissionError("issued raw scene outcomes are incomplete or reordered")

    per_family: dict[str, dict[str, dict[str, int]]] = {}
    scene_instance_counts: dict[str, int] = {}
    for outcome in normalized_outcomes:
        if set(outcome) != {
            "schema",
            "gate",
            "scene_id",
            "family",
            "model_state_sha256",
            "evaluated_checkpoint_file_sha256",
            "runner_source_sha256",
            "instances",
            "content_sha256",
        } or outcome.get("schema") != RAW_SCENE_OUTCOME_SCHEMA:
            raise ValueError("raw scene outcome fields changed")
        scene_id = str(outcome["scene_id"])
        family = str(outcome["family"])
        if (
            outcome.get("gate") != gate
            or family != scene_families[scene_id]
            or outcome.get("model_state_sha256") != expected_model_state_sha256
            or outcome.get("evaluated_checkpoint_file_sha256")
            != expected_checkpoint_file_sha256
            or outcome.get("runner_source_sha256") != expected_runner_source_sha256
        ):
            raise PermissionError("raw scene outcome authority changed")
        instances = outcome.get("instances")
        if not isinstance(instances, list) or not instances:
            raise ValueError("raw scene outcome instances are empty")
        scene_instance_counts[scene_id] = len(instances)
        family_counts = per_family.setdefault(family, {})
        for instance in instances:
            if not isinstance(instance, Mapping):
                raise ValueError("raw inference instance changed")
            metric_outcomes = instance.get("metric_outcomes")
            if not isinstance(metric_outcomes, Mapping) or set(metric_outcomes) != set(
                metric_tuple
            ):
                raise ValueError("raw inference metric inventory changed")
            for metric_name in metric_tuple:
                passed = metric_outcomes[metric_name]
                if type(passed) is not bool:
                    raise ValueError("raw inference metric outcome must be boolean")
                aggregate = family_counts.setdefault(
                    metric_name,
                    {"numerator": 0, "denominator": 0},
                )
                aggregate["numerator"] += int(passed)
                aggregate["denominator"] += 1

    ledger = validate_content(ledger_value, name="issued runner ledger")
    if set(ledger) != {
        "schema",
        "gate",
        "dataset_role_manifest_content_sha256",
        "runner_source_sha256",
        "events",
        "content_sha256",
    } or ledger.get("schema") != RUNNER_LEDGER_SCHEMA:
        raise ValueError("runner-ledger fields changed")
    if (
        ledger.get("gate") != gate
        or ledger.get("dataset_role_manifest_content_sha256")
        != roles["content_sha256"]
        or ledger.get("runner_source_sha256") != expected_runner_source_sha256
    ):
        raise PermissionError("runner-ledger authority changed")
    events = ledger.get("events")
    if not isinstance(events, list) or len(events) != len(expected_scenes):
        raise PermissionError("runner access events are incomplete")
    for sequence, (scene_id, event) in enumerate(
        zip(expected_scenes, events, strict=True),
        start=1,
    ):
        binding = file_bindings[scene_id]
        if not isinstance(event, Mapping) or event != {
            "sequence": sequence,
            "scene_id": scene_id,
            "role": gate,
            "operation": "read_and_evaluate_canonical_scene",
            "path": binding["path"],
            "file_sha256": binding["file_sha256"],
            "instance_count": scene_instance_counts[scene_id],
            "forbidden": False,
        }:
            raise PermissionError("runner event does not reproduce actual scene opens")

    metrics: dict[str, float] = {}
    for metric_name in metric_tuple:
        numerator = sum(
            family[metric_name]["numerator"] for family in per_family.values()
        )
        denominator = sum(
            family[metric_name]["denominator"] for family in per_family.values()
        )
        metrics[metric_name] = numerator / denominator
    passed = all(value >= 1.0 for value in metrics.values())
    core: dict[str, object] = {
        "schema": f"lewm_go2_shared_jepa_{gate}_final_report_v6",
        "gate": gate,
        "passed": passed,
        "model_state_sha256": expected_model_state_sha256,
        "evaluated_checkpoint_file_sha256": expected_checkpoint_file_sha256,
        "dataset_role_manifest_content_sha256": roles["content_sha256"],
        "runner_ledger_content_sha256": ledger["content_sha256"],
        "runner_source_sha256": expected_runner_source_sha256,
        "finalizer_source_sha256": finalizer_source_sha256,
        "raw_scene_outcome_content_sha256s": [
            outcome["content_sha256"] for outcome in normalized_outcomes
        ],
        "raw_scene_outcome_files": file_bindings,
        "per_family_counts": per_family,
        "metrics": metrics,
        "zero_forbidden_access": True,
    }
    return core


def _removed_finalize_gate_records(
    *,
    gate: str,
    metric_names: Sequence[str],
    runner_batch: object,
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
    finalizer_source_sha256: str,
) -> dict[str, object]:
    """Removed: the independent one-shot finalizer reopens fixed files."""

    raise PermissionError("production library finalization was removed; use the one-shot CLI")


def _finalize_gate_records_synthetic_for_tests(
    *,
    gate: str,
    metric_names: Sequence[str],
    runner_batch: SyntheticRunnerBatchV6,
    expected_model_state_sha256: str,
    expected_checkpoint_file_sha256: str,
    expected_runner_source_sha256: str,
    finalizer_source_sha256: str,
) -> dict[str, object]:
    """Finalize the separate synthetic type without production eligibility."""

    payload = _validated_synthetic_runner_batch_payload(runner_batch)
    core = _derive_gate_record_core(
        gate=gate,
        metric_names=metric_names,
        runner_batch=runner_batch,
        runner_payload=payload,
        expected_model_state_sha256=expected_model_state_sha256,
        expected_checkpoint_file_sha256=expected_checkpoint_file_sha256,
        expected_runner_source_sha256=expected_runner_source_sha256,
        finalizer_source_sha256=finalizer_source_sha256,
    )
    synthetic_core = {
        **core,
        "synthetic_only": True,
        "production_authority_eligible": False,
    }
    return {
        **synthetic_core,
        "content_sha256": canonical_sha256(synthetic_core),
    }


__all__ = [
    "ROLE_MANIFEST_SCHEMA",
    "canonical_bytes",
    "canonical_sha256",
]
