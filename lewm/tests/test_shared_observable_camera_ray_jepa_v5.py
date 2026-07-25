from __future__ import annotations

import copy
from dataclasses import FrozenInstanceError
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from lewm.models import shared_observable_camera_ray_jepa_v5 as shared_v5

from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import OUTPUT_SHAPE
from lewm.models.egomotion_bev_jepa import EgomotionBevJepa
from lewm.models.encoders import VisionEncoder
from lewm.models.observable_camera_ray_evidence_v4 import (
    IMAGE_SIZE,
    ObservableCameraRayEvidenceV4Model,
    ObservableCameraRayEvidenceV4RawOutput,
)
from lewm.models.observable_camera_ray_evidence_v4_training import (
    HierarchicalRasterCrossEntropyV4,
)
from lewm.models.shared_observable_camera_ray_jepa_v5 import (
    CheckpointProvenanceV5,
    EstablishedJepaPackageV5,
    G2_GATE_REPORT_V5_SCHEMA,
    G3_GATE_REPORT_V5_SCHEMA,
    LIFECYCLE_G3_CANDIDATE,
    LIFECYCLE_PROMOTED,
    MODEL_FAMILY,
    ObservableCameraRayEvidenceV4Head,
    ACCESS_LEDGER_V5_SCHEMA,
    DATASET_ROLE_MANIFEST_V5_SCHEMA,
    EVALUATION_PROTOCOL_V5_SCHEMA,
    PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA,
    PRODUCTION_MODEL_CONFIG_V5_SCHEMA,
    ObservableCameraRayV4FrameSupervisionV5,
    SharedHierarchicalV4LossV5,
    SharedObservableCameraRayV4FrameLossV5,
    SharedObservableCameraRayV4LossV5,
    SharedObservableCameraRayJepaV5,
    SharedObservableCameraRayJepaV5Config,
    SharedOnlineFrameV5,
    SharedTrainingPairV5,
    RAW_GATE_RESULT_V5_SCHEMA,
    ROLE_COMMITMENT_V5_SCHEMA,
    SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
    checkpoint_contract_bindings_v5,
    gate_thresholds_v5,
    shared_output_contract_v5,
    tensor_state_dict_sha256,
    TRAINING_RUN_V5_SCHEMA,
)

# The first block exercises the pure architecture/checkpoint shape contract.
# Production entry points below are tested separately and never accept these
# in-memory mappings.
build_checkpoint_v5_payload = (
    shared_v5._build_checkpoint_v5_payload_structure_only_for_tests
)
validate_checkpoint_v5_payload = (
    shared_v5._validate_checkpoint_v5_payload_structure_only_for_tests
)
checkpoint_v5_weights_only_roundtrip = (
    shared_v5._checkpoint_v5_weights_only_roundtrip_structure_only_for_tests
)
ProductionCheckpointContextV5 = object


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode("ascii")).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def test_tensor_state_dict_sha256_is_scalar_safe_and_legacy_compatible() -> None:
    scalar_state = {"step": torch.tensor(17, dtype=torch.int64)}
    first = tensor_state_dict_sha256(scalar_state)
    assert first == tensor_state_dict_sha256(scalar_state)
    assert first != tensor_state_dict_sha256(
        {"step": torch.tensor(18, dtype=torch.int64)}
    )
    assert len(first) == 64

    non_scalar_state = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "count": torch.tensor([3, 5], dtype=torch.int64),
    }
    assert tensor_state_dict_sha256(non_scalar_state) == (
        tensor_state_dict_sha256(dict(reversed(tuple(non_scalar_state.items()))))
    )
    legacy = hashlib.sha256()
    for name in sorted(non_scalar_state):
        tensor = non_scalar_state[name].detach().cpu().contiguous()
        header = {
            "name": name,
            "dtype": str(tensor.dtype),
            "shape": list(tensor.shape),
        }
        encoded = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("ascii")
        legacy.update(len(encoded).to_bytes(8, "little"))
        legacy.update(encoded)
        legacy.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    assert tensor_state_dict_sha256(non_scalar_state) == legacy.hexdigest()

    batch_norm = torch.nn.BatchNorm1d(4)
    tracked = batch_norm.state_dict()["num_batches_tracked"]
    assert tracked.ndim == 0 and tracked.dtype == torch.int64
    tracked_hash = tensor_state_dict_sha256({"num_batches_tracked": tracked})
    batch_norm.num_batches_tracked.add_(1)
    assert tracked_hash != tensor_state_dict_sha256(
        {"num_batches_tracked": batch_norm.num_batches_tracked}
    )


def _calibration(batch: int = 1):
    origin = torch.tensor((0.326, 0.02, 0.043))[None].expand(batch, -1).clone()
    basis = torch.tensor(
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0), (0.0, 0.0, 1.0))
    )[None].expand(batch, -1, -1).clone()
    ground = torch.full((batch,), -0.35)
    return origin, basis, ground


def _dev_config(
    *,
    canonical_v4: bool = False,
    canonical_bev: bool = False,
) -> SharedObservableCameraRayJepaV5Config:
    return SharedObservableCameraRayJepaV5Config(
        schema=SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA,
        encoder_depth=0,
        action_dim=3,
        bev_dim=8,
        bev_size=OUTPUT_SHAPE if canonical_bev else (4, 4),
        predictor_hidden_dim=12,
        target_ema_momentum=0.5,
        source_shape=(128, 128) if canonical_v4 else (2, 3),
        pixel_ray_shape=(84, 112) if canonical_v4 else (3, 4),
        query_chunk_size=256 if canonical_v4 else 5,
        v4_pixel_ray_chunk_size=32,
    )


def _model(**kwargs) -> SharedObservableCameraRayJepaV5:
    return SharedObservableCameraRayJepaV5(_dev_config(**kwargs))


def _pair(
    model: SharedObservableCameraRayJepaV5,
    *,
    batch: int = 2,
) -> SharedTrainingPairV5:
    current = torch.randn(batch, 3, IMAGE_SIZE, IMAGE_SIZE)
    next_image = torch.randn_like(current)
    origin, basis, ground = _calibration(batch)
    action = torch.zeros(batch, model.action_dim)
    action[:, 0] = 1.0
    if model.action_dim > 1:
        action[1::2, 0] = 0.0
        action[1::2, 1] = 1.0
    wrong_action = torch.roll(action, shifts=1, dims=1)
    realized = torch.zeros(batch, 3)
    realized[:, 0] = 0.05
    commanded = torch.zeros(batch, 3)
    commanded[:, 0] = 0.10
    wrong_delta = torch.zeros(batch, 3)
    wrong_delta[:, 1] = 0.20
    return model.forward_training_pair(
        current,
        next_image,
        action,
        realized,
        commanded_delta_pose_current=commanded,
        current_camera_origin_body_m=origin,
        current_camera_basis_body_fru=basis,
        current_ground_plane_z_body_m=ground,
        next_camera_origin_body_m=origin,
        next_camera_basis_body_fru=basis,
        next_ground_plane_z_body_m=ground,
        diagnostic_wrong_action=wrong_action,
        diagnostic_wrong_action_delta_pose_current=wrong_delta,
        diagnostic_wrong_commanded_delta_pose_current=wrong_delta,
    )


def _v4_supervision(
    frame: SharedOnlineFrameV5,
) -> ObservableCameraRayV4FrameSupervisionV5:
    hazard = frame.evidence.pixel_first_hit_hazard_logits
    pixel_shape = (hazard.shape[0], hazard.shape[2], hazard.shape[3])
    hit = torch.zeros(pixel_shape, dtype=torch.bool, device=hazard.device)
    hit[:, 0, 0] = True
    distance = torch.zeros(pixel_shape, dtype=hazard.dtype, device=hazard.device)
    distance[hit] = shared_v5.DEPTH_NEAR_EDGE_M + 0.25 * shared_v5.DEPTH_BIN_SIZE_M
    in_frustum = frame.evidence.ground_query_in_frustum.detach().clone()
    parity = torch.arange(
        in_frustum.numel(), device=in_frustum.device
    ).reshape(in_frustum.shape) % 2 == 0
    clear = in_frustum & parity
    labels = torch.zeros(
        (hazard.shape[0], *OUTPUT_SHAPE),
        dtype=torch.long,
        device=hazard.device,
    )
    labels[:, 12:40, 16:48] = 1
    labels[:, 28:32, 30:34] = 2
    return ObservableCameraRayV4FrameSupervisionV5(
        pixel_hit_mask=hit,
        pixel_first_hit_distance_m=distance,
        ground_support_in_frustum=in_frustum,
        ground_support_clear_to_target=clear,
        target_raster_labels=labels,
    )


def _provenance() -> CheckpointProvenanceV5:
    return CheckpointProvenanceV5(
        dataset_manifest_sha256=_sha("dataset"),
        corpus_plan_sha256=_sha("corpus"),
        geometry_contract_sha256=_sha("geometry"),
        camera_calibration_sha256=_sha("camera"),
        implementation_sha256=_sha("implementation"),
        fit_gate_report_sha256=_sha("fit-gate"),
        v4_fit_checkpoint_sha256=_sha("v4-fit-checkpoint"),
        training_run_sha256=_sha("training-run"),
        gate_attempt_registry_source_sha256=_sha("attempt-registry-source"),
        g2_role_commitment_sha256=_sha("g2-role"),
        g3_role_commitment_sha256=_sha("g3-role"),
        g2_evaluation_protocol_sha256=_sha("g2-protocol"),
        g3_evaluation_protocol_sha256=_sha("g3-protocol"),
        g2_finalizer_source_sha256=_sha("g2-finalizer"),
        g3_finalizer_source_sha256=_sha("g3-finalizer"),
        training_scene_ids=("scene_a", "scene_b"),
    )


def _report(
    gate: str,
    model: SharedObservableCameraRayJepaV5,
    provenance: CheckpointProvenanceV5,
) -> dict:
    bindings = checkpoint_contract_bindings_v5(model, provenance)
    thresholds = gate_thresholds_v5(gate)
    metrics = {name: 1.0 for name in thresholds["metrics"]}
    checks = {name: True for name in sorted(metrics)}
    metrics_sha = _canonical_hash(metrics)
    thresholds_sha = _canonical_hash(thresholds)
    decision_sha = _canonical_hash(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_decision_v5",
            "metrics_sha256": metrics_sha,
            "thresholds_sha256": thresholds_sha,
            "checks": checks,
            "passed": True,
        }
    )
    role_sha = getattr(provenance, f"{gate}_role_commitment_sha256")
    protocol_sha = getattr(provenance, f"{gate}_evaluation_protocol_sha256")
    attempt_id = _canonical_hash(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_attempt_identity_v5",
            "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
            "gate_role_commitment_sha256": role_sha,
            "evaluation_protocol_sha256": protocol_sha,
        }
    )
    attempt = {
        "schema": f"lewm_go2_shared_jepa_{gate}_attempt_registry_v5",
        "gate": gate,
        "status": "consumed_once_finalized",
        "attempt_id_sha256": attempt_id,
        "attempt_registry_source_sha256": (
            provenance.gate_attempt_registry_source_sha256
        ),
        "dataset_manifest_sha256": provenance.dataset_manifest_sha256,
        "gate_role_commitment_sha256": role_sha,
        "evaluation_protocol_sha256": protocol_sha,
        "threshold_contract_sha256": thresholds_sha,
        "model_state_sha256": bindings["model_state_sha256"],
        "reserved_before_payload_access": True,
        "prior_attempt_count": 0,
    }
    attempt["content_sha256"] = _canonical_hash(attempt)
    finalizer = {
        "schema": f"lewm_go2_shared_jepa_{gate}_finalizer_v5",
        "gate": gate,
        "status": "independently_finalized",
        "finalizer_source_sha256": getattr(
            provenance, f"{gate}_finalizer_source_sha256"
        ),
        "implementation_sha256": provenance.implementation_sha256,
        "attempt_registry_content_sha256": attempt["content_sha256"],
        "raw_result_content_sha256": _sha(f"{gate}-raw-result"),
        "metrics_sha256": metrics_sha,
        "thresholds_sha256": thresholds_sha,
        "decision_sha256": decision_sha,
    }
    finalizer["content_sha256"] = _canonical_hash(finalizer)
    report = {
        "schema": (
            G2_GATE_REPORT_V5_SCHEMA if gate == "g2" else G3_GATE_REPORT_V5_SCHEMA
        ),
        "gate": gate,
        "passed": True,
        "model_family": MODEL_FAMILY,
        "metrics": metrics,
        "thresholds": thresholds,
        "checks": checks,
        "decision_sha256": decision_sha,
        "attempt_registry": attempt,
        "finalizer": finalizer,
        **bindings,
    }
    report["content_sha256"] = _canonical_hash(report)
    return report


def _refresh_report_content(report: dict) -> str:
    content = dict(report)
    content.pop("content_sha256", None)
    report["content_sha256"] = _canonical_hash(content)
    return report["content_sha256"]


def _refinalize_report(report: dict) -> str:
    gate = report["gate"]
    metrics_sha = _canonical_hash(report["metrics"])
    thresholds_sha = _canonical_hash(report["thresholds"])
    report["checks"] = {
        name: float(report["metrics"][name])
        >= float(report["thresholds"]["metrics"][name]["value"])
        for name in sorted(report["metrics"])
    }
    report["decision_sha256"] = _canonical_hash(
        {
            "schema": f"lewm_go2_shared_jepa_{gate}_decision_v5",
            "metrics_sha256": metrics_sha,
            "thresholds_sha256": thresholds_sha,
            "checks": report["checks"],
            "passed": True,
        }
    )
    attempt = report["attempt_registry"]
    attempt["content_sha256"] = _canonical_hash(
        {key: value for key, value in attempt.items() if key != "content_sha256"}
    )
    finalizer = report["finalizer"]
    finalizer["attempt_registry_content_sha256"] = attempt["content_sha256"]
    finalizer["metrics_sha256"] = metrics_sha
    finalizer["thresholds_sha256"] = thresholds_sha
    finalizer["decision_sha256"] = report["decision_sha256"]
    finalizer["content_sha256"] = _canonical_hash(
        {key: value for key, value in finalizer.items() if key != "content_sha256"}
    )
    return _refresh_report_content(report)


def _candidate_payload():
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()
    g2 = _report("g2", model, provenance)
    return build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        provenance=provenance,
        g2_report=g2,
        g2_report_content_sha256=g2["content_sha256"],
    )


def _content_value(core: dict) -> dict:
    return {**core, "content_sha256": _canonical_hash(core)}


def _write_artifact_json(
    root: Path,
    relative: str,
    core: dict,
    *,
    write: bool = True,
) -> tuple[dict[str, str], dict]:
    value = _content_value(core)
    encoded = _canonical_bytes(value) + b"\n"
    path = root / relative
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(encoded)
    return (
        {
            "root": "artifact",
            "path": relative,
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
        },
        value,
    )


def _write_artifact_bytes(
    root: Path,
    relative: str,
    encoded: bytes,
) -> dict[str, str]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)
    return {
        "root": "artifact",
        "path": relative,
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _access_ledger_core(
    role: str,
    scene_ids: list[str],
    *,
    producer_source_file_sha256: str,
    forbidden_accesses: list[str] | None = None,
) -> dict:
    forbidden_accesses = [] if forbidden_accesses is None else forbidden_accesses
    return {
        "schema": ACCESS_LEDGER_V5_SCHEMA,
        "role": role,
        "allowed_scene_ids": scene_ids,
        "opened_scene_ids": scene_ids,
        "forbidden_roles": ["heldout", "sealed"],
        "forbidden_accesses": forbidden_accesses,
        "all_accesses_authorized": not forbidden_accesses,
        "producer_source_file_sha256": producer_source_file_sha256,
    }


def _raw_result_core(
    *,
    gate: str,
    model_state_sha256: str,
    dataset_content_sha256: str,
    role_content_sha256: str,
    protocol_content_sha256: str,
    ledger_content_sha256: str,
) -> dict:
    metric_names = (
        shared_v5.G2_GATE_METRICS_V5
        if gate == "g2"
        else shared_v5.G3_GATE_METRICS_V5
    )
    return {
        "schema": RAW_GATE_RESULT_V5_SCHEMA,
        "gate": gate,
        "model_state_sha256": model_state_sha256,
        "dataset_manifest_content_sha256": dataset_content_sha256,
        "role_commitment_content_sha256": role_content_sha256,
        "evaluation_protocol_content_sha256": protocol_content_sha256,
        "access_ledger_content_sha256": ledger_content_sha256,
        "families": [
            {
                "family": family,
                "metrics": {
                    name: {"numerator": numerator, "denominator": 2}
                    for name in metric_names
                },
            }
            for family, numerator in (("family_a", 2), ("family_b", 2))
        ],
    }


def _filesystem_evidence_context(
    tmp_path: Path,
    model: SharedObservableCameraRayJepaV5,
    *,
    lifecycle: str = LIFECYCLE_G3_CANDIDATE,
    artifact_root: Path | None = None,
    registry_root: Path | None = None,
    prior_payload: dict | None = None,
    prior_context: ProductionCheckpointContextV5 | None = None,
    omit_raw_result: bool = False,
    training_scene_ids: list[str] | None = None,
    mismatched_source_role: str | None = None,
    gate_forbidden_accesses: list[str] | None = None,
) -> ProductionCheckpointContextV5:
    artifact_root = (
        (tmp_path / "artifacts").resolve()
        if artifact_root is None
        else artifact_root.resolve()
    )
    registry_root = (
        (tmp_path / "registry").resolve()
        if registry_root is None
        else registry_root.resolve()
    )
    artifact_root.mkdir(parents=True, exist_ok=True)
    registry_root.mkdir(parents=True, exist_ok=True)
    for gate_name in ("g2", "g3"):
        (registry_root / gate_name).mkdir(exist_ok=True)
    gate = "g2" if lifecycle == LIFECYCLE_G3_CANDIDATE else "g3"
    generation = "synthetic_protocol_generation_v1"
    roles = {
        "train": ["train_scene_a", "train_scene_b"],
        "g2": ["g2_scene_a", "g2_scene_b"],
        "g3": ["g3_scene_a", "g3_scene_b"],
    }
    model_state_sha256 = tensor_state_dict_sha256(model.deployment_state_dict())
    artifacts: dict[str, dict[str, str]] = {}

    source_bytes = Path(shared_v5.__file__).read_bytes()
    for source_role in (
        "implementation",
        "attempt_registry_source",
        "g2_finalizer_source",
        "g3_finalizer_source",
    ):
        encoded = (
            b"synthetic mismatched source\n"
            if mismatched_source_role == source_role
            else source_bytes
        )
        artifacts[source_role] = _write_artifact_bytes(
            artifact_root,
            f"sources/{source_role}.py",
            encoded,
        )
    registry_source_sha = artifacts["attempt_registry_source"]["file_sha256"]

    dataset_spec, dataset = _write_artifact_json(
        artifact_root,
        "metadata/dataset_roles.json",
        {"schema": DATASET_ROLE_MANIFEST_V5_SCHEMA, "roles": roles},
    )
    artifacts["dataset_manifest"] = dataset_spec
    for name in (
        "corpus_plan",
        "geometry_contract",
        "camera_calibration",
        "fit_gate_report",
        "v4_fit_checkpoint",
    ):
        artifacts[name] = _write_artifact_bytes(
            artifact_root,
            f"metadata/{name}.bin",
            f"synthetic {name}\n".encode("ascii"),
        )

    role_values: dict[str, dict] = {}
    protocol_values: dict[str, dict] = {}
    for current_gate in ("g2", "g3"):
        role_spec, role_value = _write_artifact_json(
            artifact_root,
            f"metadata/{current_gate}_role.json",
            {
                "schema": ROLE_COMMITMENT_V5_SCHEMA,
                "gate": current_gate,
                "dataset_manifest_content_sha256": dataset["content_sha256"],
                "protocol_generation": generation,
                "scene_ids": roles[current_gate],
                "forbidden_roles": ["heldout", "sealed"],
            },
        )
        protocol_spec, protocol_value = _write_artifact_json(
            artifact_root,
            f"metadata/{current_gate}_protocol.json",
            {
                "schema": EVALUATION_PROTOCOL_V5_SCHEMA,
                "gate": current_gate,
                "generation": generation,
                "metric_names": list(
                    shared_v5.G2_GATE_METRICS_V5
                    if current_gate == "g2"
                    else shared_v5.G3_GATE_METRICS_V5
                ),
                "thresholds": gate_thresholds_v5(current_gate),
            },
        )
        artifacts[f"{current_gate}_role_commitment"] = role_spec
        artifacts[f"{current_gate}_evaluation_protocol"] = protocol_spec
        role_values[current_gate] = role_value
        protocol_values[current_gate] = protocol_value

    training_ledger_spec, training_ledger = _write_artifact_json(
        artifact_root,
        "metadata/training_access_ledger.json",
        _access_ledger_core(
            "train",
            roles["train"],
            producer_source_file_sha256=registry_source_sha,
        ),
    )
    artifacts["training_access_ledger"] = training_ledger_spec
    training_run_spec, _training_run = _write_artifact_json(
        artifact_root,
        "metadata/training_run.json",
        {
            "schema": TRAINING_RUN_V5_SCHEMA,
            "dataset_manifest_content_sha256": dataset["content_sha256"],
            "training_scene_ids": (
                roles["train"]
                if training_scene_ids is None
                else training_scene_ids
            ),
            "training_access_ledger_content_sha256": training_ledger[
                "content_sha256"
            ],
            "model_state_sha256": model_state_sha256,
        },
    )
    artifacts["training_run"] = training_run_spec

    gate_ledger_spec, gate_ledger = _write_artifact_json(
        artifact_root,
        f"evidence/{gate}_access_ledger.json",
        _access_ledger_core(
            gate,
            roles[gate],
            producer_source_file_sha256=registry_source_sha,
            forbidden_accesses=gate_forbidden_accesses,
        ),
    )
    artifacts[f"{gate}_access_ledger"] = gate_ledger_spec
    raw_spec, _raw = _write_artifact_json(
        artifact_root,
        f"evidence/{gate}_raw_result.json",
        _raw_result_core(
            gate=gate,
            model_state_sha256=model_state_sha256,
            dataset_content_sha256=dataset["content_sha256"],
            role_content_sha256=role_values[gate]["content_sha256"],
            protocol_content_sha256=protocol_values[gate]["content_sha256"],
            ledger_content_sha256=gate_ledger["content_sha256"],
        ),
        write=not omit_raw_result,
    )
    artifacts[f"{gate}_raw_result"] = raw_spec

    if lifecycle == LIFECYCLE_PROMOTED:
        if prior_payload is None or prior_context is None:
            raise ValueError("promoted synthetic evidence requires prior G2")
        prior_namespace = prior_payload["registry_namespace_sha256"]
        artifacts["prior_g2_report"] = {
            "root": "registry",
            "path": prior_payload["g2_report_path"],
            "file_sha256": prior_payload["g2_report_file_sha256"],
        }
        artifacts["prior_g2_finalized"] = {
            "root": "registry",
            "path": f"g2/{prior_namespace}/finalized.json",
            "file_sha256": prior_payload["g2_registry_finalized_file_sha256"],
        }
        artifacts["prior_g2_authority_manifest"] = {
            "root": "artifact",
            "path": prior_context.authority_manifest_path,
            "file_sha256": prior_context.authority_manifest_file_sha256,
        }

    authority_relative = f"authority/{gate}_authority.json"
    authority_spec, _authority = _write_artifact_json(
        artifact_root,
        authority_relative,
        {
            "schema": PRODUCTION_AUTHORITY_MANIFEST_V5_SCHEMA,
            "lifecycle": lifecycle,
            "gate": gate,
            "protocol_generation": generation,
            "artifacts": artifacts,
        },
    )
    return ProductionCheckpointContextV5(
        artifact_root=artifact_root,
        registry_root=registry_root,
        authority_manifest_path=authority_relative,
        authority_manifest_file_sha256=authority_spec["file_sha256"],
    )


def test_encoder_free_head_owns_no_visual_encoder() -> None:
    head = ObservableCameraRayEvidenceV4Head(
        source_shape=(2, 3), pixel_ray_shape=(3, 4)
    )
    assert not any(isinstance(module, VisionEncoder) for module in head.modules())


def test_training_has_one_online_and_one_frozen_ema_target_encoder() -> None:
    model = _model()
    assert isinstance(model, EgomotionBevJepa)
    encoders = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, VisionEncoder)
    ]
    assert encoders == [("encoder", model.encoder), ("target_encoder", model.target_encoder)]
    model.train()
    assert model.encoder.training
    assert not model.target_encoder.training
    assert not model.target_bev_decoder.training
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert all(
        not parameter.requires_grad for parameter in model.target_bev_decoder.parameters()
    )


def test_shared_frame_calls_only_one_online_encoder_and_no_target() -> None:
    model = _model()
    image = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
    calibration = _calibration(2)
    with mock.patch.object(
        model.encoder,
        "forward_tokens",
        wraps=model.encoder.forward_tokens,
    ) as online_call, mock.patch.object(
        model.target_encoder,
        "forward_tokens",
        wraps=model.target_encoder.forward_tokens,
    ) as target_call:
        result = model.forward_frame(image, *calibration)
    assert online_call.call_count == 1
    assert target_call.call_count == 0
    assert isinstance(result, SharedOnlineFrameV5)
    assert isinstance(result.evidence, ObservableCameraRayEvidenceV4RawOutput)
    assert result.patch_tokens.shape == (2, 256, 192)
    assert result.bev.shape == (2, 8, 4, 4)


def test_pair_uses_one_batched_online_call_and_one_no_grad_target_call() -> None:
    model = _model()
    with mock.patch.object(
        model.encoder,
        "forward_tokens",
        wraps=model.encoder.forward_tokens,
    ) as online_call, mock.patch.object(
        model.target_encoder,
        "forward_tokens",
        wraps=model.target_encoder.forward_tokens,
    ) as target_call:
        result = _pair(model)
    assert online_call.call_count == 1
    assert online_call.call_args.args[0].shape[0] == 4
    assert target_call.call_count == 1
    assert target_call.call_args.args[0].shape[0] == 2
    assert isinstance(result, SharedTrainingPairV5)
    assert not result.stop_gradient_target_next_bev.requires_grad
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_established_loss_and_health_package_is_complete() -> None:
    model = _model()
    pair = _pair(model)
    package = pair.jepa
    assert isinstance(package, EstablishedJepaPackageV5)
    expected = (
        model.jepa_weight * package.prediction
        + model.equivariance_weight * package.equivariance
        + model.action_contrast_weight * package.action_contrast
        + model.variance_weight * package.variance
    )
    torch.testing.assert_close(package.total, expected)
    for value in (
        package.prediction,
        package.equivariance,
        package.action_contrast,
        package.variance,
        package.warped_persistence,
        package.prediction_to_persistence_ratio,
        package.target_cross_sample_std_mean,
        package.target_cross_sample_effective_rank,
    ):
        assert torch.isfinite(value).all()
    counterfactuals = package.counterfactuals
    assert counterfactuals.wrong_action_contrast_loss is not None
    assert counterfactuals.zero_action_contrast_loss is not None
    assert counterfactuals.wrong_action_advantage_over_target_change is not None
    assert (
        counterfactuals.wrong_commanded_delta_advantage_over_target_change
        is not None
    )


def test_complete_v4_and_established_jepa_reach_shared_encoder() -> None:
    torch.manual_seed(11)
    model = _model(canonical_v4=True)
    pair = _pair(model, batch=1)
    current = _v4_supervision(pair.current)
    next_frame = _v4_supervision(pair.next)
    diagnostic = model.hierarchical_v4_loss(
        pair,
        current.target_raster_labels,
        next_frame.target_raster_labels,
    )
    assert isinstance(diagnostic, SharedHierarchicalV4LossV5)
    assert isinstance(diagnostic.current, HierarchicalRasterCrossEntropyV4)
    v4 = model.observable_camera_ray_v4_loss(pair, current, next_frame)
    assert isinstance(v4, SharedObservableCameraRayV4LossV5)
    assert isinstance(v4.current, SharedObservableCameraRayV4FrameLossV5)
    for frame_loss in (v4.current, v4.next):
        torch.testing.assert_close(
            frame_loss.total,
            0.25
            * (
                frame_loss.ordered_first_hit_nll
                + frame_loss.target_bin_offset_smooth_l1
                + frame_loss.ground_clear_distance_state_balanced_bce
                + frame_loss.derived_raster_hierarchical_bce.total
            ),
        )
        for component in (
            frame_loss.ordered_first_hit_nll,
            frame_loss.target_bin_offset_smooth_l1,
            frame_loss.ground_clear_distance_state_balanced_bce,
            frame_loss.derived_raster_hierarchical_bce.total,
        ):
            assert torch.isfinite(component)
    parameter = model.encoder.patch_embed.weight
    jepa_gradient = torch.autograd.grad(
        pair.jepa.total,
        parameter,
        retain_graph=True,
    )[0]
    v4_gradient = torch.autograd.grad(v4.total, parameter, retain_graph=True)[0]
    assert torch.isfinite(jepa_gradient).all() and torch.count_nonzero(jepa_gradient)
    assert torch.isfinite(v4_gradient).all() and torch.count_nonzero(v4_gradient)
    pixel_parameter = model.evidence_head.pixel_head.weight
    for component in (
        v4.current.ordered_first_hit_nll,
        v4.current.target_bin_offset_smooth_l1,
    ):
        gradient = torch.autograd.grad(
            component,
            pixel_parameter,
            retain_graph=True,
        )[0]
        assert torch.isfinite(gradient).all() and torch.count_nonzero(gradient)
    ground_parameter = model.evidence_head.ground_head[-1].weight
    ground_gradient = torch.autograd.grad(
        v4.current.ground_clear_distance_state_balanced_bce,
        ground_parameter,
        retain_graph=True,
    )[0]
    assert torch.isfinite(ground_gradient).all() and torch.count_nonzero(
        ground_gradient
    )
    joint = model.combine_joint_losses(pair, current, next_frame)
    torch.testing.assert_close(
        joint.total,
        pair.jepa.total + model.model_config.observable_camera_ray_v4_weight * v4.total,
    )
    joint.total.backward()
    assert parameter.grad is not None and torch.count_nonzero(parameter.grad)
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())


def test_output_contract_matches_runtime_tensor_axes_and_complete_v4_loss() -> None:
    model = _model()
    contract = shared_output_contract_v5(model)
    fields = contract["evidence_fields"]
    expected_pixel = [
        "B",
        shared_v5.DEPTH_BIN_COUNT,
        *model.model_config.pixel_ray_shape,
    ]
    assert fields["pixel_first_hit_hazard_logits"] == expected_pixel
    assert fields["pixel_within_bin_offset_m"] == expected_pixel
    frame = model.forward_frame(
        torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        *_calibration(),
    )
    assert list(frame.evidence.pixel_first_hit_hazard_logits.shape) == [
        1,
        shared_v5.DEPTH_BIN_COUNT,
        *model.model_config.pixel_ray_shape,
    ]
    assert contract["v4_supervision"]["required_components"] == [
        "ordered_first_hit_nll",
        "target_bin_offset_smooth_l1",
        "ground_clear_distance_state_balanced_bce",
        "derived_raster_hierarchical_bce",
    ]


def test_explicit_post_step_ema_updates_encoder_and_decoder_and_stays_eval() -> None:
    model = _model()
    encoder_before = model.target_encoder.patch_embed.weight.detach().clone()
    decoder_before = model.target_bev_decoder.query_bias.detach().clone()
    with torch.no_grad():
        model.encoder.patch_embed.weight.add_(2.0)
        model.bev_decoder.query_bias.add_(4.0)
    model.train()
    model.update_ema_target_after_optimizer_step()
    torch.testing.assert_close(
        model.target_encoder.patch_embed.weight, encoder_before + 1.0
    )
    torch.testing.assert_close(
        model.target_bev_decoder.query_bias, decoder_before + 2.0
    )
    assert not model.target_encoder.training
    assert not model.target_bev_decoder.training
    with pytest.raises(RuntimeError, match="after optimizer.step"):
        model.update_target_encoder()


def test_training_rejects_missing_counterfactual_package() -> None:
    model = _model()
    current = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    calibration = _calibration()
    with pytest.raises(ValueError, match="requires wrong-action"):
        model.forward_training_pair(
            current,
            torch.randn_like(current),
            torch.tensor(((1.0, 0.0, 0.0),)),
            torch.zeros(1, 3),
            commanded_delta_pose_current=torch.zeros(1, 3),
            current_camera_origin_body_m=calibration[0],
            current_camera_basis_body_fru=calibration[1],
            current_ground_plane_z_body_m=calibration[2],
            next_camera_origin_body_m=calibration[0],
            next_camera_basis_body_fru=calibration[1],
            next_ground_plane_z_body_m=calibration[2],
        )


def test_fit_model_migration_is_exact_and_hard_syncs_target() -> None:
    torch.manual_seed(17)
    fit = ObservableCameraRayEvidenceV4Model(
        encoder_depth=0,
        source_shape=(2, 3),
        pixel_ray_shape=(3, 4),
        query_chunk_size=5,
    ).eval()
    shared = _model().eval()
    receipt = shared.migrate_from_fit_model(fit)
    image = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    calibration = _calibration()
    with torch.no_grad():
        expected = fit(image, *calibration)
        observed = shared.forward_frame(image, *calibration).evidence
    for name in (
        "pixel_first_hit_hazard_logits",
        "pixel_within_bin_offset_m",
        "ground_clear_to_target_logits",
        "ground_query_uv_px",
        "ground_target_distance_m",
    ):
        torch.testing.assert_close(
            getattr(observed, name), getattr(expected, name), rtol=0.0, atol=0.0
        )
    assert torch.equal(observed.ground_query_in_frustum, expected.ground_query_in_frustum)
    assert receipt.shared_encoder_state_sha256 == tensor_state_dict_sha256(
        fit.encoder.state_dict()
    )
    assert tensor_state_dict_sha256(shared.target_encoder.state_dict()) == (
        tensor_state_dict_sha256(shared.encoder.state_dict())
    )


def test_deployment_export_excludes_every_training_only_module() -> None:
    model = _model()
    full = model.state_dict()
    deployment = model.deployment_state_dict()
    assert any(name.startswith("target_encoder.") for name in full)
    assert any(name.startswith("target_bev_decoder.") for name in full)
    assert all(
        name.startswith(("encoder.", "bev_decoder.", "evidence_head."))
        for name in deployment
    )
    assert all(
        not name.startswith(
            ("target_encoder.", "target_bev_decoder.", "predictor.", "occupancy_head.")
        )
        for name in deployment
    )
    restored = _model()
    restored.load_deployment_state_dict(deployment)
    assert tensor_state_dict_sha256(restored.deployment_state_dict()) == (
        tensor_state_dict_sha256(deployment)
    )


def test_deterministic_deployment_buffers_are_complete_and_bit_exact() -> None:
    model = _model()
    fresh = _model()
    keys = {
        name
        for name, _value in model.named_buffers()
        if name.startswith(("encoder.", "bev_decoder.", "evidence_head."))
        and name in model.state_dict()
    }
    expected = {
        "bev_decoder.coordinate_features",
        "evidence_head.canonical_ground_support_xy_body_m",
    }
    assert keys == expected
    model_state = model.deployment_state_dict()
    fresh_state = fresh.deployment_state_dict()
    for name in expected:
        assert torch.equal(model_state[name], fresh_state[name])

    for name in expected:
        damaged = _model()
        buffer = dict(damaged.named_buffers())[name]
        with torch.no_grad():
            buffer.reshape(-1)[0] += 1.0
        with pytest.raises(ValueError, match="deterministic buffer"):
            damaged.deployment_state_dict()


def test_config_is_complete_canonical_and_strictly_roundtrips() -> None:
    config = _dev_config(canonical_v4=True, canonical_bev=True)
    assert SharedObservableCameraRayJepaV5Config.from_mapping(config.to_dict()) == config
    assert config.content_sha256
    with pytest.raises(FrozenInstanceError):
        config.bev_dim = 9  # type: ignore[misc]
    missing = config.to_dict()
    missing.pop("target_ema_momentum")
    with pytest.raises(ValueError, match="fields changed"):
        SharedObservableCameraRayJepaV5Config.from_mapping(missing)
    extra = config.to_dict()
    extra["unknown"] = 1
    with pytest.raises(ValueError, match="fields changed"):
        SharedObservableCameraRayJepaV5Config.from_mapping(extra)
    wrong_type = config.to_dict()
    wrong_type["bev_size"] = tuple(wrong_type["bev_size"])
    with pytest.raises(ValueError, match="must be a list"):
        SharedObservableCameraRayJepaV5Config.from_mapping(wrong_type)


def test_production_schema_freezes_every_reviewed_default() -> None:
    production = SharedObservableCameraRayJepaV5Config()
    assert production.schema == PRODUCTION_MODEL_CONFIG_V5_SCHEMA
    assert production.encoder_depth == 6
    assert production.bev_dim == 64 and production.bev_size == (64, 64)
    assert production.action_dim == 9 and production.predictor_hidden_dim == 128
    assert production.target_ema_momentum == 0.996
    assert production.query_chunk_size == 4096
    assert production.v4_pixel_ray_chunk_size == 256
    assert production.forward_range_m == (-0.95, 5.35)
    assert production.left_range_m == (-3.15, 3.15)

    alternatives = {
        "encoder_depth": 5,
        "bev_dim": 32,
        "bev_size": [32, 32],
        "action_dim": 8,
        "predictor_hidden_dim": 64,
        "target_ema_momentum": 0.99,
        "jepa_weight": 2.0,
        "equivariance_weight": 0.5,
        "action_contrast_weight": 2.0,
        "action_margin_fraction": 0.2,
        "variance_weight": 0.2,
        "variance_target_std": 0.75,
        "query_chunk_size": 2048,
        "v4_pixel_ray_chunk_size": 128,
        "observable_camera_ray_v4_weight": 2.0,
        "forward_range_m": [-1.0, 5.3],
        "left_range_m": [-3.0, 3.0],
        "normalization_mean": [0.4, 0.45, 0.5],
        "normalization_std": [0.2, 0.21, 0.22],
        "projective_attention_sigma_tokens": 1,
        "projective_attention_bias_floor": -6,
    }
    for name, alternative in alternatives.items():
        changed = production.to_dict()
        changed[name] = alternative
        with pytest.raises(PermissionError, match="frozen defaults"):
            SharedObservableCameraRayJepaV5Config.from_mapping(changed)

    for kwargs in (
        {"jepa_weight": "1.0"},
        {"target_ema_momentum": "0.996"},
        {"normalization_mean": list(production.normalization_mean)},
        {"forward_range_m": list(production.forward_range_m)},
    ):
        with pytest.raises(PermissionError, match="frozen defaults at input"):
            SharedObservableCameraRayJepaV5Config(**kwargs)  # type: ignore[arg-type]


def test_checkpoint_candidate_promoted_and_weights_only_roundtrip() -> None:
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()
    g2 = _report("g2", model, provenance)
    candidate = build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        provenance=provenance,
        g2_report=g2,
        g2_report_content_sha256=g2["content_sha256"],
    )
    restored = checkpoint_v5_weights_only_roundtrip(
        candidate, expected_lifecycle=LIFECYCLE_G3_CANDIDATE
    )
    assert restored["runtime_ready"] is False
    assert restored["g3_report"] is None
    assert tensor_state_dict_sha256(restored["model_state_dict"]) == candidate[
        "model_state_sha256"
    ]
    assert all("target_" not in name for name in restored["model_state_dict"])

    with pytest.raises(PermissionError, match="requires a passing G3"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_PROMOTED,
            provenance=provenance,
            g2_report=g2,
            g2_report_content_sha256=g2["content_sha256"],
        )
    g3 = _report("g3", model, provenance)
    promoted = build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_PROMOTED,
        provenance=provenance,
        g2_report=g2,
        g2_report_content_sha256=g2["content_sha256"],
        g3_report=g3,
        g3_report_content_sha256=g3["content_sha256"],
    )
    validate_checkpoint_v5_payload(promoted, expected_lifecycle=LIFECYCLE_PROMOTED)
    assert promoted["runtime_ready"] is True


def _refresh_state_binding(payload: dict) -> None:
    state_sha = tensor_state_dict_sha256(payload["model_state_dict"])
    payload["model_state_sha256"] = state_sha
    payload["g2_report"]["model_state_sha256"] = state_sha
    attempt = payload["g2_report"]["attempt_registry"]
    attempt["model_state_sha256"] = state_sha
    attempt["content_sha256"] = _canonical_hash(
        {key: value for key, value in attempt.items() if key != "content_sha256"}
    )
    finalizer = payload["g2_report"]["finalizer"]
    finalizer["attempt_registry_content_sha256"] = attempt["content_sha256"]
    finalizer["content_sha256"] = _canonical_hash(
        {key: value for key, value in finalizer.items() if key != "content_sha256"}
    )
    payload["g2_report_content_sha256"] = _refresh_report_content(
        payload["g2_report"]
    )


def test_checkpoint_rejects_all_identity_contract_and_state_mutations() -> None:
    payload = _candidate_payload()
    mutations = []

    damaged = copy.deepcopy(payload)
    damaged["runtime_ready"] = True
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["schema"] = "wrong"
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_family"] = "wrong"
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["unexpected"] = None
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_config"].pop("normalization_std")
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_config"]["schema"] = "wrong"
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_config"]["target_ema_momentum"] = 0.1
    damaged["model_config_sha256"] = _sha("forged-config")
    damaged["g2_report"]["model_config_sha256"] = _sha("forged-config")
    damaged["g2_report_content_sha256"] = _refresh_report_content(
        damaged["g2_report"]
    )
    mutations.append(damaged)

    for contract, hash_name, field in (
        ("output_contract", "output_contract_sha256", "v4_supervision"),
        ("v4_geometry", "v4_geometry_sha256", "physical_target_inflation_m"),
        ("bev_geometry", "bev_geometry_sha256", "lift_type"),
        ("architecture_contract", "architecture_contract_sha256", "online_vision_encoder_count"),
    ):
        damaged = copy.deepcopy(payload)
        damaged[contract][field] = "wrong" if field != "online_vision_encoder_count" else 2
        damaged[hash_name] = _sha(f"forged-{contract}")
        report_name = hash_name if hash_name in damaged["g2_report"] else None
        if report_name is not None:
            damaged["g2_report"][report_name] = damaged[hash_name]
            damaged["g2_report_content_sha256"] = _refresh_report_content(
                damaged["g2_report"]
            )
        mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["provenance"]["training_run_sha256"] = _sha("other-run")
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["provenance"]["schema"] = "wrong"
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["v4_geometry"]["evidence_schema"] = "wrong"
    mutations.append(damaged)

    for receipt_field in (
        "model_state_sha256",
        "dataset_manifest_sha256",
        "model_config_sha256",
        "architecture_contract_sha256",
        "output_contract_sha256",
        "provenance_contract_sha256",
        "implementation_sha256",
        "v4_geometry_sha256",
        "bev_geometry_sha256",
    ):
        damaged = copy.deepcopy(payload)
        damaged["g2_report"][receipt_field] = _sha(f"wrong-{receipt_field}")
        damaged["g2_report_content_sha256"] = _refresh_report_content(
            damaged["g2_report"]
        )
        mutations.append(damaged)

    state_name = next(
        name
        for name, value in payload["model_state_dict"].items()
        if value.is_floating_point() and value.numel() > 1
    )
    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"][state_name].reshape(-1)[0] += 1.0
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"][state_name] = damaged["model_state_dict"][
        state_name
    ].reshape(-1)[:-1]
    _refresh_state_binding(damaged)
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"][state_name] = damaged["model_state_dict"][state_name].double()
    _refresh_state_binding(damaged)
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"][state_name].reshape(-1)[0] = float("nan")
    _refresh_state_binding(damaged)
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"]["target_encoder.forbidden"] = torch.zeros(1)
    _refresh_state_binding(damaged)
    mutations.append(damaged)

    damaged = copy.deepcopy(payload)
    damaged["model_state_dict"].pop(state_name)
    _refresh_state_binding(damaged)
    mutations.append(damaged)

    for buffer_name in (
        "bev_decoder.coordinate_features",
        "evidence_head.canonical_ground_support_xy_body_m",
    ):
        damaged = copy.deepcopy(payload)
        damaged["model_state_dict"][buffer_name].reshape(-1)[0] += 1.0
        _refresh_state_binding(damaged)
        mutations.append(damaged)

    for index, damaged in enumerate(mutations):
        with pytest.raises((ValueError, PermissionError), match="."):
            validate_checkpoint_v5_payload(damaged)


def test_synthetic_only_config_cannot_build_candidate_or_promoted_checkpoint() -> None:
    config = SharedObservableCameraRayJepaV5Config(
        schema=SYNTHETIC_ONLY_MODEL_CONFIG_V5_SCHEMA
    )
    model = SharedObservableCameraRayJepaV5(config)
    provenance = _provenance()
    g2 = _report("g2", model, provenance)
    with pytest.raises(PermissionError, match="requires production config"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=g2,
            g2_report_content_sha256=g2["content_sha256"],
        )


def test_gate_report_rejects_false_pass_bad_content_and_wrong_binding() -> None:
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()
    report = _report("g2", model, provenance)

    false_pass = copy.deepcopy(report)
    false_pass["passed"] = False
    false_pass_hash = _refresh_report_content(false_pass)
    with pytest.raises(PermissionError, match="does not record a pass"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=false_pass,
            g2_report_content_sha256=false_pass_hash,
        )

    mutated = copy.deepcopy(report)
    mutated["implementation_sha256"] = _sha("wrong-implementation")
    with pytest.raises(ValueError, match="content hash changed"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=mutated,
            g2_report_content_sha256=report["content_sha256"],
        )

    rebound_hash = _refresh_report_content(mutated)
    with pytest.raises(PermissionError, match="another candidate"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=mutated,
            g2_report_content_sha256=rebound_hash,
        )

    wrong_schema = _report("g2", model, provenance)
    wrong_schema["schema"] = G3_GATE_REPORT_V5_SCHEMA
    wrong_schema_hash = _refresh_report_content(wrong_schema)
    with pytest.raises(ValueError, match="identity changed"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=wrong_schema,
            g2_report_content_sha256=wrong_schema_hash,
        )

    extra_field = _report("g2", model, provenance)
    extra_field["unexpected"] = None
    extra_field_hash = _refresh_report_content(extra_field)
    with pytest.raises(ValueError, match="fields changed"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=extra_field,
            g2_report_content_sha256=extra_field_hash,
        )


def test_gate_report_mapping_is_snapshotted_and_caller_hash_is_required() -> None:
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()
    report = _report("g2", model, provenance)
    original_hash = report["content_sha256"]
    payload = build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        provenance=provenance,
        g2_report=report,
        g2_report_content_sha256=original_hash,
    )
    report["passed"] = False
    assert payload["g2_report"]["passed"] is True
    validate_checkpoint_v5_payload(payload)

    fresh_report = _report("g2", model, provenance)
    with pytest.raises(ValueError, match="content hash changed"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=fresh_report,
            g2_report_content_sha256=_sha("wrong-caller-content"),
        )


def test_gate_pass_cannot_be_self_asserted_without_outcome_and_one_shot_chain() -> None:
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()

    failed_metrics = _report("g2", model, provenance)
    metric_name = next(iter(failed_metrics["metrics"]))
    failed_metrics["metrics"][metric_name] = 0.0
    failed_hash = _refinalize_report(failed_metrics)
    with pytest.raises(PermissionError, match="do not pass frozen thresholds"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=failed_metrics,
            g2_report_content_sha256=failed_hash,
        )

    repeated_attempt = _report("g2", model, provenance)
    repeated_attempt["attempt_registry"]["prior_attempt_count"] = 1
    repeated_hash = _refinalize_report(repeated_attempt)
    with pytest.raises(PermissionError, match="one-shot finalization"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=repeated_attempt,
            g2_report_content_sha256=repeated_hash,
        )

    forged_finalizer = _report("g2", model, provenance)
    forged_finalizer["finalizer"]["finalizer_source_sha256"] = _sha(
        "forged-finalizer"
    )
    forged_hash = _refinalize_report(forged_finalizer)
    with pytest.raises(PermissionError, match="finalizer_source_sha256"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=forged_finalizer,
            g2_report_content_sha256=forged_hash,
        )


def test_gate_report_canonical_files_are_hashed_and_bound(
    tmp_path: Path,
) -> None:
    model = SharedObservableCameraRayJepaV5()
    provenance = _provenance()
    g2 = _report("g2", model, provenance)
    g2_path = tmp_path / "g2.json"
    g2_bytes = _canonical_bytes(g2) + b"\n"
    g2_path.write_bytes(g2_bytes)
    g2_file_sha = hashlib.sha256(g2_bytes).hexdigest()
    candidate = build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        provenance=provenance,
        g2_report=g2_path,
        g2_report_content_sha256=g2["content_sha256"],
        g2_report_file_sha256=g2_file_sha,
    )
    assert candidate["g2_report_file_sha256"] == g2_file_sha
    assert candidate["g2_report_source"] == "canonical_json_file"

    g3 = _report("g3", model, provenance)
    g3_path = tmp_path / "g3.json"
    g3_bytes = _canonical_bytes(g3) + b"\n"
    g3_path.write_bytes(g3_bytes)
    g3_file_sha = hashlib.sha256(g3_bytes).hexdigest()
    promoted = build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_PROMOTED,
        provenance=provenance,
        g2_report=g2,
        g2_report_content_sha256=g2["content_sha256"],
        g3_report=g3_path,
        g3_report_content_sha256=g3["content_sha256"],
        g3_report_file_sha256=g3_file_sha,
    )
    validate_checkpoint_v5_payload(promoted, expected_lifecycle=LIFECYCLE_PROMOTED)

    g2_path.write_bytes(g2_bytes + b" ")
    with pytest.raises(ValueError, match="file hash changed"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=g2_path,
            g2_report_content_sha256=g2["content_sha256"],
            g2_report_file_sha256=g2_file_sha,
        )

    pretty_bytes = json.dumps(g2, indent=2, sort_keys=True).encode("utf-8")
    g2_path.write_bytes(pretty_bytes)
    with pytest.raises(ValueError, match="not canonical JSON"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=g2_path,
            g2_report_content_sha256=g2["content_sha256"],
            g2_report_file_sha256=hashlib.sha256(pretty_bytes).hexdigest(),
        )

    damaged = copy.deepcopy(candidate)
    damaged["g2_report_file_sha256"] = _sha("wrong-embedded-file")
    with pytest.raises(ValueError, match="embedded gate-report file hash changed"):
        validate_checkpoint_v5_payload(damaged)

    damaged = copy.deepcopy(candidate)
    damaged["g2_report_file_sha256"] = None
    with pytest.raises(ValueError, match="lost its file hash"):
        validate_checkpoint_v5_payload(damaged)

    link = tmp_path / "g2-link.json"
    link.symlink_to(g2_path)
    with pytest.raises(ValueError, match="regular non-symlink"):
        build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            provenance=provenance,
            g2_report=link,
            g2_report_content_sha256=g2["content_sha256"],
            g2_report_file_sha256=hashlib.sha256(pretty_bytes).hexdigest(),
        )


def test_joint_loss_rejects_shallow_external_scalar() -> None:
    model = _model()
    pair = _pair(model)
    with pytest.raises(TypeError, match="ObservableCameraRayV4FrameSupervisionV5"):
        model.combine_joint_losses(  # type: ignore[arg-type]
            pair,
            torch.tensor(math.nan),
            torch.tensor(math.nan),
        )


def _legacy_removed_production_checkpoint_api_rejects_in_memory_evidence() -> None:
    model = SharedObservableCameraRayJepaV5()
    with pytest.raises(TypeError, match="production checkpoint context"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context={"g2_report": _report("g2", model, _provenance())},  # type: ignore[arg-type]
        )
    with pytest.raises(PermissionError, match="filesystem evidence"):
        shared_v5.validate_checkpoint_v5_payload(_candidate_payload())
    with pytest.raises(PermissionError, match="filesystem evidence"):
        shared_v5.checkpoint_v5_weights_only_roundtrip(_candidate_payload())


def _legacy_removed_production_filesystem_evidence_builds_candidate_and_promoted(
    tmp_path: Path,
) -> None:
    model = SharedObservableCameraRayJepaV5()
    g2_context = _filesystem_evidence_context(tmp_path, model)
    candidate = shared_v5.build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        context=g2_context,
    )
    assert candidate["runtime_ready"] is False
    assert candidate["g2_report_source"] == "filesystem_registry"
    assert candidate["g2_report"]["metrics"] == {
        name: 1.0 for name in shared_v5.G2_GATE_METRICS_V5
    }
    assert candidate["g2_report"]["access_ledger"]["forbidden_accesses"] == []
    assert candidate["g2_report"]["attempt_registry"]["status"] == (
        "role_global_filesystem_finalized"
    )
    assert candidate["g2_report"]["finalizer"]["status"] == (
        "independently_reconstructed_from_raw_counts"
    )
    shared_v5.validate_checkpoint_v5_payload(candidate, context=g2_context)

    g3_context = _filesystem_evidence_context(
        tmp_path,
        model,
        lifecycle=LIFECYCLE_PROMOTED,
        artifact_root=g2_context.artifact_root,
        registry_root=g2_context.registry_root,
        prior_payload=candidate,
        prior_context=g2_context,
    )
    promoted = shared_v5.build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_PROMOTED,
        context=g3_context,
    )
    assert promoted["runtime_ready"] is True
    assert promoted["g2_report"] == candidate["g2_report"]
    assert promoted["g3_report"]["metrics"] == {
        name: 1.0 for name in shared_v5.G3_GATE_METRICS_V5
    }
    shared_v5.validate_checkpoint_v5_payload(
        promoted,
        expected_lifecycle=LIFECYCLE_PROMOTED,
        context=g3_context,
    )
    restored = shared_v5.checkpoint_v5_weights_only_roundtrip(
        promoted,
        expected_lifecycle=LIFECYCLE_PROMOTED,
        context=g3_context,
    )
    assert restored["model_state_sha256"] == promoted["model_state_sha256"]


def _legacy_removed_production_validator_reopens_raw_result_instead_of_trusting_report(
    tmp_path: Path,
) -> None:
    model = SharedObservableCameraRayJepaV5()
    context = _filesystem_evidence_context(tmp_path, model)
    candidate = shared_v5.build_checkpoint_v5_payload(
        model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        context=context,
    )
    damaged = copy.deepcopy(candidate)
    raw = damaged["g2_report"]["raw_result"]
    metric = shared_v5.G2_GATE_METRICS_V5[0]
    raw["families"][0]["metrics"][metric]["numerator"] = 1
    raw["content_sha256"] = _canonical_hash(
        {name: value for name, value in raw.items() if name != "content_sha256"}
    )
    report = damaged["g2_report"]
    report["content_sha256"] = _canonical_hash(
        {name: value for name, value in report.items() if name != "content_sha256"}
    )
    damaged["g2_report_content_sha256"] = report["content_sha256"]
    with pytest.raises(PermissionError, match="reopened raw files"):
        shared_v5.validate_checkpoint_v5_payload(damaged, context=context)


def _legacy_removed_role_global_reservation_covers_two_distinct_model_states(
    tmp_path: Path,
) -> None:
    first_model = SharedObservableCameraRayJepaV5()
    context = _filesystem_evidence_context(tmp_path, first_model)
    first = shared_v5.build_checkpoint_v5_payload(
        first_model,
        lifecycle=LIFECYCLE_G3_CANDIDATE,
        context=context,
    )
    second_model = SharedObservableCameraRayJepaV5()
    with torch.no_grad():
        next(second_model.encoder.parameters()).view(-1)[0].add_(1.0)
    assert tensor_state_dict_sha256(second_model.deployment_state_dict()) != first[
        "model_state_sha256"
    ]
    with pytest.raises(PermissionError, match="already reserved"):
        shared_v5.build_checkpoint_v5_payload(
            second_model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=context,
        )


def _legacy_removed_missing_raw_result_consumes_role_before_any_retry(tmp_path: Path) -> None:
    model = SharedObservableCameraRayJepaV5()
    context = _filesystem_evidence_context(
        tmp_path,
        model,
        omit_raw_result=True,
    )
    with pytest.raises(ValueError, match="g2_raw_result could not be opened safely"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=context,
        )
    reservations = list((context.registry_root / "g2").glob("*/reservation.json"))
    assert len(reservations) == 1
    with pytest.raises(PermissionError, match="already reserved"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=context,
        )


def _legacy_removed_non_train_scene_ids_and_forbidden_gate_access_fail_closed(
    tmp_path: Path,
) -> None:
    model = SharedObservableCameraRayJepaV5()
    non_train_context = _filesystem_evidence_context(
        tmp_path / "non_train",
        model,
        training_scene_ids=["g2_scene_a", "train_scene_b"],
    )
    with pytest.raises(PermissionError, match="training-run provenance changed"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=non_train_context,
        )

    forbidden_context = _filesystem_evidence_context(
        tmp_path / "forbidden",
        model,
        gate_forbidden_accesses=["sealed_scene_a"],
    )
    with pytest.raises(PermissionError, match="admits forbidden or missing access"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=forbidden_context,
        )


@pytest.mark.parametrize(
    "source_role",
    (
        "implementation",
        "attempt_registry_source",
        "g2_finalizer_source",
        "g3_finalizer_source",
    ),
)
def _legacy_removed_mismatched_authorized_source_identity_fails_closed(
    tmp_path: Path,
    source_role: str,
) -> None:
    model = SharedObservableCameraRayJepaV5()
    context = _filesystem_evidence_context(
        tmp_path / source_role,
        model,
        mismatched_source_role=source_role,
    )
    with pytest.raises(PermissionError, match=f"authorized {source_role}"):
        shared_v5.build_checkpoint_v5_payload(
            model,
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=context,
        )


def _legacy_removed_production_context_rejects_noncanonical_authority_paths(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="canonical relative POSIX path"):
        ProductionCheckpointContextV5(
            artifact_root=(tmp_path / "artifacts").resolve(),
            registry_root=(tmp_path / "registry").resolve(),
            authority_manifest_path="../authority.json",
            authority_manifest_file_sha256=_sha("authority"),
        )


def _v6_content(core: dict) -> dict:
    return {**core, "content_sha256": _canonical_hash(core)}


def _v6_finalizer_fixture(
    gate: str = "g2",
    *,
    failed_metric: str | None = None,
) -> dict:
    from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_finalizer_core import (
        ROLE_MANIFEST_SCHEMA,
    )
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy

    metrics = (
        shared_v5.G2_GATE_METRICS_V5
        if gate == "g2"
        else shared_v5.G3_GATE_METRICS_V5
    )
    roles = {
        "train": ["train-a", "train-b"],
        "g2": ["g2-a", "g2-b"],
        "g3": ["g3-a", "g3-b"],
    }
    families = {
        "train-a": "family-a",
        "train-b": "family-b",
        "g2-a": "family-a",
        "g2-b": "family-b",
        "g3-a": "family-a",
        "g3-b": "family-b",
    }
    role_manifest = _v6_content(
        {
            "schema": ROLE_MANIFEST_SCHEMA,
            "protocol_generation": "synthetic-v6",
            "roles": roles,
            "scene_families": families,
        }
    )
    model_state_sha256 = _sha("v6-model-state")
    checkpoint_sha256 = _sha("v6-evaluated-checkpoint")
    runner_sha256 = hashlib.sha256(
        Path(shared_observable_camera_ray_jepa_v5_runner_policy.__file__).read_bytes()
    ).hexdigest()
    outcomes = []
    for sequence, scene_id in enumerate(roles[gate], start=1):
        outcome = _v6_content(
            {
                "schema": shared_observable_camera_ray_jepa_v5_runner_policy.RAW_SCENE_OUTCOME_SCHEMA,
                "gate": gate,
                "scene_id": scene_id,
                "family": families[scene_id],
                "model_state_sha256": model_state_sha256,
                "evaluated_checkpoint_file_sha256": checkpoint_sha256,
                "runner_source_sha256": runner_sha256,
                "instances": [
                    {
                        "instance_id": f"{scene_id}-instance-1",
                        "inference_output_sha256": _sha(
                            f"{gate}-{scene_id}-inference"
                        ),
                        "metric_outcomes": {
                            name: not (sequence == 1 and name == failed_metric)
                            for name in metrics
                        },
                    }
                ],
            }
        )
        outcomes.append(outcome)
    runner_batch = (
        shared_observable_camera_ray_jepa_v5_runner_policy._issue_synthetic_runner_batch_for_tests(
            gate=gate,
            metric_names=metrics,
            role_manifest=role_manifest,
            scene_outcomes=outcomes,
            expected_model_state_sha256=model_state_sha256,
            expected_checkpoint_file_sha256=checkpoint_sha256,
        )
    )
    return {
        "runner_batch": runner_batch,
        "expected_model_state_sha256": model_state_sha256,
        "expected_checkpoint_file_sha256": checkpoint_sha256,
        "expected_runner_source_sha256": runner_sha256,
    }


def test_production_context_and_checkpoint_library_apis_are_removed(tmp_path: Path) -> None:
    from lewm.models.shared_observable_camera_ray_jepa_v5_authority import (
        CANONICAL_AUTHORITY_FILE_SHA256,
        CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256,
        CANONICAL_G2_FINAL_REPORT_FILE_SHA256,
        CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256,
        CANONICAL_G3_FINAL_REPORT_FILE_SHA256,
        CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256,
    )

    assert not hasattr(shared_v5, "ProductionCheckpointContextV5")
    assert not hasattr(shared_v5, "load_production_checkpoint_context_v5")
    assert not hasattr(shared_v5, "build_checkpoint_v5_payload")
    assert not hasattr(shared_v5, "validate_checkpoint_v5_payload")
    assert not hasattr(shared_v5, "checkpoint_v5_weights_only_roundtrip")
    assert CANONICAL_AUTHORITY_FILE_SHA256 is None
    assert CANONICAL_G2_FINAL_REPORT_FILE_SHA256 is None
    assert CANONICAL_G3_FINAL_REPORT_FILE_SHA256 is None
    assert CANONICAL_DATASET_ROLE_MANIFEST_FILE_SHA256 is None
    assert CANONICAL_G2_RUNNER_LEDGER_FILE_SHA256 is None
    assert CANONICAL_G3_RUNNER_LEDGER_FILE_SHA256 is None
    assert "_CANONICAL_CONTEXT_CAPABILITY_V5" not in vars(shared_v5)
    assert "_ISSUED_CANONICAL_CONTEXTS_V5" not in vars(shared_v5)
    forged = object()
    with pytest.raises(PermissionError, match="permanently removed"):
        shared_v5._removed_caller_filesystem_build_checkpoint_v5_payload(
            SharedObservableCameraRayJepaV5(),
            lifecycle=LIFECYCLE_G3_CANDIDATE,
            context=forged,
        )
    with pytest.raises(PermissionError, match="permanently removed"):
        shared_v5._removed_caller_filesystem_validate_checkpoint_v5_payload(
            {},
            context=forged,
        )


def test_canonical_authority_and_registry_paths_are_fixed(tmp_path: Path) -> None:
    import inspect

    from lewm.models.shared_observable_camera_ray_jepa_v5_authority import (
        CANONICAL_ATTEMPT_REGISTRY_RELATIVE_PATH,
        CANONICAL_AUTHORITY_RELATIVE_PATH,
        CANONICAL_REPOSITORY_ROOT,
    )
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy

    assert CANONICAL_REPOSITORY_ROOT == Path(shared_v5.__file__).resolve().parents[2]
    assert CANONICAL_AUTHORITY_RELATIVE_PATH.as_posix() == (
        "docs/lewm_go2_shared_jepa_v5_production_authority.json"
    )
    assert CANONICAL_ATTEMPT_REGISTRY_RELATIVE_PATH.as_posix() == (
        ".generated/go2_shared_jepa_v5/role_global_attempt_registry"
    )
    assert not hasattr(
        shared_observable_camera_ray_jepa_v5_registry_policy,
        "acquire_canonical_attempt",
    )
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy

    assert not hasattr(
        shared_observable_camera_ray_jepa_v5_runner_policy,
        "reopen_canonical_runner_batch",
    )
    assert not hasattr(
        shared_observable_camera_ray_jepa_v5_runner_policy,
        "validated_runner_batch_payload",
    )
    assert "_install_canonical_runner_api" not in vars(
        shared_observable_camera_ray_jepa_v5_runner_policy
    )
    assert not hasattr(shared_v5, "build_checkpoint_v5_payload")

    source = Path(
        "lewm/models/shared_observable_camera_ray_jepa_v5_authority.py"
    ).resolve()
    copied = tmp_path / "copied/lewm/models/shared_observable_camera_ray_jepa_v5_authority.py"
    copied.parent.mkdir(parents=True)
    shutil.copyfile(source, copied)
    spec = importlib.util.spec_from_file_location("copied_v5_authority", copied)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.CANONICAL_REPOSITORY_ROOT == CANONICAL_REPOSITORY_ROOT
    with pytest.raises(PermissionError, match="copied or alternate root"):
        module.require_frozen_production_authority()


def test_canonical_registry_library_mutation_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy as policy

    assert not hasattr(policy, "acquire_canonical_attempt")
    assert not (tmp_path / "escaped").exists()
    assert not (tmp_path / "escaped").exists()


def test_invalid_registry_gate_is_rejected_before_authority_or_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy as policy

    def forbidden_access(*args: object, **kwargs: object) -> object:
        raise AssertionError("invalid gate reached authority or filesystem")

    monkeypatch.setattr(policy, "require_frozen_production_authority", forbidden_access)
    monkeypatch.setattr(policy.os, "lstat", forbidden_access)
    monkeypatch.setattr(policy.os, "open", forbidden_access)
    assert not hasattr(policy, "acquire_canonical_attempt")


@pytest.mark.parametrize("gate", ("g2", "g3"))
def test_distinct_finalizer_recomputes_every_scene_family_and_access_event(
    gate: str,
) -> None:
    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g2 import (
        _finalize_g2_synthetic_for_tests,
    )
    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g3 import (
        _finalize_g3_synthetic_for_tests,
    )

    fixture = _v6_finalizer_fixture(gate)
    report = (
        _finalize_g2_synthetic_for_tests
        if gate == "g2"
        else _finalize_g3_synthetic_for_tests
    )(**fixture)
    assert report["passed"] is True
    assert report["zero_forbidden_access"] is True
    assert set(report["per_family_counts"]) == {"family-a", "family-b"}
    assert all(value == 1.0 for value in report["metrics"].values())
    assert len(report["raw_scene_outcome_content_sha256s"]) == 2
    assert report["synthetic_only"] is True
    assert report["production_authority_eligible"] is False


def test_aggregate_only_and_caller_zero_evidence_are_rejected() -> None:
    fixture = _v6_finalizer_fixture("g2")
    from lewm.benchmarks import finalize_shared_observable_camera_ray_jepa_v5_g2 as g2
    assert not hasattr(g2, "finalize_g2")
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy as runner_policy

    assert not hasattr(runner_policy, "_issue_batch")
    assert not hasattr(runner_policy, "_issue_normalized_batch")
    assert type(fixture["runner_batch"]) is runner_policy.SyntheticRunnerBatchV6
    assert not hasattr(runner_policy, "CanonicalRunnerBatchV6")
    with pytest.raises((AttributeError, TypeError)):
        object.__setattr__(fixture["runner_batch"], "synthetic_only", False)

    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g2 import (
        _finalize_g2_synthetic_for_tests,
    )

    object.__setattr__(fixture["runner_batch"], "runner_ledger_bytes", b"{}\n")
    with pytest.raises(PermissionError, match="changed after issuance"):
        _finalize_g2_synthetic_for_tests(**fixture)
    fixture = _v6_finalizer_fixture("g2")


def test_finalizer_decision_is_recomputed_from_canonical_scene_counts() -> None:
    from lewm.benchmarks.finalize_shared_observable_camera_ray_jepa_v5_g2 import (
        _finalize_g2_synthetic_for_tests,
    )

    metric = shared_v5.G2_GATE_METRICS_V5[0]
    fixture = _v6_finalizer_fixture("g2", failed_metric=metric)
    report = _finalize_g2_synthetic_for_tests(**fixture)
    assert report["metrics"][metric] == 0.5
    assert report["passed"] is False


def test_caller_batch_and_same_source_substitutions_are_rejected() -> None:
    from lewm.benchmarks.shared_observable_camera_ray_jepa_v5_finalizer_core import (
        _finalize_gate_records_synthetic_for_tests,
    )

    fixture = _v6_finalizer_fixture("g2")
    with pytest.raises(PermissionError, match="must be distinct"):
        _finalize_gate_records_synthetic_for_tests(
            gate="g2",
            metric_names=shared_v5.G2_GATE_METRICS_V5,
            runner_batch=fixture["runner_batch"],
            expected_model_state_sha256=fixture["expected_model_state_sha256"],
            expected_checkpoint_file_sha256=fixture[
                "expected_checkpoint_file_sha256"
            ],
            expected_runner_source_sha256=fixture[
                "expected_runner_source_sha256"
            ],
            finalizer_source_sha256=fixture["expected_runner_source_sha256"],
        )


def test_implementation_registry_runner_and_finalizers_are_distinct_sources() -> None:
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g2 as g2_finalizer,
    )
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g3 as g3_finalizer,
    )
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_finalizer_core
    from lewm.models import shared_observable_camera_ray_jepa_v5_authority
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy

    paths = (
        Path(shared_observable_camera_ray_jepa_v5_authority.__file__),
        Path(shared_v5.__file__),
        Path(shared_observable_camera_ray_jepa_v5_registry_policy.__file__),
        Path(shared_observable_camera_ray_jepa_v5_runner_policy.__file__),
        Path(shared_observable_camera_ray_jepa_v5_finalizer_core.__file__),
        Path(g2_finalizer.__file__),
        Path(g3_finalizer.__file__),
    )
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    assert len(set(hashes)) == len(hashes)


@pytest.mark.parametrize(
    "source_role",
    (
        "authority_source",
        "implementation_source",
        "registry_policy_source",
        "runner_source",
        "finalizer_core_source",
        "g2_finalizer_source",
        "g3_finalizer_source",
    ),
)
def test_every_production_decision_source_substitution_is_rejected(
    source_role: str,
) -> None:
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g2 as g2_finalizer,
    )
    from lewm.benchmarks import (
        finalize_shared_observable_camera_ray_jepa_v5_g3 as g3_finalizer,
    )
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_finalizer_core
    from lewm.benchmarks import shared_observable_camera_ray_jepa_v5_runner_policy
    from lewm.models import shared_observable_camera_ray_jepa_v5_authority
    from lewm.models import shared_observable_camera_ray_jepa_v5_registry_policy

    root = Path(shared_v5.__file__).resolve().parents[2]
    source_paths = {
        "authority_source": Path(shared_observable_camera_ray_jepa_v5_authority.__file__),
        "implementation_source": Path(shared_v5.__file__),
        "registry_policy_source": Path(shared_observable_camera_ray_jepa_v5_registry_policy.__file__),
        "runner_source": Path(shared_observable_camera_ray_jepa_v5_runner_policy.__file__),
        "finalizer_core_source": Path(shared_observable_camera_ray_jepa_v5_finalizer_core.__file__),
        "g2_finalizer_source": Path(g2_finalizer.__file__),
        "g3_finalizer_source": Path(g3_finalizer.__file__),
    }
    artifacts = {
        name: {
            "path": path.resolve().relative_to(root).as_posix(),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, path in source_paths.items()
    }
    authority = {"artifacts": artifacts}
    context = SimpleNamespace(repository_root=root)
    assert shared_v5._validate_distinct_authorized_sources_v5(
        context, authority
    ) == artifacts["runner_source"]["file_sha256"]
    artifacts[source_role]["file_sha256"] = _sha(f"substituted-{source_role}")
    with pytest.raises(ValueError, match="file hash changed"):
        shared_v5._validate_distinct_authorized_sources_v5(context, authority)


def test_random_model_perfect_self_report_cannot_reach_production() -> None:
    model = SharedObservableCameraRayJepaV5()
    forged_report = {
        "passed": True,
        "metrics": {name: 1.0 for name in shared_v5.G2_GATE_METRICS_V5},
        "zero_forbidden_access": True,
    }
    assert forged_report["passed"] is True
    assert not hasattr(shared_v5, "build_checkpoint_v5_payload")
    assert not hasattr(shared_v5, "load_production_checkpoint_context_v5")
