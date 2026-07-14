#!/usr/bin/env python3
"""Compute-only exact verifier imported by the reviewed V9 executor.

The lifecycle-owning executor supplies the local authority and publishes the
receipt. Direct verifier execution is intentionally unavailable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.benchmarks import (  # noqa: E402
    go2_observable_camera_ray_fit_v4_n5_hierarchical_first_hit_v9 as policy,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-review", type=Path, required=True)
    parser.add_argument("--source-review-sha256", required=True)
    parser.add_argument("--reservation", required=True, help="PATH:SHA256")
    parser.add_argument("--result", required=True, help="PATH:SHA256")
    parser.add_argument("--checkpoint", required=True, help="PATH:SHA256")
    parser.add_argument("--completion", required=True, help="PATH:SHA256")
    return parser.parse_args(argv)


def _fixed_bound(value: str, expected: Path, *, name: str) -> tuple[Path, str]:
    path, digest = policy.parse_bound_path(value)
    if path != expected.resolve(strict=True):
        raise PermissionError(f"N5 full-panel {name} path is not canonical")
    return path, digest


def _review_binding(authority: policy.VerifiedAuthority) -> dict[str, str]:
    return policy.source_review_binding(authority)


def _validate_attempt_bundle(
    authority: policy.VerifiedAuthority,
    args: argparse.Namespace,
) -> dict[str, Any]:
    policy.require_verified_authority(authority)
    attempt = policy.CANONICAL_ATTEMPT_PATH
    if attempt.is_symlink() or not attempt.is_dir():
        raise PermissionError("N5 full-panel attempt is not a real directory")
    inventory = sorted(path.name for path in attempt.iterdir())
    if inventory != ["checkpoint.pt", "completed.json", "reservation.json", "result.json"]:
        raise PermissionError("N5 full-panel completed attempt inventory changed")
    reservation_path, reservation_sha = _fixed_bound(
        args.reservation,
        attempt / "reservation.json",
        name="reservation",
    )
    result_path, result_sha = _fixed_bound(
        args.result,
        attempt / "result.json",
        name="result",
    )
    checkpoint_path, checkpoint_sha = _fixed_bound(
        args.checkpoint,
        attempt / "checkpoint.pt",
        name="checkpoint",
    )
    completion_path, completion_sha = _fixed_bound(
        args.completion,
        attempt / "completed.json",
        name="completion",
    )
    reservation, reservation_raw = policy.load_hashed_json(
        reservation_path,
        reservation_sha,
        name="N5 full-panel reservation",
    )
    result, result_raw = policy.load_hashed_json(
        result_path,
        result_sha,
        name="N5 full-panel result",
    )
    checkpoint_raw = policy.read_hashed_bytes(
        checkpoint_path,
        checkpoint_sha,
        name="N5 full-panel checkpoint",
    )
    completion, completion_raw = policy.load_hashed_json(
        completion_path,
        completion_sha,
        name="N5 full-panel completion",
    )
    review = _review_binding(authority)
    policy.validate_reservation_structure(
        reservation,
        expected_source_review=review,
    )
    policy.validate_result_structure(result, expected_source_review=review)
    reservation_binding = policy.artifact_binding(
        "reservation.json",
        reservation_raw,
        content_sha256=reservation["content_sha256"],
    )
    result_binding = policy.artifact_binding(
        "result.json",
        result_raw,
        content_sha256=result["content_sha256"],
    )
    checkpoint_model = result["model"]["checkpoint"]
    checkpoint_binding = policy.artifact_binding(
        "checkpoint.pt",
        checkpoint_raw,
        content_sha256=checkpoint_model["content_sha256"],
    )
    expected_completion_fields = {
        "schema",
        "status",
        "reservation",
        "checkpoint",
        "result",
        "inventory",
        "retry_authorized",
        "licenses",
        "content_sha256",
    }
    expected_completion_licenses = {
        "checkpoint_use_authorized": False,
        "metric_verification_only_checkpoint_use_authorized": True,
        "n16_execution_authorized": False,
        "second_seed_authorized": False,
        "holdout_authorized": False,
        "g2_authorized": False,
        "runtime_authorized": False,
        "promotion_authorized": False,
    }
    if (
        result["attempt"]["reservation"] != reservation_binding
        or checkpoint_model
        != {
            **checkpoint_binding,
            "development_only": True,
        }
        or set(completion) != expected_completion_fields
        or completion.get("schema") != policy.COMPLETION_SCHEMA
        or completion.get("status") != "completed"
        or completion.get("reservation") != reservation_binding
        or completion.get("checkpoint") != checkpoint_binding
        or completion.get("result") != result_binding
        or completion.get("inventory") != inventory
        or completion.get("retry_authorized") is not False
        or completion.get("licenses") != expected_completion_licenses
    ):
        raise PermissionError("N5 full-panel completion/artifact chain changed")
    return {
        "reservation": reservation,
        "reservation_raw": reservation_raw,
        "reservation_binding": reservation_binding,
        "result": result,
        "result_raw": result_raw,
        "result_binding": result_binding,
        "checkpoint_raw": checkpoint_raw,
        "checkpoint_binding": checkpoint_binding,
        "completion": completion,
        "completion_raw": completion_raw,
        "completion_binding": policy.artifact_binding(
            "completed.json",
            completion_raw,
            content_sha256=completion["content_sha256"],
        ),
    }


def _validate_checkpoint(
    raw: bytes,
    *,
    expected_binding: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    import torch

    try:
        checkpoint = torch.load(BytesIO(raw), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(BytesIO(raw), map_location="cpu")
    if not isinstance(checkpoint, Mapping) or set(checkpoint) != {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    }:
        raise ValueError("N5 full-panel checkpoint schema changed")
    if (
        checkpoint.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or checkpoint.get("model_class") != "ObservableCameraRayEvidenceV4Model"
        or checkpoint.get("metadata") != expected_metadata
        or checkpoint.get("authoritative") is not False
        or checkpoint.get("aggregation_eligible") is not False
        or checkpoint.get("promotion_eligible") is not False
    ):
        raise PermissionError("N5 full-panel checkpoint scope/metadata changed")
    state = checkpoint.get("state_dict")
    manifest = checkpoint.get("state_manifest")
    if not isinstance(state, Mapping) or not isinstance(manifest, list):
        raise ValueError("N5 full-panel checkpoint state is malformed")
    expected_manifest = []
    for name, tensor in sorted(state.items()):
        if not isinstance(name, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError("N5 full-panel checkpoint tensor entry is malformed")
        contiguous = tensor.detach().to(device="cpu").contiguous()
        expected_manifest.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype).removeprefix("torch."),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(
                    contiguous.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    semantic_core = {
        key: checkpoint[key]
        for key in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if (
        manifest != expected_manifest
        or checkpoint.get("content_sha256")
        != policy.canonical_json_sha256(semantic_core)
        or checkpoint.get("content_sha256")
        != expected_binding["content_sha256"]
    ):
        raise ValueError("N5 full-panel checkpoint semantic hash changed")
    return checkpoint, state


def compute_four_equal_v9_losses_for_verification(
    model: Any,
    batch: Any,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    """Independently reconstruct the reviewed V9 objective and V4 branches."""

    from lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
        hierarchical_first_hit_nll_breakdown_v9,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    targets = base.derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=batch.pixel_hit_mask,
        pixel_first_hit_distance_m=batch.pixel_first_hit_distance_m,
        ground_support_in_frustum=batch.ground_support_in_frustum,
        ground_support_clear_to_target=batch.ground_support_clear_to_target,
    )
    raw_output = model(
        batch.image,
        batch.camera_origin_body_m,
        batch.camera_basis_body_fru,
        batch.ground_plane_z_body_m,
    )
    if not base.torch.equal(
        raw_output.ground_query_in_frustum,
        targets.ground_in_frustum,
    ):
        raise ValueError("verifier calibration does not reproduce V4 ground visibility")
    first_hit = hierarchical_first_hit_nll_breakdown_v9(
        raw_output.pixel_first_hit_hazard_logits,
        targets,
    ).total
    offset = base._skew_balanced_pixel_offset_loss(raw_output, targets)
    ground = base.balanced_ground_clear_bce_v4(
        raw_output.ground_clear_to_target_logits,
        targets,
        raw_output.ground_target_distance_m,
    )
    raster = base.soft_rasterize_observable_camera_ray_evidence_v4(
        raw_output,
        camera_origin_body_m=batch.camera_origin_body_m,
        camera_basis_body_fru=batch.camera_basis_body_fru,
    )
    raster_loss = base.hierarchical_raster_cross_entropy_v4(
        raster,
        batch.target_raster_labels,
    ).total
    components = {
        "hierarchical_first_hit_nll": first_hit,
        "target_bin_offset_smooth_l1": offset,
        "ground_clear_distance_state_balanced_bce": ground,
        "derived_raster_hierarchical_bce": raster_loss,
    }
    total = 0.25 * sum(components.values())
    return total, components, raw_output, targets, raster


def recompute_evaluation(
    *,
    model: Any,
    frames: Sequence[Any],
    images: Any,
    device: Any,
    wrong_rgb: bool,
) -> dict[str, Any]:
    """Independent batch-one inference and aggregation; no result reuse."""

    import torch
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    mapping = tuple(
        ((index + 1) % len(frames)) if wrong_rgb else index
        for index in range(len(frames))
    )
    accumulator = base.ObservableCameraRayFitV4MetricAccumulator()
    sums: Counter[str] = Counter()
    model.eval()
    with torch.no_grad():
        for index in range(5):
            batch = base._batch_from_indices(
                frames,
                images,
                (index,),
                image_indices=(mapping[index],),
            ).to(device)
            _batch_total, components, raw, targets, raster = (
                compute_four_equal_v9_losses_for_verification(model, batch)
            )
            for component in policy.LOSS_COMPONENTS:
                sums[component] += float(components[component].item())
            accumulator.update(
                raw_output=raw,
                targets=targets,
                soft_raster=raster,
                target_raster_labels=batch.target_raster_labels,
                families=batch.families,
            )
    means = {component: sums[component] / 5 for component in policy.LOSS_COMPONENTS}
    losses = {
        **means,
        "total": 0.25 * sum(means[component] for component in policy.LOSS_COMPONENTS),
    }
    return {
        "control": (
            "wrong_rgb_with_target_calibration" if wrong_rgb else "matched_rgb"
        ),
        "wrong_rgb_degenerate_singleton": False,
        "image_index_mapping": list(mapping),
        "image_mapping_sha256": policy.canonical_json_sha256(list(mapping)),
        "losses": losses,
        "metrics": accumulator.finalize(),
    }


def _compute_receipt(
    authority: policy.VerifiedAuthority,
    bundle: Mapping[str, Any],
) -> dict[str, Any]:
    policy.require_verified_authority(authority)
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    result = bundle["result"]
    dataset_path = ROOT / policy.DATASET_MANIFEST_RELATIVE_PATH
    audit_path = ROOT / policy.AUDIT_RECEIPT_RELATIVE_PATH
    authorization_path = ROOT / policy.TRAINER_AUTHORIZATION_RELATIVE_PATH
    review_path = ROOT / policy.TRAINER_REVIEW_RELATIVE_PATH
    inputs = base.load_exact_inputs(
        dataset_manifest_path=dataset_path,
        dataset_manifest_file_sha256=policy.DATASET_MANIFEST_FILE_SHA256,
        audit_receipt_path=audit_path,
        audit_receipt_file_sha256=policy.AUDIT_RECEIPT_FILE_SHA256,
        trainer_authorization_path=authorization_path,
        trainer_authorization_file_sha256=policy.TRAINER_AUTHORIZATION_FILE_SHA256,
        trainer_review_record_path=review_path,
        trainer_review_record_file_sha256=policy.TRAINER_REVIEW_FILE_SHA256,
        fit_size=5,
    )
    target_partition = base.validate_exact_target_partition_v4(inputs.frames, fit_size=5)
    if (
        inputs.subset_receipt != result["subset"]
        or target_partition != result["target_partition"]
    ):
        raise PermissionError("N5 full-panel verifier target reproduction changed")
    metadata = {
        "experiment": policy.experiment_contract(),
        "authority_bindings": policy.authority_bindings(),
        "source_review": _review_binding(authority),
        "inputs": result["inputs"],
        "attempt_reservation": bundle["reservation_binding"],
        "training_schedule_sha256": policy.EXPECTED_SCHEDULE_SHA256,
        "checkpoint_selection": "final_update_only",
        "loss_contract": {
            "version": "hierarchical_first_hit_v9",
            "components": list(policy.LOSS_COMPONENTS),
            "weights": {name: 0.25 for name in policy.LOSS_COMPONENTS},
        },
    }
    checkpoint, state = _validate_checkpoint(
        bundle["checkpoint_raw"],
        expected_binding=bundle["checkpoint_binding"],
        expected_metadata=metadata,
    )
    resource = base.validate_gpu0_r9700_runtime(device_text="cuda:0")
    base.configure_determinism(20260710)
    arrays = [
        base._decode_rgb_job(
            str(frame.rgb_path),
            str(frame.image_sha256),
            str(ROOT),
            policy.frozen_source_bindings()[
                "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
            ],
        )
        for frame in inputs.frames
    ]
    images = base.torch.from_numpy(base.np.stack(arrays, axis=0).copy())
    model = base.ObservableCameraRayEvidenceV4Model()
    model.load_state_dict(state, strict=True)
    device = base.torch.device("cuda:0")
    model.to(device)
    matched = recompute_evaluation(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        wrong_rgb=False,
    )
    wrong = recompute_evaluation(
        model=model,
        frames=inputs.frames,
        images=images,
        device=device,
        wrong_rgb=True,
    )
    evaluation = {
        "matched_rgb": matched,
        "wrong_rgb_with_target_calibration": wrong,
    }
    policy.validate_evaluation_structure(evaluation)
    if evaluation != result["evaluation"]:
        raise ValueError(
            "N5 full-panel independently recomputed evaluation differs from result"
        )
    matched_metrics, wrong_metrics, signature = gate._validated_metric_evaluation(
        evaluation,
        fit_size=5,
    )
    numeric_gate = gate._gate_stage(
        {"fit_size": 5, "matched": matched_metrics, "wrong": wrong_metrics}
    )
    core = {
        "schema": policy.METRIC_RECEIPT_SCHEMA,
        "authoritative": False,
        "aggregation_eligible": False,
        "promotion_eligible": False,
        "dataset_role": "train",
        "seed": 20260710,
        "fit_size": 5,
        "authority_bindings": policy.authority_bindings(),
        "source_review": _review_binding(authority),
        "artifacts": {
            "reservation": bundle["reservation_binding"],
            "result": bundle["result_binding"],
            "checkpoint": bundle["checkpoint_binding"],
            "completion": bundle["completion_binding"],
        },
        "result_content_sha256": result["content_sha256"],
        "target_partition": target_partition,
        "target_partition_signature": signature,
        "target_partition_signature_sha256": policy.canonical_json_sha256(signature),
        "recomputed_evaluation": evaluation,
        "recomputed_evaluation_sha256": policy.canonical_json_sha256(evaluation),
        "numeric_gate": numeric_gate,
        "verification": {
            "checkpoint_loaded": True,
            "checkpoint_state_manifest_rehashed": True,
            "checkpoint_final_update_binding_validated": True,
            "fresh_model_loaded_for_inference": True,
            "selected_train_targets_loaded": True,
            "selected_matched_rgb_loaded": True,
            "wrong_rgb_mapping_rerun": True,
            "evaluation_losses_recomputed": True,
            "evaluation_loss_arithmetic_validated": True,
            "all_confusions_recomputed": True,
            "depth_quantiles_and_sorted_commitments_recomputed": True,
            "raster_nll_recomputed": True,
            "family_metrics_recomputed": True,
            "frozen_thresholds_recomputed": True,
            "result_metrics_reused": False,
            "metric_repair_applied": False,
            "threshold_weakened": False,
        },
        "resource": resource,
        "access_ledger": {
            "selected_rgb_count": 5,
            "selected_rgb_hash_opens": 5,
            "selected_rgb_decodes": 5,
            "checkpoint_opens": 1,
            "heldout_opens": 0,
            "g2_opens": 0,
            "selection_opens": 0,
            "calibration_opens": 0,
            "runtime_opens": 0,
            "hardware_opens": 0,
            "production_opens": 0,
            "gpu1_uses": 0,
        },
        "licenses": {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "development_checkpoint_use_authorized": False,
            "new_model_output_authorized": False,
            "retry_authorized": False,
            "n16_execution_authorized": False,
            "second_seed_authorized": False,
            "holdout_authorized": False,
            "g2_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    return {**core, "content_sha256": policy.canonical_json_sha256(core)}


def main(argv: Sequence[str] | None = None) -> int:
    del argv
    raise PermissionError(
        "V9 verifier is compute-only; use the reviewed lifecycle-owning executor"
    )


if __name__ == "__main__":
    raise SystemExit(main())
