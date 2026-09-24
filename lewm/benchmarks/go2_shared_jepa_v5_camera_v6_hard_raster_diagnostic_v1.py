"""Pure contract for the one-shot Camera V6 hard-raster diagnostic.

The module performs no file I/O at import time.  Torch and the registered V4
evidence implementation are imported only inside the hard-adapter function so
the execution runner can validate and reserve its one output root before any
tensor runtime is loaded.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence


SCHEMA_PREFIX = "lewm_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1"
REVIEW_SCHEMA = f"{SCHEMA_PREFIX}_independent_review_v1"
AUTHORIZATION_SCHEMA = f"{SCHEMA_PREFIX}_execution_authorization_v1"
RESERVATION_SCHEMA = f"{SCHEMA_PREFIX}_reservation_v1"
ACCESS_SCHEMA = f"{SCHEMA_PREFIX}_access_v1"
RESULT_SCHEMA = f"{SCHEMA_PREFIX}_result_v1"
COMPLETION_SCHEMA = f"{SCHEMA_PREFIX}_completion_v1"
FAILURE_SCHEMA = f"{SCHEMA_PREFIX}_failure_v1"

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1_"
    "preregistration_2026-07-23.md"
)
REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1_"
    "independent_review_2026-07-23.json"
)
AUTHORIZATION_RELATIVE_PATH = (
    "docs/lewm_go2_shared_jepa_v5_camera_v6_hard_raster_diagnostic_v1_"
    "execution_authorization_2026-07-23.json"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "camera_v6_hard_raster_diagnostic_v1"
)

V6_ROOT_RELATIVE_PATH = (
    ".generated/go2_shared_observable_camera_ray_jepa_v5/"
    "protected_camera_adaptation_v6_final_fresh_update0_tail_depth_8k"
)
V6_SIDECAR_RELATIVE_PATH = (
    f"{V6_ROOT_RELATIVE_PATH}/checkpoints/update_8000.metrics.json"
)
V6_CHECKPOINT_RELATIVE_PATH = (
    f"{V6_ROOT_RELATIVE_PATH}/checkpoints/update_8000.pt"
)
V6_SIDECAR_BINDING = {
    "path": V6_SIDECAR_RELATIVE_PATH,
    "byte_count": 15_365,
    "file_sha256": "c03bc02f5c45ad8b2de0042bdb4602fe03c88ad52c2ac5b77375d9e6f956d2dc",
    "content_sha256": "7437bfee92f2b9fe9d77fd8acce1612c53ebf17c7d839786cff6f94f691bb3ee",
}
V6_CHECKPOINT_BINDING = {
    "path": V6_CHECKPOINT_RELATIVE_PATH,
    "byte_count": 29_466_305,
    "file_sha256": "01871a6495cd6ffa6cdcc97f1451014e887ac9a219360bb69ae0a866db3db20c",
    "content_sha256": "4d20f50a688efd617f31ac092a5f7019084afb67e99a064029907222a61be120",
    "state_sha256": "960854245db49a048e3a99e91b08d6746795f8c1abd52a267f592900259eee22",
    "frozen_state_sha256": "3f5cce294f840be4c6c8cfa43b2818bae68da739b13348fe45a3d5087fe2524e",
    "trainable_state_sha256": "6b01b16355d940133b6683b420f2e4f182d0535264aef595a97727d813919e96",
}
V6_SIDECAR_LOCAL_CHECKPOINT_PATH = "checkpoints/update_8000.pt"
V6_TERMINAL_AUDIT_BINDING = {
    "path": (
        "docs/lewm_go2_shared_jepa_v5_protected_camera_adaptation_v6_"
        "terminal_audit_2026-07-23.json"
    ),
    "byte_count": 20_059,
    "file_sha256": "367dd08f9a039710d61efd9ecb652134f6efbd056e126c4a51d67929f28b06b7",
    "content_sha256": "76727ada6442774412508b0ca96b1a50b5170bc75867235aecc132f28d1ac892",
}
RAW_MANIFEST_BINDING = {
    "path": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1/manifest.json"
    ),
    "byte_count": 311_598,
    "file_sha256": "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360",
    "content_sha256": "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a",
}
RAW_AUDIT_BINDING = {
    "path": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1.audit_v13.json"
    ),
    "byte_count": 26_975,
    "file_sha256": "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76",
    "content_sha256": "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca",
}

SELECTION_PAIR_COUNT = 495
SELECTION_ENDPOINT_COUNT = 924
ROUGH_SCOPE = "rough_local_dynamics"
NON_ROUGH_SCOPES = (
    "aggregate",
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ALL_SCOPES = (*NON_ROUGH_SCOPES, ROUGH_SCOPE)
SOFT_RASTER_BALANCED_ACCURACY = {
    "aggregate": 0.9009460724448773,
    "large_enclosed_maze": 0.8187028299574806,
    "local_composite_motifs": 0.8887728118230923,
    "loop_alias_stress": 0.817520212817799,
    "medium_enclosed_maze": 0.8671429422192141,
    "open_obstacle_field": 0.9085505950468774,
    "small_enclosed_maze": 0.8602078804715946,
    "visual_sensor_stress": 0.8675719422139417,
}
SOFT_AGGREGATE_FREE_RECALL = 0.91637020862468
SOFT_AGGREGATE_OCCUPIED_RECALL = 0.8059679976935274
MINIMUM_SCOPE_BALANCED_ACCURACY_GAIN = 0.05
MINIMUM_PASSING_SCOPE_COUNT = 6
MINIMUM_AGGREGATE_RECALL_GAIN = 0.05
MINIMUM_WRONG_RGB_DROP = 0.12
CLASS_NAMES = ("unknown", "free", "occupied")

DIRECT_METRIC_KEYS = (
    "pixel_first_hit_balanced_accuracy",
    "depth_median_error_m",
    "depth_p95_error_m",
    "ground_clear_balanced_accuracy",
    "distance_group_balanced_accuracy",
    "wrong_rgb_pixel_balanced_accuracy_drop",
    "wrong_rgb_depth_median_error_increase_m",
    "wrong_rgb_depth_p95_error_increase_m",
    "wrong_rgb_ground_balanced_accuracy_drop",
)

EXECUTION_AUTHORITY = {
    "one_exact_diagnostic_attempt": True,
    "rejected_v6_update8000_checkpoint_read": True,
    "rejected_v6_update8000_checkpoint_deserialization": True,
    "bound_update8000_sidecar_read": True,
    "checkpoint_selection_role_diagnostic_read": True,
    "fixed_calibration_read": True,
    "forward_only_inference": True,
    "single_r9700_gpu0": True,
    "output_root_mutation_only": True,
    "optimizer_construction": False,
    "optimizer_step": False,
    "backward": False,
    "gradient": False,
    "autocast": False,
    "ema_update": False,
    "parameter_or_buffer_mutation": False,
    "checkpoint_write_or_mutation": False,
    "train_role_read": False,
    "probability_calibration_role_read": False,
    "threshold_search": False,
    "alternate_decoder": False,
    "retry_resume_or_repair": False,
    "checkpoint_selection_decision": False,
    "camera_qualification": False,
    "checkpoint_promotion": False,
    "successor_implementation_or_training": False,
    "g2": False,
    "navigation": False,
    "runtime_or_production": False,
    "heldout": False,
}


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = dict(core)
    if "content_sha256" in normalized:
        raise ValueError("content hash is computed, not caller supplied")
    return {**normalized, "content_sha256": canonical_json_sha256(normalized)}


def validate_content_sha256(value: object, *, schema: str) -> dict[str, Any]:
    if type(value) is not dict or value.get("schema") != schema:
        raise PermissionError(f"{schema} payload changed")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        type(declared) is not str
        or len(declared) != 64
        or any(character not in "0123456789abcdef" for character in declared)
        or canonical_json_sha256(core) != declared
    ):
        raise PermissionError(f"{schema} content hash changed")
    return dict(value)


def validate_v6_sidecar_checkpoint_binding(value: object) -> dict[str, Any]:
    if type(value) is not dict:
        raise PermissionError("V6 sidecar checkpoint binding changed")
    expected = dict(V6_CHECKPOINT_BINDING)
    expected["path"] = V6_SIDECAR_LOCAL_CHECKPOINT_PATH
    if value != expected:
        raise PermissionError("V6 sidecar checkpoint binding changed")
    return dict(value)


def hard_raster_labels_from_raw_output(
    raw_output: Any,
    *,
    camera_origin_body_m: Any,
    camera_basis_body_fru: Any,
    ground_plane_z_body_m: Any,
) -> list[Any]:
    """Apply the single preregistered MAP/Boolean adapter to a batch.

    Returned entries are the public deterministic V4 raster objects.  This
    function never mutates ``raw_output`` and performs no model inference.
    """

    import numpy as np
    import torch
    from lewm.benchmarks.go2_observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4,
        rasterize_observable_camera_ray_evidence_v4,
    )
    from lewm.models.observable_camera_ray_evidence_v4 import (
        ObservableCameraRayEvidenceV4RawOutput,
    )

    if not isinstance(raw_output, ObservableCameraRayEvidenceV4RawOutput):
        raise TypeError("raw_output must be ObservableCameraRayEvidenceV4RawOutput")
    tensors = (camera_origin_body_m, camera_basis_body_fru, ground_plane_z_body_m)
    if not all(isinstance(value, torch.Tensor) for value in tensors):
        raise TypeError("hard adapter inputs must be tensors")
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        raise FloatingPointError("hard adapter input became nonfinite")
    batch = raw_output.pixel_first_hit_hazard_logits.shape[0]
    if (
        tuple(camera_origin_body_m.shape) != (batch, 3)
        or tuple(camera_basis_body_fru.shape) != (batch, 3, 3)
        or tuple(ground_plane_z_body_m.shape) != (batch,)
    ):
        raise ValueError("hard adapter calibration batch changed")

    finite_hit, selected_depth, ground_clear = decode_hard_evidence_tensors(
        raw_output
    )

    origin = camera_origin_body_m.detach().to(device="cpu", dtype=torch.float32)
    basis = camera_basis_body_fru.detach().to(device="cpu", dtype=torch.float32)
    ground = ground_plane_z_body_m.detach().to(device="cpu", dtype=torch.float32)
    in_frustum = raw_output.ground_query_in_frustum.detach().to(device="cpu")
    clear = ground_clear.detach().to(device="cpu")
    hit = finite_hit.detach().to(device="cpu")
    depth = selected_depth.detach().to(device="cpu", dtype=torch.float32)
    result = []
    for index in range(batch):
        evidence = ObservableCameraRayEvidenceV4(
            camera_origin_body_m=origin[index].numpy(),
            camera_basis_body_fru=basis[index].numpy(),
            ground_plane_z_body_m=float(ground[index].item()),
            ground_support_in_frustum=in_frustum[index].numpy(),
            ground_support_clear_to_target=clear[index].numpy(),
            pixel_hit_mask=hit[index].numpy(),
            pixel_first_hit_distance_m=np.asarray(
                depth[index].numpy(), dtype=np.float32
            ),
        )
        result.append(rasterize_observable_camera_ray_evidence_v4(evidence))
    return result


def decode_hard_evidence_tensors(raw_output: Any) -> tuple[Any, Any, Any]:
    """Return the fixed finite-hit mask, selected depth, and ground-clear mask."""

    import torch
    from lewm.models.observable_camera_ray_evidence_v4 import (
        DEPTH_BIN_SIZE_M,
        DEPTH_NEAR_EDGE_M,
        ObservableCameraRayEvidenceV4RawOutput,
        ordered_obstacle_first_hit_log_probabilities_v4,
    )

    if not isinstance(raw_output, ObservableCameraRayEvidenceV4RawOutput):
        raise TypeError("raw_output must be ObservableCameraRayEvidenceV4RawOutput")
    tensors = (
        raw_output.pixel_first_hit_hazard_logits,
        raw_output.pixel_within_bin_offset_m,
        raw_output.ground_clear_to_target_logits,
    )
    if any(not bool(torch.isfinite(value).all().item()) for value in tensors):
        raise FloatingPointError("hard evidence tensor became nonfinite")
    ordered = ordered_obstacle_first_hit_log_probabilities_v4(
        raw_output.pixel_first_hit_hazard_logits
    )
    finite_hit = -torch.expm1(ordered.no_hit) >= 0.5
    selected_bin = ordered.hit.argmax(dim=1)
    selected_offset = raw_output.pixel_within_bin_offset_m.gather(
        1, selected_bin[:, None]
    ).squeeze(1)
    selected_depth = DEPTH_NEAR_EDGE_M + (
        selected_bin.to(dtype=selected_offset.dtype) + 0.5
    ) * DEPTH_BIN_SIZE_M + selected_offset
    selected_depth = torch.where(
        finite_hit, selected_depth, torch.zeros_like(selected_depth)
    )
    if not bool(torch.isfinite(selected_depth).all().item()):
        raise FloatingPointError("hard adapter depth became nonfinite")
    ground_clear = raw_output.ground_query_in_frustum & (
        raw_output.ground_clear_to_target_logits >= 0.0
    )
    return finite_hit, selected_depth, ground_clear


class HardRasterConfusion:
    """Streaming exact 3-class confusion counts for deterministic labels."""

    def __init__(self) -> None:
        self._matrix = [[0, 0, 0] for _ in range(3)]
        self._frame_count = 0

    def update(self, predicted_labels: Sequence[Any], target_labels: Any) -> None:
        import numpy as np
        import torch

        if not isinstance(target_labels, torch.Tensor) or target_labels.ndim != 3:
            raise ValueError("target labels must have shape (B,H,W)")
        if len(predicted_labels) != int(target_labels.shape[0]):
            raise ValueError("hard prediction batch changed")
        target = target_labels.detach().to(device="cpu").numpy()
        for frame_index, prediction_value in enumerate(predicted_labels):
            prediction = np.asarray(prediction_value, dtype=np.uint8)
            if prediction.shape != target[frame_index].shape:
                raise ValueError("hard and target raster shapes differ")
            if not np.isin(prediction, (0, 1, 2)).all():
                raise ValueError("hard raster contains an unsupported class")
            for target_class in range(3):
                target_mask = target[frame_index] == target_class
                for predicted_class in range(3):
                    self._matrix[target_class][predicted_class] += int(
                        np.count_nonzero(target_mask & (prediction == predicted_class))
                    )
        self._frame_count += len(predicted_labels)

    def finalize(self) -> dict[str, Any]:
        matrix = [list(row) for row in self._matrix]
        recalls: dict[str, float | None] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            count = sum(matrix[class_index])
            recalls[class_name] = (
                None if count == 0 else matrix[class_index][class_index] / count
            )
        present = [value for value in recalls.values() if value is not None]
        if not present:
            raise RuntimeError("hard raster confusion is empty")
        return {
            "confusion_target_rows_predicted_columns": matrix,
            "class_recalls": recalls,
            "balanced_accuracy": sum(present) / len(present),
            "cell_count": sum(sum(row) for row in matrix),
            "frame_count": self._frame_count,
            "nll": None,
        }


def direct_metric_projection(
    scopes: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    if set(scopes) != set(ALL_SCOPES):
        raise PermissionError("physical metric scopes changed")
    result: dict[str, dict[str, Any]] = {}
    for scope in ALL_SCOPES:
        row = scopes[scope]
        if any(key not in row for key in DIRECT_METRIC_KEYS):
            raise PermissionError(f"direct metric projection changed in {scope}")
        result[scope] = {key: row[key] for key in DIRECT_METRIC_KEYS}
    return result


def evaluate_materiality(
    hard_scopes: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if set(hard_scopes) != set(ALL_SCOPES):
        raise PermissionError("hard raster scopes changed")
    scope_rows: dict[str, Any] = {}
    passing_gain_count = 0
    wrong_rgb_guard_count = 0
    for scope in ALL_SCOPES:
        row = hard_scopes[scope]
        matched = row.get("matched")
        wrong = row.get("wrong")
        if type(matched) is not dict or type(wrong) is not dict:
            raise PermissionError(f"hard raster arm metrics changed in {scope}")
        matched_ba = float(matched["balanced_accuracy"])
        wrong_ba = float(wrong["balanced_accuracy"])
        if not math.isfinite(matched_ba) or not math.isfinite(wrong_ba):
            raise FloatingPointError("hard raster balanced accuracy became nonfinite")
        wrong_drop = matched_ba - wrong_ba
        baseline = SOFT_RASTER_BALANCED_ACCURACY.get(scope)
        gain = None if baseline is None else matched_ba - baseline
        gain_pass = (
            None
            if gain is None
            else gain >= MINIMUM_SCOPE_BALANCED_ACCURACY_GAIN
        )
        wrong_guard = (
            None if scope == ROUGH_SCOPE else wrong_drop >= MINIMUM_WRONG_RGB_DROP
        )
        if gain_pass is True:
            passing_gain_count += 1
        if wrong_guard is True:
            wrong_rgb_guard_count += 1
        scope_rows[scope] = {
            "soft_matched_balanced_accuracy": baseline,
            "hard_matched_balanced_accuracy": matched_ba,
            "hard_wrong_balanced_accuracy": wrong_ba,
            "hard_minus_soft_matched_balanced_accuracy": gain,
            "hard_matched_minus_wrong_balanced_accuracy": wrong_drop,
            "balanced_accuracy_gain_at_least_0_05": gain_pass,
            "wrong_rgb_drop_at_least_0_12": wrong_guard,
        }

    aggregate_recalls = hard_scopes["aggregate"]["matched"]["class_recalls"]
    free_gain = (
        float(aggregate_recalls["free"]) - SOFT_AGGREGATE_FREE_RECALL
    )
    occupied_gain = (
        float(aggregate_recalls["occupied"]) - SOFT_AGGREGATE_OCCUPIED_RECALL
    )
    criteria = {
        "at_least_six_of_eight_non_rough_balanced_accuracy_gains": (
            passing_gain_count >= MINIMUM_PASSING_SCOPE_COUNT
        ),
        "aggregate_free_recall_gain_at_least_0_05": (
            free_gain >= MINIMUM_AGGREGATE_RECALL_GAIN
        ),
        "aggregate_occupied_recall_gain_at_least_0_05": (
            occupied_gain >= MINIMUM_AGGREGATE_RECALL_GAIN
        ),
        "all_eight_non_rough_wrong_rgb_drops_at_least_0_12": (
            wrong_rgb_guard_count == len(NON_ROUGH_SCOPES)
        ),
    }
    passed = all(criteria.values())
    return {
        "scope_comparisons": scope_rows,
        "non_rough_scope_gain_pass_count": passing_gain_count,
        "required_non_rough_scope_gain_pass_count": MINIMUM_PASSING_SCOPE_COUNT,
        "non_rough_wrong_rgb_guard_pass_count": wrong_rgb_guard_count,
        "required_non_rough_wrong_rgb_guard_pass_count": len(NON_ROUGH_SCOPES),
        "aggregate_free_recall_gain": free_gain,
        "aggregate_occupied_recall_gain": occupied_gain,
        "criteria": criteria,
        "scientific_verdict": (
            "PASS_MATERIAL_HARD_RASTER_LOCALIZATION"
            if passed
            else "FAIL_HYPOTHESIS_REJECTED"
        ),
        "passed": passed,
    }


__all__ = [
    "ACCESS_SCHEMA",
    "ALL_SCOPES",
    "AUTHORIZATION_RELATIVE_PATH",
    "AUTHORIZATION_SCHEMA",
    "CLASS_NAMES",
    "COMPLETION_SCHEMA",
    "DIRECT_METRIC_KEYS",
    "EXECUTION_AUTHORITY",
    "FAILURE_SCHEMA",
    "HardRasterConfusion",
    "NON_ROUGH_SCOPES",
    "OUTPUT_ROOT_RELATIVE_PATH",
    "PREREGISTRATION_RELATIVE_PATH",
    "RAW_AUDIT_BINDING",
    "RAW_MANIFEST_BINDING",
    "RESERVATION_SCHEMA",
    "RESULT_SCHEMA",
    "REVIEW_RELATIVE_PATH",
    "REVIEW_SCHEMA",
    "ROUGH_SCOPE",
    "SCHEMA_PREFIX",
    "SELECTION_ENDPOINT_COUNT",
    "SELECTION_PAIR_COUNT",
    "SOFT_AGGREGATE_FREE_RECALL",
    "SOFT_AGGREGATE_OCCUPIED_RECALL",
    "SOFT_RASTER_BALANCED_ACCURACY",
    "V6_CHECKPOINT_BINDING",
    "V6_SIDECAR_LOCAL_CHECKPOINT_PATH",
    "V6_SIDECAR_BINDING",
    "V6_TERMINAL_AUDIT_BINDING",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "decode_hard_evidence_tensors",
    "direct_metric_projection",
    "evaluate_materiality",
    "hard_raster_labels_from_raw_output",
    "validate_content_sha256",
    "validate_v6_sidecar_checkpoint_binding",
    "with_content_sha256",
]
