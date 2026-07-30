#!/usr/bin/env python3
"""Denied-by-default V23 launcher over the frozen V21 lifecycle.

V23 retains the different-scene preflight and appends one detached copy of
the already-computed train-action mean prior to each microbatch.  No image,
label, encoder, or predictor work is added.  This source grants no execution
authority.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
BASE_LAUNCHER_RELATIVE_PATH = (
    "scripts/launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21.py"
)
BASE_LAUNCHER_COMMIT = "7071a006dda3851280fbdf030e156862c4f19ab3"
BASE_LAUNCHER_FILE_SHA256 = (
    "4cb6fb3302919d6090e3ef456068ba209890237d65c87c8515a723628ee5b486"
)
BASE_LAUNCHER_BYTE_COUNT = 22_650
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v23_scene_action_private_v21_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "a7cf9692dd93212a82cb598d3175ff1c3598941b"
PREREGISTRATION_FILE_SHA256 = (
    "d5702759866138db1467778553ef8494d05f4593fcca14822050b1e0991180ae"
)
PREREGISTRATION_BYTE_COUNT = 14_294
V22_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_scene_action_contrastive_innovation_joint_jepa_v22_"
    "scientific_result_2026-07-30.json"
)
V22_SCIENTIFIC_RESULT_COMMIT = "f184a41ac99b1c66ea4db1e0b0a0845f23b48bbd"
V22_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "1f4896e8f0ae8cadbf09e6f6f34417f3fa6362f9321cfd5abd0aeb09735453d0"
)
V22_SCIENTIFIC_RESULT_BYTE_COUNT = 18_445
V22_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "d9c0376f381bb65c4246c9ff12611f4b563698a0539f81c63b95e8b083de18a2"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/LeWMQuad-v3-v23-survival-output-contrast-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23/attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_"
    "v23_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_action_prior_residualized_wrong_scene_survival_output_joint_jepa_v23"
)
EXPERIMENT_ARM_NAME = "action_prior_residualized_wrong_scene_survival_output_v23"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"

_V23_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "SOURCE_MANIFEST_RELATIVE_PATH": SOURCE_MANIFEST_RELATIVE_PATH,
    "SOURCE_REVIEW_RELATIVE_PATH": SOURCE_REVIEW_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH,
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SOURCE_EVIDENCE_SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}


def _load_private_v21_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("frozen V21 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("frozen V21 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("frozen V21 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V21 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V21 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    # V21 deliberately retains its privately loaded V20 launcher.  A public
    # V21 source-only test may therefore have occupied that fixed private name
    # earlier in the same interpreter.  The old V21 module keeps a direct
    # object reference, so release only the registry name before executing an
    # independently bound V21 instance for V23.
    inherited_private_name = "_lewm_v21_scene_innovation_private_v20_launcher"
    previous_inherited = sys.modules.pop(inherited_private_name, None)
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("frozen V21 launcher resolved another root")
        module._assert_configured_base_v21()
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        if previous_inherited is not None:
            sys.modules[inherited_private_name] = previous_inherited
        raise


_V21_LAUNCHER = _load_private_v21_launcher()
_V20_LAUNCHER = _V21_LAUNCHER._V20_LAUNCHER
_V18_LAUNCHER = _V21_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V21_LAUNCHER._BASE_LAUNCHER

# Retarget only lifecycle selectors.  V21's reviewed metadata preflight and
# microbatch hook remain the exact installed callables.
for _name, _value in _V23_BASE_OVERRIDES.items():
    setattr(_V21_LAUNCHER, _name, _value)
    _V21_LAUNCHER._V21_BASE_OVERRIDES[_name] = _value
    setattr(_V20_LAUNCHER, _name, _value)
    _V20_LAUNCHER._V19_BASE_OVERRIDES[_name] = _value
    setattr(_V18_LAUNCHER, _name, _value)
    _V18_LAUNCHER._V18_BASE_OVERRIDES[_name] = _value
    setattr(_BASE_LAUNCHER, _name, _value)

SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = (
    _V21_LAUNCHER.SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
)
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21 = (
    _V21_LAUNCHER.SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21
)
first_cyclic_different_scene_rows_v21 = (
    _V21_LAUNCHER.first_cyclic_different_scene_rows_v21
)
negative_rows_from_train_metadata_v21 = (
    _V21_LAUNCHER.negative_rows_from_train_metadata_v21
)
preflight_schedule_negative_rows_v21 = (
    _V21_LAUNCHER.preflight_schedule_negative_rows_v21
)
_build_one_microbatch_v21 = _V21_LAUNCHER._build_one_microbatch_v21
CONTROL_NAMES = _V21_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V21_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V21_LAUNCHER.COMPARISON_FIELDS
ACTION_PRIOR_M_KEY_V23 = "train_action_prior_m"
NON_HOLD_ACTION_INDICES_V23 = (0, 1, 2, 3, 4, 5, 7, 8)
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23 = (
    "_v23_state_residual_survival_schedule_preflight_receipt"
)
_SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23 = (
    "_v23_state_residual_survival_schedule_preflight_token"
)
_SCHEDULE_PREFLIGHT_TOKEN_V23 = object()
_INHERITED_BUILD_ONE_MICROBATCH_V21 = _build_one_microbatch_v21


def preflight_schedule_state_residual_survival_v23(runtime: Any) -> Mapping[str, Any]:
    """Prove every frozen microbatch has both registered comparison axes."""

    inherited = preflight_schedule_negative_rows_v21(runtime)
    cached = getattr(runtime, SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23, None)
    token = getattr(runtime, _SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23, None)
    if token is _SCHEDULE_PREFLIGHT_TOKEN_V23:
        np = getattr(runtime, "np", None)
        action_prior = getattr(runtime, "action_prior_m", None)
        if (
            not isinstance(cached, Mapping)
            or cached.get("schema")
            != f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1"
            or cached.get("microbatch_count") != 4_000
            or cached.get("scene_zero_microbatch_count") != 0
            or cached.get("prior_zero_microbatch_count") != 0
            or cached.get("passed") is not True
            or np is None
            or not isinstance(action_prior, np.ndarray)
            or action_prior.shape != (9,)
            or action_prior.dtype != np.float64
            or not np.isfinite(action_prior).all()
            or hashlib.sha256(action_prior.tobytes(order="C")).hexdigest()
            != cached.get("action_prior_sha256")
        ):
            raise RuntimeError("V23 cached schedule preflight changed")
        return cached
    if cached is not None or token is not None:
        raise PermissionError("V23 schedule preflight cache was pre-populated")

    np = getattr(runtime, "np", None)
    schedule = tuple(getattr(runtime, "schedule", ()))
    labels = getattr(runtime, "labels", None)
    pairs = getattr(runtime, "pairs", None)
    action_prior = getattr(runtime, "action_prior_m", None)
    if (
        np is None
        or len(schedule) != 16_000
        or not isinstance(labels, Mapping)
        or "train" not in labels
        or not isinstance(pairs, Mapping)
        or "train" not in pairs
    ):
        raise PermissionError("V23 frozen schedule or train metadata is absent")
    prefix = labels["train"].prefix_lengths
    if (
        not isinstance(prefix, np.ndarray)
        or prefix.shape != (4_262, 9)
        or not np.issubdtype(prefix.dtype, np.integer)
        or not isinstance(action_prior, np.ndarray)
        or action_prior.shape != (9,)
        or action_prior.dtype != np.float64
        or not np.isfinite(action_prior).all()
    ):
        raise PermissionError("V23 frozen prefix or prior arrays changed")
    expected_prior = prefix.mean(axis=0, dtype=np.float64) * 0.1
    if not np.array_equal(action_prior, expected_prior):
        raise PermissionError("V23 train-action prior changed from frozen labels")
    train_pairs = pairs["train"]
    if len(train_pairs) != 4_262:
        raise PermissionError("V23 train-pair count changed")

    scene_counts: list[int] = []
    prior_counts: list[int] = []
    mappings: list[tuple[int, ...]] = []
    for start in range(0, len(schedule), 4):
        indices = schedule[start : start + 4]
        negatives = negative_rows_from_train_metadata_v21(runtime, indices)
        mappings.append(tuple(negatives))
        batch_prefix = prefix[np.asarray(indices, dtype=np.int64)]
        action_index = np.asarray(NON_HOLD_ACTION_INDICES_V23, dtype=np.int64)
        selected = batch_prefix[:, action_index]
        negative_selected = batch_prefix[
            np.asarray(negatives, dtype=np.int64)[:, None], action_index[None, :]
        ]
        scene_count = int((selected != negative_selected).sum())
        prior_count = int(
            (selected.astype(np.float64) * 0.1 != action_prior[action_index]).sum()
        )
        if scene_count < 1 or prior_count < 1:
            raise PermissionError("V23 schedule contains an empty comparison axis")
        scene_counts.append(scene_count)
        prior_counts.append(prior_count)
    if len(scene_counts) != 4_000:
        raise RuntimeError("V23 schedule microbatch count changed")
    comparison_sha256 = hashlib.sha256(
        json.dumps(
            {
                "mappings": mappings,
                "scene_counts": scene_counts,
                "prior_counts": prior_counts,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1",
        "inherited_negative_row_mapping_sha256": inherited[
            "negative_row_mapping_sha256"
        ],
        "schedule_index_count": len(schedule),
        "microbatch_count": len(scene_counts),
        "scene_eligible_count_total": sum(scene_counts),
        "scene_eligible_count_min": min(scene_counts),
        "scene_eligible_count_max": max(scene_counts),
        "scene_zero_microbatch_count": 0,
        "prior_eligible_count_total": sum(prior_counts),
        "prior_eligible_count_min": min(prior_counts),
        "prior_eligible_count_max": max(prior_counts),
        "prior_zero_microbatch_count": 0,
        "action_prior_sha256": hashlib.sha256(
            action_prior.tobytes(order="C")
        ).hexdigest(),
        "comparison_mapping_sha256": comparison_sha256,
        "passed": True,
    }
    setattr(runtime, SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V23, receipt)
    setattr(
        runtime,
        _SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V23,
        _SCHEDULE_PREFLIGHT_TOKEN_V23,
    )
    return receipt


def _build_one_microbatch_v23(
    *, runtime: Any, indices: Sequence[int], stage: str
) -> Mapping[str, Any]:
    preflight_schedule_state_residual_survival_v23(runtime)
    inherited = _INHERITED_BUILD_ONE_MICROBATCH_V21(
        runtime=runtime, indices=indices, stage=stage
    )
    training = getattr(runtime, "training_module", None)
    inherited_keys = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V21", ()))
    required = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V23", ()))
    if (
        type(inherited) is not dict
        or tuple(inherited) != inherited_keys
        or required != (*inherited_keys, ACTION_PRIOR_M_KEY_V23)
        or getattr(training, "ACTION_PRIOR_M_KEY_V23", None)
        != ACTION_PRIOR_M_KEY_V23
    ):
        raise RuntimeError("V23 inherited microbatch or key contract changed")
    torch = getattr(runtime, "torch", None)
    anchor = inherited.get(getattr(training, "CURRENT_RGB_KEY", None))
    if torch is None or not isinstance(anchor, torch.Tensor):
        raise RuntimeError("V23 microbatch has no tensor device anchor")
    prior = torch.as_tensor(
        runtime.action_prior_m,
        dtype=torch.float32,
        device=anchor.device,
    ).clone().detach()
    if (
        tuple(prior.shape) != (9,)
        or prior.requires_grad
        or not bool(torch.isfinite(prior).all().item())
    ):
        raise RuntimeError("V23 prior tensor construction changed")
    result = {**inherited, ACTION_PRIOR_M_KEY_V23: prior}
    if tuple(result) != required:
        raise RuntimeError("V23 composed microbatch key order changed")
    return result


_BASE_LAUNCHER._build_one_microbatch_v13 = _build_one_microbatch_v23
_V21_LAUNCHER._build_one_microbatch_v21 = _build_one_microbatch_v23


def _assert_configured_base_v23() -> None:
    _V21_LAUNCHER._assert_configured_base_v21()
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V23_BASE_OVERRIDES
    }
    outer = {
        name: getattr(_V18_LAUNCHER, name, None) for name in _V23_BASE_OVERRIDES
    }
    if observed != _V23_BASE_OVERRIDES or outer != _V23_BASE_OVERRIDES:
        raise PermissionError("private V23 launcher adaptation changed after import")
    if _V21_LAUNCHER._V21_BASE_OVERRIDES != _V23_BASE_OVERRIDES:
        raise PermissionError("private V23 V21-selector binding changed")
    if _V20_LAUNCHER._V19_BASE_OVERRIDES != _V23_BASE_OVERRIDES:
        raise PermissionError("private V23 V20-selector binding changed")
    if _V18_LAUNCHER._V18_BASE_OVERRIDES != _V23_BASE_OVERRIDES:
        raise PermissionError("private V23 V18-selector binding changed")
    if (
        _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v23
        or _V21_LAUNCHER._build_one_microbatch_v21 is not _build_one_microbatch_v23
    ):
        raise PermissionError("private V23 microbatch hook changed")


def _load_authority_file_v23(path: Path) -> dict[str, Any]:
    _assert_configured_base_v23()
    return _V21_LAUNCHER._load_authority_file_v21(path)


def execute_future_authorized_v23(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v23()
    return _V21_LAUNCHER.execute_future_authorized_v21(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v23() -> dict[str, Any]:
    return {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_private_launcher_adapter_v1",
        "base_launcher": {
            "path": BASE_LAUNCHER_RELATIVE_PATH,
            "commit": BASE_LAUNCHER_COMMIT,
            "file_sha256": BASE_LAUNCHER_FILE_SHA256,
            "byte_count": BASE_LAUNCHER_BYTE_COUNT,
        },
        "preregistration": {
            "path": PREREGISTRATION_RELATIVE_PATH,
            "commit": PREREGISTRATION_COMMIT,
            "file_sha256": PREREGISTRATION_FILE_SHA256,
            "byte_count": PREREGISTRATION_BYTE_COUNT,
        },
        "predecessor_scientific_result": {
            "path": V22_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "commit": V22_SCIENTIFIC_RESULT_COMMIT,
            "file_sha256": V22_SCIENTIFIC_RESULT_FILE_SHA256,
            "byte_count": V22_SCIENTIFIC_RESULT_BYTE_COUNT,
            "content_sha256": V22_SCIENTIFIC_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "scene_negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "action_prior_batch_key": ACTION_PRIOR_M_KEY_V23,
        "scene_negative_preflight_inherited_exactly_from_v21": True,
        "state_residual_schedule_preflight": True,
        "new_batch_fields_over_v21": 1,
        "new_batch_fields_over_v22": 1,
        "prior_derived_once_from_loaded_train_labels": True,
        "extra_tensor_payload_reads": 0,
        "extra_predictor_forwards": 0,
        "numeric_comparisons_retained_without_rescoring": True,
        "update100_new_terminal_branch": False,
        "update400_and_update1000_gates_inherited": True,
        "maximum_updates": _BASE_LAUNCHER.MAXIMUM_UPDATES,
        "maximum_presentations": _BASE_LAUNCHER.MAXIMUM_PRESENTATIONS,
        "one_shot_attempt_count": 1,
        "retry_authorized": False,
        "resume_authorized": False,
        "public_base_was_loaded_before_adapter": (
            _PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER
        ),
        "public_base_loaded_by_adapter": BASE_PUBLIC_MODULE_NAME in sys.modules,
        "execution_authorized": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--future-authority", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = tuple(sys.argv[1:] if argv is None else argv)
    if not arguments:
        print(
            json.dumps(
                {
                    "schema": LAUNCHER_SCHEMA,
                    "status": "DENIED_NO_FUTURE_AUTHORITY",
                    "scientific_payload_opened": False,
                    "reservation_created": False,
                },
                sort_keys=True,
            )
        )
        return 4
    parsed = _parser().parse_args(arguments)
    authority = _load_authority_file_v23(parsed.future_authority)
    result = execute_future_authorized_v23(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V23 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
