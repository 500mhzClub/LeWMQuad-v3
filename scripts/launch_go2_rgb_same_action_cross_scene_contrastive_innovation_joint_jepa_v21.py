#!/usr/bin/env python3
"""Denied-by-default launcher for V21 cross-scene latent innovation.

The reviewed V20 launcher is loaded privately so its custody, data loading,
evaluation, paired-control capture, and write-once execution path remain
unchanged.  V21 adds exactly one train-batch field: before inherited tensor
construction, selected pair metadata determines the first cyclic row from a
different scene; after construction that mapping is appended as one int64
tensor.  This source shell grants no execution authority.
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
    "scripts/launch_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19.py"
)
BASE_LAUNCHER_COMMIT = "04c383183dab586bb3395acfceaaa749e55ff3ce"
BASE_LAUNCHER_FILE_SHA256 = (
    "be51d8da0c7f564124afa0f9f647b8f0c28e821078adc14c315fca8b91d422da"
)
BASE_LAUNCHER_BYTE_COUNT = 17_196
BASE_PUBLIC_MODULE_NAME = (
    "scripts.launch_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v19"
)
PRIVATE_BASE_MODULE_NAME = "_lewm_v21_scene_innovation_private_v20_launcher"
_PUBLIC_BASE_WAS_LOADED_BEFORE_ADAPTER = BASE_PUBLIC_MODULE_NAME in sys.modules

PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_preregistration_2026-07-30.md"
)
PREREGISTRATION_COMMIT = "c2bbce067175dd980c9ed2511dc14db5a222afe4"
PREREGISTRATION_FILE_SHA256 = (
    "f4ff1453e5cb63677dad66253d568c9204bd5504b3b3871e2b0c341402b1850e"
)
PREREGISTRATION_BYTE_COUNT = 11_594
V20_SCIENTIFIC_RESULT_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
    "semantic_grounding_joint_jepa_v20_scientific_result_2026-07-30.json"
)
V20_SCIENTIFIC_RESULT_COMMIT = "8321d76004aa1f3c87dfa04c3b18d701267a89ec"
V20_SCIENTIFIC_RESULT_FILE_SHA256 = (
    "d76fd16732d15b7637bbe8f68df65ba23990046812f4ec3d85297f7f8ea64956"
)
V20_SCIENTIFIC_RESULT_BYTE_COUNT = 17_166
V20_SCIENTIFIC_RESULT_CONTENT_SHA256 = (
    "37f683c1b2a5086c92d9cb081e9ba55b4fef4ed61f8cefea99fb0e5760e5cab2"
)

CERTIFIED_SOURCE_ROOT = (
    "/home/andrewknowles/Workspace/LeWMQuad-v3-v21-scene-innovation-source"
)
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21/attempt_v1"
)
AUTHORITY_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_execution_authorization_2026-07-30.json"
)
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_source_manifest_2026-07-30.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_source_review_2026-07-30.json"
)
CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_clean_export_certification_2026-07-30.json"
)
SOURCE_CLOSURE_CHECKER_RELATIVE_PATH = (
    "scripts/check_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21_source_closure.py"
)
EXECUTOR_MODULE_NAME = (
    "scripts.execute_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
MODEL_MODULE_NAME = (
    "lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_"
    "object_space_height_volume"
)
TRAINING_MODULE_NAME = (
    "scripts.run_go2_rgb_same_action_cross_scene_contrastive_innovation_"
    "joint_jepa_v21"
)
SOURCE_EVIDENCE_SCHEMA_PREFIX = (
    "lewm_go2_rgb_same_action_cross_scene_contrastive_innovation_joint_jepa_v21"
)
EXPERIMENT_ARM_NAME = "same_action_cross_scene_contrastive_innovation_v21"
LAUNCHER_SCHEMA = f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_launcher_v1"
SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 = "scene_innovation_negative_row"
MICROBATCH_SIZE_V21 = 4
SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21 = (
    "scene_innovation_negative_row_preflight_v21"
)
_SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V21 = (
    "_scene_innovation_negative_row_preflight_token_v21"
)
_SCHEDULE_PREFLIGHT_TOKEN_V21 = object()


_PRISTINE_V20_DEFAULTS = {
    "AUTHORITY_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20_execution_authorization_2026-07-30.json"
    ),
    "SOURCE_MANIFEST_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20_source_manifest_2026-07-30.json"
    ),
    "SOURCE_REVIEW_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20_source_review_2026-07-30.json"
    ),
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        "docs/lewm_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v20_clean_export_certification_2026-07-30.json"
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": (
        "scripts/check_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19_source_closure.py"
    ),
    "EXECUTOR_MODULE_NAME": (
        "scripts.execute_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19"
    ),
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": (
        "scripts.run_go2_rgb_object_space_height_volume_executed_successor_"
        "semantic_grounding_joint_jepa_v19"
    ),
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": (
        "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
        "grounding_joint_jepa_v20"
    ),
    "EXPERIMENT_ARM_NAME": (
        "object_space_height_volume_executed_successor_semantic_grounding_v20"
    ),
    "LAUNCHER_SCHEMA": (
        "lewm_go2_rgb_object_space_height_volume_executed_successor_semantic_"
        "grounding_joint_jepa_v20_launcher_v1"
    ),
}
_V21_BASE_OVERRIDES = {
    "AUTHORITY_RELATIVE_PATH": AUTHORITY_RELATIVE_PATH,
    "SOURCE_MANIFEST_RELATIVE_PATH": SOURCE_MANIFEST_RELATIVE_PATH,
    "SOURCE_REVIEW_RELATIVE_PATH": SOURCE_REVIEW_RELATIVE_PATH,
    "CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH": (
        CLEAN_EXPORT_CERTIFICATION_RELATIVE_PATH
    ),
    "SOURCE_CLOSURE_CHECKER_RELATIVE_PATH": SOURCE_CLOSURE_CHECKER_RELATIVE_PATH,
    "EXECUTOR_MODULE_NAME": EXECUTOR_MODULE_NAME,
    "MODEL_MODULE_NAME": MODEL_MODULE_NAME,
    "TRAINING_MODULE_NAME": TRAINING_MODULE_NAME,
    "SOURCE_EVIDENCE_SCHEMA_PREFIX": SOURCE_EVIDENCE_SCHEMA_PREFIX,
    "EXPERIMENT_ARM_NAME": EXPERIMENT_ARM_NAME,
    "LAUNCHER_SCHEMA": LAUNCHER_SCHEMA,
}


def _load_private_v20_launcher() -> Any:
    path = ROOT / BASE_LAUNCHER_RELATIVE_PATH
    try:
        root = ROOT.resolve(strict=True)
        resolved = path.resolve(strict=True)
    except (FileNotFoundError, OSError) as error:
        raise PermissionError("reviewed V20 launcher is absent") from error
    if (
        resolved != path.absolute()
        or not resolved.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
    ):
        raise PermissionError("reviewed V20 launcher escaped or is not regular")
    source = path.read_bytes()
    if (
        len(source) != BASE_LAUNCHER_BYTE_COUNT
        or hashlib.sha256(source).hexdigest() != BASE_LAUNCHER_FILE_SHA256
    ):
        raise PermissionError("reviewed V20 launcher binding changed")
    if PRIVATE_BASE_MODULE_NAME in sys.modules:
        raise RuntimeError("private V20 launcher module name is already occupied")
    spec = importlib.util.spec_from_file_location(PRIVATE_BASE_MODULE_NAME, path)
    if spec is None or spec.loader is None:
        raise ImportError("cannot load reviewed V20 launcher")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PRIVATE_BASE_MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
        if Path(module.ROOT).resolve(strict=True) != root:
            raise PermissionError("reviewed V20 launcher resolved another root")
        module._assert_configured_base_v19()
        observed = {
            name: getattr(module, name, None) for name in _PRISTINE_V20_DEFAULTS
        }
        if observed != _PRISTINE_V20_DEFAULTS:
            raise PermissionError("reviewed V20 launcher defaults changed")
        return module
    except BaseException:
        sys.modules.pop(PRIVATE_BASE_MODULE_NAME, None)
        raise


_V20_LAUNCHER = _load_private_v20_launcher()
_V18_LAUNCHER = _V20_LAUNCHER._V18_LAUNCHER
_BASE_LAUNCHER = _V20_LAUNCHER._BASE_LAUNCHER
_ORIGINAL_BUILD_ONE_MICROBATCH_V13 = _BASE_LAUNCHER._build_one_microbatch_v13

for _name, _value in _V21_BASE_OVERRIDES.items():
    setattr(_V20_LAUNCHER, _name, _value)
    _V20_LAUNCHER._V19_BASE_OVERRIDES[_name] = _value
    setattr(_V18_LAUNCHER, _name, _value)
    _V18_LAUNCHER._V18_BASE_OVERRIDES[_name] = _value
    setattr(_BASE_LAUNCHER, _name, _value)


def first_cyclic_different_scene_rows_v21(
    scene_ids: Sequence[str],
) -> tuple[int, ...]:
    """Select the first cyclic different-scene row for every B=4 row."""

    values = tuple(scene_ids)
    if len(values) != MICROBATCH_SIZE_V21 or any(
        not isinstance(scene_id, str) or not scene_id for scene_id in values
    ):
        raise ValueError("V21 scene IDs must be four nonempty strings")
    negative_rows: list[int] = []
    for row, scene_id in enumerate(values):
        selected = next(
            (
                (row + offset) % MICROBATCH_SIZE_V21
                for offset in range(1, MICROBATCH_SIZE_V21)
                if values[(row + offset) % MICROBATCH_SIZE_V21] != scene_id
            ),
            None,
        )
        if selected is None:
            raise PermissionError(
                "V21 microbatch has no different-scene negative for every row"
            )
        negative_rows.append(selected)
    result = tuple(negative_rows)
    if any(
        index == row or values[index] == values[row]
        for row, index in enumerate(result)
    ):
        raise RuntimeError("V21 different-scene negative-row selection failed")
    return result


def negative_rows_from_train_metadata_v21(
    runtime: Any,
    indices: Sequence[int],
) -> tuple[int, ...]:
    """Validate selected train-pair metadata without opening tensor payloads."""

    selected_indices = tuple(indices)
    if len(selected_indices) != MICROBATCH_SIZE_V21 or any(
        type(index) is not int or index < 0 for index in selected_indices
    ):
        raise PermissionError("V21 train indices must be four nonnegative integers")
    pairs = getattr(runtime, "pairs", None)
    if not isinstance(pairs, Mapping) or "train" not in pairs:
        raise PermissionError("V21 runtime train-pair registry is absent")
    train_pairs = pairs["train"]
    if not isinstance(train_pairs, Sequence) or isinstance(
        train_pairs, (str, bytes, bytearray)
    ):
        raise PermissionError("V21 train-pair registry is malformed")
    selected: list[Mapping[str, Any]] = []
    for index in selected_indices:
        if index >= len(train_pairs):
            raise PermissionError("V21 selected train index escaped the registry")
        row = train_pairs[index]
        if type(row) is not dict or "scene_id" not in row:
            raise PermissionError("V21 selected train-pair metadata is malformed")
        selected.append(row)
    scene_ids = tuple(row["scene_id"] for row in selected)
    try:
        return first_cyclic_different_scene_rows_v21(scene_ids)
    except (TypeError, ValueError) as error:
        raise PermissionError("V21 selected train scene metadata is malformed") from error


def preflight_schedule_negative_rows_v21(runtime: Any) -> Mapping[str, Any]:
    """Validate all 4,000 scheduled microbatches before the first tensor read."""

    cached = getattr(runtime, SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21, None)
    token = getattr(runtime, _SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V21, None)
    if token is _SCHEDULE_PREFLIGHT_TOKEN_V21:
        expected_fixed = {
            "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1",
            "schedule_index_count": 16_000,
            "microbatch_count": 4_000,
            "negative_row_index_count": 16_000,
            "all_rows_nonself_different_scene": True,
            "passed": True,
        }
        if (
            not isinstance(cached, Mapping)
            or any(cached.get(name) != value for name, value in expected_fixed.items())
            or not isinstance(cached.get("negative_row_mapping_sha256"), str)
            or len(cached["negative_row_mapping_sha256"]) != 64
        ):
            raise RuntimeError("V21 cached schedule preflight receipt changed")
        return cached
    if cached is not None or token is not None:
        raise PermissionError("V21 schedule preflight cache was pre-populated")

    schedule = getattr(runtime, "schedule", None)
    if not isinstance(schedule, Sequence) or isinstance(
        schedule, (str, bytes, bytearray)
    ):
        raise PermissionError("V21 frozen schedule is malformed")
    scheduled_indices = tuple(schedule)
    if len(scheduled_indices) != _BASE_LAUNCHER.MAXIMUM_PRESENTATIONS:
        raise PermissionError("V21 frozen schedule is not exactly 16000 indices")

    mappings: list[tuple[int, ...]] = []
    for start in range(0, len(scheduled_indices), MICROBATCH_SIZE_V21):
        chunk = scheduled_indices[start : start + MICROBATCH_SIZE_V21]
        mappings.append(negative_rows_from_train_metadata_v21(runtime, chunk))
    flattened_count = sum(len(mapping) for mapping in mappings)
    if len(mappings) != 4_000 or flattened_count != 16_000:
        raise RuntimeError("V21 schedule negative-row accounting changed")
    mapping_sha256 = hashlib.sha256(
        json.dumps(
            mappings,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    receipt = {
        "schema": f"{SOURCE_EVIDENCE_SCHEMA_PREFIX}_schedule_preflight_v1",
        "schedule_index_count": len(scheduled_indices),
        "microbatch_count": len(mappings),
        "negative_row_index_count": flattened_count,
        "negative_row_mapping_sha256": mapping_sha256,
        "all_rows_nonself_different_scene": True,
        "passed": True,
    }
    setattr(runtime, SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21, receipt)
    setattr(
        runtime,
        _SCHEDULE_PREFLIGHT_TOKEN_ATTRIBUTE_V21,
        _SCHEDULE_PREFLIGHT_TOKEN_V21,
    )
    return receipt


def _build_one_microbatch_v21(
    *,
    runtime: Any,
    indices: Sequence[int],
    stage: str,
) -> Mapping[str, Any]:
    # The entire schedule and then the current batch are checked before the
    # inherited builder can open or construct a tensor.
    preflight_schedule_negative_rows_v21(runtime)
    negative_rows = negative_rows_from_train_metadata_v21(runtime, indices)
    training = getattr(runtime, "training_module", None)
    if (
        getattr(training, "SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21", None)
        != SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21
    ):
        raise RuntimeError("V21 training negative-row key changed")
    inherited_keys = tuple(getattr(training, "REQUIRED_BATCH_KEYS", ()))
    required_v21 = tuple(getattr(training, "REQUIRED_BATCH_KEYS_V21", ()))
    if (
        not inherited_keys
        or SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21 in inherited_keys
        or required_v21
        != (*inherited_keys, SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21)
    ):
        raise RuntimeError("V21 batch schema is not the exact one-field extension")

    inherited = _ORIGINAL_BUILD_ONE_MICROBATCH_V13(
        runtime=runtime,
        indices=indices,
        stage=stage,
    )
    if type(inherited) is not dict or tuple(inherited) != inherited_keys:
        raise RuntimeError("V21 inherited microbatch schema changed")
    current_rgb_key = getattr(training, "CURRENT_RGB_KEY", None)
    anchor = inherited.get(current_rgb_key)
    torch = getattr(runtime, "torch", None)
    if torch is None or not isinstance(anchor, torch.Tensor):
        raise RuntimeError("V21 inherited microbatch has no tensor device anchor")
    negative_tensor = torch.tensor(
        negative_rows,
        dtype=torch.int64,
        device=anchor.device,
    )
    if (
        tuple(negative_tensor.shape) != (MICROBATCH_SIZE_V21,)
        or negative_tensor.dtype != torch.int64
        or negative_tensor.device != anchor.device
    ):
        raise RuntimeError("V21 negative-row tensor construction changed")
    result = {
        **inherited,
        SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21: negative_tensor,
    }
    if tuple(result) != required_v21:
        raise RuntimeError("V21 composed microbatch key order changed")
    return result


_BASE_LAUNCHER._build_one_microbatch_v13 = _build_one_microbatch_v21

# Preserve the inherited one-pass comparison capture without rescoring.
CONTROL_NAMES = _V20_LAUNCHER.CONTROL_NAMES
REGISTERED_FAMILIES = _V20_LAUNCHER.REGISTERED_FAMILIES
COMPARISON_FIELDS = _V20_LAUNCHER.COMPARISON_FIELDS
_sanitize_comparison_v19 = _V20_LAUNCHER._sanitize_comparison_v19
_v12_observation_v19 = _V20_LAUNCHER._v12_observation_v19


def _assert_configured_base_v21() -> None:
    _V20_LAUNCHER._assert_configured_base_v19()
    observed = {
        name: getattr(_BASE_LAUNCHER, name, None) for name in _V21_BASE_OVERRIDES
    }
    outer = {
        name: getattr(_V18_LAUNCHER, name, None) for name in _V21_BASE_OVERRIDES
    }
    if observed != _V21_BASE_OVERRIDES or outer != _V21_BASE_OVERRIDES:
        raise PermissionError("private V21 launcher adaptation changed after import")
    if _V20_LAUNCHER._V19_BASE_OVERRIDES != _V21_BASE_OVERRIDES:
        raise PermissionError("private V21 V20-selector binding changed after import")
    if _V18_LAUNCHER._V18_BASE_OVERRIDES != _V21_BASE_OVERRIDES:
        raise PermissionError("private V21 V18-selector binding changed after import")
    if _BASE_LAUNCHER._build_one_microbatch_v13 is not _build_one_microbatch_v21:
        raise PermissionError("private V21 microbatch hook changed after import")
    if _BASE_LAUNCHER._v12_observation_v13 is not _v12_observation_v19:
        raise PermissionError("private V21 comparison hook changed after import")


def _load_authority_file_v21(path: Path) -> dict[str, Any]:
    _assert_configured_base_v21()
    return _V20_LAUNCHER._load_authority_file_v19(path)


def execute_future_authorized_v21(
    *,
    repository_root: Path,
    authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    _assert_configured_base_v21()
    return _V20_LAUNCHER.execute_future_authorized_v19(
        repository_root=repository_root,
        authority=authority,
    )


def private_launcher_adapter_receipt_v21() -> dict[str, Any]:
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
            "path": V20_SCIENTIFIC_RESULT_RELATIVE_PATH,
            "commit": V20_SCIENTIFIC_RESULT_COMMIT,
            "file_sha256": V20_SCIENTIFIC_RESULT_FILE_SHA256,
            "byte_count": V20_SCIENTIFIC_RESULT_BYTE_COUNT,
            "content_sha256": V20_SCIENTIFIC_RESULT_CONTENT_SHA256,
        },
        "model_module": MODEL_MODULE_NAME,
        "training_module": TRAINING_MODULE_NAME,
        "executor_module": EXECUTOR_MODULE_NAME,
        "certified_source_root": CERTIFIED_SOURCE_ROOT,
        "output_root": OUTPUT_ROOT_RELATIVE_PATH,
        "negative_row_key": SCENE_INNOVATION_NEGATIVE_ROW_KEY_V21,
        "negative_row_selection": "first_cyclic_different_scene_offset_1_to_3",
        "metadata_validated_before_inherited_tensor_construction": True,
        "full_schedule_metadata_preflight_before_first_tensor": True,
        "schedule_preflight_receipt_attribute": (
            SCHEDULE_PREFLIGHT_RECEIPT_ATTRIBUTE_V21
        ),
        "exact_one_field_batch_extension": True,
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
    authority = _load_authority_file_v21(parsed.future_authority)
    result = execute_future_authorized_v21(
        repository_root=ROOT,
        authority=authority,
    )
    print(json.dumps(dict(result), sort_keys=True))
    status = result.get("status")
    if status == "PASS_DEVELOPMENT_UPDATE1000_TERMINAL":
        return 0
    if isinstance(status, str) and status.startswith("FAIL_"):
        return 2
    raise RuntimeError("V21 controller returned a nonterminal status")


if __name__ == "__main__":
    raise SystemExit(main())
