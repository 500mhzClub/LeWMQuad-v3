"""Offline projective-support corridor labels for the 2026-07-28 JEPA probe.

This module is deliberately model-free.  It opens no RGB, checkpoints, runner
outputs, or evaluation material.  Its pure geometry functions are also used by
the training runner so that labels and score masks cannot acquire different
pose sampling or polygon rasterization rules.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
from typing import Any, BinaryIO, Iterable, Iterator, Mapping, MutableMapping, Sequence

import numpy as np
import yaml

from lewm.benchmarks import (
    go2_post_action_projective_support_corridor_contract_v1 as contract,
)
from lewm.planning.geometry_contract import GeometryContract, load_geometry_contract
from lewm.planning.oriented_footprint import (
    DirectionalSupportFootprint,
    ManifestDirectionalFootprintFeasibility,
    OrientedRectangle,
    Pose2D,
    convex_polygon_intersects_rectangle,
    wrap_angle_pi,
)
from lewm_worlds.manifest import (
    SceneManifest,
    manifest_sha256,
    parse_scene_manifest_dict,
)


PREREGISTRATION_COMMIT = "8a52adb77d30cb98a6dd086037e6f7c296d76d63"
PREREGISTRATION_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "preregistration_2026-07-28.md"
)
PREREGISTRATION_FILE_SHA256 = (
    "fe39daa2ff2f19624d67910d60a6da640f6351a6b5a6135db7b877bbc784e045"
)
PREREGISTRATION_BYTE_COUNT = 29_487
LABEL_EXECUTION_BINDING_RELATIVE_PATH = (
    "docs/lewm_go2_post_action_projective_support_labels_v4_"
    "execution_binding_2026-07-28.json"
)
LABEL_OUTPUT_RELATIVE_PATH = ".generated/go2_post_action_projective_support_labels_v4"
LABEL_RESERVATION_RELATIVE_PATH = f"{LABEL_OUTPUT_RELATIVE_PATH}/reservation.json"
LABEL_BUILDER_CLAIM_RELATIVE_PATH = f"{LABEL_OUTPUT_RELATIVE_PATH}/builder_claim.json"
SOURCE_MANIFEST_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "source_manifest_v4_2026-07-28.json"
)
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1_"
    "source_review_v4_2026-07-28.json"
)
LABEL_RESERVATION_SCHEMA = (
    "lewm_go2_post_action_projective_support_labels_v1_reservation_v1"
)
LABEL_BUILDER_CLAIM_SCHEMA = (
    "lewm_go2_post_action_projective_support_labels_v1_builder_claim_v1"
)
LABEL_FAILURE_SCHEMA = "lewm_go2_post_action_projective_support_labels_v1_failure_v1"
SCHEDULE_PREFIX_SHA256 = (
    "3f7b5799e855c3d218dcc62428f26ae0f9577c0dd4b04af5156d439a6f81e528"
)
REGISTERED_SELECTION_FAMILIES = (
    "large_enclosed_maze",
    "local_composite_motifs",
    "loop_alias_stress",
    "medium_enclosed_maze",
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "visual_sensor_stress",
)
ACTION_ORDER = (
    "arc_left",
    "arc_right",
    "backward",
    "forward_fast",
    "forward_medium",
    "forward_slow",
    "hold",
    "yaw_left",
    "yaw_right",
)
RAW_ROLE_ORDER = ("train", "checkpoint_selection", "probability_calibration")
OUTPUT_ROLE_ORDER = ("train", "probability_calibration", "checkpoint_selection")
ROLE_STATE_COUNTS = {
    "train": 4_262,
    "probability_calibration": 415,
    "checkpoint_selection": 495,
}
ROLE_SCENE_COUNTS = {
    "train": 72,
    "probability_calibration": 8,
    "checkpoint_selection": 8,
}
ROLE_ACTION_ROW_COUNTS = {
    role: count * len(ACTION_ORDER) for role, count in ROLE_STATE_COUNTS.items()
}

RAW_DATASET_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_dataset_v1"
RAW_PAIR_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_pair_v1"
RAW_ENDPOINT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_endpoint_index_v1"
RAW_AUDIT_SCHEMA = "lewm_go2_shared_jepa_v5_raw_supervision_audit_v13"
EXECUTION_BINDING_SCHEMA = (
    "lewm_go2_post_action_projective_support_labels_v1_execution_binding_v1"
)
LABEL_ROW_SCHEMA = "lewm_go2_post_action_projective_support_label_row_v1"
LABEL_MANIFEST_SCHEMA = "lewm_go2_post_action_projective_support_labels_v1"

RAW_MANIFEST_FILE_SHA256 = (
    "e102b3c64e99029f118597353966edaaaddbc11efe49b9081d5d7a9c9d974360"
)
RAW_MANIFEST_CONTENT_SHA256 = (
    "74ae5799919ff4d9a06f56d98929cb4cb702d64db52ecdfc93cfa9a8e82fb35a"
)
RAW_PAIRS_FILE_SHA256 = (
    "5a6f7de405206aba855051bd9e14cab5262cfbfebc070ed02ef81d8cf62afc8d"
)
RAW_ENDPOINTS_FILE_SHA256 = (
    "34e47ddcc40ad8c1f092c73193d16773cf4dedae05e7f4f684abb385cc2c0d01"
)
RAW_AUDIT_FILE_SHA256 = (
    "0680e1680f30c45feda60498792c3f208c28313e8f087dfbdd1c5807bcf1fe76"
)
RAW_AUDIT_CONTENT_SHA256 = (
    "0c16e368c9de258d0fbf46e3123d7a3cfcdf60162fd9efa6440d4a7773056aca"
)
SCHEDULE_FILE_SHA256 = (
    "08f54578febbc182d936a999d6cf86263b8cd03a5f640da064c1538dd53dc270"
)
SCHEDULE_CONTENT_SHA256 = (
    "274c0cbd9a87cbbc5bbc3123fff046f02ac3555014b5ec750d4a32b552650a15"
)
MATCHED_TRAINING_V4_SCHEDULE_SCHEMA = (
    "lewm_go2_shared_jepa_v5_matched_training_v4_schedule_v1"
)
MATCHED_TRAINING_V4_SCHEDULE_FIELDS = frozenset(
    {
        "schema",
        "seed",
        "train_pair_count",
        "presentation_count",
        "update_count",
        "microbatch_size",
        "accumulation_steps",
        "effective_batch_size",
        "ordered_pair_ids_sha256",
        "indices_sha256",
        "presentation_pair_ids_sha256",
        "per_update_pair_ids_sha256",
        "presentation_indices",
        "content_sha256",
    }
)
MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS = {
    "seed": 20_260_713,
    "train_pair_count": 4_262,
    "presentation_count": 128_000,
    "update_count": 8_000,
    "microbatch_size": 4,
    "accumulation_steps": 4,
    "effective_batch_size": 16,
}
MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256 = {
    "ordered_pair_ids_sha256": (
        "74b90f10347a89d2151c4f65f76d6fc3c6a94fb3e8caa350d2a92e934e80840a"
    ),
    "indices_sha256": (
        "a6f4fda5eb570336fb360631af3629832cccbe4cba21bdbb325dcb8a21963663"
    ),
    "presentation_pair_ids_sha256": (
        "1534dcdd85feb8421639a0dc433473913f6674556e22e0fa9f515be455b7b79a"
    ),
    "per_update_pair_ids_sha256": (
        "fe4aab82bd05b5e3438e8623319211ae75220f8bf3143223f6b6e375d91d46f0"
    ),
}
RAW_ORDERED_PAIR_SHA256 = (
    "76810dba883f3aaffb92fccb593d382daf7edca74a9bb5559a977e7e88b7b5ea"
)
RAW_ENDPOINT_INDEX_ORDER_SHA256 = (
    "ab21c1a89b37ef60a056de390d59d3983705ab2e40de061d0cb163d1837e850f"
)
RAW_METADATA_PLAN_ENDPOINT_ORDER_SHA256 = (
    "8130e961b7b5c04944b178fa4f73c1fa157776f7702ab5cdc213cf16c922f698"
)

GEOMETRY_CONTRACT_FILE_SHA256 = (
    "e7d0627d1de259c6e01dabe142aa55e69fed3e75c9c745974d437d7682d40a52"
)
GEOMETRY_CONTRACT_CONTENT_SHA256 = (
    "e06830cbffa67dedec4c20ecd3c1fb9873fe814f212bfa09ec0f160b6514d0ca"
)
DIRECTIONAL_POLICY_FILE_SHA256 = (
    "750d8afe47ee3edd5988cdea443f19703efad7a3266218932671b9fdfbe43828"
)
DIRECTIONAL_POLICY_CONTENT_SHA256 = (
    "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc"
)
DIRECTIONAL_POLICY_ID = "go2-directional-observed-max-margin-v1"
DIRECTIONAL_PROFILE = "observed_max_plus_margin"
PRIMITIVE_REGISTRY_FILE_SHA256 = (
    "cb83acf61d0e958b90d5dcd98e2ad11c630426bf480bd948aeb77242d84293f8"
)

COMMAND_DT_S = 0.10
COMMAND_COUNT = 5
MAXIMUM_CORNER_STEP_M = 0.025
MAXIMUM_YAW_STEP_RAD = math.radians(5.0)
REMOTE_SAMPLE_OFFSETS = (0, 1, 10, 19, 28, 37, 46, 55, 64, 73, 82, 91)
REMOTE_SAMPLE_SHA256 = (
    "df96a4d23e9f2a297467c7384e54e9d7f8eac64609e937392f0db51e3c87abc3"
)

GRID_SHAPE = (64, 64)
GRID_CELL_SIZE_M = 0.10
GRID_FORWARD_MIN_EDGE_M = -1.0
GRID_LEFT_MIN_EDGE_M = -3.2
PREDICTED_NEXT_MASK_SHA256 = (
    "63648c9c157d032db943b4dea5168879c287c847101606c56c97688f06e69da4"
)
PREDICTED_NEXT_MASK_COUNTS = (49, 61, 61, 61, 61, 61, 61, 61, 61, 61, 61)
PERSISTENCE_MASK_SHA256 = {
    "arc_left": "ea6e49053b653dd84250647f6ca51d5aa929df7cf84a214203a6c5822f186740",
    "arc_right": "77bf4e01900e559387a11f36a2c66a9859caee93c139032bb7e74c2296f3a1c2",
    "backward": "dfc0aeac0f6f8b44a8e37c7eac16dcfbd06ee98a7e1e3bf308f78413a472b08f",
    "forward_fast": "17a8e0b66a03c8d0210a7b0bf1665daa71ba8d355df2d344d5bf06feb3f6f773",
    "forward_medium": "4b78889928776d40f0c344d37dd942f91356da333b5a98ebb843fc966bb617d9",
    "forward_slow": "f651df5fead03d200477f1bfc418f17ed3bd613918c77a7615d65fbfdc75853f",
    "hold": PREDICTED_NEXT_MASK_SHA256,
    "yaw_left": "bcba50e628bd4557840db74e2e47b9a0513d5bd0b454cd3c863d4883e1d1e6f2",
    "yaw_right": "c91dc19501891039bee3d3b9a536de655a243f9b3e4e74a88e9d9da2888f180f",
}
PERSISTENCE_STACK_SHA256 = (
    "983577015f2822bbf60d89cd633baa9958afd624410e1a3e4390422647e59e34"
)
PROJECTIVE_SUPPORT_CELL_COUNT = 1_964

INPUT_RELATIVE_PATHS = {
    "raw_manifest": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1/manifest.json"
    ),
    "raw_pairs": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1/pairs.jsonl"
    ),
    "raw_endpoints": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1/endpoints.jsonl"
    ),
    "raw_audit": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "development_raw_supervision_v1.audit_v13.json"
    ),
    "schedule": (
        ".generated/go2_shared_observable_camera_ray_jepa_v5/"
        "matched_training_v4/schedule.json"
    ),
    "geometry_contract": "config/go2_generalization_geometry_v2.json",
    "directional_policy": (
        "config/go2_geometry_v2_artifacts/"
        "go2_directional_footprint_policy_v1_"
        "c57650326e8b7d302498bbfe93b9e3d15c36d56d55ae9e1f339507ece0a9f1fc.json"
    ),
    "primitive_registry": "config/go2_primitive_registry.yaml",
}

_FRAME_NAME = re.compile(r"^frame_([0-9]{6})_env_([0-9]{2})\.png$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_PURPOSES = (
    "source_frames_jsonl",
    "render_summary",
    "source_scene_manifest",
)


class LabelContractError(ValueError):
    """Raised before publication when a frozen label contract is violated."""


@dataclass(frozen=True)
class RawIndexesV1:
    manifest: Mapping[str, Any]
    pairs: tuple[Mapping[str, Any], ...]
    endpoints: tuple[Mapping[str, Any], ...]
    endpoint_by_sha256: Mapping[str, Mapping[str, Any]]
    shard_by_scene: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class GeometryInputsV1:
    geometry: GeometryContract
    footprint: DirectionalSupportFootprint
    commands_by_action: Mapping[str, tuple[tuple[float, float, float], ...]]
    source_bindings: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class JoinedCurrentStateV1:
    pair: Mapping[str, Any]
    endpoint: Mapping[str, Any]
    source_pose_world: Pose2D
    source_line_number: int
    source_bindings: Mapping[str, Mapping[str, Any]]


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def with_content_sha256(core: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(core)
    if "content_sha256" in value:
        raise LabelContractError("content hash cannot be supplied inside its core")
    return {**value, "content_sha256": canonical_json_sha256(value)}


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


_ACCESS_LEDGER_KEYS = (
    "execution_binding_opens",
    "source_manifest_opens",
    "independent_source_review_opens",
    "source_authority_validation_calls",
    "raw_manifest_opens",
    "raw_pairs_opens",
    "raw_endpoints_opens",
    "raw_audit_opens",
    "geometry_contract_opens",
    "geometry_contract_validation_calls",
    "directional_policy_opens",
    "primitive_registry_opens",
    "schedule_opens",
    "scene_join_calls_started",
    "render_summary_opens",
    "source_frames_jsonl_opens",
    "scene_manifest_opens",
    "rgb_opens",
    "checkpoint_opens",
    "runtime_output_opens",
    "g2_opens",
    "navigation_opens",
    "heldout_opens",
    "sealed_opens",
    "production_opens",
)


def new_access_ledger_v1() -> dict[str, int]:
    """Return the fixed, explicit access counters used by terminal receipts."""

    return {name: 0 for name in _ACCESS_LEDGER_KEYS}


def _normalize_access_ledger(value: Mapping[str, Any] | None) -> dict[str, int]:
    if value is None:
        return new_access_ledger_v1()
    if set(value) != set(_ACCESS_LEDGER_KEYS):
        raise LabelContractError("label access-ledger fields changed")
    return {
        key: _exact_int(value[key], name=f"access_ledger.{key}", minimum=0)
        for key in _ACCESS_LEDGER_KEYS
    }


def _record_access(
    access_ledger: MutableMapping[str, int] | None,
    key: str,
) -> None:
    if access_ledger is None:
        return
    if key not in _ACCESS_LEDGER_KEYS:
        raise LabelContractError(f"unknown access-ledger key: {key}")
    access_ledger[key] = int(access_ledger.get(key, 0)) + 1


def _normalize_authority_artifact(
    value: object,
    *,
    name: str,
    exact_path: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "path",
        "byte_count",
        "file_sha256",
        "content_sha256",
    }:
        raise LabelContractError(f"{name} artifact fields changed")
    if (
        value.get("path") != exact_path
        or _exact_int(value.get("byte_count"), name=f"{name}.byte_count", minimum=1)
        < 1
        or not _is_sha256(value.get("file_sha256"))
        or not _is_sha256(value.get("content_sha256"))
    ):
        raise LabelContractError(f"{name} artifact binding changed")
    return dict(value)


def _label_output_path(*, repository_root: Path) -> Path:
    return Path(repository_root).absolute() / LABEL_OUTPUT_RELATIVE_PATH


def _canonical_write_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _preregistration_binding_v1() -> dict[str, Any]:
    return {
        "path": PREREGISTRATION_RELATIVE_PATH,
        "commit": PREREGISTRATION_COMMIT,
        "file_sha256": PREREGISTRATION_FILE_SHA256,
        "byte_count": PREREGISTRATION_BYTE_COUNT,
    }


def reserve_label_root_v1(
    repository_root: Path,
    *,
    source_manifest: Mapping[str, Any],
    independent_source_review: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Atomically consume the sole label attempt before any runtime-input open."""

    manifest_record = _normalize_authority_artifact(
        source_manifest,
        name="source_manifest",
        exact_path=SOURCE_MANIFEST_RELATIVE_PATH,
    )
    review_record = _normalize_authority_artifact(
        independent_source_review,
        name="independent_source_review",
        exact_path=SOURCE_REVIEW_RELATIVE_PATH,
    )
    output = _label_output_path(repository_root=repository_root)
    output.parent.mkdir(parents=True, exist_ok=True)
    # mkdir without exist_ok is the one-shot compare-and-set.  No caller may
    # replace, reuse, or resume an existing target, regardless of its contents.
    output.mkdir(mode=0o700)
    reservation = with_content_sha256(
        {
            "schema": LABEL_RESERVATION_SCHEMA,
            "status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
            "preregistration": _preregistration_binding_v1(),
            "execution_binding_path": LABEL_EXECUTION_BINDING_RELATIVE_PATH,
            "source_manifest": manifest_record,
            "independent_source_review": review_record,
            "output_directory": LABEL_OUTPUT_RELATIVE_PATH,
            "attempt": {
                "index": 1,
                "maximum_attempts": 1,
                "retry_authorized": False,
                "resume_authorized": False,
                "second_invocation_authorized": False,
            },
            "access_ledger": new_access_ledger_v1(),
            "authority": {
                "development_label_preflight_authorized": True,
                "training_authorized": False,
                "g2_authorized": False,
                "navigation_authorized": False,
                "heldout_authorized": False,
                "sealed_authorized": False,
                "production_authorized": False,
                "promotion_authorized": False,
            },
        }
    )
    _canonical_write_exclusive(output / "reservation.json", reservation)
    return reservation


def load_label_reservation_v1(
    repository_root: Path,
    *,
    source_manifest: Mapping[str, Any] | None = None,
    independent_source_review: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    output = _label_output_path(repository_root=repository_root)
    if output.is_symlink() or not output.is_dir():
        raise PermissionError("exact label root was not atomically reserved")
    path = output / "reservation.json"
    if path.is_symlink() or not path.is_file():
        raise PermissionError("label reservation receipt is absent")
    value = _parse_canonical_object(path.read_bytes(), name="label reservation")
    manifest_record = _normalize_authority_artifact(
        value.get("source_manifest"),
        name="source_manifest",
        exact_path=SOURCE_MANIFEST_RELATIVE_PATH,
    )
    review_record = _normalize_authority_artifact(
        value.get("independent_source_review"),
        name="independent_source_review",
        exact_path=SOURCE_REVIEW_RELATIVE_PATH,
    )
    expected_core = {
        "schema": LABEL_RESERVATION_SCHEMA,
        "status": "RESERVED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT",
        "preregistration": _preregistration_binding_v1(),
        "execution_binding_path": LABEL_EXECUTION_BINDING_RELATIVE_PATH,
        "source_manifest": manifest_record,
        "independent_source_review": review_record,
        "output_directory": LABEL_OUTPUT_RELATIVE_PATH,
        "attempt": {
            "index": 1,
            "maximum_attempts": 1,
            "retry_authorized": False,
            "resume_authorized": False,
            "second_invocation_authorized": False,
        },
        "access_ledger": new_access_ledger_v1(),
        "authority": {
            "development_label_preflight_authorized": True,
            "training_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "heldout_authorized": False,
            "sealed_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    expected = with_content_sha256(expected_core)
    if value != expected:
        raise PermissionError("label reservation identity changed")
    if source_manifest is not None and dict(source_manifest) != manifest_record:
        raise PermissionError("source manifest escaped the label reservation")
    if independent_source_review is not None and dict(independent_source_review) != review_record:
        raise PermissionError("source review escaped the label reservation")
    return value


def claim_label_builder_v1(
    repository_root: Path,
    *,
    execution_binding_content_sha256: str,
) -> Mapping[str, Any]:
    """Atomically admit the sole builder invocation inside a reserved root."""

    if not _is_sha256(execution_binding_content_sha256):
        raise LabelContractError("execution binding content SHA-256 is malformed")
    reservation = load_label_reservation_v1(repository_root)
    output = _label_output_path(repository_root=repository_root)
    if any((output / name).exists() for name in ("failure.json", "manifest.json")):
        raise PermissionError("label attempt is already terminal")
    claim = with_content_sha256(
        {
            "schema": LABEL_BUILDER_CLAIM_SCHEMA,
            "status": "CLAIMED_ONE_EXACT_LABEL_BUILDER_INVOCATION",
            "reservation_content_sha256": reservation["content_sha256"],
            "execution_binding_content_sha256": execution_binding_content_sha256,
            "retry_authorized": False,
            "resume_authorized": False,
            "second_invocation_authorized": False,
        }
    )
    _canonical_write_exclusive(output / "builder_claim.json", claim)
    return claim


def write_label_failure_v1(
    repository_root: Path,
    *,
    phase: str,
    error: BaseException,
    source_manifest: Mapping[str, Any],
    independent_source_review: Mapping[str, Any],
    binding_content_sha256: str | None = None,
    schedule_prefix_sha256: str | None = None,
    access_ledger: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Publish the shared canonical terminal receipt after reservation."""

    phase = _exact_str(phase, name="failure phase")
    manifest_record = _normalize_authority_artifact(
        source_manifest,
        name="source_manifest",
        exact_path=SOURCE_MANIFEST_RELATIVE_PATH,
    )
    review_record = _normalize_authority_artifact(
        independent_source_review,
        name="independent_source_review",
        exact_path=SOURCE_REVIEW_RELATIVE_PATH,
    )
    reservation = load_label_reservation_v1(
        repository_root,
        source_manifest=manifest_record,
        independent_source_review=review_record,
    )
    if binding_content_sha256 is not None and not _is_sha256(binding_content_sha256):
        raise LabelContractError("failure binding content SHA-256 is malformed")
    if schedule_prefix_sha256 is not None and not _is_sha256(schedule_prefix_sha256):
        raise LabelContractError("failure schedule-prefix SHA-256 is malformed")
    failure = with_content_sha256(
        {
            "schema": LABEL_FAILURE_SCHEMA,
            "status": "FAILED_TERMINAL_NO_RETRY",
            "phase": phase,
            "error": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "preregistration": _preregistration_binding_v1(),
            "reservation_content_sha256": reservation["content_sha256"],
            "source_manifest": manifest_record,
            "independent_source_review": review_record,
            "execution_binding_content_sha256": binding_content_sha256,
            "schedule_prefix_sha256": schedule_prefix_sha256,
            "access_ledger": _normalize_access_ledger(access_ledger),
            "terminal": {
                "retry_authorized": False,
                "resume_authorized": False,
                "second_invocation_authorized": False,
                "same_root_replacement_authorized": False,
            },
            "downstream_authority": {
                "training_authorized": False,
                "g2_authorized": False,
                "navigation_authorized": False,
                "heldout_authorized": False,
                "sealed_authorized": False,
                "production_authorized": False,
                "promotion_authorized": False,
            },
        }
    )
    output = _label_output_path(repository_root=repository_root)
    if (output / "manifest.json").exists():
        raise PermissionError("cannot append failure to a completed label attempt")
    _canonical_write_exclusive(output / "failure.json", failure)
    return failure


def _exact_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LabelContractError(f"{name} must be an exact integer")
    if minimum is not None and value < minimum:
        raise LabelContractError(f"{name} must be at least {minimum}")
    return value


def _exact_str(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise LabelContractError(f"{name} must be a nonempty string")
    return value


def _finite(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelContractError(f"{name} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise LabelContractError(f"{name} must be finite")
    return result


def _validate_content_hash(value: Mapping[str, Any], *, name: str) -> None:
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not _is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise LabelContractError(f"{name} content hash changed")


def _json_loads_strict(raw: bytes, *, name: str) -> Any:
    def reject_constant(value: str) -> None:
        raise LabelContractError(f"{name} contains non-finite {value}")

    def reject_duplicates(pairs: Sequence[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise LabelContractError(f"{name} repeats key {key!r}")
            result[key] = value
        return result

    try:
        return json.loads(
            raw,
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise LabelContractError(f"{name} is invalid JSON") from error


def _read_bound_file(
    path: Path,
    *,
    expected_sha256: str,
    access_ledger: MutableMapping[str, int] | None = None,
    access_key: str | None = None,
) -> bytes:
    if not _is_sha256(expected_sha256):
        raise LabelContractError("expected file SHA-256 is malformed")
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise LabelContractError(f"bound source is not a regular non-symlink: {path}")
    if access_key is not None:
        _record_access(access_ledger, access_key)
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != expected_sha256:
        raise LabelContractError(
            f"file hash mismatch for {path}: expected {expected_sha256}, got {actual}"
        )
    return raw


def _parse_canonical_object(raw: bytes, *, name: str) -> dict[str, Any]:
    value = _json_loads_strict(raw, name=name)
    if not isinstance(value, dict) or raw != canonical_json_bytes(value) + b"\n":
        raise LabelContractError(f"{name} must be canonical one-line JSON")
    _validate_content_hash(value, name=name)
    return value


def _parse_canonical_jsonl(raw: bytes, *, name: str) -> tuple[dict[str, Any], ...]:
    if not raw or not raw.endswith(b"\n"):
        raise LabelContractError(f"{name} must be nonempty terminal-newline JSONL")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        value = _json_loads_strict(line, name=f"{name}:{line_number}")
        if not isinstance(value, dict) or line != canonical_json_bytes(value):
            raise LabelContractError(f"{name}:{line_number} is not canonical JSON")
        _validate_content_hash(value, name=f"{name}:{line_number}")
        rows.append(value)
    return tuple(rows)


def validate_raw_audit_v1(
    path: Path,
    *,
    access_ledger: MutableMapping[str, int] | None = None,
) -> Mapping[str, Any]:
    report = _parse_canonical_object(
        _read_bound_file(
            path,
            expected_sha256=RAW_AUDIT_FILE_SHA256,
            access_ledger=access_ledger,
            access_key="raw_audit_opens",
        ),
        name="raw audit V13",
    )
    if (
        report.get("schema") != RAW_AUDIT_SCHEMA
        or report.get("verdict") != "PASS"
        or report.get("content_sha256") != RAW_AUDIT_CONTENT_SHA256
        or report.get("dataset_manifest_file_sha256") != RAW_MANIFEST_FILE_SHA256
        or report.get("dataset_manifest_content_sha256")
        != RAW_MANIFEST_CONTENT_SHA256
        or report.get("pair_count") != 5_172
        or report.get("unique_endpoint_count") != 9_460
        or report.get("scene_shard_count") != 88
    ):
        raise LabelContractError("raw audit V13 identity or PASS population changed")
    return report


def load_execution_binding_file_v1(
    path: Path,
    *,
    repository_root: Path | None = None,
    access_ledger: MutableMapping[str, int] | None = None,
) -> Mapping[str, Any]:
    path = Path(path)
    if repository_root is not None:
        expected = Path(repository_root).absolute() / LABEL_EXECUTION_BINDING_RELATIVE_PATH
        if path.absolute() != expected:
            raise PermissionError("label execution binding path changed")
    if path.is_symlink() or not path.is_file():
        raise LabelContractError("execution binding is not a regular non-symlink")
    _record_access(access_ledger, "execution_binding_opens")
    return _parse_canonical_object(path.read_bytes(), name="label execution binding")


def load_schedule_indices_v1(
    path: Path,
    *,
    raw_indexes: RawIndexesV1,
    access_ledger: MutableMapping[str, int] | None = None,
) -> tuple[int, ...]:
    schedule = _parse_canonical_object(
        _read_bound_file(
            path,
            expected_sha256=SCHEDULE_FILE_SHA256,
            access_ledger=access_ledger,
            access_key="schedule_opens",
        ),
        name="frozen presentation schedule",
    )
    return _validate_matched_training_v4_schedule_v1(
        schedule,
        raw_indexes=raw_indexes,
    )


def _validate_matched_training_v4_schedule_v1(
    schedule: Mapping[str, Any],
    *,
    raw_indexes: RawIndexesV1,
) -> tuple[int, ...]:
    if type(schedule) is not dict or set(schedule) != set(
        MATCHED_TRAINING_V4_SCHEDULE_FIELDS
    ):
        raise LabelContractError("frozen presentation schedule fields changed")
    _validate_content_hash(schedule, name="frozen presentation schedule")
    if (
        schedule.get("schema") != MATCHED_TRAINING_V4_SCHEDULE_SCHEMA
        or schedule.get("content_sha256") != SCHEDULE_CONTENT_SHA256
    ):
        raise LabelContractError("frozen presentation schedule identity changed")
    for name, expected in MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS.items():
        observed = _exact_int(schedule.get(name), name=f"schedule.{name}", minimum=0)
        if observed != expected:
            raise LabelContractError("frozen presentation schedule dimensions changed")
    if {
        name: schedule.get(name)
        for name in MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256
    } != MATCHED_TRAINING_V4_SCHEDULE_IDENTITY_SHA256:
        raise LabelContractError("frozen presentation schedule hashes changed")

    train_pair_ids = [
        str(pair["content_sha256"])
        for pair in raw_indexes.pairs
        if pair["dataset_role"] == "train"
    ]
    ordered_pair_ids_sha256 = canonical_json_sha256(train_pair_ids)
    if (
        len(train_pair_ids)
        != MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS["train_pair_count"]
        or ordered_pair_ids_sha256 != schedule["ordered_pair_ids_sha256"]
    ):
        raise LabelContractError("frozen schedule train-pair identity changed")

    indices = schedule.get("presentation_indices")
    if type(indices) is not list or len(indices) != MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS[
        "presentation_count"
    ]:
        raise LabelContractError("frozen presentation schedule indices changed")
    normalized_indices = tuple(
        _exact_int(value, name=f"schedule index {position}", minimum=0)
        for position, value in enumerate(indices)
    )
    if any(
        index >= MATCHED_TRAINING_V4_SCHEDULE_DIMENSIONS["train_pair_count"]
        for index in normalized_indices
    ):
        raise LabelContractError("frozen presentation schedule escaped the train role")
    if (
        canonical_json_sha256(list(normalized_indices))
        != schedule["indices_sha256"]
    ):
        raise LabelContractError("frozen presentation schedule index hash changed")
    prefix, _ = _schedule_prefix_identity_v1(
        normalized_indices,
        require_frozen=True,
    )
    return prefix


def _schedule_prefix_identity_v1(
    presentation_indices: Sequence[object],
    *,
    require_frozen: bool,
) -> tuple[tuple[int, ...], str]:
    if len(presentation_indices) < 16_000:
        raise LabelContractError("schedule has fewer than 16,000 presentations")
    prefix = tuple(
        _exact_int(value, name=f"schedule index {position}", minimum=0)
        for position, value in enumerate(presentation_indices[:16_000])
    )
    digest = canonical_json_sha256(list(prefix))
    if require_frozen and digest != SCHEDULE_PREFIX_SHA256:
        raise LabelContractError("frozen first-16,000 schedule prefix changed")
    return prefix, digest


def _validate_raw_manifest_endpoint_order_v1(value: object) -> None:
    if value != RAW_ENDPOINT_INDEX_ORDER_SHA256:
        raise LabelContractError("raw manifest identity or population changed")


def _validate_raw_endpoint_content_order_v1(
    endpoints: Sequence[Mapping[str, Any]],
) -> None:
    digest = canonical_json_sha256([row["content_sha256"] for row in endpoints])
    if digest != RAW_ENDPOINT_INDEX_ORDER_SHA256:
        raise LabelContractError("raw endpoint ordering changed")


def load_and_validate_raw_indexes(
    manifest_path: Path,
    pairs_path: Path,
    endpoints_path: Path,
    *,
    access_ledger: MutableMapping[str, int] | None = None,
) -> RawIndexesV1:
    """Load the three exact raw index leaves; no shard array or RGB is opened."""

    manifest_path = Path(manifest_path)
    pairs_path = Path(pairs_path)
    endpoints_path = Path(endpoints_path)
    manifest = _parse_canonical_object(
        _read_bound_file(
            manifest_path,
            expected_sha256=RAW_MANIFEST_FILE_SHA256,
            access_ledger=access_ledger,
            access_key="raw_manifest_opens",
        ),
        name="raw manifest",
    )
    if (
        manifest.get("schema") != RAW_DATASET_SCHEMA
        or manifest.get("status") != "complete_pending_independent_audit"
        or manifest.get("content_sha256") != RAW_MANIFEST_CONTENT_SHA256
        or manifest.get("roles") != list(RAW_ROLE_ORDER)
        or manifest.get("pair_counts")
        != {role: ROLE_STATE_COUNTS[role] for role in RAW_ROLE_ORDER}
        or manifest.get("scene_shard_count") != 88
        or manifest.get("ordered_pair_sha256") != RAW_ORDERED_PAIR_SHA256
    ):
        raise LabelContractError("raw manifest identity or population changed")
    _validate_raw_manifest_endpoint_order_v1(manifest.get("ordered_endpoint_sha256"))

    pair_record = manifest.get("pair_index")
    endpoint_record = manifest.get("endpoint_index")
    if not isinstance(pair_record, Mapping) or not isinstance(endpoint_record, Mapping):
        raise LabelContractError("raw manifest index bindings are absent")
    expected_pair_path = manifest_path.parent / "pairs.jsonl"
    expected_endpoint_path = manifest_path.parent / "endpoints.jsonl"
    if (
        pair_record != {
            "path": "pairs.jsonl",
            "row_count": 5_172,
            "file_sha256": RAW_PAIRS_FILE_SHA256,
        }
        or endpoint_record != {
            "path": "endpoints.jsonl",
            "row_count": 9_460,
            "file_sha256": RAW_ENDPOINTS_FILE_SHA256,
        }
        or pairs_path.absolute() != expected_pair_path.absolute()
        or endpoints_path.absolute() != expected_endpoint_path.absolute()
    ):
        raise LabelContractError("raw pair/endpoint path binding changed")

    pairs = _parse_canonical_jsonl(
        _read_bound_file(
            pairs_path,
            expected_sha256=RAW_PAIRS_FILE_SHA256,
            access_ledger=access_ledger,
            access_key="raw_pairs_opens",
        ),
        name="raw pairs",
    )
    endpoints = _parse_canonical_jsonl(
        _read_bound_file(
            endpoints_path,
            expected_sha256=RAW_ENDPOINTS_FILE_SHA256,
            access_ledger=access_ledger,
            access_key="raw_endpoints_opens",
        ),
        name="raw endpoints",
    )
    if len(pairs) != 5_172 or len(endpoints) != 9_460:
        raise LabelContractError("raw pair/endpoint row counts changed")
    if canonical_json_sha256([row["content_sha256"] for row in pairs]) != RAW_ORDERED_PAIR_SHA256:
        raise LabelContractError("raw pair ordering changed")
    _validate_raw_endpoint_content_order_v1(endpoints)

    shards = manifest.get("shards")
    if not isinstance(shards, list) or len(shards) != 88:
        raise LabelContractError("raw scene-shard inventory changed")
    shard_by_scene: dict[str, Mapping[str, Any]] = {}
    for shard in shards:
        if not isinstance(shard, Mapping):
            raise LabelContractError("raw scene-shard record is malformed")
        scene = _exact_str(shard.get("scene_id"), name="shard.scene_id")
        role = _exact_str(shard.get("dataset_role"), name="shard.dataset_role")
        family = _exact_str(shard.get("family"), name="shard.family")
        if role not in RAW_ROLE_ORDER or scene in shard_by_scene or not family:
            raise LabelContractError("raw scene-shard identity is invalid or repeated")
        shard_by_scene[scene] = shard

    endpoint_by_sha256: dict[str, Mapping[str, Any]] = {}
    previous_endpoint_key: tuple[int, str] | None = None
    for endpoint in endpoints:
        if endpoint.get("schema") != RAW_ENDPOINT_SCHEMA:
            raise LabelContractError("raw endpoint schema changed")
        digest = endpoint.get("endpoint_identity_sha256")
        if not _is_sha256(digest) or digest in endpoint_by_sha256:
            raise LabelContractError("raw endpoint identity is malformed or repeated")
        role = endpoint.get("dataset_role")
        scene = endpoint.get("scene_id")
        family = endpoint.get("family")
        if role not in RAW_ROLE_ORDER or not isinstance(scene, str) or not isinstance(family, str):
            raise LabelContractError("raw endpoint role/scene/family is malformed")
        shard = shard_by_scene.get(scene)
        if shard is None or (shard.get("dataset_role"), shard.get("family")) != (role, family):
            raise LabelContractError("raw endpoint crossed its scene shard")
        if not _is_sha256(endpoint.get("image_sha256_commitment_only")):
            raise LabelContractError("raw endpoint image commitment is malformed")
        parse_rendered_filename(str(endpoint.get("image_path_metadata_only", "")))
        key = (RAW_ROLE_ORDER.index(role), str(digest))
        if previous_endpoint_key is not None and key <= previous_endpoint_key:
            raise LabelContractError("raw endpoint order changed")
        previous_endpoint_key = key
        endpoint_by_sha256[str(digest)] = endpoint

    pair_counts: Counter[str] = Counter()
    scenes_by_role: dict[str, set[str]] = defaultdict(set)
    seen_global_rows: set[int] = set()
    seen_current: set[str] = set()
    previous_pair_key: tuple[int, int] | None = None
    for pair in pairs:
        if pair.get("schema") != RAW_PAIR_SCHEMA:
            raise LabelContractError("raw pair schema changed")
        role = pair.get("dataset_role")
        action = pair.get("primitive")
        global_row = _exact_int(pair.get("global_row"), name="pair.global_row", minimum=0)
        scene = _exact_str(pair.get("scene_id"), name="pair.scene_id")
        family = _exact_str(pair.get("family"), name="pair.family")
        if role not in RAW_ROLE_ORDER or action not in ACTION_ORDER:
            raise LabelContractError("raw pair role or primitive changed")
        key = (RAW_ROLE_ORDER.index(str(role)), global_row)
        if previous_pair_key is not None and key <= previous_pair_key:
            raise LabelContractError("raw pair order changed")
        previous_pair_key = key
        if global_row in seen_global_rows:
            raise LabelContractError("raw global row repeated")
        seen_global_rows.add(global_row)
        shard = shard_by_scene.get(scene)
        if shard is None or (shard.get("dataset_role"), shard.get("family")) != (role, family):
            raise LabelContractError("raw pair crossed its scene shard")
        for side in ("current", "next"):
            digest = pair.get(f"{side}_endpoint_sha256")
            endpoint = endpoint_by_sha256.get(str(digest))
            if endpoint is None or (
                endpoint.get("dataset_role"),
                endpoint.get("scene_id"),
                endpoint.get("family"),
            ) != (role, scene, family):
                raise LabelContractError("raw pair crossed its endpoint binding")
        current = str(pair["current_endpoint_sha256"])
        if current in seen_current:
            raise LabelContractError("raw current endpoint is not state-unique")
        seen_current.add(current)
        if not _is_sha256(pair.get("frames_jsonl_sha256")) or not _is_sha256(
            pair.get("scene_manifest_sha256")
        ):
            raise LabelContractError("raw pair source hash is malformed")
        pair_counts[str(role)] += 1
        scenes_by_role[str(role)].add(scene)
    if dict(pair_counts) != {role: ROLE_STATE_COUNTS[role] for role in RAW_ROLE_ORDER}:
        raise LabelContractError("raw pair role counts changed")
    if {role: len(scenes_by_role[role]) for role in RAW_ROLE_ORDER} != {
        role: ROLE_SCENE_COUNTS[role] for role in RAW_ROLE_ORDER
    }:
        raise LabelContractError("raw pair scene counts changed")
    return RawIndexesV1(
        manifest=manifest,
        pairs=pairs,
        endpoints=endpoints,
        endpoint_by_sha256=endpoint_by_sha256,
        shard_by_scene=shard_by_scene,
    )


def parse_rendered_filename(path: str) -> tuple[int, int]:
    candidate = Path(path)
    match = _FRAME_NAME.fullmatch(candidate.name)
    if match is None or candidate.parent.name != "rgb":
        raise LabelContractError("endpoint does not use the canonical rendered filename")
    return int(match.group(1)), int(match.group(2))


def load_geometry_inputs_v1(
    *,
    repository_root: Path,
    geometry_path: Path,
    directional_policy_path: Path,
    primitive_registry_path: Path,
    access_ledger: MutableMapping[str, int] | None = None,
) -> GeometryInputsV1:
    """Hash-check and reconstruct only the frozen clean geometry inputs."""

    geometry_raw = _read_bound_file(
        geometry_path,
        expected_sha256=GEOMETRY_CONTRACT_FILE_SHA256,
        access_ledger=access_ledger,
        access_key="geometry_contract_opens",
    )
    geometry_payload = _json_loads_strict(geometry_raw, name="geometry contract")
    if not isinstance(geometry_payload, dict) or canonical_json_sha256(geometry_payload) != GEOMETRY_CONTRACT_CONTENT_SHA256:
        raise LabelContractError("geometry contract content hash changed")
    _record_access(access_ledger, "geometry_contract_validation_calls")
    geometry = load_geometry_contract(
        Path(geometry_path), repository_root=Path(repository_root), verify_sources=True
    )
    if geometry.sha256 != GEOMETRY_CONTRACT_CONTENT_SHA256:
        raise LabelContractError("parsed geometry contract content hash changed")

    policy_raw = _read_bound_file(
        directional_policy_path,
        expected_sha256=DIRECTIONAL_POLICY_FILE_SHA256,
        access_ledger=access_ledger,
        access_key="directional_policy_opens",
    )
    policy = _json_loads_strict(policy_raw, name="directional policy")
    if not isinstance(policy, dict):
        raise LabelContractError("directional policy must be an object")
    policy_core = dict(policy)
    policy_declared = policy_core.pop("content_sha256", None)
    if (
        policy.get("schema") != "lewm_go2_directional_footprint_policy_v1"
        or policy.get("policy_id") != DIRECTIONAL_POLICY_ID
        or policy.get("recommended_profile") != DIRECTIONAL_PROFILE
        or policy_declared != DIRECTIONAL_POLICY_CONTENT_SHA256
        or canonical_json_sha256(policy_core) != DIRECTIONAL_POLICY_CONTENT_SHA256
        or geometry.swept_footprint.directional_policy_id != DIRECTIONAL_POLICY_ID
        or geometry.swept_footprint.directional_policy_content_sha256
        != DIRECTIONAL_POLICY_CONTENT_SHA256
        or geometry.swept_footprint.directional_profile != DIRECTIONAL_PROFILE
    ):
        raise LabelContractError("directional policy identity changed")
    profiles = policy.get("profiles")
    profile = profiles.get(DIRECTIONAL_PROFILE) if isinstance(profiles, Mapping) else None
    if not isinstance(profile, Mapping) or not isinstance(profile.get("support_planes"), list):
        raise LabelContractError("directional policy profile is malformed")
    footprint = DirectionalSupportFootprint.from_directional_support(
        {
            _finite(plane.get("angle_deg"), name="support angle"): _finite(
                plane.get("raw_support_m"), name="raw directional support"
            )
            for plane in profile["support_planes"]
            if isinstance(plane, Mapping)
        },
        margin_m=_finite(profile.get("margin_m"), name="directional margin"),
    )
    artifact_vertices = tuple(
        (float(vertex[0]), float(vertex[1])) for vertex in profile.get("vertices_xy_body_m", ())
    )
    if len(artifact_vertices) != len(footprint.vertices_xy_m) or any(
        math.dist(actual, expected) > 1e-10
        for actual, expected in zip(footprint.vertices_xy_m, artifact_vertices, strict=True)
    ):
        raise LabelContractError("directional support reconstruction changed")

    registry_raw = _read_bound_file(
        primitive_registry_path,
        expected_sha256=PRIMITIVE_REGISTRY_FILE_SHA256,
        access_ledger=access_ledger,
        access_key="primitive_registry_opens",
    )
    registry = yaml.safe_load(registry_raw)
    if not isinstance(registry, Mapping):
        raise LabelContractError("primitive registry must be an object")
    if (
        int(registry.get("command_dim", -1)) != 3
        or registry.get("command_order")
        != ["vx_body_mps", "vy_body_mps", "yaw_rate_radps"]
        or _exact_int(registry.get("block_size"), name="registry.block_size")
        != COMMAND_COUNT
        or not math.isclose(
            _finite(registry.get("command_dt_s"), name="registry.command_dt_s"),
            COMMAND_DT_S,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise LabelContractError("primitive registry command schema changed")
    primitives = registry.get("primitives")
    if not isinstance(primitives, Mapping):
        raise LabelContractError("primitive registry lacks primitives")
    commands_by_action: dict[str, tuple[tuple[float, float, float], ...]] = {}
    for action in ACTION_ORDER:
        primitive = primitives.get(action)
        command = primitive.get("command") if isinstance(primitive, Mapping) else None
        if not isinstance(primitive, Mapping) or primitive.get("type") != "velocity_block" or not isinstance(command, Mapping):
            raise LabelContractError(f"primitive {action!r} is not a velocity block")
        row = (
            _finite(command.get("vx_body_mps"), name=f"{action}.vx"),
            _finite(command.get("vy_body_mps"), name=f"{action}.vy"),
            _finite(command.get("yaw_rate_radps"), name=f"{action}.yaw_rate"),
        )
        commands_by_action[action] = tuple(row for _ in range(COMMAND_COUNT))
    return GeometryInputsV1(
        geometry=geometry,
        footprint=footprint,
        commands_by_action=commands_by_action,
        source_bindings={
            "geometry_contract": {
                "path": str(Path(geometry_path)),
                "file_sha256": GEOMETRY_CONTRACT_FILE_SHA256,
                "content_sha256": GEOMETRY_CONTRACT_CONTENT_SHA256,
            },
            "directional_policy": {
                "path": str(Path(directional_policy_path)),
                "file_sha256": DIRECTIONAL_POLICY_FILE_SHA256,
                "content_sha256": DIRECTIONAL_POLICY_CONTENT_SHA256,
            },
            "primitive_registry": {
                "path": str(Path(primitive_registry_path)),
                "file_sha256": PRIMITIVE_REGISTRY_FILE_SHA256,
            },
        },
    )


def integrate_action_v1(
    commands: Sequence[Sequence[float]],
    *,
    start: Pose2D = Pose2D(0.0, 0.0, 0.0),
) -> tuple[Pose2D, ...]:
    """Integrate the exact five-command block, returning start plus five poses."""

    if len(commands) != COMMAND_COUNT:
        raise LabelContractError("an action must contain exactly five commands")
    poses = [start]
    for command_index, command in enumerate(commands):
        if len(command) != 3:
            raise LabelContractError(f"command {command_index} must contain vx/vy/yaw-rate")
        vx, vy, yaw_rate = (
            _finite(value, name=f"command[{command_index}]") for value in command
        )
        previous = poses[-1]
        cos_yaw = math.cos(previous.yaw_rad)
        sin_yaw = math.sin(previous.yaw_rad)
        poses.append(
            Pose2D(
                previous.x_m + (vx * cos_yaw - vy * sin_yaw) * COMMAND_DT_S,
                previous.y_m + (vx * sin_yaw + vy * cos_yaw) * COMMAND_DT_S,
                wrap_angle_pi(previous.yaw_rad + yaw_rate * COMMAND_DT_S),
            )
        )
    return tuple(poses)


def transform_pose_v1(origin: Pose2D, local: Pose2D) -> Pose2D:
    cos_yaw = math.cos(origin.yaw_rad)
    sin_yaw = math.sin(origin.yaw_rad)
    return Pose2D(
        origin.x_m + cos_yaw * local.x_m - sin_yaw * local.y_m,
        origin.y_m + sin_yaw * local.x_m + cos_yaw * local.y_m,
        wrap_angle_pi(origin.yaw_rad + local.yaw_rad),
    )


def remote_corridor_pose_samples_v1() -> tuple[tuple[Pose2D, ...], ...]:
    groups: list[tuple[Pose2D, ...]] = [(Pose2D(290 / 200, 0.0, 0.0),)]
    for station in range(1, 11):
        groups.append(
            tuple(
                Pose2D((290 + 40 * (station - 1) + 5 * sample) / 200, 0.0, 0.0)
                for sample in range(9)
            )
        )
    poses = np.asarray(
        [(pose.x_m, pose.y_m, pose.yaw_rad) for group in groups for pose in group],
        dtype="<f8",
        order="C",
    )
    offsets = np.asarray(REMOTE_SAMPLE_OFFSETS, dtype="<i8", order="C")
    digest = hashlib.sha256(offsets.tobytes(order="C") + poses.tobytes(order="C")).hexdigest()
    if poses.shape != (91, 3) or digest != REMOTE_SAMPLE_SHA256:
        raise AssertionError("frozen remote-corridor sampler changed")
    return tuple(groups)


def rasterize_corridor_masks_v1(
    footprint: DirectionalSupportFootprint,
    pose_groups: Sequence[Sequence[Pose2D]],
) -> np.ndarray:
    """Rasterize unions of closed polygon/cell SAT intersections in row-major order."""

    if len(pose_groups) != 11 or any(not group for group in pose_groups):
        raise LabelContractError("corridor masks require eleven nonempty pose groups")
    result = np.zeros((11, *GRID_SHAPE), dtype=np.uint8, order="C")
    half = 0.5 * GRID_CELL_SIZE_M
    cells: dict[tuple[int, int], OrientedRectangle] = {}
    for station, group in enumerate(pose_groups):
        for pose in group:
            vertices = footprint.vertices_at(pose)
            min_x = min(vertex[0] for vertex in vertices)
            max_x = max(vertex[0] for vertex in vertices)
            min_y = min(vertex[1] for vertex in vertices)
            max_y = max(vertex[1] for vertex in vertices)
            row_first = max(0, math.floor((min_x - GRID_FORWARD_MIN_EDGE_M) / GRID_CELL_SIZE_M) - 1)
            row_last = min(GRID_SHAPE[0] - 1, math.floor((max_x - GRID_FORWARD_MIN_EDGE_M) / GRID_CELL_SIZE_M) + 1)
            column_first = max(0, math.floor((min_y - GRID_LEFT_MIN_EDGE_M) / GRID_CELL_SIZE_M) - 1)
            column_last = min(GRID_SHAPE[1] - 1, math.floor((max_y - GRID_LEFT_MIN_EDGE_M) / GRID_CELL_SIZE_M) + 1)
            for row in range(row_first, row_last + 1):
                for column in range(column_first, column_last + 1):
                    key = (row, column)
                    cell = cells.get(key)
                    if cell is None:
                        cell = OrientedRectangle(
                            center_xy_m=(
                                GRID_FORWARD_MIN_EDGE_M + (row + 0.5) * GRID_CELL_SIZE_M,
                                GRID_LEFT_MIN_EDGE_M + (column + 0.5) * GRID_CELL_SIZE_M,
                            ),
                            half_extent_x_m=half,
                            half_extent_y_m=half,
                            yaw_rad=0.0,
                        )
                        cells[key] = cell
                    if convex_polygon_intersects_rectangle(vertices, cell):
                        result[station, row, column] = 1
    return np.ascontiguousarray(result, dtype=np.uint8)


def predicted_next_corridor_masks_v1(
    footprint: DirectionalSupportFootprint,
) -> np.ndarray:
    masks = rasterize_corridor_masks_v1(footprint, remote_corridor_pose_samples_v1())
    counts = tuple(int(value) for value in masks.sum(axis=(1, 2)))
    digest = hashlib.sha256(masks.tobytes(order="C")).hexdigest()
    if counts != PREDICTED_NEXT_MASK_COUNTS or digest != PREDICTED_NEXT_MASK_SHA256:
        raise LabelContractError("predicted-next corridor mask identity changed")
    return masks


def persistence_corridor_masks_v1(
    footprint: DirectionalSupportFootprint,
    commands_by_action: Mapping[str, Sequence[Sequence[float]]],
) -> np.ndarray:
    remote = remote_corridor_pose_samples_v1()
    action_masks: list[np.ndarray] = []
    for action in ACTION_ORDER:
        post = integrate_action_v1(commands_by_action[action])[-1]
        groups = tuple(
            tuple(transform_pose_v1(post, pose) for pose in group) for group in remote
        )
        masks = rasterize_corridor_masks_v1(footprint, groups)
        digest = hashlib.sha256(masks.tobytes(order="C")).hexdigest()
        if digest != PERSISTENCE_MASK_SHA256[action]:
            raise LabelContractError(f"persistence mask identity changed for {action}")
        action_masks.append(masks)
    result = np.ascontiguousarray(np.stack(action_masks, axis=0), dtype=np.uint8)
    if hashlib.sha256(result.tobytes(order="C")).hexdigest() != PERSISTENCE_STACK_SHA256:
        raise LabelContractError("persistence mask-stack identity changed")
    return result


def projective_support_mask_v1() -> np.ndarray:
    """Return the frozen model's fixed level-camera learned-support mask."""

    forward = np.linspace(-0.95, 5.35, 64, dtype=np.float64)
    left = np.linspace(-3.15, 3.15, 64, dtype=np.float64)
    forward_grid, left_grid = np.meshgrid(forward, left, indexing="ij")
    camera_forward = forward_grid - 0.326
    camera_right = -left_grid
    camera_up = np.full(GRID_SHAPE, -0.333 - 0.043, dtype=np.float64)
    grid_x = camera_right / np.maximum(camera_forward, np.finfo(np.float64).eps)
    grid_x /= math.tan(math.radians(78.323) / 2.0)
    grid_y = -camera_up / np.maximum(camera_forward, np.finfo(np.float64).eps)
    grid_y /= math.tan(math.radians(62.8370386364) / 2.0)
    result = (
        (camera_forward > 0.05)
        & (grid_x >= -1.0)
        & (grid_x <= 1.0)
        & (grid_y >= -1.0)
        & (grid_y <= 1.0)
    )
    if int(result.sum()) != PROJECTIVE_SUPPORT_CELL_COUNT:
        raise AssertionError("frozen projective-support population changed")
    return np.ascontiguousarray(result, dtype=np.uint8)


def _pose_values(pose: Pose2D) -> list[float]:
    return [float(pose.x_m), float(pose.y_m), float(pose.yaw_rad)]


def _sampled_immediate_poses_v1(
    checker: ManifestDirectionalFootprintFeasibility,
    source_pose_world: Pose2D,
    commands: Sequence[Sequence[float]],
) -> tuple[Pose2D, ...]:
    local = integrate_action_v1(commands)
    world = tuple(transform_pose_v1(source_pose_world, pose) for pose in local)
    samples: list[Pose2D] = []
    for segment, (start, end) in enumerate(
        zip(world[:-1], world[1:], strict=True)
    ):
        swept = checker.interpolated_sweep(
            start,
            end,
            maximum_corner_step_m=MAXIMUM_CORNER_STEP_M,
            maximum_yaw_step_rad=MAXIMUM_YAW_STEP_RAD,
        )
        samples.extend(pose for _, pose in swept[(1 if segment else 0) :])
    return tuple(samples)


def _feasibility_summary_v1(
    checker: ManifestDirectionalFootprintFeasibility,
    poses: Sequence[Pose2D],
) -> dict[str, Any]:
    if not poses:
        raise LabelContractError("a feasibility query must contain at least one pose")
    reports = tuple(checker.pose_feasibility(pose) for pose in poses)
    colliding = sorted(
        {
            object_id
            for report in reports
            for object_id in report.colliding_object_ids
        }
    )
    inside = all(report.inside_world_bounds for report in reports)
    return {
        "feasible": inside and not colliding,
        "inside_world_bounds": inside,
        "colliding_object_ids": colliding,
        "sample_count": len(reports),
    }


def _remote_prefix(station_safe: Sequence[bool]) -> int:
    prefix = 0
    for safe in station_safe:
        if not safe:
            break
        prefix += 1
    return prefix


def label_state_v1(
    *,
    pair: Mapping[str, Any],
    endpoint: Mapping[str, Any],
    source_pose_world: Pose2D,
    source_line_number: int,
    scene_manifest: SceneManifest,
    footprint: DirectionalSupportFootprint,
    commands_by_action: Mapping[str, Sequence[Sequence[float]]],
    source_bindings: Mapping[str, Mapping[str, Any]],
    role_state_index: int,
) -> tuple[dict[str, Any], ...]:
    """Create all nine flat action rows for one current RGB state."""

    role = _exact_str(pair.get("dataset_role"), name="pair.dataset_role")
    if role not in OUTPUT_ROLE_ORDER:
        raise LabelContractError("label state escaped development roles")
    scene_id = _exact_str(pair.get("scene_id"), name="pair.scene_id")
    family = _exact_str(pair.get("family"), name="pair.family")
    if scene_manifest.scene_id != scene_id or scene_manifest.family != family:
        raise LabelContractError("label state crossed its parsed scene manifest")
    if endpoint.get("endpoint_identity_sha256") != pair.get("current_endpoint_sha256"):
        raise LabelContractError("label state crossed its current endpoint")
    checker = ManifestDirectionalFootprintFeasibility(scene_manifest, footprint)
    remote_local = remote_corridor_pose_samples_v1()

    candidates: list[dict[str, Any]] = []
    for action_index, action in enumerate(ACTION_ORDER):
        commands = commands_by_action.get(action)
        if commands is None:
            raise LabelContractError(f"missing command block for {action}")
        integrated_local = integrate_action_v1(commands)
        post_world = transform_pose_v1(source_pose_world, integrated_local[-1])
        immediate = _feasibility_summary_v1(
            checker,
            _sampled_immediate_poses_v1(
                checker, source_pose_world, commands
            ),
        )
        blind_end = Pose2D(
            post_world.x_m + 1.45 * math.cos(post_world.yaw_rad),
            post_world.y_m + 1.45 * math.sin(post_world.yaw_rad),
            post_world.yaw_rad,
        )
        blind_samples = checker.interpolated_sweep(
            post_world,
            blind_end,
            maximum_corner_step_m=MAXIMUM_CORNER_STEP_M,
            maximum_yaw_step_rad=MAXIMUM_YAW_STEP_RAD,
        )
        blind = _feasibility_summary_v1(
            checker, tuple(pose for _, pose in blind_samples)
        )
        station_reports = tuple(
            _feasibility_summary_v1(
                checker,
                tuple(transform_pose_v1(post_world, pose) for pose in group),
            )
            for group in remote_local
        )
        station_safe = tuple(bool(report["feasible"]) for report in station_reports)
        candidates.append(
            {
                "action": action,
                "action_index": action_index,
                "post_local": integrated_local[-1],
                "immediate": immediate,
                "blind": blind,
                "station_reports": station_reports,
                "station_safe": station_safe,
                "prefix": _remote_prefix(station_safe),
            }
        )

    non_hold = tuple(candidate for candidate in candidates if candidate["action"] != "hold")
    primary = all(
        candidate["immediate"]["feasible"] and candidate["blind"]["feasible"]
        for candidate in non_hold
    )
    oracle = max(non_hold, key=lambda item: (item["prefix"], -item["action_index"]))
    distinct_prefixes = {int(candidate["prefix"]) for candidate in non_hold}
    informative = primary and int(oracle["prefix"]) > 0 and len(distinct_prefixes) >= 2
    eligible_pair_count = sum(
        int(first["prefix"] > second["prefix"])
        for first in non_hold
        for second in non_hold
    )
    participating = {
        candidate["action"]
        for candidate in non_hold
        if any(
            candidate["prefix"] != other["prefix"]
            for other in non_hold
            if other is not candidate
        )
    }
    source_line = _exact_int(source_line_number, name="source_line_number", minimum=1)
    state_index = _exact_int(role_state_index, name="role_state_index", minimum=0)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        station_reports = candidate["station_reports"]
        core = {
            "schema": LABEL_ROW_SCHEMA,
            "dataset_role": role,
            "role_state_index": state_index,
            "global_row": _exact_int(pair.get("global_row"), name="pair.global_row", minimum=0),
            "pair_content_sha256": pair.get("content_sha256"),
            "current_endpoint_sha256": pair.get("current_endpoint_sha256"),
            "scene_id": scene_id,
            "family": family,
            "action_index": candidate["action_index"],
            "action": candidate["action"],
            "nominal_post_action_se2_current_frame": _pose_values(candidate["post_local"]),
            "immediate_primitive": candidate["immediate"],
            "blind_bridge": candidate["blind"],
            "station_safe": list(candidate["station_safe"]),
            "station_inside_world_bounds": [
                bool(report["inside_world_bounds"]) for report in station_reports
            ],
            "station_colliding_object_ids": [
                list(report["colliding_object_ids"]) for report in station_reports
            ],
            "station_sample_counts": [
                int(report["sample_count"]) for report in station_reports
            ],
            "remote_safe_prefix_length": int(candidate["prefix"]),
            "primary_subset_eligible": primary,
            "informative_state": informative,
            "oracle_action": oracle["action"],
            "oracle_remote_safe_prefix_length": int(oracle["prefix"]),
            "eligible_ordered_ranking_pair_count": eligible_pair_count,
            "action_participates_in_ranking_pair": candidate["action"] in participating,
            "provenance": {
                "endpoint_index_content_sha256": endpoint.get("content_sha256"),
                "source_frame_line_number": source_line,
                "source_pose_world_xy_yaw": _pose_values(source_pose_world),
                "executed_pair_primitive": pair.get("primitive"),
                "source_bindings_sha256": canonical_json_sha256(
                    {
                        key: dict(value)
                        for key, value in sorted(source_bindings.items())
                    }
                ),
                "source_frames_jsonl_sha256": pair.get("frames_jsonl_sha256"),
                "scene_manifest_content_sha256": pair.get("scene_manifest_sha256"),
            },
        }
        if not _is_sha256(core["pair_content_sha256"]) or not _is_sha256(
            core["current_endpoint_sha256"]
        ) or not _is_sha256(
            core["provenance"]["source_frames_jsonl_sha256"]
        ) or not _is_sha256(
            core["provenance"]["scene_manifest_content_sha256"]
        ):
            raise LabelContractError("label row input hashes are malformed")
        rows.append(with_content_sha256(core))
    return tuple(rows)


def validate_label_rows_v1(
    rows: Sequence[Mapping[str, Any]],
    *,
    role: str,
    enforce_frozen_count: bool = True,
) -> tuple[Mapping[str, Any], ...]:
    if role not in OUTPUT_ROLE_ORDER:
        raise LabelContractError("unsupported label role")
    normalized = tuple(rows)
    if enforce_frozen_count and len(normalized) != ROLE_ACTION_ROW_COUNTS[role]:
        raise LabelContractError(f"{role} label action-row count changed")
    previous_key: tuple[int, int] | None = None
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in normalized:
        if row.get("schema") != LABEL_ROW_SCHEMA or row.get("dataset_role") != role:
            raise LabelContractError("label row schema or role changed")
        _validate_content_hash(row, name="label row")
        state_index = _exact_int(row.get("role_state_index"), name="role_state_index", minimum=0)
        action_index = _exact_int(row.get("action_index"), name="action_index", minimum=0)
        key = (state_index, action_index)
        if previous_key is not None and key <= previous_key:
            raise LabelContractError("label row order changed")
        previous_key = key
        if action_index >= len(ACTION_ORDER) or row.get("action") != ACTION_ORDER[action_index]:
            raise LabelContractError("label action vocabulary changed")
        station_safe = row.get("station_safe")
        if (
            not isinstance(station_safe, list)
            or len(station_safe) != 11
            or any(type(value) is not bool for value in station_safe)
            or row.get("remote_safe_prefix_length") != _remote_prefix(station_safe)
        ):
            raise LabelContractError("label station/prefix contract changed")
        grouped[state_index].append(row)
    if any(
        len(group) != len(ACTION_ORDER)
        or [row["action"] for row in group] != list(ACTION_ORDER)
        for group in grouped.values()
    ):
        raise LabelContractError("label state does not contain exactly nine actions")
    if enforce_frozen_count and sorted(grouped) != list(range(ROLE_STATE_COUNTS[role])):
        raise LabelContractError("label role-state indices changed")
    return normalized


def summarize_preflight_v1(
    rows_by_role: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    enforce_frozen_gates: bool = True,
) -> dict[str, Any]:
    """Summarize and, by default, enforce the data-only preregistered gates."""

    if set(rows_by_role) != set(OUTPUT_ROLE_ORDER):
        raise LabelContractError("preflight requires exactly three development roles")
    validated = {
        role: validate_label_rows_v1(
            rows_by_role[role], role=role, enforce_frozen_count=enforce_frozen_gates
        )
        for role in OUTPUT_ROLE_ORDER
    }
    informative_counts: dict[str, int] = {}
    scene_sets: dict[str, set[str]] = {}
    endpoint_sets: dict[str, set[str]] = {}
    scene_state_counts: dict[str, Counter[str]] = {}
    scene_endpoint_sets: dict[str, dict[str, set[str]]] = {}
    family_informative: Counter[str] = Counter()
    for role, rows in validated.items():
        first_rows = tuple(row for row in rows if row["action_index"] == 0)
        informative_counts[role] = sum(bool(row["informative_state"]) for row in first_rows)
        scene_sets[role] = {str(row["scene_id"]) for row in first_rows}
        endpoint_sets[role] = {str(row["current_endpoint_sha256"]) for row in first_rows}
        scene_state_counts[role] = Counter(str(row["scene_id"]) for row in first_rows)
        scene_endpoint_sets[role] = defaultdict(set)
        for row in first_rows:
            scene_endpoint_sets[role][str(row["scene_id"])].add(
                str(row["current_endpoint_sha256"])
            )
        if role == "checkpoint_selection":
            family_informative.update(
                str(row["family"]) for row in first_rows if row["informative_state"]
            )
    role_disjoint = all(
        scene_sets[first].isdisjoint(scene_sets[second])
        and endpoint_sets[first].isdisjoint(endpoint_sets[second])
        for first_index, first in enumerate(OUTPUT_ROLE_ORDER)
        for second in OUTPUT_ROLE_ORDER[first_index + 1 :]
    )

    train_first = tuple(
        row for row in validated["train"] if row["action_index"] == 0
    )
    train_participation = {
        action: sum(
            any(
                candidate["action"] == action
                and candidate["action_participates_in_ranking_pair"]
                for candidate in validated["train"][offset : offset + 9]
            )
            for offset in range(0, len(validated["train"]), 9)
        )
        for action in ACTION_ORDER
        if action != "hold"
    }
    support_population: dict[str, dict[str, list[dict[str, int]]]] = {}
    for population_name, roles in (
        ("train", ("train",)),
        ("calibration_plus_selection", ("probability_calibration", "checkpoint_selection")),
    ):
        population = tuple(row for role in roles for row in validated[role])
        support_population[population_name] = {
            action: [
                {
                    "safe": sum(bool(row["station_safe"][station]) for row in population if row["action"] == action),
                    "unsafe": sum(not bool(row["station_safe"][station]) for row in population if row["action"] == action),
                }
                for station in range(11)
            ]
            for action in ACTION_ORDER
            if action != "hold"
        }
    all_safe_unsafe = all(
        counts["safe"] > 0 and counts["unsafe"] > 0
        for population in support_population.values()
        for action in population.values()
        for counts in action
    )
    checks = {
        "exact_state_count": sum(len(rows) // 9 for rows in validated.values()),
        "exact_action_row_count": sum(len(rows) for rows in validated.values()),
        "exact_station_label_count": sum(len(rows) * 11 for rows in validated.values()),
        "informative_state_counts": informative_counts,
        "train_action_ranking_participation_counts": train_participation,
        "selection_family_informative_counts": dict(sorted(family_informative.items())),
        "role_scene_and_endpoint_disjoint": role_disjoint,
        "role_scene_counts": {
            role: len(counts) for role, counts in scene_state_counts.items()
        },
        "minimum_states_per_role_scene": {
            role: min(
                (len(endpoints) for endpoints in scene_endpoint_sets[role].values()),
                default=0,
            )
            for role in OUTPUT_ROLE_ORDER
        },
        "safe_unsafe_support": support_population,
        "every_non_hold_action_station_has_safe_and_unsafe_support": all_safe_unsafe,
    }
    if enforce_frozen_gates:
        _validate_registered_selection_family_counts_v1(family_informative)
    if enforce_frozen_gates and (
        checks["exact_state_count"] != 5_172
        or checks["exact_action_row_count"] != 46_548
        or checks["exact_station_label_count"] != 512_028
        or informative_counts["train"] < 512
        or informative_counts["probability_calibration"] < 128
        or informative_counts["checkpoint_selection"] < 128
        or any(count == 0 for count in train_participation.values())
        or not role_disjoint
        or {
            role: len(counts) for role, counts in scene_state_counts.items()
        }
        != ROLE_SCENE_COUNTS
        or any(
            min(
                (len(endpoints) for endpoints in scene_endpoint_sets[role].values()),
                default=0,
            )
            < 2
            for role in OUTPUT_ROLE_ORDER
        )
        or not all_safe_unsafe
    ):
        raise LabelContractError("development label preflight gates did not pass")
    return checks


def _validate_registered_selection_family_counts_v1(
    family_informative: Mapping[str, int],
) -> None:
    if set(family_informative) != set(REGISTERED_SELECTION_FAMILIES) or any(
        _exact_int(
            family_informative[family],
            name=f"selection informative count {family}",
            minimum=0,
        )
        < 8
        for family in REGISTERED_SELECTION_FAMILIES
    ):
        raise LabelContractError(
            "selection informative counts do not cover the exact registered families"
        )


def scheduled_preflight_v1(
    train_rows: Sequence[Mapping[str, Any]],
    presentation_indices: Sequence[int],
    *,
    enforce_frozen_count: bool = True,
) -> dict[str, Any]:
    """Apply the two frozen 16k schedule gates without opening RGB."""

    rows = validate_label_rows_v1(
        train_rows, role="train", enforce_frozen_count=enforce_frozen_count
    )
    prefix_tuple, prefix_sha256 = _schedule_prefix_identity_v1(
        presentation_indices,
        require_frozen=enforce_frozen_count,
    )
    prefix_indices = list(prefix_tuple)
    state_groups = tuple(tuple(rows[offset : offset + 9]) for offset in range(0, len(rows), 9))
    informative_presentations = 0
    participation = {action: 0 for action in ACTION_ORDER if action != "hold"}
    for index in prefix_indices:
        if index >= len(state_groups):
            raise LabelContractError("schedule escaped the train role")
        group = state_groups[index]
        if bool(group[0]["informative_state"]):
            informative_presentations += 1
        for row in group:
            action = str(row["action"])
            if action != "hold" and bool(row["action_participates_in_ranking_pair"]):
                participation[action] += 1
    result = {
        "presentation_count": 16_000,
        "presentation_indices_sha256": prefix_sha256,
        "informative_presentation_count": informative_presentations,
        "ranking_participation_presentations_by_action": participation,
    }
    if informative_presentations < 512 or any(value < 32 for value in participation.values()):
        raise LabelContractError("frozen schedule informative/participation gates failed")
    return result


def _normalize_source_record(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": _exact_str(record.get("path"), name="source path"),
        "byte_count": _exact_int(record.get("byte_count"), name="source byte_count", minimum=1),
        "file_sha256": _exact_str(record.get("file_sha256"), name="source file_sha256"),
        "purpose": _exact_str(record.get("purpose"), name="source purpose"),
        "scene_id": _exact_str(record.get("scene_id"), name="source scene_id"),
    }


def validate_execution_binding_envelope_v1(
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate all source-independent one-shot binding fields."""

    _validate_content_hash(value, name="label execution binding")
    if (
        value.get("schema") != EXECUTION_BINDING_SCHEMA
        or value.get("status") != "AUTHORIZED_ONE_EXACT_DEVELOPMENT_LABEL_PREFLIGHT"
        or value.get("preregistration_commit") != PREREGISTRATION_COMMIT
    ):
        raise LabelContractError("label execution binding identity changed")
    if value.get(
        "integrity_adapter_amendment"
    ) != contract.integrity_adapter_amendment_binding():
        raise LabelContractError("execution integrity_adapter_amendment changed")
    if (
        value.get("label_v1_terminal_predecessor_bindings")
        != contract.LABEL_V1_TERMINAL_PREDECESSOR_BINDINGS
    ):
        raise LabelContractError(
            "execution label_v1_terminal_predecessor_bindings changed"
        )
    if value.get(
        "schedule_schema_adapter_amendment"
    ) != contract.schedule_schema_adapter_amendment_binding():
        raise LabelContractError(
            "execution schedule_schema_adapter_amendment changed"
        )
    if (
        value.get("label_v2_terminal_predecessor_bindings")
        != contract.LABEL_V2_TERMINAL_PREDECESSOR_BINDINGS
    ):
        raise LabelContractError(
            "execution label_v2_terminal_predecessor_bindings changed"
        )
    if value.get(
        "source_episode_id_adapter_amendment"
    ) != contract.source_episode_id_adapter_amendment_binding():
        raise LabelContractError(
            "execution source_episode_id_adapter_amendment changed"
        )
    if (
        value.get("label_v3_terminal_predecessor_bindings")
        != contract.LABEL_V3_TERMINAL_PREDECESSOR_BINDINGS
    ):
        raise LabelContractError(
            "execution label_v3_terminal_predecessor_bindings changed"
        )
    authority = value.get("authority")
    if (
        not isinstance(authority, Mapping)
        or authority.get("development_label_preflight_authorized") is not True
        or any(
            flag is True
            for name, flag in authority.items()
            if name != "development_label_preflight_authorized"
            and str(name).endswith("_authorized")
        )
    ):
        raise PermissionError("execution binding grants absent or broader authority")
    inputs = value.get("inputs")
    expected_inputs = {
        "raw_manifest": (RAW_MANIFEST_FILE_SHA256, RAW_MANIFEST_CONTENT_SHA256),
        "raw_pairs": (RAW_PAIRS_FILE_SHA256, None),
        "raw_endpoints": (RAW_ENDPOINTS_FILE_SHA256, None),
        "raw_audit": (RAW_AUDIT_FILE_SHA256, RAW_AUDIT_CONTENT_SHA256),
        "schedule": (SCHEDULE_FILE_SHA256, SCHEDULE_CONTENT_SHA256),
        "geometry_contract": (
            GEOMETRY_CONTRACT_FILE_SHA256,
            GEOMETRY_CONTRACT_CONTENT_SHA256,
        ),
        "directional_policy": (
            DIRECTIONAL_POLICY_FILE_SHA256,
            DIRECTIONAL_POLICY_CONTENT_SHA256,
        ),
        "primitive_registry": (PRIMITIVE_REGISTRY_FILE_SHA256, None),
    }
    if not isinstance(inputs, Mapping) or set(inputs) != set(expected_inputs):
        raise LabelContractError("execution binding input set changed")
    for name, (file_sha256, content_sha256) in expected_inputs.items():
        record = inputs[name]
        if (
            not isinstance(record, Mapping)
            or record.get("path") != INPUT_RELATIVE_PATHS[name]
            or _exact_int(record.get("byte_count"), name=f"inputs.{name}.byte_count", minimum=1) < 1
            or record.get("file_sha256") != file_sha256
            or (
                content_sha256 is not None
                and record.get("content_sha256") != content_sha256
            )
        ):
            raise LabelContractError(f"execution input binding changed for {name}")
    if value.get("output_directory") != LABEL_OUTPUT_RELATIVE_PATH:
        raise LabelContractError("execution output_directory changed")
    source_manifest = _normalize_authority_artifact(
        value.get("source_manifest"),
        name="source_manifest",
        exact_path=SOURCE_MANIFEST_RELATIVE_PATH,
    )
    source_review = _normalize_authority_artifact(
        value.get("independent_source_review"),
        name="independent_source_review",
        exact_path=SOURCE_REVIEW_RELATIVE_PATH,
    )
    return {
        "source_manifest": source_manifest,
        "independent_source_review": source_review,
        "inputs": inputs,
    }


def validate_execution_binding_v1(
    value: Mapping[str, Any],
    *,
    raw_indexes: RawIndexesV1,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Bind exactly three named source leaves per development scene."""

    validate_execution_binding_envelope_v1(value)

    provenance = raw_indexes.manifest.get("input_provenance")
    inventory = provenance.get("source_payload_inventory") if isinstance(provenance, Mapping) else None
    if not isinstance(inventory, list):
        raise LabelContractError("raw source payload inventory is absent")
    inventory_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for raw_record in inventory:
        if not isinstance(raw_record, Mapping):
            raise LabelContractError("raw source inventory record is malformed")
        purpose = raw_record.get("purpose")
        scene = raw_record.get("scene_id")
        if purpose in (*_SOURCE_PURPOSES, "render_plan") and isinstance(scene, str):
            normalized = _normalize_source_record(raw_record)
            key = (scene, str(purpose))
            if key in inventory_by_key:
                raise LabelContractError("raw source inventory repeats a scene/purpose")
            inventory_by_key[key] = normalized

    records = value.get("source_records")
    if not isinstance(records, list):
        raise LabelContractError("execution binding source_records must be a list")
    by_scene: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    normalized_bound: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {
            "path",
            "byte_count",
            "file_sha256",
            "purpose",
            "scene_id",
            "dataset_role",
            "family",
        }:
            raise LabelContractError(f"execution source record {index} fields changed")
        normalized = _normalize_source_record(record)
        scene = normalized["scene_id"]
        purpose = normalized["purpose"]
        shard = raw_indexes.shard_by_scene.get(scene)
        if (
            purpose not in _SOURCE_PURPOSES
            or shard is None
            or record.get("dataset_role") != shard.get("dataset_role")
            or record.get("family") != shard.get("family")
            or inventory_by_key.get((scene, purpose)) != normalized
            or purpose in by_scene[scene]
        ):
            raise LabelContractError("execution source record crossed its raw binding")
        enriched = {
            **normalized,
            "dataset_role": str(record["dataset_role"]),
            "family": str(record["family"]),
        }
        by_scene[scene][purpose] = enriched
        normalized_bound.append(enriched)
    expected_keys = {
        (scene, purpose) for scene in raw_indexes.shard_by_scene for purpose in _SOURCE_PURPOSES
    }
    observed_keys = {(scene, purpose) for scene, values in by_scene.items() for purpose in values}
    if observed_keys != expected_keys or len(records) != 264:
        raise LabelContractError("execution binding is not the exact 88-scene/264-file set")
    expected_order = sorted(
        normalized_bound,
        key=lambda item: (
            OUTPUT_ROLE_ORDER.index(item["dataset_role"]),
            item["scene_id"],
            _SOURCE_PURPOSES.index(item["purpose"]),
        ),
    )
    if normalized_bound != expected_order:
        raise LabelContractError("execution source records are not in canonical role/scene/purpose order")
    return {scene: dict(values) for scene, values in by_scene.items()}


def _bound_path(path: str, *, repository_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else Path(repository_root) / candidate


def _read_bound_record(
    record: Mapping[str, Any],
    *,
    repository_root: Path,
    access_ledger: MutableMapping[str, int] | None = None,
    access_key: str | None = None,
) -> bytes:
    raw = _read_bound_file(
        _bound_path(str(record["path"]), repository_root=repository_root),
        expected_sha256=str(record["file_sha256"]),
        access_ledger=access_ledger,
        access_key=access_key,
    )
    if len(raw) != int(record["byte_count"]):
        raise LabelContractError("bound source byte count changed")
    return raw


def _summary_path(value: object, *, repository_root: Path) -> Path:
    candidate = Path(str(value))
    return (candidate if candidate.is_absolute() else repository_root / candidate).absolute()


def _render_plan_inventory_record(
    raw_indexes: RawIndexesV1, *, scene_id: str
) -> Mapping[str, Any]:
    provenance = raw_indexes.manifest["input_provenance"]
    matches = [
        record
        for record in provenance["source_payload_inventory"]
        if isinstance(record, Mapping)
        and record.get("scene_id") == scene_id
        and record.get("purpose") == "render_plan"
    ]
    if len(matches) != 1:
        raise LabelContractError("raw inventory does not bind one render plan")
    return matches[0]


def load_joined_scene_v1(
    *,
    raw_indexes: RawIndexesV1,
    scene_id: str,
    source_records: Mapping[str, Mapping[str, Any]],
    repository_root: Path,
    access_ledger: MutableMapping[str, int] | None = None,
) -> tuple[SceneManifest, tuple[JoinedCurrentStateV1, ...]]:
    """Scan one explicitly bound frames file once and join only current endpoints."""

    if set(source_records) != set(_SOURCE_PURPOSES):
        raise LabelContractError("scene source set changed")
    _record_access(access_ledger, "scene_join_calls_started")
    shard = raw_indexes.shard_by_scene.get(scene_id)
    if shard is None:
        raise LabelContractError("scene is absent from raw shard inventory")
    role = str(shard["dataset_role"])
    family = str(shard["family"])
    pairs = tuple(pair for pair in raw_indexes.pairs if pair["scene_id"] == scene_id)
    if not pairs:
        raise LabelContractError("scene has no current states")

    summary_record = source_records["render_summary"]
    summary = _json_loads_strict(
        _read_bound_record(
            summary_record,
            repository_root=repository_root,
            access_ledger=access_ledger,
            access_key="render_summary_opens",
        ),
        name=f"render summary {scene_id}",
    )
    if (
        not isinstance(summary, Mapping)
        or summary.get("schema") != "lewm_rendered_vision_v04"
        or summary.get("render_status") != "complete"
        or summary.get("scene_id") != scene_id
        or summary.get("family") != family
        or summary.get("g2_model_outputs_opened", False) is not False
    ):
        raise LabelContractError("render summary identity/status changed")
    summary_sources = summary.get("source")
    if not isinstance(summary_sources, Mapping):
        raise LabelContractError("render summary source inventory is absent")
    expected_source_records = {
        "frames_jsonl": source_records["source_frames_jsonl"],
        "scene_manifest": source_records["source_scene_manifest"],
        "plan": _render_plan_inventory_record(raw_indexes, scene_id=scene_id),
    }
    for name, expected in expected_source_records.items():
        entry = summary_sources.get(name)
        if (
            not isinstance(entry, Mapping)
            or set(entry) != {"path", "sha256"}
            or _summary_path(entry.get("path"), repository_root=repository_root)
            != _summary_path(expected.get("path"), repository_root=repository_root)
            or entry.get("sha256") != expected.get("file_sha256")
        ):
            raise LabelContractError(f"render summary source.{name} binding changed")

    rendered = summary.get("rendered_frames")
    if not isinstance(rendered, list):
        raise LabelContractError("render summary has no rendered-frame commitments")
    rendered_by_key: dict[tuple[int, int], Mapping[str, Any]] = {}
    for item in rendered:
        if not isinstance(item, Mapping) or set(item) != {
            "frame_index",
            "env_index",
            "timestamp_ns",
            "image_sha256",
        }:
            raise LabelContractError("rendered-frame commitment is malformed")
        key = (
            _exact_int(item.get("frame_index"), name="rendered frame_index", minimum=0),
            _exact_int(item.get("env_index"), name="rendered env_index", minimum=0),
        )
        _exact_int(item.get("timestamp_ns"), name="rendered timestamp_ns", minimum=0)
        if not _is_sha256(item.get("image_sha256")) or key in rendered_by_key:
            raise LabelContractError("rendered-frame key/hash is malformed or repeated")
        rendered_by_key[key] = item

    endpoint_context: dict[str, dict[str, Any]] = {}
    wanted: dict[tuple[int, int, int], str] = {}
    for pair in pairs:
        digest = str(pair["current_endpoint_sha256"])
        endpoint = raw_indexes.endpoint_by_sha256[digest]
        frame_index, env_index = parse_rendered_filename(
            str(endpoint["image_path_metadata_only"])
        )
        rendered_item = rendered_by_key.get((frame_index, env_index))
        if rendered_item is None:
            raise LabelContractError("current endpoint is absent from render summary")
        timestamp_ns = int(rendered_item["timestamp_ns"])
        if endpoint["image_sha256_commitment_only"] != rendered_item["image_sha256"]:
            raise LabelContractError("endpoint/render-summary image commitment changed")
        image_parent = Path(str(endpoint["image_path_metadata_only"])).absolute().parent
        summary_parent = _bound_path(
            str(summary_record["path"]), repository_root=repository_root
        ).absolute().parent
        if image_parent != summary_parent / "rgb":
            raise PermissionError("endpoint image metadata escapes the bound render summary")
        key = (frame_index, env_index, timestamp_ns)
        if key in wanted:
            raise LabelContractError("two current endpoints share one source-frame key")
        wanted[key] = digest
        endpoint_context[digest] = {
            "pair": pair,
            "endpoint": endpoint,
            "frame_index": frame_index,
            "env_index": env_index,
            "timestamp_ns": timestamp_ns,
            "image_sha256": str(rendered_item["image_sha256"]),
        }

    frames_record = source_records["source_frames_jsonl"]
    frames_path = _bound_path(str(frames_record["path"]), repository_root=repository_root)
    if frames_path.is_symlink() or not frames_path.is_file():
        raise LabelContractError("frames source is not a regular non-symlink")
    selected: dict[str, tuple[int, Mapping[str, Any]]] = {}
    digest = hashlib.sha256()
    byte_count = 0
    _record_access(access_ledger, "source_frames_jsonl_opens")
    with frames_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            digest.update(line)
            byte_count += len(line)
            if not line.endswith(b"\n") or not line.strip():
                raise LabelContractError("source frames must be nonblank terminal-newline JSONL")
            frame = _json_loads_strict(line, name=f"source frame {scene_id}:{line_number}")
            if not isinstance(frame, Mapping):
                raise LabelContractError("source frame must be an object")
            key = (
                _exact_int(frame.get("frame_index"), name="source frame_index", minimum=0),
                _exact_int(frame.get("env_index"), name="source env_index", minimum=0),
                _exact_int(frame.get("timestamp_ns"), name="source timestamp_ns", minimum=0),
            )
            endpoint_digest = wanted.get(key)
            if endpoint_digest is not None:
                if endpoint_digest in selected:
                    raise LabelContractError("source frame matched a current endpoint twice")
                selected[endpoint_digest] = (line_number, frame)
    if (
        byte_count != int(frames_record["byte_count"])
        or digest.hexdigest() != frames_record["file_sha256"]
        or set(selected) != set(endpoint_context)
    ):
        raise LabelContractError("source frames hash/count/selection changed")

    manifest_record = source_records["source_scene_manifest"]
    manifest_payload = _json_loads_strict(
        _read_bound_record(
            manifest_record,
            repository_root=repository_root,
            access_ledger=access_ledger,
            access_key="scene_manifest_opens",
        ),
        name=f"scene manifest {scene_id}",
    )
    if not isinstance(manifest_payload, dict):
        raise LabelContractError("scene manifest must be an object")
    scene_manifest = parse_scene_manifest_dict(manifest_payload)
    semantic_sha256 = manifest_sha256(scene_manifest)
    if scene_manifest.scene_id != scene_id or scene_manifest.family != family:
        raise LabelContractError("parsed scene manifest identity changed")

    joined_by_pair: dict[str, JoinedCurrentStateV1] = {}
    for endpoint_digest, context in endpoint_context.items():
        pair = context["pair"]
        endpoint = context["endpoint"]
        line_number, frame = selected[endpoint_digest]
        if (
            pair.get("frames_jsonl_sha256") != frames_record["file_sha256"]
            or pair.get("scene_manifest_sha256") != semantic_sha256
            or endpoint.get("dataset_role") != role
            or endpoint.get("scene_id") != scene_id
            or endpoint.get("family") != family
        ):
            raise LabelContractError("pair/endpoint/source hash matrix changed")
        episode = frame.get("episode")
        if not isinstance(episode, Mapping):
            raise LabelContractError("source frame lacks episode provenance")
        source_episode_id = _exact_int(
            episode.get("episode_id"), name="source episode_id", minimum=0
        )
        episode_id = str(source_episode_id)
        reset_count = _exact_int(episode.get("reset_count"), name="source reset_count", minimum=0)
        episode_step = _exact_int(episode.get("episode_step"), name="source episode_step", minimum=0)
        if (
            pair.get("episode_id") != episode_id
            or pair.get("env_index") != context["env_index"]
            or pair.get("reset_count") != reset_count
        ):
            raise LabelContractError("pair/source episode join changed")
        identity = {
            "dataset_role": role,
            "scene_id": scene_id,
            "episode_id": episode_id,
            "env_index": context["env_index"],
            "episode_step": episode_step,
            "frame_index": context["frame_index"],
            "timestamp_ns": context["timestamp_ns"],
            "image_sha256": context["image_sha256"],
        }
        if canonical_json_sha256(identity) != endpoint_digest:
            raise LabelContractError("current endpoint identity did not reconstruct")
        base_pose = frame.get("base_pose_world")
        base_rpy = frame.get("base_rpy_rad")
        position = base_pose.get("position") if isinstance(base_pose, Mapping) else None
        if (
            not isinstance(position, Mapping)
            or set(position) != {"x", "y", "z"}
            or not isinstance(base_rpy, Mapping)
            or "yaw" not in base_rpy
            or set(base_rpy) - {"roll", "pitch", "yaw"}
        ):
            raise LabelContractError("source frame lacks its base position/yaw")
        source_pose = Pose2D(
            _finite(position["x"], name="source base x"),
            _finite(position["y"], name="source base y"),
            _finite(base_rpy["yaw"], name="source base yaw"),
        )
        _finite(position["z"], name="source base z")
        joined_by_pair[str(pair["content_sha256"])] = JoinedCurrentStateV1(
            pair=pair,
            endpoint=endpoint,
            source_pose_world=source_pose,
            source_line_number=line_number,
            source_bindings={purpose: dict(record) for purpose, record in source_records.items()},
        )
    ordered = tuple(joined_by_pair[str(pair["content_sha256"])] for pair in pairs)
    return scene_manifest, ordered


def _file_record(path: Path, *, relative_to: Path, **extra: Any) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "path": str(path.relative_to(relative_to)),
        "byte_count": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        **extra,
    }


def _publish_staging_manifest_last_v1(
    staging: Path,
    output: Path,
    filenames: Sequence[str],
) -> None:
    for filename in filenames:
        if Path(filename).name != filename or filename == "manifest.json":
            raise LabelContractError("staged label payload filename changed")
        os.rename(staging / filename, output / filename)
    os.rename(staging / "manifest.json", output / "manifest.json")
    staging.rmdir()


def materialize_role_labels_v1(
    *,
    raw_indexes: RawIndexesV1,
    geometry_inputs: GeometryInputsV1,
    source_records_by_scene: Mapping[str, Mapping[str, Mapping[str, Any]]],
    execution_binding: Mapping[str, Any],
    repository_root: Path,
    output_directory: Path,
    schedule_indices: Sequence[int] | None = None,
    access_ledger: MutableMapping[str, int] | None = None,
) -> Mapping[str, Any]:
    """Run one model-free source scan and publish three canonical role files."""

    output = Path(output_directory).absolute()
    expected_output = _label_output_path(repository_root=repository_root)
    if output != expected_output:
        raise PermissionError("materializer output escaped the exact reserved label root")
    reservation = load_label_reservation_v1(
        repository_root,
        source_manifest=execution_binding.get("source_manifest"),
        independent_source_review=execution_binding.get(
            "independent_source_review"
        ),
    )
    claim_path = output / "builder_claim.json"
    if claim_path.is_symlink() or not claim_path.is_file():
        raise PermissionError("label materializer has no one-shot builder claim")
    claim = _parse_canonical_object(claim_path.read_bytes(), name="label builder claim")
    if (
        claim.get("schema") != LABEL_BUILDER_CLAIM_SCHEMA
        or claim.get("status") != "CLAIMED_ONE_EXACT_LABEL_BUILDER_INVOCATION"
        or claim.get("reservation_content_sha256")
        != reservation.get("content_sha256")
        or claim.get("execution_binding_content_sha256")
        != execution_binding.get("content_sha256")
        or claim.get("retry_authorized") is not False
        or claim.get("resume_authorized") is not False
        or claim.get("second_invocation_authorized") is not False
    ):
        raise PermissionError("label builder claim identity changed")
    if set(source_records_by_scene) != set(raw_indexes.shard_by_scene):
        raise LabelContractError("materializer scene source set changed")
    role_state_index: dict[str, int] = {}
    for role in OUTPUT_ROLE_ORDER:
        role_pairs = tuple(pair for pair in raw_indexes.pairs if pair["dataset_role"] == role)
        role_state_index.update(
            {str(pair["content_sha256"]): index for index, pair in enumerate(role_pairs)}
        )

    rows_by_role: dict[str, list[dict[str, Any]]] = {
        role: [] for role in OUTPUT_ROLE_ORDER
    }
    scenes = sorted(
        raw_indexes.shard_by_scene,
        key=lambda scene: (
            OUTPUT_ROLE_ORDER.index(str(raw_indexes.shard_by_scene[scene]["dataset_role"])),
            scene,
        ),
    )
    for scene_id in scenes:
        scene_manifest, joined_states = load_joined_scene_v1(
            raw_indexes=raw_indexes,
            scene_id=scene_id,
            source_records=source_records_by_scene[scene_id],
            repository_root=repository_root,
            access_ledger=access_ledger,
        )
        for state in joined_states:
            pair_sha256 = str(state.pair["content_sha256"])
            role = str(state.pair["dataset_role"])
            rows_by_role[role].extend(
                label_state_v1(
                    pair=state.pair,
                    endpoint=state.endpoint,
                    source_pose_world=state.source_pose_world,
                    source_line_number=state.source_line_number,
                    scene_manifest=scene_manifest,
                    footprint=geometry_inputs.footprint,
                    commands_by_action=geometry_inputs.commands_by_action,
                    source_bindings=state.source_bindings,
                    role_state_index=role_state_index[pair_sha256],
                )
            )
    for role in OUTPUT_ROLE_ORDER:
        rows_by_role[role].sort(
            key=lambda row: (int(row["role_state_index"]), int(row["action_index"]))
        )
    preflight = summarize_preflight_v1(rows_by_role)
    if schedule_indices is not None:
        preflight["frozen_schedule"] = scheduled_preflight_v1(
            rows_by_role["train"], schedule_indices
        )

    predicted_masks = predicted_next_corridor_masks_v1(geometry_inputs.footprint)
    persistence_masks = persistence_corridor_masks_v1(
        geometry_inputs.footprint, geometry_inputs.commands_by_action
    )
    support = projective_support_mask_v1()
    if (
        np.any(predicted_masks & (1 - support)[None, :, :])
        or np.any(persistence_masks & (1 - support)[None, None, :, :])
    ):
        raise LabelContractError("corridor mask escaped learned projective support")

    staging = output / "staging"
    staging.mkdir(mode=0o700)
    try:
        files: list[dict[str, Any]] = []
        for role in OUTPUT_ROLE_ORDER:
            filename = f"{role}.jsonl"
            path = staging / filename
            payload = b"".join(
                canonical_json_bytes(row) + b"\n" for row in rows_by_role[role]
            )
            path.write_bytes(payload)
            files.append(
                _file_record(
                    path,
                    relative_to=staging,
                    schema=LABEL_ROW_SCHEMA,
                    dataset_role=role,
                    state_count=len(rows_by_role[role]) // len(ACTION_ORDER),
                    action_row_count=len(rows_by_role[role]),
                    ordered_row_content_sha256=canonical_json_sha256(
                        [row["content_sha256"] for row in rows_by_role[role]]
                    ),
                )
            )
        predicted_path = staging / "predicted_next_corridor_masks.u1"
        predicted_path.write_bytes(predicted_masks.tobytes(order="C"))
        files.append(
            _file_record(
                predicted_path,
                relative_to=staging,
                dtype="|u1",
                shape=[11, 64, 64],
                set_cell_count=int(predicted_masks.sum()),
            )
        )
        persistence_path = staging / "persistence_corridor_masks.u1"
        persistence_path.write_bytes(persistence_masks.tobytes(order="C"))
        files.append(
            _file_record(
                persistence_path,
                relative_to=staging,
                dtype="|u1",
                shape=[9, 11, 64, 64],
                set_cell_count=int(persistence_masks.sum()),
            )
        )
        support_path = staging / "projective_support_mask.u1"
        support_path.write_bytes(support.tobytes(order="C"))
        files.append(
            _file_record(
                support_path,
                relative_to=staging,
                dtype="|u1",
                shape=[64, 64],
                set_cell_count=int(support.sum()),
            )
        )
        files.sort(key=lambda record: str(record["path"]))
        manifest = with_content_sha256(
            {
                "schema": LABEL_MANIFEST_SCHEMA,
                "status": "complete_pre_gpu_development_labels",
                "preregistration_commit": PREREGISTRATION_COMMIT,
                "roles": list(OUTPUT_ROLE_ORDER),
                "action_order": list(ACTION_ORDER),
                "state_count": sum(len(rows) // 9 for rows in rows_by_role.values()),
                "action_row_count": sum(len(rows) for rows in rows_by_role.values()),
                "station_label_count": sum(len(rows) * 11 for rows in rows_by_role.values()),
                "remote_sampler": {
                    "pose_count": 91,
                    "offsets": list(REMOTE_SAMPLE_OFFSETS),
                    "binary_sha256": REMOTE_SAMPLE_SHA256,
                },
                "files": files,
                "preflight": preflight,
                "input_bindings": {
                    "label_reservation": dict(reservation),
                    "label_builder_claim": dict(claim),
                    "integrity_adapter_amendment": dict(
                        execution_binding["integrity_adapter_amendment"]
                    ),
                    "label_v1_terminal_predecessor_bindings": {
                        name: dict(binding)
                        for name, binding in execution_binding[
                            "label_v1_terminal_predecessor_bindings"
                        ].items()
                    },
                    "schedule_schema_adapter_amendment": dict(
                        execution_binding["schedule_schema_adapter_amendment"]
                    ),
                    "label_v2_terminal_predecessor_bindings": {
                        name: dict(binding)
                        for name, binding in execution_binding[
                            "label_v2_terminal_predecessor_bindings"
                        ].items()
                    },
                    "source_episode_id_adapter_amendment": dict(
                        execution_binding["source_episode_id_adapter_amendment"]
                    ),
                    "label_v3_terminal_predecessor_bindings": {
                        name: dict(binding)
                        for name, binding in execution_binding[
                            "label_v3_terminal_predecessor_bindings"
                        ].items()
                    },
                    "source_manifest": dict(execution_binding["source_manifest"]),
                    "independent_source_review": dict(
                        execution_binding["independent_source_review"]
                    ),
                    "raw_manifest": {
                        "file_sha256": RAW_MANIFEST_FILE_SHA256,
                        "content_sha256": RAW_MANIFEST_CONTENT_SHA256,
                    },
                    "raw_pairs": {"file_sha256": RAW_PAIRS_FILE_SHA256},
                    "raw_endpoints": {"file_sha256": RAW_ENDPOINTS_FILE_SHA256},
                    "raw_audit": {
                        "file_sha256": RAW_AUDIT_FILE_SHA256,
                        "content_sha256": RAW_AUDIT_CONTENT_SHA256,
                    },
                    **{
                        key: dict(value)
                        for key, value in geometry_inputs.source_bindings.items()
                    },
                    "execution_binding_content_sha256": execution_binding.get(
                        "content_sha256"
                    ),
                    "source_records_sha256": canonical_json_sha256(
                        execution_binding.get("source_records")
                    ),
                    "schedule_prefix_sha256": SCHEDULE_PREFIX_SHA256,
                },
                "access_ledger": _normalize_access_ledger(access_ledger),
                "authority": {
                    "training_authorized": False,
                    "g2_authorized": False,
                    "navigation_authorized": False,
                    "heldout_authorized": False,
                    "production_authorized": False,
                    "promotion_authorized": False,
                },
            }
        )
        (staging / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        # Publish every payload from inside the consumed root, then make the
        # bundle visible as complete with the manifest as the final rename.
        _publish_staging_manifest_last_v1(
            staging,
            output,
            [str(record["path"]) for record in files],
        )
        return manifest
    except BaseException:
        # Leave a private staging directory as a failure receipt; never overwrite
        # or silently delete partial evidence.
        raise


def load_label_manifest_v1(
    path: Path, *, expected_file_sha256: str
) -> Mapping[str, Any]:
    """Load a hash-bound published label manifest for a later runner."""

    manifest = _parse_canonical_object(
        _read_bound_file(path, expected_sha256=expected_file_sha256),
        name="projective-support label manifest",
    )
    if (
        manifest.get("schema") != LABEL_MANIFEST_SCHEMA
        or manifest.get("status") != "complete_pre_gpu_development_labels"
        or manifest.get("preregistration_commit") != PREREGISTRATION_COMMIT
        or manifest.get("roles") != list(OUTPUT_ROLE_ORDER)
        or manifest.get("action_order") != list(ACTION_ORDER)
        or manifest.get("state_count") != 5_172
        or manifest.get("action_row_count") != 46_548
        or manifest.get("station_label_count") != 512_028
    ):
        raise LabelContractError("published label manifest identity/count changed")
    preflight = manifest.get("preflight")
    frozen_schedule = (
        preflight.get("frozen_schedule") if isinstance(preflight, Mapping) else None
    )
    if (
        not isinstance(frozen_schedule, Mapping)
        or frozen_schedule.get("presentation_count") != 16_000
        or frozen_schedule.get("presentation_indices_sha256")
        != SCHEDULE_PREFIX_SHA256
    ):
        raise LabelContractError("published label schedule-prefix identity changed")
    files = manifest.get("files")
    if not isinstance(files, list) or len(files) != 6:
        raise LabelContractError("published label file inventory changed")
    by_path = {
        str(record.get("path")): record
        for record in files
        if isinstance(record, Mapping)
    }
    if set(by_path) != {
        *(f"{role}.jsonl" for role in OUTPUT_ROLE_ORDER),
        "predicted_next_corridor_masks.u1",
        "persistence_corridor_masks.u1",
        "projective_support_mask.u1",
    }:
        raise LabelContractError("published label filenames changed")
    if (
        by_path["predicted_next_corridor_masks.u1"].get("file_sha256")
        != PREDICTED_NEXT_MASK_SHA256
        or by_path["persistence_corridor_masks.u1"].get("file_sha256")
        != PERSISTENCE_STACK_SHA256
        or by_path["projective_support_mask.u1"].get("set_cell_count")
        != PROJECTIVE_SUPPORT_CELL_COUNT
    ):
        raise LabelContractError("published static mask identity changed")
    return manifest


def load_role_labels_v1(
    path: Path,
    *,
    role: str,
    expected_file_sha256: str,
) -> tuple[Mapping[str, Any], ...]:
    """Load one hash-bound role file in its canonical state/action order."""

    rows = _parse_canonical_jsonl(
        _read_bound_file(path, expected_sha256=expected_file_sha256),
        name=f"{role} projective-support labels",
    )
    return validate_label_rows_v1(rows, role=role)


__all__ = [
    "ACTION_ORDER",
    "EXECUTION_BINDING_SCHEMA",
    "GeometryInputsV1",
    "JoinedCurrentStateV1",
    "LABEL_BUILDER_CLAIM_RELATIVE_PATH",
    "LABEL_BUILDER_CLAIM_SCHEMA",
    "LABEL_MANIFEST_SCHEMA",
    "LABEL_EXECUTION_BINDING_RELATIVE_PATH",
    "LABEL_FAILURE_SCHEMA",
    "LABEL_OUTPUT_RELATIVE_PATH",
    "LABEL_RESERVATION_RELATIVE_PATH",
    "LABEL_RESERVATION_SCHEMA",
    "LABEL_ROW_SCHEMA",
    "LabelContractError",
    "OUTPUT_ROLE_ORDER",
    "REGISTERED_SELECTION_FAMILIES",
    "RawIndexesV1",
    "canonical_json_bytes",
    "canonical_json_sha256",
    "claim_label_builder_v1",
    "integrate_action_v1",
    "label_state_v1",
    "load_and_validate_raw_indexes",
    "load_execution_binding_file_v1",
    "load_geometry_inputs_v1",
    "load_joined_scene_v1",
    "load_label_manifest_v1",
    "load_label_reservation_v1",
    "load_role_labels_v1",
    "load_schedule_indices_v1",
    "materialize_role_labels_v1",
    "new_access_ledger_v1",
    "persistence_corridor_masks_v1",
    "predicted_next_corridor_masks_v1",
    "projective_support_mask_v1",
    "rasterize_corridor_masks_v1",
    "remote_corridor_pose_samples_v1",
    "reserve_label_root_v1",
    "SCHEDULE_PREFIX_SHA256",
    "scheduled_preflight_v1",
    "summarize_preflight_v1",
    "transform_pose_v1",
    "validate_execution_binding_v1",
    "validate_execution_binding_envelope_v1",
    "validate_label_rows_v1",
    "validate_raw_audit_v1",
    "with_content_sha256",
    "write_label_failure_v1",
]
