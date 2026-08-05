#!/usr/bin/env python3
"""Run the frozen fit-only N32 camera-frustum observability audit.

The runner never opens RGB, a checkpoint, a holdout, G2, runtime, or sealed
bytes.  Image paths and hashes are commitments only.  Label shards are read
once into verified in-memory bytes; source geometry is allowlisted by a
committed parent before it is parsed and is rehashed afterwards.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import types
from typing import Any, Mapping, Sequence
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
LEWM_WORLDS = ROOT / "lewm_worlds"
if str(LEWM_WORLDS) not in sys.path:
    sys.path.insert(0, str(LEWM_WORLDS))

EXECUTION_BINDING_SHA256 = (
    "c045a5566e53686ab80fdc86c2de910d312c02c5f03f253dfda13be7a85a16c9"
)
RESULT_SCHEMA = "lewm_go2_n32_camera_frustum_observability_audit_result_v1"
FAMILIES = (
    "open_obstacle_field",
    "rough_local_dynamics",
    "small_enclosed_maze",
    "medium_enclosed_maze",
    "large_enclosed_maze",
)
ENDPOINT_SIDES = ("current", "next")
UNKNOWN_CLASS = 0
FREE_CLASS = 1

labels_v3: Any = None
BoxObject: Any = None
parse_scene_manifest_dict: Any = None
InflatedOccupancyGrid: Any = None
aggregate_label_observability: Any = None
analyze_frame_labels: Any = None
audit_camera_centered_mapping: Any = None
authorization_decision: Any = None
build_camera_centered_mapping: Any = None
frozen_camera_geometry_contract: Any = None
old_body_column_span_audit: Any = None
_SEMANTICS_LOADED = False


BINDING_PATH = (
    ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_binding_2026-07-11.md"
)
IMPLEMENTATION_MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.md"
)
MACHINE_IMPLEMENTATION_MANIFEST_PATH = (
    ROOT
    / "docs/lewm_go2_n32_camera_frustum_observability_audit_v2_implementation_manifest_2026-07-11.json"
)
PREFLIGHT_INCIDENT_PATH = (
    ROOT
    / "docs/lewm_go2_n32_camera_frustum_manifest_preparation_failure_2026-07-11.md"
)
PREFLIGHT_INCIDENT_SHA256 = (
    "5c3fad3b8e296aed239c3573e263af766b52e391fb9fe86e0e31d26c94845db3"
)
PREFLIGHT_INCIDENT_STATUS = "acknowledged_pre_authoritative_run"
MACHINE_MANIFEST_SCHEMA = (
    "lewm_go2_n32_camera_frustum_observability_audit_implementation_manifest_v1"
)
PANEL_PATH = ROOT / ".generated/go2_physical_micro_overfit/patch7_v1/panel.json"
OUTPUT_PATH = (
    ROOT
    / ".generated/go2_n32_camera_frustum_observability_audit/v2/result.json"
)
PANEL_FILE_SHA256 = (
    "c3f44c6b1147efbb6a5fbc2294c6431c72e25da877cab6884972d25c1ffdb16c"
)
PANEL_CONTENT_SHA256 = (
    "f3e5198b81ac48c06f6c8e4b21e8bf24d62200e3830b1d6685d949a668349d5f"
)
FIT_ROWS_SHA256 = "5a75e202c8f7a803aafaad093c7f474137dd2e69f50ecdb7fb4e97765afb659d"
EXPECTED_TRANSITIONS = 160
EXPECTED_FRAMES = 320
EXPECTED_SHARDS = 20

V4_REPORT_PATH = ROOT / "docs/lewm_go2_categorical_radial_n32_v4_result_2026-07-11.md"
V4_REPORT_SHA256 = (
    "dd0842d1c59b42a985eaf0843f0d6f6adc41286a2a1a2b4b1f95111a9c0efa50"
)
KNOWN_BIAS_PROOF_PATH = ROOT / "docs/lewm_go2_n32_known_bias_impossibility_2026-07-11.md"
KNOWN_BIAS_PROOF_SHA256 = (
    "e214bb80bcccf9ae5051231d90f7a5d8c2bfa33ca799e7db3eb969698fa2108a"
)

SUMMARY_ROOT = ROOT / ".generated/go2_render_selected_v04/scenes"
EXPECTED_SUMMARY_SHA256 = {
    "scene_074f19f0608afca2/summary.json": "7a5d3b1e6ff5a8acb914ae5226326084c2b951517c110ffc19d7a99945fe0413",
    "scene_142dbd9b0428f16f/summary.json": "995e192cc1830f32bd2dc6d358da91f5bdaec48bd585ac2dadecc45517cbd2b0",
    "scene_4931dab75d2ceee8/summary.json": "7800d0d6a14ea54b9970d1dac36472446cd525af8c893736ebe1c4b4bf57cc23",
    "scene_49db95fc9ed0ce8f/summary.json": "80a035ceecf56f2c668fed3ab1dbabeeca181cb2886fedafa7116ec26bc0566d",
    "scene_4af4d0549179a705/summary.json": "bcb3866fe141c0c629368eefee8e228630ca8f3b30e1c2810b34e68fd61347b4",
    "scene_7239d51aced24ee3/summary.json": "5c6785479b9a302fcffb1d7532e450af10d2e2625a030eff872edf22b23aef6f",
    "scene_7f390beda8f5070f/summary.json": "2dc1f874130cb733be4f28eccae3359aac7bdc4e2947718391182ad651d027e7",
    "scene_9ff98ead4f1a2e96/summary.json": "203ffca9205f68dc74e6135718d3fec4bfb55e9c841bf7a4eb49964930309cc0",
    "scene_a81215e4d326a2a2/summary.json": "7b9c5dff08be0876327f8b625d225e4b1729320f98b9ccb1efcbd1c68cc2e3c1",
    "scene_b1355439db03d8f8/summary.json": "d21cd06b202422ecce81c009c08b13ab4e92be86bdc93f6571e69ac265f33fa9",
    "scene_b748962d390baeca/summary.json": "a3a90172486dc08f3e7a1728da71e43ae224aefddc22ba32e1de5b4fa6ab7f38",
    "scene_b75bb34744434970/summary.json": "64bcf8f57c55cb3456f6dd04be23bbdc417865b2ee8dbad914b5eaa387d61b6b",
    "scene_bc5a05ec9fce8d9c/summary.json": "41377a7619560162b7fd4453ca302321d2f5f22aee1a8c7397ff32626bbb1a92",
    "scene_c60650f53aaae4a6/summary.json": "be319a4b1a6e456367c3a6b4d9eee5059380ef83ebe720416b7f292a959c2d6e",
    "scene_cfcadb2bd44cce85/summary.json": "fa5a9049889a10700cd678fea78ecfb6f91545403ebfdfd304d1dc59a4b6d40a",
    "scene_d8b06cdfb1f739ed/summary.json": "6f06ee751ec3a26de741bdafcf39cb044e49734cb5a2ab1103ab2834e3edf3c2",
    "scene_ddc88df212918857/summary.json": "7b1deec174715696d4a3dd653610886e1244edfa993a8c0dc0e91176b728488f",
    "scene_df1c6b34503f2ae1/summary.json": "deed15024342195754b9022522c048624ab09a1d55e2727f615822d5b6f658e8",
    "scene_e0c2fe611e747d90/summary.json": "df2fde293612833f00f15a25a8c81c799e15e4674f5ad7f29a0d7ea06e9fd341",
    "scene_ebc33be3e6a87264/summary.json": "12b5825f4dc2388631190cc80dd42f9cea1bbbbf002f666f12ca53ddde704a35",
}

FRAME_IDENTITY_FIELDS = (
    "family",
    "scene_id",
    "global_row",
    "side",
    "image_sha256",
    "label_shard_sha256",
    "label_row",
)
FORBIDDEN_ACCESS_FIELDS = (
    "rgb_byte_opens",
    "rgb_decodes",
    "holdout_label_or_geometry_opens",
    "selection_or_calibration_opens",
    "physical_nontrain_role_opens",
    "g2_opens",
    "runtime_opens",
    "sealed_opens",
    "model_checkpoint_or_output_opens",
    "generated_v4_result_opens",
    "seed_20260711_opens",
)
PRIMARY_DENIAL_REASONS = (
    "sealed",
    "g2",
    "seed_20260711",
    "generated_v4_result",
    "model",
    "runtime",
    "physical_nontrain",
    "selection_or_calibration",
    "holdout",
    "image_or_depth",
    "unregistered_role",
    "forbidden_modality",
    "path_alias_or_escape",
    "unallowlisted",
)
DENIAL_MODALITIES = (
    "markdown",
    "python_source",
    "json",
    "jsonl",
    "npz",
    "image",
    "video",
    "raster_array",
    "point_cloud",
    "model",
    "archive",
    "unknown",
)
ALLOWED_PATH_ROLES = frozenset(
    {
        "binding",
        "incident_record",
        "v4_adjudication_report",
        "known_bias_proof",
        "human_implementation_manifest",
        "machine_implementation_manifest",
        "implementation_source",
        "fit_panel",
        "fit_label_shard",
        "physical_geometry_contract",
        "render_audit_contract",
        "fit_render_summary",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
        "fit_frame_selection",
        "audit_output",
    }
)
ALLOWED_MODALITY_BY_ROLE = {
    "binding": "markdown",
    "incident_record": "markdown",
    "v4_adjudication_report": "markdown",
    "known_bias_proof": "markdown",
    "human_implementation_manifest": "markdown",
    "machine_implementation_manifest": "json",
    "implementation_source": None,
    "fit_panel": "json",
    "fit_label_shard": "npz",
    "physical_geometry_contract": "json",
    "render_audit_contract": "json",
    "fit_render_summary": "json",
    "render_source_plan": "json",
    "source_frames_jsonl": "jsonl",
    "source_scene_manifest": "json",
    "renderer_source": "python_source",
    "fit_frame_selection": "json",
    "audit_output": "json",
}
DISTANCE_BINS = (
    ("0.0_to_0.5", 0.0, 0.5),
    ("0.5_to_1.0", 0.5, 1.0),
    ("1.0_to_2.0", 1.0, 2.0),
    ("2.0_to_3.0", 2.0, 3.0),
    ("3.0_plus", 3.0, None),
)
REGISTERED_LABEL_ARRAY_NAMES = (
    "current_labels",
    "current_supervision_mask",
    "next_labels",
    "next_supervision_mask",
)
REGISTERED_AUXILIARY_ARRAY_NAMES = (
    "current_observed_mask",
    "next_observed_mask",
    "relative_se2_current_frame",
    "primitive",
    "current_image_path",
    "next_image_path",
    "current_image_sha256",
    "next_image_sha256",
)
REGISTERED_SHARD_ARRAY_NAMES = frozenset(
    (*REGISTERED_LABEL_ARRAY_NAMES, *REGISTERED_AUXILIARY_ARRAY_NAMES)
)
NOMINAL_CAMERA_MOUNT_BODY = {
    "parent_link": "camera_link",
    "rpy_body_rad": [0.0, 0.0, 0.0],
    "xyz_body_m": [0.326, 0.0, 0.043],
}
CAMERA_COMPOSITION_TOLERANCE = 1e-5
EXPECTED_AUDIT_TEST_COUNT = 145
REQUIRED_VERIFICATION_COMMANDS = (
    {
        "category": "pytest",
        "command": (
            "PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "
            "PYTHONPATH=/home/andrewknowles/Workspace/LeWMQuad-v3:"
            "/home/andrewknowles/Workspace/LeWMQuad-v3/lewm_worlds "
            "/usr/bin/python3 -m pytest -q "
            "lewm/tests/test_go2_n32_camera_frustum_observability.py "
            "lewm/tests/test_go2_n32_camera_frustum_fit_evidence_stdlib.py "
            "lewm/tests/test_audit_go2_n32_camera_frustum_observability.py "
            "lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py"
        ),
        "deterministic_result": {
            "kind": "passed_test_count",
            "count": EXPECTED_AUDIT_TEST_COUNT,
        },
    },
    {
        "category": "py_compile",
        "command": (
            "/home/andrewknowles/TinyQuadJEPA/bin/python -m py_compile "
            "lewm/benchmarks/go2_n32_camera_frustum_observability.py "
            "lewm/benchmarks/go2_n32_camera_frustum_fit_evidence_stdlib.py "
            "scripts/audit_go2_n32_camera_frustum_observability.py "
            "scripts/finalize_go2_n32_camera_frustum_observability.py"
        ),
        "deterministic_result": {
            "kind": "compiled_file_count",
            "count": 4,
        },
    },
    {
        "category": "import_isolation",
        "command": (
            "PYTHONNOUSERSITE=1 /usr/bin/python3 -c \"import sys; "
            "from scripts import finalize_go2_n32_camera_frustum_observability; "
            "assert 'numpy' not in sys.modules; assert 'torch' not in sys.modules\""
        ),
        "deterministic_result": {
            "kind": "forbidden_import_count",
            "count": 2,
        },
    },
    {
        "category": "diff_check",
        "command": "git diff --check",
        "deterministic_result": {
            "kind": "whitespace_error_count",
            "count": 0,
        },
    },
)


def _default_source_paths() -> dict[str, Path]:
    return {
        "binding": BINDING_PATH,
        "audit_core": ROOT / "lewm/benchmarks/go2_n32_camera_frustum_observability.py",
        "audit_core_test": ROOT / "lewm/tests/test_go2_n32_camera_frustum_observability.py",
        "audit_runner": Path(__file__).resolve(),
        "audit_runner_test": ROOT / "lewm/tests/test_audit_go2_n32_camera_frustum_observability.py",
        "audit_finalizer": ROOT / "scripts/finalize_go2_n32_camera_frustum_observability.py",
        "audit_finalizer_test": ROOT / "lewm/tests/test_finalize_go2_n32_camera_frustum_observability.py",
        "label_semantics": ROOT / "lewm/datasets/go2_paired_navigation.py",
        "geometry_contract_semantics": ROOT / "lewm/planning/geometry_contract.py",
        "scene_manifest_semantics": ROOT / "lewm_worlds/lewm_worlds/manifest.py",
        "planning_grid_semantics": ROOT / "lewm_worlds/lewm_worlds/planning_grid.py",
    }


def _install_semantic_modules(
    core: Any,
    label_module: Any,
    manifest_module: Any,
    planning_module: Any,
) -> None:
    global BoxObject
    global InflatedOccupancyGrid
    global aggregate_label_observability
    global analyze_frame_labels
    global audit_camera_centered_mapping
    global authorization_decision
    global build_camera_centered_mapping
    global frozen_camera_geometry_contract
    global labels_v3
    global old_body_column_span_audit
    global parse_scene_manifest_dict
    global _SEMANTICS_LOADED

    if tuple(core.FAMILIES) != FAMILIES or tuple(core.ENDPOINT_SIDES) != ENDPOINT_SIDES:
        raise RuntimeError("authorized audit-core family/side constants changed")
    if (
        str(core.EXECUTION_BINDING_SHA256) != EXECUTION_BINDING_SHA256
        or str(core.RESULT_SCHEMA) != RESULT_SCHEMA
        or int(core.UNKNOWN_CLASS) != UNKNOWN_CLASS
        or int(core.FREE_CLASS) != FREE_CLASS
    ):
        raise RuntimeError("authorized audit-core frozen constants changed")
    labels_v3 = label_module
    BoxObject = manifest_module.BoxObject
    parse_scene_manifest_dict = manifest_module.parse_scene_manifest_dict
    InflatedOccupancyGrid = planning_module.InflatedOccupancyGrid
    aggregate_label_observability = core.aggregate_label_observability
    analyze_frame_labels = core.analyze_frame_labels
    audit_camera_centered_mapping = core.audit_camera_centered_mapping
    authorization_decision = core.authorization_decision
    build_camera_centered_mapping = core.build_camera_centered_mapping
    frozen_camera_geometry_contract = core.geometry_contract
    old_body_column_span_audit = core.old_body_column_span_audit
    _SEMANTICS_LOADED = True


def _load_authorized_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load authorized semantic module {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_namespace(name: str, path: Path) -> None:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]  # type: ignore[attr-defined]
    module.__package__ = name
    sys.modules[name] = module


def _load_authorized_semantics(
    source_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    """Load only the pre-hashed closed semantic graph, without package init files."""

    if _SEMANTICS_LOADED:
        raise RuntimeError("repository semantics were loaded before authorization")
    if any(
        name == "lewm"
        or name.startswith("lewm.")
        or name == "lewm_worlds"
        or name.startswith("lewm_worlds.")
        for name in sys.modules
    ):
        raise RuntimeError("repository semantic module was imported before authorization")
    _install_namespace("lewm", ROOT / "lewm")
    _install_namespace("lewm.benchmarks", ROOT / "lewm/benchmarks")
    _install_namespace("lewm.datasets", ROOT / "lewm/datasets")
    _install_namespace("lewm.planning", ROOT / "lewm/planning")
    _install_namespace("lewm_worlds", ROOT / "lewm_worlds/lewm_worlds")
    geometry = _load_authorized_module(
        "lewm.planning.geometry_contract",
        Path(source_hashes["geometry_contract_semantics"]["path"]),
    )
    manifest = _load_authorized_module(
        "lewm_worlds.manifest",
        Path(source_hashes["scene_manifest_semantics"]["path"]),
    )
    planning = _load_authorized_module(
        "lewm_worlds.planning_grid",
        Path(source_hashes["planning_grid_semantics"]["path"]),
    )
    labels = _load_authorized_module(
        "lewm.datasets.go2_paired_navigation",
        Path(source_hashes["label_semantics"]["path"]),
    )
    core = _load_authorized_module(
        "lewm.benchmarks.go2_n32_camera_frustum_observability",
        Path(source_hashes["audit_core"]["path"]),
    )
    del geometry
    _install_semantic_modules(core, labels, manifest, planning)


@dataclass(frozen=True)
class AuditSpec:
    """Frozen inputs; tests may supply a smaller synthetic equivalent."""

    panel_path: Path = PANEL_PATH
    panel_file_sha256: str = PANEL_FILE_SHA256
    panel_content_sha256: str = PANEL_CONTENT_SHA256
    fit_rows_sha256: str = FIT_ROWS_SHA256
    summary_root: Path = SUMMARY_ROOT
    summary_sha256: Mapping[str, str] | None = None
    output_path: Path = OUTPUT_PATH
    expected_transitions: int = EXPECTED_TRANSITIONS
    expected_frames: int = EXPECTED_FRAMES
    expected_shards: int = EXPECTED_SHARDS
    source_paths: Mapping[str, Path] | None = None

    def summaries(self) -> Mapping[str, str]:
        return EXPECTED_SUMMARY_SHA256 if self.summary_sha256 is None else self.summary_sha256

    def sources(self) -> Mapping[str, Path]:
        return _default_source_paths() if self.source_paths is None else self.source_paths


def new_access_ledger() -> dict[str, Any]:
    ledger: dict[str, Any] = {
        "panel_metadata_byte_opens": 0,
        "label_shard_hash_byte_opens": 0,
        "label_shard_npz_opens": 0,
        "registered_arrays_decompressed": 0,
        "materialized_label_rows": 0,
        "materialized_supervision_rows": 0,
        "per_shard_materialization": [],
        "selected_label_rows_read": 0,
        "selected_supervision_rows_read": 0,
        "unselected_row_values_inspected": 0,
        "unselected_row_metrics_computed": 0,
        "unselected_rows_retained": 0,
        "derivative_shard_or_cache_writes": 0,
        "source_geometry_hash_byte_opens": 0,
        "source_geometry_json_parses": 0,
        "source_geometry_jsonl_records": 0,
        "source_frame_records_selected": 0,
        "implementation_source_hash_byte_opens": 0,
        "document_hash_byte_opens": 0,
        "unexpected_path_attempts": 0,
        "denied_attempts_total": 0,
        "denied_primary_reasons": {name: 0 for name in PRIMARY_DENIAL_REASONS},
        "denied_modality_attempts": {name: 0 for name in DENIAL_MODALITIES},
        "denied_attempt_records": [],
    }
    ledger.update({name: 0 for name in FORBIDDEN_ACCESS_FIELDS})
    return ledger


def _is_sha256(value: object) -> bool:
    return bool(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _strict_json_int(value: object, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{label} is not a strict integer")
    return value


def _strict_json_number(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} is not a strict number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} is not finite")
    return result


def _canonical_json_bytes(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("value is not strict canonical JSON") from exc


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        return stream.read()


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(payload: bytes, *, name: str) -> dict[str, Any]:
    def object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{name} contains duplicate JSON key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ValueError(f"{name} contains forbidden JSON constant {value}")

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} is not strict UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    canonical_json_sha256(value)
    return value


def _infer_modality(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".md":
        return "markdown"
    if suffix == ".py":
        return "python_source"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".npz":
        return "npz"
    if suffix in FORBIDDEN_IMAGE_SUFFIXES:
        return "image"
    if suffix in FORBIDDEN_VIDEO_SUFFIXES:
        return "video"
    if suffix in {".npy", ".bin", ".raw"}:
        return "raster_array"
    if suffix in {".pcd", ".ply", ".las", ".laz"}:
        return "point_cloud"
    if suffix in {".pt", ".pth", ".ckpt", ".safetensors", ".onnx", ".pkl", ".pickle", ".joblib"}:
        return "model"
    if suffix in {".zip", ".tar", ".gz", ".bz2", ".xz", ".7z"}:
        return "archive"
    return "unknown"


FORBIDDEN_IMAGE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff", ".exr", ".hdr", ".gif"}
)
FORBIDDEN_VIDEO_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm"})


def _lexical_primary_denial(
    path: Path,
    *,
    requested_role: str,
    declared_role: str | None,
    modality: str,
) -> str | None:
    text = str(path).replace("\\", "/").lower()
    tokens = tuple(token for token in re.split(r"[^a-z0-9]+", text) if token)
    token_set = set(tokens)
    requested_tokens = set(
        token
        for token in re.split(r"[^a-z0-9]+", str(requested_role).lower())
        if token
    )
    declared_text = "" if declared_role is None else str(declared_role).lower()
    declared_tokens = set(
        token for token in re.split(r"[^a-z0-9]+", declared_text) if token
    )
    role_tokens = requested_tokens | declared_tokens
    if (
        "sealed" in token_set
        or "sealed" in role_tokens
        or any(token.startswith("sealed") for token in tokens)
    ):
        return "sealed"
    if "g2" in token_set or "g2" in role_tokens or "g2payload" in token_set:
        return "g2"
    if (
        "seed20260711" in token_set
        or "seed20260711" in role_tokens
        or ("seed" in token_set and "20260711" in token_set)
        or ("seed" in role_tokens and "20260711" in role_tokens)
    ):
        return "seed_20260711"
    if (
        requested_role == "generated_v4_result"
        or (
            requested_role != "v4_adjudication_report"
            and "generated" in token_set
            and "v4" in token_set
            and "result" in token_set
        )
        or "generatedv4result" in token_set
        or "generatedv4result" in role_tokens
        or (
            "result" in token_set
            and (
                "seed20260710" in token_set
                or ("seed" in token_set and "20260710" in token_set)
            )
        )
    ):
        return "generated_v4_result"
    model_tokens = {
        "model",
        "models",
        "checkpoint",
        "checkpoints",
        "activation",
        "logit",
        "probability",
        "prediction",
        "parameter",
        "parameters",
    }
    if modality == "model" or bool(role_tokens & model_tokens) or bool(
        token_set & model_tokens
    ):
        return "model"
    if (
        token_set & {"runtime", "development", "closedloop"}
        or role_tokens & {"runtime", "development", "closedloop"}
        or ("closed" in token_set and "loop" in token_set)
        or ("closed" in role_tokens and "loop" in role_tokens)
    ):
        return "runtime"
    if "nontrain" in token_set or "nontrain" in role_tokens or bool(
        declared_tokens & {"validation", "test", "testeasy", "testhard"}
    ):
        return "physical_nontrain"
    if bool(token_set & {"calibration", "calib"}) or bool(
        role_tokens & {"calibration", "calib"}
    ):
        return "selection_or_calibration"
    if (
        ("selection" in token_set or "selection" in role_tokens)
        and requested_role != "fit_frame_selection"
    ):
        return "selection_or_calibration"
    if bool(token_set & {"holdout", "heldout"}) or bool(
        role_tokens & {"holdout", "heldout"}
    ):
        return "holdout"
    image_tokens = {"rgb", "image", "images", "depth", "pixels", "pointcloud"}
    if modality in {"image", "video", "raster_array", "point_cloud"} or bool(
        token_set & image_tokens
    ) or bool(role_tokens & image_tokens):
        return "image_or_depth"
    train_data_roles = {
        "fit_panel",
        "fit_label_shard",
        "fit_render_summary",
        "render_audit_contract",
        "fit_frame_selection",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
        "physical_geometry_contract",
    }
    declared_role_allowed = (
        declared_role is None
        or declared_role == requested_role
        or (requested_role in train_data_roles and declared_role == "train")
    )
    if requested_role not in ALLOWED_PATH_ROLES or not declared_role_allowed:
        return "unregistered_role"
    expected_modality = ALLOWED_MODALITY_BY_ROLE[requested_role]
    if expected_modality is None:
        if modality != "python_source":
            return "forbidden_modality"
    elif modality != expected_modality:
        return "forbidden_modality"
    return None


def _record_path_denial(
    ledger: dict[str, Any],
    *,
    path: Path,
    requested_role: str,
    declared_role: str | None,
    modality: str,
    primary_reason: str,
    resolved_path: Path | None,
) -> None:
    ledger["denied_attempts_total"] += 1
    ledger["unexpected_path_attempts"] += 1
    ledger["denied_primary_reasons"][primary_reason] += 1
    ledger["denied_modality_attempts"][
        modality if modality in DENIAL_MODALITIES else "unknown"
    ] += 1
    ledger["denied_attempt_records"].append(
        {
            "lexical_path": str(path),
            "resolved_path": None if resolved_path is None else str(resolved_path),
            "requested_role": requested_role,
            "declared_role": declared_role,
            "modality": modality,
            "primary_reason": primary_reason,
        }
    )


def _authorize_path(
    path: Path,
    root: Path,
    *,
    ledger: dict[str, Any],
    requested_role: str,
    declared_role: str | None = None,
    expected_resolved_path: Path | None = None,
    label: str,
) -> Path:
    """Authorize semantics and modality before resolving or touching a path."""

    lexical = Path(path)
    modality = _infer_modality(lexical)
    denial = _lexical_primary_denial(
        lexical,
        requested_role=requested_role,
        declared_role=declared_role,
        modality=modality,
    )
    if denial is not None:
        _record_path_denial(
            ledger,
            path=lexical,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            primary_reason=denial,
            resolved_path=None,
        )
        raise PermissionError(f"{label} denied as {denial}")

    lexical_absolute = Path(os.path.abspath(os.fspath(lexical)))
    root_absolute = Path(os.path.abspath(os.fspath(root)))
    try:
        lexical_relative = lexical_absolute.relative_to(root_absolute)
    except ValueError as exc:
        _record_path_denial(
            ledger,
            path=lexical,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            primary_reason="path_alias_or_escape",
            resolved_path=None,
        )
        raise PermissionError(f"{label} escapes the repository access root") from exc

    component = root_absolute
    symlink_component = component.is_symlink()
    for part in lexical_relative.parts:
        component = component / part
        symlink_component = symlink_component or component.is_symlink()
    if symlink_component:
        _record_path_denial(
            ledger,
            path=lexical,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            primary_reason="path_alias_or_escape",
            resolved_path=None,
        )
        raise PermissionError(
            f"{label} denied as path_alias_or_escape (symlink component)"
        )

    resolved = lexical_absolute.resolve(strict=False)
    root_resolved = root_absolute.resolve(strict=True)
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        _record_path_denial(
            ledger,
            path=lexical,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            primary_reason="path_alias_or_escape",
            resolved_path=resolved,
        )
        raise PermissionError(f"{label} escapes the repository access root") from exc
    if expected_resolved_path is not None and resolved != expected_resolved_path.resolve(
        strict=False
    ):
        _record_path_denial(
            ledger,
            path=lexical,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=modality,
            primary_reason="unallowlisted",
            resolved_path=resolved,
        )
        raise PermissionError(f"{label} denied as unallowlisted")
    return resolved


def _frame_key(record: Mapping[str, Any]) -> dict[str, Any]:
    return {name: record[name] for name in FRAME_IDENTITY_FIELDS}


def _frame_identity_values(record: Mapping[str, Any]) -> list[Any]:
    return [record[name] for name in FRAME_IDENTITY_FIELDS]


def _canonical_frame_sort_key(record: Mapping[str, Any]) -> tuple[int, str, int, int]:
    family_rank = {family: index for index, family in enumerate(FAMILIES)}
    side_rank = {side: index for index, side in enumerate(ENDPOINT_SIDES)}
    return (
        family_rank[str(record["family"])],
        str(record["scene_id"]),
        int(record["global_row"]),
        side_rank[str(record["side"])],
    )


def _canonicalize_fit_panel(
    panel: Mapping[str, Any],
    *,
    spec: AuditSpec,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if panel.get("schema") != "lewm_go2_physical_micro_overfit_panel_v1":
        raise ValueError("fit panel schema changed")
    core = dict(panel)
    declared = str(core.pop("content_sha256", ""))
    if declared != spec.panel_content_sha256 or canonical_json_sha256(core) != declared:
        raise ValueError("fit panel canonical content hash mismatch")
    if tuple(map(str, panel.get("families", ()))) != FAMILIES:
        raise ValueError("fit panel family order changed")
    if _strict_json_int(
        panel.get("rows_per_family_panel"), label="fit panel rows_per_family_panel"
    ) * len(FAMILIES) != spec.expected_transitions:
        raise ValueError("fit panel transition budget changed")
    panels = panel.get("panels")
    if not isinstance(panels, Mapping) or not isinstance(panels.get("fit"), Mapping):
        raise ValueError("fit panel lacks its fit partition")
    fit = panels["fit"]
    rows = fit.get("rows")
    if not isinstance(rows, list) or len(rows) != spec.expected_transitions:
        raise ValueError("fit panel transition count changed")
    if _strict_json_int(fit.get("row_count"), label="fit panel row_count") != spec.expected_transitions:
        raise ValueError("fit panel declared row count changed")
    if _strict_json_int(fit.get("frame_count"), label="fit panel frame_count") != spec.expected_frames:
        raise ValueError("fit panel declared frame count changed")
    if str(fit.get("rows_sha256", "")) != spec.fit_rows_sha256:
        raise ValueError("fit rows declared hash changed")
    if canonical_json_sha256(rows) != spec.fit_rows_sha256:
        raise ValueError("fit rows canonical hash changed")

    family_counts: Counter[str] = Counter()
    records: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("fit row is not an object")
        family = str(row.get("family", ""))
        if family not in FAMILIES:
            raise ValueError("fit row has an unregistered family")
        if str(row.get("dataset_role", "")) != "train":
            raise PermissionError("fit row is not current physical train role")
        family_counts[family] += 1
        shard_hash = str(row.get("label_shard_sha256", ""))
        if not _is_sha256(shard_hash):
            raise ValueError("fit row label-shard hash is malformed")
        label_row = _strict_json_int(
            row.get("label_shard_row"), label="fit row label-shard row"
        )
        for side in ENDPOINT_SIDES:
            image_hash = str(row.get(f"{side}_image_sha256", ""))
            if not _is_sha256(image_hash):
                raise ValueError("fit row image commitment is malformed")
            record = {
                "family": family,
                "scene_id": str(row["scene_id"]),
                "global_row": _strict_json_int(row.get("global_row"), label="fit row global_row"),
                "side": side,
                "image_path_metadata_only": str(row[f"{side}_image_path"]),
                "image_sha256": image_hash,
                "label_shard_path": str(row["label_shard_path"]),
                "label_shard_sha256": shard_hash,
                "label_row": label_row,
                "frame_index": _strict_json_int(
                    row.get(f"{side}_frame_index"), label=f"fit row {side}_frame_index"
                ),
                "env_index": _strict_json_int(row.get("env_index"), label="fit row env_index"),
                "timestamp_ns": _strict_json_int(
                    row.get(f"{side}_timestamp_ns"), label=f"fit row {side}_timestamp_ns"
                ),
                "episode_id": str(row["episode_id"]),
                "reset_count": _strict_json_int(
                    row.get("reset_count"), label="fit row reset_count"
                ),
                "episode_step": _strict_json_int(
                    row.get(f"{side}_episode_step"), label=f"fit row {side}_episode_step"
                ),
            }
            records.append(record)
    expected_per_family = spec.expected_transitions // len(FAMILIES)
    if family_counts != Counter({family: expected_per_family for family in FAMILIES}):
        raise ValueError("fit transitions are not family balanced")
    records.sort(key=_canonical_frame_sort_key)
    canonical_coordinates = [_canonical_frame_sort_key(record) for record in records]
    if len(set(canonical_coordinates)) != len(canonical_coordinates):
        raise ValueError("fit panel repeats a canonical frame coordinate")
    identities = [tuple(_frame_identity_values(record)) for record in records]
    if len(records) != spec.expected_frames or len(set(identities)) != spec.expected_frames:
        raise ValueError("fit frame identity set is not exactly unique")
    if len({record["image_sha256"] for record in records}) != spec.expected_frames:
        raise ValueError("fit endpoint image commitments are not unique")

    local = panel.get("local_grid")
    if not isinstance(local, Mapping):
        raise ValueError("fit panel lacks local-grid geometry")
    if (
        list(local.get("shape", ())) != [64, 64]
        or _strict_json_number(local.get("cell_size_m"), label="fit panel cell_size_m") != 0.10
        or list(local.get("forward_edge_range_m", ())) != [-1.0, 5.4]
        or list(local.get("left_edge_range_m", ())) != [-3.2, 3.2]
    ):
        raise ValueError("fit panel local-grid geometry changed")
    projection = panel.get("source_camera_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("fit panel lacks camera projection")
    if (
        not math.isclose(
            _strict_json_number(
                projection.get("horizontal_fov_deg"), label="fit panel horizontal_fov_deg"
            ),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or not math.isclose(
            _strict_json_number(projection.get("near_m"), label="fit panel near_m"),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError("fit panel camera calibration changed")
    inputs = panel.get("inputs")
    geometry = inputs.get("geometry_contract") if isinstance(inputs, Mapping) else None
    render_audit = inputs.get("render_audit_contract") if isinstance(inputs, Mapping) else None
    if not isinstance(geometry, Mapping) or not isinstance(render_audit, Mapping):
        raise ValueError("fit panel lacks committed geometry/render-audit contracts")
    return records, {
        "local_grid": dict(local),
        "camera_projection": dict(projection),
        "geometry_contract": {
            "path": str(geometry.get("path", "")),
            "file_sha256": str(geometry.get("file_sha256", "")),
            "semantic_sha256": str(geometry.get("semantic_sha256", "")),
        },
        "render_audit_contract": {
            "path": str(render_audit.get("path", "")),
            "file_sha256": str(render_audit.get("file_sha256", "")),
            "content_sha256": str(render_audit.get("content_sha256", "")),
        },
    }


def _load_panel(spec: AuditSpec, ledger: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = _authorize_path(
        spec.panel_path,
        ROOT,
        ledger=ledger,
        requested_role="fit_panel",
        declared_role="train",
        expected_resolved_path=(
            PANEL_PATH if spec.panel_path == PANEL_PATH else spec.panel_path
        ),
        label="fit panel",
    )
    ledger["panel_metadata_byte_opens"] += 1
    payload = _read_bytes(path)
    if _sha256_bytes(payload) != spec.panel_file_sha256:
        raise ValueError("fit panel file SHA-256 changed")
    return _canonicalize_fit_panel(_strict_json_bytes(payload, name="fit panel"), spec=spec)


def _label_shard_manifest(
    records: Sequence[Mapping[str, Any]], *, spec: AuditSpec, ledger: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[Path, list[Mapping[str, Any]]]]:
    grouped: dict[Path, list[Mapping[str, Any]]] = {}
    commitments: dict[Path, str] = {}
    for record in records:
        lexical = Path(str(record["label_shard_path"]))
        path = _authorize_path(
            lexical,
            ROOT,
            ledger=ledger,
            requested_role="fit_label_shard",
            declared_role="train",
            expected_resolved_path=lexical,
            label="fit label shard",
        )
        if path.suffix.lower() != ".npz":
            raise PermissionError("fit label shard is not NPZ")
        digest = str(record["label_shard_sha256"])
        previous = commitments.setdefault(path, digest)
        if previous != digest:
            raise ValueError("one fit label shard has conflicting commitments")
        grouped.setdefault(path, []).append(record)
    if len(grouped) != spec.expected_shards:
        raise ValueError("distinct fit label-shard count changed")
    selected_storage_rows: set[tuple[Path, int, str]] = set()
    entries = []
    for path in sorted(grouped, key=str):
        selected_records = list(grouped[path])
        for record in selected_records:
            storage_identity = (
                path,
                int(record["label_row"]),
                str(record["side"]),
            )
            if storage_identity in selected_storage_rows:
                raise ValueError("fit panel selects one shard row/side more than once")
            selected_storage_rows.add(storage_identity)
        family_side_counts = {
            family: {
                side: sum(
                    str(record["family"]) == family and str(record["side"]) == side
                    for record in selected_records
                )
                for side in ENDPOINT_SIDES
            }
            for family in FAMILIES
        }
        entries.append(
            {
                "path": str(path),
                "sha256": commitments[path],
                "selected_tuples": [
                    [
                        str(record["family"]),
                        str(record["scene_id"]),
                        int(record["global_row"]),
                        str(record["side"]),
                        int(record["label_row"]),
                    ]
                    for record in selected_records
                ],
                "selected_row_count": len(selected_records),
                "family_side_counts": family_side_counts,
            }
        )
    if len(selected_storage_rows) != spec.expected_frames:
        raise ValueError("fit label-shard selected rows do not reconcile to fit frames")
    return entries, grouped


def _validate_npz_archive_inventory(raw: bytes, *, label: str) -> None:
    expected_names = {f"{name}.npy" for name in REGISTERED_SHARD_ARRAY_NAMES}
    try:
        archive = zipfile.ZipFile(io.BytesIO(raw), mode="r")
    except (zipfile.BadZipFile, OSError) as exc:
        raise ValueError(f"{label} is not a valid NPZ container") from exc
    with archive:
        infos = archive.infolist()
        names = [str(info.filename) for info in infos]
        if len(names) != len(set(names)):
            raise ValueError(f"{label} contains duplicate archive member names")
        if set(names) != expected_names:
            raise ValueError(f"{label} archive inventory differs from the frozen 12 names")
        for info in infos:
            if (
                info.is_dir()
                or info.filename != Path(info.filename).name
                or info.filename.startswith(("/", "\\"))
                or ".." in Path(info.filename).parts
            ):
                raise ValueError(f"{label} contains a path-qualified archive member")
            offset = int(info.header_offset)
            if offset < 0 or offset + 30 > len(raw) or raw[offset : offset + 4] != b"PK\x03\x04":
                raise ValueError(f"{label}:{info.filename} has an invalid local ZIP header")
            local_flags = int(struct.unpack_from("<H", raw, offset + 6)[0])
            local_name_length = int(struct.unpack_from("<H", raw, offset + 26)[0])
            local_extra_length = int(struct.unpack_from("<H", raw, offset + 28)[0])
            name_start = offset + 30
            name_end = name_start + local_name_length
            if name_end + local_extra_length > len(raw):
                raise ValueError(f"{label}:{info.filename} has a truncated local ZIP header")
            encoding = "utf-8" if local_flags & 0x800 else "cp437"
            try:
                local_name = raw[name_start:name_end].decode(encoding)
            except UnicodeDecodeError as exc:
                raise ValueError(f"{label}:{info.filename} has an invalid local ZIP name") from exc
            if local_name != info.filename:
                raise ValueError(f"{label}:{info.filename} central/local names differ")
            if bool(local_flags & 0x1) != bool(info.flag_bits & 0x1) or local_flags & 0x1:
                raise ValueError(f"{label}:{info.filename} is encrypted")


def _read_selected_labels_once(
    grouped: Mapping[Path, Sequence[Mapping[str, Any]]],
    *,
    ledger: dict[str, Any],
) -> dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]]:
    selected: dict[tuple[Any, ...], tuple[np.ndarray, np.ndarray]] = {}
    for path in sorted(grouped, key=str):
        path = _authorize_path(
            path,
            ROOT,
            ledger=ledger,
            requested_role="fit_label_shard",
            declared_role="train",
            expected_resolved_path=path,
            label="fit label shard byte open",
        )
        records = grouped[path]
        expected_hashes = {str(record["label_shard_sha256"]) for record in records}
        if len(expected_hashes) != 1:
            raise ValueError("one label shard has conflicting selected commitments")
        ledger["label_shard_hash_byte_opens"] += 1
        raw = _read_bytes(path)
        if _sha256_bytes(raw) != next(iter(expected_hashes)):
            raise ValueError("fit label shard SHA-256 changed")
        ledger["label_shard_npz_opens"] += 1
        _validate_npz_archive_inventory(raw, label="fit label shard")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            archive_names = list(map(str, archive.files))
            if len(archive_names) != len(set(archive_names)):
                raise ValueError("fit label shard contains duplicate array names")
            if set(archive_names) != REGISTERED_SHARD_ARRAY_NAMES:
                raise ValueError("fit label shard inventory differs from the registered arrays")
            arrays = {
                name: np.asarray(archive[name])
                for name in REGISTERED_LABEL_ARRAY_NAMES
            }
            ledger["registered_arrays_decompressed"] += len(arrays)
        row_counts = {name: int(array.shape[0]) if array.ndim else -1 for name, array in arrays.items()}
        if len(set(row_counts.values())) != 1 or next(iter(row_counts.values())) <= 0:
            raise ValueError("fit label shard arrays have inconsistent row counts")
        storage_rows = next(iter(row_counts.values()))
        for side in ENDPOINT_SIDES:
            labels = arrays[f"{side}_labels"]
            masks = arrays[f"{side}_supervision_mask"]
            if labels.shape != (storage_rows, 64, 64) or labels.dtype != np.dtype(np.uint8):
                raise ValueError("fit label shard labels have an unregistered dtype or shape")
            if masks.shape != (storage_rows, 64, 64) or masks.dtype != np.dtype(bool):
                raise ValueError("fit label shard supervision has an unregistered dtype or shape")
        ledger["materialized_label_rows"] += 2 * storage_rows
        ledger["materialized_supervision_rows"] += 2 * storage_rows
        ledger["per_shard_materialization"].append(
            {
                "path": str(path),
                "storage_rows_per_array": storage_rows,
                "materialized_label_rows": 2 * storage_rows,
                "materialized_supervision_rows": 2 * storage_rows,
                "selected_endpoint_rows": len(records),
            }
        )
        for record in records:
            side = str(record["side"])
            row = int(record["label_row"])
            labels = arrays[f"{side}_labels"]
            masks = arrays[f"{side}_supervision_mask"]
            if row >= labels.shape[0] or row >= masks.shape[0]:
                raise ValueError("selected fit label row is outside its shard")
            target = np.asarray(labels[row])
            mask = np.asarray(masks[row])
            if target.shape != (64, 64) or mask.shape != (64, 64):
                raise ValueError("selected fit target shape changed")
            if not np.issubdtype(target.dtype, np.integer) or not np.isin(target, (0, 1, 2)).all():
                raise ValueError("selected fit target class values changed")
            if not np.all(mask == 1):
                raise ValueError("selected fit supervision is not the full finite grid")
            identity = tuple(_frame_identity_values(record))
            if identity in selected:
                raise ValueError("selected fit label identity repeated")
            selected[identity] = (
                np.array(target, dtype=np.uint8, order="C", copy=True),
                np.array(mask, dtype=bool, order="C", copy=True),
            )
            ledger["selected_label_rows_read"] += 1
            ledger["selected_supervision_rows_read"] += 1
        # Full shard arrays are a storage-boundary necessity only. Release
        # them before the next shard can be opened; selected rows above are
        # independent contiguous copies.
        del target, mask, labels, masks, arrays, archive
    return selected


def _add_allowlist_entry(
    allowlist: dict[Path, str],
    path: Path,
    expected_sha256: str,
    *,
    ledger: dict[str, Any],
    label: str,
    requested_role: str,
    declared_role: str | None = "train",
) -> Path:
    if not _is_sha256(expected_sha256):
        raise ValueError(f"{label} has a malformed committed SHA-256")
    resolved = _authorize_path(
        path,
        ROOT,
        ledger=ledger,
        requested_role=requested_role,
        declared_role=declared_role,
        expected_resolved_path=path,
        label=label,
    )
    previous = allowlist.setdefault(resolved, str(expected_sha256))
    if previous != str(expected_sha256):
        raise ValueError(f"{label} has conflicting commitments")
    return resolved


def _require_allowlisted(
    path: Path,
    expected_sha256: str,
    *,
    allowlist: Mapping[Path, str],
    ledger: dict[str, Any],
    label: str,
    requested_role: str,
    declared_role: str | None = "train",
) -> Path:
    resolved = _authorize_path(
        path,
        ROOT,
        ledger=ledger,
        requested_role=requested_role,
        declared_role=declared_role,
        label=label,
    )
    if allowlist.get(resolved) != str(expected_sha256):
        _record_path_denial(
            ledger,
            path=path,
            requested_role=requested_role,
            declared_role=declared_role,
            modality=_infer_modality(path),
            primary_reason="unallowlisted",
            resolved_path=resolved,
        )
        raise PermissionError(f"{label} was not allowlisted before byte access")
    return resolved


def _read_allowlisted_json(
    path: Path,
    expected_sha256: str,
    *,
    allowlist: Mapping[Path, str],
    ledger: dict[str, Any],
    label: str,
    requested_role: str,
) -> dict[str, Any]:
    resolved = _require_allowlisted(
        path,
        expected_sha256,
        allowlist=allowlist,
        ledger=ledger,
        label=label,
        requested_role=requested_role,
    )
    # Hash opens and the intervening JSON parse open are deliberately separate
    # ledger dimensions. This records two integrity passes around one parse.
    ledger["source_geometry_hash_byte_opens"] += 1
    before = _hash_file(resolved)
    if before != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 changed before parse")
    ledger["source_geometry_json_parses"] += 1
    value = _strict_json_bytes(_read_bytes(resolved), name=label)
    ledger["source_geometry_hash_byte_opens"] += 1
    after = _hash_file(resolved)
    if after != before:
        raise ValueError(f"{label} changed while it was parsed")
    return value


def _verify_allowlisted_file(
    path: Path,
    expected_sha256: str,
    *,
    allowlist: Mapping[Path, str],
    ledger: dict[str, Any],
    label: str,
    requested_role: str,
) -> None:
    resolved = _require_allowlisted(
        path,
        expected_sha256,
        allowlist=allowlist,
        ledger=ledger,
        label=label,
        requested_role=requested_role,
    )
    ledger["source_geometry_hash_byte_opens"] += 1
    before = _hash_file(resolved)
    if before != str(expected_sha256):
        raise ValueError(f"{label} SHA-256 changed")
    ledger["source_geometry_hash_byte_opens"] += 1
    if _hash_file(resolved) != before:
        raise ValueError(f"{label} changed during verification")


def _summary_path_for_record(
    record: Mapping[str, Any],
    *,
    spec: AuditSpec,
    ledger: dict[str, Any],
) -> Path:
    image = Path(str(record["image_path_metadata_only"]))
    if image.parent.name != "rgb":
        raise PermissionError("committed image path is not under a render rgb directory")
    summary = image.parent.parent / "summary.json"
    return _authorize_path(
        summary,
        ROOT,
        ledger=ledger,
        requested_role="fit_render_summary",
        declared_role="train",
        label="committed fit render summary",
    )


def _geometry_semantic_sha256(payload: Mapping[str, Any]) -> str:
    return canonical_json_sha256(payload)


def _geometry_flags(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "lewm_go2_generalization_geometry_v2":
        raise ValueError("physical geometry contract schema changed")
    camera = payload.get("camera")
    configuration = payload.get("configuration_space")
    if not isinstance(camera, Mapping) or not isinstance(configuration, Mapping):
        raise ValueError("geometry contract lacks camera/configuration geometry")
    result = {
        "oracle_cell_size_m": _strict_json_number(
            configuration.get("oracle_cell_size_m"),
            label="geometry oracle_cell_size_m",
        ),
        "landmarks_are_obstacles": configuration.get("landmarks_are_obstacles"),
        "distractors_are_obstacles": configuration.get("distractors_are_obstacles"),
        "horizontal_fov_deg": _strict_json_number(
            camera.get("horizontal_fov_deg"), label="geometry horizontal_fov_deg"
        ),
        "near_m": _strict_json_number(camera.get("near_m"), label="geometry near_m"),
    }
    if not math.isclose(result["oracle_cell_size_m"], 0.05, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("geometry contract physical cell size changed")
    if not isinstance(result["landmarks_are_obstacles"], bool) or not isinstance(
        result["distractors_are_obstacles"], bool
    ):
        raise ValueError("geometry obstacle-role flags are not booleans")
    if (
        not math.isclose(result["horizontal_fov_deg"], 78.323, rel_tol=0.0, abs_tol=1e-12)
        or not math.isclose(result["near_m"], 0.05, rel_tol=0.0, abs_tol=1e-12)
    ):
        raise ValueError("geometry camera calibration changed")
    return result


def _validate_embedded_content_hash(
    payload: Mapping[str, Any],
    *,
    expected_sha256: str,
    label: str,
) -> None:
    core = dict(payload)
    declared = str(core.pop("content_sha256", ""))
    if (
        not _is_sha256(expected_sha256)
        or declared != expected_sha256
        or canonical_json_sha256(core) != expected_sha256
    ):
        raise ValueError(f"{label} canonical content SHA-256 changed")


def _validate_render_audit_contract(
    payload: Mapping[str, Any],
    *,
    expected_content_sha256: str,
) -> None:
    _validate_embedded_content_hash(
        payload,
        expected_sha256=expected_content_sha256,
        label="render audit contract",
    )
    camera = payload.get("camera_projection")
    objects = payload.get("object_contract")
    if payload.get("schema") != "lewm_go2_selected_render_audit_v1":
        raise ValueError("render audit contract schema changed")
    if (
        not isinstance(camera, Mapping)
        or list(camera.get("resolution_wh", ())) != [224, 168]
        or not math.isclose(
            _strict_json_number(
                camera.get("horizontal_fov_deg"), label="render audit horizontal_fov_deg"
            ),
            78.323,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _strict_json_number(
                camera.get("vertical_fov_deg"), label="render audit vertical_fov_deg"
            ),
            62.837038636424516,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or not math.isclose(
            _strict_json_number(camera.get("near_m"), label="render audit near_m"),
            0.05,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or camera.get("runtime_rectification_required") is not False
    ):
        raise ValueError("render audit camera projection contract changed")
    if (
        not isinstance(objects, Mapping)
        or objects.get("rendered_groups")
        != ["wall", "obstacle", "landmark", "distractor"]
        or objects.get("collision_distractors_rendered") is not True
        or objects.get("full_box_roll_pitch_yaw_rendered") is not True
    ):
        raise ValueError("render audit object contract changed")
    contact = {
        "g2_row_metadata_read": True,
        "g2_image_bytes_hashed_for_integrity": True,
        "g2_images_decoded_or_inspected": False,
        "g2_image_content_metrics_computed": False,
        "g2_label_shards_opened": False,
        "g2_model_outputs_opened": False,
    }
    if any(payload.get(name) is not expected for name, expected in contact.items()):
        raise ValueError("render audit contact contract changed")


def _source_record_identity(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (
        int(record["frame_index"]),
        int(record["env_index"]),
        int(record["timestamp_ns"]),
    )


def _finite_vector(value: object, *, length: int, label: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        raise ValueError(f"{label} must contain exactly {length} values")
    result = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"{label} contains a non-numeric value")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError(f"{label} contains a nonfinite value")
        result.append(number)
    return result


def _camera_mount_record(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "parent_link",
        "rpy_body_rad",
        "xyz_body_m",
    }:
        raise ValueError(f"{label} fields changed")
    parent_link = value.get("parent_link")
    if not isinstance(parent_link, str) or not parent_link:
        raise ValueError(f"{label}.parent_link is malformed")
    return {
        "parent_link": parent_link,
        "rpy_body_rad": _finite_vector(
            value.get("rpy_body_rad"), length=3, label=f"{label}.rpy_body_rad"
        ),
        "xyz_body_m": _finite_vector(
            value.get("xyz_body_m"), length=3, label=f"{label}.xyz_body_m"
        ),
    }


def _camera_mount_composition_evidence(
    *,
    base_position_world: Sequence[float],
    base_quat_world_xyzw: Sequence[float],
    stored_base_yaw_rad: float,
    plan_camera_mount_body: Mapping[str, Any],
    frame_camera_mount_body: Mapping[str, Any],
    recorded_camera_pose_world: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    base_position = _finite_vector(
        base_position_world, length=3, label="base position world"
    )
    quaternion = _finite_vector(
        base_quat_world_xyzw, length=4, label="base_quat_world_xyzw"
    )
    if isinstance(stored_base_yaw_rad, bool) or not isinstance(
        stored_base_yaw_rad, (int, float)
    ):
        raise ValueError("stored base yaw is not numeric")
    stored_yaw = float(stored_base_yaw_rad)
    if not math.isfinite(stored_yaw):
        raise ValueError("stored base yaw is nonfinite")
    plan_mount = _camera_mount_record(
        plan_camera_mount_body, label="plan camera.mount_body"
    )
    frame_mount = _camera_mount_record(
        frame_camera_mount_body, label="frame camera_mount_body"
    )
    if not isinstance(recorded_camera_pose_world, Mapping) or set(
        recorded_camera_pose_world
    ) != {"position", "lookat", "up"}:
        raise ValueError("recorded camera pose fields changed")
    recorded_position = _finite_vector(
        recorded_camera_pose_world.get("position"),
        length=3,
        label="recorded camera position",
    )
    recorded_lookat = _finite_vector(
        recorded_camera_pose_world.get("lookat"),
        length=3,
        label="recorded camera lookat",
    )
    recorded_up = _finite_vector(
        recorded_camera_pose_world.get("up"),
        length=3,
        label="recorded camera up",
    )

    qx, qy, qz, qw = quaternion
    quaternion_norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    quaternion_norm_abs_residual = abs(quaternion_norm - 1.0)
    rotation = (
        (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy - qz * qw),
            2.0 * (qx * qz + qy * qw),
        ),
        (
            2.0 * (qx * qy + qz * qw),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz - qx * qw),
        ),
        (
            2.0 * (qx * qz - qy * qw),
            2.0 * (qy * qz + qx * qw),
            1.0 - 2.0 * (qx * qx + qy * qy),
        ),
    )
    nominal_xyz = NOMINAL_CAMERA_MOUNT_BODY["xyz_body_m"]
    expected_position = [
        base_position[row]
        + sum(rotation[row][column] * float(nominal_xyz[column]) for column in range(3))
        for row in range(3)
    ]
    expected_forward = [rotation[row][0] for row in range(3)]
    expected_up = [rotation[row][2] for row in range(3)]
    expected_lookat = [
        expected_position[index] + expected_forward[index] for index in range(3)
    ]
    recorded_forward = [
        recorded_lookat[index] - recorded_position[index] for index in range(3)
    ]
    look_distance = math.sqrt(sum(value * value for value in recorded_forward))
    recorded_up_norm = math.sqrt(sum(value * value for value in recorded_up))
    expected_forward_norm = math.sqrt(sum(value * value for value in expected_forward))
    expected_up_norm = math.sqrt(sum(value * value for value in expected_up))
    if min(look_distance, recorded_up_norm, expected_forward_norm, expected_up_norm) <= 0.0:
        raise ValueError("camera pose contains a zero-length direction vector")

    def angular_error(left: Sequence[float], right: Sequence[float]) -> float:
        left_norm = math.sqrt(sum(value * value for value in left))
        right_norm = math.sqrt(sum(value * value for value in right))
        dot = sum(
            left[index] * right[index] for index in range(3)
        ) / (left_norm * right_norm)
        return math.acos(max(-1.0, min(1.0, dot)))

    quaternion_yaw = math.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )
    wrapped_yaw_residual = abs(
        math.atan2(
            math.sin(stored_yaw - quaternion_yaw),
            math.cos(stored_yaw - quaternion_yaw),
        )
    )
    position_residual = max(
        abs(recorded_position[index] - expected_position[index]) for index in range(3)
    )
    lookat_residual = max(
        abs(recorded_lookat[index] - expected_lookat[index]) for index in range(3)
    )
    up_residual = max(
        abs(recorded_up[index] - expected_up[index]) for index in range(3)
    )
    look_distance_residual = abs(look_distance - 1.0)
    forward_angular_error = angular_error(recorded_forward, expected_forward)
    up_angular_error = angular_error(recorded_up, expected_up)
    tolerance = CAMERA_COMPOSITION_TOLERANCE
    passes = bool(
        plan_mount == NOMINAL_CAMERA_MOUNT_BODY
        and frame_mount == NOMINAL_CAMERA_MOUNT_BODY
        and quaternion_norm_abs_residual <= tolerance
        and wrapped_yaw_residual <= tolerance
        and position_residual <= tolerance
        and lookat_residual <= tolerance
        and up_residual <= tolerance
        and look_distance_residual <= tolerance
        and forward_angular_error <= tolerance
        and up_angular_error <= tolerance
    )
    return {
        "base_position_world": base_position,
        "base_quat_world_xyzw": quaternion,
        "stored_base_yaw_rad": stored_yaw,
        "plan_camera_mount_body": plan_mount,
        "frame_camera_mount_body": frame_mount,
        "recorded_camera_pose_world": {
            "position": recorded_position,
            "lookat": recorded_lookat,
            "up": recorded_up,
        },
        "expected_camera_pose_world": {
            "position": expected_position,
            "lookat": expected_lookat,
            "up": expected_up,
        },
        "quaternion_norm": quaternion_norm,
        "quaternion_norm_abs_residual": quaternion_norm_abs_residual,
        "quaternion_yaw_rad": quaternion_yaw,
        "wrapped_yaw_abs_residual_rad": wrapped_yaw_residual,
        "position_max_abs_residual_m": position_residual,
        "lookat_max_abs_residual_m": lookat_residual,
        "up_max_abs_residual": up_residual,
        "look_distance_m": look_distance,
        "look_distance_abs_residual_m": look_distance_residual,
        "forward_angular_error_rad": forward_angular_error,
        "up_angular_error_rad": up_angular_error,
        "passes": passes,
    }


def _extract_source_frame(
    frame: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    plan_camera_mount_body: Mapping[str, Any],
) -> dict[str, Any]:
    frame_index = _strict_json_int(
        frame.get("frame_index"), label="source frame frame_index"
    )
    env_index = _strict_json_int(frame.get("env_index"), label="source frame env_index")
    timestamp_ns = _strict_json_int(
        frame.get("timestamp_ns"), label="source frame timestamp_ns"
    )
    if (
        frame_index != int(record["frame_index"])
        or env_index != int(record["env_index"])
        or timestamp_ns != int(record["timestamp_ns"])
    ):
        raise ValueError("source-frame key disagrees with fit record")
    episode = frame.get("episode")
    if not isinstance(episode, Mapping):
        raise ValueError("source frame lacks episode provenance")
    episode_reset_count = _strict_json_int(
        episode.get("reset_count"), label="source frame reset_count"
    )
    episode_step = _strict_json_int(
        episode.get("episode_step"), label="source frame episode_step"
    )
    if (
        str(episode.get("episode_id", "")) != str(record["episode_id"])
        or episode_reset_count != int(record["reset_count"])
        or episode_step != int(record["episode_step"])
    ):
        raise ValueError("source-frame episode provenance disagrees with fit record")
    base_pose = frame.get("base_pose_world")
    base_rpy = frame.get("base_rpy_rad")
    base_quaternion = frame.get("base_quat_world_xyzw")
    frame_mount = frame.get("camera_mount_body")
    camera = frame.get("camera_pose_world")
    if not isinstance(base_pose, Mapping) or not isinstance(base_rpy, Mapping):
        raise ValueError("source frame lacks base pose/yaw")
    if not isinstance(camera, Mapping):
        raise ValueError("source frame lacks camera pose")
    position = base_pose.get("position")
    if not isinstance(position, Mapping) or set(position) != {"x", "y", "z"}:
        raise ValueError("source frame lacks base position")
    if "yaw" not in base_rpy or set(base_rpy) - {"roll", "pitch", "yaw"}:
        raise ValueError("source frame base_rpy_rad fields changed")
    base_xyz = tuple(
        _finite_vector(
            [position[axis] for axis in ("x", "y", "z")],
            length=3,
            label="source base position",
        )
    )
    raw_base_yaw = base_rpy.get("yaw")
    camera_position = tuple(
        _finite_vector(camera.get("position"), length=3, label="source camera position")
    )
    camera_lookat = tuple(
        _finite_vector(camera.get("lookat"), length=3, label="source camera lookat")
    )
    camera_up = tuple(
        _finite_vector(camera.get("up"), length=3, label="source camera up")
    )
    if (
        not all(
            math.isfinite(value)
            for value in (*base_xyz, *camera_position, *camera_lookat, *camera_up)
        )
    ):
        raise ValueError("source frame has malformed finite pose values")
    composition = _camera_mount_composition_evidence(
        base_position_world=base_xyz,
        base_quat_world_xyzw=base_quaternion,
        stored_base_yaw_rad=raw_base_yaw,
        plan_camera_mount_body=plan_camera_mount_body,
        frame_camera_mount_body=frame_mount,
        recorded_camera_pose_world={
            "position": camera_position,
            "lookat": camera_lookat,
            "up": camera_up,
        },
    )
    base_yaw = float(composition["stored_base_yaw_rad"])
    return {
        "base_pose_world": {"position": dict(zip(("x", "y", "z"), base_xyz))},
        "base_rpy_rad": {"yaw": base_yaw},
        "base_quat_world_xyzw": list(composition["base_quat_world_xyzw"]),
        "camera_mount_body": dict(composition["frame_camera_mount_body"]),
        "camera_pose_world": {
            "position": list(camera_position),
            "lookat": list(camera_lookat),
            "up": list(camera_up),
        },
        "camera_mount_composition": composition,
    }


def _scan_allowlisted_frames(
    path: Path,
    expected_sha256: str,
    records: Sequence[Mapping[str, Any]],
    *,
    allowlist: Mapping[Path, str],
    ledger: dict[str, Any],
    expected_rendered_timestamps: Mapping[tuple[int, int], int],
    plan_camera_mount_body: Mapping[str, Any],
) -> dict[tuple[Any, ...], dict[str, Any]]:
    resolved = _require_allowlisted(
        path,
        expected_sha256,
        allowlist=allowlist,
        ledger=ledger,
        label="source frames JSONL",
        requested_role="source_frames_jsonl",
    )
    ledger["source_geometry_hash_byte_opens"] += 1
    before = _hash_file(resolved)
    if before != expected_sha256:
        raise ValueError("source frames JSONL SHA-256 changed")
    wanted: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for record in records:
        wanted.setdefault(_source_record_identity(record), []).append(record)
    found: dict[tuple[Any, ...], dict[str, Any]] = {}
    rendered_occurrences: Counter[tuple[int, int]] = Counter()
    ledger["source_geometry_json_parses"] += 1
    with resolved.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ValueError("source frames JSONL contains a blank record")
            if not line.endswith("\n"):
                raise ValueError("source frames JSONL lacks a terminal newline")
            ledger["source_geometry_jsonl_records"] += 1
            try:
                frame = _strict_json_bytes(
                    line.encode("utf-8"),
                    name=f"source frame JSONL line {line_number}",
                )
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid source frame JSON at line {line_number}") from exc
            if not isinstance(frame, Mapping):
                raise ValueError("source frame JSONL record is not an object")
            raw_key_values = tuple(
                frame.get(name) for name in ("frame_index", "env_index", "timestamp_ns")
            )
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in raw_key_values
            ):
                raise ValueError("source frame key/timestamp is not a strict integer")
            key = tuple(map(int, raw_key_values))
            rendered_key = (key[0], key[1])
            if rendered_key not in expected_rendered_timestamps:
                # The committed source JSONL is a larger corpus. Only the
                # exact selected/rendered key set is authorized for geometry.
                continue
            rendered_occurrences[rendered_key] += 1
            if rendered_occurrences[rendered_key] != 1:
                raise ValueError("source frame JSONL repeats a planned frame key")
            if key[2] != int(expected_rendered_timestamps[rendered_key]):
                raise ValueError(
                    "source frame timestamp disagrees with rendered-frame commitment"
                )
            matches = wanted.get(key, ())
            for record in matches:
                identity = tuple(_frame_identity_values(record))
                if identity in found:
                    raise ValueError("source frame matched one fit identity more than once")
                found[identity] = _extract_source_frame(
                    frame,
                    record,
                    plan_camera_mount_body=plan_camera_mount_body,
                )
                ledger["source_frame_records_selected"] += 1
    ledger["source_geometry_hash_byte_opens"] += 1
    if _hash_file(resolved) != before:
        raise ValueError("source frames JSONL changed while it was scanned")
    expected_identities = {tuple(_frame_identity_values(record)) for record in records}
    if set(found) != expected_identities:
        raise ValueError("source frame scan did not match the selected fit identities once")
    if rendered_occurrences != Counter({key: 1 for key in expected_rendered_timestamps}):
        raise ValueError(
            "source frame JSONL does not contain every selected render key exactly once"
        )
    return found


def _rendered_boxes(manifest: Any) -> tuple[BoxObject, ...]:
    distractors: tuple[BoxObject, ...] = ()
    if manifest.visual_randomization is not None:
        distractors = tuple(manifest.visual_randomization.distractor_objects)
    return tuple((*manifest.walls, *manifest.obstacles, *manifest.landmarks, *distractors))


def _validate_raw_scene_object_records(payload: Mapping[str, Any]) -> None:
    groups: list[tuple[str, object]] = [
        ("wall", payload.get("walls", [])),
        ("obstacle", payload.get("obstacles", [])),
        ("landmark", payload.get("landmarks", [])),
    ]
    visual = payload.get("visual_randomization")
    distractors: object = []
    if isinstance(visual, Mapping):
        distractors = visual.get("distractor_objects", [])
    groups.append(("distractor", distractors))
    for group, raw_boxes in groups:
        if not isinstance(raw_boxes, list):
            raise ValueError(f"scene manifest {group} boxes are not a list")
        for index, raw_box in enumerate(raw_boxes):
            if not isinstance(raw_box, Mapping):
                raise ValueError(f"scene manifest {group}[{index}] is not an object")
            for name in ("object_id", "kind"):
                if not isinstance(raw_box.get(name), str) or not raw_box.get(name):
                    raise ValueError(f"scene manifest {group} {name} is malformed")
            for name in ("center_xyz_m", "size_xyz_m"):
                values = raw_box.get(name)
                if not isinstance(values, list) or len(values) != 3:
                    raise ValueError(
                        f"scene manifest {group} {name} must contain three values"
                    )
                for value in values:
                    _strict_json_number(
                        value, label=f"scene manifest {group} {name} value"
                    )
            for name in ("roll_rad", "pitch_rad", "yaw_rad"):
                _strict_json_number(
                    raw_box.get(name, 0.0),
                    label=f"scene manifest {group} {name}",
                )


def _box_geometry(box: BoxObject) -> tuple[float, ...]:
    rotation = labels_v3._box_rotation_matrix(box)
    values = (
        *map(float, box.center_xyz_m),
        *map(float, box.size_xyz_m),
        *map(float, rotation.reshape(-1)),
    )
    if len(values) != 15 or not all(math.isfinite(value) for value in values):
        raise ValueError("box geometry is not a finite center/size/rotation tuple")
    return tuple(values)


def _match_boxes(
    rendered: Sequence[BoxObject], collision: Sequence[BoxObject]
) -> dict[str, Any]:
    rendered_records = sorted(
        ((_box_geometry(box), index, box) for index, box in enumerate(rendered)),
        key=lambda item: (item[0], item[1]),
    )
    collision_records = sorted(
        ((_box_geometry(box), index, box) for index, box in enumerate(collision)),
        key=lambda item: (item[0], item[1]),
    )
    available = set(range(len(rendered_records)))
    matches: list[tuple[int, int]] = []
    unmatched_collision: list[int] = []
    for collision_geometry, collision_index, _box in collision_records:
        candidates = [
            position
            for position in sorted(available)
            if all(
                abs(left - right) <= 1e-12
                for left, right in zip(rendered_records[position][0], collision_geometry)
            )
        ]
        if not candidates:
            unmatched_collision.append(collision_index)
            continue
        position = candidates[0]
        available.remove(position)
        matches.append((rendered_records[position][1], collision_index))
    unmatched_rendered = [rendered_records[position][1] for position in sorted(available)]
    multiplicities = Counter(_box_geometry(collision[index]) for _, index in matches)
    return {
        "matches": matches,
        "unmatched_rendered_indices": unmatched_rendered,
        "unmatched_collision_indices": unmatched_collision,
        "unmatched_rendered_boxes": [
            {
                "index": int(index),
                "canonical_geometry": list(_box_geometry(rendered[index])),
            }
            for index in unmatched_rendered
        ],
        "unmatched_collision_boxes": [
            {
                "index": int(index),
                "canonical_geometry": list(_box_geometry(collision[index])),
            }
            for index in unmatched_collision
        ],
        "matched_multiplicities": [
            {"canonical_geometry": list(geometry), "multiplicity": count}
            for geometry, count in sorted(multiplicities.items())
        ],
    }


def _reconstruct_label_stages(
    manifest: Any,
    frame: Mapping[str, Any],
    *,
    rendered_boxes: Sequence[BoxObject],
    collision_boxes: Sequence[BoxObject],
    geometry_flags: Mapping[str, Any],
    camera_projection: Mapping[str, Any],
    physical_grid: InflatedOccupancyGrid | None = None,
) -> dict[str, np.ndarray]:
    local_grid = labels_v3.DEFAULT_LOCAL_GRID
    base_x, base_y, base_yaw = labels_v3._base_xy_yaw(frame)
    camera = labels_v3._camera_observation(
        frame,
        horizontal_fov_deg=float(camera_projection["horizontal_fov_deg"]),
        near_m=float(camera_projection["near_m"]),
        vertical_fov_deg=float(camera_projection["vertical_fov_deg"]),
        require_recorded_up=True,
        image_width_px=int(camera_projection.get("resolution_wh", [224, 168])[0]),
        image_height_px=int(camera_projection.get("resolution_wh", [224, 168])[1]),
        obstacle_ray_stride_px=2,
    )
    if physical_grid is None:
        physical_grid = InflatedOccupancyGrid(
            manifest,
            cell_size_m=float(geometry_flags["oracle_cell_size_m"]),
            inflation_m=0.0,
            treat_landmarks_as_obstacles=bool(geometry_flags["landmarks_are_obstacles"]),
            treat_distractors_as_obstacles=bool(geometry_flags["distractors_are_obstacles"]),
        )
    forward, left = np.meshgrid(
        local_grid.forward_centers_m(), local_grid.left_centers_m(), indexing="ij"
    )
    cos_yaw = math.cos(base_yaw)
    sin_yaw = math.sin(base_yaw)
    output_x = base_x + cos_yaw * forward - sin_yaw * left
    output_y = base_y + sin_yaw * forward + cos_yaw * left

    cell_size = float(physical_grid.cell_size_m)
    half_cell = 0.5 * cell_size
    output_half = 0.5 * float(local_grid.cell_size_m)
    origin_x, origin_y = physical_grid.origin_xy
    x_low = float(np.min(output_x)) - output_half - half_cell
    x_high = float(np.max(output_x)) + output_half + half_cell
    y_low = float(np.min(output_y)) - output_half - half_cell
    y_high = float(np.max(output_y)) + output_half + half_cell
    ix_low = int(math.floor((x_low - origin_x) / cell_size - 0.5)) - 1
    ix_high = int(math.ceil((x_high - origin_x) / cell_size - 0.5)) + 1
    iy_low = int(math.floor((y_low - origin_y) / cell_size - 0.5)) - 1
    iy_high = int(math.ceil((y_high - origin_y) / cell_size - 0.5)) + 1
    ix = np.arange(ix_low, ix_high + 1, dtype=np.int64)
    iy = np.arange(iy_low, iy_high + 1, dtype=np.int64)
    x_centers = origin_x + (ix.astype(np.float64) + 0.5) * cell_size
    y_centers = origin_y + (iy.astype(np.float64) + 0.5) * cell_size
    world_x, world_y = np.meshgrid(x_centers, y_centers, indexing="ij")
    inside = (
        (ix[:, None] >= 0)
        & (ix[:, None] < physical_grid.shape[0])
        & (iy[None, :] >= 0)
        & (iy[None, :] < physical_grid.shape[1])
    )
    physical_free = np.zeros(inside.shape, dtype=bool)
    valid_rows, valid_cols = np.nonzero(inside)
    physical_free[valid_rows, valid_cols] = physical_grid.free_mask[
        ix[valid_rows], iy[valid_cols]
    ]
    physical_labels = np.full(inside.shape, UNKNOWN_CLASS, dtype=np.uint8)
    free_rows, free_cols = np.nonzero(inside & physical_free)
    if free_rows.size:
        free_center_x = world_x[free_rows, free_cols]
        free_center_y = world_y[free_rows, free_cols]
        offsets = np.asarray(
            ((0.0, 0.0), (-half_cell, -half_cell), (-half_cell, half_cell),
             (half_cell, -half_cell), (half_cell, half_cell)),
            dtype=np.float64,
        )
        floor_points = np.column_stack(
            (
                (free_center_x[:, None] + offsets[None, :, 0]).ravel(),
                (free_center_y[:, None] + offsets[None, :, 1]).ravel(),
                np.full(free_rows.size * offsets.shape[0], float(camera.ground_plane_z_m)),
            )
        )
        in_frustum, distances, nearest, _nearest_index = labels_v3._first_box_hits(
            floor_points, camera, rendered_boxes
        )
        visible = (in_frustum & (nearest >= distances - 1e-9)).reshape(
            free_rows.size, offsets.shape[0]
        ).all(axis=1)
        physical_labels[free_rows[visible], free_cols[visible]] = FREE_CLASS
    witnesses = labels_v3._visible_obstacle_camera_ray_witnesses_xy(camera, rendered_boxes)
    pre_veto = labels_v3.observable_physical_labels_from_raster(
        physical_labels,
        physical_x_centers_m=x_centers,
        physical_y_centers_m=y_centers,
        output_world_x_m=output_x,
        output_world_y_m=output_y,
        output_yaw_rad=base_yaw,
        physical_cell_size_m=cell_size,
        output_cell_size_m=local_grid.cell_size_m,
        visible_obstacle_first_hit_xy_m=witnesses,
    )
    collision_overlap = labels_v3._output_cells_intersect_collision_geometry(
        output_x,
        output_y,
        output_yaw_rad=base_yaw,
        output_cell_size_m=local_grid.cell_size_m,
        obstacle_boxes=collision_boxes,
    )
    rendered_overlap = labels_v3._output_cells_intersect_collision_geometry(
        output_x,
        output_y,
        output_yaw_rad=base_yaw,
        output_cell_size_m=local_grid.cell_size_m,
        obstacle_boxes=rendered_boxes,
    )
    final = np.asarray(pre_veto, dtype=np.uint8).copy()
    final[(final == FREE_CLASS) & collision_overlap] = UNKNOWN_CLASS
    return {
        "pre_veto": pre_veto,
        "collision_overlap": collision_overlap,
        "rendered_overlap": rendered_overlap,
        "final": final,
        "output_x": output_x,
        "output_y": output_y,
    }


def _committed_source_entry(summary: Mapping[str, Any], name: str) -> tuple[Path, str]:
    source = summary.get("source")
    record = source.get(name) if isinstance(source, Mapping) else None
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"render summary lacks committed source.{name}")
    path = Path(str(record.get("path", "")))
    digest = str(record.get("sha256", ""))
    if not _is_sha256(digest):
        raise ValueError(f"render summary source.{name} hash is malformed")
    return path, digest


def _validate_summary_records(
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    summary_path: Path,
) -> None:
    if not records:
        raise ValueError("render summary has no selected fit records")
    family = str(records[0]["family"])
    scene_id = str(records[0]["scene_id"])
    if any(
        str(record["family"]) != family or str(record["scene_id"]) != scene_id
        for record in records
    ):
        raise ValueError("one render summary was assigned multiple scene identities")
    if (
        summary.get("schema") != "lewm_rendered_vision_v04"
        or summary.get("render_status") != "complete"
        or str(summary.get("family", "")) != family
        or str(summary.get("scene_id", "")) != scene_id
    ):
        raise ValueError("render summary identity/status changed")
    if bool(summary.get("g2_model_outputs_opened", False)):
        raise PermissionError("render summary declares forbidden G2 model-output access")
    rendered = summary.get("rendered_frames")
    if not isinstance(rendered, list):
        raise ValueError("render summary lacks rendered-frame commitments")
    by_key: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for item in rendered:
        if not isinstance(item, Mapping) or set(item) != {
            "frame_index",
            "env_index",
            "timestamp_ns",
            "image_sha256",
        }:
            raise ValueError("rendered-frame commitment is malformed")
        if not _is_sha256(item.get("image_sha256")):
            raise ValueError("rendered-frame image commitment is malformed")
        frame_index = _strict_json_int(
            item.get("frame_index"), label="rendered frame_index"
        )
        env_index = _strict_json_int(item.get("env_index"), label="rendered env_index")
        _strict_json_int(item.get("timestamp_ns"), label="rendered timestamp_ns")
        by_key.setdefault((frame_index, env_index), []).append(item)
    for record in records:
        matches = by_key.get((int(record["frame_index"]), int(record["env_index"])), ())
        if len(matches) != 1:
            raise ValueError("selected fit frame does not match render metadata once")
        match = matches[0]
        if (
            _strict_json_int(
                match.get("timestamp_ns"), label="rendered timestamp_ns"
            )
            != int(record["timestamp_ns"])
            or str(match.get("image_sha256", "")) != str(record["image_sha256"])
        ):
            raise ValueError("selected fit frame commitment changed")
        image_parent = Path(str(record["image_path_metadata_only"])).parent
        if image_parent != summary_path.parent / "rgb":
            raise PermissionError("committed image path escapes its render summary")


def _validate_frame_selection_and_rendered_set(
    selection: Mapping[str, Any],
    summary: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
    *,
    scene_id: str,
    expected_content_sha256: str,
) -> dict[tuple[int, int], int]:
    _validate_embedded_content_hash(
        selection,
        expected_sha256=expected_content_sha256,
        label="fit frame selection",
    )
    frame_keys = selection.get("frame_keys")
    expected_selection_fields = {
        "schema",
        "scene_id",
        "scene_id_sha256",
        "dataset_role",
        "row_count",
        "frame_count",
        "frame_keys",
        "frame_key_set_sha256",
        "source_rows",
        "g2_images_opened",
        "g2_label_shards_opened",
        "content_sha256",
    }
    if (
        set(selection) != expected_selection_fields
        or selection.get("schema") != "lewm_go2_selected_render_frames_v1"
        or str(selection.get("scene_id", "")) != scene_id
        or str(selection.get("scene_id_sha256", ""))
        != hashlib.sha256(scene_id.encode("utf-8")).hexdigest()
        or selection.get("dataset_role") != "train"
        or selection.get("g2_images_opened") is not False
        or selection.get("g2_label_shards_opened") is not False
        or not isinstance(frame_keys, list)
    ):
        raise ValueError("fit frame-selection contract changed")
    source_rows = selection.get("source_rows")
    if (
        not isinstance(source_rows, Mapping)
        or set(source_rows) != {"path", "sha256"}
        or not str(source_rows.get("path", ""))
        or not _is_sha256(source_rows.get("sha256"))
    ):
        raise ValueError("fit frame-selection source-row commitment changed")
    normalized_keys = []
    for index, key in enumerate(frame_keys):
        if (
            not isinstance(key, list)
            or len(key) != 2
            or any(isinstance(value, bool) or not isinstance(value, int) for value in key)
        ):
            raise ValueError(f"fit frame-selection key {index} is malformed")
        normalized_keys.append([int(key[0]), int(key[1])])
    if normalized_keys != sorted(normalized_keys) or len(normalized_keys) != len(
        {tuple(key) for key in normalized_keys}
    ):
        raise ValueError("fit frame-selection keys are not canonical and unique")
    key_hash = canonical_json_sha256(normalized_keys)
    summary_selection = summary.get("frame_selection")
    if (
        _strict_json_int(
            selection.get("frame_count"), label="fit frame-selection frame_count"
        )
        != len(normalized_keys)
        or str(selection.get("frame_key_set_sha256", "")) != key_hash
        or not isinstance(summary_selection, Mapping)
        or str(summary_selection.get("frame_key_set_sha256", "")) != key_hash
    ):
        raise ValueError("fit frame-selection key-set commitment changed")
    selected_keys = {
        (int(record["frame_index"]), int(record["env_index"])) for record in records
    }
    full_selection_keys = {tuple(key) for key in normalized_keys}
    if len(selected_keys) != len(records) or not selected_keys.issubset(
        full_selection_keys
    ):
        raise ValueError(
            "fit panel endpoints are not a unique subset of the frame selection"
        )
    selected_rows = {
        (str(record["scene_id"]), int(record["global_row"])) for record in records
    }
    if _strict_json_int(
        selection.get("row_count"), label="fit frame-selection row_count"
    ) < len(selected_rows):
        raise ValueError("fit frame-selection row count is smaller than its fit subset")

    rendered = summary.get("rendered_frames")
    if not isinstance(rendered, list):
        raise ValueError("fit render summary lacks rendered frames")
    normalized_rendered = []
    for item in rendered:
        if not isinstance(item, Mapping) or set(item) != {
            "frame_index",
            "env_index",
            "timestamp_ns",
            "image_sha256",
        }:
            raise ValueError("fit rendered-frame record is malformed")
        if not _is_sha256(item.get("image_sha256")):
            raise ValueError("fit rendered-frame image SHA-256 is malformed")
        normalized_rendered.append(
            {
                "frame_index": _strict_json_int(
                    item.get("frame_index"), label="fit rendered frame_index"
                ),
                "env_index": _strict_json_int(
                    item.get("env_index"), label="fit rendered env_index"
                ),
                "timestamp_ns": _strict_json_int(
                    item.get("timestamp_ns"), label="fit rendered timestamp_ns"
                ),
                "image_sha256": str(item["image_sha256"]),
            }
        )
    normalized_rendered.sort(key=lambda item: (item["frame_index"], item["env_index"]))
    rendered_keys = [
        [item["frame_index"], item["env_index"]] for item in normalized_rendered
    ]
    if (
        rendered_keys != normalized_keys
        or len(normalized_rendered)
        != _strict_json_int(summary.get("frame_count"), label="render summary frame_count")
        or canonical_json_sha256(normalized_rendered)
        != str(summary.get("rendered_image_set_sha256", ""))
    ):
        raise ValueError("fit rendered-frame set commitment changed")
    return {
        (item["frame_index"], item["env_index"]): item["timestamp_ns"]
        for item in normalized_rendered
    }


def _read_source_geometry(
    records: Sequence[Mapping[str, Any]],
    panel_inputs: Mapping[str, Any],
    *,
    spec: AuditSpec,
    ledger: dict[str, Any],
    authorized_source_entries: Sequence[Mapping[str, Any]] | None = None,
    inventory_only: bool = False,
) -> tuple[
    dict[tuple[Any, ...], dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, Any],
    list[dict[str, str]],
]:
    allowlist: dict[Path, str] = {}
    opened: set[Path] = set()
    source_assignments: set[tuple[str, str, str, str]] = set()
    authorized_assignments = (
        None
        if authorized_source_entries is None
        else {
            (
                str(entry["path"]),
                str(entry["sha256"]),
                str(entry["semantic_role"]),
                str(entry["scene_id"]),
            )
            for entry in authorized_source_entries
        }
    )
    selected_scene_ids_frozen = {str(record["scene_id"]) for record in records}

    def require_pre_authorized(
        path: Path,
        digest: str,
        role: str,
        scene_ids: Sequence[str],
    ) -> None:
        if authorized_assignments is None:
            return
        lexical = path if path.is_absolute() else ROOT / path
        expected = {
            (str(lexical), str(digest), role, str(scene_id))
            for scene_id in scene_ids
        }
        if not expected.issubset(authorized_assignments):
            raise PermissionError(
                f"{role} path/hash/scene was not authorized by the machine manifest"
            )
    panel_geometry = panel_inputs.get("geometry_contract")
    panel_render_audit = panel_inputs.get("render_audit_contract")
    if not isinstance(panel_geometry, Mapping) or not isinstance(
        panel_render_audit, Mapping
    ):
        raise ValueError("panel source inputs are incomplete")
    geometry_path_raw = Path(str(panel_geometry.get("path", "")))
    geometry_path = geometry_path_raw if geometry_path_raw.is_absolute() else ROOT / geometry_path_raw
    geometry_digest = str(panel_geometry.get("file_sha256", ""))
    require_pre_authorized(
        geometry_path,
        geometry_digest,
        "physical_geometry_contract",
        sorted(selected_scene_ids_frozen),
    )
    geometry_path = _add_allowlist_entry(
        allowlist,
        geometry_path,
        geometry_digest,
        ledger=ledger,
        label="physical geometry contract",
        requested_role="physical_geometry_contract",
    )
    geometry_payload = _read_allowlisted_json(
        geometry_path,
        geometry_digest,
        allowlist=allowlist,
        ledger=ledger,
        label="physical geometry contract",
        requested_role="physical_geometry_contract",
    )
    opened.add(geometry_path)
    semantic = _geometry_semantic_sha256(geometry_payload)
    if semantic != str(panel_geometry.get("semantic_sha256", "")):
        raise ValueError("physical geometry semantic SHA-256 changed")
    geometry_flags = _geometry_flags(geometry_payload)

    render_audit_path_raw = Path(str(panel_render_audit.get("path", "")))
    render_audit_path = (
        render_audit_path_raw
        if render_audit_path_raw.is_absolute()
        else ROOT / render_audit_path_raw
    )
    render_audit_digest = str(panel_render_audit.get("file_sha256", ""))
    require_pre_authorized(
        render_audit_path,
        render_audit_digest,
        "render_audit_contract",
        sorted(selected_scene_ids_frozen),
    )
    render_audit_path = _add_allowlist_entry(
        allowlist,
        render_audit_path,
        render_audit_digest,
        ledger=ledger,
        label="render audit contract",
        requested_role="render_audit_contract",
    )
    render_audit_payload = _read_allowlisted_json(
        render_audit_path,
        render_audit_digest,
        allowlist=allowlist,
        ledger=ledger,
        label="render audit contract",
        requested_role="render_audit_contract",
    )
    _validate_render_audit_contract(
        render_audit_payload,
        expected_content_sha256=str(panel_render_audit.get("content_sha256", "")),
    )
    opened.add(render_audit_path)

    summary_root = spec.summary_root
    summary_commitments: dict[Path, str] = {}
    for relative, digest in spec.summaries().items():
        summary_candidate = summary_root / relative
        summary_scene_ids = {
            str(record["scene_id"])
            for record in records
            if Path(str(record["image_path_metadata_only"])).parent.parent
            == summary_candidate.parent
        }
        require_pre_authorized(
            summary_candidate,
            digest,
            "fit_render_summary",
            sorted(summary_scene_ids),
        )
        path = _add_allowlist_entry(
            allowlist,
            summary_root / relative,
            digest,
            ledger=ledger,
            label="fit render summary",
            requested_role="fit_render_summary",
        )
        summary_commitments[path] = digest
    by_summary: dict[Path, list[Mapping[str, Any]]] = {}
    for record in records:
        path = _summary_path_for_record(record, spec=spec, ledger=ledger)
        if path not in summary_commitments:
            _record_path_denial(
                ledger,
                path=path,
                requested_role="fit_render_summary",
                declared_role="train",
                modality="json",
                primary_reason="unallowlisted",
                resolved_path=path,
            )
            raise PermissionError("fit frame names a render summary outside the frozen allowlist")
        by_summary.setdefault(path, []).append(record)
    if set(by_summary) != set(summary_commitments):
        raise ValueError("fit frame summaries differ from the frozen allowlist")

    source_frames: dict[tuple[Any, ...], dict[str, Any]] = {}
    scenes: dict[str, dict[str, Any]] = {}
    cached_json: dict[Path, dict[str, Any]] = {geometry_path: geometry_payload}
    verified_files: set[Path] = set()
    for summary_path in sorted(by_summary, key=str):
        summary = _read_allowlisted_json(
            summary_path,
            summary_commitments[summary_path],
            allowlist=allowlist,
            ledger=ledger,
            label="fit render summary",
            requested_role="fit_render_summary",
        )
        opened.add(summary_path)
        selected = by_summary[summary_path]
        _validate_summary_records(summary, selected, summary_path=summary_path)
        scene_id = str(selected[0]["scene_id"])
        if scene_id in scenes:
            raise ValueError("one selected scene is represented by multiple summaries")
        source_assignments.update(
            {
                (str(geometry_path), geometry_digest, "physical_geometry_contract", scene_id),
                (str(render_audit_path), render_audit_digest, "render_audit_contract", scene_id),
                (
                    str(summary_path),
                    summary_commitments[summary_path],
                    "fit_render_summary",
                    scene_id,
                ),
            }
        )
        selection_record = summary.get("frame_selection")
        if not isinstance(selection_record, Mapping) or set(selection_record) != {
            "path",
            "sha256",
            "frame_key_set_sha256",
        }:
            raise ValueError("fit render summary lacks frame-selection provenance")
        selection_path_raw = Path(str(selection_record.get("path", "")))
        selection_path = (
            selection_path_raw
            if selection_path_raw.is_absolute()
            else ROOT / selection_path_raw
        )
        selection_digest = str(selection_record.get("sha256", ""))
        require_pre_authorized(
            selection_path,
            selection_digest,
            "fit_frame_selection",
            [scene_id],
        )
        selection_path = _add_allowlist_entry(
            allowlist,
            selection_path,
            selection_digest,
            ledger=ledger,
            label="fit frame selection",
            requested_role="fit_frame_selection",
        )
        selection_payload = _read_allowlisted_json(
            selection_path,
            selection_digest,
            allowlist=allowlist,
            ledger=ledger,
            label="fit frame selection",
            requested_role="fit_frame_selection",
        )
        expected_rendered_timestamps = _validate_frame_selection_and_rendered_set(
            selection_payload,
            summary,
            selected,
            scene_id=scene_id,
            expected_content_sha256=str(selection_payload.get("content_sha256", "")),
        )
        opened.add(selection_path)
        source_assignments.add(
            (str(selection_path), selection_digest, "fit_frame_selection", scene_id)
        )
        projection = summary.get("camera_projection")
        if not isinstance(projection, Mapping) or set(projection) != {
            "model",
            "renderer_fov_axis",
            "horizontal_fov_deg",
            "vertical_fov_deg",
            "near_m",
            "far_m",
            "runtime_rectification_required",
        }:
            raise ValueError("render summary lacks camera projection")
        summary_horizontal = _strict_json_number(
            projection.get("horizontal_fov_deg"),
            label="render summary horizontal_fov_deg",
        )
        summary_near = _strict_json_number(
            projection.get("near_m"), label="render summary near_m"
        )
        summary_vertical = _strict_json_number(
            projection.get("vertical_fov_deg"),
            label="render summary vertical_fov_deg",
        )
        summary_far = _strict_json_number(
            projection.get("far_m"), label="render summary far_m"
        )
        if (
            not math.isclose(
                summary_horizontal,
                78.323,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                summary_near,
                0.05,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                summary_vertical,
                62.8370386364,
                rel_tol=0.0,
                abs_tol=1e-10,
            )
            or list(summary.get("resolution_wh", ())) != [224, 168]
            or summary_far <= summary_near
        ):
            raise ValueError("render summary camera projection changed")
        camera_projection = {
            "horizontal_fov_deg": summary_horizontal,
            "vertical_fov_deg": summary_vertical,
            "near_m": summary_near,
            "far_m": summary_far,
            "resolution_wh": [224, 168],
        }

        committed: dict[str, tuple[Path, str]] = {}
        source_roles = {
            "plan": "render_source_plan",
            "frames_jsonl": "source_frames_jsonl",
            "scene_manifest": "source_scene_manifest",
            "renderer_source": "renderer_source",
        }
        summary_source = summary.get("source")
        if not isinstance(summary_source, Mapping) or set(summary_source) != set(
            source_roles
        ):
            raise ValueError("render summary source inventory changed")
        for source_name in ("plan", "frames_jsonl", "scene_manifest", "renderer_source"):
            raw_path, digest = _committed_source_entry(summary, source_name)
            path = raw_path if raw_path.is_absolute() else ROOT / raw_path
            require_pre_authorized(
                path,
                digest,
                source_roles[source_name],
                [scene_id],
            )
            committed[source_name] = (
                _add_allowlist_entry(
                    allowlist,
                    path,
                    digest,
                    ledger=ledger,
                    label=f"render source {source_name}",
                    requested_role=source_roles[source_name],
                ),
                digest,
            )
            source_assignments.add(
                (str(committed[source_name][0]), digest, source_roles[source_name], scene_id)
            )

        plan_path, plan_digest = committed["plan"]
        if plan_path in cached_json:
            plan = cached_json[plan_path]
        else:
            plan = _read_allowlisted_json(
                plan_path,
                plan_digest,
                allowlist=allowlist,
                ledger=ledger,
                label="render source plan",
                requested_role="render_source_plan",
            )
            cached_json[plan_path] = plan
            opened.add(plan_path)
        if (
            plan.get("schema") != "lewm_render_replay_plan_v0"
            or str(plan.get("scene_id", "")) != scene_id
        ):
            raise ValueError("render source plan scene identity changed")
        plan_camera = plan.get("camera")
        if not isinstance(plan_camera, Mapping) or set(plan_camera) != {
            "native_resolution",
            "training_resolution",
            "fov_axis",
            "fov_deg",
            "near_m",
            "far_m",
            "encoding",
            "mount_body",
        }:
            raise ValueError("render source plan lacks camera contract")
        plan_camera_mount = _camera_mount_record(
            plan_camera.get("mount_body"), label="render source plan camera.mount_body"
        )
        plan_horizontal = _strict_json_number(
            plan_camera.get("fov_deg"), label="render plan fov_deg"
        )
        plan_near = _strict_json_number(
            plan_camera.get("near_m"), label="render plan near_m"
        )
        plan_far = _strict_json_number(
            plan_camera.get("far_m"), label="render plan far_m"
        )
        expected_vertical = math.degrees(
            2.0
            * math.atan(
                math.tan(math.radians(plan_horizontal) * 0.5) * (168.0 / 224.0)
            )
        )
        if (
            plan_camera.get("fov_axis") != "horizontal"
            or
            projection.get("model") != "pinhole"
            or projection.get("renderer_fov_axis") != "vertical"
            or projection.get("runtime_rectification_required") is not False
            or not math.isclose(
                summary_horizontal,
                plan_horizontal,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                summary_vertical,
                expected_vertical,
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            or not math.isclose(
                summary_near,
                plan_near,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            or not math.isclose(
                summary_far,
                plan_far,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            raise ValueError("render source plan/summary projection mismatch")
        plan_frames = _authorize_path(
            Path(str(plan.get("frames_jsonl", ""))),
            ROOT,
            ledger=ledger,
            requested_role="source_frames_jsonl",
            declared_role="train",
            expected_resolved_path=committed["frames_jsonl"][0],
            label="render plan frames JSONL",
        )
        if plan_frames != committed["frames_jsonl"][0]:
            raise ValueError("render source plan/summary frame paths disagree")

        manifest_path, manifest_digest = committed["scene_manifest"]
        if manifest_path in cached_json:
            manifest_payload = cached_json[manifest_path]
        else:
            manifest_payload = _read_allowlisted_json(
                manifest_path,
                manifest_digest,
                allowlist=allowlist,
                ledger=ledger,
                label="source scene manifest",
                requested_role="source_scene_manifest",
            )
            cached_json[manifest_path] = manifest_payload
            opened.add(manifest_path)
        _validate_raw_scene_object_records(manifest_payload)
        manifest = None
        if inventory_only:
            if (
                not isinstance(manifest_payload.get("scene_id"), str)
                or manifest_payload.get("scene_id") != scene_id
                or not isinstance(manifest_payload.get("family"), str)
                or manifest_payload.get("family") != str(selected[0]["family"])
            ):
                raise ValueError("source scene manifest identity changed")
        else:
            manifest = parse_scene_manifest_dict(manifest_payload)
            if manifest.scene_id != scene_id or manifest.family != str(selected[0]["family"]):
                raise ValueError("source scene manifest identity changed")

        renderer_path, renderer_digest = committed["renderer_source"]
        if renderer_path not in verified_files:
            _verify_allowlisted_file(
                renderer_path,
                renderer_digest,
                allowlist=allowlist,
                ledger=ledger,
                label="renderer source",
                requested_role="renderer_source",
            )
            verified_files.add(renderer_path)
            opened.add(renderer_path)

        frames_path, frames_digest = committed["frames_jsonl"]
        extracted = _scan_allowlisted_frames(
            frames_path,
            frames_digest,
            selected,
            allowlist=allowlist,
            ledger=ledger,
            expected_rendered_timestamps=expected_rendered_timestamps,
            plan_camera_mount_body=plan_camera_mount,
        )
        opened.add(frames_path)
        if set(source_frames) & set(extracted):
            raise ValueError("source frame identities overlap between scenes")
        source_frames.update(extracted)

        if inventory_only:
            scenes[scene_id] = {
                "scene_id": scene_id,
                "family": str(selected[0]["family"]),
            }
            continue

        rendered_boxes = _rendered_boxes(manifest)
        collision_boxes = labels_v3._physical_obstacle_boxes(
            manifest,
            treat_landmarks_as_obstacles=bool(geometry_flags["landmarks_are_obstacles"]),
            treat_distractors_as_obstacles=bool(geometry_flags["distractors_are_obstacles"]),
        )
        parity = _match_boxes(rendered_boxes, collision_boxes)
        parity_record = summary.get("object_parity")
        provenance_missing = 0
        provenance_nonunique = 0
        render_records = labels_v3._render_object_records(manifest)
        object_ids = [str(record["object_id"]) for record in render_records]
        if len(object_ids) != len(set(object_ids)):
            provenance_nonunique += 1
        if not isinstance(parity_record, Mapping) or set(parity_record) != {
            "schema",
            "rendered_groups",
            "rendered_object_count",
            "rendered_object_ids",
            "rendered_object_ids_sha256",
            "rendered_object_records_sha256",
            "collision_distractors_rendered",
            "full_box_roll_pitch_yaw_rendered",
        }:
            provenance_missing += 1
        else:
            expected_record_hash = canonical_json_sha256(render_records)
            expected_ids = sorted(object_ids)
            rendered_ids = parity_record.get("rendered_object_ids")
            if (
                parity_record.get("schema") != "lewm_render_object_parity_v1"
                or parity_record.get("rendered_groups")
                != ["wall", "obstacle", "landmark", "distractor"]
                or _strict_json_int(
                    parity_record.get("rendered_object_count"),
                    label="rendered object count",
                )
                != len(render_records)
                or not isinstance(rendered_ids, list)
                or any(not isinstance(value, str) for value in rendered_ids)
                or rendered_ids != expected_ids
                or parity_record.get("rendered_object_ids_sha256")
                != canonical_json_sha256(expected_ids)
                or parity_record.get("rendered_object_records_sha256")
                != expected_record_hash
                or parity_record.get("collision_distractors_rendered") is not True
                or parity_record.get("full_box_roll_pitch_yaw_rendered") is not True
            ):
                provenance_missing += 1
        scenes[scene_id] = {
            "scene_id": scene_id,
            "family": str(selected[0]["family"]),
            "manifest": manifest,
            "rendered_boxes": rendered_boxes,
            "collision_boxes": collision_boxes,
            "physical_grid": InflatedOccupancyGrid(
                manifest,
                cell_size_m=float(geometry_flags["oracle_cell_size_m"]),
                inflation_m=0.0,
                treat_landmarks_as_obstacles=bool(
                    geometry_flags["landmarks_are_obstacles"]
                ),
                treat_distractors_as_obstacles=bool(
                    geometry_flags["distractors_are_obstacles"]
                ),
            ),
            "box_matching": parity,
            "camera_projection": camera_projection,
            "required_provenance_missing_count": provenance_missing,
            "required_provenance_nonunique_count": provenance_nonunique,
        }

    expected_identities = {tuple(_frame_identity_values(record)) for record in records}
    if set(source_frames) != expected_identities:
        raise ValueError("source geometry did not reconcile to all fit frames")
    if {Path(record[0]) for record in source_assignments} != opened:
        raise ValueError("source role assignments do not match opened source paths")
    source_entries = [
        {
            "path": path,
            "sha256": digest,
            "semantic_role": role,
            "scene_id": scene_id,
        }
        for path, digest, role, scene_id in sorted(source_assignments)
    ]
    if authorized_assignments is not None and source_assignments != authorized_assignments:
        raise ValueError("observed source assignments differ from the machine manifest")
    return source_frames, scenes, {
        "path": str(geometry_path),
        "file_sha256": geometry_digest,
        "semantic_sha256": semantic,
        "flags": geometry_flags,
        "render_audit_contract": {
            "path": str(render_audit_path),
            "file_sha256": render_audit_digest,
            "content_sha256": str(panel_render_audit["content_sha256"]),
        },
    }, source_entries


def _distance_bin_counts(mask: np.ndarray) -> dict[str, int]:
    forward = labels_v3.DEFAULT_LOCAL_GRID.forward_centers_m()
    left = labels_v3.DEFAULT_LOCAL_GRID.left_centers_m()
    distance = np.hypot(forward[:, None], left[None, :])
    return {
        name: int(
            np.count_nonzero(
                mask
                & (distance >= low)
                & (True if high is None else distance < high)
            )
        )
        for name, low, high in DISTANCE_BINS
    }


def _overlap_for_boxes(
    stages: Mapping[str, np.ndarray],
    frame: Mapping[str, Any],
    boxes: Sequence[BoxObject],
) -> np.ndarray:
    _base_x, _base_y, yaw = labels_v3._base_xy_yaw(frame)
    return labels_v3._output_cells_intersect_collision_geometry(
        stages["output_x"],
        stages["output_y"],
        output_yaw_rad=yaw,
        output_cell_size_m=labels_v3.DEFAULT_LOCAL_GRID.cell_size_m,
        obstacle_boxes=boxes,
    )


def _scope_collision(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    scoped_partitions = {}
    for name in ATTRIBUTION_PARTITION_NAMES:
        identities = [
            [_frame_identity_values(record["record_key"]), int(row), int(column)]
            for record in records
            for row, column in record["attribution_partitions"][name]["identities"]
        ]
        scoped_partitions[name] = {
            "count": len(identities),
            "identities_sha256": canonical_json_sha256(identities),
        }
    return {
        "frame_count": len(records),
        "veto_only_unknown_count": sum(int(record["collision_veto_only_unknown_count"]) for record in records),
        "distance_bin_counts": {
            name: sum(int(record["collision_veto_only_unknown_distance_bin_counts"][name]) for record in records)
            for name, _low, _high in DISTANCE_BINS
        },
        "attributed_to_matched_box_count": sum(int(record["attributed_to_matched_box_count"]) for record in records),
        "depends_on_unmatched_collision_box_count": sum(int(record["depends_on_unmatched_collision_box_count"]) for record in records),
        "attribution_partitions": scoped_partitions,
    }


ATTRIBUTION_PARTITION_NAMES = (
    "matched_true_unmatched_false",
    "matched_false_unmatched_true",
    "matched_true_unmatched_true",
    "matched_false_unmatched_false",
)


def _mask_row_column_identities(mask: np.ndarray) -> list[list[int]]:
    return [[int(row), int(column)] for row, column in np.argwhere(mask)]


def _attribution_partition_records(
    veto: np.ndarray,
    matched_overlap: np.ndarray,
    unmatched_overlap: np.ndarray,
) -> dict[str, dict[str, Any]]:
    masks = {
        "matched_true_unmatched_false": veto & matched_overlap & ~unmatched_overlap,
        "matched_false_unmatched_true": veto & ~matched_overlap & unmatched_overlap,
        "matched_true_unmatched_true": veto & matched_overlap & unmatched_overlap,
        "matched_false_unmatched_false": veto & ~matched_overlap & ~unmatched_overlap,
    }
    result = {}
    for name in ATTRIBUTION_PARTITION_NAMES:
        identities = _mask_row_column_identities(masks[name])
        result[name] = {
            "count": len(identities),
            "identities": identities,
            "identities_sha256": canonical_json_sha256(identities),
        }
    if sum(record["count"] for record in result.values()) != int(np.count_nonzero(veto)):
        raise RuntimeError("veto attribution partitions are not exhaustive")
    return result


def _compact_core_frame(report: Mapping[str, Any]) -> dict[str, Any]:
    rays = report["ray_sequences"]
    return {
        "record_key": dict(report["frame_key"]),
        "label_support": dict(report["label_support"]),
        "ray_sequences": {
            "schema": rays["schema"],
            "summary": rays["summary"],
            "sequence_summary_records_sha256": rays["sequence_summary_records_sha256"],
            "transition_table_sha256": rays["transition_table_sha256"],
        },
    }


def _representative_ray_violations(
    reports: Sequence[Mapping[str, Any]], *, limit: int = 32
) -> dict[str, Any]:
    representatives = []
    total = 0
    for report in reports:
        for ray in report["ray_sequences"]["records"]:
            if bool(ray["scalar_first_hit_regular"]) and int(ray["transition_count"]) < 3:
                continue
            total += 1
            if len(representatives) < int(limit):
                representatives.append(
                    {
                        "frame_key": dict(ray["frame_key"]),
                        "angular_bin": int(ray["angular_bin"]),
                        "range_bins": list(map(int, ray["range_bins"])),
                        "class_sequence": list(map(int, ray["class_sequence"])),
                    }
                )
    return {
        "selection": "first_in_canonical_frame_then_angular_bin_order",
        "limit": int(limit),
        "total_violation_count": total,
        "records": representatives,
        "records_sha256": canonical_json_sha256(representatives),
    }


def _source_hashes(
    paths: Mapping[str, Path],
    *,
    ledger: dict[str, Any] | None = None,
) -> dict[str, dict[str, str]]:
    required = {
        "binding",
        "audit_core",
        "audit_core_test",
        "audit_runner",
        "audit_runner_test",
        "audit_finalizer",
        "audit_finalizer_test",
        "label_semantics",
        "geometry_contract_semantics",
        "scene_manifest_semantics",
        "planning_grid_semantics",
    }
    if set(paths) != required:
        raise ValueError("implementation source map keys changed")
    result = {}
    for name, path in sorted(paths.items()):
        if ledger is None:
            resolved = path.resolve(strict=True)
        else:
            resolved = _authorize_path(
                path,
                ROOT,
                ledger=ledger,
                requested_role=("binding" if name == "binding" else "implementation_source"),
                expected_resolved_path=path,
                label=f"implementation source {name}",
            )
            ledger["implementation_source_hash_byte_opens"] += 1
        result[name] = {"path": str(resolved), "sha256": _hash_file(resolved)}
    return result


def _runtime_environment() -> dict[str, Any]:
    version = sys.implementation.version
    return {
        "python_implementation_name": str(sys.implementation.name),
        "python_implementation_version": [
            int(version.major),
            int(version.minor),
            int(version.micro),
            str(version.releaselevel),
            int(version.serial),
        ],
        "python_version": str(sys.version),
        "numpy_version": str(np.__version__),
    }


def _validate_utc_timestamp(value: object) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError("created_at_utc is not an ISO-8601 UTC timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("created_at_utc is not an ISO-8601 UTC timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("created_at_utc is not an ISO-8601 UTC timestamp")


def _validate_runtime_environment(value: object) -> None:
    expected = _runtime_environment()
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("machine manifest runtime environment changed")
    version = value.get("python_implementation_version")
    if not isinstance(version, list) or len(version) != 5:
        raise ValueError("machine runtime implementation version is malformed")
    for index in (0, 1, 2, 4):
        _strict_json_int(
            version[index], label=f"machine runtime implementation version {index}"
        )
    if not isinstance(version[3], str):
        raise ValueError("machine runtime release level is malformed")
    for name in ("python_implementation_name", "python_version", "numpy_version"):
        if not isinstance(value.get(name), str):
            raise ValueError(f"machine runtime {name} is malformed")
    if value != expected:
        raise ValueError("machine manifest runtime environment changed")


def _canonical_manifest(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    copied = [dict(entry) for entry in entries]
    hashes = [canonical_json_sha256(entry) for entry in copied]
    if len(hashes) != len(set(hashes)):
        raise ValueError("canonical manifest contains a duplicate entry")
    return {
        "entry_count": len(copied),
        "entries": copied,
        "manifest_sha256": canonical_json_sha256(copied),
    }


def _validate_canonical_manifest(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {
        "entry_count",
        "entries",
        "manifest_sha256",
    }:
        raise ValueError(f"{label} manifest fields changed")
    entries = value.get("entries")
    if not isinstance(entries, list) or _strict_json_int(
        value.get("entry_count"), label=f"{label} entry_count"
    ) != len(entries):
        raise ValueError(f"{label} manifest count changed")
    if str(value.get("manifest_sha256", "")) != canonical_json_sha256(entries):
        raise ValueError(f"{label} manifest hash changed")
    if any(not isinstance(entry, Mapping) for entry in entries):
        raise ValueError(f"{label} manifest contains a non-object entry")
    entry_hashes = [canonical_json_sha256(entry) for entry in entries]
    if len(entry_hashes) != len(set(entry_hashes)):
        raise ValueError(f"{label} manifest contains duplicate entries")
    return {"entry_count": len(entries), "entries": entries, "manifest_sha256": value["manifest_sha256"]}


def _validate_machine_label_inventory(inventory: Mapping[str, Any]) -> None:
    entries = inventory["entries"]
    paths: list[str] = []
    canonical_coordinates: set[tuple[str, str, int, str]] = set()
    storage_rows: set[tuple[str, int, str]] = set()
    family_side_totals: Counter[tuple[str, str]] = Counter()
    selected_total = 0
    for index, entry in enumerate(entries):
        if set(entry) != {
            "path",
            "sha256",
            "selected_tuples",
            "selected_row_count",
            "family_side_counts",
        }:
            raise ValueError("machine label-shard entry fields changed")
        path = str(entry.get("path", ""))
        if not Path(path).is_absolute() or not _is_sha256(entry.get("sha256")):
            raise ValueError("machine label-shard path/hash is malformed")
        paths.append(path)
        selected = entry.get("selected_tuples")
        if not isinstance(selected, list) or _strict_json_int(
            entry.get("selected_row_count"), label="label shard selected_row_count"
        ) != len(selected):
            raise ValueError("machine label-shard selected tuple count changed")
        normalized: list[tuple[str, str, int, str, int]] = []
        for value in selected:
            if (
                not isinstance(value, list)
                or len(value) != 5
                or str(value[0]) not in FAMILIES
                or str(value[3]) not in ENDPOINT_SIDES
                or isinstance(value[2], bool)
                or not isinstance(value[2], int)
                or isinstance(value[4], bool)
                or not isinstance(value[4], int)
                or int(value[4]) < 0
            ):
                raise ValueError("machine label-shard selected tuple is malformed")
            item = (
                str(value[0]),
                str(value[1]),
                int(value[2]),
                str(value[3]),
                int(value[4]),
            )
            coordinate = item[:4]
            if coordinate in canonical_coordinates:
                raise ValueError("machine label inventory repeats a canonical fit frame")
            if (path, item[4], item[3]) in storage_rows:
                raise ValueError("machine label inventory repeats a shard row/side")
            canonical_coordinates.add(coordinate)
            storage_rows.add((path, item[4], item[3]))
            family_side_totals[(item[0], item[3])] += 1
            normalized.append(item)
        if normalized != sorted(
            normalized,
            key=lambda item: (
                FAMILIES.index(item[0]),
                item[1],
                item[2],
                ENDPOINT_SIDES.index(item[3]),
            ),
        ):
            raise ValueError("machine label-shard selected tuples are not canonical")
        declared_counts = entry.get("family_side_counts")
        expected_counts = {
            family: {
                side: sum(item[0] == family and item[3] == side for item in normalized)
                for side in ENDPOINT_SIDES
            }
            for family in FAMILIES
        }
        counts_valid = isinstance(declared_counts, Mapping) and set(
            declared_counts
        ) == set(FAMILIES)
        if counts_valid:
            for family in FAMILIES:
                side_counts = declared_counts.get(family)
                if not isinstance(side_counts, Mapping) or set(side_counts) != set(
                    ENDPOINT_SIDES
                ):
                    counts_valid = False
                    break
                for side in ENDPOINT_SIDES:
                    _strict_json_int(
                        side_counts.get(side),
                        label=f"label shard family_side_counts {family}.{side}",
                    )
        if not counts_valid or declared_counts != expected_counts:
            raise ValueError("machine label-shard family/side counts do not reconcile")
        selected_total += len(normalized)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("machine label-shard paths are not canonical and unique")
    if selected_total != EXPECTED_FRAMES or family_side_totals != Counter(
        {(family, side): EXPECTED_FRAMES // len(FAMILIES) // len(ENDPOINT_SIDES)
         for family in FAMILIES for side in ENDPOINT_SIDES}
    ):
        raise ValueError("machine label-shard selections do not equal the frozen fit scope")


def _validate_machine_source_inventory(inventory: Mapping[str, Any]) -> None:
    allowed_roles = {
        "physical_geometry_contract",
        "render_audit_contract",
        "fit_render_summary",
        "fit_frame_selection",
        "render_source_plan",
        "source_frames_jsonl",
        "source_scene_manifest",
        "renderer_source",
    }
    normalized: list[tuple[str, str, str, str]] = []
    role_scenes: set[tuple[str, str]] = set()
    for entry in inventory["entries"]:
        if set(entry) != {"path", "sha256", "semantic_role", "scene_id"}:
            raise ValueError("machine source-geometry entry fields changed")
        item = (
            str(entry.get("path", "")),
            str(entry.get("sha256", "")),
            str(entry.get("semantic_role", "")),
            str(entry.get("scene_id", "")),
        )
        if (
            not Path(item[0]).is_absolute()
            or not _is_sha256(item[1])
            or item[2] not in allowed_roles
            or not item[3]
        ):
            raise ValueError("machine source-geometry entry is malformed")
        if (item[2], item[3]) in role_scenes:
            raise ValueError("machine source geometry repeats a role/scene assignment")
        role_scenes.add((item[2], item[3]))
        normalized.append(item)
    if normalized != sorted(normalized) or len(normalized) != len(set(normalized)):
        raise ValueError("machine source-geometry entries are not canonical and unique")
    scene_sets = {
        role: {scene for entry_role, scene in role_scenes if entry_role == role}
        for role in allowed_roles
    }
    expected_scenes = scene_sets["fit_render_summary"]
    if len(expected_scenes) != len(EXPECTED_SUMMARY_SHA256) or any(
        scenes != expected_scenes for scenes in scene_sets.values()
    ):
        raise ValueError("machine source-geometry role/scene coverage changed")


def _validate_preparation_ledger(
    value: object,
    *,
    source_inventory: Mapping[str, Any],
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError("machine manifest preparation ledger is incomplete")
    expected_fields = set(new_access_ledger()) | {"passes", "forbidden_counters_zero"}
    if set(value) != expected_fields:
        raise ValueError("machine manifest preparation ledger fields changed")
    if value.get("passes") is not True or value.get("forbidden_counters_zero") is not True:
        raise ValueError("machine manifest preparation ledger is incomplete")
    if value.get("per_shard_materialization") != [] or value.get("denied_attempt_records") != []:
        raise ValueError("machine manifest preparation ledger contains forbidden records")
    primary = value.get("denied_primary_reasons")
    modalities = value.get("denied_modality_attempts")
    if (
        not isinstance(primary, Mapping)
        or set(primary) != set(PRIMARY_DENIAL_REASONS)
        or not isinstance(modalities, Mapping)
        or set(modalities) != set(DENIAL_MODALITIES)
    ):
        raise ValueError("machine manifest preparation denial tables changed")
    for table_name, table in (
        ("denied_primary_reasons", primary),
        ("denied_modality_attempts", modalities),
    ):
        for name, count in table.items():
            if _strict_json_int(
                count, label=f"preparation ledger {table_name}.{name}"
            ) != 0:
                raise ValueError(
                    "machine manifest preparation denial counters are nonzero"
                )
    scalar_fields = expected_fields - {
        "passes",
        "forbidden_counters_zero",
        "per_shard_materialization",
        "denied_primary_reasons",
        "denied_modality_attempts",
        "denied_attempt_records",
    }
    for name in scalar_fields:
        count = value.get(name)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"machine manifest preparation ledger {name} is malformed")
    unique_source_paths = {str(entry["path"]) for entry in source_inventory["entries"]}
    exact_counts = {
        "panel_metadata_byte_opens": 1,
        "implementation_source_hash_byte_opens": 2 * len(_default_source_paths()),
        "document_hash_byte_opens": 4,
        "source_geometry_hash_byte_opens": 2 * len(unique_source_paths),
        "source_geometry_json_parses": sum(
            Path(path).suffix.lower() in {".json", ".jsonl"}
            for path in unique_source_paths
        ),
        "source_frame_records_selected": EXPECTED_FRAMES,
    }
    for name, expected in exact_counts.items():
        if value[name] != expected:
            raise ValueError(f"machine manifest preparation ledger {name} count changed")
    zero_fields = {
        "label_shard_hash_byte_opens",
        "label_shard_npz_opens",
        "registered_arrays_decompressed",
        "materialized_label_rows",
        "materialized_supervision_rows",
        "selected_label_rows_read",
        "selected_supervision_rows_read",
        "unselected_row_values_inspected",
        "unselected_row_metrics_computed",
        "unselected_rows_retained",
        "derivative_shard_or_cache_writes",
        "denied_attempts_total",
        "unexpected_path_attempts",
        *FORBIDDEN_ACCESS_FIELDS,
    }
    for name in zero_fields:
        if value[name] != 0:
            raise ValueError(f"machine manifest preparation ledger {name} is nonzero")
    if value["source_geometry_jsonl_records"] < EXPECTED_FRAMES:
        raise ValueError("machine manifest preparation scanned too few source frames")


def _validate_machine_manifest(
    payload: Mapping[str, Any],
    *,
    machine_file_sha256: str,
    source_hashes: Mapping[str, Mapping[str, str]],
) -> None:
    required = {
        "schema",
        "created_at_utc",
        "binding",
        "preflight_access_incident",
        "human_implementation_manifest",
        "authorized_inputs",
        "source_map",
        "runtime_environment",
        "verification_evidence",
        "exclusive_output",
        "preparation_access_ledger",
        "review",
        "authoritative_fit_audit_authorized",
        "content_sha256",
    }
    if set(payload) != required or payload.get("schema") != MACHINE_MANIFEST_SCHEMA:
        raise ValueError("machine implementation manifest schema/fields changed")
    _validate_utc_timestamp(payload.get("created_at_utc"))
    _validate_embedded_content_hash(
        payload,
        expected_sha256=str(payload.get("content_sha256", "")),
        label="machine implementation manifest",
    )
    if not _is_sha256(machine_file_sha256):
        raise ValueError("machine implementation manifest file hash is malformed")
    binding = payload.get("binding")
    incident = payload.get("preflight_access_incident")
    human = payload.get("human_implementation_manifest")
    if binding != {
        "path": str(BINDING_PATH),
        "file_sha256": EXECUTION_BINDING_SHA256,
    }:
        raise ValueError("machine manifest binding record changed")
    if incident != {
        "path": str(PREFLIGHT_INCIDENT_PATH),
        "file_sha256": PREFLIGHT_INCIDENT_SHA256,
        "status": PREFLIGHT_INCIDENT_STATUS,
    }:
        raise ValueError("machine manifest incident record changed")
    if (
        not isinstance(human, Mapping)
        or human.get("path") != str(IMPLEMENTATION_MANIFEST_PATH)
        or not _is_sha256(human.get("file_sha256"))
    ):
        raise ValueError("machine manifest human-report record changed")

    source_map = payload.get("source_map")
    if not isinstance(source_map, Mapping):
        raise ValueError("machine manifest lacks source map")
    expected_entries = [
        {"role": role, "path": record["path"], "sha256": record["sha256"]}
        for role, record in sorted(source_hashes.items())
    ]
    if source_map != {
        "entry_count": len(expected_entries),
        "entries": expected_entries,
        "source_map_sha256": canonical_json_sha256(expected_entries),
    }:
        raise ValueError("machine manifest source map differs from current sources")
    _validate_runtime_environment(payload.get("runtime_environment"))

    inputs = payload.get("authorized_inputs")
    expected_input_keys = {
        "fit_panel",
        "v4_adjudication_report",
        "known_bias_proof",
        "physical_geometry_contract",
        "label_shards",
        "render_summaries",
        "source_geometry",
    }
    if not isinstance(inputs, Mapping) or set(inputs) != expected_input_keys:
        raise ValueError("machine manifest authorized-input inventory changed")
    if inputs["fit_panel"] != {
        "semantic_role": "fit_panel",
        "path": str(PANEL_PATH),
        "file_sha256": PANEL_FILE_SHA256,
        "content_sha256": PANEL_CONTENT_SHA256,
        "fit_rows_sha256": FIT_ROWS_SHA256,
        "schema": "lewm_go2_physical_micro_overfit_panel_v1",
    }:
        raise ValueError("machine manifest fit-panel record changed")
    fixed_documents = {
        "v4_adjudication_report": (
            "v4_adjudication_report",
            V4_REPORT_PATH,
            V4_REPORT_SHA256,
        ),
        "known_bias_proof": (
            "known_bias_proof",
            KNOWN_BIAS_PROOF_PATH,
            KNOWN_BIAS_PROOF_SHA256,
        ),
    }
    for name, (semantic_role, path, digest) in fixed_documents.items():
        if inputs[name] != {
            "semantic_role": semantic_role,
            "path": str(path),
            "file_sha256": digest,
        }:
            raise ValueError(f"machine manifest {name} record changed")
    for name in ("physical_geometry_contract", "label_shards", "render_summaries", "source_geometry"):
        if not isinstance(inputs[name], Mapping):
            raise ValueError(f"machine manifest {name} inventory is malformed")
    if set(inputs["physical_geometry_contract"]) != {
        "semantic_role",
        "path",
        "file_sha256",
        "semantic_sha256",
        "schema",
    } or (
        inputs["physical_geometry_contract"].get("semantic_role")
        != "physical_geometry_contract"
        or not Path(str(inputs["physical_geometry_contract"].get("path", ""))).is_absolute()
        or not _is_sha256(inputs["physical_geometry_contract"].get("file_sha256"))
        or not _is_sha256(inputs["physical_geometry_contract"].get("semantic_sha256"))
        or inputs["physical_geometry_contract"].get("schema")
        != "lewm_go2_generalization_geometry_v2"
    ):
        raise ValueError("machine manifest physical geometry record changed")
    label_inventory = _validate_canonical_manifest(
        inputs["label_shards"], label="machine label-shard"
    )
    summary_inventory = _validate_canonical_manifest(
        inputs["render_summaries"], label="machine render-summary"
    )
    source_inventory = _validate_canonical_manifest(
        inputs["source_geometry"], label="machine source-geometry"
    )
    _validate_machine_label_inventory(label_inventory)
    _validate_machine_source_inventory(source_inventory)
    source_summaries = [
        entry
        for entry in source_inventory["entries"]
        if isinstance(entry, Mapping)
        and entry.get("semantic_role") == "fit_render_summary"
    ]
    if summary_inventory != _canonical_manifest(source_summaries):
        raise ValueError("machine render summaries do not reconcile to source geometry")
    if label_inventory["entry_count"] != EXPECTED_SHARDS:
        raise ValueError("machine label-shard inventory count changed")

    verification = payload.get("verification_evidence")
    if not isinstance(verification, Mapping) or set(verification) != {
        "all_passed",
        "commands",
    } or verification.get("all_passed") is not True:
        raise ValueError("machine manifest verification evidence is incomplete")
    commands = verification.get("commands")
    if not isinstance(commands, list) or len(commands) != len(
        REQUIRED_VERIFICATION_COMMANDS
    ):
        raise ValueError("machine manifest verification command count changed")
    for record, expected in zip(commands, REQUIRED_VERIFICATION_COMMANDS):
        deterministic = record.get("deterministic_result") if isinstance(
            record, Mapping
        ) else None
        if (
            not isinstance(record, Mapping)
            or set(record)
            != {
                "category",
                "command",
                "exit_code",
                "deterministic_result",
                "captured_output_sha256",
            }
            or record.get("category") != expected["category"]
            or record.get("command") != expected["command"]
            or not isinstance(deterministic, Mapping)
            or set(deterministic) != {"kind", "count"}
            or not isinstance(deterministic.get("kind"), str)
            or _strict_json_int(
                deterministic.get("count"), label="verification deterministic count"
            )
            != expected["deterministic_result"]["count"]
            or deterministic.get("kind")
            != expected["deterministic_result"]["kind"]
            or _strict_json_int(
                record.get("exit_code"), label="verification command exit_code"
            )
            != 0
            or not _is_sha256(record.get("captured_output_sha256"))
        ):
            raise ValueError("machine manifest verification evidence differs from the frozen suite")
    exclusive_output = payload.get("exclusive_output")
    if (
        not isinstance(exclusive_output, Mapping)
        or set(exclusive_output)
        != {"path", "schema", "absent_before_authorization", "zero_output_state"}
        or exclusive_output.get("absent_before_authorization") is not True
        or exclusive_output.get("zero_output_state") is not True
        or exclusive_output
        != {
        "path": str(OUTPUT_PATH),
        "schema": RESULT_SCHEMA,
        "absent_before_authorization": True,
        "zero_output_state": True,
        }
    ):
        raise ValueError("machine manifest exclusive-output record changed")
    preparation = payload.get("preparation_access_ledger")
    _validate_preparation_ledger(preparation, source_inventory=source_inventory)
    review = payload.get("review")
    if (
        not isinstance(review, Mapping)
        or not str(review.get("reviewer_identity", ""))
        or review.get("status") != "reviewed_and_authorized"
        or payload.get("authoritative_fit_audit_authorized") is not True
    ):
        raise PermissionError("machine manifest does not explicitly authorize the fit audit")


def _load_machine_manifest(
    expected_file_sha256: str,
    *,
    source_hashes: Mapping[str, Mapping[str, str]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    path = _authorize_path(
        MACHINE_IMPLEMENTATION_MANIFEST_PATH,
        ROOT,
        ledger=ledger,
        requested_role="machine_implementation_manifest",
        expected_resolved_path=MACHINE_IMPLEMENTATION_MANIFEST_PATH,
        label="machine implementation manifest",
    )
    ledger["document_hash_byte_opens"] += 1
    raw = _read_bytes(path)
    if _sha256_bytes(raw) != expected_file_sha256:
        raise ValueError("machine implementation manifest file SHA-256 changed")
    payload = _strict_json_bytes(raw, name="machine implementation manifest")
    if raw != _canonical_json_bytes(payload):
        raise ValueError(
            "machine implementation manifest is not canonical compact sorted-key UTF-8 JSON"
        )
    _validate_machine_manifest(
        payload,
        machine_file_sha256=expected_file_sha256,
        source_hashes=source_hashes,
    )
    return payload


def _verify_bound_document(
    path: Path,
    expected_sha256: str,
    *,
    requested_role: str,
    ledger: dict[str, Any],
    label: str,
) -> Path:
    resolved = _authorize_path(
        path,
        ROOT,
        ledger=ledger,
        requested_role=requested_role,
        expected_resolved_path=path,
        label=label,
    )
    ledger["document_hash_byte_opens"] += 1
    if _hash_file(resolved) != expected_sha256:
        raise ValueError(f"{label} file SHA-256 changed")
    return resolved


def prepare_manifest_inventory(
    *,
    authorization_sha256: str,
    human_manifest_sha256: str,
    spec: AuditSpec | None = None,
    ledger: dict[str, Any] | None = None,
    synthetic_test_only: bool = False,
) -> dict[str, Any]:
    """Discover only authorized metadata needed by the machine companion."""

    active = AuditSpec() if spec is None else spec
    access = new_access_ledger() if ledger is None else ledger
    if authorization_sha256 != EXECUTION_BINDING_SHA256:
        raise PermissionError("camera-frustum inventory authorization SHA-256 is wrong")
    if not _is_sha256(human_manifest_sha256):
        raise PermissionError("human-manifest authorization is malformed")
    if spec is not None and not synthetic_test_only:
        raise PermissionError("AuditSpec overrides are synthetic-test-only")

    source_start = _source_hashes(active.sources(), ledger=access)
    if source_start["binding"]["sha256"] != EXECUTION_BINDING_SHA256:
        raise ValueError("camera-frustum execution binding changed")
    _verify_bound_document(
        PREFLIGHT_INCIDENT_PATH,
        PREFLIGHT_INCIDENT_SHA256,
        requested_role="incident_record",
        ledger=access,
        label="preflight access incident",
    )
    _verify_bound_document(
        V4_REPORT_PATH,
        V4_REPORT_SHA256,
        requested_role="v4_adjudication_report",
        ledger=access,
        label="V4 adjudication report",
    )
    _verify_bound_document(
        KNOWN_BIAS_PROOF_PATH,
        KNOWN_BIAS_PROOF_SHA256,
        requested_role="known_bias_proof",
        ledger=access,
        label="KNOWN-bias proof",
    )
    _verify_bound_document(
        IMPLEMENTATION_MANIFEST_PATH,
        human_manifest_sha256,
        requested_role="human_implementation_manifest",
        ledger=access,
        label="human implementation manifest",
    )
    output_root = ROOT if not synthetic_test_only else active.output_path.parent
    output_path = _authorize_path(
        active.output_path,
        output_root,
        ledger=access,
        requested_role="audit_output",
        expected_resolved_path=(OUTPUT_PATH if not synthetic_test_only else active.output_path),
        label="exclusive audit output",
    )
    if output_path.exists():
        raise FileExistsError(f"immutable audit output already exists: {output_path}")

    records, panel_metadata = _load_panel(active, access)
    shard_entries, _grouped = _label_shard_manifest(
        records,
        spec=active,
        ledger=access,
    )
    source_frames, scenes, geometry, source_entries = _read_source_geometry(
        records,
        panel_metadata,
        spec=active,
        ledger=access,
        inventory_only=True,
    )
    if len(source_frames) != active.expected_frames or set(scenes) != {
        str(record["scene_id"]) for record in records
    }:
        raise ValueError("manifest inventory does not reconcile to selected fit frames")
    if any(
        int(access[name]) != 0
        for name in (
            "label_shard_hash_byte_opens",
            "label_shard_npz_opens",
            "registered_arrays_decompressed",
            "selected_label_rows_read",
            "selected_supervision_rows_read",
            *FORBIDDEN_ACCESS_FIELDS,
        )
    ):
        raise PermissionError("manifest preparation crossed its metadata-only boundary")
    source_end = _source_hashes(active.sources(), ledger=access)
    if source_end != source_start:
        raise RuntimeError("implementation sources changed during manifest preparation")
    forbidden_zero = all(int(access[name]) == 0 for name in FORBIDDEN_ACCESS_FIELDS)
    preparation_ledger = {
        **access,
        "passes": bool(
            forbidden_zero
            and int(access["unexpected_path_attempts"]) == 0
            and int(access["denied_attempts_total"]) == 0
        ),
        "forbidden_counters_zero": forbidden_zero,
    }
    render_summary_entries = [
        entry
        for entry in source_entries
        if entry["semantic_role"] == "fit_render_summary"
    ]
    source_map_entries = [
        {"role": role, "path": record["path"], "sha256": record["sha256"]}
        for role, record in sorted(source_end.items())
    ]
    authorized_inputs = {
        "fit_panel": {
            "semantic_role": "fit_panel",
            "path": str(PANEL_PATH.resolve()),
            "file_sha256": PANEL_FILE_SHA256,
            "content_sha256": PANEL_CONTENT_SHA256,
            "fit_rows_sha256": FIT_ROWS_SHA256,
            "schema": "lewm_go2_physical_micro_overfit_panel_v1",
        },
        "v4_adjudication_report": {
            "semantic_role": "v4_adjudication_report",
            "path": str(V4_REPORT_PATH.resolve()),
            "file_sha256": V4_REPORT_SHA256,
        },
        "known_bias_proof": {
            "semantic_role": "known_bias_proof",
            "path": str(KNOWN_BIAS_PROOF_PATH.resolve()),
            "file_sha256": KNOWN_BIAS_PROOF_SHA256,
        },
        "physical_geometry_contract": {
            "semantic_role": "physical_geometry_contract",
            "path": str(geometry["path"]),
            "file_sha256": str(geometry["file_sha256"]),
            "semantic_sha256": str(geometry["semantic_sha256"]),
            "schema": "lewm_go2_generalization_geometry_v2",
        },
        "label_shards": _canonical_manifest(shard_entries),
        "render_summaries": _canonical_manifest(render_summary_entries),
        "source_geometry": _canonical_manifest(source_entries),
    }
    core = {
        "schema": "lewm_go2_n32_camera_frustum_manifest_preparation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "binding": {
            "path": str(BINDING_PATH.resolve()),
            "file_sha256": EXECUTION_BINDING_SHA256,
        },
        "preflight_access_incident": {
            "path": str(PREFLIGHT_INCIDENT_PATH.resolve()),
            "file_sha256": PREFLIGHT_INCIDENT_SHA256,
            "status": PREFLIGHT_INCIDENT_STATUS,
        },
        "human_implementation_manifest": {
            "path": str(IMPLEMENTATION_MANIFEST_PATH.resolve()),
            "file_sha256": human_manifest_sha256,
        },
        "authorized_inputs": authorized_inputs,
        "source_map": {
            "entry_count": len(source_map_entries),
            "entries": source_map_entries,
            "source_map_sha256": canonical_json_sha256(source_map_entries),
        },
        "runtime_environment": _runtime_environment(),
        "exclusive_output": {
            "path": str(OUTPUT_PATH.resolve()),
            "schema": RESULT_SCHEMA,
            "absent_before_authorization": True,
            "zero_output_state": True,
        },
        "preparation_access_ledger": preparation_ledger,
    }
    return {**core, "content_sha256": canonical_json_sha256(core)}


def _family_box_scope(scene_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    fields = (
        "rendered_box_count",
        "collision_box_count",
        "matched_box_count",
        "unmatched_rendered_box_count",
        "unmatched_collision_box_count",
        "collision_boxes_affecting_selected_target_without_rendered_match_count",
        "rendered_collision_overlap_xor_cell_count",
        "required_provenance_missing_count",
        "required_provenance_nonunique_count",
    )
    return {
        "scene_count": len(scene_records),
        **{
            field: sum(int(record[field]) for record in scene_records)
            for field in fields
        },
    }


def _family_class_count_table(
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    def row(scope: str, selected: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "scope": scope,
            "frame_count": len(selected),
            **{
                class_name: sum(
                    int(report["label_support"]["class_counts"][class_name])
                    for report in selected
                )
                for class_name in ("unknown", "free", "occupied")
            },
        }

    rows = [row("aggregate", reports)] + [
        row(
            family,
            [report for report in reports if report["record_key"]["family"] == family],
        )
        for family in FAMILIES
    ]
    return {
        "family_order": list(FAMILIES),
        "rows": rows,
        "table_sha256": canonical_json_sha256(rows),
    }


def _atomic_write_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Publish one immutable JSON file with an atomic no-replace link."""

    destination = path.resolve(strict=False)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    nonce = hashlib.sha256(
        f"{os.getpid()}:{destination}:{len(encoded)}".encode("utf-8")
    ).hexdigest()[:16]
    temporary = destination.parent / f".{destination.name}.{nonce}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise FileExistsError(f"immutable audit output already exists: {destination}") from exc
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def run_authoritative_audit(
    *,
    authorization_sha256: str,
    machine_manifest_sha256: str,
    spec: AuditSpec | None = None,
    ledger: dict[str, Any] | None = None,
    synthetic_test_only: bool = False,
) -> dict[str, Any]:
    """Execute the frozen audit after exact binding/manifest authorization."""

    active = AuditSpec() if spec is None else spec
    access = new_access_ledger() if ledger is None else ledger
    if authorization_sha256 != EXECUTION_BINDING_SHA256:
        raise PermissionError("camera-frustum audit authorization SHA-256 is wrong")
    if not _is_sha256(machine_manifest_sha256):
        raise PermissionError("machine-manifest authorization is malformed")
    if spec is not None and not synthetic_test_only:
        raise PermissionError("AuditSpec overrides are synthetic-test-only")

    source_start = _source_hashes(active.sources(), ledger=access)
    if source_start["binding"]["sha256"] != EXECUTION_BINDING_SHA256:
        raise ValueError("camera-frustum execution binding changed")
    machine_manifest = _load_machine_manifest(
        machine_manifest_sha256,
        source_hashes=source_start,
        ledger=access,
    )
    human_record = machine_manifest["human_implementation_manifest"]
    _verify_bound_document(
        PREFLIGHT_INCIDENT_PATH,
        PREFLIGHT_INCIDENT_SHA256,
        requested_role="incident_record",
        ledger=access,
        label="preflight access incident",
    )
    _verify_bound_document(
        V4_REPORT_PATH,
        V4_REPORT_SHA256,
        requested_role="v4_adjudication_report",
        ledger=access,
        label="V4 adjudication report",
    )
    _verify_bound_document(
        KNOWN_BIAS_PROOF_PATH,
        KNOWN_BIAS_PROOF_SHA256,
        requested_role="known_bias_proof",
        ledger=access,
        label="KNOWN-bias proof",
    )
    _verify_bound_document(
        IMPLEMENTATION_MANIFEST_PATH,
        str(human_record["file_sha256"]),
        requested_role="human_implementation_manifest",
        ledger=access,
        label="human implementation manifest",
    )
    output_root = ROOT if not synthetic_test_only else active.output_path.parent
    output_path = _authorize_path(
        active.output_path,
        output_root,
        ledger=access,
        requested_role="audit_output",
        expected_resolved_path=(OUTPUT_PATH if not synthetic_test_only else active.output_path),
        label="exclusive audit output",
    )
    if output_path.exists():
        raise FileExistsError(f"immutable audit output already exists: {output_path}")
    if not synthetic_test_only:
        _load_authorized_semantics(source_start)
    elif not _SEMANTICS_LOADED:
        raise RuntimeError("synthetic test did not install semantic modules")

    records, panel_metadata = _load_panel(active, access)
    shard_entries, grouped_shards = _label_shard_manifest(
        records,
        spec=active,
        ledger=access,
    )
    if not synthetic_test_only:
        authorized_inputs = machine_manifest["authorized_inputs"]
        if authorized_inputs["label_shards"] != _canonical_manifest(shard_entries):
            raise ValueError("selected label shards differ from the machine manifest")
        panel_geometry = panel_metadata["geometry_contract"]
        geometry_lexical = Path(str(panel_geometry["path"]))
        if not geometry_lexical.is_absolute():
            geometry_lexical = ROOT / geometry_lexical
        if authorized_inputs["physical_geometry_contract"] != {
            "semantic_role": "physical_geometry_contract",
            "path": str(geometry_lexical),
            "file_sha256": str(panel_geometry["file_sha256"]),
            "semantic_sha256": str(panel_geometry["semantic_sha256"]),
            "schema": "lewm_go2_generalization_geometry_v2",
        }:
            raise ValueError("physical geometry differs from the machine manifest")
    selected_labels = _read_selected_labels_once(grouped_shards, ledger=access)
    source_frames, scenes, source_geometry_contract, source_entries = _read_source_geometry(
        records,
        panel_metadata,
        spec=active,
        ledger=access,
        authorized_source_entries=(
            None
            if synthetic_test_only
            else machine_manifest["authorized_inputs"]["source_geometry"]["entries"]
        ),
    )

    mapping = build_camera_centered_mapping()
    mapping_audit = audit_camera_centered_mapping(mapping)
    old_span = old_body_column_span_audit()
    full_core_reports: list[dict[str, Any]] = []
    compact_reports: list[dict[str, Any]] = []
    veto_identities: list[list[Any]] = []
    mismatch_identities: list[list[Any]] = []
    attribution_partition_identities: dict[str, list[list[Any]]] = {
        name: [] for name in ATTRIBUTION_PARTITION_NAMES
    }
    selected_label_digest = hashlib.sha256()
    frame_geometry_records: list[dict[str, Any]] = []
    camera_evidence_records: list[dict[str, Any]] = []
    scene_xor_counts: Counter[str] = Counter()
    scene_unmatched_affecting: dict[str, set[int]] = {
        scene_id: set() for scene_id in scenes
    }

    for record in records:
        identity = tuple(_frame_identity_values(record))
        target, supervision = selected_labels[identity]
        selected_label_digest.update(target.tobytes(order="C"))
        key = _frame_key(record)
        core_report = analyze_frame_labels(
            target,
            supervision,
            frame_key=key,
            family=str(record["family"]),
            endpoint_side=str(record["side"]),
            mapping=mapping,
        )
        full_core_reports.append(core_report)
        scene = scenes[str(record["scene_id"])]
        frame = source_frames[identity]
        camera_evidence = dict(frame["camera_mount_composition"])
        camera_evidence_records.append(
            {"record_key": key, "camera_mount_composition": camera_evidence}
        )
        stages = _reconstruct_label_stages(
            scene["manifest"],
            frame,
            rendered_boxes=scene["rendered_boxes"],
            collision_boxes=scene["collision_boxes"],
            geometry_flags=source_geometry_contract["flags"],
            camera_projection=scene["camera_projection"],
            physical_grid=scene["physical_grid"],
        )
        mismatch = np.asarray(stages["final"] != target, dtype=bool)
        veto = (
            (target == UNKNOWN_CLASS)
            & (stages["pre_veto"] == FREE_CLASS)
            & stages["collision_overlap"]
        )
        matching = scene["box_matching"]
        matched_collision = tuple(
            scene["collision_boxes"][collision_index]
            for _rendered_index, collision_index in matching["matches"]
        )
        unmatched_collision = tuple(
            scene["collision_boxes"][index]
            for index in matching["unmatched_collision_indices"]
        )
        matched_overlap = _overlap_for_boxes(stages, frame, matched_collision)
        unmatched_overlap = _overlap_for_boxes(stages, frame, unmatched_collision)
        attribution_partitions = _attribution_partition_records(
            veto,
            matched_overlap,
            unmatched_overlap,
        )
        xor = np.asarray(stages["rendered_overlap"] ^ stages["collision_overlap"], dtype=bool)
        scene_xor_counts[str(record["scene_id"])] += int(np.count_nonzero(xor))
        for collision_index in matching["unmatched_collision_indices"]:
            one = _overlap_for_boxes(
                stages, frame, (scene["collision_boxes"][collision_index],)
            )
            if bool(np.any(one & supervision)):
                scene_unmatched_affecting[str(record["scene_id"])].add(int(collision_index))
        ambiguity_cells = mismatch | (unmatched_overlap & supervision)

        for row, column in np.argwhere(veto):
            veto_identities.append([_frame_identity_values(record), int(row), int(column)])
        for row, column in np.argwhere(mismatch):
            mismatch_identities.append([_frame_identity_values(record), int(row), int(column)])
        for name, partition in attribution_partitions.items():
            for row, column in partition["identities"]:
                attribution_partition_identities[name].append(
                    [_frame_identity_values(record), int(row), int(column)]
                )
        compact = _compact_core_frame(core_report)
        per_frame_veto_identities = _mask_row_column_identities(veto)
        per_frame_mismatch_identities = _mask_row_column_identities(mismatch)
        geometry_record = {
            "record_key": key,
            "collision_veto_only_unknown_count": len(per_frame_veto_identities),
            "collision_veto_only_unknown_distance_bin_counts": _distance_bin_counts(veto),
            "collision_veto_only_unknown_identities": per_frame_veto_identities,
            "attributed_to_matched_box_count": (
                attribution_partitions["matched_true_unmatched_false"]["count"]
                + attribution_partitions["matched_true_unmatched_true"]["count"]
            ),
            "depends_on_unmatched_collision_box_count": (
                attribution_partitions["matched_false_unmatched_true"]["count"]
                + attribution_partitions["matched_true_unmatched_true"]["count"]
            ),
            "attribution_partitions": attribution_partitions,
            "reconstruction_mismatch_cell_count": len(per_frame_mismatch_identities),
            "reconstruction_mismatch_identities": per_frame_mismatch_identities,
            "rendered_collision_overlap_xor_cell_count": int(np.count_nonzero(xor)),
            "geometry_ambiguity_cell_count": int(np.count_nonzero(ambiguity_cells)),
        }
        frame_geometry_records.append(geometry_record)
        compact.update(
            {
                name: geometry_record[name]
                for name in (
                    "collision_veto_only_unknown_count",
                    "collision_veto_only_unknown_distance_bin_counts",
                    "collision_veto_only_unknown_identities",
                    "attributed_to_matched_box_count",
                    "depends_on_unmatched_collision_box_count",
                    "attribution_partitions",
                    "reconstruction_mismatch_cell_count",
                    "reconstruction_mismatch_identities",
                    "rendered_collision_overlap_xor_cell_count",
                    "geometry_ambiguity_cell_count",
                )
            }
        )
        compact["camera_mount_composition"] = camera_evidence
        compact_reports.append(compact)

    label_observability = aggregate_label_observability(full_core_reports)
    label_observability["representative_scalar_first_hit_violations"] = (
        _representative_ray_violations(full_core_reports)
    )
    collision_veto = {
        "aggregate": _scope_collision(frame_geometry_records),
        "families": {
            family: _scope_collision(
                [record for record in frame_geometry_records if record["record_key"]["family"] == family]
            )
            for family in FAMILIES
        },
        "attribution_partitions": {
            name: {
                "count": len(attribution_partition_identities[name]),
                "identities": attribution_partition_identities[name],
                "identities_sha256": canonical_json_sha256(
                    attribution_partition_identities[name]
                ),
            }
            for name in ATTRIBUTION_PARTITION_NAMES
        },
        "ordered_cell_identities": veto_identities,
        "ordered_cell_identities_sha256": canonical_json_sha256(veto_identities),
    }

    scene_box_records = []
    for scene_id in sorted(
        scenes,
        key=lambda value: (FAMILIES.index(str(scenes[value]["family"])), str(value)),
    ):
        scene = scenes[scene_id]
        matching = scene["box_matching"]
        scene_box_records.append(
            {
                "scene_id": scene_id,
                "family": scene["family"],
                "rendered_box_count": len(scene["rendered_boxes"]),
                "collision_box_count": len(scene["collision_boxes"]),
                "matched_box_count": len(matching["matches"]),
                "unmatched_rendered_box_count": len(matching["unmatched_rendered_indices"]),
                "unmatched_collision_box_count": len(matching["unmatched_collision_indices"]),
                "unmatched_rendered_boxes": matching["unmatched_rendered_boxes"],
                "unmatched_collision_boxes": matching["unmatched_collision_boxes"],
                "matched_multiplicities": matching["matched_multiplicities"],
                "collision_boxes_affecting_selected_target_without_rendered_match_count": len(
                    scene_unmatched_affecting[scene_id]
                ),
                "collision_boxes_affecting_selected_target_without_rendered_match": [
                    record
                    for record in matching["unmatched_collision_boxes"]
                    if int(record["index"]) in scene_unmatched_affecting[scene_id]
                ],
                "rendered_collision_overlap_xor_cell_count": int(scene_xor_counts[scene_id]),
                "required_provenance_missing_count": int(scene["required_provenance_missing_count"]),
                "required_provenance_nonunique_count": int(scene["required_provenance_nonunique_count"]),
            }
        )
    selected_scene_pairs = {
        (str(record["scene_id"]), str(record["family"])) for record in records
    }
    box_scene_pairs = {
        (str(record["scene_id"]), str(record["family"]))
        for record in scene_box_records
    }
    if box_scene_pairs != selected_scene_pairs:
        raise RuntimeError("box-parity scenes do not equal selected fit scenes")
    selected_scene_ids = {scene_id for scene_id, _family in selected_scene_pairs}
    if any(str(entry["scene_id"]) not in selected_scene_ids for entry in source_entries):
        raise RuntimeError("source geometry contains an unselected scene assignment")
    box_parity = {
        "aggregate": _family_box_scope(scene_box_records),
        "families": {
            family: _family_box_scope(
                [record for record in scene_box_records if record["family"] == family]
            )
            for family in FAMILIES
        },
        "scenes": scene_box_records,
        "ordered_box_parity_table_sha256": canonical_json_sha256(scene_box_records),
    }
    reconstruction = {
        "frame_count": len(records),
        "passes": not mismatch_identities,
        "mismatch_frame_count": sum(
            int(record["reconstruction_mismatch_cell_count"]) > 0
            for record in frame_geometry_records
        ),
        "mismatch_cell_count": len(mismatch_identities),
        "mismatch_identities": mismatch_identities,
        "mismatch_identities_sha256": canonical_json_sha256(mismatch_identities),
    }
    camera_mount_composition = {
        "frame_count": len(camera_evidence_records),
        "pass_count": sum(
            record["camera_mount_composition"]["passes"] is True
            for record in camera_evidence_records
        ),
        "failure_count": sum(
            record["camera_mount_composition"]["passes"] is not True
            for record in camera_evidence_records
        ),
        "passes": all(
            record["camera_mount_composition"]["passes"] is True
            for record in camera_evidence_records
        ),
        "ordered_frame_evidence_sha256": canonical_json_sha256(
            camera_evidence_records
        ),
    }
    rendered_collision_target_ambiguity = bool(
        mismatch_identities
        or not camera_mount_composition["passes"]
        or collision_veto["aggregate"]["depends_on_unmatched_collision_box_count"]
        or box_parity["aggregate"][
            "collision_boxes_affecting_selected_target_without_rendered_match_count"
        ]
        or box_parity["aggregate"]["required_provenance_missing_count"]
        or box_parity["aggregate"]["required_provenance_nonunique_count"]
    )

    source_end = _source_hashes(active.sources(), ledger=access)
    source_hashes_pass = source_end == source_start
    if not source_hashes_pass:
        raise RuntimeError("audit implementation sources changed during execution")
    forbidden_zero = all(int(access[name]) == 0 for name in FORBIDDEN_ACCESS_FIELDS)
    denial_arithmetic_passes = (
        sum(int(value) for value in access["denied_primary_reasons"].values())
        == int(access["denied_attempts_total"])
        == int(access["unexpected_path_attempts"])
    )
    unselected_zero = all(
        int(access[name]) == 0
        for name in (
            "unselected_row_values_inspected",
            "unselected_row_metrics_computed",
            "unselected_rows_retained",
            "derivative_shard_or_cache_writes",
        )
    )
    access_reconciliation = {
        "passes": bool(
            access["panel_metadata_byte_opens"] == 1
            and access["label_shard_hash_byte_opens"] == active.expected_shards
            and access["label_shard_npz_opens"] == active.expected_shards
            and access["registered_arrays_decompressed"] == 4 * active.expected_shards
            and access["materialized_label_rows"]
            == access["materialized_supervision_rows"]
            and access["selected_label_rows_read"] == active.expected_frames
            and access["selected_supervision_rows_read"] == active.expected_frames
            and access["source_frame_records_selected"] == active.expected_frames
            and forbidden_zero
            and unselected_zero
            and denial_arithmetic_passes
            and access["unexpected_path_attempts"] == 0
        ),
        "expected_distinct_label_shards": active.expected_shards,
        "expected_selected_label_rows": active.expected_frames,
        "forbidden_counters_zero": forbidden_zero,
        "unselected_row_counters_zero": unselected_zero,
        "denial_arithmetic_passes": denial_arithmetic_passes,
        "unexpected_paths_zero": access["unexpected_path_attempts"] == 0,
        "one_npz_open_per_shard": access["label_shard_npz_opens"] == active.expected_shards,
    }
    provenance_complete = (
        box_parity["aggregate"]["required_provenance_missing_count"] == 0
        and box_parity["aggregate"]["required_provenance_nonunique_count"] == 0
    )
    provenance = {
        "passes": bool(provenance_complete and camera_mount_composition["passes"]),
        "fit_panel_file_hash_pass": True,
        "fit_panel_content_hash_pass": True,
        "current_physical_dataset_role_train_only": True,
        "fit_frame_identity_unique": True,
        "one_to_one_frame_match": access["source_frame_records_selected"] == active.expected_frames,
        "source_hashes_pass": source_hashes_pass,
        "source_geometry_allowlisted_before_parse": access["unexpected_path_attempts"] == 0,
        "source_geometry_rehashed_after_parse": True,
        "rendered_collision_provenance_complete": provenance_complete,
        "fixed_camera_mount_composition_complete": camera_mount_composition["passes"],
        "legacy_source_split_used_for_selection": False,
    }
    decision = authorization_decision(
        provenance_passes=provenance["passes"],
        source_hashes_pass=source_hashes_pass,
        reconstruction_passes=reconstruction["passes"],
        access_reconciliation_passes=access_reconciliation["passes"],
        mapping_audit=mapping_audit,
        label_observability=label_observability,
        rendered_collision_target_ambiguity=rendered_collision_target_ambiguity,
    )

    frame_identity_values = [_frame_identity_values(record) for record in records]
    human_manifest_record = dict(machine_manifest["human_implementation_manifest"])
    machine_manifest_record = {
        "path": str(MACHINE_IMPLEMENTATION_MANIFEST_PATH.resolve()),
        "file_sha256": machine_manifest_sha256,
        "content_sha256": str(machine_manifest["content_sha256"]),
        "schema": MACHINE_MANIFEST_SCHEMA,
    }
    incident_record = {
        "path": str(PREFLIGHT_INCIDENT_PATH.resolve()),
        "file_sha256": PREFLIGHT_INCIDENT_SHA256,
        "status": PREFLIGHT_INCIDENT_STATUS,
    }
    preparation_ledger = dict(machine_manifest["preparation_access_ledger"])
    unique_source_paths = {str(entry["path"]) for entry in source_entries}
    expected_finalizer_ledger = {
        "panel_metadata_byte_opens": 1,
        "document_hash_byte_opens": 16,
        "label_shard_hash_byte_opens": active.expected_shards,
        "label_shard_npz_opens": active.expected_shards,
        "registered_arrays_decompressed": 4 * active.expected_shards,
        "materialized_label_rows": int(access["materialized_label_rows"]),
        "materialized_supervision_rows": int(
            access["materialized_supervision_rows"]
        ),
        "selected_label_rows_read": active.expected_frames,
        "selected_supervision_rows_read": active.expected_frames,
        "source_geometry_hash_byte_opens": len(unique_source_paths),
        "source_geometry_json_parses": sum(
            Path(path).suffix.lower() in {".json", ".jsonl"}
            for path in unique_source_paths
        ),
        "source_geometry_jsonl_records": int(
            access["source_geometry_jsonl_records"]
        ),
        "denied_attempts_total": 0,
        "unexpected_path_attempts": 0,
        "unselected_row_values_inspected": 0,
        "unselected_row_metrics_computed": 0,
        "unselected_rows_retained": 0,
        "derivative_shard_or_cache_writes": 0,
        **{name: 0 for name in FORBIDDEN_ACCESS_FIELDS},
    }
    two_phase_access_reconciliation = {
        "phase_names": ["preparation", "runner"],
        "passes": bool(
            preparation_ledger.get("passes") is True
            and access_reconciliation["passes"]
        ),
        "forbidden_counters_zero": bool(
            preparation_ledger.get("forbidden_counters_zero", True)
            and forbidden_zero
        ),
        "unexpected_paths_zero": bool(
            int(preparation_ledger.get("unexpected_path_attempts", 0)) == 0
            and int(access["unexpected_path_attempts"]) == 0
        ),
        "incident_separate": True,
        "expected_distinct_label_shards": active.expected_shards,
        "selected_label_rows_each": active.expected_frames,
        "selected_supervision_rows_each": active.expected_frames,
        "source_geometry_unique_path_count": len(unique_source_paths),
    }
    core = {
        "schema": RESULT_SCHEMA,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": {
            "dataset_role": "train",
            "transition_count": active.expected_transitions,
            "frame_count": active.expected_frames,
            "families": list(FAMILIES),
            "endpoint_sides": list(ENDPOINT_SIDES),
            "learning_performed": False,
        },
        "execution_binding": {
            "path": str(BINDING_PATH.resolve()),
            "sha256": EXECUTION_BINDING_SHA256,
        },
        "preflight_access_incident": incident_record,
        "human_implementation_manifest": human_manifest_record,
        "machine_implementation_manifest": machine_manifest_record,
        "runtime_environment": _runtime_environment(),
        "inputs": {
            "fit_panel": {
                "path": str(active.panel_path.resolve()),
                "file_sha256": active.panel_file_sha256,
                "content_sha256": active.panel_content_sha256,
                "fit_rows_sha256": active.fit_rows_sha256,
            },
            "human_implementation_manifest": human_manifest_record,
            "machine_implementation_manifest": machine_manifest_record,
            "preflight_access_incident": incident_record,
            "v4_adjudication_report": {
                "path": str(V4_REPORT_PATH.resolve()),
                "file_sha256": V4_REPORT_SHA256,
            },
            "known_bias_proof": {
                "path": str(KNOWN_BIAS_PROOF_PATH.resolve()),
                "file_sha256": KNOWN_BIAS_PROOF_SHA256,
            },
            "geometry_contract": {
                key: source_geometry_contract[key]
                for key in ("path", "file_sha256", "semantic_sha256")
            },
        },
        "source_hashes": source_end,
        "geometry_contract": frozen_camera_geometry_contract(),
        "old_body_column_span_audit": old_span,
        "mapping_audit": mapping_audit,
        "frame_identity": {
            "count": len(records),
            "encoding_fields": list(FRAME_IDENTITY_FIELDS),
            "sha256": canonical_json_sha256(frame_identity_values),
        },
        "source_geometry_manifest": {
            "entry_count": len(source_entries),
            "entries": source_entries,
            "manifest_sha256": canonical_json_sha256(source_entries),
        },
        "label_shard_manifest": {
            "entry_count": len(shard_entries),
            "entries": shard_entries,
            "manifest_sha256": canonical_json_sha256(shard_entries),
        },
        "selected_label_bytes": {
            "frame_count": len(records),
            "encoding": "canonical_frame_order_contiguous_row_major_uint8_targets_only",
            "byte_count": len(records) * 64 * 64,
            "sha256": selected_label_digest.hexdigest(),
        },
        "family_class_count_table": _family_class_count_table(compact_reports),
        "frame_reports": compact_reports,
        "label_observability": label_observability,
        "reconstruction": reconstruction,
        "collision_veto": collision_veto,
        "box_parity": box_parity,
        "camera_mount_composition": camera_mount_composition,
        "provenance": provenance,
        "phase_ledgers": {
            "preparation": preparation_ledger,
            "runner": access,
        },
        "expected_finalizer_ledger": expected_finalizer_ledger,
        "two_phase_access_reconciliation": two_phase_access_reconciliation,
        "rendered_collision_target_ambiguity": rendered_collision_target_ambiguity,
        "authorization_decision": decision,
        "licenses": {
            "n32_passed": False,
            "g2_passed": False,
            "trained_model_output_authorized": False,
            "holdout_access_authorized": False,
            "seed_20260711_authorized": False,
            "runtime_authorized": False,
            "promotion_authorized": False,
        },
    }
    payload = {**core, "content_sha256": canonical_json_sha256(core)}
    _atomic_write_json_exclusive(active.output_path, payload)
    return payload


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authorization-sha256", required=True)
    parser.add_argument("--machine-manifest-sha256")
    parser.add_argument("--prepare-manifest-inventory", action="store_true")
    parser.add_argument("--human-manifest-sha256")
    args = parser.parse_args(argv)
    if args.prepare_manifest_inventory:
        if args.human_manifest_sha256 is None or args.machine_manifest_sha256 is not None:
            parser.error(
                "manifest preparation requires --human-manifest-sha256 and forbids "
                "--machine-manifest-sha256"
            )
    elif args.machine_manifest_sha256 is None or args.human_manifest_sha256 is not None:
        parser.error(
            "authoritative audit requires --machine-manifest-sha256 and forbids "
            "--human-manifest-sha256"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.prepare_manifest_inventory:
        inventory = prepare_manifest_inventory(
            authorization_sha256=str(args.authorization_sha256),
            human_manifest_sha256=str(args.human_manifest_sha256),
        )
        print(
            json.dumps(inventory, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
        return 0
    result = run_authoritative_audit(
        authorization_sha256=str(args.authorization_sha256),
        machine_manifest_sha256=str(args.machine_manifest_sha256),
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_PATH),
                "content_sha256": result["content_sha256"],
                "camera_frustum_representation_implementation_authorized": result[
                    "authorization_decision"
                ]["camera_frustum_representation_implementation_authorized"],
                "target_amendment_required_before_model_output": result[
                    "authorization_decision"
                ]["target_amendment_required_before_model_output"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
