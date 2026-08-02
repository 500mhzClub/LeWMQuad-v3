#!/usr/bin/env python3
"""Strict join for one bounded Go2 counterfactual pilot collection.

The joiner consumes one caller-bound train/eval physics collection plus one
caller-bound successful calibration receipt.  For textured-v03 evidence it
opens every bound RGB leaf through the checker's no-follow reader and
independently recomputes the decoded raw-pixel identity before emitting a
manifest.  It never opens runtime inputs or checkpoints.  All emitted rows
preserve requested action identity separately from the future executed command
tape.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as producer_contract  # noqa: E402
from scripts import analyze_go2_world_model_counterfactual_calibration_v1 as calibration  # noqa: E402
from scripts import check_go2_world_model_counterfactual_pilot_v1 as checker  # noqa: E402


MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_pilot_manifest_v1"
RGB_MANIFEST_SCHEMA = "lewm_go2_world_model_counterfactual_rgb_manifest_v1"
GROUP_SCHEMA = "lewm_go2_world_model_counterfactual_group_v1"
ACTION_COUNT = 9
ROLE_NAMES = ("train", "eval")
LEGACY_RENDER_PROFILE = "legacy_v1"
TEXTURED_V03_RENDER_PROFILE = "textured_v03_v3"


class PilotJoinError(RuntimeError):
    """Raised before a malformed collection can mint a final pilot manifest."""


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _relative_binding(path: Path, *, root: Path) -> dict[str, object]:
    root_input = Path(root)
    if root_input.is_symlink():
        raise PilotJoinError(f"joined root is a symlink: {root_input}")
    root_resolved = root_input.resolve(strict=True)
    selected = Path(os.path.abspath(os.fspath(path)))
    try:
        relative = selected.relative_to(root_resolved)
    except ValueError as exc:
        raise PilotJoinError(f"joined file escapes pilot root: {selected}") from exc
    if not relative.parts:
        raise PilotJoinError(f"joined file is not a leaf: {selected}")
    directory_flags = (
        os.O_RDONLY
        | os.O_DIRECTORY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(root_resolved, directory_flags)
    file_descriptor = None
    try:
        for component in relative.parts[:-1]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        file_descriptor = os.open(
            relative.parts[-1], file_flags, dir_fd=descriptor
        )
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise PilotJoinError(f"joined file is not regular: {selected}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
    except OSError as exc:
        raise PilotJoinError(
            f"cannot safely bind joined file: {selected}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    identity_fields = (
        "st_dev",
        "st_ino",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if tuple(getattr(before, field) for field in identity_fields) != tuple(
        getattr(after, field) for field in identity_fields
    ):
        raise PilotJoinError(f"joined file changed while binding: {selected}")
    raw = b"".join(chunks)
    if len(raw) != before.st_size:
        raise PilotJoinError(f"joined file length changed while binding: {selected}")
    return {
        "path": relative.as_posix(),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
    }


def _write_bytes_exclusive(path: Path, raw: bytes) -> None:
    selected = Path(path)
    if selected.exists() or selected.is_symlink():
        raise FileExistsError(f"refusing to overwrite joined receipt: {selected}")
    selected.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        selected,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o644,
    )
    try:
        written = 0
        while written < len(raw):
            written += os.write(descriptor, raw[written:])
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json_exclusive(path: Path, value: Mapping[str, object]) -> None:
    _write_bytes_exclusive(path, _canonical_json_bytes(value) + b"\n")


def _quantize(value: float, tolerance: float) -> int:
    magnitude = math.floor(abs(value) / tolerance + 0.5)
    return magnitude if value >= 0.0 else -magnitude


def _dense_ranks(
    branches: Sequence[Mapping[str, Any]],
    *,
    progress_tolerance_m: float,
    path_length_tolerance_m: float,
) -> list[int]:
    keys = []
    for branch in branches:
        keys.append((
            int(branch["physical_fell"]),
            int(branch["physical_tipped"]),
            -_quantize(
                float(branch["physical_target_progress_m"]),
                progress_tolerance_m,
            ),
            _quantize(
                float(branch["physical_path_length_m"]),
                path_length_tolerance_m,
            ),
        ))
    mapping = {key: rank for rank, key in enumerate(sorted(set(keys)))}
    return [mapping[key] for key in keys]


def _artifact_from_frame_receipt(receipt: Mapping[str, Any]) -> dict[str, object]:
    expected = {
        "artifact_id",
        "frame_identity",
        "path",
        "file_sha256",
        "byte_count",
        "width",
        "height",
        "mode",
        "format",
        "camera_valid",
        "low_information",
        "low_info_reasons",
    }
    observed = set(receipt)
    if observed not in (expected, expected | {"pixel_sha256"}):
        raise PilotJoinError("frame receipt field set changed")
    return dict(receipt)


def _inert_binding(value: object, *, label: str) -> dict[str, object]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "file_sha256", "byte_count"}
        or not isinstance(value.get("path"), str)
        or not value["path"]
        or not isinstance(value.get("file_sha256"), str)
        or len(value["file_sha256"]) != 64
        or any(character not in "0123456789abcdef" for character in value["file_sha256"])
        or type(value.get("byte_count")) is not int
        or int(value["byte_count"]) <= 0
    ):
        raise PilotJoinError(f"{label} binding changed")
    return dict(value)


def _collection_render_lineage(
    collection: Mapping[str, Any], *, label: str
) -> tuple[
    str,
    dict[str, object] | None,
    dict[str, object] | None,
    dict[str, object] | None,
]:
    """Derive render generation from checker-validated plan and state receipts."""

    plan = collection.get("plan")
    document = plan.get("document") if isinstance(plan, Mapping) else None
    states = collection.get("states")
    if (
        not isinstance(document, Mapping)
        or not isinstance(states, Sequence)
        or isinstance(states, (str, bytes))
        or not states
        or any(not isinstance(state, Mapping) for state in states)
    ):
        raise PilotJoinError(f"{label} render lineage is absent")
    state_schemas = [
        state.get("document", {}).get("schema")
        if isinstance(state.get("document"), Mapping)
        else None
        for state in states
    ]
    render_contract = document.get("render_contract")
    parity_fields = (
        "visual_domain_parity_result_binding",
        "visual_domain_parity_terminal_binding",
        "visual_domain_parity_review_binding",
    )
    present_parity_fields = {
        field for field in parity_fields if field in document
    }
    if render_contract == producer_contract.TEXTURED_V03_RENDER_CONTRACT:
        if any(
            schema != producer_contract.TEXTURED_V03_STATE_RECEIPT_SCHEMA
            for schema in state_schemas
        ) or present_parity_fields != set(parity_fields):
            raise PilotJoinError(f"{label} textured-v03 identity is only partial")
        return (
            TEXTURED_V03_RENDER_PROFILE,
            _inert_binding(
                document["visual_domain_parity_result_binding"],
                label=f"{label} visual-domain parity result",
            ),
            _inert_binding(
                document["visual_domain_parity_terminal_binding"],
                label=f"{label} visual-domain parity terminal",
            ),
            _inert_binding(
                document["visual_domain_parity_review_binding"],
                label=f"{label} visual-domain parity review",
            ),
        )
    if render_contract == producer_contract.RENDER_CONTRACT:
        if any(
            schema != producer_contract.STATE_RECEIPT_SCHEMA
            for schema in state_schemas
        ) or present_parity_fields:
            raise PilotJoinError(f"{label} legacy identity is only partial")
        return LEGACY_RENDER_PROFILE, None, None, None
    raise PilotJoinError(f"{label} render contract is unsupported")


def _calibration_render_profile(receipt: Mapping[str, Any]) -> str:
    schema = receipt.get("schema")
    if schema == calibration.TEXTURED_V03_CALIBRATION_RECEIPT_SCHEMA:
        return TEXTURED_V03_RENDER_PROFILE
    if schema == calibration.CALIBRATION_RECEIPT_SCHEMA:
        return LEGACY_RENDER_PROFILE
    raise PilotJoinError("calibration receipt schema is unsupported")


def _load_pixel_verified_collection(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_byte_count: int,
) -> dict[str, Any]:
    """Cross the RGB boundary only through the exact decoded-pixel verifier."""

    try:
        return checker.load_bound_collection_receipts(
            path,
            expected_file_sha256=expected_file_sha256,
            expected_byte_count=expected_byte_count,
            verify_textured_pixels=True,
        )
    except checker.PilotReceiptError as exc:
        raise PilotJoinError(str(exc)) from exc


def build_joined_documents_v1(
    collection: Mapping[str, Any],
    calibration_receipt: Mapping[str, Any],
    *,
    calibration_visual_domain_parity_result_binding: Mapping[str, Any] | None = None,
    calibration_visual_domain_parity_terminal_binding: Mapping[str, Any] | None = None,
    calibration_visual_domain_parity_review_binding: Mapping[str, Any] | None = None,
) -> tuple[
    dict[str, object],
    dict[str, list[dict[str, object]]],
    dict[str, object],
]:
    """Build deterministic manifest-independent rows from validated receipts."""

    if collection.get("purpose") != "bounded_wm_a_pilot":
        raise PilotJoinError("pilot join requires a bounded_wm_a_pilot collection")
    if calibration_receipt.get("decision") != "FREEZE_PILOT_CONTRACT":
        raise PilotJoinError("calibration did not freeze the pilot contract")
    (
        collection_profile,
        collection_parity_result,
        collection_parity_terminal,
        collection_parity_review,
    ) = _collection_render_lineage(
        collection, label="bounded collection"
    )
    calibration_profile = _calibration_render_profile(calibration_receipt)
    calibration_collection_binding = _inert_binding(
        calibration_receipt.get("calibration_collection_receipt"),
        label="calibration collection receipt",
    )
    if collection_profile != calibration_profile:
        raise PilotJoinError(
            "bounded collection and calibration receipt render profiles differ"
        )
    if collection_profile == TEXTURED_V03_RENDER_PROFILE:
        calibration_parity_result = _inert_binding(
            calibration_visual_domain_parity_result_binding,
            label="calibration visual-domain parity result",
        )
        calibration_parity_terminal = _inert_binding(
            calibration_visual_domain_parity_terminal_binding,
            label="calibration visual-domain parity terminal",
        )
        calibration_parity_review = _inert_binding(
            calibration_visual_domain_parity_review_binding,
            label="calibration visual-domain parity review",
        )
        receipt_prerequisites = calibration_receipt.get(
            "visual_domain_parity_prerequisites"
        )
        if (
            not isinstance(receipt_prerequisites, Mapping)
            or set(receipt_prerequisites)
            != {"result_binding", "terminal_binding", "review_binding"}
        ):
            raise PilotJoinError(
                "textured-v03 calibration parity prerequisites changed"
            )
        receipt_parity_result = _inert_binding(
            receipt_prerequisites["result_binding"],
            label="calibration receipt visual-domain parity result",
        )
        receipt_parity_terminal = _inert_binding(
            receipt_prerequisites["terminal_binding"],
            label="calibration receipt visual-domain parity terminal",
        )
        receipt_parity_review = _inert_binding(
            receipt_prerequisites["review_binding"],
            label="calibration receipt visual-domain parity review",
        )
        if (
            calibration_parity_result != collection_parity_result
            or calibration_parity_terminal != collection_parity_terminal
            or calibration_parity_review != collection_parity_review
            or receipt_parity_result != collection_parity_result
            or receipt_parity_terminal != collection_parity_terminal
            or receipt_parity_review != collection_parity_review
        ):
            raise PilotJoinError(
                "bounded and calibration visual-domain parity lineage differs"
            )
    elif any(
        binding is not None
        for binding in (
            calibration_visual_domain_parity_result_binding,
            calibration_visual_domain_parity_terminal_binding,
            calibration_visual_domain_parity_review_binding,
        )
    ):
        raise PilotJoinError("legacy calibration cannot carry textured-v03 parity")
    contract = calibration_receipt.get("calibration_contract")
    if not isinstance(contract, Mapping):
        raise PilotJoinError("calibration contract is absent")
    excluded = set(contract["excluded_scene_ids"])
    states = collection.get("states")
    plan = collection.get("plan")
    if not isinstance(states, Sequence) or isinstance(states, (str, bytes)):
        raise PilotJoinError("pilot collection states are absent")
    if not isinstance(plan, Mapping) or not isinstance(plan.get("document"), Mapping):
        raise PilotJoinError("pilot collection plan is absent")
    plan_document = plan["document"]
    action_catalog = plan_document.get("action_catalog")
    if not isinstance(action_catalog, list) or len(action_catalog) != ACTION_COUNT:
        raise PilotJoinError("pilot action catalog changed")

    rows: dict[str, list[dict[str, object]]] = {role: [] for role in ROLE_NAMES}
    artifacts: list[dict[str, object]] = []
    artifact_ids: set[str] = set()
    scene_ids: dict[str, set[str]] = {role: set() for role in ROLE_NAMES}
    for state in states:
        if not isinstance(state, Mapping):
            raise PilotJoinError("pilot state is malformed")
        state_identity = state.get("state")
        context = state.get("context")
        branches = state.get("branches")
        sync = state.get("document", {}).get("synchronization_audit")
        if (
            not isinstance(state_identity, Mapping)
            or not isinstance(context, Mapping)
            or not isinstance(branches, Sequence)
            or isinstance(branches, (str, bytes))
            or len(branches) != ACTION_COUNT
            or not isinstance(sync, Mapping)
        ):
            raise PilotJoinError("pilot state receipt shape changed")
        role = str(state_identity["role"])
        scene_id = str(state_identity["scene_id"])
        if role not in ROLE_NAMES or scene_id in excluded:
            raise PilotJoinError("pilot role includes a calibration scene")
        scene_ids[role].add(scene_id)
        ranks = _dense_ranks(
            branches,
            progress_tolerance_m=float(contract["progress_tolerance_m"]),
            path_length_tolerance_m=float(contract["path_length_tolerance_m"]),
        )
        context_ids = list(context["rgb_artifact_ids"])
        for identity in context["frame_identities"]:
            frame = collection["frame_receipts"].get(identity)
            if not isinstance(frame, Mapping):
                raise PilotJoinError("context frame receipt is absent")
            artifact = _artifact_from_frame_receipt(frame)
            if artifact["artifact_id"] in artifact_ids:
                raise PilotJoinError("RGB artifact identity repeats")
            artifact_ids.add(str(artifact["artifact_id"]))
            artifacts.append(artifact)
        joined_branches = []
        for action_id, (source_branch, dense_rank) in enumerate(
            zip(branches, ranks, strict=True)
        ):
            if not isinstance(source_branch, Mapping):
                raise PilotJoinError("pilot branch is malformed")
            frame = source_branch.get("frame_receipt")
            if not isinstance(frame, Mapping):
                raise PilotJoinError("target frame receipt is absent")
            artifact = _artifact_from_frame_receipt(frame)
            if artifact["artifact_id"] in artifact_ids:
                raise PilotJoinError("RGB artifact identity repeats")
            artifact_ids.add(str(artifact["artifact_id"]))
            artifacts.append(artifact)
            joined_branch = dict(source_branch)
            if joined_branch.pop("duplicates_candidate_action_id", None) is not None:
                raise PilotJoinError("train/eval candidate carries a repeat marker")
            joined_branch["declared_oracle_dense_rank"] = dense_rank
            if (
                joined_branch["action_id"] != action_id
                or joined_branch["requested_block"]
                != action_catalog[action_id]["requested_block"]
            ):
                raise PilotJoinError("requested candidate action identity changed")
            # The future executed tape remains an outcome/audit field.  It is
            # deliberately not transformed into a model-input action identity.
            joined_branches.append(joined_branch)
        rows[role].append({
            "schema": GROUP_SCHEMA,
            "role": role,
            "state_id": str(state_identity["state_id"]),
            "family": str(state_identity["family"]),
            "scene_id": scene_id,
            "group_index": int(state_identity["group_index"]),
            "state_index_in_scene": int(state_identity["state_index_in_scene"]),
            "task": {
                "target_present": True,
                "relative_target_xy_body_m": list(
                    state["relative_target_xy_body_m"]
                ),
            },
            "context": {
                "rgb_artifact_ids": context_ids,
                "frame_identities": list(context["frame_identities"]),
                "history_action_ids": list(context["history_action_ids"]),
                "history_executed_blocks": list(
                    context["history_executed_blocks"]
                ),
                "executed_block_sha256s": list(
                    context["executed_block_sha256s"]
                ),
                "endpoint_command_ticks": list(context["endpoint_command_ticks"]),
                "prebranch_state_sha256": str(
                    context["prebranch_state_sha256"]
                ),
            },
            "synchronization_audit": dict(sync),
            "branches": joined_branches,
        })
    if not rows["train"] or not rows["eval"]:
        raise PilotJoinError("pilot join requires nonempty train and eval roles")
    if scene_ids["train"] & scene_ids["eval"]:
        raise PilotJoinError("pilot train/eval scenes overlap")
    return (
        {
            "schema": RGB_MANIFEST_SCHEMA,
            "artifacts": artifacts,
        },
        rows,
        {
            "action_catalog": action_catalog,
            "calibration_contract": dict(contract),
            "calibration_collection_receipt_binding": (
                calibration_collection_binding
            ),
            "render_profile": collection_profile,
            "visual_domain_parity_result_binding": collection_parity_result,
            "visual_domain_parity_terminal_binding": collection_parity_terminal,
            "visual_domain_parity_review_binding": collection_parity_review,
            "scene_ids": {
                role: sorted(scene_ids[role]) for role in ROLE_NAMES
            },
        },
    )


def join_pilot(
    *,
    collection_path: Path,
    expected_collection_sha256: str,
    expected_collection_byte_count: int,
    calibration_receipt_path: Path,
    expected_calibration_sha256: str,
    expected_calibration_byte_count: int,
) -> tuple[dict[str, object], dict[str, object]]:
    collection = _load_pixel_verified_collection(
        collection_path,
        expected_file_sha256=expected_collection_sha256,
        expected_byte_count=expected_collection_byte_count,
    )
    calibration_receipt, _, calibration_raw = (
        calibration.load_bound_calibration_receipt_v1(
            calibration_receipt_path,
            expected_sha256=expected_calibration_sha256,
            expected_byte_count=expected_calibration_byte_count,
        )
    )
    calibration_collection_binding = _inert_binding(
        calibration_receipt["calibration_collection_receipt"],
        label="calibration collection receipt",
    )
    calibration_collection_path = Path(
        str(calibration_collection_binding["path"])
    )
    if not calibration_collection_path.is_absolute():
        raise PilotJoinError("calibration collection receipt path must be absolute")
    calibration_collection = _load_pixel_verified_collection(
        calibration_collection_path,
        expected_file_sha256=str(
            calibration_collection_binding["file_sha256"]
        ),
        expected_byte_count=int(calibration_collection_binding["byte_count"]),
    )
    (
        calibration_collection_profile,
        calibration_parity_result,
        calibration_parity_terminal,
        calibration_parity_review,
    ) = (
        _collection_render_lineage(
            calibration_collection, label="calibration collection"
        )
    )
    if calibration_collection_profile != _calibration_render_profile(
        calibration_receipt
    ):
        raise PilotJoinError(
            "calibration receipt and calibration collection render profiles differ"
        )
    rgb_manifest, rows, metadata = build_joined_documents_v1(
        collection,
        calibration_receipt,
        calibration_visual_domain_parity_result_binding=calibration_parity_result,
        calibration_visual_domain_parity_terminal_binding=calibration_parity_terminal,
        calibration_visual_domain_parity_review_binding=calibration_parity_review,
    )
    root = Path(collection_path).resolve(strict=True).parent
    if Path(collection["document"]["output_root"]).resolve(strict=True) != root:
        raise PilotJoinError("pilot collection output root changed")
    joined_root = root / "joined_receipts_v1"
    if joined_root.exists() or joined_root.is_symlink():
        raise FileExistsError(f"joined receipt root already exists: {joined_root}")
    joined_root.mkdir(parents=False)

    calibration_copy = joined_root / "calibration_receipt.json"
    _write_bytes_exclusive(calibration_copy, calibration_raw)
    rgb_path = joined_root / "rgb_manifest.json"
    _write_json_exclusive(rgb_path, rgb_manifest)
    role_contracts: dict[str, object] = {}
    for role in ROLE_NAMES:
        role_path = joined_root / f"{role}.jsonl"
        role_raw = b"".join(
            _canonical_json_bytes(row) + b"\n" for row in rows[role]
        )
        _write_bytes_exclusive(role_path, role_raw)
        role_contracts[role] = {
            "index": _relative_binding(role_path, root=root),
            "group_count": len(rows[role]),
            "branch_count": len(rows[role]) * ACTION_COUNT,
            "scene_ids": metadata["scene_ids"][role],
        }
    collection_document = collection["document"]
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "attempt_id": str(collection_document["attempt_id"]),
        "purpose": "bounded_wm_a_pilot",
        "status": "COMPLETE",
        "physics_validated": True,
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "evidence_scope": "physics_executed",
        "receipt_root": str(root),
        "output_root": str(root),
        "action_catalog": metadata["action_catalog"],
        "action_contract": {
            "primitive_names": [
                str(entry["name"]) for entry in metadata["action_catalog"]
            ],
            "command_ticks_per_block": 5,
            "executed_tape_shape": [5, 3],
            "candidate_model_input": "requested_action_id",
            "future_executed_tape_usage": "target_and_audit_only",
        },
        "calibration_contract": metadata["calibration_contract"],
        "calibration_collection_receipt_binding": metadata[
            "calibration_collection_receipt_binding"
        ],
        "render_profile": metadata["render_profile"],
        "visual_domain_parity_result_binding": metadata[
            "visual_domain_parity_result_binding"
        ],
        "visual_domain_parity_terminal_binding": metadata[
            "visual_domain_parity_terminal_binding"
        ],
        "visual_domain_parity_review_binding": metadata[
            "visual_domain_parity_review_binding"
        ],
        "calibration_receipt": _relative_binding(calibration_copy, root=root),
        "roles": role_contracts,
        "rgb_artifact_manifest": _relative_binding(rgb_path, root=root),
        "source_bindings": list(collection_document["source_bindings"]),
        "collection_receipt": _relative_binding(collection_path, root=root),
    }
    manifest_path = root / "manifest.json"
    _write_json_exclusive(manifest_path, manifest)
    manifest_binding = _relative_binding(manifest_path, root=root)
    return manifest, manifest_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", required=True, type=Path)
    parser.add_argument("--expected-collection-sha256", required=True)
    parser.add_argument("--expected-collection-byte-count", required=True, type=int)
    parser.add_argument("--calibration-receipt", required=True, type=Path)
    parser.add_argument("--expected-calibration-sha256", required=True)
    parser.add_argument("--expected-calibration-byte-count", required=True, type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _, binding = join_pilot(
        collection_path=args.collection,
        expected_collection_sha256=args.expected_collection_sha256,
        expected_collection_byte_count=args.expected_collection_byte_count,
        calibration_receipt_path=args.calibration_receipt,
        expected_calibration_sha256=args.expected_calibration_sha256,
        expected_calibration_byte_count=args.expected_calibration_byte_count,
    )
    print(json.dumps({"status": "COMPLETE", "manifest": binding}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "PilotJoinError",
    "build_joined_documents_v1",
    "join_pilot",
]
