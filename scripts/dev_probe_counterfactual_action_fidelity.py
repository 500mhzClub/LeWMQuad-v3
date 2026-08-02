#!/usr/bin/env python3
"""DEVELOPMENT-TIER probe: counterfactual action fidelity (WM-A diagnostic).

The decisive question for a world model, and the one this campaign has never
measured: given one state, does the model predict the *correct different future*
for each *different action* -- not merely notice that the executed action's
embedding matches the observed outcome?

Discrimination (what `wrong_action_ratio` measures) is necessary but weak: it
only asks whether swapping in a wrong action makes the prediction worse.
Fidelity asks whether the prediction conditioned on `a_i` is closest to the
TRUE RENDERED OUTCOME of `a_i` among all candidate outcomes from that state.
Planning needs fidelity, because planning only ever scores untaken actions.

The executable entry point accepts only one caller-pinned, render-joined
physical pilot manifest.  It requires the corrected five-tick cadence, two
executed history blocks, exact nine-action groups, and scene-disjoint roles.
The older kinematic loaders remain pure legacy-test helpers and are never a
fallback from the executable path.

Method: group rows by (scene_id, source_index). Within a group take the distinct
first primitives {a_1..a_K} and their true first future frames {y_1..y_K}. For
each i predict from the same context conditioned on a_i, then score the full
K x K energy matrix:

  row-wise  (FIDELITY)       argmin_j energy(p_i, y_j) == i ?
  col-wise  (DISCRIMINATION) argmin_i energy(p_i, y_j) == j ?

Chance for both is 1/K. Controls: the untrained-temporal predecessor, an
action-blind arm that conditions every prediction on HOLD, and a deterministic
action-shuffled arm.  Train and evaluation roles are never pooled.

Writes only under `.generated/dev/`. Not citable.
"""
from __future__ import annotations

import argparse
import collections
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import random
import stat
import statistics
import sys
from typing import Mapping

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

model_module = importlib.import_module(
    "lewm.models.rgb_recurrent_patch_memory_temporal_jepa_v1")
evaluation = importlib.import_module(
    "scripts.evaluate_go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
metrics = importlib.import_module(
    "lewm.benchmarks.go2_rgb_recurrent_patch_memory_temporal_jepa_v1")
h6 = importlib.import_module(
    "lewm.datasets.go2_explicit_plan_discounted_successor_state_v27")
pilot = importlib.import_module(
    "lewm.datasets.go2_world_model_counterfactual_pilot_v1")
trainer = importlib.import_module("scripts.dev_train_temporal_jepa_scaled")

PREDECESSOR = (REPO_ROOT
               / ".generated/go2_rgb_single_frame_multiblock_masked_spatial_jepa_v1"
               / "attempt_v1/snapshots/update_1000.pt")
PREDECESSOR_BYTE_COUNT = 52_282_877
PREDECESSOR_SHA256 = (
    "f5aac23cf275d73b92ce5609a583dea89f6686a624d4889d9762740535aab873"
)
SOURCES = {
    "eval": ".generated/jepa_counterfactual/phase2b_eval_8scene_spatial_v1.jsonl",
    "train": ".generated/jepa_counterfactual/phase2b_train_8scene_spatial_v1.jsonl",
}
SOURCE_SPLITS = {"eval": "val", "train": "train"}
H6_ROLES = {"eval": "val", "train": "train"}
RGB_ROOT = REPO_ROOT / ".generated/datagen_full/render_textured_v03"
# Alphabetical, matching lewm/datasets/go2_shared_jepa_v5_raw_supervision_plan.py
# PRIMITIVE_VOCABULARY; index 6 == "hold" == metrics.HOLD_ACTION_ID.
PRIMITIVES = ("arc_left", "arc_right", "backward", "forward_fast",
              "forward_medium", "forward_slow", "hold", "yaw_left", "yaw_right")
ACTION_ID = {name: i for i, name in enumerate(PRIMITIVES)}
HOLD = ACTION_ID["hold"]


FRAME_RE = re.compile(r"frame_(\d+)_env_(\d+)\.png$")
EXPECTED_H6_FRAME_DELTA = int(h6.BLOCK_SIZE * h6.ENVS_PER_SOURCE)
EVIDENCE_SCOPES = ("physics_validated", "kinematic_diagnostic")
DEV_OUTPUT_ROOT = (REPO_ROOT / ".generated/dev").resolve()


class CounterfactualProtocolError(RuntimeError):
    """Raised when a row cannot satisfy the temporal-model input contract."""


@dataclass(frozen=True)
class ContextProvenance:
    source_role: str
    scene_id: str
    start_frame: str
    context_frames: tuple[str, str, str]
    historical_actions: tuple[int, int]


@dataclass(frozen=True)
class GroupLoadResult:
    groups_by_role: Mapping[str, tuple[dict, ...]]
    audit: Mapping[str, object]


def load_pilot_groups(
    pilot_root: Path,
    *,
    expected_manifest_byte_count: int,
    expected_manifest_sha256: str,
) -> tuple[GroupLoadResult, object]:
    """Load the one caller-pinned physical pilot; there is no legacy fallback."""
    bundle = pilot.load_bound_pilot_v1(
        pilot_root,
        expected_manifest_byte_count=expected_manifest_byte_count,
        expected_manifest_sha256=expected_manifest_sha256,
    )
    converted: dict[str, tuple[dict, ...]] = {}
    for role in ("train", "eval"):
        rows = []
        for group in bundle.groups_by_role[role]:
            rows.append({
                "source_role": role,
                "source_split": role,
                "group_id": group.state_id,
                "scene_id": group.scene_id,
                "source_index": group.state_index_in_scene,
                "family": group.family,
                "actions": [branch.action_id for branch in group.branches],
                "start_frame": group.context_rgb_artifact_ids[-1],
                "context_frames": list(group.context_rgb_artifact_ids),
                "historical_actions": list(group.history_action_ids),
                "targets": [
                    branch.target_rgb_artifact_id for branch in group.branches
                ],
                "progress": [
                    branch.labels.target_progress_m for branch in group.branches
                ],
                "target_evidence_class": "physics_executed",
                "physical_oracle_dense_ranks": [
                    branch.oracle_dense_rank for branch in group.branches
                ],
            })
        converted[role] = tuple(rows)
    audit = {
        "manifest_binding": dict(bundle.manifest_binding),
        "rgb_manifest_binding": dict(bundle.rgb_manifest_binding),
        "role_bindings": {
            role: dict(bundle.role_bindings[role]) for role in ("train", "eval")
        },
        "access_audit": dict(bundle.access_audit),
        "evidence_scope": "physics_executed",
        "candidate_actions_per_group": pilot.ACTION_COUNT,
        "train_eval_scene_overlap_count": 0,
        "legacy_source_fallback_used": False,
    }
    return GroupLoadResult(converted, audit), bundle


def _render_leaf(path: str) -> str:
    """Return a stable render-pool-relative identity without opening the path."""
    value = str(path).replace("\\", "/")
    marker = "/render_textured_v03/"
    if marker in value:
        return value.split(marker, 1)[1]
    return value.removeprefix("./")


def _frame_coordinates(path: str) -> tuple[int, int]:
    match = FRAME_RE.search(str(path).replace("\\", "/"))
    if match is None:
        raise CounterfactualProtocolError(f"unparseable RGB frame identity: {path!r}")
    return int(match.group(1)), int(match.group(2))


def validate_context_provenance(value: ContextProvenance) -> None:
    if value.source_role not in SOURCES:
        raise CounterfactualProtocolError(
            f"unknown counterfactual source role {value.source_role!r}")
    if len(value.context_frames) != 3 or len(value.historical_actions) != 2:
        raise CounterfactualProtocolError(
            "counterfactual provenance requires three context frames and two actions")
    if _render_leaf(value.context_frames[-1]) != _render_leaf(value.start_frame):
        raise CounterfactualProtocolError(
            "counterfactual context does not terminate at the start frame")
    coordinates = [_frame_coordinates(path) for path in value.context_frames]
    frame_indices = [item[0] for item in coordinates]
    env_indices = [item[1] for item in coordinates]
    if len(set(env_indices)) != 1:
        raise CounterfactualProtocolError("counterfactual context crosses environments")
    if any(index % h6.ENVS_PER_SOURCE != env_indices[0] for index in frame_indices):
        raise CounterfactualProtocolError(
            "counterfactual context frame/env interleave identity is inconsistent")
    deltas = [right - left for left, right in zip(frame_indices, frame_indices[1:])]
    if deltas != [EXPECTED_H6_FRAME_DELTA, EXPECTED_H6_FRAME_DELTA]:
        raise CounterfactualProtocolError(
            "counterfactual context is not sampled at corrected five-tick H6 endpoints")
    if any(
        isinstance(action, bool)
        or not isinstance(action, int)
        or not 0 <= action < len(PRIMITIVES)
        for action in value.historical_actions
    ):
        raise CounterfactualProtocolError("historical action provenance is invalid")


def build_bound_h6_provenance() -> dict[str, dict[str, ContextProvenance]]:
    """Index bound H6 metadata by current frame; no RGB leaf is opened."""
    result: dict[str, dict[str, ContextProvenance]] = {}
    for source_role, h6_role in H6_ROLES.items():
        rows, _audit = h6.load_bound_index(REPO_ROOT, role=h6_role)
        indexed: dict[str, ContextProvenance] = {}
        for row in rows:
            context = tuple(str(RGB_ROOT / row.rgb[index]) for index in range(3))
            value = ContextProvenance(
                source_role=source_role,
                scene_id=row.scene_id,
                start_frame=context[-1],
                context_frames=context,
                historical_actions=(row.actions[0], row.actions[1]),
            )
            validate_context_provenance(value)
            key = _render_leaf(value.start_frame)
            previous = indexed.get(key)
            if previous is not None and previous != value:
                raise CounterfactualProtocolError(
                    f"conflicting H6 provenance for current frame {key!r}")
            indexed[key] = value
        result[source_role] = indexed
    return result


def _action_id(value: object) -> int:
    if isinstance(value, str) and value in ACTION_ID:
        return ACTION_ID[value]
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 9:
        raise CounterfactualProtocolError(f"invalid action provenance value {value!r}")
    return value


def _embedded_provenance(row: Mapping[str, object], source_role: str) -> ContextProvenance | None:
    frames = row.get("h6_context_frames", row.get("context_frames"))
    actions = row.get("h6_historical_actions", row.get("historical_actions"))
    if frames is None and actions is None:
        return None
    if not isinstance(frames, list) or len(frames) != 3:
        raise CounterfactualProtocolError("embedded H6 context provenance is malformed")
    if not isinstance(actions, list) or len(actions) != 2:
        raise CounterfactualProtocolError("embedded H6 action provenance is malformed")
    if any(not isinstance(path, str) for path in frames):
        raise CounterfactualProtocolError("embedded H6 context paths must be strings")
    value = ContextProvenance(
        source_role=source_role,
        scene_id=str(row["scene_id"]),
        start_frame=str(row["start_frame"]),
        context_frames=(frames[0], frames[1], frames[2]),
        historical_actions=(_action_id(actions[0]), _action_id(actions[1])),
    )
    validate_context_provenance(value)
    return value


def _resolve_provenance(
    row: Mapping[str, object],
    source_role: str,
    indexed: Mapping[str, ContextProvenance],
    *,
    allow_unbound_embedded_provenance: bool,
) -> ContextProvenance:
    embedded = _embedded_provenance(row, source_role)
    key = _render_leaf(str(row["start_frame"]))
    bound = indexed.get(key)
    if embedded is not None and bound is not None and embedded != bound:
        raise CounterfactualProtocolError(
            "embedded H6 provenance disagrees with the bound H6 index")
    if bound is not None:
        value = bound
    elif embedded is not None and allow_unbound_embedded_provenance:
        value = embedded
    else:
        value = None
    if value is None:
        raise CounterfactualProtocolError(
            "no reset-safe H6 block-cadence/action provenance exists for "
            f"{source_role}:{row.get('scene_id')}:{row.get('source_index')}")
    validate_context_provenance(value)
    if value.scene_id != row["scene_id"]:
        raise CounterfactualProtocolError("H6 provenance crosses scene identity")
    return value


def _first_target_evidence(row: Mapping[str, object]) -> str:
    flags = row.get("future_frame_physics_validated")
    if not isinstance(flags, list) or not flags or any(not isinstance(flag, bool) for flag in flags):
        raise CounterfactualProtocolError("future physics-validity provenance is absent")
    return "physics_validated" if flags[0] else "kinematic_render_only"


def load_groups(
    min_actions: int,
    *,
    sources: Mapping[str, Path] | None = None,
    provenance_by_role: Mapping[str, Mapping[str, ContextProvenance]] | None = None,
    evidence_scope: str = "physics_validated",
    allow_unbound_embedded_provenance: bool = False,
) -> GroupLoadResult:
    """Load split-preserving groups under the corrected temporal protocol."""
    if min_actions < 2:
        raise ValueError("min_actions must be at least two")
    if evidence_scope not in EVIDENCE_SCOPES:
        raise ValueError(f"unsupported evidence scope {evidence_scope!r}")
    resolved_sources = (
        sources
        if sources is not None
        else {role: REPO_ROOT / relative for role, relative in SOURCES.items()}
    )
    indexed = (
        provenance_by_role
        if provenance_by_role is not None
        else build_bound_h6_provenance()
    )
    unknown_roles = set(resolved_sources) - set(SOURCES)
    if unknown_roles:
        raise CounterfactualProtocolError(
            f"unknown source roles: {sorted(unknown_roles)}")

    groups_by_role: dict[str, tuple[dict, ...]] = {}
    audit: dict[str, object] = {
        "evidence_scope": evidence_scope,
        "expected_h6_frame_delta": EXPECTED_H6_FRAME_DELTA,
        "unbound_embedded_provenance_allowed": (
            allow_unbound_embedded_provenance
        ),
        "roles": {},
    }
    for source_role in ("train", "eval"):
        path = resolved_sources.get(source_role)
        role_audit = collections.Counter()
        buckets: dict[tuple[str, int], dict[int, tuple[dict, ContextProvenance, str]]] = (
            collections.defaultdict(dict)
        )
        if path is None or not path.exists():
            groups_by_role[source_role] = ()
            audit["roles"][source_role] = dict(role_audit)
            continue
        if path.is_symlink() or not path.is_file():
            raise CounterfactualProtocolError(
                f"counterfactual source is not a regular non-symlink file: {path}"
            )
        role_index = indexed.get(source_role, {})
        source_digest = hashlib.sha256()
        source_byte_count = 0
        with path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            for line_number, line in enumerate(stream, 1):
                source_digest.update(line)
                source_byte_count += len(line)
                role_audit["rows_seen"] += 1
                row = json.loads(line)
                if row.get("split") != SOURCE_SPLITS[source_role]:
                    raise CounterfactualProtocolError(
                        f"{path}:{line_number}: split does not match source role")
                if not row.get("complete_valid_future_sequence"):
                    role_audit["incomplete_rows_rejected"] += 1
                    continue
                evidence = _first_target_evidence(row)
                role_audit[f"{evidence}_rows"] += 1
                if evidence_scope == "physics_validated" and evidence != "physics_validated":
                    role_audit["nonphysics_rows_rejected"] += 1
                    continue
                provenance = _resolve_provenance(
                    row,
                    source_role,
                    role_index,
                    allow_unbound_embedded_provenance=(
                        allow_unbound_embedded_provenance
                    ),
                )
                primitive_sequence = row.get("primitive_sequence")
                targets = row.get("future_frames")
                if (
                    not isinstance(primitive_sequence, list)
                    or not primitive_sequence
                    or not isinstance(targets, list)
                    or not targets
                    or not isinstance(targets[0], str)
                ):
                    raise CounterfactualProtocolError(
                        f"{path}:{line_number}: candidate action/target is malformed")
                action = _action_id(primitive_sequence[0])
                key = (str(row["scene_id"]), int(row["source_index"]))
                previous = buckets[key].get(action)
                candidate = (row, provenance, evidence)
                if previous is not None:
                    previous_row = previous[0]
                    if previous_row["future_frames"][0] != targets[0]:
                        raise CounterfactualProtocolError(
                            f"{path}:{line_number}: one action has conflicting first outcomes")
                    role_audit["duplicate_first_action_rows"] += 1
                    continue
                buckets[key][action] = candidate
                role_audit["candidate_rows_accepted"] += 1
            after = os.fstat(stream.fileno())
        if (
            (before.st_dev, before.st_ino, before.st_size)
            != (after.st_dev, after.st_ino, after.st_size)
            or source_byte_count != before.st_size
        ):
            raise CounterfactualProtocolError(
                f"counterfactual source changed while it was read: {path}"
            )
        role_audit["source_binding"] = {
            "path": str(path.resolve()),
            "byte_count": source_byte_count,
            "sha256": source_digest.hexdigest(),
        }

        groups: list[dict] = []
        for (scene, source_index), by_action in sorted(buckets.items()):
            if len(by_action) < min_actions:
                role_audit["groups_below_min_actions"] += 1
                continue
            actions = sorted(by_action)
            first_row, provenance, _first_evidence = by_action[actions[0]]
            if any(by_action[action][1] != provenance for action in actions):
                raise CounterfactualProtocolError(
                    "candidate branches disagree on H6 context provenance")
            evidence_classes = {by_action[action][2] for action in actions}
            evidence_class = (
                "physics_validated"
                if evidence_classes == {"physics_validated"}
                else "kinematic_render_only"
            )
            groups.append({
                "source_role": source_role,
                "source_split": SOURCE_SPLITS[source_role],
                "group_id": f"{source_role}:{scene}:{source_index}",
                "scene_id": scene,
                "source_index": source_index,
                "family": first_row["family"],
                "actions": actions,
                "start_frame": provenance.start_frame,
                "context_frames": list(provenance.context_frames),
                "historical_actions": list(provenance.historical_actions),
                "targets": [by_action[action][0]["future_frames"][0] for action in actions],
                "progress": [by_action[action][0]["consequence_labels"].get(
                    "target_progress_m", 0.0) for action in actions],
                "target_evidence_class": evidence_class,
            })
        role_audit["groups_accepted"] = len(groups)
        groups_by_role[source_role] = tuple(groups)
        audit["roles"][source_role] = dict(role_audit)
    train_scenes = {group["scene_id"] for group in groups_by_role["train"]}
    eval_scenes = {group["scene_id"] for group in groups_by_role["eval"]}
    scene_overlap = train_scenes & eval_scenes
    if scene_overlap:
        raise CounterfactualProtocolError(
            "train/eval scene overlap violates capacity-diagnostic separation: "
            f"{sorted(scene_overlap)}"
        )
    audit["train_eval_scene_overlap_count"] = 0
    return GroupLoadResult(groups_by_role=groups_by_role, audit=audit)


def decode(path: str, device, *, pilot_bundle=None) -> torch.Tensor:
    if pilot_bundle is not None:
        raw = pilot.read_bound_rgb_bytes_v1(pilot_bundle, path)
        return h6.rectify_h6_rgb_bytes(raw).to(device)
    selected = Path(path)
    candidate = selected if selected.is_absolute() else REPO_ROOT / selected
    resolved = candidate.resolve()
    try:
        resolved.relative_to(RGB_ROOT.resolve())
    except ValueError as exc:
        raise CounterfactualProtocolError(
            f"counterfactual RGB input escapes the bound render root: {path}"
        ) from exc
    if candidate.is_symlink() or not resolved.is_file():
        raise CounterfactualProtocolError(
            f"counterfactual RGB input is not a regular non-symlink file: {path}"
        )
    relative = resolved.relative_to(RGB_ROOT.resolve())
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
    descriptor = os.open(RGB_ROOT.resolve(), directory_flags)
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
            raise CounterfactualProtocolError(
                f"counterfactual RGB input is not regular: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(file_descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(file_descriptor)
        if (before.st_dev, before.st_ino, before.st_size) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
        ):
            raise CounterfactualProtocolError(
                f"counterfactual RGB input changed while read: {path}"
            )
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise CounterfactualProtocolError(
                f"counterfactual RGB byte count changed while read: {path}"
            )
    except OSError as exc:
        raise CounterfactualProtocolError(
            f"cannot safely open counterfactual RGB input: {path}"
        ) from exc
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(descriptor)
    return h6.rectify_h6_rgb_bytes(raw).to(device)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one complete JSON result without exposing a partial final file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if (
        path.exists()
        or path.is_symlink()
        or temporary.exists()
        or temporary.is_symlink()
    ):
        raise FileExistsError(f"refusing to overwrite diagnostic JSON: {path}")
    try:
        with temporary.open("x") as stream:
            stream.write(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        os.link(temporary, path)
    except FileExistsError as exc:
        raise FileExistsError(
            f"refusing to overwrite diagnostic JSON: {path}"
        ) from exc
    finally:
        temporary.unlink(missing_ok=True)


def require_development_output(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(DEV_OUTPUT_ROOT):
        raise ValueError(f"development output must remain under {DEV_OUTPUT_ROOT}")
    return resolved


def require_development_checkpoint(path: Path) -> Path:
    selected = Path(path)
    if selected.is_symlink():
        raise ValueError(
            f"development checkpoint must be a non-symlink file: {selected}"
        )
    resolved = selected.resolve()
    if not resolved.is_relative_to(DEV_OUTPUT_ROOT):
        raise ValueError(
            f"development checkpoint must remain under {DEV_OUTPUT_ROOT}"
        )
    return resolved


def file_binding(path: Path) -> dict[str, object]:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise ValueError(f"input is not a regular non-symlink file: {selected}")
    digest = hashlib.sha256()
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return {
        "path": str(selected.resolve()),
        "byte_count": selected.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def assert_file_bindings_unchanged(
    bindings: list[Mapping[str, object]], *, kind: str
) -> None:
    for expected in bindings:
        current = file_binding(Path(str(expected["path"])))
        if current != dict(expected):
            raise RuntimeError(f"{kind} changed during evaluation: {expected['path']}")


def assert_counterfactual_sources_unchanged(audit: Mapping[str, object]) -> None:
    roles = audit.get("roles")
    if not isinstance(roles, Mapping):
        raise CounterfactualProtocolError("counterfactual source audit is absent")
    for role in ("train", "eval"):
        role_audit = roles.get(role)
        if not isinstance(role_audit, Mapping):
            continue
        expected = role_audit.get("source_binding")
        if expected is None:
            continue
        if not isinstance(expected, Mapping):
            raise CounterfactualProtocolError(
                f"counterfactual source binding is malformed for {role}"
            )
        current = file_binding(Path(str(expected["path"])))
        if current != dict(expected):
            raise CounterfactualProtocolError(
                f"counterfactual source changed after loading: {role}"
            )


ARM_SNAPSHOT_SCHEMAS = frozenset({
    "lewm_go2_world_model_existing_pool_three_arm_snapshot_v1",
    "lewm_go2_world_model_action_alignment_successor_v1_snapshot_v1",
    "lewm_go2_world_model_progression_v1_snapshot_v1",
})
_ARM_EXACT_STATE_KEYS = frozenset({
    "predictor_position",
    "predictor_mask_token",
})
_ARM_STATE_PREFIXES = (
    "predictor_blocks.",
    "predictor_norm.",
    "predictor_output.",
    "action_embedding.",
    "time_embedding.",
    "temporal_gru.",
)


def load_predictor_arm_state_v1(model, payload, *, expected_update: int) -> dict:
    """Load a current predictor-only arm snapshot into its frozen substrate.

    The snapshot does not carry the encoder/target-encoder tensors.  Those stay
    at the exact frozen predecessor initialization; every and only trainable
    predictor/memory key must be supplied by ``arm_state_dict``.
    """

    if (
        not isinstance(payload, dict)
        or payload.get("schema") not in ARM_SNAPSHOT_SCHEMAS
        or payload.get("status") not in {"COMPLETE", "INERT_AUDIT_SNAPSHOT"}
        or type(payload.get("update")) is not int
        or payload.get("update") != expected_update
        or not isinstance(payload.get("arm"), str)
        or not payload["arm"]
        or not isinstance(payload.get("arm_state_dict"), dict)
    ):
        raise ValueError("predictor-arm snapshot schema or update changed")
    if payload.get("citable_as_scientific_evidence", False) is not False:
        raise ValueError("predictor-arm snapshot unexpectedly claims scientific evidence")
    if payload.get("authorizes_retry_or_resume", False) is not False:
        raise ValueError("predictor-arm snapshot unexpectedly authorizes retry/resume")
    full_state = model.state_dict()
    expected_arm_keys = {
        name
        for name in full_state
        if name in _ARM_EXACT_STATE_KEYS
        or name.startswith(_ARM_STATE_PREFIXES)
    }
    arm_state = payload["arm_state_dict"]
    if set(arm_state) != expected_arm_keys:
        raise ValueError("predictor-arm state inventory changed")
    merged = dict(full_state)
    for name in sorted(expected_arm_keys):
        value = arm_state[name]
        expected = full_state[name]
        if (
            not isinstance(value, torch.Tensor)
            or value.device.type != "cpu"
            or value.layout != torch.strided
            or value.dtype != expected.dtype
            or tuple(value.shape) != tuple(expected.shape)
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError(f"predictor-arm tensor changed: {name}")
        merged[name] = value.detach()
    model.load_state_dict(merged, strict=True)
    return {
        "kind": "predictor_arm_snapshot",
        "snapshot_schema": payload["schema"],
        "selected_arm": payload["arm"],
        "selected_model_update": payload["update"],
        "state_key": "arm_state_dict",
        "frozen_substrate_source": "migrated_predecessor_initialization",
        "arm_tensor_count": len(expected_arm_keys),
    }


def build_model(
    checkpoint,
    device,
    expected_checkpoint_sha256=None,
    expected_update=None,
):
    predecessor_binding = file_binding(PREDECESSOR)
    if (
        predecessor_binding["byte_count"] != PREDECESSOR_BYTE_COUNT
        or predecessor_binding["sha256"] != PREDECESSOR_SHA256
    ):
        raise ValueError("predecessor checkpoint disagrees with its frozen binding")
    base = torch.load(PREDECESSOR, map_location="cpu", weights_only=True)
    if file_binding(PREDECESSOR) != predecessor_binding:
        raise RuntimeError("predecessor checkpoint changed while it was loaded")
    if not isinstance(base, dict) or not isinstance(base.get("model_state_dict"), dict):
        raise ValueError("predecessor checkpoint schema changed")
    state = {k: v.detach() for k, v in base["model_state_dict"].items()}
    model = model_module.RGBRecurrentPatchMemoryTemporalJepaV1(state)
    label = "predecessor_init"
    identity = {
        "kind": "migrated_predecessor_initialization",
        "predecessor": predecessor_binding,
        "scaled_snapshot": None,
    }
    if checkpoint:
        checkpoint_binding = file_binding(Path(checkpoint))
        if checkpoint_binding["sha256"] != expected_checkpoint_sha256:
            raise ValueError("scaled snapshot SHA-256 disagrees with expectation")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if file_binding(Path(checkpoint)) != checkpoint_binding:
            raise RuntimeError("scaled snapshot changed while it was loaded")
        if payload.get("schema") == trainer.SNAPSHOT_SCHEMA:
            if (
                payload.get("citable_as_scientific_evidence") is not False
                or payload.get("authorizes_retry_or_resume") is not False
                or not isinstance(payload.get("model_state_dict"), dict)
                or type(payload.get("update")) is not int
                or payload.get("update") != expected_update
                or not isinstance(payload.get("pack_bindings"), dict)
                or not isinstance(payload.get("source_bindings"), list)
                or payload.get("predecessor_binding") != predecessor_binding
            ):
                raise ValueError("scaled snapshot schema or provenance changed")
            model.load_state_dict(payload["model_state_dict"])
            label = f"temporal_update_{payload.get('update')}"
            identity = {
                "kind": "scaled_temporal_snapshot",
                "predecessor": predecessor_binding,
                "scaled_snapshot": checkpoint_binding,
                "selected_model_update": payload["update"],
                "snapshot_pack_bindings": payload.get("pack_bindings"),
                "snapshot_source_bindings": payload.get("source_bindings"),
            }
        else:
            arm_identity = load_predictor_arm_state_v1(
                model, payload, expected_update=expected_update
            )
            label = f"{payload['arm']}_update_{payload['update']}"
            identity = {
                **arm_identity,
                "predecessor": predecessor_binding,
                "scaled_snapshot": checkpoint_binding,
                "snapshot_substrate_receipt": payload.get("substrate"),
                "snapshot_authority_binding": payload.get("authority_binding"),
            }
    return model.to(device).eval(), label, identity


def conditioned_candidate_actions(actions: list[int], mode: str) -> list[int]:
    if mode == "factual":
        return list(actions)
    if mode == "action_blind":
        return [HOLD] * len(actions)
    if mode == "action_shuffled":
        if len(actions) < 2:
            raise ValueError("action-shuffled control requires at least two actions")
        return list(actions[1:]) + [actions[0]]
    raise ValueError(f"unknown action-conditioning mode {mode!r}")


def _matrix_metrics(energy: torch.Tensor, group: Mapping[str, object]) -> dict:
    k = int(energy.shape[0])
    if tuple(energy.shape) != (k, k) or k < 2:
        raise ValueError("counterfactual energy must be a square K>=2 matrix")
    diag = torch.arange(k)
    fidelity_choice = energy.argmin(dim=1)
    discrimination_choice = energy.argmin(dim=0)
    fidelity_hit_flags = fidelity_choice == diag
    discrimination_hit_flags = discrimination_choice == diag
    fidelity_hits = int(fidelity_hit_flags.sum())
    discrimination_hits = int(discrimination_hit_flags.sum())
    off_diagonal = energy.clone()
    off_diagonal[diag, diag] = torch.inf
    fidelity_margins = off_diagonal.min(dim=1).values - energy.diagonal()
    discrimination_margins = off_diagonal.min(dim=0).values - energy.diagonal()
    return {
        "group_id": group["group_id"],
        "source_role": group["source_role"],
        "scene_id": group["scene_id"],
        "source_index": group["source_index"],
        "family": group["family"],
        "target_evidence_class": group["target_evidence_class"],
        "actions": list(group["actions"]),
        "k": k,
        "fidelity_hit_count": fidelity_hits,
        "discrimination_hit_count": discrimination_hits,
        "fidelity_rate": fidelity_hits / k,
        "discrimination_rate": discrimination_hits / k,
        "fidelity_strict_win_count": int((fidelity_margins > 0.0).sum()),
        "discrimination_strict_win_count": int(
            (discrimination_margins > 0.0).sum()
        ),
        "fidelity_margin_sum": float(fidelity_margins.sum()),
        "discrimination_margin_sum": float(discrimination_margins.sum()),
        "fidelity_margin_mean": float(fidelity_margins.mean()),
        "discrimination_margin_mean": float(discrimination_margins.mean()),
        "branch_results": [
            {
                "action": int(group["actions"][index]),
                "fidelity_hit": bool(fidelity_hit_flags[index]),
                "discrimination_hit": bool(discrimination_hit_flags[index]),
                "fidelity_margin": float(fidelity_margins[index]),
                "discrimination_margin": float(discrimination_margins[index]),
                "fidelity_selected_target_action": int(
                    group["actions"][int(fidelity_choice[index])]
                ),
                "discrimination_selected_prediction_action": int(
                    group["actions"][int(discrimination_choice[index])]
                ),
            }
            for index in range(k)
        ],
        "chance": 1.0 / k,
        "diag_mean": float(energy.diagonal().mean()),
        "offdiag_mean": float(
            (energy.sum() - energy.diagonal().sum()) / (k * k - k)),
        "energy_matrix": energy.tolist(),
    }


def scene_cluster_bootstrap_lower_95(
    results: list[dict],
    field: str,
    *,
    seed: int = 20260731,
    resamples: int = 2000,
) -> float | None:
    """Lower percentile bound with scenes, rather than rows, as the unit."""
    if resamples <= 0:
        raise ValueError("bootstrap resamples must be positive")
    if not results:
        return None
    by_scene: dict[str, list[dict]] = collections.defaultdict(list)
    for result in results:
        by_scene[str(result["scene_id"])].append(result)
    scenes = sorted(by_scene)
    if not scenes:
        return None
    rng = random.Random(seed)
    samples = []
    for _ in range(resamples):
        rows = [
            row
            for _scene in scenes
            for row in by_scene[rng.choice(scenes)]
        ]
        samples.append(statistics.fmean(float(row[field]) for row in rows))
    samples.sort()
    return samples[max(0, math.floor(0.025 * len(samples)))]


@torch.no_grad()
def score_group(
    model,
    group,
    device,
    mask_indices,
    action_mode: str = "factual",
    *,
    pilot_bundle=None,
):
    if not group.get("context_frames") or not group.get("historical_actions"):
        raise CounterfactualProtocolError(
            "counterfactual scoring requires verified context/action provenance")
    context = torch.stack(
        [
            decode(path, device, pilot_bundle=pilot_bundle)
            for path in group["context_frames"]
        ]).unsqueeze(0)
    actions = group["actions"]
    k = len(actions)
    targets = torch.stack([
        evaluation._target_tokens(
            model,
            decode(p, device, pilot_bundle=pilot_bundle).unsqueeze(0),
                                  mask_indices)[0]
        for p in group["targets"]])
    conditioned = conditioned_candidate_actions(actions, action_mode)
    history = list(group["historical_actions"])
    preds = []
    for used in conditioned:
        seq = torch.tensor(
            [[history[0], history[1], used]], dtype=torch.long, device=device)
        fields = evaluation._predict_future(model, context, seq, mask_indices)
        preds.append(fields.prediction[0])
    preds = torch.stack(preds)
    # energy[i, j] = normalized squared error of prediction i against target j
    energy = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        for j in range(k):
            energy[i, j] = evaluation._energy(
                preds[i].unsqueeze(0), targets[j].unsqueeze(0)).double().mean()
    # Descriptive separation of the true outcomes.  This contextualizes branch
    # difficulty but does not by itself identify the cause of a fidelity result.
    tsep = torch.zeros(k, k, dtype=torch.float64)
    for i in range(k):
        for j in range(k):
            tsep[i, j] = evaluation._energy(
                targets[i].unsqueeze(0), targets[j].unsqueeze(0)).double().mean()
    target_separation = float(
        (tsep.sum() - tsep.diagonal().sum()) / (k * k - k)) if k > 1 else 0.0
    result = _matrix_metrics(energy, group)
    result.update({
        "action_mode": action_mode,
        "target_separation": target_separation,
        "pred_spread": float(torch.cdist(
            preds.flatten(1).double(), preds.flatten(1).double()).mean()),
    })
    return result


@torch.no_grad()
def score_group_four_masks(
    model,
    group,
    device,
    masks,
    action_mode: str = "factual",
    *,
    pilot_bundle=None,
):
    """Average energies from four separately evaluated deterministic masks."""
    if len(masks) != 4:
        raise ValueError("fixed four-mask scoring requires exactly four masks")
    per_mask = [
        score_group(
            model,
            group,
            device,
            mask,
            action_mode,
            pilot_bundle=pilot_bundle,
        )
        for mask in masks
    ]
    energy = torch.tensor(
        [result["energy_matrix"] for result in per_mask], dtype=torch.float64
    ).mean(dim=0)
    result = _matrix_metrics(energy, group)
    result.update({
        "action_mode": action_mode,
        "target_separation": statistics.fmean(
            float(item["target_separation"]) for item in per_mask
        ),
        "pred_spread": statistics.fmean(
            float(item["pred_spread"]) for item in per_mask
        ),
        "mask_row_indices": [0, 1, 2, 3],
    })
    return result


def summarize(results, label, *, bootstrap_resamples: int = 2000):
    if not results:
        return {"label": label, "groups": 0}
    fam = collections.defaultdict(list)
    for r in results:
        fam[r["family"]].append(r)
    macro_fid = statistics.fmean(r["fidelity_rate"] for r in results)
    macro_dis = statistics.fmean(r["discrimination_rate"] for r in results)
    macro_chance = statistics.fmean(r["chance"] for r in results)
    total_branches = sum(int(r["k"]) for r in results)
    micro_fid = sum(int(r["fidelity_hit_count"]) for r in results) / total_branches
    micro_dis = sum(int(r["discrimination_hit_count"]) for r in results) / total_branches
    micro_chance = len(results) / total_branches
    micro_fidelity_margin = (
        sum(float(r["fidelity_margin_sum"]) for r in results) / total_branches
    )
    micro_discrimination_margin = (
        sum(float(r["discrimination_margin_sum"]) for r in results)
        / total_branches
    )
    fam_pass = sum(
        1 for rs in fam.values()
        if statistics.fmean(r["fidelity_rate"] for r in rs)
        > statistics.fmean(r["chance"] for r in rs) + 1e-12)
    by_action: dict[int, list[dict]] = collections.defaultdict(list)
    selected_target_actions: collections.Counter[int] = collections.Counter()
    for result in results:
        for branch in result.get("branch_results", []):
            by_action[int(branch["action"])].append(branch)
            selected_target_actions[
                int(branch["fidelity_selected_target_action"])
            ] += 1
    summary = {
        "label": label, "groups": len(results), "families": len(fam),
        "branch_count": total_branches,
        "macro": {
            "fidelity_rate": macro_fid,
            "discrimination_rate": macro_dis,
            "chance": macro_chance,
            "fidelity_over_chance": macro_fid / macro_chance,
            "discrimination_over_chance": macro_dis / macro_chance,
        },
        "micro": {
            "fidelity_rate": micro_fid,
            "discrimination_rate": micro_dis,
            "chance": micro_chance,
            "fidelity_over_chance": micro_fid / micro_chance,
            "discrimination_over_chance": micro_dis / micro_chance,
            "fidelity_strict_win_rate": sum(
                int(r["fidelity_strict_win_count"]) for r in results
            ) / total_branches,
            "discrimination_strict_win_rate": sum(
                int(r["discrimination_strict_win_count"]) for r in results
            ) / total_branches,
            "fidelity_margin_mean": micro_fidelity_margin,
            "discrimination_margin_mean": micro_discrimination_margin,
        },
        "families_above_chance_fidelity": fam_pass,
        "diag_mean": statistics.fmean(r["diag_mean"] for r in results),
        "offdiag_mean": statistics.fmean(r["offdiag_mean"] for r in results),
        "per_family_fidelity": {
            f: statistics.fmean(r["fidelity_rate"] for r in rs)
            for f, rs in sorted(fam.items())},
        "per_action": {
            PRIMITIVES[action]: {
                "action_id": action,
                "count": len(branches),
                "fidelity_rate": statistics.fmean(
                    bool(branch["fidelity_hit"]) for branch in branches
                ),
                "discrimination_rate": statistics.fmean(
                    bool(branch["discrimination_hit"]) for branch in branches
                ),
                "fidelity_margin_mean": statistics.fmean(
                    float(branch["fidelity_margin"]) for branch in branches
                ),
            }
            for action, branches in sorted(by_action.items())
        },
        "fidelity_selected_target_action_counts": {
            PRIMITIVES[action]: count
            for action, count in sorted(selected_target_actions.items())
        },
    }
    spreads = [r["pred_spread"] for r in results if "pred_spread" in r]
    if spreads:
        summary["prediction_spread"] = statistics.fmean(spreads)
    if bootstrap_resamples > 0:
        summary["scene_bootstrap_lower_95"] = {
            "resamples": bootstrap_resamples,
            "group_fidelity_advantage_over_chance": (
                scene_cluster_bootstrap_lower_95(
                    [
                        {
                            **r,
                            "fidelity_advantage": r["fidelity_rate"] - r["chance"],
                        }
                        for r in results
                    ],
                    "fidelity_advantage",
                    resamples=bootstrap_resamples,
                )
            ),
            "group_fidelity_margin_mean": scene_cluster_bootstrap_lower_95(
                results,
                "fidelity_margin_mean",
                resamples=bootstrap_resamples,
            ),
        }
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--checkpoint",
        required=True,
        help="Exact immutable development checkpoint; no mutable latest alias.",
    )
    ap.add_argument(
        "--expected-checkpoint-sha256",
        required=True,
        help="Pinned lowercase SHA-256 of the immutable development checkpoint.",
    )
    ap.add_argument("--expected-update", type=int, required=True)
    ap.add_argument("--pilot-root", type=Path, required=True)
    ap.add_argument(
        "--expected-pilot-manifest-byte-count", type=int, required=True
    )
    ap.add_argument("--expected-pilot-manifest-sha256", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out",
                    default=".generated/dev/counterfactual/wm_a_probe.json")
    args = ap.parse_args()

    if args.expected_update < 0:
        raise ValueError("expected-update must be non-negative")
    if Path(args.checkpoint).name == "latest.pt":
        raise ValueError("mutable latest.pt checkpoints are forbidden")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_checkpoint_sha256):
        raise ValueError("expected checkpoint SHA-256 must be lowercase hex")
    if not re.fullmatch(r"[0-9a-f]{64}", args.expected_pilot_manifest_sha256):
        raise ValueError("expected pilot manifest SHA-256 must be lowercase hex")
    checkpoint = require_development_checkpoint(Path(args.checkpoint))
    out = require_development_output(Path(args.out))
    code_bindings = [
        file_binding(Path(path))
        for path in (
            __file__, model_module.__file__, evaluation.__file__,
            metrics.__file__, h6.__file__, trainer.__file__, pilot.__file__,
        )
    ]
    device = torch.device(args.device)
    loaded, pilot_bundle = load_pilot_groups(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    print(json.dumps(loaded.audit, sort_keys=True), flush=True)
    groups = [
        group
        for role in ("train", "eval")
        for group in loaded.groups_by_role[role]
    ]
    print(f"counterfactual groups (exact nine-action pilot): {len(groups)}", flush=True)
    if not groups:
        print("no protocol-valid groups; refusing to open model or RGB", flush=True)
        return 1
    sizes = collections.Counter(len(g["actions"]) for g in groups)
    print(f"  group sizes: {dict(sorted(sizes.items()))}", flush=True)

    mask_indices = [
        metrics.batched_mask_indices("val", [row], device=device)[0]
        for row in (0, 1, 2, 3)
    ]
    report = {"schema": "dev_counterfactual_action_fidelity_v3",
              "status": "COMPLETE",
              "citable_as_scientific_evidence": False,
              "authorizes_retry_or_resume": False,
              "source_bindings": code_bindings,
              "evidence_scope": "physics_executed",
              "claim_scope": "physical_counterfactual_pilot_development_only",
              "mask_protocol": {
                  "role": "val",
                  "row_indices": [0, 1, 2, 3],
                  "fixed_four_mask_contract": True,
              },
              "protocol_audit": loaded.audit,
              "group_count": len(groups),
              "group_size_histogram": {str(k): v for k, v in sorted(sizes.items())},
              "arms": []}

    model_specs = [
        (
            checkpoint,
            (
                ("factual", "temporal"),
                ("action_blind", "temporal_action_blind"),
                ("action_shuffled", "temporal_action_shuffled"),
            ),
        ),
        (None, (("factual", "predecessor"),)),
    ]
    for ckpt, arm_specs in model_specs:
        expected_sha256 = (
            args.expected_checkpoint_sha256 if ckpt is not None else None
        )
        model, label, model_identity = build_model(
            ckpt,
            device,
            expected_checkpoint_sha256=expected_sha256,
            expected_update=(args.expected_update if ckpt is not None else None),
        )
        for action_mode, name in arm_specs:
            results = [score_group_four_masks(
                model,
                g,
                device,
                mask_indices,
                action_mode,
                pilot_bundle=pilot_bundle,
            )
                       for g in groups]
            summaries = {
                role: summarize(
                    [
                        result
                        for result in results
                        if result["source_role"] == role
                    ],
                    f"{name} ({label}) {role}",
                )
                for role in ("train", "eval")
            }
            arm_report = {
                "name": name,
                "label": label,
                "model_identity": model_identity,
                "action_mode": action_mode,
                "summaries_by_role": summaries,
                "group_results": results,
            }
            report["arms"].append(arm_report)
            print(json.dumps({
                "name": name,
                "label": label,
                "summaries_by_role": summaries,
            }), flush=True)
        del model
        torch.cuda.empty_cache()

    reloaded, _ = load_pilot_groups(
        args.pilot_root,
        expected_manifest_byte_count=args.expected_pilot_manifest_byte_count,
        expected_manifest_sha256=args.expected_pilot_manifest_sha256,
    )
    if reloaded.audit != loaded.audit:
        raise CounterfactualProtocolError("pilot receipts changed during evaluation")
    assert_file_bindings_unchanged(code_bindings, kind="diagnostic source")
    write_json_atomic(out, report)
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
