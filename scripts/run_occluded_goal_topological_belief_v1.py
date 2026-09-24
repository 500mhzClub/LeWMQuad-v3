#!/home/andrewknowles/TinyQuadJEPA/bin/python
"""Direct staged runner for OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1.

The benchmark is a prospectively fixed synthetic graph assay.  Exact visual
aliases make the current observation and a four-frame window insufficient at
each query, while a persistent Bayes filter can retain an earlier marker and
propagate it through action-labelled graph transitions.  The graph is an
oracle used to construct the challenge and reduce evidence; it is never an
input to a learned visual model.

Importing this module performs no filesystem discovery, rendering, model
construction, checkpoint opening, or device initialisation.  Each major stage
is invoked explicitly in one foreground process.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, deque
from collections.abc import Iterable, Mapping, Sequence
import hashlib
import heapq
import io
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import sys
import time
import traceback
from typing import Any, Callable
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
for _package_root in (REPO_ROOT, REPO_ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.safety import occluded_goal_topological_belief_v1_contract as CONTRACT


PARENT_COMMIT = "6250d282f1d1edeee411f8ef11407cc7c7a445d3"
OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v1"
)
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "occluded_goal_topological_belief_v1_regeneration_receipt.json"
)

DOC_PATHS = {
    "contract": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v1_contract_2026-09-01.json",
    "fixture": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v1_fixture_2026-09-01.json",
    "output_schema": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v1_output_schema_2026-09-01.json",
    "preregistration": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v1_preregistration_2026-09-01.md",
    "source_closure": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v1_source_closure_2026-09-01.json",
}
SOURCE_PATHS = (
    "lewm/safety/occluded_goal_topological_belief_metrics_v1.py",
    "lewm/safety/occluded_goal_topological_belief_v1_contract.py",
    "lewm/tests/test_evaluate_occluded_goal_topological_belief_v1.py",
    "lewm/tests/test_occluded_goal_topological_belief_metrics_v1.py",
    "lewm/tests/test_occluded_goal_topological_belief_v1_contract.py",
    "lewm/tests/test_run_occluded_goal_topological_belief_v1.py",
    "scripts/evaluate_occluded_goal_topological_belief_v1.py",
    "scripts/run_occluded_goal_topological_belief_v1.py",
    "AGENTS.md",
    "scripts/dev_frozen_dense_representation_encoders_v1.py",
    "lewm/safety/non_greedy_local_subgoal_jepa_planning_v1_contract.py",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.json",
    "docs/lewm_go2_non_greedy_local_subgoal_jepa_planning_v1_result_2026-08-31.md",
)

FAMILIES = tuple(
    getattr(
        CONTRACT,
        "FAMILY_IDS",
        (
            "REPEATED_CORRIDOR",
            "MIRRORED_JUNCTION",
            "LOOP_ALIAS",
            "REPEATED_ROOM",
        ),
    )
)
ROLES = ("FIT", "CALIBRATION", "DEVELOPMENT_HELDOUT")
ROLE_COUNTS_PER_FAMILY = {"FIT": 16, "CALIBRATION": 4, "DEVELOPMENT_HELDOUT": 4}
PORT_LABELS = tuple(
    getattr(CONTRACT, "PORT_LABEL_ORDER", ("LEFT", "STRAIGHT", "RIGHT", "REVERSE"))
)
QUERY_PORTS = ("LEFT", "RIGHT")
CONDITION_IDS = tuple(
    getattr(
        CONTRACT,
        "CONDITION_IDS",
        (
            "CURRENT_FRAME_NEAREST_NODE",
            "FIXED_WINDOW_SEQUENCE",
            "MAP_FILTER",
            "TOP_K_BELIEF",
            "FULL_BELIEF",
            "ORACLE_PLACE_IDENTITY",
            "NO_ACTION_CONSISTENCY",
            "SHUFFLED_ACTION_HISTORY",
            "NO_OBSERVATION_LIKELIHOOD",
        ),
    )
)
CALIBRATION_GRID = getattr(
    CONTRACT,
    "CALIBRATION_GRID",
    {
        "observation_softmax_temperature": (0.01, 0.03, 0.10, 0.30),
        "action_compatible_edge_probability": (0.70, 0.85, 0.95),
        "transition_noise_probability": (0.01, 0.05, 0.15),
        "normalized_entropy_abstention_threshold": (0.25, 0.40, 0.55, 0.70),
    },
)
FIXED_WINDOW = int(getattr(CONTRACT, "FIXED_WINDOW_OBSERVATIONS", 4))
TOP_K = int(getattr(CONTRACT, "TOP_K", 3))
IMAGE_HEIGHT = 168
IMAGE_WIDTH = 224
TOKEN_SHAPE = (768, 1024)
LAYER_NORM_EPSILON = 1.0e-5


class ExperimentError(RuntimeError):
    """The direct experiment contract or a durable stage invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    builder = getattr(CONTRACT, "canonical_json_bytes", None)
    if callable(builder):
        payload = builder(value)
    else:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    return payload.rstrip(b"\n") + b"\n"


def content_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_digest", None)
    return hashlib.sha256(canonical_bytes(payload)[:-1]).hexdigest()


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    helper = getattr(CONTRACT, "attach_content_digest", None)
    if callable(helper):
        return dict(helper(value))
    result = dict(value)
    result["content_digest"] = content_digest(result)
    return result


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular(path: Path, *, absent_ok: bool = False) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if absent_ok:
            return
        raise
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ExperimentError(f"expected an ordinary nlink-1 file: {path}")


def atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ExperimentError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, canonical_bytes(value))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    payload = b"".join(canonical_bytes(dict(row)) for row in rows)
    atomic_bytes(path, payload)


def load_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ExperimentError(f"JSON root is not an object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_regular(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), 1):
        if not raw:
            raise ExperimentError(f"blank JSONL row: {path}:{line_number}")
        value = json.loads(raw)
        if not isinstance(value, dict) or canonical_bytes(value)[:-1] != raw:
            raise ExperimentError(f"noncanonical JSONL row: {path}:{line_number}")
        rows.append(value)
    return rows


def _npy_bytes(value: Any) -> bytes:
    import numpy as np

    stream = io.BytesIO()
    np.lib.format.write_array(stream, np.asarray(value), allow_pickle=False)
    return stream.getvalue()


def atomic_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    """Write a byte-reproducible, pickle-free NPZ with fixed ZIP metadata."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ExperimentError(f"temporary output already exists: {temporary}")
    with temporary.open("xb") as raw:
        with zipfile.ZipFile(raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise ExperimentError(f"invalid NPZ member name: {name!r}")
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(arrays[name]))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _freeze_subject() -> str:
    return str(
        getattr(
            CONTRACT,
            "CONTRACT_FREEZE_COMMIT_SUBJECT",
            "Freeze occluded-goal topological belief experiment",
        )
    )


def _result_subject() -> str:
    return str(
        getattr(
            CONTRACT,
            "RESULT_COMMIT_SUBJECT",
            "Evaluate occluded-goal topological belief experiment",
        )
    )


def _validate_published_source_closure() -> dict[str, Any]:
    closure = load_json(DOC_PATHS["source_closure"])
    if closure.get("schema") != "occluded_goal_topological_belief_v1.source_closure.v1":
        raise ExperimentError("published source-closure schema drift")
    if closure.get("parent_commit") != PARENT_COMMIT:
        raise ExperimentError("published source-closure parent drift")
    if closure.get("content_digest") != content_digest(closure):
        raise ExperimentError("published source-closure content digest drift")
    rows = closure.get("rows")
    if not isinstance(rows, list) or int(closure.get("row_count", -1)) != len(rows):
        raise ExperimentError("published source-closure row count drift")
    if len(rows) != len(SOURCE_PATHS):
        raise ExperimentError("published source-closure path count drift")
    expected_paths = list(SOURCE_PATHS)
    observed_paths = [row.get("path") if isinstance(row, Mapping) else None for row in rows]
    if observed_paths != expected_paths or len(set(observed_paths)) != len(observed_paths):
        raise ExperimentError("published source-closure path/order drift")
    for row, relative in zip(rows, SOURCE_PATHS):
        if set(row) != {"path", "bytes", "sha256"}:
            raise ExperimentError(f"published source-closure row schema drift: {relative}")
        path = REPO_ROOT / relative
        _require_regular(path)
        if (
            isinstance(row["bytes"], bool)
            or not isinstance(row["bytes"], int)
            or path.stat().st_size != row["bytes"]
            or sha256_file(path) != row["sha256"]
        ):
            raise ExperimentError(f"source byte binding differs from closure: {relative}")
    return closure


def _validate_runtime_contract(head: str) -> None:
    contract_path = OUTPUT_ROOT / "contract.json"
    if not contract_path.exists():
        return
    contract = load_json(contract_path)
    expected_fields = {
        "schema",
        "source_freeze_commit",
        "parent_commit",
        "scientific_contract",
        "content_digest",
    }
    if set(contract) != expected_fields:
        raise ExperimentError("runtime contract field set drift")
    if contract.get("schema") != "occluded_goal_topological_belief_v1.runtime_contract.v1":
        raise ExperimentError("runtime contract schema drift")
    if contract.get("content_digest") != content_digest(contract):
        raise ExperimentError("runtime contract content digest drift")
    expected_scientific = CONTRACT.build_contract()
    validator = getattr(CONTRACT, "validate_contract", None)
    if callable(validator):
        validator(expected_scientific)
    if canonical_bytes(contract.get("scientific_contract")) != canonical_bytes(
        expected_scientific
    ):
        raise ExperimentError("runtime scientific contract differs from frozen module")
    if (
        contract.get("source_freeze_commit") != head
        or contract.get("parent_commit") != PARENT_COMMIT
        or expected_scientific.get("source_parent_commit") != PARENT_COMMIT
        or expected_scientific.get("output", {}).get("root") != str(OUTPUT_ROOT)
    ):
        raise ExperimentError("runtime contract source/root binding drift")


def require_source_freeze() -> str:
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", "HEAD^")
    subject = git("show", "-s", "--format=%s", "HEAD")
    if parent != PARENT_COMMIT or subject != _freeze_subject():
        raise ExperimentError(
            "HEAD is not the exact descendant source-freeze commit"
        )
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("source-freeze worktree is not clean")
    changed = tuple(
        sorted(
            value
            for value in git(
                "diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD"
            ).splitlines()
            if value
        )
    )
    expected_changed = tuple(sorted(CONTRACT.TRACKED_SOURCE_PATHS))
    if changed != expected_changed:
        raise ExperimentError("source-freeze changed-path set drift")
    if tuple(SOURCE_PATHS) != tuple(
        value
        for value in CONTRACT.TRACKED_SOURCE_PATHS
        if value not in {
            str(path.relative_to(REPO_ROOT)) for path in DOC_PATHS.values()
        }
    ) + tuple(CONTRACT.SOURCE_DEPENDENCY_PATHS):
        raise ExperimentError("runner source-closure declaration drift")
    _validate_published_source_closure()
    _validate_runtime_contract(head)
    return head


def require_new_output_root() -> None:
    if OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise ExperimentError(f"official output root must be absent: {OUTPUT_ROOT}")
    parent = OUTPUT_ROOT.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ExperimentError(f"output parent must be an ordinary directory: {parent}")


def _role_for_family_rank(rank: int) -> str:
    if rank < 16:
        return "FIT"
    if rank < 20:
        return "CALIBRATION"
    if rank < 24:
        return "DEVELOPMENT_HELDOUT"
    raise ExperimentError(f"family rank out of range: {rank}")


def teacher_side_paths() -> tuple[tuple[int, ...], ...]:
    """Return 96 unique, exactly bit-balanced deterministic eight-bit paths."""

    representatives: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []
    for value in range(256):
        bits = tuple((value >> index) & 1 for index in range(8))
        complement = tuple(1 - bit for bit in bits)
        if bits > complement:
            continue
        key = hashlib.sha256(
            f"OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1/PATH/{value}".encode("ascii")
        ).hexdigest()
        representatives.append((key, bits, complement))
    selected = sorted(representatives)[:48]
    paths = [path for _key, left, right in selected for path in (left, right)]
    ordered = sorted(
        paths,
        key=lambda path: hashlib.sha256(
            ("OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1/ORDER/" + "".join(map(str, path))).encode("ascii")
        ).hexdigest(),
    )
    if len(set(ordered)) != 96 or any(sum(path[index] for path in ordered) != 48 for index in range(8)):
        raise ExperimentError("teacher path construction lost uniqueness/balance")
    return tuple(ordered)


def _identity_projection_digest(values: Sequence[Any], *, numeric: bool = False) -> str:
    unique = sorted(set(values))
    if numeric and any(isinstance(value, bool) or not isinstance(value, int) for value in unique):
        raise ExperimentError("numeric identity projection contains a non-integer")
    return hashlib.sha256(canonical_bytes(unique)[:-1]).hexdigest()


def _structured_identity_projection_digest(values: Sequence[Sequence[Any]]) -> str:
    encoded = sorted({canonical_bytes(list(value))[:-1] for value in values})
    payload = b"[" + b",".join(encoded) + b"]"
    return hashlib.sha256(payload).hexdigest()


def _authority_path(relative_or_absolute: str) -> Path:
    path = Path(relative_or_absolute)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_bound_prior_authorities() -> dict[str, Any]:
    """Open only the ten hash-bound predecessor identity authorities."""

    documents: dict[str, Any] = {}
    for binding in CONTRACT.PRIOR_PANEL_EXCLUSION_AUTHORITY["authorities"]:
        name = str(binding["name"])
        path = _authority_path(str(binding["path"]))
        try:
            info = path.lstat()
        except FileNotFoundError as exc:
            raise ExperimentError(f"prior identity authority is absent: {path}") from exc
        if path.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ExperimentError(f"prior identity authority is not an ordinary file: {path}")
        raw = path.read_bytes()
        if (
            len(raw) != int(binding["bytes"])
            or hashlib.sha256(raw).hexdigest() != str(binding["sha256"])
        ):
            raise ExperimentError(f"prior identity authority bytes/hash drift: {name}")
        try:
            if path.suffix == ".jsonl":
                if not raw or not raw.endswith(b"\n"):
                    raise ExperimentError(f"prior JSONL authority is not LF terminated: {name}")
                rows = []
                for line_number, line in enumerate(raw[:-1].split(b"\n"), 1):
                    if not line:
                        raise ExperimentError(
                            f"prior JSONL authority contains a blank row: {name}:{line_number}"
                        )
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ExperimentError(
                            f"prior JSONL row is not an object: {name}:{line_number}"
                        )
                    rows.append(row)
                documents[name] = rows
            else:
                document = json.loads(raw)
                if not isinstance(document, dict):
                    raise ExperimentError(f"prior JSON authority is not an object: {name}")
                documents[name] = document
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ExperimentError(f"prior identity authority cannot be parsed: {name}") from exc
    if set(documents) != {
        str(row["name"])
        for row in CONTRACT.PRIOR_PANEL_EXCLUSION_AUTHORITY["authorities"]
    }:
        raise ExperimentError("prior identity authority identity set drift")
    return documents


def _required_list(parent: Mapping[str, Any], key: str, label: str) -> list[Any]:
    value = parent.get(key)
    if not isinstance(value, list) or not value:
        raise ExperimentError(f"{label}.{key} must be a nonempty JSON array")
    return value


def _required_text(parent: Mapping[str, Any], key: str, label: str) -> str:
    value = parent.get(key)
    if not isinstance(value, str) or not value:
        raise ExperimentError(f"{label}.{key} must be a nonempty JSON string")
    return value


def _required_seed(parent: Mapping[str, Any], key: str, label: str) -> int:
    value = parent.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ExperimentError(f"{label}.{key} must be a JSON integer")
    return value


def _required_object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ExperimentError(f"{label} must be a JSON object")
    return value


def _project_prior_identities(documents: Mapping[str, Any]) -> dict[str, Any]:
    """Reproduce the six prospectively frozen predecessor identity unions."""

    scenes: set[str] = set()
    supplied_scene_hashes: set[str] = set()
    episode_or_state: set[str] = set()
    seeds: set[int] = set()
    textual_paths: set[str] = set()
    structured_paths: dict[bytes, list[Any]] = {}

    def rows(document_name: str, collection: str) -> list[Mapping[str, Any]]:
        document = _required_object(documents[document_name], document_name)
        return [
            _required_object(value, f"{document_name}.{collection}[]")
            for value in _required_list(document, collection, document_name)
        ]

    def add_scene(row: Mapping[str, Any], label: str) -> None:
        scenes.add(_required_text(row, "scene_id", label))

    def add_state(row: Mapping[str, Any], label: str) -> None:
        episode_or_state.add(_required_text(row, "state_id", label))

    def add_structured(value: Any, label: str, element_type: type) -> None:
        if not isinstance(value, list) or not value:
            raise ExperimentError(f"{label} must be a nonempty JSON array")
        if element_type is int:
            if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
                raise ExperimentError(f"{label} must contain only JSON integers")
        elif any(not isinstance(item, str) or not item for item in value):
            raise ExperimentError(f"{label} must contain only nonempty strings")
        structured_paths[canonical_bytes(value)[:-1]] = list(value)

    safe = rows("safe_local_waypoint_panel", "state_candidates")
    for index, row in enumerate(safe):
        label = f"safe.state_candidates[{index}]"
        add_scene(row, label); add_state(row, label)
        seeds.add(_required_seed(row, "seed", label))
        textual_paths.add(_required_text(row, "scene_dir", label))
        add_structured(row.get("waypoint_path_cells"), f"{label}.waypoint_path_cells", int)

    oracle_12 = rows("oracle_branch_pilot_v1_2", "states")
    for index, row in enumerate(oracle_12):
        label = f"oracle_v1_2.states[{index}]"
        add_scene(row, label); add_state(row, label)
        seeds.add(_required_seed(row, "drive_seed", label))
        textual_paths.add(_required_text(row, "scene_dir", label))

    oracle_1: list[tuple[str, Mapping[str, Any]]] = []
    for collection in ("pilot_states", "replay_states"):
        oracle_1.extend(
            (collection, row)
            for row in rows("oracle_branch_pilot_v1", collection)
        )
    for index, (collection, row) in enumerate(oracle_1):
        label = f"oracle_v1.{collection}[{index}]"
        add_scene(row, label); add_state(row, label)
        seeds.add(_required_seed(row, "drive_seed", label))
        textual_paths.add(_required_text(row, "scene_dir", label))
        if collection == "replay_states":
            add_structured(row.get("sequence"), f"{label}.sequence", str)

    counterfactual = rows("counterfactual_predictor_panel", "scenes")
    for index, scene in enumerate(counterfactual):
        label = f"counterfactual.scenes[{index}]"
        add_scene(scene, label)
        for binding_name in ("scene_manifest_binding", "scene_genesis_binding"):
            binding = _required_object(scene.get(binding_name), f"{label}.{binding_name}")
            textual_paths.add(_required_text(binding, "path", f"{label}.{binding_name}"))
        for state_index, state in enumerate(_required_list(scene, "states", label)):
            add_state(
                _required_object(state, f"{label}.states[{state_index}]"),
                f"{label}.states[{state_index}]",
            )

    non_greedy_document = _required_object(
        documents["non_greedy_local_subgoal_panel"], "non_greedy_local_subgoal_panel"
    )
    non_greedy = [
        _required_object(value, "non_greedy.states[]")
        for value in _required_list(non_greedy_document, "states", "non_greedy")
    ]
    for index, row in enumerate(non_greedy):
        label = f"non_greedy.states[{index}]"
        add_scene(row, label)
        episode_or_state.add(_required_text(row, "episode_id", label))
        add_state(row, label)
        seeds.add(_required_seed(row, "procedural_seed", label))
        textual_paths.add(_required_text(row, "geometry_digest", label))
    exclusion = _required_object(
        non_greedy_document.get("predecessor_scene_exclusion"),
        "non_greedy.predecessor_scene_exclusion",
    )
    for value in _required_list(exclusion, "scene_ids", "non_greedy.predecessor_scene_exclusion"):
        if not isinstance(value, str) or not value:
            raise ExperimentError("non-greedy excluded scene identity is not text")
        scenes.add(value)
    for value in _required_list(
        exclusion, "scene_id_sha256", "non_greedy.predecessor_scene_exclusion"
    ):
        if (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            raise ExperimentError("non-greedy excluded scene hash is malformed")
        supplied_scene_hashes.add(value)

    for document_name in ("memory_role_place_train", "memory_role_place_checkpoint_selection"):
        document_rows = documents[document_name]
        if not isinstance(document_rows, list) or not document_rows:
            raise ExperimentError(f"{document_name} must contain nonempty JSONL rows")
        for index, original in enumerate(document_rows):
            row = _required_object(original, f"{document_name}[{index}]")
            add_scene(row, f"{document_name}[{index}]")
            for role in ("anchor", "positive", "negative"):
                occurrence = _required_object(row.get(role), f"{document_name}[{index}].{role}")
                textual_paths.add(
                    _required_text(occurrence, "rgb_path", f"{document_name}[{index}].{role}")
                )

    for document_name in ("recurrent_predictor_train", "recurrent_predictor_validation"):
        document_rows = documents[document_name]
        if not isinstance(document_rows, list) or not document_rows:
            raise ExperimentError(f"{document_name} must contain nonempty JSONL rows")
        for index, original in enumerate(document_rows):
            row = _required_object(original, f"{document_name}[{index}]")
            add_scene(row, f"{document_name}[{index}]")
            for value in _required_list(row, "rgb", f"{document_name}[{index}]"):
                if not isinstance(value, str) or not value:
                    raise ExperimentError("recurrent RGB path is not text")
                textual_paths.add(value)

    # The memory manifest is provenance-only but was already byte/hash checked.
    scene_hashes = supplied_scene_hashes | {
        hashlib.sha256(value.encode("utf-8")).hexdigest() for value in scenes
    }
    projection = {
        "scene_identity": scenes,
        "scene_identity_sha256": scene_hashes,
        "episode_or_state_identity": episode_or_state,
        "numeric_seed": seeds,
        "textual_path_geometry_or_source_identity": textual_paths,
        "structured_waypoint_or_sequence_path": list(structured_paths.values()),
    }
    authority = CONTRACT.PRIOR_PANEL_EXCLUSION_AUTHORITY["prior_projection"]
    checks = (
        ("scene_identity", "scene_identity_count", "scene_identity_canonical_json_sha256", False),
        ("scene_identity_sha256", "scene_identity_sha256_count", "scene_identity_sha256_canonical_json_sha256", False),
        ("episode_or_state_identity", "episode_or_state_identity_count", "episode_or_state_identity_canonical_json_sha256", False),
        ("numeric_seed", "numeric_seed_count", "numeric_seed_canonical_json_sha256", True),
        ("textual_path_geometry_or_source_identity", "textual_path_geometry_or_source_identity_count", "textual_path_geometry_or_source_identity_canonical_json_sha256", False),
    )
    for key, count_key, digest_key, numeric in checks:
        values = projection[key]
        if len(values) != int(authority[count_key]) or _identity_projection_digest(
            list(values), numeric=numeric
        ) != str(authority[digest_key]):
            raise ExperimentError(f"frozen prior identity projection drift: {key}")
    structured = projection["structured_waypoint_or_sequence_path"]
    if (
        len(structured) != int(authority["structured_waypoint_or_sequence_path_count"])
        or _structured_identity_projection_digest(structured)
        != str(authority["structured_waypoint_or_sequence_path_canonical_json_sha256"])
    ):
        raise ExperimentError("frozen prior structured path projection drift")
    return projection


def build_identity_disjointness(
    graphs: Sequence[Mapping[str, Any]], teacher_paths: Sequence[Sequence[int]]
) -> dict[str, Any]:
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    documents = _load_bound_prior_authorities()
    prior = _project_prior_identities(documents)
    current_scenes = {str(graph["scene_id"]) for graph in graphs}
    current_episode_state = {
        str(value)
        for graph in graphs
        for value in (graph["episode_id"], graph["graph_id"])
    }
    current_seeds = {int(graph["procedural_seed"]) for graph in graphs}
    current_paths = {str(graph["episode_path_id"]) for graph in graphs}
    current_structured = {canonical_bytes(list(value))[:-1] for value in teacher_paths}
    prior_structured = {
        canonical_bytes(list(value))[:-1]
        for value in prior["structured_waypoint_or_sequence_path"]
    }
    comparisons = {
        "scene_id_overlap_count": len(current_scenes & prior["scene_identity"]),
        "scene_id_sha256_overlap_count": len(
            {
                hashlib.sha256(value.encode("utf-8")).hexdigest()
                for value in current_scenes
            }
            & prior["scene_identity_sha256"]
        ),
        "episode_or_state_id_overlap_count": len(
            current_episode_state & prior["episode_or_state_identity"]
        ),
        "procedural_seed_overlap_count": len(current_seeds & prior["numeric_seed"]),
        "textual_path_identity_overlap_count": len(
            current_paths & prior["textual_path_geometry_or_source_identity"]
        ),
        "structured_path_overlap_count": len(current_structured & prior_structured),
    }
    if any(comparisons.values()):
        raise ExperimentError(f"new panel identity overlaps prior authorities: {comparisons}")
    evidence = METRICS.build_identity_disjointness_evidence(graphs, comparisons)
    METRICS.validate_identity_disjointness(evidence, graphs)
    return evidence


def _teacher_paths_from_queries(
    graphs: Sequence[Mapping[str, Any]], queries: Sequence[Mapping[str, Any]]
) -> list[list[int]]:
    by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for query in queries:
        by_episode[str(query["episode_id"])].append(query)
    paths: list[list[int]] = []
    for graph in graphs:
        ordered = sorted(
            by_episode[str(graph["episode_id"])], key=lambda row: int(row["query_index"])
        )
        if len(ordered) != 8:
            raise ExperimentError("cannot rebuild the eight-step teacher identity path")
        paths.append(
            [0 if row["true_next_port_label"] == "LEFT" else 1 for row in ordered]
        )
    return paths


def verify_persisted_identity_disjointness(
    graph_manifest: Mapping[str, Any], query_rows: Sequence[Mapping[str, Any]]
) -> None:
    graphs = list(graph_manifest["graphs"])
    rebuilt = build_identity_disjointness(
        graphs, _teacher_paths_from_queries(graphs, query_rows)
    )
    if rebuilt != graph_manifest.get("identity_disjointness"):
        raise ExperimentError("persisted prior identity-disjointness evidence drift")


def observation_recipe(
    *, family: str, kind: str, module: int = -1, step: int = -1, side: int = -1
) -> dict[str, Any]:
    if family not in FAMILIES:
        raise ExperimentError(f"unknown family {family!r}")
    if kind in {"TAIL", "QUERY"}:
        side = -1  # Exact alias: no side information survives in the recipe.
        if family == "LOOP_ALIAS":
            module = -1
        elif family in {"REPEATED_CORRIDOR", "REPEATED_ROOM"}:
            module //= 2
    recipe = {
        "schema": "occluded_goal_topological_belief_v1.observation_recipe.v1",
        "family": family,
        "kind": kind,
        "module": int(module),
        "step": int(step),
        "side": int(side),
        "height": IMAGE_HEIGHT,
        "width": IMAGE_WIDTH,
        "palette": "neutral_fixed_v1",
        "illumination": "fixed_v1",
        "camera": "fixed_224x168_v1",
        "goal_marker": kind == "GOAL_KEYFRAME",
    }
    template_id = "pixel-template-" + hashlib.sha256(canonical_bytes(recipe)).hexdigest()[:24]
    return {**recipe, "pixel_template_id": template_id}


def _capture_id(episode_id: str, local_node_id: str, phase: str) -> str:
    return f"{episode_id}:{phase}:{local_node_id}"


def _query_node_kind(family: str) -> str:
    return {
        "REPEATED_CORRIDOR": "CORRIDOR_SEGMENT",
        "MIRRORED_JUNCTION": "MIRRORED_JUNCTION",
        "LOOP_ALIAS": "LOOP_LOCATION",
        "REPEATED_ROOM": "REPEATED_ROOM",
    }[family]


def _alias_partner(module: int, side: int, family: str) -> tuple[int, int]:
    if family == "MIRRORED_JUNCTION":
        return module, 1 - side
    return module ^ 1, 1 - side


def _query_alias_group(episode_id: str, module: int, side: int, family: str) -> str:
    if family == "MIRRORED_JUNCTION":
        return f"{episode_id}:alias-query:{module}"
    pair_base = module - module % 2
    parity_class = side if module % 2 == 0 else 1 - side
    return f"{episode_id}:alias-query:{pair_base}:{parity_class}"


def _node(
    episode_id: str,
    node_id: str,
    kind: str,
    recipe: Mapping[str, Any],
    *,
    module: int = -1,
    side: int = -1,
    alias_group_id: str | None = None,
) -> dict[str, Any]:
    full_node_id = f"{episode_id}:{node_id}"
    pixel_sha256 = hashlib.sha256(
        render_observation(recipe).tobytes(order="C")
    ).hexdigest()
    return {
        "node_id": full_node_id,
        "alias_group_id": alias_group_id or f"singleton:{full_node_id}",
        "node_kind": kind,
        "module_index": max(0, int(module)),
        "side_label": "LEFT" if side == 0 else "RIGHT" if side == 1 else "NONE",
        "keyframe_observation_id": _capture_id(episode_id, node_id, "PHASE_A_KEYFRAME"),
        "observation_descriptor": canonical_bytes(dict(recipe)).decode("utf-8").strip(),
        "teacher_visit_index": -1,
        "goal_keyframe": kind == "GOAL",
        "pixel_template_id": str(recipe["pixel_template_id"]),
        "pixel_sha256": pixel_sha256,
        "keyframe_timestamp_s": -1.0,
    }


def _edge(
    episode_id: str,
    source: str,
    target: str,
    port: str,
    *,
    executed_action: str | None = None,
    teacher: bool = False,
    edge_cost: float = 1.0,
    physically_executable: bool = True,
    oracle_admissible: bool = True,
) -> dict[str, Any]:
    if port not in PORT_LABELS:
        raise ExperimentError(f"unknown port label {port!r}")
    return {
        "edge_id": f"{episode_id}:edge:{source}:{port}:{target}",
        "source_node_id": f"{episode_id}:{source}",
        "target_node_id": f"{episode_id}:{target}",
        "port_label": port,
        "executed_action_label": str(executed_action or port),
        "relative_waypoint": {
            "LEFT": [0.20, 0.0, 0.45],
            "STRAIGHT": [0.25, 0.0, 0.0],
            "RIGHT": [0.20, 0.0, -0.45],
            "REVERSE": [-0.20, 0.0, 0.0],
        }[port],
        "edge_cost": float(edge_cost),
        "teacher_edge": bool(teacher),
        "physically_executable": bool(physically_executable),
        "oracle_admissible": bool(oracle_admissible),
    }


def _unused_port(edges: Sequence[Mapping[str, Any]], source: str) -> str:
    occupied = {
        str(edge["port_label"])
        for edge in edges
        if str(edge["source_node_id"]) == source
    }
    for port in PORT_LABELS:
        if port not in occupied:
            return port
    raise ExperimentError(f"no unused Phase-A traversal port at {source}")


def _deterministic_shortest_edge_path(
    edges: Sequence[Mapping[str, Any]],
    source_node_id: str,
    target_node_id: str,
) -> list[str]:
    """Return the frozen admissible shortest path with complete-ID tie break."""

    source = str(source_node_id)
    target = str(target_node_id)
    if source == target:
        return []
    outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        if bool(edge["physically_executable"]) and bool(edge["oracle_admissible"]):
            outgoing[str(edge["source_node_id"])].append(edge)
    queue: list[tuple[float, tuple[str, ...], str]] = [(0.0, (), source)]
    best: dict[str, tuple[float, tuple[str, ...]]] = {source: (0.0, ())}
    while queue:
        cost, path, node_id = heapq.heappop(queue)
        if (cost, path) != best.get(node_id):
            continue
        if node_id == target:
            return list(path)
        for edge in sorted(outgoing.get(node_id, ()), key=lambda row: str(row["edge_id"])):
            edge_id = str(edge["edge_id"])
            successor = str(edge["target_node_id"])
            candidate = (cost + float(edge["edge_cost"]), (*path, edge_id))
            if successor not in best or candidate < best[successor]:
                best[successor] = candidate
                heapq.heappush(queue, (*candidate, successor))
    raise ExperimentError(
        f"no executable oracle-admissible path from {source!r} to {target!r}"
    )


def _stage_b_candidate_port(
    name: str, primitive_blocks: Sequence[str], old_contract: Any
) -> str | None:
    """Derive a frozen candidate's graph port from its post-slew geometry."""

    import numpy as np

    _requested, applied = _slew_stage_b_commands(primitive_blocks, old_contract)
    applied_array = np.asarray(applied, dtype=np.float64)
    tick_seconds = float(old_contract.PANEL_GEOMETRY_AUTHORITY["command_tick_seconds"])
    x = y = yaw = 0.0
    for velocity_x, velocity_y, yaw_rate in applied_array:
        x += (
            math.cos(yaw) * float(velocity_x)
            - math.sin(yaw) * float(velocity_y)
        ) * tick_seconds
        y += (
            math.sin(yaw) * float(velocity_x)
            + math.cos(yaw) * float(velocity_y)
        ) * tick_seconds
        yaw += float(yaw_rate) * tick_seconds
    if abs(yaw) > 1.0e-12:
        return "LEFT" if yaw > 0.0 else "RIGHT"
    if x > 1.0e-12:
        return "STRAIGHT"
    if x < -1.0e-12:
        return "REVERSE"
    if abs(y) <= 1.0e-12 and name == "hold":
        return None
    raise ExperimentError(f"candidate endpoint geometry has no registered port: {name}")


def _stage_b_local_candidate_outcomes(
    query: Mapping[str, Any], edges: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Prospectively bind all 12 local candidates to actual graph outcomes."""

    from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as OLD

    actual_node = str(query["true_node_id"])
    outgoing = [
        edge for edge in edges if str(edge["source_node_id"]) == actual_node
    ]
    all_outgoing: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for edge in edges:
        all_outgoing[str(edge["source_node_id"])].append(edge)
    outcomes: list[dict[str, Any]] = []
    for candidate_index, (candidate_name, primitive_blocks) in enumerate(OLD.CANDIDATE_BANK):
        port = _stage_b_candidate_port(
            str(candidate_name), primitive_blocks, OLD
        )
        matches = [
            edge
            for edge in outgoing
            if port is not None and str(edge["port_label"]) == port
        ]
        if len(matches) > 1:
            raise ExperimentError("candidate geometry maps to multiple actual query edges")
        edge = matches[0] if matches else None
        endpoint = actual_node if edge is None else str(edge["target_node_id"])
        successor_viable = bool(
            endpoint is not None
            and any(
                bool(option["physically_executable"])
                and bool(option["oracle_admissible"])
                for option in all_outgoing.get(endpoint, ())
            )
        )
        immediate_contact = False
        stuck = str(candidate_name) == "hold"
        admissible = bool(
            edge is not None
            and edge["physically_executable"]
            and edge["oracle_admissible"]
            and successor_viable
            and not immediate_contact
            and not stuck
        )
        outcomes.append(
            {
                "candidate_index": candidate_index,
                "candidate_name": str(candidate_name),
                "actual_edge_id": None if edge is None else str(edge["edge_id"]),
                "actual_port_label": None if edge is None else port,
                "endpoint_node_id": endpoint,
                "prefix_distance_m": 0.0 if edge is None else float(edge["edge_cost"]),
                "immediate_contact": immediate_contact,
                "successor_viable": successor_viable,
                "stuck": stuck,
                "oracle_admissible": admissible,
            }
        )
    if len(outcomes) != 12 or [row["candidate_index"] for row in outcomes] != list(range(12)):
        raise ExperimentError("Stage-B candidate-outcome bank drift")
    return outcomes


def build_episode_graph(
    *, family: str, family_rank: int, teacher_path: Sequence[int]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build one graph and its exact eight query rows without visual inference."""

    if len(teacher_path) != 8 or any(bit not in (0, 1) for bit in teacher_path):
        raise ExperimentError("teacher path must contain exactly eight bits")
    family_index = FAMILIES.index(family)
    global_index = family_index * 24 + int(family_rank)
    episode_id = f"{CONTRACT.EPISODE_ID_PREFIX}{family_index:02d}-{family_rank:02d}"
    graph_id = f"{CONTRACT.IDENTITY_DOMAIN}-graph-{family_index:02d}-{family_rank:02d}"
    role = _role_for_family_rank(family_rank)
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    nodes.append(
        _node(
            episode_id,
            "goal",
            "GOAL",
            observation_recipe(family=family, kind="GOAL_KEYFRAME"),
        )
    )
    for prefix in range(3):
        nodes.append(
            _node(
                episode_id,
                f"prefix_{prefix}",
                "PREFIX",
                observation_recipe(family=family, kind="PREFIX", step=prefix),
            )
        )
    edges.extend(
        [
            _edge(episode_id, "goal", "prefix_0", "STRAIGHT", teacher=True),
            _edge(episode_id, "prefix_0", "prefix_1", "LEFT", teacher=True),
            _edge(episode_id, "prefix_1", "prefix_2", "RIGHT", teacher=True),
            _edge(episode_id, "prefix_2", "anchor_0", "STRAIGHT", teacher=True),
        ]
    )

    for module in range(8):
        alias_depth = 4 + module
        tail_count = alias_depth - 1
        nodes.append(
            _node(
                episode_id,
                f"anchor_{module}",
                "ANCHOR",
                observation_recipe(family=family, kind="ANCHOR", module=module),
                module=module,
            )
        )
        next_anchor = "goal" if module == 7 else f"anchor_{module + 1}"
        for side in (0, 1):
            side_name = "left" if side == 0 else "right"
            marker = f"module_{module}_{side_name}_marker"
            nodes.append(
                _node(
                    episode_id,
                    marker,
                    "MARKER",
                    observation_recipe(
                        family=family, kind="BRANCH_MARKER", module=module, side=side
                    ),
                    module=module,
                    side=side,
                )
            )
            # Both anchor branches deliberately share the same executed action.
            edges.append(
                _edge(
                    episode_id,
                    f"anchor_{module}",
                    marker,
                    "LEFT" if side == 0 else "RIGHT",
                    executed_action="STRAIGHT",
                    teacher=True,
                )
            )
            previous = marker
            tail_ports = tuple(
                "STRAIGHT" if index % 2 == 0 else "REVERSE"
                for index in range(tail_count)
            )
            for step, port in enumerate(tail_ports):
                tail = f"module_{module}_{side_name}_tail_{step}"
                nodes.append(
                    _node(
                        episode_id,
                        tail,
                        "ALIAS_TAIL",
                        observation_recipe(
                            family=family, kind="TAIL", module=module, step=step, side=side
                        ),
                        module=module,
                        side=side,
                        # The images alias exactly, but only the registered
                        # query pair is a place-identity alias target.
                        alias_group_id=f"singleton:{episode_id}:{tail}",
                    )
                )
                edges.append(
                    _edge(
                        episode_id,
                        previous,
                        tail,
                        port,
                        teacher=True,
                    )
                )
                previous = tail
            query_local = f"module_{module}_{side_name}_query"
            nodes.append(
                _node(
                    episode_id,
                    query_local,
                    _query_node_kind(family),
                    observation_recipe(
                        family=family, kind="QUERY", module=module, step=4, side=side
                    ),
                    module=module,
                    side=side,
                    alias_group_id=_query_alias_group(
                        episode_id, module, side, family
                    ),
                )
            )
            edges.append(
                _edge(
                    episode_id,
                    previous,
                    query_local,
                    "STRAIGHT",
                    teacher=True,
                )
            )
            correct_port = "LEFT" if side == 0 else "RIGHT"
            wrong_port = "RIGHT" if side == 0 else "LEFT"
            edges.append(
                _edge(
                    episode_id,
                    query_local,
                    next_anchor,
                    correct_port,
                    teacher=True,
                )
            )
            dead_0 = f"module_{module}_{side_name}_dead_0"
            dead_1 = f"module_{module}_{side_name}_dead_1"
            for dead_step, dead_local in enumerate((dead_0, dead_1)):
                nodes.append(
                    _node(
                        episode_id,
                        dead_local,
                        "LONGER_ALTERNATE",
                        observation_recipe(
                            family=family,
                            kind="DEAD_END",
                            module=module,
                            step=dead_step,
                            side=side,
                        ),
                        module=module,
                        side=side,
                    )
                )
            edges.extend(
                [
                    _edge(episode_id, query_local, dead_0, wrong_port, teacher=True),
                    _edge(episode_id, dead_0, dead_1, "REVERSE", teacher=True),
                    _edge(
                        episode_id,
                        dead_1,
                        f"anchor_{module}",
                        "STRAIGHT",
                        teacher=True,
                    ),
                ]
            )

    if family == "LOOP_ALIAS":
        # A substantive cycle crosses two module identities whose query/tail
        # pixel templates alias. The Phase-A traversal below executes it.
        edges.append(
            _edge(
                episode_id,
                "module_1_right_dead_1",
                "anchor_0",
                "REVERSE",
                teacher=True,
                oracle_admissible=False,
            )
        )

    # Phase A is a genuine connected survey over registered executable edges.
    # It starts at prefix_0, traverses both sides of every module (returning via
    # each longer branch), and encounters the goal exactly once as its final
    # visit. Phase B/C then begins from that just-observed goal.
    node_by_local = {str(node["node_id"]).split(":", 1)[1]: node for node in nodes}
    edge_by_triplet = {
        (
            str(edge["source_node_id"]).split(":", 1)[1],
            str(edge["target_node_id"]).split(":", 1)[1],
            str(edge["port_label"]),
        ): edge
        for edge in edges
    }
    phase_a: list[dict[str, Any]] = []
    phase_a_seen: set[str] = set()

    def phase_a_observation_id(local_node_id: str, visit_index: int) -> str:
        node = node_by_local[local_node_id]
        node_id = str(node["node_id"])
        if node_id not in phase_a_seen:
            phase_a_seen.add(node_id)
            return str(node["keyframe_observation_id"])
        return _capture_id(
            episode_id, local_node_id, f"PHASE_A_REVISIT_{visit_index:04d}"
        )

    def phase_a_start(local_node_id: str) -> None:
        node = node_by_local[local_node_id]
        phase_a.append(
            {
                "visit_index": 0,
                "node_id": node["node_id"],
                "observation_id": phase_a_observation_id(local_node_id, 0),
                "arrival_edge_id": None,
                "executed_action_label": None,
                "timestamp_s": 0.0,
            }
        )

    def phase_a_step(source: str, target: str, port: str) -> None:
        edge = edge_by_triplet[(source, target, port)]
        if not edge["physically_executable"] or not edge["teacher_edge"]:
            raise ExperimentError("Phase-A traversal used an ineligible edge")
        node = node_by_local[target]
        visit_index = len(phase_a)
        phase_a.append(
            {
                "visit_index": visit_index,
                "node_id": node["node_id"],
                "observation_id": phase_a_observation_id(target, visit_index),
                "arrival_edge_id": edge["edge_id"],
                "executed_action_label": edge["executed_action_label"],
                "timestamp_s": round(0.5 * visit_index, 6),
            }
        )

    phase_a_start("prefix_0")
    phase_a_step("prefix_0", "prefix_1", "LEFT")
    phase_a_step("prefix_1", "prefix_2", "RIGHT")
    phase_a_step("prefix_2", "anchor_0", "STRAIGHT")
    if family == "LOOP_ALIAS":
        # Execute the registered cross-module alias cycle before continuing the
        # survey. Its two query locations have identical local templates and
        # opposite shortest-path ports.
        previous = "anchor_0"
        for module, side in ((0, 0), (1, 1)):
            side_name = "left" if side == 0 else "right"
            marker = f"module_{module}_{side_name}_marker"
            phase_a_step(previous, marker, "LEFT" if side == 0 else "RIGHT")
            previous = marker
            for step in range((4 + module) - 1):
                tail = f"module_{module}_{side_name}_tail_{step}"
                port = "STRAIGHT" if step % 2 == 0 else "REVERSE"
                phase_a_step(previous, tail, port)
                previous = tail
            query_local = f"module_{module}_{side_name}_query"
            phase_a_step(previous, query_local, "STRAIGHT")
            if module == 0:
                phase_a_step(query_local, "anchor_1", "LEFT")
                previous = "anchor_1"
            else:
                dead_0 = "module_1_right_dead_0"
                dead_1 = "module_1_right_dead_1"
                phase_a_step(query_local, dead_0, "LEFT")
                phase_a_step(dead_0, dead_1, "REVERSE")
                phase_a_step(dead_1, "anchor_0", "REVERSE")
                previous = "anchor_0"
    for module, teacher_side in enumerate(teacher_path):
        alias_depth = 4 + module
        tail_count = alias_depth - 1
        anchor = f"anchor_{module}"
        for side in (0, 1):
            side_name = "left" if side == 0 else "right"
            marker = f"module_{module}_{side_name}_marker"
            phase_a_step(anchor, marker, "LEFT" if side == 0 else "RIGHT")
            previous = marker
            for step in range(tail_count):
                tail = f"module_{module}_{side_name}_tail_{step}"
                port = "STRAIGHT" if step % 2 == 0 else "REVERSE"
                phase_a_step(previous, tail, port)
                previous = tail
            query_local = f"module_{module}_{side_name}_query"
            phase_a_step(previous, query_local, "STRAIGHT")
            wrong_port = "RIGHT" if side == 0 else "LEFT"
            dead_0 = f"module_{module}_{side_name}_dead_0"
            dead_1 = f"module_{module}_{side_name}_dead_1"
            phase_a_step(query_local, dead_0, wrong_port)
            phase_a_step(dead_0, dead_1, "REVERSE")
            phase_a_step(dead_1, anchor, "STRAIGHT")
        # Revisit the deterministic teacher side and take its shortest edge to
        # the next module (or the final goal at module seven).
        side_name = "left" if int(teacher_side) == 0 else "right"
        marker = f"module_{module}_{side_name}_marker"
        phase_a_step(anchor, marker, "LEFT" if int(teacher_side) == 0 else "RIGHT")
        previous = marker
        for step in range(tail_count):
            tail = f"module_{module}_{side_name}_tail_{step}"
            port = "STRAIGHT" if step % 2 == 0 else "REVERSE"
            phase_a_step(previous, tail, port)
            previous = tail
        query_local = f"module_{module}_{side_name}_query"
        phase_a_step(previous, query_local, "STRAIGHT")
        correct_port = "LEFT" if int(teacher_side) == 0 else "RIGHT"
        next_anchor = "goal" if module == 7 else f"anchor_{module + 1}"
        phase_a_step(query_local, next_anchor, correct_port)
    first_visit: dict[str, int] = {}
    for visit in phase_a:
        first_visit.setdefault(str(visit["node_id"]), int(visit["visit_index"]))
    if set(first_visit) != {str(node["node_id"]) for node in nodes}:
        raise ExperimentError("Phase-A traversal did not encounter every belief-bank node")
    if [visit["node_id"] for visit in phase_a].count(f"{episode_id}:goal") != 1:
        raise ExperimentError("Phase-A must encounter the goal exactly once")
    if phase_a[-1]["node_id"] != f"{episode_id}:goal":
        raise ExperimentError("Phase-A goal encounter must be the final visit")
    for node in nodes:
        node["teacher_visit_index"] = first_visit[str(node["node_id"])]
        node["keyframe_timestamp_s"] = float(
            phase_a[first_visit[str(node["node_id"]) ]]["timestamp_s"]
        )

    # anchor_8 is represented by the already-observed GOAL keyframe.
    node_ids = [str(node["node_id"]) for node in nodes]
    trace_nodes = ["goal", "prefix_0", "prefix_1", "prefix_2", "anchor_0"]
    trace_actions = ["STRAIGHT", "LEFT", "RIGHT", "STRAIGHT"]
    queries: list[dict[str, Any]] = []
    for module, side in enumerate(teacher_path):
        alias_depth = 4 + module
        tail_count = alias_depth - 1
        side_name = "left" if side == 0 else "right"
        other_name = "right" if side == 0 else "left"
        marker = f"module_{module}_{side_name}_marker"
        trace_actions.append("STRAIGHT")
        trace_nodes.append(marker)
        tail_ports = tuple(
            "STRAIGHT" if index % 2 == 0 else "REVERSE"
            for index in range(tail_count)
        )
        for step, port in enumerate(tail_ports):
            trace_actions.append(port)
            trace_nodes.append(f"module_{module}_{side_name}_tail_{step}")
        trace_actions.append("STRAIGHT")
        true_query = f"module_{module}_{side_name}_query"
        alias_query = f"module_{module}_{other_name}_query"
        trace_nodes.append(true_query)
        correct_port = "LEFT" if side == 0 else "RIGHT"
        correct_edge = next(
            edge
            for edge in edges
            if edge["source_node_id"] == f"{episode_id}:{true_query}"
            and edge["port_label"] == correct_port
        )
        partner_module, partner_side = _alias_partner(
            module, int(side), family
        )
        partner_side_name = "left" if partner_side == 0 else "right"
        alias_ids = [
            f"{episode_id}:{true_query}",
            f"{episode_id}:module_{partner_module}_{partner_side_name}_query",
        ]
        history_node_ids = [f"{episode_id}:{value}" for value in trace_nodes]
        history_observation_ids = [
            _capture_id(episode_id, value, "PHASE_C_OBSERVATION")
            for value in trace_nodes
        ]
        query_node = node_by_local[true_query]
        query_timestamp = round(
            float(phase_a[-1]["timestamp_s"]) + 1.0 + 0.1 * (len(trace_nodes) - 1),
            6,
        )
        admissible_edges = [
            edge["edge_id"]
            for edge in edges
            if edge["source_node_id"] == f"{episode_id}:{true_query}"
            and edge["physically_executable"]
            and edge["oracle_admissible"]
        ]
        if len(admissible_edges) < 2:
            raise ExperimentError("query has fewer than two executable admissible route edges")
        queries.append(
            {
                "query_id": f"{episode_id}:query_{module}",
                "episode_id": episode_id,
                "family": family,
                "role": role,
                "graph_id": graph_id,
                "query_index": module,
                "depth_since_last_unambiguous_observation": alias_depth,
                "true_node_id": f"{episode_id}:{true_query}",
                "true_next_edge_id": correct_edge["edge_id"],
                "true_next_port_label": correct_port,
                "prior_visited_node_ids": history_node_ids[:-1],
                "history_observation_ids": history_observation_ids,
                "history_executed_action_labels": list(trace_actions),
                "unresolved_condition_ids": [
                    "CURRENT_FRAME_NEAREST_NODE",
                    "FIXED_WINDOW_SEQUENCE",
                    "NO_ACTION_CONSISTENCY",
                    "SHUFFLED_ACTION_HISTORY",
                    "NO_OBSERVATION_LIKELIHOOD",
                ],
                "candidate_node_ids": list(node_ids),
                "alias_node_ids": alias_ids,
                "alias_distractor_node_ids": [
                    value for value in alias_ids if value != f"{episode_id}:{true_query}"
                ],
                "oracle_admissible_edge_ids": admissible_edges,
                "query_observation_id": history_observation_ids[-1],
                "query_pixel_template_id": query_node["pixel_template_id"],
                "query_pixel_sha256": query_node["pixel_sha256"],
                "query_timestamp_s": query_timestamp,
                "goal_visible": False,
            }
        )
        next_anchor = "goal" if module == 7 else f"anchor_{module + 1}"
        trace_actions.append(correct_port)
        trace_nodes.append(next_anchor)

    if len(queries) != 8 or trace_nodes.count("goal") != 2:
        raise ExperimentError("teacher trace structure drift")
    if len(set(trace_nodes[:-1])) != len(trace_nodes) - 1:
        raise ExperimentError("teacher trace repeats a nonterminal place")
    edge_by_id = {str(edge["edge_id"]): edge for edge in edges}
    prelude_edge_ids = _deterministic_shortest_edge_path(
        edges,
        f"{episode_id}:goal",
        str(queries[0]["true_node_id"]),
    )
    prelude_actions = [
        str(edge_by_id[edge_id]["executed_action_label"])
        for edge_id in prelude_edge_ids
    ]
    if prelude_actions != list(queries[0]["history_executed_action_labels"]):
        raise ExperimentError("Stage-B prelude differs from the registered complete history")
    for query_index, query in enumerate(queries):
        correct_edge_id = str(query["true_next_edge_id"])
        correct_edge = edge_by_id[correct_edge_id]
        correct_destination = (
            f"{episode_id}:goal"
            if query_index == 7
            else str(queries[query_index + 1]["true_node_id"])
        )
        correct_macro = [
            correct_edge_id,
            *_deterministic_shortest_edge_path(
                edges,
                str(correct_edge["target_node_id"]),
                correct_destination,
            ),
        ]
        alternate_edges = [
            edge_by_id[str(edge_id)]
            for edge_id in query["oracle_admissible_edge_ids"]
            if str(edge_id) != correct_edge_id
        ]
        if len(alternate_edges) != 1:
            raise ExperimentError("Stage-B requires one prospectively fixed wrong query edge")
        wrong_edge = alternate_edges[0]
        wrong_macro = [
            str(wrong_edge["edge_id"]),
            *_deterministic_shortest_edge_path(
                edges,
                str(wrong_edge["target_node_id"]),
                str(query["true_node_id"]),
            ),
        ]
        query["stage_b_choice_macros"] = {
            correct_edge_id: {
                "outcome": "CORRECT_ADVANCE",
                "destination_node_id": correct_destination,
                "constituent_edge_ids": correct_macro,
            },
            str(wrong_edge["edge_id"]): {
                "outcome": "WRONG_RETURN",
                "destination_node_id": str(query["true_node_id"]),
                "constituent_edge_ids": wrong_macro,
            },
        }
        query["stage_b_local_candidate_outcomes"] = _stage_b_local_candidate_outcomes(
            query, edges
        )
    procedural_seed = int(CONTRACT.PROCEDURAL_SEED_BASE) + global_index
    episode_path_id = str(CONTRACT.EPISODE_PATH_ID_PREFIX) + hashlib.sha256(
        canonical_bytes({"episode_id": episode_id, "teacher_path": list(teacher_path)})
    ).hexdigest()[:24]
    witness_partner_module, witness_partner_side = _alias_partner(0, 0, family)
    witness_alias_ids = [
        f"{episode_id}:module_0_left_query",
        f"{episode_id}:module_{witness_partner_module}_{'left' if witness_partner_side == 0 else 'right'}_query",
    ]
    witness_edges = [
        edge for edge in edges if edge["source_node_id"] in witness_alias_ids
        and edge["port_label"] in QUERY_PORTS
    ]
    cycle_nodes: list[str] = []
    cycle_edges: list[str] = []
    if family == "LOOP_ALIAS":
        cycle_locals = [
            "anchor_0",
            "module_0_left_marker",
            *[f"module_0_left_tail_{step}" for step in range(3)],
            "module_0_left_query",
            "anchor_1",
            "module_1_right_marker",
            *[f"module_1_right_tail_{step}" for step in range(4)],
            "module_1_right_query",
            "module_1_right_dead_0",
            "module_1_right_dead_1",
        ]
        cycle_nodes = [f"{episode_id}:{value}" for value in cycle_locals]
        for source, target in zip(cycle_nodes, cycle_nodes[1:] + cycle_nodes[:1]):
            cycle_edges.append(
                min(
                    (
                        edge for edge in edges
                        if edge["source_node_id"] == source
                        and edge["target_node_id"] == target
                    ),
                    key=lambda edge: (edge["edge_cost"], edge["edge_id"]),
                )["edge_id"]
            )
    episode = {
        "episode_id": episode_id,
        "scene_id": str(CONTRACT.SCENE_ID_PREFIX) + hashlib.sha256(
            canonical_bytes(
                {
                    "family": family,
                    "rank": family_rank,
                    "teacher_path": list(teacher_path),
                }
            )
        ).hexdigest()[:24],
        "procedural_seed": procedural_seed,
        "episode_path_id": episode_path_id,
        "identity_domain": CONTRACT.IDENTITY_DOMAIN,
        "family": family,
        "role": role,
        "graph_id": graph_id,
        "goal_node_id": f"{episode_id}:goal",
        "stage_b_start_node_id": f"{episode_id}:goal",
        "stage_b_decision_start_node_id": str(queries[0]["true_node_id"]),
        "stage_b_prelude_edge_ids": prelude_edge_ids,
        "nodes": nodes,
        "edges": edges,
        "phase_a_traversal": phase_a,
        "family_adequacy_witness": {
            "family": family,
            "alias_group_id": _query_alias_group(episode_id, 0, 0, family),
            "alias_node_ids": witness_alias_ids,
            "different_next_port_labels": ["LEFT", "RIGHT"],
            "witness_node_ids": witness_alias_ids,
            "witness_edge_ids": sorted(edge["edge_id"] for edge in witness_edges),
            "cycle_node_ids": cycle_nodes,
            "cycle_edge_ids": cycle_edges,
        },
        "constructed_set_caveat": getattr(
            CONTRACT,
            "CONSTRUCTED_SET_CAVEAT",
            "This is a constructed perceptual-aliasing challenge set, not an estimate of natural alias prevalence.",
        ),
    }
    del node_by_local
    return episode, queries


def _graph_distances_to_goal(episode: Mapping[str, Any]) -> dict[str, int | None]:
    reverse: dict[str, list[str]] = defaultdict(list)
    for edge in episode["edges"]:
        reverse[str(edge["target_node_id"])].append(str(edge["source_node_id"]))
    goal = str(episode["goal_node_id"])
    distances: dict[str, int | None] = {
        str(node["node_id"]): None for node in episode["nodes"]
    }
    distances[goal] = 0
    queue: deque[str] = deque((goal,))
    while queue:
        node = queue.popleft()
        assert distances[node] is not None
        for previous in reverse.get(node, ()):
            if distances[previous] is None:
                distances[previous] = int(distances[node]) + 1
                queue.append(previous)
    return distances


def validate_episode_structure(
    episode: Mapping[str, Any], queries: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    node_by_id = {str(node["node_id"]): node for node in episode["nodes"]}
    if len(node_by_id) != len(episode["nodes"]):
        raise ExperimentError("duplicate graph node identity")
    edges_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in episode["edges"]:
        source = str(edge["source_node_id"])
        target = str(edge["target_node_id"])
        if source not in node_by_id or target not in node_by_id:
            raise ExperimentError("edge endpoint absent from graph")
        edges_by_source[source].append(dict(edge))
    distances = _graph_distances_to_goal(episode)
    if any(value is None for value in distances.values()):
        raise ExperimentError("episode graph contains a node that cannot reach goal")
    unresolved = 0
    for query in queries:
        candidates = [str(value) for value in query["alias_node_ids"]]
        left, right = (node_by_id[value] for value in candidates)
        if left["observation_descriptor"] != right["observation_descriptor"]:
            raise ExperimentError("alias query observation IDs are not byte-identical")
        depth = int(query["depth_since_last_unambiguous_observation"])
        module = int(query["query_index"])
        window_pair = [
            f"{query['episode_id']}:module_{module}_left_query",
            f"{query['episode_id']}:module_{module}_right_query",
        ]
        for step in range(depth - 1):
            left_tail = node_by_id[window_pair[0].replace("_query", f"_tail_{step}")]
            right_tail = node_by_id[window_pair[1].replace("_query", f"_tail_{step}")]
            if left_tail["observation_descriptor"] != right_tail["observation_descriptor"]:
                raise ExperimentError("registered alias tail is not exact")
        true_node = str(query["true_node_id"])
        port = str(query["true_next_port_label"])
        selected = [edge for edge in edges_by_source[true_node] if edge["port_label"] == port]
        alternate = [edge for edge in edges_by_source[true_node] if edge["port_label"] in QUERY_PORTS and edge["port_label"] != port]
        if len(selected) != 1 or len(alternate) != 1:
            raise ExperimentError("query port cardinality drift")
        if int(distances[str(selected[0]["target_node_id"])]) >= int(distances[str(alternate[0]["target_node_id"])]):
            raise ExperimentError("teacher query edge is not strictly shorter to goal")
        if not {"CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE"}.issubset(
            set(query["unresolved_condition_ids"])
        ):
            raise ExperimentError("current/fixed query is not registered unresolved")
        if "FULL_BELIEF" in query["unresolved_condition_ids"]:
            raise ExperimentError("full history is incorrectly marked unresolved")
        unresolved += 1
    goal_node = node_by_id[str(episode["goal_node_id"])]
    if json.loads(goal_node["observation_descriptor"])["goal_marker"] is not True:
        raise ExperimentError("goal keyframe lacks its recorded goal marker")
    if any(
        json.loads(node["observation_descriptor"])["goal_marker"]
        for node in episode["nodes"]
        if node["node_kind"] in {
            "CORRIDOR_SEGMENT", "MIRRORED_JUNCTION", "LOOP_LOCATION", "REPEATED_ROOM"
        }
    ):
        raise ExperimentError("goal marker leaked into a query")
    return {
        "nodes": len(node_by_id),
        "edges": len(episode["edges"]),
        "queries": len(queries),
        "structurally_unresolved_queries": unresolved,
        "goal_prefix_edge_separation": 4,
        "minimum_alias_depth": min(
            int(query["depth_since_last_unambiguous_observation"])
            for query in queries
        ),
        "maximum_alias_depth": max(
            int(query["depth_since_last_unambiguous_observation"])
            for query in queries
        ),
        "phase_a_visits": len(episode["phase_a_traversal"]),
        "phase_a_all_nodes_covered": {
            str(visit["node_id"]) for visit in episode["phase_a_traversal"]
        } == set(node_by_id),
        "loop_cycle_present": bool(episode["family_adequacy_witness"]["cycle_edge_ids"]),
    }


def build_panel_documents() -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    paths = teacher_side_paths()
    episodes: list[dict[str, Any]] = []
    queries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    cursor = 0
    for family in FAMILIES:
        for family_rank in range(24):
            episode, episode_queries = build_episode_graph(
                family=family,
                family_rank=family_rank,
                teacher_path=paths[cursor],
            )
            cursor += 1
            checks.append(validate_episode_structure(episode, episode_queries))
            episodes.append(episode)
            queries.extend(episode_queries)
    role_counts = {role: sum(ep["role"] == role for ep in episodes) for role in ROLES}
    family_counts = {family: sum(ep["family"] == family for ep in episodes) for family in FAMILIES}
    if role_counts != {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16}:
        raise ExperimentError(f"role count drift: {role_counts}")
    if family_counts != {family: 24 for family in FAMILIES}:
        raise ExperimentError(f"family count drift: {family_counts}")
    if len(queries) != 768 or len({row["query_id"] for row in queries}) != 768:
        raise ExperimentError("query count or identity drift")
    if len({episode["scene_id"] for episode in episodes}) != 96:
        raise ExperimentError("scene identity collision")
    if len({episode["procedural_seed"] for episode in episodes}) != 96:
        raise ExperimentError("procedural seed collision")
    if len({episode["episode_path_id"] for episode in episodes}) != 96:
        raise ExperimentError("episode path identity collision")
    identity_disjointness = build_identity_disjointness(episodes, paths)
    panel = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.panel_manifest.v1",
            "experiment_id": "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "episode_count": len(episodes),
            "queries_per_episode": 8,
            "query_count": len(queries),
            "family_counts": family_counts,
            "role_counts": role_counts,
            "teacher_paths_unique": len(set(paths)),
            "teacher_path_bit_count_per_module": [
                sum(int(path[index]) for path in paths)
                for index in range(8)
            ],
            "adequacy": {
                "pass": True,
                "all_phase_a_surveys_end_at_single_goal_keyframe": True,
                "all_phase_c_histories_begin_at_goal": True,
                "minimum_goal_to_first_anchor_edges": min(row["goal_prefix_edge_separation"] for row in checks),
                "alias_depths": list(range(4, 12)),
                "structurally_unresolved_queries": sum(row["structurally_unresolved_queries"] for row in checks),
                "opposite_next_port_alias_pairs": len(queries),
                "goal_marker_query_leak_count": 0,
                "phase_a_all_graph_nodes_observed": all(
                    row["phase_a_all_nodes_covered"] for row in checks
                ),
                "loop_alias_cycle_episode_count": sum(
                    row["loop_cycle_present"] for row in checks
                ),
                "distinct_procedural_seeds": 96,
                "distinct_episode_paths": 96,
                "prior_identity_overlap_counts": dict(
                    identity_disjointness["comparisons"]
                ),
            },
            "identity_disjointness": identity_disjointness,
            "episodes": [
                {
                    key: episode[key]
                    for key in (
                        "episode_id",
                        "scene_id",
                        "family",
                        "role",
                        "graph_id",
                        "goal_node_id",
                        "procedural_seed",
                        "episode_path_id",
                        "identity_domain",
                    )
                }
                for episode in episodes
            ],
        }
    )
    graph_manifest = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.graph_manifest.v1",
            "experiment_id": "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "identity_disjointness": identity_disjointness,
            "graphs": episodes,
        }
    )
    return panel, graph_manifest, queries


def build_split_manifest(panel: Mapping[str, Any]) -> dict[str, Any]:
    rows = [
        {
            "episode_id": episode["episode_id"],
            "scene_id": episode["scene_id"],
            "family": episode["family"],
            "role": episode["role"],
        }
        for episode in panel["episodes"]
    ]
    return attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.split_manifest.v1",
            "roles": list(ROLES),
            "counts": dict(panel["role_counts"]),
            "rows": rows,
        }
    )


def panel_stage() -> dict[str, Any]:
    require_new_output_root()
    source_freeze = require_source_freeze()
    contract_builder = getattr(CONTRACT, "build_contract", None)
    if not callable(contract_builder):
        raise ExperimentError("contract module lacks build_contract")
    scientific_contract = dict(contract_builder())
    validator = getattr(CONTRACT, "validate_contract", None)
    if callable(validator):
        validator(scientific_contract)
    runtime_contract = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.runtime_contract.v1",
            "source_freeze_commit": source_freeze,
            "parent_commit": PARENT_COMMIT,
            "scientific_contract": scientific_contract,
        }
    )
    panel, graph, queries = build_panel_documents()
    split = build_split_manifest(panel)
    # Only after all ten predecessor authorities, six projection digests, and
    # six zero-overlap comparisons pass may the official output root exist.
    require_new_output_root()
    OUTPUT_ROOT.mkdir(mode=0o755)
    atomic_json(OUTPUT_ROOT / "contract.json", runtime_contract)
    atomic_json(OUTPUT_ROOT / "panel_manifest.json", panel)
    atomic_json(OUTPUT_ROOT / "split_manifest.json", split)
    atomic_json(OUTPUT_ROOT / "graph_manifest.json", graph)
    write_jsonl(OUTPUT_ROOT / "query_ledger.jsonl", queries)
    return panel


def _fill_rect(image: Any, x0: int, y0: int, x1: int, y1: int, color: Sequence[int]) -> None:
    image[max(0, y0):min(IMAGE_HEIGHT, y1), max(0, x0):min(IMAGE_WIDTH, x1)] = color


def render_observation(recipe: Mapping[str, Any]) -> Any:
    """Render one neutral deterministic observation; alias recipes are exact."""

    import numpy as np

    if int(recipe["height"]) != IMAGE_HEIGHT or int(recipe["width"]) != IMAGE_WIDTH:
        raise ExperimentError("observation recipe image shape drift")
    family = str(recipe["family"])
    family_index = FAMILIES.index(family)
    kind = str(recipe["kind"])
    module = int(recipe["module"])
    step = int(recipe["step"])
    side = int(recipe["side"])
    image = np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=np.uint8)
    image[:84] = (118, 139, 151)
    for row in range(84, IMAGE_HEIGHT):
        shade = 102 + (row - 84) // 3
        image[row] = (shade, shade, shade - 7)
    # Fixed neutral corridor geometry. Family affects geometry, never palette,
    # lighting, or the left/right alias pair.
    inset = (family_index * 7 + max(module, 0) * 3 + max(step, 0) * 2) % 24
    wall = (126, 126, 122)
    _fill_rect(image, 0, 30 + inset // 2, 36 + inset, 168, wall)
    _fill_rect(image, 188 - inset, 24 + inset // 3, 224, 168, wall)
    if family in {"MIRRORED_JUNCTION", "LOOP_ALIAS"}:
        _fill_rect(image, 82, 42 + inset // 4, 142, 58 + inset // 4, wall)
    if family in {"REPEATED_ROOM", "LOOP_ALIAS"}:
        _fill_rect(image, 103, 72, 121, 132, (114, 114, 111))
    # A route marker is visual evidence but never the goal marker.
    if kind == "BRANCH_MARKER":
        if side not in (0, 1):
            raise ExperimentError("branch marker requires side 0/1")
        cue_color = (80, 105, 138) if side == 0 else (138, 101, 80)
        cue_x = 54 if side == 0 else 150
        _fill_rect(image, cue_x - 14, 45, cue_x + 14, 105, cue_color)
        _fill_rect(image, cue_x - 20, 51, cue_x + 20, 59, cue_color)
    if kind == "GOAL_KEYFRAME":
        _fill_rect(image, 95, 38, 129, 116, (155, 137, 65))
        _fill_rect(image, 86, 52, 138, 66, (155, 137, 65))
    if bool(recipe["goal_marker"]) != (kind == "GOAL_KEYFRAME"):
        raise ExperimentError("goal-marker recipe invariant drift")
    return np.ascontiguousarray(image)


def render_stage() -> dict[str, Any]:
    require_source_freeze()
    for leaf in ("keyframe_index.json", "observations.npz"):
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError(f"render output already exists: {leaf}")
    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    queries = load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl")
    verify_persisted_identity_disjointness(graph, queries)
    recipes: dict[str, dict[str, Any]] = {}
    occurrences: dict[str, dict[str, Any]] = {}
    nodes_by_episode: dict[str, dict[str, dict[str, Any]]] = {}
    for episode in graph["graphs"]:
        episode_id = str(episode["episode_id"])
        nodes_by_episode[episode_id] = {
            str(node["node_id"]): dict(node) for node in episode["nodes"]
        }
        for node in episode["nodes"]:
            recipe = json.loads(str(node["observation_descriptor"]))
            template_id = str(recipe["pixel_template_id"])
            if template_id in recipes and recipes[template_id] != recipe:
                raise ExperimentError("pixel-template identity collision")
            recipes[template_id] = recipe
        for visit in episode["phase_a_traversal"]:
            node = nodes_by_episode[episode_id][str(visit["node_id"])]
            capture_id = str(visit["observation_id"])
            if capture_id in occurrences:
                raise ExperimentError("duplicate Phase-A capture occurrence")
            occurrences[capture_id] = {
                "capture_id": capture_id,
                "episode_id": episode_id,
                "node_id": node["node_id"],
                "phase": (
                    "PHASE_A_KEYFRAME"
                    if capture_id == node["keyframe_observation_id"]
                    else "PHASE_A_REVISIT"
                ),
                "timestamp_s": float(visit["timestamp_s"]),
                "pixel_template_id": node["pixel_template_id"],
                "query_ids": [],
            }
    for query in queries:
        episode_id = str(query["episode_id"])
        graph_episode = next(
            graph_row for graph_row in graph["graphs"]
            if graph_row["episode_id"] == episode_id
        )
        phase_a_final_timestamp = float(graph_episode["phase_a_traversal"][-1]["timestamp_s"])
        history_nodes = [*query["prior_visited_node_ids"], query["true_node_id"]]
        history_ids = list(query["history_observation_ids"])
        if len(history_nodes) != len(history_ids):
            raise ExperimentError("Phase-C node/capture history alignment drift")
        for history_index, (node_id, capture_id) in enumerate(zip(history_nodes, history_ids)):
            node = nodes_by_episode[episode_id][str(node_id)]
            recipe = json.loads(str(node["observation_descriptor"]))
            row = {
                "capture_id": str(capture_id),
                "episode_id": episode_id,
                "node_id": str(node_id),
                "phase": "PHASE_C_QUERY_TRACE",
                "timestamp_s": round(
                    phase_a_final_timestamp + 1.0 + 0.1 * history_index, 6
                ),
                "pixel_template_id": str(recipe["pixel_template_id"]),
                "query_ids": [],
            }
            existing = occurrences.get(str(capture_id))
            if existing is not None:
                if {key: existing[key] for key in row if key != "query_ids"} != {
                    key: row[key] for key in row if key != "query_ids"
                }:
                    raise ExperimentError("Phase-C capture identity collision")
                row = existing
            occurrences[str(capture_id)] = row
        occurrences[str(history_ids[-1])]["query_ids"].append(str(query["query_id"]))
        if not math.isclose(
            float(occurrences[str(history_ids[-1])]["timestamp_s"]),
            float(query["query_timestamp_s"]),
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ExperimentError("query capture timestamp binding drift")
    ids = sorted(recipes)
    images = [render_observation(recipes[template_id]) for template_id in ids]
    import numpy as np

    image_array = np.stack(images).astype(np.uint8, copy=False)
    hashes = [hashlib.sha256(image.tobytes(order="C")).hexdigest() for image in image_array]
    # Every alias node pair shares the same recipe ID, and therefore the exact
    # same stored byte row. Distinct marker sides must not accidentally alias.
    for family in FAMILIES:
        for module in range(8):
            left = observation_recipe(family=family, kind="BRANCH_MARKER", module=module, side=0)["pixel_template_id"]
            right = observation_recipe(family=family, kind="BRANCH_MARKER", module=module, side=1)["pixel_template_id"]
            if hashes[ids.index(left)] == hashes[ids.index(right)]:
                raise ExperimentError("visually distinct route markers rendered identically")
    atomic_npz(
        OUTPUT_ROOT / "observations.npz",
        {
            "schema": np.asarray(["occluded_goal_topological_belief_v1.observations.v1"]),
            "pixel_template_ids": np.asarray(ids),
            "image_sha256": np.asarray(hashes),
            "images": image_array,
        },
    )
    index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.keyframe_index.v1",
            "observations_file": {
                "path": "observations.npz",
                "bytes": (OUTPUT_ROOT / "observations.npz").stat().st_size,
                "sha256": sha256_file(OUTPUT_ROOT / "observations.npz"),
            },
            "image_shape": [IMAGE_HEIGHT, IMAGE_WIDTH, 3],
            "unique_pixel_template_count": len(ids),
            "capture_count": len(occurrences),
            "phase_a_capture_count": sum(
                row["phase"].startswith("PHASE_A") for row in occurrences.values()
            ),
            "phase_c_capture_count": sum(
                row["phase"] == "PHASE_C_QUERY_TRACE" for row in occurrences.values()
            ),
            "query_reference_disjoint": not (
                {
                    capture_id for capture_id, row in occurrences.items()
                    if row["phase"].startswith("PHASE_A")
                }
                & {
                    capture_id for capture_id, row in occurrences.items()
                    if row["phase"] == "PHASE_C_QUERY_TRACE"
                }
            ),
            "records": [
                {
                    "pixel_template_id": template_id,
                    "row_index": index,
                    "image_sha256": hashes[index],
                    "recipe": recipes[template_id],
                }
                for index, template_id in enumerate(ids)
            ],
            "captures": [occurrences[key] for key in sorted(occurrences)],
        }
    )
    atomic_json(OUTPUT_ROOT / "keyframe_index.json", index)
    return index


def _validate_npz_member_names(path: Path, expected: set[str]) -> None:
    _require_regular(path)
    with zipfile.ZipFile(path, "r") as archive:
        observed = {Path(name).stem for name in archive.namelist()}
        if observed != expected or any(Path(name).name != name for name in archive.namelist()):
            raise ExperimentError(f"NPZ member drift: {path}")


def token_layernorm_l2(value: Any) -> Any:
    """Return the exact registered aligned-token spatial descriptor.

    LayerNorm is applied independently to every 1,024-dimensional token, then
    every token is L2-normalized.  The result retains the 768-token spatial
    order; it is not a globally pooled image descriptor.
    """

    import numpy as np

    raw = np.asarray(value, dtype=np.float32)
    if raw.shape[-2:] != TOKEN_SHAPE or not np.isfinite(raw).all():
        raise ExperimentError(f"raw V-JEPA token shape/value drift: {raw.shape}")
    mean = raw.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = np.square(raw - mean, dtype=np.float32).mean(
        axis=-1, keepdims=True, dtype=np.float32
    )
    normalized = (raw - mean) / np.sqrt(variance + LAYER_NORM_EPSILON)
    norm = np.linalg.norm(normalized, axis=-1, keepdims=True)
    normalized = normalized / np.maximum(norm, np.float32(1.0e-12))
    if not np.isfinite(normalized).all():
        raise ExperimentError("non-finite token-normalized spatial descriptor")
    return np.ascontiguousarray(normalized, dtype=np.float32)


class SpatialDescriptorStore(Mapping[str, Any]):
    """Capture-indexed descriptors with cached aligned-token similarities."""

    def __init__(
        self,
        *,
        descriptors_by_template: Mapping[str, Any],
        template_by_capture: Mapping[str, str],
    ) -> None:
        self._descriptors = dict(descriptors_by_template)
        self._template_by_capture = dict(template_by_capture)
        self._similarity_cache: dict[tuple[str, tuple[str, ...]], Any] = {}

    def __getitem__(self, capture_id: str) -> Any:
        return self._descriptors[self._template_by_capture[str(capture_id)]]

    def __iter__(self):
        return iter(self._template_by_capture)

    def __len__(self) -> int:
        return len(self._template_by_capture)

    def observation_evidence(
        self,
        observation_id: str,
        reference_ids: Sequence[str],
        *,
        temperature: float,
        uniform: bool,
    ) -> tuple[Any, Any]:
        import numpy as np

        query_template = self._template_by_capture[str(observation_id)]
        reference_templates = tuple(
            self._template_by_capture[str(reference_id)]
            for reference_id in reference_ids
        )
        key = (query_template, reference_templates)
        similarity = self._similarity_cache.get(key)
        if similarity is None:
            query = np.asarray(self._descriptors[query_template], dtype=np.float32)
            unique_templates = list(dict.fromkeys(reference_templates))
            unique_values = np.stack(
                [self._descriptors[value] for value in unique_templates]
            ).astype(np.float32, copy=False)
            unique_similarity = (
                np.einsum("td,utd->u", query, unique_values, optimize=True)
                / np.float32(TOKEN_SHAPE[0])
            )
            lookup = dict(zip(unique_templates, unique_similarity.tolist()))
            similarity = np.asarray(
                [lookup[value] for value in reference_templates], dtype=np.float64
            )
            self._similarity_cache[key] = similarity
        if uniform:
            likelihood = np.full(
                len(reference_templates), 1.0 / len(reference_templates), dtype=np.float64
            )
        else:
            logits = (similarity - float(similarity.max())) / float(temperature)
            weights = np.exp(np.clip(logits, -80.0, 0.0))
            likelihood = weights / max(float(weights.sum()), 1.0e-300)
        return similarity.copy(), likelihood


def verify_external_vjepa_source() -> dict[str, Any]:
    """Fail closed on the exact clean V-JEPA source imported by the encoder."""

    binding = dict(CONTRACT.FROZEN_SOURCE_BINDINGS["vjepa_encoder_helper"])
    commit = str(binding["external_repository_commit"])
    repository = Path.home() / f".cache/vjepa2-{commit}"
    try:
        info = repository.lstat()
    except FileNotFoundError as exc:
        raise ExperimentError(f"frozen V-JEPA source repository is absent: {repository}") from exc
    if repository.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ExperimentError("frozen V-JEPA source repository is not an ordinary directory")
    try:
        observed = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=repository,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ExperimentError("frozen V-JEPA source custody check failed") from exc
    if observed != commit or dirty:
        raise ExperimentError("frozen V-JEPA source commit/cleanliness drift")
    return {
        "path": str(repository),
        "commit": observed,
        "worktree_clean": True,
    }


def encode_stage(
    batch_size: int = 8,
    *,
    arm_factory: Callable[[], Any] | None = None,
) -> dict[str, Any]:
    require_source_freeze()
    for leaf in ("latent_index.json", "latents.npz"):
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError(f"encode output already exists: {leaf}")
    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    queries = load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl")
    verify_persisted_identity_disjointness(graph, queries)
    import numpy as np
    import torch

    observation_path = OUTPUT_ROOT / "observations.npz"
    _validate_npz_member_names(
        observation_path, {"schema", "pixel_template_ids", "image_sha256", "images"}
    )
    with np.load(observation_path, allow_pickle=False) as data:
        ids = [str(value) for value in data["pixel_template_ids"].tolist()]
        images = np.asarray(data["images"], dtype=np.uint8)
        hashes = [str(value) for value in data["image_sha256"].tolist()]
    if images.shape != (len(ids), IMAGE_HEIGHT, IMAGE_WIDTH, 3):
        raise ExperimentError("observation tensor shape drift")
    if any(hashlib.sha256(image.tobytes(order="C")).hexdigest() != digest for image, digest in zip(images, hashes)):
        raise ExperimentError("observation image hash drift")
    if arm_factory is None:
        external_source = verify_external_vjepa_source()
        from scripts.dev_frozen_dense_representation_encoders_v1 import (
            VJepa21Arm,
            preprocessing_hash,
        )

        arm = VJepa21Arm()
        preprocess_digest_fn = preprocessing_hash
    else:
        external_source = {
            "path": "FAKE_TEST_ENCODER",
            "commit": "FAKE_TEST_ENCODER",
            "worktree_clean": True,
        }
        arm = arm_factory()
        preprocess_digest_fn = lambda value: str(getattr(value, "preprocessing_digest", "FAKE_TEST_ENCODER"))
    expected_checkpoint = str(
        getattr(CONTRACT, "OBSERVATION_LIKELIHOOD", {}).get(
            "checkpoint_sha256",
            "7ea9b7cb4a75d10644a8a8d42cff9e177b10dca8f02173f0eaf2b0bed82838c6",
        )
    )
    if arm_factory is None:
        checkpoint = Path(getattr(arm, "checkpoint"))
        if sha256_file(checkpoint) != expected_checkpoint:
            raise ExperimentError("frozen V-JEPA checkpoint binding mismatch")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arm.build(device, torch.float32)
    raw_tokens: list[np.ndarray] = []
    spatial_descriptors: list[np.ndarray] = []
    started = time.time()
    for offset in range(0, len(images), int(batch_size)):
        selected = images[offset:offset + int(batch_size)]
        if hasattr(arm, "preprocess_array"):
            pixels = torch.stack([arm.preprocess_array(image) for image in selected])
        else:
            from PIL import Image
            import tempfile

            prepared: list[torch.Tensor] = []
            with tempfile.TemporaryDirectory(prefix="ogtb-vjepa-") as directory:
                root = Path(directory)
                for index, image in enumerate(selected):
                    path = root / f"{index:04d}.png"
                    Image.fromarray(image, mode="RGB").save(path)
                    prepared.append(arm.preprocess(str(path)))
            pixels = torch.stack(prepared)
        pixels = pixels.to(device)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            encoded = arm.tokens(pixels)
        value = encoded.detach().float().cpu().numpy()
        if value.shape[1:] != TOKEN_SHAPE or not np.isfinite(value).all():
            raise ExperimentError(f"V-JEPA token contract drift: {value.shape}")
        for frame in value:
            raw_tokens.append(np.ascontiguousarray(frame, dtype=np.float16))
            spatial_descriptors.append(token_layernorm_l2(frame))
    raw_token_array = np.stack(raw_tokens).astype(np.float16, copy=False)
    descriptor_array = np.stack(spatial_descriptors).astype(np.float32, copy=False)
    atomic_npz(
        OUTPUT_ROOT / "latents.npz",
        {
            "schema": np.asarray(["occluded_goal_topological_belief_v1.latents.v1"]),
            "pixel_template_ids": np.asarray(ids),
            "raw_tokens": raw_token_array,
            "spatial_descriptors": descriptor_array,
        },
    )
    latent_hash = sha256_file(OUTPUT_ROOT / "latents.npz")
    index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.latent_index.v1",
            "complete": True,
            "encoder_checkpoint_sha256": expected_checkpoint,
            "external_encoder_source": external_source,
            "preprocessing_digest": preprocess_digest_fn(arm),
            "raw_token_shape": list(TOKEN_SHAPE),
            "raw_token_dtype": "float16",
            "token_normalized_representation": "spatial_descriptors",
            "spatial_descriptor_shape": list(TOKEN_SHAPE),
            "spatial_descriptor_dtype": "float32",
            "spatial_descriptor_rule": (
                "token-wise LayerNorm over 1024 with epsilon 1e-5; token-wise "
                "L2 normalization; preserve aligned 768-token order"
            ),
            "records": [
                {
                    "pixel_template_id": template_id,
                    "row_index": index,
                    "raw_token_sha256": hashlib.sha256(
                        raw_token_array[index].tobytes(order="C")
                    ).hexdigest(),
                    "spatial_descriptor_sha256": hashlib.sha256(
                        descriptor_array[index].tobytes(order="C")
                    ).hexdigest(),
                }
                for index, template_id in enumerate(ids)
            ],
            "captures": [
                {
                    "capture_id": capture["capture_id"],
                    "pixel_template_id": capture["pixel_template_id"],
                    "row_index": ids.index(str(capture["pixel_template_id"])),
                    "phase": capture["phase"],
                    "timestamp_s": capture["timestamp_s"],
                    "episode_id": capture["episode_id"],
                    "node_id": capture["node_id"],
                    "query_ids": capture["query_ids"],
                }
                for capture in load_json(OUTPUT_ROOT / "keyframe_index.json")["captures"]
            ],
            "latents_file": {
                "path": "latents.npz",
                "bytes": (OUTPUT_ROOT / "latents.npz").stat().st_size,
                "sha256": latent_hash,
            },
            "unique_pixel_template_count": len(ids),
            "batch_size": int(batch_size),
            "device": str(device),
            "runtime_s": time.time() - started,
        }
    )
    atomic_json(OUTPUT_ROOT / "latent_index.json", index)
    del arm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return index


def _episode_indices(episode: Mapping[str, Any]) -> tuple[list[str], dict[str, int], dict[str, list[dict[str, Any]]]]:
    node_ids = [str(value["node_id"]) for value in episode["nodes"]]
    index = {node_id: position for position, node_id in enumerate(node_ids)}
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in episode["edges"]:
        outgoing[str(edge["source_node_id"])].append(dict(edge))
    return node_ids, index, outgoing


def _observation_likelihood(
    observed: Any,
    prototypes: Any,
    *,
    temperature: float,
    uniform: bool = False,
) -> tuple[Any, Any]:
    """Aligned-token cosine similarities and their exact softmax likelihood."""

    import numpy as np

    query = np.asarray(observed, dtype=np.float64)
    reference = np.asarray(prototypes, dtype=np.float64)
    if query.shape != TOKEN_SHAPE or reference.ndim != 3 or reference.shape[1:] != TOKEN_SHAPE:
        raise ExperimentError("spatial descriptor shape drift")
    similarity = np.einsum("td,ntd->n", query, reference, optimize=True) / TOKEN_SHAPE[0]
    if uniform:
        return similarity, np.full(len(reference), 1.0 / len(reference), dtype=np.float64)
    logits = (similarity - float(similarity.max())) / float(temperature)
    weights = np.exp(np.clip(logits, -80.0, 0.0))
    weights /= max(float(weights.sum()), 1.0e-300)
    return similarity, weights


def _transition(
    belief: Any,
    action: str,
    *,
    node_ids: Sequence[str],
    node_index: Mapping[str, int],
    outgoing: Mapping[str, Sequence[Mapping[str, Any]]],
    compatibility: float,
    noise: float,
    action_consistency: bool = True,
) -> Any:
    import numpy as np

    count = len(node_ids)
    predicted = np.zeros(count, dtype=np.float64)
    for source_index, source in enumerate(node_ids):
        mass = float(belief[source_index])
        edges = list(outgoing.get(source, ()))
        if not edges:
            predicted[source_index] += mass
            continue
        if not action_consistency:
            for edge in edges:
                predicted[node_index[str(edge["target_node_id"])]] += mass / len(edges)
            continue
        matching = [
            edge for edge in edges if str(edge["executed_action_label"]) == action
        ]
        incompatible = [edge for edge in edges if edge not in matching]
        if not matching:
            matching_mass, incompatible_mass = 0.0, 1.0
        elif not incompatible:
            matching_mass, incompatible_mass = 1.0, 0.0
        else:
            matching_mass = float(compatibility)
            incompatible_mass = 1.0 - float(compatibility)
        for edge in matching:
            predicted[node_index[str(edge["target_node_id"])]] += (
                mass * matching_mass / len(matching)
            )
        for edge in incompatible:
            predicted[node_index[str(edge["target_node_id"])]] += (
                mass * incompatible_mass / len(incompatible)
            )
    predicted = (1.0 - float(noise)) * predicted + float(noise) / count
    return predicted / max(float(predicted.sum()), 1.0e-300)


def _truncate(belief: Any, mode: str, node_ids: Sequence[str]) -> Any:
    import numpy as np

    value = np.asarray(belief, dtype=np.float64).copy()
    if len(value) != len(node_ids):
        raise ExperimentError("belief/node identity length drift")
    order = sorted(
        range(len(value)), key=lambda index: (-value[index], str(node_ids[index]))
    )
    if mode == "MAP_FILTER":
        best = order[0]; value[:] = 0.0; value[best] = 1.0
    elif mode == "TOP_K_BELIEF" and len(value) > TOP_K:
        keep = order[:TOP_K]
        mask = np.ones(len(value), dtype=bool); mask[keep] = False; value[mask] = 0.0
        value /= max(float(value.sum()), 1.0e-300)
    return value


def _shuffled_actions(query_id: str, actions: Sequence[str]) -> tuple[list[str], list[int]]:
    if len(actions) < 2:
        raise ExperimentError("shuffled action history requires at least two positions")
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    mapping = list(
        METRICS.deterministic_action_history_position_mapping(
            str(query_id), len(actions)
        )
    )
    if sorted(mapping) != list(range(len(actions))) or any(index == source for index, source in enumerate(mapping)):
        raise ExperimentError("action-history mapping is not a complete position derangement")
    return [str(actions[index]) for index in mapping], mapping


def infer_query_belief(
    *,
    episode: Mapping[str, Any],
    query: Mapping[str, Any],
    descriptors_by_observation: Mapping[str, Any],
    parameters: Mapping[str, float],
    condition_id: str,
) -> dict[str, Any]:
    """Infer graph-order belief evidence without reading non-oracle labels.

    For every non-oracle condition this function reads only the registered
    observation/action history and the immutable graph transition structure.
    Ground-truth place identity is opened exclusively inside the explicit
    ORACLE_PLACE_IDENTITY branch.
    """

    import numpy as np

    node_ids, node_index, outgoing = _episode_indices(episode)
    if condition_id not in CONDITION_IDS:
        raise ExperimentError(f"unknown condition {condition_id!r}")
    node_by_id = {str(node["node_id"]): node for node in episode["nodes"]}
    reference_ids = [
        str(node_by_id[node_id]["keyframe_observation_id"])
        for node_id in node_ids
    ]
    prototypes = None
    if not hasattr(descriptors_by_observation, "observation_evidence"):
        prototypes = np.stack(
            [descriptors_by_observation[value] for value in reference_ids]
        ).astype(np.float64)
    history_ids = [str(value) for value in query["history_observation_ids"]]
    full_actions = [str(value) for value in query["history_executed_action_labels"]]
    if len(history_ids) != len(full_actions) + 1:
        raise ExperimentError("observation/action history length drift")
    actions = list(full_actions)
    observation_ids = list(history_ids)
    temperature = float(parameters["observation_softmax_temperature"])
    compatibility = float(parameters["action_compatible_edge_probability"])
    noise = float(parameters["transition_noise_probability"])
    action_mapping = list(range(len(full_actions)))
    if condition_id == "CURRENT_FRAME_NEAREST_NODE":
        observation_ids = observation_ids[-1:]
        actions = []
    elif condition_id == "FIXED_WINDOW_SEQUENCE":
        start = max(0, len(observation_ids) - FIXED_WINDOW)
        observation_ids = observation_ids[start:]
        required_actions = max(0, len(observation_ids) - 1)
        actions = actions[-required_actions:] if required_actions else []
    elif condition_id == "SHUFFLED_ACTION_HISTORY":
        actions, action_mapping = _shuffled_actions(str(query["query_id"]), actions)

    belief = np.full(len(node_ids), 1.0 / len(node_ids), dtype=np.float64)
    final_similarity = np.zeros(len(node_ids), dtype=np.float64)
    final_likelihood = np.full(len(node_ids), 1.0 / len(node_ids), dtype=np.float64)
    final_prior = belief.copy()
    final_preprojection = belief.copy()
    for step, observation_id in enumerate(observation_ids):
        if step:
            action = actions[step - 1]
            belief = _transition(
                belief,
                action,
                node_ids=node_ids,
                node_index=node_index,
                outgoing=outgoing,
                compatibility=compatibility,
                noise=noise,
                action_consistency=condition_id != "NO_ACTION_CONSISTENCY",
            )
        final_prior = belief.copy()
        if hasattr(descriptors_by_observation, "observation_evidence"):
            final_similarity, final_likelihood = (
                descriptors_by_observation.observation_evidence(
                    observation_id,
                    reference_ids,
                    temperature=temperature,
                    uniform=condition_id == "NO_OBSERVATION_LIKELIHOOD",
                )
            )
        else:
            assert prototypes is not None
            final_similarity, final_likelihood = _observation_likelihood(
                descriptors_by_observation[observation_id],
                prototypes,
                temperature=temperature,
                uniform=condition_id == "NO_OBSERVATION_LIKELIHOOD",
            )
        belief = belief * final_likelihood
        belief /= max(float(belief.sum()), 1.0e-300)
        final_preprojection = belief.copy()
        if condition_id in {"MAP_FILTER", "TOP_K_BELIEF"}:
            belief = _truncate(belief, condition_id, node_ids)
    if condition_id in {"CURRENT_FRAME_NEAREST_NODE", "FIXED_WINDOW_SEQUENCE"}:
        belief = _truncate(belief, "MAP_FILTER", node_ids)
    elif condition_id == "ORACLE_PLACE_IDENTITY":
        # This is the only label opening in belief inference.
        true_index = node_index[str(query["true_node_id"])]
        belief[:] = 0.0
        belief[true_index] = 1.0
    return {
        "node_ids": node_ids,
        "observation_similarities": final_similarity.tolist(),
        "observation_likelihoods": final_likelihood.tolist(),
        "transition_prior_probabilities": final_prior.tolist(),
        "preprojection_probabilities": final_preprojection.tolist(),
        "posterior_probabilities": belief.tolist(),
        "action_history_position_mapping": action_mapping,
    }


def _load_latent_context() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, Any],
]:
    import numpy as np

    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    episodes = {str(row["episode_id"]): row for row in graph["graphs"]}
    queries_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl"):
        queries_by_episode[str(query["episode_id"])].append(query)
    _validate_npz_member_names(
        OUTPUT_ROOT / "latents.npz",
        {"schema", "pixel_template_ids", "raw_tokens", "spatial_descriptors"},
    )
    with np.load(OUTPUT_ROOT / "latents.npz", allow_pickle=False) as data:
        ids = [str(value) for value in data["pixel_template_ids"].tolist()]
        descriptors = np.asarray(data["spatial_descriptors"], dtype=np.float32)
    if descriptors.shape != (len(ids), *TOKEN_SHAPE) or not np.isfinite(descriptors).all():
        raise ExperimentError("spatial descriptor contract drift")
    descriptor_by_template = dict(zip(ids, descriptors))
    latent_index = load_json(OUTPUT_ROOT / "latent_index.json")
    template_by_capture: dict[str, str] = {}
    for capture in latent_index["captures"]:
        capture_id = str(capture["capture_id"])
        template_id = str(capture["pixel_template_id"])
        if capture_id in template_by_capture or template_id not in descriptor_by_template:
            raise ExperimentError("latent capture binding drift")
        template_by_capture[capture_id] = template_id
    return graph, episodes, queries_by_episode, SpatialDescriptorStore(
        descriptors_by_template=descriptor_by_template,
        template_by_capture=template_by_capture,
    )


def _parameter_grid() -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for temperature in CALIBRATION_GRID["observation_softmax_temperature"]:
        for compatibility in CALIBRATION_GRID["action_compatible_edge_probability"]:
            for noise in CALIBRATION_GRID["transition_noise_probability"]:
                for entropy in CALIBRATION_GRID["normalized_entropy_abstention_threshold"]:
                    rows.append(
                        {
                            "observation_softmax_temperature": float(temperature),
                            "action_compatible_edge_probability": float(compatibility),
                            "transition_noise_probability": float(noise),
                            "normalized_entropy_abstention_threshold": float(entropy),
                        }
                    )
    return rows


_ROUTE_TABLE_CACHE: dict[str, dict[str, Any]] = {}


def _route_tables(episode: Mapping[str, Any]) -> dict[str, Any]:
    graph_id = str(episode["graph_id"])
    if graph_id in _ROUTE_TABLE_CACHE:
        return _ROUTE_TABLE_CACHE[graph_id]
    node_ids = [str(node["node_id"]) for node in episode["nodes"]]
    reverse: dict[str, list[tuple[str, float]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in episode["edges"]:
        edge = dict(raw)
        outgoing[str(edge["source_node_id"])].append(edge)
        reverse[str(edge["target_node_id"])].append(
            (str(edge["source_node_id"]), float(edge["edge_cost"]))
        )
    goal = str(episode["goal_node_id"])
    distances: dict[str, float] = {goal: 0.0}
    queue: list[tuple[float, str]] = [(0.0, goal)]
    while queue:
        cost, target = heapq.heappop(queue)
        if cost != distances.get(target):
            continue
        for source, edge_cost in reverse.get(target, ()):
            candidate = cost + edge_cost
            if candidate < distances.get(source, math.inf):
                distances[source] = candidate
                heapq.heappush(queue, (candidate, source))
    if set(distances) != set(node_ids):
        raise ExperimentError("route table contains a node that cannot reach goal")
    best_edge: dict[str, dict[str, Any] | None] = {}
    port_distance: dict[str, dict[str, float]] = {}
    for node_id in node_ids:
        options = sorted(
            outgoing.get(node_id, ()),
            key=lambda edge: (
                float(edge["edge_cost"]) + distances[str(edge["target_node_id"])],
                str(edge["edge_id"]),
            ),
        )
        best_edge[node_id] = options[0] if options else None
        port_distance[node_id] = {
            str(edge["port_label"]): float(edge["edge_cost"])
            + distances[str(edge["target_node_id"])]
            for edge in options
        }
    result = {
        "node_ids": node_ids,
        "distances": distances,
        "best_edge": best_edge,
        "best_port": {
            node_id: (
                None
                if best_edge[node_id] is None
                else str(best_edge[node_id]["port_label"])
            )
            for node_id in node_ids
        },
        "port_distance": port_distance,
    }
    _ROUTE_TABLE_CACHE[graph_id] = result
    return result


def _route_choice(
    node_ids: Sequence[str], probabilities: Sequence[float], tables: Mapping[str, Any]
) -> tuple[str | None, str | None]:
    route_mass = {port: 0.0 for port in PORT_LABELS}
    for node_id, probability in zip(node_ids, probabilities):
        edge = tables["best_edge"][node_id]
        if edge is not None:
            route_mass[str(edge["port_label"])] += float(probability)
    if max(route_mass.values(), default=0.0) <= 0.0:
        return None, None
    selected_port = min(
        PORT_LABELS,
        key=lambda port: (-route_mass[port], PORT_LABELS.index(port)),
    )
    witnesses = [
        (float(probability), str(node_id))
        for node_id, probability in zip(node_ids, probabilities)
        if tables["best_edge"][node_id] is not None
        and tables["best_edge"][node_id]["port_label"] == selected_port
    ]
    witness = min(witnesses, key=lambda item: (-item[0], item[1]))[1]
    return selected_port, str(tables["best_edge"][witness]["edge_id"])


def materialize_belief_row(
    *,
    episode: Mapping[str, Any],
    query: Mapping[str, Any],
    condition_id: str,
    evidence: Mapping[str, Any],
    entropy_threshold: float,
) -> dict[str, Any]:
    import numpy as np

    node_ids = [str(value) for value in evidence["node_ids"]]
    value = np.asarray(evidence["posterior_probabilities"], dtype=np.float64)
    order = sorted(range(len(node_ids)), key=lambda index: (-value[index], node_ids[index]))
    entropy = -float(np.sum(np.where(value > 0.0, value * np.log(np.maximum(value, 1e-300)), 0.0)))
    normalized_entropy = entropy / math.log(max(len(value), 2))
    abstained = normalized_entropy > float(entropy_threshold)
    selected_port, selected_edge = _route_choice(node_ids, value, _route_tables(episode))
    return {
        "query_id": str(query["query_id"]),
        "condition_id": condition_id,
        **{key: evidence[key] for key in (
            "node_ids", "observation_similarities", "observation_likelihoods",
            "transition_prior_probabilities", "preprojection_probabilities",
            "posterior_probabilities", "action_history_position_mapping",
        )},
        "selected_node_id": node_ids[order[0]],
        "selected_edge_id": selected_edge,
        "selected_port_label": selected_port,
        "normalized_entropy": normalized_entropy,
        "abstained": abstained,
    }


def calibration_query_outcome(
    *, episode: Mapping[str, Any], query: Mapping[str, Any], belief: Mapping[str, Any]
) -> dict[str, Any]:
    probabilities = [float(value) for value in belief["posterior_probabilities"]]
    node_ids = [str(value) for value in belief["node_ids"]]
    order = sorted(range(len(node_ids)), key=lambda index: (-probabilities[index], node_ids[index]))
    true_node = str(query["true_node_id"])
    true_index = node_ids.index(true_node)
    tables = _route_tables(episode)
    selected_port = belief["selected_port_label"]
    available = tables["port_distance"][true_node]
    if belief["abstained"] or selected_port not in available:
        regret = 1.0
    else:
        best = min(available.values())
        worst = max(available.values())
        regret = 0.0 if worst == best else (available[selected_port] - best) / (worst - best)
        regret = min(1.0, max(0.0, regret))
    return {
        "edge_correct": bool(
            not belief["abstained"]
            and selected_port == str(query["true_next_port_label"])
        ),
        "localisation_top3": bool(true_index in order[: min(3, len(order))]),
        "normalized_regret": regret,
        "false_confident": bool(
            node_ids[order[0]] != true_node and not belief["abstained"]
        ),
        "abstained": bool(belief["abstained"]),
    }


def calibrate_stage() -> dict[str, Any]:
    require_source_freeze()
    path = OUTPUT_ROOT / "calibration.json"
    if path.exists():
        raise ExperimentError("calibration output already exists")
    graph, episodes, queries_by_episode, descriptors = _load_latent_context()
    all_queries = [row for values in queries_by_episode.values() for row in values]
    calibration_queries = sorted(
        (row for row in all_queries if row["role"] == "CALIBRATION"),
        key=lambda row: str(row["query_id"]),
    )
    if len(calibration_queries) != 128:
        raise ExperimentError("calibration query population drift")
    grid_results: list[dict[str, Any]] = []
    grid_query_results: list[dict[str, Any]] = []
    for grid_index, parameters in enumerate(_parameter_grid()):
        outcomes: list[dict[str, Any]] = []
        for query in calibration_queries:
            episode = episodes[str(query["episode_id"])]
            evidence = infer_query_belief(
                episode=episode,
                query=query,
                descriptors_by_observation=descriptors,
                parameters=parameters,
                condition_id="FULL_BELIEF",
            )
            belief = materialize_belief_row(
                episode=episode,
                query=query,
                condition_id="FULL_BELIEF",
                evidence=evidence,
                entropy_threshold=parameters[
                    "normalized_entropy_abstention_threshold"
                ],
            )
            outcome = calibration_query_outcome(
                episode=episode, query=query, belief=belief
            )
            outcomes.append(outcome)
            grid_query_results.append(
                {
                    "grid_index": grid_index,
                    "query_id": query["query_id"],
                    **outcome,
                }
            )
        grid_results.append(
            {
                "grid_index": grid_index,
                "parameters": parameters,
                "correct_next_edge_accuracy": sum(
                    bool(row["edge_correct"]) for row in outcomes
                ) / len(outcomes),
                "localisation_top3": sum(
                    bool(row["localisation_top3"]) for row in outcomes
                ) / len(outcomes),
                "normalized_graph_distance_regret": sum(
                    float(row["normalized_regret"]) for row in outcomes
                ) / len(outcomes),
                "false_confident_localisation_rate": sum(
                    bool(row["false_confident"]) for row in outcomes
                ) / len(outcomes),
                "abstention_rate": sum(
                    bool(row["abstained"]) for row in outcomes
                ) / len(outcomes),
            }
        )
    selected = min(
        grid_results,
        key=lambda row: (
            -float(row["correct_next_edge_accuracy"]),
            -float(row["localisation_top3"]),
            float(row["normalized_graph_distance_regret"]),
            float(row["false_confident_localisation_rate"]),
            float(row["abstention_rate"]),
            int(row["grid_index"]),
        ),
    )
    receipt = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.calibration.v1",
            "experiment_id": "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "selection_role": "CALIBRATION",
            "calibration_episode_ids": sorted(
                episode_id for episode_id, episode in episodes.items()
                if episode["role"] == "CALIBRATION"
            ),
            "selected_parameters": selected["parameters"],
            "selected_grid_index": selected["grid_index"],
            "grid_results": grid_results,
            "grid_query_results": grid_query_results,
        }
    )
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    METRICS.validate_calibration(receipt, graph, all_queries)
    atomic_json(path, receipt)
    return receipt


def stage_a() -> dict[str, Any]:
    require_source_freeze()
    for leaf in ("stage_a_beliefs.jsonl", "stage_a_metrics.json"):
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError(f"Stage-A output already exists: {leaf}")
    graph, episodes, queries_by_episode, descriptors = _load_latent_context()
    calibration = load_json(OUTPUT_ROOT / "calibration.json")
    parameters = dict(calibration["selected_parameters"])
    rows: list[dict[str, Any]] = []
    for episode_id, episode in episodes.items():
        if episode["role"] != "DEVELOPMENT_HELDOUT":
            continue
        for query in sorted(queries_by_episode[episode_id], key=lambda row: row["query_id"]):
            for condition_id in CONDITION_IDS:
                evidence = infer_query_belief(
                    episode=episode,
                    query=query,
                    descriptors_by_observation=descriptors,
                    parameters=parameters,
                    condition_id=condition_id,
                )
                rows.append(
                    materialize_belief_row(
                        episode=episode,
                        query=query,
                        condition_id=condition_id,
                        evidence=evidence,
                        entropy_threshold=parameters[
                            "normalized_entropy_abstention_threshold"
                        ],
                    )
                )
    expected = 16 * 8 * len(CONDITION_IDS)
    if len(rows) != expected:
        raise ExperimentError(f"Stage-A belief row count drift: {len(rows)} != {expected}")
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    metrics = METRICS.recompute_stage_a_metrics(
        graph,
        [row for values in queries_by_episode.values() for row in values],
        rows,
        calibration,
    )
    write_jsonl(OUTPUT_ROOT / "stage_a_beliefs.jsonl", rows)
    atomic_json(OUTPUT_ROOT / "stage_a_metrics.json", metrics)
    return metrics


def _stage_a_authorizes_stage_b(metrics: Mapping[str, Any]) -> bool:
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    try:
        return bool(METRICS.stage_a_authorizes_stage_b(metrics))
    except (ValueError, TypeError, KeyError) as exc:
        raise ExperimentError("Stage-A metrics lack valid Stage-B authority") from exc


def _prior_current_visual_binding() -> dict[str, Any]:
    bindings = getattr(CONTRACT, "FROZEN_SOURCE_BINDINGS", None)
    value = (
        bindings.get("stage_b_current_visual_ranker")
        if isinstance(bindings, Mapping)
        else None
    )
    if not isinstance(value, Mapping):
        raise ExperimentError("contract lacks prior CURRENT_VISUAL checkpoint binding")
    return dict(value)


def _slew_stage_b_commands(
    primitive_blocks: Sequence[str],
    old_contract: Any,
    *,
    initial_previous_command: Sequence[float] | None = None,
) -> tuple[list[list[float]], list[list[float]]]:
    import numpy as np

    requested: list[list[float]] = []
    applied: list[list[float]] = []
    previous = np.asarray(
        [0.0, 0.0, 0.0]
        if initial_previous_command is None
        else list(initial_previous_command),
        dtype=np.float64,
    )
    if previous.shape != (3,) or not np.isfinite(previous).all():
        raise ExperimentError("Stage-B previous command shape/value drift")
    slew = old_contract.PANEL_GEOMETRY_AUTHORITY["slew_limits_per_command_tick"]
    delta = np.asarray(
        [
            slew["delta_vx_max_mps"],
            slew["delta_vy_max_mps"],
            slew["delta_yaw_rate_max_radps"],
        ],
        dtype=np.float64,
    )
    lower = np.asarray([-0.3, 0.0, -0.5], dtype=np.float64)
    upper = np.asarray([0.3, 0.0, 0.5], dtype=np.float64)
    for primitive in primitive_blocks[:3]:
        command = np.asarray(old_contract.PRIMITIVES[str(primitive)], dtype=np.float64)
        for _ in range(int(old_contract.TICKS_PER_BLOCK)):
            requested.append(command.tolist())
            bounded = np.clip(command, lower, upper)
            bounded = np.clip(bounded, previous - delta, previous + delta)
            applied.append(bounded.tolist())
            previous = bounded
    if len(requested) != 15 or len(applied) != 15:
        raise ExperimentError("Stage-B candidate horizon is not H3/15 ticks")
    return requested, applied


def build_stage_b_ranker_inputs(
    relative_waypoint: Sequence[float],
    old_contract: Any,
    *,
    previous_command: Sequence[float] | None = None,
    control_history: Sequence[Sequence[float]] | None = None,
) -> tuple[list[str], Any, Any, Any]:
    """Build the frozen old-ranker features for one oracle-graph waypoint."""

    import numpy as np

    if len(relative_waypoint) != 3:
        raise ExperimentError("Stage-B relative waypoint must contain x/y/yaw")
    goal_x, goal_y, goal_yaw = (float(value) for value in relative_waypoint)
    goal = np.asarray(
        [goal_x, goal_y, math.sin(goal_yaw), math.cos(goal_yaw)],
        dtype=np.float32,
    )
    previous = np.asarray(
        [0.0, 0.0, 0.0] if previous_command is None else list(previous_command),
        dtype=np.float32,
    )
    history = np.asarray(
        np.zeros((15, 2), dtype=np.float32)
        if control_history is None
        else control_history,
        dtype=np.float32,
    )
    if previous.shape != (3,) or history.shape != (15, 2):
        raise ExperimentError("Stage-B carried command/history shape drift")
    if not np.isfinite(previous).all() or not np.isfinite(history).all():
        raise ExperimentError("Stage-B carried command/history contains non-finite values")
    tick_seconds = float(old_contract.PANEL_GEOMETRY_AUTHORITY["command_tick_seconds"])
    names: list[str] = []
    base_rows: list[Any] = []
    query_rows: list[Any] = []
    anchors: list[float] = []
    for name, primitive_blocks in old_contract.CANDIDATE_BANK:
        requested, applied = _slew_stage_b_commands(
            primitive_blocks,
            old_contract,
            initial_previous_command=previous,
        )
        requested_array = np.asarray(requested, dtype=np.float32)
        applied_array = np.asarray(applied, dtype=np.float32)
        x = y = yaw = 0.0
        for velocity_x, velocity_y, yaw_rate in applied_array:
            x += (
                math.cos(yaw) * float(velocity_x)
                - math.sin(yaw) * float(velocity_y)
            ) * tick_seconds
            y += (
                math.sin(yaw) * float(velocity_x)
                + math.cos(yaw) * float(velocity_y)
            ) * tick_seconds
            yaw += float(yaw_rate) * tick_seconds
        direct_progress = math.hypot(goal_x, goal_y) - math.hypot(
            goal_x - x, goal_y - y
        )
        base = np.concatenate(
            [
                goal,
                requested_array.reshape(-1),
                applied_array.reshape(-1),
                previous,
                history.reshape(-1),
                np.asarray([direct_progress], dtype=np.float32),
                np.asarray([x, y, yaw], dtype=np.float32),
            ]
        )
        query_features = np.concatenate(
            [
                goal,
                applied_array[:, (0, 2)].reshape(-1),
                previous[[0, 2]],
                history.reshape(-1),
            ]
        )
        if base.shape != (old_contract.BASE_FEATURE_DIM,) or query_features.shape != (
            old_contract.QUERY_FEATURE_DIM,
        ):
            raise ExperimentError("Stage-B old-ranker feature width drift")
        names.append(str(name))
        base_rows.append(base)
        query_rows.append(query_features)
        anchors.append(direct_progress)
    return (
        names,
        np.stack(base_rows).astype(np.float32, copy=False),
        np.stack(query_rows).astype(np.float32, copy=False),
        np.asarray(anchors, dtype=np.float32),
    )


class StageBObservationRuntime:
    """Fresh-occurrence renderer with exact-byte frozen-encoder cache reuse."""

    def __init__(self) -> None:
        import numpy as np

        observation_path = OUTPUT_ROOT / "observations.npz"
        latent_path = OUTPUT_ROOT / "latents.npz"
        _validate_npz_member_names(
            observation_path,
            {"schema", "pixel_template_ids", "image_sha256", "images"},
        )
        _validate_npz_member_names(
            latent_path,
            {"schema", "pixel_template_ids", "raw_tokens", "spatial_descriptors"},
        )
        with np.load(observation_path, allow_pickle=False) as data:
            observation_ids = [str(value) for value in data["pixel_template_ids"].tolist()]
            image_hashes = [str(value) for value in data["image_sha256"].tolist()]
            images = np.asarray(data["images"], dtype=np.uint8)
        with np.load(latent_path, allow_pickle=False) as data:
            latent_ids = [str(value) for value in data["pixel_template_ids"].tolist()]
            raw_tokens = np.asarray(data["raw_tokens"], dtype=np.float16)
            descriptors = np.asarray(data["spatial_descriptors"], dtype=np.float32)
        if observation_ids != latent_ids or images.shape != (
            len(observation_ids), IMAGE_HEIGHT, IMAGE_WIDTH, 3
        ):
            raise ExperimentError("Stage-B observation/latent template order drift")
        if raw_tokens.shape != (len(latent_ids), *TOKEN_SHAPE) or descriptors.shape != (
            len(latent_ids), *TOKEN_SHAPE
        ):
            raise ExperimentError("Stage-B frozen token-cache shape drift")
        latent_index = load_json(OUTPUT_ROOT / "latent_index.json")
        if (
            latent_index["encoder_checkpoint_sha256"]
            != CONTRACT.OBSERVATION_LIKELIHOOD["checkpoint_sha256"]
            or latent_index["latents_file"]["sha256"] != sha256_file(latent_path)
        ):
            raise ExperimentError("Stage-B frozen encoder/cache binding drift")
        self.encoder_cache_binding = (
            str(latent_index["encoder_checkpoint_sha256"]),
            str(latent_index["preprocessing_digest"]),
        )
        self._by_template: dict[str, dict[str, Any]] = {}
        by_hash: dict[str, tuple[Any, Any]] = {}
        for index, template_id in enumerate(observation_ids):
            image = np.ascontiguousarray(images[index])
            pixel_sha = hashlib.sha256(image.tobytes(order="C")).hexdigest()
            if pixel_sha != image_hashes[index]:
                raise ExperimentError("Stage-B cached image hash drift")
            raw = np.ascontiguousarray(raw_tokens[index])
            descriptor = np.ascontiguousarray(descriptors[index])
            existing = by_hash.get(pixel_sha)
            if existing is not None and (
                not np.array_equal(existing[0], raw)
                or not np.array_equal(existing[1], descriptor)
            ):
                raise ExperimentError(
                    "exact-byte observations have inconsistent frozen token-cache rows"
                )
            by_hash[pixel_sha] = (raw, descriptor)
            self._by_template[template_id] = {
                "pixel_sha256": pixel_sha,
                "image": image,
                "raw_tokens": raw,
                "spatial_descriptor": descriptor,
            }

    def observe(
        self,
        *,
        episode: Mapping[str, Any],
        node_id: str,
        occurrence_id: str,
    ) -> dict[str, Any]:
        node = next(
            (row for row in episode["nodes"] if str(row["node_id"]) == str(node_id)),
            None,
        )
        if node is None:
            raise ExperimentError("Stage-B fresh observation escaped the graph")
        recipe = json.loads(str(node["observation_descriptor"]))
        image = render_observation(recipe)
        pixel_sha = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        template_id = str(node["pixel_template_id"])
        cached = self._by_template.get(template_id)
        if (
            cached is None
            or pixel_sha != str(node["pixel_sha256"])
            or pixel_sha != cached["pixel_sha256"]
            or not (image == cached["image"]).all()
        ):
            raise ExperimentError("Stage-B fresh observation/cache identity drift")
        return {
            "observation_id": str(occurrence_id),
            "node_id": str(node_id),
            "pixel_sha256": pixel_sha,
            "raw_tokens": cached["raw_tokens"],
            "spatial_descriptor": cached["spatial_descriptor"],
            "encoder_cache_binding": self.encoder_cache_binding,
        }

    def observation_evidence(
        self,
        *,
        episode: Mapping[str, Any],
        observation: Mapping[str, Any],
        temperature: float,
    ) -> tuple[Any, Any]:
        import numpy as np

        references = []
        for node in episode["nodes"]:
            template_id = str(node["pixel_template_id"])
            if template_id not in self._by_template:
                raise ExperimentError("Stage-B node reference is absent from frozen cache")
            references.append(self._by_template[template_id]["spatial_descriptor"])
        similarity, _likelihood = _observation_likelihood(
            observation["spatial_descriptor"],
            np.stack(references).astype(np.float32, copy=False),
            temperature=float(temperature),
        )
        similarity = np.clip(np.asarray(similarity, dtype=np.float64), -1.0, 1.0)
        logits = (similarity - float(similarity.max())) / float(temperature)
        weights = np.exp(np.clip(logits, -80.0, 0.0))
        likelihood = weights / max(float(weights.sum()), 1.0e-300)
        return similarity, likelihood


def _stage_b_observation_update(
    *,
    episode: Mapping[str, Any],
    observation: Mapping[str, Any],
    prior_probabilities: Any,
    visual_runtime: Any,
    temperature: float,
    projection: str,
    oracle_node_id: str | None = None,
) -> dict[str, Any]:
    import numpy as np

    node_ids = [str(node["node_id"]) for node in episode["nodes"]]
    prior = np.asarray(prior_probabilities, dtype=np.float64)
    if prior.shape != (len(node_ids),) or not np.isfinite(prior).all():
        raise ExperimentError("Stage-B filter prior shape/value drift")
    prior = prior / max(float(prior.sum()), 1.0e-300)
    similarities, likelihoods = visual_runtime.observation_evidence(
        episode=episode,
        observation=observation,
        temperature=float(temperature),
    )
    similarities = np.asarray(similarities, dtype=np.float64)
    likelihoods = np.asarray(likelihoods, dtype=np.float64)
    if similarities.shape != prior.shape or likelihoods.shape != prior.shape:
        raise ExperimentError("Stage-B observation evidence width drift")
    preprojection = prior * likelihoods
    preprojection /= max(float(preprojection.sum()), 1.0e-300)
    if oracle_node_id is not None:
        posterior = np.zeros_like(preprojection)
        posterior[node_ids.index(str(oracle_node_id))] = 1.0
    elif projection == "MAP_FILTER":
        best = min(
            range(len(node_ids)),
            key=lambda index: (-float(preprojection[index]), node_ids[index]),
        )
        posterior = np.zeros_like(preprojection)
        posterior[best] = 1.0
    elif projection == "FULL_BELIEF":
        posterior = preprojection
    else:
        raise ExperimentError(f"unregistered Stage-B filter projection {projection!r}")
    return {
        "node_ids": node_ids,
        "observation_similarities": similarities.tolist(),
        "observation_likelihoods": likelihoods.tolist(),
        "transition_prior_probabilities": prior.tolist(),
        "preprojection_probabilities": preprojection.tolist(),
        "posterior_probabilities": posterior.tolist(),
    }


def _stage_b_roll_control_history(
    previous_command: Sequence[float],
    control_history: Sequence[Sequence[float]],
    applied_commands: Sequence[Sequence[float]],
) -> tuple[list[float], list[list[float]]]:
    import numpy as np

    previous = np.asarray(previous_command, dtype=np.float64)
    history = np.asarray(control_history, dtype=np.float64)
    applied = np.asarray(applied_commands, dtype=np.float64)
    if previous.shape != (3,) or history.shape != (15, 2) or applied.ndim != 2 or applied.shape[1:] != (3,):
        raise ExperimentError("Stage-B control-history update shape drift")
    active = applied[:, (0, 2)]
    combined = np.concatenate((history, active), axis=0)[-15:]
    return applied[-1].tolist(), combined.tolist()


def _stage_b_edge_applied_commands(
    action_label: str, previous_command: Sequence[float], old_contract: Any
) -> list[list[float]]:
    primitive = {
        "LEFT": "yaw_left",
        "STRAIGHT": "forward_medium",
        "RIGHT": "yaw_right",
        "REVERSE": "backward",
    }.get(str(action_label))
    if primitive is None:
        raise ExperimentError(f"unregistered Stage-B graph action {action_label!r}")
    _requested, applied = _slew_stage_b_commands(
        (primitive, primitive, primitive),
        old_contract,
        initial_previous_command=previous_command,
    )
    return applied[: int(old_contract.TICKS_PER_BLOCK)]


class FrozenCurrentVisualLocalRanker:
    """The one frozen old current-frame ranker authorized for conditional B."""

    def __init__(self) -> None:
        import numpy as np
        import torch
        from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as OLD

        binding = _prior_current_visual_binding()
        source = REPO_ROOT / str(binding["model_source_path"])
        checkpoint = Path(str(binding["checkpoint_path"]))
        _require_regular(source)
        _require_regular(checkpoint)
        if sha256_file(source) != str(binding["model_source_sha256"]):
            raise ExperimentError("prior current-visual model source binding mismatch")
        if (
            checkpoint.stat().st_size != int(binding["checkpoint_size_bytes"])
            or sha256_file(checkpoint) != str(binding["checkpoint_sha256"])
        ):
            raise ExperimentError("prior CURRENT_VISUAL checkpoint binding mismatch")
        model = OLD.MatchedNonGreedyRanker(OLD.CURRENT_VISUAL_REACTIVE_RANKER)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping):
            raise ExperimentError("prior CURRENT_VISUAL checkpoint payload drift")
        if (
            payload.get("model_id") != OLD.CURRENT_VISUAL_REACTIVE_RANKER
            or int(payload.get("epoch", -1)) != 60
            or int(payload.get("parameter_count", -1)) != int(binding["parameter_count"])
        ):
            raise ExperimentError("prior CURRENT_VISUAL checkpoint identity drift")
        model.load_state_dict(payload["model_state_dict"], strict=True)
        if sum(parameter.numel() for parameter in model.parameters()) != int(
            binding["parameter_count"]
        ):
            raise ExperimentError("prior CURRENT_VISUAL parameter count drift")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.old_contract = OLD
        self.binding = binding

    def __call__(
        self,
        current_tokens: Any,
        relative_waypoint: Sequence[float],
        *,
        previous_command: Sequence[float],
        control_history: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        import numpy as np
        import torch

        names, base, query, anchor = build_stage_b_ranker_inputs(
            relative_waypoint,
            self.old_contract,
            previous_command=previous_command,
            control_history=control_history,
        )
        current = torch.from_numpy(
            np.asarray(current_tokens, dtype=np.float32)
        ).to(self.device)
        if tuple(current.shape) != TOKEN_SHAPE:
            raise ExperimentError("Stage-B fresh current-token shape drift")
        with torch.inference_mode():
            output = self.model(
                base_features=torch.from_numpy(base).to(self.device),
                query_features=torch.from_numpy(query).to(self.device),
                kinematic_anchor=torch.from_numpy(anchor).to(self.device),
                current_tokens=current,
                future_tokens=None,
            )
        scores = [float(value) for value in output.score.detach().cpu().tolist()]
        if len(scores) != len(names) or not all(math.isfinite(value) for value in scores):
            raise ExperimentError("Stage-B ranker score vector drift")
        return {
            "candidate_names": names,
            "candidate_scores": scores,
        }


def build_stage_b_trace_rows(
    *,
    graph_manifest: Mapping[str, Any],
    query_rows: Sequence[Mapping[str, Any]],
    stage_a_metrics: Mapping[str, Any],
    local_ranker: Callable[..., Mapping[str, Any]],
    visual_runtime: Any,
) -> list[dict[str, Any]]:
    """Execute fresh macro-choice rollouts for the three frozen B conditions."""

    import numpy as np
    from lewm.safety import non_greedy_local_subgoal_jepa_planning_v1_contract as OLD
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS

    strongest = stage_a_metrics["decision"].get("strongest_memory_condition")
    if strongest not in {"MAP_FILTER", "FULL_BELIEF"}:
        raise ExperimentError("authorized Stage B lacks a registered strongest memory")
    parameters = dict(stage_a_metrics["calibration_binding"]["selected_parameters"])
    temperature = float(parameters["observation_softmax_temperature"])
    compatibility = float(parameters["action_compatible_edge_probability"])
    noise = float(parameters["transition_noise_probability"])
    entropy_threshold = float(
        parameters["normalized_entropy_abstention_threshold"]
    )
    source_condition = {
        "CURRENT_FRAME_NEAREST_NODE": "CURRENT_FRAME_NEAREST_NODE",
        "STRONGEST_STAGE_A_MEMORY": str(strongest),
        "ORACLE_PLACE_IDENTITY": "ORACLE_PLACE_IDENTITY",
    }
    episodes = {
        str(episode["episode_id"]): episode
        for episode in graph_manifest["graphs"]
        if episode["role"] == "DEVELOPMENT_HELDOUT"
    }
    queries_by_episode: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for query in query_rows:
        if query["role"] == "DEVELOPMENT_HELDOUT":
            queries_by_episode[str(query["episode_id"])].append(query)
    if len(episodes) != 16:
        raise ExperimentError("Stage B requires the 16 development-heldout episodes")
    traces: list[dict[str, Any]] = []
    for condition_id in CONTRACT.STAGE_B_CONDITION_IDS:
        for episode_id in sorted(episodes):
            episode = episodes[episode_id]
            edges = {str(edge["edge_id"]): edge for edge in episode["edges"]}
            nodes = {str(node["node_id"]): node for node in episode["nodes"]}
            node_ids, node_index, outgoing = _episode_indices(episode)
            uniform = np.full(len(node_ids), 1.0 / len(node_ids), dtype=np.float64)
            episode_queries = sorted(
                queries_by_episode[episode_id], key=lambda row: int(row["query_index"])
            )
            if [int(row["query_index"]) for row in episode_queries] != list(range(8)):
                raise ExperimentError("Stage-B episode query sequence drift")
            query_by_node = {
                str(query["true_node_id"]): query for query in episode_queries
            }
            execution_id = f"ogtb-v1-stage-b:{condition_id}:{episode_id}"
            occurrence_index = 0
            carried = uniform.copy()
            previous_command = [0.0, 0.0, 0.0]
            control_history = [[0.0, 0.0] for _ in range(15)]

            def fresh_update(node_id: str, incoming_action: str | None) -> dict[str, Any]:
                nonlocal carried, occurrence_index
                occurrence_id = (
                    f"{execution_id}:fresh-{occurrence_index:04d}:"
                    f"{str(node_id).rsplit(':', 1)[-1]}"
                )
                occurrence_index += 1
                observation = visual_runtime.observe(
                    episode=episode,
                    node_id=str(node_id),
                    occurrence_id=occurrence_id,
                )
                if condition_id == "CURRENT_FRAME_NEAREST_NODE":
                    prior = uniform.copy()
                    projection = "MAP_FILTER"
                    oracle_node = None
                elif condition_id == "ORACLE_PLACE_IDENTITY":
                    prior = np.zeros(len(node_ids), dtype=np.float64)
                    prior[node_index[str(node_id)]] = 1.0
                    projection = "FULL_BELIEF"
                    oracle_node = str(node_id)
                else:
                    if incoming_action is None or incoming_action == "NO_OP":
                        prior = np.asarray(carried, dtype=np.float64)
                    else:
                        prior = _transition(
                            carried,
                            str(incoming_action),
                            node_ids=node_ids,
                            node_index=node_index,
                            outgoing=outgoing,
                            compatibility=compatibility,
                            noise=noise,
                        )
                    projection = str(strongest)
                    oracle_node = None
                evidence = _stage_b_observation_update(
                    episode=episode,
                    observation=observation,
                    prior_probabilities=prior,
                    visual_runtime=visual_runtime,
                    temperature=temperature,
                    projection=projection,
                    oracle_node_id=oracle_node,
                )
                carried = np.asarray(
                    evidence["posterior_probabilities"], dtype=np.float64
                )
                return {
                    "observation": observation,
                    "incoming_action": incoming_action,
                    "evidence": evidence,
                }

            actual_node = str(episode["stage_b_start_node_id"])
            pending_updates = [fresh_update(actual_node, None)]
            prelude_nodes = [actual_node]
            for edge_id in episode["stage_b_prelude_edge_ids"]:
                edge = edges[str(edge_id)]
                if str(edge["source_node_id"]) != actual_node:
                    raise ExperimentError("Stage-B prelude is not contiguous")
                applied = _stage_b_edge_applied_commands(
                    str(edge["executed_action_label"]), previous_command, OLD
                )
                previous_command, control_history = _stage_b_roll_control_history(
                    previous_command, control_history, applied
                )
                actual_node = str(edge["target_node_id"])
                prelude_nodes.append(actual_node)
                pending_updates.append(
                    fresh_update(actual_node, str(edge["executed_action_label"]))
                )
            if actual_node != str(episode["stage_b_decision_start_node_id"]):
                raise ExperimentError("Stage-B prelude did not terminate at query zero")
            visited_nodes: list[str] = []
            for node_id in prelude_nodes[:-1]:
                if node_id not in visited_nodes:
                    visited_nodes.append(node_id)
            oracle_edge_ids = list(episode["stage_b_prelude_edge_ids"])
            prelude_distance = sum(
                float(edges[str(edge_id)]["edge_cost"])
                for edge_id in episode["stage_b_prelude_edge_ids"]
            )
            for query in episode_queries:
                oracle_edge_ids.extend(
                    query["stage_b_choice_macros"][str(query["true_next_edge_id"])][
                        "constituent_edge_ids"
                    ]
                )
            oracle_path_distance = sum(
                float(edges[str(edge_id)]["edge_cost"])
                for edge_id in oracle_edge_ids
            )
            nonmovement_count = 0
            recovery_pending = False

            for decision_index in range(int(CONTRACT.STAGE_B_EXECUTION_POLICY["decision_budget"])):
                query = query_by_node.get(actual_node)
                if query is None:
                    raise ExperimentError("Stage-B decision state is not a registered query")
                decision_started = time.perf_counter()
                final_update = pending_updates[-1]
                evidence = final_update["evidence"]
                belief = np.asarray(evidence["posterior_probabilities"], dtype=np.float64)
                selected_node = min(
                    range(len(node_ids)),
                    key=lambda index: (-float(belief[index]), node_ids[index]),
                )
                selected_node_id = node_ids[selected_node]
                entropy = -float(
                    np.sum(
                        np.where(
                            belief > 0.0,
                            belief * np.log(np.maximum(belief, 1.0e-300)),
                            0.0,
                        )
                    )
                ) / math.log(max(len(belief), 2))
                proposed_port, witness_edge_id = _route_choice(
                    node_ids, belief, _route_tables(episode)
                )
                abstained = entropy > entropy_threshold
                selected_port = None if abstained else proposed_port
                waypoint = (
                    None
                    if witness_edge_id is None
                    else list(edges[str(witness_edge_id)]["relative_waypoint"])
                )
                candidate_outcomes = list(query["stage_b_local_candidate_outcomes"])
                mask = [
                    int(outcome["candidate_index"])
                    for outcome in candidate_outcomes
                    if bool(outcome["oracle_admissible"])
                ]
                if abstained:
                    disposition = "ENTROPY_ABSTENTION"
                elif witness_edge_id is None:
                    disposition = "NO_ROUTE_WITNESS"
                elif not mask:
                    disposition = "NO_ORACLE_ADMISSIBLE_LOCAL_CANDIDATE"
                else:
                    disposition = "EXECUTED_CANDIDATE"
                acting = disposition == "EXECUTED_CANDIDATE"
                ranker_called = False
                candidate_scores: list[float] = []
                candidate_index: int | None = None
                candidate_name: str | None = None
                candidate_score: float | None = None
                executed_port: str | None = None
                executed_choice_edge: str | None = None
                local_execution_success = False
                oracle_execution = False
                ranker_latency_ms = 0.0
                constituent_edges: list[str] = []
                constituent_nodes = [actual_node]
                constituent_actions: list[str] = []
                constituent_costs: list[float] = []
                constituent_observations: list[str] = []
                constituent_shas: list[str] = []
                next_updates: list[dict[str, Any]] = []
                candidate_contact = False
                ranker_previous_command = list(previous_command)
                ranker_control_history = [list(value) for value in control_history]
                selected_candidate_applied_commands: list[list[float]] = []
                if acting:
                    rank_started = time.perf_counter()
                    local = dict(
                        local_ranker(
                            final_update["observation"]["raw_tokens"],
                            waypoint,
                            previous_command=previous_command,
                            control_history=control_history,
                        )
                    )
                    observed_ranker_ms = (time.perf_counter() - rank_started) * 1000.0
                    ranker_latency_ms = float(
                        local.get("ranker_latency_ms", observed_ranker_ms)
                    )
                    if list(local.get("candidate_names", ())) != list(
                        CONTRACT.STAGE_B_LOCAL_CANDIDATE_IDS
                    ):
                        raise ExperimentError("Stage-B ranker candidate identity order drift")
                    candidate_scores = [
                        float(value) for value in local.get("candidate_scores", ())
                    ]
                    if len(candidate_scores) != 12 or not all(
                        math.isfinite(value) for value in candidate_scores
                    ):
                        raise ExperimentError("Stage-B ranker did not return 12 finite scores")
                    candidate_index = min(
                        mask,
                        key=lambda index: (-candidate_scores[index], index),
                    )
                    outcome = candidate_outcomes[candidate_index]
                    candidate_name = str(outcome["candidate_name"])
                    candidate_score = candidate_scores[candidate_index]
                    executed_port = str(outcome["actual_port_label"])
                    executed_choice_edge = str(outcome["actual_edge_id"])
                    local_execution_success = executed_port == proposed_port
                    oracle_execution = True
                    candidate_contact = bool(outcome["immediate_contact"])
                    macro = query["stage_b_choice_macros"][executed_choice_edge]
                    constituent_edges = [
                        str(value) for value in macro["constituent_edge_ids"]
                    ]
                    for edge_position, edge_id in enumerate(constituent_edges):
                        edge = edges[edge_id]
                        if str(edge["source_node_id"]) != constituent_nodes[-1]:
                            raise ExperimentError("Stage-B executed macro is not contiguous")
                        if edge_position == 0:
                            primitive_blocks = OLD.CANDIDATE_BANK[candidate_index][1]
                            _requested, candidate_applied = _slew_stage_b_commands(
                                primitive_blocks,
                                OLD,
                                initial_previous_command=previous_command,
                            )
                            applied = candidate_applied[: int(OLD.TICKS_PER_BLOCK)]
                            selected_candidate_applied_commands = [
                                [float(value) for value in command]
                                for command in applied
                            ]
                        else:
                            applied = _stage_b_edge_applied_commands(
                                str(edge["executed_action_label"]),
                                previous_command,
                                OLD,
                            )
                        previous_command, control_history = _stage_b_roll_control_history(
                            previous_command, control_history, applied
                        )
                        actual_node = str(edge["target_node_id"])
                        constituent_nodes.append(actual_node)
                        constituent_actions.append(str(edge["executed_action_label"]))
                        constituent_costs.append(float(edge["edge_cost"]))
                        update = fresh_update(
                            actual_node, str(edge["executed_action_label"])
                        )
                        next_updates.append(update)
                        constituent_observations.append(
                            str(update["observation"]["observation_id"])
                        )
                        constituent_shas.append(
                            str(update["observation"]["pixel_sha256"])
                        )
                    if actual_node != str(macro["destination_node_id"]):
                        raise ExperimentError("Stage-B macro destination drift")
                    nonmovement_count = 0
                else:
                    update = fresh_update(actual_node, "NO_OP")
                    next_updates = [update]
                    nonmovement_count += 1

                reobservation = next_updates[-1]["observation"]
                distance_before = float(
                    _route_tables(episode)["distances"][str(query["true_node_id"])]
                )
                distance_after = float(_route_tables(episode)["distances"][actual_node])
                progress = distance_before - distance_after
                recovery_before = recovery_pending
                executed_correct = bool(
                    acting
                    and executed_choice_edge == str(query["true_next_edge_id"])
                    and progress > 0.0
                )
                recovery = bool(recovery_before and executed_correct)
                recovery_pending = False if executed_correct else True
                wrong_confident = bool(
                    not abstained
                    and selected_port != str(query["true_next_port_label"])
                )
                false_merge = bool(
                    wrong_confident
                    and selected_node_id != str(query["true_node_id"])
                    and nodes[selected_node_id]["alias_group_id"]
                    == nodes[str(query["true_node_id"])]["alias_group_id"]
                )
                false_loop = bool(
                    wrong_confident
                    and selected_node_id != str(query["true_node_id"])
                    and selected_node_id in visited_nodes
                )
                goal_reached = actual_node == str(episode["goal_node_id"])
                if goal_reached:
                    terminal_reason: str | None = "GOAL_REACHED"
                elif nonmovement_count == int(
                    CONTRACT.STAGE_B_EXECUTION_POLICY[
                        "maximum_consecutive_nonmovement_decisions"
                    ]
                ):
                    terminal_reason = "NONMOVEMENT_LIMIT"
                elif decision_index == int(
                    CONTRACT.STAGE_B_EXECUTION_POLICY["decision_budget"]
                ) - 1:
                    terminal_reason = "DECISION_BUDGET_EXHAUSTED"
                else:
                    terminal_reason = None
                terminal = terminal_reason is not None
                planning_latency_ms = max(
                    ranker_latency_ms,
                    float(
                        local.get("planning_latency_ms", 0.0)
                        if acting
                        else 0.0
                    ),
                    (time.perf_counter() - decision_started) * 1000.0,
                )
                row = {
                    "step_id": f"{execution_id}:decision-{decision_index:02d}",
                    "execution_id": execution_id,
                    "episode_id": episode_id,
                    "family": episode["family"],
                    "condition_id": condition_id,
                    "source_condition_id": source_condition[condition_id],
                    "decision_index": decision_index,
                    "query_id": query["query_id"],
                    "query_index": query["query_index"],
                    "selected_port_label": selected_port,
                    "true_next_port_label": query["true_next_port_label"],
                    "abstained": abstained,
                    "observation_id": final_update["observation"]["observation_id"],
                    "observation_pixel_sha256": final_update["observation"]["pixel_sha256"],
                    "actual_node_id_before": query["true_node_id"],
                    "actual_node_id_after": actual_node,
                    "actual_alias_group_id": nodes[str(query["true_node_id"])]["alias_group_id"],
                    "selected_node_id": selected_node_id,
                    "selected_alias_group_id": nodes[selected_node_id]["alias_group_id"],
                    "prior_visited_node_ids": list(visited_nodes),
                    "belief_node_ids": list(node_ids),
                    "belief_probabilities": belief.tolist(),
                    "normalized_entropy": entropy,
                    "entropy_threshold": entropy_threshold,
                    "belief_proposed_port_label": proposed_port,
                    "belief_selected_edge_id": witness_edge_id,
                    "filter_update_observation_ids": [
                        update["observation"]["observation_id"] for update in pending_updates
                    ],
                    "filter_update_pixel_sha256s": [
                        update["observation"]["pixel_sha256"] for update in pending_updates
                    ],
                    "filter_update_node_ids": [
                        update["observation"]["node_id"] for update in pending_updates
                    ],
                    "filter_update_incoming_action_labels": [
                        update["incoming_action"] for update in pending_updates
                    ],
                    "filter_update_observation_similarities": [
                        update["evidence"]["observation_similarities"] for update in pending_updates
                    ],
                    "filter_update_observation_likelihoods": [
                        update["evidence"]["observation_likelihoods"] for update in pending_updates
                    ],
                    "filter_update_transition_prior_probabilities": [
                        update["evidence"]["transition_prior_probabilities"] for update in pending_updates
                    ],
                    "filter_update_preprojection_probabilities": [
                        update["evidence"]["preprojection_probabilities"] for update in pending_updates
                    ],
                    "filter_update_posterior_probabilities": [
                        update["evidence"]["posterior_probabilities"] for update in pending_updates
                    ],
                    "stored_relative_waypoint": waypoint,
                    "action_disposition": disposition,
                    "ranker_called": acting,
                    "local_candidate_scores": candidate_scores,
                    "local_oracle_admissible_candidate_indices": mask,
                    "local_candidate_index": candidate_index,
                    "local_candidate_name": candidate_name,
                    "local_candidate_score": candidate_score,
                    "local_execution_success": local_execution_success,
                    "ranker_latency_ms": ranker_latency_ms,
                    "planning_latency_ms": planning_latency_ms,
                    "ranker_previous_applied_command": ranker_previous_command,
                    "ranker_control_history": ranker_control_history,
                    "selected_candidate_applied_commands": selected_candidate_applied_commands,
                    "post_decision_previous_applied_command": list(previous_command),
                    "post_decision_control_history": [
                        list(value) for value in control_history
                    ],
                    "executed_port_label": executed_port,
                    "executed_choice_edge_id": executed_choice_edge,
                    "constituent_node_ids": constituent_nodes,
                    "constituent_edge_ids": constituent_edges,
                    "constituent_action_labels": constituent_actions,
                    "constituent_edge_costs_m": constituent_costs,
                    "constituent_observation_ids": constituent_observations,
                    "constituent_observation_pixel_sha256s": constituent_shas,
                    "reobservation_id": reobservation["observation_id"],
                    "reobservation_pixel_sha256": reobservation["pixel_sha256"],
                    "oracle_admissible_execution": oracle_execution,
                    "geodesic_distance_before_m": distance_before,
                    "geodesic_distance_after_m": distance_after,
                    "decision_budget": int(CONTRACT.STAGE_B_EXECUTION_POLICY["decision_budget"]),
                    "prelude_distance_m": prelude_distance,
                    "consecutive_nonmovement_decisions": nonmovement_count,
                    "recovery_pending_before": recovery_before,
                    "recovery_pending_after": recovery_pending,
                    "terminal_reason": terminal_reason,
                    "false_confident_wrong_turn": wrong_confident,
                    "false_place_merge": false_merge,
                    "false_loop_closure": false_loop,
                    "geodesic_progress_m": progress,
                    "replan": not terminal,
                    "recovery": recovery,
                    "immediate_contact": candidate_contact,
                    "stuck": terminal_reason == "NONMOVEMENT_LIMIT",
                    "executed_distance_m": sum(constituent_costs),
                    "oracle_path_distance_m": oracle_path_distance,
                    "terminal": terminal,
                    "goal_reached": goal_reached,
                }
                if set(row) != set(METRICS.STAGE_B_TRACE_FIELDS):
                    raise ExperimentError("Stage-B trace row field projection drift")
                traces.append(row)
                for node_id in constituent_nodes[:-1]:
                    if node_id not in visited_nodes:
                        visited_nodes.append(node_id)
                if terminal:
                    break
                pending_updates = next_updates
    return traces


def stage_b(
    *,
    local_ranker: Callable[..., Mapping[str, Any]] | None = None,
    visual_runtime: Any | None = None,
) -> dict[str, Any]:
    """Conditionally execute the authorized current-frame local-ranker interface."""

    require_source_freeze()
    metrics_a = load_json(OUTPUT_ROOT / "stage_a_metrics.json")
    if not _stage_a_authorizes_stage_b(metrics_a):
        if (OUTPUT_ROOT / "stage_b_trace.jsonl").exists() or (OUTPUT_ROOT / "stage_b_metrics.json").exists():
            raise ExperimentError("conditional Stage-B evidence exists after failed gate")
        return {"stage_b_executed": False, "reason": "STAGE_A_GATE_DID_NOT_AUTHORIZE"}
    for leaf in ("stage_b_trace.jsonl", "stage_b_metrics.json"):
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError(f"Stage-B output already exists: {leaf}")
    from lewm.safety import occluded_goal_topological_belief_metrics_v1 as METRICS
    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    queries = load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl")
    scorer = local_ranker if local_ranker is not None else FrozenCurrentVisualLocalRanker()
    visual = visual_runtime if visual_runtime is not None else StageBObservationRuntime()
    traces = build_stage_b_trace_rows(
        graph_manifest=graph,
        query_rows=queries,
        stage_a_metrics=metrics_a,
        local_ranker=scorer,
        visual_runtime=visual,
    )
    METRICS.validate_stage_b_trace_rows(graph, queries, traces, metrics_a)
    metrics_b = METRICS.recompute_stage_b_metrics(
        graph, queries, traces, metrics_a
    )
    write_jsonl(OUTPUT_ROOT / "stage_b_trace.jsonl", traces)
    atomic_json(OUTPUT_ROOT / "stage_b_metrics.json", metrics_b)
    return metrics_b


def _external_regeneration_mapping() -> dict[str, Any]:
    _require_regular(EXTERNAL_REGENERATION_RECEIPT)
    from scripts import evaluate_occluded_goal_topological_belief_v1 as REDUCER

    try:
        receipt = REDUCER.validate_existing_regeneration_receipt(
            OUTPUT_ROOT, EXTERNAL_REGENERATION_RECEIPT
        )
    except (ValueError, OSError) as exc:
        raise ExperimentError(
            "external independent regeneration exact-rebuild validation failed"
        ) from exc
    if receipt.get("pass") is not True:
        raise ExperimentError("external independent regeneration did not pass")
    return {
        "path": str(EXTERNAL_REGENERATION_RECEIPT),
        "bytes": EXTERNAL_REGENERATION_RECEIPT.stat().st_size,
        "sha256": sha256_file(EXTERNAL_REGENERATION_RECEIPT),
        "validated_mapping": receipt,
    }


def _decision_field(metrics: Mapping[str, Any], field: str, default: Any = None) -> Any:
    if field in metrics:
        return metrics[field]
    decision = metrics.get("decision")
    if isinstance(decision, Mapping) and field in decision:
        return decision[field]
    return default


def report_stage() -> dict[str, Any]:
    require_source_freeze()
    for leaf in ("result.json", "result.md", "file_hashes.json"):
        if (OUTPUT_ROOT / leaf).exists():
            raise ExperimentError(f"publication output already exists: {leaf}")
    metrics_a = load_json(OUTPUT_ROOT / "stage_a_metrics.json")
    stage_b_expected = _stage_a_authorizes_stage_b(metrics_a)
    stage_b_path = OUTPUT_ROOT / "stage_b_metrics.json"
    trace_b_path = OUTPUT_ROOT / "stage_b_trace.jsonl"
    if stage_b_expected != (stage_b_path.is_file() and trace_b_path.is_file()):
        raise ExperimentError("conditional Stage-B disposition drift")
    metrics_b = load_json(stage_b_path) if stage_b_expected else None
    regeneration = _external_regeneration_mapping()
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    runtime_contract = load_json(OUTPUT_ROOT / "contract.json")
    calibration = load_json(OUTPUT_ROOT / "calibration.json")
    latent_index = load_json(OUTPUT_ROOT / "latent_index.json")
    primary = _decision_field(metrics_b or metrics_a, "primary_classification", "UNRESOLVED")
    stage_a_primary = _decision_field(metrics_a, "primary_classification", "UNRESOLVED")
    secondary = [stage_a_primary] if stage_b_expected else []
    next_experiment = _decision_field(metrics_b or metrics_a, "next_experiment", "UNRESOLVED")
    prohibited = {key: 0 for key in CONTRACT.PROHIBITIONS}
    claims = dict(CONTRACT.CLAIMS)
    predecessor = dict(CONTRACT.PREDECESSOR_RESULT_BINDING)
    result = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.result.v1",
            "experiment_id": "OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "status": "DEVELOPMENT_EXPLORATORY_COMPLETE",
            "development_only": True,
            "source_freeze_commit": git("rev-parse", "HEAD"),
            "runtime_contract_digest": runtime_contract["content_digest"],
            "panel_digest": panel["content_digest"],
            "panel_summary": {
                "episodes": panel["episode_count"],
                "queries": panel["query_count"],
                "family_counts": panel["family_counts"],
                "role_counts": panel["role_counts"],
                "adequacy": panel["adequacy"],
            },
            "encoder_binding": {
                "checkpoint_sha256": latent_index["encoder_checkpoint_sha256"],
                "preprocessing_digest": latent_index["preprocessing_digest"],
                "external_encoder_source": latent_index["external_encoder_source"],
                "latent_npz_sha256": latent_index["latents_file"]["sha256"],
            },
            "calibration_binding": {
                "content_digest": calibration["content_digest"],
                "selected_grid_index": calibration["selected_grid_index"],
                "selected_parameters": calibration["selected_parameters"],
                "grid_results": len(calibration["grid_results"]),
                "grid_query_results": len(calibration["grid_query_results"]),
            },
            "stage_a_metrics": metrics_a,
            "stage_b_executed": stage_b_expected,
            "stage_b_disposition": (
                "EXECUTED_AFTER_FROZEN_STAGE_A_GATE"
                if stage_b_expected
                else "NOT_AUTHORIZED_BY_FROZEN_STAGE_A_GATE"
            ),
            "stage_b_metrics": metrics_b,
            "stage_b_current_visual_ranker_binding": (
                _prior_current_visual_binding() if stage_b_expected else None
            ),
            "independent_regeneration": regeneration,
            "primary_classification": primary,
            "secondary_classifications": secondary,
            "next_experiment": next_experiment,
            "claims_boundary": claims,
            "constructed_set_caveat": CONTRACT.CONSTRUCTED_SET_CAVEAT,
            "predecessor_development_context": predecessor,
            "safety_workstream": "REQUIREMENTS_ACQUISITION_REQUIRED",
            "planning_result_resolves_deployment_safety": False,
            "prohibited_action_counters": prohibited,
            "runtime_framework": {
                "direct_foreground_stages": True,
                "custom_python_audit_hooks": False,
                "custom_startup_or_forensic_framework": False,
                "launcher_child_role_hierarchy": False,
                "self_referential_receipts": False,
            },
        }
    )
    atomic_json(OUTPUT_ROOT / "result.json", result)
    stage_a_lines = [
        "| Condition | Top-1 | Top-3 | Edge accuracy | Regret | False-confident | Abstention |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id in CONDITION_IDS:
        aggregate = metrics_a["conditions"][condition_id]["aggregate"]
        stage_a_lines.append(
            "| {condition} | {top1:.6f} | {top3:.6f} | {edge:.6f} | "
            "{regret:.6f} | {false_confident:.6f} | {abstention:.6f} |".format(
                condition=condition_id,
                top1=aggregate["localisation_top1"],
                top3=aggregate["localisation_top3"],
                edge=aggregate["correct_next_edge_accuracy"],
                regret=aggregate["normalized_graph_distance_regret"],
                false_confident=aggregate["false_confident_localisation_rate"],
                abstention=aggregate["abstention_rate"],
            )
        )
    stage_b_lines: list[str]
    if metrics_b is None:
        stage_b_lines = ["Stage B did not run because the frozen Stage-A gate did not authorize it."]
    else:
        stage_b_lines = [
            "Stage B ran on the same 16 development-heldout scene identities with no true- or predicted-future inputs.",
            "",
            "| Condition | Goals | Edge accuracy | Path efficiency | Wrong turns | False-confident wrong turns |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for condition_id in CONTRACT.STAGE_B_CONDITION_IDS:
            aggregate = metrics_b["conditions"][condition_id]
            stage_b_lines.append(
                "| {condition} | {goals}/16 | {edge:.6f} | {efficiency:.6f} | "
                "{wrong:.6f} | {false_confident:.6f} |".format(
                    condition=condition_id,
                    goals=aggregate["goals_reached"],
                    edge=aggregate["correct_next_edge_accuracy"],
                    efficiency=aggregate["path_efficiency"],
                    wrong=aggregate["wrong_turn_rate"],
                    false_confident=aggregate["false_confident_wrong_turn_rate"],
                )
            )
    preserved = predecessor["preserved_development_classifications"]
    report = "\n".join(
        [
            "# OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "",
            "Development-only exploratory result on a constructed, scene-disjoint perceptual-alias challenge.",
            "",
            f"> {CONTRACT.CONSTRUCTED_SET_CAVEAT}",
            "",
            "## Decision",
            "",
            f"Primary classification: `{primary}`.",
            f"Secondary classifications: {', '.join(f'`{value}`' for value in secondary) if secondary else 'none'}.",
            f"Exact next experiment: `{next_experiment}`.",
            "",
            "## Panel and calibration",
            "",
            "The panel contains 96 unique episodes (24 per family; 64 fit, 16 calibration, 16 development-heldout) and eight queries per episode. Every posterior-bank keyframe was encountered during Phase A; Phase-C query captures are distinct later occurrences. Goal markers occur only at recorded goal keyframes.",
            "",
            f"Calibration selected grid index `{calibration['selected_grid_index']}` from 144 registered tuples and 18,432 persisted calibration-query outcomes.",
            "",
            "## Stage A",
            "",
            *stage_a_lines,
            "",
            f"Stage-A classification: `{stage_a_primary}`. Strongest memory condition: `{metrics_a['decision']['strongest_memory_condition']}`. Stage B authorized: `{stage_b_expected}`.",
            "",
            "## Conditional Stage B",
            "",
            *stage_b_lines,
            "",
            "## Context and claims boundary",
            "",
            "The predecessor development result remains bound and unchanged. Its preserved classifications are: "
            + ", ".join(f"`{value}`" for value in preserved)
            + ".",
            "",
            f"Predecessor conclusion: {predecessor['exact_conclusion']}",
            "",
            f"Positive wording is limited to: **{claims['positive_wording']}**",
            "",
            "Oracle graph identities, shortest paths, and admissibility were used only for benchmark construction, evidence reduction, and the explicitly conditional navigation interface. They were never visual-model inputs. This result does not establish deployment safety, learned contact avoidance, physical Go2 safety, online map construction, hidden beacon discovery, novelty, or complete maze navigation.",
            "",
            "The separate safety workstream remains `REQUIREMENTS_ACQUISITION_REQUIRED`.",
            "",
            "## Evidence custody",
            "",
            f"Independent reducer receipt: `{regeneration['sha256']}` ({regeneration['bytes']} bytes), exact-rebuild validated.",
            "",
            "All prohibited-action counters are zero. No custom Python audit hook, startup/forensic framework, launcher hierarchy, predictor training, safety-model training, route-ranker training, online topology mutation, memory construction, novelty, or beacon discovery was introduced.",
            "",
        ]
    )
    atomic_bytes(OUTPUT_ROOT / "result.md", report.encode("utf-8"))
    expected = {
        "contract.json",
        "panel_manifest.json",
        "split_manifest.json",
        "graph_manifest.json",
        "keyframe_index.json",
        "query_ledger.jsonl",
        "observations.npz",
        "latent_index.json",
        "latents.npz",
        "calibration.json",
        "stage_a_beliefs.jsonl",
        "stage_a_metrics.json",
        "result.json",
        "result.md",
    }
    if stage_b_expected:
        expected |= {"stage_b_trace.jsonl", "stage_b_metrics.json"}
    observed = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
    if observed != expected:
        raise ExperimentError(f"official output leaf set drift: {sorted(observed ^ expected)}")
    files: list[dict[str, Any]] = []
    for name in sorted(expected):
        path = OUTPUT_ROOT / name
        _require_regular(path)
        files.append({"path": name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    manifest = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.file_hashes.v1",
            "root": str(OUTPUT_ROOT),
            "file_hashes_self_sha256_excluded": True,
            "file_count_excluding_self": len(files),
            "bytes_excluding_self": sum(row["bytes"] for row in files),
            "files": files,
        }
    )
    atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    return result


def build_freeze_documents() -> dict[str, Any]:
    if git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("freeze documents must be generated on the registered parent")
    contract_builder = getattr(CONTRACT, "build_contract", None)
    if not callable(contract_builder):
        raise ExperimentError("contract module lacks build_contract")
    scientific_contract = dict(contract_builder())
    contract_document = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.contract_document.v1",
            "status": "FROZEN_BEFORE_PANEL_MATERIALIZATION",
            "parent_commit": PARENT_COMMIT,
            "required_freeze_subject": _freeze_subject(),
            "required_result_subject": _result_subject(),
            "scientific_contract": scientific_contract,
        }
    )
    fixture = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.fixture.v1",
            "families": list(FAMILIES),
            "episode_count": 96,
            "queries_per_episode": 8,
            "role_counts": {"FIT": 64, "CALIBRATION": 16, "DEVELOPMENT_HELDOUT": 16},
            "condition_ids": list(CONDITION_IDS),
            "teacher_side_paths": [list(path) for path in teacher_side_paths()],
            "example_graph_digests": {
                family: hashlib.sha256(
                    canonical_bytes(build_episode_graph(family=family, family_rank=0, teacher_path=teacher_side_paths()[FAMILIES.index(family) * 24])[0])
                ).hexdigest()
                for family in FAMILIES
            },
        }
    )
    required_leaves = [
        "contract.json",
        "panel_manifest.json",
        "split_manifest.json",
        "graph_manifest.json",
        "keyframe_index.json",
        "query_ledger.jsonl",
        "observations.npz",
        "latent_index.json",
        "latents.npz",
        "calibration.json",
        "stage_a_beliefs.jsonl",
        "stage_a_metrics.json",
        "result.json",
        "result.md",
        "file_hashes.json",
    ]
    output_schema = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.output_schema.v1",
            "root": str(OUTPUT_ROOT),
            "required_files": required_leaves,
            "conditional_files": {
                "stage_b_trace.jsonl": "present exactly when frozen Stage-A gate authorizes Stage B",
                "stage_b_metrics.json": "present exactly when frozen Stage-A gate authorizes Stage B",
            },
            "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
            "report_publication_part_of_scientific_identity": False,
        }
    )
    preregistration = "\n".join(
        [
            "# OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V1",
            "",
            "Development-only, exploratory, scene-disjoint synthetic graph evaluation.",
            "",
            "The frozen question is whether a persistent belief over visually aliased graph places changes the correct next-edge decision after current-frame and four-frame evidence have become structurally ambiguous.",
            "",
            "Exact graph identities are construction and reduction oracles only. Positive wording is limited to topological belief under an oracle graph in a constructed simulated alias challenge. It is not deployment safety or complete maze navigation.",
            "",
        ]
    )
    atomic_json(DOC_PATHS["contract"], contract_document)
    atomic_json(DOC_PATHS["fixture"], fixture)
    atomic_json(DOC_PATHS["output_schema"], output_schema)
    atomic_bytes(DOC_PATHS["preregistration"], preregistration.encode("utf-8"))
    rows: list[dict[str, Any]] = []
    for relative in SOURCE_PATHS:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise ExperimentError(f"source closure path absent: {relative}")
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    closure = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v1.source_closure.v1",
            "parent_commit": PARENT_COMMIT,
            "row_count": len(rows),
            "rows": rows,
        }
    )
    atomic_json(DOC_PATHS["source_closure"], closure)
    return contract_document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "freeze-docs",
            "panel",
            "render",
            "encode",
            "calibrate",
            "stage-a",
            "stage-b",
            "report",
        ),
    )
    parser.add_argument("--batch-size", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import faulthandler

    faulthandler.enable()
    args = build_parser().parse_args(argv)
    if args.stage == "freeze-docs":
        result = build_freeze_documents()
    elif args.stage == "panel":
        result = panel_stage()
    elif args.stage == "render":
        result = render_stage()
    elif args.stage == "encode":
        result = encode_stage(args.batch_size)
    elif args.stage == "calibrate":
        result = calibrate_stage()
    elif args.stage == "stage-a":
        result = stage_a()
    elif args.stage == "stage-b":
        result = stage_b()
    else:
        result = report_stage()
    print(
        json.dumps(
            {
                "stage": args.stage,
                "status": "PASS",
                "content_digest": result.get("content_digest") if isinstance(result, Mapping) else None,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        _exit_code = main()
    except Exception:
        traceback.print_exc()
        raise
    raise SystemExit(_exit_code)
