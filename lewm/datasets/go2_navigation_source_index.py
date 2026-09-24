"""Deterministic, fail-closed source indexing for Go2 navigation frames.

The paired-navigation dataset builder deliberately accepts an explicit source
index instead of discovering directories itself.  This module builds that
index from the historical ``datagen_full`` layout while preserving the
cross-artifact provenance needed to audit every accepted scene.

Scene directory names are the only scene-specific values inspected before the
opaque held-out SHA-256 commitments are checked.  A forbidden scene is
recorded by digest only and none of its files are opened.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from lewm_worlds.manifest import manifest_sha256, parse_scene_manifest_dict


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_DIR_RE = re.compile(r"^[0-9]+_(?P<scene_id>.+)$")
_RENDER_SCHEMA = "lewm_rendered_vision_v03"
_PLAN_SCHEMA = "lewm_render_replay_plan_v0"
_RUN_SCHEMA = "lewm_genesis_bulk_rollout_run_v0"


class SourceIndexContractError(ValueError):
    """Raised when global source-index inputs are unsafe or malformed."""


@dataclass(frozen=True)
class _RejectedSource(Exception):
    code: str
    detail: str


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )


def _canonical_json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise _RejectedSource("invalid_json_object", f"{path} is not an object")
    return payload


def _scene_id_sha256(scene_id: str) -> str:
    return hashlib.sha256(scene_id.encode("utf-8")).hexdigest()


def _read_commitments(path: Path, *, label: str) -> frozenset[str]:
    values: set[str] = set()
    for line_number, raw in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        value = raw.strip().lower()
        if not value or value.startswith("#"):
            continue
        if not _SHA256_RE.fullmatch(value):
            raise SourceIndexContractError(
                f"{path}:{line_number}: {label} must contain only SHA-256 digests"
            )
        values.add(value)
    if not values:
        raise SourceIndexContractError(f"{label} commitment set is empty: {path}")
    return frozenset(values)


def _resolved_within(path: Path, root: Path, *, label: str) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise _RejectedSource(
            "path_outside_root", f"{label} path {resolved} is outside {root}"
        ) from exc
    return resolved


def _declared_path(value: Any, *, parent: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise _RejectedSource("missing_path", f"missing {label}")
    candidate = Path(value)
    return (candidate if candidate.is_absolute() else parent / candidate).resolve()


def _require_equal(actual: Any, expected: Any, *, code: str, label: str) -> None:
    if actual != expected:
        raise _RejectedSource(code, f"{label}: {actual!r} != {expected!r}")


def _hash_jsonl_and_endpoints(
    path: Path,
) -> tuple[str, int, dict[str, Any], dict[str, Any]]:
    digest = hashlib.sha256()
    count = 0
    first_raw: bytes | None = None
    last_raw: bytes | None = None
    with path.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            if not raw.strip():
                raise _RejectedSource(
                    "blank_frame_metadata_line", f"blank JSONL line in {path}"
                )
            if first_raw is None:
                first_raw = raw
            last_raw = raw
            count += 1
    if first_raw is None or last_raw is None:
        raise _RejectedSource("empty_frame_metadata", f"no rows in {path}")
    try:
        first = json.loads(first_raw)
        last = json.loads(last_raw)
    except json.JSONDecodeError as exc:
        raise _RejectedSource(
            "invalid_frame_endpoint_json", f"invalid endpoint row in {path}: {exc}"
        ) from exc
    if not isinstance(first, dict) or not isinstance(last, dict):
        raise _RejectedSource(
            "invalid_frame_endpoint_json", f"endpoint rows in {path} are not objects"
        )
    return digest.hexdigest(), count, first, last


def _frame_image_path(rgb_dir: Path, frame: Mapping[str, Any]) -> Path:
    try:
        frame_index = int(frame["frame_index"])
        env_index = int(frame["env_index"])
    except (KeyError, TypeError, ValueError) as exc:
        raise _RejectedSource(
            "invalid_frame_endpoint", "frame endpoint lacks integer frame/env indices"
        ) from exc
    return rgb_dir / f"frame_{frame_index:06d}_env_{env_index:02d}.png"


def _validate_frame_endpoint(
    frame: Mapping[str, Any],
    *,
    expected_index: int,
    manifest_digest: str,
    split: str,
) -> None:
    _require_equal(
        int(frame.get("frame_index", -1)),
        expected_index,
        code="frame_index_mismatch",
        label="frame endpoint index",
    )
    episode = frame.get("episode")
    if not isinstance(episode, Mapping):
        raise _RejectedSource(
            "missing_frame_episode", "frame endpoint has no episode metadata"
        )
    if episode.get("manifest_sha256") is not None:
        _require_equal(
            str(episode["manifest_sha256"]),
            manifest_digest,
            code="frame_manifest_hash_mismatch",
            label="frame endpoint manifest hash",
        )
    if episode.get("split") is not None:
        _require_equal(
            str(episode["split"]),
            split,
            code="frame_split_mismatch",
            label="frame endpoint split",
        )


def _discover_plans(
    rollout_root: Path, forbidden: frozenset[str]
) -> dict[str, list[Path]]:
    plans: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(rollout_root.rglob("render_replay_plan.json")):
        match = _PLAN_DIR_RE.fullmatch(path.parent.name)
        if match is None:
            continue
        scene_id = match.group("scene_id")
        if _scene_id_sha256(scene_id) in forbidden:
            continue
        plans[scene_id].append(path.resolve())
    return plans


def _manifest_index(
    corpus_root: Path,
    *,
    forbidden: frozenset[str],
) -> dict[str, list[Path]]:
    manifests: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(corpus_root.glob("*/*/*/manifest.json")):
        scene_id = path.parent.name
        if _scene_id_sha256(scene_id) in forbidden:
            continue
        manifests[scene_id].append(path.resolve())
    return manifests


def _render_selection_identity(
    *, scene_id: str, render_dir: Path, render_root: Path
) -> tuple[str, str]:
    """Read only the small render summary needed for filters and hash ranking."""

    render_dir = _resolved_within(render_dir, render_root, label="render scene")
    summary_path = render_dir / "summary.json"
    if not summary_path.is_file():
        raise _RejectedSource("missing_render_summary", f"missing {summary_path}")
    summary = _read_json(summary_path)
    _require_equal(
        str(summary.get("schema")),
        _RENDER_SCHEMA,
        code="render_schema_mismatch",
        label="render summary schema",
    )
    _require_equal(
        str(summary.get("scene_id")),
        scene_id,
        code="render_scene_id_mismatch",
        label="render summary scene ID",
    )
    _require_equal(
        str(summary.get("render_status")),
        "complete",
        code="render_incomplete",
        label="render status",
    )
    family = str(summary.get("family", ""))
    split = str(summary.get("split", ""))
    if not family or not split:
        raise _RejectedSource(
            "missing_family_or_split", "render summary lacks family or split"
        )
    return family, split


def _selection_rank(*, scene_id: str, family: str, seed: str) -> str:
    payload = f"{seed}\0{family}\0{scene_id}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_one_source(
    *,
    scene_id: str,
    render_dir: Path,
    render_root: Path,
    rollout_root: Path,
    scene_corpus_root: Path,
    plan_paths: Mapping[str, Sequence[Path]],
    manifest_indices: dict[Path, dict[str, list[Path]]],
    forbidden: frozenset[str],
    families: frozenset[str],
    splits: frozenset[str],
) -> tuple[dict[str, Any] | None, str | None]:
    render_dir = _resolved_within(render_dir, render_root, label="render scene")
    render_summary_path = render_dir / "summary.json"
    if not render_summary_path.is_file():
        raise _RejectedSource(
            "missing_render_summary", f"missing {render_summary_path}"
        )
    render_summary = _read_json(render_summary_path)
    _require_equal(
        str(render_summary.get("schema")),
        _RENDER_SCHEMA,
        code="render_schema_mismatch",
        label="render summary schema",
    )
    _require_equal(
        str(render_summary.get("scene_id")),
        scene_id,
        code="render_scene_id_mismatch",
        label="render summary scene ID",
    )
    _require_equal(
        str(render_summary.get("render_status")),
        "complete",
        code="render_incomplete",
        label="render status",
    )
    family = str(render_summary.get("family", ""))
    split = str(render_summary.get("split", ""))
    if not family or not split:
        raise _RejectedSource(
            "missing_family_or_split", "render summary lacks family or split"
        )
    if families and family not in families:
        return None, "family"
    if splits and split not in splits:
        return None, "split"

    discovered_plans = list(plan_paths.get(scene_id, ()))
    if len(discovered_plans) != 1:
        raise _RejectedSource(
            "duplicate_or_missing_render_plan",
            f"found {len(discovered_plans)} rollout plans for {scene_id}",
        )
    plan_path = discovered_plans[0]
    summary_plan_path = _declared_path(
        render_summary.get("plan"),
        parent=render_summary_path.parent,
        label="render-summary plan",
    )
    _require_equal(
        summary_plan_path,
        plan_path,
        code="render_plan_path_mismatch",
        label="render-summary and discovered plan paths",
    )
    _resolved_within(plan_path, rollout_root, label="render plan")
    plan = _read_json(plan_path)
    _require_equal(
        str(plan.get("schema")),
        _PLAN_SCHEMA,
        code="plan_schema_mismatch",
        label="render plan schema",
    )
    _require_equal(
        str(plan.get("scene_id")),
        scene_id,
        code="plan_scene_id_mismatch",
        label="render plan scene ID",
    )
    _require_equal(
        str(plan.get("scene_family")),
        family,
        code="plan_family_mismatch",
        label="render plan family",
    )
    _require_equal(
        str(plan.get("split")),
        split,
        code="plan_split_mismatch",
        label="render plan split",
    )
    manifest_digest = str(plan.get("manifest_sha256", ""))
    if not _SHA256_RE.fullmatch(manifest_digest):
        raise _RejectedSource(
            "invalid_plan_manifest_hash", "render plan lacks a canonical manifest hash"
        )

    frame_count = int(plan.get("frame_count", -1))
    if frame_count <= 0:
        raise _RejectedSource("invalid_frame_count", "render plan frame count is not positive")
    _require_equal(
        int(render_summary.get("frame_count", -1)),
        frame_count,
        code="render_frame_count_mismatch",
        label="render summary and plan frame counts",
    )
    frames_path = _declared_path(
        plan.get("frames_jsonl"), parent=plan_path.parent, label="frames JSONL"
    )
    _resolved_within(frames_path, rollout_root, label="frames JSONL")
    if not frames_path.is_file():
        raise _RejectedSource("missing_frames_jsonl", f"missing {frames_path}")
    frames_sha256, actual_count, first_frame, last_frame = (
        _hash_jsonl_and_endpoints(frames_path)
    )
    _require_equal(
        actual_count,
        frame_count,
        code="frames_line_count_mismatch",
        label="frames JSONL and declared frame counts",
    )
    _validate_frame_endpoint(
        first_frame,
        expected_index=0,
        manifest_digest=manifest_digest,
        split=split,
    )
    _validate_frame_endpoint(
        last_frame,
        expected_index=frame_count - 1,
        manifest_digest=manifest_digest,
        split=split,
    )

    rgb_dir = render_dir / "rgb"
    if not rgb_dir.is_dir():
        raise _RejectedSource("missing_rgb_dir", f"missing {rgb_dir}")
    for endpoint in (first_frame, last_frame):
        image = _frame_image_path(rgb_dir, endpoint)
        if not image.is_file():
            raise _RejectedSource(
                "missing_endpoint_image", f"missing endpoint image {image}"
            )

    rendered_metadata_candidates = [
        path
        for path in (
            render_dir / "rendered_frames.jsonl",
            render_dir / "frames_rendered.jsonl",
            render_dir / "frames.jsonl",
        )
        if path.is_file()
    ]
    if len(rendered_metadata_candidates) > 1:
        raise _RejectedSource(
            "duplicate_rendered_frame_metadata",
            f"multiple rendered-frame metadata files in {render_dir}",
        )

    if len(plan_path.parents) < 3:
        raise _RejectedSource("invalid_plan_layout", f"unexpected plan path {plan_path}")
    chunk_root = plan_path.parents[2]
    run_summary_path = chunk_root / "rollout" / "run_summary.json"
    _resolved_within(run_summary_path, rollout_root, label="rollout run summary")
    if not run_summary_path.is_file():
        raise _RejectedSource(
            "missing_rollout_run_summary", f"missing {run_summary_path}"
        )
    run_summary = _read_json(run_summary_path)
    _require_equal(
        str(run_summary.get("schema")),
        _RUN_SCHEMA,
        code="run_schema_mismatch",
        label="rollout run schema",
    )
    _require_equal(
        str(run_summary.get("family")),
        family,
        code="run_family_mismatch",
        label="rollout run family",
    )
    _require_equal(
        str(run_summary.get("split")),
        split,
        code="run_split_mismatch",
        label="rollout run split",
    )
    origin_corpus = _declared_path(
        run_summary.get("scene_corpus"),
        parent=run_summary_path.parent,
        label="originating scene corpus",
    )
    origin_corpus = _resolved_within(
        origin_corpus, scene_corpus_root, label="originating scene corpus"
    )
    if origin_corpus not in manifest_indices:
        manifest_indices[origin_corpus] = _manifest_index(
            origin_corpus, forbidden=forbidden
        )
    origin_manifests = manifest_indices[origin_corpus].get(scene_id, [])
    if len(origin_manifests) != 1:
        raise _RejectedSource(
            "duplicate_or_missing_scene_manifest",
            f"found {len(origin_manifests)} manifests for {scene_id} in {origin_corpus}",
        )
    manifest_path = origin_manifests[0]
    expected_manifest_path = (
        origin_corpus / split / family / scene_id / "manifest.json"
    ).resolve()
    _require_equal(
        manifest_path,
        expected_manifest_path,
        code="manifest_layout_mismatch",
        label="originating manifest path",
    )
    manifest_payload = _read_json(manifest_path)
    try:
        manifest = parse_scene_manifest_dict(manifest_payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise _RejectedSource(
            "invalid_scene_manifest", f"invalid manifest {manifest_path}: {exc}"
        ) from exc
    _require_equal(
        manifest.scene_id,
        scene_id,
        code="manifest_scene_id_mismatch",
        label="manifest scene ID",
    )
    _require_equal(
        manifest.family,
        family,
        code="manifest_family_mismatch",
        label="manifest family",
    )
    _require_equal(
        str(manifest.split),
        split,
        code="manifest_split_mismatch",
        label="manifest split",
    )
    canonical_manifest_digest = manifest_sha256(manifest)
    _require_equal(
        canonical_manifest_digest,
        manifest_digest,
        code="manifest_hash_mismatch",
        label="canonical manifest and render plan hashes",
    )

    row: dict[str, Any] = {
        "schema": "lewm_go2_navigation_source_v1",
        "scene_id": scene_id,
        "scene_id_sha256": _scene_id_sha256(scene_id),
        "family": family,
        "split": split,
        "scene_manifest_path": str(manifest_path),
        "render_plan_path": str(plan_path),
        "rgb_dir": str(rgb_dir.resolve()),
        "frames_jsonl_path": str(frames_path),
        "render_summary_path": str(render_summary_path.resolve()),
        "rollout_run_summary_path": str(run_summary_path.resolve()),
        "origin_scene_corpus": str(origin_corpus),
        "frame_count": frame_count,
        "hashes": {
            "scene_manifest_sha256": canonical_manifest_digest,
            "scene_manifest_file_sha256": _sha256_file(manifest_path),
            "render_plan_file_sha256": _sha256_file(plan_path),
            "render_summary_file_sha256": _sha256_file(render_summary_path),
            "frames_jsonl_file_sha256": frames_sha256,
            "rollout_run_summary_file_sha256": _sha256_file(run_summary_path),
        },
        "frame_metadata_validation": "sha256_line_count_and_endpoint_contracts",
        "image_validation": "first_and_last_declared_frames_present",
    }
    if rendered_metadata_candidates:
        rendered_path = rendered_metadata_candidates[0].resolve()
        rendered_sha, rendered_count, _first, _last = _hash_jsonl_and_endpoints(
            rendered_path
        )
        _require_equal(
            rendered_count,
            frame_count,
            code="rendered_metadata_count_mismatch",
            label="rendered metadata and declared frame counts",
        )
        row["rendered_frames_jsonl_path"] = str(rendered_path)
        row["hashes"]["rendered_frames_jsonl_file_sha256"] = rendered_sha
    return row, None


def _duplicate_artifact_scene_ids(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, set[str]]:
    fields: dict[str, dict[str, list[str]]] = {
        "scene_manifest_path": defaultdict(list),
        "render_plan_path": defaultdict(list),
        "frames_jsonl_path": defaultdict(list),
        "scene_manifest_sha256": defaultdict(list),
        "render_plan_file_sha256": defaultdict(list),
        "frames_jsonl_file_sha256": defaultdict(list),
    }
    for row in rows:
        scene_id = str(row["scene_id"])
        hashes = row["hashes"]
        for field in ("scene_manifest_path", "render_plan_path", "frames_jsonl_path"):
            fields[field][str(row[field])].append(scene_id)
        for field in (
            "scene_manifest_sha256",
            "render_plan_file_sha256",
            "frames_jsonl_file_sha256",
        ):
            fields[field][str(hashes[field])].append(scene_id)
    duplicate_by_scene: dict[str, set[str]] = defaultdict(set)
    for field, values in fields.items():
        for scene_ids in values.values():
            unique = sorted(set(scene_ids))
            if len(unique) > 1:
                for scene_id in unique:
                    duplicate_by_scene[scene_id].add(field)
    return duplicate_by_scene


def _write_content_addressed(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise SourceIndexContractError(
                f"content-address collision at existing artifact {path}"
            )
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def build_navigation_source_index(
    *,
    render_root: Path,
    rollout_root: Path,
    scene_corpus_root: Path,
    output_dir: Path,
    exclusion_commitment_files: Sequence[tuple[str, Path]] = (),
    development_commitments_path: Path | None = None,
    sealed_commitments_path: Path | None = None,
    families: Iterable[str] = (),
    splits: Iterable[str] = (),
    max_scenes_per_family: int | None = None,
    selection_seed: str = "go2_navigation_source_index_v1",
) -> dict[str, Any]:
    """Validate corpus joins and emit content-addressed source artifacts."""

    render_root = render_root.resolve()
    rollout_root = rollout_root.resolve()
    scene_corpus_root = scene_corpus_root.resolve()
    for root, label in (
        (render_root, "render root"),
        (rollout_root, "rollout root"),
        (scene_corpus_root, "scene corpus root"),
    ):
        if not root.is_dir():
            raise SourceIndexContractError(f"{label} is not a directory: {root}")

    commitment_files = list(exclusion_commitment_files)
    if development_commitments_path is not None:
        commitment_files.append(("v3_development", development_commitments_path))
    if sealed_commitments_path is not None:
        commitment_files.append(("v3_sealed", sealed_commitments_path))
    if not commitment_files:
        raise SourceIndexContractError(
            "at least one labeled scene-ID commitment file is required"
        )
    commitment_sets: dict[str, tuple[Path, frozenset[str]]] = {}
    digest_labels: dict[str, set[str]] = defaultdict(set)
    for raw_label, raw_path in commitment_files:
        label = str(raw_label).strip()
        if not label or not re.fullmatch(r"[A-Za-z0-9_.-]+", label):
            raise SourceIndexContractError(
                f"invalid commitment label {raw_label!r}; use letters, digits, '.', '-', '_'"
            )
        if label in commitment_sets:
            raise SourceIndexContractError(f"duplicate commitment label: {label}")
        path = Path(raw_path).resolve()
        values = _read_commitments(path, label=label)
        commitment_sets[label] = (path, values)
        for digest in values:
            digest_labels[digest].add(label)
    forbidden = frozenset(digest_labels)
    family_filter = frozenset(str(value) for value in families if str(value))
    split_filter = frozenset(str(value) for value in splits if str(value))
    selection_seed = str(selection_seed)
    if not selection_seed:
        raise SourceIndexContractError("selection seed must be non-empty")
    if max_scenes_per_family is not None and int(max_scenes_per_family) <= 0:
        raise SourceIndexContractError("max scenes per family must be positive")

    render_dirs = sorted(
        path
        for path in render_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    )
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    filtered: list[dict[str, str]] = []
    forbidden_records: list[dict[str, Any]] = []
    candidates_by_family: dict[str, list[tuple[str, str, Path]]] = defaultdict(list)

    for render_dir in render_dirs:
        scene_id = render_dir.name
        scene_digest = _scene_id_sha256(scene_id)
        if scene_digest in forbidden:
            forbidden_records.append(
                {
                    "scene_id_sha256": scene_digest,
                    "labels": sorted(digest_labels[scene_digest]),
                }
            )
            continue
        try:
            family, split = _render_selection_identity(
                scene_id=scene_id,
                render_dir=render_dir,
                render_root=render_root,
            )
            if family_filter and family not in family_filter:
                filtered.append({"scene_id": scene_id, "reason": "family"})
                continue
            if split_filter and split not in split_filter:
                filtered.append({"scene_id": scene_id, "reason": "split"})
                continue
            rank = _selection_rank(
                scene_id=scene_id, family=family, seed=selection_seed
            )
            candidates_by_family[family].append((rank, scene_id, render_dir))
        except _RejectedSource as exc:
            rejected.append(
                {"scene_id": scene_id, "code": exc.code, "detail": exc.detail}
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "scene_id": scene_id,
                    "code": "artifact_read_or_parse_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    selected_render_dirs: list[Path] = []
    candidate_counts: dict[str, int] = {}
    selected_counts: dict[str, int] = {}
    for family, candidates in sorted(candidates_by_family.items()):
        ordered = sorted(candidates)
        candidate_counts[family] = len(ordered)
        if max_scenes_per_family is None:
            selected = ordered
            omitted: Sequence[tuple[str, str, Path]] = ()
        else:
            selected = ordered[: int(max_scenes_per_family)]
            omitted = ordered[int(max_scenes_per_family) :]
        selected_counts[family] = len(selected)
        selected_render_dirs.extend(item[2] for item in selected)
        filtered.extend(
            {"scene_id": scene_id, "reason": "max_scenes_per_family"}
            for _rank, scene_id, _render_dir in omitted
        )

    # Deep validation (including large frame-plan hashing) is restricted to
    # the deterministic selection above.
    plan_paths = _discover_plans(rollout_root, forbidden)
    manifest_indices: dict[Path, dict[str, list[Path]]] = {}
    for render_dir in sorted(selected_render_dirs, key=lambda path: path.name):
        scene_id = render_dir.name
        try:
            row, filter_reason = _validate_one_source(
                scene_id=scene_id,
                render_dir=render_dir,
                render_root=render_root,
                rollout_root=rollout_root,
                scene_corpus_root=scene_corpus_root,
                plan_paths=plan_paths,
                manifest_indices=manifest_indices,
                forbidden=forbidden,
                families=family_filter,
                splits=split_filter,
            )
            if filter_reason is not None:
                filtered.append({"scene_id": scene_id, "reason": filter_reason})
            elif row is not None:
                accepted.append(row)
        except _RejectedSource as exc:
            rejected.append(
                {"scene_id": scene_id, "code": exc.code, "detail": exc.detail}
            )
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            rejected.append(
                {
                    "scene_id": scene_id,
                    "code": "artifact_read_or_parse_error",
                    "detail": f"{type(exc).__name__}: {exc}",
                }
            )

    duplicates = _duplicate_artifact_scene_ids(accepted)
    if duplicates:
        retained: list[dict[str, Any]] = []
        for row in accepted:
            scene_id = str(row["scene_id"])
            duplicate_fields = duplicates.get(scene_id)
            if duplicate_fields:
                rejected.append(
                    {
                        "scene_id": scene_id,
                        "code": "duplicate_artifact_identity",
                        "detail": "duplicate fields: "
                        + ", ".join(sorted(duplicate_fields)),
                    }
                )
            else:
                retained.append(row)
        accepted = retained

    accepted.sort(key=lambda row: str(row["scene_id"]))
    rejected.sort(key=lambda row: (str(row["scene_id"]), str(row["code"])))
    filtered.sort(key=lambda row: (str(row["scene_id"]), str(row["reason"])))
    forbidden_records.sort(key=lambda row: str(row["scene_id_sha256"]))
    index_bytes = b"".join(_canonical_json_bytes(row) + b"\n" for row in accepted)
    index_sha256 = hashlib.sha256(index_bytes).hexdigest()
    index_filename = f"go2_navigation_sources_{index_sha256}.jsonl"

    rejection_counts = Counter(str(row["code"]) for row in rejected)
    accepted_by_family_split = Counter(
        (str(row["family"]), str(row["split"])) for row in accepted
    )
    report: dict[str, Any] = {
        "schema": "lewm_go2_navigation_source_index_report_v1",
        "roots": {
            "render": str(render_root),
            "rollout": str(rollout_root),
            "scene_corpus": str(scene_corpus_root),
        },
        "filters": {
            "families": sorted(family_filter),
            "splits": sorted(split_filter),
        },
        "selection": {
            "method": "sha256(seed\\0family\\0scene_id)_ascending",
            "seed": selection_seed,
            "max_scenes_per_family": max_scenes_per_family,
            "candidate_count": sum(candidate_counts.values()),
            "selected_for_deep_validation_count": sum(selected_counts.values()),
            "candidate_by_family": dict(sorted(candidate_counts.items())),
            "selected_for_deep_validation_by_family": dict(
                sorted(selected_counts.items())
            ),
        },
        "exclusions": {
            "comparison": "sha256(utf8(scene_id))",
            "union_count": len(forbidden),
            "union_commitment_set_sha256": _canonical_json_sha256(
                sorted(forbidden)
            ),
            "sets": {
                label: {
                    "count": len(values),
                    "commitment_set_sha256": _canonical_json_sha256(sorted(values)),
                    "file": str(path),
                    "file_sha256": _sha256_file(path),
                }
                for label, (path, values) in sorted(commitment_sets.items())
            },
            "forbidden_raw_scene_ids_persisted": False,
        },
        "counts": {
            "render_directories_discovered": len(render_dirs),
            "eligible_selection_candidates": sum(candidate_counts.values()),
            "selected_for_deep_validation": sum(selected_counts.values()),
            "accepted": len(accepted),
            "filtered": len(filtered),
            "forbidden_before_artifact_open": len(forbidden_records),
            "rejected": len(rejected),
        },
        "accepted_by_family_split": {
            f"{family}/{split}": count
            for (family, split), count in sorted(accepted_by_family_split.items())
        },
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "forbidden": forbidden_records,
        "filtered": filtered,
        "rejected": rejected,
        "index": {
            "filename": index_filename,
            "row_count": len(accepted),
            "sha256": index_sha256,
            "rows_canonical_json_sha256": _canonical_json_sha256(accepted),
        },
    }
    report_bytes = json.dumps(report, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    report_sha256 = hashlib.sha256(report_bytes).hexdigest()
    report_filename = f"go2_navigation_sources_report_{report_sha256}.json"

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / index_filename
    report_path = output_dir / report_filename
    _write_content_addressed(index_path, index_bytes)
    _write_content_addressed(report_path, report_bytes)
    return {
        "index_path": str(index_path),
        "index_sha256": index_sha256,
        "report_path": str(report_path),
        "report_sha256": report_sha256,
        "accepted": len(accepted),
        "filtered": len(filtered),
        "forbidden": len(forbidden_records),
        "rejected": len(rejected),
    }


__all__ = ["SourceIndexContractError", "build_navigation_source_index"]
