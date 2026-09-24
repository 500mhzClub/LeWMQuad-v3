#!/home/andrewknowles/TinyQuadJEPA/bin/python
"""Direct canonical-cache replacement for occluded-goal belief V1.

V2 changes no benchmark identity, split, graph, query, calibration rule,
belief condition, metric, gate, or conditional Stage-B policy.  It copies the
six explicitly reusable V1 input leaves byte-for-byte into a fresh root and
re-encodes each *unique byte image* twice, one image per fresh encoder call.
All later inference resolves an occurrence through
``template -> canonical pixel -> one token/descriptor row``.

Importing this module performs no runtime-root reads, model construction,
checkpoint opening, device initialisation, or scientific computation.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from contextlib import contextmanager
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Iterator
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.safety import occluded_goal_topological_belief_v2_contract as CONTRACT
from scripts import run_occluded_goal_topological_belief_v1 as V1


PARENT_COMMIT = "dafff4b5cfdb7a18009f8719506bcdf337c962b6"
V1_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v1"
)
OUTPUT_ROOT = Path(
    "/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/"
    "occluded_goal_topological_belief_v2"
)
EXTERNAL_REGENERATION_RECEIPT = OUTPUT_ROOT.parent / (
    "occluded_goal_topological_belief_v2_regeneration_receipt.json"
)

FREEZE_SUBJECT = "Freeze canonical occluded-goal topological belief replacement"
RESULT_SUBJECT = "Evaluate canonical occluded-goal topological belief replacement"

DOC_PATHS = {
    "contract": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v2_contract_2026-09-01.json",
    "fixture": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v2_fixture_2026-09-01.json",
    "output_schema": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v2_output_schema_2026-09-01.json",
    "preregistration": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v2_preregistration_2026-09-01.md",
    "source_closure": REPO_ROOT / "docs/lewm_go2_occluded_goal_topological_belief_v2_source_closure_2026-09-01.json",
}

REUSABLE_V1_LEAVES = (
    "panel_manifest.json",
    "split_manifest.json",
    "graph_manifest.json",
    "query_ledger.jsonl",
    "keyframe_index.json",
    "observations.npz",
)
NONREUSABLE_V1_LEAVES = (
    "contract.json",
    "latent_index.json",
    "latents.npz",
    "calibration.json",
    "stage_a_beliefs.jsonl",
    "stage_a_metrics.json",
)
CANONICAL_CACHE_LEAVES = (
    "pixel_index.json",
    "template_to_pixel_index.json",
    "occurrence_index.json",
    "canonical_tokens.npz",
    "canonical_descriptors.npz",
    "canonical_latent_index.json",
    "canonical_descriptor_index.json",
    "encoding_determinism_receipt.json",
    "cache_integrity_receipt.json",
)

EXPECTED_CAPTURE_COUNT = 33_384
EXPECTED_TEMPLATE_COUNT = 375
EXPECTED_UNIQUE_PIXEL_COUNT = 157
EXPECTED_REUSE_COUNT = EXPECTED_TEMPLATE_COUNT - EXPECTED_UNIQUE_PIXEL_COUNT
TOKEN_SHAPE = (768, 1024)
IMAGE_SHAPE = (168, 224, 3)


class ExperimentError(RuntimeError):
    """A V2 source, custody, canonical-cache, or stage invariant failed."""


def canonical_bytes(value: Any) -> bytes:
    payload = CONTRACT.canonical_json_bytes(value)
    if not isinstance(payload, bytes):
        raise ExperimentError("contract canonicalizer did not return bytes")
    return payload.rstrip(b"\n") + b"\n"


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    return dict(CONTRACT.attach_content_digest(value))


def sha256_file(path: Path, chunk_size: int = 4 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_array_sha256(value: Any) -> str:
    """Hash exact shape, NumPy dtype string, and contiguous array bytes."""

    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    header = canonical_bytes(
        {
            "dtype": array.dtype.str,
            "layout": "C",
            "shape": [int(size) for size in array.shape],
        }
    )[:-1]
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _canonical_array_stream_hasher(*, shape: Sequence[int], dtype: Any) -> Any:
    import numpy as np

    header = canonical_bytes(
        {
            "dtype": np.dtype(dtype).str,
            "layout": "C",
            "shape": [int(size) for size in shape],
        }
    )[:-1]
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    return digest


def content_digest(value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("content_digest", None)
    return hashlib.sha256(canonical_bytes(payload)[:-1]).hexdigest()


def _require_regular(
    path: Path, *, absent_ok: bool = False, single_link: bool = False
) -> os.stat_result | None:
    try:
        info = path.lstat()
    except FileNotFoundError:
        if absent_ok:
            return None
        raise ExperimentError(f"required path is absent: {path}") from None
    if path.is_symlink() or not stat.S_ISREG(info.st_mode):
        raise ExperimentError(f"path is not an ordinary regular file: {path}")
    if single_link and info.st_nlink != 1:
        raise ExperimentError(f"official copied leaf does not have nlink==1: {path}")
    return info


def _require_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise ExperimentError(f"required directory is absent: {path}") from exc
    if path.is_symlink() or not stat.S_ISDIR(info.st_mode):
        raise ExperimentError(f"path is not an ordinary directory: {path}")


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
    atomic_bytes(path, canonical_bytes(dict(value)))


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    atomic_bytes(path, b"".join(canonical_bytes(dict(row)) for row in rows))


def atomic_copy(source: Path, target: Path) -> None:
    """Copy bytes without reflinks/hardlinks and require a single-link target."""

    _require_regular(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    if temporary.exists() or target.exists():
        raise ExperimentError(f"copy target already exists: {target}")
    with source.open("rb") as reader, temporary.open("xb") as writer:
        shutil.copyfileobj(reader, writer, length=4 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    os.replace(temporary, target)
    _require_regular(target, single_link=True)


def load_json(path: Path) -> dict[str, Any]:
    _require_regular(path)
    raw = path.read_bytes()
    value = json.loads(raw)
    if not isinstance(value, dict) or raw != canonical_bytes(value):
        raise ExperimentError(f"noncanonical JSON document: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    _require_regular(path)
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_bytes().splitlines(), start=1):
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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or path.exists():
        raise ExperimentError(f"NPZ output already exists: {path}")
    with temporary.open("xb") as raw:
        with zipfile.ZipFile(
            raw, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as archive:
            for name in sorted(arrays):
                if not name or "/" in name or "\\" in name:
                    raise ExperimentError(f"invalid NPZ member name: {name!r}")
                info = zipfile.ZipInfo(
                    f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0)
                )
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, _npy_bytes(arrays[name]))
        raw.flush()
        os.fsync(raw.fileno())
    os.replace(temporary, path)


def _validate_npz_members(path: Path, expected: set[str]) -> None:
    _require_regular(path)
    with zipfile.ZipFile(path, "r") as archive:
        observed = {
            name[:-4]
            for name in archive.namelist()
            if name.endswith(".npy") and "/" not in name
        }
        if len(archive.namelist()) != len(observed) or observed != expected:
            raise ExperimentError(f"NPZ member drift: {path}: {sorted(observed)}")


def git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.STDOUT
    ).strip()


def _contract_document() -> dict[str, Any]:
    document = CONTRACT.build_contract()
    if not isinstance(document, Mapping):
        raise ExperimentError("V2 build_contract did not return a mapping")
    CONTRACT.validate_contract(document)
    return dict(document)


def _v1_binding() -> dict[str, Any]:
    scientific = _contract_document()
    binding = scientific.get("v1_retained_root_binding")
    if not isinstance(binding, Mapping):
        raise ExperimentError("V2 contract lacks v1_retained_root_binding")
    return dict(binding)


def _inventory_rows(binding: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = binding.get("leaves")
    if not isinstance(rows, list) or len(rows) != 12:
        raise ExperimentError("V1 binding must contain exactly 12 inventory rows")
    result = [dict(row) for row in rows if isinstance(row, Mapping)]
    if len(result) != len(rows):
        raise ExperimentError("V1 inventory row type drift")
    return result


def verify_v1_root() -> dict[str, Any]:
    """Rehash the immutable V1 twelve-leaf root without opening science values."""

    binding = _v1_binding()
    if Path(str(binding.get("root"))) != V1_ROOT:
        raise ExperimentError("V1 root binding drift")
    _require_directory(V1_ROOT)
    rows = _inventory_rows(binding)
    expected_names = {str(row.get("path")) for row in rows}
    if expected_names != set(REUSABLE_V1_LEAVES) | set(NONREUSABLE_V1_LEAVES):
        raise ExperimentError("V1 exact inventory name drift")
    observed_names = {path.name for path in V1_ROOT.iterdir() if path.is_file()}
    if observed_names != expected_names or any(
        not path.is_file() for path in V1_ROOT.iterdir()
    ):
        raise ExperimentError("V1 twelve-leaf root inventory changed")
    for row in rows:
        name = str(row["path"])
        path = V1_ROOT / name
        info = _require_regular(path)
        assert info is not None
        if info.st_size != int(row["bytes"]) or sha256_file(path) != str(row["sha256"]):
            raise ExperimentError(f"immutable V1 leaf binding mismatch: {name}")
    if tuple(binding.get("reusable_leaves", ())) != REUSABLE_V1_LEAVES:
        raise ExperimentError("V1 reusable-leaf allow-list drift")
    if tuple(binding.get("invalid_nonreusable_leaves", ())) != NONREUSABLE_V1_LEAVES[1:]:
        raise ExperimentError("V1 nonreuse disposition drift")
    if tuple(binding.get("custody_only_leaves", ())) != ("contract.json",):
        raise ExperimentError("V1 custody-only disposition drift")
    return binding


def require_new_output_root() -> None:
    if not OUTPUT_ROOT.is_absolute() or OUTPUT_ROOT.exists() or OUTPUT_ROOT.is_symlink():
        raise ExperimentError(f"fresh V2 output root is not absent: {OUTPUT_ROOT}")
    if EXTERNAL_REGENERATION_RECEIPT.exists() or EXTERNAL_REGENERATION_RECEIPT.is_symlink():
        raise ExperimentError("fresh V2 external reducer namespace is not absent")
    _require_directory(OUTPUT_ROOT.parent)


def _validate_published_source_closure() -> dict[str, Any]:
    expected_paths = tuple(CONTRACT.TRACKED_SOURCE_PATHS[5:]) + tuple(
        CONTRACT.SOURCE_DEPENDENCY_PATHS
    )
    if len(expected_paths) != 17 or len(set(expected_paths)) != len(expected_paths):
        raise ExperimentError("published source-closure authority cardinality drift")
    closure = load_json(DOC_PATHS["source_closure"])
    if (
        closure.get("schema")
        != "occluded_goal_topological_belief_v2.source_closure.v1"
        or closure.get("parent_commit") != PARENT_COMMIT
        or closure.get("row_count") != 17
        or closure.get("content_digest") != content_digest(closure)
    ):
        raise ExperimentError("published V2 source-closure root drift")
    rows = closure.get("rows")
    if not isinstance(rows, list) or [row.get("path") for row in rows] != list(
        expected_paths
    ):
        raise ExperimentError("published V2 source-closure path/order drift")
    for row in rows:
        path = REPO_ROOT / str(row["path"])
        info = _require_regular(path)
        assert info is not None
        if info.st_size != int(row["bytes"]) or sha256_file(path) != str(
            row["sha256"]
        ):
            raise ExperimentError(
                f"source differs from published closure: {row['path']}"
            )
    contract_document = load_json(DOC_PATHS["contract"])
    if (
        contract_document.get("schema")
        != "occluded_goal_topological_belief_v2.contract_document.v1"
        or contract_document.get("parent_commit") != PARENT_COMMIT
        or contract_document.get("required_freeze_subject") != FREEZE_SUBJECT
        or contract_document.get("required_result_subject") != RESULT_SUBJECT
        or canonical_bytes(contract_document.get("scientific_contract"))
        != canonical_bytes(_contract_document())
    ):
        raise ExperimentError("published V2 scientific contract document drift")
    for key in ("fixture", "output_schema"):
        document = load_json(DOC_PATHS[key])
        if document.get("content_digest") != content_digest(document):
            raise ExperimentError(f"published V2 {key} digest drift")
    _require_regular(DOC_PATHS["preregistration"])
    return closure


def require_source_freeze() -> str:
    head = git("rev-parse", "HEAD")
    if git("rev-parse", "HEAD^") != PARENT_COMMIT:
        raise ExperimentError("V2 freeze is not a direct descendant of the bound parent")
    if git("show", "-s", "--format=%s", "HEAD") != FREEZE_SUBJECT:
        raise ExperimentError("V2 freeze subject drift")
    if git("status", "--porcelain=v1", "--untracked-files=all"):
        raise ExperimentError("V2 source worktree is not clean")
    expected = tuple(getattr(CONTRACT, "TRACKED_SOURCE_PATHS", ()))
    observed = tuple(
        line
        for line in git("diff-tree", "--no-commit-id", "--name-only", "-r", "HEAD").splitlines()
        if line
    )
    if not expected or set(observed) != set(expected) or len(observed) != len(expected):
        raise ExperimentError("V2 freeze changed-path allow-list drift")
    _validate_published_source_closure()
    return head


def _validate_runtime_contract(head: str) -> dict[str, Any]:
    runtime = load_json(OUTPUT_ROOT / "contract.json")
    if runtime.get("source_freeze_commit") != head or runtime.get("parent_commit") != PARENT_COMMIT:
        raise ExperimentError("V2 runtime source binding drift")
    scientific = _contract_document()
    if canonical_bytes(runtime.get("scientific_contract")) != canonical_bytes(scientific):
        raise ExperimentError("V2 runtime scientific contract drift")
    if runtime.get("content_digest") != content_digest(runtime):
        raise ExperimentError("V2 runtime contract content digest drift")
    return runtime


def _verify_copied_inputs(*, recheck_v1: bool = False) -> None:
    binding = verify_v1_root() if recheck_v1 else _v1_binding()
    rows = {str(row["path"]): row for row in _inventory_rows(binding)}
    for name in REUSABLE_V1_LEAVES:
        target = OUTPUT_ROOT / name
        target_info = _require_regular(target, single_link=True)
        assert target_info is not None
        expected = rows[name]
        if (
            target_info.st_size != int(expected["bytes"])
            or sha256_file(target) != str(expected["sha256"])
        ):
            raise ExperimentError(f"V2 copied input differs from V1: {name}")
    for name in NONREUSABLE_V1_LEAVES:
        target = OUTPUT_ROOT / name
        if not target.exists() and not target.is_symlink():
            continue
        if name in {"latent_index.json", "latents.npz"}:
            raise ExperimentError(f"invalid V1 cache filename entered V2: {name}")
        _require_regular(target, single_link=True)
        if sha256_file(target) == str(rows[name]["sha256"]):
            raise ExperimentError(f"invalid V1 science leaf was reused in V2: {name}")


def require_runtime() -> str:
    head = require_source_freeze()
    _require_directory(OUTPUT_ROOT)
    _validate_runtime_contract(head)
    _verify_copied_inputs()
    return head


def bind_inputs_stage() -> dict[str, Any]:
    """Create the fresh root only after the complete V1 binding passes."""

    head = require_source_freeze()
    binding = verify_v1_root()
    require_new_output_root()
    # Close the absent-root and immutable-input races immediately before mkdir.
    binding = verify_v1_root()
    require_new_output_root()
    OUTPUT_ROOT.mkdir(mode=0o700)
    runtime = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.runtime_contract.v1",
            "source_freeze_commit": head,
            "parent_commit": PARENT_COMMIT,
            "scientific_contract": _contract_document(),
            "v1_binding_verified_before_copy": binding,
        }
    )
    atomic_json(OUTPUT_ROOT / "contract.json", runtime)
    for name in REUSABLE_V1_LEAVES:
        atomic_copy(V1_ROOT / name, OUTPUT_ROOT / name)
    _verify_copied_inputs()
    return runtime


def token_layernorm_l2(value: Any) -> Any:
    import numpy as np

    raw = np.asarray(value, dtype=np.float32)
    if raw.shape != TOKEN_SHAPE or not np.isfinite(raw).all():
        raise ExperimentError(f"raw token shape/value drift: {raw.shape}")
    mean = raw.mean(axis=-1, keepdims=True, dtype=np.float32)
    variance = np.square(raw - mean, dtype=np.float32).mean(
        axis=-1, keepdims=True, dtype=np.float32
    )
    normalized = (raw - mean) / np.sqrt(variance + np.float32(1.0e-5))
    norm = np.linalg.norm(normalized, axis=-1, keepdims=True)
    normalized = normalized / np.maximum(norm, np.float32(1.0e-12))
    if not np.isfinite(normalized).all():
        raise ExperimentError("non-finite canonical spatial descriptor")
    return np.ascontiguousarray(normalized, dtype=np.float32)


def _load_pixel_population() -> dict[str, Any]:
    """Validate V1 raw observations and construct the canonical pixel groups."""

    import numpy as np

    observation_path = OUTPUT_ROOT / "observations.npz"
    _validate_npz_members(
        observation_path, {"schema", "pixel_template_ids", "image_sha256", "images"}
    )
    with np.load(observation_path, allow_pickle=False) as data:
        schema = [str(value) for value in data["schema"].tolist()]
        template_ids = [str(value) for value in data["pixel_template_ids"].tolist()]
        source_hashes = [str(value) for value in data["image_sha256"].tolist()]
        images = np.asarray(data["images"])
    if len(schema) != 1 or images.dtype != np.dtype("uint8"):
        raise ExperimentError("V1 observation schema/dtype drift")
    if images.shape != (EXPECTED_TEMPLATE_COUNT, *IMAGE_SHAPE):
        raise ExperimentError(f"V1 observation tensor shape drift: {images.shape}")
    if len(template_ids) != EXPECTED_TEMPLATE_COUNT or len(set(template_ids)) != len(template_ids):
        raise ExperimentError("V1 template identity cardinality drift")
    if len(source_hashes) != len(template_ids):
        raise ExperimentError("V1 source image hash cardinality drift")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source_index, (template_id, source_hash, raw_image) in enumerate(
        zip(template_ids, source_hashes, images)
    ):
        image = np.ascontiguousarray(raw_image, dtype=np.uint8)
        if hashlib.sha256(image.tobytes(order="C")).hexdigest() != source_hash:
            raise ExperimentError("V1 source image hash mismatch")
        pixel_sha = canonical_array_sha256(image)
        groups[pixel_sha].append(
            {
                "pixel_template_id": template_id,
                "source_observation_row_index": source_index,
                "source_image_sha256": source_hash,
                "image": image,
            }
        )
    if len(groups) != EXPECTED_UNIQUE_PIXEL_COUNT:
        raise ExperimentError(
            f"canonical unique-pixel cardinality drift: {len(groups)} != "
            f"{EXPECTED_UNIQUE_PIXEL_COUNT}"
        )

    keyframes = load_json(OUTPUT_ROOT / "keyframe_index.json")
    records = keyframes.get("records")
    captures = keyframes.get("captures")
    if not isinstance(records, list) or len(records) != EXPECTED_TEMPLATE_COUNT:
        raise ExperimentError("V1 keyframe record cardinality drift")
    if not isinstance(captures, list) or len(captures) != EXPECTED_CAPTURE_COUNT:
        raise ExperimentError("V1 occurrence cardinality drift")
    keyframe_by_template = {
        str(row["pixel_template_id"]): dict(row)
        for row in records
        if isinstance(row, Mapping)
    }
    if set(keyframe_by_template) != set(template_ids):
        raise ExperimentError("V1 keyframe/template identity alignment drift")
    for index, template_id in enumerate(template_ids):
        record = keyframe_by_template[template_id]
        if (
            int(record["row_index"]) != index
            or str(record["image_sha256"]) != source_hashes[index]
        ):
            raise ExperimentError("V1 keyframe observation binding drift")

    sorted_hashes = sorted(groups)
    pixel_rows: list[dict[str, Any]] = []
    template_rows: list[dict[str, Any]] = []
    image_by_hash: dict[str, Any] = {}
    template_lookup: dict[str, dict[str, Any]] = {}
    for pixel_index, pixel_sha in enumerate(sorted_hashes):
        members = sorted(groups[pixel_sha], key=lambda row: row["pixel_template_id"])
        canonical_template = str(members[0]["pixel_template_id"])
        first_image = members[0]["image"]
        if any(not np.array_equal(first_image, row["image"]) for row in members[1:]):
            raise ExperimentError("canonical pixel-hash group contains unequal image bytes")
        source_hash_set = {str(row["source_image_sha256"]) for row in members}
        if len(source_hash_set) != 1:
            raise ExperimentError("canonical pixel group disagrees with V1 raw-byte hash")
        image_by_hash[pixel_sha] = first_image
        pixel_rows.append(
            {
                "pixel_index": pixel_index,
                "pixel_sha256": pixel_sha,
                "shape": list(IMAGE_SHAPE),
                "dtype": np.dtype("uint8").str,
                "canonical_template_id": canonical_template,
                "template_ids": [str(row["pixel_template_id"]) for row in members],
                "source_observation_row_indices": [
                    int(row["source_observation_row_index"]) for row in members
                ],
                "source_image_sha256": next(iter(source_hash_set)),
            }
        )
        for member in members:
            template_row = {
                "pixel_template_id": str(member["pixel_template_id"]),
                "source_observation_row_index": int(
                    member["source_observation_row_index"]
                ),
                "source_image_sha256": str(member["source_image_sha256"]),
                "pixel_sha256": pixel_sha,
                "pixel_index": pixel_index,
                "canonical_template_id": canonical_template,
                "reused": str(member["pixel_template_id"]) != canonical_template,
            }
            template_rows.append(template_row)
            template_lookup[template_row["pixel_template_id"]] = template_row
    template_rows.sort(key=lambda row: row["pixel_template_id"])
    if sum(bool(row["reused"]) for row in template_rows) != EXPECTED_REUSE_COUNT:
        raise ExperimentError("canonical template reuse count drift")

    occurrence_rows: list[dict[str, Any]] = []
    seen_capture_ids: set[str] = set()
    for raw_capture in captures:
        if not isinstance(raw_capture, Mapping):
            raise ExperimentError("V1 occurrence row type drift")
        capture = dict(raw_capture)
        capture_id = str(capture["capture_id"])
        template_id = str(capture["pixel_template_id"])
        mapping = template_lookup.get(template_id)
        if capture_id in seen_capture_ids or mapping is None:
            raise ExperimentError("V1 occurrence identity/template drift")
        seen_capture_ids.add(capture_id)
        occurrence_rows.append(
            {
                "capture_id": capture_id,
                "episode_id": str(capture["episode_id"]),
                "node_id": str(capture["node_id"]),
                "phase": str(capture["phase"]),
                "timestamp_s": float(capture["timestamp_s"]),
                "query_ids": [str(value) for value in capture["query_ids"]],
                "pixel_template_id": template_id,
                "pixel_sha256": str(mapping["pixel_sha256"]),
                "pixel_index": int(mapping["pixel_index"]),
                "canonical_template_id": str(mapping["canonical_template_id"]),
            }
        )
    occurrence_rows.sort(key=lambda row: row["capture_id"])
    if len(occurrence_rows) != EXPECTED_CAPTURE_COUNT:
        raise ExperimentError("canonical occurrence row count drift")
    return {
        "source_schema": schema[0],
        "pixel_hashes": sorted_hashes,
        "image_by_hash": image_by_hash,
        "pixel_rows": pixel_rows,
        "template_rows": template_rows,
        "occurrence_rows": occurrence_rows,
        "keyframe_index": keyframes,
    }


def _preprocess_one(arm: Any, image: Any, temporary_root: Path) -> Any:
    import numpy as np
    import torch

    if hasattr(arm, "preprocess_array"):
        value = arm.preprocess_array(np.ascontiguousarray(image))
    else:
        from PIL import Image

        path = temporary_root / "canonical-frame.png"
        Image.fromarray(np.ascontiguousarray(image), mode="RGB").save(path)
        value = arm.preprocess(str(path))
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != (3, 384, 512)
        or value.dtype != torch.float32
    ):
        raise ExperimentError("encoder preprocessing output contract drift")
    if not torch.isfinite(value).all():
        raise ExperimentError("encoder preprocessing produced non-finite values")
    return value.detach().cpu().contiguous()


def _default_arm_factory() -> Any:
    from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm

    return VJepa21Arm()


def _preprocessing_binding(arm: Any, *, fake: bool) -> str:
    if fake:
        value = getattr(arm, "preprocessing_digest", None)
        if not isinstance(value, str) or not value:
            raise ExperimentError("fake encoder lacks preprocessing_digest")
        return value
    from scripts.dev_frozen_dense_representation_encoders_v1 import preprocessing_hash

    return str(preprocessing_hash(arm))


def _encoder_expected_checkpoint() -> str:
    value = getattr(CONTRACT, "ENCODER_CHECKPOINT_SHA256", None)
    if isinstance(value, str) and len(value) == 64:
        return value
    return str(V1.CONTRACT.OBSERVATION_LIKELIHOOD["checkpoint_sha256"])


def _run_encoder_pass(
    *,
    pass_index: int,
    pixel_hashes: Sequence[str],
    image_by_hash: Mapping[str, Any],
    arm_factory: Callable[[], Any],
    fake_encoder: bool,
    expected_raw: Any | None = None,
    expected_descriptors: Any | None = None,
    expected_preprocessed_hashes: Sequence[str] | None = None,
    prior_encoder_instance: Any | None = None,
) -> dict[str, Any]:
    import numpy as np
    import torch

    if list(pixel_hashes) != sorted(pixel_hashes) or len(set(pixel_hashes)) != len(pixel_hashes):
        raise ExperimentError("encoder invocation order is not sorted unique pixel SHA")
    external_source = (
        {"path": "FAKE_TEST_ENCODER", "commit": "FAKE_TEST_ENCODER", "worktree_clean": True}
        if fake_encoder
        else V1.verify_external_vjepa_source()
    )
    arm = arm_factory()
    if arm is prior_encoder_instance:
        raise ExperimentError("encoder factory reused the prior encoder instance")
    instance_id = (
        f"{arm.__class__.__module__}.{arm.__class__.__qualname__}@{id(arm):x}"
    )
    preprocessing_digest = _preprocessing_binding(arm, fake=fake_encoder)
    checkpoint_sha = _encoder_expected_checkpoint()
    if not fake_encoder:
        checkpoint = Path(getattr(arm, "checkpoint"))
        if sha256_file(checkpoint) != checkpoint_sha:
            raise ExperimentError("frozen V-JEPA checkpoint binding mismatch")
        if not torch.cuda.is_available():
            raise ExperimentError("supported V-JEPA GPU path is unavailable")
    device = torch.device("cpu" if fake_encoder else "cuda:0")
    built_module = arm.build(device, torch.float32)
    if not fake_encoder:
        module = built_module if built_module is not None else getattr(arm, "_module", None)
        if module is None or bool(getattr(module, "training", True)):
            raise ExperimentError("frozen encoder was not built in eval mode")
        if any(bool(parameter.requires_grad) for parameter in module.parameters()):
            raise ExperimentError("frozen encoder contains trainable parameters")
    raw_rows: list[Any] = []
    descriptor_rows: list[Any] = []
    preprocessing_hashes: list[str] = []
    comparison_rows: list[dict[str, Any]] = []
    raw_cache_hasher = _canonical_array_stream_hasher(
        shape=(len(pixel_hashes), *TOKEN_SHAPE), dtype=np.float16
    )
    descriptor_cache_hasher = _canonical_array_stream_hasher(
        shape=(len(pixel_hashes), *TOKEN_SHAPE), dtype=np.float32
    )
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"ogtb-v2-encode-pass{pass_index}-") as directory:
        temporary_root = Path(directory)
        for pixel_index, pixel_sha in enumerate(pixel_hashes):
            image = image_by_hash[pixel_sha]
            preprocessed = _preprocess_one(arm, image, temporary_root)
            preprocessed_sha = canonical_array_sha256(preprocessed.numpy())
            pixels = preprocessed.unsqueeze(0).to(device=device, dtype=torch.float32)
            with torch.inference_mode(), torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                encoded = arm.tokens(pixels)
            value = encoded.detach().float().cpu().numpy()
            if value.shape != (1, *TOKEN_SHAPE) or not np.isfinite(value).all():
                raise ExperimentError(f"singleton V-JEPA token contract drift: {value.shape}")
            raw = np.ascontiguousarray(value[0], dtype=np.float16)
            descriptor = token_layernorm_l2(value[0])
            raw_sha = canonical_array_sha256(raw)
            descriptor_sha = canonical_array_sha256(descriptor)
            raw_cache_hasher.update(raw.tobytes(order="C"))
            descriptor_cache_hasher.update(descriptor.tobytes(order="C"))
            preprocessing_hashes.append(preprocessed_sha)
            if pass_index == 1:
                raw_rows.append(raw)
                descriptor_rows.append(descriptor)
            else:
                if (
                    expected_raw is None
                    or expected_descriptors is None
                    or expected_preprocessed_hashes is None
                ):
                    raise ExperimentError("second encoder pass lacks first-pass evidence")
                preprocessed_equal = preprocessed_sha == expected_preprocessed_hashes[pixel_index]
                raw_equal = np.array_equal(raw, expected_raw[pixel_index])
                descriptor_equal = np.array_equal(
                    descriptor, expected_descriptors[pixel_index]
                )
                comparison_rows.append(
                    {
                        "pixel_index": pixel_index,
                        "pixel_sha256": pixel_sha,
                        "pass_1_preprocessed_sha256": expected_preprocessed_hashes[pixel_index],
                        "pass_2_preprocessed_sha256": preprocessed_sha,
                        "preprocessed_exact_equal": preprocessed_equal,
                        "pass_1_raw_token_sha256": canonical_array_sha256(
                            expected_raw[pixel_index]
                        ),
                        "pass_2_raw_token_sha256": raw_sha,
                        "raw_token_exact_equal": raw_equal,
                        "pass_1_spatial_descriptor_sha256": canonical_array_sha256(
                            expected_descriptors[pixel_index]
                        ),
                        "pass_2_spatial_descriptor_sha256": descriptor_sha,
                        "spatial_descriptor_exact_equal": descriptor_equal,
                    }
                )
                if not (preprocessed_equal and raw_equal and descriptor_equal):
                    raise ExperimentError(
                        f"{CONTRACT.CANONICAL_SINGLETON_ENCODER_NONDETERMINISM}: "
                        f"fresh encoder loads disagree for canonical pixel {pixel_sha}"
                    )
    result: dict[str, Any] = {
        "pass_index": pass_index,
        "fresh_encoder_instance_id": instance_id,
        "fresh_encoder_load": True,
        "singleton_batch_size": 1,
        "invocation_count": len(pixel_hashes),
        "invocation_pixel_sha256": list(pixel_hashes),
        "invocation_order_sha256": hashlib.sha256(
            canonical_bytes(list(pixel_hashes))[:-1]
        ).hexdigest(),
        "preprocessed_sha256": preprocessing_hashes,
        "ordered_preprocessed_sha256": hashlib.sha256(
            canonical_bytes(preprocessing_hashes)[:-1]
        ).hexdigest(),
        "checkpoint_sha256": checkpoint_sha,
        "preprocessing_digest": preprocessing_digest,
        "external_encoder_source": external_source,
        "device": str(device),
        "runtime_s": time.time() - started,
        "whole_raw_token_cache_sha256": raw_cache_hasher.hexdigest(),
        "whole_descriptor_cache_sha256": descriptor_cache_hasher.hexdigest(),
    }
    if pass_index == 1:
        raw_array = np.stack(raw_rows).astype(np.float16, copy=False)
        descriptor_array = np.stack(descriptor_rows).astype(np.float32, copy=False)
        result.update(
            {
                "raw_tokens": raw_array,
                "spatial_descriptors": descriptor_array,
            }
        )
    else:
        assert expected_raw is not None and expected_descriptors is not None
        result.update(
            {
                "comparisons": comparison_rows,
            }
        )
    # Drop the model before another fresh construction.  No checkpoint state is reused.
    if hasattr(arm, "_module"):
        delattr(arm, "_module")
    del built_module
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # Retain only the lightweight, unloaded object until the next factory call
    # so Python cannot reuse its identity.  This is custody evidence, never
    # model state reuse.
    result["_encoder_instance_guard"] = arm
    return result


def _ordinary_binding(path: Path, relative_name: str | None = None) -> dict[str, Any]:
    info = _require_regular(path)
    assert info is not None
    return {
        "path": relative_name if relative_name is not None else path.name,
        "bytes": info.st_size,
        "sha256": sha256_file(path),
    }


def _cache_record_digest(records: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(CONTRACT.canonical_json_bytes(list(records))[:-1]).hexdigest()


def _preoutcome_leaf_guard() -> dict[str, Any]:
    forbidden = {
        "calibration.json",
        "stage_a_beliefs.jsonl",
        "stage_a_metrics.json",
        "stage_b_trace.jsonl",
        "stage_b_metrics.json",
        "result.json",
        "result.md",
        "file_hashes.json",
    }
    observed = sorted(name for name in forbidden if (OUTPUT_ROOT / name).exists())
    if observed:
        raise ExperimentError(
            f"pre-canonical-cache root contains outcome/publication leaves: {observed}"
        )
    if EXTERNAL_REGENERATION_RECEIPT.exists() or EXTERNAL_REGENERATION_RECEIPT.is_symlink():
        raise ExperimentError("external reducer evidence exists before cache qualification")
    expected_present = sorted(["contract.json", *REUSABLE_V1_LEAVES])
    entries = list(OUTPUT_ROOT.iterdir())
    observed_present = sorted(path.name for path in entries)
    if observed_present != expected_present:
        raise ExperimentError(
            f"pre-outcome V2 leaf inventory drift: {observed_present}"
        )
    for path in entries:
        _require_regular(path, single_link=True)
    evidence = copy.deepcopy(CONTRACT.PRE_OUTCOME_BOUNDARY_AUTHORITY)
    if evidence["present_v2_leaf_names"] != expected_present:
        raise ExperimentError("contract pre-outcome present-leaf authority drift")
    return evidence


def encode_stage(
    *, arm_factory: Callable[[], Any] | None = None
) -> dict[str, Any]:
    import numpy as np
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    head = require_runtime()
    pre_outcome_boundary = _preoutcome_leaf_guard()
    if any((OUTPUT_ROOT / name).exists() for name in CANONICAL_CACHE_LEAVES):
        raise ExperimentError("canonical encoding output already exists")
    population = _load_pixel_population()
    pixel_hashes = list(population["pixel_hashes"])
    fake_encoder = arm_factory is not None
    factory = arm_factory if arm_factory is not None else _default_arm_factory
    first = _run_encoder_pass(
        pass_index=1,
        pixel_hashes=pixel_hashes,
        image_by_hash=population["image_by_hash"],
        arm_factory=factory,
        fake_encoder=fake_encoder,
    )
    second = _run_encoder_pass(
        pass_index=2,
        pixel_hashes=pixel_hashes,
        image_by_hash=population["image_by_hash"],
        arm_factory=factory,
        fake_encoder=fake_encoder,
        expected_raw=first["raw_tokens"],
        expected_descriptors=first["spatial_descriptors"],
        expected_preprocessed_hashes=first["preprocessed_sha256"],
        prior_encoder_instance=first["_encoder_instance_guard"],
    )
    if (
        first["checkpoint_sha256"] != second["checkpoint_sha256"]
        or first["preprocessing_digest"] != second["preprocessing_digest"]
        or first["external_encoder_source"] != second["external_encoder_source"]
        or first["whole_raw_token_cache_sha256"]
        != second["whole_raw_token_cache_sha256"]
        or first["whole_descriptor_cache_sha256"]
        != second["whole_descriptor_cache_sha256"]
    ):
        raise ExperimentError("two fresh encoder-pass bindings disagree")

    source_observation_binding = _ordinary_binding(
        OUTPUT_ROOT / "observations.npz", "observations.npz"
    )
    pixel_index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.pixel_index.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "observations_binding": source_observation_binding,
            "identity_rule": CONTRACT.CANONICAL_HASH_DOMAINS["rgb_pixel_sha256"],
            "records": [
                {
                    "pixel_index": int(row["pixel_index"]),
                    "pixel_sha256": str(row["pixel_sha256"]),
                    "canonical_template_id": str(row["canonical_template_id"]),
                    "member_template_ids": list(row["template_ids"]),
                    "observation_row_indices": list(
                        row["source_observation_row_indices"]
                    ),
                }
                for row in population["pixel_rows"]
            ],
        }
    )
    template_projection = sorted(
        population["template_rows"],
        key=lambda row: int(row["source_observation_row_index"]),
    )
    template_index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.template_to_pixel_index.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "pixel_index_binding": pixel_index["content_digest"],
            "records": [
                {
                    "template_row_index": index,
                    "pixel_template_id": str(row["pixel_template_id"]),
                    "pixel_sha256": str(row["pixel_sha256"]),
                    "pixel_index": int(row["pixel_index"]),
                    "canonical_template_id": str(row["canonical_template_id"]),
                }
                for index, row in enumerate(template_projection)
            ],
        }
    )
    keyframe_path = OUTPUT_ROOT / "keyframe_index.json"
    occurrence_index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.occurrence_index.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "keyframe_index_binding": _ordinary_binding(
                keyframe_path, "keyframe_index.json"
            ),
            "template_index_binding": template_index["content_digest"],
            "records": [
                {
                    "occurrence_index": index,
                    **{
                        key: row[key]
                        for key in (
                            "capture_id",
                            "episode_id",
                            "node_id",
                            "phase",
                            "timestamp_s",
                            "pixel_template_id",
                            "pixel_sha256",
                            "pixel_index",
                            "query_ids",
                        )
                    },
                }
                for index, row in enumerate(population["occurrence_rows"])
            ],
        }
    )
    raw_tokens = first["raw_tokens"]
    descriptors = first["spatial_descriptors"]
    atomic_json(OUTPUT_ROOT / "pixel_index.json", pixel_index)
    atomic_json(OUTPUT_ROOT / "template_to_pixel_index.json", template_index)
    atomic_json(OUTPUT_ROOT / "occurrence_index.json", occurrence_index)
    atomic_npz(
        OUTPUT_ROOT / "canonical_tokens.npz",
        {
            "schema": np.asarray(
                ["occluded_goal_topological_belief_v2.canonical_tokens.v1"]
            ),
            "pixel_sha256": np.asarray(pixel_hashes),
            "raw_tokens": raw_tokens,
        },
    )
    atomic_npz(
        OUTPUT_ROOT / "canonical_descriptors.npz",
        {
            "schema": np.asarray(
                ["occluded_goal_topological_belief_v2.canonical_descriptors.v1"]
            ),
            "pixel_sha256": np.asarray(pixel_hashes),
            "spatial_descriptors": descriptors,
        },
    )
    canonical_by_hash = {
        str(row["pixel_sha256"]): str(row["canonical_template_id"])
        for row in population["pixel_rows"]
    }
    pass_1_records: list[dict[str, Any]] = [
        {
            "pixel_index": index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": canonical_by_hash[pixel_sha],
            "preprocessed_tensor_sha256": first["preprocessed_sha256"][index],
            "raw_token_sha256": canonical_array_sha256(raw_tokens[index]),
            "spatial_descriptor_sha256": canonical_array_sha256(descriptors[index]),
        }
        for index, pixel_sha in enumerate(pixel_hashes)
    ]
    if len(second["comparisons"]) != len(pixel_hashes):
        raise ExperimentError("second-pass comparison cardinality drift")
    pass_2_records: list[dict[str, Any]] = [
        {
            "pixel_index": index,
            "pixel_sha256": pixel_sha,
            "canonical_template_id": canonical_by_hash[pixel_sha],
            "preprocessed_tensor_sha256": second["comparisons"][index][
                "pass_2_preprocessed_sha256"
            ],
            "raw_token_sha256": second["comparisons"][index][
                "pass_2_raw_token_sha256"
            ],
            "spatial_descriptor_sha256": second["comparisons"][index][
                "pass_2_spatial_descriptor_sha256"
            ],
        }
        for index, pixel_sha in enumerate(pixel_hashes)
    ]
    pixel_order_exact = (
        first["invocation_pixel_sha256"] == second["invocation_pixel_sha256"]
        == pixel_hashes
    )
    preprocessed_exact = all(
        bool(row["preprocessed_exact_equal"]) for row in second["comparisons"]
    )
    raw_exact = all(bool(row["raw_token_exact_equal"]) for row in second["comparisons"])
    descriptor_exact = all(
        bool(row["spatial_descriptor_exact_equal"])
        for row in second["comparisons"]
    )
    cache_digest_exact = (
        _cache_record_digest(pass_1_records) == _cache_record_digest(pass_2_records)
        and first["whole_raw_token_cache_sha256"]
        == second["whole_raw_token_cache_sha256"]
        and first["whole_descriptor_cache_sha256"]
        == second["whole_descriptor_cache_sha256"]
    )
    comparisons_pass = all(
        (
            pixel_order_exact,
            preprocessed_exact,
            raw_exact,
            descriptor_exact,
            cache_digest_exact,
        )
    )
    if not comparisons_pass:
        raise ExperimentError(
            f"{CONTRACT.CANONICAL_SINGLETON_ENCODER_NONDETERMINISM}: "
            "two-pass exact determinism evidence did not pass"
        )
    determinism = {
        "schema": "occluded_goal_topological_belief_v2.encoding_determinism_receipt.v1",
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "source_freeze_commit": head,
        "observations_binding": source_observation_binding,
        "pixel_index_content_digest": pixel_index["content_digest"],
        "pre_outcome_boundary": pre_outcome_boundary,
        "encoder_binding": {
            **{
                field: CONTRACT.VJEPA_ENCODER_BINDING[field]
                for field in (
                    "constructor",
                    "checkpoint_sha256",
                    "checkpoint_size_bytes",
                    "helper_path",
                    "helper_sha256",
                    "external_repository_commit",
                )
            },
            "preprocessing_digest": first["preprocessing_digest"],
        },
        "hash_domains": dict(CONTRACT.CANONICAL_HASH_DOMAINS),
        "counts": {
            "template_rows": EXPECTED_TEMPLATE_COUNT,
            "unique_pixel_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
            "reused_template_rows": EXPECTED_REUSE_COUNT,
            "pass_count": 2,
            "encoder_invocations_per_pass": EXPECTED_UNIQUE_PIXEL_COUNT,
            "singleton_batch_size": 1,
        },
        "passes": [
            {
                "pass_index": pass_index,
                "fresh_encoder_instance_id": pass_result[
                    "fresh_encoder_instance_id"
                ],
                "ordered_pixel_sha256s": pixel_hashes,
                "records": records,
                "canonical_cache_content_digest": _cache_record_digest(records),
            }
            for pass_index, pass_result, records in (
                (1, first, pass_1_records),
                (2, second, pass_2_records),
            )
        ],
        "comparisons": {
            "pixel_order_exact": pixel_order_exact,
            "preprocessed_tensors_exact": preprocessed_exact,
            "raw_tokens_exact": raw_exact,
            "spatial_descriptors_exact": descriptor_exact,
            "canonical_cache_content_digest_exact": cache_digest_exact,
            "pass": comparisons_pass,
        },
    }
    # Canonical evidence receipts deliberately carry no content/self digest.
    atomic_json(OUTPUT_ROOT / "encoding_determinism_receipt.json", determinism)
    determinism_binding = _ordinary_binding(
        OUTPUT_ROOT / "encoding_determinism_receipt.json",
        "encoding_determinism_receipt.json",
    )
    latent_index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.canonical_latent_index.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "source_freeze_commit": head,
            "pixel_index_binding": pixel_index["content_digest"],
            "encoding_determinism_receipt_binding": determinism_binding,
            "tokens_file": _ordinary_binding(
                OUTPUT_ROOT / "canonical_tokens.npz", "canonical_tokens.npz"
            ),
            "records": [
                {
                    "pixel_index": index,
                    "pixel_sha256": pixel_sha,
                    "canonical_template_id": canonical_by_hash[pixel_sha],
                    "raw_token_row_index": index,
                    "raw_token_sha256": pass_1_records[index]["raw_token_sha256"],
                }
                for index, pixel_sha in enumerate(pixel_hashes)
            ],
        }
    )
    descriptor_index = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.canonical_descriptor_index.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "source_freeze_commit": head,
            "pixel_index_binding": pixel_index["content_digest"],
            "encoding_determinism_receipt_binding": determinism_binding,
            "descriptors_file": _ordinary_binding(
                OUTPUT_ROOT / "canonical_descriptors.npz",
                "canonical_descriptors.npz",
            ),
            "records": [
                {
                    "pixel_index": index,
                    "pixel_sha256": pixel_sha,
                    "canonical_template_id": canonical_by_hash[pixel_sha],
                    "spatial_descriptor_row_index": index,
                    "spatial_descriptor_sha256": pass_1_records[index][
                        "spatial_descriptor_sha256"
                    ],
                }
                for index, pixel_sha in enumerate(pixel_hashes)
            ],
        }
    )
    atomic_json(OUTPUT_ROOT / "canonical_latent_index.json", latent_index)
    atomic_json(OUTPUT_ROOT / "canonical_descriptor_index.json", descriptor_index)

    copied_bindings = [
        dict(row)
        for row in CONTRACT.V1_RETAINED_LEAVES
        if row["path"] in CONTRACT.V1_REUSABLE_LEAVES
    ]
    cache_integrity = {
        "schema": "occluded_goal_topological_belief_v2.cache_integrity_receipt.v1",
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "source_freeze_commit": head,
        "v1_retained_root_binding": CONTRACT.v1_retained_root_authority(),
        "copied_v1_input_bindings": copied_bindings,
        "pixel_index_content_digest": pixel_index["content_digest"],
        "template_index_content_digest": template_index["content_digest"],
        "occurrence_index_content_digest": occurrence_index["content_digest"],
        "encoding_determinism_receipt_binding": determinism_binding,
        "canonical_latent_index_content_digest": latent_index["content_digest"],
        "canonical_descriptor_index_content_digest": descriptor_index[
            "content_digest"
        ],
        "counts": {
            "template_rows": EXPECTED_TEMPLATE_COUNT,
            "unique_pixel_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
            "reused_template_rows": EXPECTED_REUSE_COUNT,
            "occurrence_rows": EXPECTED_CAPTURE_COUNT,
            "canonical_token_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
            "canonical_descriptor_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
            "encoder_invocations_per_pass": EXPECTED_UNIQUE_PIXEL_COUNT,
            "multi_template_pixel_groups": 76,
            "singleton_pixel_groups": 81,
            "templates_in_multi_template_groups": 294,
        },
        "gates": {
            field: True for field in METRICS.CACHE_GATE_FIELDS
        },
    }
    atomic_json(OUTPUT_ROOT / "cache_integrity_receipt.json", cache_integrity)
    validation = validate_canonical_cache()
    if validation.get("pass") is not True:
        raise ExperimentError("canonical cache gate did not pass")
    return cache_integrity


def validate_canonical_cache() -> dict[str, Any]:
    """Rebuild the strict cache gate from disk without model inference."""

    import numpy as np
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    require_runtime()
    pixel_index = load_json(OUTPUT_ROOT / "pixel_index.json")
    template_index = load_json(OUTPUT_ROOT / "template_to_pixel_index.json")
    occurrence_index = load_json(OUTPUT_ROOT / "occurrence_index.json")
    latent_index = load_json(OUTPUT_ROOT / "canonical_latent_index.json")
    descriptor_index = load_json(OUTPUT_ROOT / "canonical_descriptor_index.json")
    encoding_receipt = load_json(OUTPUT_ROOT / "encoding_determinism_receipt.json")
    cache_receipt = load_json(OUTPUT_ROOT / "cache_integrity_receipt.json")
    if "content_digest" in encoding_receipt or "content_digest" in cache_receipt:
        raise ExperimentError("canonical evidence receipt contains a forbidden self-digest")
    validation = METRICS.validate_cache_gate(
        pixel_index,
        template_index,
        occurrence_index,
        latent_index,
        descriptor_index,
        encoding_receipt,
        cache_receipt,
    )

    token_path = OUTPUT_ROOT / "canonical_tokens.npz"
    descriptor_path = OUTPUT_ROOT / "canonical_descriptors.npz"
    _validate_npz_members(token_path, {"schema", "pixel_sha256", "raw_tokens"})
    _validate_npz_members(
        descriptor_path, {"schema", "pixel_sha256", "spatial_descriptors"}
    )
    with np.load(token_path, allow_pickle=False) as data:
        token_schema = [str(value) for value in data["schema"].tolist()]
        token_hashes = [str(value) for value in data["pixel_sha256"].tolist()]
        raw_tokens = np.asarray(data["raw_tokens"])
    with np.load(descriptor_path, allow_pickle=False) as data:
        descriptor_schema = [str(value) for value in data["schema"].tolist()]
        descriptor_hashes = [str(value) for value in data["pixel_sha256"].tolist()]
        descriptors = np.asarray(data["spatial_descriptors"])
    expected_hashes = [str(row["pixel_sha256"]) for row in pixel_index["records"]]
    if token_schema != ["occluded_goal_topological_belief_v2.canonical_tokens.v1"]:
        raise ExperimentError("canonical token NPZ schema drift")
    if descriptor_schema != [
        "occluded_goal_topological_belief_v2.canonical_descriptors.v1"
    ]:
        raise ExperimentError("canonical descriptor NPZ schema drift")
    if token_hashes != expected_hashes or descriptor_hashes != expected_hashes:
        raise ExperimentError("canonical NPZ pixel order drift")
    if raw_tokens.shape != (EXPECTED_UNIQUE_PIXEL_COUNT, *TOKEN_SHAPE):
        raise ExperimentError(f"canonical raw-token shape drift: {raw_tokens.shape}")
    if raw_tokens.dtype != np.dtype("float16") or not np.isfinite(raw_tokens).all():
        raise ExperimentError("canonical raw-token dtype/value drift")
    if descriptors.shape != (EXPECTED_UNIQUE_PIXEL_COUNT, *TOKEN_SHAPE):
        raise ExperimentError(f"canonical descriptor shape drift: {descriptors.shape}")
    if descriptors.dtype != np.dtype("float32") or not np.isfinite(descriptors).all():
        raise ExperimentError("canonical descriptor dtype/value drift")
    for index, (raw_record, descriptor_record) in enumerate(
        zip(latent_index["records"], descriptor_index["records"])
    ):
        if (
            canonical_array_sha256(raw_tokens[index])
            != raw_record["raw_token_sha256"]
            or canonical_array_sha256(descriptors[index])
            != descriptor_record["spatial_descriptor_sha256"]
        ):
            raise ExperimentError("canonical cache array/index digest drift")
    if latent_index["tokens_file"] != _ordinary_binding(
        token_path, "canonical_tokens.npz"
    ) or descriptor_index["descriptors_file"] != _ordinary_binding(
        descriptor_path, "canonical_descriptors.npz"
    ):
        raise ExperimentError("canonical cache file binding drift")

    # Recompute the pixel/template/occurrence projection from the copied raw
    # observations.  This catches a coordinated index-only tamper.
    population = _load_pixel_population()
    regenerated_pixels = [
        {
            "pixel_index": int(row["pixel_index"]),
            "pixel_sha256": str(row["pixel_sha256"]),
            "canonical_template_id": str(row["canonical_template_id"]),
            "member_template_ids": list(row["template_ids"]),
            "observation_row_indices": list(row["source_observation_row_indices"]),
        }
        for row in population["pixel_rows"]
    ]
    regenerated_templates = [
        {
            "template_row_index": index,
            "pixel_template_id": str(row["pixel_template_id"]),
            "pixel_sha256": str(row["pixel_sha256"]),
            "pixel_index": int(row["pixel_index"]),
            "canonical_template_id": str(row["canonical_template_id"]),
        }
        for index, row in enumerate(
            sorted(
                population["template_rows"],
                key=lambda value: int(value["source_observation_row_index"]),
            )
        )
    ]
    regenerated_occurrences = [
        {
            "occurrence_index": index,
            **{
                key: row[key]
                for key in (
                    "capture_id",
                    "episode_id",
                    "node_id",
                    "phase",
                    "timestamp_s",
                    "pixel_template_id",
                    "pixel_sha256",
                    "pixel_index",
                    "query_ids",
                )
            },
        }
        for index, row in enumerate(population["occurrence_rows"])
    ]
    if (
        pixel_index["records"] != regenerated_pixels
        or template_index["records"] != regenerated_templates
        or occurrence_index["records"] != regenerated_occurrences
    ):
        raise ExperimentError(
            "canonical pixel/template/occurrence projection regeneration drift"
        )
    return validation


def _load_canonical_context() -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    Any,
]:
    import numpy as np
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    queries = load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl")
    METRICS.validate_graph_manifest(graph)
    METRICS.validate_query_rows(queries, graph)
    episodes = {str(row["episode_id"]): row for row in graph["graphs"]}
    queries_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for query in queries:
        queries_by_episode[str(query["episode_id"])].append(query)

    descriptor_path = OUTPUT_ROOT / "canonical_descriptors.npz"
    with np.load(descriptor_path, allow_pickle=False) as data:
        pixel_hashes = [str(value) for value in data["pixel_sha256"].tolist()]
        descriptors = np.asarray(data["spatial_descriptors"], dtype=np.float32)
    if descriptors.shape != (EXPECTED_UNIQUE_PIXEL_COUNT, *TOKEN_SHAPE):
        raise ExperimentError("canonical descriptor context shape drift")
    pixel_index = load_json(OUTPUT_ROOT / "pixel_index.json")
    template_index = load_json(OUTPUT_ROOT / "template_to_pixel_index.json")
    occurrence_index = load_json(OUTPUT_ROOT / "occurrence_index.json")
    if pixel_hashes != [str(row["pixel_sha256"]) for row in pixel_index["records"]]:
        raise ExperimentError("descriptor context pixel order drift")
    descriptors_by_template = {
        str(row["pixel_template_id"]): descriptors[int(row["pixel_index"])]
        for row in template_index["records"]
    }
    template_by_capture = {
        str(row["capture_id"]): str(row["pixel_template_id"])
        for row in occurrence_index["records"]
    }
    if (
        len(descriptors_by_template) != EXPECTED_TEMPLATE_COUNT
        or len(template_by_capture) != EXPECTED_CAPTURE_COUNT
    ):
        raise ExperimentError("canonical descriptor fanout cardinality drift")
    store = V1.SpatialDescriptorStore(
        descriptors_by_template=descriptors_by_template,
        template_by_capture=template_by_capture,
    )
    return graph, episodes, queries_by_episode, store


def calibrate_stage() -> dict[str, Any]:
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    require_runtime()
    validate_canonical_cache()
    path = OUTPUT_ROOT / "calibration.json"
    if path.exists() or path.is_symlink():
        raise ExperimentError("V2 calibration output already exists")
    for name in (
        "stage_a_beliefs.jsonl",
        "stage_a_metrics.json",
        "stage_b_trace.jsonl",
        "stage_b_metrics.json",
        "result.json",
        "result.md",
        "file_hashes.json",
    ):
        if (OUTPUT_ROOT / name).exists() or (OUTPUT_ROOT / name).is_symlink():
            raise ExperimentError(f"post-calibration leaf exists before calibration: {name}")
    graph, episodes, queries_by_episode, descriptors = _load_canonical_context()
    all_queries = [row for values in queries_by_episode.values() for row in values]
    calibration_queries = sorted(
        (row for row in all_queries if row["role"] == "CALIBRATION"),
        key=lambda row: str(row["query_id"]),
    )
    if len(calibration_queries) != CONTRACT.CALIBRATION_QUERY_COUNT:
        raise ExperimentError("V2 calibration query population drift")
    grid_results: list[dict[str, Any]] = []
    grid_query_results: list[dict[str, Any]] = []
    for grid_index, parameters in enumerate(V1._parameter_grid()):
        outcomes: list[dict[str, Any]] = []
        for query in calibration_queries:
            episode = episodes[str(query["episode_id"])]
            evidence = V1.infer_query_belief(
                episode=episode,
                query=query,
                descriptors_by_observation=descriptors,
                parameters=parameters,
                condition_id="FULL_BELIEF",
            )
            belief = V1.materialize_belief_row(
                episode=episode,
                query=query,
                condition_id="FULL_BELIEF",
                evidence=evidence,
                entropy_threshold=parameters[
                    "normalized_entropy_abstention_threshold"
                ],
            )
            outcome = V1.calibration_query_outcome(
                episode=episode, query=query, belief=belief
            )
            outcomes.append(outcome)
            grid_query_results.append(
                {"grid_index": grid_index, "query_id": query["query_id"], **outcome}
            )
        grid_results.append(
            {
                "grid_index": grid_index,
                "parameters": parameters,
                "correct_next_edge_accuracy": sum(
                    bool(row["edge_correct"]) for row in outcomes
                )
                / len(outcomes),
                "localisation_top3": sum(
                    bool(row["localisation_top3"]) for row in outcomes
                )
                / len(outcomes),
                "normalized_graph_distance_regret": sum(
                    float(row["normalized_regret"]) for row in outcomes
                )
                / len(outcomes),
                "false_confident_localisation_rate": sum(
                    bool(row["false_confident"]) for row in outcomes
                )
                / len(outcomes),
                "abstention_rate": sum(bool(row["abstained"]) for row in outcomes)
                / len(outcomes),
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
    calibration = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.calibration.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "selection_role": "CALIBRATION",
            "calibration_episode_ids": sorted(
                episode_id
                for episode_id, episode in episodes.items()
                if episode["role"] == "CALIBRATION"
            ),
            "selected_parameters": selected["parameters"],
            "selected_grid_index": selected["grid_index"],
            "grid_results": grid_results,
            "grid_query_results": grid_query_results,
        }
    )
    METRICS.validate_calibration(calibration, graph, all_queries)
    atomic_json(path, calibration)
    return calibration


def stage_a() -> dict[str, Any]:
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    require_runtime()
    validate_canonical_cache()
    for leaf in ("stage_a_beliefs.jsonl", "stage_a_metrics.json"):
        if (OUTPUT_ROOT / leaf).exists() or (OUTPUT_ROOT / leaf).is_symlink():
            raise ExperimentError(f"V2 Stage-A output already exists: {leaf}")
    for leaf in (
        "stage_b_trace.jsonl",
        "stage_b_metrics.json",
        "result.json",
        "result.md",
        "file_hashes.json",
    ):
        if (OUTPUT_ROOT / leaf).exists() or (OUTPUT_ROOT / leaf).is_symlink():
            raise ExperimentError(f"post-Stage-A output exists before Stage A: {leaf}")
    graph, episodes, queries_by_episode, descriptors = _load_canonical_context()
    calibration = load_json(OUTPUT_ROOT / "calibration.json")
    all_queries = [row for values in queries_by_episode.values() for row in values]
    METRICS.validate_calibration(calibration, graph, all_queries)
    parameters = dict(calibration["selected_parameters"])
    rows: list[dict[str, Any]] = []
    for episode_id, episode in episodes.items():
        if episode["role"] != "DEVELOPMENT_HELDOUT":
            continue
        for query in sorted(
            queries_by_episode[episode_id], key=lambda row: str(row["query_id"])
        ):
            for condition_id in CONTRACT.CONDITION_IDS:
                evidence = V1.infer_query_belief(
                    episode=episode,
                    query=query,
                    descriptors_by_observation=descriptors,
                    parameters=parameters,
                    condition_id=condition_id,
                )
                rows.append(
                    V1.materialize_belief_row(
                        episode=episode,
                        query=query,
                        condition_id=condition_id,
                        evidence=evidence,
                        entropy_threshold=parameters[
                            "normalized_entropy_abstention_threshold"
                        ],
                    )
                )
    expected = CONTRACT.HELDOUT_QUERY_COUNT * len(CONTRACT.CONDITION_IDS)
    if len(rows) != expected:
        raise ExperimentError(f"V2 Stage-A row count drift: {len(rows)} != {expected}")
    metrics = METRICS.recompute_stage_a_metrics(
        graph, all_queries, rows, calibration
    )
    write_jsonl(OUTPUT_ROOT / "stage_a_beliefs.jsonl", rows)
    atomic_json(OUTPUT_ROOT / "stage_a_metrics.json", metrics)
    return metrics


class CanonicalStageBObservationRuntime:
    """Fresh Stage-B observations backed by the one-row-per-pixel cache."""

    def __init__(self) -> None:
        import numpy as np

        validate_canonical_cache()
        with np.load(OUTPUT_ROOT / "observations.npz", allow_pickle=False) as data:
            template_ids = [
                str(value) for value in data["pixel_template_ids"].tolist()
            ]
            source_hashes = [str(value) for value in data["image_sha256"].tolist()]
            images = np.asarray(data["images"], dtype=np.uint8)
        with np.load(OUTPUT_ROOT / "canonical_tokens.npz", allow_pickle=False) as data:
            pixel_hashes = [str(value) for value in data["pixel_sha256"].tolist()]
            raw_tokens = np.asarray(data["raw_tokens"], dtype=np.float16)
        with np.load(
            OUTPUT_ROOT / "canonical_descriptors.npz", allow_pickle=False
        ) as data:
            descriptor_hashes = [
                str(value) for value in data["pixel_sha256"].tolist()
            ]
            descriptors = np.asarray(data["spatial_descriptors"], dtype=np.float32)
        template_index = load_json(OUTPUT_ROOT / "template_to_pixel_index.json")
        determinism = load_json(OUTPUT_ROOT / "encoding_determinism_receipt.json")
        if pixel_hashes != descriptor_hashes:
            raise ExperimentError("Stage-B canonical cache order drift")
        if raw_tokens.shape != (EXPECTED_UNIQUE_PIXEL_COUNT, *TOKEN_SHAPE) or descriptors.shape != (
            EXPECTED_UNIQUE_PIXEL_COUNT,
            *TOKEN_SHAPE,
        ):
            raise ExperimentError("Stage-B canonical cache shape drift")
        source_by_template = {
            template_id: (np.ascontiguousarray(images[index]), source_hashes[index])
            for index, template_id in enumerate(template_ids)
        }
        self._by_template: dict[str, dict[str, Any]] = {}
        for row in template_index["records"]:
            template_id = str(row["pixel_template_id"])
            pixel_index = int(row["pixel_index"])
            image, source_hash = source_by_template[template_id]
            observed_pixel_hash = canonical_array_sha256(image)
            if (
                observed_pixel_hash != str(row["pixel_sha256"])
                or observed_pixel_hash != pixel_hashes[pixel_index]
                or hashlib.sha256(image.tobytes(order="C")).hexdigest() != source_hash
            ):
                raise ExperimentError("Stage-B template/pixel cache binding drift")
            self._by_template[template_id] = {
                "source_image_sha256": source_hash,
                "pixel_sha256": observed_pixel_hash,
                "image": image,
                "raw_tokens": np.ascontiguousarray(raw_tokens[pixel_index]),
                "spatial_descriptor": np.ascontiguousarray(descriptors[pixel_index]),
            }
        if len(self._by_template) != EXPECTED_TEMPLATE_COUNT:
            raise ExperimentError("Stage-B template cache cardinality drift")
        encoder = determinism["encoder_binding"]
        self.encoder_cache_binding = (
            str(encoder["checkpoint_sha256"]),
            str(encoder["preprocessing_digest"]),
        )

    def observe(
        self,
        *,
        episode: Mapping[str, Any],
        node_id: str,
        occurrence_id: str,
    ) -> dict[str, Any]:
        import numpy as np

        node = next(
            (row for row in episode["nodes"] if str(row["node_id"]) == str(node_id)),
            None,
        )
        if node is None:
            raise ExperimentError("Stage-B fresh observation escaped the graph")
        recipe = json.loads(str(node["observation_descriptor"]))
        image = np.ascontiguousarray(V1.render_observation(recipe), dtype=np.uint8)
        source_hash = hashlib.sha256(image.tobytes(order="C")).hexdigest()
        pixel_sha = canonical_array_sha256(image)
        template_id = str(node["pixel_template_id"])
        cached = self._by_template.get(template_id)
        if (
            cached is None
            or source_hash != str(node["pixel_sha256"])
            or source_hash != cached["source_image_sha256"]
            or pixel_sha != cached["pixel_sha256"]
            or not np.array_equal(image, cached["image"])
        ):
            raise ExperimentError("Stage-B fresh observation/cache identity drift")
        return {
            "observation_id": str(occurrence_id),
            "node_id": str(node_id),
            # Trace schema remains scientifically identical to V1 and binds
            # the registered raw-RGB digest.  Canonical cache identity was
            # independently checked above.
            "pixel_sha256": source_hash,
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
            cached = self._by_template.get(str(node["pixel_template_id"]))
            if cached is None:
                raise ExperimentError("Stage-B node reference is absent from cache")
            references.append(cached["spatial_descriptor"])
        similarity, _ = V1._observation_likelihood(
            observation["spatial_descriptor"],
            np.stack(references).astype(np.float32, copy=False),
            temperature=float(temperature),
        )
        similarity = np.clip(np.asarray(similarity, dtype=np.float64), -1.0, 1.0)
        logits = (similarity - float(similarity.max())) / float(temperature)
        weights = np.exp(np.clip(logits, -80.0, 0.0))
        likelihood = weights / max(float(weights.sum()), 1.0e-300)
        return similarity, likelihood


def stage_b(
    *,
    local_ranker: Callable[..., Mapping[str, Any]] | None = None,
    visual_runtime: Any | None = None,
) -> dict[str, Any]:
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    require_runtime()
    metrics_a = load_json(OUTPUT_ROOT / "stage_a_metrics.json")
    if not METRICS.stage_a_authorizes_stage_b(metrics_a):
        if (OUTPUT_ROOT / "stage_b_trace.jsonl").exists() or (
            OUTPUT_ROOT / "stage_b_metrics.json"
        ).exists():
            raise ExperimentError("conditional Stage-B evidence exists after failed gate")
        return {
            "stage_b_executed": False,
            "reason": "STAGE_A_GATE_DID_NOT_AUTHORIZE",
        }
    for leaf in ("stage_b_trace.jsonl", "stage_b_metrics.json"):
        if (OUTPUT_ROOT / leaf).exists() or (OUTPUT_ROOT / leaf).is_symlink():
            raise ExperimentError(f"V2 Stage-B output already exists: {leaf}")
    # This is intentionally immediately before constructing/opening the frozen
    # local ranker.  Any cache drift stops without a ranker score call.
    validate_canonical_cache()
    graph = load_json(OUTPUT_ROOT / "graph_manifest.json")
    queries = load_jsonl(OUTPUT_ROOT / "query_ledger.jsonl")
    scorer = local_ranker if local_ranker is not None else V1.FrozenCurrentVisualLocalRanker()
    visual = (
        visual_runtime
        if visual_runtime is not None
        else CanonicalStageBObservationRuntime()
    )
    traces = V1.build_stage_b_trace_rows(
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
    _require_regular(EXTERNAL_REGENERATION_RECEIPT, single_link=True)
    from scripts import evaluate_occluded_goal_topological_belief_v2 as REDUCER

    try:
        receipt = REDUCER.validate_existing_regeneration_receipt(
            OUTPUT_ROOT, EXTERNAL_REGENERATION_RECEIPT
        )
    except (ValueError, OSError) as exc:
        raise ExperimentError(
            "external V2 regeneration exact-rebuild validation failed"
        ) from exc
    if receipt.get("pass") is not True:
        raise ExperimentError("external V2 regeneration did not pass")
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
    from lewm.safety import occluded_goal_topological_belief_metrics_v2 as METRICS

    head = require_runtime()
    for leaf in ("result.json", "result.md", "file_hashes.json"):
        if (OUTPUT_ROOT / leaf).exists() or (OUTPUT_ROOT / leaf).is_symlink():
            raise ExperimentError(f"V2 publication output already exists: {leaf}")
    cache_validation = validate_canonical_cache()
    metrics_a = load_json(OUTPUT_ROOT / "stage_a_metrics.json")
    stage_b_expected = METRICS.stage_a_authorizes_stage_b(metrics_a)
    stage_b_path = OUTPUT_ROOT / "stage_b_metrics.json"
    trace_b_path = OUTPUT_ROOT / "stage_b_trace.jsonl"
    if stage_b_expected != (stage_b_path.is_file() and trace_b_path.is_file()):
        raise ExperimentError("V2 conditional Stage-B disposition drift")
    metrics_b = load_json(stage_b_path) if stage_b_expected else None
    regeneration = _external_regeneration_mapping()
    # Recheck the retained root and all copied leaves after the independent
    # reducer and immediately before terminal publication.
    verify_v1_root()
    _verify_copied_inputs()
    panel = load_json(OUTPUT_ROOT / "panel_manifest.json")
    calibration = load_json(OUTPUT_ROOT / "calibration.json")
    determinism = load_json(OUTPUT_ROOT / "encoding_determinism_receipt.json")
    primary = _decision_field(metrics_b or metrics_a, "primary_classification", "UNRESOLVED")
    stage_a_primary = _decision_field(metrics_a, "primary_classification", "UNRESOLVED")
    secondary = [stage_a_primary] if stage_b_expected else []
    next_experiment = _decision_field(metrics_b or metrics_a, "next_experiment", "UNRESOLVED")
    prohibited = {key: 0 for key in CONTRACT.PROHIBITIONS}
    prepublication_files = [path for path in OUTPUT_ROOT.iterdir() if path.is_file()]
    contract_mtime_ns = (OUTPUT_ROOT / "contract.json").stat().st_mtime_ns
    latest_mtime_ns = max(path.stat().st_mtime_ns for path in prepublication_files)
    runtime_storage = {
        "scope": "prepublication official-root observation",
        "leaf_count": len(prepublication_files),
        "bytes": sum(path.stat().st_size for path in prepublication_files),
        "elapsed_from_contract_write_s": max(
            0.0, (latest_mtime_ns - contract_mtime_ns) / 1_000_000_000.0
        ),
        "excludes": ["result.json", "result.md", "file_hashes.json"],
    }
    result = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.result.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "status": "OUTCOME_OBSERVED_CORRECTED_DEVELOPMENT_REPLACEMENT_COMPLETE",
            "development_only": True,
            "outcome_observed": True,
            "source_freeze_commit": head,
            "parent_commit": PARENT_COMMIT,
            "v1_retained_root_binding": CONTRACT.v1_retained_root_authority(),
            "v1_invalid_science_reused": False,
            "panel_binding": {
                "content_digest": panel["content_digest"],
                "episodes": panel["episode_count"],
                "queries": panel["query_count"],
                "family_counts": panel["family_counts"],
                "role_counts": panel["role_counts"],
                "adequacy": panel["adequacy"],
            },
            "canonical_cache_gate": cache_validation,
            "encoding_determinism_receipt": {
                "binding": _ordinary_binding(
                    OUTPUT_ROOT / "encoding_determinism_receipt.json",
                    "encoding_determinism_receipt.json",
                ),
                "encoder_binding": determinism["encoder_binding"],
                "counts": determinism["counts"],
                "comparisons": determinism["comparisons"],
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
                dict(CONTRACT.STAGE_B_CURRENT_VISUAL_BINDING)
                if stage_b_expected
                else None
            ),
            "independent_regeneration": regeneration,
            "primary_classification": primary,
            "secondary_classifications": secondary,
            "next_experiment": next_experiment,
            "claims_boundary": dict(CONTRACT.OUTCOME_OBSERVED_CLAIMS_BOUNDARY),
            "constructed_set_caveat": CONTRACT.CONSTRUCTED_SET_CAVEAT,
            "safety_workstream": "REQUIREMENTS_ACQUISITION_REQUIRED",
            "planning_result_resolves_deployment_safety": False,
            "prohibited_action_counters": prohibited,
            "runtime_storage_observation": runtime_storage,
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

    localisation_lines = [
        "| Condition | Top-1 | Top-3 | MRR | NLL | Brier | ECE-10 | Entropy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    route_lines = [
        "| Condition | Edge acc. | Regret | Route top-3 | Wrong turn | False-conf. | False merge | False loop | Abstain | Correct abstain |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition_id in CONTRACT.CONDITION_IDS:
        aggregate = metrics_a["conditions"][condition_id]["aggregate"]
        localisation_lines.append(
            "| {condition} | {top1:.6f} | {top3:.6f} | {mrr:.6f} | {nll:.6f} | "
            "{brier:.6f} | {ece:.6f} | {entropy:.6f} |".format(
                condition=condition_id,
                top1=aggregate["localisation_top1"],
                top3=aggregate["localisation_top3"],
                mrr=aggregate["mean_reciprocal_rank"],
                nll=aggregate["localisation_nll"],
                brier=aggregate["multiclass_brier"],
                ece=aggregate["ece_10_bin"],
                entropy=aggregate["normalized_belief_entropy"],
            )
        )
        correct_abstention = aggregate["correct_abstention_when_unresolved"]
        route_lines.append(
            "| {condition} | {edge:.6f} | {regret:.6f} | {route_top3:.6f} | "
            "{wrong:.6f} | {false_confident:.6f} | {false_merge:.6f} | "
            "{false_loop:.6f} | {abstention:.6f} | {correct_abstention} |".format(
                condition=condition_id,
                edge=aggregate["correct_next_edge_accuracy"],
                regret=aggregate["normalized_graph_distance_regret"],
                route_top3=aggregate["route_top3"],
                wrong=aggregate["wrong_turn_rate"],
                false_confident=aggregate["false_confident_localisation_rate"],
                false_merge=aggregate["false_place_merge_rate"],
                false_loop=aggregate["false_loop_closure_rate"],
                abstention=aggregate["abstention_rate"],
                correct_abstention=(
                    "n/a"
                    if correct_abstention is None
                    else f"{float(correct_abstention):.6f}"
                ),
            )
        )
    if metrics_b is None:
        stage_b_lines = [
            "Stage B did not run because the unchanged Stage-A gate did not authorize it."
        ]
    else:
        stage_b_lines = [
            "Stage B ran conditionally with the frozen prior current-visual ranker and no future predictor.",
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
                    false_confident=aggregate[
                        "false_confident_wrong_turn_rate"
                    ],
                )
            )
    report = "\n".join(
        [
            "# OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2",
            "",
            "Outcome-observed corrected development replacement on the unchanged constructed alias benchmark.",
            "",
            f"> {CONTRACT.OUTCOME_OBSERVED_CLAIMS_BOUNDARY['must_disclose']}",
            "",
            f"> {CONTRACT.CONSTRUCTED_SET_CAVEAT}",
            "",
            "## Canonical encoding gate",
            "",
            "The 375 template rows resolve to 157 unique exact-pixel rows and 33,384 occurrences. Each unique pixel was encoded alone, in sorted hash order, through two fresh encoder loads. Preprocessed tensors, raw tokens, spatial descriptors, and the canonical cache digest agreed exactly; tolerance was zero.",
            "",
            "## Decision",
            "",
            f"Primary classification: `{primary}`.",
            f"Secondary classifications: {', '.join(f'`{value}`' for value in secondary) if secondary else 'none'}.",
            f"Exact next experiment: `{next_experiment}`.",
            "",
            "## Stage A",
            "",
            *localisation_lines,
            "",
            *route_lines,
            "",
            f"Stage-A classification: `{stage_a_primary}`. Strongest memory condition: `{metrics_a['decision']['strongest_memory_condition']}`. Stage B authorized: `{stage_b_expected}`.",
            "",
            "Fresh calibration selected grid index "
            f"`{calibration['selected_grid_index']}` with parameters "
            f"`{json.dumps(calibration['selected_parameters'], sort_keys=True, separators=(',', ':'))}`.",
            "",
            "FULL versus MAP gate: "
            f"values `{json.dumps(metrics_a['gate']['full_incremental_over_map']['values'], sort_keys=True, separators=(',', ':'))}`, "
            f"checks `{json.dumps(metrics_a['gate']['full_incremental_over_map']['checks'], sort_keys=True, separators=(',', ':'))}`, "
            f"pass `{metrics_a['gate']['full_incremental_over_map']['pass']}`.",
            "",
            "Incremental-over-current dispositions for FIXED_WINDOW_SEQUENCE, MAP_FILTER, TOP_K_BELIEF, and FULL_BELIEF: "
            f"`{json.dumps(metrics_a['gate']['incremental_over_current'], sort_keys=True, separators=(',', ':'))}`.",
            "",
            f"Fixed window matches strongest persistent condition: `{metrics_a['gate']['fixed_window_matches_strongest_persistent']}`; strongest absolute-gate-passing persistent condition: `{metrics_a['gate']['strongest_gate_passing_persistent_condition']}`.",
            "",
            "## Conditional Stage B",
            "",
            *stage_b_lines,
            "",
            "## Claims and safety boundary",
            "",
            f"Positive wording is limited to: **{CONTRACT.CLAIMS['positive_wording']}**",
            "",
            "This result does not establish deployment safety, learned contact avoidance, physical Go2 safety, online map construction, hidden beacon discovery, novelty, or complete maze navigation. The separate safety workstream remains `REQUIREMENTS_ACQUISITION_REQUIRED`.",
            "",
            "All prohibited-action counters are zero. V1 latent-derived science was not reused; no predictor, ranker, or safety model was trained; and no custom audit/startup/forensic framework was introduced.",
            "",
            "Prepublication runtime/storage observation: "
            f"`{runtime_storage['elapsed_from_contract_write_s']:.6f}` seconds, "
            f"`{runtime_storage['leaf_count']}` official leaves, `{runtime_storage['bytes']}` bytes (before result/report/hash-manifest publication).",
            "",
            f"Independent reducer receipt SHA-256: `{regeneration['sha256']}`.",
            "",
        ]
    )
    atomic_bytes(OUTPUT_ROOT / "result.md", report.encode("utf-8"))

    expected = set(CONTRACT.UNCONDITIONAL_OUTPUT_LEAVES) - {"file_hashes.json"}
    if stage_b_expected:
        expected |= {"stage_b_trace.jsonl", "stage_b_metrics.json"}
    observed = {path.name for path in OUTPUT_ROOT.iterdir() if path.is_file()}
    if observed != expected:
        raise ExperimentError(f"V2 official output leaf-set drift: {sorted(observed ^ expected)}")
    files = [_ordinary_binding(OUTPUT_ROOT / name, name) for name in sorted(expected)]
    manifest = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.file_hashes.v1",
            "root": str(OUTPUT_ROOT),
            "file_hashes_self_sha256_excluded": True,
            "file_count_excluding_self": len(files),
            "bytes_excluding_self": sum(int(row["bytes"]) for row in files),
            "files": files,
        }
    )
    atomic_json(OUTPUT_ROOT / "file_hashes.json", manifest)
    verify_v1_root()
    _verify_copied_inputs()
    return result


def build_freeze_documents() -> dict[str, Any]:
    """Generate the five prospective docs and exact 17-row source closure."""

    if git("rev-parse", "HEAD") != PARENT_COMMIT:
        raise ExperimentError("V2 freeze docs must be generated on the bound parent")
    scientific = _contract_document()
    contract_document = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.contract_document.v1",
            "status": "FROZEN_BEFORE_CORRECTED_ENCODING",
            "parent_commit": PARENT_COMMIT,
            "required_freeze_subject": FREEZE_SUBJECT,
            "required_result_subject": RESULT_SUBJECT,
            "scientific_contract": scientific,
        }
    )
    fixture = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.fixture.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "v1_retained_root_binding": CONTRACT.v1_retained_root_authority(),
            "reusable_leaves": list(REUSABLE_V1_LEAVES),
            "invalid_nonreusable_leaves": list(CONTRACT.V1_INVALID_NONREUSABLE_LEAVES),
            "canonical_encoding": {
                "template_rows": EXPECTED_TEMPLATE_COUNT,
                "unique_pixel_rows": EXPECTED_UNIQUE_PIXEL_COUNT,
                "reused_template_rows": EXPECTED_REUSE_COUNT,
                "occurrence_rows": EXPECTED_CAPTURE_COUNT,
                "passes": 2,
                "batch_size": 1,
                "hash_domains": dict(CONTRACT.CANONICAL_HASH_DOMAINS),
                "terminal_failure": CONTRACT.CANONICAL_SINGLETON_ENCODER_NONDETERMINISM,
            },
            "conditions": list(CONTRACT.CONDITION_IDS),
            "scientific_design_unchanged": True,
        }
    )
    output_schema = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.output_schema.v1",
            "experiment_id": CONTRACT.EXPERIMENT_ID,
            "root": str(OUTPUT_ROOT),
            "unconditional_files": list(CONTRACT.UNCONDITIONAL_OUTPUT_LEAVES),
            "conditional_files": {
                "stage_b_trace.jsonl": "present exactly when the unchanged Stage-A gate authorizes Stage B",
                "stage_b_metrics.json": "present exactly when the unchanged Stage-A gate authorizes Stage B",
            },
            "external_regeneration_receipt": str(EXTERNAL_REGENERATION_RECEIPT),
            "report_publication_part_of_scientific_identity": False,
            "receipt_self_digests": False,
        }
    )
    preregistration = "\n".join(
        [
            "# OCCLUDED_GOAL_TOPOLOGICAL_BELIEF_V2",
            "",
            "Development-only, outcome-observed canonical-cache replacement on the unchanged V1 panel and scientific design.",
            "",
            f"> {CONTRACT.OUTCOME_OBSERVED_CLAIMS_BOUNDARY['must_disclose']}",
            "",
            "V2 reuses only the V1 panel, split, graph, query histories, keyframe index, and raw observations. It never opens or reuses V1 latents, calibration, beliefs, or metrics for V2 science.",
            "",
            "Each of the 157 exact unique RGB arrays is encoded alone, in sorted canonical-pixel-hash order, through two fresh frozen-encoder loads. Both passes must agree bitwise before calibration. A mismatch terminates as `CANONICAL_SINGLETON_ENCODER_NONDETERMINISM` with no scientific result.",
            "",
            "All V1 calibration, Stage-A, Stage-B, metric, gate, classification, and next-decision rules are unchanged. Positive wording remains limited to JEPA place belief under oracle topology; this is not deployment safety or complete maze navigation.",
            "",
        ]
    )
    atomic_json(DOC_PATHS["contract"], contract_document)
    atomic_json(DOC_PATHS["fixture"], fixture)
    atomic_json(DOC_PATHS["output_schema"], output_schema)
    atomic_bytes(DOC_PATHS["preregistration"], preregistration.encode("utf-8"))

    source_paths = tuple(CONTRACT.TRACKED_SOURCE_PATHS[5:]) + tuple(
        CONTRACT.SOURCE_DEPENDENCY_PATHS
    )
    if len(source_paths) != 17 or len(set(source_paths)) != len(source_paths):
        raise ExperimentError("V2 source-closure path cardinality/uniqueness drift")
    rows: list[dict[str, Any]] = []
    for relative in source_paths:
        path = REPO_ROOT / relative
        _require_regular(path)
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    closure = attach_digest(
        {
            "schema": "occluded_goal_topological_belief_v2.source_closure.v1",
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
            "bind-inputs",
            "encode",
            "calibrate",
            "stage-a",
            "stage-b",
            "report",
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    import faulthandler

    faulthandler.enable()
    args = build_parser().parse_args(argv)
    if args.stage == "freeze-docs":
        result = build_freeze_documents()
    elif args.stage == "bind-inputs":
        result = bind_inputs_stage()
    elif args.stage == "encode":
        result = encode_stage()
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
                "content_digest": (
                    result.get("content_digest")
                    if isinstance(result, Mapping)
                    else None
                ),
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
