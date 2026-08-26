#!/usr/bin/env python3
"""GPU-only encoder/predictor worker for planning-cost qualification V1.

This process never imports Genesis.  ``preflight`` hashes but does not
deserialize checkpoint files.  ``materialize`` is legal only after the frozen
CPU preexecution and context/fanout indices exist; it persists every raw FP16
grid before any scientific cost is reduced by the CPU coordinator.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import struct
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lewm.safety import (  # noqa: E402
    jepa_local_waypoint_planning_cost_qualification_v1_contract as CONTRACT,
)


TENSOR_SHAPE = (768, 1024)
TENSOR_DTYPE = np.dtype(np.float16)
ENCODER_BATCH = 16
PREDICTOR_BATCH = 12
STATE_COUNT = 48
CANDIDATE_COUNT = 12
LATENT_INDEX_REL = Path("latents/tensor_index.json")
CONTEXT_INDEX_REL = Path("materialization/context_reconstruction_index.json")
GOAL_INDEX_REL = Path("goal_views/index.json")
BATCH_MANIFEST_REL = Path("latents/batch_manifest.json")
GPU_INFERENCE_REL = Path("receipts/gpu_inference.json")
GPU_ENVIRONMENT_REL = Path("receipts/gpu_environment.json")
TARGET_INDEX = ROOT / ".generated/safe_local_waypoint_route_intent_v2/target_latent_index.json"
DENSE_INDEX = ROOT / ".generated/dense_temporal_true_future_safety_observability_v1/token_index.json"
GPU_IMPORT_CLOSURE = (
    "torch",
    "numpy",
    "scipy",
    "PIL",
    "yaml",
    "scripts.dev_frozen_dense_representation_encoders_v1",
    "scripts.dev_proprio_predictor_v1",
    "scripts.run_dev_v03_temporal_action_jepa_v1",
    "scripts.build_dev_v03_proprio_action_manifest_v1",
    "scripts.dev_action_slew_reconstruction_v1",
    "scripts.dev_checkpoint_v1",
    "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract",
)
GPU_LOCAL_SOURCE_MODULE_IDS = GPU_IMPORT_CLOSURE[5:]


class InferenceError(RuntimeError):
    """Fail-closed GPU worker error."""


def sha256_file(path: str | Path, chunk_size: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    def ready(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): ready(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, np.ndarray):
            return ready(item.tolist())
        if isinstance(item, np.generic):
            return ready(item.item())
        if isinstance(item, (list, tuple)):
            return [ready(child) for child in item]
        if isinstance(item, float):
            if not math.isfinite(item):
                raise InferenceError("canonical JSON forbids non-finite floats")
            return item
        if item is None or isinstance(item, (str, int, bool)):
            return item
        raise InferenceError(f"unsupported JSON value {type(item).__name__}")

    return (
        json.dumps(
            ready(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        + "\n"
    ).encode("utf-8")


def attach_digest(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(value))
    payload.pop("content_digest", None)
    payload["content_digest"] = hashlib.sha256(
        canonical_json_bytes(payload)[:-1]
    ).hexdigest()
    return payload


def validate_digest(value: Mapping[str, Any]) -> None:
    payload = copy.deepcopy(dict(value))
    declared = payload.pop("content_digest", None)
    if not isinstance(declared, str) or len(declared) != 64:
        raise InferenceError("missing or malformed content_digest")
    observed = hashlib.sha256(canonical_json_bytes(payload)[:-1]).hexdigest()
    if observed != declared:
        raise InferenceError("content_digest mismatch")


def atomic_bytes(path: str | Path, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    return atomic_bytes(path, canonical_json_bytes(value))


def atomic_npy(path: str | Path, value: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.tmp-", dir=target.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, np.ascontiguousarray(value, dtype=np.float16), allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, target)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return target


def _artifact_reference(path: str | Path, output_root: Path) -> str:
    """Return a publication-stable output-root-relative artifact path."""

    target = Path(path).resolve()
    root = Path(output_root).resolve()
    try:
        return str(target.relative_to(root))
    except ValueError as exc:
        raise InferenceError(f"artifact is outside output root: {target}") from exc


def _artifact_path(reference: str | Path, output_root: Path) -> Path:
    value = Path(reference)
    return value if value.is_absolute() else Path(output_root) / value


def load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_bytes())
    except (OSError, json.JSONDecodeError) as exc:
        raise InferenceError(f"cannot load JSON {path}") from exc
    if not isinstance(value, dict):
        raise InferenceError(f"JSON root is not an object: {path}")
    return value


def phase_core(schema: str, source_freeze_commit: str) -> dict[str, Any]:
    return {
        "schema": schema,
        "experiment_id": CONTRACT.EXPERIMENT_ID,
        "source_freeze_commit": source_freeze_commit,
        "contract_sha256": CONTRACT.CONTRACT_SHA256,
        "output_schema_sha256": CONTRACT.OUTPUT_SCHEMA_SHA256,
    }


def validate_source_commit(
    source_freeze_commit: str, *, require_live_head: bool = False
) -> None:
    if len(source_freeze_commit) != 40:
        raise InferenceError("source-freeze commit must be full-length")
    result = subprocess.run(
        ["git", "rev-parse", f"{source_freeze_commit}^{{commit}}"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    if result.stdout.strip() != source_freeze_commit:
        raise InferenceError("source-freeze commit does not resolve exactly")
    if require_live_head:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if head != source_freeze_commit:
            raise InferenceError("live HEAD differs from source-freeze commit")


def validate_frozen_source_closure() -> dict[str, Any]:
    path = ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH
    value = load_json(path)
    CONTRACT.validate_content_digest(value)
    if value.get("complete") is not True or value.get("missing_paths") != []:
        raise InferenceError("frozen source closure is incomplete")
    try:
        regenerated = CONTRACT.build_source_closure(ROOT, require_complete=True)
    except CONTRACT.ContractError as exc:
        raise InferenceError("source-closure domain regeneration failed") from exc
    if canonical_json_bytes(regenerated) != canonical_json_bytes(value):
        raise InferenceError(
            "stored source closure differs from the complete frozen path domain"
        )
    for row in value.get("rows", []):
        source = ROOT / str(row["path"])
        if not source.is_file() or sha256_file(source) != row["sha256"]:
            raise InferenceError(f"frozen source closure drift: {source}")
    return value


def _validated_interpreter_binary_binding(executable: str | Path) -> dict[str, Any]:
    resolved = Path(executable).resolve()
    if not resolved.is_file():
        raise InferenceError("resolved GPU interpreter binary is missing")
    observed = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }
    if observed != CONTRACT.INTERPRETER_BINARY_BINDING:
        raise InferenceError("resolved GPU interpreter binary binding drift")
    return observed


def validate_phase(value: Mapping[str, Any], source_freeze_commit: str) -> None:
    validate_digest(value)
    if value.get("experiment_id") != CONTRACT.EXPERIMENT_ID:
        raise InferenceError("phase experiment mismatch")
    if value.get("source_freeze_commit") != source_freeze_commit:
        raise InferenceError("phase source-freeze commit mismatch")
    if value.get("contract_sha256") != CONTRACT.CONTRACT_SHA256:
        raise InferenceError("phase contract digest mismatch")
    if value.get("output_schema_sha256") != CONTRACT.OUTPUT_SCHEMA_SHA256:
        raise InferenceError("phase output-schema digest mismatch")


def _checkpoint_bindings() -> dict[str, Any]:
    contract = CONTRACT.build_contract()["predictor_bindings"]
    output: dict[str, Any] = {}
    for name in ("one_step", "two_step"):
        path = Path(contract[name]["path"])
        if not path.is_file():
            raise InferenceError(f"missing {name} checkpoint")
        observed = sha256_file(path)
        if observed != contract[name]["sha256"]:
            raise InferenceError(f"{name} checkpoint SHA drift")
        if path.stat().st_size != int(contract[name]["bytes"]):
            raise InferenceError(f"{name} checkpoint byte drift")
        output[name] = {
            "path": str(path),
            "sha256": observed,
            "bytes": path.stat().st_size,
        }
    encoder = Path.home() / ".cache/vjepa2_1_vitl_dist_vitG_384.pt"
    if not encoder.is_file() or sha256_file(encoder) != CONTRACT.ENCODER_SHA256:
        raise InferenceError("encoder checkpoint binding drift")
    output["encoder"] = {
        "path": str(encoder),
        "sha256": CONTRACT.ENCODER_SHA256,
        "bytes": encoder.stat().st_size,
    }
    return output


def _encoder_source_repository_binding() -> dict[str, Any]:
    expected = CONTRACT.build_contract()["latent_bindings"]["encoder"]
    repository = expected["source_repository"]
    root = Path(repository["path"]).resolve()
    backbones = root / repository["backbones_path"]
    if not root.is_dir() or not backbones.is_file():
        raise InferenceError("frozen encoder source repository is missing")
    observed_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
        cwd=root,
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    observed = {
        "path": str(root),
        "git_commit": observed_commit,
        "worktree_clean": status == "",
        "constructor": expected["constructor"],
        "backbones_path": repository["backbones_path"],
        "backbones_sha256": sha256_file(backbones),
        "backbones_bytes": backbones.stat().st_size,
    }
    required = {
        "path": str(Path(repository["path"]).resolve()),
        "git_commit": repository["git_commit"],
        "worktree_clean": bool(repository["worktree_clean_required"]),
        "constructor": expected["constructor"],
        "backbones_path": repository["backbones_path"],
        "backbones_sha256": repository["backbones_sha256"],
        "backbones_bytes": repository["backbones_bytes"],
    }
    if observed != required:
        raise InferenceError(
            f"frozen encoder source repository binding drift: {observed!r}"
        )
    return observed


PARAMETER_DIGEST_NAMESPACE = b"JEPA_LOCAL_WAYPOINT_PARAMETER_STATE_V1\x00"


def _parameter_state_digest(model: Any) -> str:
    """Digest parameters and buffers in a deterministic, device-neutral form."""

    digest = hashlib.sha256()
    digest.update(PARAMETER_DIGEST_NAMESPACE)
    state = model.state_dict()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        key_bytes = key.encode("utf-8")
        dtype_bytes = str(tensor.dtype).encode("utf-8")
        digest.update(struct.pack(">Q", len(key_bytes)))
        digest.update(key_bytes)
        digest.update(struct.pack(">Q", len(dtype_bytes)))
        digest.update(dtype_bytes)
        digest.update(struct.pack(">Q", tensor.ndim))
        for dimension in tensor.shape:
            digest.update(struct.pack(">q", int(dimension)))
        raw = tensor.numpy().tobytes(order="C")
        digest.update(struct.pack(">Q", len(raw)))
        digest.update(raw)
    return digest.hexdigest()


def _assert_gpu_process_separation() -> None:
    imported = sorted(
        name for name in sys.modules if name == "genesis" or name.startswith("genesis.")
    )
    if imported:
        raise InferenceError(f"GPU worker imported forbidden Genesis modules: {imported}")


def _runtime_import_resolution(import_name: str, expected_root: Path) -> dict[str, Any]:
    expected_root = expected_root.resolve()
    spec = importlib.util.find_spec(import_name)
    if spec is None or spec.origin is None:
        raise InferenceError(f"GPU runtime import is unresolved: {import_name}")
    origin = Path(spec.origin).resolve()
    locations = sorted(
        str(Path(value).resolve()) for value in (spec.submodule_search_locations or ())
    )
    if not origin.is_relative_to(expected_root) or any(
        not Path(value).is_relative_to(expected_root) for value in locations
    ):
        raise InferenceError(f"GPU runtime import shadowing detected: {import_name}")
    module = importlib.import_module(import_name)
    module_file_raw = getattr(module, "__file__", None)
    if module_file_raw is None:
        raise InferenceError(f"GPU runtime import lacks __file__: {import_name}")
    module_file = Path(str(module_file_raw)).resolve()
    if not module_file.is_relative_to(expected_root):
        raise InferenceError(f"GPU live module root drift: {import_name}")
    return {
        "import_name": import_name,
        "find_spec_origin": str(origin),
        "submodule_search_locations": locations,
        "live_module_file": str(module_file),
        "expected_package_root": str(expected_root),
        "resolved_inside_frozen_package_root": True,
        "pass": True,
    }


def _foundational_package_roots() -> dict[str, Any]:
    output: dict[str, Any] = {}
    for package_id, expected in CONTRACT.GPU_FOUNDATIONAL_PACKAGE_BINDINGS.items():
        version = importlib.metadata.version(expected["distribution"])
        root = Path(expected["package_root"]).resolve()
        if version != expected["version"] or not root.is_dir():
            raise InferenceError(f"GPU foundational package drift: {package_id}")
        output[package_id] = {
            **copy.deepcopy(expected),
            "package_root": str(root),
            "import_resolution": _runtime_import_resolution(
                expected["import_name"], root
            ),
        }
    return output


def _live_gpu_environment_identity() -> dict[str, Any]:
    """Return the exact live interpreter/package/device identity."""

    import torch
    import yaml
    from PIL import __version__ as pillow_version
    from scipy import __version__ as scipy_version

    if not torch.cuda.is_available():
        raise InferenceError("frozen ROCm/CUDA-compatible GPU is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    entrypoint = Path(sys.executable).absolute()
    interpreter_binding = _validated_interpreter_binary_binding(entrypoint)
    return {
        "python": sys.version.split()[0],
        "executable": str(entrypoint),
        "interpreter_binding": interpreter_binding,
        "torch": torch.__version__,
        "packages": {
            "numpy": np.__version__,
            "scipy": scipy_version,
            "pillow": pillow_version,
            "pyyaml": yaml.__version__,
        },
        "cuda_available": True,
        "device": {
            "type": "cuda",
            "index": 0,
            "name": str(properties.name),
            "total_memory_bytes": int(properties.total_memory),
            "hip": torch.version.hip,
        },
    }


def _gpu_environment_binding_for_inference(
    source_freeze_commit: str, *, output_root: Path
) -> dict[str, Any]:
    """Revalidate the preflight GPU environment in the live GPU child.

    Preflight and inference are separate processes.  The long CPU replay sits
    between them, so the inference process independently proves that every
    foundational import still resolves to the frozen root before any
    checkpoint tensor is deserialized.
    """

    path = output_root / GPU_ENVIRONMENT_REL
    value = load_json(path)
    validate_phase(value, source_freeze_commit)
    live = _live_gpu_environment_identity()
    observed_foundational = _foundational_package_roots()
    if value.get("foundational_package_roots") != observed_foundational:
        raise InferenceError(
            "GPU foundational package custody drift between preflight and inference"
        )
    if value.get("foundational_package_closure_policy") != (
        CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
    ):
        raise InferenceError("GPU foundational package closure-policy drift")
    expected = CONTRACT.build_contract()["execution"]["environments"][
        "encoder_predictor"
    ]
    expected_packages = {
        "numpy": expected["numpy"],
        "scipy": expected["scipy"],
        "pillow": expected["pillow"],
        "pyyaml": expected["pyyaml"],
    }
    if (
        live["python"] != expected["python"]
        or Path(live["executable"]).absolute()
        != Path(expected["interpreter"]).absolute()
        or live["interpreter_binding"] != CONTRACT.INTERPRETER_BINARY_BINDING
        or live["torch"] != expected["torch"]
        or live["packages"] != expected_packages
        or live["device"]["name"] != "AMD Radeon AI PRO R9700"
        or value.get("python") != live["python"]
        or value.get("executable") != live["executable"]
        or value.get("interpreter_binding") != live["interpreter_binding"]
        or value.get("torch") != live["torch"]
        or value.get("packages") != live["packages"]
        or value.get("device") != live["device"]
        or value.get("checkpoint_tensor_open_count") != 0
        or value.get("predictor_inference_calls") != 0
        or value.get("genesis_imported") is not False
        or value.get("pass") is not True
    ):
        raise InferenceError("GPU preflight environment receipt drift")
    _assert_gpu_process_separation()
    return {
        "path": str(GPU_ENVIRONMENT_REL),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "content_digest": value["content_digest"],
    }


def preflight(
    source_freeze_commit: str, *, output_root: Path, receipt: Path
) -> dict[str, Any]:
    """Hash and import closure only; checkpoint tensors remain unopened."""

    validate_source_commit(source_freeze_commit, require_live_head=True)
    bindings = _checkpoint_bindings()
    import torch
    import yaml
    from PIL import __version__ as pillow_version
    from scipy import __version__ as scipy_version
    from scripts import build_dev_v03_proprio_action_manifest_v1 as M
    from scripts import dev_action_slew_reconstruction_v1 as SLEW
    from scripts import dev_checkpoint_v1 as CK
    from scripts import dev_frozen_dense_representation_encoders_v1 as E
    from scripts import dev_proprio_predictor_v1 as P
    from scripts import run_dev_v03_temporal_action_jepa_v1 as T

    _ = (
        E.VJepa21CroppedV03Arm,
        P.unroll,
        T.normalise,
        M.__name__,
        SLEW.ACTION_DIM,
        CK.__name__,
    )
    _assert_gpu_process_separation()
    encoder_source = _encoder_source_repository_binding()
    expected = CONTRACT.build_contract()["execution"]["environments"][
        "encoder_predictor"
    ]
    observed = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "scipy": scipy_version,
        "pillow": pillow_version,
        "pyyaml": importlib.metadata.version("PyYAML"),
    }
    expected_versions = {key: expected[key] for key in observed}
    if observed != expected_versions:
        raise InferenceError(
            f"GPU environment version drift: {observed!r} != {expected_versions!r}"
        )
    if not torch.cuda.is_available():
        raise InferenceError("frozen ROCm/CUDA-compatible GPU is unavailable")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    if str(properties.name) != "AMD Radeon AI PRO R9700":
        raise InferenceError(f"unexpected frozen GPU device: {properties.name}")
    entrypoint = Path(sys.executable).absolute()
    if entrypoint != Path(expected["interpreter"]).absolute():
        raise InferenceError("GPU interpreter entrypoint differs from frozen contract")
    interpreter_binding = _validated_interpreter_binary_binding(entrypoint)
    local_modules = {
        "scripts.dev_frozen_dense_representation_encoders_v1": E,
        "scripts.dev_proprio_predictor_v1": P,
        "scripts.run_dev_v03_temporal_action_jepa_v1": T,
        "scripts.build_dev_v03_proprio_action_manifest_v1": M,
        "scripts.dev_action_slew_reconstruction_v1": SLEW,
        "scripts.dev_checkpoint_v1": CK,
        "lewm.safety.jepa_local_waypoint_planning_cost_qualification_v1_contract": CONTRACT,
    }
    if tuple(local_modules) != GPU_LOCAL_SOURCE_MODULE_IDS:
        raise InferenceError("GPU transitive local import closure is incomplete")
    local_sources = {
        name: Path(module.__file__).resolve()
        for name, module in local_modules.items()
    }
    if "R9700" not in properties.name:
        raise InferenceError(f"unexpected GPU device {properties.name!r}")
    foundational_packages = _foundational_package_roots()
    value = attach_digest(
        {
            **phase_core(
                "jepa_local_waypoint_planning_cost_gpu_environment_v1",
                source_freeze_commit,
            ),
            "python": sys.version.split()[0],
            "executable": str(entrypoint),
            "interpreter_binding": interpreter_binding,
            "torch": torch.__version__,
            "device": {
                "type": "cuda",
                "index": 0,
                "name": properties.name,
                "total_memory_bytes": int(properties.total_memory),
                "hip": torch.version.hip,
            },
            "packages": {
                "numpy": np.__version__,
                "scipy": scipy_version,
                "pillow": pillow_version,
                "pyyaml": yaml.__version__,
            },
            "foundational_package_roots": foundational_packages,
            "foundational_package_closure_policy": copy.deepcopy(
                CONTRACT.FOUNDATIONAL_PACKAGE_CLOSURE_POLICY
            ),
            "import_closure": list(GPU_IMPORT_CLOSURE),
            "import_source_bindings": {
                name: {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "bytes": path.stat().st_size,
                }
                for name, path in local_sources.items()
            },
            "genesis_imported": False,
            "checkpoint_file_hashes": bindings,
            "encoder_source_repository_binding": encoder_source,
            "checkpoint_tensor_open_count": 0,
            "predictor_inference_calls": 0,
            "pass": True,
        }
    )
    if receipt.resolve().is_relative_to(ROOT.resolve()):
        raise InferenceError("GPU environment receipt must live on the output filesystem")
    atomic_json(receipt, value)
    return value


def _tensor_record(
    *,
    kind: str,
    state_id: str,
    family: str,
    role: str,
    candidate_index: int | None,
    horizon: int | None,
    source: str | None,
    path: Path,
    external: Any,
    output_root: Path,
) -> dict[str, Any]:
    array = np.load(path, allow_pickle=False)
    if array.shape != TENSOR_SHAPE or array.dtype != TENSOR_DTYPE:
        raise InferenceError(f"tensor payload contract mismatch: {path}")
    if not np.isfinite(array).all():
        raise InferenceError(f"tensor contains non-finite values: {path}")
    return {
        "kind": kind,
        "state_id": state_id,
        "family": family,
        "role": role,
        "candidate_index_or_null": candidate_index,
        "horizon_or_null": horizon,
        "source_or_null": source,
        "path": _artifact_reference(path, output_root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": [768, 1024],
        "dtype": "float16",
        "external_existing_artifact": external,
    }


def _numeric_state_key(state_id: str) -> tuple[str, int]:
    prefix, separator, suffix = state_id.rpartition("-")
    if not separator or not suffix.isdigit():
        raise InferenceError(f"state ID lacks numeric suffix: {state_id}")
    return prefix, int(suffix)


def _canonical_frame_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        str(row["rgb_sha256"]),
        str(row["kind"]),
        _numeric_state_key(str(row["state_id"])),
        int(row.get("slot", -1)),
    )


def _context_frame_map(
    mixed_frames: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, int], Mapping[str, Any]]:
    output = {
        (str(row["state_id"]), int(row["slot"])): row
        for row in mixed_frames
        if row.get("kind") == "CONTEXT"
    }
    if len(output) != 144 or {slot for (_state, slot) in output} != {0, 1, 2}:
        raise InferenceError("mixed frame inventory lacks exactly 144 context slots")
    if len({state for state, _slot in output}) != 48:
        raise InferenceError("mixed frame inventory lacks exactly 48 context states")
    return output


def _encode_frames(
    frames: list[dict[str, Any]],
    *,
    output_root: Path,
    source_freeze_commit: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Any, dict[str, Any]]:
    import torch
    from scripts.dev_frozen_dense_representation_encoders_v1 import (
        VJepa21CroppedV03Arm,
        preprocessing_hash,
    )

    if len(frames) != 192:
        raise InferenceError(f"encoder expected 192 context/goal frames, got {len(frames)}")
    # Exact canonical encoder order: RGB identity first, then semantic identity.
    frames.sort(key=_canonical_frame_key)
    if any(
        sha256_file(_artifact_path(row["rgb_path"], output_root))
        != row["rgb_sha256"]
        for row in frames
    ):
        raise InferenceError("encoder input RGB SHA mismatch")
    device = torch.device("cuda:0")
    arm = VJepa21CroppedV03Arm()
    if sha256_file(Path(arm.checkpoint)) != CONTRACT.ENCODER_SHA256:
        raise InferenceError("encoder checkpoint changed after preflight")
    arm.build(device, torch.float32)
    observed_preprocessing_digest = preprocessing_hash(arm)
    expected_preprocessing_digest = CONTRACT.build_contract()["latent_bindings"][
        "encoder"
    ]["preprocessing_digest"]
    if observed_preprocessing_digest != expected_preprocessing_digest:
        raise InferenceError("encoder preprocessing digest drift")
    if arm._module.training or any(  # noqa: SLF001 - source-closed encoder custody.
        parameter.requires_grad for parameter in arm._module.parameters()  # noqa: SLF001
    ):
        raise InferenceError("encoder did not enter frozen evaluation mode")
    parameter_digest_before = _parameter_state_digest(arm._module)  # noqa: SLF001
    records: list[dict[str, Any]] = []
    batches: list[dict[str, Any]] = []
    for offset in range(0, len(frames), ENCODER_BATCH):
        batch = frames[offset : offset + ENCODER_BATCH]
        if len(batch) != ENCODER_BATCH:
            raise InferenceError("192-frame encoder phase must have no partial batch")
        pixels = torch.stack(
            [
                arm.preprocess(str(_artifact_path(row["rgb_path"], output_root)))
                for row in batch
            ]
        ).to(device=device, dtype=torch.float32)
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            encoded = arm.tokens(pixels)
        values = encoded.float().cpu().numpy().astype(np.float16)
        if values.shape != (ENCODER_BATCH, *TENSOR_SHAPE):
            raise InferenceError("encoder output shape mismatch")
        batch_ids = []
        for row, value in zip(batch, values, strict=True):
            state_id = str(row["state_id"])
            if row["kind"] == "CONTEXT":
                slot = int(row["slot"])
                path = output_root / "latents/context" / state_id / f"slot_{slot}.npy"
                horizon = slot - 2
            else:
                path = output_root / "latents/goal" / f"{state_id}.npy"
                horizon = None
            atomic_npy(path, value)
            records.append(
                _tensor_record(
                    kind=str(row["kind"]),
                    state_id=state_id,
                    family=str(row["family"]),
                    role=str(row["role"]),
                    candidate_index=None,
                    horizon=horizon,
                    source=None,
                    path=path,
                    external=False,
                    output_root=output_root,
                )
            )
            row["token_path"] = _artifact_reference(path, output_root)
            row["token_sha256"] = sha256_file(path)
            batch_ids.append(
                {
                    "kind": row["kind"],
                    "state_id": state_id,
                    "slot": row.get("slot"),
                    "rgb_sha256": row["rgb_sha256"],
                }
            )
        batches.append(
            {
                "batch_index": offset // ENCODER_BATCH,
                "size": len(batch),
                "records": batch_ids,
            }
        )
    parameter_digest_after = _parameter_state_digest(arm._module)  # noqa: SLF001
    if parameter_digest_after != parameter_digest_before:
        raise InferenceError("encoder parameter state changed during inference")
    return records, batches, arm, {
        "parameter_state_digest_before": parameter_digest_before,
        "parameter_state_digest_after": parameter_digest_after,
        "parameter_state_unchanged": True,
        "eval_mode": True,
        "requires_grad_all_false": True,
        "preprocessing_digest": observed_preprocessing_digest,
    }


def _dense_current_authority() -> dict[str, dict[str, Any]]:
    if sha256_file(DENSE_INDEX) != (
        "3cf2d42f52525ce8291f76ee5af0bd58ef0928d62a8f879efb40a9ac6530cd15"
    ):
        raise InferenceError("dense current-token index binding drift")
    index = load_json(DENSE_INDEX)
    records = {str(row["rgb_sha256"]): row for row in index["records"]}
    output: dict[str, dict[str, Any]] = {}
    for occurrence in index["occurrences"]:
        if occurrence.get("kind") != "current":
            continue
        state_id = str(occurrence["state_id"])
        digest = str(occurrence["rgb_sha256"])
        if state_id in output or digest not in records:
            raise InferenceError("dense current-token authority is ambiguous")
        output[state_id] = {"rgb_sha256": digest, **records[digest]}
    if len(output) != STATE_COUNT:
        raise InferenceError("dense current-token authority does not cover all 48 states")
    return output


def _copy_current_and_validate(
    context_records: list[dict[str, Any]],
    tensor_records: list[dict[str, Any]],
    *,
    output_root: Path,
) -> list[dict[str, Any]]:
    authority = _dense_current_authority()
    output: list[dict[str, Any]] = []
    context_by_state_slot = _context_frame_map(context_records)
    metadata = {
        (str(row["state_id"]), int(row["horizon_or_null"])): row
        for row in tensor_records
        if row["kind"] == "CONTEXT"
    }
    for state_id in sorted(authority, key=_numeric_state_key):
        context = context_by_state_slot[(state_id, 2)]
        external = authority[state_id]
        if context["rgb_sha256"] != external["rgb_sha256"]:
            raise InferenceError(f"{state_id}: rerendered current RGB SHA differs exactly")
        current_context_path = _artifact_path(
            metadata[(state_id, 0)]["path"], output_root
        )
        observed = np.load(current_context_path, allow_pickle=False)
        external_path = Path(external["token_path"])
        if external_path.suffix == ".npy":
            expected = np.load(external_path, allow_pickle=False)
        else:
            expected = np.memmap(
                external_path, mode="r", dtype=np.float16, shape=TENSOR_SHAPE
            )
        if expected.shape != TENSOR_SHAPE or expected.dtype != TENSOR_DTYPE:
            raise InferenceError(f"{state_id}: current authority token contract mismatch")
        if not np.array_equal(observed, expected):
            raise InferenceError(f"{state_id}: re-encoded current token differs byte-exactly")
        observed_raw_sha = hashlib.sha256(
            np.ascontiguousarray(observed).tobytes(order="C")
        ).hexdigest()
        expected_raw_sha = str(external["token_sha256"])
        if observed_raw_sha != expected_raw_sha:
            raise InferenceError(
                f"{state_id}: re-encoded current raw FP16 payload SHA differs exactly"
            )
        # CURRENT is a logical alias of context slot at elapsed 0 s.  The
        # contract forbids duplicating the identical 48 payloads.
        path = current_context_path
        template = metadata[(state_id, 0)]
        output.append(
            _tensor_record(
                kind="CURRENT",
                state_id=state_id,
                family=str(template["family"]),
                role=str(template["role"]),
                candidate_index=None,
                horizon=None,
                source=None,
                path=path,
                external={
                    "path": str(external_path),
                    "sha256": str(external["token_sha256"]),
                    "rgb_sha256": str(external["rgb_sha256"]),
                    "byte_exact_array_equality": True,
                    "raw_payload_sha256_expected": expected_raw_sha,
                    "raw_payload_sha256_observed": observed_raw_sha,
                    "raw_payload_byte_exact": True,
                },
                output_root=output_root,
            )
        )
    return output


def _copy_true_future(
    context_by_state: Mapping[str, Mapping[str, Any]], *, output_root: Path
) -> list[dict[str, Any]]:
    expected_sha = "df5e55b6606b0a914603ec99db9f91d1898bfd460e0b83cbd33abb0772da4874"
    if sha256_file(TARGET_INDEX) != expected_sha:
        raise InferenceError("true-future target index binding drift")
    index = load_json(TARGET_INDEX)
    entries = index.get("entries")
    if not isinstance(entries, list) or len(entries) != 1728:
        raise InferenceError("true-future target index cardinality mismatch")
    entries = sorted(
        entries,
        key=lambda row: (
            _numeric_state_key(str(row["state_id"])),
            int(row["candidate_index"]),
            int(row["horizon"]),
        ),
    )
    output: list[dict[str, Any]] = []
    identities: set[tuple[str, int, int]] = set()
    for row in entries:
        state_id = str(row["state_id"])
        candidate = int(row["candidate_index"])
        horizon = int(row["horizon"])
        identity = (state_id, candidate, horizon)
        if identity in identities or state_id not in context_by_state:
            raise InferenceError(f"invalid true-future identity {identity}")
        identities.add(identity)
        source_path = Path(row["latent_path"])
        if not source_path.is_file() or sha256_file(source_path) != row["sha256"]:
            raise InferenceError(f"true-future tensor drift {identity}")
        value = np.load(source_path, allow_pickle=False)
        if value.shape != TENSOR_SHAPE or value.dtype != TENSOR_DTYPE:
            raise InferenceError(f"true-future tensor contract mismatch {identity}")
        target = (
            output_root
            / "latents/true_future"
            / state_id
            / f"candidate_{candidate:02d}_h{horizon}.npy"
        )
        atomic_npy(target, value)
        context = context_by_state[state_id]
        output.append(
            _tensor_record(
                kind="TRUE_FUTURE",
                state_id=state_id,
                family=str(context["family"]),
                role=str(context["role"]),
                candidate_index=candidate,
                horizon=horizon,
                source="TRUE_FUTURE",
                path=target,
                external={
                    "path": str(source_path),
                    "sha256": str(row["sha256"]),
                    "index_sha256": expected_sha,
                    "array_equality": True,
                },
                output_root=output_root,
            )
        )
    if len(identities) != 1728:
        raise InferenceError("true-future identities are incomplete")
    return output


def _load_predictor(checkpoint_name: str, device: Any) -> tuple[Any, dict[str, Any]]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P

    binding = CONTRACT.build_contract()["predictor_bindings"][checkpoint_name]
    path = Path(binding["path"])
    if sha256_file(path) != binding["sha256"]:
        raise InferenceError(f"{checkpoint_name} checkpoint drift before open")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    expected_cell = "rgb_one_step" if checkpoint_name == "one_step" else "rgb_rollout"
    expected_rollout = checkpoint_name == "two_step"
    model_config = checkpoint.get("model_config")
    required = {
        "cell": expected_cell,
        "use_proprio": False,
        "rollout": expected_rollout,
        "width": 384,
    }
    if model_config != required:
        raise InferenceError(
            f"{checkpoint_name} model config {model_config!r} != {required!r}"
        )
    model = P.build_paired(
        CONTRACT.SEED, use_proprio=False, width=384, depth=6, heads=6
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    if model.training or any(parameter.requires_grad for parameter in model.parameters()):
        raise InferenceError(f"{checkpoint_name} did not enter frozen evaluation mode")
    del checkpoint
    return model, {
        "checkpoint_name": checkpoint_name,
        "model_config": required,
        "strict_state_dict_load": True,
        "eval_mode": True,
        "requires_grad_all_false": True,
        "parameter_state_digest_before": _parameter_state_digest(model),
    }


def _predict_source(
    checkpoint_name: str,
    context_by_state: Mapping[str, Mapping[str, Any]],
    context_tensor_records: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, dict[str, Any]]:
    import torch
    from scripts import dev_proprio_predictor_v1 as P
    from scripts import run_dev_v03_temporal_action_jepa_v1 as T

    source = "ONE_STEP_PREDICTED" if checkpoint_name == "one_step" else "TWO_STEP_PREDICTED"
    device = torch.device("cuda:0")
    model, custody = _load_predictor(checkpoint_name, device)
    by_state_slot = {
        (str(row["state_id"]), int(row["horizon_or_null"])): row
        for row in context_tensor_records
        if row["kind"] == "CONTEXT"
    }
    output: list[dict[str, Any]] = []
    calls: list[dict[str, Any]] = []
    call_count = 0
    for state_index, state_id in enumerate(sorted(context_by_state, key=_numeric_state_key)):
        state = context_by_state[state_id]
        raw_context = np.stack(
            [
                np.load(
                    _artifact_path(
                        by_state_slot[(state_id, horizon)]["path"], output_root
                    ),
                    allow_pickle=False,
                )
                for horizon in (-2, -1, 0)
            ],
            axis=0,
        )
        if raw_context.shape != (3, *TENSOR_SHAPE) or raw_context.dtype != np.float16:
            raise InferenceError(f"{state_id}: predictor context payload mismatch")
        # Required normalization order: persisted FP16 -> FP32 -> T.normalise;
        # model execution then follows the frozen FP32-weights/BF16-autocast path.
        context = torch.from_numpy(raw_context).to(device=device, dtype=torch.float32)
        context = T.normalise(context).unsqueeze(0).repeat(PREDICTOR_BATCH, 1, 1, 1)
        actions = np.asarray(
            state["action_blocks_raw_3x10_by_candidate"], dtype=np.float32
        )
        if actions.shape != (12, 3, 10):
            raise InferenceError(f"{state_id}: predictor action shape mismatch")
        action_blocks = [
            torch.from_numpy(actions[:, horizon]).to(device=device, dtype=torch.float32)
            for horizon in range(3)
        ]
        control_array = np.asarray(
            state["control_history_normalized_3x5x2"], dtype=np.float32
        )
        if control_array.shape != (3, 5, 2):
            raise InferenceError(f"{state_id}: predictor control shape mismatch")
        control = (
            torch.from_numpy(control_array)
            .to(device=device, dtype=torch.float32)
            .unsqueeze(0)
            .repeat(PREDICTOR_BATCH, 1, 1, 1)
        )
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16, enabled=True
        ):
            predicted = P.unroll(
                model,
                context,
                action_blocks,
                control=control,
                max_h=3,
            )
        if len(predicted) != 3:
            raise InferenceError(f"{state_id}: predictor returned wrong horizon count")
        values = [tensor.float().cpu().numpy().astype(np.float16) for tensor in predicted]
        for horizon, tensor in enumerate(values, 1):
            if tensor.shape != (12, *TENSOR_SHAPE):
                raise InferenceError(f"{state_id}: prediction tensor shape mismatch")
            for candidate in range(12):
                path = (
                    output_root
                    / "latents/predicted"
                    / source.lower()
                    / state_id
                    / f"candidate_{candidate:02d}_h{horizon}.npy"
                )
                atomic_npy(path, tensor[candidate])
                output.append(
                    _tensor_record(
                        kind=source,
                        state_id=state_id,
                        family=str(state["family"]),
                        role=str(state["role"]),
                        candidate_index=candidate,
                        horizon=horizon,
                        source=source,
                        path=path,
                        external=False,
                        output_root=output_root,
                    )
                )
        calls.append(
            {
                "call_index": state_index,
                "state_id": state_id,
                "batch_size": 12,
                "horizons": [1, 2, 3],
                "checkpoint": checkpoint_name,
                "source": source,
            }
        )
        call_count += 1
    after = _parameter_state_digest(model)
    custody["parameter_state_digest_after"] = after
    custody["parameter_state_unchanged"] = (
        after == custody["parameter_state_digest_before"]
    )
    if not custody["parameter_state_unchanged"]:
        raise InferenceError(f"{checkpoint_name}: model parameter state changed")
    custody["unroll_calls"] = call_count
    custody["model_forward_calls"] = call_count * 3
    custody["candidate_horizon_outputs"] = len(output)
    del model
    torch.cuda.empty_cache()
    if call_count != 48 or len(output) != 1728:
        raise InferenceError(f"{source}: predictor call/tensor cardinality mismatch")
    return output, calls, call_count, custody


def _update_goal_index(
    goal_index: dict[str, Any], frame_rows: Sequence[Mapping[str, Any]], output_root: Path
) -> dict[str, Any]:
    tokens = {
        str(row["state_id"]): row for row in frame_rows if row["kind"] == "GOAL"
    }
    if len(tokens) != 48:
        raise InferenceError("goal token map is incomplete")
    for record in goal_index["records"]:
        token = tokens[str(record["state_id"])]
        record["token_path"] = token["token_path"]
        record["token_sha256"] = token["token_sha256"]
        record["token_shape"] = [768, 1024]
    updated = attach_digest(goal_index)
    atomic_json(output_root / GOAL_INDEX_REL, updated)
    return updated


def _update_context_index(
    context_index: dict[str, Any],
    current_records: Sequence[Mapping[str, Any]],
    *,
    output_root: Path,
) -> dict[str, Any]:
    if len(current_records) != 48:
        raise InferenceError("current authority audit requires exactly 48 records")
    audit_rows = []
    for row in sorted(current_records, key=lambda item: _numeric_state_key(str(item["state_id"]))):
        external = row.get("external_existing_artifact")
        if not isinstance(external, Mapping):
            raise InferenceError("current record lacks external authority binding")
        audit_rows.append(
            {
                "state_id": row["state_id"],
                "rgb_sha256": external["rgb_sha256"],
                "external_token_path": external["path"],
                "external_token_sha256": external["sha256"],
                "persisted_current_path": row["path"],
                "persisted_current_sha256": row["sha256"],
                "rgb_exact": True,
                "token_array_byte_exact": True,
                "raw_payload_sha256_expected": external[
                    "raw_payload_sha256_expected"
                ],
                "raw_payload_sha256_observed": external[
                    "raw_payload_sha256_observed"
                ],
                "raw_payload_byte_exact": external["raw_payload_byte_exact"],
            }
        )
    context_index["current_rgb_authority_reproduction"] = {
        "status": "PASS",
        "states_expected": 48,
        "states_exact": 48,
        "mismatches": 0,
        "records": audit_rows,
    }
    context_index["current_token_authority_reproduction"] = {
        "status": "PASS",
        "states_expected": 48,
        "states_exact": 48,
        "mismatches": 0,
        "array_equality": "exact float16 payload equality",
    }
    updated = attach_digest(context_index)
    atomic_json(output_root / CONTEXT_INDEX_REL, updated)
    return updated


def materialize(source_freeze_commit: str, *, output_root: Path) -> dict[str, Any]:
    """Encode 192 frames, copy true targets, then run both frozen predictors."""

    validate_source_commit(source_freeze_commit, require_live_head=True)
    source_closure = validate_frozen_source_closure()
    gpu_environment_binding = _gpu_environment_binding_for_inference(
        source_freeze_commit, output_root=output_root
    )
    live_gpu_environment = _live_gpu_environment_identity()
    bindings = _checkpoint_bindings()
    encoder_source = _encoder_source_repository_binding()
    context_index = load_json(output_root / CONTEXT_INDEX_REL)
    goal_index = load_json(output_root / GOAL_INDEX_REL)
    validate_phase(context_index, source_freeze_commit)
    validate_phase(goal_index, source_freeze_commit)
    if len(context_index.get("records", [])) != 48 or len(goal_index.get("records", [])) != 48:
        raise InferenceError("CPU context/goal index cardinality mismatch")
    context_by_state = {
        str(row["state_id"]): row for row in context_index["records"]
    }
    if len(context_by_state) != 48:
        raise InferenceError("CPU context state identities are incomplete")
    frames: list[dict[str, Any]] = []
    for state in context_index["records"]:
        for slot, (path, digest) in enumerate(
            zip(
                state["context_rgb_paths"],
                state["context_rgb_sha256s"],
                strict=True,
            )
        ):
            frames.append(
                {
                    "kind": "CONTEXT",
                    "state_id": state["state_id"],
                    "family": state["family"],
                    "role": state["role"],
                    "slot": slot,
                    "rgb_path": path,
                    "rgb_sha256": digest,
                }
            )
    for state in goal_index["records"]:
        frames.append(
            {
                "kind": "GOAL",
                "state_id": state["state_id"],
                "family": state["family"],
                "role": state["role"],
                "rgb_path": state["rgb_path"],
                "rgb_sha256": state["rgb_sha256"],
            }
        )
    started = time.time()
    import torch

    torch.cuda.reset_peak_memory_stats(torch.device("cuda:0"))
    tensor_records, encoder_batches, _arm, encoder_custody = _encode_frames(
        frames, output_root=output_root, source_freeze_commit=source_freeze_commit
    )
    current_records = _copy_current_and_validate(
        frames, tensor_records, output_root=output_root
    )
    tensor_records.extend(current_records)
    updated_context = _update_context_index(
        context_index, current_records, output_root=output_root
    )
    true_records = _copy_true_future(context_by_state, output_root=output_root)
    tensor_records.extend(true_records)
    one_records, one_calls, one_count, one_custody = _predict_source(
        "one_step", context_by_state, tensor_records, output_root=output_root
    )
    tensor_records.extend(one_records)
    two_records, two_calls, two_count, two_custody = _predict_source(
        "two_step", context_by_state, tensor_records, output_root=output_root
    )
    tensor_records.extend(two_records)
    if len(tensor_records) != 5424:
        raise InferenceError(f"latent record count {len(tensor_records)} != 5424")
    counts = {
        kind: sum(row["kind"] == kind for row in tensor_records)
        for kind in (
            "CONTEXT",
            "CURRENT",
            "GOAL",
            "TRUE_FUTURE",
            "ONE_STEP_PREDICTED",
            "TWO_STEP_PREDICTED",
        )
    }
    expected_counts = {
        "CONTEXT": 144,
        "CURRENT": 48,
        "GOAL": 48,
        "TRUE_FUTURE": 1728,
        "ONE_STEP_PREDICTED": 1728,
        "TWO_STEP_PREDICTED": 1728,
    }
    if counts != expected_counts:
        raise InferenceError(f"latent kind counts {counts} != {expected_counts}")
    tensor_records.sort(
        key=lambda row: (
            str(row["kind"]),
            _numeric_state_key(str(row["state_id"])),
            -1 if row["candidate_index_or_null"] is None else int(row["candidate_index_or_null"]),
            -99 if row["horizon_or_null"] is None else int(row["horizon_or_null"]),
        )
    )
    unique_payloads = {
        (str(row["path"]), str(row["sha256"])): int(row["bytes"])
        for row in tensor_records
    }
    batch_manifest = attach_digest(
        {
            **phase_core(
                "jepa_local_waypoint_planning_cost_gpu_batch_manifest_v1",
                source_freeze_commit,
            ),
            "encoder": {
                "order": "ascending rgb_sha256, kind, numeric state, slot",
                "batch_size": 16,
                "padding": False,
                "dynamic_fallback": False,
                "calls": len(encoder_batches),
                "batches": encoder_batches,
            },
            "predictor": {
                "order": ["one_step checkpoint all 48 states", "two_step checkpoint all 48 states"],
                "state_order": "numeric frozen state order",
                "batch_size": 12,
                "horizons": [1, 2, 3],
                "dynamic_fallback": False,
                "one_step_calls": one_count,
                "two_step_calls": two_count,
                "calls": [*one_calls, *two_calls],
            },
            "checkpoint_tensor_open_count": 3,
            "training_steps": 0,
        }
    )
    atomic_json(output_root / BATCH_MANIFEST_REL, batch_manifest)
    updated_goal = _update_goal_index(goal_index, frames, output_root)
    index = attach_digest(
        {
            **phase_core(
                "jepa_local_waypoint_latent_tensor_index_v1", source_freeze_commit
            ),
            "records": tensor_records,
            "counts_by_kind": counts,
            "total_records": len(tensor_records),
            "total_bytes": sum(unique_payloads.values()),
            "unique_tensor_payload_count": len(unique_payloads),
            "external_true_future_index_binding": {
                "path": str(TARGET_INDEX),
                "sha256": sha256_file(TARGET_INDEX),
                "count": 1728,
            },
            "external_current_index_binding": {
                "path": str(DENSE_INDEX),
                "sha256": sha256_file(DENSE_INDEX),
                "states": 48,
                "rgb_and_token_array_exact_equality": True,
            },
            "encoder_binding": bindings["encoder"],
            "checkpoint_bindings": {
                "one_step": bindings["one_step"],
                "two_step": bindings["two_step"],
            },
            "batch_manifest": {
                "path": str(BATCH_MANIFEST_REL),
                "sha256": sha256_file(output_root / BATCH_MANIFEST_REL),
                "content_digest": batch_manifest["content_digest"],
            },
            "shape_dtype_validation": {
                "records": 5424,
                "passed": 5424,
                "shape": [768, 1024],
                "dtype": "float16",
            },
            "failed_records": [],
            "runtime_s": time.time() - started,
            "peak_vram_bytes": int(torch.cuda.max_memory_allocated()),
            "checkpoint_tensor_open_count": 3,
            "predictor_inference_calls": 96,
            "predictor_model_forward_calls": 288,
            "encoder_inference_calls": len(encoder_batches),
            "training_steps": 0,
            "goal_index_content_digest": updated_goal["content_digest"],
            "context_index_content_digest": updated_context["content_digest"],
        }
    )
    atomic_json(output_root / LATENT_INDEX_REL, index)
    gpu_receipt = attach_digest(
        {
            **phase_core(
                "jepa_local_waypoint_planning_cost_gpu_inference_v1",
                source_freeze_commit,
            ),
            "interpreter_entrypoint": live_gpu_environment["executable"],
            "interpreter": copy.deepcopy(
                live_gpu_environment["interpreter_binding"]
            ),
            "environment": {
                "python": live_gpu_environment["python"],
                "torch": live_gpu_environment["torch"],
                **live_gpu_environment["packages"],
            },
            "device": {
                "requested": "cuda:0",
                "resolved": str(torch.device("cuda:0")),
                **live_gpu_environment["device"],
            },
            "autocast": {
                "device_type": "cuda",
                "dtype": "bfloat16",
                "enabled": True,
                "weights_dtype": "float32",
                "persisted_dtype": "float16",
            },
            "model_config": {
                "one_step": one_custody["model_config"],
                "two_step": two_custody["model_config"],
            },
            "checkpoint_bindings": bindings,
            "encoder_source_repository_binding": encoder_source,
            "source_closure_binding": {
                "path": CONTRACT.TRACKED_SOURCE_CLOSURE_PATH,
                "sha256": sha256_file(ROOT / CONTRACT.TRACKED_SOURCE_CLOSURE_PATH),
                "content_digest": source_closure["content_digest"],
                "complete": True,
            },
            "gpu_environment_receipt_binding": gpu_environment_binding,
            "gpu_environment_revalidation": {
                "interpreter_exact": True,
                "versions_exact": True,
                "device_exact": True,
                "foundational_roots_exact": True,
                "precheckpoint": True,
                "pass": True,
            },
            "gpu_watchdog_status": copy.deepcopy(
                CONTRACT.GPU_WATCHDOG_STATUS_SUCCESS
            ),
            "strict_state_dict_load": {
                "encoder": True,
                "one_step": True,
                "two_step": True,
            },
            "eval_mode": {"encoder": True, "one_step": True, "two_step": True},
            "inference_mode": True,
            "requires_grad_all_false": {
                "encoder": True,
                "one_step": True,
                "two_step": True,
            },
            "optimizer_absent": True,
            "training_steps": 0,
            "parameter_state_digest_before": {
                "encoder": encoder_custody["parameter_state_digest_before"],
                "one_step": one_custody["parameter_state_digest_before"],
                "two_step": two_custody["parameter_state_digest_before"],
            },
            "parameter_state_digest_after": {
                "encoder": encoder_custody["parameter_state_digest_after"],
                "one_step": one_custody["parameter_state_digest_after"],
                "two_step": two_custody["parameter_state_digest_after"],
            },
            "parameter_state_unchanged": {
                "encoder": encoder_custody["parameter_state_unchanged"],
                "one_step": one_custody["parameter_state_unchanged"],
                "two_step": two_custody["parameter_state_unchanged"],
            },
            "encoder_call_counts": {
                "frames": 192,
                "batch_size": 16,
                "batches": len(encoder_batches),
                "preprocessing_digest": encoder_custody["preprocessing_digest"],
            },
            "preprocessing_digest": {
                "expected": CONTRACT.build_contract()["latent_bindings"]["encoder"][
                    "preprocessing_digest"
                ],
                "observed": encoder_custody["preprocessing_digest"],
                "match": True,
            },
            "predictor_call_counts_by_source": {
                "ONE_STEP_PREDICTED": {
                    "unroll_calls": one_count,
                    "model_forward_calls": one_custody["model_forward_calls"],
                    "candidate_horizon_outputs": len(one_records),
                },
                "TWO_STEP_PREDICTED": {
                    "unroll_calls": two_count,
                    "model_forward_calls": two_custody["model_forward_calls"],
                    "candidate_horizon_outputs": len(two_records),
                },
            },
            "batch_manifest": {
                "path": str(BATCH_MANIFEST_REL),
                "sha256": sha256_file(output_root / BATCH_MANIFEST_REL),
                "content_digest": batch_manifest["content_digest"],
            },
            "batch_order_validation": {
                "encoder": True,
                "one_step_all_states_before_two_step": True,
                "numeric_state_order": True,
                "no_padding": True,
            },
            "dynamic_oom_batch_fallback_used": False,
            "future_input_fields": [],
            "latent_tensor_index_binding": {
                "path": str(LATENT_INDEX_REL),
                "sha256": sha256_file(output_root / LATENT_INDEX_REL),
                "bytes": (output_root / LATENT_INDEX_REL).stat().st_size,
                "content_digest": index["content_digest"],
            },
            "pass": True,
        }
    )
    atomic_json(output_root / GPU_INFERENCE_REL, gpu_receipt)
    _assert_gpu_process_separation()
    return {
        "pass": True,
        "source_freeze_commit": source_freeze_commit,
        "tensor_records": 5424,
        "encoder_calls": len(encoder_batches),
        "predictor_calls": 96,
        "content_digest": index["content_digest"],
    }


def check(source_freeze_commit: str, *, output_root: Path, deep: bool) -> dict[str, Any]:
    validate_source_commit(source_freeze_commit)
    value = load_json(output_root / LATENT_INDEX_REL)
    validate_phase(value, source_freeze_commit)
    if int(value.get("total_records", -1)) != 5424:
        raise InferenceError("latent index total is not 5424")
    identities: set[tuple[Any, ...]] = set()
    for row in value["records"]:
        identity = (
            row["kind"],
            row["state_id"],
            row["candidate_index_or_null"],
            row["horizon_or_null"],
        )
        if identity in identities:
            raise InferenceError(f"duplicate tensor identity {identity}")
        identities.add(identity)
        path = _artifact_path(row["path"], output_root)
        if not path.is_file() or path.stat().st_size != int(row["bytes"]):
            raise InferenceError(f"tensor byte drift {identity}")
        if deep and sha256_file(path) != row["sha256"]:
            raise InferenceError(f"tensor SHA drift {identity}")
    gpu_receipt = load_json(output_root / GPU_INFERENCE_REL)
    validate_phase(gpu_receipt, source_freeze_commit)
    gpu_environment_binding = _gpu_environment_binding_for_inference(
        source_freeze_commit, output_root=output_root
    )
    live_checkpoint_bindings = _checkpoint_bindings()
    gpu_environment = load_json(output_root / GPU_ENVIRONMENT_REL)
    if (
        gpu_receipt.get("checkpoint_bindings") != live_checkpoint_bindings
        or gpu_environment.get("checkpoint_file_hashes")
        != live_checkpoint_bindings
        or value.get("encoder_binding") != live_checkpoint_bindings["encoder"]
        or value.get("checkpoint_bindings")
        != {
            "one_step": live_checkpoint_bindings["one_step"],
            "two_step": live_checkpoint_bindings["two_step"],
        }
    ):
        raise InferenceError("terminal checkpoint file binding drift")
    if gpu_receipt.get("gpu_environment_receipt_binding") != gpu_environment_binding:
        raise InferenceError("GPU inference environment-receipt binding drift")
    expected_entrypoint = CONTRACT.build_contract()["execution"]["environments"][
        "encoder_predictor"
    ]["interpreter"]
    if (
        gpu_receipt.get("interpreter_entrypoint") != expected_entrypoint
        or gpu_receipt.get("interpreter") != CONTRACT.INTERPRETER_BINARY_BINDING
    ):
        raise InferenceError("GPU inference interpreter custody drift")
    if gpu_receipt.get("gpu_environment_revalidation") != {
        "interpreter_exact": True,
        "versions_exact": True,
        "device_exact": True,
        "foundational_roots_exact": True,
        "precheckpoint": True,
        "pass": True,
    }:
        raise InferenceError("GPU inference environment revalidation drift")
    if gpu_receipt.get("gpu_watchdog_status") != (
        CONTRACT.GPU_WATCHDOG_STATUS_SUCCESS
    ):
        raise InferenceError("GPU inference watchdog receipt drift")
    if gpu_receipt.get("training_steps") != 0:
        raise InferenceError("GPU inference receipt reports training")
    if gpu_receipt.get("parameter_state_unchanged") != {
        "encoder": True,
        "one_step": True,
        "two_step": True,
    }:
        raise InferenceError("GPU parameter-state custody is not unchanged")
    latent_binding = gpu_receipt.get("latent_tensor_index_binding", {})
    if latent_binding.get("sha256") != sha256_file(output_root / LATENT_INDEX_REL):
        raise InferenceError("GPU receipt latent-index binding drift")
    if gpu_receipt.get(
        "encoder_source_repository_binding"
    ) != _encoder_source_repository_binding():
        raise InferenceError("GPU encoder source repository receipt drift")
    _assert_gpu_process_separation()
    return {
        "pass": True,
        "source_freeze_commit": source_freeze_commit,
        "tensor_records": len(identities),
        "checkpoint_opened_during_check": 0,
        "predictor_inference_calls_during_check": 0,
        "genesis_imported": False,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    pre = subparsers.add_parser("preflight")
    pre.add_argument("--source-freeze-commit", required=True)
    pre.add_argument("--output-root", type=Path, required=True)
    pre.add_argument("--receipt", type=Path, required=True)
    materialize_parser = subparsers.add_parser("materialize")
    materialize_parser.add_argument("--source-freeze-commit", required=True)
    materialize_parser.add_argument("--output-root", type=Path, required=True)
    terminal = subparsers.add_parser("check")
    terminal.add_argument("--source-freeze-commit", required=True)
    terminal.add_argument("--output-root", type=Path, required=True)
    terminal.add_argument("--shallow", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "preflight":
        value = preflight(
            args.source_freeze_commit,
            output_root=args.output_root,
            receipt=args.receipt,
        )
    elif args.command == "materialize":
        value = materialize(args.source_freeze_commit, output_root=args.output_root)
    elif args.command == "check":
        value = check(
            args.source_freeze_commit,
            output_root=args.output_root,
            deep=not args.shallow,
        )
    else:  # pragma: no cover
        raise InferenceError(f"unknown command {args.command}")
    print(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
