#!/usr/bin/env python3
"""Run the one authorized RGB multiresolution perception probe.

Importing this module is source-only: Torch, image decoders, generated inputs,
and tensor checkpoints are deferred until exact authority validation and a
mode-0700 attempt reservation have both succeeded.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import hashlib
import importlib.util
import io
import math
import os
from pathlib import Path
import stat
import sys
from types import SimpleNamespace
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
_CONTRACT_PATH = (
    ROOT / "lewm/benchmarks/go2_shared_jepa_v5_multires_probe_v1.py"
)
_CONTRACT_SPEC = importlib.util.spec_from_file_location(
    "_lewm_go2_shared_jepa_v5_multires_probe_v1_contract",
    _CONTRACT_PATH,
)
if _CONTRACT_SPEC is None or _CONTRACT_SPEC.loader is None:
    raise ImportError("cannot load multires probe contract")
contract = importlib.util.module_from_spec(_CONTRACT_SPEC)
_CONTRACT_SPEC.loader.exec_module(contract)

PREFLIGHT_ENVIRONMENT_KEY = "LEWM_MULTIRES_PROBE_PREFLIGHT_JSON"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
CONFLICTING_ACCELERATOR_ENVIRONMENT = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
)


def _fingerprint(value: os.stat_result) -> tuple[int, ...]:
    return (
        int(value.st_dev),
        int(value.st_ino),
        int(value.st_mode),
        int(value.st_size),
        int(value.st_mtime_ns),
        int(value.st_ctime_ns),
    )


def _read_regular(path: Path, *, expected_sha256: str | None = None) -> bytes:
    if path.is_symlink():
        raise PermissionError(f"symlink input forbidden: {path}")
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise PermissionError(f"input is not regular: {path}")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        chunks: list[bytes] = []
        while chunk := os.read(descriptor, 1024 * 1024):
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _fingerprint(before) != _fingerprint(after):
        raise RuntimeError(f"input changed while read: {path}")
    raw = b"".join(chunks)
    if (
        expected_sha256 is not None
        and hashlib.sha256(raw).hexdigest() != expected_sha256
    ):
        raise PermissionError(f"input hash changed: {path}")
    return raw


def _write_exclusive(path: Path, raw: bytes, *, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        mode,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(descriptor, mode)
    finally:
        os.close(descriptor)
    directory = os.open(
        path.parent,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _publish_json(
    path: Path,
    core: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    value = contract.with_content_sha256(dict(core))
    raw = contract.canonical_json_bytes(value) + b"\n"
    _write_exclusive(path, raw)
    return value, raw


def _binding(
    relative: str,
    value: Mapping[str, Any],
    raw: bytes,
) -> dict[str, Any]:
    return contract.artifact_binding(
        relative,
        raw,
        content_sha256=str(value["content_sha256"]),
    )


def _publish_readonly_atomic(path: Path, raw: bytes) -> None:
    """Publish complete immutable bytes without overwriting any prior path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.publishing")
    _write_exclusive(temporary, raw, mode=0o444)
    try:
        os.link(
            temporary,
            path,
            src_dir_fd=None,
            dst_dir_fd=None,
            follow_symlinks=False,
        )
        os.unlink(temporary)
        directory = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except BaseException:
        # Leave a mode-0444 publication remnant as evidence if final linking
        # fails.  The enclosing attempt is terminal and cannot be retried.
        raise
    if stat.S_IMODE(path.stat(follow_symlinks=False).st_mode) != 0o444:
        raise PermissionError("atomic sidecar did not publish mode 0444")


def _load_authority_pre_reservation(
    review_sha256: str,
    authorization_sha256: str,
) -> tuple[
    dict[str, Any],
    bytes,
    dict[str, Any],
    bytes,
    dict[str, str],
]:
    sources = contract.current_source_bindings(ROOT)
    review_raw = _read_regular(
        ROOT / contract.REVIEW_RELATIVE_PATH,
        expected_sha256=review_sha256,
    )
    review = contract.validate_review(
        contract.parse_canonical_json(review_raw, name="source review"),
        expected_sources=sources,
    )
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_raw = _read_regular(
        ROOT / contract.AUTHORIZATION_RELATIVE_PATH,
        expected_sha256=authorization_sha256,
    )
    authorization = contract.validate_authorization(
        contract.parse_canonical_json(
            authorization_raw, name="execution authorization"
        ),
        review_binding=review_binding,
        reviewer=str(review["reviewer"]),
    )
    return review, review_raw, authorization, authorization_raw, sources


def _source_authority_receipt(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "source_binding_count": len(sources),
        "source_bindings_sha256": contract.canonical_json_sha256(sources),
        "source_review": contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=str(review["content_sha256"]),
        ),
        "execution_authorization": contract.artifact_binding(
            contract.AUTHORIZATION_RELATIVE_PATH,
            authorization_raw,
            content_sha256=str(authorization["content_sha256"]),
        ),
        "generated_runtime_input_open_count": 0,
        "torch_imported": False,
    }


def _validate_preflight(
    *,
    expected_sha256: str,
    launcher_source_sha256: str,
    expected_source_authority: Mapping[str, Any],
) -> dict[str, Any]:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("probe runner requires python -I -B")
    if "torch" in sys.modules or any(name.startswith("torch.") for name in sys.modules):
        raise PermissionError("Torch was imported before attempt reservation")
    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("probe runner requires HIP_VISIBLE_DEVICES=0")
    conflicting = [
        name for name in CONFLICTING_ACCELERATOR_ENVIRONMENT if name in os.environ
    ]
    threads = {name: os.environ.get(name) for name in THREAD_ENVIRONMENT}
    if conflicting or any(value != "1" for value in threads.values()):
        raise PermissionError("accelerator or native-thread environment changed")
    encoded = os.environ.get(PREFLIGHT_ENVIRONMENT_KEY)
    if type(encoded) is not str:
        raise PermissionError("isolated no-tensor preflight receipt is absent")
    try:
        raw = encoded.encode("ascii") + b"\n"
    except UnicodeEncodeError as error:
        raise PermissionError("preflight receipt is not ASCII") from error
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PermissionError("preflight receipt file hash changed")
    value = contract.parse_canonical_json(raw, name="hardware preflight receipt")
    fields = {
        "schema",
        "status",
        "launcher_process_id",
        "source_authority",
        "preflight_child_process_id",
        "visible_device_count",
        "visible_device_index",
        "visible_device_name",
        "total_memory_bytes",
        "torch_version",
        "hip_version",
        "tensor_allocation_count",
        "payload_open_count",
        "torch_device_api_call_count",
        "launcher_source_sha256",
        "immediate_exec_required",
        "intervening_gpu_query_count",
        "content_sha256",
    }
    name = value.get("visible_device_name")
    if (
        set(value) != fields
        or value["schema"] != f"{contract.SCHEMA_PREFIX}_hardware_preflight_v1"
        or value["status"] != "PASS_EXACTLY_ONE_VISIBLE_DISCRETE_R9700"
        or value["launcher_process_id"] != os.getpid()
        or value["source_authority"] != dict(expected_source_authority)
        or type(value["preflight_child_process_id"]) is not int
        or value["preflight_child_process_id"] <= 0
        or value["visible_device_count"] != 1
        or value["visible_device_index"] != 0
        or type(name) is not str
        or "r9700" not in name.casefold().replace(" ", "")
        or type(value["total_memory_bytes"]) is not int
        or value["total_memory_bytes"] < 32_000_000_000
        or type(value["torch_version"]) is not str
        or not value["torch_version"]
        or type(value["hip_version"]) is not str
        or not value["hip_version"]
        or value["tensor_allocation_count"] != 0
        or value["payload_open_count"] != 0
        or value["torch_device_api_call_count"] != 3
        or value["launcher_source_sha256"] != launcher_source_sha256
        or value["immediate_exec_required"] is not True
        or value["intervening_gpu_query_count"] != 0
    ):
        raise PermissionError("hardware preflight receipt changed")
    return value


def _reserve(
    output_root: Path,
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    preflight: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    if output_root.exists() or output_root.is_symlink():
        raise RuntimeError("the sole multires probe attempt is already consumed")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    review_binding = contract.artifact_binding(
        contract.REVIEW_RELATIVE_PATH,
        review_raw,
        content_sha256=str(review["content_sha256"]),
    )
    authorization_binding = contract.artifact_binding(
        contract.AUTHORIZATION_RELATIVE_PATH,
        authorization_raw,
        content_sha256=str(authorization["content_sha256"]),
    )
    attempt_identity = contract.canonical_json_sha256({
        "schema": f"{contract.SCHEMA_PREFIX}_attempt_identity_v1",
        "review": review_binding,
        "authorization": authorization_binding,
        "science_contract_sha256":
            contract.canonical_json_sha256(contract.science_contract()),
    })
    reservation_core = {
        "schema": contract.RESERVATION_SCHEMA,
        "status": "RESERVED_0700_BEFORE_TORCH_OR_RUNTIME_INPUTS",
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt_identity,
        "independent_source_review": review_binding,
        "execution_authorization": authorization_binding,
        "reviewed_sources": dict(sources),
        "preflight": dict(preflight),
        "science_contract": contract.science_contract(),
        "lifecycle_contract": contract.lifecycle_contract(),
        "output_root_absent_before_reservation": True,
        "output_root_mode": "0700",
        "torch_imported_before_reservation": False,
        "runtime_input_opened_before_reservation": False,
        "reservation_consumes_attempt": True,
        "retry_authorized": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    os.mkdir(output_root, mode=0o700)
    try:
        if (
            stat.S_IMODE(output_root.stat(follow_symlinks=False).st_mode)
            != 0o700
        ):
            raise PermissionError("attempt output root was not reserved mode 0700")
        return _publish_json(output_root / "reservation.json", reservation_core)
    except BaseException as error:
        failure_error: BaseException | None = None
        try:
            _publish_json(output_root / "reservation_failed.json", {
                "schema": contract.FAILURE_SCHEMA,
                "status": "TERMINAL_RESERVATION_COMMIT_FAILURE",
                "stage": "reservation_commit",
                "attempt_identity": attempt_identity,
                "error": {"type": type(error).__name__, "message": str(error)},
                "torch_imported": False,
                "runtime_input_opened": False,
                "retry_authorized": False,
                "authority": dict(contract.DOWNSTREAM_DENIALS),
            })
        except BaseException as terminal_error:
            failure_error = terminal_error
        finally:
            _seal_terminal(output_root)
        if failure_error is not None:
            raise RuntimeError(
                "reservation commit and terminalization both failed"
            ) from failure_error
        raise


def _load_path(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {path.relative_to(ROOT).as_posix()}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_post_reservation_stack() -> tuple[Any, Any, Any]:
    """First Torch-capable import point; caller must already own reservation."""
    matched_path = ROOT / contract.MATCHED_V1_RUNNER_RELATIVE_PATH
    matched_raw = _read_regular(
        matched_path,
        expected_sha256=contract.FROZEN_SOURCE_SHA256[
            contract.MATCHED_V1_RUNNER_RELATIVE_PATH
        ],
    )
    if not matched_raw:
        raise PermissionError("matched V1 reusable runner is empty")
    matched = _load_path(
        "_lewm_multires_probe_matched_v1_loader",
        matched_path,
    )
    base_runtime = matched._load_runtime()

    original_path = list(sys.path)
    try:
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_multires_v1 as multires,
        )
        from lewm.models import (  # type: ignore[import-not-found]
            shared_observable_camera_ray_jepa_v5_protected_camera_adaptation_v4_tail_depth
            as tail_depth,
        )
    finally:
        sys.path[:] = original_path
    expected_model = ROOT / contract.MODEL_RELATIVE_PATH
    observed_model = Path(multires.__file__)
    if (
        observed_model.is_symlink()
        or expected_model.is_symlink()
        or observed_model.resolve() != expected_model.resolve()
    ):
        raise PermissionError("imported multires model source changed")
    _read_regular(
        observed_model,
        expected_sha256=contract.MODEL_FILE_SHA256,
    )
    loss_adapter = SimpleNamespace(
        observable_camera_ray_v4_loss_v4=(
            tail_depth.observable_camera_ray_v4_tail_depth_loss_v4
        )
    )
    runtime = SimpleNamespace(
        **{
            **vars(base_runtime),
            "loss_adapter": loss_adapter,
        }
    )
    return matched, runtime, multires


def _read_bound(
    path: Path,
    binding: Mapping[str, Any],
) -> bytes:
    validated = contract.validate_binding(
        binding,
        path=path.relative_to(ROOT).as_posix(),
    )
    raw = _read_regular(path, expected_sha256=validated["file_sha256"])
    if len(raw) != validated["byte_count"]:
        raise PermissionError(f"bound byte count changed: {path}")
    return raw


def _rehash_deferred_runtime_and_authority(
    *,
    authorization: Mapping[str, Any],
    reservation: Mapping[str, Any],
) -> dict[str, Any]:
    runtime_inputs = authorization["runtime_inputs"]
    deferred = (
        runtime_inputs["camera"]["gate"],
        runtime_inputs["camera"]["checkpoint"],
        runtime_inputs["schedule"],
    )
    runtime_records: list[dict[str, Any]] = []
    for binding in deferred:
        raw = _read_bound(ROOT / binding["path"], binding)
        runtime_records.append({
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    authority_records: list[dict[str, Any]] = []
    for kind in ("independent_source_review", "execution_authorization"):
        binding = reservation[kind]
        raw = _read_regular(
            ROOT / binding["path"],
            expected_sha256=binding["file_sha256"],
        )
        if len(raw) != binding["byte_count"]:
            raise PermissionError(f"{kind} byte count changed")
        authority_records.append({
            "kind": kind,
            **dict(binding),
            "observed_file_sha256": hashlib.sha256(raw).hexdigest(),
            "observed_byte_count": len(raw),
        })
    return {
        "deferred_runtime_records": runtime_records,
        "authority_records": authority_records,
        "all_rehashed": True,
    }


def _load_schedule(
    matched: Any,
    authorization: Mapping[str, Any],
    train_pairs: Sequence[Mapping[str, Any]],
) -> tuple[list[int], dict[str, Any]]:
    binding = authorization["runtime_inputs"]["schedule"]
    raw = _read_bound(ROOT / binding["path"], binding)
    schedule = matched.contract.parse_canonical_json(
        raw, name="bound matched-training schedule"
    )
    if schedule.get("content_sha256") != contract.RUNTIME_CONTENT_SHA256[
        contract.SCHEDULE_RELATIVE_PATH
    ]:
        raise PermissionError("bound schedule content hash changed")
    indices = schedule.get("presentation_indices")
    if not isinstance(indices, list):
        raise PermissionError("bound schedule indices are absent")
    pair_ids = [str(item["content_sha256"]) for item in train_pairs]
    recomputed = matched.contract.with_content_sha256({
        **matched.contract.schedule_core(indices, pair_ids),
        "presentation_indices": indices,
    })
    if recomputed != schedule or len(indices) < contract.MAXIMUM_PRESENTATIONS:
        raise PermissionError("bound schedule does not recompute exactly")
    for update, expected in contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256.items():
        observed = contract.canonical_json_sha256(
            list(indices[: update * contract.EFFECTIVE_BATCH_SIZE])
        )
        if observed != expected:
            raise PermissionError(f"schedule prefix changed at update {update}")
    return list(indices[: contract.MAXIMUM_PRESENTATIONS]), {
        **dict(binding),
        "used_prefix_presentations": contract.MAXIMUM_PRESENTATIONS,
        "used_prefix_sha256": contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[1_000],
    }


def _state_sha(runtime: Any, state_or_model: Any) -> str:
    state = (
        state_or_model.state_dict()
        if hasattr(state_or_model, "state_dict")
        else state_or_model
    )
    return runtime.model_module.tensor_state_dict_sha256(state)


def _subset_sha(runtime: Any, model: Any, prefixes: Sequence[str]) -> str:
    state = {
        name: value
        for name, value in model.state_dict().items()
        if name.startswith(tuple(prefixes))
    }
    if not state:
        raise RuntimeError("fixed state subset is empty")
    return _state_sha(runtime, state)


def _receipt_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        observed = asdict(value)
    elif hasattr(value, "to_dict"):
        observed = value.to_dict()
    elif type(value) is dict:
        observed = dict(value)
    else:
        raise TypeError("migration receipt is not structured")
    if type(observed) is not dict:
        raise TypeError("migration receipt did not normalize to a dict")
    return observed


def _validate_migration_receipt(
    runtime: Any,
    multires: Any,
    model: Any,
    fit: Any,
    value: object,
) -> dict[str, Any]:
    receipt = _receipt_dict(value)
    fields = {
        "schema",
        "model_family",
        "base_initialization_seed",
        "decoder_initialization_seed",
        "initialization_input_role",
        "n320_checkpoint_file_sha256",
        "n320_checkpoint_content_sha256",
        "fit_model_state_sha256",
        "shared_encoder_state_sha256",
        "pixel_head_state_sha256",
        "ground_head_state_sha256",
        "decoder_state_sha256",
        "evidence_head_state_sha256",
        "copied_state_keys",
        "copied_state_entry_count",
        "copied_predecessor_dense_decoder_entry_count",
        "canonical_ground_support_exact",
        "hard_sync_count",
        "caller_cpu_rng_restored",
        "rejected_adaptation_checkpoint_open_count",
        "torch_version",
    }
    copied = receipt.get("copied_state_keys")
    expected_copied = sorted((
        *(f"encoder.{name}" for name in model.encoder.state_dict()),
        *(
            f"evidence_head.pixel_head.{name}"
            for name in model.evidence_head.pixel_head.state_dict()
        ),
        *(
            f"evidence_head.ground_head.{name}"
            for name in model.evidence_head.ground_head.state_dict()
        ),
    ))
    if (
        set(receipt) != fields
        or receipt["schema"] != multires.INITIALIZATION_SCHEMA
        or receipt["model_family"] != multires.MODEL_FAMILY
        or receipt["base_initialization_seed"]
        != contract.BASE_INITIALIZATION_SEED
        or receipt["decoder_initialization_seed"]
        != contract.DECODER_INITIALIZATION_SEED
        or receipt["initialization_input_role"]
        != "n320_fit_initialization_only"
        or receipt["n320_checkpoint_file_sha256"]
        != contract.RUNTIME_FILE_SHA256[contract.N320_CHECKPOINT_RELATIVE_PATH]
        or receipt["n320_checkpoint_content_sha256"]
        != contract.RUNTIME_CONTENT_SHA256[
            contract.N320_CHECKPOINT_RELATIVE_PATH
        ]
        or receipt["fit_model_state_sha256"] != _state_sha(runtime, fit)
        or receipt["shared_encoder_state_sha256"]
        != _state_sha(runtime, model.encoder)
        or receipt["pixel_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.pixel_head)
        or receipt["ground_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head.ground_head)
        or receipt["decoder_state_sha256"]
        != _state_sha(runtime, model.evidence_head.dense_decoder)
        or receipt["evidence_head_state_sha256"]
        != _state_sha(runtime, model.evidence_head)
        or type(copied) is not list
        or copied != expected_copied
        or len(copied) != 84
        or len(set(copied)) != 84
        or receipt["copied_state_entry_count"] != 84
        or receipt["copied_predecessor_dense_decoder_entry_count"] != 0
        or any("dense_decoder" in name for name in copied)
        or receipt["canonical_ground_support_exact"] is not True
        or receipt["hard_sync_count"] != 1
        or receipt["caller_cpu_rng_restored"] is not True
        or receipt["rejected_adaptation_checkpoint_open_count"] != 0
        or receipt["torch_version"] != str(runtime.torch.__version__)
        or not bool(getattr(model, "_n320_initialization_complete", False))
        or _state_sha(runtime, model.encoder)
        != _state_sha(runtime, model.target_encoder)
    ):
        raise PermissionError("N320 multires initialization receipt changed")
    return receipt


def _prepare_model(
    runtime: Any,
    multires: Any,
    fit: Any,
    device: Any,
) -> tuple[Any, list[Any], list[Any], list[Any], dict[str, Any]]:
    caller_rng = runtime.torch.random.get_rng_state().clone()
    model, raw_migration = (
        multires.SharedObservableCameraRayJepaV5MultiresV1
        .initialize_from_n320_fit_model(
            fit,
            n320_checkpoint_file_sha256=contract.RUNTIME_FILE_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
            n320_checkpoint_content_sha256=contract.RUNTIME_CONTENT_SHA256[
                contract.N320_CHECKPOINT_RELATIVE_PATH
            ],
        )
    )
    if not bool(runtime.torch.equal(
        caller_rng, runtime.torch.random.get_rng_state()
    )):
        raise RuntimeError("N320 model initialization changed caller CPU RNG")
    migration = _validate_migration_receipt(
        runtime, multires, model, fit, raw_migration
    )
    if (
        getattr(multires, "MODEL_FAMILY", None)
        != "shared_observable_camera_ray_jepa_v5_multires_v1"
    ):
        raise PermissionError("multires model runtime identity changed")
    declared_trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    expected_trainable = {
        name for name, _parameter in model.named_parameters()
        if name.startswith(contract.TRAINABLE_PARAMETER_PREFIXES)
    }
    if declared_trainable != expected_trainable:
        raise PermissionError("constructor trainable partition changed")
    model = model.to(device)
    model.requires_grad_(False)
    groups: dict[str, list[tuple[str, Any]]] = {
        "evidence_head": [],
        "encoder": [],
        "frozen": [],
    }
    for name in model.state_dict():
        contract.parameter_partition(name)
    for name, parameter in model.named_parameters():
        component = contract.parameter_partition(name)
        if component in ("evidence_head", "encoder"):
            parameter.requires_grad_(True)
            groups[component].append((name, parameter))
        else:
            groups["frozen"].append((name, parameter))
    counts = {
        name: sum(parameter.numel() for _, parameter in groups[name])
        for name in ("evidence_head", "encoder")
    }
    tensor_counts = {
        name: len(groups[name]) for name in ("evidence_head", "encoder")
    }
    if (
        counts != contract.EXPECTED_PARAMETER_COUNTS
        or tensor_counts != contract.EXPECTED_PARAMETER_TENSOR_COUNTS
        or not groups["frozen"]
        or any(parameter.requires_grad for _, parameter in groups["frozen"])
    ):
        raise PermissionError("multires trainable/frozen partition changed")
    names = {
        name: [parameter_name for parameter_name, _ in values]
        for name, values in groups.items()
    }
    partition = {
        "parameter_counts": counts,
        "parameter_tensor_counts": tensor_counts,
        "parameter_names_sha256": {
            name: contract.canonical_json_sha256(values)
            for name, values in names.items()
        },
        "migration": migration,
        "model_runtime_version": contract.MODEL_RUNTIME_VERSION,
        "initial_state_sha256": _state_sha(runtime, model),
    }
    return (
        model,
        [parameter for _, parameter in groups["evidence_head"]],
        [parameter for _, parameter in groups["encoder"]],
        [parameter for _, parameter in groups["frozen"]],
        partition,
    )


def _assert_frozen_grads_none(frozen: Sequence[Any]) -> None:
    if any(parameter.grad is not None for parameter in frozen):
        raise RuntimeError("a frozen parameter acquired a gradient")


def _gradient_group_norm(
    runtime: Any,
    parameters: Sequence[Any],
    group: str,
    *,
    maximum: float | None = None,
) -> float:
    if len(parameters) != contract.EXPECTED_PARAMETER_TENSOR_COUNTS[group]:
        raise RuntimeError(f"{group} gradient tensor count changed")
    gradients = [parameter.grad for parameter in parameters]
    if any(gradient is None for gradient in gradients):
        raise RuntimeError(f"{group} parameter has no gradient")
    if not bool(runtime.torch.stack([
        runtime.torch.isfinite(gradient).all() for gradient in gradients
    ]).all().item()):
        raise FloatingPointError(f"{group} gradient became nonfinite")
    squared = runtime.torch.stack([
        gradient.detach().float().square().sum() for gradient in gradients
    ]).sum()
    norm = math.sqrt(float(squared.detach().cpu()))
    if (
        not math.isfinite(norm)
        or (
            maximum is not None
            and norm > maximum + contract.POST_CLIP_NORM_ASSERTION_TOLERANCE
        )
    ):
        raise RuntimeError(f"{group} gradient norm is invalid")
    return norm


def _camera_pair(runtime: Any, model: Any, batch: Mapping[str, Any]) -> Any:
    forward = batch["forward"]
    current = model.forward_frame(
        forward["current_image"],
        forward["current_camera_origin_body_m"],
        forward["current_camera_basis_body_fru"],
        forward["current_ground_plane_z_body_m"],
    )
    next_frame = model.forward_frame(
        forward["next_image"],
        forward["next_camera_origin_body_m"],
        forward["next_camera_basis_body_fru"],
        forward["next_ground_plane_z_body_m"],
    )
    overlap = runtime.torch.ones_like(
        current.bev[:, :1], dtype=runtime.torch.bool
    )
    return runtime.model_module.SharedTrainingPairV5(
        current=current,
        next=next_frame,
        predicted_next_bev=next_frame.bev,
        stop_gradient_target_next_bev=next_frame.bev.detach(),
        commanded_warped_current_bev=current.bev,
        commanded_overlap_mask=overlap,
        realized_warped_current_bev=current.bev,
        realized_overlap_mask=overlap,
        jepa=None,
    )


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu())
    if not math.isfinite(result):
        raise FloatingPointError("probe scalar became nonfinite")
    return result


def _camera_components(loss: Any) -> dict[str, float]:
    result = {"camera_total": _scalar(loss.total)}
    for side in ("current", "next"):
        frame = getattr(loss, side)
        result.update({
            f"{side}_hierarchical_first_hit_nll":
                _scalar(frame.hierarchical_first_hit_nll),
            f"{side}_tail_depth_p95_cvar":
                _scalar(frame.tail_depth_p95_cvar),
            f"{side}_ground_clear_distance_state_balanced_bce":
                _scalar(frame.ground_clear_distance_state_balanced_bce),
            f"{side}_derived_raster_hierarchical_bce":
                _scalar(frame.derived_raster_hierarchical_bce.total),
            f"{side}_derived_raster_cell_nll":
                _scalar(frame.derived_raster_cell_nll),
        })
    return result


def _snapshot(
    runtime: Any,
    model: Any,
    output_root: Path,
    *,
    update: int,
    frozen_sha256: str,
    initial_state_sha256: str,
    migration: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        name: value.detach().cpu().contiguous().clone()
        for name, value in sorted(model.state_dict().items())
    }
    state_sha256 = _state_sha(runtime, state)
    frozen_observed = _state_sha(runtime, {
        name: value
        for name, value in state.items()
        if name.startswith(contract.FROZEN_STATE_PREFIXES)
    })
    if frozen_observed != frozen_sha256:
        raise RuntimeError("frozen state changed before snapshot")
    semantic = {
        "schema": contract.SNAPSHOT_SCHEMA,
        "update": update,
        "model_family": "shared_observable_camera_ray_jepa_v5_multires_v1",
        "model_config": model.model_config.to_dict(),
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
        "initial_state_sha256": initial_state_sha256,
        "migration": dict(migration),
        "schedule_prefix_indices_sha256":
            contract.CHECKPOINT_SCHEDULE_PREFIX_SHA256[update],
        "development_only": True,
        "resume_authorized": False,
        "runtime_ready": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    content_sha256 = contract.canonical_json_sha256(semantic)
    buffer = io.BytesIO()
    runtime.torch.save({
        **semantic,
        "content_sha256": content_sha256,
        "model_state_dict": state,
    }, buffer)
    raw = buffer.getvalue()
    relative = f"checkpoints/update_{update}.pt"
    _write_exclusive(output_root / relative, raw)
    return {
        "path": relative,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
        "state_sha256": state_sha256,
        "frozen_state_sha256": frozen_sha256,
    }


def _evaluate(
    runtime: Any,
    trainer: Any,
    model: Any,
    selection_pairs: Sequence[Mapping[str, Any]],
    device: Any,
    *,
    update: int,
    frozen_sha256: str,
) -> dict[str, Any]:
    before = _state_sha(runtime, model)
    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha256:
        raise RuntimeError("frozen state changed before inline evaluation")
    model.eval()
    physical, camera_loss = trainer.physical_metrics(
        model,
        selection_pairs,
        device,
        arm="multires_probe_v1",
        stage=f"inline_checkpoint_selection_update_{update}",
    )
    model.train()
    after = _state_sha(runtime, model)
    frozen_after = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    if before != after or frozen_after != frozen_sha256:
        raise RuntimeError("inline evaluation mutated model state")
    evaluation = contract.evaluate_physical_scopes(physical)
    return {
        "update": update,
        "role": "checkpoint_selection",
        "pair_count": contract.SELECTION_ROLE_COUNTS["pairs"],
        "unique_endpoint_count":
            contract.SELECTION_ROLE_COUNTS["unique_endpoints"],
        "scopes": physical,
        "aggregate_complete_v4_tail_depth_loss": float(camera_loss),
        "evaluation": evaluation,
        "integrity_pass": True,
        "state_sha256_before": before,
        "state_sha256_after": after,
        "frozen_state_sha256_before_and_after": frozen_sha256,
        "state_mutation_count": 0,
    }


def _publish_metric_sidecar(
    output_root: Path,
    *,
    update: int,
    checkpoint: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    continuation = contract.checkpoint_control_decision(
        update=update,
        evaluation=metric["evaluation"],
        integrity_pass=metric["integrity_pass"],
    )
    core = {
        "schema": contract.METRIC_SIDECAR_SCHEMA,
        "status": "PUBLISHED_0444_AFTER_INLINE_EVALUATION_BEFORE_CONTROL",
        "update": update,
        "checkpoint": dict(checkpoint),
        "metric": dict(metric),
        "inline_evaluation_count": 1,
        "state_mutation_count": 0,
        "publication_order": [
            "cpu_snapshot",
            "inline_nonmutating_selection_evaluation",
            "atomic_mode_0444_sidecar",
            "control_branch",
        ],
        "continuation": continuation,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    }
    value = contract.with_content_sha256(core)
    raw = contract.canonical_json_bytes(value) + b"\n"
    relative = contract.metric_sidecar_relative_path(update)
    _publish_readonly_atomic(output_root / relative, raw)
    contract.validate_metric_sidecar(value, update=update)
    return _binding(relative, value, raw), continuation


def _train(
    runtime: Any,
    trainer: Any,
    model: Any,
    head: Sequence[Any],
    encoder: Sequence[Any],
    frozen: Sequence[Any],
    train_pairs: Sequence[Mapping[str, Any]],
    selection_pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    vocabulary: Sequence[str],
    commanded: Any,
    device: Any,
    output_root: Path,
    partition: Mapping[str, Any],
) -> dict[str, Any]:
    frozen_sha256 = _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES)
    initial_state_sha256 = _state_sha(runtime, model)
    optimizer = runtime.torch.optim.AdamW(
        [
            {
                "params": list(head),
                "lr": contract.learning_rates(1)[0],
                "group_name": "evidence_head",
            },
            {
                "params": list(encoder),
                "lr": contract.learning_rates(1)[1],
                "group_name": "encoder",
            },
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4,
        amsgrad=False,
    )
    trace: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []
    sidecars: list[dict[str, Any]] = []
    controls: list[dict[str, Any]] = []
    for update in range(1, contract.MAXIMUM_UPDATE + 1):
        head_lr, encoder_lr = contract.learning_rates(update)
        optimizer.param_groups[0]["lr"] = head_lr
        optimizer.param_groups[1]["lr"] = encoder_lr
        _assert_frozen_grads_none(frozen)
        optimizer.zero_grad(set_to_none=True)
        sums: dict[str, float] = {}
        start = (update - 1) * contract.EFFECTIVE_BATCH_SIZE
        update_indices = indices[start : start + contract.EFFECTIVE_BATCH_SIZE]
        if len(update_indices) != contract.EFFECTIVE_BATCH_SIZE:
            raise PermissionError("fixed presentation schedule ended early")
        for microbatch in range(contract.MICROBATCHES_PER_UPDATE):
            low = microbatch * contract.MICROBATCH_SIZE
            batch = trainer.batch(
                train_pairs,
                update_indices[low : low + contract.MICROBATCH_SIZE],
                vocabulary,
                commanded,
                device,
                role="train",
                arm="multires_probe_v1",
                stage="camera_gradient",
            )
            pair = _camera_pair(runtime, model, batch)
            camera = runtime.loss_adapter.observable_camera_ray_v4_loss_v4(
                model,
                pair,
                batch["current_supervision"],
                batch["next_supervision"],
            )
            if not bool(runtime.torch.isfinite(camera.total).item()):
                raise FloatingPointError("probe backward scalar became nonfinite")
            (camera.total / contract.MICROBATCHES_PER_UPDATE).backward()
            for name, value in _camera_components(camera).items():
                sums[name] = (
                    sums.get(name, 0.0)
                    + value / contract.MICROBATCHES_PER_UPDATE
                )
        _assert_frozen_grads_none(frozen)
        head_pre = _gradient_group_norm(runtime, head, "evidence_head")
        encoder_pre = _gradient_group_norm(runtime, encoder, "encoder")
        head_clip = runtime.torch.nn.utils.clip_grad_norm_(head, max_norm=1.0)
        encoder_clip = runtime.torch.nn.utils.clip_grad_norm_(
            encoder, max_norm=1.0
        )
        if (
            not bool(runtime.torch.isfinite(head_clip).item())
            or not bool(runtime.torch.isfinite(encoder_clip).item())
        ):
            raise FloatingPointError("probe clip norm became nonfinite")
        head_post = _gradient_group_norm(
            runtime, head, "evidence_head", maximum=1.0
        )
        encoder_post = _gradient_group_norm(
            runtime, encoder, "encoder", maximum=1.0
        )
        optimizer.step()
        _assert_frozen_grads_none(frozen)
        trace.append({
            "schema": f"{contract.SCHEMA_PREFIX}_trace_row_v1",
            "update": update,
            "presentation_indices_sha256":
                contract.canonical_json_sha256(list(update_indices)),
            "head_learning_rate": head_lr,
            "encoder_learning_rate": encoder_lr,
            "microbatch_count": contract.MICROBATCHES_PER_UPDATE,
            "camera_objective_count": contract.MICROBATCHES_PER_UPDATE,
            "backward_call_count": contract.MICROBATCHES_PER_UPDATE,
            "optimizer_step_count": update,
            "head_clip_invocation_count": update,
            "encoder_clip_invocation_count": update,
            "head_gradient_norm_before_clip": head_pre,
            "encoder_gradient_norm_before_clip": encoder_pre,
            "head_clip_return_norm": _scalar(head_clip),
            "encoder_clip_return_norm": _scalar(encoder_clip),
            "head_gradient_norm_after_clip": head_post,
            "encoder_gradient_norm_after_clip": encoder_post,
            "losses": sums,
            "jepa_objective_count": 0,
            "jepa_backward_count": 0,
            "ema_update_count": 0,
        })
        if update not in contract.CHECKPOINT_UPDATES:
            continue
        if _subset_sha(
            runtime, model, contract.FROZEN_STATE_PREFIXES
        ) != frozen_sha256:
            raise RuntimeError("frozen state changed during probe training")
        snapshot = _snapshot(
            runtime,
            model,
            output_root,
            update=update,
            frozen_sha256=frozen_sha256,
            initial_state_sha256=initial_state_sha256,
            migration=partition["migration"],
        )
        snapshots.append(snapshot)
        metric = _evaluate(
            runtime,
            trainer,
            model,
            selection_pairs,
            device,
            update=update,
            frozen_sha256=frozen_sha256,
        )
        metrics.append(metric)
        sidecar, control = _publish_metric_sidecar(
            output_root,
            update=update,
            checkpoint=snapshot,
            metric=metric,
        )
        sidecars.append(sidecar)
        # The branch occurs only after the immutable sidecar is visible.
        controls.append(control)
        if update in (100, 400):
            if control["action"] != contract.CONTROL_CONTINUE:
                raise RuntimeError("informational checkpoint stopped the probe")
        elif control["action"] not in (
            contract.CONTROL_PASS,
            contract.CONTROL_FAIL,
        ):
            raise RuntimeError("terminal probe control is invalid")
    if [row["update"] for row in metrics] != list(contract.CHECKPOINT_UPDATES):
        raise RuntimeError("probe did not evaluate the exact checkpoint set")
    if _subset_sha(runtime, model, contract.FROZEN_STATE_PREFIXES) != frozen_sha256:
        raise RuntimeError("frozen state changed at probe terminal")
    return {
        "trace": trace,
        "metrics": metrics,
        "snapshots": snapshots,
        "sidecars": sidecars,
        "controls": controls,
        "terminal_control": controls[-1],
        "frozen_state_sha256": frozen_sha256,
        "final_state_sha256": _state_sha(runtime, model),
        "operation_counts": contract.operation_counts(
            contract.MAXIMUM_UPDATE, contract.CHECKPOINT_UPDATES
        ),
    }


def _publish_training_records(
    output_root: Path,
    training: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    trace = training["trace"]
    trace_raw = b"".join(
        contract.canonical_json_bytes(row) + b"\n" for row in trace
    )
    _write_exclusive(output_root / "training_trace.jsonl", trace_raw)
    trace_binding = {
        "path": "training_trace.jsonl",
        "file_sha256": hashlib.sha256(trace_raw).hexdigest(),
        "content_sha256": contract.canonical_json_sha256(trace),
        "byte_count": len(trace_raw),
        "row_count": len(trace),
    }
    value, raw = _publish_json(output_root / "checkpoint_metrics.json", {
        "schema": f"{contract.SCHEMA_PREFIX}_checkpoint_metrics_v1",
        "status": "COLLATED_FROM_THREE_IMMUTABLE_INLINE_SIDECARS",
        "checkpoint_updates": list(contract.CHECKPOINT_UPDATES),
        "rows": list(training["metrics"]),
        "sidecars": list(training["sidecars"]),
        "controls": list(training["controls"]),
        "inline_evaluation_count": 3,
        "observer_evaluation_rerun_count": 0,
        "threshold_equality_passes": False,
        "authority": dict(contract.DOWNSTREAM_DENIALS),
    })
    return trace_binding, _binding("checkpoint_metrics.json", value, raw)


def _terminal_inventory(
    output_root: Path,
    *,
    exclude: Sequence[str] = (),
) -> tuple[list[str], list[str]]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise PermissionError("terminal output contains a symlink")
    excluded = set(exclude)
    files = sorted(
        item.relative_to(output_root).as_posix()
        for item in entries
        if item.is_file()
        and item.relative_to(output_root).as_posix() not in excluded
    )
    directories = [
        ".",
        *sorted(
            item.relative_to(output_root).as_posix()
            for item in entries
            if item.is_dir()
        ),
    ]
    return files, directories


def _seal_terminal(output_root: Path) -> dict[str, Any]:
    entries = list(output_root.rglob("*"))
    if any(item.is_symlink() for item in entries):
        raise PermissionError("cannot seal a symlinked terminal")
    for path in (item for item in entries if item.is_file()):
        os.chmod(path, 0o444, follow_symlinks=False)
    directories = sorted(
        (item for item in entries if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    )
    for path in directories:
        os.chmod(path, 0o555, follow_symlinks=False)
    os.chmod(output_root, 0o555, follow_symlinks=False)
    files, observed_directories = _terminal_inventory(output_root)
    if any(
        stat.S_IMODE((output_root / relative).stat().st_mode) != 0o444
        for relative in files
    ):
        raise PermissionError("terminal file sealing failed")
    if any(
        stat.S_IMODE(
            (output_root if relative == "." else output_root / relative).stat().st_mode
        )
        != 0o555
        for relative in observed_directories
    ):
        raise PermissionError("terminal directory sealing failed")
    return {
        "files": files,
        "directories_including_root": observed_directories,
        "file_mode": "0444",
        "directory_mode": "0555",
    }


def _terminal_failure(
    output_root: Path,
    reservation: Mapping[str, Any],
    *,
    stage: str,
    error: BaseException,
) -> None:
    try:
        files, directories = _terminal_inventory(
            output_root, exclude=("failed.json",)
        )
        _publish_json(output_root / "failed.json", {
            "schema": contract.FAILURE_SCHEMA,
            "status": "TERMINAL_LIFECYCLE_OR_INTEGRITY_FAILURE",
            "stage": stage,
            "attempt_identity": reservation["attempt_identity"],
            "published_prefix": files,
            "directories_including_root": directories,
            "error": {"type": type(error).__name__, "message": str(error)},
            "retry_authorized": False,
            "g2_navigation_or_heldout_attempted": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
    finally:
        _seal_terminal(output_root)


def _execute_after_reservation(
    *,
    review: Mapping[str, Any],
    review_raw: bytes,
    authorization: Mapping[str, Any],
    authorization_raw: bytes,
    sources: Mapping[str, str],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    preflight: Mapping[str, Any],
    output_root: Path,
) -> int:
    del preflight
    stage = "post_reservation_source_and_authority_rehash"
    try:
        if contract.current_source_bindings(ROOT) != dict(sources):
            raise PermissionError("reviewed source changed across reservation")
        observed_review = contract.validate_review(
            contract.parse_canonical_json(review_raw, name="source review rehash"),
            expected_sources=sources,
        )
        review_binding = contract.artifact_binding(
            contract.REVIEW_RELATIVE_PATH,
            review_raw,
            content_sha256=str(observed_review["content_sha256"]),
        )
        observed_authorization = contract.validate_authorization(
            contract.parse_canonical_json(
                authorization_raw, name="execution authorization rehash"
            ),
            review_binding=review_binding,
            reviewer=str(observed_review["reviewer"]),
        )
        if (
            observed_review != dict(review)
            or observed_authorization != dict(authorization)
        ):
            raise PermissionError("authority changed across reservation")

        stage = "deferred_torch_and_reusable_v1_loader_import"
        matched, runtime, multires = _load_post_reservation_stack()
        stage = "deferred_n320_direct_reconstruction"
        runtime_authority = authorization["runtime_inputs"]
        adapted_authorization = {
            "raw": runtime_authority["raw"],
            "camera": runtime_authority["camera"],
        }
        fit, gate, camera_binding = matched._camera_model_after_reservation(
            runtime, adapted_authorization
        )
        inputs = matched.RawInputs(runtime, adapted_authorization)
        trainer = matched.Trainer(runtime, inputs, output_root, reservation)
        device, hardware = trainer.device()
        if (
            hardware["visible_device_count"] != 1
            or "r9700" not in hardware["name"].casefold().replace(" ", "")
        ):
            raise PermissionError("reserved runtime device differs from preflight")
        train_pairs = inputs.role_pairs("train")
        selection_pairs = inputs.role_pairs("checkpoint_selection")
        vocabulary, commanded_cpu = trainer.commanded_table(train_pairs)
        indices, schedule_binding = _load_schedule(
            matched, authorization, train_pairs
        )
        model, head, encoder, frozen, partition = _prepare_model(
            runtime, multires, fit, device
        )
        del fit
        commanded = commanded_cpu.to(device)

        stage = "bounded_1000_update_training_and_inline_selection"
        training = _train(
            runtime,
            trainer,
            model,
            head,
            encoder,
            frozen,
            train_pairs,
            selection_pairs,
            indices,
            vocabulary,
            commanded,
            device,
            output_root,
            partition,
        )
        trace_binding, metrics_binding = _publish_training_records(
            output_root, training
        )

        stage = "all_consumed_input_and_source_rehash"
        consumed = inputs.rehash_consumed()
        observed_roles = {
            role for row in consumed["records"] for role in row["roles"]
        }
        if (
            not {"train", "checkpoint_selection"}.issubset(observed_roles)
            or not observed_roles.issubset(
                {"authority", "index", "train", "checkpoint_selection"}
            )
            or contract.current_source_bindings(ROOT) != dict(sources)
        ):
            raise PermissionError("probe consumed an unauthorized role or source")
        final_rehash = _rehash_deferred_runtime_and_authority(
            authorization=authorization,
            reservation=reservation,
        )
        access, access_raw = _publish_json(output_root / "access.json", {
            "schema": contract.ACCESS_SCHEMA,
            "status": "ALL_CONSUMED_DEVELOPMENT_INPUTS_REHASHED",
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "roles_opened": ["train", "checkpoint_selection"],
            "probability_calibration_open_count": 0,
            "n320": {
                "gate_content_sha256": gate["content_sha256"],
                "checkpoint": camera_binding,
                "initialization_only": True,
            },
            "schedule": schedule_binding,
            "consumed": consumed,
            "deferred_runtime_and_authority_rehash": final_rehash,
            "reviewed_sources": {
                "count": len(sources),
                "bindings": dict(sources),
                "all_rehashed": True,
            },
            "rejected_adaptation_checkpoint_open_count": 0,
            "g2_navigation_or_heldout_open_count": 0,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        terminal = training["terminal_control"]
        passed = terminal["action"] == contract.CONTROL_PASS
        stage = "terminal_scientific_result_publication"
        result, result_raw = _publish_json(output_root / "result.json", {
            "schema": contract.RESULT_SCHEMA,
            "status": (
                "PASS_BOUNDED_FALSIFICATION_SEPARATE_QUALIFICATION_PREREG_ONLY"
                if passed
                else "FAIL_BOUNDED_FALSIFICATION_MECHANISM_TERMINATED"
            ),
            "reservation": _binding(
                "reservation.json", reservation, reservation_raw
            ),
            "access": _binding("access.json", access, access_raw),
            "terminal_control": terminal,
            "snapshots": list(training["snapshots"]),
            "checkpoint_metrics": metrics_binding,
            "training_trace": trace_binding,
            "partition": partition,
            "state": {
                "initial_state_sha256": partition["initial_state_sha256"],
                "frozen_state_sha256": training["frozen_state_sha256"],
                "final_state_sha256": training["final_state_sha256"],
            },
            "operation_counts": training["operation_counts"],
            "probe_pass_authorizes":
                "separate_bounded_perception_qualification_preregistration_only"
                if passed
                else "nothing",
            "checkpoint_qualified": False,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        stage = "completion_publication"
        files, directories = _terminal_inventory(output_root)
        completed, _ = _publish_json(output_root / "completed.json", {
            "schema": contract.COMPLETION_SCHEMA,
            "status": "TERMINAL_PASS" if passed else "TERMINAL_FAIL",
            "attempt_identity": reservation["attempt_identity"],
            "result": _binding("result.json", result, result_raw),
            "terminal_control": terminal,
            "operation_counts": training["operation_counts"],
            "exact_precompletion_files": files,
            "exact_terminal_files": sorted([*files, "completed.json"]),
            "exact_terminal_directories_including_root": directories,
            "all_inputs_rehashed": True,
            "all_terminal_files_sealed_read_only": True,
            "retry_authorized": False,
            "authority": dict(contract.DOWNSTREAM_DENIALS),
        })
        if completed["operation_counts"] != contract.operation_counts(
            1_000, contract.CHECKPOINT_UPDATES
        ):
            raise RuntimeError("terminal operation counts changed")
        _seal_terminal(output_root)
        return 0 if passed else 2
    except BaseException as error:
        _terminal_failure(
            output_root,
            reservation,
            stage=stage,
            error=error,
        )
        raise


def run_parent(
    *,
    review_file_sha256: str,
    authorization_file_sha256: str,
    preflight_file_sha256: str,
) -> int:
    # Immutable pre-reservation order: authority first, preflight second,
    # namespace reservation third.  No call above imports Torch or opens a
    # generated runtime input.
    review, review_raw, authorization, authorization_raw, sources = (
        _load_authority_pre_reservation(
            review_file_sha256,
            authorization_file_sha256,
        )
    )
    preflight = _validate_preflight(
        expected_sha256=preflight_file_sha256,
        launcher_source_sha256=sources[contract.LAUNCHER_RELATIVE_PATH],
        expected_source_authority=_source_authority_receipt(
            review=review,
            review_raw=review_raw,
            authorization=authorization,
            authorization_raw=authorization_raw,
            sources=sources,
        ),
    )
    output_root = ROOT / contract.OUTPUT_ROOT_RELATIVE_PATH
    reservation, reservation_raw = _reserve(
        output_root,
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
        preflight=preflight,
    )
    return _execute_after_reservation(
        review=review,
        review_raw=review_raw,
        authorization=authorization,
        authorization_raw=authorization_raw,
        sources=sources,
        reservation=reservation,
        reservation_raw=reservation_raw,
        preflight=preflight,
        output_root=output_root,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--review-sha256")
    parser.add_argument("--authorization-sha256")
    parser.add_argument("--preflight-sha256")
    args = parser.parse_args(argv)
    if (
        not args.run
        or not contract.is_sha256(args.review_sha256)
        or not contract.is_sha256(args.authorization_sha256)
        or not contract.is_sha256(args.preflight_sha256)
    ):
        parser.error("--run and all three exact SHA-256 arguments are required")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    return run_parent(
        review_file_sha256=args.review_sha256,
        authorization_file_sha256=args.authorization_sha256,
        preflight_file_sha256=args.preflight_sha256,
    )


if __name__ == "__main__":
    raise SystemExit(main())
