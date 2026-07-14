#!/usr/bin/env python3
"""Lean runtime-only recovery wrapper for the frozen Camera V15 operation.

V16 changes no data, model, objective, schedule, seed, or gate.  It validates
the consumed V15 terminal incident, requires an independent V16 source review,
redirects the reviewed V15 lifecycle to a fresh V16 namespace, and makes the
retained determinism helper safe after the V15 visibility check has already
configured PyTorch's one-shot inter-op thread setting.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Iterator, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
V15_EXECUTOR_PATH = ROOT / (
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_"
    "gate_aligned_raster_nll_v15.py"
)
V15_OUTPUT_ROOT = ROOT / (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v15"
)
V16_OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v16"
)
V16_OUTPUT_ROOT = ROOT / V16_OUTPUT_ROOT_RELATIVE_PATH
V16_ATTEMPT_PATH = V16_OUTPUT_ROOT / "attempts/seed_20260710/n5"
V16_METRIC_RECEIPT_PATH = (
    V16_OUTPUT_ROOT / "metric_verifications/seed_20260710_n5.json"
)
V16_GATE_PATH = V16_OUTPUT_ROOT / "gates/seed_20260710_n5.json"
V15_GPU_VISIBILITY_RECEIPT_PATH = Path(
    "/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_"
    "gate_aligned_raster_nll_v15_gpu_visibility_preflight_2026-07-14.json"
)
V16_GPU_VISIBILITY_RECEIPT_PATH = Path(
    "/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_"
    "gate_aligned_raster_nll_v16_gpu_visibility_preflight_2026-07-14.json"
)
V15_GPU_VISIBILITY_RECEIPT_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_"
    "gpu_visibility_preflight_v1"
)
V16_GPU_VISIBILITY_RECEIPT_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16_"
    "gpu_visibility_preflight_v1"
)
V15_ATTEMPT_SCOPE = "one_exclusive_fresh_gate_aligned_raster_nll_v15_attempt"
V16_ATTEMPT_SCOPE = "one_exclusive_fresh_gate_aligned_raster_nll_v16_attempt"

V15_TERMINAL_RESERVATION_PATH = (
    V15_OUTPUT_ROOT / "attempts/seed_20260710/n5/reservation.json"
)
V15_TERMINAL_FAILURE_PATH = (
    V15_OUTPUT_ROOT / "attempts/seed_20260710/n5/failed.json"
)
V15_TERMINAL_LOCK_PATH = (
    V15_OUTPUT_ROOT / "attempts/seed_20260710/.n5.reservation-v15.lock"
)
V15_TERMINAL_BINDINGS = {
    "reservation": {
        "file_sha256": (
            "bae23223289aa07ae1951f6d7d1202780856aa555d51e385f1961a229c1ae706"
        ),
        "content_sha256": (
            "ecccab261a3e9d5bcb2fb6b3f0fe52c864abd4fd6c5ed07d6dcce53347b17b29"
        ),
    },
    "failure": {
        "file_sha256": (
            "797280760654144a156d96148664d956d67b38e3d70cfc07afe9936ad6c3b2fe"
        ),
        "content_sha256": (
            "73862b8c640bd4aacaf68917d263a0265a0359fbf26219421729fe04da4e31a0"
        ),
    },
    "lock": {
        "file_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "byte_count": 0,
    },
    "seed_root_inventory": [".n5.reservation-v15.lock", "n5"],
    "output_root_inventory": ["attempts", "gates", "metric_verifications"],
}
V15_RETAINED_FILES = {
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v15.py": (
        "17677435731779c9549b5fb8f08b3268f223bc7a945d40f4f2f572a3b652e0ed"
    ),
    "scripts/preflight_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v15_gpu_visibility.py": (
        "fe913bd04448ea5ddae39186c805c8448c72a4f0bd12b430c26dd29a991b3051"
    ),
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v15.py": (
        "8879a42bd091609e4d48aa8ff743d0ab5adcb595caead3507c4393afcc8a7d6d"
    ),
    "scripts/train_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v15.py": (
        "62f5d9d5072bb83f6c8fd9af4c8bb32a96357d3365ba87a5258a529ae1ddcaf1"
    ),
    "scripts/verify_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v15.py": (
        "bb3e8838689105ab2ee1e4e5525d1de341525439aa83e526ff834efce89a1584"
    ),
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_"
    "nll_v15_independent_review_2026-07-14.json": (
        "49c7c4405ef98e955464e73291e2a77e1942836bbaec06b3d4d826920958a624"
    ),
}
V15_SOURCE_REVIEW_FILE_SHA256 = (
    "49c7c4405ef98e955464e73291e2a77e1942836bbaec06b3d4d826920958a624"
)
V16_REVIEW_PATH = ROOT / (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v16_independent_review_2026-07-14.json"
)
V16_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v16_runtime_recovery_source_review_v1"
)
V16_REVIEW_ENVIRONMENT = "LEWM_CAMERA_V16_SOURCE_REVIEW_SHA256"
V16_IMPLEMENTATION_AUTHOR = "/root/camera_v16_lean_recovery"
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
V16_SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v16_independent_review_2026-07-14.json"
)
V16_SUCCESSOR_SOURCE_PATHS = (
    "scripts/execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v16.py",
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
    "raster_nll_v16_runtime_recovery.py",
)
V16_SUCCESSOR_PROOF_PATHS = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_"
    "nll_v16_runtime_recovery_amendment_and_handoff_2026-07-14.md",
)
V16_RUNTIME_DELTA = {
    "science": "bit_for_bit_retained_v15",
    "output_root": V16_OUTPUT_ROOT_RELATIVE_PATH,
    "attempt_scope": V16_ATTEMPT_SCOPE,
    "gpu_visibility_receipt_path": str(V16_GPU_VISIBILITY_RECEIPT_PATH),
    "gpu_visibility_receipt_schema": V16_GPU_VISIBILITY_RECEIPT_SCHEMA,
    "determinism_thread_configuration": (
        "set_intra_and_inter_op_to_one_only_when_not_already_one"
    ),
    "retained_v15_schemas_and_transaction": "compatibility_only",
}
V16_REVIEW_LICENSES = {
    "v16_exact_attempt_authorized": True,
    "retry_authorized": False,
    "later_rung_execution_authorized": False,
    "checkpoint_use_authorized": False,
    "heldout_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "hardware_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
}


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _read_regular(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise PermissionError("V16 requires no-follow regular-file opens")
    descriptor = os.open(path, flags | nofollow)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"V16 rejected non-private regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise PermissionError(f"V16 observed a changing file: {path}")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise PermissionError(f"V16 observed a truncated file: {path}")
        return raw
    finally:
        os.close(descriptor)


def _load_bound_json(
    path: Path,
    *,
    file_sha256: str,
    content_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path)
    if hashlib.sha256(raw).hexdigest() != file_sha256:
        raise PermissionError(f"V16 file binding changed: {path}")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"V16 JSON is malformed: {path}") from error
    if type(value) is not dict or raw != _canonical_json_bytes(value) + b"\n":
        raise ValueError(f"V16 JSON is not a canonical object: {path}")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if (
        not isinstance(declared, str)
        or hashlib.sha256(_canonical_json_bytes(core)).hexdigest() != declared
        or (content_sha256 is not None and declared != content_sha256)
    ):
        raise PermissionError(f"V16 content binding changed: {path}")
    return value, raw


def _validate_terminal_v15(
    *,
    reservation_path: Path = V15_TERMINAL_RESERVATION_PATH,
    failure_path: Path = V15_TERMINAL_FAILURE_PATH,
    lock_path: Path = V15_TERMINAL_LOCK_PATH,
    bindings: Mapping[str, Any] = V15_TERMINAL_BINDINGS,
    require_inventory: bool = True,
) -> dict[str, str]:
    reservation, reservation_raw = _load_bound_json(
        reservation_path,
        file_sha256=bindings["reservation"]["file_sha256"],
        content_sha256=bindings["reservation"]["content_sha256"],
    )
    failure, failure_raw = _load_bound_json(
        failure_path,
        file_sha256=bindings["failure"]["file_sha256"],
        content_sha256=bindings["failure"]["content_sha256"],
    )
    lock_raw = _read_regular(lock_path)
    lock_metadata = lock_path.stat(follow_symlinks=False)
    if (
        hashlib.sha256(lock_raw).hexdigest()
        != bindings["lock"]["file_sha256"]
        or len(lock_raw) != bindings["lock"]["byte_count"]
        or stat.S_IMODE(lock_metadata.st_mode) != 0o600
    ):
        raise PermissionError("V16 rejected the V15 terminal lock")
    expected_reservation = {
        "path": "reservation.json",
        "file_sha256": bindings["reservation"]["file_sha256"],
        "content_sha256": bindings["reservation"]["content_sha256"],
        "byte_count": len(reservation_raw),
    }
    if (
        reservation.get("schema")
        != (
            "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_reservation_v1"
        )
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or reservation.get("seed") != 20260710
        or reservation.get("fit_size") != 5
        or failure.get("schema")
        != (
            "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_failure_v1"
        )
        or failure.get("failure_stage") != "training"
        or failure.get("reservation") != expected_reservation
        or failure.get("retry_authorized") is not False
        or not isinstance(failure.get("licenses"), Mapping)
        or failure["licenses"].get("retry_authorized") is not False
    ):
        raise PermissionError("V16 rejected the V15 terminal incident semantics")
    if require_inventory:
        if sorted(item.name for item in reservation_path.parent.iterdir()) != [
            "failed.json",
            "reservation.json",
        ]:
            raise PermissionError("V15 terminal attempt gained a numeric artifact")
        seed_root = reservation_path.parent.parent
        if sorted(item.name for item in seed_root.iterdir()) != sorted(
            bindings["seed_root_inventory"]
        ):
            raise PermissionError("V15 terminal seed-root inventory changed")
        output_root = reservation_path.parents[3]
        if sorted(item.name for item in output_root.iterdir()) != sorted(
            bindings["output_root_inventory"]
        ):
            raise PermissionError("V15 terminal root gained a derived artifact")
        for empty_name in ("gates", "metric_verifications"):
            if any((output_root / empty_name).iterdir()):
                raise PermissionError("V15 terminal derived directory is not empty")
    return {
        "reservation_file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "reservation_content_sha256": str(reservation["content_sha256"]),
        "failure_file_sha256": hashlib.sha256(failure_raw).hexdigest(),
        "failure_content_sha256": str(failure["content_sha256"]),
        "lock_file_sha256": hashlib.sha256(lock_raw).hexdigest(),
    }


def _validate_retained_v15_files(
    *, root: Path = ROOT, bindings: Mapping[str, str] = V15_RETAINED_FILES
) -> bytes:
    executor_raw = b""
    for relative, digest in bindings.items():
        raw = _read_regular(root / relative)
        if hashlib.sha256(raw).hexdigest() != digest:
            raise PermissionError(f"V16 retained V15 file changed: {relative}")
        if root / relative == V15_EXECUTOR_PATH:
            executor_raw = raw
    if not executor_raw:
        raise PermissionError("V16 retained executor binding is absent")
    return executor_raw


def _validate_v16_review(
    file_sha256: str,
    *,
    path: Path = V16_REVIEW_PATH,
    root: Path = ROOT,
    successor_source_paths: Sequence[str] = V16_SUCCESSOR_SOURCE_PATHS,
    successor_proof_paths: Sequence[str] = V16_SUCCESSOR_PROOF_PATHS,
) -> dict[str, Any]:
    if (
        type(file_sha256) is not str
        or len(file_sha256) != 64
        or any(character not in "0123456789abcdef" for character in file_sha256)
    ):
        raise ValueError("V16 source review SHA-256 is malformed")
    review, _ = _load_bound_json(path, file_sha256=file_sha256)
    reviewer = review.get("reviewer")
    if (
        set(review)
        != {
            "schema",
            "status",
            "implementation_author",
            "reviewer",
            "review_completed",
            "source_closure_approved",
            "exact_attempt_authorized",
            "exactly_one_v16_attempt_authorized",
            "v15_retry_authorized",
            "scientific_change_authorized",
            "output_root",
            "terminal_v15",
            "retained_v15_files",
            "runtime_delta",
            "successor_sources",
            "successor_proofs",
            "licenses",
            "content_sha256",
        }
        or review.get("schema") != V16_REVIEW_SCHEMA
        or review.get("status")
        != "different_agent_review_passed_v16_runtime_only_recovery"
        or review.get("implementation_author") != V16_IMPLEMENTATION_AUTHOR
        or not isinstance(reviewer, str)
        or not reviewer.startswith("/root/")
        or reviewer in {"/root", V16_IMPLEMENTATION_AUTHOR}
        or review.get("review_completed") is not True
        or review.get("source_closure_approved") is not True
        or review.get("exact_attempt_authorized") is not True
        or review.get("exactly_one_v16_attempt_authorized") is not True
        or review.get("v15_retry_authorized") is not False
        or review.get("scientific_change_authorized") is not False
        or review.get("output_root") != V16_OUTPUT_ROOT_RELATIVE_PATH
        or review.get("terminal_v15") != V15_TERMINAL_BINDINGS
        or review.get("retained_v15_files") != V15_RETAINED_FILES
        or review.get("runtime_delta") != V16_RUNTIME_DELTA
        or review.get("licenses") != V16_REVIEW_LICENSES
    ):
        raise PermissionError("V16 independent source review contract changed")
    sources = review.get("successor_sources")
    proofs = review.get("successor_proofs")
    if (
        not isinstance(sources, Mapping)
        or set(sources) != set(successor_source_paths)
        or not isinstance(proofs, Mapping)
        or set(proofs) != set(successor_proof_paths)
    ):
        raise PermissionError("V16 reviewed successor source set changed")
    for relative in (*successor_source_paths, *successor_proof_paths):
        binding = (sources if relative in sources else proofs)[relative]
        raw = _read_regular(root / relative)
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or binding.get("file_sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise PermissionError(f"V16 reviewed source changed: {relative}")
    return review


def _v16_review_binding(review: Mapping[str, Any], file_sha256: str) -> dict[str, str]:
    return {
        "path": str(V16_REVIEW_PATH.relative_to(ROOT)),
        "file_sha256": file_sha256,
        "content_sha256": str(review["content_sha256"]),
    }


def _make_idempotent_determinism(
    original: Callable[[int], Mapping[str, Any]], torch_module: Any
) -> Callable[[int], Mapping[str, Any]]:
    """Preserve V15 determinism while making its thread setters idempotent."""

    original_intra = torch_module.set_num_threads
    original_inter = torch_module.set_num_interop_threads

    def configure(seed: int) -> Mapping[str, Any]:
        def set_intra(value: int) -> None:
            if type(value) is not int or value != 1:
                raise PermissionError("V16 forbids a non-one intra-op request")
            observed = torch_module.get_num_threads()
            if type(observed) is not int:
                raise TypeError("V16 intra-op thread count is not an integer")
            if observed != 1:
                original_intra(1)
            observed = torch_module.get_num_threads()
            if type(observed) is not int or observed != 1:
                raise RuntimeError("V16 could not configure one intra-op thread")

        def set_inter(value: int) -> None:
            if type(value) is not int or value != 1:
                raise PermissionError("V16 forbids a non-one inter-op request")
            observed = torch_module.get_num_interop_threads()
            if type(observed) is not int:
                raise TypeError("V16 inter-op thread count is not an integer")
            if observed != 1:
                original_inter(1)
            observed = torch_module.get_num_interop_threads()
            if type(observed) is not int or observed != 1:
                raise RuntimeError("V16 could not configure one inter-op thread")

        torch_module.set_num_threads = set_intra
        torch_module.set_num_interop_threads = set_inter
        try:
            receipt = original(seed)
        finally:
            torch_module.set_num_threads = original_intra
            torch_module.set_num_interop_threads = original_inter
        if (
            type(torch_module.get_num_threads()) is not int
            or torch_module.get_num_threads() != 1
            or type(torch_module.get_num_interop_threads()) is not int
            or torch_module.get_num_interop_threads() != 1
        ):
            raise RuntimeError("V16 determinism returned with non-one threads")
        return receipt

    return configure


@contextmanager
def _retained_review_paths(policy: Any, retained: Mapping[str, Path]) -> Iterator[None]:
    names = tuple(retained)
    current = {name: getattr(policy, name) for name in names}
    try:
        for name, value in retained.items():
            setattr(policy, name, value)
        yield
    finally:
        for name, value in current.items():
            setattr(policy, name, value)


def _install_v16_runtime_adapter(
    policy: Any,
    base: Any | None,
    gpu_visibility: Any,
    *,
    v16_review_binding: Mapping[str, str],
) -> Callable[[], None]:
    retained_paths = {
        "CANONICAL_OUTPUT_ROOT": policy.CANONICAL_OUTPUT_ROOT,
        "CANONICAL_ATTEMPT_PATH": policy.CANONICAL_ATTEMPT_PATH,
        "CANONICAL_METRIC_RECEIPT_PATH": policy.CANONICAL_METRIC_RECEIPT_PATH,
        "CANONICAL_GATE_PATH": policy.CANONICAL_GATE_PATH,
        "GPU_VISIBILITY_RECEIPT_PATH": policy.GPU_VISIBILITY_RECEIPT_PATH,
        "GPU_VISIBILITY_RECEIPT_SCHEMA": policy.GPU_VISIBILITY_RECEIPT_SCHEMA,
        "ATTEMPT_SCOPE": policy.ATTEMPT_SCOPE,
        "authority_bindings": policy.authority_bindings,
        "SOURCE_REVIEW_RELATIVE_PATH": policy.SOURCE_REVIEW_RELATIVE_PATH,
        "CANONICAL_SOURCE_REVIEW_PATH": policy.CANONICAL_SOURCE_REVIEW_PATH,
        "SOURCE_REVIEW_SCHEMA": policy.SOURCE_REVIEW_SCHEMA,
    }
    if retained_paths != {
        "CANONICAL_OUTPUT_ROOT": V15_OUTPUT_ROOT,
        "CANONICAL_ATTEMPT_PATH": (
            V15_OUTPUT_ROOT / "attempts/seed_20260710/n5"
        ),
        "CANONICAL_METRIC_RECEIPT_PATH": (
            V15_OUTPUT_ROOT / "metric_verifications/seed_20260710_n5.json"
        ),
        "CANONICAL_GATE_PATH": (
            V15_OUTPUT_ROOT / "gates/seed_20260710_n5.json"
        ),
        "GPU_VISIBILITY_RECEIPT_PATH": V15_GPU_VISIBILITY_RECEIPT_PATH,
        "GPU_VISIBILITY_RECEIPT_SCHEMA": V15_GPU_VISIBILITY_RECEIPT_SCHEMA,
        "ATTEMPT_SCOPE": V15_ATTEMPT_SCOPE,
        "authority_bindings": policy.authority_bindings,
        "SOURCE_REVIEW_RELATIVE_PATH": (
            "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_independent_review_2026-07-14.json"
        ),
        "CANONICAL_SOURCE_REVIEW_PATH": ROOT / (
            "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_independent_review_2026-07-14.json"
        ),
        "SOURCE_REVIEW_SCHEMA": (
            "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_source_review_v1"
        ),
    }:
        raise PermissionError("V16 retained V15 output constants changed")
    original_review = policy.preflight_source_review
    original_determinism = None if base is None else base.configure_determinism
    original_authority_bindings = policy.authority_bindings
    original_cached_receipt_path = gpu_visibility.CANONICAL_RECEIPT_PATH

    def v16_review(path: Path, file_sha256: str) -> tuple[dict[str, Any], bytes]:
        if (
            Path(path) != V16_REVIEW_PATH
            or file_sha256 != v16_review_binding["file_sha256"]
        ):
            raise PermissionError("V16 executor bound another source review")
        with _retained_review_paths(policy, retained_paths):
            original_review(
                retained_paths["CANONICAL_SOURCE_REVIEW_PATH"],
                V15_SOURCE_REVIEW_FILE_SHA256,
            )
        review = _validate_v16_review(file_sha256)
        if review.get("content_sha256") != v16_review_binding["content_sha256"]:
            raise PermissionError("V16 source review content binding changed")
        return review, _read_regular(V16_REVIEW_PATH)

    policy.CANONICAL_OUTPUT_ROOT = V16_OUTPUT_ROOT
    policy.CANONICAL_ATTEMPT_PATH = V16_ATTEMPT_PATH
    policy.CANONICAL_METRIC_RECEIPT_PATH = V16_METRIC_RECEIPT_PATH
    policy.CANONICAL_GATE_PATH = V16_GATE_PATH
    policy.GPU_VISIBILITY_RECEIPT_PATH = V16_GPU_VISIBILITY_RECEIPT_PATH
    policy.GPU_VISIBILITY_RECEIPT_SCHEMA = V16_GPU_VISIBILITY_RECEIPT_SCHEMA
    policy.ATTEMPT_SCOPE = V16_ATTEMPT_SCOPE
    policy.SOURCE_REVIEW_RELATIVE_PATH = V16_SOURCE_REVIEW_RELATIVE_PATH
    policy.CANONICAL_SOURCE_REVIEW_PATH = V16_REVIEW_PATH
    policy.SOURCE_REVIEW_SCHEMA = V16_REVIEW_SCHEMA
    gpu_visibility.CANONICAL_RECEIPT_PATH = V16_GPU_VISIBILITY_RECEIPT_PATH

    def v16_authority_bindings() -> dict[str, Any]:
        return {
            **original_authority_bindings(),
            "v16_runtime_recovery_review": dict(v16_review_binding),
            "v16_terminal_v15_evidence": dict(V15_TERMINAL_BINDINGS),
            "v16_compatibility_boundary": {
                "retained_v15_schemas": True,
                "retained_v15_transaction": True,
                "retained_v15_science": True,
                "retained_v15_license_is_not_v16_authority": True,
                "v15_retry_authorized": False,
            },
        }

    policy.authority_bindings = v16_authority_bindings
    policy.preflight_source_review = v16_review
    if base is not None and original_determinism is not None:
        base.configure_determinism = _make_idempotent_determinism(
            original_determinism, base.torch
        )

    def restore() -> None:
        if base is not None and original_determinism is not None:
            base.configure_determinism = original_determinism
        policy.preflight_source_review = original_review
        gpu_visibility.CANONICAL_RECEIPT_PATH = original_cached_receipt_path
        for name, value in retained_paths.items():
            setattr(policy, name, value)

    return restore


def _source_review_argument(argv: Sequence[str]) -> str | None:
    positions = [
        index for index, value in enumerate(argv) if value == "--source-review-sha256"
    ]
    if len(positions) > 1:
        raise ValueError("V16 source review argument is duplicated")
    if not positions:
        return None
    index = positions[0]
    if index + 1 >= len(argv):
        raise ValueError("V16 source review argument lacks a value")
    return argv[index + 1]


def _mode_requires_review(argv: Sequence[str]) -> bool:
    if argv in (["--cpu-contract-smoke"], ["--cpu-verifier-contract-smoke"]):
        return False
    if argv in (["--help"], ["-h"]):
        return False
    if argv == ["--verification-child"] and os.environ.get(
        "LEWM_V15_VERIFIER_MODE"
    ) == "cpu_contract_smoke_v1":
        return False
    return True


def _isolated_visibility_preflight(argv: Sequence[str]) -> int:
    environment = dict(os.environ)
    for name in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
        "CUDA_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_DEVICE_ORDINAL",
        "HSA_VISIBLE_DEVICES",
        "HSA_OVERRIDE_GFX_VERSION",
        "NVIDIA_VISIBLE_DEVICES",
        "ONEAPI_DEVICE_SELECTOR",
        "ZE_AFFINITY_MASK",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["HIP_VISIBLE_DEVICES"] = "0"
    for name in THREAD_ENVIRONMENT:
        environment[name] = "1"
    completed = subprocess.run(
        [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *argv],
        cwd=ROOT,
        env=environment,
        check=False,
    )
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    visibility_preflight = bool(
        raw_argv and raw_argv[0] == "--v16-gpu-visibility-preflight"
    )
    if visibility_preflight and (
        len(raw_argv) != 3 or raw_argv[1] != "--source-review-sha256"
    ):
        raise ValueError(
            "V16 visibility preflight requires exactly one source-review digest"
        )
    if visibility_preflight and not sys.flags.isolated:
        return _isolated_visibility_preflight(raw_argv)
    retained_argv = raw_argv[1:] if visibility_preflight else raw_argv
    needs_review = _mode_requires_review(retained_argv)
    review_digest = _source_review_argument(retained_argv)
    if retained_argv == ["--verification-child"] and review_digest is None:
        review_digest = os.environ.get(V16_REVIEW_ENVIRONMENT)
    terminal_before: dict[str, str] | None = None
    review_binding: dict[str, str] | None = None
    if needs_review:
        if review_digest is None:
            raise PermissionError("V16 exact execution lacks its independent review")
        review = _validate_v16_review(review_digest)
        review_binding = _v16_review_binding(review, review_digest)
        terminal_before = _validate_terminal_v15()
        os.environ[V16_REVIEW_ENVIRONMENT] = review_digest

    executor_raw = _validate_retained_v15_files()
    if not needs_review:
        previous_argv = sys.argv
        sys.argv = [str(V15_EXECUTOR_PATH), *retained_argv]
        try:
            exec(
                compile(executor_raw, str(V15_EXECUTOR_PATH), "exec"),
                {
                    "__name__": "__main__",
                    "__file__": str(V15_EXECUTOR_PATH),
                    "__package__": None,
                    "__cached__": None,
                },
            )
        finally:
            sys.argv = previous_argv
        return 0
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from lewm.benchmarks import (
        go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15 as policy,
    )
    from scripts import (
        preflight_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_gpu_visibility
        as gpu_visibility,
    )

    base = None
    if not visibility_preflight:
        from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base
    if review_binding is None:
        raise PermissionError("V16 internal review binding is absent")
    restore = _install_v16_runtime_adapter(
        policy,
        base,
        gpu_visibility,
        v16_review_binding=review_binding,
    )
    if visibility_preflight:
        try:
            receipt, passed = gpu_visibility.run_diagnostic(review_digest)
            print(policy.canonical_json_bytes(receipt).decode("ascii"))
            return 0 if passed else 2
        finally:
            restore()
            if terminal_before is not None and _validate_terminal_v15() != terminal_before:
                raise PermissionError("V15 terminal evidence changed during V16 preflight")
    previous_argv = sys.argv
    sys.argv = [str(Path(__file__).resolve()), *retained_argv]
    namespace = {
        "__name__": "__main__",
        "__file__": str(Path(__file__).resolve()),
        "__package__": None,
        "__cached__": None,
    }
    try:
        exec(compile(executor_raw, str(V15_EXECUTOR_PATH), "exec"), namespace)
    finally:
        sys.argv = previous_argv
        restore()
        if terminal_before is not None and _validate_terminal_v15() != terminal_before:
            raise PermissionError("V15 terminal evidence changed during V16 execution")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
