#!/usr/bin/env python3
"""Lean, serialized Camera V4 gate-aligned development ladder.

One reviewed runner derives and executes exactly one of eight fixed rows per
invocation.  It binds terminal V16, uses fresh initialization and fixed compute,
and advances only after an isolated strict checkpoint reload reproduces the
matched/wrong controls and the frozen retained numeric gate.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from io import BytesIO
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence
import warnings


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/gate_aligned_ladder_v1"
)
OUTPUT_ROOT = ROOT / OUTPUT_ROOT_RELATIVE_PATH
ROWS_ROOT = OUTPUT_ROOT / "rows"
SOURCE_REVIEW_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_"
    "independent_review_2026-07-14.json"
)
SOURCE_REVIEW_PATH = ROOT / SOURCE_REVIEW_RELATIVE_PATH
SOURCE_REVIEW_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_"
    "source_review_v1"
)
IMPLEMENTATION_AUTHOR = "/root"
RUNNER_RELATIVE_PATH = (
    "scripts/run_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1.py"
)
TEST_RELATIVE_PATH = (
    "lewm/tests/test_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1.py"
)
PROOF_RELATIVE_PATH = (
    "docs/lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_"
    "preregistration_and_handoff_2026-07-14.md"
)
SUCCESSOR_SOURCES = (RUNNER_RELATIVE_PATH, TEST_RELATIVE_PATH)
SUCCESSOR_PROOFS = (PROOF_RELATIVE_PATH,)
RUNTIME_SOURCE_BINDINGS = {
    "scripts/train_go2_observable_camera_ray_fit_v4_v2.py": (
        "c9d22fb38acdf5fd3099271661dc65bb9cea989426a3b6021ad28649d6dd74d3"
    ),
    "scripts/launch_go2_observable_camera_ray_fit_v4_v2.py": (
        "65c58e36cb97d155a58ec1cbc93a1f2f42a75e62f049b5d8e874481a435a614b"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_ladder_gate.py": (
        "aa51413edfea10a2d7c04b034033c83c78c27b1c08d2be1413f5917dc32e36ad"
    ),
    "lewm/benchmarks/go2_observable_camera_ray_fit_v4_metrics.py": (
        "6a0e40f9dcb496831553dc5bbc6d1efcdf6d82676d6f18aa20e417f8de4fa6a0"
    ),
    "lewm/models/observable_camera_ray_evidence_v4.py": (
        "6238f7fb2b9c0c5201c9d7ebb5343ceef72fa97b423dddb466465b6c594cc882"
    ),
    "lewm/models/observable_camera_ray_evidence_v4_training.py": (
        "c0f3f944883987950edb7579a9e108171486122a9a3ae9d84d2a1abb6ac015ed"
    ),
    "lewm/models/observable_camera_ray_evidence_v4_hierarchical_first_hit_v9.py": (
        "52bc99f0ba59c2cf7444221931169ba57af61f343308b85625877c7a257adffd"
    ),
    "lewm/models/observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12.py": (
        "735563f811c5d7b9efb9e37dca8348825a8467bd0a059f83ab94d41d45d57662"
    ),
}
THRESHOLD_CONTRACT_SHA256 = (
    "408b10d8dc4f3734acb8ba17e974da4a84108a9c964d9b10787e7df59b165c60"
)
ROW_THRESHOLD_SHA256 = {
    5: "1fdf04cab1a0359b14509f1c0ade53a83e58b69e2e32e47315c919ea37687811",
    16: "5935a458c9331ef299de9f54a4d0774bb25053f332f1c8d6925dc09a665ba46c",
    32: "c238eaa65ba378552a317d6111514c92aa3613895e37dadf631cfe237590f392",
    320: "9b402dfaf0339dca5085e26b455c746b68e19332034233a8676432a1b8bac4ff",
}

ROW_GATE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_row_gate_v1"
)
ROW_FAILURE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_failure_v1"
)
ROW_RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_reservation_v1"
)
ROW_RESULT_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_result_v1"
)
ROW_COMPLETION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_completion_v1"
)
ROW_METRIC_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_"
    "metric_verification_v1"
)
FINAL_GATE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_gate_aligned_ladder_v1_"
    "final_gate_v1"
)
FINAL_GATE_FILENAME = "ladder_gate.json"
V16_RESERVATION_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_"
    "reservation_v1"
)
V16_FAILURE_SCHEMA = (
    "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_"
    "failure_v1"
)
THREAD_ENVIRONMENT = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)
UNSET_DEVICE_SELECTORS = (
    "CUDA_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "GPU_DEVICE_ORDINAL",
    "HSA_VISIBLE_DEVICES",
    "NVIDIA_VISIBLE_DEVICES",
    "ONEAPI_DEVICE_SELECTOR",
    "ZE_AFFINITY_MASK",
    "HSA_OVERRIDE_GFX_VERSION",
)
TERMINAL_V16_VISIBILITY_PATH = Path(
    "/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_"
    "v16_gpu_visibility_preflight_2026-07-14.json"
)
EXPECTED_GATE_CHECK_COUNT = 26
LOSS_COMPONENTS = (
    "hierarchical_first_hit_nll",
    "target_bin_offset_smooth_l1",
    "ground_clear_distance_state_balanced_bce",
    "derived_raster_hierarchical_bce",
    "derived_raster_cell_nll",
)
RETAINED_LOSS_COMPONENTS = LOSS_COMPONENTS[:-1]
DATA_BINDINGS = {
    "dataset_manifest": {
        "path": ".generated/go2_observable_camera_ray_fit_v4/v1/manifest.json",
        "file_sha256": "2ed32d0c385756ae1b56b2d4bd8871f8d6e6513aac97d19f737cdba2b8668c85",
        "content_sha256": "9be0c1539897bd731d4dfaf96e03b5d5c1d31d8cb8c723a2b77ffde57baf2812",
    },
    "audit_receipt": {
        "path": ".generated/go2_observable_camera_ray_fit_v4/v1/audit_result.json",
        "file_sha256": "2d6c81d6603d1baad03c4a9dadf26cf7d0ad0bfe5c2f45eb1742eb4c3d869f7c",
        "content_sha256": "a922114b7e42552043a487bae527c35fb511804d4e8683c5a3f64a2bf499cf76",
    },
    "trainer_authorization": {
        "path": "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_authorization_bound_2026-07-12.json",
        "file_sha256": "d0de4c81bce27f38ea4a477808eae7dcbb1cf8bac15e9294c3dabbf08d05d802",
        "content_sha256": "18a285e80252d41de7daadba918a00223d8770b71c533f74807e0ace5444ac1e",
    },
    "trainer_review": {
        "path": "docs/lewm_go2_observable_camera_ray_fit_v4_trainer_review_record_2026-07-12.json",
        "file_sha256": "c93b01bdc4220c5d8e70bfcb5181b4239525c9de152f95d109aae207144733ea",
        "content_sha256": "ab55270986268c5a326eeb6ba191cd9a0531112b1b742812d2cbd549f67158be",
    },
    "rgb_receipt_content_sha256": (
        "d763d7ae294e4e5a9e5f2352672913bc06411388d92abe1fb0f5090dfc41d5c3"
    ),
}
SUBSET_CONTENT_SHA256 = {
    5: "3595dff9d24dbb44f3e73086fce3be4ec53eb8659684738defa8591c4a375f15",
    16: "3e3706c4d46476c9d6682e92bd80aa97bd7b0f0bd5bc2c9b69b9aa3605f9d4ba",
    32: "19ae70495e7a21e4ecacd7846672145ffc0187ced6b4f9296c7f9e5b4d46ed73",
    320: "be4b8863120d67132180228982f0631f5f8f6042b581ee5f8a61559fa58188b1",
}
TARGET_PARTITION_CONTENT_SHA256 = {
    5: "ac9d6e1c91ca58c1182fa5e05d3189a6dc319013c3dc07e2f229f88c55cca429",
    16: "93d69ccdaf528d91bb43d1131f19886fbd29c4d249157dfc8bdcc191732c686f",
    32: "6552c94f2c737fe586093d603b8a9fd0fae4d036ec2d800e81265ecbd913c92d",
    320: "acd7d9d610c55d6a2c2efc9b54925309d361f648450c39006a3aa4bc8637b442",
}


@dataclass(frozen=True)
class LadderRow:
    index: int
    seed: int
    fit_size: int
    updates: int
    batch_size: int
    frame_exposures: int
    schedule_sha256: str

    @property
    def key(self) -> str:
        return f"row_{self.index:02d}_seed_{self.seed}_n{self.fit_size}"


_ROW_VALUES = (
    (0, 20260710, 5, "fb5a6c13708944b6ce514960c11c063eece704676276b352bd5233080f4fd380"),
    (1, 20260710, 16, "06f3ab002349bb8726d1abd7ae5350de711938b67a8ea7e7da7ae66145f9248e"),
    (2, 20260710, 32, "5d93d4d4f4697635170a3739557ccdddc7da0e0bc9e874438802cf65298627fc"),
    (3, 20260710, 320, "4084f8d5c14989cb76df4f01e4de46b0b6a88537ba607ccc4152795304bc3bd6"),
    (4, 20260711, 5, "829d366eb9dcefdaad66596413939da455209909009e29177271ff5ed9c76c2e"),
    (5, 20260711, 16, "57d5cd679ab7eb99654430a166a53985c21ffcf261faa70c8df0357ac7dc80f3"),
    (6, 20260711, 32, "405632e5e6c8e26590debfa5139090ca89c4ac262930ca286ce82e9b9db1f10c"),
    (7, 20260711, 320, "2b5475b725f1ae3c956adaef0a72153b0fdafd6d1ba36d827ed23792ac6a0b9a"),
)
LADDER_ROWS = tuple(
    LadderRow(
        index=index,
        seed=seed,
        fit_size=fit_size,
        updates=4000,
        batch_size=5,
        frame_exposures=20000,
        schedule_sha256=schedule_sha256,
    )
    for index, seed, fit_size, schedule_sha256 in _ROW_VALUES
)

TERMINAL_V16_ROOT_RELATIVE_PATH = (
    ".generated/go2_observable_camera_ray_fit_v4/"
    "n5_gate_aligned_raster_nll_v16"
)
TERMINAL_V16_BINDINGS: dict[str, Any] = {
    "reservation": {
        "path": "attempts/seed_20260710/n5/reservation.json",
        "file_sha256": "1769d282f528c6c64b1fb67ad229c6ebf2dbc55ae61b1a53451a76538a69bf1c",
        "content_sha256": "00fdc565b3791579ca4c6bbc090eac8db2d87b3e54d37e647b46dc9780a28e15",
        "byte_count": 30965,
    },
    "failure": {
        "path": "attempts/seed_20260710/n5/failed.json",
        "file_sha256": "06ae522dc0748d6d0857e5d8cfd22d96fbc78e5e1463c30c8928670d2c22dd51",
        "content_sha256": "c861eca6b88abe469ab73b29f0499f2a6e549d16c6f4ad266aaa9eb3dc8f49d5",
        "byte_count": 1312,
    },
    "lock": {
        "path": "attempts/seed_20260710/.n5.reservation-v15.lock",
        "file_sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "byte_count": 0,
    },
    "attempt_inventory": ["failed.json", "reservation.json"],
    "seed_inventory": [".n5.reservation-v15.lock", "n5"],
    "root_inventory": ["attempts", "gates", "metric_verifications"],
    "visibility_receipt": {
        "path": str(TERMINAL_V16_VISIBILITY_PATH),
        "file_sha256": "cbda43a1b251d48eb400e263bd6e81645d02d44c9630a513b368de821c87545a",
        "content_sha256": "06c72c6275bbb9101753774189b0987e12cfcf4e57cbcbe1329299f12b6df2ec",
        "byte_count": 3817,
    },
}

REVIEW_LICENSES = {
    "eight_serial_development_rows_authorized": True,
    "v16_retry_authorized": False,
    "threshold_change_authorized": False,
    "data_change_authorized": False,
    "predecessor_checkpoint_use_authorized": False,
    "heldout_authorized": False,
    "g2_authorized": False,
    "navigation_authorized": False,
    "production_authorized": False,
    "promotion_authorized": False,
}


class LadderStopped(RuntimeError):
    """The fixed ladder has a terminal failure or incomplete prior row."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def read_regular(path: Path) -> bytes:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise PermissionError("ladder requires no-follow file opens")
    descriptor = os.open(path, os.O_RDONLY | nofollow | getattr(os, "O_CLOEXEC", 0))
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"not a private regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        fingerprint = lambda value: (
            value.st_dev,
            value.st_ino,
            value.st_mode,
            value.st_nlink,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
        if fingerprint(before) != fingerprint(after):
            raise PermissionError(f"file changed while read: {path}")
        raw = b"".join(chunks)
        if len(raw) != before.st_size:
            raise PermissionError(f"file was truncated: {path}")
        return raw
    finally:
        os.close(descriptor)


def load_bound_json(
    path: Path,
    *,
    file_sha256: str | None = None,
    content_sha256: str | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path)
    if file_sha256 is not None and hashlib.sha256(raw).hexdigest() != file_sha256:
        raise PermissionError(f"file binding changed: {path}")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"malformed JSON: {path}") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"JSON is not a canonical object: {path}")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError(f"content hash changed: {path}")
    if content_sha256 is not None and declared != content_sha256:
        raise PermissionError(f"content binding changed: {path}")
    return value, raw


def artifact_binding(path: str, raw: bytes, *, content_sha256: str) -> dict[str, Any]:
    return {
        "path": path,
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "content_sha256": content_sha256,
        "byte_count": len(raw),
    }


def terminal_v16_summary_contract(
    bindings: Mapping[str, Any] = TERMINAL_V16_BINDINGS,
    *,
    visibility_path: Path | None = None,
) -> dict[str, Any]:
    receipt_path = (
        str(bindings["visibility_receipt"]["path"])
        if visibility_path is None
        else str(visibility_path)
    )
    return {
        "reservation": {
            key: bindings["reservation"][key]
            for key in ("path", "file_sha256", "content_sha256", "byte_count")
        },
        "failure": {
            key: bindings["failure"][key]
            for key in ("path", "file_sha256", "content_sha256", "byte_count")
        },
        "lock_file_sha256": bindings["lock"]["file_sha256"],
        "visibility_receipt": {
            "path": receipt_path,
            **{
                key: bindings["visibility_receipt"][key]
                for key in ("file_sha256", "content_sha256", "byte_count")
            },
        },
        "numeric_outputs_observed_or_persisted": False,
        "terminal_failure_stage": "training",
        "v16_retry_authorized": False,
    }


def row_contract() -> list[dict[str, Any]]:
    return [
        asdict(row)
        | {
            "key": row.key,
            "threshold_sha256": ROW_THRESHOLD_SHA256[row.fit_size],
            "subset_content_sha256": SUBSET_CONTENT_SHA256[row.fit_size],
            "target_partition_content_sha256": TARGET_PARTITION_CONTENT_SHA256[
                row.fit_size
            ],
        }
        for row in LADDER_ROWS
    ]


def science_contract() -> dict[str, Any]:
    observed = {
        "model_class": "ObservableCameraRayEvidenceV4Model",
        "objective": "gate_aligned_raster_nll_v15_five_equal_terms",
        "loss_components": list(LOSS_COMPONENTS),
        "loss_weights": {name: 0.25 for name in LOSS_COMPONENTS},
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "precision": "float32",
        "autocast": False,
        "evaluation_batch_size": 1,
        "wrong_rgb_mapping": "cyclic_plus_one",
        "threshold_contract_sha256": THRESHOLD_CONTRACT_SHA256,
        "runtime_source_bindings": dict(RUNTIME_SOURCE_BINDINGS),
        "data_bindings": DATA_BINDINGS,
    }
    return observed


def initialization_identity(
    row: LadderRow,
    attempt_identity: str,
    initial_state_sha256: str,
) -> str:
    if not is_sha256(attempt_identity) or not is_sha256(initial_state_sha256):
        raise ValueError("attempt or initial-state digest is malformed")
    return canonical_json_sha256(
        {
            "schema": "lewm_go2_camera_ladder_initialization_identity_v1",
            "row": asdict(row) | {"key": row.key},
            "attempt_identity": attempt_identity,
            "initial_state_sha256": initial_state_sha256,
            "fresh_model_construction": True,
            "predecessor_checkpoint_opens": 0,
        }
    )


def validate_terminal_v16(
    *,
    root: Path = ROOT / TERMINAL_V16_ROOT_RELATIVE_PATH,
    bindings: Mapping[str, Any] = TERMINAL_V16_BINDINGS,
    visibility_path: Path | None = None,
) -> dict[str, Any]:
    reservation_binding = bindings["reservation"]
    failure_binding = bindings["failure"]
    lock_binding = bindings["lock"]
    reservation, reservation_raw = load_bound_json(
        root / reservation_binding["path"],
        file_sha256=reservation_binding["file_sha256"],
        content_sha256=reservation_binding["content_sha256"],
    )
    failure, failure_raw = load_bound_json(
        root / failure_binding["path"],
        file_sha256=failure_binding["file_sha256"],
        content_sha256=failure_binding["content_sha256"],
    )
    lock_path = root / lock_binding["path"]
    lock_raw = read_regular(lock_path)
    lock_stat = lock_path.stat(follow_symlinks=False)
    if (
        hashlib.sha256(lock_raw).hexdigest() != lock_binding["file_sha256"]
        or len(lock_raw) != lock_binding["byte_count"]
        or stat.S_IMODE(lock_stat.st_mode) != 0o600
    ):
        raise PermissionError("terminal V16 lock changed")
    expected_reservation = {
        "path": "reservation.json",
        "file_sha256": reservation_binding["file_sha256"],
        "content_sha256": reservation_binding["content_sha256"],
        "byte_count": reservation_binding["byte_count"],
    }
    if (
        len(reservation_raw) != reservation_binding["byte_count"]
        or reservation.get("schema") != V16_RESERVATION_SCHEMA
        or reservation.get("scope")
        != "one_exclusive_fresh_gate_aligned_raster_nll_v16_attempt"
        or reservation.get("seed") != 20260710
        or reservation.get("fit_size") != 5
        or len(failure_raw) != failure_binding["byte_count"]
        or failure.get("schema") != V16_FAILURE_SCHEMA
        or failure.get("status") != "failed"
        or failure.get("failure_stage") != "training"
        or failure.get("failure")
        != {"class": "permission", "code": "scope_or_authorization_failure"}
        or failure.get("reservation") != expected_reservation
        or failure.get("partial_artifacts_removed") is not True
        or failure.get("artifact_cleanup") != []
        or failure.get("diagnostic_publication_succeeded") is not False
        or failure.get("verification_failure") is not None
        or failure.get("retry_authorized") is not False
    ):
        raise PermissionError("terminal V16 semantics changed")
    attempt = root / "attempts/seed_20260710/n5"
    seed_root = attempt.parent
    if sorted(item.name for item in attempt.iterdir()) != sorted(
        bindings["attempt_inventory"]
    ):
        raise PermissionError("terminal V16 attempt inventory changed")
    if sorted(item.name for item in seed_root.iterdir()) != sorted(
        bindings["seed_inventory"]
    ):
        raise PermissionError("terminal V16 seed inventory changed")
    if sorted(item.name for item in root.iterdir()) != sorted(
        bindings["root_inventory"]
    ):
        raise PermissionError("terminal V16 root inventory changed")
    for name in ("gates", "metric_verifications"):
        directory = root / name
        if directory.is_symlink() or not directory.is_dir() or any(directory.iterdir()):
            raise PermissionError("terminal V16 derived directory changed")
    visibility_binding = bindings["visibility_receipt"]
    receipt_path = (
        Path(str(visibility_binding["path"]))
        if visibility_path is None
        else visibility_path
    )
    visibility, visibility_raw = load_bound_json(
        receipt_path,
        file_sha256=visibility_binding["file_sha256"],
        content_sha256=visibility_binding["content_sha256"],
    )
    zero_access = visibility.get("zero_access_evidence")
    authority = visibility.get("authority")
    selectors = visibility.get("selector_observation")
    runtime = visibility.get("runtime_observation")
    threads = visibility.get("native_thread_observation")
    if (
        len(visibility_raw) != visibility_binding["byte_count"]
        or visibility.get("schema")
        != (
            "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_"
            "nll_v16_gpu_visibility_preflight_v1"
        )
        or visibility.get("status") != "passed"
        or visibility.get("disposition") != "pass_exactly_one_r9700"
        or type(zero_access) is not dict
        or not zero_access
        or any(type(value) is not int or value != 0 for value in zero_access.values())
        or type(authority) is not dict
        or not authority
        or any(value is not False for value in authority.values())
        or type(selectors) is not dict
        or selectors.get("hip_visible_devices") != "0"
        or selectors.get("hsa_override_gfx_version") is not None
        or any(
            value is not None
            for value in selectors.get("conflicting_selectors", {}).values()
        )
        or runtime
        != {
            "enumeration_completed": True,
            "gpu1_absent": True,
            "ordered_devices": [
                {"logical_ordinal": 0, "name": "AMD Radeon AI PRO R9700"}
            ],
            "raphael_absent": True,
            "runtime_available": True,
            "visible_device_count": 1,
        }
        or type(threads) is not dict
        or threads.get("environment") != {name: "1" for name in THREAD_ENVIRONMENT}
        or threads.get("torch_inter_op") != 1
        or threads.get("torch_intra_op") != 1
    ):
        raise PermissionError("terminal V16 visibility receipt changed")
    observed = {
        "reservation": artifact_binding(
            reservation_binding["path"],
            reservation_raw,
            content_sha256=reservation["content_sha256"],
        ),
        "failure": artifact_binding(
            failure_binding["path"],
            failure_raw,
            content_sha256=failure["content_sha256"],
        ),
        "lock_file_sha256": hashlib.sha256(lock_raw).hexdigest(),
        "visibility_receipt": artifact_binding(
            str(receipt_path),
            visibility_raw,
            content_sha256=visibility["content_sha256"],
        ),
        "numeric_outputs_observed_or_persisted": False,
        "terminal_failure_stage": "training",
        "v16_retry_authorized": False,
    }
    if observed != terminal_v16_summary_contract(
        bindings,
        visibility_path=receipt_path,
    ):
        raise RuntimeError("terminal V16 summary construction changed")
    return observed


def validate_source_review(
    file_sha256: str,
    *,
    path: Path = SOURCE_REVIEW_PATH,
    root: Path = ROOT,
) -> tuple[dict[str, Any], bytes]:
    if not is_sha256(file_sha256):
        raise ValueError("source review digest is malformed")
    review, raw = load_bound_json(path, file_sha256=file_sha256)
    expected_fields = {
        "schema",
        "status",
        "implementation_author",
        "reviewer",
        "review_completed",
        "source_closure_approved",
        "runtime_complete",
        "eight_serial_rows_authorized",
        "output_root",
        "rows",
        "terminal_v16",
        "successor_sources",
        "successor_proofs",
        "runtime_sources",
        "science_contract",
        "licenses",
        "content_sha256",
    }
    reviewer = review.get("reviewer")
    if (
        set(review) != expected_fields
        or review.get("schema") != SOURCE_REVIEW_SCHEMA
        or review.get("status") != "different_agent_review_passed_lean_ladder_v1"
        or review.get("implementation_author") != IMPLEMENTATION_AUTHOR
        or type(reviewer) is not str
        or not reviewer.startswith("/root/")
        or reviewer in {"/root", IMPLEMENTATION_AUTHOR}
        or review.get("review_completed") is not True
        or review.get("source_closure_approved") is not True
        or review.get("runtime_complete") is not True
        or review.get("eight_serial_rows_authorized") is not True
        or review.get("output_root") != OUTPUT_ROOT_RELATIVE_PATH
        or review.get("rows") != row_contract()
        or review.get("terminal_v16") != TERMINAL_V16_BINDINGS
        or review.get("science_contract") != science_contract()
        or review.get("licenses") != REVIEW_LICENSES
    ):
        raise PermissionError("source review contract changed")
    sources = review.get("successor_sources")
    proofs = review.get("successor_proofs")
    runtime_sources = review.get("runtime_sources")
    if (
        type(sources) is not dict
        or set(sources) != set(SUCCESSOR_SOURCES)
        or type(proofs) is not dict
        or set(proofs) != set(SUCCESSOR_PROOFS)
        or type(runtime_sources) is not dict
        or set(runtime_sources) != set(RUNTIME_SOURCE_BINDINGS)
    ):
        raise PermissionError("reviewed closure inventory changed")
    for relative in (*SUCCESSOR_SOURCES, *SUCCESSOR_PROOFS):
        binding = (sources if relative in sources else proofs)[relative]
        source_raw = read_regular(root / relative)
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or binding.get("file_sha256") != hashlib.sha256(source_raw).hexdigest()
        ):
            raise PermissionError(f"reviewed source changed: {relative}")
    for relative, expected_sha256 in RUNTIME_SOURCE_BINDINGS.items():
        binding = runtime_sources[relative]
        source_raw = read_regular(root / relative)
        if (
            type(binding) is not dict
            or set(binding) != {"path", "file_sha256"}
            or binding.get("path") != relative
            or binding.get("file_sha256") != expected_sha256
            or hashlib.sha256(source_raw).hexdigest() != expected_sha256
        ):
            raise PermissionError(f"reviewed runtime source changed: {relative}")
    return review, raw


def source_review_binding(review: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return artifact_binding(
        SOURCE_REVIEW_RELATIVE_PATH,
        raw,
        content_sha256=str(review["content_sha256"]),
    )


def _validate_artifact_map(value: object) -> dict[str, dict[str, Any]]:
    expected = {
        "reservation",
        "checkpoint",
        "result",
        "completion",
        "metric_verification",
    }
    if type(value) is not dict or set(value) != expected:
        raise PermissionError("row artifact map changed")
    normalized: dict[str, dict[str, Any]] = {}
    expected_paths = {
        "reservation": "reservation.json",
        "checkpoint": "checkpoint.pt",
        "result": "result.json",
        "completion": "completed.json",
        "metric_verification": "metric_verification.json",
    }
    for role, binding in value.items():
        if (
            type(binding) is not dict
            or set(binding)
            != {"path", "file_sha256", "content_sha256", "byte_count"}
            or not is_sha256(binding.get("file_sha256"))
            or not is_sha256(binding.get("content_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
            or binding.get("path") != expected_paths[role]
        ):
            raise PermissionError(f"row {role} binding changed")
        normalized[role] = dict(binding)
    return normalized


def expected_numeric_check_contract(
    thresholds: Mapping[str, Any],
) -> dict[str, tuple[str, float]]:
    minimum = "greater_than_or_equal"
    maximum = "less_than_or_equal"
    result = {
        "matched.pixel_hit_balanced_accuracy": (
            minimum,
            thresholds["pixel_hit_balanced_accuracy_min"],
        ),
        "matched.pixel_hit_depth_median_absolute_error_m": (
            maximum,
            thresholds["pixel_hit_depth_median_error_m_max"],
        ),
        "matched.pixel_hit_depth_p95_absolute_error_m": (
            maximum,
            thresholds["pixel_hit_depth_p95_error_m_max"],
        ),
        "matched.ground_overall_balanced_accuracy": (
            minimum,
            thresholds["ground_overall_balanced_accuracy_min"],
        ),
        "matched.raster_nll": (maximum, thresholds["raster_nll_max"]),
        "matched.raster_balanced_accuracy": (
            minimum,
            thresholds["raster_balanced_accuracy_min"],
        ),
        "wrong_rgb.pixel_balanced_accuracy_drop": (
            minimum,
            thresholds["wrong_pixel_balanced_accuracy_drop_min"],
        ),
        "wrong_rgb.depth_median_error_increase_m": (
            minimum,
            thresholds["wrong_depth_median_error_increase_m_min"],
        ),
        "wrong_rgb.depth_p95_error_increase_m": (
            minimum,
            thresholds["wrong_depth_p95_error_increase_m_min"],
        ),
        "wrong_rgb.ground_balanced_accuracy_drop": (
            minimum,
            thresholds["wrong_ground_balanced_accuracy_drop_min"],
        ),
        "wrong_rgb.raster_nll_increase": (
            minimum,
            thresholds["wrong_raster_nll_increase_min"],
        ),
        "wrong_rgb.raster_balanced_accuracy_drop": (
            minimum,
            thresholds["wrong_raster_balanced_accuracy_drop_min"],
        ),
    }
    for group in (
        "0.0_to_1.0",
        "1.0_to_2.0",
        "2.0_to_3.0",
        "3.0_to_4.0",
        "4.0_to_5.0",
        "5.0_plus",
    ):
        result[f"matched.ground_distance.{group}.balanced_accuracy"] = (
            minimum,
            thresholds["ground_distance_balanced_accuracy_min"],
        )
    for family in (
        "open_obstacle_field",
        "rough_local_dynamics",
        "small_enclosed_maze",
        "medium_enclosed_maze",
        "large_enclosed_maze",
    ):
        result[f"matched.ground_family.{family}.balanced_accuracy"] = (
            minimum,
            thresholds["ground_family_balanced_accuracy_min"],
        )
    for class_name in ("unknown", "free", "occupied"):
        result[f"matched.raster_recall.{class_name}"] = (
            minimum,
            thresholds["raster_class_recall_min"],
        )
    if len(result) != EXPECTED_GATE_CHECK_COUNT:
        raise AssertionError("frozen numeric check inventory is not 26")
    return {name: (comparison, float(threshold)) for name, (comparison, threshold) in result.items()}


def validate_numeric_gate(value: object, *, row: LadderRow) -> dict[str, Any]:
    fields = {
        "fit_size",
        "thresholds",
        "wrong_rgb_dependence_assessable",
        "check_count",
        "checks",
        "failure_count",
        "failed_checks",
        "passes",
    }
    if type(value) is not dict or set(value) != fields:
        raise PermissionError(f"numeric gate fields changed: {row.key}")
    checks = value.get("checks")
    thresholds = value.get("thresholds")
    if (
        value.get("fit_size") != row.fit_size
        or canonical_json_sha256(thresholds)
        != ROW_THRESHOLD_SHA256[row.fit_size]
        or value.get("wrong_rgb_dependence_assessable") is not True
        or value.get("check_count") != EXPECTED_GATE_CHECK_COUNT
        or type(checks) is not list
        or len(checks) != EXPECTED_GATE_CHECK_COUNT
    ):
        raise PermissionError(f"numeric gate contract changed: {row.key}")
    expected_checks = expected_numeric_check_contract(thresholds)
    normalized_checks = []
    seen_names: set[str] = set()
    for check in checks:
        if type(check) is not dict or set(check) != {
            "name",
            "comparison",
            "value",
            "threshold",
            "passes",
        }:
            raise PermissionError(f"numeric check fields changed: {row.key}")
        name = check.get("name")
        comparison = check.get("comparison")
        observed = check.get("value")
        threshold = check.get("threshold")
        if (
            type(name) is not str
            or not name
            or name in seen_names
            or name not in expected_checks
            or comparison != expected_checks.get(name, (None, None))[0]
            or isinstance(observed, bool)
            or not isinstance(observed, (int, float))
            or not math.isfinite(float(observed))
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(float(threshold))
            or float(threshold) != expected_checks.get(name, (None, None))[1]
        ):
            raise PermissionError(f"numeric check malformed: {row.key}")
        seen_names.add(name)
        computed = (
            float(observed) >= float(threshold)
            if comparison == "greater_than_or_equal"
            else float(observed) <= float(threshold)
        )
        if check.get("passes") is not computed:
            raise PermissionError(f"numeric check arithmetic changed: {row.key}")
        normalized_checks.append(dict(check))
    if seen_names != set(expected_checks):
        raise PermissionError(f"numeric check inventory changed: {row.key}")
    failed = [check for check in normalized_checks if not check["passes"]]
    if (
        value.get("failure_count") != len(failed)
        or value.get("failed_checks") != failed
        or value.get("passes") is not (not failed)
    ):
        raise PermissionError(f"numeric gate disposition changed: {row.key}")
    return dict(value)


def validate_row_gate(
    gate_path: Path,
    *,
    row: LadderRow,
    expected_source_review: Mapping[str, Any],
    expected_prerequisite_gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate, _ = load_bound_json(gate_path)
    expected_fields = {
        "schema",
        "status",
        "row",
        "source_review",
        "prerequisite_gates",
        "artifacts",
        "threshold_contract_sha256",
        "numeric_gate",
        "check_count",
        "failure_count",
        "passes",
        "retry_authorized",
        "content_sha256",
    }
    if (
        set(gate) != expected_fields
        or gate.get("schema") != ROW_GATE_SCHEMA
        or gate.get("status") not in {"passed", "failed_numeric_gate"}
        or gate.get("row") != (asdict(row) | {"key": row.key})
        or gate.get("source_review") != dict(expected_source_review)
        or gate.get("prerequisite_gates")
        != [dict(binding) for binding in expected_prerequisite_gates]
        or gate.get("threshold_contract_sha256") != THRESHOLD_CONTRACT_SHA256
        or gate.get("retry_authorized") is not False
    ):
        raise PermissionError(f"row gate changed: {row.key}")
    _validate_artifact_map(gate.get("artifacts"))
    numeric = validate_numeric_gate(gate.get("numeric_gate"), row=row)
    if (
        gate.get("check_count") != numeric["check_count"]
        or gate.get("failure_count") != numeric["failure_count"]
        or gate.get("passes") is not numeric["passes"]
        or gate.get("status")
        != ("passed" if numeric["passes"] else "failed_numeric_gate")
    ):
        raise PermissionError(f"row gate summary changed: {row.key}")
    return gate


def _json_file_binding(path: str, value: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return artifact_binding(
        path,
        raw,
        content_sha256=str(value["content_sha256"]),
    )


def _gate_binding(row: LadderRow, gate: Mapping[str, Any], raw: bytes) -> dict[str, Any]:
    return _json_file_binding(f"rows/{row.key}/gate.json", gate, raw)


def _final_gate_core(
    *,
    expected_source_review: Mapping[str, Any],
    expected_row_gates: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(expected_row_gates) != len(LADDER_ROWS):
        raise PermissionError("final gate requires all eight row gates")
    return {
        "schema": FINAL_GATE_SCHEMA,
        "status": "all_eight_rows_passed",
        "source_review": dict(expected_source_review),
        "rows": row_contract(),
        "row_gates": [dict(binding) for binding in expected_row_gates],
        "threshold_contract_sha256": THRESHOLD_CONTRACT_SHA256,
        "row_count": len(LADDER_ROWS),
        "all_rows_passed": True,
        "retry_authorized": False,
        "licenses": {
            "checkpoint_use_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }


def validate_final_gate(
    path: Path,
    *,
    expected_source_review: Mapping[str, Any],
    expected_row_gates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], bytes]:
    gate, raw = load_bound_json(path)
    expected = _self_hashed(
        _final_gate_core(
            expected_source_review=expected_source_review,
            expected_row_gates=expected_row_gates,
        )
    )
    if gate != expected:
        raise PermissionError("final ladder gate changed")
    return gate, raw


def publish_final_gate(
    *,
    expected_source_review: Mapping[str, Any],
    expected_row_gates: Sequence[Mapping[str, Any]],
    output_root: Path = OUTPUT_ROOT,
) -> tuple[dict[str, Any], bytes]:
    path = output_root / FINAL_GATE_FILENAME
    gate, raw = _publish_json_exclusive(
        path,
        _final_gate_core(
            expected_source_review=expected_source_review,
            expected_row_gates=expected_row_gates,
        ),
    )
    _fsync_directory(output_root)
    validated, validated_raw = validate_final_gate(
        path,
        expected_source_review=expected_source_review,
        expected_row_gates=expected_row_gates,
    )
    if validated != gate or validated_raw != raw:
        raise RuntimeError("final ladder gate did not byte-revalidate")
    return gate, raw


def _expected_prerequisite_gates(
    row: LadderRow,
    passed_gate_bindings: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    if row.index == 0:
        return []
    if row.index == 4:
        if len(passed_gate_bindings) != 4:
            raise PermissionError("second seed lacks all four first-seed gates")
        return [dict(binding) for binding in passed_gate_bindings]
    if len(passed_gate_bindings) != row.index:
        raise PermissionError("row prerequisite history is incomplete")
    return [dict(passed_gate_bindings[-1])]


def _actual_artifact_binding(
    row_directory: Path,
    role: str,
    claimed: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None, bytes]:
    path = row_directory / str(claimed["path"])
    raw = read_regular(path)
    if (
        hashlib.sha256(raw).hexdigest() != claimed["file_sha256"]
        or len(raw) != claimed["byte_count"]
    ):
        raise PermissionError(f"row {role} bytes changed")
    if role == "checkpoint":
        return dict(claimed), None, raw
    value, loaded_raw = load_bound_json(
        path,
        file_sha256=str(claimed["file_sha256"]),
        content_sha256=str(claimed["content_sha256"]),
    )
    if loaded_raw != raw:
        raise RuntimeError(f"row {role} changed between reads")
    return dict(claimed), value, raw


def _validate_reservation_record(
    reservation: Mapping[str, Any],
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    prerequisite_gates: Sequence[Mapping[str, Any]],
) -> None:
    fields = {
        "schema",
        "status",
        "row",
        "attempt_index",
        "maximum_attempts",
        "attempt_identity",
        "source_review",
        "prerequisite_gates",
        "terminal_v16",
        "science_contract_sha256",
        "inputs",
        "initialization",
        "resource",
        "determinism",
        "retry_authorized",
        "licenses",
        "content_sha256",
    }
    inputs = reservation.get("inputs")
    initialization = reservation.get("initialization")
    resource = reservation.get("resource")
    determinism = reservation.get("determinism")
    if (
        set(reservation) != fields
        or reservation.get("schema") != ROW_RESERVATION_SCHEMA
        or reservation.get("status") != "reserved"
        or reservation.get("row") != (asdict(row) | {"key": row.key})
        or reservation.get("attempt_index") != 1
        or reservation.get("maximum_attempts") != 1
        or not is_sha256(reservation.get("attempt_identity"))
        or reservation.get("source_review") != dict(source_review)
        or reservation.get("prerequisite_gates")
        != [dict(value) for value in prerequisite_gates]
        or (
            row.index == 0
            and reservation.get("terminal_v16") != terminal_v16_summary_contract()
        )
        or (row.index != 0 and reservation.get("terminal_v16") is not None)
        or reservation.get("science_contract_sha256")
        != canonical_json_sha256(science_contract())
        or type(inputs) is not dict
        or set(inputs) != {"data_bindings", "subset", "target_partition"}
        or inputs.get("data_bindings") != DATA_BINDINGS
        or inputs.get("subset", {}).get("content_sha256")
        != SUBSET_CONTENT_SHA256[row.fit_size]
        or inputs.get("target_partition", {}).get("content_sha256")
        != TARGET_PARTITION_CONTENT_SHA256[row.fit_size]
        or type(initialization) is not dict
        or set(initialization)
        != {
            "attempt_identity",
            "initial_state_sha256",
            "initialization_identity",
            "fresh_model_construction",
            "predecessor_checkpoint_opens",
        }
        or initialization.get("attempt_identity")
        != reservation.get("attempt_identity")
        or not is_sha256(initialization.get("initial_state_sha256"))
        or not is_sha256(initialization.get("initialization_identity"))
        or initialization.get("fresh_model_construction") is not True
        or initialization.get("predecessor_checkpoint_opens") != 0
        or type(resource) is not dict
        or resource.get("device") != "cuda:0"
        or resource.get("visible_device_count") != 1
        or "r9700" not in str(resource.get("device_name", "")).casefold()
        or resource.get("native_thread_environment")
        != {name: "1" for name in THREAD_ENVIRONMENT}
        or resource.get("all_conflicting_selectors_unset") is not True
        or type(determinism) is not dict
        or determinism.get("seed") != row.seed
        or determinism.get("torch_num_threads") != 1
        or determinism.get("torch_num_interop_threads") != 1
        or reservation.get("retry_authorized") is not False
        or reservation.get("licenses")
        != {
            "development_checkpoint_creation_authorized": True,
            "metric_verification_checkpoint_use_authorized": True,
            "predecessor_checkpoint_use_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError(f"row reservation changed: {row.key}")


def _validate_result_record(
    result: Mapping[str, Any],
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
) -> None:
    fields = {
        "schema",
        "status",
        "row",
        "attempt_identity",
        "source_review",
        "reservation",
        "subset",
        "target_partition",
        "initialization",
        "model",
        "training",
        "evaluation",
        "gate_evaluation",
        "gate_adapter",
        "resource",
        "determinism",
        "access_ledger",
        "licenses",
        "content_sha256",
    }
    model = result.get("model")
    training = result.get("training")
    access = result.get("access_ledger")
    if (
        set(result) != fields
        or result.get("schema") != ROW_RESULT_SCHEMA
        or result.get("status") != "completed_training"
        or result.get("row") != (asdict(row) | {"key": row.key})
        or result.get("source_review") != dict(source_review)
        or result.get("attempt_identity") != reservation["attempt_identity"]
        or result.get("reservation") != dict(reservation_binding)
        or result.get("subset") != reservation["inputs"]["subset"]
        or result.get("target_partition")
        != reservation["inputs"]["target_partition"]
        or result.get("initialization") != reservation["initialization"]
        or model
        != {
            "class": "ObservableCameraRayEvidenceV4Model",
            "fresh_initialization": True,
            "parameter_count": 3_105_513,
            "checkpoint": {**dict(checkpoint_binding), "development_only": True},
        }
        or type(training) is not dict
        or training.get("steps") != row.updates
        or training.get("batch_size") != row.batch_size
        or training.get("frame_exposures") != row.frame_exposures
        or training.get("schedule_sha256") != row.schedule_sha256
        or training.get("checkpoint_selection") != "final_update_only"
        or training.get("fresh_model_initialization") is not True
        or training.get("predecessor_checkpoint_opens") != 0
        or result.get("gate_adapter")
        != "v15_native_diagnostics_excluded_then_hierarchical_to_ordered_key_v1"
        or result.get("resource") != reservation["resource"]
        or type(result.get("determinism")) is not dict
        or type(access) is not dict
        or access.get("predecessor_checkpoint_opens") != 0
        or any(
            access.get(name) != 0
            for name in (
                "heldout_opens",
                "g2_opens",
                "navigation_opens",
                "runtime_opens",
                "production_opens",
                "gpu1_uses",
            )
        )
        or result.get("licenses")
        != {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "metric_verification_only_checkpoint_use_authorized": True,
            "retry_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError(f"row result chain changed: {row.key}")


def _validate_completion_record(
    completion: Mapping[str, Any],
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    checkpoint_binding: Mapping[str, Any],
    result_binding: Mapping[str, Any],
) -> None:
    if (
        set(completion)
        != {
            "schema",
            "status",
            "row",
            "source_review",
            "reservation",
            "checkpoint",
            "result",
            "inventory",
            "retry_authorized",
            "licenses",
            "content_sha256",
        }
        or completion.get("schema") != ROW_COMPLETION_SCHEMA
        or completion.get("status") != "completed_training"
        or completion.get("row") != (asdict(row) | {"key": row.key})
        or completion.get("source_review") != dict(source_review)
        or completion.get("reservation") != dict(reservation_binding)
        or completion.get("checkpoint") != dict(checkpoint_binding)
        or completion.get("result") != dict(result_binding)
        or completion.get("inventory")
        != [
            "checkpoint.pt",
            "completed.json",
            "gate.json",
            "metric_verification.json",
            "reservation.json",
            "result.json",
        ]
        or completion.get("retry_authorized") is not False
        or completion.get("licenses")
        != {
            "checkpoint_use_authorized": False,
            "metric_verification_only_checkpoint_use_authorized": True,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError(f"row completion chain changed: {row.key}")


def validate_completed_row_bundle(
    row_directory: Path,
    *,
    row: LadderRow,
    expected_source_review: Mapping[str, Any],
    expected_prerequisite_gates: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    gate, gate_raw = load_bound_json(row_directory / "gate.json")
    validate_row_gate(
        row_directory / "gate.json",
        row=row,
        expected_source_review=expected_source_review,
        expected_prerequisite_gates=expected_prerequisite_gates,
    )
    artifacts = _validate_artifact_map(gate["artifacts"])
    loaded: dict[str, dict[str, Any]] = {}
    raw_by_role: dict[str, bytes] = {}
    for role, claimed in artifacts.items():
        _binding, value, raw = _actual_artifact_binding(
            row_directory,
            role,
            claimed,
        )
        raw_by_role[role] = raw
        if value is not None:
            loaded[role] = value
    reservation = loaded["reservation"]
    result = loaded["result"]
    completion = loaded["completion"]
    metric = loaded["metric_verification"]
    reservation_binding = artifacts["reservation"]
    checkpoint_binding = artifacts["checkpoint"]
    result_binding = artifacts["result"]
    completion_binding = artifacts["completion"]
    _validate_reservation_record(
        reservation,
        row=row,
        source_review=expected_source_review,
        prerequisite_gates=expected_prerequisite_gates,
    )
    _validate_result_record(
        result,
        row=row,
        source_review=expected_source_review,
        reservation=reservation,
        reservation_binding=reservation_binding,
        checkpoint_binding=checkpoint_binding,
    )
    _validate_completion_record(
        completion,
        row=row,
        source_review=expected_source_review,
        reservation_binding=reservation_binding,
        checkpoint_binding=checkpoint_binding,
        result_binding=result_binding,
    )
    metric_artifacts = metric.get("artifacts")
    verification = metric.get("verification")
    metric_access = metric.get("access_ledger")
    if (
        set(metric)
        != {
            "schema",
            "status",
            "row",
            "source_review",
            "artifacts",
            "target_partition",
            "target_partition_signature",
            "target_partition_signature_sha256",
            "recomputed_evaluation",
            "recomputed_evaluation_sha256",
            "recomputed_gate_evaluation",
            "recomputed_gate_evaluation_sha256",
            "numeric_gate",
            "verification",
            "resource",
            "determinism",
            "access_ledger",
            "retry_authorized",
            "licenses",
            "content_sha256",
        }
        or metric.get("schema") != ROW_METRIC_SCHEMA
        or metric.get("status") != "verified"
        or metric.get("row") != (asdict(row) | {"key": row.key})
        or metric.get("source_review") != dict(expected_source_review)
        or metric_artifacts
        != {
            "reservation": reservation_binding,
            "checkpoint": checkpoint_binding,
            "result": result_binding,
            "completion": completion_binding,
        }
        or metric.get("target_partition") != result.get("target_partition")
        or metric.get("target_partition")
        != reservation["inputs"]["target_partition"]
        or metric.get("target_partition_signature_sha256")
        != canonical_json_sha256(metric.get("target_partition_signature"))
        or metric.get("recomputed_evaluation") != result.get("evaluation")
        or metric.get("recomputed_evaluation_sha256")
        != canonical_json_sha256(metric.get("recomputed_evaluation"))
        or metric.get("recomputed_gate_evaluation")
        != result.get("gate_evaluation")
        or metric.get("recomputed_gate_evaluation_sha256")
        != canonical_json_sha256(metric.get("recomputed_gate_evaluation"))
        or metric.get("numeric_gate") != gate.get("numeric_gate")
        or verification
        != {
            "checkpoint_bytes_rehashed": True,
            "checkpoint_state_manifest_rehashed": True,
            "checkpoint_semantic_hash_recomputed": True,
            "fresh_model_strict_loaded": True,
            "matched_evaluation_recomputed": True,
            "wrong_rgb_evaluation_recomputed": True,
            "result_metrics_reused": False,
            "metric_repair_applied": False,
            "threshold_weakened": False,
        }
        or metric.get("resource", {}).get("native_thread_environment")
        != {name: "1" for name in THREAD_ENVIRONMENT}
        or type(metric_access) is not dict
        or metric_access.get("checkpoint_opens") != 1
        or any(
            metric_access.get(name) != 0
            for name in (
                "heldout_opens",
                "g2_opens",
                "navigation_opens",
                "runtime_opens",
                "production_opens",
                "gpu1_uses",
            )
        )
        or metric.get("retry_authorized") is not False
        or metric.get("licenses")
        != {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "development_checkpoint_use_authorized": False,
            "new_model_output_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError(f"row metric chain changed: {row.key}")
    return gate, _gate_binding(row, gate, gate_raw)


def _validate_failed_row(
    row_directory: Path,
    row: LadderRow,
    *,
    expected_source_review: Mapping[str, Any],
    expected_prerequisite_gates: Sequence[Mapping[str, Any]],
) -> None:
    reservation, reservation_raw = load_bound_json(row_directory / "reservation.json")
    failure, _ = load_bound_json(row_directory / "failed.json")
    reservation_binding = _json_file_binding(
        "reservation.json",
        reservation,
        reservation_raw,
    )
    _validate_reservation_record(
        reservation,
        row=row,
        source_review=expected_source_review,
        prerequisite_gates=expected_prerequisite_gates,
    )
    if (
        set(failure)
        != {
            "schema",
            "status",
            "row",
            "source_review",
            "reservation",
            "failure_stage",
            "failure",
            "removed_owned_partials",
            "partial_artifacts_removed",
            "retry_authorized",
            "licenses",
            "content_sha256",
        }
        or failure.get("schema") != ROW_FAILURE_SCHEMA
        or failure.get("status") != "failed"
        or failure.get("row") != (asdict(row) | {"key": row.key})
        or failure.get("source_review") != dict(expected_source_review)
        or failure.get("reservation") != reservation_binding
        or failure.get("failure_stage")
        not in {
            "reservation_claim",
            "selected_rgb_decode",
            "training_and_evaluation",
            "input_revalidation_and_checkpoint",
            "training_bundle_publication",
            "isolated_metric_verification",
            "metric_and_gate_publication",
        }
        or type(failure.get("failure")) is not dict
        or set(failure["failure"]) != {"class", "code"}
        or type(failure.get("removed_owned_partials")) is not list
        or failure.get("partial_artifacts_removed") is not True
        or failure.get("retry_authorized") is not False
        or failure.get("licenses")
        != {
            "checkpoint_use_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        }
    ):
        raise PermissionError(f"row failure changed: {row.key}")


def derive_next_row(
    *,
    output_root: Path = OUTPUT_ROOT,
    expected_source_review: Mapping[str, Any],
) -> LadderRow | None:
    if not output_root.exists():
        return LADDER_ROWS[0]
    if output_root.is_symlink() or not output_root.is_dir():
        raise PermissionError("ladder output root is not a real directory")
    root_inventory = sorted(item.name for item in output_root.iterdir())
    if root_inventory not in (["rows"], [FINAL_GATE_FILENAME, "rows"]):
        raise PermissionError("ladder output-root inventory changed")
    has_final_gate = FINAL_GATE_FILENAME in root_inventory
    rows_root = output_root / "rows"
    if rows_root.is_symlink() or not rows_root.is_dir():
        raise PermissionError("ladder rows root is not a real directory")
    entries = {item.name: item for item in rows_root.iterdir()}
    allowed = {row.key for row in LADDER_ROWS}
    if not set(entries).issubset(allowed):
        raise PermissionError("ladder contains an unknown row")
    missing_seen = False
    passed_gate_bindings: list[dict[str, Any]] = []
    for row in LADDER_ROWS:
        directory = entries.get(row.key)
        if directory is None:
            missing_seen = True
            continue
        if missing_seen:
            raise PermissionError("ladder row order is non-contiguous")
        if directory.is_symlink() or not directory.is_dir():
            raise PermissionError(f"row is not a real directory: {row.key}")
        inventory = sorted(item.name for item in directory.iterdir())
        prerequisites = _expected_prerequisite_gates(row, passed_gate_bindings)
        if "failed.json" in inventory:
            if has_final_gate:
                raise PermissionError("final gate exists beside a failed row")
            if inventory != ["failed.json", "reservation.json"]:
                raise PermissionError(f"failed row inventory changed: {row.key}")
            _validate_failed_row(
                directory,
                row,
                expected_source_review=expected_source_review,
                expected_prerequisite_gates=prerequisites,
            )
            raise LadderStopped(f"terminal failed row: {row.key}")
        success_inventory = [
            "checkpoint.pt",
            "completed.json",
            "gate.json",
            "metric_verification.json",
            "reservation.json",
            "result.json",
        ]
        if inventory != success_inventory:
            if has_final_gate:
                raise PermissionError("final gate exists before row completion")
            raise LadderStopped(f"incomplete terminal row: {row.key}")
        gate, gate_binding = validate_completed_row_bundle(
            directory,
            row=row,
            expected_source_review=expected_source_review,
            expected_prerequisite_gates=prerequisites,
        )
        if gate["passes"] is not True:
            if has_final_gate:
                raise PermissionError("final gate exists beside a numeric miss")
            raise LadderStopped(f"terminal numeric gate failure: {row.key}")
        passed_gate_bindings.append(gate_binding)
    for row in LADDER_ROWS:
        if row.key not in entries:
            if has_final_gate:
                raise PermissionError("final gate exists before all eight rows")
            return row
    if has_final_gate:
        validate_final_gate(
            output_root / FINAL_GATE_FILENAME,
            expected_source_review=expected_source_review,
            expected_row_gates=passed_gate_bindings,
        )
    return None


@dataclass(frozen=True)
class RuntimeModules:
    base: Any
    gate: Any
    v9_first_hit: Any
    v12: Any
    np: Any
    torch: Any


_RUNTIME_MODULES: RuntimeModules | None = None
_TORCH_THREADS_CONFIGURED = False
_TORCH_RUNTIME_CONFIGURED = False


def validate_exact_process_environment(
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    environment = os.environ if environ is None else environ
    if environment.get("HIP_VISIBLE_DEVICES") != "0":
        raise PermissionError("ladder exact execution requires HIP_VISIBLE_DEVICES=0")
    present = {
        name: environment.get(name)
        for name in UNSET_DEVICE_SELECTORS
        if name in environment
    }
    if present:
        raise PermissionError(f"conflicting GPU selectors must be unset: {present}")
    wrong_threads = {
        name: environment.get(name)
        for name in THREAD_ENVIRONMENT
        if environment.get(name) != "1"
    }
    if wrong_threads:
        raise PermissionError(f"all six native thread caps must equal one: {wrong_threads}")
    return {
        "hip_visible_devices": "0",
        "conflicting_selectors_unset": list(UNSET_DEVICE_SELECTORS),
        "native_thread_environment": {name: "1" for name in THREAD_ENVIRONMENT},
    }


def load_runtime_modules() -> RuntimeModules:
    global _RUNTIME_MODULES, _TORCH_THREADS_CONFIGURED
    for relative, expected_sha256 in RUNTIME_SOURCE_BINDINGS.items():
        if hashlib.sha256(read_regular(ROOT / relative)).hexdigest() != expected_sha256:
            raise PermissionError(f"runtime source changed before import: {relative}")
    if _RUNTIME_MODULES is not None:
        return _RUNTIME_MODULES
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import torch

    if _TORCH_THREADS_CONFIGURED:
        raise RuntimeError("Torch thread setup repeated before runtime caching")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    _TORCH_THREADS_CONFIGURED = True
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("Torch thread setup did not take effect")
    import numpy as np
    from lewm.benchmarks import go2_observable_camera_ray_fit_v4_ladder_gate as gate
    from lewm.models.observable_camera_ray_evidence_v4_hierarchical_first_hit_v9 import (
        hierarchical_first_hit_nll_breakdown_v9,
    )
    from lewm.models import (
        observable_camera_ray_evidence_v4_gate_aligned_raster_nll_v12 as v12,
    )
    from scripts import train_go2_observable_camera_ray_fit_v4_v2 as base

    threshold_contract = {
        str(size): asdict(gate.FIT_THRESHOLDS[size])
        for size in gate.LADDER_FIT_SIZES
    }
    if (
        gate.canonical_json_sha256(threshold_contract)
        != THRESHOLD_CONTRACT_SHA256
        or {
            size: gate.canonical_json_sha256(asdict(gate.FIT_THRESHOLDS[size]))
            for size in gate.LADDER_FIT_SIZES
        }
        != ROW_THRESHOLD_SHA256
    ):
        raise PermissionError("frozen retained numeric thresholds changed")
    _RUNTIME_MODULES = RuntimeModules(
        base=base,
        gate=gate,
        v9_first_hit=hierarchical_first_hit_nll_breakdown_v9,
        v12=v12,
        np=np,
        torch=torch,
    )
    return _RUNTIME_MODULES


def configure_row_runtime(runtime: RuntimeModules, seed: int) -> dict[str, Any]:
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        raise RuntimeError("Torch runtime configuration may occur only once per process")
    _TORCH_RUNTIME_CONFIGURED = True
    value = int(seed)
    random.seed(value)
    runtime.np.random.seed(value % (2**32))
    runtime.torch.manual_seed(value)
    if runtime.torch.cuda.is_available():
        runtime.torch.cuda.manual_seed_all(value)
    if (
        not _TORCH_THREADS_CONFIGURED
        or runtime.torch.get_num_threads() != 1
        or runtime.torch.get_num_interop_threads() != 1
    ):
        raise RuntimeError("Torch threads were not configured before RNG work")
    runtime.torch.backends.cudnn.benchmark = False
    runtime.torch.backends.cudnn.deterministic = True
    runtime.torch.use_deterministic_algorithms(True, warn_only=True)
    return {
        "seed": value,
        "requested": "strict_deterministic_algorithms",
        "effective": "strict_where_supported_warn_on_exact_allowlisted_kernels",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "torch_num_threads": runtime.torch.get_num_threads(),
        "torch_num_interop_threads": runtime.torch.get_num_interop_threads(),
    }


def validate_live_resource(runtime: RuntimeModules) -> dict[str, Any]:
    resource = dict(runtime.base.validate_gpu0_r9700_runtime(device_text="cuda:0"))
    if (
        runtime.torch.cuda.device_count() != 1
        or "r9700"
        not in "".join(
            character
            for character in runtime.torch.cuda.get_device_name(0).casefold()
            if character.isalnum()
        )
        or "raphael" in runtime.torch.cuda.get_device_name(0).casefold()
    ):
        raise PermissionError("live ladder device is not exactly one R9700")
    resource["native_thread_environment"] = {
        name: os.environ[name] for name in THREAD_ENVIRONMENT
    }
    resource["all_conflicting_selectors_unset"] = all(
        name not in os.environ for name in UNSET_DEVICE_SELECTORS
    )
    if (
        resource["native_thread_environment"]
        != {name: "1" for name in THREAD_ENVIRONMENT}
        or resource["all_conflicting_selectors_unset"] is not True
    ):
        raise PermissionError("live six-thread or selector receipt changed")
    return resource


def _exact_data_paths() -> dict[str, Path]:
    return {
        role: ROOT / str(binding["path"])
        for role, binding in DATA_BINDINGS.items()
        if type(binding) is dict
    }


def load_row_inputs(runtime: RuntimeModules, row: LadderRow) -> tuple[Any, dict[str, Any]]:
    paths = _exact_data_paths()
    runtime.base.preflight_exact_frozen_dataset_provenance(
        dataset_manifest_path=paths["dataset_manifest"],
        dataset_manifest_file_sha256=DATA_BINDINGS["dataset_manifest"][
            "file_sha256"
        ],
    )
    inputs = runtime.base.load_exact_inputs(
        dataset_manifest_path=paths["dataset_manifest"],
        dataset_manifest_file_sha256=DATA_BINDINGS["dataset_manifest"][
            "file_sha256"
        ],
        audit_receipt_path=paths["audit_receipt"],
        audit_receipt_file_sha256=DATA_BINDINGS["audit_receipt"]["file_sha256"],
        trainer_authorization_path=paths["trainer_authorization"],
        trainer_authorization_file_sha256=DATA_BINDINGS["trainer_authorization"][
            "file_sha256"
        ],
        trainer_review_record_path=paths["trainer_review"],
        trainer_review_record_file_sha256=DATA_BINDINGS["trainer_review"][
            "file_sha256"
        ],
        fit_size=row.fit_size,
    )
    target_partition = runtime.base.validate_exact_target_partition_v4(
        inputs.frames,
        fit_size=row.fit_size,
    )
    if (
        len(inputs.frames) != row.fit_size
        or inputs.subset_receipt.get("content_sha256")
        != SUBSET_CONTENT_SHA256[row.fit_size]
        or target_partition.get("content_sha256")
        != TARGET_PARTITION_CONTENT_SHA256[row.fit_size]
    ):
        raise PermissionError(f"frozen row input partition changed: {row.key}")
    return inputs, target_partition


def serial_decode_selected_rgb(
    runtime: RuntimeModules,
    frames: Sequence[Any],
) -> tuple[Any, dict[str, Any]]:
    images, receipt = runtime.base.decode_selected_rgb(
        frames,
        maximum_workers=1,
        allowed_rgb_root=ROOT,
        expected_trainer_source_sha256=RUNTIME_SOURCE_BINDINGS[
            "scripts/train_go2_observable_camera_ray_fit_v4_v2.py"
        ],
    )
    if receipt != {
        "selected_rgb_count": len(frames),
        "nonselected_rgb_opens": 0,
        "rgb_hash_opens": len(frames),
        "rgb_decodes": len(frames),
        "worker_start_method": "inline",
        "worker_count": 1,
        "native_threads_per_worker": 1,
    }:
        raise PermissionError("serial selected-RGB access receipt changed")
    return images, receipt


def model_state_sha256(model: Any) -> str:
    manifest = []
    for name, tensor in sorted(model.state_dict().items()):
        contiguous = tensor.detach().to(device="cpu").contiguous()
        manifest.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype).removeprefix("torch."),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(
                    contiguous.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    return canonical_json_sha256(manifest)


def row_attempt_identity(
    row: LadderRow,
    source_review: Mapping[str, Any],
    prerequisite_gates: Sequence[Mapping[str, Any]],
) -> str:
    return canonical_json_sha256(
        {
            "schema": "lewm_go2_camera_ladder_attempt_identity_v1",
            "output_root": OUTPUT_ROOT_RELATIVE_PATH,
            "row": asdict(row) | {"key": row.key},
            "attempt_index": 1,
            "maximum_attempts": 1,
            "source_review": dict(source_review),
            "prerequisite_gates": [dict(value) for value in prerequisite_gates],
        }
    )


class CompactWarningCollector:
    def __init__(self, base: Any) -> None:
        self.base = base
        self.normalized: Counter[str] = Counter()
        self.context_trailer_count = 0

    def __call__(
        self,
        message: object,
        category: type[Warning],
        filename: str,
        lineno: int,
        file: Any = None,
        line: str | None = None,
    ) -> None:
        del category, filename, lineno, file, line
        validated = self.base.validate_determinism_warnings([message])
        normalized = str(validated["normalized_messages"][0])
        self.normalized[normalized] += 1
        self.context_trailer_count += int(
            validated["normalization"][0]["trailer_removed"]
        )

    def receipt(self) -> dict[str, Any]:
        histogram = [
            {"message": message, "count": count}
            for message, count in sorted(self.normalized.items())
        ]
        return {
            "warning_count": sum(self.normalized.values()),
            "normalized_histogram": histogram,
            "normalized_histogram_sha256": canonical_json_sha256(histogram),
            "context_trailer_count": self.context_trailer_count,
            "whitelist": list(self.base.DETERMINISM_WARNING_WHITELIST),
            "kernel_inventory": list(self.base.DETERMINISM_WARNING_KERNELS),
            "kernel_counts": {
                kernel: sum(
                    count
                    for message, count in self.normalized.items()
                    if message.startswith(kernel)
                )
                for kernel in self.base.DETERMINISM_WARNING_KERNELS
            },
        }


@contextmanager
def capture_compact_determinism_warnings(base: Any) -> Any:
    collector = CompactWarningCollector(base)
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        original = warnings.showwarning
        warnings.showwarning = collector
        try:
            yield collector
        finally:
            warnings.showwarning = original


def compute_gate_aligned_training_loss(
    runtime: RuntimeModules,
    model: Any,
    batch: Any,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    base = runtime.base
    targets = base.derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=batch.pixel_hit_mask,
        pixel_first_hit_distance_m=batch.pixel_first_hit_distance_m,
        ground_support_in_frustum=batch.ground_support_in_frustum,
        ground_support_clear_to_target=batch.ground_support_clear_to_target,
    )
    raw = model(
        batch.image,
        batch.camera_origin_body_m,
        batch.camera_basis_body_fru,
        batch.ground_plane_z_body_m,
    )
    if not runtime.torch.equal(raw.ground_query_in_frustum, targets.ground_in_frustum):
        raise ValueError("model calibration does not reproduce ground visibility")
    first_hit = runtime.v9_first_hit(
        raw.pixel_first_hit_hazard_logits,
        targets,
    ).total
    offset = base._skew_balanced_pixel_offset_loss(raw, targets)
    ground = base.balanced_ground_clear_bce_v4(
        raw.ground_clear_to_target_logits,
        targets,
        raw.ground_target_distance_m,
    )
    raster = base.soft_rasterize_observable_camera_ray_evidence_v4(
        raw,
        camera_origin_body_m=batch.camera_origin_body_m,
        camera_basis_body_fru=batch.camera_basis_body_fru,
    )
    raster_hierarchical = base.hierarchical_raster_cross_entropy_v4(
        raster,
        batch.target_raster_labels,
    ).total
    retained = {
        "hierarchical_first_hit_nll": first_hit,
        "target_bin_offset_smooth_l1": offset,
        "ground_clear_distance_state_balanced_bce": ground,
        "derived_raster_hierarchical_bce": raster_hierarchical,
    }
    cell_nll = runtime.v12.derived_raster_cell_nll_v12(
        raster.class_probabilities,
        batch.target_raster_labels,
    )
    objective = runtime.v12.compose_gate_aligned_objective_v12(retained, cell_nll)
    components = {
        **retained,
        "derived_raster_cell_nll": objective.derived_raster_cell_nll,
    }
    return objective.total, components, raw, targets, raster


def _independent_cell_nll(runtime: RuntimeModules, probabilities: Any, labels: Any) -> Any:
    torch = runtime.torch
    if (
        probabilities.dtype != torch.float32
        or probabilities.ndim != 4
        or probabilities.shape[1] != 3
        or labels.ndim != 3
        or tuple(labels.shape)
        != (probabilities.shape[0], probabilities.shape[2], probabilities.shape[3])
        or probabilities.device != labels.device
    ):
        raise ValueError("independent raster-NLL inputs changed")
    labels = labels.to(dtype=torch.long)
    if bool((labels < 0).any().item()) or bool((labels >= 3).any().item()):
        raise ValueError("independent raster targets are invalid")
    selected = probabilities.gather(1, labels[:, None]).squeeze(1)
    value = -selected.clamp_min(torch.finfo(probabilities.dtype).eps).log().mean()
    if not bool(torch.isfinite(value).item()):
        raise FloatingPointError("independent raster NLL is non-finite")
    return value


def compute_gate_aligned_verification_loss(
    runtime: RuntimeModules,
    model: Any,
    batch: Any,
) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    base = runtime.base
    targets = base.derive_observable_camera_ray_evidence_v4_targets(
        pixel_hit_mask=batch.pixel_hit_mask,
        pixel_first_hit_distance_m=batch.pixel_first_hit_distance_m,
        ground_support_in_frustum=batch.ground_support_in_frustum,
        ground_support_clear_to_target=batch.ground_support_clear_to_target,
    )
    raw = model(
        batch.image,
        batch.camera_origin_body_m,
        batch.camera_basis_body_fru,
        batch.ground_plane_z_body_m,
    )
    if not runtime.torch.equal(raw.ground_query_in_frustum, targets.ground_in_frustum):
        raise ValueError("verifier calibration does not reproduce ground visibility")
    raster = base.soft_rasterize_observable_camera_ray_evidence_v4(
        raw,
        camera_origin_body_m=batch.camera_origin_body_m,
        camera_basis_body_fru=batch.camera_basis_body_fru,
    )
    components = {
        "hierarchical_first_hit_nll": runtime.v9_first_hit(
            raw.pixel_first_hit_hazard_logits,
            targets,
        ).total,
        "target_bin_offset_smooth_l1": base._skew_balanced_pixel_offset_loss(
            raw,
            targets,
        ),
        "ground_clear_distance_state_balanced_bce": (
            base.balanced_ground_clear_bce_v4(
                raw.ground_clear_to_target_logits,
                targets,
                raw.ground_target_distance_m,
            )
        ),
        "derived_raster_hierarchical_bce": (
            base.hierarchical_raster_cross_entropy_v4(
                raster,
                batch.target_raster_labels,
            ).total
        ),
        "derived_raster_cell_nll": _independent_cell_nll(
            runtime,
            raster.class_probabilities,
            batch.target_raster_labels,
        ),
    }
    total = 0.25 * sum(components.values())
    return total, components, raw, targets, raster


def train_row_model(
    runtime: RuntimeModules,
    *,
    row: LadderRow,
    model: Any,
    frames: Sequence[Any],
    images: Any,
    device: Any,
) -> dict[str, Any]:
    schedule = runtime.base._deterministic_training_batches(
        frame_count=len(frames),
        batch_size=row.batch_size,
        steps=row.updates,
        seed=row.seed,
    )
    schedule_sha256 = runtime.base.canonical_json_sha256(schedule)
    if schedule_sha256 != row.schedule_sha256:
        raise PermissionError(f"frozen training schedule changed: {row.key}")
    model.to(device)
    model.train()
    optimizer = runtime.torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4,
    )
    trace: list[dict[str, Any]] = []
    for step, indices in enumerate(schedule, start=1):
        batch = runtime.base._batch_from_indices(frames, images, indices).to(device)
        optimizer.zero_grad(set_to_none=True)
        total, components, _raw, _targets, _raster = (
            compute_gate_aligned_training_loss(runtime, model, batch)
        )
        if not bool(runtime.torch.isfinite(total).item()):
            raise FloatingPointError("camera ladder training loss became non-finite")
        total.backward()
        gradient_norm = runtime.torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0,
        )
        if not bool(runtime.torch.isfinite(gradient_norm).item()):
            raise FloatingPointError("camera ladder gradient norm became non-finite")
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append(
                {
                    "step": step,
                    "total": float(total.detach().item()),
                    "components": {
                        name: float(components[name].detach().item())
                        for name in LOSS_COMPONENTS
                    },
                    "gradient_norm_before_clip": float(
                        gradient_norm.detach().item()
                    ),
                }
            )
    expected_steps = [1, *range(100, row.updates + 1, 100)]
    if [value["step"] for value in trace] != expected_steps:
        raise RuntimeError("compact training trace schedule changed")
    return {
        "steps": row.updates,
        "batch_size": row.batch_size,
        "frame_exposures": row.frame_exposures,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4,
        "optimizer": "AdamW",
        "precision": "float32",
        "autocast": False,
        "gradient_clip_norm": 1.0,
        "loss_weights": {name: 0.25 for name in LOSS_COMPONENTS},
        "initial": trace[0],
        "final": trace[-1],
        "trace": trace,
        "diagnostic_updates": expected_steps,
        "schedule_algorithm": (
            "torch_cpu_generator_manual_seed_then_concatenated_randperm_cycles_"
            "take_steps_times_batch_v1"
        ),
        "schedule_sha256": schedule_sha256,
        "checkpoint_selection": "final_update_only",
        "fresh_model_initialization": True,
        "predecessor_checkpoint_opens": 0,
    }


def evaluate_row_model(
    runtime: RuntimeModules,
    *,
    model: Any,
    frames: Sequence[Any],
    images: Any,
    device: Any,
    wrong_rgb: bool,
    independent: bool,
) -> dict[str, Any]:
    model.eval()
    accumulator = runtime.base.ObservableCameraRayFitV4MetricAccumulator()
    sums: Counter[str] = Counter()
    diagnostic_rows: list[dict[str, Any]] = []
    mapping = tuple(
        ((index + 1) % len(frames)) if wrong_rgb else index
        for index in range(len(frames))
    )
    loss_function = (
        compute_gate_aligned_verification_loss
        if independent
        else compute_gate_aligned_training_loss
    )
    with runtime.torch.no_grad():
        for target_index in range(len(frames)):
            batch = runtime.base._batch_from_indices(
                frames,
                images,
                (target_index,),
                image_indices=(mapping[target_index],),
            ).to(device)
            _total, components, raw, targets, raster = loss_function(
                runtime,
                model,
                batch,
            )
            for name in LOSS_COMPONENTS:
                sums[name] += float(components[name].item())
            accumulator.update(
                raw_output=raw,
                targets=targets,
                soft_raster=raster,
                target_raster_labels=batch.target_raster_labels,
                families=batch.families,
            )
            diagnostic_rows.append(
                runtime.v12.raster_nll_diagnostics_v12(
                    raster.class_probabilities,
                    batch.target_raster_labels,
                    batch.families,
                )
            )
    means = {name: sums[name] / len(frames) for name in LOSS_COMPONENTS}
    retained_total = 0.25 * sum(means[name] for name in RETAINED_LOSS_COMPONENTS)
    diagnostics = runtime.v12.merge_raster_nll_diagnostics_v12(diagnostic_rows)
    cell_nll = float(diagnostics["overall"]["mean"])
    metrics = accumulator.finalize()
    if not runtime.torch.isclose(
        runtime.torch.tensor(metrics["derived_raster"]["nll"], dtype=runtime.torch.float64),
        runtime.torch.tensor(cell_nll, dtype=runtime.torch.float64),
        rtol=0.0,
        atol=2e-7,
    ):
        raise ValueError("raster diagnostics and metric accumulator disagree")
    if not math.isclose(means["derived_raster_cell_nll"], cell_nll, rel_tol=0.0, abs_tol=2e-7):
        raise ValueError("batch raster NLL and merged diagnostics disagree")
    return {
        "control": (
            "wrong_rgb_with_target_calibration" if wrong_rgb else "matched_rgb"
        ),
        "wrong_rgb_degenerate_singleton": False,
        "image_index_mapping": list(mapping),
        "image_mapping_sha256": canonical_json_sha256(list(mapping)),
        "losses": {
            **{name: means[name] for name in RETAINED_LOSS_COMPONENTS},
            "total": retained_total,
        },
        "native_v15_objective": {
            "derived_raster_cell_nll": cell_nll,
            "v11_base_total": retained_total,
            "total": retained_total + 0.25 * cell_nll,
        },
        "raster_nll_diagnostics": diagnostics,
        "metrics": metrics,
    }


def gate_evaluation_view(evaluation: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for role in ("matched_rgb", "wrong_rgb_with_target_calibration"):
        native = evaluation[role]
        losses = native["losses"]
        result[role] = {
            "control": native["control"],
            "wrong_rgb_degenerate_singleton": native[
                "wrong_rgb_degenerate_singleton"
            ],
            "image_index_mapping": list(native["image_index_mapping"]),
            "image_mapping_sha256": native["image_mapping_sha256"],
            "losses": {
                "ordered_first_hit_nll": losses["hierarchical_first_hit_nll"],
                "target_bin_offset_smooth_l1": losses[
                    "target_bin_offset_smooth_l1"
                ],
                "ground_clear_distance_state_balanced_bce": losses[
                    "ground_clear_distance_state_balanced_bce"
                ],
                "derived_raster_hierarchical_bce": losses[
                    "derived_raster_hierarchical_bce"
                ],
                "total": losses["total"],
            },
            "metrics": native["metrics"],
        }
    return result


def reconstruct_numeric_gate(
    runtime: RuntimeModules,
    *,
    row: LadderRow,
    gate_evaluation: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    matched, wrong, signature = runtime.gate._validated_metric_evaluation(
        gate_evaluation,
        fit_size=row.fit_size,
    )
    numeric = runtime.gate._gate_stage(
        {"fit_size": row.fit_size, "matched": matched, "wrong": wrong}
    )
    validate_numeric_gate(numeric, row=row)
    return signature, numeric


def _write_exclusive(path: Path, raw: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _self_hashed(core: Mapping[str, Any]) -> dict[str, Any]:
    normalized = json.loads(canonical_json_bytes(core))
    return {**normalized, "content_sha256": canonical_json_sha256(normalized)}


def _json_payload(value: Mapping[str, Any]) -> bytes:
    return canonical_json_bytes(value) + b"\n"


def _publish_json_exclusive(path: Path, core: Mapping[str, Any]) -> tuple[dict[str, Any], bytes]:
    value = _self_hashed(core)
    raw = _json_payload(value)
    _write_exclusive(path, raw)
    return value, raw


def ensure_output_root() -> None:
    OUTPUT_ROOT.parent.mkdir(parents=True, exist_ok=True)
    ROWS_ROOT.mkdir(parents=True, exist_ok=True)
    for directory, name in ((OUTPUT_ROOT, "output root"), (ROWS_ROOT, "rows root")):
        if directory.is_symlink() or not directory.is_dir():
            raise PermissionError(f"ladder {name} is not a real directory")
    if sorted(item.name for item in OUTPUT_ROOT.iterdir()) not in (
        ["rows"],
        [FINAL_GATE_FILENAME, "rows"],
    ):
        raise PermissionError("ladder output-root inventory changed")


def next_row_prerequisite_bindings(row: LadderRow) -> list[dict[str, Any]]:
    passed = []
    for prior in LADDER_ROWS[: row.index]:
        gate, raw = load_bound_json(ROWS_ROOT / prior.key / "gate.json")
        if gate.get("passes") is not True:
            raise LadderStopped(f"prior gate did not pass: {prior.key}")
        passed.append(_gate_binding(prior, gate, raw))
    return _expected_prerequisite_gates(row, passed)


def all_passed_row_gate_bindings(
    expected_source_review: Mapping[str, Any],
) -> list[dict[str, Any]]:
    passed: list[dict[str, Any]] = []
    for row in LADDER_ROWS:
        prerequisites = _expected_prerequisite_gates(row, passed)
        gate, binding = validate_completed_row_bundle(
            ROWS_ROOT / row.key,
            row=row,
            expected_source_review=expected_source_review,
            expected_prerequisite_gates=prerequisites,
        )
        if gate["passes"] is not True:
            raise LadderStopped(f"row did not pass: {row.key}")
        passed.append(binding)
    return passed


def checkpoint_metadata(
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    reservation_binding: Mapping[str, Any],
    initialization: Mapping[str, Any],
    subset: Mapping[str, Any],
    target_partition: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment": {
            "row": asdict(row) | {"key": row.key},
            "science_contract_sha256": canonical_json_sha256(science_contract()),
        },
        "source_review": dict(source_review),
        "attempt_reservation": dict(reservation_binding),
        "initialization": dict(initialization),
        "subset_content_sha256": subset["content_sha256"],
        "target_partition_content_sha256": target_partition["content_sha256"],
        "training_schedule_sha256": row.schedule_sha256,
        "checkpoint_selection": "final_update_only",
        "loss_contract": {
            "version": "gate_aligned_raster_nll_v15",
            "components": list(LOSS_COMPONENTS),
            "weights": {name: 0.25 for name in LOSS_COMPONENTS},
            "retained_v11_components": list(RETAINED_LOSS_COMPONENTS),
            "predecessor_checkpoint_input": False,
        },
    }


def reserve_row(
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    prerequisite_gates: Sequence[Mapping[str, Any]],
    terminal_v16: Mapping[str, Any],
    inputs: Any,
    target_partition: Mapping[str, Any],
    resource: Mapping[str, Any],
    determinism: Mapping[str, Any],
    initialization: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], bytes]:
    directory = ROWS_ROOT / row.key
    attempt_identity = str(initialization["attempt_identity"])
    core = {
        "schema": ROW_RESERVATION_SCHEMA,
        "status": "reserved",
        "row": asdict(row) | {"key": row.key},
        "attempt_index": 1,
        "maximum_attempts": 1,
        "attempt_identity": attempt_identity,
        "source_review": dict(source_review),
        "prerequisite_gates": [dict(value) for value in prerequisite_gates],
        "terminal_v16": dict(terminal_v16) if row.index == 0 else None,
        "science_contract_sha256": canonical_json_sha256(science_contract()),
        "inputs": {
            "data_bindings": DATA_BINDINGS,
            "subset": inputs.subset_receipt,
            "target_partition": dict(target_partition),
        },
        "initialization": dict(initialization),
        "resource": dict(resource),
        "determinism": dict(determinism),
        "retry_authorized": False,
        "licenses": {
            "development_checkpoint_creation_authorized": True,
            "metric_verification_checkpoint_use_authorized": True,
            "predecessor_checkpoint_use_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    value = _self_hashed(core)
    raw = _json_payload(value)
    os.mkdir(directory, 0o700)
    try:
        _write_exclusive(directory / "reservation.json", raw)
        _fsync_directory(directory)
        _fsync_directory(ROWS_ROOT)
        return directory, value, raw
    except BaseException as error:
        if (directory / "reservation.json").exists():
            try:
                terminate_row_failure(
                    directory,
                    row=row,
                    source_review=source_review,
                    reservation=value,
                    reservation_raw=raw,
                    error=error,
                    stage="reservation_claim",
                )
            except BaseException as terminal_error:
                raise RuntimeError(
                    "reservation failed after commit and terminalization failed"
                ) from terminal_error
        else:
            os.rmdir(directory)
        raise


def _failure_classification(error: BaseException) -> dict[str, str]:
    if isinstance(error, FloatingPointError):
        return {"class": "numeric", "code": "nonfinite_training_failure"}
    if isinstance(error, PermissionError):
        return {"class": "permission", "code": "scope_or_authorization_failure"}
    if isinstance(error, ValueError):
        return {"class": "validation", "code": "structural_validation_failure"}
    if isinstance(error, OSError):
        return {"class": "io", "code": "filesystem_or_device_failure"}
    if isinstance(error, KeyboardInterrupt):
        return {"class": "interruption", "code": "operator_interruption"}
    if isinstance(error, RuntimeError):
        return {"class": "runtime", "code": "execution_failure"}
    return {"class": "internal", "code": "unexpected_internal_failure"}


def terminate_row_failure(
    directory: Path,
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    error: BaseException,
    stage: str,
) -> dict[str, Any]:
    removed = []
    for name in (
        "gate.json",
        "metric_verification.json",
        "completed.json",
        "result.json",
        "checkpoint.pt",
    ):
        path = directory / name
        if path.exists():
            if path.is_symlink() or not path.is_file():
                raise PermissionError("owned row partial is not a regular file")
            path.unlink()
            removed.append(name)
    core = {
        "schema": ROW_FAILURE_SCHEMA,
        "status": "failed",
        "row": asdict(row) | {"key": row.key},
        "source_review": dict(source_review),
        "reservation": _json_file_binding(
            "reservation.json",
            reservation,
            reservation_raw,
        ),
        "failure_stage": stage,
        "failure": _failure_classification(error),
        "removed_owned_partials": sorted(removed),
        "partial_artifacts_removed": True,
        "retry_authorized": False,
        "licenses": {
            "checkpoint_use_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    failed, _raw = _publish_json_exclusive(directory / "failed.json", core)
    _fsync_directory(directory)
    _fsync_directory(ROWS_ROOT)
    return failed


def validate_checkpoint_bytes(
    runtime: RuntimeModules,
    raw: bytes,
    *,
    expected_binding: Mapping[str, Any],
    expected_metadata: Mapping[str, Any],
) -> Mapping[str, Any]:
    if (
        hashlib.sha256(raw).hexdigest() != expected_binding.get("file_sha256")
        or len(raw) != expected_binding.get("byte_count")
    ):
        raise PermissionError("checkpoint byte binding changed")
    try:
        checkpoint = runtime.torch.load(
            BytesIO(raw),
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = runtime.torch.load(BytesIO(raw), map_location="cpu")
    fields = {
        "schema",
        "model_class",
        "state_manifest",
        "metadata",
        "authoritative",
        "aggregation_eligible",
        "promotion_eligible",
        "state_dict",
        "content_sha256",
    }
    if type(checkpoint) is not dict or set(checkpoint) != fields:
        raise ValueError("checkpoint schema changed")
    if (
        checkpoint.get("schema")
        != "lewm_go2_observable_camera_ray_fit_v4_development_checkpoint_v2"
        or checkpoint.get("model_class") != "ObservableCameraRayEvidenceV4Model"
        or checkpoint.get("metadata") != dict(expected_metadata)
        or checkpoint.get("authoritative") is not False
        or checkpoint.get("aggregation_eligible") is not False
        or checkpoint.get("promotion_eligible") is not False
    ):
        raise PermissionError("checkpoint metadata or scope changed")
    state = checkpoint.get("state_dict")
    if type(state) is not dict or type(checkpoint.get("state_manifest")) is not list:
        raise ValueError("checkpoint state is malformed")
    expected_manifest = []
    for name, tensor in sorted(state.items()):
        if type(name) is not str or not isinstance(tensor, runtime.torch.Tensor):
            raise ValueError("checkpoint tensor entry is malformed")
        contiguous = tensor.detach().to(device="cpu").contiguous()
        expected_manifest.append(
            {
                "name": name,
                "dtype": str(contiguous.dtype).removeprefix("torch."),
                "shape": list(contiguous.shape),
                "sha256": hashlib.sha256(
                    contiguous.numpy().tobytes(order="C")
                ).hexdigest(),
            }
        )
    semantic_core = {
        key: checkpoint[key]
        for key in (
            "schema",
            "model_class",
            "state_manifest",
            "metadata",
            "authoritative",
            "aggregation_eligible",
            "promotion_eligible",
        )
    }
    if (
        checkpoint["state_manifest"] != expected_manifest
        or checkpoint["content_sha256"] != canonical_json_sha256(semantic_core)
        or checkpoint["content_sha256"] != expected_binding.get("content_sha256")
    ):
        raise ValueError("checkpoint semantic hash changed")
    model = runtime.base.ObservableCameraRayEvidenceV4Model()
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ValueError("strict checkpoint load reported incompatible keys")
    return model


def _validate_preverification_bundle(
    request: Mapping[str, Any],
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        request.get("schema")
        != "lewm_go2_camera_ladder_internal_verification_request_v1"
        or request.get("row_index") != row.index
        or request.get("source_review") != dict(source_review)
    ):
        raise PermissionError("internal verification request changed")
    artifacts = request.get("artifacts")
    expected_roles = {"reservation", "checkpoint", "result", "completion"}
    if type(artifacts) is not dict or set(artifacts) != expected_roles:
        raise PermissionError("internal verification artifact map changed")
    expected_paths = {
        "reservation": "reservation.json",
        "checkpoint": "checkpoint.pt",
        "result": "result.json",
        "completion": "completed.json",
    }
    for role, binding in artifacts.items():
        if (
            type(binding) is not dict
            or set(binding)
            != {"path", "file_sha256", "content_sha256", "byte_count"}
            or binding.get("path") != expected_paths[role]
            or not is_sha256(binding.get("file_sha256"))
            or not is_sha256(binding.get("content_sha256"))
            or type(binding.get("byte_count")) is not int
            or binding["byte_count"] <= 0
        ):
            raise PermissionError(f"internal {role} binding changed")
    directory = ROWS_ROOT / row.key
    if sorted(path.name for path in directory.iterdir()) != [
        "checkpoint.pt",
        "completed.json",
        "reservation.json",
        "result.json",
    ]:
        raise PermissionError("preverification row inventory changed")
    loaded: dict[str, dict[str, Any]] = {}
    raw_by_role: dict[str, bytes] = {}
    for role, binding in artifacts.items():
        _claimed, value, raw = _actual_artifact_binding(directory, role, binding)
        raw_by_role[role] = raw
        if value is not None:
            loaded[role] = value
    reservation = loaded["reservation"]
    result = loaded["result"]
    completion = loaded["completion"]
    prerequisites = reservation.get("prerequisite_gates")
    if type(prerequisites) is not list:
        raise PermissionError("preverification prerequisite gate list changed")
    _validate_reservation_record(
        reservation,
        row=row,
        source_review=source_review,
        prerequisite_gates=prerequisites,
    )
    _validate_result_record(
        result,
        row=row,
        source_review=source_review,
        reservation=reservation,
        reservation_binding=artifacts["reservation"],
        checkpoint_binding=artifacts["checkpoint"],
    )
    _validate_completion_record(
        completion,
        row=row,
        source_review=source_review,
        reservation_binding=artifacts["reservation"],
        checkpoint_binding=artifacts["checkpoint"],
        result_binding=artifacts["result"],
    )
    return {
        "directory": directory,
        "artifacts": artifacts,
        "reservation": reservation,
        "result": result,
        "completion": completion,
        "checkpoint_raw": raw_by_role["checkpoint"],
    }


def revalidate_row_inputs(runtime: RuntimeModules, inputs: Any) -> dict[str, int]:
    paths = _exact_data_paths()
    return runtime.base.revalidate_exact_inputs_after_training(
        inputs,
        dataset_manifest_path=paths["dataset_manifest"],
        dataset_manifest_file_sha256=DATA_BINDINGS["dataset_manifest"][
            "file_sha256"
        ],
        audit_receipt_path=paths["audit_receipt"],
        audit_receipt_file_sha256=DATA_BINDINGS["audit_receipt"]["file_sha256"],
        trainer_authorization_path=paths["trainer_authorization"],
        trainer_authorization_file_sha256=DATA_BINDINGS["trainer_authorization"][
            "file_sha256"
        ],
        trainer_review_record_path=paths["trainer_review"],
        trainer_review_record_file_sha256=DATA_BINDINGS["trainer_review"][
            "file_sha256"
        ],
    )


def _parse_canonical_object_bytes(raw: bytes, *, name: str) -> dict[str, Any]:
    if len(raw) > 1024 * 1024:
        raise ValueError(f"{name} is too large")
    try:
        value = json.loads(raw, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is malformed") from error
    if type(value) is not dict or raw != canonical_json_bytes(value) + b"\n":
        raise ValueError(f"{name} is not one canonical JSON object")
    core = dict(value)
    declared = core.pop("content_sha256", None)
    if not is_sha256(declared) or canonical_json_sha256(core) != declared:
        raise PermissionError(f"{name} self-hash changed")
    return value


def discover_internal_verification_bundle() -> tuple[
    LadderRow,
    dict[str, Any],
    bytes,
    dict[str, Any],
]:
    if OUTPUT_ROOT.is_symlink() or not OUTPUT_ROOT.is_dir():
        raise PermissionError("internal verifier output root is not a real directory")
    if sorted(item.name for item in OUTPUT_ROOT.iterdir()) != ["rows"]:
        raise PermissionError("internal verifier output-root inventory changed")
    entries = {item.name: item for item in ROWS_ROOT.iterdir()}
    if not set(entries).issubset({row.key for row in LADDER_ROWS}):
        raise PermissionError("internal verifier found an unknown row")
    candidate: LadderRow | None = None
    for row in LADDER_ROWS:
        directory = entries.get(row.key)
        if directory is None:
            if candidate is None and any(
                later.key in entries for later in LADDER_ROWS[row.index + 1 :]
            ):
                raise PermissionError("internal verifier row order is non-contiguous")
            continue
        if directory.is_symlink() or not directory.is_dir():
            raise PermissionError("internal verifier row is not a real directory")
        inventory = sorted(path.name for path in directory.iterdir())
        if inventory == [
            "checkpoint.pt",
            "completed.json",
            "gate.json",
            "metric_verification.json",
            "reservation.json",
            "result.json",
        ]:
            if candidate is not None:
                raise PermissionError("completed row follows verifier candidate")
            continue
        if inventory == [
            "checkpoint.pt",
            "completed.json",
            "reservation.json",
            "result.json",
        ]:
            if candidate is not None:
                raise PermissionError("multiple internal verifier candidates exist")
            candidate = row
            continue
        raise PermissionError("internal verifier found a non-candidate row state")
    if candidate is None:
        raise PermissionError("internal verifier found no unique completed ungated row")
    if any(row.key in entries for row in LADDER_ROWS[candidate.index + 1 :]):
        raise PermissionError("internal verifier candidate is not the final row")
    directory = ROWS_ROOT / candidate.key
    reservation, reservation_raw = load_bound_json(directory / "reservation.json")
    claimed_review = reservation.get("source_review")
    if type(claimed_review) is not dict or not is_sha256(
        claimed_review.get("file_sha256")
    ):
        raise PermissionError("candidate source-review binding changed")
    review, review_raw = validate_source_review(claimed_review["file_sha256"])
    review_binding = source_review_binding(review, review_raw)
    if review_binding != claimed_review:
        raise PermissionError("candidate source-review bytes changed")
    passed: list[dict[str, Any]] = []
    for prior in LADDER_ROWS[: candidate.index]:
        prerequisites = _expected_prerequisite_gates(prior, passed)
        gate, binding = validate_completed_row_bundle(
            ROWS_ROOT / prior.key,
            row=prior,
            expected_source_review=review_binding,
            expected_prerequisite_gates=prerequisites,
        )
        if gate["passes"] is not True:
            raise PermissionError("verifier candidate follows a failed numeric gate")
        passed.append(binding)
    prerequisites = _expected_prerequisite_gates(candidate, passed)
    result, result_raw = load_bound_json(directory / "result.json")
    completion, completion_raw = load_bound_json(directory / "completed.json")
    checkpoint_model = result.get("model", {}).get("checkpoint")
    if (
        type(checkpoint_model) is not dict
        or checkpoint_model.get("development_only") is not True
    ):
        raise PermissionError("candidate checkpoint model binding changed")
    checkpoint_binding = dict(checkpoint_model)
    checkpoint_binding.pop("development_only")
    artifacts = {
        "reservation": _json_file_binding(
            "reservation.json",
            reservation,
            reservation_raw,
        ),
        "checkpoint": checkpoint_binding,
        "result": _json_file_binding("result.json", result, result_raw),
        "completion": _json_file_binding(
            "completed.json",
            completion,
            completion_raw,
        ),
    }
    request = {
        "schema": "lewm_go2_camera_ladder_internal_verification_request_v1",
        "row_index": candidate.index,
        "source_review": review_binding,
        "artifacts": artifacts,
    }
    bundle = _validate_preverification_bundle(
        request,
        row=candidate,
        source_review=review_binding,
    )
    if reservation.get("prerequisite_gates") != prerequisites:
        raise PermissionError("candidate prerequisite gate chain changed")
    return candidate, review, review_raw, bundle


def run_internal_verifier() -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("internal verifier requires python -I -B")
    validate_exact_process_environment()
    row, review, review_raw, bundle = discover_internal_verification_bundle()
    review_binding = source_review_binding(review, review_raw)
    runtime = load_runtime_modules()
    determinism = configure_row_runtime(runtime, row.seed)
    resource = validate_live_resource(runtime)
    inputs, target_partition = load_row_inputs(runtime, row)
    reservation = bundle["reservation"]
    if (
        reservation.get("inputs", {}).get("subset") != inputs.subset_receipt
        or reservation.get("inputs", {}).get("target_partition")
        != target_partition
    ):
        raise PermissionError("internal verifier input reproduction changed")
    images, rgb_access = serial_decode_selected_rgb(runtime, inputs.frames)
    metadata = checkpoint_metadata(
        row=row,
        source_review=review_binding,
        reservation_binding=bundle["artifacts"]["reservation"],
        initialization=reservation["initialization"],
        subset=inputs.subset_receipt,
        target_partition=target_partition,
    )
    model = validate_checkpoint_bytes(
        runtime,
        bundle["checkpoint_raw"],
        expected_binding=bundle["artifacts"]["checkpoint"],
        expected_metadata=metadata,
    )
    device = runtime.torch.device("cuda:0")
    model.to(device)
    with capture_compact_determinism_warnings(runtime.base) as warning_collector:
        matched = evaluate_row_model(
            runtime,
            model=model,
            frames=inputs.frames,
            images=images,
            device=device,
            wrong_rgb=False,
            independent=True,
        )
        wrong = evaluate_row_model(
            runtime,
            model=model,
            frames=inputs.frames,
            images=images,
            device=device,
            wrong_rgb=True,
            independent=True,
        )
    evaluation = {
        "matched_rgb": matched,
        "wrong_rgb_with_target_calibration": wrong,
    }
    if evaluation != bundle["result"].get("evaluation"):
        raise ValueError("independent checkpoint evaluation differs from result")
    gate_evaluation = gate_evaluation_view(evaluation)
    if gate_evaluation != bundle["result"].get("gate_evaluation"):
        raise ValueError("independent gate adapter differs from result")
    signature, numeric_gate = reconstruct_numeric_gate(
        runtime,
        row=row,
        gate_evaluation=gate_evaluation,
    )
    post_input = revalidate_row_inputs(runtime, inputs)
    core = {
        "schema": ROW_METRIC_SCHEMA,
        "status": "verified",
        "row": asdict(row) | {"key": row.key},
        "source_review": review_binding,
        "artifacts": {
            role: bundle["artifacts"][role]
            for role in ("reservation", "checkpoint", "result", "completion")
        },
        "target_partition": target_partition,
        "target_partition_signature": signature,
        "target_partition_signature_sha256": canonical_json_sha256(signature),
        "recomputed_evaluation": evaluation,
        "recomputed_evaluation_sha256": canonical_json_sha256(evaluation),
        "recomputed_gate_evaluation": gate_evaluation,
        "recomputed_gate_evaluation_sha256": canonical_json_sha256(
            gate_evaluation
        ),
        "numeric_gate": numeric_gate,
        "verification": {
            "checkpoint_bytes_rehashed": True,
            "checkpoint_state_manifest_rehashed": True,
            "checkpoint_semantic_hash_recomputed": True,
            "fresh_model_strict_loaded": True,
            "matched_evaluation_recomputed": True,
            "wrong_rgb_evaluation_recomputed": True,
            "result_metrics_reused": False,
            "metric_repair_applied": False,
            "threshold_weakened": False,
        },
        "resource": resource,
        "determinism": {**determinism, **warning_collector.receipt()},
        "access_ledger": {
            **rgb_access,
            **post_input,
            "checkpoint_opens": 1,
            "heldout_opens": 0,
            "g2_opens": 0,
            "navigation_opens": 0,
            "runtime_opens": 0,
            "production_opens": 0,
            "gpu1_uses": 0,
        },
        "retry_authorized": False,
        "licenses": {
            "checkpoint_use_authorized_for_metric_verification_only": True,
            "development_checkpoint_use_authorized": False,
            "new_model_output_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    receipt = _self_hashed(core)
    sys.stdout.buffer.write(_json_payload(receipt))
    sys.stdout.buffer.flush()
    return 0


def invoke_internal_verifier(
    *,
    row: LadderRow,
    source_review: Mapping[str, Any],
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-I",
        "-B",
        str(ROOT / RUNNER_RELATIVE_PATH),
        "--internal-verify",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=3600,
    )
    if completed.returncode != 0 or completed.stderr:
        raise RuntimeError(
            "isolated metric verifier failed: "
            + completed.stderr.decode("utf-8", errors="replace")[-2000:]
        )
    receipt = _parse_canonical_object_bytes(
        completed.stdout,
        name="internal verification receipt",
    )
    if (
        receipt.get("schema") != ROW_METRIC_SCHEMA
        or receipt.get("status") != "verified"
        or receipt.get("row") != (asdict(row) | {"key": row.key})
        or receipt.get("source_review") != dict(source_review)
        or receipt.get("artifacts") != dict(artifacts)
        or receipt.get("retry_authorized") is not False
    ):
        raise PermissionError("internal metric receipt changed")
    validate_numeric_gate(receipt.get("numeric_gate"), row=row)
    return receipt


def validate_replication_initial_state(row: LadderRow, state_sha256: str) -> None:
    for prior in LADDER_ROWS[: row.index]:
        if prior.seed != row.seed:
            continue
        reservation, _raw = load_bound_json(ROWS_ROOT / prior.key / "reservation.json")
        if reservation.get("initialization", {}).get("initial_state_sha256") != state_sha256:
            raise PermissionError("same-seed replication initial state changed across N")


def _publish_training_bundle(
    *,
    directory: Path,
    row: LadderRow,
    source_review: Mapping[str, Any],
    reservation: Mapping[str, Any],
    reservation_raw: bytes,
    inputs: Any,
    target_partition: Mapping[str, Any],
    initialization: Mapping[str, Any],
    resource: Mapping[str, Any],
    determinism: Mapping[str, Any],
    warning_receipt: Mapping[str, Any],
    rgb_access: Mapping[str, Any],
    post_input: Mapping[str, Any],
    training: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    checkpoint_raw: bytes,
    checkpoint_content_sha256: str,
) -> dict[str, Any]:
    reservation_binding = _json_file_binding(
        "reservation.json",
        reservation,
        reservation_raw,
    )
    checkpoint_binding = artifact_binding(
        "checkpoint.pt",
        checkpoint_raw,
        content_sha256=checkpoint_content_sha256,
    )
    gate_evaluation = gate_evaluation_view(evaluation)
    result_core = {
        "schema": ROW_RESULT_SCHEMA,
        "status": "completed_training",
        "row": asdict(row) | {"key": row.key},
        "attempt_identity": reservation["attempt_identity"],
        "source_review": dict(source_review),
        "reservation": reservation_binding,
        "subset": inputs.subset_receipt,
        "target_partition": dict(target_partition),
        "initialization": dict(initialization),
        "model": {
            "class": "ObservableCameraRayEvidenceV4Model",
            "fresh_initialization": True,
            "parameter_count": 3_105_513,
            "checkpoint": {**checkpoint_binding, "development_only": True},
        },
        "training": dict(training),
        "evaluation": dict(evaluation),
        "gate_evaluation": gate_evaluation,
        "gate_adapter": (
            "v15_native_diagnostics_excluded_then_hierarchical_to_ordered_key_v1"
        ),
        "resource": dict(resource),
        "determinism": {**dict(determinism), **dict(warning_receipt)},
        "access_ledger": {
            **dict(rgb_access),
            **dict(post_input),
            "selected_rgb_rehashes_before_publication": len(inputs.frames),
            "predecessor_checkpoint_opens": 0,
            "heldout_opens": 0,
            "g2_opens": 0,
            "navigation_opens": 0,
            "runtime_opens": 0,
            "production_opens": 0,
            "gpu1_uses": 0,
        },
        "licenses": {
            "development_checkpoint_creation_authorized": True,
            "checkpoint_use_authorized": False,
            "metric_verification_only_checkpoint_use_authorized": True,
            "retry_authorized": False,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    result = _self_hashed(result_core)
    result_raw = _json_payload(result)
    result_binding = _json_file_binding("result.json", result, result_raw)
    completion_core = {
        "schema": ROW_COMPLETION_SCHEMA,
        "status": "completed_training",
        "row": asdict(row) | {"key": row.key},
        "source_review": dict(source_review),
        "reservation": reservation_binding,
        "checkpoint": checkpoint_binding,
        "result": result_binding,
        "inventory": [
            "checkpoint.pt",
            "completed.json",
            "gate.json",
            "metric_verification.json",
            "reservation.json",
            "result.json",
        ],
        "retry_authorized": False,
        "licenses": {
            "checkpoint_use_authorized": False,
            "metric_verification_only_checkpoint_use_authorized": True,
            "heldout_authorized": False,
            "g2_authorized": False,
            "navigation_authorized": False,
            "production_authorized": False,
            "promotion_authorized": False,
        },
    }
    completion = _self_hashed(completion_core)
    completion_raw = _json_payload(completion)
    completion_binding = _json_file_binding(
        "completed.json",
        completion,
        completion_raw,
    )
    _write_exclusive(directory / "checkpoint.pt", checkpoint_raw)
    _write_exclusive(directory / "result.json", result_raw)
    _write_exclusive(directory / "completed.json", completion_raw)
    _fsync_directory(directory)
    return {
        "reservation": reservation_binding,
        "checkpoint": checkpoint_binding,
        "result": result_binding,
        "completion": completion_binding,
    }


def _publish_metric_and_gate(
    *,
    directory: Path,
    row: LadderRow,
    source_review: Mapping[str, Any],
    prerequisite_gates: Sequence[Mapping[str, Any]],
    training_artifacts: Mapping[str, Any],
    metric: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_raw = _json_payload(metric)
    _write_exclusive(directory / "metric_verification.json", metric_raw)
    metric_binding = _json_file_binding(
        "metric_verification.json",
        metric,
        metric_raw,
    )
    artifacts = {**dict(training_artifacts), "metric_verification": metric_binding}
    numeric = validate_numeric_gate(metric["numeric_gate"], row=row)
    gate_core = {
        "schema": ROW_GATE_SCHEMA,
        "status": "passed" if numeric["passes"] else "failed_numeric_gate",
        "row": asdict(row) | {"key": row.key},
        "source_review": dict(source_review),
        "prerequisite_gates": [dict(value) for value in prerequisite_gates],
        "artifacts": artifacts,
        "threshold_contract_sha256": THRESHOLD_CONTRACT_SHA256,
        "numeric_gate": numeric,
        "check_count": numeric["check_count"],
        "failure_count": numeric["failure_count"],
        "passes": numeric["passes"],
        "retry_authorized": False,
    }
    gate, gate_raw = _publish_json_exclusive(directory / "gate.json", gate_core)
    _fsync_directory(directory)
    _fsync_directory(ROWS_ROOT)
    return gate, _gate_binding(row, gate, gate_raw)


def run_next(
    review: Mapping[str, Any],
    review_raw: bytes,
    terminal_v16: Mapping[str, Any],
) -> int:
    if not sys.flags.isolated or not sys.dont_write_bytecode:
        raise PermissionError("exact ladder execution requires python -I -B")
    validate_exact_process_environment()
    source_binding = source_review_binding(review, review_raw)
    row = derive_next_row(expected_source_review=source_binding)
    if row is None:
        row_gates = all_passed_row_gate_bindings(source_binding)
        final_path = OUTPUT_ROOT / FINAL_GATE_FILENAME
        if final_path.exists():
            final_gate, final_raw = validate_final_gate(
                final_path,
                expected_source_review=source_binding,
                expected_row_gates=row_gates,
            )
        else:
            final_gate, final_raw = publish_final_gate(
                expected_source_review=source_binding,
                expected_row_gates=row_gates,
            )
        final_binding = _json_file_binding(
            FINAL_GATE_FILENAME,
            final_gate,
            final_raw,
        )
        summary = _self_hashed(
            {
                "schema": "lewm_go2_camera_ladder_v1_execution_summary",
                "status": "all_eight_rows_passed",
                "next_row": None,
                "final_gate": final_binding,
            }
        )
        print(canonical_json_bytes(summary).decode("ascii"))
        return 0
    runtime = load_runtime_modules()
    determinism = configure_row_runtime(runtime, row.seed)
    inputs, target_partition = load_row_inputs(runtime, row)
    schedule = runtime.base._deterministic_training_batches(
        frame_count=row.fit_size,
        batch_size=row.batch_size,
        steps=row.updates,
        seed=row.seed,
    )
    if runtime.base.canonical_json_sha256(schedule) != row.schedule_sha256:
        raise PermissionError("pre-reservation row schedule changed")
    ensure_output_root()
    if derive_next_row(expected_source_review=source_binding) != row:
        raise PermissionError("next row changed during pre-reservation validation")
    prerequisites = next_row_prerequisite_bindings(row)
    attempt_identity = row_attempt_identity(row, source_binding, prerequisites)
    model = runtime.base.ObservableCameraRayEvidenceV4Model()
    initial_state_sha256 = model_state_sha256(model)
    validate_replication_initial_state(row, initial_state_sha256)
    initialization = {
        "attempt_identity": attempt_identity,
        "initial_state_sha256": initial_state_sha256,
        "initialization_identity": initialization_identity(
            row,
            attempt_identity,
            initial_state_sha256,
        ),
        "fresh_model_construction": True,
        "predecessor_checkpoint_opens": 0,
    }
    resource = validate_live_resource(runtime)
    directory, reservation, reservation_raw = reserve_row(
        row=row,
        source_review=source_binding,
        prerequisite_gates=prerequisites,
        terminal_v16=terminal_v16,
        inputs=inputs,
        target_partition=target_partition,
        resource=resource,
        determinism=determinism,
        initialization=initialization,
    )
    stage = "selected_rgb_decode"
    row_committed = False
    try:
        images, rgb_access = serial_decode_selected_rgb(runtime, inputs.frames)
        device = runtime.torch.device("cuda:0")
        stage = "training_and_evaluation"
        with capture_compact_determinism_warnings(runtime.base) as warning_collector:
            training = train_row_model(
                runtime,
                row=row,
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
            )
            matched = evaluate_row_model(
                runtime,
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                wrong_rgb=False,
                independent=False,
            )
            wrong = evaluate_row_model(
                runtime,
                model=model,
                frames=inputs.frames,
                images=images,
                device=device,
                wrong_rgb=True,
                independent=False,
            )
        evaluation = {
            "matched_rgb": matched,
            "wrong_rgb_with_target_calibration": wrong,
        }
        gate_view = gate_evaluation_view(evaluation)
        parent_signature, parent_numeric_gate = reconstruct_numeric_gate(
            runtime,
            row=row,
            gate_evaluation=gate_view,
        )
        warning_receipt = warning_collector.receipt()
        stage = "input_revalidation_and_checkpoint"
        post_input = revalidate_row_inputs(runtime, inputs)
        runtime.base._verify_file_commitments(
            tuple((frame.rgb_path, frame.image_sha256) for frame in inputs.frames),
            name="camera ladder selected train RGB before publication",
        )
        reservation_binding = _json_file_binding(
            "reservation.json",
            reservation,
            reservation_raw,
        )
        metadata = checkpoint_metadata(
            row=row,
            source_review=source_binding,
            reservation_binding=reservation_binding,
            initialization=initialization,
            subset=inputs.subset_receipt,
            target_partition=target_partition,
        )
        checkpoint_raw, checkpoint_content_sha256 = runtime.base._checkpoint_bytes(
            model,
            metadata=metadata,
        )
        stage = "training_bundle_publication"
        artifacts = _publish_training_bundle(
            directory=directory,
            row=row,
            source_review=source_binding,
            reservation=reservation,
            reservation_raw=reservation_raw,
            inputs=inputs,
            target_partition=target_partition,
            initialization=initialization,
            resource=resource,
            determinism=determinism,
            warning_receipt=warning_receipt,
            rgb_access=rgb_access,
            post_input=post_input,
            training=training,
            evaluation=evaluation,
            checkpoint_raw=checkpoint_raw,
            checkpoint_content_sha256=checkpoint_content_sha256,
        )
        model.to(runtime.torch.device("cpu"))
        del model
        runtime.torch.cuda.empty_cache()
        stage = "isolated_metric_verification"
        metric = invoke_internal_verifier(
            row=row,
            source_review=source_binding,
            artifacts=artifacts,
        )
        if (
            metric.get("target_partition_signature") != parent_signature
            or metric.get("numeric_gate") != parent_numeric_gate
        ):
            raise RuntimeError("isolated verifier differs from parent reconstruction")
        stage = "metric_and_gate_publication"
        gate, gate_binding = _publish_metric_and_gate(
            directory=directory,
            row=row,
            source_review=source_binding,
            prerequisite_gates=prerequisites,
            training_artifacts=artifacts,
            metric=metric,
        )
        validated_gate, validated_binding = validate_completed_row_bundle(
            directory,
            row=row,
            expected_source_review=source_binding,
            expected_prerequisite_gates=prerequisites,
        )
        if validated_gate != gate or validated_binding != gate_binding:
            raise RuntimeError("published row did not byte-revalidate")
        row_committed = True
        summary = _self_hashed(
            {
                "schema": "lewm_go2_camera_ladder_v1_execution_summary",
                "status": (
                    "row_passed" if gate["passes"] else "row_failed_numeric_gate"
                ),
                "row": asdict(row) | {"key": row.key},
                "gate": gate_binding,
                "retry_authorized": False,
            }
        )
        print(canonical_json_bytes(summary).decode("ascii"))
        return 0 if gate["passes"] else 3
    except BaseException as error:
        if row_committed:
            raise
        terminate_row_failure(
            directory,
            row=row,
            source_review=source_binding,
            reservation=reservation,
            reservation_raw=reservation_raw,
            error=error,
            stage=stage,
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    raw = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--next", action="store_true")
    modes.add_argument("--cpu-contract-smoke", action="store_true")
    modes.add_argument("--internal-verify", action="store_true")
    parser.add_argument("--source-review-sha256")
    args = parser.parse_args(raw)
    if args.next:
        if raw != ["--next", "--source-review-sha256", args.source_review_sha256]:
            raise ValueError("--next accepts only the canonical source review digest")
        if not is_sha256(args.source_review_sha256):
            raise ValueError("source review digest is malformed")
    elif args.cpu_contract_smoke:
        if raw != ["--cpu-contract-smoke"]:
            raise ValueError("CPU smoke accepts no other argument")
    elif raw != ["--internal-verify"]:
        raise ValueError("internal verifier accepts no caller arguments")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.cpu_contract_smoke:
        print(
            canonical_json_bytes(
                {
                    "schema": "lewm_go2_camera_ladder_v1_cpu_contract_smoke",
                    "status": "passed_runtime_complete_increment_two",
                    "rows": row_contract(),
                    "science_contract_sha256": canonical_json_sha256(
                        science_contract()
                    ),
                    "runtime_complete": True,
                }
            ).decode("ascii")
        )
        return 0
    if args.internal_verify:
        return run_internal_verifier()
    review, review_raw = validate_source_review(args.source_review_sha256)
    binding = source_review_binding(review, review_raw)
    candidate = derive_next_row(expected_source_review=binding)
    terminal_v16 = (
        validate_terminal_v16()
        if candidate is not None and candidate.index == 0
        else {"transitively_bound_by_passing_row_zero_gate": True}
    )
    return run_next(review, review_raw, terminal_v16)


if __name__ == "__main__":
    raise SystemExit(main())
