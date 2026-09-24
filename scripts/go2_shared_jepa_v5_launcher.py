#!/usr/bin/env python3
"""Fixed source-capturing launcher for the Shared JEPA V5 lifecycle."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
from types import ModuleType
from typing import Mapping, Sequence


CANONICAL_REPOSITORY_ROOT = Path(
    "/home/andrewknowles/Workspace/LeWMQuad-v3"
).resolve()
CANONICAL_LAUNCHER_PATH = (
    CANONICAL_REPOSITORY_ROOT / "scripts/go2_shared_jepa_v5_launcher.py"
)
CANONICAL_CORE_PATH = (
    CANONICAL_REPOSITORY_ROOT / "scripts/go2_shared_jepa_v5_one_shot.py"
)
EXPECTED_CORE_FILE_SHA256 = (
    "62a19f3028e9152120af990528752431b996f56b4bc9b62db32eba47ae235a1f"
)
WRAPPER_PATHS = {
    "runner": CANONICAL_REPOSITORY_ROOT / "scripts/run_go2_shared_jepa_v5_gate.py",
    "finalizer": (
        CANONICAL_REPOSITORY_ROOT / "scripts/finalize_go2_shared_jepa_v5_gate.py"
    ),
    "publisher": (
        CANONICAL_REPOSITORY_ROOT / "scripts/publish_go2_shared_jepa_v5_checkpoint.py"
    ),
}


def _read_source(path: Path, *, name: str) -> tuple[bytes, str]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"{name} is not a singly-linked regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise RuntimeError(f"{name} changed while open")
    finally:
        os.close(descriptor)
    encoded = b"".join(chunks)
    return encoded, hashlib.sha256(encoded).hexdigest()


def launch_captured_stage(
    stage: str,
    *,
    wrapper_identity: Mapping[str, str],
    launcher_file_sha256: str,
    production_authorities: object,
    argv: Sequence[str] | None = None,
) -> int:
    if stage not in WRAPPER_PATHS:
        raise ValueError("unknown captured V5 stage")
    if Path(__file__).resolve() != CANONICAL_LAUNCHER_PATH:
        raise PermissionError("V5 launcher was executed from a copied path")
    if (
        not isinstance(wrapper_identity, Mapping)
        or set(wrapper_identity) != {"path", "file_sha256"}
        or wrapper_identity.get("path")
        != WRAPPER_PATHS[stage].relative_to(CANONICAL_REPOSITORY_ROOT).as_posix()
        or type(wrapper_identity.get("file_sha256")) is not str
    ):
        raise PermissionError("V5 wrapper execution identity changed")
    launcher_bytes, actual_launcher_hash = _read_source(
        CANONICAL_LAUNCHER_PATH,
        name="V5 launcher source",
    )
    if actual_launcher_hash != launcher_file_sha256:
        raise PermissionError("captured V5 launcher source hash changed")
    core_bytes, actual_core_hash = _read_source(
        CANONICAL_CORE_PATH,
        name="V5 core source",
    )
    if actual_core_hash != EXPECTED_CORE_FILE_SHA256:
        raise PermissionError("captured V5 core source hash changed")

    module = ModuleType("_lewm_go2_shared_jepa_v5_captured_core")
    module.__file__ = str(CANONICAL_CORE_PATH)
    exec(compile(core_bytes, module.__file__, "exec"), module.__dict__)
    entrypoint = getattr(module, "main_for_stage", None)
    if not callable(entrypoint):
        raise TypeError("captured V5 core lacks main_for_stage")
    execution_identity = {
        "schema": "lewm_go2_shared_jepa_v5_execution_identity_v1",
        "entrypoint_wrapper": dict(wrapper_identity),
        "captured_launcher": {
            "path": CANONICAL_LAUNCHER_PATH.relative_to(
                CANONICAL_REPOSITORY_ROOT
            ).as_posix(),
            "file_sha256": actual_launcher_hash,
        },
        "captured_core": {
            "path": CANONICAL_CORE_PATH.relative_to(
                CANONICAL_REPOSITORY_ROOT
            ).as_posix(),
            "file_sha256": actual_core_hash,
        },
    }
    return entrypoint(
        stage,
        argv,
        production_authorities=production_authorities,
        execution_identity=execution_identity,
    )


if __name__ == "__main__":
    raise SystemExit("invoke one of the fixed V5 entrypoint wrappers")
