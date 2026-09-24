#!/usr/bin/env -S python3 -I
"""Fixed source-capturing entrypoint for Shared JEPA V5 G2/G3 runners."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import sys
from types import ModuleType


CANONICAL_REPOSITORY_ROOT = Path(
    "/home/andrewknowles/Workspace/LeWMQuad-v3"
).resolve()
CANONICAL_WRAPPER_PATH = (
    CANONICAL_REPOSITORY_ROOT / "scripts/run_go2_shared_jepa_v5_gate.py"
)
CANONICAL_LAUNCHER_PATH = (
    CANONICAL_REPOSITORY_ROOT / "scripts/go2_shared_jepa_v5_launcher.py"
)
EXPECTED_LAUNCHER_FILE_SHA256 = (
    "7f273649fa6c8b4256c552359927fc20bb59d1bfbd5b47194a3f5a941c5b8958"
)

CANONICAL_G2_RUNNER_AUTHORITY_FILE_SHA256: str | None = None
CANONICAL_G3_RUNNER_AUTHORITY_FILE_SHA256: str | None = None


def _read_source(path: Path, *, name: str) -> tuple[bytes, str]:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PermissionError(f"{name} is not a singly-linked regular file")
        encoded = b""
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            encoded += chunk
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
    return encoded, hashlib.sha256(encoded).hexdigest()


def _main() -> int:
    if not sys.flags.isolated or not sys.flags.no_user_site:
        raise PermissionError("V5 runner wrapper requires isolated Python")
    if Path(__file__).resolve() != CANONICAL_WRAPPER_PATH:
        raise PermissionError("V5 runner wrapper was executed from a copied path")
    wrapper_bytes, wrapper_hash = _read_source(
        CANONICAL_WRAPPER_PATH,
        name="V5 runner wrapper",
    )
    del wrapper_bytes
    launcher_bytes, launcher_hash = _read_source(
        CANONICAL_LAUNCHER_PATH,
        name="V5 launcher",
    )
    if launcher_hash != EXPECTED_LAUNCHER_FILE_SHA256:
        raise PermissionError("captured V5 launcher source hash changed")
    launcher = ModuleType("_lewm_go2_shared_jepa_v5_captured_runner_launcher")
    launcher.__file__ = str(CANONICAL_LAUNCHER_PATH)
    exec(compile(launcher_bytes, launcher.__file__, "exec"), launcher.__dict__)
    return launcher.launch_captured_stage(
        "runner",
        wrapper_identity={
            "path": CANONICAL_WRAPPER_PATH.relative_to(
                CANONICAL_REPOSITORY_ROOT
            ).as_posix(),
            "file_sha256": wrapper_hash,
        },
        launcher_file_sha256=launcher_hash,
        production_authorities={
            "runner_g2_inputs_v2": (
                CANONICAL_REPOSITORY_ROOT
                / "docs/lewm_go2_shared_jepa_v5_runner_g2_authority_v2.json",
                CANONICAL_G2_RUNNER_AUTHORITY_FILE_SHA256,
            ),
            "runner_g3_inputs_v2": (
                CANONICAL_REPOSITORY_ROOT
                / "docs/lewm_go2_shared_jepa_v5_runner_g3_authority_v2.json",
                CANONICAL_G3_RUNNER_AUTHORITY_FILE_SHA256,
            ),
        },
    )


if __name__ == "__main__":
    raise SystemExit(_main())
