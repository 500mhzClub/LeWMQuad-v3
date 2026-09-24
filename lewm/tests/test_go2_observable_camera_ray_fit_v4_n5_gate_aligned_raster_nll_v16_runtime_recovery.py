"""CPU/synthetic proofs for the lean Camera V16 runtime recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess
import sys
from typing import Any

import pytest

from scripts import (
    execute_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v16
    as v16,
)


ROOT = Path(__file__).resolve().parents[2]


def _self_hashed(core: dict[str, Any]) -> dict[str, Any]:
    return {
        **core,
        "content_sha256": hashlib.sha256(v16._canonical_json_bytes(core)).hexdigest(),
    }


def _write_json(path: Path, value: dict[str, Any]) -> bytes:
    raw = v16._canonical_json_bytes(value) + b"\n"
    path.write_bytes(raw)
    return raw


class _Torch:
    def __init__(self, intra: int = 1, inter: int = 1, *, preserve: bool = False):
        self.intra = intra
        self.inter = inter
        self.preserve = preserve
        self.intra_sets: list[int] = []
        self.inter_sets: list[int] = []

    def set_num_threads(self, value: int) -> None:
        self.intra_sets.append(value)
        if not self.preserve:
            self.intra = value

    def set_num_interop_threads(self, value: int) -> None:
        self.inter_sets.append(value)
        if not self.preserve:
            self.inter = value

    def get_num_threads(self) -> int:
        return self.intra

    def get_num_interop_threads(self) -> int:
        return self.inter


def test_idempotent_determinism_preserves_calls_receipt_and_skips_setters() -> None:
    torch = _Torch()
    events: list[object] = []
    receipt = {"seed": 20260710, "all_other_determinism_calls": True}

    def retained(seed: int) -> dict[str, Any]:
        events.extend(("random_seed", "numpy_seed", "torch_seed", seed))
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        events.extend(("cudnn", "deterministic_algorithms"))
        return receipt

    observed = v16._make_idempotent_determinism(retained, torch)(20260710)
    assert observed is receipt
    assert events == [
        "random_seed",
        "numpy_seed",
        "torch_seed",
        20260710,
        "cudnn",
        "deterministic_algorithms",
    ]
    assert torch.intra_sets == []
    assert torch.inter_sets == []


def test_idempotent_determinism_sets_non_one_once_and_fails_closed() -> None:
    torch = _Torch(intra=4, inter=3)

    def retained(_seed: int) -> dict[str, bool]:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        return {"retained": True}

    assert v16._make_idempotent_determinism(retained, torch)(1) == {
        "retained": True
    }
    assert torch.intra_sets == [1]
    assert torch.inter_sets == [1]

    stuck = _Torch(intra=2, inter=2, preserve=True)

    def stuck_retained(_seed: int) -> dict[str, bool]:
        stuck.set_num_threads(1)
        stuck.set_num_interop_threads(1)
        return {"retained": True}

    with pytest.raises(RuntimeError, match="could not configure"):
        v16._make_idempotent_determinism(stuck_retained, stuck)(1)

    boolean = _Torch()

    def bad_retained(_seed: int) -> dict[str, bool]:
        boolean.set_num_threads(True)
        return {"retained": True}

    with pytest.raises(PermissionError, match="non-one intra-op"):
        v16._make_idempotent_determinism(bad_retained, boolean)(1)


def test_terminal_v15_binding_includes_lock_and_exact_inventories(tmp_path: Path) -> None:
    output = tmp_path / "v15"
    attempt = output / "attempts/seed_20260710/n5"
    attempt.mkdir(parents=True)
    (output / "gates").mkdir()
    (output / "metric_verifications").mkdir()
    lock = attempt.parent / ".n5.reservation-v15.lock"
    lock.write_bytes(b"")
    lock.chmod(0o600)

    reservation = _self_hashed(
        {
            "schema": (
                "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
                "raster_nll_v15_reservation_v1"
            ),
            "attempt_index": 1,
            "maximum_attempts": 1,
            "seed": 20260710,
            "fit_size": 5,
        }
    )
    reservation_raw = _write_json(attempt / "reservation.json", reservation)
    reservation_binding = {
        "path": "reservation.json",
        "file_sha256": hashlib.sha256(reservation_raw).hexdigest(),
        "content_sha256": reservation["content_sha256"],
        "byte_count": len(reservation_raw),
    }
    failure = _self_hashed(
        {
            "schema": (
                "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
                "raster_nll_v15_failure_v1"
            ),
            "failure_stage": "training",
            "reservation": reservation_binding,
            "retry_authorized": False,
            "licenses": {"retry_authorized": False},
        }
    )
    failure_raw = _write_json(attempt / "failed.json", failure)
    bindings = {
        "reservation": {
            "file_sha256": reservation_binding["file_sha256"],
            "content_sha256": reservation["content_sha256"],
        },
        "failure": {
            "file_sha256": hashlib.sha256(failure_raw).hexdigest(),
            "content_sha256": failure["content_sha256"],
        },
        "lock": {
            "file_sha256": hashlib.sha256(b"").hexdigest(),
            "byte_count": 0,
        },
        "seed_root_inventory": [".n5.reservation-v15.lock", "n5"],
        "output_root_inventory": ["attempts", "gates", "metric_verifications"],
    }
    observed = v16._validate_terminal_v15(
        reservation_path=attempt / "reservation.json",
        failure_path=attempt / "failed.json",
        lock_path=lock,
        bindings=bindings,
    )
    assert observed["lock_file_sha256"] == hashlib.sha256(b"").hexdigest()
    (output / "gates/forbidden.json").write_text("{}", encoding="ascii")
    with pytest.raises(PermissionError, match="not empty"):
        v16._validate_terminal_v15(
            reservation_path=attempt / "reservation.json",
            failure_path=attempt / "failed.json",
            lock_path=lock,
            bindings=bindings,
        )


def test_v16_review_binds_sources_proof_terminal_and_runtime(tmp_path: Path) -> None:
    sources = ("wrapper.py", "test.py")
    proofs = ("amendment.md",)
    for relative in (*sources, *proofs):
        (tmp_path / relative).write_text(relative + "\n", encoding="ascii")
    source_bindings = {
        relative: {
            "path": relative,
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for relative in sources
    }
    proof_bindings = {
        relative: {
            "path": relative,
            "file_sha256": hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest(),
        }
        for relative in proofs
    }
    review = _self_hashed(
        {
            "schema": v16.V16_REVIEW_SCHEMA,
            "status": "different_agent_review_passed_v16_runtime_only_recovery",
            "implementation_author": v16.V16_IMPLEMENTATION_AUTHOR,
            "reviewer": "/root/independent_v16_reviewer",
            "review_completed": True,
            "source_closure_approved": True,
            "exact_attempt_authorized": True,
            "exactly_one_v16_attempt_authorized": True,
            "v15_retry_authorized": False,
            "scientific_change_authorized": False,
            "output_root": v16.V16_OUTPUT_ROOT_RELATIVE_PATH,
            "terminal_v15": v16.V15_TERMINAL_BINDINGS,
            "retained_v15_files": v16.V15_RETAINED_FILES,
            "runtime_delta": v16.V16_RUNTIME_DELTA,
            "successor_sources": source_bindings,
            "successor_proofs": proof_bindings,
            "licenses": v16.V16_REVIEW_LICENSES,
        }
    )
    raw = _write_json(tmp_path / "review.json", review)
    digest = hashlib.sha256(raw).hexdigest()
    assert v16._validate_v16_review(
        digest,
        path=tmp_path / "review.json",
        root=tmp_path,
        successor_source_paths=sources,
        successor_proof_paths=proofs,
    )["exact_attempt_authorized"] is True
    (tmp_path / "wrapper.py").write_text("changed\n", encoding="ascii")
    with pytest.raises(PermissionError, match="reviewed source changed"):
        v16._validate_v16_review(
            digest,
            path=tmp_path / "review.json",
            root=tmp_path,
            successor_source_paths=sources,
            successor_proof_paths=proofs,
        )


def test_adapter_redirects_only_v16_and_persists_outer_review(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_authority = lambda: {"retained_v15": True}
    policy = SimpleNamespace(
        CANONICAL_OUTPUT_ROOT=v16.V15_OUTPUT_ROOT,
        CANONICAL_ATTEMPT_PATH=(
            v16.V15_OUTPUT_ROOT / "attempts/seed_20260710/n5"
        ),
        CANONICAL_METRIC_RECEIPT_PATH=(
            v16.V15_OUTPUT_ROOT / "metric_verifications/seed_20260710_n5.json"
        ),
        CANONICAL_GATE_PATH=(
            v16.V15_OUTPUT_ROOT / "gates/seed_20260710_n5.json"
        ),
        GPU_VISIBILITY_RECEIPT_PATH=v16.V15_GPU_VISIBILITY_RECEIPT_PATH,
        GPU_VISIBILITY_RECEIPT_SCHEMA=v16.V15_GPU_VISIBILITY_RECEIPT_SCHEMA,
        ATTEMPT_SCOPE=v16.V15_ATTEMPT_SCOPE,
        authority_bindings=original_authority,
        SOURCE_REVIEW_RELATIVE_PATH=(
            "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_independent_review_2026-07-14.json"
        ),
        CANONICAL_SOURCE_REVIEW_PATH=v16.ROOT / (
            "docs/lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_independent_review_2026-07-14.json"
        ),
        SOURCE_REVIEW_SCHEMA=(
            "lewm_go2_observable_camera_ray_fit_v4_n5_gate_aligned_"
            "raster_nll_v15_source_review_v1"
        ),
    )

    def retained_review(path: Path, digest: str):
        assert path == policy.CANONICAL_SOURCE_REVIEW_PATH
        assert digest == v16.V15_SOURCE_REVIEW_FILE_SHA256
        assert policy.CANONICAL_OUTPUT_ROOT == v16.V15_OUTPUT_ROOT
        assert policy.GPU_VISIBILITY_RECEIPT_PATH == (
            v16.V15_GPU_VISIBILITY_RECEIPT_PATH
        )
        assert policy.authority_bindings() == {"retained_v15": True}
        return {"retained": True}, b"retained\n"

    policy.preflight_source_review = retained_review
    torch = _Torch()
    base = SimpleNamespace(
        torch=torch,
        configure_determinism=lambda seed: {"seed": seed},
    )
    gpu = SimpleNamespace(
        CANONICAL_RECEIPT_PATH=v16.V15_GPU_VISIBILITY_RECEIPT_PATH
    )
    outer = {
        "content_sha256": "b" * 64,
        "successor_sources": {},
        "successor_proofs": {},
    }
    binding = {
        "path": v16.V16_SOURCE_REVIEW_RELATIVE_PATH,
        "file_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }
    monkeypatch.setattr(v16, "_validate_v16_review", lambda _digest: outer)
    monkeypatch.setattr(v16, "_read_regular", lambda _path: b"outer\n")

    restore = v16._install_v16_runtime_adapter(
        policy, base, gpu, v16_review_binding=binding
    )
    assert policy.CANONICAL_OUTPUT_ROOT == v16.V16_OUTPUT_ROOT
    assert policy.CANONICAL_ATTEMPT_PATH == v16.V16_ATTEMPT_PATH
    assert policy.ATTEMPT_SCOPE == v16.V16_ATTEMPT_SCOPE
    assert gpu.CANONICAL_RECEIPT_PATH == v16.V16_GPU_VISIBILITY_RECEIPT_PATH
    review, raw = policy.preflight_source_review(v16.V16_REVIEW_PATH, "a" * 64)
    assert (review, raw) == (outer, b"outer\n")
    assert policy.CANONICAL_OUTPUT_ROOT == v16.V16_OUTPUT_ROOT
    authority = policy.authority_bindings()
    assert authority["v16_runtime_recovery_review"] == binding
    assert authority["v16_compatibility_boundary"]["v15_retry_authorized"] is False
    restore()
    assert policy.CANONICAL_OUTPUT_ROOT == v16.V15_OUTPUT_ROOT
    assert gpu.CANONICAL_RECEIPT_PATH == v16.V15_GPU_VISIBILITY_RECEIPT_PATH

    restore = v16._install_v16_runtime_adapter(
        policy, None, gpu, v16_review_binding=binding
    )
    assert policy.CANONICAL_OUTPUT_ROOT == v16.V16_OUTPUT_ROOT
    restore()
    assert policy.CANONICAL_OUTPUT_ROOT == v16.V15_OUTPUT_ROOT


def test_visibility_preflight_rejects_every_extra_argument() -> None:
    with pytest.raises(ValueError, match="exactly one source-review digest"):
        v16.main(
            [
                "--v16-gpu-visibility-preflight",
                "--source-review-sha256",
                "a" * 64,
                "--ignored-before-fix",
            ]
        )


@pytest.mark.parametrize(
    "mode", ["--cpu-contract-smoke", "--cpu-verifier-contract-smoke"]
)
def test_v15_cpu_smokes_are_byte_identical_through_wrapper(mode: str) -> None:
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for name in v16.THREAD_ENVIRONMENT:
        environment[name] = "1"
    python = Path("/home/andrewknowles/TinyQuadJEPA/bin/python")
    assert python.is_file()
    direct = subprocess.run(
        [str(python), str(v16.V15_EXECUTOR_PATH), mode],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    wrapped = subprocess.run(
        [str(python), str(Path(v16.__file__).resolve()), mode],
        cwd=ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=90,
    )
    assert (direct.returncode, direct.stderr) == (0, b"")
    assert (wrapped.returncode, wrapped.stderr) == (0, b"")
    assert json.loads(wrapped.stdout) == json.loads(direct.stdout)
