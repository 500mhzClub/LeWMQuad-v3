"""CPU-only proofs for the Camera V15 pre-reservation visibility boundary."""
from __future__ import annotations

import ast
from datetime import datetime, timedelta, timezone
import hashlib
import inspect
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any

import pytest

from lewm.benchmarks import (
    go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15 as policy,
)
from scripts import (
    preflight_go2_observable_camera_ray_fit_v4_n5_gate_aligned_raster_nll_v15_gpu_visibility
    as visibility,
)


ROOT = Path(__file__).resolve().parents[2]
COMMIT = "c" * 64
FILE_SHA = "a" * 64


def _environment() -> dict[str, str]:
    return {
        "HIP_VISIBLE_DEVICES": "0",
        **{name: "1" for name in policy.GPU_VISIBILITY_THREAD_ENVIRONMENT},
    }


class _Cuda:
    def __init__(
        self,
        *,
        available: bool = True,
        names: list[str] | None = None,
        availability_error: BaseException | None = None,
        enumeration_error: BaseException | None = None,
    ) -> None:
        self.available = available
        self.names = [policy.EXPECTED_GPU_DEVICE_NAME] if names is None else names
        self.availability_error = availability_error
        self.enumeration_error = enumeration_error
        self.calls: list[str] = []

    def is_available(self) -> bool:
        self.calls.append("is_available")
        if self.availability_error is not None:
            raise self.availability_error
        return self.available

    def device_count(self) -> int:
        self.calls.append("device_count")
        if self.enumeration_error is not None:
            raise self.enumeration_error
        return len(self.names)

    def get_device_name(self, ordinal: int) -> str:
        self.calls.append(f"get_device_name:{ordinal}")
        return self.names[ordinal]


class _Torch:
    def __init__(
        self,
        cuda: _Cuda | None = None,
        *,
        intra: int = 1,
        inter: int = 1,
        preserve_threads: bool = False,
    ) -> None:
        self.cuda = _Cuda() if cuda is None else cuda
        self.intra = intra
        self.inter = inter
        self.preserve_threads = preserve_threads
        self.calls: list[str] = []

    def set_num_threads(self, value: int) -> None:
        self.calls.append(f"set_num_threads:{value}")
        if not self.preserve_threads:
            self.intra = value

    def set_num_interop_threads(self, value: int) -> None:
        self.calls.append(f"set_num_interop_threads:{value}")
        if not self.preserve_threads:
            self.inter = value

    def get_num_threads(self) -> int:
        self.calls.append("get_num_threads")
        return self.intra

    def get_num_interop_threads(self) -> int:
        self.calls.append("get_num_interop_threads")
        return self.inter


def _review() -> dict[str, Any]:
    sources = {
        relative: {"path": relative, "file_sha256": "1" * 64}
        for relative in policy.SUCCESSOR_SOURCE_PATHS
    }
    proofs = {
        relative: {"path": relative, "file_sha256": "2" * 64}
        for relative in policy.SUCCESSOR_PROOF_PATHS
    }
    return {
        "reviewer": "/root/independent_v15_visibility_reviewer",
        "status": (
            "different_agent_review_passed_n5_gate_aligned_raster_nll_v15_"
            "runtime_visibility_successor"
        ),
        "source_closure_approved": True,
        "exact_attempt_authorized": True,
        "successor_sources": sources,
        "successor_proofs": proofs,
        "content_sha256": "b" * 64,
    }


def _passing_observation() -> dict[str, Any]:
    return visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: _Torch(),
    )


def _receipt(**overrides: Any) -> dict[str, Any]:
    return visibility.build_receipt(
        observation=_passing_observation(),
        source_review=_review(),
        source_review_file_sha256=FILE_SHA,
        repository_commit=COMMIT,
        hostname=overrides.pop("hostname", "host-a"),
        boot_id=overrides.pop("boot_id", "00000000-0000-0000-0000-000000000001"),
        process_id=overrides.pop("process_id", 123),
        utc_timestamp=overrides.pop("utc_timestamp", "2026-07-14T12:00:00.000000Z"),
        monotonic_seconds=overrides.pop("monotonic_seconds", 1000.0),
        **overrides,
    )


def _rehash(value: dict[str, Any]) -> dict[str, Any]:
    core = dict(value)
    core.pop("content_sha256", None)
    value["content_sha256"] = policy.canonical_json_sha256(core)
    return value


def _validate(value: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    return visibility.validate_receipt_value(
        value,
        expected_source_review=_review(),
        expected_source_review_file_sha256=FILE_SHA,
        expected_repository_commit=COMMIT,
        hostname=overrides.pop("hostname", "host-a"),
        boot_id=overrides.pop(
            "boot_id", "00000000-0000-0000-0000-000000000001"
        ),
        utc_now=overrides.pop(
            "utc_now", datetime(2026, 7, 14, 12, 5, tzinfo=timezone.utc)
        ),
        monotonic_now=overrides.pop("monotonic_now", 1300.0),
        **overrides,
    )


def _git_show(commit: str, relative: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    return completed.stdout


def _review_bound_to_git_commit(commit: str) -> dict[str, Any]:
    def bindings(paths: tuple[str, ...]) -> dict[str, dict[str, str]]:
        return {
            relative: {
                "path": relative,
                "file_sha256": hashlib.sha256(
                    _git_show(commit, relative)
                ).hexdigest(),
            }
            for relative in paths
        }

    return {
        "successor_sources": bindings(policy.SUCCESSOR_SOURCE_PATHS),
        "successor_proofs": bindings(policy.SUCCESSOR_PROOF_PATHS),
    }


@pytest.mark.parametrize(
    ("torch_loader", "expected_reason"),
    (
        (lambda: (_ for _ in ()).throw(ImportError("missing runtime")), "torch_runtime_import_failed"),
        (
            lambda: _Torch(
                _Cuda(availability_error=PermissionError("missing /dev/kfd"))
            ),
            "gpu_runtime_availability_check_failed",
        ),
        (lambda: _Torch(_Cuda(available=False)), "gpu_runtime_reported_unavailable"),
        (
            lambda: _Torch(_Cuda(enumeration_error=RuntimeError("runtime init"))),
            "gpu_runtime_enumeration_failed",
        ),
    ),
)
def test_runtime_import_permission_availability_and_enumeration_failures_are_unavailable(
    torch_loader: Any,
    expected_reason: str,
) -> None:
    observed = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=torch_loader,
    )
    assert observed["disposition"] == "gpu_runtime_unavailable"
    assert observed["reason_code"] == expected_reason
    assert observed["runtime_observation"]["enumeration_completed"] is False


@pytest.mark.parametrize("names", ([], [policy.EXPECTED_GPU_DEVICE_NAME] * 2))
def test_successful_zero_or_multiple_device_enumeration_is_count_mismatch(
    names: list[str],
) -> None:
    observed = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: _Torch(_Cuda(names=names)),
    )
    assert observed["disposition"] == "gpu_device_count_mismatch"
    assert observed["runtime_observation"]["enumeration_completed"] is True
    assert observed["runtime_observation"]["visible_device_count"] == len(names)


def test_unavailable_runtime_and_successful_zero_enumeration_are_distinct() -> None:
    unavailable = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: _Torch(_Cuda(available=False)),
    )
    zero = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: _Torch(_Cuda(names=[])),
    )
    assert unavailable["disposition"] == "gpu_runtime_unavailable"
    assert zero["disposition"] == "gpu_device_count_mismatch"


@pytest.mark.parametrize("name", ("AMD Radeon AI PRO R9700 ", "AMD Raphael", "Raphael iGPU"))
def test_wrong_or_raphael_single_device_is_identity_mismatch(name: str) -> None:
    observed = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: _Torch(_Cuda(names=[name])),
    )
    assert observed["disposition"] == "gpu_device_identity_mismatch"


def test_exact_single_r9700_passes_without_allocation_or_kernel_api() -> None:
    torch_module = _Torch()
    observed = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: torch_module,
    )
    visibility.require_passing_observation(observed)
    assert observed["disposition"] == "pass_exactly_one_r9700"
    assert torch_module.cuda.calls == [
        "is_available",
        "device_count",
        "get_device_name:0",
    ]
    assert all("tensor" not in call and "kernel" not in call for call in torch_module.calls)


@pytest.mark.parametrize(
    ("changed", "value"),
    (
        ("HIP_VISIBLE_DEVICES", "1"),
        ("CUDA_VISIBLE_DEVICES", "0"),
        ("ROCR_VISIBLE_DEVICES", "0"),
        ("HSA_OVERRIDE_GFX_VERSION", "11.0.0"),
        ("ONEAPI_DEVICE_SELECTOR", "gpu"),
    ),
)
def test_every_selector_deviation_rejects_before_runtime_import(
    changed: str,
    value: str,
) -> None:
    environment = _environment()
    environment[changed] = value
    calls = 0

    def loader() -> _Torch:
        nonlocal calls
        calls += 1
        return _Torch()

    observed = visibility.observe_visibility(
        environment=environment,
        torch_loader=loader,
    )
    assert observed["disposition"] == "gpu_selector_mismatch"
    assert calls == 0


@pytest.mark.parametrize("thread_name", policy.GPU_VISIBILITY_THREAD_ENVIRONMENT)
def test_every_native_thread_deviation_rejects_before_runtime_import(
    thread_name: str,
) -> None:
    environment = _environment()
    environment[thread_name] = "2"
    calls = 0

    def loader() -> _Torch:
        nonlocal calls
        calls += 1
        return _Torch()

    observed = visibility.observe_visibility(
        environment=environment,
        torch_loader=loader,
    )
    assert observed["disposition"] == "native_thread_mismatch"
    assert calls == 0


def test_torch_thread_mismatch_is_distinct_and_stops_before_cuda() -> None:
    torch_module = _Torch(intra=2, preserve_threads=True)
    observed = visibility.observe_visibility(
        environment=_environment(),
        torch_loader=lambda: torch_module,
    )
    assert observed["disposition"] == "native_thread_mismatch"
    assert torch_module.cuda.calls == []


def test_publication_failure_is_explicit_and_never_touches_fixed_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = _review()
    monkeypatch.setattr(
        policy,
        "preflight_source_review",
        lambda *_args, **_kwargs: (review, b"review"),
    )
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "0")
    for name in policy.GPU_VISIBILITY_UNSET_SELECTORS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("HSA_OVERRIDE_GFX_VERSION", raising=False)
    for name in policy.GPU_VISIBILITY_THREAD_ENVIRONMENT:
        monkeypatch.setenv(name, "1")

    def fail(_receipt: Any) -> tuple[str, str]:
        raise OSError("injected publication failure")

    receipt, passed = visibility.run_diagnostic(
        FILE_SHA,
        publisher=fail,
        torch_loader=lambda: _Torch(),
        repository_commit=COMMIT,
    )
    assert passed is False
    assert receipt["status"] == "failed"
    assert receipt["disposition"] == "gpu_visibility_receipt_publication_failure"
    assert receipt["reason"]["sanitized_exception_class"] == "OSError"


def test_receipt_is_canonical_self_hashed_and_plain_schema() -> None:
    receipt = _receipt()
    core = dict(receipt)
    declared = core.pop("content_sha256")
    assert policy.canonical_json_sha256(core) == declared
    assert type(receipt) is dict
    assert _validate(receipt) == receipt


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value.update(status="failed"),
        lambda value: value.update(disposition="gpu_runtime_unavailable"),
        lambda value: value["selector_observation"].update(hip_visible_devices="1"),
        lambda value: value["native_thread_observation"].update(torch_intra_op=2),
        lambda value: value["runtime_observation"].update(visible_device_count=0),
        lambda value: value["runtime_observation"]["ordered_devices"][0].update(name="Raphael"),
        lambda value: value["source_review"].update(file_sha256="d" * 64),
        lambda value: value["repository"].update(git_commit="e" * 64),
        lambda value: value["zero_access_evidence"].update(tensor_allocation_count=1),
    ),
)
def test_tampered_receipt_fields_reject_even_after_self_consistent_rehash(
    mutation: Any,
) -> None:
    receipt = json.loads(json.dumps(_receipt()))
    mutation(receipt)
    _rehash(receipt)
    with pytest.raises(PermissionError):
        _validate(receipt)


def test_stale_future_wrong_host_and_wrong_boot_reject() -> None:
    with pytest.raises(PermissionError, match="stale or future"):
        _validate(
            _receipt(),
            utc_now=datetime(2026, 7, 14, 12, 11, tzinfo=timezone.utc),
            monotonic_now=1660.0,
        )
    with pytest.raises(PermissionError, match="stale or future"):
        _validate(
            _receipt(),
            utc_now=datetime(2026, 7, 14, 11, 59, tzinfo=timezone.utc),
            monotonic_now=999.0,
        )
    with pytest.raises(PermissionError, match="host or boot"):
        _validate(_receipt(), hostname="host-b")
    with pytest.raises(PermissionError, match="host or boot"):
        _validate(
            _receipt(),
            boot_id="00000000-0000-0000-0000-000000000002",
        )


def test_private_no_clobber_publication_and_fixed_reader_round_trip(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "receipt.json"
    monkeypatch.setattr(visibility, "CANONICAL_RECEIPT_PATH", path)
    receipt = _receipt(
        utc_timestamp=(
            datetime.now(timezone.utc)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z")
        ),
        monotonic_seconds=5000.0,
    )
    file_sha, content_sha = visibility.publish_receipt(receipt)
    metadata = path.lstat()
    assert stat.S_ISREG(metadata.st_mode)
    assert stat.S_IMODE(metadata.st_mode) == 0o600
    assert metadata.st_nlink == 1
    assert path.read_bytes() == policy.canonical_json_bytes(receipt) + b"\n"
    assert visibility.validate_fixed_receipt(
        expected_file_sha256=file_sha,
        expected_content_sha256=content_sha,
        expected_source_review=_review(),
        expected_source_review_file_sha256=FILE_SHA,
        expected_repository_commit=COMMIT,
        hostname="host-a",
        boot_id="00000000-0000-0000-0000-000000000001",
        utc_now=datetime.now(timezone.utc),
        monotonic_now=5001.0,
    ) == receipt
    with pytest.raises(FileExistsError):
        visibility.publish_receipt(receipt)


def test_fixed_reader_rejects_symlink_hardlink_mode_and_caller_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt()
    raw = policy.canonical_json_bytes(receipt) + b"\n"
    target = tmp_path / "target"
    target.write_bytes(raw)
    target.chmod(0o600)
    path = tmp_path / "receipt"
    path.symlink_to(target)
    monkeypatch.setattr(visibility, "CANONICAL_RECEIPT_PATH", path)
    with pytest.raises(OSError):
        visibility.validate_fixed_receipt(
            expected_file_sha256=hashlib.sha256(raw).hexdigest(),
            expected_content_sha256=receipt["content_sha256"],
            expected_source_review=_review(),
            expected_source_review_file_sha256=FILE_SHA,
            expected_repository_commit=COMMIT,
        )
    path.unlink()
    os.link(target, path)
    with pytest.raises(PermissionError, match="insecure"):
        visibility._read_receipt_bytes()
    path.unlink()
    target.unlink()
    path.write_bytes(raw)
    path.chmod(0o644)
    with pytest.raises(PermissionError, match="insecure"):
        visibility._read_receipt_bytes()
    path.chmod(0o600)
    with pytest.raises(PermissionError, match="file hash"):
        visibility.validate_fixed_receipt(
            expected_file_sha256="f" * 64,
            expected_content_sha256=receipt["content_sha256"],
            expected_source_review=_review(),
            expected_source_review_file_sha256=FILE_SHA,
            expected_repository_commit=COMMIT,
        )


def test_receipt_path_is_fixed_and_no_alternate_path_is_callable() -> None:
    assert visibility.CANONICAL_RECEIPT_PATH == Path(
        "/tmp/lewm_go2_observable_camera_ray_fit_v4_n5_"
        "gate_aligned_raster_nll_v15_gpu_visibility_preflight_2026-07-14.json"
    )
    assert "path" not in inspect.signature(visibility.publish_receipt).parameters
    assert "path" not in inspect.signature(visibility.validate_fixed_receipt).parameters
    parser_source = inspect.getsource(visibility.parse_args)
    assert "receipt-path" not in parser_source


def test_import_surface_is_stdlib_only_until_injected_runtime_loader() -> None:
    tree = ast.parse((ROOT / policy.GPU_VISIBILITY_PREFLIGHT_RELATIVE_PATH).read_text())
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    assert "torch" not in imported
    source = inspect.getsource(visibility.observe_visibility)
    for forbidden in ("torch.tensor", ".to(", "Model(", "Optimizer(", "backward("):
        assert forbidden not in source


def test_natural_cli_bootstrap_suppresses_repo_bytecode_before_policy_import(
    tmp_path: Path,
) -> None:
    copied_root = tmp_path / "repo"
    script_path = copied_root / policy.GPU_VISIBILITY_PREFLIGHT_RELATIVE_PATH
    policy_path = copied_root / policy.POLICY_RELATIVE_PATH
    script_path.parent.mkdir(parents=True)
    policy_path.parent.mkdir(parents=True)
    shutil.copy2(ROOT / policy.GPU_VISIBILITY_PREFLIGHT_RELATIVE_PATH, script_path)
    shutil.copy2(ROOT / policy.POLICY_RELATIVE_PATH, policy_path)
    (copied_root / "lewm" / "__init__.py").write_text("", encoding="utf-8")
    (copied_root / "lewm" / "benchmarks" / "__init__.py").write_text(
        "", encoding="utf-8"
    )

    environment = dict(os.environ)
    for name in (
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONUSERBASE",
    ):
        environment.pop(name, None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(script_path), "--help"],
        cwd=copied_root,
        env=environment,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr.decode("utf-8", errors="replace")
    assert b"--source-review-sha256" in completed.stdout
    assert list(copied_root.rglob("__pycache__")) == []
    assert list(copied_root.rglob("*.py[co]")) == []


def test_actual_sha1_head_passes_complete_reviewed_closure_containment() -> None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode(
        "utf-8", errors="replace"
    )
    commit = completed.stdout.decode("ascii", errors="strict").strip()
    assert len(commit) == 40
    assert visibility._is_git_object_id(commit) is True
    assert visibility.current_reviewed_git_commit(
        _review_bound_to_git_commit(commit)
    ) == commit


def test_synthetic_sha256_git_object_id_passes_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    commit = "d" * 64
    amendment = b"synthetic amendment\n"
    clarification = b"synthetic clarification\n"
    payloads = {
        policy.V15_AMENDMENT_RELATIVE_PATH: amendment,
        policy.V15_TERMINAL_V14_PROOF_CLARIFICATION_RELATIVE_PATH: clarification,
    }
    monkeypatch.setattr(
        policy,
        "V15_AMENDMENT_FILE_SHA256",
        hashlib.sha256(amendment).hexdigest(),
    )
    monkeypatch.setattr(
        policy,
        "V15_TERMINAL_V14_PROOF_CLARIFICATION_FILE_SHA256",
        hashlib.sha256(clarification).hexdigest(),
    )

    def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        if command == ["git", "rev-parse", "--show-toplevel", "HEAD"]:
            output = f"{policy.ROOT}\n{commit}\n".encode("ascii")
            return subprocess.CompletedProcess(command, 0, output, b"")
        assert command[:2] == ["git", "show"]
        shown_commit, relative = command[2].split(":", 1)
        assert shown_commit == commit
        return subprocess.CompletedProcess(command, 0, payloads[relative], b"")

    monkeypatch.setattr(visibility.subprocess, "run", fake_run)
    review = {"successor_sources": {}, "successor_proofs": {}}
    assert visibility.current_reviewed_git_commit(review) == commit


def test_git_object_id_rejects_nonexact_strings_bad_lengths_and_nonhex() -> None:
    class StringSubclass(str):
        pass

    values = (
        None,
        False,
        40,
        "",
        "a" * 39,
        "a" * 41,
        "a" * 63,
        "a" * 65,
        "A" * 40,
        "g" * 40,
        StringSubclass("a" * 40),
    )
    assert all(visibility._is_git_object_id(value) is False for value in values)


def test_executor_order_is_static_review_receipt_live_freshness_then_reservation() -> None:
    source = (ROOT / policy.EXECUTOR_RELATIVE_PATH).read_text()
    tree = ast.parse(source)
    execute = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "execute_exact"
    )
    rendered = ast.unparse(execute)
    ordered = (
        "policy.preflight_static_authority()",
        "policy.preflight_source_review",
        "gpu_visibility.validate_fixed_receipt",
        "gpu_visibility.observe_visibility()",
        "gpu_visibility.require_passing_observation",
        "_preflight_generated_mutator_quiescence()",
        "_preflight_exact_output_freshness()",
        "_run_frozen_training",
    )
    offsets = [rendered.index(token) for token in ordered]
    assert offsets == sorted(offsets)


def test_trainer_reserves_before_any_dataset_or_rgb_open() -> None:
    tree = ast.parse((ROOT / policy.TRAINER_RELATIVE_PATH).read_text())
    training = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_run_training"
    )
    rendered = ast.unparse(training)
    reserve = rendered.index("reservation = _reserve_attempt(authority)")
    for token in (
        "DATASET_MANIFEST_RELATIVE_PATH",
        "preflight_exact_frozen_dataset_provenance",
        "load_exact_inputs",
        "decode_selected_rgb",
        "ObservableCameraRayEvidenceV4Model()",
    ):
        assert reserve < rendered.index(token)


def test_visibility_failures_have_zero_scientific_access_evidence() -> None:
    review = _review()
    for observation in (
        visibility.observe_visibility(
            environment={**_environment(), "HIP_VISIBLE_DEVICES": "1"},
            torch_loader=lambda: _Torch(),
        ),
        visibility.observe_visibility(
            environment=_environment(),
            torch_loader=lambda: _Torch(_Cuda(available=False)),
        ),
        visibility.observe_visibility(
            environment=_environment(),
            torch_loader=lambda: _Torch(_Cuda(names=[])),
        ),
    ):
        receipt = visibility.build_receipt(
            observation=observation,
            source_review=review,
            source_review_file_sha256=FILE_SHA,
            repository_commit=COMMIT,
            hostname="host-a",
            boot_id="00000000-0000-0000-0000-000000000001",
            process_id=1,
            utc_timestamp="2026-07-14T12:00:00.000000Z",
            monotonic_seconds=1.0,
        )
        assert receipt["zero_access_evidence"] == visibility.ZERO_ACCESS_EVIDENCE
        assert all(value == 0 for value in receipt["zero_access_evidence"].values())
        assert not any(receipt["authority"].values())
