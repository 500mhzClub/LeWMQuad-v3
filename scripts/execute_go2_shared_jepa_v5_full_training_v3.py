#!/usr/bin/env python3
"""Fail-closed exact reserver for Shared JEPA V5 full training V3.

This entry point is standard-library only.  The fixed execution manifest must
be complete before the canonical namespace is touched.  A successful
reservation is durable before the fixed trainer process is spawned.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping, Sequence


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from lewm.benchmarks import go2_shared_jepa_v5_full_training_v3_policy as policy


if __name__ == "__main__":
    ROOT = SCRIPT_ROOT

    def _directory_flags() -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        directory = getattr(os, "O_DIRECTORY", 0)
        if not nofollow or not directory:
            raise PermissionError("exact execution requires no-follow directory opens")
        return os.O_RDONLY | nofollow | directory | getattr(os, "O_CLOEXEC", 0)


    def _file_flags() -> int:
        nofollow = getattr(os, "O_NOFOLLOW", 0)
        if not nofollow:
            raise PermissionError("exact execution requires no-follow file opens")
        return (
            os.O_RDONLY
            | nofollow
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0)
        )


    def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_uid),
            int(metadata.st_gid),
        )


    def _stable_file_fingerprint(metadata: os.stat_result) -> tuple[int, ...]:
        return (
            int(metadata.st_dev),
            int(metadata.st_ino),
            int(metadata.st_mode),
            int(metadata.st_nlink),
            int(metadata.st_uid),
            int(metadata.st_gid),
            int(metadata.st_size),
            int(metadata.st_mtime_ns),
            int(metadata.st_ctime_ns),
        )


    @dataclass(frozen=True)
    class ChainEntry:
        parent_fd: int
        name: str
        child_fd: int
        identity: tuple[int, int, int, int, int]


    @dataclass
    class DirectoryChain:
        descriptors: list[int]
        entries: list[ChainEntry]
        path_fds: dict[Path, int]
        closed: bool = False


    @dataclass(frozen=True)
    class ExactReservation:
        directory_fd: int
        parent_fd: int
        directory_name: str
        directory_identity: tuple[int, int]
        chain: DirectoryChain
        manifest: Mapping[str, Any]
        manifest_file_sha256: str
        reservation: Mapping[str, Any]
        reservation_raw: bytes


    def _assert_chain(chain: DirectoryChain) -> None:
        if chain.closed:
            raise PermissionError("exact directory chain is closed")
        for entry in chain.entries:
            named = os.stat(entry.name, dir_fd=entry.parent_fd, follow_symlinks=False)
            opened = os.fstat(entry.child_fd)
            if (
                stat.S_ISLNK(named.st_mode)
                or not stat.S_ISDIR(named.st_mode)
                or not stat.S_ISDIR(opened.st_mode)
                or _identity(named) != entry.identity
                or _identity(opened) != entry.identity
            ):
                raise PermissionError("exact canonical ancestry identity changed")


    def _open_parent_chain(final_parent: Path) -> DirectoryChain:
        final_parent = Path(final_parent)
        if (
            not final_parent.is_absolute()
            or not final_parent.is_relative_to(ROOT)
            or any(part in {"", ".", ".."} for part in final_parent.parts[1:])
        ):
            raise PermissionError("exact parent escaped the repository")
        filesystem_root = Path(final_parent.anchor)
        anchor_fd = os.open(filesystem_root, _directory_flags())
        chain = DirectoryChain(
            descriptors=[anchor_fd],
            entries=[],
            path_fds={filesystem_root: anchor_fd},
        )
        try:
            parent_fd = anchor_fd
            current = filesystem_root
            repository_depth = len(ROOT.parts) - 1
            for index, component in enumerate(final_parent.parts[1:]):
                try:
                    before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                except FileNotFoundError:
                    if index < repository_depth:
                        raise PermissionError("repository component is missing")
                    os.mkdir(component, 0o700, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    before = os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
                if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                    raise PermissionError("exact parent component is not a directory")
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                chain.descriptors.append(child_fd)
                entry = ChainEntry(parent_fd, component, child_fd, _identity(before))
                if _identity(os.fstat(child_fd)) != entry.identity:
                    raise PermissionError("exact parent component changed during open")
                chain.entries.append(entry)
                current = current / component
                chain.path_fds[current] = child_fd
                parent_fd = child_fd
            _assert_chain(chain)
            return chain
        except BaseException:
            _close_chain(chain)
            raise


    def _close_chain(chain: DirectoryChain) -> None:
        if chain.closed:
            return
        chain.closed = True
        for descriptor in reversed(chain.descriptors):
            os.close(descriptor)


    def _read_repo_authority(relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("authority path escaped the repository")
        root_fd = os.open(ROOT, _directory_flags())
        descriptors = [root_fd]
        parent_fd = root_fd
        try:
            for component in path.parts[:-1]:
                child_fd = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(child_fd)
                parent_fd = child_fd
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent_fd)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise PermissionError("authority file is not singly linked")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if _stable_file_fingerprint(before) != _stable_file_fingerprint(after):
                    raise RuntimeError("authority file changed while read")
                return b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _load_ready_manifest(expected_file_sha256: str) -> tuple[dict[str, Any], bytes]:
        if not policy.is_sha256(expected_file_sha256):
            raise ValueError("execution-manifest SHA-256 is malformed")
        raw = _read_repo_authority(policy.EXACT_EXECUTION_MANIFEST_RELATIVE_PATH)
        if hashlib.sha256(raw).hexdigest() != expected_file_sha256:
            raise PermissionError("execution-manifest file hash changed")
        value = policy.parse_canonical_json(raw, name="exact execution manifest")
        return policy.validate_execution_manifest(value, require_ready=True), raw


    def _assert_claim(reservation: ExactReservation) -> None:
        _assert_chain(reservation.chain)
        opened = os.fstat(reservation.directory_fd)
        named = os.stat(
            reservation.directory_name,
            dir_fd=reservation.parent_fd,
            follow_symlinks=False,
        )
        if (
            not stat.S_ISDIR(opened.st_mode)
            or not stat.S_ISDIR(named.st_mode)
            or (opened.st_dev, opened.st_ino) != reservation.directory_identity
            or (named.st_dev, named.st_ino) != reservation.directory_identity
        ):
            raise PermissionError("exact claim identity changed")


    def _write_leaf_exclusive(
        reservation: ExactReservation,
        name: str,
        raw: bytes,
    ) -> None:
        _assert_claim(reservation)
        if Path(name).name != name or name in {"", ".", ".."}:
            raise PermissionError("exact publication leaf escaped")
        descriptor = os.open(
            name,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=reservation.directory_fd,
        )
        try:
            view = memoryview(raw)
            while view:
                count = os.write(descriptor, view)
                if count <= 0:
                    raise OSError("exact publication write made no progress")
                view = view[count:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(reservation.directory_fd)
        _assert_claim(reservation)


    def _reserve_exact(
        manifest: Mapping[str, Any],
        manifest_file_sha256: str,
    ) -> ExactReservation:
        final = policy.CANONICAL_EXACT_ROOT
        chain = _open_parent_chain(final.parent)
        parent_fd = chain.path_fds[final.parent]
        directory_fd = -1
        try:
            os.mkdir(final.name, 0o700, dir_fd=parent_fd)
            os.fsync(parent_fd)
            _assert_chain(chain)
            directory_fd = os.open(final.name, _directory_flags(), dir_fd=parent_fd)
            metadata = os.fstat(directory_fd)
            directory_identity = (int(metadata.st_dev), int(metadata.st_ino))
            core = {
                "schema": policy.EXACT_RESERVATION_SCHEMA,
                "status": "reserved_before_torch_model_checkpoint_or_payload",
                "namespace": policy.EXACT_ROOT_RELATIVE_PATH,
                "directory_identity": list(directory_identity),
                "execution_manifest_file_sha256": manifest_file_sha256,
                "execution_manifest_content_sha256": manifest["content_sha256"],
                "required_exact_bindings": dict(manifest["required_exact_bindings"]),
                "preflight_process_state_inherited": False,
                "torch_or_payload_opened_before_reservation": False,
                "attempt_index": 1,
                "maximum_attempts": 1,
                "retry_authorized": False,
                "g2_authorized": False,
                "heldout_authorized": False,
                "runtime_navigation_hardware_authorized": False,
                "production_or_promotion_authorized": False,
            }
            reservation_value = policy.content_value(core)
            reservation_raw = policy.canonical_json_bytes(reservation_value) + b"\n"
            reservation = ExactReservation(
                directory_fd=directory_fd,
                parent_fd=parent_fd,
                directory_name=final.name,
                directory_identity=directory_identity,
                chain=chain,
                manifest=dict(manifest),
                manifest_file_sha256=manifest_file_sha256,
                reservation=reservation_value,
                reservation_raw=reservation_raw,
            )
            _write_leaf_exclusive(reservation, "reservation.json", reservation_raw)
            return reservation
        except BaseException:
            if directory_fd >= 0:
                os.close(directory_fd)
            _close_chain(chain)
            raise


    def _fresh_exact_environment() -> dict[str, str]:
        environment = dict(os.environ)
        for name in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE"):
            environment.pop(name, None)
        environment["PYTHONNOUSERSITE"] = "1"
        environment["HIP_VISIBLE_DEVICES"] = "0"
        environment["ROCR_VISIBLE_DEVICES"] = "0"
        environment.pop("HSA_OVERRIDE_GFX_VERSION", None)
        for name in (
            "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            environment[name] = "1"
        return environment


    def _run_fixed_child(
        reservation: ExactReservation,
        relative_source: str,
        role: str,
    ) -> dict[str, Any]:
        if relative_source not in {
            policy.EXACT_TRAINER_RELATIVE_PATH,
            policy.EXACT_VERIFIER_RELATIVE_PATH,
        }:
            raise PermissionError("exact child source is not fixed")
        source_hash_name = (
            "exact_trainer_source_sha256"
            if role == "trainer"
            else "exact_verifier_source_sha256"
        )
        expected_source_sha256 = reservation.manifest["required_exact_bindings"][
            source_hash_name
        ]
        child = ROOT / relative_source
        completed = subprocess.run(
            [
                sys.executable,
                "-I",
                "-B",
                str(child),
                "--claim-fd",
                str(reservation.directory_fd),
                "--parent-fd",
                str(reservation.parent_fd),
                "--expected-directory-name",
                reservation.directory_name,
                "--execution-manifest-sha256",
                reservation.manifest_file_sha256,
                "--execution-manifest-content-sha256",
                str(reservation.manifest["content_sha256"]),
                "--expected-source-sha256",
                str(expected_source_sha256),
            ],
            cwd=ROOT,
            env=_fresh_exact_environment(),
            pass_fds=(reservation.directory_fd, reservation.parent_fd),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"fixed exact {role} process failed")
        return policy.parse_canonical_json(
            completed.stdout,
            name=f"fixed exact {role} summary",
        )


    def _read_relative_artifact(directory_fd: int, relative: str) -> bytes:
        path = Path(relative)
        if path.is_absolute() or not path.parts or ".." in path.parts:
            raise PermissionError("exact artifact path escaped")
        descriptors: list[int] = []
        parent_fd = directory_fd
        try:
            for component in path.parts[:-1]:
                descriptor = os.open(component, _directory_flags(), dir_fd=parent_fd)
                descriptors.append(descriptor)
                parent_fd = descriptor
            descriptor = os.open(path.name, _file_flags(), dir_fd=parent_fd)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
                    raise PermissionError("exact artifact is not singly linked")
                chunks: list[bytes] = []
                while chunk := os.read(descriptor, 1024 * 1024):
                    chunks.append(chunk)
                after = os.fstat(descriptor)
                if _stable_file_fingerprint(before) != _stable_file_fingerprint(after):
                    raise RuntimeError("exact artifact changed while read")
                return b"".join(chunks)
            finally:
                os.close(descriptor)
        finally:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


    def _inventory_files(directory_fd: int) -> set[str]:
        observed: set[str] = set()
        allowed_directories = {
            parent.as_posix()
            for relative in policy.EXACT_INVENTORY
            for parent in Path(relative).parents
            if parent != Path(".")
        }

        def walk(parent_fd: int, prefix: Path) -> None:
            for name in os.listdir(parent_fd):
                if Path(name).name != name or name in {"", ".", ".."}:
                    raise PermissionError("exact inventory name escaped")
                metadata = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
                relative = prefix / name
                if stat.S_ISDIR(metadata.st_mode):
                    if relative.as_posix() not in allowed_directories:
                        raise PermissionError("exact inventory contains an extra directory")
                    child_fd = os.open(name, _directory_flags(), dir_fd=parent_fd)
                    try:
                        walk(child_fd, relative)
                    finally:
                        os.close(child_fd)
                elif stat.S_ISREG(metadata.st_mode) and metadata.st_nlink == 1:
                    observed.add(relative.as_posix())
                else:
                    raise PermissionError("exact inventory contains a special or linked entry")

        walk(directory_fd, Path())
        return observed


    def _publish_completion(
        reservation: ExactReservation,
        trainer_summary: Mapping[str, Any],
        verification: Mapping[str, Any],
    ) -> dict[str, Any]:
        _assert_claim(reservation)
        if (
            trainer_summary.get("schema")
            != "lewm_go2_shared_jepa_v5_full_training_v3_trainer_summary_v1"
            or trainer_summary.get("status") != "trainer_publication_complete"
            or verification.get("schema")
            != "lewm_go2_shared_jepa_v5_full_training_v3_verification_v1"
            or verification.get("status") != "independently_reconstructed_pass"
            or trainer_summary.get("directory_identity")
            != list(reservation.directory_identity)
            or trainer_summary.get("execution_manifest_content_sha256")
            != reservation.manifest["content_sha256"]
            or verification.get("claim_identity")
            != list(reservation.directory_identity)
            or verification.get("execution_manifest_file_sha256")
            != reservation.manifest_file_sha256
            or verification.get("execution_manifest_content_sha256")
            != reservation.manifest["content_sha256"]
            or verification.get("trainer_metrics_trusted") is not False
            or verification.get("g2_open_count") != 0
            or verification.get("heldout_open_count") != 0
            or verification.get("runtime_navigation_hardware_open_count") != 0
            or verification.get("production_or_promotion_open_count") != 0
            or verification.get("runtime_ready") is not False
        ):
            raise PermissionError("exact trainer or verifier did not pass")
        declared = trainer_summary.get("artifacts_before_completion")
        if not isinstance(declared, Mapping):
            raise ValueError("exact trainer artifact bindings are missing")
        expected_paths = list(policy.EXACT_INVENTORY[:-1])
        if list(declared) != expected_paths:
            raise PermissionError("exact trainer inventory changed")
        if _inventory_files(reservation.directory_fd) != set(expected_paths):
            raise PermissionError("exact directory inventory changed before completion")
        observed: dict[str, Any] = {}
        for relative in expected_paths:
            raw = _read_relative_artifact(reservation.directory_fd, relative)
            binding = policy.artifact_binding(relative, raw)
            declared_binding = declared[relative]
            if (
                not isinstance(declared_binding, Mapping)
                or binding["file_sha256"] != declared_binding.get("file_sha256")
                or binding["byte_count"] != declared_binding.get("byte_count")
            ):
                raise PermissionError(f"exact artifact binding changed: {relative}")
            observed[relative] = dict(declared_binding)
        if trainer_summary.get(
            "artifacts_before_completion_sha256"
        ) != policy.canonical_json_sha256(observed):
            raise PermissionError("trainer inventory commitment changed")
        if verification.get("artifacts_before_completion_sha256") != policy.canonical_json_sha256(observed):
            raise PermissionError("independent verifier inventory commitment changed")
        reconstruction = verification.get("reconstruction")
        if (
            not isinstance(reconstruction, Mapping)
            or reconstruction.get("selected_update")
            != trainer_summary.get("selected_update")
            or reconstruction.get("pre_g2_candidate_state_sha256")
            != trainer_summary.get("pre_g2_candidate_state_sha256")
            or reconstruction.get("trainer_metrics_trusted") is not False
            or reconstruction.get("raw_inputs_and_checkpoints_reopened") is not True
        ):
            raise PermissionError("trainer and independent reconstruction differ")
        core = {
            "schema": policy.EXACT_COMPLETION_SCHEMA,
            "status": "completed_after_independent_reconstruction",
            "namespace": policy.EXACT_ROOT_RELATIVE_PATH,
            "directory_identity": list(reservation.directory_identity),
            "execution_manifest_file_sha256": reservation.manifest_file_sha256,
            "execution_manifest_content_sha256": reservation.manifest["content_sha256"],
            "ordered_inventory": list(policy.EXACT_INVENTORY),
            "artifacts_before_completion": observed,
            "trainer_summary_content_sha256": trainer_summary["content_sha256"],
            "independent_verification": dict(verification),
            "runtime_ready": False,
            "g2_authorized": False,
            "heldout_authorized": False,
            "production_or_promotion_authorized": False,
            "retry_authorized": False,
        }
        completed = policy.content_value(core)
        raw = policy.canonical_json_bytes(completed) + b"\n"
        _write_leaf_exclusive(reservation, "completed.json", raw)
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.parent_fd)
        return completed


    def _terminalize_failure(
        reservation: ExactReservation,
        error: BaseException,
    ) -> None:
        core = {
            "schema": policy.EXACT_FAILURE_SCHEMA,
            "status": "terminal_failure_no_retry",
            "failure_class": type(error).__name__,
            "namespace": policy.EXACT_ROOT_RELATIVE_PATH,
            "execution_manifest_file_sha256": reservation.manifest_file_sha256,
            "directory_identity": list(reservation.directory_identity),
            "g2_authorized": False,
            "heldout_authorized": False,
            "runtime_navigation_hardware_authorized": False,
            "production_or_promotion_authorized": False,
            "retry_authorized": False,
        }
        raw = policy.canonical_json_bytes(policy.content_value(core)) + b"\n"
        try:
            _write_leaf_exclusive(reservation, "failed.json", raw)
        except FileExistsError:
            pass
        os.fsync(reservation.directory_fd)
        os.fsync(reservation.parent_fd)


    def _close_reservation(reservation: ExactReservation) -> None:
        os.close(reservation.directory_fd)
        _close_chain(reservation.chain)


    def _orchestrate_exact(
        reserve_operation: Callable[[], ExactReservation],
        trainer_operation: Callable[[ExactReservation], Mapping[str, Any]],
        verifier_operation: Callable[[ExactReservation], Mapping[str, Any]],
        completion_operation: Callable[[ExactReservation, Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]],
        failure_operation: Callable[[ExactReservation, BaseException], None],
        close_operation: Callable[[ExactReservation], None],
    ) -> Mapping[str, Any]:
        reservation = reserve_operation()
        try:
            trainer_summary = trainer_operation(reservation)
            verification = verifier_operation(reservation)
            return completion_operation(reservation, trainer_summary, verification)
        except BaseException as error:
            failure_operation(reservation, error)
            raise
        finally:
            close_operation(reservation)


    def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--execution-manifest-sha256", required=True)
        args = parser.parse_args(argv)
        if not policy.is_sha256(args.execution_manifest_sha256):
            raise ValueError("execution-manifest SHA-256 is malformed")
        return args


    arguments = parse_args()
    exact_manifest, _manifest_raw = _load_ready_manifest(
        arguments.execution_manifest_sha256
    )
    completion = _orchestrate_exact(
        lambda: _reserve_exact(exact_manifest, arguments.execution_manifest_sha256),
        lambda reservation: _run_fixed_child(
            reservation,
            policy.EXACT_TRAINER_RELATIVE_PATH,
            "trainer",
        ),
        lambda reservation: _run_fixed_child(
            reservation,
            policy.EXACT_VERIFIER_RELATIVE_PATH,
            "verifier",
        ),
        _publish_completion,
        _terminalize_failure,
        _close_reservation,
    )
    print(policy.canonical_json_bytes(completion).decode("ascii"))
