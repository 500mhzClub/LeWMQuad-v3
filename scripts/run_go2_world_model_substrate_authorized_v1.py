#!/usr/bin/env python3
"""Supervise one exactly authorized world-model substrate development run.

This runner does not create authority.  It accepts only a separately committed
authorization document, verifies its caller-supplied byte/SHA identity and all
of its bound source/contract files, then executes the already reviewed manifest
DAG once.  The first runtime-input access remains inside S0, after S0 has
exclusively created the fresh pack root and thereby consumed the attempt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTHORITY_SCHEMA = "lewm_go2_world_model_substrate_execution_authority_v1"
AUTHORITY_STATUS = "AUTHORIZED_ONE_EXACT_DEVELOPMENT_ATTEMPT"
TERMINAL_SCHEMA = "lewm_go2_world_model_substrate_supervision_terminal_v1"
FOLLOW_ON_REVIEW_SCHEMA = (
    "lewm_go2_world_model_follow_on_independent_source_review_v1"
)
FOLLOW_ON_REVIEW_STATUS = "PASS_SOURCE_ONLY_NOT_AUTHORITY"
EXPECTED_MANIFEST_SCHEMA = (
    "lewm_go2_world_model_substrate_u700_sched3000_run_manifest_v1"
)
REVIEWED_PACKAGE_COMMIT = "9eeff2d030e73f9210fb140fe407f5cfd132b68d"
EXPECTED_REVIEWED_ARTIFACTS: dict[str, dict[str, Any]] = {
    "run_manifest": {
        "path": "docs/lewm_go2_world_model_substrate_u700_sched3000_run_manifest_2026-07-31.json",
        "byte_count": 19_819,
        "sha256": "90a2058f28e49030651090c3e0ae5f98d578b81bec95837be2c32bfeb9959805",
    },
    "proposal": {
        "path": "docs/lewm_go2_world_model_substrate_development_authority_proposal_2026-07-31.json",
        "byte_count": 8_306,
        "sha256": "d9f1193980a944d6401bdef05f848840c7c16ed98484e8914b060624babe62f9",
    },
    "independent_review": {
        "path": "docs/lewm_go2_world_model_next_tranche_independent_source_review_2026-07-31.json",
        "byte_count": 6_581,
        "sha256": "0e61c5b80bf8ef852b8f244e762a365026d48030b5f5074fc5ddbe696fcdd06a",
    },
    "sizing_witness": {
        "path": "docs/lewm_go2_world_model_counterfactual_pilot_sizing_decision_2026-07-31.md",
        "byte_count": 13_671,
        "sha256": "185f17e1fca2edd1d553fc59b9881f014fed647d1b724e8cb79fe8449cee788c",
    },
}
EXPECTED_SELECTOR_CORRECTION = {
    "path": "docs/lewm_go2_world_model_substrate_gpu_selector_correction_amendment_2026-07-31.json",
    "byte_count": 4_287,
    "sha256": "e961240b1f93dc4d60c62e5b8e12480d39ada239eca9dc740c8505e1d78a0166",
}


class SupervisionError(RuntimeError):
    """Raised when an authority, phase, receipt, or hard cap fails closed."""


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SupervisionError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def strict_json_bytes(payload: bytes, *, label: str) -> Any:
    """Decode strict JSON, rejecting duplicate keys and non-finite numbers."""

    try:
        return json.loads(
            payload,
            object_pairs_hook=_strict_object,
            parse_constant=lambda value: (_ for _ in ()).throw(
                SupervisionError(f"non-finite JSON value in {label}: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupervisionError(f"invalid JSON in {label}") from exc


def file_binding(path: Path) -> dict[str, Any]:
    selected = Path(path)
    if selected.is_symlink() or not selected.is_file():
        raise SupervisionError(f"bound file is absent, non-regular, or a symlink: {selected}")
    digest = hashlib.sha256()
    before = selected.stat()
    with selected.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = selected.stat()
    if (before.st_dev, before.st_ino, before.st_size) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
    ):
        raise SupervisionError(f"bound file changed while read: {selected}")
    return {
        "path": str(selected.resolve()),
        "byte_count": int(after.st_size),
        "sha256": digest.hexdigest(),
    }


def _resolve_bound_path(value: str) -> Path:
    selected = Path(value)
    return selected if selected.is_absolute() else REPO_ROOT / selected


def verify_binding(spec: Mapping[str, Any]) -> dict[str, Any]:
    expected_path = _resolve_bound_path(str(spec["path"]))
    actual = file_binding(expected_path)
    if actual["byte_count"] != spec.get("byte_count"):
        raise SupervisionError(f"byte-count mismatch for {spec['path']}")
    if actual["sha256"] != spec.get("sha256"):
        raise SupervisionError(f"SHA-256 mismatch for {spec['path']}")
    return actual


def validate_follow_on_source_review(
    record: Any,
    *,
    source_commit: str,
) -> dict[str, Any]:
    """Require the exact non-authorizing PASS semantics used by this tranche."""

    if (
        not isinstance(record, dict)
        or record.get("schema") != FOLLOW_ON_REVIEW_SCHEMA
        or record.get("status") != FOLLOW_ON_REVIEW_STATUS
        or record.get("authority_granted_by_this_document") is not False
        or record.get("reviewed_source_commit") != source_commit
        or record.get("remaining_findings") != []
    ):
        raise SupervisionError("follow-on source review is not an exact PASS")
    return record


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _require_commit_ancestor(commit: str, *, label: str) -> None:
    if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
        raise SupervisionError(f"{label} commit is not lowercase full-length hex")
    result = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=REPO_ROOT,
        check=False,
    )
    if result.returncode != 0:
        raise SupervisionError(f"{label} commit is not an ancestor of HEAD")


def _require_reviewed_package_ancestor() -> None:
    _require_commit_ancestor(REVIEWED_PACKAGE_COMMIT, label="reviewed package")


def _require_authority_committed_at_head(
    authority_path: Path,
    authority_binding: Mapping[str, Any],
) -> None:
    resolved = authority_path.resolve()
    try:
        relative = resolved.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise SupervisionError("authority must be tracked inside the repository") from exc
    result = subprocess.run(
        ["git", "show", f"HEAD:{relative.as_posix()}"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        raise SupervisionError("authority is not committed at HEAD")
    payload = result.stdout
    if len(payload) != authority_binding["byte_count"]:
        raise SupervisionError("committed authority byte count differs from working tree")
    if hashlib.sha256(payload).hexdigest() != authority_binding["sha256"]:
        raise SupervisionError("committed authority SHA-256 differs from working tree")


def load_and_validate_authority(
    authority_path: Path,
    *,
    expected_byte_count: int,
    expected_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate the non-self-authorizing execution envelope before S0."""

    binding = file_binding(authority_path)
    if binding["byte_count"] != expected_byte_count:
        raise SupervisionError("authority byte count disagrees with caller binding")
    if binding["sha256"] != expected_sha256:
        raise SupervisionError("authority SHA-256 disagrees with caller binding")
    _require_authority_committed_at_head(authority_path, binding)
    authority = strict_json_bytes(authority_path.read_bytes(), label="authority")
    if not isinstance(authority, dict):
        raise SupervisionError("authority must be a JSON object")
    if authority.get("schema") != AUTHORITY_SCHEMA:
        raise SupervisionError("authority schema is not executable by this supervisor")
    if authority.get("status") != AUTHORITY_STATUS:
        raise SupervisionError("authority status is not authorized")
    if authority.get("authority_granted_by_this_document") is not True:
        raise SupervisionError("authority grant is absent")
    authorizer = authority.get("authorizer")
    if not isinstance(authorizer, dict) or not authorizer.get("identity"):
        raise SupervisionError("durable authorizer identity is absent")
    if not authority.get("issued_at"):
        raise SupervisionError("authority issue timestamp is absent")
    source_commit = str(authority.get("source_commit") or "")
    _require_commit_ancestor(source_commit, label="authorized source")
    if authority.get("reviewed_package_commit") != REVIEWED_PACKAGE_COMMIT:
        raise SupervisionError("authority does not bind the reviewed package commit")
    _require_reviewed_package_ancestor()

    artifacts = authority.get("bound_artifacts")
    if not isinstance(artifacts, dict):
        raise SupervisionError("bound artifact map is absent")
    for name in ("run_manifest", "proposal", "independent_review", "sizing_witness"):
        spec = artifacts.get(name)
        if not isinstance(spec, dict):
            raise SupervisionError(f"authority omits bound artifact {name}")
        if spec != EXPECTED_REVIEWED_ARTIFACTS[name]:
            raise SupervisionError(f"authority changes reviewed artifact {name}")
        verify_binding(spec)
    source_review = artifacts.get("source_tranche_review")
    if not isinstance(source_review, dict):
        raise SupervisionError("authority omits the follow-on source review")
    verify_binding(source_review)
    source_review_path = _resolve_bound_path(str(source_review["path"]))
    source_review_record = strict_json_bytes(
        source_review_path.read_bytes(), label="follow-on source review"
    )
    validate_follow_on_source_review(
        source_review_record,
        source_commit=source_commit,
    )
    selector_correction = artifacts.get("gpu_selector_correction")
    if selector_correction != EXPECTED_SELECTOR_CORRECTION:
        raise SupervisionError("authority omits or changes the GPU correction")
    correction_path = _resolve_bound_path(str(selector_correction["path"]))
    verify_binding(selector_correction)
    correction = strict_json_bytes(
        correction_path.read_bytes(), label="GPU selector correction"
    )
    if (
        not isinstance(correction, dict)
        or correction.get("schema")
        != "lewm_go2_world_model_substrate_gpu_selector_correction_amendment_v1"
        or correction.get("authority_granted_by_this_document") is not False
        or correction.get("sole_effective_contract_delta", {}).get("old_value")
        != "HIP_VISIBLE_DEVICES=1"
        or correction.get("sole_effective_contract_delta", {}).get("new_value")
        != "HIP_VISIBLE_DEVICES=0"
    ):
        raise SupervisionError("GPU selector correction contract is invalid")

    supervisor = authority.get("external_supervisor")
    if not isinstance(supervisor, dict):
        raise SupervisionError("external supervisor contract is absent")
    supervisor_binding = verify_binding(supervisor["source_binding"])
    if Path(str(supervisor_binding["path"])) != Path(__file__).resolve():
        raise SupervisionError("authority binds a different supervisor source")
    if not supervisor.get("terminal_reviewer"):
        raise SupervisionError("terminal reviewer is absent")

    manifest_spec = artifacts["run_manifest"]
    manifest_path = _resolve_bound_path(str(manifest_spec["path"]))
    manifest = strict_json_bytes(manifest_path.read_bytes(), label="run manifest")
    if not isinstance(manifest, dict) or manifest.get("schema") != EXPECTED_MANIFEST_SCHEMA:
        raise SupervisionError("run manifest schema is invalid")
    if manifest.get("authority_granted_by_this_manifest") is not False:
        raise SupervisionError("reviewed manifest authority boundary changed")
    source_bindings = manifest.get("source_bindings")
    if not isinstance(source_bindings, list) or len(source_bindings) != 16:
        raise SupervisionError("reviewed manifest must bind exactly 16 runtime sources")
    for source in source_bindings:
        verify_binding(source)

    caps = authority.get("caps")
    runtime = manifest.get("runtime", {})
    accounting = manifest.get("accounting", {})
    if not isinstance(caps, dict):
        raise SupervisionError("authority caps are absent")
    required_caps = {
        "maximum_wall_seconds": runtime.get("wall_clock_ceiling_seconds"),
        "maximum_gpu_seconds": runtime.get("gpu_wall_clock_ceiling_seconds"),
        "maximum_training_updates": accounting.get("training_updates"),
        "schedule_horizon_updates": 3000,
        "maximum_total_rgb_leaf_opens": accounting.get("maximum_total_rgb_leaf_opens"),
    }
    if any(caps.get(key) != value for key, value in required_caps.items()):
        raise SupervisionError("authority caps differ from the reviewed manifest")
    if authority.get("attempt") != {
        "attempt_id": manifest["attempt"]["attempt_id"],
        "maximum_attempts": 1,
        "attempt_root": manifest["output_roots"]["attempt_root"],
        "must_be_absent_before_reservation": True,
        "reservation_consumes_attempt": True,
        "retry": False,
        "resume": False,
        "overwrite": False,
        "reuse_or_cleanup": False,
    }:
        raise SupervisionError("authority attempt contract differs from manifest")
    if authority.get("network_access") is not False:
        raise SupervisionError("network must remain disabled")
    if authority.get("scientific_claim_authorized") is not False:
        raise SupervisionError("substrate run cannot authorize a scientific claim")
    effective_manifest = copy.deepcopy(manifest)
    effective_manifest["runtime"]["environment"]["HIP_VISIBLE_DEVICES"] = "0"
    for phase in effective_manifest["phases"]:
        if phase["phase_id"] in {"S1_train", "S2_retention_baseline"}:
            phase["environment"]["HIP_VISIBLE_DEVICES"] = "0"
    runtime_authority = authority.get("runtime")
    if runtime_authority != {
        "physical_gpu_selector": "HIP_VISIBLE_DEVICES=0",
        "logical_torch_device": "cuda:0",
        "visible_device_count": 1,
        "device_name": "AMD Radeon AI PRO R9700",
        "gcn_arch": "gfx1201",
        "total_memory_bytes": 34208743424,
    }:
        raise SupervisionError("authority runtime does not bind the corrected device")
    return authority, effective_manifest


def validate_corrected_device_preflight() -> dict[str, Any]:
    """Verify selector 0 resolves to the exact reviewed physical device."""

    if os.environ.get("HIP_VISIBLE_DEVICES") != "0":
        raise SupervisionError("supervisor must itself start with HIP_VISIBLE_DEVICES=0")
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SupervisionError("corrected authority requires exactly one visible GPU")
    properties = torch.cuda.get_device_properties(0)
    receipt = {
        "selector": "0",
        "logical_device": "cuda:0",
        "visible_device_count": int(torch.cuda.device_count()),
        "device_name": str(properties.name),
        "gcn_arch": str(getattr(properties, "gcnArchName", "")),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": str(torch.__version__),
        "torch_hip": str(torch.version.hip),
    }
    if (
        receipt["device_name"] != "AMD Radeon AI PRO R9700"
        or receipt["gcn_arch"] != "gfx1201"
        or receipt["total_memory_bytes"] != 34208743424
    ):
        raise SupervisionError(f"corrected device preflight mismatch: {receipt}")
    return receipt


def _remaining_timeout(
    *,
    wall_start: float,
    gpu_elapsed: float,
    wall_cap: float,
    gpu_cap: float,
    gpu_phase: bool,
) -> float:
    wall_remaining = wall_cap - (time.monotonic() - wall_start)
    gpu_remaining = gpu_cap - gpu_elapsed if gpu_phase else math.inf
    remaining = min(wall_remaining, gpu_remaining)
    if remaining <= 0.0:
        raise SupervisionError("hard wall/GPU time ceiling exhausted")
    return remaining


def run_phase(
    phase_id: str,
    argv: list[str],
    *,
    environment: Mapping[str, str],
    wall_start: float,
    gpu_elapsed: float,
    wall_cap: float,
    gpu_cap: float,
    gpu_phase: bool,
) -> tuple[dict[str, Any], float]:
    """Run one phase once under the remaining hard wall/GPU ceiling."""

    timeout = _remaining_timeout(
        wall_start=wall_start,
        gpu_elapsed=gpu_elapsed,
        wall_cap=wall_cap,
        gpu_cap=gpu_cap,
        gpu_phase=gpu_phase,
    )
    env = os.environ.copy()
    env.update({str(key): str(value) for key, value in environment.items()})
    started = time.monotonic()
    process = subprocess.Popen(
        argv,
        cwd=REPO_ROOT,
        env=env,
        start_new_session=True,
    )
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process_group(process)
        raise SupervisionError(f"phase {phase_id} exceeded a hard time ceiling") from exc
    except BaseException:
        _terminate_process_group(process)
        raise
    elapsed = time.monotonic() - started
    if returncode != 0:
        _terminate_process_group(process)
        raise SupervisionError(f"phase {phase_id} exited {returncode}")
    if gpu_phase:
        gpu_elapsed += elapsed
    return {
        "phase_id": phase_id,
        "argv": argv,
        "elapsed_seconds": elapsed,
        "gpu_phase": gpu_phase,
        "exit_code": int(returncode),
    }, gpu_elapsed


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    """Terminate every surviving child in a supervised phase process group."""

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def _raise_on_termination_signal(signum: int, _frame: Any) -> None:
    raise SupervisionError(f"supervisor received signal {signum}")


def _finite_scalars(value: Any, prefix: str = "") -> dict[str, float]:
    result: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_finite_scalars(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            result.update(_finite_scalars(child, f"{prefix}[{index}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        number = float(value)
        if math.isfinite(number):
            result[prefix] = number
    return result


def retention_parity_differences(
    predecessor_receipt: Mapping[str, Any],
    update_zero_receipt: Mapping[str, Any],
) -> dict[str, float]:
    """Return absolute differences for shared finite retention scalars."""

    left = _finite_scalars(predecessor_receipt.get("spatial_retention", {}).get("evaluation", {}))
    right = _finite_scalars(update_zero_receipt.get("spatial_retention", {}).get("evaluation", {}))
    shared = sorted(set(left) & set(right))
    if not shared:
        raise SupervisionError("u0/predecessor receipts have no shared retention scalars")
    return {key: abs(left[key] - right[key]) for key in shared}


def _load_complete_receipt(path: Path, schema: str) -> dict[str, Any]:
    binding = file_binding(path)
    value = strict_json_bytes(path.read_bytes(), label=str(path))
    if not isinstance(value, dict) or value.get("schema") != schema or value.get("status") != "COMPLETE":
        raise SupervisionError(f"required receipt is not complete: {path}")
    value["_independent_binding"] = binding
    return value


def _write_terminal(attempt_root: Path, payload: Mapping[str, Any]) -> Path | None:
    if not attempt_root.is_dir() or attempt_root.is_symlink():
        return None
    path = attempt_root / "terminal_supervision.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(dict(payload), stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return path


def _snapshot_by_update(trace: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    snapshots: dict[int, dict[str, Any]] = {}
    for record in trace.get("records", []):
        if not isinstance(record, dict) or not isinstance(record.get("snapshot"), dict):
            continue
        update = record.get("update")
        if isinstance(update, bool) or not isinstance(update, int):
            continue
        snapshots[update] = dict(record["snapshot"])
    return snapshots


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--expected-authority-byte-count", type=int, required=True)
    parser.add_argument("--expected-authority-sha256", required=True)
    args = parser.parse_args()
    if args.expected_authority_byte_count <= 0:
        parser.error("authority byte count must be positive")
    if len(args.expected_authority_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in args.expected_authority_sha256
    ):
        parser.error("authority SHA-256 must be lowercase hexadecimal")

    authority, manifest = load_and_validate_authority(
        args.authority,
        expected_byte_count=args.expected_authority_byte_count,
        expected_sha256=args.expected_authority_sha256,
    )
    device_preflight = validate_corrected_device_preflight()
    attempt_root = REPO_ROOT / manifest["output_roots"]["attempt_root"]
    if attempt_root.exists() or attempt_root.is_symlink():
        raise SupervisionError(f"attempt root is not fresh: {attempt_root}")

    signal.signal(signal.SIGINT, _raise_on_termination_signal)
    signal.signal(signal.SIGTERM, _raise_on_termination_signal)

    caps = authority["caps"]
    wall_cap = float(caps["maximum_wall_seconds"])
    gpu_cap = float(caps["maximum_gpu_seconds"])
    wall_start = time.monotonic()
    gpu_elapsed = 0.0
    phase_receipts: list[dict[str, Any]] = []
    failure: str | None = None
    try:
        phases = {phase["phase_id"]: phase for phase in manifest["phases"]}
        for phase_id in ("S0_pack", "S1_train", "S2_retention_baseline"):
            phase = phases[phase_id]
            receipt, gpu_elapsed = run_phase(
                phase_id,
                [str(value) for value in phase["argv"]],
                environment=phase.get("environment", {}),
                wall_start=wall_start,
                gpu_elapsed=gpu_elapsed,
                wall_cap=wall_cap,
                gpu_cap=gpu_cap,
                gpu_phase=phase_id != "S0_pack",
            )
            phase_receipts.append(receipt)

        trace_path = REPO_ROOT / phases["S1_train"]["required_receipt"]
        trace = _load_complete_receipt(trace_path, "dev_temporal_jepa_scaled_v4")
        snapshots = _snapshot_by_update(trace)
        s3 = phases["S3_retention_snapshots"]
        template = [str(value) for value in s3["argv_template"]]
        for update in s3["ordered_updates"]:
            snapshot = snapshots.get(int(update))
            if snapshot is None:
                raise SupervisionError(f"S1 trace omits update {update} snapshot")
            verified = verify_binding(snapshot)
            argv = [
                value.replace("{CHECKPOINT_PATH_FROM_S1_FINAL_TRACE}", verified["path"])
                .replace("{CHECKPOINT_SHA256_FROM_S1_FINAL_TRACE}", verified["sha256"])
                .replace("{U:06d}", f"{int(update):06d}")
                .replace("{U}", str(int(update)))
                for value in template
            ]
            receipt, gpu_elapsed = run_phase(
                f"S3_retention_update_{int(update):06d}",
                argv,
                environment=manifest["runtime"]["environment"],
                wall_start=wall_start,
                gpu_elapsed=gpu_elapsed,
                wall_cap=wall_cap,
                gpu_cap=gpu_cap,
                gpu_phase=True,
            )
            phase_receipts.append(receipt)

        retention_root = REPO_ROOT / manifest["output_roots"]["retention_root"]
        predecessor = _load_complete_receipt(
            retention_root / "predecessor.json",
            "dev_temporal_retention_composability_v4",
        )
        update_zero = _load_complete_receipt(
            retention_root / "update_000000.json",
            "dev_temporal_retention_composability_v4",
        )
        differences = retention_parity_differences(predecessor, update_zero)
        maximum_difference = max(differences.values())
        if maximum_difference > 1e-7:
            raise SupervisionError(
                f"u0/predecessor parity failed: max absolute difference {maximum_difference}"
            )
        for required in s3["required_receipts"]:
            _load_complete_receipt(
                REPO_ROOT / required,
                "dev_temporal_retention_composability_v4",
            )
    except BaseException as exc:
        failure = f"{type(exc).__name__}: {exc}"

    wall_elapsed = time.monotonic() - wall_start
    if failure is None and wall_elapsed > wall_cap:
        failure = (
            "SupervisionError: terminal validation exceeded the hard wall "
            f"ceiling ({wall_elapsed:.6f} > {wall_cap:.6f} seconds)"
        )
    if failure is None and gpu_elapsed > gpu_cap:
        failure = (
            "SupervisionError: GPU phases exceeded the hard GPU ceiling "
            f"({gpu_elapsed:.6f} > {gpu_cap:.6f} seconds)"
        )
    terminal = {
        "schema": TERMINAL_SCHEMA,
        "status": (
            "COMPLETE_PENDING_TERMINAL_REVIEW"
            if failure is None
            else "CONSUMED_TERMINAL_FAILURE"
        ),
        "citable_as_scientific_evidence": False,
        "authorizes_retry_or_resume": False,
        "authority_binding": file_binding(args.authority),
        "source_commit": authority["source_commit"],
        "execution_head": _git_head(),
        "device_preflight": device_preflight,
        "attempt_root": str(attempt_root.resolve()),
        "wall_elapsed_seconds": wall_elapsed,
        "gpu_phase_elapsed_seconds": gpu_elapsed,
        "wall_ceiling_seconds": wall_cap,
        "gpu_ceiling_seconds": gpu_cap,
        "phase_receipts": phase_receipts,
        "failure": failure,
        "terminal_reviewer": authority["external_supervisor"]["terminal_reviewer"],
        "automatic_checkpoint_selection_performed": False,
        "scientific_verdict_emitted": False,
    }
    terminal_path = _write_terminal(attempt_root, terminal)
    if terminal_path is None:
        print(f"pre-reservation failure: {failure}", file=sys.stderr)
        return 2
    print(json.dumps({**terminal, "terminal_path": str(terminal_path)}, indent=2))
    return 0 if failure is None else 1


if __name__ == "__main__":
    raise SystemExit(main())
