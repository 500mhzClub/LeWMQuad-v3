#!/usr/bin/env python3
"""Run the lean Shared V5 controller on one development maze.

This entrypoint loads only explicit qualified artifacts and uses the committed
kinematic Genesis development stack.  Maze scoring is imported and called only
after the fixed controller loop has returned.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.models.shared_observable_camera_ray_jepa_v5 import (  # noqa: E402
    SharedObservableCameraRayJepaV5,
    SharedObservableCameraRayJepaV5Config,
    shared_output_contract_v5,
    tensor_state_dict_sha256,
)
from lewm.models.shared_v5_target_observation_head_v1 import (  # noqa: E402
    SharedV5TargetObservationHeadConfigV1,
    SharedV5TargetObservationHeadV1,
)
from lewm.models.two_resolution_frontier_value_head_v1 import (  # noqa: E402
    TwoResolutionFrontierValueHeadConfigV1,
    TwoResolutionFrontierValueHeadV1,
)
from lewm.navigation.shared_v5_dev_runtime import (  # noqa: E402
    ControllerRun,
    RuntimeArtifactBindings,
    SharedV5DevMazeRuntime,
    SharedV5DevRuntimeConfigurationError,
    TargetConfirmationCalibration,
)
from lewm.benchmarks.go2_shared_jepa_v5_full_training_v4_policy import pre_g2_candidate_checkpoint_core  # noqa: E402
from scripts.go2_shared_jepa_v5_one_shot import _validate_publication  # noqa: E402


_ACTUAL_OPENS: list[dict[str, object]] = []
def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _exact_self_hashed(value: object, *, schema: str, keys: set[str], purpose: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != keys or value.get("schema") != schema: raise SharedV5DevRuntimeConfigurationError(f"{purpose} schema/fields changed")
    core = dict(value)
    if core.pop("content_sha256", None) != _canonical_sha256(core): raise SharedV5DevRuntimeConfigurationError(f"{purpose} content hash changed")
    return value


def _file_bytes(path: Path, *, purpose: str) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} must not be a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {purpose}: {resolved}")
    raw = resolved.read_bytes()
    _ACTUAL_OPENS.append({"purpose": purpose, "path": str(resolved), "byte_count": len(raw),
                          "file_sha256": hashlib.sha256(raw).hexdigest()})
    return raw


def file_sha256(path: Path, *, purpose: str = "artifact") -> str:
    return hashlib.sha256(_file_bytes(path, purpose=purpose)).hexdigest()


def _json_artifact(path: Path, *, purpose: str) -> tuple[dict[str, Any], str]:
    raw = _file_bytes(path, purpose=purpose)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} is not UTF-8 JSON: {path}"
        ) from exc
    if type(value) is not dict:
        raise SharedV5DevRuntimeConfigurationError(f"{purpose} root must be an object")
    return value, hashlib.sha256(raw).hexdigest()


def _trained_checkpoint(path: Path, *, purpose: str) -> tuple[dict[str, Any], str]:
    raw = _file_bytes(path, purpose=purpose)
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    except Exception as exc:
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} is not a safe weights-only torch checkpoint: {path}"
        ) from exc
    if type(payload) is not dict:
        raise SharedV5DevRuntimeConfigurationError(f"{purpose} root must be a dict")
    if payload.get("trained") is not True:
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} is not explicitly marked trained"
        )
    return payload, hashlib.sha256(raw).hexdigest()


def load_shared_v5(
    checkpoint_path: Path,
    g2_report_path: Path,
    *,
    device: torch.device,
    repo_root: Path = REPO_ROOT,
) -> tuple[SharedObservableCameraRayJepaV5, str, str, dict[str, str]]:
    report, report_sha = _json_artifact(g2_report_path, purpose="G2 publication")
    _validate_publication(report, mode="g2-candidate", synthetic=False)
    core = dict(report)
    if core.pop("content_sha256", None) != _canonical_sha256(core): raise SharedV5DevRuntimeConfigurationError("G2 publication content hash changed")
    evaluated = report["evaluated_checkpoint"]; expected_checkpoint = (repo_root / evaluated["path"]).resolve()
    if expected_checkpoint != checkpoint_path.resolve(): raise SharedV5DevRuntimeConfigurationError("G2 publication names another checkpoint")
    final_binding = report["final_reports"]["g2"]; final, final_sha = _json_artifact(repo_root / final_binding["path"], purpose="G2 final report")
    final = _exact_self_hashed(
        final, schema="lewm_go2_shared_jepa_final_report_v9", purpose="G2 final report",
        keys={"schema", "gate", "passed", "metrics", "per_family_counts", "dataset_role_manifest", "evaluated_checkpoint_file_sha256",
              "captured_runtime_sources", "runner_authority_path", "runner_authority_file_sha256", "runner_execution_identity", "finalizer_authority_file_sha256",
              "finalizer_execution_identity", "runner_ledger_path", "runner_ledger_file_sha256", "raw_scene_outcome_file_sha256s", "total_instance_count", "g2_candidate_predecessor",
              "synthetic_only", "production_authority_eligible", "content_sha256"},
    )
    checkpoint_raw = _file_bytes(checkpoint_path, purpose="Shared V5 checkpoint"); checkpoint_sha = hashlib.sha256(checkpoint_raw).hexdigest()
    if (checkpoint_sha != evaluated["file_sha256"] or final_sha != final_binding["file_sha256"]
            or final["content_sha256"] != final_binding["content_sha256"] or final["dataset_role_manifest"] != report["dataset_role_manifest"]
            or final["g2_candidate_predecessor"] is not None
            or final["finalizer_authority_file_sha256"] != final_binding["finalizer_authority_file_sha256"]
            or final["gate"] != "g2" or final["passed"] is not True
            or final["evaluated_checkpoint_file_sha256"] != checkpoint_sha
            or final["synthetic_only"] is not False or final["production_authority_eligible"] is not True):
        raise SharedV5DevRuntimeConfigurationError("G2 evidence/checkpoint cross-binding changed")
    payload = torch.load(io.BytesIO(checkpoint_raw), map_location="cpu", weights_only=True)
    if type(payload) is not dict: raise SharedV5DevRuntimeConfigurationError("Shared V5 checkpoint root changed")
    config_value = payload.get("model_config")
    state = payload.get("deployment_state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError("Shared V5 checkpoint lacks model_config or deployment_state_dict")
    config = SharedObservableCameraRayJepaV5Config.from_mapping(config_value)
    model = SharedObservableCameraRayJepaV5(config)
    model.load_deployment_state_dict(state)
    actual_state_sha = tensor_state_dict_sha256(model.deployment_state_dict())
    if payload.get("deployment_state_sha256") != actual_state_sha:
        raise SharedV5DevRuntimeConfigurationError("Shared V5 deployment-state digest does not match its checkpoint")
    expected_core = pre_g2_candidate_checkpoint_core(model_config=config.to_dict(), deployment_state_sha256=actual_state_sha,
                                                      selection=payload.get("selection"), calibration=payload.get("calibration"))
    if (set(payload) != {*expected_core, "content_sha256", "deployment_state_dict"}
            or {key: payload[key] for key in expected_core} != expected_core
            or payload.get("content_sha256") != _canonical_sha256(expected_core)):
        raise SharedV5DevRuntimeConfigurationError("Shared V5 pre-G2 checkpoint fields changed")
    feature_sha = _canonical_sha256(shared_output_contract_v5(model))
    physical_head_sha = tensor_state_dict_sha256({key: value for key, value in model.deployment_state_dict().items() if key.startswith("evidence_head.")})
    model.eval().to(device)
    return model, checkpoint_sha, report_sha, {"state": actual_state_sha, "config": config.content_sha256,
                                               "feature": feature_sha, "physical_head": physical_head_sha, "g2_final_report": final_sha}


def load_target_head(
    checkpoint_path: Path,
    *,
    device: torch.device,
    shared_feature_sha256: str,
) -> tuple[SharedV5TargetObservationHeadV1, str, str, str]:
    payload, checkpoint_sha = _trained_checkpoint(
        checkpoint_path, purpose="target-head checkpoint"
    )
    if set(payload) != {"schema", "trained", "config", "config_sha256", "state_dict",
                        "state_dict_sha256", "shared_feature_contract_sha256"} or payload["schema"] != "lewm_go2_shared_v5_target_head_checkpoint_v1" or payload["shared_feature_contract_sha256"] != shared_feature_sha256:
        raise SharedV5DevRuntimeConfigurationError("target-head checkpoint fields/binding changed")
    config_value = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "target-head checkpoint lacks config or state_dict"
        )
    config = SharedV5TargetObservationHeadConfigV1(**dict(config_value))
    if set(config_value) != set(asdict(config)) or payload.get("config_sha256") != config.content_sha256:
        raise SharedV5DevRuntimeConfigurationError(
            "target-head architecture digest does not match its checkpoint"
        )
    head = SharedV5TargetObservationHeadV1(config)
    head.load_state_dict(state, strict=True)
    state_sha = tensor_state_dict_sha256(head.state_dict())
    if payload["state_dict_sha256"] != state_sha: raise SharedV5DevRuntimeConfigurationError("target-head state digest changed")
    head.eval().to(device)
    return head, checkpoint_sha, config.content_sha256, state_sha


def load_g4_head(
    checkpoint_path: Path,
    *,
    device: torch.device,
    shared_feature_sha256: str,
) -> tuple[TwoResolutionFrontierValueHeadV1, str, str, str]:
    payload, checkpoint_sha = _trained_checkpoint(
        checkpoint_path, purpose="G4-head checkpoint"
    )
    if set(payload) != {"schema", "trained", "config", "config_sha256", "state_dict",
                        "state_dict_sha256", "shared_feature_contract_sha256"} or payload["schema"] != "lewm_go2_shared_v5_g4_head_checkpoint_v1" or payload["shared_feature_contract_sha256"] != shared_feature_sha256:
        raise SharedV5DevRuntimeConfigurationError("G4-head checkpoint fields/binding changed")
    config_value = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "G4-head checkpoint lacks config or state_dict"
        )
    config = TwoResolutionFrontierValueHeadConfigV1(**dict(config_value))
    if set(config_value) != set(asdict(config)) or payload.get("config_sha256") != config.content_sha256:
        raise SharedV5DevRuntimeConfigurationError(
            "G4-head architecture digest does not match its checkpoint"
        )
    head = TwoResolutionFrontierValueHeadV1(config)
    head.load_state_dict(state, strict=True)
    state_sha = tensor_state_dict_sha256(head.state_dict())
    if payload["state_dict_sha256"] != state_sha: raise SharedV5DevRuntimeConfigurationError("G4-head state digest changed")
    head.eval().to(device)
    return head, checkpoint_sha, config.content_sha256, state_sha


def load_qualified_calibration(
    path: Path,
    *,
    purpose: str,
    role: str,
    checkpoint_sha256: str,
    config_sha256: str,
    shared_feature_sha256: str,
) -> tuple[dict[str, Any], str]:
    value, digest = _json_artifact(path, purpose=purpose)
    value = _exact_self_hashed(
        value, schema="lewm_go2_shared_v5_calibration_binding_v1", purpose=purpose,
        keys={"schema", "role", "qualified", "checkpoint_sha256", "config_sha256",
              "shared_feature_contract_sha256", "payload", "content_sha256"},
    )
    if (value["qualified"] is not True or value["role"] != role
            or value["checkpoint_sha256"] != checkpoint_sha256
            or value["config_sha256"] != config_sha256
            or value["shared_feature_contract_sha256"] != shared_feature_sha256
            or type(value["payload"]) is not dict):
        raise SharedV5DevRuntimeConfigurationError(f"{purpose} identity cross-binding changed")
    return value["payload"], digest


def _load_callable(spec: str, *, purpose: str) -> Any:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} must use module:callable syntax"
        )
    module = importlib.import_module(module_name)
    value = getattr(module, attribute, None)
    if not callable(value):
        raise SharedV5DevRuntimeConfigurationError(f"{purpose} is not callable: {spec}")
    return value


def load_development_authority(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    value, digest = _json_artifact(args.development_authority, purpose="development authority")
    value = _exact_self_hashed(value, schema="lewm_go2_shared_v5_development_scene_authority_v1", purpose="development authority",
                               keys={"schema", "authorized", "role", "scene", "platform_manifest", "observer", "content_sha256"})
    if value["authorized"] is not True or value["role"] != "development": raise SharedV5DevRuntimeConfigurationError("scene role is not authorized development")
    scene = value["scene"]; platform = value["platform_manifest"]
    if (type(scene) is not dict or set(scene) != {"path", "scene_id", "manifest_sha256", "manifest_file_sha256", "genesis_file_sha256"}
            or type(platform) is not dict or set(platform) != {"path", "file_sha256"}
            or (args.repo_root / scene["path"]).resolve() != args.scene.resolve() or (args.repo_root / platform["path"]).resolve() != args.platform_manifest.resolve()
            or file_sha256(args.platform_manifest, purpose="platform manifest") != platform["file_sha256"]):
        raise SharedV5DevRuntimeConfigurationError("development scene/platform binding changed")
    manifest, manifest_sha = _json_artifact(args.scene / "manifest.json", purpose="authorized scene manifest")
    if (manifest_sha != scene["manifest_file_sha256"] or manifest.get("split") != "development" or manifest.get("scene_id") != scene["scene_id"]
            or manifest.get("manifest_sha256") != scene["manifest_sha256"]): raise SharedV5DevRuntimeConfigurationError("authorized scene manifest is not development")
    if file_sha256(args.scene / "genesis_scene.json", purpose="authorized Genesis scene") != scene["genesis_file_sha256"]: raise SharedV5DevRuntimeConfigurationError("authorized Genesis scene changed")
    observer = value["observer"]
    if observer is None and args.observer is not None: raise SharedV5DevRuntimeConfigurationError("observer was not authorized")
    elif observer is not None and (type(observer) is not dict or set(observer) != {"spec", "source_path", "source_sha256"} or observer["spec"] != args.observer
          or file_sha256(args.repo_root / observer["source_path"], purpose="observer source") != observer["source_sha256"]): raise SharedV5DevRuntimeConfigurationError("fixed observer identity changed")
    return value, digest


def build_runtime_stack(args: argparse.Namespace) -> tuple[SharedV5DevMazeRuntime, object]:
    _ACTUAL_OPENS.clear()
    development_authority, _authority_sha = load_development_authority(args)
    device = torch.device(args.device)
    shared, shared_sha, g2_sha, shared_binding = load_shared_v5(
        args.shared_checkpoint,
        args.g2_report,
        device=device,
        repo_root=args.repo_root,
    )
    target, target_sha, target_config_sha, _target_state_sha = load_target_head(
        args.target_head_checkpoint, device=device,
        shared_feature_sha256=shared_binding["feature"],
    )
    target_calibration_value, target_calibration_sha = load_qualified_calibration(
        args.target_calibration,
        purpose="target calibration",
        role="target_confirmation", checkpoint_sha256=target_sha,
        config_sha256=target_config_sha, shared_feature_sha256=shared_binding["feature"],
    )
    if set(target_calibration_value) != {"target_confirmation"}:
        raise SharedV5DevRuntimeConfigurationError("target calibration payload changed")
    target_thresholds = target_calibration_value.get("target_confirmation")
    if not isinstance(target_thresholds, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "target calibration lacks target_confirmation thresholds"
        )
    target_calibration = TargetConfirmationCalibration.from_mapping(target_thresholds)
    physical_calibration, physical_calibration_sha = load_qualified_calibration(
        args.physical_calibration,
        purpose="physical calibration",
        role="shared_physical_admission", checkpoint_sha256=shared_sha,
        config_sha256=shared_binding["config"], shared_feature_sha256=shared_binding["feature"],
    )
    if (set(physical_calibration) != {"free_probability_threshold", "occupied_probability_threshold",
                                     "camera_transform_sha256", "physical_head_state_sha256"}
            or physical_calibration["physical_head_state_sha256"] != shared_binding["physical_head"]):
        raise SharedV5DevRuntimeConfigurationError("physical calibration payload binding changed")
    physical_calibration = {"qualified": True, **physical_calibration}

    g4_head = None
    g4_checkpoint_sha = None
    g4_calibration = None
    g4_calibration_sha = None
    g4_config_sha = None
    if (args.g4_head_checkpoint is None) != (args.g4_calibration is None):
        raise SharedV5DevRuntimeConfigurationError(
            "--g4-head-checkpoint and --g4-calibration must be supplied together"
        )
    if args.g4_head_checkpoint is not None:
        g4_head, g4_checkpoint_sha, g4_config_sha, _g4_state_sha = load_g4_head(
            args.g4_head_checkpoint,
            device=device,
            shared_feature_sha256=shared_binding["feature"],
        )
        g4_calibration, g4_calibration_sha = load_qualified_calibration(
            args.g4_calibration,
            purpose="G4 calibration",
            role="g4_frontier_value", checkpoint_sha256=g4_checkpoint_sha,
            config_sha256=g4_config_sha, shared_feature_sha256=shared_binding["feature"],
        )
        expected_candidate = _canonical_sha256({"schema": g4_calibration.get("candidate_feature_schema"),
                                                "candidate_feature_dim": g4_head.config.candidate_feature_dim})
        if set(g4_calibration) != {"candidate_feature_schema", "candidate_contract_sha256"} or g4_calibration["candidate_contract_sha256"] != expected_candidate:
            raise SharedV5DevRuntimeConfigurationError("G4 candidate calibration binding changed")

    artifacts = RuntimeArtifactBindings(
        shared_checkpoint_sha256=shared_sha,
        g2_report_sha256=g2_sha,
        physical_calibration_sha256=physical_calibration_sha,
        target_head_checkpoint_sha256=target_sha,
        target_calibration_sha256=target_calibration_sha,
        g4_head_checkpoint_sha256=g4_checkpoint_sha,
        g4_calibration_sha256=g4_calibration_sha,
    )

    # Scene loading stays on the existing Genesis path.  The factory only has
    # to assemble the backend and existing navigation objects around this pack.
    from lewm_genesis.scene_loader import load_scene_pack

    scene_pack = load_scene_pack(
        args.scene,
        platform_manifest=args.platform_manifest,
        workspace_root=args.repo_root,
    )
    scene_binding = development_authority["scene"]
    if (scene_pack.scene_id != scene_binding["scene_id"] or scene_pack.manifest_sha256 != scene_binding["manifest_sha256"]
            or scene_pack.split != "development"):
        raise SharedV5DevRuntimeConfigurationError("loaded scene is not the authorized development scene")
    from lewm.navigation.genesis_shared_v5_dev_stack import (
        build_kinematic_development_stack,
    )

    stack = build_kinematic_development_stack(
        scene_pack=scene_pack,
        repo_root=args.repo_root,
        genesis_backend=args.genesis_backend,
        device=device,
        shared_model=shared,
        target_head=target,
        physical_calibration=physical_calibration,
        g4_head=g4_head,
        g4_calibration=g4_calibration,
    )
    if not isinstance(stack, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "built-in stack must return a mapping of existing runtime components"
        )
    required = {
        "backend",
        "physical_fuser",
        "physical_memory",
        "projection",
        "planner",
    }
    missing = sorted(required - set(stack))
    if missing:
        raise SharedV5DevRuntimeConfigurationError(
            "built-in stack omitted: " + ", ".join(missing)
        )
    runtime = SharedV5DevMazeRuntime(
        model=shared,
        target_head=target,
        physical_fuser=stack["physical_fuser"],
        physical_memory=stack["physical_memory"],
        projection=stack["projection"],
        planner=stack["planner"],
        target_calibration=target_calibration,
        artifacts=artifacts,
        target_color=args.target_color,
        g4_head=g4_head,
        g4_candidate_builder=stack.get("g4_candidate_builder"),
        frontier_cap=args.frontier_cap,
    )
    runtime.development_authority = development_authority
    return runtime, stack["backend"]


def run_controller_then_observer(
    runtime: SharedV5DevMazeRuntime,
    backend: object,
    *,
    visual_ticks: int,
    observer_spec: str | None,
) -> tuple[ControllerRun, object | None]:
    """Seal the controller before importing or calling any evaluator code."""

    controller_run = runtime.run_controller(backend, visual_ticks=visual_ticks)
    observer_result = None
    if observer_spec is not None:
        # Explicit second stop at the controller/observer transition.  The
        # committed backend contract makes this idempotent.
        stop = getattr(backend, "stop", None)
        if not callable(stop):
            raise SharedV5DevRuntimeConfigurationError(
                "backend must expose idempotent stop() before observation"
            )
        stop()
        observer = _load_callable(observer_spec, purpose="post-controller observer")
        binding = runtime.development_authority["observer"]
        module_path = Path(sys.modules[observer.__module__].__file__).resolve()
        if module_path != (REPO_ROOT / binding["source_path"]).resolve():
            raise SharedV5DevRuntimeConfigurationError("loaded observer source changed")
        snapshot_fn = getattr(backend, "observer_snapshot", None)
        snapshot = snapshot_fn() if callable(snapshot_fn) else None
        observer_result = observer(controller_run=controller_run, observer_snapshot=snapshot)
    return controller_run, observer_result


def _verified_input_open_journal() -> dict[str, object]:
    previous = hashlib.sha256(b"shared-v5-dev-open-chain-v1").hexdigest()
    rows = []
    for row in _ACTUAL_OPENS:
        content = _canonical_sha256(row); previous = _canonical_sha256({"previous": previous, "content": content})
        rows.append({**row, "content_sha256": content, "chain_sha256": previous})
    return {"rows": rows, "final_chain_sha256": previous}
def _jsonable_run(run: ControllerRun, observer_result: object | None, authority: Mapping[str, Any]) -> dict[str, Any]:
    core = {
        "schema": "lewm_go2_shared_v5_development_maze_run_v1",
        "controller": asdict(run),
        "development_authority": authority,
        "verified_input_open_journal": _verified_input_open_journal(),
        "observer": observer_result,
        "observer_called_after_controller_seal": True,
    }
    return {**core, "content_sha256": _canonical_sha256(core)}
def _jsonable_fault(fault: object, authority: Mapping[str, Any]) -> dict[str, Any]:
    core = {"schema": "lewm_go2_shared_v5_development_fault_v1", "terminal_fault": asdict(fault), "development_authority": authority,
            "verified_input_open_journal": _verified_input_open_journal()}
    return {**core, "content_sha256": _canonical_sha256(core)}
def _publish_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    output = path.expanduser().resolve(); output.parent.mkdir(parents=True, exist_ok=True); raw = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    descriptor, name = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            if stream.write(raw) != len(raw): raise OSError("short development-result write")
            stream.flush(); os.fchmod(stream.fileno(), 0o444); os.fsync(stream.fileno())
        os.link(temporary, output, follow_symlinks=False)
        directory = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try: os.fsync(directory)
        finally: os.close(directory)
    finally: temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--development-authority", type=Path, required=True)
    parser.add_argument("--platform-manifest", type=Path, required=True)
    parser.add_argument("--shared-checkpoint", type=Path, required=True)
    parser.add_argument("--g2-report", type=Path, required=True)
    parser.add_argument("--physical-calibration", type=Path, required=True)
    parser.add_argument("--target-head-checkpoint", type=Path, required=True)
    parser.add_argument("--target-calibration", type=Path, required=True)
    parser.add_argument("--g4-head-checkpoint", type=Path, default=None)
    parser.add_argument("--g4-calibration", type=Path, default=None)
    parser.add_argument("--observer", default=None, help="optional module:callable scorer")
    parser.add_argument("--target-color", choices=("red", "yellow", "blue", "green"), required=True)
    parser.add_argument("--visual-ticks", type=int, default=64)
    parser.add_argument("--frontier-cap", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--genesis-backend", default="auto")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    runtime, backend = build_runtime_stack(args)
    try:
        run, observer = run_controller_then_observer(
            runtime, backend, visual_ticks=args.visual_ticks, observer_spec=args.observer)
    except BaseException as exc:
        fault = getattr(exc, "lewm_terminal_fault_record", None)
        if fault is not None:
            _publish_exclusive(args.output, _jsonable_fault(fault, runtime.development_authority))
        raise
    _publish_exclusive(args.output, _jsonable_run(run, observer, runtime.development_authority))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
