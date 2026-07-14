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
from pathlib import Path
import sys
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lewm.models.shared_observable_camera_ray_jepa_v5 import (  # noqa: E402
    SharedObservableCameraRayJepaV5,
    SharedObservableCameraRayJepaV5Config,
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


def _file_bytes(path: Path, *, purpose: str) -> bytes:
    expanded = path.expanduser()
    if expanded.is_symlink():
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} must not be a symlink: {expanded}"
        )
    resolved = expanded.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"missing {purpose}: {resolved}")
    return resolved.read_bytes()


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
) -> tuple[SharedObservableCameraRayJepaV5, str, str]:
    """Load a post-G2 deployment state; pre-G2 candidates fail closed."""

    report, report_sha = _json_artifact(g2_report_path, purpose="G2 report")
    if report.get("status") != "PASS" or report.get("post_g2_qualified") is not True:
        raise SharedV5DevRuntimeConfigurationError(
            "G2 report must explicitly say status=PASS and post_g2_qualified=true"
        )
    payload, checkpoint_sha = _trained_checkpoint(
        checkpoint_path, purpose="Shared V5 checkpoint"
    )
    if payload.get("post_g2_qualified") is not True:
        raise SharedV5DevRuntimeConfigurationError(
            "Shared V5 checkpoint is trained but not post-G2 qualified"
        )
    config_value = payload.get("model_config")
    state = payload.get("deployment_state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "Shared V5 checkpoint lacks model_config or deployment_state_dict"
        )
    config = SharedObservableCameraRayJepaV5Config.from_mapping(config_value)
    model = SharedObservableCameraRayJepaV5(config)
    model.load_deployment_state_dict(state)
    actual_state_sha = tensor_state_dict_sha256(model.deployment_state_dict())
    expected_state_sha = payload.get("deployment_state_sha256")
    if expected_state_sha != actual_state_sha:
        raise SharedV5DevRuntimeConfigurationError(
            "Shared V5 deployment-state digest does not match its checkpoint"
        )
    model.eval().to(device)
    return model, checkpoint_sha, report_sha


def load_target_head(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[SharedV5TargetObservationHeadV1, str]:
    payload, checkpoint_sha = _trained_checkpoint(
        checkpoint_path, purpose="target-head checkpoint"
    )
    config_value = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "target-head checkpoint lacks config or state_dict"
        )
    config = SharedV5TargetObservationHeadConfigV1(**dict(config_value))
    if payload.get("config_sha256") != config.content_sha256:
        raise SharedV5DevRuntimeConfigurationError(
            "target-head architecture digest does not match its checkpoint"
        )
    head = SharedV5TargetObservationHeadV1(config)
    head.load_state_dict(state, strict=True)
    head.eval().to(device)
    return head, checkpoint_sha


def load_g4_head(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[TwoResolutionFrontierValueHeadV1, str]:
    payload, checkpoint_sha = _trained_checkpoint(
        checkpoint_path, purpose="G4-head checkpoint"
    )
    config_value = payload.get("config")
    state = payload.get("state_dict")
    if not isinstance(config_value, Mapping) or not isinstance(state, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "G4-head checkpoint lacks config or state_dict"
        )
    config = TwoResolutionFrontierValueHeadConfigV1(**dict(config_value))
    if payload.get("config_sha256") != config.content_sha256:
        raise SharedV5DevRuntimeConfigurationError(
            "G4-head architecture digest does not match its checkpoint"
        )
    head = TwoResolutionFrontierValueHeadV1(config)
    head.load_state_dict(state, strict=True)
    head.eval().to(device)
    return head, checkpoint_sha


def load_qualified_calibration(
    path: Path,
    *,
    purpose: str,
) -> tuple[dict[str, Any], str]:
    value, digest = _json_artifact(path, purpose=purpose)
    if value.get("qualified") is not True:
        raise SharedV5DevRuntimeConfigurationError(
            f"{purpose} must be explicitly marked qualified"
        )
    return value, digest


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


def build_runtime_stack(args: argparse.Namespace) -> tuple[SharedV5DevMazeRuntime, object]:
    device = torch.device(args.device)
    shared, shared_sha, g2_sha = load_shared_v5(
        args.shared_checkpoint,
        args.g2_report,
        device=device,
    )
    target, target_sha = load_target_head(args.target_head_checkpoint, device=device)
    target_calibration_value, target_calibration_sha = load_qualified_calibration(
        args.target_calibration,
        purpose="target calibration",
    )
    target_thresholds = target_calibration_value.get("target_confirmation")
    if not isinstance(target_thresholds, Mapping):
        raise SharedV5DevRuntimeConfigurationError(
            "target calibration lacks target_confirmation thresholds"
        )
    target_calibration = TargetConfirmationCalibration.from_mapping(target_thresholds)
    physical_calibration, physical_calibration_sha = load_qualified_calibration(
        args.physical_calibration,
        purpose="physical calibration",
    )

    g4_head = None
    g4_checkpoint_sha = None
    g4_calibration = None
    g4_calibration_sha = None
    if (args.g4_head_checkpoint is None) != (args.g4_calibration is None):
        raise SharedV5DevRuntimeConfigurationError(
            "--g4-head-checkpoint and --g4-calibration must be supplied together"
        )
    if args.g4_head_checkpoint is not None:
        g4_head, g4_checkpoint_sha = load_g4_head(
            args.g4_head_checkpoint,
            device=device,
        )
        g4_calibration, g4_calibration_sha = load_qualified_calibration(
            args.g4_calibration,
            purpose="G4 calibration",
        )

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
        snapshot_fn = getattr(backend, "observer_snapshot", None)
        snapshot = snapshot_fn() if callable(snapshot_fn) else None
        observer_result = observer(controller_run=controller_run, observer_snapshot=snapshot)
    return controller_run, observer_result


def _jsonable_run(run: ControllerRun, observer_result: object | None) -> dict[str, Any]:
    return {
        "schema": "lewm_go2_shared_v5_development_maze_run_v1",
        "controller": asdict(run),
        "observer": observer_result,
        "observer_called_after_controller_seal": True,
        "development_only": True,
        "heldout": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--scene", type=Path, required=True)
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
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> int:
    args = _parser().parse_args()
    runtime, backend = build_runtime_stack(args)
    run, observer = run_controller_then_observer(
        runtime,
        backend,
        visual_ticks=args.visual_ticks,
        observer_spec=args.observer,
    )
    encoded = json.dumps(_jsonable_run(run, observer), sort_keys=True, indent=2) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        output = args.output.expanduser().resolve()
        if output.exists():
            raise FileExistsError(f"refusing to overwrite development result: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
