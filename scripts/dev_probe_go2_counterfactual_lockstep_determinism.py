#!/usr/bin/env python3
"""Non-citable common-prefix determinism probe for Genesis batched Go2 lanes.

This probe runs exactly one shared HOLD block in ten lanes, with no branch and
no RGB persistence.  It exists only to compare explicitly selected Genesis
parallelization levels after the V2 source-integration smoke lost bitwise lane
equality.  Results remain development-tier and are never training inputs.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
for _package_root in (ROOT, ROOT / "lewm_genesis", ROOT / "lewm_worlds"):
    if str(_package_root) not in sys.path:
        sys.path.insert(0, str(_package_root))

from lewm.benchmarks import go2_world_model_counterfactual_pilot_v1 as pilot  # noqa: E402
from scripts import collect_go2_world_model_counterfactual_pilot_v1 as collector  # noqa: E402


PROBE_SCHEMA = "lewm_go2_counterfactual_lockstep_determinism_probe_v1"


def _configure_environment(plan: Mapping[str, Any], *, para_level: int) -> dict[str, str]:
    expected = {
        str(key): str(value)
        for key, value in plan["execution_contract"]["environment"].items()
    }
    for key in {*collector._SANITIZED_SELECTOR_KEYS, "GS_PARA_LEVEL"}:  # noqa: SLF001
        os.environ.pop(key, None)
    os.environ.update(expected)
    os.environ["GS_PARA_LEVEL"] = str(para_level)
    return {**expected, "GS_PARA_LEVEL": str(para_level)}


def _source_bindings() -> dict[str, dict[str, Any]]:
    return {
        "contract": pilot.file_binding(Path(pilot.__file__)),
        "collector": pilot.file_binding(Path(collector.__file__)),
        "probe": pilot.file_binding(Path(__file__)),
        "rollout": pilot.file_binding(ROOT / "lewm_genesis/lewm_genesis/rollout.py"),
        "scene_builder": pilot.file_binding(
            ROOT / "lewm_genesis/lewm_genesis/scene_builder.py"
        ),
    }


def run_probe(
    *, plan_path: Path, output_root: Path, para_level: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan_binding_before = pilot.file_binding(plan_path)
    plan = copy.deepcopy(pilot.validate_plan(json.loads(plan_path.read_text())))
    if pilot.file_binding(plan_path) != plan_binding_before:
        raise pilot.PilotContractError("probe plan changed while being loaded")
    if plan["purpose"] != "source_integration_smoke" or len(plan["states"]) != 1:
        raise pilot.PilotContractError("probe requires the one-state source smoke plan")
    if para_level not in {0, 2}:
        raise pilot.PilotContractError("probe para level must be 0 or 2")

    selected_root = pilot.fresh_development_output_root(
        output_root, development_root=ROOT / ".generated" / "dev"
    )
    environment = _configure_environment(plan, para_level=para_level)
    sources = _source_bindings()
    plan["output_root"] = str(selected_root)
    plan["attempt_id"] = f"dev-lockstep-para-{para_level}"
    plan["states"][0]["scene_generation"]["scene_generator_binding"] = sources[
        "collector"
    ]

    initial_audits: list[dict[str, Any]] | None = None
    policy_step_audits: list[dict[str, Any]] = []
    scene_materialization: dict[str, Any] | None = None
    failure: dict[str, Any] | None = None
    sim_time_ns = 0
    executed_block_sha256: str | None = None
    observed_para_level: int | None = None
    try:
        pilot.require_plan_bindings(plan)
        scene_materialization = collector._materialize_smoke_scene(plan)  # noqa: SLF001
        runtime = collector._runtime_imports()  # noqa: SLF001
        platform = runtime["load_platform_manifest"](
            plan["runtime_bindings"]["platform_manifest"]["path"]
        )
        registry = runtime["PrimitiveRegistry"].from_yaml(
            plan["runtime_bindings"]["primitive_registry"]["path"]
        )
        action_blocks = collector._load_action_blocks(  # noqa: SLF001
            plan=plan,
            registry=registry,
            expand=runtime["expand_primitive_to_block"],
        )
        state = plan["states"][0]
        scene_dir = Path(state["scene_manifest_binding"]["path"]).parent
        pack = runtime["load_scene_pack"](
            scene_dir,
            platform_manifest=platform,
            workspace_root=ROOT,
        )
        build = runtime["build_scene_from_pack"](
            pack,
            n_envs=pilot.CALIBRATION_LANES_PER_STATE,
            backend=str(plan["execution_contract"]["backend"]),
            show_viewer=False,
            render_robot=False,
            apply_textures=False,
            batched_camera=False,
        )
        observed_para_level = int(build.scene._para_level)  # noqa: SLF001
        if observed_para_level != para_level:
            raise pilot.PilotContractError(
                "Genesis scene parallelization level disagrees with probe"
            )
        runner = collector._build_rollout_runner(  # noqa: SLF001
            plan=plan,
            runtime=runtime,
            platform=platform,
            build=build,
            registry=registry,
        )
        collector._initialize_exact_clones(runner)  # noqa: SLF001
        state_ids = [str(state["state_id"])]
        roles = [str(state["role"])]
        initial_audits = pilot.audit_prebranch_synchronization(
            collector._capture_components(runner),  # noqa: SLF001
            state_ids=state_ids,
            roles=roles,
        )
        if any(not row["passed"] for row in initial_audits):
            raise pilot.PilotDiagnosticError(
                "probe initial lanes are not exactly equal",
                diagnostics={
                    "phase": "initial_clone",
                    "sim_time_ns": 0,
                    "synchronization_audits": initial_audits,
                },
            )

        hold = np.asarray(action_blocks[6], dtype=np.float32)
        requested = np.tile(
            hold[None, :, :], (pilot.CALIBRATION_LANES_PER_STATE, 1, 1)
        )

        def after_policy_step(command_tick_index: int, policy_step_index: int) -> None:
            audits = pilot.audit_prebranch_synchronization(
                collector._capture_components(runner),  # noqa: SLF001
                state_ids=state_ids,
                roles=roles,
            )
            policy_step_audits.append(
                {
                    "command_tick_index": int(command_tick_index),
                    "policy_step_index": int(policy_step_index),
                    "block_policy_step_index": (
                        int(command_tick_index)
                        * int(runner.policy_steps_per_command_tick)
                        + int(policy_step_index)
                    ),
                    "sim_time_ns": int(runner._sim_time_ns),  # noqa: SLF001
                    "synchronization_audits": audits,
                }
            )
            failed = [row["state_id"] for row in audits if not row["passed"]]
            if failed:
                raise pilot.PilotDiagnosticError(
                    "probe first common-prefix divergence for states: "
                    + ", ".join(failed),
                    diagnostics={
                        "phase": "common_history_policy_step",
                        **policy_step_audits[-1],
                    },
                )

        trajectory = runner.execute_requested_block(
            requested, after_policy_step=after_policy_step
        )
        sim_time_ns = int(runner._sim_time_ns)  # noqa: SLF001
        executed_block_sha256 = hashlib.sha256(
            np.asarray(trajectory.executed, dtype="<f4").tobytes(order="C")
        ).hexdigest()
    except Exception as exc:  # development result still fails closed
        failure = collector._failure_receipt(exc)  # noqa: SLF001
        runner_value = locals().get("runner")
        if runner_value is not None:
            sim_time_ns = int(runner_value._sim_time_ns)  # noqa: SLF001

    result = {
        "schema": PROBE_SCHEMA,
        "status": "EXACT_COMMON_PREFIX_PASS" if failure is None else "DIVERGED",
        "citable_as_scientific_evidence": False,
        "eligible_for_training": False,
        "authorizes_branch_generation": False,
        "para_level": para_level,
        "observed_para_level": observed_para_level,
        "environment": environment,
        "plan_witness_binding": plan_binding_before,
        "source_bindings": sources,
        "scene_materialization": scene_materialization,
        "lane_count": pilot.CALIBRATION_LANES_PER_STATE,
        "command": "hold",
        "requested_policy_steps_per_lane": 25,
        "observed_policy_steps_per_lane": len(policy_step_audits),
        "observed_sim_time_ns": sim_time_ns,
        "initial_synchronization_audits": initial_audits,
        "policy_step_audits": policy_step_audits,
        "executed_block_sha256": executed_block_sha256,
        "failure": failure,
    }
    result_binding = pilot.write_json_exclusive(
        selected_root / "probe_result.json", result
    )
    return result, result_binding


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--para-level", required=True, type=int, choices=(0, 2))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result, binding = run_probe(
        plan_path=args.plan,
        output_root=args.output_root,
        para_level=args.para_level,
    )
    print(json.dumps({"status": result["status"], "result": binding}, sort_keys=True))
    return 0 if result["status"] == "EXACT_COMMON_PREFIX_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
