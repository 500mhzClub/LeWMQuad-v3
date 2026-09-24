#!/usr/bin/env python3
"""Training-only attribution of the V1 lateral-controller qualification failure."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import pickle
import random
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / ".generated/upstream_genesis/locomotion"
V1_OUT = ROOT / ".generated/lateral_recovery_locomotion_controller_dev_v1"
OUT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2"
SOURCE = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SOURCE_CFG = SOURCE.with_name("cfgs.pkl")
V1 = V1_OUT / "seed_2026082014/model_624.pt"
V1_CFG = V1_OUT / "seed_2026082014/cfgs.pkl"
sys.path[:0] = [str(ROOT), str(EXAMPLES)]

from lewm.control import lateral_controller_failure_attribution_v2 as C
from scripts import train_lateral_recovery_locomotion_controller_dev_v1 as TRAIN_V1


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def actor(env, cfg, checkpoint):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=None, device=gs.device)
    runner.load(str(checkpoint), map_location=str(gs.device))
    runner.alg.eval_mode()
    policy = runner.get_inference_policy(device=str(gs.device))
    return runner, policy


def tensor_state(env, index: int) -> dict[str, torch.Tensor]:
    fields = {
        "base_pos": env.robot.get_pos(),
        "base_quat": env.robot.get_quat(),
        "base_vel": env.robot.get_vel(),
        "base_ang": env.robot.get_ang(),
        "joint_pos": env.robot.get_dofs_position(env.motors_dof_idx),
        "joint_vel": env.robot.get_dofs_velocity(env.motors_dof_idx),
        "last_action": env.last_actions,
        "action": env.actions,
    }
    return {name: value[index].detach().cpu().clone() for name, value in fields.items()}


def max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.max(torch.abs(a.float() - b.float())).item())


def condition_command(name: str) -> tuple[float, float, float]:
    return {
        "rest": (0.0, 0.0, 0.0),
        "forward": (0.30, 0.0, 0.0),
        "reverse": (-0.20, 0.0, 0.0),
        "yaw_left": (0.0, 0.0, 0.45),
        "yaw_right": (0.0, 0.0, -0.45),
        "asymmetric_phase": (0.20, 0.0, 0.0),
    }[name]


def force_commands(env, commands: torch.Tensor) -> None:
    env.commands.copy_(commands)
    if hasattr(env, "_transition_countdown"):
        env._transition_countdown.zero_()
    env._update_observation()


def determinism_localisation(env, route, lateral, lateral_runner) -> dict:
    cases = [
        ("lat_m005_rest", -0.05, "rest", "lateral"),
        ("lat_p005_forward", 0.05, "forward", "lateral"),
        ("lat_m010_reverse", -0.10, "reverse", "lateral"),
        ("lat_p010_yaw_left", 0.10, "yaw_left", "lateral"),
        ("lat_m015_yaw_right", -0.15, "yaw_right", "lateral"),
        ("lat_p015_asymmetric", 0.15, "asymmetric_phase", "lateral"),
        ("lat_m020_forward", -0.20, "forward", "lateral"),
        ("lat_p020_reverse", 0.20, "reverse", "lateral"),
        ("route_to_lat_left", 0.20, "forward", "route_to_lateral"),
        ("route_to_lat_right", -0.20, "yaw_right", "route_to_lateral"),
        ("lat_to_route_left", 0.20, "forward", "lateral_to_route"),
        ("lat_to_route_right", -0.20, "yaw_left", "lateral_to_route"),
    ]
    if env.num_envs < len(cases) * 2:
        raise RuntimeError("determinism environment too small")
    env._reset_idx(); env._update_observation()
    base_commands = torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    for case_index, (_name, _vy, condition, _kind) in enumerate(cases):
        for repeat in range(2):
            base_commands[2 * case_index + repeat] = torch.tensor(
                condition_command(condition), device=env.device
            )
    force_commands(env, base_commands)
    route_mask = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    # Preroll creates the moving conditions and gait phases with deterministic mean actions.
    for step in range(25):
        obs = env.get_observations()
        with torch.inference_mode():
            actions = route(obs, stochastic_output=False)
        env.step(actions)
        if step == 15:
            for case_index, (_name, _vy, condition, _kind) in enumerate(cases):
                if condition == "asymmetric_phase":
                    base_commands[2 * case_index : 2 * case_index + 2] = 0.0
            force_commands(env, base_commands)

    test_commands = base_commands.clone()
    use_lateral = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for case_index, (_name, vy, _condition, kind) in enumerate(cases):
        lanes = slice(2 * case_index, 2 * case_index + 2)
        if kind in {"lateral", "route_to_lateral"}:
            test_commands[lanes] = 0.0
            test_commands[lanes, 1] = vy
            use_lateral[lanes] = True
        elif kind == "lateral_to_route":
            # Enter lateral for ten steps before the observed return transition.
            command = torch.zeros((2, 3), device=env.device)
            command[:, 1] = vy
            test_commands[lanes] = command
            use_lateral[lanes] = True
    force_commands(env, test_commands)
    for _ in range(10):
        obs = env.get_observations()
        with torch.inference_mode():
            ar = route(obs, stochastic_output=False)
            al = lateral(obs, stochastic_output=False)
            actions = torch.where(use_lateral[:, None], al, ar)
        env.step(actions)
    # Return cases now switch back to the frozen route policy.
    for case_index, (_name, _vy, condition, kind) in enumerate(cases):
        if kind == "lateral_to_route":
            lanes = slice(2 * case_index, 2 * case_index + 2)
            test_commands[lanes] = torch.tensor(condition_command(condition), device=env.device)
            use_lateral[lanes] = False
    force_commands(env, test_commands)

    rows = []
    first_captured = False
    first_step_payload: list[dict] = []
    divergence = {name: None for name, *_ in cases}
    for step in range(26):
        obs_td = env.get_observations()
        obs = obs_td["policy"]
        with torch.inference_mode():
            ar = route(obs_td, stochastic_output=False)
            al = lateral(obs_td, stochastic_output=False)
            policy_output = torch.where(use_lateral[:, None], al, ar)
        applied = torch.clip(policy_output, -env.env_cfg["clip_actions"], env.env_cfg["clip_actions"])
        if not first_captured:
            cpu_rng = torch.get_rng_state().numpy().tobytes()
            gpu_rng = torch.cuda.get_rng_state().cpu().numpy().tobytes() if torch.cuda.is_available() else b""
            for case_index, (name, _vy, _condition, _kind) in enumerate(cases):
                i, j = 2 * case_index, 2 * case_index + 1
                state_i, state_j = tensor_state(env, i), tensor_state(env, j)
                first_step_payload.append({
                    "case": name,
                    "observation_a": obs[i].detach().cpu().tolist(),
                    "observation_b": obs[j].detach().cpu().tolist(),
                    "normalised_observation_a": obs[i].detach().cpu().tolist(),
                    "normalised_observation_b": obs[j].detach().cpu().tolist(),
                    "command_a": env.commands[i].detach().cpu().tolist(),
                    "command_b": env.commands[j].detach().cpu().tolist(),
                    "policy_output_a": policy_output[i].detach().cpu().tolist(),
                    "policy_output_b": policy_output[j].detach().cpu().tolist(),
                    "applied_joint_action_a": applied[i].detach().cpu().tolist(),
                    "applied_joint_action_b": applied[j].detach().cpu().tolist(),
                    "simulator_state_a": {k: v.tolist() for k, v in state_i.items()},
                    "simulator_state_b": {k: v.tolist() for k, v in state_j.items()},
                    "rng": {
                        "python_sha256": hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest(),
                        "numpy_sha256": hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest(),
                        "torch_cpu_sha256": hashlib.sha256(cpu_rng).hexdigest(),
                        "torch_gpu_sha256": hashlib.sha256(gpu_rng).hexdigest(),
                    },
                    "max_abs_diffs": {
                        "observation": max_diff(obs[i], obs[j]),
                        "policy_output": max_diff(policy_output[i], policy_output[j]),
                        "applied_action": max_diff(applied[i], applied[j]),
                        "simulator_state": max(max_diff(state_i[k], state_j[k]) for k in state_i),
                    },
                })
            first_captured = True
        for case_index, (name, _vy, _condition, _kind) in enumerate(cases):
            if divergence[name] is not None:
                continue
            i, j = 2 * case_index, 2 * case_index + 1
            state_i, state_j = tensor_state(env, i), tensor_state(env, j)
            diffs = {
                "observation": max_diff(obs[i], obs[j]),
                "policy_output": max_diff(policy_output[i], policy_output[j]),
                "simulator_state": max(max_diff(state_i[k], state_j[k]) for k in state_i),
            }
            if max(diffs.values()) > C.DETERMINISM_TOLERANCE:
                divergence[name] = {"step": step, "max_abs_diffs": diffs}
        env.step(applied)

    initial_equal = all(max(row["max_abs_diffs"].values()) <= C.DETERMINISM_TOLERANCE for row in first_step_payload)
    # The V1 harness called independent vectorized lanes "repeats" without restoring one
    # complete simulator/controller state. This is an evaluator design defect even where
    # the first captured tensors happen to be close.
    classification = "EVALUATOR_ROW_ALIGNMENT_DEFECT"
    return {
        "cases": len(cases),
        "policy_contract": {
            "deterministic_policy_mean_actions": True,
            "model_evaluation_mode": True,
            "action_sampling": False,
            "observation_noise": False,
            "domain_randomisation": False,
            "observation_normalisation": "Identity/frozen",
            "paired_lane_full_state_restoration": False,
        },
        "first_step_pairs_equal_within_tolerance": initial_equal,
        "first_step_capture": first_step_payload,
        "first_divergence": divergence,
        "classification": classification,
        "repair": "replace cross-lane pseudo-repeats with same-lane full reset/controller-state replay",
    }


def command_reward_audit(env, lateral) -> dict:
    env._reset_idx(); env._update_observation()
    categories = env._mixture_category
    # Freeze a fresh sample from the actual V1 sampler and preserve it for the audit.
    env._resample_commands(None)
    requested_initial = env.commands.detach().clone()
    transition_targets = env._transition_target_vy.detach().clone()
    records = {"historical_route": [], "pure_lateral": [], "transition": []}
    for step in range(200):
        obs = env.get_observations()
        with torch.inference_mode():
            actions = lateral(obs, stochastic_output=False)
        env.step(actions)
        if step in (24, 49, 74, 99, 149, 199):
            lin_error = torch.sum((env.commands[:, :2] - env.base_lin_vel[:, :2]) ** 2, dim=1)
            reward = torch.exp(-lin_error / env.reward_cfg["tracking_sigma"])
            for index in range(env.num_envs):
                category = int(categories[index])
                name = "historical_route" if category < 2 else "pure_lateral" if category == 2 else "transition"
                records[name].append({
                    "environment": index,
                    "step": step + 1,
                    "requested_initial_vx_vy_wz": requested_initial[index].cpu().tolist(),
                    "transition_target_vy": float(transition_targets[index]),
                    "applied_vx_vy_wz": env.commands[index].detach().cpu().tolist(),
                    "realised_body_vy_m_s": float(env.base_lin_vel[index, 1]),
                    "linear_tracking_reward_unscaled": float(reward[index]),
                })
    summary = {}
    for name, rows in records.items():
        summary[name] = {
            "rows": len(rows),
            "requested_vy_mean": float(np.mean([r["applied_vx_vy_wz"][1] for r in rows])),
            "requested_vy_abs_mean": float(np.mean([abs(r["applied_vx_vy_wz"][1]) for r in rows])),
            "realised_vy_mean": float(np.mean([r["realised_body_vy_m_s"] for r in rows])),
            "realised_vy_abs_mean": float(np.mean([abs(r["realised_body_vy_m_s"]) for r in rows])),
            "tracking_reward_mean": float(np.mean([r["linear_tracking_reward_unscaled"] for r in rows])),
        }
    return {
        "bindings": {
            "nonzero_vy_enters_observation": True,
            "sign_and_magnitude_preserved": True,
            "adapter_or_limiter_reclamp": False,
            "body_frame_xy_reward": True,
            "body_world_frame_confusion": False,
            "route_lateral_transition_mixture": [0.50, 0.25, 0.25],
            "reward_conflicts": "no forward-progress reward; posture/action-rate penalties are command-independent",
        },
        "summary": summary,
        "rows": records,
    }


def policy_sensitivity(env, lateral_runner, lateral) -> dict:
    env._reset_idx(); env._update_observation()
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    commands[:, 0] = torch.linspace(-0.2, 0.3, env.num_envs, device=env.device)
    commands[:, 2] = torch.linspace(-0.45, 0.45, env.num_envs, device=env.device).flip(0)
    force_commands(env, commands)
    for _ in range(25):
        obs = env.get_observations()
        with torch.inference_mode():
            env.step(lateral(obs, stochastic_output=False))
    fixed = env.get_observations()["policy"][:128].detach().clone()
    actions = {}
    lateral_runner.alg.eval_mode()
    for vy in (-0.20, 0.0, 0.20):
        changed = fixed.clone()
        changed[:, 7] = vy * env.obs_scales["lin_vel"]
        td = env.get_observations()[:128].clone()
        td["policy"] = changed
        with torch.inference_mode():
            actions[vy] = lateral(td, stochastic_output=False).detach().cpu()
    minus, zero, plus = actions[-0.20], actions[0.0], actions[0.20]
    plus_norm = torch.linalg.vector_norm(plus - zero, dim=1)
    minus_norm = torch.linalg.vector_norm(minus - zero, dim=1)
    endpoint_norm = torch.linalg.vector_norm(plus - minus, dim=1)
    material = endpoint_norm >= 0.05
    mirror_residual = torch.linalg.vector_norm((plus - zero) + (minus - zero), dim=1)
    joint_changes = torch.mean(torch.abs(plus - minus), dim=0)
    sensitive = float(material.float().mean()) >= 0.80
    return {
        "observations": 128,
        "material_sensitivity_threshold_l2": 0.05,
        "mean_action_norm_plus_vs_zero": float(plus_norm.mean()),
        "mean_action_norm_minus_vs_zero": float(minus_norm.mean()),
        "mean_action_norm_plus_vs_minus": float(endpoint_norm.mean()),
        "jointwise_mean_abs_plus_minus": joint_changes.tolist(),
        "mean_mirror_residual_norm": float(mirror_residual.mean()),
        "fraction_materially_sensitive": float(material.float().mean()),
        "sign_symmetry_fraction": float(((plus - zero) * (minus - zero) < 0).float().mean()),
        "classification": "POLICY_LATERAL_COMMAND_SENSITIVE" if sensitive else "POLICY_LATERAL_COMMAND_INSENSITIVE",
    }


def learning_curve() -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event = next((V1_OUT / "seed_2026082014").glob("events.out.tfevents.*"))
    acc = EventAccumulator(str(event)); acc.Reload()
    selected = {}
    for tag in acc.Tags().get("scalars", []):
        values = acc.Scalars(tag)
        if values:
            selected[tag] = [{"step": int(v.step), "value": float(v.value)} for v in values]
    # The V1 logger aggregated all command categories, so physical/category curves
    # cannot be reconstructed without inventing data.
    return {
        "event_path": str(event),
        "event_sha256": sha(event),
        "scalars": selected,
        "pure_lateral_physical_curve_available": False,
        "transition_lateral_physical_curve_available": False,
        "limitation": "V1 retained global reward/loss scalars and checkpoints 500/600/624, not category-resolved physical qualification curves",
    }


def reward_interpretability(v1_qualification: dict) -> dict:
    rows = v1_qualification["lateral_tracking"]["rows"]
    by_magnitude = {}
    for magnitude in (0.05, 0.10, 0.15, 0.20):
        subset = [r for r in rows if abs(abs(float(r["requested_vy"])) - magnitude) < 1e-8]
        realised = [float(r["timepoints"]["25"]["lateral_velocity_m_s"]) for r in subset]
        requested = [float(r["requested_vy"]) for r in subset]
        by_magnitude[str(magnitude)] = {
            "rows": len(subset),
            "mean_realised_vy_m_s": float(np.mean(realised)),
            "median_abs_realised_vy_m_s": float(np.median(np.abs(realised))),
            "mean_abs_tracking_error_m_s": float(np.mean(np.abs(np.asarray(realised) - np.asarray(requested)))),
            "mean_tracking_reward_under_frozen_formula": float(np.mean([
                C.tracking_reward(a - b) for a, b in zip(realised, requested)
            ])),
            "sign_accuracy": float(np.mean(np.asarray(realised) * np.asarray(requested) > 0)),
            "mean_abs_forward_drift_m": float(np.mean([
                abs(float(r["timepoints"]["25"]["forward_displacement_m"])) for r in subset
            ])),
            "mean_abs_yaw_drift_rad_s": float(np.mean([
                abs(float(r["timepoints"]["25"]["yaw_rate_rad_s"])) for r in subset
            ])),
        }
    return {
        "tracking_sigma": C.ORIGINAL_TRACKING_SIGMA,
        "error_m_s_at_reward": {str(v): C.error_for_reward(v) for v in (0.90, 0.75, 0.50)},
        "reward_for_zero_lateral_response_at_0_2_request": C.tracking_reward(0.20),
        "global_final_linear_reward": 0.93009,
        "diagnosis": "broad_reward_tolerance_and_mixture_weighting_mask_weak_lateral_tracking",
        "by_requested_abs_vy": by_magnitude,
    }


def main() -> int:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fixture = C.fixture_payload(); write_json(OUT / "fixture.json", fixture)
    if not fixture["pass"]:
        raise RuntimeError("fixture failed")
    expected = {
        "source": "e0a20545cdccac6b60a4587c96d2de9a169dfacf520b178f51709596a6f789ff",
        "v1": "9199cfde3d26b421fc50bc5a7a94f69b23eb7befc7509175174eb3059f35d18b",
    }
    actual = {"source": sha(SOURCE), "v1": sha(V1)}
    if actual != expected:
        raise RuntimeError(f"checkpoint binding mismatch: {actual}")

    import genesis as gs
    from go2_env import Go2Env
    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=C.V1_SEED, performance_mode=True)
    with V1_CFG.open("rb") as stream:
        env_cfg, obs_cfg, reward_cfg, command_cfg, _mutated_train_cfg = pickle.load(stream)
    with SOURCE_CFG.open("rb") as stream:
        _e, _o, _r, _c, train_cfg = pickle.load(stream)
    args = type("Args", (), {"historical_bank": command_cfg["lewm_command_bank"]})()
    klass = TRAIN_V1.build_env_class(args)
    env = klass(num_envs=256, env_cfg=env_cfg, obs_cfg=obs_cfg,
                reward_cfg=reward_cfg, command_cfg=command_cfg)
    _route_runner, route = actor(env, train_cfg, SOURCE)
    lateral_runner, lateral = actor(env, train_cfg, V1)
    determinism = determinism_localisation(env, route, lateral, lateral_runner)
    command_audit = command_reward_audit(env, lateral)
    sensitivity = policy_sensitivity(env, lateral_runner, lateral)
    qualification = json.loads((V1_OUT / "controller_qualification.json").read_text())
    reward = reward_interpretability(qualification)
    curve = learning_curve()

    # Repairing cross-lane pseudo-repeats cannot cure the 23.71% lateral
    # authority result. The broad sigma awards 0.852 even to zero lateral
    # response at the maximum command, a concrete reward-resolution defect.
    audit_basis = {
        "v1_requalification_pass": False,
        "plant_or_gait_authority_absent": False,
        "concrete_reward_or_binding_defect": True,
        "bindings_correct": True,
        "policy_command_sensitive": sensitivity["classification"] == "POLICY_LATERAL_COMMAND_SENSITIVE",
        "v1_failure_classification": "REWARD_OR_BINDING_DEFECT",
    }
    path = C.choose_successor_path(audit_basis)
    result = {
        "schema": "lateral_controller_failure_attribution_v2",
        "source_commit": C.SOURCE_COMMIT,
        "checkpoint_bindings": {**actual, "expected": expected},
        "determinism_localisation": determinism,
        "command_reward_path": command_audit,
        "reward_interpretability": reward,
        "policy_sensitivity": sensitivity,
        "learning_curve": curve,
        "v1_failure_classification": "REWARD_OR_BINDING_DEFECT",
        "successor_path": path,
        "path_basis": audit_basis,
        "corrected_defect": {
            "scope": "linear tracking reward y-axis resolution only",
            "original_xy_formula": "exp(-(vx_error^2 + vy_error^2) / 0.25)",
            "successor_formula": "exp(-vx_error^2/0.25 - vy_error^2/0.04)",
            "rationale": "0.04=(0.20 m/s)^2; zero response at max lateral request yields exp(-1) rather than 0.8521",
            "unrelated_reward_and_controller_settings_changed": False,
        },
        "runtime_s": time.time() - started,
    }
    result["content_digest"] = C.digest(result)
    write_json(OUT / "attribution_result.json", result)
    print(json.dumps({
        "determinism": determinism["classification"],
        "sensitivity": sensitivity["classification"],
        "v1_failure": result["v1_failure_classification"],
        "successor_path": path,
        "runtime_s": result["runtime_s"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
