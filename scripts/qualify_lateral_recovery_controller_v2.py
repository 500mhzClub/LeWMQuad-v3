#!/usr/bin/env python3
"""Same-lane reset/replay qualification for V1 or the bounded V2 successor."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / ".generated/upstream_genesis/locomotion"
SOURCE = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SOURCE_CFG = SOURCE.with_name("cfgs.pkl")
V1_CFG = ROOT / ".generated/lateral_recovery_locomotion_controller_dev_v1/seed_2026082014/cfgs.pkl"
DEFAULT_OUT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2"
sys.path[:0] = [str(ROOT), str(EXAMPLES)]

from lewm.control import lateral_controller_failure_attribution_v2 as C
from scripts.qualify_lateral_recovery_locomotion_controller_dev_v1 import Monitor, actor


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def force_commands(env, commands):
    env.commands.copy_(commands)
    if hasattr(env, "_transition_countdown"):
        env._transition_countdown.zero_()
    env._update_observation()


def reset(env):
    env._reset_idx(); env._update_observation()
    return Monitor(env)


def step(env, policy, monitor):
    obs = env.get_observations()
    with torch.inference_mode():
        actions = policy(obs, stochastic_output=False)
    _obs, _reward, dones, _extras = env.step(actions)
    monitor.update(actions, dones)
    return actions


def quantized_equal(a, b, tolerance=C.DETERMINISM_TOLERANCE):
    return abs(float(a) - float(b)) <= tolerance


def route_once(env, policy):
    bank = [
        ("hold", 0.0, 0.0), ("forward_slow", 0.2, 0.0),
        ("forward_medium", 0.25, 0.0), ("forward_fast", 0.3, 0.0),
        ("backward", -0.2, 0.0), ("yaw_left", 0.0, 0.45),
        ("yaw_right", 0.0, -0.45), ("arc_left", 0.2, 0.45),
        ("arc_right", 0.2, -0.45),
    ]
    monitor = reset(env); n = len(bank)
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    for i, (_name, vx, wz) in enumerate(bank): commands[i] = torch.tensor([vx, 0.0, wz], device=env.device)
    force_commands(env, commands)
    velocity = []; yaw = []
    for tick in range(100):
        step(env, policy, monitor)
        if tick >= 50:
            velocity.append(env.base_lin_vel[:n].detach().cpu().numpy())
            yaw.append(env.base_ang_vel[:n, 2].detach().cpu().numpy())
    mean_vel = np.mean(velocity, axis=0); mean_yaw = np.mean(yaw, axis=0)
    rows = []
    for i, (name, vx, wz) in enumerate(bank):
        rows.append({
            "command": name, "requested": [vx, 0.0, wz],
            "mean_body_velocity": mean_vel[i].tolist(),
            "vx_abs_error": abs(float(mean_vel[i, 0]) - vx),
            "yaw_abs_error": abs(float(mean_yaw[i]) - wz),
            "unintended_vy_abs": abs(float(mean_vel[i, 1])),
            "contact": bool(monitor.contact[i]), "fall": bool(monitor.fall[i]),
            "joint_limit_violation": bool(monitor.joint_violation[i]),
            "torque_limit_violation": bool(monitor.torque_violation[i]),
            "energy": float(monitor.energy[i]),
            "action_smoothness": float(monitor.smoothness[i] / 99),
            "peak_tilt": float(monitor.peak_tilt[i]),
        })
    return rows


CONDITIONS = ("rest", "forward", "reverse", "yaw_left", "yaw_right", "asymmetric_phase")


def condition_commands(env, conditions):
    mapping = {"rest": (0, 0), "forward": (.3, 0), "reverse": (-.2, 0),
               "yaw_left": (0, .45), "yaw_right": (0, -.45), "asymmetric_phase": (.2, 0)}
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    for i, name in enumerate(conditions): commands[i, 0], commands[i, 2] = mapping[name]
    return commands


def preroll(env, route, conditions, monitor):
    commands = condition_commands(env, conditions); force_commands(env, commands)
    for tick in range(25):
        if tick == 15:
            for i, name in enumerate(conditions):
                if name == "asymmetric_phase": commands[i] = 0.0
            force_commands(env, commands)
        step(env, route, monitor)


def lateral_once(env, route, lateral):
    cases = [(vy, condition) for vy in (-.05, .05, -.1, .1, -.15, .15, -.2, .2) for condition in CONDITIONS]
    n = len(cases); monitor = reset(env); preroll(env, route, [c[1] for c in cases], monitor)
    start_pos = env.base_pos[:n].detach().clone(); start_yaw = torch.deg2rad(env.base_euler[:n, 2].detach().clone())
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    for i, (vy, _condition) in enumerate(cases): commands[i, 1] = vy
    force_commands(env, commands); snapshots = {}
    for tick in range(1, 51):
        step(env, lateral, monitor)
        if tick in (5, 10, 25, 50):
            delta = env.base_pos[:n, :2] - start_pos[:, :2]
            dy = -torch.sin(start_yaw) * delta[:, 0] + torch.cos(start_yaw) * delta[:, 1]
            dx = torch.cos(start_yaw) * delta[:, 0] + torch.sin(start_yaw) * delta[:, 1]
            snapshots[tick] = (dy.cpu().numpy(), dx.cpu().numpy(),
                               env.base_lin_vel[:n, 1].cpu().numpy(), env.base_ang_vel[:n, 2].cpu().numpy())
    rows = []
    for i, (vy, condition) in enumerate(cases):
        rows.append({
            "requested_vy": vy, "condition": condition,
            "timepoints": {str(tick): {
                "lateral_displacement_m": float(value[0][i]),
                "forward_displacement_m": float(value[1][i]),
                "lateral_velocity_m_s": float(value[2][i]),
                "yaw_rate_rad_s": float(value[3][i]),
                "tracking_error_m_s": abs(float(value[2][i]) - vy),
            } for tick, value in snapshots.items()},
            "contact": bool(monitor.contact[i]), "fall": bool(monitor.fall[i]),
            "joint_limit_violation": bool(monitor.joint_violation[i]),
            "torque_limit_violation": bool(monitor.torque_violation[i]),
            "peak_torque": float(monitor.peak_torque[i]), "peak_tilt": float(monitor.peak_tilt[i]),
        })
    return rows


def transition_once(env, route, lateral):
    cases = [(condition, direction * .2) for condition in CONDITIONS for direction in (-1, 1)]
    n = len(cases); monitor = reset(env); preroll(env, route, [c[0] for c in cases], monitor)
    mapping = {"rest": (0, 0), "forward": (.3, 0), "reverse": (-.2, 0),
               "yaw_left": (0, .45), "yaw_right": (0, -.45), "asymmetric_phase": (.2, 0)}
    base = torch.zeros((env.num_envs, 3), device=env.device)
    lateral_commands = torch.zeros_like(base)
    for i, (condition, vy) in enumerate(cases):
        base[i, 0], base[i, 2] = mapping[condition]; lateral_commands[i, 1] = vy
    force_commands(env, lateral_commands)
    previous = monitor.previous_action.clone(); first_lateral = None
    peak_lin = torch.zeros(env.num_envs, device=env.device); peak_ang = torch.zeros_like(peak_lin)
    prior_v = env.base_lin_vel.clone(); prior_w = env.base_ang_vel.clone()
    for tick in range(10):
        actions = step(env, lateral, monitor)
        if tick == 0: first_lateral = actions.clone()
        peak_lin = torch.maximum(peak_lin, torch.linalg.vector_norm(env.base_lin_vel-prior_v, dim=-1)/env.dt)
        peak_ang = torch.maximum(peak_ang, torch.linalg.vector_norm(env.base_ang_vel-prior_w, dim=-1)/env.dt)
        prior_v = env.base_lin_vel.clone(); prior_w = env.base_ang_vel.clone()
    last_lateral = actions.clone(); force_commands(env, base); first_return = None; velocities=[]; yaws=[]
    for tick in range(25):
        actions = step(env, route, monitor)
        if tick == 0: first_return = actions.clone()
        peak_lin = torch.maximum(peak_lin, torch.linalg.vector_norm(env.base_lin_vel-prior_v, dim=-1)/env.dt)
        peak_ang = torch.maximum(peak_ang, torch.linalg.vector_norm(env.base_ang_vel-prior_w, dim=-1)/env.dt)
        prior_v = env.base_lin_vel.clone(); prior_w = env.base_ang_vel.clone()
        if tick >= 15: velocities.append(env.base_lin_vel[:n].cpu().numpy()); yaws.append(env.base_ang_vel[:n,2].cpu().numpy())
    mean_v=np.mean(velocities,axis=0); mean_w=np.mean(yaws,axis=0); rows=[]
    for i,(condition,vy) in enumerate(cases):
        vx,wz=mapping[condition]
        rows.append({
            "condition":condition,"lateral_vy":vy,
            "entry_action_discontinuity":float((first_lateral[i]-previous[i]).abs().max()),
            "return_action_discontinuity":float((first_return[i]-last_lateral[i]).abs().max()),
            "peak_base_acceleration_m_s2":float(peak_lin[i]),
            "peak_angular_acceleration_rad_s2":float(peak_ang[i]),
            "return_vx_error":abs(float(mean_v[i,0])-vx),"return_yaw_error":abs(float(mean_w[i])-wz),
            "contact":bool(monitor.contact[i]),"fall":bool(monitor.fall[i]),
            "joint_limit_violation":bool(monitor.joint_violation[i]),
            "torque_limit_violation":bool(monitor.torque_violation[i]),
        })
    return rows


def compare_repeats(first, second, keys):
    failures=[]
    for index,(a,b) in enumerate(zip(first,second)):
        diffs={key:abs(float(a[key])-float(b[key])) for key in keys}
        if max(diffs.values(),default=0)>C.DETERMINISM_TOLERANCE: failures.append({"row":index,"diffs":diffs})
    return {"pass":not failures,"failures":failures,"tolerance":C.DETERMINISM_TOLERANCE}


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--checkpoint",type=Path,required=True);parser.add_argument("--cfg",type=Path,default=V1_CFG);parser.add_argument("--label",required=True);parser.add_argument("--out",type=Path,default=DEFAULT_OUT)
    args=parser.parse_args(); started=time.time()
    import genesis as gs
    from go2_env import Go2Env
    gs.init(backend=gs.gpu,precision="32",logging_level="warning",seed=2026082014,performance_mode=True)
    with args.cfg.open("rb") as stream: env_cfg,obs_cfg,reward_cfg,command_cfg,_=pickle.load(stream)
    with SOURCE_CFG.open("rb") as stream: _e,_o,_r,_c,train_cfg=pickle.load(stream)
    env=Go2Env(num_envs=128,env_cfg=env_cfg,obs_cfg=obs_cfg,reward_cfg=reward_cfg,command_cfg=command_cfg)
    route=actor(env,train_cfg,SOURCE); successor=actor(env,train_cfg,args.checkpoint)
    route_source_a=route_once(env,route); route_source_b=route_once(env,route)
    route_new_a=route_once(env,successor); route_new_b=route_once(env,successor)
    lateral_a=lateral_once(env,route,successor); lateral_b=lateral_once(env,route,successor)
    transition_a=transition_once(env,route,successor); transition_b=transition_once(env,route,successor)

    def agg(rows):
        return {key:float(np.mean([r[key] for r in rows])) for key in ("vx_abs_error","yaw_abs_error","unintended_vy_abs","energy","action_smoothness","peak_tilt")} | {
            "contacts":sum(r["contact"] for r in rows),"falls":sum(r["fall"] for r in rows),
            "joint_limit_violations":sum(r["joint_limit_violation"] for r in rows),"torque_limit_violations":sum(r["torque_limit_violation"] for r in rows)}
    old,new=agg(route_source_a),agg(route_new_a)
    vx_allowed=max(old["vx_abs_error"]*1.10,old["vx_abs_error"]+.02);yaw_allowed=max(old["yaw_abs_error"]*1.10,old["yaw_abs_error"]+.05)
    route_det=compare_repeats(route_new_a,route_new_b,("vx_abs_error","yaw_abs_error","unintended_vy_abs","energy","action_smoothness","peak_tilt"))
    route_pass=(new["vx_abs_error"]<=vx_allowed and new["yaw_abs_error"]<=yaw_allowed and new["falls"]==new["contacts"]==new["joint_limit_violations"]==new["torque_limit_violations"]==0 and new["peak_tilt"]<=max(old["peak_tilt"]*1.25,old["peak_tilt"]+.05) and route_det["pass"])

    lat_det=[]
    for i,(a,b) in enumerate(zip(lateral_a,lateral_b)):
        diffs=[]
        for tick in ("5","10","25","50"):
            for key in ("lateral_displacement_m","forward_displacement_m","lateral_velocity_m_s","yaw_rate_rad_s"):
                diffs.append(abs(a["timepoints"][tick][key]-b["timepoints"][tick][key]))
        if max(diffs)>C.DETERMINISM_TOLERANCE:lat_det.append({"row":i,"max_abs_diff":max(diffs)})
    correct=all(r["timepoints"]["10"]["lateral_displacement_m"]*r["requested_vy"]>0 for r in lateral_a)
    measurable=all(abs(r["timepoints"]["10"]["lateral_displacement_m"])>=.001 for r in lateral_a)
    max_rows=[r for r in lateral_a if abs(r["requested_vy"])==.2]
    ratio=float(np.median([abs(r["timepoints"]["25"]["lateral_velocity_m_s"])/.2 for r in max_rows]))
    lat_safe=not any(r["contact"] or r["fall"] or r["joint_limit_violation"] or r["torque_limit_violation"] for r in lateral_a)
    lateral_pass=correct and measurable and ratio>=.5 and not lat_det and lat_safe

    trans_det=compare_repeats(transition_a,transition_b,("entry_action_discontinuity","return_action_discontinuity","peak_base_acceleration_m_s2","peak_angular_acceleration_rad_s2","return_vx_error","return_yaw_error"))
    resumes=all(r["return_vx_error"]<=.15 and r["return_yaw_error"]<=.20 for r in transition_a)
    trans_safe=not any(r["contact"] or r["fall"] or r["joint_limit_violation"] or r["torque_limit_violation"] for r in transition_a)
    transition_pass=trans_det["pass"] and resumes and trans_safe
    if not route_pass:classification="LATERAL_CONTROLLER_ROUTE_NON_REGRESSION_FAILURE"
    elif not lateral_pass:classification="LATERAL_TRACKING_AUTHORITY_NO_GO" if not lat_det else "LATERAL_CONTROLLER_DETERMINISM_NO_GO"
    elif not transition_pass:classification="LATERAL_CONTROLLER_MODE_TRANSITION_NO_GO" if trans_det["pass"] else "LATERAL_CONTROLLER_DETERMINISM_NO_GO"
    else:classification="LATERAL_RECOVERY_CONTROLLER_QUALIFIED"
    result={"schema":"lateral_recovery_controller_qualification_v2","label":args.label,"checkpoint":{"path":str(args.checkpoint),"sha256":sha(args.checkpoint)},
            "determinism_contract":"same vectorized lane, complete environment/controller reset, identical command/preroll replay; tolerance 1e-4",
            "route_non_regression":{"source":old,"successor":new,"allowed":{"vx":vx_allowed,"yaw":yaw_allowed},"determinism":route_det,"pass":route_pass,"rows":route_new_a},
            "lateral_tracking":{"correct_sign_every_fixture_at_0_2_s":correct,"measurable_every_fixture_at_0_2_s":measurable,"median_0_5_s_velocity_fraction_for_abs_0_2":ratio,"determinism":{"pass":not lat_det,"failures":lat_det,"tolerance":C.DETERMINISM_TOLERANCE},"safe":lat_safe,"pass":lateral_pass,"rows":lateral_a},
            "mode_transition":{"determinism":trans_det,"route_tracking_resumes":resumes,"safe":trans_safe,"pass":transition_pass,"rows":transition_a},
            "classification":classification,"pass":classification=="LATERAL_RECOVERY_CONTROLLER_QUALIFIED","runtime_s":time.time()-started}
    args.out.mkdir(parents=True,exist_ok=True);path=args.out/f"qualification_{args.label}.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+"\n")
    print(json.dumps({"classification":classification,"route":new,"lateral_ratio":ratio,"lateral_deterministic":not lat_det,"transition_deterministic":trans_det["pass"],"runtime_s":result["runtime_s"]},indent=2))


if __name__=="__main__":main()
