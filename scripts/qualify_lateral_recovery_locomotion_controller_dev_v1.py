#!/usr/bin/env python3
"""Training-only controller qualification for lateral recovery PPO V1."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
import copy
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / ".generated/upstream_genesis/locomotion"
OUT = ROOT / ".generated/lateral_recovery_locomotion_controller_dev_v1"
SOURCE = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SOURCE_CFG = SOURCE.with_name("cfgs.pkl")
NEW = OUT / "seed_2026082014/model_624.pt"
CFG = OUT / "seed_2026082014/cfgs.pkl"
sys.path[:0] = [str(ROOT), str(EXAMPLES)]


def sha(path: Path) -> str:
    h = hashlib.sha256(path.read_bytes()).hexdigest()
    return h


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def actor(env, cfg, checkpoint):
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir=None, device=gs.device)
    runner.load(str(checkpoint), map_location=str(gs.device))
    return runner.get_inference_policy(device=str(gs.device))


def row_key(row):
    return tuple(row[key] for key in ("controller", "command_name", "repeat"))


class Monitor:
    def __init__(self, env):
        self.env = env
        self.n = env.num_envs
        self.contact = torch.zeros(self.n, dtype=torch.bool, device=env.device)
        self.fall = torch.zeros_like(self.contact)
        self.joint_violation = torch.zeros_like(self.contact)
        self.torque_violation = torch.zeros_like(self.contact)
        self.energy = torch.zeros(self.n, device=env.device)
        self.smoothness = torch.zeros(self.n, device=env.device)
        self.peak_torque = torch.zeros(self.n, device=env.device)
        self.peak_tilt = torch.zeros(self.n, device=env.device)
        self.previous_action = None
        solver = env.scene.rigid_solver
        self.disallowed_links = torch.tensor(
            [int(link.idx) for link in solver.links
             if getattr(link, "entity", None) is env.robot
             and not any(token in str(link.name).lower() for token in ("calf", "foot"))],
            device=env.device,
        )
        self.lower_q, self.upper_q = env.robot.get_dofs_limit(env.motors_dof_idx)
        self.lower_f, self.upper_f = env.robot.get_dofs_force_range(env.motors_dof_idx)

    def update(self, actions, dones):
        contacts = self.env.robot.get_contacts(exclude_self_contact=True)
        valid = contacts.get("valid_mask")
        if valid is not None and self.disallowed_links.numel():
            la, lb = contacts["link_a"], contacts["link_b"]
            bad = ((la[..., None] == self.disallowed_links).any(-1)
                   | (lb[..., None] == self.disallowed_links).any(-1)) & valid
            self.contact |= bad.any(-1)
        self.fall |= dones.bool()
        q = self.env.dof_pos
        force = self.env.robot.get_dofs_control_force(self.env.motors_dof_idx)
        self.joint_violation |= ((q < self.lower_q - 1e-4) | (q > self.upper_q + 1e-4)).any(-1)
        finite_force = torch.isfinite(self.lower_f) & torch.isfinite(self.upper_f)
        self.torque_violation |= (((force < self.lower_f - 1e-3) | (force > self.upper_f + 1e-3))
                                  & finite_force).any(-1)
        self.energy += (force.abs() * self.env.dof_vel.abs()).sum(-1) * self.env.dt
        self.peak_torque = torch.maximum(self.peak_torque, force.abs().max(-1).values)
        self.peak_tilt = torch.maximum(self.peak_tilt, torch.linalg.vector_norm(self.env.projected_gravity[:, :2], dim=-1))
        if self.previous_action is not None:
            self.smoothness += (actions - self.previous_action).abs().mean(-1)
        self.previous_action = actions.detach().clone()


def step_policies(env, route, lateral, use_lateral, monitor):
    obs = env.get_observations()
    with torch.inference_mode():
        ar = route(obs, stochastic_output=False)
        al = lateral(obs, stochastic_output=False)
        actions = torch.where(use_lateral[:, None], al, ar)
    _obs, _reward, dones, _extras = env.step(actions)
    monitor.update(actions, dones)
    return actions


def reset(env):
    env._reset_idx()
    env._update_observation()
    return Monitor(env)


def force_commands(env, commands):
    env.commands.copy_(commands)
    if hasattr(env, "_transition_countdown"):
        env._transition_countdown.zero_()


def route_panel(env, route, lateral):
    bank = [
        ("hold", 0.0, 0.0), ("forward_slow", 0.2, 0.0), ("forward_medium", 0.25, 0.0),
        ("forward_fast", 0.3, 0.0), ("backward", -0.2, 0.0), ("yaw_left", 0.0, 0.45),
        ("yaw_right", 0.0, -0.45), ("arc_left", 0.2, 0.45), ("arc_right", 0.2, -0.45),
    ]
    rows = []
    for controller in ("frozen_route", "lateral_successor"):
        for name, vx, wz in bank:
            for repeat in range(2):
                rows.append({"controller": controller, "command_name": name, "repeat": repeat,
                             "requested": [vx, 0.0, wz]})
    n = len(rows); monitor = reset(env)
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    use = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for i, row in enumerate(rows):
        commands[i] = torch.tensor(row["requested"], device=env.device)
        use[i] = row["controller"] == "lateral_successor"
    force_commands(env, commands)
    velocities = []
    for step in range(100):
        step_policies(env, route, lateral, use, monitor)
        if step >= 50:
            velocities.append(env.base_lin_vel[:n].detach().cpu().numpy())
    mean_vel = np.mean(velocities, axis=0)
    ang = env.base_ang_vel[:n, 2].detach().cpu().numpy()
    for i, row in enumerate(rows):
        req = row["requested"]
        row.update({"mean_body_velocity": mean_vel[i].tolist(),
                    "vx_abs_error": abs(mean_vel[i, 0] - req[0]),
                    "yaw_abs_error": abs(float(ang[i]) - req[2]),
                    "unintended_vy_abs": abs(mean_vel[i, 1]),
                    "contact": bool(monitor.contact[i]), "fall": bool(monitor.fall[i]),
                    "joint_limit_violation": bool(monitor.joint_violation[i]),
                    "torque_limit_violation": bool(monitor.torque_violation[i]),
                    "energy": float(monitor.energy[i]), "action_smoothness": float(monitor.smoothness[i] / 99),
                    "peak_tilt": float(monitor.peak_tilt[i])})
    agg = {}
    for controller in ("frozen_route", "lateral_successor"):
        rr = [r for r in rows if r["controller"] == controller]
        agg[controller] = {key: float(np.mean([r[key] for r in rr])) for key in
                           ("vx_abs_error", "yaw_abs_error", "unintended_vy_abs", "energy", "action_smoothness", "peak_tilt")}
        agg[controller].update({"falls": sum(r["fall"] for r in rr),
                                "contacts": sum(r["contact"] for r in rr),
                                "joint_limit_violations": sum(r["joint_limit_violation"] for r in rr),
                                "torque_limit_violations": sum(r["torque_limit_violation"] for r in rr)})
    old, new = agg["frozen_route"], agg["lateral_successor"]
    vx_allowed = max(old["vx_abs_error"] * 1.10, old["vx_abs_error"] + 0.02)
    yaw_allowed = max(old["yaw_abs_error"] * 1.10, old["yaw_abs_error"] + 0.05)
    passed = (new["vx_abs_error"] <= vx_allowed and new["yaw_abs_error"] <= yaw_allowed
              and new["falls"] == old["falls"] == 0 and new["contacts"] == 0
              and new["joint_limit_violations"] == 0 and new["torque_limit_violations"] == 0
              and new["peak_tilt"] <= max(old["peak_tilt"] * 1.25, old["peak_tilt"] + 0.05))
    return {"rows": rows, "aggregate": agg, "allowed": {"vx_abs_error": vx_allowed, "yaw_abs_error": yaw_allowed}, "pass": passed}


def preroll(env, route, conditions, monitor):
    commands = torch.zeros((env.num_envs, 3), device=env.device)
    mapping = {"rest": (0, 0), "forward": (0.3, 0), "reverse": (-0.2, 0),
               "yaw_left": (0, 0.45), "yaw_right": (0, -0.45), "asymmetric_phase": (0.2, 0)}
    for i, name in enumerate(conditions):
        commands[i, 0], commands[i, 2] = mapping[name]
    use = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    force_commands(env, commands)
    for step in range(25):
        if step == 15:
            for i, name in enumerate(conditions):
                if name == "asymmetric_phase": commands[i] = 0.0
            force_commands(env, commands)
        step_policies(env, route, route, use, monitor)


def lateral_panel(env, route, lateral):
    rows = []
    conditions = ("rest", "forward", "reverse", "yaw_left", "yaw_right", "asymmetric_phase")
    for vy in (-0.05, 0.05, -0.10, 0.10, -0.15, 0.15, -0.20, 0.20):
        for condition in conditions:
            for repeat in range(2): rows.append({"requested_vy": vy, "condition": condition, "repeat": repeat})
    n=len(rows); monitor=reset(env); preroll(env, route, [r["condition"] for r in rows], monitor)
    start_pos=env.base_pos[:n].detach().clone(); start_yaw=torch.deg2rad(env.base_euler[:n,2].detach().clone())
    commands=torch.zeros((env.num_envs,3),device=env.device); use=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
    for i,row in enumerate(rows): commands[i,1]=row["requested_vy"]; use[i]=True
    force_commands(env,commands); snapshots={}
    for step in range(1,51):
        step_policies(env,route,lateral,use,monitor)
        if step in (5,10,25,50):
            delta=env.base_pos[:n,:2]-start_pos[:,:2]
            dy=-torch.sin(start_yaw)*delta[:,0]+torch.cos(start_yaw)*delta[:,1]
            dx=torch.cos(start_yaw)*delta[:,0]+torch.sin(start_yaw)*delta[:,1]
            snapshots[step]=(dy.detach().cpu().numpy(),dx.detach().cpu().numpy(),
                             env.base_lin_vel[:n,1].detach().cpu().numpy(),
                             env.base_ang_vel[:n,2].detach().cpu().numpy())
    for i,row in enumerate(rows):
        row["timepoints"]={str(step): {"lateral_displacement_m":float(v[0][i]),"forward_displacement_m":float(v[1][i]),
                                        "lateral_velocity_m_s":float(v[2][i]),"yaw_rate_rad_s":float(v[3][i]),
                                        "tracking_error_m_s":abs(float(v[2][i])-row["requested_vy"])} for step,v in snapshots.items()}
        row.update({"contact":bool(monitor.contact[i]),"fall":bool(monitor.fall[i]),
                    "joint_limit_violation":bool(monitor.joint_violation[i]),"torque_limit_violation":bool(monitor.torque_violation[i]),
                    "peak_torque":float(monitor.peak_torque[i]),"peak_tilt":float(monitor.peak_tilt[i])})
    correct=all(r["timepoints"]["10"]["lateral_displacement_m"]*r["requested_vy"]>0 for r in rows)
    measurable=all(abs(r["timepoints"]["10"]["lateral_displacement_m"])>=0.001 for r in rows)
    max_rows=[r for r in rows if abs(r["requested_vy"])==0.2]
    ratio=float(np.median([abs(r["timepoints"]["25"]["lateral_velocity_m_s"])/.2 for r in max_rows]))
    deterministic=True
    for vy in (-.05,.05,-.1,.1,-.15,.15,-.2,.2):
        for condition in conditions:
            pair=[r for r in rows if r["requested_vy"]==vy and r["condition"]==condition]
            a,b=pair
            av=round(a["timepoints"]["25"]["lateral_displacement_m"],4); bv=round(b["timepoints"]["25"]["lateral_displacement_m"],4)
            deterministic &= av==bv
    passed=(correct and measurable and ratio>=.5 and deterministic and not any(r["contact"] or r["fall"] or r["joint_limit_violation"] or r["torque_limit_violation"] for r in rows))
    return {"rows":rows,"correct_sign_every_fixture_at_0_2_s":correct,"measurable_every_fixture_at_0_2_s":measurable,
            "median_0_5_s_velocity_fraction_for_abs_0_2":ratio,"deterministic_reduction":deterministic,"pass":passed}


def transition_panel(env, route, lateral):
    conditions=("rest","forward","reverse","yaw_left","yaw_right","asymmetric_phase")
    rows=[]
    for condition in conditions:
        for direction in (-1,1):
            for repeat in range(2): rows.append({"condition":condition,"lateral_vy":direction*.2,"repeat":repeat})
    n=len(rows);monitor=reset(env);preroll(env,route,[r["condition"] for r in rows],monitor)
    base_commands=torch.zeros((env.num_envs,3),device=env.device)
    mapping={"rest":(0,0),"forward":(.3,0),"reverse":(-.2,0),"yaw_left":(0,.45),"yaw_right":(0,-.45),"asymmetric_phase":(.2,0)}
    for i,row in enumerate(rows): base_commands[i,0],base_commands[i,2]=mapping[row["condition"]]
    lateral_commands=torch.zeros_like(base_commands)
    for i,row in enumerate(rows): lateral_commands[i,1]=row["lateral_vy"]
    use=torch.ones(env.num_envs,dtype=torch.bool,device=env.device);force_commands(env,lateral_commands)
    previous=monitor.previous_action.detach().clone(); first_lat=None
    peak_lin_acc=torch.zeros(env.num_envs,device=env.device);peak_ang_acc=torch.zeros_like(peak_lin_acc)
    pv=env.base_lin_vel.detach().clone();pw=env.base_ang_vel.detach().clone()
    for step in range(10):
        a=step_policies(env,route,lateral,use,monitor)
        if step==0:first_lat=a.detach().clone()
        peak_lin_acc=torch.maximum(peak_lin_acc,torch.linalg.vector_norm(env.base_lin_vel-pv,dim=-1)/env.dt)
        peak_ang_acc=torch.maximum(peak_ang_acc,torch.linalg.vector_norm(env.base_ang_vel-pw,dim=-1)/env.dt);pv=env.base_lin_vel.clone();pw=env.base_ang_vel.clone()
    last_lat=a.detach().clone();use.zero_();force_commands(env,base_commands);first_return=None;vel=[];yaw=[]
    for step in range(25):
        a=step_policies(env,route,lateral,use,monitor)
        if step==0:first_return=a.detach().clone()
        peak_lin_acc=torch.maximum(peak_lin_acc,torch.linalg.vector_norm(env.base_lin_vel-pv,dim=-1)/env.dt)
        peak_ang_acc=torch.maximum(peak_ang_acc,torch.linalg.vector_norm(env.base_ang_vel-pw,dim=-1)/env.dt);pv=env.base_lin_vel.clone();pw=env.base_ang_vel.clone()
        if step>=15:vel.append(env.base_lin_vel[:n].detach().cpu().numpy());yaw.append(env.base_ang_vel[:n,2].detach().cpu().numpy())
    mv=np.mean(vel,axis=0);mw=np.mean(yaw,axis=0)
    for i,row in enumerate(rows):
        vx,wz=mapping[row["condition"]]
        row.update({"entry_action_discontinuity":float((first_lat[i]-previous[i]).abs().max()),
                    "return_action_discontinuity":float((first_return[i]-last_lat[i]).abs().max()),
                    "peak_base_acceleration_m_s2":float(peak_lin_acc[i]),"peak_angular_acceleration_rad_s2":float(peak_ang_acc[i]),
                    "return_vx_error":abs(float(mv[i,0])-vx),"return_yaw_error":abs(float(mw[i])-wz),
                    "contact":bool(monitor.contact[i]),"fall":bool(monitor.fall[i]),
                    "joint_limit_violation":bool(monitor.joint_violation[i]),"torque_limit_violation":bool(monitor.torque_violation[i])})
    deterministic=True
    for condition in conditions:
        for vy in (-.2,.2):
            pair=[r for r in rows if r["condition"]==condition and r["lateral_vy"]==vy]
            deterministic &= round(pair[0]["entry_action_discontinuity"],4)==round(pair[1]["entry_action_discontinuity"],4)
    resumes=all(r["return_vx_error"]<=.15 and r["return_yaw_error"]<=.20 for r in rows)
    passed=deterministic and resumes and not any(r["contact"] or r["fall"] or r["joint_limit_violation"] or r["torque_limit_violation"] for r in rows)
    return {"rows":rows,"deterministic_reduction":deterministic,"route_tracking_resumes":resumes,
            "frozen_return_thresholds":{"vx_abs_error_m_s":.15,"yaw_abs_error_rad_s":.20},"pass":passed}


def main():
    import genesis as gs
    from go2_env import Go2Env
    started=time.time();gs.init(backend=gs.gpu,precision="32",logging_level="warning",seed=2026082014,performance_mode=True)
    with CFG.open("rb") as f: env_cfg,obs_cfg,reward_cfg,command_cfg,_mutated_train_cfg=pickle.load(f)
    with SOURCE_CFG.open("rb") as f: _e,_o,_r,_c,train_cfg=pickle.load(f)
    env=Go2Env(num_envs=256,env_cfg=env_cfg,obs_cfg=obs_cfg,reward_cfg=reward_cfg,command_cfg=command_cfg)
    route=actor(env,train_cfg,SOURCE);lateral=actor(env,train_cfg,NEW)
    route_result=route_panel(env,route,lateral); lateral_result=lateral_panel(env,route,lateral); transition_result=transition_panel(env,route,lateral)
    if not route_result["pass"]: classification="LATERAL_CONTROLLER_ROUTE_NON_REGRESSION_FAILURE"
    elif not lateral_result["pass"]: classification="LATERAL_TRACKING_AUTHORITY_NO_GO"
    elif not transition_result["pass"]: classification="LATERAL_CONTROLLER_MODE_TRANSITION_NO_GO"
    else: classification="LATERAL_RECOVERY_CONTROLLER_QUALIFIED"
    result={"schema":"lateral_recovery_controller_qualification_v1","source_checkpoint_sha256":sha(SOURCE),
            "lateral_checkpoint_sha256":sha(NEW),"route_non_regression":route_result,"lateral_tracking":lateral_result,
            "mode_transition":transition_result,"classification":classification,"pass":classification=="LATERAL_RECOVERY_CONTROLLER_QUALIFIED",
            "runtime_s":time.time()-started}
    path=OUT/"controller_qualification.json";path.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False,default=json_default)+"\n")
    print(json.dumps({"classification":classification,"route":route_result["aggregate"],
                      "lateral_ratio":lateral_result["median_0_5_s_velocity_fraction_for_abs_0_2"],
                      "transition_resumes":transition_result["route_tracking_resumes"],"runtime_s":result["runtime_s"]},indent=2,default=json_default))


if __name__=="__main__": main()
