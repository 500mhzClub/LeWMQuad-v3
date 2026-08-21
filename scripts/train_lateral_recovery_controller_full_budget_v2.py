#!/usr/bin/env python3
"""One corrected 500-update lateral-controller successor (Path C)."""
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
OUT = ROOT / ".generated/lateral_controller_failure_attribution_and_full_budget_successor_v2"
RUN = OUT / "seed_2026082015"
SOURCE = ROOT / "models/tier_a_go2_locomotion/20260516_contract_ppo/model_500.pt"
SOURCE_CFG = SOURCE.with_name("cfgs.pkl")
sys.path[:0] = [str(ROOT), str(EXAMPLES)]

from lewm.control import lateral_controller_failure_attribution_v2 as C
from scripts import train_lateral_recovery_locomotion_controller_dev_v1 as V1
from scripts.qualify_lateral_recovery_locomotion_controller_dev_v1 import Monitor


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1<<20),b""):h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")


CORRECTION = {
    "old": "exp(-(vx_error^2 + vy_error^2) / 0.25)",
    "new": "exp(-vx_error^2 / 0.25 - vy_error^2 / 0.04)",
    "scope": "only the y-axis resolution of the existing linear tracking reward",
    "lateral_sigma_m2_s2": C.CORRECTED_LATERAL_TRACKING_SIGMA,
}
CORRECTION["digest"] = C.digest(CORRECTION)


def build_corrected_env_class(args):
    base = V1.build_env_class(args)

    class CorrectedLateralMixtureGo2Env(base):
        def _reward_tracking_lin_vel(self):
            x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
            y_error = torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1])
            return torch.exp(
                -x_error / self.reward_cfg["tracking_sigma"]
                -y_error / C.CORRECTED_LATERAL_TRACKING_SIGMA
            )

    return CorrectedLateralMixtureGo2Env


def configs(run_name: str):
    with SOURCE_CFG.open("rb") as stream:
        env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg=pickle.load(stream)
    command_cfg=dict(command_cfg)
    command_cfg["lin_vel_y_range"]=[-C.VY_LIMIT_M_S,C.VY_LIMIT_M_S]
    command_cfg["sampler"]="fixed_50_route_25_lateral_25_transition_v1"
    command_cfg["transition_delay_policy_steps"]=50
    train_cfg=dict(train_cfg);train_cfg["run_name"]=run_name;train_cfg["save_interval"]=1000
    return env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg


def force_commands(env, commands):
    env.commands.copy_(commands)
    if hasattr(env,"_transition_countdown"):env._transition_countdown.zero_()
    env._update_observation()


def fixed_monitor(env, runner, update_count: int) -> dict:
    runner.alg.eval_mode();policy=runner.get_inference_policy(device=str(env.device))
    env._reset_idx();env._update_observation();monitor=Monitor(env)
    route=[("hold",0,0),("forward_slow",.2,0),("forward_medium",.25,0),("forward_fast",.3,0),("backward",-.2,0),("yaw_left",0,.45),("yaw_right",0,-.45),("arc_left",.2,.45),("arc_right",.2,-.45)]
    lateral=[vy for vy in (-.05,.05,-.1,.1,-.15,.15,-.2,.2) for _ in range(2)]
    commands=torch.zeros((env.num_envs,3),device=env.device)
    for i,(_name,vx,wz) in enumerate(route):commands[i]=torch.tensor([vx,0,wz],device=env.device)
    offset=len(route)
    for i,vy in enumerate(lateral):commands[offset+i,1]=vy
    force_commands(env,commands);vel=[];ang=[];positions=[]
    start=env.base_pos.detach().clone()
    for tick in range(25):
        obs=env.get_observations()
        with torch.inference_mode():actions=policy(obs,stochastic_output=False)
        _obs,_rew,dones,_extras=env.step(actions);monitor.update(actions,dones)
        if tick>=14:
            vel.append(env.base_lin_vel.detach().cpu().numpy());ang.append(env.base_ang_vel[:,2].detach().cpu().numpy())
    mean_vel=np.mean(vel,axis=0);mean_ang=np.mean(ang,axis=0)
    lateral_vel=mean_vel[offset:offset+len(lateral),1];requested=np.asarray(lateral)
    route_vx=np.asarray([x[1] for x in route]);route_wz=np.asarray([x[2] for x in route])
    force=env.robot.get_dofs_control_force(env.motors_dof_idx)
    lower,upper=env.robot.get_dofs_force_range(env.motors_dof_idx);finite=torch.isfinite(lower)&torch.isfinite(upper)
    torque_viol=int((((force<lower-1e-3)|(force>upper+1e-3))&finite).any(-1).sum().item())
    row={
        "updates_completed":update_count,
        "pure_lateral_achieved_requested_ratio_median":float(np.median(np.abs(lateral_vel)/np.abs(requested))),
        "pure_lateral_sign_accuracy":float(np.mean(lateral_vel*requested>0)),
        "pure_lateral_forward_drift_abs_mean_m_s":float(np.mean(np.abs(mean_vel[offset:offset+len(lateral),0]))),
        "pure_lateral_yaw_drift_abs_mean_rad_s":float(np.mean(np.abs(mean_ang[offset:offset+len(lateral)]))),
        "route_vx_abs_error_mean_m_s":float(np.mean(np.abs(mean_vel[:len(route),0]-route_vx))),
        "route_yaw_abs_error_mean_rad_s":float(np.mean(np.abs(mean_ang[:len(route)]-route_wz))),
        "falls":int(monitor.fall[:offset+len(lateral)].sum().item()),
        "contacts":int(monitor.contact[:offset+len(lateral)].sum().item()),
        "joint_limit_violations":int(monitor.joint_violation[:offset+len(lateral)].sum().item()),
        "torque_limit_violations":torque_viol,
    }
    runner.alg.train_mode()
    return row


def contract() -> dict:
    with SOURCE_CFG.open("rb") as stream:
        env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg=pickle.load(stream)
    value={
        "schema":"lateral_controller_full_budget_successor_v2_contract",
        "source_commit":C.SOURCE_COMMIT,"path":"PATH_C_CORRECTED_SUCCESSOR_TRAINING",
        "seed":C.SUCCESSOR_SEED,"updates":C.SUCCESSOR_UPDATES,"start_checkpoint":{"path":str(SOURCE),"sha256":sha(SOURCE)},
        "parallel_environments":4096,"command_mixture":{"historical_route":.50,"pure_lateral":.25,"route_to_lateral":.25},
        "vy_range_m_s":[-.2,.2],"correction":CORRECTION,"ppo":train_cfg,"environment":env_cfg,"observation":obs_cfg,
        "reward":reward_cfg,"original_command":command_cfg,"monitor_interval_updates":25,"final_update_only":True,
        "second_successor_path":False,"hyperparameter_sweep":False,
    }
    value["content_digest"]=C.digest(value);return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--freeze",action="store_true");parser.add_argument("--train",action="store_true");args=parser.parse_args()
    if args.freeze==args.train:parser.error("choose exactly one")
    OUT.mkdir(parents=True,exist_ok=True);frozen=contract();write_json(OUT/"successor_training_contract.json",frozen)
    if args.freeze:print(json.dumps(frozen,indent=2));return
    if RUN.exists():raise RuntimeError(f"single successor output exists: {RUN}")
    import genesis as gs
    from rsl_rl.runners import OnPolicyRunner
    gs.init(backend=gs.gpu,precision="32",logging_level="warning",seed=C.SUCCESSOR_SEED,performance_mode=True)
    env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg=configs("seed_2026082015")
    args_env=type("Args",(),{"historical_bank":command_cfg["lewm_command_bank"]})()
    klass=build_corrected_env_class(args_env)
    RUN.mkdir(parents=True,exist_ok=False)
    frozen_cfg=(env_cfg,obs_cfg,reward_cfg,command_cfg,train_cfg)
    with (RUN/"cfgs.pkl").open("wb") as stream:pickle.dump(frozen_cfg,stream)
    train_env=klass(num_envs=4096,env_cfg=copy.deepcopy(env_cfg),obs_cfg=copy.deepcopy(obs_cfg),reward_cfg=copy.deepcopy(reward_cfg),command_cfg=copy.deepcopy(command_cfg))
    runner=OnPolicyRunner(train_env,train_cfg,str(RUN),device=gs.device);runner.load(str(SOURCE),map_location=str(gs.device))
    monitor_env=klass(num_envs=64,env_cfg=copy.deepcopy(env_cfg),obs_cfg=copy.deepcopy(obs_cfg),reward_cfg=copy.deepcopy(reward_cfg),command_cfg=copy.deepcopy(command_cfg))
    monitoring=[];original_log=runner.logger.log;start_iteration=int(runner.current_learning_iteration)
    def hooked_log(*pos,**kw):
        original_log(*pos,**kw)
        iteration=int(kw.get("it",runner.current_learning_iteration));completed=iteration-start_iteration+1
        if completed%25==0:
            monitoring.append(fixed_monitor(monitor_env,runner,completed));write_json(OUT/"monitoring_records.json",monitoring)
    runner.logger.log=hooked_log
    started=time.time();runner.learn(num_learning_iterations=C.SUCCESSOR_UPDATES,init_at_random_ep_len=True);runtime=time.time()-started
    checkpoint=RUN/"model_999.pt"
    if not checkpoint.is_file():raise RuntimeError(f"final checkpoint missing: {checkpoint}")
    result={"schema":"lateral_controller_full_budget_successor_v2_training_result","contract":frozen,"seed":C.SUCCESSOR_SEED,"updates":C.SUCCESSOR_UPDATES,
            "start_iteration":500,"final_iteration":999,"checkpoint":{"path":str(checkpoint),"bytes":checkpoint.stat().st_size,"sha256":sha(checkpoint)},
            "cfg":{"path":str(RUN/"cfgs.pkl"),"sha256":sha(RUN/"cfgs.pkl")},"monitoring_records":len(monitoring),"monitoring":monitoring,"runtime_s":runtime}
    write_json(OUT/"successor_training_result.json",result);print(json.dumps({"checkpoint":result["checkpoint"],"monitoring_records":len(monitoring),"runtime_s":runtime},indent=2))


if __name__=="__main__":main()
