"""Six-action diagnosis of the first wrong-direction decision on fresh task 01."""
import argparse
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from lewm.dense_metric_goal_control_development import MetricGoalControl
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm_genesis.lewm_contract import apply_safety_limits_single
from scripts import run_go2_fresh_visual_goal_comparison_development as live

base = live.pilot.base
OUTPUT = live.previous.previous.OUTPUT.parent/'go2_fresh_goal_direction_v1_attempt_001'
PLAN = Path('docs/go2_fresh_goal_direction_plan_2026-09-17.json')
RESULT = Path('docs/go2_fresh_goal_direction_result_2026-09-17.json')
ORIGINAL = live.OUTPUT/'case_02'
NEW_ACTIONS = (0, 1, 2, 3, 5)  # The actual left-turn successor already exists.


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    decision=json.loads((ORIGINAL/'decisions.json').read_text())[0]
    assert decision['tick']==10 and decision['action']=='left_turn'
    result=json.loads((ORIGINAL/'result.json').read_text())
    assert result['status']=='COMPLETE' and result['completed_budget']
    free=shutil.disk_usage(OUTPUT.parent).free;assert free>(512+48)*1024**2,free
    plan=dict(source_sha256={p:base.digest(p) for p in (list(live.SOURCES)+[__file__])},
        live_plan_sha256=base.digest(live.PLAN),original_result_sha256=base.digest(ORIGINAL/'result.json'),
        original_decisions_sha256=base.digest(ORIGINAL/'decisions.json'),original_decision=decision,
        scene='fresh_01',case=2,departure_tick=10,context_frames=[0,5,10],suffix_ticks=5,
        actions=list(ACTIONS),new_actions=list(NEW_ACTIONS),reused_action='left_turn',
        new_native_branches=5,selection='first decision on first right-opening fresh task; post-hoc wrong-direction diagnosis',
        source=str(ORIGINAL),goal=str(live.GOALS/'fresh_01/rgb_0023.png'),
        resources=dict(output_free_bytes=free,allowance_bytes=48*1024**2,reserve_bytes=512*1024**2,
            available_ram_gib=72,concurrency=2,cpu_groups=[[4,5,6,7],[8,9,10,11]]),
        no_training=True,depth_recorded=False,new_navigation=False,
        limitations=['one exposed post-hoc state','five new branches plus one reused factual successor',
            '500-ms local ranking does not prove navigation or long-horizon optimality'])
    OUTPUT.mkdir();base.save(PLAN,plan);base.save(OUTPUT/'plan.json',plan)
    print('DIRECTION_DIAGNOSIS_PREPARED',flush=True)


def validate():
    plan=json.loads(PLAN.read_text())
    assert all(base.digest(p)==h for p,h in plan['source_sha256'].items())
    assert base.digest(ORIGINAL/'decisions.json')==plan['original_decisions_sha256']
    return plan


def branch(index):
    from scripts.run_go2_contact_attributed_execution_development_v1 import PhysicalStop
    validate();assert index in NEW_ACTIONS
    directory=OUTPUT/f'action_{index:02d}';directory.mkdir()
    cv2.setNumThreads(1);cv2.ocl.setUseOpenCL(False);torch.set_num_threads(4)
    torch.manual_seed(2026091706);start=time.monotonic();session=None;stop=None;matched=False
    base.save(directory/'launch.json',dict(pid=os.getpid(),action=ACTIONS[index],affinity=sorted(os.sched_getaffinity(0))))
    try:
        spec=live.specification('fresh_01');base.save(directory/'specification.json',spec)
        live.initialize_genesis(backend='cpu',seed=spec['procedural_seed'],logging_level='warning')
        session=live.FreshVisualGoalSession(spec,directory);session.install_contact_identity()
        gains=live.configure_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist(),session.ctx.policy.env_cfg,'checkpoint')
        base.save(directory/'actuator_identity.json',gains)
        try:
            session.settle_recorded();session.capture_current();live.admit_context_setup(session,base.digest(PLAN))
            tape=[[0.,0.,0.]]*10+[candidate_commands(ACTIONS[index])[0]]*5
            for tick in range(16):
                session.sensor_packets()
                if tick==10:
                    with np.load(ORIGINAL/'physics_trace.npz',allow_pickle=False) as old:
                        for key in ('base_pose_world','joint_position','joint_velocity','applied_command'):
                            np.testing.assert_array_equal(np.stack([s[key] for s in session.samples]),old[key][:len(session.samples)])
                    for frame in (0,5,10):
                        assert (directory/f'rgb_{frame:04d}.png').read_bytes()==(ORIGINAL/f'rgb_{frame:04d}.png').read_bytes()
                    matched=True
                if tick==15:break
                if shutil.disk_usage(OUTPUT).free<512*1024**2:raise RuntimeError('storage reserve reached')
                session.phase=1 if tick<10 else 2;session.command_tick(tape[tick])
        except PhysicalStop as error:
            stop=str(error);session.capture_current()
        assert matched
        assert live.read_gains(session.ctx.build.robot,session.ctx.runner._leg_dof_idx.tolist())==gains['effective']
        result=dict(status='COMPLETE',action=ACTIONS[index],action_index=index,
            complete_500ms=stop is None and len(session.image_audit)==16,physical_stop=stop,
            disallowed_contact=any(bool(s['physics_contact']) for s in session.samples),
            exact_native_and_context_prefix=True,wall_s=time.monotonic()-start)
    except Exception as error:
        base.save(directory/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()));raise
    finally:
        if session is not None:
            try:
                session.persist(directory);session.persist_observations(directory)
                base.save(directory/'native_guard_rows.json',session.guard_rows)
            finally:session.ctx.build.scene.destroy()
        live.shutdown_genesis()
    base.save(directory/'result.json',result);print('DIRECTION_BRANCH_COMPLETE',json.dumps(result),flush=True)


@torch.inference_mode()
def score():
    plan=validate();assert not RESULT.exists();torch.set_num_threads(4);start=time.monotonic()
    controller=MetricGoalControl('action',Path(plan['goal']))
    def encode(path):
        with Image.open(path) as im:rgb=np.asarray(im.convert('RGB'))
        return controller.encode(rgb)
    context=torch.stack([encode(ORIGINAL/f'rgb_{i:04d}.png') for i in (0,5,10)])[None]
    with np.load(ORIGINAL/'policy_histories.npz',allow_pickle=False) as a:
        commands=a['applied_command_values'][10].astype(np.float32)
        assert a['applied_command_valid'][10].all()
    control=torch.from_numpy((commands[:,[0,2]].reshape(3,5,2)-controller.mean)/controller.std).cuda()[None]
    applied=np.asarray([apply_safety_limits_single([candidate_commands(name)[0]]*5,tuple(commands[-1]),controller.limits)[0]
        for name in ACTIONS],np.float32)
    actions=torch.from_numpy(applied[:,:,[0,2]].reshape(6,10)).cuda()
    mask=torch.ones(6,768,dtype=torch.bool,device='cuda')
    pred=F.layer_norm(controller.model(context.expand(6,-1,-1,-1),actions,mask,control=control.expand(6,-1,-1,-1)).float(),(1024,))
    embeddings=controller.metric.embed(pool_tokens(pred))
    costs=(embeddings-controller.goal_embedding).square().mean(-1)
    np.testing.assert_allclose(costs.cpu().numpy(),plan['original_decision']['costs'],rtol=0,atol=1e-6)
    base.save(OUTPUT/'forecasts_complete.json',dict(original_costs_reproduced=True,future_images_loaded=False,
        predicted_costs=costs.cpu().tolist()))
    # Only now load actual successors and evaluator-only goal/robot poses.
    original=json.loads((ORIGINAL/'result.json').read_text());goal=np.asarray(original['goal_pose_evaluator_only'])
    goal_rotation=live.pilot.rotation_xyzw(goal[3:]);goal_yaw=np.arctan2(goal_rotation[1,0],goal_rotation[0,0])
    rows=[];targets=[]
    for index,name in enumerate(ACTIONS):
        directory=ORIGINAL if index==4 else OUTPUT/f'action_{index:02d}'
        if index!=4:
            r=json.loads((directory/'result.json').read_text())
            assert r['status']=='COMPLETE' and r['complete_500ms'] and not r['disallowed_contact']
        with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
            np.testing.assert_allclose(a['applied_command_values'][15][-5:],applied[index],atol=1e-6,rtol=0)
        cameras=json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as a:pose=a['base_pose_world'][cameras[15]['physical_sample_index']]
        target=encode(directory/'rgb_0015.png');targets.append(target)
        z=controller.metric.embed(pool_tokens(target[None]));rotation=live.pilot.rotation_xyzw(pose[3:])
        yaw=np.arctan2(rotation[1,0],rotation[0,0])-goal_yaw;yaw=np.arctan2(np.sin(yaw),np.cos(yaw))
        relative=goal_rotation.T@rotation;native_yaw=abs(np.arctan2(relative[1,0],relative[0,0]))
        xy=float(np.linalg.norm(pose[:2]-goal[:2]))
        rows.append(dict(action=name,predicted_cost=float(costs[index]),
            actual_image_cost=float((z-controller.goal_embedding).square().mean()),
            actual_raw_goal_mse=float((target-controller.goal).square().mean()),
            xy_error_cm=xy*100,yaw_error_deg=float(np.rad2deg(native_yaw)),
            physical_cost=float((xy/.03)**2+(yaw/np.deg2rad(5))**2),
            future_dense_mse=float((pred[index]-target).square().mean()),
            persistence_mse=float((context[0,-1]-target).square().mean()),
            reused_factual_successor=index==4))
    target=torch.stack(targets);matrix=(pred[:,None]-target[None]).square().mean((-1,-2))
    wins=[bool(matrix[i,i]<torch.cat((matrix[:i,i],matrix[i+1:,i])).min()) for i in range(6)]
    report=dict(status='COMPLETE',rows=rows,plan_sha256=base.digest(PLAN),
        original_forecast_costs_reproduced=True,all_native_contexts_and_applied_tapes_matched=True,
        predicted_choice=min(rows,key=lambda r:r['predicted_cost'])['action'],
        actual_image_choice=min(rows,key=lambda r:r['actual_image_cost'])['action'],
        physical_choice=min(rows,key=lambda r:r['physical_cost'])['action'],
        actual_raw_goal_choice=min(rows,key=lambda r:r['actual_raw_goal_mse'])['action'],
        mean_future_dense_mse=float(matrix.diag().mean()),
        mean_persistence_mse=float((target-context[0,-1]).square().mean()),
        correct_action_retrieval=sum(wins),correct_action_wins=wins,mse_matrix=matrix.cpu().tolist(),
        wall_s=time.monotonic()-start,new_native_branches=5,reused_factual_branches=1,
        post_hoc_diagnostic=True,new_navigation=False,encoder_predictor_cost_unchanged=True,
        limitations=plan['limitations'])
    base.save(OUTPUT/'result.json',report);base.save(RESULT,report)
    print('DIRECTION_DIAGNOSIS_COMPLETE',json.dumps({k:v for k,v in report.items() if k!='mse_matrix'}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--prepare',action='store_true');g.add_argument('--branch',type=int,choices=NEW_ACTIONS);g.add_argument('--score',action='store_true');a=p.parse_args()
    if a.prepare:prepare()
    elif a.score:score()
    else:branch(a.branch)
