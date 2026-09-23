"""Replay only a snapshot's own frozen source selection, never an audit row."""
import copy
from dataclasses import asdict
import threading
from types import SimpleNamespace

import numpy as np
import torch

from lewm.delayed_action_planning_development import ScheduledCommand
from lewm_genesis.lewm_contract import apply_safety_limits_batch
from scripts import run_go2_dense_horizon_navigation_development as native


def differences(actual, expected, *, atol, rtol, path='root'):
    """Discrete/schema equality, finite numerical agreement; no silent keys."""
    if isinstance(actual, np.ndarray): actual = actual.tolist()
    if isinstance(expected, np.ndarray): expected = expected.tolist()
    if isinstance(actual, torch.Tensor): actual = actual.detach().cpu().tolist()
    if isinstance(expected, torch.Tensor): expected = expected.detach().cpu().tolist()
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            return [dict(path=path, reason='KEY_SET', actual_keys=list(actual) if isinstance(actual,dict) else None, expected_keys=list(expected))]
        return [r for k in expected for r in differences(actual[k],expected[k],atol=atol,rtol=rtol,path=path+'.'+str(k))]
    if isinstance(expected, (list,tuple)):
        if not isinstance(actual,(list,tuple)) or len(actual)!=len(expected):
            return [dict(path=path,reason='SEQUENCE_LENGTH')]
        return [r for i,(a,e) in enumerate(zip(actual,expected)) for r in differences(a,e,atol=atol,rtol=rtol,path=f'{path}[{i}]')]
    if isinstance(expected,(bool,str,int,type(None))):
        if type(actual) is not type(expected) or actual != expected:
            return [dict(path=path,reason='DISCRETE',actual=actual,expected=expected)]
    elif isinstance(expected,(float,np.floating)):
        if not isinstance(actual,(float,int,np.number)) or not np.isfinite([actual,expected]).all() or not np.isclose(actual,expected,atol=atol,rtol=rtol):
            return [dict(path=path,reason='NUMERICAL',actual=float(actual),expected=float(expected))]
    elif actual != expected:
        return [dict(path=path,reason='UNSUPPORTED_OR_UNEQUAL',type=type(expected).__name__)]
    return []


@torch.inference_mode()
def qualify(packet, physical, model, controller_name, original_plan, original_request, tolerances):
    cls=native.DenseReactiveNavigationRuntime if controller_name=='reactive_feedback' else native.DenseNavigationRuntime
    controller=object.__new__(cls)
    controller.__dict__.update(copy.deepcopy(packet['controller_state']))
    controller.model=model
    controller.lock=threading.RLock();controller.correction_pose_lock=threading.RLock()
    controller.profile_current=None;controller.plan_profile_rows=[]
    controller.visual_dispatch_events=[];controller.events=[]
    controller.clock_ns=lambda: original_plan['completed_ns']
    controller.planning=[]
    item=SimpleNamespace(**copy.deepcopy(packet['packet']))
    # Hook is observational and is the unchanged source's input-contract hook.
    if model._forward_pre_hooks:
        raise ValueError('live source model input hook must be detached after collection before replay')
    hook=model.register_forward_pre_hook(controller._check_model_inputs,with_kwargs=True)
    try:
        selected,correction=controller._select_action(item,copy.deepcopy(packet['evidence']),
            copy.deepcopy(packet['committed_prefix']),copy.deepcopy(packet['route_target_body_xy']),
            packet['scan_error_rad'],copy.deepcopy(packet['observed_map']),
            np.array(packet['observed_position']),np.array(packet['observed_rotation']))
    finally:
        hook.remove()
    compare=lambda a,e,p: differences(a,e,atol=tolerances['absolute'],rtol=tolerances['relative'],path=p)
    failures=compare(selected,packet['source_selection'],'selection')+compare(correction,packet['source_correction'],'correction')
    receipt=copy.deepcopy(model.receipts[-1]);expected=copy.deepcopy(packet['source_model_receipt'])
    receipt.pop('wall_ns');expected.pop('wall_ns')
    failures+=compare(receipt,expected,'model_receipt')
    failures+=compare(receipt['requested_commands'],packet['candidate_requested_commands'],'requested_tapes')
    failures+=compare(receipt['applied_commands'],packet['candidate_applied_commands'],'limited_tapes')
    # Recreate plan installation and the current-time request from the same
    # pre-selection state. No later obstacle observation or true pose is used.
    plan=ScheduledCommand.prepare(selected['action'],observed_ns=item.measured_ns,
        completed_ns=original_plan['completed_ns'],delay_ticks=controller.planning_delay_ticks,
        commit_ticks=controller.commit_ticks)
    controller.planning.append(dict(frame=item.frame,measured_ns=item.measured_ns,
        completed_ns=original_plan['completed_ns'],action=selected['action'],on_time=plan is not None,
        selection=selected,committed_prefix=packet['committed_prefix'],route_status=packet['route']['status']))
    if plan is not None:
        controller._store_plan(plan,original_plan['completed_ns'],packet['committed_prefix'])
    request=controller.request(now_ns=item.measured_ns)
    expected_request={k:v for k,v in original_request.items() if k not in ('simulator_ns','pre_sample_index','post_sample_index','applied_command')}
    failures+=compare(request,expected_request,'final_request_at_snapshot')
    window=item.measured_ns//100_000_000
    previous=(physical['session']['_dispatch_previous'] if physical['session'].get('_dispatch_window')==window
        else physical['runner_arrays']['_last_executed'])
    applied,_=apply_safety_limits_batch(np.array(request['requested_command'],np.float32)[None,None,:],previous,model.limits)
    failures+=compare(applied[0,0].tolist(),original_request['applied_command'],'final_limited_command_at_snapshot')
    return dict(status='PASS' if not failures else 'FAIL',controller=controller_name,frame=item.frame,
        tolerance=tolerances,failures=failures,source_selected_action=packet['source_selection']['action'],
        replay_selected_action=selected['action'],candidate_scores_applicability='not applicable to reactive final selector' if controller_name=='reactive_feedback' else 'compared all recorded candidate scores',
        selected=selected,correction=correction,request=request,applied=applied[0,0].tolist(),
        installed_plans=[asdict(p) for p in controller.plans],
        later_dispatch_with_new_observations_reproduced=False,
        later_dispatch_scope='Current-time dispatch and the complete planned post-limiter candidate tapes checked; future re-planning is tested only by source command-trace restoration.',
        qualification_attaches_to_packet=True,future_R4_restricted_to_learned_sources=False,
        cross_method_rankings=False,reference_regret=False)
