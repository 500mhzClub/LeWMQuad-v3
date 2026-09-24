"""Repeated half-second directional choices from frozen temporal model ensembles."""
import copy
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.causal_relative_orientation_development import CausalRelativeOrientation
from lewm.causal_subtrajectory_learning_development import remaining_plan
from lewm.counterfactual_maze_development import ACTIONS
from lewm.online_rgb_history_development import OnlineRGBHistory
from lewm.temporal_rgb_body_jepa_development import TemporalRGBBodyJEPA

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'.generated/go2_temporal_rgb_body_learning_comparison_development_v1_attempt_001'
LAUNCH_SHA='a4e63ced7df5b8fef20b449037577ea9e473a0d40e5eccb44670e28f7dfc4055'
AUDIT_SHA='9f65ca24c3c9ba4197ed7e3a5e6ef633226381c2030202c098fc368cee1f873d'
SEEDS=(2026091700,2026091701,2026091702)
METHODS={'always_stop':(None,None),'direct_direct':('direct','direct'),
    'supervised_direct':('supervised_rollout','direct'),'supervised_rollout':('supervised_rollout','rollout'),
    'jepa_direct':('jepa','direct'),'jepa_rollout':('jepa','rollout')}


def _read_bound(path,expected):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts or any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in path.parts):
        raise ValueError('explicit nonprotected bound path required')
    if path.resolve()!=path: raise ValueError('symlinked evidence forbidden')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=expected: raise ValueError('source/model/evidence binding changed')
    return content


def rank_half_second_ensemble(member_predictions,direction_current_body):
    values=np.asarray(member_predictions,dtype=float); direction=np.asarray(direction_current_body,dtype=float)
    if values.shape!=(3,5,5) or not np.isfinite(values).all(): raise ValueError('three finite five-candidate half-second predictions required')
    if direction.shape!=(2,) or not np.isfinite(direction).all() or not 0<np.linalg.norm(direction)<=.8000001:
        raise ValueError('finite transported directional cue within .8m required')
    motion=values[...,:4].mean(0)
    probabilities=(1/(1+np.exp(-np.clip(values[...,4],-60,60)))).mean(0)
    costs=10*probabilities+np.linalg.norm(motion[:,:2]-direction,axis=-1)
    return {'selected_action_index':int(np.argmin(costs)),'candidate_costs':costs.tolist(),
        'mean_motion_sin_cos':motion.tolist(),'mean_contact_probability':probabilities.tolist()}


class OnlineTemporalChoice:
    """Explicit observation updates, direction initialization, then .5-s choices.

    A fault is latched for the episode. The caller must explicitly send a stop;
    stale state or exceptions are never treated as permission to retain motion.
    """
    def __init__(self,method,models,model_bindings):
        if method not in METHODS: raise ValueError('unknown temporal method')
        expected=0 if method=='always_stop' else 3
        if len(models)!=expected or len(model_bindings)!=expected: raise ValueError('fixed ensemble size')
        self.method=method; self.condition,self.head=METHODS[method]
        self.models=tuple(models); self.bindings=copy.deepcopy(list(model_bindings))
        for model in self.models: model.eval().requires_grad_(False)
        self.history=OnlineRGBHistory(); self.orientation=None; self._episode=None
        self._packet=None; self._clock=None; self._image_bindings=[]; self._direction=None
        self._start_ns=None; self._last_select=None; self._decisions=0; self._fault=False

    @classmethod
    def from_completed_study(cls,method):
        if method not in METHODS: raise ValueError('unknown temporal method')
        launch=json.loads(_read_bound(STUDY/'launch.json',LAUNCH_SHA))
        audit=json.loads(_read_bound(STUDY/'raw_artifact_audit.json',AUDIT_SHA))
        if audit['status']!='PASS' or not audit['full_study'] or audit['audited_models']!=9:
            raise ValueError('full nine-model audit required')
        _read_bound(STUDY/'result.json',audit['study_result_sha256'])
        for name,sha in launch['source_sha256'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('source path escape')
            _read_bound(ROOT/relative,sha)
        condition,_=METHODS[method]
        if condition is None: return cls(method,(),())
        models=[]; bindings=[]
        for seed in SEEDS:
            matches=[m for m in audit['models'] if m['seed']==seed and m['condition']==condition]
            if len(matches)!=1 or matches[0]['status']!='PASS': raise ValueError('missing audited ensemble member')
            sha=matches[0]['checkpoint_sha256']
            payload=_read_bound(STUDY/f'{seed}-{condition}'/'final.pt',sha)
            checkpoint=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
            if (checkpoint['seed']!=seed or checkpoint['condition']!=condition or checkpoint['updates']!=1200
                    or checkpoint['launch_sha256']!=LAUNCH_SHA or checkpoint['schedule_sha256']!=audit['schedule_sha256']):
                raise ValueError('checkpoint identity')
            model=TemporalRGBBodyJEPA(); model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError('nonfinite fitted model')
            models.append(model); bindings.append({'seed':seed,'condition':condition,'checkpoint_sha256':sha})
        return cls(method,models,bindings)

    def begin_episode(self,identity):
        identity=_identity(identity); self.history.begin_episode(identity)
        self._episode=identity; self.orientation=None; self._packet=None; self._clock=None
        self._image_bindings=[]; self._direction=None; self._start_ns=None; self._last_select=None
        self._decisions=0; self._fault=False

    def observe(self,packet,*,now_ns):
        if self._fault: raise SensorContractError('episode has a latched input/selection failure')
        try:
            status=self.history.push(packet,now_ns=now_ns)
            if self.orientation is not None: self.orientation.step(packet,now_ns=now_ns)
            self._packet=copy.deepcopy(packet); self._clock=_ns(now_ns,'temporal observation clock')
            if status['reset_for_gap']: self._image_bindings=[]
            self._image_bindings.append({'measured_ns':self._clock,
                'rgb_sha256':hashlib.sha256(np.asarray(packet['image']['rgb']).tobytes()).hexdigest()})
            self._image_bindings=self._image_bindings[-4:]
            return status
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self._fault=True; raise SensorContractError('temporal observation failed; caller must stop') from error

    def begin_control(self,direction_initial_body,*,now_ns):
        if self._fault or self._direction is not None: raise SensorContractError('fresh valid control start required')
        try:
            now_ns=_ns(now_ns,'temporal control start clock'); self.history.tensors(now_ns=now_ns)
            direction=np.asarray(direction_initial_body,dtype=float)
            if direction.shape!=(2,) or not np.isfinite(direction).all() or abs(np.linalg.norm(direction)-.8)>1e-8:
                raise ValueError('fixed .8-m initial-frame directional cue required')
            tracker=CausalRelativeOrientation(); state=tracker.begin(self._packet,now_ns=now_ns)
            self.orientation=tracker; self._direction=direction.copy(); self._start_ns=now_ns
            return state
        except (ValueError,TypeError,KeyError,IndexError) as error:
            self._fault=True; raise SensorContractError('temporal control start failed; caller must stop') from error

    @torch.no_grad()
    def select(self,*,now_ns):
        started=time.perf_counter()
        if self._fault or self._direction is None: raise SensorContractError('active fault-free directional control required')
        try:
            now_ns=_ns(now_ns,'temporal decision clock')
            expected=self._start_ns if self._last_select is None else self._last_select+500_000_000
            if now_ns!=expected or self._clock!=now_ns: raise SensorContractError('fresh fixed .5-second decision cadence required')
            context=self.history.tensors(now_ns=now_ns)
            direction=self.orientation.transport_xy(self._direction,now_ns=now_ns)
            prior=self._packet['sensor_state']['control']['applied_command']['values'][-1]
            plans=[remaining_plan(command,prior,5) for _,command in ACTIONS]
            blocks=torch.stack([p['known_action_blocks'] for p in plans]); valid=torch.stack([p['known_action_valid'] for p in plans])
            observation={k:v[None].expand(5,*v.shape) for k,v in context.items()}
            inference_start=time.perf_counter()
            if self.method=='always_stop':
                members=[]; ranked={'selected_action_index':0,'candidate_costs':None,'mean_motion_sin_cos':None,'mean_contact_probability':None}
            else:
                members=[]
                for model in self.models:
                    if self.head=='direct':
                        z,_=model.encode_history(observation); output=model.direct(z,blocks)
                    else: output=model(observation,blocks,valid)['rollout_outcomes']
                    members.append(output[:,0].cpu().numpy())
                ranked=rank_half_second_ensemble(members,direction)
            inference_ms=(time.perf_counter()-inference_start)*1000
            selected=ranked['selected_action_index']; name,command=ACTIONS[selected]
            applied=blocks[:,0].numpy()*[.3,1.,.5]
            result={'schema':'online_temporal_direction_choice_development.v1','method':self.method,
                'training_condition':self.condition,'inference_head':self.head,'episode_identity':list(self._episode),
                'decision_ns':now_ns,'decision_index':self._decisions,'initial_direction_xy':self._direction.tolist(),
                'direction_current_body_xy':direction.tolist(),'orientation':self.orientation.snapshot(now_ns=now_ns),
                'horizon_ns':500_000_000,'command_period_ns':100_000_000,'branch_ticks':5,
                'selected_action_name':name,**ranked,'requested_command_tape':np.tile(command,(5,1)).tolist(),
                'expected_applied_command_tape':applied[selected].tolist(),'candidate_applied_plans':applied.tolist(),
                'member_predictions':np.asarray(members).tolist(),'model_bindings':copy.deepcopy(self.bindings),
                'input_images':copy.deepcopy(self._image_bindings),
                'input_tensor_sha256':{k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in context.items()},
                'inference_ms':inference_ms,'adapter_ms':(time.perf_counter()-started)*1000,
                'scope':'half-second directional choice; no translation estimate, point-goal arrival, clearance or hardware guarantee'}
            self._last_select=now_ns; self._decisions+=1
            return result
        except (ValueError,TypeError,KeyError,IndexError,RuntimeError) as error:
            self._fault=True; raise SensorContractError('temporal selection failed; caller must stop') from error
