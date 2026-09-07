"""One causal local choice from fixed, audited three-seed development models.

No runtime dataset discovery, learned updates, simulator pose or task-map inputs.
This strict zero-latency ideal-sensor contract is not hardware qualification.
"""
import hashlib
import io
import json
from pathlib import Path
import time

import numpy as np
import torch

from lewm.causal_sensor_state import SensorContractError,_identity,_ns
from lewm.counterfactual_learning_data_development import prospective_plan
from lewm.counterfactual_maze_development import ACTIONS
from lewm.rgb_body_jepa_reference_development import RGBBodyJEPAReference
from lewm.rgb_body_tensor_interface_development import observation_tensors
from lewm.simulated_body_observation_development import validate_policy_packet

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'.generated/go2_rgb_body_learning_comparison_development_v1_attempt_001'
LAUNCH_SHA256='0cb16ea1bdec658aa086ac052810502f9d0392f6a332c7f1e03ae50baae8c9f3'
AUDIT_SHA256='ae29a92c2e9a5abc721a738a42b7f19443f294a8750b89bd959dd73e4a1081e1'
SEEDS=(2026091200,2026091201,2026091202)
CHECKPOINTS={
    'supervised_rollout':(
        'b3486866340121bdf6a8459e2ab00d12e0d5aaf436764813e12e7ffb42931a8d',
        '87452a4756126f4ec196831a169b2042f20eb619c429759d5a19cf999725145e',
        '602a302e7ff4e1961bb755d9b26c22ea31128c53f06c6827794d18bcebe613e8'),
    'jepa':(
        '97ec646e57c7939a369a1fad943561e699c74c73936fea3b31ce000bff16c0a5',
        '1ca9569337005fab1c696ca7891de69830d7159601f431bd59c83cfb180675a0',
        '98c61778d8c24a4e9347b96c98cd48ba500b4b8720b8a303730b1da6cdcf8e19'),
}


def _read_bound(path,expected):
    path=Path(path)
    if not path.is_absolute() or any(p in ('sealed','sealed_test.json') or p.startswith('sealed_') for p in path.parts):
        raise ValueError('explicit nonprotected path required')
    if path.resolve()!=path: raise ValueError('symlinked model/source material forbidden')
    content=path.read_bytes()
    if hashlib.sha256(content).hexdigest()!=expected: raise ValueError('model/source binding mismatch')
    return content


def rank_ensemble(member_predictions,intent_xy):
    values=np.asarray(member_predictions,dtype=np.float64)
    intent=np.asarray(intent_xy,dtype=np.float64)
    if values.shape!=(3,5,8,5) or not np.isfinite(values).all(): raise ValueError('three finite five-candidate rollout predictions required')
    if intent.shape!=(2,) or not np.isfinite(intent).all() or np.linalg.norm(intent)>.8000001:
        raise ValueError('local intent must be finite body-frame xy within 0.8 metres')
    mean=values.mean(axis=0)
    # Average probabilities, NOT logits. The ensemble is not claimed calibrated.
    probabilities=(1/(1+np.exp(-np.clip(values[...,4],-60,60)))).mean(axis=0)
    costs=10*probabilities[:,-1]+np.linalg.norm(mean[:,-1,:2]-intent,axis=-1)
    return {'selected_action_index':int(np.argmin(costs)),'candidate_costs':costs.tolist(),
        'mean_motion_sin_cos':mean[...,:4].tolist(),'mean_contact_probability':probabilities.tolist()}


class OnlineLocalChoice:
    """Explicit-episode one-shot interface; no hidden state estimator or replanning."""
    def __init__(self,condition,models,model_bindings):
        if condition not in ('supervised_rollout','jepa','always_stop'): raise ValueError('unknown fixed condition')
        expected=0 if condition=='always_stop' else 3
        if len(models)!=expected or len(model_bindings)!=expected: raise ValueError('fixed ensemble population required')
        self.condition=condition; self.models=tuple(models); self.bindings=tuple(model_bindings)
        for model in self.models: model.eval().requires_grad_(False)
        self._seen=set(); self._episode=None; self._selected=False

    @classmethod
    def from_completed_study(cls,condition):
        if condition=='always_stop': return cls(condition,(),())
        if condition not in CHECKPOINTS: raise ValueError('unknown fixed learned condition')
        launch=json.loads(_read_bound(STUDY/'launch.json',LAUNCH_SHA256))
        _read_bound(STUDY/'prediction_artifact_audit.json',AUDIT_SHA256)
        for name,expected in launch['source_sha256'].items():
            relative=Path(name)
            if relative.is_absolute() or '..' in relative.parts: raise ValueError('source binding path escape')
            _read_bound(ROOT/relative,expected)
        models=[]; bindings=[]
        for seed,expected in zip(SEEDS,CHECKPOINTS[condition],strict=True):
            path=STUDY/f'{seed}-{condition}'/'final.pt'
            content=_read_bound(path,expected)
            checkpoint=torch.load(io.BytesIO(content),map_location='cpu',weights_only=True)
            if (checkpoint['seed']!=seed or checkpoint['condition']!=condition or checkpoint['updates']!=300
                    or checkpoint['launch_sha256']!=LAUNCH_SHA256): raise ValueError('checkpoint identity mismatch')
            model=RGBBodyJEPAReference(); model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            if not all(torch.isfinite(p).all() for p in model.parameters()): raise ValueError('nonfinite checkpoint')
            models.append(model); bindings.append({'seed':seed,'sha256':expected})
        return cls(condition,models,bindings)

    def begin_episode(self,identity):
        identity=_identity(identity)
        if identity in self._seen: raise SensorContractError('episode/reset identity already used')
        self._seen.add(identity); self._episode=identity; self._selected=False

    @torch.no_grad()
    def select(self,packet,intent_xy,*,now_ns):
        start=time.perf_counter()
        validate_policy_packet(packet)
        now_ns=_ns(now_ns,'online clock'); state=packet['sensor_state']
        if self._episode is None or _identity(state['identity'])!=self._episode: raise SensorContractError('inactive episode/reset')
        if self._selected: raise SensorContractError('only one local choice is qualified per episode/reset')
        if state['decision_ns']!=now_ns or packet['image']['measured_ns']!=now_ns:
            raise SensorContractError('stale packet/image; fixed pilot requires current zero-latency observation')
        if now_ns%100_000_000: raise SensorContractError('choice must lie on the command clock')
        intent=np.asarray(intent_xy,dtype=np.float64)
        if intent.shape!=(2,) or not np.isfinite(intent).all() or np.linalg.norm(intent)>.8000001:
            raise ValueError('local intent must be finite body-frame xy within 0.8 metres')
        for role in ('sensed','control'):
            for row in state[role].values():
                if not np.asarray(row['valid']).all() or row['measured_ns'][-1]!=now_ns:
                    raise SensorContractError('complete fresh history required by the fitted reference')
        prior=state['control']['applied_command']['values'][-1]
        plans=np.stack([prospective_plan(command,prior) for _,command in ACTIONS])
        context=observation_tensors(packet)
        observation={k:v[None].expand(5,*v.shape) for k,v in context.items()}
        blocks=torch.from_numpy((plans/np.array([.3,1.,.5],dtype=np.float32)).reshape(5,8,5,3))
        inference_start=time.perf_counter()
        if self.condition=='always_stop':
            members=[]; ranking={'selected_action_index':0,'candidate_costs':None,
                'mean_motion_sin_cos':None,'mean_contact_probability':None}
        else:
            predictions=[model(observation,blocks)['rollout_outcomes'].cpu().numpy() for model in self.models]
            members=np.stack(predictions).tolist(); ranking=rank_ensemble(predictions,intent)
        inference_ms=(time.perf_counter()-inference_start)*1000
        selected=ranking['selected_action_index']; name,command=ACTIONS[selected]
        requested=np.vstack([np.tile(command,(40,1)),np.zeros((5,3))])
        applied=np.vstack([plans[selected],prospective_plan([0.,0.,0.],plans[selected,-1])[:5]])
        context_sha={k:hashlib.sha256(v.contiguous().numpy().tobytes()).hexdigest() for k,v in context.items()}
        result={'schema':'online_local_choice_development.v1','condition':self.condition,'episode_identity':list(self._episode),
            'decision_ns':now_ns,'intent_xy_body_start_m':intent.tolist(),'selected_action_name':name,**ranking,
            'command_period_ns':100_000_000,'branch_ticks':40,'release_ticks':5,
            'requested_command_tape':requested.tolist(),'expected_applied_command_tape':applied.tolist(),
            'candidate_applied_plans':plans.tolist(),'member_predictions':members,'model_bindings':list(self.bindings),
            'input_rgb_sha256':hashlib.sha256(packet['image']['rgb'].tobytes()).hexdigest(),
            'input_tensor_sha256':context_sha,'inference_ms':inference_ms,
            'adapter_ms':(time.perf_counter()-start)*1000,
            'scope':'one conditional open-loop choice, ideal zero-latency simulated sensors; no hardware or receding-horizon qualification'}
        self._selected=True
        return result
