"""Policy-only fitting/inference materialization from caller-authenticated receipts.

The caller checks complete dataset/derivation gates and all bytes before and
after a fit. Each cache admission additionally checks its exact consumed leaves
before and after loading. Cached tensors are private; batches are fresh stacks.
Inference has a separate past-only path and never opens future images or native
state. No fitting, role reassignment, checkpoint access or output writes here.
"""
from copy import deepcopy
from pathlib import Path
from lewm.geometry_progress_family_learning_view_development import FamilyWindowView
from lewm.geometry_progress_family_causal_windows_development import materialize,remaining_candidate
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.navigation_artifact_root_development import BASE,validate_root,verify_artifacts
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

INPUT=BASE/'go2_geometry_progress_family_v1_attempt_001'


def policy_leaves(window,*,include_future):
    if type(include_future) is not bool or window['available'] is not True:raise ValueError('explicit available policy scope required')
    frame=3+window['offset_ticks'];past=window['history_observation_indices']
    if past!=list(range(frame-3,frame+1)) or not 3<=frame<=38:
        raise ValueError('exact four-packet causal history required')
    indices=set(past)
    if include_future:
        for i,t in enumerate(window['targets']):
            if t['future_image_valid']:
                index=t['future_observation_index']
                if type(index) is not int or index!=frame+5*(i+1) or index>43:
                    raise ValueError('actual prospective future image index required')
                indices.add(index)
    return ['policy_observations.json','policy_histories.npz']+[f'rgb_{i:04d}.png' for i in sorted(indices)]


class _PolicyReader:
    def __init__(self,directory,indices):self.directory=directory;self.indices=frozenset(indices)
    def packet(self,index):
        if type(index) is not int or index not in self.indices:raise ValueError('packet outside explicit materialization scope')
        return (load_route_observation(self.directory,index),)


class FamilyPolicyStream:
    def __init__(self,view,*,output,bindings,tensor_index):
        if not isinstance(view,FamilyWindowView) or validate_root(output)!=INPUT:
            raise ValueError('exact prospective family view/root required')
        self.view=FamilyWindowView(view.windows);self.output=Path(output);self.bindings=deepcopy(bindings)
        if [i['window_id'] for i in tensor_index]!=[w['window_id'] for w in view.windows]:
            raise ValueError('complete exact tensor index required')
        self.index=deepcopy(tensor_index);self._cache={};self.failed=False

    def _one(self,index,*,include_future):
        row=self.view.windows[index];names=policy_leaves(row,include_future=include_future)
        paths=[row['trial']+'/'+n for n in names]
        if not set(paths)<=set(self.bindings):raise ValueError('every consumed policy leaf must be bound')
        selected={n:self.bindings[n] for n in paths};verify_artifacts(self.output,selected)
        indices=[int(n[4:8]) for n in names if n.startswith('rgb_')]
        reader=_PolicyReader(self.output/row['trial'],indices)
        if include_future:sample=materialize(reader,row);inputs=sample['inputs']
        else:
            history=causal_history_tensors([reader.packet(i)[0] for i in row['history_observation_indices']],row['decision_ns'])
            blocks,valid=remaining_candidate(row['action'],row['offset_ticks'])
            inputs=dict(observation_history=history,known_action_blocks=blocks,known_action_valid=valid);sample=inputs
        witness=self.index[index]
        if witness['materialized'] is not True:raise ValueError('original derivation did not materialize this context')
        if ({k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()}!=witness['history_sha256']
                or fingerprint(inputs['known_action_blocks'].numpy())!=witness['known_action_sha256']
                or fingerprint(inputs['known_action_valid'].numpy())!=witness['known_action_valid_sha256']):
            raise ValueError('policy-only tensors differ from authenticated derivation')
        verify_artifacts(self.output,selected);return sample

    def _batch(self,indices,role,*,include_future):
        if self.failed:raise ValueError('stream failure latched; no retry')
        try:
            if (role not in ('train','geometry_transfer') or not isinstance(indices,list) or not 1<=len(indices)<=16
                    or include_future and role!='train'
                    or any(type(i) is not int or not 0<=i<len(self.view.windows) or not self.view.windows[i]['available']
                        or self.view.windows[i]['data_role']!=role for i in indices)):
                raise ValueError('bounded available indices must match explicit materialization role')
            samples=[]
            for i in indices:
                if include_future:
                    if i not in self._cache:
                        if len(self._cache)>=384:raise ValueError('bounded training-only cache exceeded')
                        self._cache[i]=self._one(i,include_future=True)
                    samples.append(self._cache[i])
                else:samples.append(self._one(i,include_future=False))
            return stack_samples(samples)
        except Exception:self.failed=True;raise

    def training_batch(self,indices):return self._batch(indices,'train',include_future=True)
    def inference_batch(self,indices,*,role):return self._batch(indices,role,include_future=False)
