"""Bound policy leaves with separate past-only inference and training targets."""
from copy import deepcopy
from pathlib import Path
from lewm.moving_action_switch_learning_sample_development import inference_inputs,materialize_training,validate_assignment
from lewm.moving_action_switch_learning_view_development import MovingActionSwitchView
from lewm.pulse_timed_dataset_development import stack_samples
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.navigation_artifact_root_development import BASE,validate_root,verify_artifacts
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

INPUT=BASE/'go2_moving_action_switch_family_v1_attempt_001'


def policy_leaves(report,*,include_future):
    validate_assignment(report)
    if type(include_future) is not bool or not report['outcome']['branch_available']:raise ValueError('explicit available policy scope required')
    if include_future and report['data_role']!='train':raise ValueError('future images are training targets only')
    indices={10,11,12,13}
    if include_future:
        for i,t in enumerate(report['targets']['targets']):
            if t['future_image_valid']:
                index=t['future_observation_index']
                if type(index) is not int or index!=13+5*(i+1) or index>53:raise ValueError('exact cumulative future frame required')
                indices.add(index)
    return ['policy_observations.json','policy_histories.npz']+[f'rgb_{i:04d}.png' for i in sorted(indices)]


class _PolicyReader:
    def __init__(self,directory,indices):self.directory=directory;self.indices=frozenset(indices)
    def packet(self,index):
        if type(index) is not int or index not in self.indices:raise ValueError('packet outside explicit policy scope')
        return (load_route_observation(self.directory,index),)


class MovingActionSwitchPolicyStream:
    def __init__(self,view,*,output,bindings,tensor_index):
        if not isinstance(view,MovingActionSwitchView) or validate_root(output)!=INPUT:raise ValueError('exact new branch view/root required')
        self.view=MovingActionSwitchView(view.reports);self.output=Path(output);self.bindings=deepcopy(bindings)
        if [r['trial'] for r in tensor_index]!=[r['trial'] for r in view.reports]:raise ValueError('complete exact tensor index required')
        self.index=deepcopy(tensor_index);self._cache={};self.failed=False

    def _one(self,index,*,include_future):
        row=self.view.reports[index];names=policy_leaves(row,include_future=include_future)
        paths=[row['trial']+'/'+n for n in names]
        if not set(paths)<=set(self.bindings):raise ValueError('all consumed policy leaves must be bound')
        selected={n:self.bindings[n] for n in paths};verify_artifacts(self.output,selected)
        reader=_PolicyReader(self.output/row['trial'],[int(n[4:8]) for n in names if n.startswith('rgb_')])
        sample=materialize_training(reader,row) if include_future else inference_inputs(reader,row)
        inputs=sample['inputs'] if include_future else sample;witness=self.index[index]
        if witness['materialized'] is not True:raise ValueError('input check did not materialize this branch')
        if ({k:fingerprint(v.numpy()) for k,v in inputs['observation_history'].items()}!=witness['history_sha256']
                or fingerprint(inputs['known_action_blocks'].numpy())!=witness['known_action_sha256']
                or fingerprint(inputs['known_action_valid'].numpy())!=witness['known_action_valid_sha256']):
            raise ValueError('policy tensors differ from authenticated input check')
        verify_artifacts(self.output,selected);return sample

    def _batch(self,indices,role,*,include_future):
        if self.failed:raise ValueError('stream failure latched; no retry')
        try:
            if (role not in ('train','geometry_transfer') or not isinstance(indices,list) or not 1<=len(indices)<=16
                    or include_future and role!='train'
                    or any(type(i) is not int or i not in self.view.indices(role) for i in indices)):
                raise ValueError('bounded available indices must match explicit role')
            samples=[]
            for i in indices:
                if include_future:
                    if i not in self._cache:
                        if len(self._cache)>=72:raise ValueError('bounded training-only cache exceeded')
                        self._cache[i]=self._one(i,include_future=True)
                    samples.append(self._cache[i])
                else:samples.append(self._one(i,include_future=False))
            return stack_samples(samples)
        except Exception:self.failed=True;raise

    def training_batch(self,indices):return self._batch(indices,'train',include_future=True)
    def inference_batch(self,indices,*,role):return self._batch(indices,role,include_future=False)
