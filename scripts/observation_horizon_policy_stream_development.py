"""Explicit past-only inference and private training-only future packets."""
from copy import deepcopy
from lewm.observation_horizon_view_development import ObservationHorizonView
from lewm.observation_horizon_sample_development import inference_inputs,materialize_training
from lewm.pulse_timed_dataset_development import stack_samples
from scripts.augmented_family_switch_stream_development import AugmentedFamilySwitchStream
from scripts.geometry_progress_family_policy_stream_development import _PolicyReader
from scripts.navigation_artifact_root_development import verify_artifacts


class ObservationHorizonPolicyStream:
    def __init__(self,original,rows):
        if not isinstance(original,AugmentedFamilySwitchStream):
            raise ValueError('both authenticated original policy streams required')
        self.original=original;self.view=ObservationHorizonView(original.view,rows);self.failed=False;self._cache={}

    def _one(self,index,*,training):
        row=self.view.rows[index]
        old=self.original.inference_batch([index],role=row['data_role'])
        def unbatch(value):return {k:unbatch(v) for k,v in value.items()} if isinstance(value,dict) else value[0]
        inputs=inference_inputs(unbatch(old),row)
        if not training:return inputs
        if row['data_role']!='train':raise ValueError('future packet reader is training-only')
        source=self.original.family if row['source']=='family' else self.original.switch
        indices=[t['future_observation_index'] for t in row['targets'] if t['future_image_valid']]
        names=[row['trial']+'/'+n for n in ('policy_observations.json','policy_histories.npz',
            *(f'rgb_{i:04d}.png' for i in sorted(set(indices))))]
        if not set(names)<=set(source.bindings):raise ValueError('all consumed future policy leaves must be bound')
        selected={n:source.bindings[n] for n in names};verify_artifacts(source.output,selected)
        reader=_PolicyReader(source.output/row['trial'],indices)
        sample=materialize_training(reader,row,inputs);verify_artifacts(source.output,selected);return sample

    def _batch(self,indices,role,*,training):
        if self.failed:raise ValueError('short-horizon stream failure latched')
        try:
            if (role not in ('train','geometry_transfer') or training and role!='train'
                    or not isinstance(indices,list) or not 1<=len(indices)<=16
                    or any(type(i) is not int or i not in self.view.indices(role) for i in indices)):
                raise ValueError('bounded available indices and exact materialization role required')
            samples=[]
            for i in indices:
                if training:
                    if i not in self._cache:
                        if len(self._cache)>=408:raise ValueError('fixed training-context cache bound exceeded')
                        self._cache[i]=self._one(i,training=True)
                    samples.append(self._cache[i])
                else:samples.append(self._one(i,training=False))
            return stack_samples(samples)
        except Exception:self.failed=True;raise

    def training_batch(self,indices):return self._batch(indices,'train',training=True)

    def inference_batch(self,indices,*,role):return self._batch(indices,role,training=False)
