"""Dispatch homogeneous batches through unchanged source-specific policy paths."""
from lewm.augmented_family_switch_view_development import AugmentedFamilySwitchView
from scripts.geometry_progress_family_policy_stream_development import FamilyPolicyStream
from scripts.moving_action_switch_policy_stream_development import MovingActionSwitchPolicyStream


class AugmentedFamilySwitchStream:
    def __init__(self,family,switch):
        if not isinstance(family,FamilyPolicyStream) or not isinstance(switch,MovingActionSwitchPolicyStream):
            raise ValueError('authenticated private streams for both fixed sources required')
        self.family=family;self.switch=switch;self.view=AugmentedFamilySwitchView(family.view,switch.view);self.failed=False

    def _batch(self,indices,role,*,training):
        if self.failed:raise ValueError('augmented stream failure latched')
        try:
            if (not isinstance(indices,list) or not 1<=len(indices)<=16 or role not in ('train','geometry_transfer')
                    or training and role!='train' or any(type(i) is not int or i not in self.view.indices(role) for i in indices)):
                raise ValueError('bounded available indices and explicit source role required')
            rows=[self.view.rows[i] for i in indices]
            if len({r['source'] for r in rows})!=1:raise ValueError('source-homogeneous batch required')
            stream=self.family if rows[0]['source']=='family' else self.switch
            ids=[r['local_index'] for r in rows]
            return stream.training_batch(ids) if training else stream.inference_batch(ids,role=role)
        except Exception:self.failed=True;raise

    def training_batch(self,indices):return self._batch(indices,'train',training=True)
    def inference_batch(self,indices,*,role):return self._batch(indices,role,training=False)
