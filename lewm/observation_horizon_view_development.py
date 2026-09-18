"""Original complete context/schedule accounting with separately bound targets."""
from copy import deepcopy
from lewm.augmented_family_switch_view_development import AugmentedFamilySwitchView
from lewm.observation_horizon_targets_development import verify_half_second_overlap


class ObservationHorizonView:
    def __init__(self,original,rows):
        if not isinstance(original,AugmentedFamilySwitchView) or len(rows)!=912:
            raise ValueError('complete original augmented population and all new target slots required')
        self.original=AugmentedFamilySwitchView(original.family,original.switch)
        for old,new in zip(self.original.rows,rows,strict=True):
            added={'observation_horizon_receipt','shared_half_second_native_target_exact'}
            if set(new)!=set(old)|added or any(new[k]!=old[k] for k in old if k!='targets'):
                raise ValueError('changed original context, role, availability or metadata')
            frame=3+old['offset_ticks'] if old['source']=='family' else 13
            receipt=new['observation_horizon_receipt']
            expected=dict(target_only=True,departure_tick=frame,departure_ns=1_500_000_000+100_000_000*frame,
                history_observation_indices=list(range(frame-3,frame+1)),target_cadence_ns=100_000_000,
                maximum_horizon_ns=800_000_000,available=old['available'],
                reason=None if old['available'] else old.get('reason'))
            if receipt!=expected or new['shared_half_second_native_target_exact']!=old['available']:
                raise ValueError('exact original departure and short-clock receipt required')
            if old['available']:
                if len(new['targets'])!=8:raise ValueError('eight target slots required')
                verify_half_second_overlap(receipt|dict(targets=new['targets']),old['targets'])
                known=min(8,40-old['offset_ticks']) if old['source']=='family' else 8
                for h,target in enumerate(new['targets'],1):
                    active=h<=known
                    if target['in_plan']!=active or target['offset_ns']!=(h*100_000_000 if active else 0):
                        raise ValueError('actual known short target offsets required')
                    if target['future_image_valid'] and (not target['motion_valid'] or target['future_observation_index']!=frame+h):
                        raise ValueError('contact-free actual future boundary required')
            elif new['targets'] is not None:raise ValueError('unavailable context cannot acquire targets')
        self.rows=deepcopy(rows)

    def indices(self,role,*,source=None):return self.original.indices(role,source=source)

    def schedule(self,*,updates,batch_size,seed):
        return self.original.schedule(updates=updates,batch_size=batch_size,seed=seed)
