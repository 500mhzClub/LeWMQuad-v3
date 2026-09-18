"""Training-only, draw-weighted XY intercepts; yaw and contact remain unchanged."""
from collections import Counter
import numpy as np
import torch
from torch import nn
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA

HEADS=('direct_outcomes','rollout_outcomes')
OFFSETS=np.arange(1,9,dtype=np.int64)*100_000_000


def validate_arrays(arrays,head):
    if head not in HEADS:raise ValueError('explicit trained outcome head required')
    p=arrays[head];active=arrays['prediction_valid'];clocks=arrays['target_offsets_ns'];ids=arrays['indices']
    if (ids.ndim!=1 or ids.dtype!=np.int64 or len(ids)==0 or len(set(ids.tolist()))!=len(ids)
            or p.shape!=(len(ids),8,5) or p.dtype!=np.float32 or not np.isfinite(p).all()
            or active.shape!=(len(ids),8) or active.dtype!=bool
            or not active[:,0].all() or (active[:,1:]&~active[:,:-1]).any()
            or clocks.dtype!=np.int64 or clocks.shape!=active.shape
            or not np.array_equal(clocks,np.where(active,OFFSETS,0)) or np.any(p[~active]!=0.)):
        raise ValueError('complete finite float32 predictions, exact known clocks and zero padding required')


def fit_translation_bias(rows,arrays,schedule,*,head):
    validate_arrays(arrays,head)
    ids=arrays['indices'].tolist()
    expected=[i for source in ('family','switch') for i,r in enumerate(rows)
        if r['available'] and r['data_role']=='train' and r['source']==source]
    if ids!=expected:raise ValueError('all and only available training contexts in original source order required')
    batches=schedule['batches']
    if len(batches)!=1200 or any(len(b)!=6 for b in batches):raise ValueError('exact original 1200-by-six draw schedule required')
    counts=Counter(int(i) for batch in batches for i in batch)
    if set(counts)!=set(ids):raise ValueError('every scheduled context must be available training data')
    numerator=np.zeros((8,2),np.float64);denominator=np.zeros(8,np.float64);valid_counts=np.zeros(8,np.int64)
    for j,i in enumerate(ids):
        targets=rows[i]['targets'];valid=np.asarray([t['motion_valid'] for t in targets],bool)
        if len(targets)!=8 or not valid.any():raise ValueError('known training motion support required')
        weight=counts[i]/int(valid.sum())
        for h,t in enumerate(targets):
            if (bool(arrays['prediction_valid'][j,h])!=t['in_plan'] or arrays['target_offsets_ns'][j,h]!=t['offset_ns']):
                raise ValueError('training labels and forecast clocks differ')
            if not t['motion_valid']:continue
            motion=np.asarray(t['motion'],float)
            if not t['in_plan'] or motion.shape!=(3,) or not np.isfinite(motion).all() or t['contact']!=0.:
                raise ValueError('finite pre-contact training motion required')
            numerator[h]+=weight*(arrays[head][j,h,:2].astype(float)-motion[:2])
            denominator[h]+=weight;valid_counts[h]+=1
    if not np.all(denominator>0):raise ValueError('training support at every corrected horizon required')
    bias=numerator/denominator[:,None];applied=bias.astype(np.float32)
    if not np.isfinite(applied).all():raise ValueError('representable finite fitted intercepts required')
    return dict(schema='training_translation_bias.v1',head=head,estimator='draw_count_divided_by_window_motion_count_weighted_mean',
        training_examples=len(ids),training_draws=sum(counts.values()),motion_counts=valid_counts.tolist(),
        effective_horizon_weights=denominator.tolist(),residual_mean_xy_m=bias.tolist(),
        applied_bias_xy_m=applied.tolist(),target_offsets_ns=OFFSETS.tolist(),fitted_scalar_parameters=16,
        yaw_changed=False,contact_changed=False,probability_calibrated=False,native_data_used=False)


def bias_array(record):
    a=np.asarray(record['applied_bias_xy_m'],np.float32)
    if (record['schema']!='training_translation_bias.v1' or record['head'] not in HEADS
            or record['target_offsets_ns']!=OFFSETS.tolist() or a.shape!=(8,2) or not np.isfinite(a).all()
            or record['yaw_changed'] or record['contact_changed'] or record['native_data_used']):
        raise ValueError('exact finite training-only XY correction receipt required')
    return a


def correct_arrays(arrays,records):
    if not records or any(k!=r['head'] for k,r in records.items()):raise ValueError('explicit nonempty head corrections required')
    result={k:v.copy() for k,v in arrays.items()}
    for head,record in records.items():
        validate_arrays(arrays,head);bias=bias_array(record)
        result[head][...,:2]=np.where(arrays['prediction_valid'][...,None],arrays[head][...,:2]-bias,np.float32(0.))
    return result


class TrainingTranslationBiasModel(nn.Module):
    """Evaluation-only wrapper; inference accepts only original causal inputs."""
    def __init__(self,base,records):
        super().__init__()
        if not isinstance(base,ObservationHorizonRGBBodyJEPA) or base.training or not records:
            raise ValueError('admitted short-horizon base in evaluation mode required')
        if any(k not in HEADS or k!=r['head'] for k,r in records.items()):raise ValueError('explicit trained-head records required')
        self.base=base;self.corrected_heads=tuple(sorted(records))
        for head,record in records.items():self.register_buffer(head+'_xy_bias',torch.from_numpy(bias_array(record).copy()))
        self.eval()

    def train(self,mode=True):
        if mode:raise ValueError('training-only fitted correction is evaluation-only; no resume')
        return super().train(False)

    @torch.inference_mode()
    def forward(self,observation_history,known_action_blocks,known_action_valid):
        if self.training or self.base.training:raise ValueError('evaluation-only corrected model required')
        output=self.base(observation_history,known_action_blocks,known_action_valid)
        active=output['prediction_valid']
        for head in self.corrected_heads:
            original=output[head];corrected=original.clone();bias=getattr(self,head+'_xy_bias')
            if bias.dtype!=original.dtype or bias.device!=original.device:raise ValueError('unchanged correction dtype/device required')
            corrected[...,:2]=torch.where(active[...,None],original[...,:2]-bias,torch.zeros_like(original[...,:2]))
            output[head]=corrected
        return output
