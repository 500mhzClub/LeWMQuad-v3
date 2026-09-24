"""Draw-weighted training-only XY intercepts with explicit censored rows.

Available contexts with zero valid motion horizons remain accounted for. They
contribute no XY residual; their neural contact training is unchanged. Native
training labels are used, but no evaluator-only runtime data enters inference.
"""
from collections import Counter
import numpy as np
import torch
from torch import nn
from lewm.training_translation_bias_development import validate_arrays, HEADS, OFFSETS
from lewm.observation_horizon_rgb_body_jepa_development import ObservationHorizonRGBBodyJEPA


def fit_translation_bias(rows, arrays, schedule, *, head):
    validate_arrays(arrays, head); ids = arrays['indices'].tolist()
    expected = [i for source in ('family','switch') for i,r in enumerate(rows)
        if r['source']==source and r['data_role']=='train' and r['available']]
    if ids != expected: raise ValueError('all available training contexts in source order required')
    batches = schedule['batches']
    if len(batches)!=1200 or any(len(batch)!=6 for batch in batches):
        raise ValueError('exact1200-by-six training draw schedule required')
    if any(type(i) is not int for batch in batches for i in batch):
        raise ValueError('integer training indices required')
    counts = Counter(i for batch in batches for i in batch)
    if set(counts) != set(ids): raise ValueError('all and only available training contexts must be scheduled')
    numerator = np.zeros((8,2),np.float64); denominator = np.zeros(8,np.float64)
    valid_counts = np.zeros(8,np.int64); motionless = []; motionless_draws = 0
    for j,i in enumerate(ids):
        targets = rows[i]['targets']
        if len(targets)!=8: raise ValueError('all eight target slots required')
        valid = np.asarray([t['motion_valid'] for t in targets],bool)
        # Check every clock even for a context without valid motion targets.
        for h,target in enumerate(targets):
            if (bool(arrays['prediction_valid'][j,h]) != target['in_plan']
                    or arrays['target_offsets_ns'][j,h] != target['offset_ns']):
                raise ValueError('exact training forecast and target clocks required')
            if target['motion_valid']:
                motion = np.asarray(target['motion'],np.float64)
                if (not target['in_plan'] or not target['contact_valid'] or target['contact']!=0.
                        or motion.shape!=(3,) or not np.isfinite(motion).all()):
                    raise ValueError('finite pre-contact native training motion target required')
        if not valid.any():
            motionless.append(i); motionless_draws += counts[i]
            continue
        weight = counts[i]/int(valid.sum())
        for h,target in enumerate(targets):
            if not target['motion_valid']: continue
            motion = np.asarray(target['motion'],np.float64)
            numerator[h] += weight*(arrays[head][j,h,:2].astype(np.float64)-motion[:2])
            denominator[h] += weight; valid_counts[h] += 1
    if not np.all(denominator>0): raise ValueError('training motion support at every corrected horizon required')
    bias = numerator/denominator[:,None]; applied = bias.astype(np.float32)
    if not np.isfinite(applied).all(): raise ValueError('finite representable training intercepts required')
    return dict(schema='all_phase_training_translation_bias.v1',head=head,
        estimator='draw_count_divided_by_window_motion_count_weighted_mean',
        training_examples=len(ids),training_draws=sum(counts.values()),
        motionless_training_indices=motionless,motionless_training_examples=len(motionless),
        motionless_training_draws=motionless_draws,motion_contributing_examples=len(ids)-len(motionless),
        motion_counts=valid_counts.tolist(),effective_horizon_weights=denominator.tolist(),
        residual_mean_xy_m=bias.tolist(),applied_bias_xy_m=applied.tolist(),
        target_offsets_ns=OFFSETS.tolist(),fitted_scalar_parameters=16,
        yaw_changed=False,contact_changed=False,probability_calibrated=False,
        native_training_targets_used=True,native_artifacts_opened_by_estimator=False,
        transfer_targets_used=False,runtime_native_state_used=False)


def bias_array(record):
    bias = np.asarray(record['applied_bias_xy_m'],np.float32)
    if (record['schema']!='all_phase_training_translation_bias.v1' or record['head'] not in HEADS
            or record['target_offsets_ns']!=OFFSETS.tolist() or bias.shape!=(8,2) or not np.isfinite(bias).all()
            or record['yaw_changed'] is not False or record['contact_changed'] is not False
            or record['native_training_targets_used'] is not True
            or record['native_artifacts_opened_by_estimator'] is not False
            or record['transfer_targets_used'] is not False or record['runtime_native_state_used'] is not False):
        raise ValueError('exact finite training-only XY correction with explicit data scope required')
    return bias


def correct_arrays(arrays, records):
    if not records or any(head!=r['head'] for head,r in records.items()):
        raise ValueError('explicit nonempty trained-head corrections required')
    result = {k:v.copy() for k,v in arrays.items()}
    for head,record in records.items():
        validate_arrays(arrays,head); bias = bias_array(record)
        result[head][...,:2] = np.where(arrays['prediction_valid'][...,None],
            arrays[head][...,:2]-bias,np.float32(0.))
    return result


class AllPhaseTranslationBiasModel(nn.Module):
    def __init__(self, base, records):
        super().__init__()
        if not isinstance(base,ObservationHorizonRGBBodyJEPA) or base.training or not records:
            raise ValueError('admitted evaluation-only short-horizon model required')
        if any(head not in HEADS or head!=r['head'] for head,r in records.items()):
            raise ValueError('explicit trained-head corrections required')
        self.base = base; self.corrected_heads = tuple(sorted(records))
        for head,record in records.items():
            self.register_buffer(head+'_xy_bias',torch.from_numpy(bias_array(record).copy()))
        self.eval()

    def train(self, mode=True):
        if mode: raise ValueError('corrected model is evaluation-only; no resume')
        return super().train(False)

    @torch.inference_mode()
    def forward(self, observation_history, known_action_blocks, known_action_valid):
        if self.training or self.base.training: raise ValueError('evaluation-only corrected model required')
        output = self.base(observation_history,known_action_blocks,known_action_valid)
        for head in self.corrected_heads:
            raw = output[head]; bias = getattr(self,head+'_xy_bias')
            if bias.dtype!=raw.dtype or bias.device!=raw.device: raise ValueError('exact correction dtype/device required')
            corrected = raw.clone()
            corrected[...,:2] = torch.where(output['prediction_valid'][...,None],raw[...,:2]-bias,
                torch.zeros_like(raw[...,:2]))
            output[head] = corrected
        return output
