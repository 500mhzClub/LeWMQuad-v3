"""Study routing and admitted tensor checks without fitting or new collection."""
from copy import deepcopy
import torch
from lewm.all_phase_training_view_development import AllPhaseTrainingView
from lewm.observation_horizon_plan_development import plan as original_plan, validate_plan
from scripts.all_phase_training_policy_stream_development import AllPhaseTrainingStream, plan
from scripts.observation_horizon_fit_inputs_development import CheckedObservationHorizonStream
from scripts.check_go2_all_phase_training_inputs_v1 import input_identity
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint


class CheckedAllPhaseTrainingStream(AllPhaseTrainingStream):
    def __init__(self, rows, bindings, tensor_index, **kwargs):
        super().__init__(rows, bindings, **kwargs)
        if (len(tensor_index) != len(rows) or any(
                w['index'] != i or w['sample_id'] != r['sample_id']
                or w['source'] != r['source'] or w['data_role'] != 'train'
                or w['materialized'] is not r['available']
                for i, (r, w) in enumerate(zip(rows, tensor_index, strict=True)))):
            raise ValueError('complete exact expanded input-check tensor index required')
        self.index = deepcopy(tensor_index)

    def materialize(self, index, *, training):
        try:
            value = super().materialize(index, training=training)
            witness = self.index[index]
            inputs = value['inputs'] if training else value
            identity = input_identity(inputs)
            if (witness['materialized'] is not True
                    or any(witness[k] != v for k, v in identity.items())
                    or witness['inference_and_training_inputs_exact'] is not True):
                raise ValueError('expanded input tensors differ from completed admission')
            scope = witness['training_access' if training else 'past_access']
            if self.last_access != scope:
                raise ValueError('reader scope differs from completed input admission')
            if training:
                targets = value['targets']
                future = {k: fingerprint(v.numpy()) for k, v in targets['future_observations'].items()}
                labels = {k: fingerprint(v.numpy()) for k, v in targets.items() if k != 'future_observations'}
                if future != witness['future_tensor_sha256'] or labels != witness['target_tensor_sha256']:
                    raise ValueError('training targets differ from completed input admission')
            return value
        except Exception:
            self.failed = True
            raise


class AllPhaseStudyStream:
    def __init__(self, training, original):
        if (not isinstance(training, CheckedAllPhaseTrainingStream)
                or not isinstance(original, CheckedObservationHorizonStream)
                or training.failed or original.failed):
            raise ValueError('nonfailed admitted training and original transfer streams required')
        self.view = AllPhaseTrainingView(original.view, training.rows)
        self.training = training; self.original = original; self.failed = False

    def _indices(self, indices, role):
        if self.failed or self.training.failed or self.original.failed:
            raise ValueError('study stream failure latched')
        if (role not in ('train', 'geometry_transfer') or not isinstance(indices, list)
                or not 1 <= len(indices) <= 16
                or any(type(i) is not int or i not in self.view.indices(role) for i in indices)):
            raise ValueError('bounded available indices from exactly one requested role required')
        return indices if role == 'train' else [self.view.transfer_index(i) for i in indices]

    def training_batch(self, indices):
        try:
            ids = self._indices(indices, 'train')
            return self.training.training_batch(ids)
        except Exception:
            self.failed = True
            raise

    def inference_batch(self, indices, *, role):
        try:
            ids = self._indices(indices, role)
            if role == 'train': return self.training.inference_batch(ids, role='train')
            return self.original.inference_batch(ids, role='geometry_transfer')
        except Exception:
            self.failed = True
            raise


def verified_plan(view, indices, inputs):
    if (not isinstance(view, AllPhaseTrainingView) or not isinstance(indices, list)
            or not 1 <= len(indices) <= 16
            or any(type(i) is not int or not 0 <= i < len(view.rows)
                or view.rows[i]['available'] is not True for i in indices)):
        raise ValueError('complete study view and bounded available indices required')
    pairs = []
    for i in indices:
        row = view.rows[i]
        if row['data_role'] == 'train': pairs.append(plan(row))
        else:
            pairs.append(original_plan(row['action'],
                offset_ticks=row['offset_ticks'] if row['source'] == 'family' else 0))
    blocks = torch.stack([p[0] for p in pairs]); valid = torch.stack([p[1] for p in pairs])
    if (not torch.equal(inputs['known_action_blocks'], blocks)
            or not torch.equal(inputs['known_action_valid'], valid)):
        raise ValueError('exact assigned training or original transfer command suffix required')
    return validate_plan(blocks, valid, len(indices))
