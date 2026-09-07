"""Bounded policy-only materialization from authenticated study metadata.

No directory discovery, fitting, persistent image cache or raw-world-state
inference. Verify every consumed policy artifact before and after each sample.
Full raw provenance is established separately by the terminal study loader.
"""
from copy import deepcopy
from pathlib import Path

from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.independent_pulse_evaluation_development import IndependentPulseEvaluation
from lewm.pulse_timed_dataset_development import ROLES, stack_samples
from lewm.pulse_timed_rgb_body_jepa_development import pulse_brake_plan
from lewm.route_rgb_dataset_development import load_route_observation
from scripts.independent_rgb_body_batch_development import BATCHES, output_root
from scripts.independent_rgb_body_study_data_development import StudyData, AUDIT, require
from scripts.navigation_artifact_root_development import verify_artifacts

MAX_BATCH = 16


def policy_artifacts(window, *, include_future):
    """Exact consumed leaves; no depth, native state or future-command tape."""
    require(type(include_future) is bool and window['history_ready'] is True,
        'explicit materialization mode and complete causal history required')
    past = window['history_observation_indices']
    require(len(past) == 4 and all(type(i) is int and 0 <= i < 34 for i in past),
        'four actual bounded history indices required')
    require(past == [5, 6, 7, 8] and window['decision_ns'] == 2_300_000_000,
        'exact fixed departure history required before reading any RGB')
    indices = set(past)
    if include_future:
        for row in window['targets']:
            if row['future_valid']:
                i = row['future_observation_index']
                require(type(i) is int and 0 <= i < 34, 'actual bounded future frame required')
                indices.add(i)
    return ['policy_observations.json', 'policy_histories.npz'] + [f'rgb_{i:04d}.png' for i in sorted(indices)]


class _PolicyReader:
    def __init__(self, directory, indices):
        self.directory = directory; self.indices = frozenset(indices)

    def packet(self, index):
        require(type(index) is int and index in self.indices, 'packet outside bound sample frame set')
        # The existing dataset interface requests only tuple element zero. No
        # depth/fast/shadow reader is initialized just to discard its output.
        return (load_route_observation(self.directory, index),)


def materialize_policy_sample(dataset, index, output, bindings, *, include_future):
    """One sample from caller-authenticated metadata; not an eligibility grant."""
    require(type(index) is int and 0 <= index < len(dataset), 'exact dataset sample index required')
    window = dataset.windows[index]; c = window['condition']
    names = policy_artifacts(window, include_future=include_future)
    required = {c + '/' + n for n in names}
    require(required <= set(bindings), 'every consumed policy artifact must have an existing binding')
    selected = {n: bindings[n] for n in sorted(required)}
    verify_artifacts(output, selected)
    indices = [int(n[4:8]) for n in names if n.startswith('rgb_')]
    reader = _PolicyReader(Path(output) / c, indices)
    if include_future:
        result = dataset.sample(index, {c: reader})
    else:
        # Do not call dataset.sample here: inference never materializes future
        # images, native targets or masks derived from outcomes.
        history = causal_history_tensors([reader.packet(i)[0] for i in window['history_observation_indices']],
            window['decision_ns'])
        blocks, valid = pulse_brake_plan(tuple(window['command']), window['pulse_ticks'])
        result = dict(observation_history=history, known_action_blocks=blocks, known_action_valid=valid)
    verify_artifacts(output, selected)
    return result


class AuditedStudyStream:
    """Private metadata snapshot; keep at most one bounded tensor batch alive."""
    def __init__(self, study):
        require(isinstance(study, StudyData) and set(study.batches) == set(BATCHES)
            and set(study.receipts) == set(BATCHES), 'complete receipt-loaded study required')
        self.evaluation = IndependentPulseEvaluation(study.evaluation.inventory, study.evaluation.dataset)
        self._batches = deepcopy(study.batches); self._receipts = deepcopy(study.receipts)
        self._membership = {}
        for batch in BATCHES:
            row = self._batches[batch]; root = output_root(batch)
            require(row['source_and_artifact_bindings_verified'] is True and row['output_root'] == str(root)
                and row['receipt'] == self._receipts[batch], 'authenticated fresh batch identity required')
            require(set(self._receipts[batch]) == {'launch.json', AUDIT}, 'exact terminal receipt required')
            verify_artifacts(root, self._receipts[batch])
            for c in self.evaluation.inventory.episode_ids(batch):
                self._membership[c] = batch
        self.failed = False

    def _batch(self, indices, role, *, include_future):
        require(not self.failed, 'stream failure latched; no implicit retry')
        try:
            require(role in ROLES and isinstance(indices, list) and 1 <= len(indices) <= MAX_BATCH,
                'explicit role and bounded list of sample indices required')
            dataset = self.evaluation.dataset
            require(all(type(i) is int and 0 <= i < len(dataset) for i in indices), 'actual integer dataset indices required')
            require(all(dataset.windows[i]['history_ready'] and
                dataset.episode_roles[dataset.windows[i]['condition']]['role'] == role for i in indices),
                'every sample must belong to the requested role with complete history')
            samples = []
            for i in indices:
                c = dataset.windows[i]['condition']; batch = self._membership[c]; root = output_root(batch)
                verify_artifacts(root, self._receipts[batch])
                samples.append(materialize_policy_sample(dataset, i, root,
                    self._batches[batch]['artifact_sha256'], include_future=include_future))
                verify_artifacts(root, self._receipts[batch])
            return stack_samples(samples)
        except Exception:
            self.failed = True
            raise

    def training_batch(self, indices):
        return self._batch(indices, 'train', include_future=True)

    def inference_batch(self, indices, *, role):
        return self._batch(indices, role, include_future=False)
