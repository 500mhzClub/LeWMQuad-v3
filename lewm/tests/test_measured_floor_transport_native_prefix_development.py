from copy import deepcopy
import numpy as np
import pytest
from lewm.tests.test_measured_floor_transport_native_development import prefix_report, candidate
from lewm.measured_floor_transport_prefix_development import normalize_labels
from scripts import measured_floor_transport_native_prefix_development as comparison


@pytest.mark.parametrize('fault', [None, 'physical_boundary', 'after_boundary', 'short_physics', 'prior_command',
    'changed_command', 'public', 'decision', 'raw_visual', 'short_stream'])
def test_all_prior_physics_and_intervention_observation_are_required(tmp_path, monkeypatch, fault):
    prior, current, bound = [tmp_path/n for n in ('prior','current','bound')]
    for p in (prior, current, bound): p.mkdir()
    frames, changed, count = 1905, 1904, 95950
    arrays = {prior:np.zeros((96000,3)), current:np.zeros((count-1 if fault=='short_physics' else 96000,3))}
    if fault == 'physical_boundary': arrays[current][count-1,0] = 1.
    if fault == 'after_boundary': arrays[current][count:,0] = 1.
    for p,a in arrays.items(): np.savez(p/'physics_trace.npz', base_pose_world=a)
    new = [dict(tick=i, decision=candidate(i), preintervention_complete_decision_exact=i<changed) for i in range(frames)]
    old = [dict(tick=i, decision=normalize_labels(new[i]['decision'])) for i in range(frames)]
    old[-1]['decision'].update(requested_command=[0.,0.,0.], terminal='SENSOR_OR_MODEL_FAILURE')
    bound_rows = deepcopy(new); rows = {prior:old, current:new, bound:bound_rows}
    tapes = {p:[dict(requested_command=deepcopy(r['decision']['requested_command'])) for r in rows[p]] for p in (prior,current)}
    if fault == 'prior_command': tapes[current][changed-1]['requested_command'] = [.2,0.,0.]
    if fault == 'changed_command': tapes[current][changed]['requested_command'] = [.2,0.,0.]
    if fault == 'decision': new[changed]['decision']['extra_field'] = True
    if fault == 'raw_visual': old[changed]['decision']['original_visual_evidence'] = {'different':True}
    if fault == 'short_stream': new.pop()
    monkeypatch.setattr(comparison, 'read_rows', lambda p:iter(rows[p]))
    monkeypatch.setattr(comparison, 'read_json', lambda p,n:tapes[p] if n=='command_tape.json' else [{} for _ in range(frames)])
    class Reader:
        def __init__(self, directory): self.directory = directory
        def packet(self, i):
            return dict(frame=i, changed=fault=='public' and self.directory==current and i==changed), {}, {}, i
    monkeypatch.setattr(comparison, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(comparison, 'public_acquisition', lambda a:a)
    monkeypatch.setattr(comparison, 'packet', lambda *args,**kwargs:({},{}))
    if fault in (None, 'after_boundary'):
        result = comparison.compare(prior, current, bound, prefix_report())
        assert result['physical_prefix_samples'] == count and result['common_prefix_frames'] == frames
        assert result['complete_candidate_decisions_match_prospective_prefix']
        assert result['candidate_intervention_command'] != result['original_intervention_command']
    else:
        with pytest.raises(ValueError): comparison.compare(prior, current, bound, prefix_report())
