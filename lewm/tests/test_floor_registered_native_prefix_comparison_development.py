from copy import deepcopy
import numpy as np
import pytest
from scripts import floor_registered_native_prefix_comparison_development as module


@pytest.fixture
def pair(tmp_path, monkeypatch):
    old, new, bound = [tmp_path/n for n in ('old', 'new', 'prefix')]
    for root in (old, new, bound): root.mkdir()
    for root in (old, new): np.savez(root/'physics_trace.npz', timestamp_s=np.arange(850)*.002)
    report = dict(frames=2, first_requested_command_difference=1,
        prior_requested_command=[0., 0., 0.], final_requested_command=[0., 0., .45])
    tapes = {old: [dict(requested_command=[0., 0., 0.]) for _ in range(2)],
        new: [dict(requested_command=[0., 0., 0.]), dict(requested_command=[0., 0., .45])]}
    monkeypatch.setattr(module, 'read_json', lambda root, name:
        tapes[root] if name == 'command_tape.json' else [dict(frame=i) for i in range(2)])
    class Reader:
        def __init__(self, directory): self.directory = directory
        def packet(self, index): return dict(frame=index), {}, {}, index
    monkeypatch.setattr(module, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(module, 'public_acquisition', lambda r: r)
    monkeypatch.setattr(module, 'packet', lambda directory, i, p, a, **k: dict(frame=i))
    rows = {old: [dict(tick=i, decision=dict(evidence={'frame': i})) for i in range(2)]}
    rows[new] = [dict(tick=i, decision=dict(original_visual_evidence={'frame': i},
        corrected_pose={'position': i}, map_receipt={'height': .32})) for i in range(2)]
    rows[bound] = deepcopy(rows[new])
    monkeypatch.setattr(module, 'read_rows', lambda directory: iter(deepcopy(rows[directory])))
    return old, new, bound, report, rows, tapes


def test_exact_actual_prefix_includes_changed_decision_but_excludes_its_outcome(pair):
    old, new, bound, report, _, _ = pair
    values = np.arange(850)*.002; values[800:] += .5
    np.savez(new/'physics_trace.npz', timestamp_s=values)
    result = module.compare(old, new, bound, report)
    assert result['physical_and_public_prefix_exact']
    assert result['complete_candidate_decisions_match_prospective_prefix']
    assert not result['unexecuted_outcomes_inferred']


@pytest.mark.parametrize('fault', ['physics', 'map', 'pose', 'raw_witness', 'clock', 'missing', 'command', 'intervention', 'public'])
def test_any_prefix_mismatch_rejects_attribution(pair, fault, monkeypatch):
    old, new, bound, report, rows, tapes = pair
    if fault == 'physics': np.savez(new/'physics_trace.npz', timestamp_s=np.arange(850)*.003)
    elif fault == 'map': rows[new][1]['decision']['map_receipt']['height'] += .01
    elif fault == 'pose': rows[new][1]['decision']['corrected_pose']['position'] += .01
    elif fault == 'raw_witness': rows[old][1]['decision']['evidence']['frame'] = 99
    elif fault == 'clock': rows[new][1]['tick'] = 2
    elif fault == 'missing': rows[bound].pop()
    elif fault == 'command': tapes[new][0]['requested_command'][0] = .1
    elif fault == 'intervention': tapes[new][1]['requested_command'][2] = -.45
    else: monkeypatch.setattr(module, 'packet', lambda directory, *a, **k: str(directory))
    with pytest.raises(ValueError): module.compare(old, new, bound, report)
