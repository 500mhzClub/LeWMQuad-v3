from copy import deepcopy
import numpy as np
import pytest
from scripts import nominal_reentry_native_prefix_comparison_development as module


@pytest.fixture
def pair(tmp_path, monkeypatch):
    old = tmp_path/'old'; new = tmp_path/'new'; old.mkdir(); new.mkdir()
    for root in (old, new): np.savez(root/'physics_trace.npz', timestamp_s=np.arange(800)*.002)
    prefix = dict(frames=2, first_requested_command_difference=1,
        prior_requested_command=[0., 0., 0.], final_requested_command=[0., 0., .45])
    tapes = {old: [dict(requested_command=[0., 0., 0.]) for _ in range(2)],
        new: [dict(requested_command=[0., 0., 0.]), dict(requested_command=[0., 0., .45])]}
    def read(root, name):
        return tapes[root] if name == 'command_tape.json' else [dict(frame=i) for i in range(2)]
    monkeypatch.setattr(module, 'read_json', read)
    class Reader:
        def __init__(self, directory): self.directory = directory
        def packet(self, index): return dict(frame=index), {}, {}, index
    monkeypatch.setattr(module, 'IntentReturnRGBDReplay', Reader)
    monkeypatch.setattr(module, 'public_acquisition', lambda r: r)
    monkeypatch.setattr(module, 'packet', lambda directory, i, p, a, **k: dict(frame=i))
    rows = {root: [dict(decision=dict(evidence={'frame': i}, memory_receipt={}, mission_receipt={},
        auxiliary_floor_partition_receipt={}, new_selection={'prediction': [i]})) for i in range(2)] for root in (old, new)}
    monkeypatch.setattr(module, 'read_rows', lambda directory: iter(deepcopy(rows[directory])))
    return old, new, prefix, rows, tapes


def test_comparison_includes_intervention_observation_but_not_its_outcome(pair):
    old, new, prefix, rows, tapes = pair
    r = module.compare(old, new, prefix)
    assert r['common_prefix_frames'] == 2 and r['first_changed_command'] == 1
    assert r['physical_and_public_prefix_exact'] and not r['unexecuted_outcomes_inferred']


@pytest.mark.parametrize('mismatch', ['physics', 'forecast', 'missing_observation', 'command', 'public_packet'])
def test_prefix_mismatch_cannot_support_attribution(pair, mismatch, monkeypatch):
    old, new, prefix, rows, tapes = pair
    if mismatch == 'physics': np.savez(new/'physics_trace.npz', timestamp_s=np.arange(800)*.003)
    elif mismatch == 'forecast': rows[new][1]['decision']['new_selection']['prediction'] = [999]
    elif mismatch == 'missing_observation': rows[new].pop()
    elif mismatch == 'command': tapes[new][0]['requested_command'] = [.2, 0., 0.]
    else: monkeypatch.setattr(module, 'packet', lambda directory, *a, **k: str(directory))
    with pytest.raises(ValueError): module.compare(old, new, prefix)
