from copy import deepcopy
import numpy as np
import pytest
from scripts import confirmed_floor_native_prefix_comparison_development as module


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
    rows = {root: [dict(decision=dict(evidence={'frame': i}, memory_receipt={'auxiliary_receipt': {'raw_count': 100}}, mission_receipt={},
        auxiliary_floor_partition_receipt={}, causal_residual_receipt={}, observed_goal_distance_m=3., new_selection={'prediction': [i]})) for i in range(2)] for root in (old, new)}
    for row in rows[new]:
        row['decision']['current_primary_floor_confirmation_enabled'] = True
        row['decision']['memory_receipt']['auxiliary_receipt']['current_primary_floor_confirmation'] = {'added': 10}
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


@pytest.mark.parametrize('mismatch', ['raw_map', 'old_partition', 'primary', 'witness', 'unknown', 'nonfoot', 'nominal'])
def test_floor_confirmation_cannot_hide_other_constraint_changes(pair, mismatch):
    old, new, prefix, rows, tapes = pair
    a, b = rows[old][1]['decision'], rows[new][1]['decision']
    surface = dict(shapes=[{'id': 'primary', 'count': 1}], primary_possible_intersection=True,
        auxiliary_shapes=[{'id': 'FL_foot', 'count': 1}])
    a['new_selection']['surface_checks'] = [deepcopy(surface)]
    revised = deepcopy(surface) | dict(original_auxiliary_floor_contact_check=deepcopy(surface),
        non_foot_contacts_exempted=False, non_floor_or_unknown_contacts_exempted=False)
    b['new_selection']['surface_checks'] = [revised]
    module.compare(old, new, prefix)
    if mismatch == 'raw_map': b['memory_receipt']['auxiliary_receipt']['raw_count'] = 99
    elif mismatch == 'old_partition': b['auxiliary_floor_partition_receipt'] = {'changed': True}
    elif mismatch == 'primary': revised['primary_possible_intersection'] = False
    elif mismatch == 'witness': revised['original_auxiliary_floor_contact_check']['auxiliary_shapes'] = []
    elif mismatch == 'unknown': revised['non_floor_or_unknown_contacts_exempted'] = True
    elif mismatch == 'nonfoot': revised['non_foot_contacts_exempted'] = True
    else: b['new_selection']['nominal_path_checks'] = ['changed']
    with pytest.raises(ValueError): module.compare(old, new, prefix)
