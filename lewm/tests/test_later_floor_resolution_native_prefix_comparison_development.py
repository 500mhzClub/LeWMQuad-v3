from copy import deepcopy
import numpy as np
import pytest
from scripts import later_floor_resolution_native_prefix_comparison_development as module


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
    rows = {old: [], new: []}
    for i in range(2):
        d = dict(failure=None, terminal=None, evidence=dict(frame=i, registered=True),
            original_visual_evidence=dict(frame=i, raw=True), memory_receipt=dict(floor=i),
            mission_receipt=dict(phase='outbound'), requested_command=[0., 0., 0.],
            new_selection=dict(prediction=[[[.1, .2]]], surface_checks=[dict(possible_intersection=True)],
                nominal_path_checks=[True]*8))
        rows[old].append(dict(tick=i, decision=deepcopy(d)))
        d['requested_command'] = tapes[new][i]['requested_command'].copy()
        d['new_selection']['surface_checks'] = [dict(possible_intersection=False,
            original_contact_check_before_later_floor_resolution=dict(possible_intersection=True))]
        rows[new].append(dict(tick=i, decision=d))
    rows[bound] = deepcopy(rows[new])
    monkeypatch.setattr(module, 'read_rows', lambda directory: iter(deepcopy(rows[directory])))
    return old, new, bound, report, rows, tapes


def test_exact_prefix_excludes_the_changed_commands_outcome(pair):
    old, new, bound, report, _, _ = pair
    values = np.arange(850)*.002; values[800:] += .5
    np.savez(new/'physics_trace.npz', timestamp_s=values)
    result = module.compare(old, new, bound, report)
    assert result['physical_and_public_prefix_exact']
    assert result['complete_candidate_decisions_match_prospective_prefix']
    assert result['original_pose_map_mission_forecasts_contact_queries_exact']
    assert not result['unexecuted_outcomes_inferred']


@pytest.mark.parametrize('fault', ['physics', 'map', 'pose', 'raw_witness', 'clock',
    'missing', 'command', 'intervention', 'public', 'forecast', 'contact', 'nominal', 'terminal'])
def test_prefix_mismatch_rejects_attribution(pair, fault, monkeypatch):
    old, new, bound, report, rows, tapes = pair
    d = rows[new][1]['decision']
    if fault == 'physics': np.savez(new/'physics_trace.npz', timestamp_s=np.arange(850)*.003)
    elif fault == 'map': d['memory_receipt']['floor'] += 1
    elif fault == 'pose': d['evidence']['registered'] = False
    elif fault == 'raw_witness': rows[old][1]['decision']['original_visual_evidence']['frame'] = 99
    elif fault == 'clock': rows[new][1]['tick'] = 2
    elif fault == 'missing': rows[bound].pop()
    elif fault == 'command': tapes[new][0]['requested_command'][0] = .1
    elif fault == 'intervention': tapes[new][1]['requested_command'][2] = -.45
    elif fault == 'forecast': d['new_selection']['prediction'][0][0][0] += .1
    elif fault == 'contact': d['new_selection']['surface_checks'][0]['original_contact_check_before_later_floor_resolution']['possible_intersection'] = False
    elif fault == 'nominal': d['new_selection']['nominal_path_checks'].pop()
    elif fault == 'terminal': d['terminal'] = 'stopped'
    else: monkeypatch.setattr(module, 'packet', lambda directory, *a, **k: str(directory))
    # Even a matching prospective stream cannot legitimize a changed original
    # pose/map/forecast/contact/horizon or an unexpected terminal policy.
    if fault in ('map', 'pose', 'forecast', 'contact', 'nominal', 'terminal'):
        rows[bound] = deepcopy(rows[new])
    with pytest.raises(ValueError): module.compare(old, new, bound, report)
