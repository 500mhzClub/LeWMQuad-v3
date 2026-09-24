"""Saved expectations stop before any later original observation is consumed."""
from copy import deepcopy
import pytest
from lewm.tests.test_commitment_contact_anchored_development import pair
from scripts.check_go2_commitment_contact_anchored_saved_prefix_v1 import check_rows


def population():
    old, _ = pair(); rows = []; tape = []
    for frame in range(4):
        d = deepcopy(old); d['tick'] = frame
        if frame < 3: d.update(new_selection=None, selected_action=None, requested_command=[0., 0., 0.])
        rows.append(dict(tick=frame, decision=d))
        tape.append(dict(tick=frame, completed=True, pre_sample_index=749+50*frame,
            post_sample_index=799+50*frame, requested_command=d['requested_command']))
    return rows, tape


def test_first_boundary_stops_generator_before_unexecuted_future():
    rows, tape = population(); records = []
    def guarded_rows():
        yield from rows
        raise AssertionError('later original observation consumed as candidate future')
    result = check_rows(guarded_rows(), tape, records.append)
    assert result['frame'] == 3 and result['consumed_frames'] == 4
    assert [r['changed'] for r in records] == [False, False, False, True]
    assert result['original_requested_command'] == [0., 0., 0.]
    assert result['candidate_requested_command'] != result['original_requested_command']
    assert not result['full_controller_reconstructed'] and not result['model_inference_performed']


@pytest.mark.parametrize('fault', ['order', 'terminal', 'model', 'command', 'endpoint', 'incomplete', 'no_change'])
def test_unbound_or_incomplete_original_expectations_are_rejected(fault):
    rows, tape = population()
    if fault == 'order': rows[1]['tick'] = 2
    elif fault == 'terminal': rows[1]['decision']['terminal'] = 'STOP'
    elif fault == 'model': rows[1]['decision']['model_condition'] = 'jepa'
    elif fault == 'command': tape[1]['requested_command'] = [1., 0., 0.]
    elif fault == 'endpoint': tape[1]['post_sample_index'] += 1
    elif fault == 'incomplete': tape[1]['completed'] = False
    else: rows.pop()
    with pytest.raises(ValueError): check_rows(rows, tape, lambda row: None)
