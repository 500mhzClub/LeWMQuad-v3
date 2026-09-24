"""Reject altered command endpoints, raw-input identities and saved outcomes."""
from copy import deepcopy
import pytest
from lewm.tests.test_chained_anchor_controller_comparison_development import rows, boundary
from scripts import verify_go2_chained_anchor_controller_completion_v1 as check


def fixture(frame=3):
    old, new = boundary() if frame == 853 else rows(frame)
    command = dict(tick=frame, pre_sample_index=749+50*frame, post_sample_index=799+50*frame,
                   completed=True, requested_command=old['requested_command'])
    original = dict(tick=frame, observation_index=frame, pre_sample_index=749+50*frame, decision=old)
    visual = dict(tick=frame, public_packet_sha256='actual', candidate=deepcopy(new['original_visual_evidence']))
    saved = dict(tick=frame, decision=new, public_input_sha256='actual', public_input_arrays_unchanged=True,
                 original_requested_command=deepcopy(command['requested_command']),
                 comparison=check.run.compare(old, new, command['requested_command'], visual['candidate'],
                                              frame=frame, boundary=853))
    return original, saved, visual, command


def test_positive_prefix_and_boundary_are_distinct():
    for frame in (3, 853):
        result = check.check_row(*fixture(frame), frame=frame, public_sha='actual')
        assert result['stop'] is (frame == 853)
        assert result['navigation_recovered'] is False


@pytest.mark.parametrize('index,key,value', [
    (0,'tick',True), (0,'observation_index',4), (0,'pre_sample_index',0),
    (1,'tick',4), (1,'public_input_arrays_unchanged',1), (1,'public_input_sha256','other'),
    (1,'original_requested_command',[.2,0.,0.]), (2,'tick',4),
    (2,'public_packet_sha256','other'), (3,'completed',False), (3,'completed',1),
    (3,'pre_sample_index',0), (3,'post_sample_index',0)])
def test_forged_packet_or_execution_witness_rejected(index,key,value):
    data = fixture()
    data[index][key] = value
    with pytest.raises(ValueError): check.check_row(*data, frame=3, public_sha='actual')


def test_saved_boolean_cannot_be_replaced_with_integer():
    data = fixture()
    data[1]['comparison']['stop'] = 0
    with pytest.raises(ValueError, match='entire saved comparison'):
        check.check_row(*data, frame=3, public_sha='actual')


def test_changed_forecast_is_rejected():
    data = fixture()
    data[1]['decision']['new_selection']['prediction']['x'] = [99]
    with pytest.raises(ValueError, match='complete original decision'):
        check.check_row(*data, frame=3, public_sha='actual')


@pytest.mark.parametrize('counts', [(853,853,850),(855,853,850),(854,852,850),(854,853,849)])
def test_partial_or_excess_populations_rejected(counts):
    with pytest.raises(ValueError, match='populations'): check.reconstruct_report(*counts, {})


def test_reconstructed_report_keeps_scope_and_actual_boundary():
    data = fixture(853)
    data[1]['decision']['selected_action'] = None
    report = check.reconstruct_report(854,853,850,data[1])
    assert report['boundary_requested_command'] == data[1]['decision']['requested_command']
    for key in ('new_command_executed','native_execution','navigation_qualified','goal_achieved',
                'following_recorded_observations_consumed'):
        assert report[key] is False
    assert report['exact_original_decisions'] == 853
