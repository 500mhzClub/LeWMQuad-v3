import numpy as np
import pytest

from lewm.action_coverage_diagnostic_development import command_context,support_table,stratified_errors
from lewm.tests.test_relative_gyro_turn_development import initialized,packet


def test_actual_command_context_does_not_infer_movement_from_request():
    value=packet(initialized(),80); control=value['sensor_state']['control']['applied_command']
    control['values'][-5:]=[.3,0,0]; control['values'][-5]=[.25,0,0]
    result=command_context(value)
    assert result['last_action_index']==1 and not result['five_applied_ticks_constant']
    control['values'][-5]=[.3,0,0]
    assert command_context(value)['five_applied_ticks_constant']


def test_nonbank_command_and_missing_history_are_explicit():
    value=packet(initialized(),80); control=value['sensor_state']['control']['applied_command']
    control['values'][-1]=[.1,0,.1]
    assert command_context(value)['last_action_name']=='nonbank'
    control['valid'][-3]=False; control['values'][-3]=0
    with pytest.raises(ValueError): command_context(value)


def test_support_counts_windows_without_inflating_layouts():
    rows=[{'stage':'later','prior_index':1,'future_index':1,'layout_id':'a','constant_past':True} for _ in range(7)]
    table=support_table(rows); diagonal=next(r for r in table if r['stage']=='later' and r['prior_action_index']==r['future_action_index']==1)
    assert diagonal['windows']==7 and diagonal['layouts']==1
    assert all(r['windows']==0 for r in table if r['prior_action_index']!=r['future_action_index'])


def test_error_strata_keep_empty_masks_and_layout_weighting():
    common={'method':'jepa_rollout','group':'switch_previous_selection','contact':False,
        'training_later_pair_windows':0,'yaw_error_rad':None,'contact_brier':.1}
    rows=[common|{'layout_id':'a','position_error_m':1.} for _ in range(5)]+[common|{'layout_id':'b','position_error_m':3.}]
    report=stratified_errors(rows); switch=next(r for r in report if r['group']=='switch_previous_selection')
    assert switch['layout_macro']['position_error_m']['mean']==2.
    assert switch['layout_macro']['yaw_error_rad']=={'contributing_layouts':0,'mean':None}
    assert next(r for r in report if r['group']=='initial')['choices']==0
