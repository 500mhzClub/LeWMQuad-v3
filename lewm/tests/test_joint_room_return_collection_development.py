"""Timing, custody, pairing and preserved native scoring for the new collection."""
import ast
from copy import deepcopy
from pathlib import Path

import pytest

from scripts import run_go2_joint_room_return_v1 as run
from scripts import audit_go2_joint_room_return_v1 as audit
from scripts import audit_go2_inner_arrival_room_return_v1 as old
from scripts.navigation_artifact_root_development import BASE,validate_root


def test_explicit_new_root_and_exact_preserved_native_audit_functions():
    assert run.OUTPUT==BASE/'go2_joint_room_return_v1_attempt_001'
    assert run.PREVIOUS==BASE/'go2_inner_arrival_room_return_v1_attempt_001'
    assert run.OUTPUT!=run.PREVIOUS and audit.OUTPUT==run.OUTPUT
    validate_root(run.OUTPUT,must_exist=False)
    assert audit.audit_sensors is old.audit_sensors
    assert audit.score_hold is old.score_hold and audit.score_winding is old.score_winding
    assert audit.TRIALS==old.TRIALS==('nominal_left','nominal_right','lower_friction_left')
    def fn(path,name):
        return ast.dump(next(n for n in ast.parse(Path(path).read_text()).body
            if isinstance(n,ast.FunctionDef) and n.name==name))
    assert fn(audit.__file__,'verify_paired_setup')==fn(old.__file__,'verify_paired_setup')
    assert fn(run.__file__,'artifacts')==fn('scripts/run_go2_inner_arrival_room_return_v1.py','artifacts')


def timing_fixture():
    rows=[dict(command_ticks_before=0,command_ticks_after=1,acquisition_wall_ms=12.,controller_wall_ms=15.,
        observation_and_control_wall_ms=28.,decision_elapsed_wall_ms=30.,iteration_wall_ms=140.,
        decision=dict(terminal=None)),
        dict(command_ticks_before=1,command_ticks_after=3,acquisition_wall_ms=12.,controller_wall_ms=15.,
        observation_and_control_wall_ms=28.,decision_elapsed_wall_ms=30.,iteration_wall_ms=340.,
        decision=dict(terminal='ROOM_RETURN_FAILED'))]
    tape=[dict(wall_ms=t) for t in (100.,150.,150.)]
    return rows,tape


def test_timing_keeps_normal_iteration_terminal_drain_and_deadline_misses():
    rows,tape=timing_fixture();before=deepcopy((rows,tape));r=audit.verify_timings(rows,tape)
    assert (rows,tape)==before
    assert r['normal_iteration']['count']==r['normal_iteration']['above100ms']==1
    assert r['terminal_iteration']['count']==1 and r['physics_command_interval']['count']==3
    assert r['total_recorded_iteration_wall_ms']==480.
    assert not r['real_time_qualified'] and r['physics_paused_during_compute']


@pytest.mark.parametrize('fault',['overlap','omission','unaccounted','nan','negative','short_total','short_decision'])
def test_incomplete_or_inconsistent_timing_rejected(fault):
    rows,tape=timing_fixture()
    if fault=='overlap':rows[1]['command_ticks_before']=0
    if fault=='omission':rows[1]['command_ticks_before']=2
    if fault=='unaccounted':tape.append(dict(wall_ms=1.))
    if fault=='nan':rows[0]['controller_wall_ms']=float('nan')
    if fault=='negative':tape[0]['wall_ms']=-1
    if fault=='short_total':rows[0]['iteration_wall_ms']=100.
    if fault=='short_decision':rows[0]['observation_and_control_wall_ms']=26.
    with pytest.raises(AssertionError):audit.verify_timings(rows,tape)


def test_collection_refuses_existing_output_before_preflight(monkeypatch,tmp_path):
    monkeypatch.setattr(run,'OUTPUT',tmp_path)
    monkeypatch.setattr(run,'preflight',lambda:pytest.fail('existing attempt must not reach preflight'))
    with pytest.raises(ValueError,match='exclusive'):run.main()
