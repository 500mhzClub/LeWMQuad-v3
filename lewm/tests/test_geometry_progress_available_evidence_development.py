"""Insufficient rays remain failed measurements, with all other checks retained."""
import ast
from pathlib import Path
import numpy as np
import pytest
from scripts import geometry_progress_available_sensor_evidence_development as sensor
from scripts import pulse_context_sensor_audit_development as old_sensor
from scripts import read_go2_geometry_progress_available_evidence_v1 as new
from scripts import read_go2_geometry_progress_commands_v1 as old


@pytest.mark.parametrize('n',[0,1,999,1000])
def test_original_minimum_ray_population_still_required_for_passing_frame(n):
    r=sensor.depth_evidence(np.zeros(n))
    assert r['within1mm']==r['sufficient_native_depth_rays']==(n>=1000)
    assert r['maximum_error_m']==(0. if n else None)
    assert r['measurement_status']==('PASS' if n>=1000 else 'INSUFFICIENT_NATIVE_DEPTH_RAYS')


def test_nonfinite_error_still_fails_and_large_finite_error_is_not_accepted():
    with pytest.raises(ValueError):sensor.depth_evidence(np.array([float('nan')]))
    r=sensor.depth_evidence(np.full(1000,.0011))
    assert not r['within1mm'] and r['measurement_status']=='DEPTH_GEOMETRY_ERROR'


def fn(source,name):
    return ast.dump(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_only_insufficient_ray_failure_accounting_changes_in_raw_sensor_audit():
    before=Path(old_sensor.__file__).read_text();after=Path(sensor.__file__).read_text()
    after=after.replace("require(np.isfinite(errors).all(), 'finite interior depth errors required')",
        "require(len(errors) >= 1000 and np.isfinite(errors).all(), 'sufficient finite interior depth rays')")
    after=after.replace('row = dict(frame=frame, **depth_evidence(errors),',
        'row = dict(frame=frame, rays=len(errors), maximum_error_m=float(errors.max()), within1mm=bool(errors.max() <= .001),')
    assert fn(after,'audit_sensors')==fn(before,'audit_sensors')
    assert fn(after,'native_depth_comparison_mask')==fn(before,'native_depth_comparison_mask')


def test_exact_command_stop_and_outcome_checks_are_unchanged_from_separate_readout():
    before=Path(old.__file__).read_text();after=Path(new.__file__).read_text()
    for name in ('audit_commands','audit_condition','native_horizons','prefix_witness','compare_prefixes'):
        assert fn(after,name)==fn(before,name)
    assert new.INPUT==old.INPUT and new.FAILED_READOUT==old.OUTPUT and new.OUTPUT!=old.OUTPUT
    assert new.audit_setup is old.audit_setup and new.audit_stops is old.audit_stops
