import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import auxiliary_downward45_frame_acquisition_development as acquisition
from scripts import auxiliary_downward45_depth_capture_development as capture
from scripts import auxiliary_tilted_depth_capture_integrity_development as old_capture
from scripts import auxiliary_downward45_sensor_audit_development as audit
from scripts import auxiliary_depth_goal_sensor_audit_development as old_audit
from scripts import capture_go2_auxiliary_downward45_depth_prefix_v1 as runner
from scripts import capture_go2_auxiliary_tilted_depth_prefix_integrity_v2 as old_runner
from lewm.auxiliary_downward45_depth_geometry_development import body_from_optical,CALIBRATION_ID


def body(module,name):
    return ast.dump(next(n for n in ast.parse(Path(module.__file__).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name))


def test_same_native_collection_capture_and_raw_gates_with_explicit_new_calibration():
    assert capture.body_from_optical is audit.body_from_optical is body_from_optical
    assert capture.CALIBRATION_ID==audit.CALIBRATION_ID==CALIBRATION_ID
    assert body(capture,'capture')==body(old_capture,'capture')
    assert body(runner,'collect')==body(old_runner,'collect')
    for name in ('audit_auxiliary','audit_rasters_and_footprints'):assert body(audit,name)==body(old_audit,name)
    assert runner.COMMAND_TICKS==17 and runner.VisibleRobotFamilySession is old_runner.VisibleRobotFamilySession
    assert runner.PREVIOUS_CAPTURE==old_runner.OUTPUT


def test_eighteen_lazy_paired_observations_stop_before_translation(monkeypatch):
    events=[];s=SimpleNamespace(samples=[None]*750,model_manifest=[],tick=0)
    def primary():
        s.model_manifest.append({});events.append(('primary',s.tick));return s.tick
    s.capture_current=primary
    def auxiliary(session,directory,tick):
        assert len(session.samples)==750+50*tick and len(session.model_manifest)==tick+1
        events.append(('auxiliary',tick));return dict(frame=tick)
    monkeypatch.setattr(acquisition,'capture',auxiliary)
    for i in range(18):
        s.tick=i
        if i:s.samples.extend([None]*50)
        assert acquisition.capture_frame(s,None,i)==dict(frame=i)
    assert events==[(k,i) for i in range(18) for k in ('primary','auxiliary')]
    s.tick=18;s.samples.extend([None]*50)
    with pytest.raises(ValueError):acquisition.capture_frame(s,None,18)
    assert events[-1]==('primary',18)
