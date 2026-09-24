import ast
from pathlib import Path
from types import SimpleNamespace
import pytest
from scripts import auxiliary_depth_goal_session_development as session_module
from scripts import auxiliary_depth_goal_audit_development as audit
from scripts import training_bias_goal_audit_development as prior_audit
from scripts import auxiliary_depth_goal_episode_development as episode
from scripts import training_bias_goal_episode_development as prior_episode
from scripts import run_go2_auxiliary_depth_goal_probe_v1 as runner
from scripts import auxiliary_depth_goal_sensor_audit_development as sensors


def test_original_native_arrival_actuator_and_primary_sensor_gates():
    assert audit.native_goal is prior_audit.native_goal
    assert audit.audit_commands is prior_audit.audit_commands
    assert audit.audit_sensors is prior_audit.audit_sensors
    assert audit.audit_setup is prior_audit.audit_setup and audit.audit_stops is prior_audit.audit_stops
    assert runner.CASES == __import__('scripts.run_go2_training_bias_goal_probe_v1',fromlist=['CASES']).CASES


def test_collector_preserves_execution_and_terminal_contract():
    source=Path(episode.__file__).read_text()
    for new,old in (
        ('AuxiliaryDepthGoalSession(', 'GeometryProgressFamilySession('),
        ('AuxiliaryDepthGoalProbe(', 'TrainingBiasGoalProbe('),
        ('auxiliary_depth_goal_probe_v1;', 'training_bias_goal_probe_v1;'),
        ('AUXILIARY_DEPTH_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED','TRAINING_BIAS_GOAL_PROBE_TERMINAL_AUDIT_REQUIRED'),
        ('p, d, f, auxiliary, now =','p, d, f, now ='),
        (', auxiliary_depth=auxiliary',''),
        ('auxiliary_frames=len(session.auxiliary_audit), ','')):
        source=source.replace(new,old)
    def body(s):
        return ast.dump(next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name=='collect'))
    assert body(source)==body(Path(prior_episode.__file__).read_text())


def test_live_lazy_pairing_once_per_frame_through_terminal_bound(monkeypatch):
    events=[]
    session=session_module.AuxiliaryDepthGoalSession.__new__(session_module.AuxiliaryDepthGoalSession)
    session.model_manifest=[];session.auxiliary_audit=[];session.output=None;session.tick=0
    def primary(s):
        if len(s.model_manifest)==s.tick:s.model_manifest.append({})
        events.append(('primary',s.tick))
        return {},{},{},1500000000+100000000*s.tick
    def capture(s,d,i):
        assert len(s.model_manifest)==i+1
        events.append(('auxiliary',i))
        return dict(frame=i,physical_sample_index=749+50*i,measured_ns=1500000000+100000000*i)
    monkeypatch.setattr(session_module.VisibleRobotFamilySession,'sensor_packets',primary)
    monkeypatch.setattr(session_module,'capture',capture)
    monkeypatch.setattr(session_module,'public_acquisition',lambda r:r)
    monkeypatch.setattr(session_module,'packet',lambda d,i,p,r,now_ns:dict(frame=i,time=now_ns))
    for tick in range(254):
        session.tick=tick
        first=session.sensor_packets();second=session.sensor_packets()
        assert first==second and first[3]['frame']==tick
    assert [i for k,i in events if k=='auxiliary']==list(range(254))
    assert events==[(k,i) for i in range(254) for k in ('primary','auxiliary','primary')]
    session.tick=254
    with pytest.raises(ValueError,match='bounded'):session.sensor_packets()


def test_live_pairing_rejects_wrong_physical_sample_before_public_packet(monkeypatch):
    session=session_module.AuxiliaryDepthGoalSession.__new__(session_module.AuxiliaryDepthGoalSession)
    session.model_manifest=[{}];session.auxiliary_audit=[];session.output=None
    monkeypatch.setattr(session_module.VisibleRobotFamilySession,'sensor_packets',lambda s:({},{},{},1500000000))
    monkeypatch.setattr(session_module,'capture',lambda *a:dict(physical_sample_index=750,measured_ns=1500000000))
    monkeypatch.setattr(session_module,'packet',lambda *a,**k:pytest.fail('mispaired acquisition must not become public'))
    with pytest.raises(ValueError,match='pairing'):session.sensor_packets()
    assert session.auxiliary_audit==[]


@pytest.mark.parametrize('change',[dict(robot_visual_geometries=32),dict(total_nodes=34),dict(order='floor_first')])
def test_incomplete_robot_raster_never_reaches_precision_normalization(monkeypatch,change):
    order=dict(order='floor_walls_robot_visual_geometry_order',surfaces=[],robot_visual_geometries=33,total_nodes=35)|change
    monkeypatch.setattr(sensors,'read_json',lambda *a:dict(order=order))
    monkeypatch.setattr(sensors,'validate_primary_precision',lambda *a:pytest.fail('incomplete raster accepted'))
    with pytest.raises(ValueError,match='33-robot-node'):sensors.audit_rasters_and_footprints(None,{},[{}],{})
