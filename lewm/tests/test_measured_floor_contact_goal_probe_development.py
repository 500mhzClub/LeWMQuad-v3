import ast
from pathlib import Path
import numpy as np
from lewm.measured_floor_partition_development import MeasuredFloorPartition
from lewm.measured_floor_contact_development import apply_floor_contact_rule
from lewm.measured_floor_contact_goal_probe_development import MeasuredFloorContactGoalProbe


class Geometry:
    _shapes=[dict(shape_id='FL_foot:0',kind='sphere',dimensions=[.022]),dict(shape_id='calf:0',kind='box',dimensions=[.01]*3)]
    def supports(self,q,R):return dict(shapes=[dict(shape_id=s['shape_id'],center_body_m=[.025,.025,0.]) for s in self._shapes])


def run(mask,floor,body_hit=False):
    p=MeasuredFloorPartition();p.insert([[.025,.025,0.]],np.array([mask]),dict(frame=0))
    original=dict(shapes=[dict(shape_id='FL_foot:0',intersecting_voxels=1),dict(shape_id='calf:0',intersecting_voxels=int(body_hit))],possible_intersection=True)
    return apply_floor_contact_rule(original,Geometry(),None,np.eye(3),np.zeros(3),np.eye(3),floor,p)


def test_only_measured_floor_with_full_projection_can_receive_foot_contact():
    result=run(True,{(0,0)})
    assert not result['possible_intersection'] and result['ground_contact_waiver']
    assert result['all_return_possible_intersection'] and not result['ground_support_approved']
    assert run(True,set())['possible_intersection']
    assert run(False,{(0,0)})['possible_intersection']


def test_nonfoot_body_contact_is_never_exempted():
    assert run(True,{(0,0)},body_hit=True)['possible_intersection']


def test_failure_latch_still_stops():
    c=MeasuredFloorContactGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    r=c.observe({}, {}, {},now_ns=1)
    assert r['terminal']=='SENSOR_OR_MODEL_FAILURE' and r['requested_command']==[0.,0.,0.]
    assert r['floor_partition_receipt'] is None
    assert c.observe({}, {}, {},now_ns=2)['terminal']==r['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import measured_floor_contact_goal_audit_development as new
    from scripts import sample_bounds_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)
