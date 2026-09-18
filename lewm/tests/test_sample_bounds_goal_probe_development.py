import ast
from pathlib import Path
import numpy as np
from lewm.sample_bounds_surface_memory_development import SampleBoundsSurfaceMemory
from lewm.sample_bounds_goal_probe_development import SampleBoundsGoalProbe
from lewm.nominal_action_goal_probe_development import NominalActionWaypointSelector


class SphereGeometry:
    _shapes=[dict(shape_id='test:0',kind='sphere',dimensions=[.02])]
    def supports(self,joints,R):
        return dict(shapes=[dict(shape_id='test:0',center_body_m=[0.,0.,0.],lower=[-.02]*3,upper=[.02]*3)])


def test_sphere_corner_refinement_preserves_raw_box_checks_and_unknown_semantics():
    memory=SampleBoundsSurfaceMemory(identity=(0,0,0));memory.last_ns=1
    memory.position=np.zeros(3);memory.rotation=np.eye(3);memory.joints=np.zeros(12)
    memory.index.insert([[.019,.019,0]],dict(frame=0))
    r=memory.footprint(SphereGeometry(),[0,0],0.,now_ns=1,persistent=True)
    assert r['whole_voxel_possible_intersection'] and r['sample_bounds_aabb_possible_intersection']
    assert not r['possible_intersection'] and not r['free_space_established']
    assert not r['motion_permitted'] and not r['ground_contact_waiver']
    memory.index.insert([[.001,.001,0]],dict(frame=1))
    assert memory.footprint(SphereGeometry(),[0,0],0.,now_ns=1,persistent=True)['possible_intersection']


def test_failure_latch_and_nominal_selector_are_inherited():
    controller=SampleBoundsGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    assert type(controller.selector) is NominalActionWaypointSelector
    row=controller.observe({}, {}, {},now_ns=1)
    assert row['controller']=='sample_bounds_goal_probe_v1' and row['terminal']=='SENSOR_OR_MODEL_FAILURE'
    assert row['requested_command']==[0.,0.,0.]
    assert controller.observe({}, {}, {},now_ns=2)['terminal']==row['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import sample_bounds_goal_audit_development as new
    from scripts import nominal_action_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)
