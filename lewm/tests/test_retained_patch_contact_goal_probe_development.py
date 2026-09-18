import ast
from pathlib import Path
from lewm.retained_patch_contact_goal_probe_development import RetainedPatchContactGoalProbe
from lewm.retained_patch_contact_goal_probe_development import RetainedPatchContactMemory
import numpy as np


def test_failure_latch_still_stops_without_stale_classification():
    c=RetainedPatchContactGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    r=c.observe({}, {}, {},now_ns=1)
    assert r['terminal']=='SENSOR_OR_MODEL_FAILURE' and r['requested_command']==[0.,0.,0.]
    assert r['floor_partition_receipt'] is None and c.memory.failed
    assert c.observe({}, {}, {},now_ns=2)['terminal']==r['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import retained_patch_contact_goal_audit_development as new
    from scripts import measured_floor_contact_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)


def test_retained_patch_resolves_only_floor_contact_and_keeps_unknown_return():
    from lewm.tests.test_observed_geometry_refinement_development import floor
    class Geometry:
        _shapes=[dict(shape_id='FL_foot:0',kind='sphere',dimensions=[.022])]
        def supports(self,q,R):
            return dict(shapes=[dict(shape_id='FL_foot:0',center_body_m=[1.,0.,-.32],
                lower=[.978,-.022,-.342],upper=[1.022,.022,-.298])])
    m=RetainedPatchContactMemory(identity=(0,0,0));w=dict(frame=0,measured_ns=0)
    m.position=np.zeros(3);m.rotation=np.eye(3);m.joints=np.zeros(12);m.last_ns=m.classified_ns=0
    m.map_from_initial=np.eye(3);m.floor_cells=set();m.route=[w]
    point=[[1.,0.,-.32]];m.index.insert(point,w);m.partition.insert(point,np.array([True]),w)
    d,v=floor();m.patches.append(d,v,np.eye(3),np.zeros(3),-.32,w)
    r=m.footprint(Geometry(),[0,0],0.,now_ns=0)
    assert r['grid_contact_possible_intersection'] and not r['possible_intersection']
    f=r['foot_floor_contacts'][0]
    assert not f['original_grid_contact_rule_eligible'] and f['retained_patch']['complete_nominal_foot_patch']
    assert not r['ground_support_approved'] and not m.floor_cells
    m.index.insert(point,w);m.partition.insert(point,np.array([False]),w)
    assert m.footprint(Geometry(),[0,0],0.,now_ns=0)['possible_intersection']
