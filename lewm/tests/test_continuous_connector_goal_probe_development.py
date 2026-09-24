import ast
from pathlib import Path
import numpy as np
from lewm.observed_floor_waypoint_development import propose as original,centre
from lewm.continuous_connector_waypoint_development import propose
from lewm.continuous_connector_goal_probe_development import ContinuousConnectorGoalProbe
from lewm.commitment_pose_goal_probe_development import CommitmentPoseWaypointSelector
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory


def test_same_radius_connector_avoids_cell_start_overapproximation_without_claiming_coverage():
    floor={(1,16),(2,16),(3,16)};occupied={(11,0)};p=[.107899,.181206];g=centre((3,16))
    assert original(floor,occupied,p,g)['status']=='ADDITIONAL_VIEW_REQUIRED'
    result=propose(floor,occupied,p,g)
    assert result['status']=='OBSERVED_FLOOR_ROUTE_TO_GOAL_CELL'
    assert result['nominal_radius_m']==.45 and result['connector_clearance']['minimum_observed_cell_distance_m']>.45
    assert result['unknown_connector_cells'] and not result['complete_route_floor_coverage']
    assert not result['motion_permitted'] and not result['ground_support_approved']
    assert floor=={(1,16),(2,16),(3,16)} and occupied=={(11,0)}


def test_real_nominal_start_conflict_still_rejects_every_connector():
    result=propose({(1,16),(2,16)}, {(11,0)}, [.11,.01], [.1,.9])
    assert result['status']=='ADDITIONAL_VIEW_REQUIRED' and not result['start_clearance']['nominal_disk_connector_clear']


def test_floor_grid_routing_unchanged_when_original_connector_was_clear():
    floor={(x,y) for x in range(5) for y in range(5)};args=(floor,set(),[.025,.025],[.225,.225])
    old=original(*args);new=propose(*args)
    assert all(new[k]==v for k,v in old.items())


def test_selector_surface_memory_and_failure_latch_remain_inherited():
    controller=ContinuousConnectorGoalProbe(object(),object(),condition='direct',variant='full',persistent=True)
    assert type(controller.selector) is CommitmentPoseWaypointSelector
    assert type(controller.memory) is JointVisualSurfaceMemory
    row=controller.observe({}, {}, {},now_ns=1)
    assert row['controller']=='continuous_connector_goal_probe_v1'
    assert row['terminal']=='SENSOR_OR_MODEL_FAILURE' and row['requested_command']==[0.,0.,0.]
    assert row['goal_initial_body_xy_m']==[1.2,0.]
    assert controller.observe({}, {}, {},now_ns=2)['terminal']==row['terminal']


def test_native_goal_and_actuator_audits_remain_identical():
    from scripts import continuous_connector_goal_audit_development as new
    from scripts import commitment_pose_goal_audit_development as old
    def extract(module,name):
        tree=ast.parse(Path(module.__file__).read_text())
        return ast.dump(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name))
    for name in ('audit_commands','native_goal'):assert extract(new,name)==extract(old,name)
