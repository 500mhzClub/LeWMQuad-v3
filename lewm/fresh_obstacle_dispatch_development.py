"""Current sampled-obstacle veto for expiring development commands.

This nominal disk test is not a whole-body or ground-support certificate. It
does not infer free space from absent returns. Fresh collision and visibility
outcomes must be measured in the prospective simulator experiment.
"""
from dataclasses import dataclass
import numpy as np

from lewm.causal_depth_observation_development import body_points, validate_depth
from lewm.auxiliary_downward45_depth_observation_development import (
    body_points as auxiliary_points, validate_depth as validate_auxiliary)
from lewm.extended_return_budget_transport_development import current_measured_floor_pose
from lewm.observed_geometry_refinement_development import nominal_connector
from lewm.joint_visual_floor_map_development import CELL_M
from lewm.delayed_action_planning_development import PERIOD_NS

MAX_OBSERVATION_AGE_NS = 200_000_000
NOMINAL_RADIUS_M = .45


@dataclass(frozen=True)
class CurrentObstacles:
    frame: int
    measured_ns: int
    position_map: tuple
    rotation_map_from_body: tuple
    occupied: frozenset
    valid_return_counts: tuple
    coordinate_frame: str = 'fixed_routing_map'


def observe_body_obstacles(policy, depth, evidence, *, auxiliary_depth, measured_ns,
        identity=(0, 0, 0)):
    """Current paired returns above the current measured plane, before mapping."""
    from lewm.joint_sensor_anchored_goal_development import current_supported_rgbd_pose as current_joint_pose
    from lewm.joint_floor_registered_evidence_development import depth_hash
    from lewm.joint_measured_floor_plane_development import validate_joint_plane
    _,_,pose=current_joint_pose(evidence,identity=identity,now_ns=measured_ns)
    receipt=evidence['measured_plane_evidence']
    if receipt['measured_ns']!=measured_ns or receipt['frame']!=pose['frame']:
        raise ValueError('current pose and measured plane must share acquisition')
    for camera,packet,validate in (('primary',depth,validate_depth),
            ('auxiliary',auxiliary_depth,validate_auxiliary)):
        validate(packet,policy,now_ns=measured_ns)
        if packet['measured_ns']!=measured_ns or depth_hash(packet)!=receipt['depth_sha256'][camera]:
            raise ValueError('current plane must bind actual paired depth')
    plane=receipt['joint_plane']
    if not plane['available']:return None
    validate_joint_plane(plane,plane['up_body'])
    normal=np.asarray(plane['normal_body']);offset=plane['offset_body_m']
    cells=set();counts=[]
    for packet,project in ((depth,body_points),(auxiliary_depth,auxiliary_points)):
        cloud=project(packet,policy,now_ns=measured_ns,stride=4)
        points=cloud['points_body_m'][cloud['valid']];counts.append(len(points))
        if not len(points):return None
        height=points@normal+offset
        above=points[(height>.03)&(height<.65)]
        keys=np.unique(np.floor(above[:,:2]/CELL_M).astype(int),axis=0)
        cells.update(tuple(map(int,cell)) for cell in keys)
    return CurrentObstacles(pose['frame'],measured_ns,(0.,0.,0.),
        tuple(map(tuple,np.eye(3))),frozenset(cells),tuple(counts),'current_body')


def observe_obstacles(policy, depth, evidence, routing_snapshot, *, auxiliary_depth,
        measured_ns, identity=(0, 0, 0)):
    routing_snapshot.age_ns(now_ns=measured_ns)
    validate_depth(depth,policy,now_ns=measured_ns)
    validate_auxiliary(auxiliary_depth,policy,now_ns=measured_ns)
    p,R,pose=current_measured_floor_pose(evidence,identity=identity,now_ns=measured_ns)
    if depth['measured_ns']!=measured_ns or auxiliary_depth['measured_ns']!=measured_ns:
        raise ValueError('actual paired current-depth acquisition required')
    B=np.asarray(routing_snapshot.map_from_initial);Q,q=B@R,B@p
    cells=set();counts=[]
    for auxiliary,packet in ((False,depth),(True,auxiliary_depth)):
        cloud=(auxiliary_points if auxiliary else body_points)(packet,policy,now_ns=measured_ns,stride=4)
        points=cloud['points_body_m'][cloud['valid']];counts.append(len(points))
        if not len(points):raise ValueError('observed current returns required from both cameras')
        mapped=((points@R.T+p)@B.T) if auxiliary else (points@Q.T+q)
        above=mapped[(mapped[:,2]>routing_snapshot.floor_height+.03)&(mapped[:,2]<routing_snapshot.floor_height+.65)]
        keys=np.unique(np.floor(above[:,:2]/CELL_M).astype(int),axis=0)
        cells.update(tuple(map(int,cell)) for cell in keys if np.all(cell>=-100) and np.all(cell<100))
    return CurrentObstacles(pose['frame'],measured_ns,tuple(map(float,q)),
        tuple(tuple(map(float,row)) for row in Q),frozenset(cells),tuple(counts))


def dispatch_request(plan, current, *, now_ns):
    if type(now_ns) is not int:raise ValueError('actual integer dispatch clock required')
    common=dict(now_ns=now_ns,observation_measured_ns=None if current is None else current.measured_ns,
        obstacle_coordinate_frame=None if current is None else current.coordinate_frame,
        clearance_certified=False,whole_body_contact_checked=False,ground_support_checked=False,
        unknown_space_inferred_free=False,native_state_used=False)
    if plan is None:return common|dict(requested_command=[0.,0.,0.],reason='NO_ON_TIME_PLAN')
    if not plan.dispatch_ns<=now_ns<plan.expires_ns:
        return common|dict(requested_command=[0.,0.,0.],reason='OUTSIDE_COMMITTED_INTERVAL')
    if current is None or not 0<=now_ns-current.measured_ns<=MAX_OBSERVATION_AGE_NS:
        return common|dict(requested_command=[0.,0.,0.],reason='CURRENT_OBSERVATION_UNAVAILABLE_OR_STALE')
    p=np.asarray(current.position_map);R=np.asarray(current.rotation_map_from_body)
    # This is explicitly a requested-speed nominal connector, not a measured
    # future pose. Check the full remaining interval from the latest measured pose.
    seconds=(plan.expires_ns-now_ns)/1e9
    endpoint=p+R@np.array([plan.command[0]*seconds,plan.command[1]*seconds,0.])
    check=nominal_connector(p[:2],endpoint[:2],sorted(current.occupied),radius_m=NOMINAL_RADIUS_M)
    allowed=bool(check['nominal_disk_connector_clear'])
    return common|dict(requested_command=plan.request(now_ns=now_ns,fresh_observation_allows_motion=allowed),
        reason='CURRENT_NOMINAL_OBSTACLE_TEST_PASSED' if allowed else 'CURRENT_OBSERVED_OBSTACLE_VETO',
        observation_age_ns=now_ns-current.measured_ns,nominal_connector=check,
        endpoint_is_requested_speed_projection=True,command_expires_ns=plan.expires_ns,
        command_observation_ns=plan.observed_ns)
