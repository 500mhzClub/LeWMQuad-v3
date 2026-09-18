"""One explicitly admitted pose for mapping, mission, residuals and contact geometry.

Only the three pose accessors differ from the frozen consumer methods. The
live dual-camera controller is unchanged; this successor requires raw prefix
and fresh physical execution before any navigation claim.
"""
from copy import deepcopy
import hashlib
import numpy as np
from lewm.causal_sensor_state import SensorContractError
from lewm.causal_depth_observation_development import validate_depth, body_points
from lewm.joint_visual_surface_memory_development import SurfaceIndex, MAX_FRAMES, STRIDE
from lewm.causal_executed_residual_diagnosis_development import WINDOW_TICKS
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.auxiliary_depth_reobserve_goal_probe_development import NO_FEASIBLE, MAX_CONSECUTIVE_WAIT_COMMANDS
from lewm.joint_floor_registered_controller_development import JointFloorRegisteredResidual
from lewm.later_floor_resolution_controller_development import LaterResolvedFloorMemory, LaterResolvedFloorMap
from lewm.dual_camera_settled_controller_development import DualCameraSettledController
from lewm.settled_boundary_round_trip_development import SettledBoundaryRoundTripMission
from lewm.measured_floor_transport_development import current_measured_floor_pose
from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration


class MeasuredFloorTransportMemory(LaterResolvedFloorMemory):
    def observe(self, policy, depth, evidence, *, now_ns):
        if self.failed:
            raise SensorContractError('visual surface memory failure latched')
        try:
            validate_depth(depth, policy, now_ns=now_ns)
            p, R, pose = current_measured_floor_pose(evidence, identity=self.identity, now_ns=now_ns)
            h = hashlib.sha256(depth['depth_m'].tobytes()+depth['valid'].tobytes()).hexdigest()
            if (pose['frame'] != len(self.route) or depth['measured_ns'] != now_ns
                    or pose['rgb_sha256'] != depth['rgb_sha256'] or pose['depth_sha256'] != h
                    or (self.last_ns is not None and now_ns-self.last_ns != 100_000_000)
                    or len(self.route) >= MAX_FRAMES):
                raise SensorContractError('uninterrupted same-image pose/depth history required')
            joints = policy['sensor_state']['sensed']['joints']
            if joints['measured_ns'][-1] != now_ns or not joints['valid'][-1, :12].all():
                raise SensorContractError('current measured joint posture required')
            witness = dict(frame=pose['frame'], measured_ns=now_ns,
                rgb_sha256=pose['rgb_sha256'], depth_sha256=h)
            cloud = body_points(depth, policy, now_ns=now_ns, stride=STRIDE)
            points = cloud['points_body_m'][cloud['valid']] @ R.T + p
            latest = SurfaceIndex(); latest.insert(points, witness)
            self.index.insert(points, witness)
            self.latest = latest
            self.route.append(witness | dict(position_initial_body_m=p.tolist(),
                rotation_initial_body_from_body=R.tolist()))
            self.position, self.rotation = p.copy(), R.copy()
            self.joints = joints['values'][-1, :12].copy()
            self.last_ns = now_ns
            return dict(**witness, sampled_returns=len(points), retained_voxels=len(self.index.cells),
                current_voxels=len(latest.cells), visited_poses=len(self.route),
                static_scene_assumed=True, uncertainty_calibrated=False,
                free_space_established=False, navigation_qualified=False)
        except (ValueError, TypeError, KeyError, IndexError, RuntimeError) as error:
            self.failed = True
            raise SensorContractError('visual surface memory unavailable; stop') from error


class MeasuredFloorTransportResidual(JointFloorRegisteredResidual):
    def observe(self, evidence, *, now_ns):
        p, R, pose = current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
        frame = pose['frame']
        if frame != self.frame+1 or now_ns != 1_500_000_000+frame*100_000_000:
            raise ValueError('uninterrupted observed residual clock required')
        new = None
        if self.pending is not None:
            old = self.pending
            if old['tick'] != frame-1 or old['measured_ns']+100_000_000 != now_ns:
                raise ValueError('only immediately preceding command may receive an observed label')
            observed = (np.asarray(old['rotation']).T@(p-np.asarray(old['position'])))[:2]
            prediction = np.asarray(old['predicted_body_xy_m'])
            new = dict(tick=old['tick'], available_tick=frame, measured_ns=now_ns,
                action=old['action'], requested_command=old['requested_command'],
                predicted_body_xy_m=prediction.tolist(), observed_body_xy_m=observed.tolist(),
                residual_xy_m=(prediction-observed).tolist(),
                start_rgb_sha256=old['rgb_sha256'], start_depth_sha256=old['depth_sha256'],
                end_rgb_sha256=pose['rgb_sha256'], end_depth_sha256=pose['depth_sha256'])
        if new is not None: self.history.append(new)
        while self.history and frame-self.history[0]['tick'] > WINDOW_TICKS: self.history.popleft()
        self.frame = frame; self.now_ns = now_ns; self.pending = None
        self.pose = dict(position=p.tolist(), rotation=R.tolist(),
            rgb_sha256=pose['rgb_sha256'], depth_sha256=pose['depth_sha256'])


class MeasuredFloorTransportMap(LaterResolvedFloorMap):
    def __init__(self, *, identity=(0, 0, 0)):
        super().__init__(identity=identity)
        self.surface = MeasuredFloorTransportMemory(identity=identity)


class MeasuredFloorTransportMission(SettledBoundaryRoundTripMission):
    def advance(self, *args, **kwargs):
        result = super().advance(*args, **kwargs)
        if result.get('observed_settling') is not None:
            result['observed_settling']['motion_source'] = 'consecutive_admitted_visual_positions_in_floor_reference'
        self.last = deepcopy(result)
        return result


class MeasuredFloorTransportController(DualCameraSettledController):
    def __init__(self, model, geometry, *, public_mission, navigation_ticks, **kwargs):
        super().__init__(model, geometry, public_mission=public_mission, navigation_ticks=navigation_ticks, **kwargs)
        self.mission = MeasuredFloorTransportMission(public_mission, navigation_ticks=navigation_ticks)
        self.registration = MeasuredFloorTransportRegistration(identity=(0, 0, 0))
        self.mapper = MeasuredFloorTransportMap(identity=(0, 0, 0))
        self.memory = self.mapper.surface
        self.residual = MeasuredFloorTransportResidual()
        self.selector.residual = self.residual

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None:
            return self._result([0., 0., 0.], None, None)
        self.infeasible_wait_active = False
        try:
            if type(now_ns) is not int or now_ns != 1_500_000_000+(self.tick+1)*100_000_000:
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, _, pose = current_measured_floor_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick+1:
                raise SensorContractError('uninterrupted measured mission frame required')
            self.history.append(deepcopy(policy))
            history = causal_history_tensors(list(self.history), now_ns) if len(self.history) == 4 else None
            self.residual.observe(evidence, now_ns=now_ns)
            self.tick += 1; self.last_ns = now_ns
            self.mission_receipt = self.mission.advance(p, frame=self.tick, now_ns=now_ns,
                previous_requested_command=self.previous_command)
            mission = self.mission_receipt
            self.terminal = mission['terminal']; self.failure = mission['failure']
            self.quiet = mission.get('quiet_intervals', 0)
            # Only target-specific scanning state changes. Map, tracker, learned
            # history and residual memory stay alive across the return transition.
            self.selector.set_goal(mission['active_goal_initial_body_xy_m'])
            self.action = None; self.plan_offset = 0
            selection = None; requested = [0., 0., 0.]
            if not mission['hold_required']:
                selection = self.selector.choose(self.model, history, self.mapper, self.geometry, now_ns=now_ns)
                if selection['view_budget_exhausted']:
                    self.terminal = 'VIEW_BUDGET_EXHAUSTED'
                elif selection['action'] is None:
                    if 'prediction' not in selection:
                        raise ValueError('infeasibility requires a valid observed forecast')
                    self.infeasible_wait_count += 1
                    if self.infeasible_wait_count <= MAX_CONSECUTIVE_WAIT_COMMANDS:
                        self.infeasible_wait_active = True
                    else:
                        self.terminal = NO_FEASIBLE
                else:
                    if self.infeasible_wait_count: self.feasible_action_recoveries += 1
                    self.infeasible_wait_count = 0
                    self.action = selection['action']; self.plan_offset = 1
                    requested = list(candidate_commands(self.action)[0])
            self.previous_command = requested.copy()
            result = self._result(requested, selection, mission.get('observed_goal_distance_m'))
            if self.terminal is None: self.residual.remember(result)
            return result | dict(causal_residual_receipt=self.residual.snapshot())
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'; self.failure = str(error)
            self.previous_command = [0., 0., 0.]
            return self._result([0., 0., 0.], None, None)

    def _result(self, command, selection, distance):
        return super()._result(command, selection, distance) | dict(
            controller='measured_floor_transport_round_trip_controller_v1',
            floor_transport_during_missingness_enabled=True,
            pose_uncertainty_calibrated=False)
