"""Bounded online bootstrap probe: learned plans, measured pose, explicit goal.

The fixed family-JEPA model is unqualified for arbitrary moving action switches.
Only robot URDF geometry is used; no native state, scene geometry or labels.
The 1.2m mission target is an initial-frame instruction, not a .4m local servo
goal and not an observation of free space. Observed arrival is only a candidate.
"""
from collections import deque
from copy import deepcopy
import numpy as np

from lewm.causal_sensor_state import SensorContractError
from lewm.causal_subtrajectory_learning_development import causal_history_tensors
from lewm.geometry_progress_predictive_selection_development import select
from lewm.geometry_progress_pilot_development import candidate_commands
from lewm.joint_sensor_anchored_goal_development import current_joint_pose
from lewm.corner_support_joint_observer_development import CornerSupportVisualLedMotion
from lewm.joint_visual_surface_memory_development import JointVisualSurfaceMemory
from lewm.surface_memory_candidate_filter_development import filter_selection

GOAL_XY_M = (1.2, 0.)
WARMUP_TICKS = 3
NAVIGATION_TICKS = 240
COMMIT_TICKS = 5
ARRIVAL_RADIUS_M = .04
ARRIVAL_QUIET_INTERVALS = 10
CONTACT_PENALTY_M = 1.2
DRAIN_TICKS = 10


class SurfaceMemoryGoalProbe:
    def __init__(self, model, geometry, *, persistent):
        if type(persistent) is not bool:
            raise ValueError("explicit memory variant required")
        self.geometry = geometry
        self.persistent = persistent
        self.memory = JointVisualSurfaceMemory(identity=(0, 0, 0))
        self.memory_receipt = None
        self.model = model
        self.motion = CornerSupportVisualLedMotion(identity=(0, 0, 0))
        self.history = deque(maxlen=4)
        self.tick = -1
        self.last_ns = None
        self.terminal = None
        self.failure = None
        self.action = None
        self.plan_offset = 0
        self.quiet = 0
        self.was_within_goal = False
        self.previous_command = [0., 0., 0.]

    def observe(self, policy, depth, fast, *, now_ns):
        evidence = None
        self.memory_receipt = None
        try:
            if self.terminal is None:
                evidence = self.motion.observe(policy, depth, fast, now_ns=now_ns)
                self.memory_receipt = self.memory.observe(policy, depth, evidence, now_ns=now_ns)
            result = self.advance(policy, evidence, now_ns=now_ns)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'
            self.failure = str(error)
            result = self._result([0., 0., 0.], None, None)
        return result | dict(evidence=evidence)

    def advance(self, policy, evidence, *, now_ns):
        if self.terminal is not None:
            return self._result([0., 0., 0.], None, None)
        try:
            if (type(now_ns) is not int or now_ns != 1_500_000_000 + (self.tick + 1) * 100_000_000):
                raise SensorContractError('uninterrupted exact mission observation clock required')
            p, R, pose = current_joint_pose(evidence, identity=(0, 0, 0), now_ns=now_ns)
            if pose['frame'] != self.tick + 1:
                raise SensorContractError('uninterrupted measured mission frame required')
            self.tick += 1
            self.last_ns = now_ns
            self.history.append(deepcopy(policy))
            # Full tensor admission also on committed ticks, before commands.
            history = causal_history_tensors(list(self.history), now_ns) if len(self.history) == 4 else None
            delta = np.r_[np.asarray(GOAL_XY_M) - p[:2], 0.]
            goal_body = (R.T @ delta)[:2]
            distance = float(np.linalg.norm(delta[:2]))
            selection = None
            requested = [0., 0., 0.]
            if self.tick < WARMUP_TICKS:
                pass
            elif distance <= ARRIVAL_RADIUS_M:
                self.quiet = self.quiet + 1 if self.was_within_goal and self.previous_command == [0., 0., 0.] else 0
                self.action = None
                if self.quiet >= ARRIVAL_QUIET_INTERVALS:
                    self.terminal = 'OBSERVED_GOAL_CANDIDATE'
            elif self.tick >= WARMUP_TICKS + NAVIGATION_TICKS:
                self.terminal = 'MISSION_TICK_BUDGET_EXHAUSTED'
            else:
                self.quiet = 0
                if self.action is None or self.plan_offset == COMMIT_TICKS:
                    selection = select(self.model, history, head='rollout_outcomes', input_variant='full',
                        goal_body_xy_m=goal_body, contact_penalty_m=CONTACT_PENALTY_M)
                    selection.pop('selection_wall_ms')
                    selection = filter_selection(selection, self.memory, self.geometry,
                        now_ns=now_ns, persistent=self.persistent)
                    self.action = selection['action']
                    self.plan_offset = 0
                if self.action is None:
                    self.terminal = "ALL_CANDIDATES_HAVE_SURFACE_INTERSECTION"
                else:
                    requested = list(candidate_commands(self.action)[self.plan_offset])
                    self.plan_offset += 1
            # Hard time budget also applies during an unfinished arrival dwell.
            if self.tick >= WARMUP_TICKS + NAVIGATION_TICKS and self.terminal is None:
                self.terminal = 'MISSION_TICK_BUDGET_EXHAUSTED'
                requested = [0., 0., 0.]
            self.previous_command = requested.copy()
            self.was_within_goal = distance <= ARRIVAL_RADIUS_M
            return self._result(requested, selection, distance)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            self.terminal = 'SENSOR_OR_MODEL_FAILURE'
            self.failure = str(error)
            return self._result([0., 0., 0.], None, None)

    def _result(self, command, selection, distance):
        return dict(controller='surface_memory_goal_probe_v1',
            memory_variant='persistent' if self.persistent else 'current_frame',
            memory_receipt=deepcopy(self.memory_receipt),
            experimental_motion_without_clearance_certificate=True, requested_command=command,
            tick=self.tick, terminal=self.terminal, failure=self.failure,
            selected_action=self.action, plan_offset=self.plan_offset, new_selection=selection,
            observed_goal_distance_m=distance, quiet_intervals=self.quiet,
            goal_initial_body_xy_m=list(GOAL_XY_M), native_state_used=False,
            contact_probability_calibrated=False, navigation_qualified=False)
