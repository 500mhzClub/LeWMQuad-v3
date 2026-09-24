"""Explicit committed-prefix forecasts and expiring commands for a delayed planner."""
from dataclasses import dataclass
import numpy as np
import torch

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.observation_horizon_predictive_selection_development import candidate_inputs
from lewm.extended_return_budget_transport_development import current_measured_floor_pose
from lewm.observed_floor_waypoint_development import propose
from lewm.mission_coordinate_metric_development import position_distance

PERIOD_NS = 100_000_000
DELAY_TICKS = 2
COMMIT_TICKS = 1


def route_from_current_pose(snapshot, evidence, goal_initial_xy, *, measured_ns, identity=(0, 0, 0)):
    age = snapshot.age_ns(now_ns=measured_ns)
    p, _, pose = current_measured_floor_pose(evidence, identity=identity, now_ns=measured_ns)
    goal = np.asarray(goal_initial_xy, float)
    if goal.shape != (2,) or not np.isfinite(goal).all(): raise ValueError('finite routing goal required')
    B = np.asarray(snapshot.map_from_initial)
    result = propose(snapshot.floor, snapshot.occupied, (B@p)[:2], (B@np.r_[goal, 0.])[:2])
    return result | dict(map_frame=snapshot.frame, map_measured_ns=snapshot.measured_ns,
        pose_frame=pose['frame'], pose_measured_ns=measured_ns, map_age_at_pose_ns=age,
        command_authorized=False, current_robot_pose_assumed=False)


def delayed_candidate_inputs(history, committed_commands, *, delay_ticks=DELAY_TICKS, commit_ticks=COMMIT_TICKS):
    """Two already-committed intervals, one candidate interval, then known zero.

    The caller must supply its actual execution ledger. A historical measured
    command is not automatically a promise about the future committed prefix.
    """
    if type(delay_ticks) is not int or delay_ticks not in (2,3):raise ValueError('two or three delay ticks required')
    if type(commit_ticks) is not int or commit_ticks not in (1,4):raise ValueError('one or four commitment ticks required')
    prefix = np.asarray(committed_commands, float)
    if (prefix.shape != (delay_ticks, 3) or not np.isfinite(prefix).all()
            or np.any(prefix[:, 1] != 0.) or np.any(np.abs(prefix) > [.3, 1., .5])):
        raise ValueError('two explicit bounded fixed-axis committed commands required')
    inputs = candidate_inputs(history)
    commands = torch.zeros((6, 8, 1, 3), dtype=torch.float32)
    commands[:, :delay_ticks, 0] = torch.as_tensor(prefix, dtype=torch.float32)
    for index, action in enumerate(ACTIONS):
        commands[index, delay_ticks:delay_ticks+commit_ticks, 0] = torch.tensor(candidate_commands(action)[0])
    inputs['known_action_blocks'] = commands/torch.tensor([.3, 1., .5])
    inputs['known_action_valid'] = torch.ones((6, 8, 1), dtype=torch.bool)
    return inputs


def score_delayed_predictions(prediction, goal_body_xy_m, *, contact_penalty_m=1.2, delay_ticks=DELAY_TICKS, commit_ticks=COMMIT_TICKS, position_metric_matrix=None):
    if type(delay_ticks) is not int or delay_ticks not in (2,3):raise ValueError('two or three delay ticks required')
    if type(commit_ticks) is not int or commit_ticks not in (1,4):raise ValueError('one or four commitment ticks required')
    p = np.asarray(prediction); goal = np.asarray(goal_body_xy_m, float)
    if (p.shape != (6, 8, 5) or not np.issubdtype(p.dtype, np.floating)
            or not np.isfinite(p).all() or goal.shape != (2,) or not np.isfinite(goal).all()
            or not np.isfinite(contact_penalty_m) or contact_penalty_m < 0
            or (np.diff(p[:, :, 4], axis=1) < -1e-6).any()):
        raise ValueError('finite complete cumulative forecasts and explicit goal/cost required')
    start = p[:, delay_ticks-1, :2]; end = p[:, delay_ticks+commit_ticks-1, :2]
    progress = position_distance(goal-start, position_metric_matrix)-position_distance(goal-end, position_metric_matrix)
    contact = np.exp(-np.logaddexp(0., -p[:, delay_ticks+commit_ticks-1, 4].astype(float)))
    utility = progress-contact_penalty_m*contact
    selected = int(np.argmax(utility))
    result = dict(action=ACTIONS[selected], action_index=selected,
        requested_command=candidate_commands(ACTIONS[selected])[0],
        candidates=[dict(action=a, predicted_progress_during_commit_m=float(progress[i]),
            predicted_contact_by_commit_end=float(contact[i]), utility_m=float(utility[i]))
            for i,a in enumerate(ACTIONS)],
        common_prefix_max_absolute_spread=np.ptp(p[:, :delay_ticks],axis=0).max(axis=0).tolist(),
        dispatch_offset_ns=delay_ticks*PERIOD_NS, command_duration_ns=commit_ticks*PERIOD_NS,
        scoring_endpoint_offset_ns=(delay_ticks+commit_ticks)*PERIOD_NS,
        contact_penalty_m=float(contact_penalty_m), command_authorized=False,
        forecast_accuracy_under_delayed_dispatch_proven=False)
    if position_metric_matrix is not None:
        result['position_metric_matrix'] = np.asarray(position_metric_matrix).tolist()
        result['position_distance_metric'] = 'mission_initial_xy'
    return result


@dataclass(frozen=True)
class ScheduledCommand:
    observed_ns: int
    dispatch_ns: int
    expires_ns: int
    command: tuple

    @classmethod
    def prepare(cls, action, *, observed_ns, completed_ns, delay_ticks=DELAY_TICKS, commit_ticks=COMMIT_TICKS):
        if type(delay_ticks) is not int or delay_ticks not in (2,3):raise ValueError('two or three delay ticks required')
        if type(commit_ticks) is not int or commit_ticks not in (1,4):raise ValueError('one or four commitment ticks required')
        if (action not in ACTIONS or type(observed_ns) is not int or observed_ns < 0
                or type(completed_ns) is not int or completed_ns < observed_ns):
            raise ValueError('valid action and explicit causal completion clock required')
        dispatch = observed_ns+delay_ticks*PERIOD_NS
        if completed_ns > dispatch: return None
        return cls(observed_ns, dispatch, dispatch+commit_ticks*PERIOD_NS,
            tuple(candidate_commands(action)[0]))

    def request(self, *, now_ns, fresh_observation_allows_motion):
        if type(now_ns) is not int or type(fresh_observation_allows_motion) is not bool:
            raise ValueError('explicit dispatch clock and fresh-observation decision required')
        if self.dispatch_ns <= now_ns < self.expires_ns and fresh_observation_allows_motion:
            return list(self.command)
        return [0., 0., 0.]
