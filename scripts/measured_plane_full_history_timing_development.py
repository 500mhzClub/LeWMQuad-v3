"""Complete-population accounting for a semantics-preserving timing replay."""
import math
import statistics
from types import FunctionType

import cv2

from lewm.measured_floor_transport_registration_development import MeasuredFloorTransportRegistration
from lewm.tiled_density_floor_registration_development import TiledDensityFloorRegistration
from scripts.measured_plane_single_pass_comparison_development import normalize, state, STATE_TYPE_PATHS
from scripts.replay_go2_residual_current_observation_planning_prefix_v1 import state_tree

MAX_FRAMES = 4014
FIXED_STATE_FRAMES = (0, 3, 61, 122, 255, 511, 1023, 2047, 3071)
WINDOWS = ((0, 256), (256, 1024), (1024, 2048), (2048, 3072), (3072, MAX_FRAMES))


def observer_state_tree(value):
    if type(value) is cv2.KeyPoint:
        return dict(type='cv2.KeyPoint', fields=dict(pt=list(value.pt), size=value.size,
            angle=value.angle, response=value.response, octave=value.octave, class_id=value.class_id))
    return _observer_state_tree(value)


_observer_state_tree = FunctionType(state_tree.__code__, state_tree.__globals__ | {'state_tree': observer_state_tree},
    state_tree.__name__, state_tree.__defaults__, state_tree.__closure__)


def state_frames(count):
    if type(count) is not int or not 4 <= count <= MAX_FRAMES:
        raise ValueError('complete admitted four-to-4014 observation population required')
    return sorted({count-1, *(f for f in FIXED_STATE_FRAMES if f < count)})


def observed_state(controller):
    # The existing ten type normalizations apply only to the declared retained
    # map/residual/history state. The existing optimized registration adds one
    # explicit implementation type tag; retain every registration field.
    kind = type(controller.registration)
    if kind not in (MeasuredFloorTransportRegistration, TiledDensityFloorRegistration):
        raise ValueError('exact original or existing tiled registration implementation required')
    observer = observer_state_tree(dict(motion=controller.motion,
        registration=controller.registration, mission=controller.mission))
    registration = observer['registration']
    if set(registration) != {'type', 'fields'} or registration['type'] != kind.__module__+'.'+kind.__name__:
        raise ValueError('complete explicitly typed registration state required')
    registration['type'] = MeasuredFloorTransportRegistration.__module__+'.'+MeasuredFloorTransportRegistration.__name__
    return dict(retained_map_residual_history=state(controller),
        observer_registration_mission=observer)


def reference_endpoint(row, tape, frame, count):
    state_frames(count)
    if (row['tick'] != frame or row['observation_index'] != frame
            or row['pre_sample_index'] != 749+50*frame or len(tape) not in (count-1, count)):
        raise ValueError('complete original physical observation endpoints required')
    if frame == len(tape):
        if frame != count-1 or row['decision']['terminal'] is None:
            raise ValueError('only final terminal observation may omit a command')
        return
    command = tape[frame]; end = command['post_sample_index']
    if (command['tick'] != frame or command['pre_sample_index'] != 749+50*frame
            or command['requested_command'] != row['decision']['requested_command']
            or type(end) is not int or not 749+50*frame <= end <= 799+50*frame
            or type(command['completed']) is not bool):
        raise ValueError('same actual command and physical interval required')
    if command['completed']:
        if end != 799+50*frame: raise ValueError('completed command requires every physical sample')
    elif frame != count-1:
        raise ValueError('no observation may follow an incomplete physical command')


def timings(rows):
    before = [r['baseline_controller_s'] for r in rows]
    after = [r['candidate_controller_s'] for r in rows]
    def percentile(values):
        if not values: return None
        ordered = sorted(values); position = .95*(len(ordered)-1)
        low = int(position); high = min(low+1, len(ordered)-1)
        return ordered[low]+(position-low)*(ordered[high]-ordered[low])
    return dict(observations=len(rows), baseline_total_s=math.fsum(before), candidate_total_s=math.fsum(after),
        baseline_median_s=statistics.median(before) if before else None,
        candidate_median_s=statistics.median(after) if after else None,
        baseline_p95_s=percentile(before), candidate_p95_s=percentile(after),
        baseline_over_100ms=sum(v > .1 for v in before), candidate_over_100ms=sum(v > .1 for v in after))


def summarize(rows, states, *, frames, model_sha, input_result_sha):
    expected_states = state_frames(frames)
    if len(rows) != frames or [r['frame'] for r in rows] != list(range(frames)):
        raise ValueError('all admitted observations must be compared in original order')
    calls = [0, 0]
    for row in rows:
        frame = row['frame']
        if (row['execution_order'] != ([0, 1] if frame % 2 == 0 else [1, 0])
                or row['complete_original_decision_equal'] is not True
                or row['complete_normalized_candidate_equal'] is not True
                or row['public_inputs_unchanged'] is not True
                or not row['original_decision_sha256'] == row['baseline_decision_sha256'] == row['normalized_candidate_decision_sha256']):
            raise ValueError('complete equal decisions and alternating execution order required')
        forwards = row['actual_model_forward_calls']
        if (not isinstance(forwards, list) or len(forwards) != 2
                or any(type(v) is not int or v not in (0, 1) for v in forwards)
                or forwards[0] != forwards[1]
                or row['forecast_compared'] is not bool(forwards[0])):
            raise ValueError('same actual model forward population in both arms required')
        calls = [a+b for a, b in zip(calls, forwards, strict=True)]
        for key in ('baseline_controller_s', 'candidate_controller_s'):
            value = row[key]
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError('finite positive measured controller times required')
    if ([s['frame'] for s in states] != expected_states
            or any(s['observed_state_equal'] is not True or not isinstance(s['state_sha256'], str)
                or len(s['state_sha256']) != 64 for s in states)):
        raise ValueError('all fixed state checkpoints and actual final state required')
    return dict(frames=frames, input_result_sha256=input_result_sha, complete_actual_native_history=True,
        actual_model_forward_calls=calls, raw_model_forecast_comparisons=calls[0],
        observed_state_checks=states,
        normalized_state_type_paths=['retained_map_residual_history.'+p for p in STATE_TYPE_PATHS]
            + ['observer_registration_mission.registration.type'],
        motion_and_mission_types_normalized=False, registration_implementation_type_normalized=True,
        baseline='MeasuredPlaneResidualController', candidate='MeasuredPlaneSinglePassController',
        timing=dict(all_observations=timings(rows), forecasts=timings([r for r in rows if r['forecast_compared']]),
            fixed_frame_windows=[dict(start=a, stop=min(b, frames), **timings(rows[a:min(b, frames)]))
                for a, b in WINDOWS if a < frames]),
        model_state_sha256=model_sha, model_states_unchanged=True,
        controller_observe_only_timed=True, sensor_acquisition_timed=False,
        isolated_benchmark=False, automatic_retry=False, native_execution=False,
        navigation_outcomes_inferred=False, navigation_qualified=False, real_time_qualified=False, goal_achieved=False)
