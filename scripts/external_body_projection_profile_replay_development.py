"""Original complete replay with external-sampling markers and seven state witnesses.

No launch entry point. The caller must authenticate completed original inputs,
models and reference rows before replay, then verify the externally saved trace
only after both the Python child and profiler have ended.
"""
import builtins
import json
from types import FunctionType
from scripts import profile_go2_frozen_footprint_late_history_v1 as original
from scripts import replay_go2_body_projected_tiled_late_history_v1 as paired
from scripts.profile_go2_receipt_copied_footprint_late_history_v1 import compare_profile_rows
from scripts.progressive_batched_floor_state_development import normalized_state_tree as state_tree, STATE_TYPE_PATHS
from scripts.navigation_artifact_root_development import BASE
from lewm.body_projected_tiled_controller_development import BodyProjectedTiledController

SOURCE = 'scripts/external_body_projection_profile_replay_development.py'
TEST = 'lewm/tests/test_external_body_projection_profile_replay_development.py'
OUTPUT = BASE/'go2_body_projected_external_sampling_v1_attempt_001'
WINDOWS = original.WINDOWS
STATE_FRAMES = paired.original.STATE_FRAMES
normalize_candidate = paired.harness.normalize_candidate
MARKER_WINDOWS = {frame:name for name,(first,last) in WINDOWS.items() for frame in range(first,last+1)}
MARKERS = {frame:f'controller_observation_{frame:04d}' for frame in MARKER_WINDOWS}


def _marker_template(call, *args, **kwargs):
    return call(*args, **kwargs)


_MARKER_FUNCTIONS = {frame:FunctionType(_marker_template.__code__.replace(co_name=name, co_qualname=name),
    globals(), name) for frame,name in MARKERS.items()}


def observe_window(frame, window, call, *args, **kwargs):
    if type(frame) is not int or window != MARKER_WINDOWS.get(frame):
        raise ValueError('original frame and declared sampling window required')
    if frame in _MARKER_FUNCTIONS:
        return _MARKER_FUNCTIONS[frame](call, *args, **kwargs)
    return call(*args, **kwargs)


def progress(*args, **kwargs):
    if args and args[0] == 'LATE_HISTORY_CONTROLLER_PROFILE_FRAME':
        args = ('EXTERNAL_BODY_PROJECTION_REPLAY_FRAME', *args[1:])
    builtins.print(*args, **kwargs)


def replay_body():
    original = reference.original; case = reference.CASE
    model = original.assigned_model(read_json(original.OUTPUT, 'launch.json'), case)
    if state_digest(model.state_dict()) != reference.MODEL_SHA:
        raise ValueError('unchanged assigned JEPA model required')
    controller = FrozenFootprintAnchoredController(model, ArticulatedCollisionGeometry(reference.URDF),
        public_mission=public_mission(case[1]), navigation_ticks=NAVIGATION_TICKS,
        condition=case[3], variant=case[2], persistent=True)
    directory = original.OUTPUT/case[0]; reader = IntentReturnRGBDReplay(directory)
    acquisitions = read_json(directory, 'auxiliary_camera_audit.json'); tape = read_json(directory, 'command_tape.json')
    captured = {name:[] for name in WINDOWS}
    count = forecasts = 0; sizes = {}; states = []
    with (OUTPUT/'comparison.jsonl').open('x') as stream:
        for row in islice(read_rows(directory), FRAMES):
            frame = row['tick']
            if (frame != count or tape[frame]['tick'] != frame or tape[frame]['completed'] is not True
                    or tape[frame]['pre_sample_index'] != 749+50*frame or tape[frame]['post_sample_index'] != 799+50*frame):
                raise ValueError('ordered completed original command endpoints required')
            p,d,fast,now = reader.packet(frame)
            image,aux = packet(directory,frame,p,public_acquisition(acquisitions[frame]),now_ns=now)
            before = fingerprint((p,d,fast,aux,image,now))
            window = next((n for n,(first,last) in WINDOWS.items() if first <= frame <= last), None)
            start = time.perf_counter()
            actual = observe_window(frame, window, controller.observe, p,d,fast,now_ns=now,auxiliary_depth=aux,auxiliary_rgb=image)
            elapsed = time.perf_counter()-start
            actual = json.loads(json.dumps(actual))
            candidate_sha = reference.saved.identity(actual)
            actual = normalize_candidate(actual)
            if (actual != row['decision'] or actual['requested_command'] != tape[frame]['requested_command']
                    or actual['terminal'] is not None or before != fingerprint((p,d,fast,aux,image,now))):
                raise ValueError('complete original decision or public inputs changed at frame '+str(frame))
            selection = actual['new_selection']; action = None if selection is None else selection['action']
            if window == 'repeated_hold' and (action != 'hold' or actual['mission_receipt']['hold_required']):
                raise ValueError('original discretionary hold window required')
            if window:
                captured[window].append(dict(frame=frame, action=action, controller_wall_s_with_profiling=elapsed))
                if frame == WINDOWS[window][1]: sizes[window] = state_sizes(controller)
            stream.write(json.dumps(dict(frame=frame, original_decision_sha256=reference.saved.identity(actual),
                candidate_decision_sha256=candidate_sha, candidate_normalized_decision_exact=True,
                public_input_sha256=before, complete_original_decision_reconstructed=True,
                public_input_arrays_unchanged=True, profiled_window=window,
                controller_wall_s=elapsed), allow_nan=False)+'\n')
            if frame in STATE_FRAMES:
                state_sha = fingerprint(state_tree(dict(memory=controller.memory, floor=controller.mapper.floor,
                    occupied=controller.mapper.occupied, residual=controller.residual, history=controller.history)))
                states.append(dict(frame=frame, state_sha256=state_sha, retained_observed_state_equal=True))
            count += 1; forecasts += int(bool(selection and 'prediction' in selection))
            if frame % 50 == 0: print('LATE_HISTORY_CONTROLLER_PROFILE_FRAME',frame,flush=True)
    if count != FRAMES or forecasts != FRAMES-3:
        raise ValueError('complete 1428-observation history and 1425 forecasts required')
    if state_digest(model.state_dict()) != reference.MODEL_SHA or any(p.grad is not None for p in model.parameters()):
        raise ValueError('profiled model state or gradients changed')
    summaries = {}
    for name, (first,last) in WINDOWS.items():
        if [r['frame'] for r in captured[name]] != list(range(first,last+1)):
            raise ValueError('all three fixed ten-observation windows required')
        summaries[name] = dict(observations=captured[name],
            python_stack_markers=[MARKERS[frame] for frame in range(first,last+1)])
    return dict(frames=count, raw_model_forecast_comparisons=forecasts, windows=summaries, observed_state_checks=states,
        state_size_snapshots=sizes, retained_state_identity_established=False,
        complete_original_decisions_reconstructed=True, model_state_sha256=reference.MODEL_SHA,
        model_state_unchanged=True, sensor_acquisition_profiled=True, controller_observe_only_profiled=False,
        last_replayed_observation=FRAMES-1, no_observation_1428_consumed=True,
        profiler_overhead_removed=False, isolated_benchmark=False, speedup_established=False,
        native_execution=False, policy_changed=False, real_time_qualified=False, navigation_qualified=False,
        invocation_frozen_footprint_receipts=True, normalization_outside_profiled_region=False,
        complete_normalized_candidate_decisions_exact=True)


def isolated_replay():
    bindings = dict(OUTPUT=OUTPUT, FrozenFootprintAnchoredController=BodyProjectedTiledController,
        normalize_candidate=normalize_candidate, observe_window=observe_window, MARKERS=MARKERS,
        STATE_FRAMES=STATE_FRAMES, state_tree=state_tree, print=progress)
    return FunctionType(replay_body.__code__, original.replay.__globals__ | bindings,
        replay_body.__name__, replay_body.__defaults__)


def replay(prior_rows, prior_report):
    report = isolated_replay()()
    rows = [json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
    compare_profile_rows(rows, prior_rows)
    if (report['observed_state_checks'] != prior_report['observed_state_checks']
            or [row['frame'] for row in report['observed_state_checks']] != list(STATE_FRAMES)):
        raise ValueError('all seven original retained-state identities required')
    return report | dict(controller='BodyProjectedTiledController',
        retained_state_identity_established=True, normalized_state_type_paths=STATE_TYPE_PATHS,
        external_profiler_covers_entire_child=True, controller_window_markers_provided=True,
        normalization_outside_marked_controller_windows=True, cprofile_hooks_enabled=False,
        external_profile_file_verified=False)
