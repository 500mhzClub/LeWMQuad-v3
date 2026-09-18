"""Evaluator-only camera-context witnesses at existing acquisition boundaries."""
from copy import deepcopy
import time
import numpy as np
from lewm.novel_maze_round_trip_contract_development import MAX_OBSERVATIONS
from lewm_genesis.camera_renderer_identity_development import renderer_identity_readback
from lewm_genesis.core_raster_precision_development import precision_readback
from lewm_genesis.optical_camera_readback_development import check_optical_pose
from scripts.single_sample_rgbd_session_development import sampling_readback

PHASES = ('after_primary_capture', 'after_paired_capture_with_primary_pose_restored')
PRIMARY_HASHES = {'primary_rgb_sha256', 'primary_native_depth_sha256'}
PAIRED_HASHES = PRIMARY_HASHES | {'auxiliary_rgb_sha256', 'auxiliary_native_depth_sha256'}
SAMPLING = dict(draw_framebuffer_is_single_sample_target=True,
    draw_framebuffer_is_multisample_target=False, samples=0, sample_buffers=0,
    multisample_enabled=False, pixel_scale=1)


def _hashes(values, phase):
    if set(values) != (PRIMARY_HASHES if phase == PHASES[0] else PAIRED_HASHES):
        raise ValueError('exact acquisition pixel identities required')
    for value in values.values():
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('lowercase pixel SHA256 required')


def _state(session):
    camera = session.ctx.build.camera
    matrix = np.array(camera.transform, copy=True)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError('finite actual camera transform required')
    return int(session.ctx.runner._sim_time_ns), len(session.samples), matrix


def capture_witness(session, *, frame, phase, pixel_hashes):
    if phase not in PHASES or type(frame) is not int or not 0 <= frame < MAX_OBSERVATIONS:
        raise ValueError('bounded acquisition witness identity required')
    _hashes(pixel_hashes, phase)
    before = _state(session)
    if before[:2] != (1_500_000_000+100_000_000*frame, 750+50*frame):
        raise ValueError('actual complete maze observation boundary required')
    camera = session.ctx.build.camera; started = time.perf_counter_ns()
    context = dict(identity=renderer_identity_readback(camera),
        sampling=sampling_readback(camera), precision=precision_readback(camera))
    after = _state(session)
    if before[:2] != after[:2] or not np.array_equal(before[2], after[2]):
        raise ValueError('renderer query changed camera pose or physical state')
    if (context['sampling'] != SAMPLING or context['identity']['camera_uid'] != str(camera.uid)
            or context['identity']['framebuffer_matches_camera_depth_target'] is not True):
        raise ValueError('actual unchanged camera depth target required')
    return dict(frame=frame, phase=phase, measured_ns=before[0], physical_sample_index=before[1]-1,
        camera_transform=before[2].tolist(), pixel_hashes=deepcopy(pixel_hashes), context=context,
        query_wall_ms=(time.perf_counter_ns()-started)/1e6,
        camera_pose_and_physical_state_unchanged=True, render_calls_added=0,
        evaluator_only=True, historical_context_inferred=False, raster_error_bound_proven=False)


def validate_pair(primary, paired):
    if primary['phase'] != PHASES[0] or paired['phase'] != PHASES[1]:
        raise ValueError('both declared acquisition endpoints required')
    for key in ('frame', 'measured_ns', 'physical_sample_index', 'camera_transform', 'context'):
        if primary[key] != paired[key]: raise ValueError('paired renderer endpoint drift: '+key)
    for row in (primary, paired):
        _hashes(row['pixel_hashes'], row['phase'])
        if (row['camera_pose_and_physical_state_unchanged'] is not True
                or type(row['render_calls_added']) is not int or row['render_calls_added'] != 0
                or row['evaluator_only'] is not True or row['historical_context_inferred'] is not False
                or row['raster_error_bound_proven'] is not False):
            raise ValueError('observed evaluator-only readback without an inferred error bound required')
        if (row['context']['sampling'] != SAMPLING
                or row['context']['identity']['framebuffer_matches_camera_depth_target'] is not True
                or row['context']['identity']['raster_error_bound_proven'] is not False
                or row['context']['identity']['source_implementation_equivalence_proven'] is not False
                or not np.isfinite(row['query_wall_ms']) or row['query_wall_ms'] < 0):
            raise ValueError('measured camera context without unproved precision claims required')
    if any(primary['pixel_hashes'][key] != paired['pixel_hashes'][key] for key in PRIMARY_HASHES):
        raise ValueError('paired witness must bind the same primary pixels')


def audit_witnesses(document, captures):
    """Bindings must come from the separately verified actual raw acquisitions."""
    primary = document['primary']; paired = document['paired']
    if (document['failures'] or not 0 < len(captures) <= MAX_OBSERVATIONS
            or len(primary) != len(paired) or len(primary) != len(captures)):
        raise ValueError('complete successful renderer witnesses for every acquisition required')
    for frame, (a, b, capture) in enumerate(zip(primary, paired, captures, strict=True)):
        validate_pair(a, b)
        if (a['frame'] != frame or a['measured_ns'] != 1_500_000_000+100_000_000*frame
                or a['physical_sample_index'] != 749+50*frame
                or a['context'] != primary[0]['context']):
            raise ValueError('consecutive unchanged-context acquisition witnesses required')
        for key in ('frame', 'measured_ns', 'physical_sample_index'):
            if b[key] != capture[key]: raise ValueError('witness differs from audited acquisition: '+key)
        if b['pixel_hashes'] != {key:capture[key] for key in PAIRED_HASHES}:
            raise ValueError('witness pixel identities differ from audited raw capture')
        check_optical_pose(a['camera_transform'], capture['primary_world_from_optical'])
    return dict(frames=len(captures), capture_endpoints=2*len(captures),
        paired_context_readbacks_equal=True, all_witnesses_match_raw_acquisitions=True,
        evaluator_only=True, historical_context_inferred=False,
        per_draw_shader_arithmetic_reconstructed=False, raster_error_bound_proven=False,
        visibility_outcomes_replaced=False, navigation_qualified=False)
