"""One exclusive compact-format check on four declared recorded observations."""
import json
import time

import numpy as np
import psutil

from lewm.causal_depth_observation_development import from_native_depth as primary_packet
from lewm.auxiliary_downward45_depth_observation_development import from_native_depth as auxiliary_packet
from lewm.causal_auxiliary_rgb_observation_development import from_captured_rgb
from scripts import compact_native_depth_archive_development as compact
from scripts import replay_go2_extended_return_budget_controller_prefix_v1 as prefix
from scripts import extended_budget_anchored_maze_development as packets
from scripts.novel_maze_auxiliary_rgb_packet_development import public_acquisition

run = prefix.run
SOURCE = 'scripts/check_go2_compact_depth_recorded_sample_v1.py'
PROTOCOL = 'docs/go2_compact_depth_recorded_sample_v1_2026-09-12.md'
SEEDS = (SOURCE, PROTOCOL, 'scripts/compact_native_depth_archive_development.py',
    'lewm/tests/test_compact_native_depth_archive_development.py',
    'docs/go2_compact_depth_recorded_sample_preflight_correction_2026-09-12.md')
FRAMES = (0, 3062, 4003, 4013)
NATIVE_RESULT_SHA = '163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849'
NATIVE_ROOT = run.BASE/'go2_measured_plane_chained_maze02_v1_attempt_001'
CASE = 'no_rgb_direct_measured_plane_chained_maze_02'
INPUT = NATIVE_ROOT/CASE
OUTPUT = run.BASE/'go2_compact_depth_recorded_sample_v1_attempt_001'
MAX_OUTPUT_BYTES = 64*1024**2


def input_identities():
    run.verify_artifacts(NATIVE_ROOT, {'result.json': NATIVE_RESULT_SHA})
    result = run.read_json(NATIVE_ROOT, 'result.json')
    if result['status'] != 'MEASURED_PLANE_CHAINED_MAZE02_V1_COMPLETE':
        raise ValueError('exact completed native input required')
    names = ['policy_observations.json', 'policy_histories.npz', 'depth_observations.json',
        'fast_gyro_histories.npz', 'depth_camera_audit.json', 'auxiliary_camera_audit.json']
    for frame in FRAMES:
        names.extend(f'{stem}_{frame:04d}.{suffix}' for stem, suffix in (
            ('rgb', 'png'), ('depth', 'npz'), ('native_depth', 'npz'),
            ('auxiliary_depth', 'npz'), ('auxiliary_rgb', 'png')))
    identities = {name: result['artifact_sha256'][CASE+'/'+name] for name in names}
    run.verify_artifacts(NATIVE_ROOT, {CASE+'/'+name: sha for name, sha in identities.items()})
    return identities


def compare(reader, acquisitions, depth_audit, frame):
    policy, old_primary, fast, now = reader.packet(frame)
    old_image, old_auxiliary = packets.rgb_packet(INPUT, frame, policy,
        public_acquisition(acquisitions[frame]), now_ns=now)
    original_packet = (policy, old_primary, fast, old_image, old_auxiliary)
    before = run.fingerprint(original_packet)
    with np.load(INPUT/f'native_depth_{frame:04d}.npz', allow_pickle=False) as archive:
        native_primary = archive['optical_depth_m']
    with np.load(INPUT/f'auxiliary_depth_{frame:04d}.npz', allow_pickle=False) as archive:
        native_auxiliary = archive['native_optical_depth_m']
        segmentation = archive['diagnostic_segmentation']
    if (compact.digest(native_primary.tobytes()) != depth_audit[frame]['native_depth_sha256']
            or compact.digest(native_auxiliary.tobytes()) != acquisitions[frame]['native_depth_sha256']
            or compact.digest(segmentation.tobytes()) != acquisitions[frame]['diagnostic_segmentation_sha256']):
        raise ValueError('actual native depth and segmentation acquisition identities required')
    primary, _ = compact.write(OUTPUT, role='primary', frame=frame, native_depth=native_primary)
    auxiliary, evaluator = compact.write(OUTPUT, role='auxiliary', frame=frame,
        native_depth=native_auxiliary, diagnostic_segmentation=segmentation)
    decoded_primary = compact.read_native(OUTPUT, primary)
    decoded_auxiliary = compact.read_native(OUTPUT, auxiliary)
    decoded_segmentation = compact.read_evaluator_segmentation(OUTPUT, auxiliary, evaluator)
    if (decoded_primary.tobytes() != native_primary.tobytes()
            or decoded_auxiliary.tobytes() != native_auxiliary.tobytes()
            or decoded_segmentation.dtype != segmentation.dtype
            or decoded_segmentation.tobytes() != segmentation.tobytes()):
        raise ValueError('complete native array bits must survive compact storage')
    new_primary = primary_packet(decoded_primary, policy, measured_ns=old_primary['measured_ns'],
        available_ns=old_primary['available_ns'], now_ns=now)
    new_auxiliary = auxiliary_packet(decoded_auxiliary, policy,
        measured_ns=old_auxiliary['measured_ns'], available_ns=old_auxiliary['available_ns'], now_ns=now)
    new_image = from_captured_rgb(old_image['rgb'], new_auxiliary, policy,
        measured_ns=old_image['measured_ns'], available_ns=old_image['available_ns'], now_ns=now)
    after = run.fingerprint((policy, new_primary, fast, new_image, new_auxiliary))
    if before != after or run.fingerprint(original_packet) != before:
        raise ValueError('complete actual public packets must match without input mutation')
    legacy_bytes = sum((INPUT/f'{stem}_{frame:04d}.npz').stat().st_size
        for stem in ('native_depth', 'depth', 'auxiliary_depth'))
    return dict(frame=frame, observation_now_ns=now, primary_binding=primary,
        auxiliary_binding=auxiliary, evaluator_binding=evaluator,
        complete_public_packet_sha256=before, complete_public_packets_equal=True,
        raw_depth_and_segmentation_bits_equal=True, original_public_inputs_unchanged=True,
        legacy_depth_archive_bytes=legacy_bytes,
        compact_depth_archive_bytes=primary['archive_bytes']+auxiliary['archive_bytes'])


def main():
    if any(run.os.environ.get(k) != v for k, v in run.ENV.items()) or run.cv2.ocl.useOpenCL():
        raise ValueError('original single-thread environment and disabled OpenCL required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive sample check; no retry, overwrite or resume')
    sources = prefix.inputs.prepared_sources(SEEDS)
    identities = input_identities()
    hardware = run.hardware()
    if psutil.virtual_memory().available < 8*1024**3 or psutil.disk_usage(run.BASE).free < 40*1024**3+MAX_OUTPUT_BYTES:
        raise ValueError('sample-check memory and artifact reserve required')
    run.create_output(OUTPUT); owner = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, protocol=PROTOCOL,
        original_native_result_sha256=NATIVE_RESULT_SHA, selected_input_sha256=identities,
        frames=list(FRAMES), maximum_output_bytes=MAX_OUTPUT_BYTES, hardware=hardware,
        owner=dict(pid=owner.pid, created=owner.create_time(), command=owner.cmdline()),
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        concurrent_controller_prefix_replay=run.owner_live(run.read_json(prefix.OUTPUT, 'launch.json')['owner']),
        performance_benchmark=False,
        controller_execution=False, native_execution=False, automatic_retry=False))
    print('COMPACT_DEPTH_RECORDED_SAMPLE_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        reader = packets.ExtendedBudgetRGBDReplay(INPUT)
        acquisitions = run.read_json(INPUT, 'auxiliary_camera_audit.json')
        depth_audit = run.read_json(INPUT, 'depth_camera_audit.json')
        if len(reader.frames) != 4014 or len(acquisitions) != 4014 or len(depth_audit) != 4014:
            raise ValueError('complete original sensor population required')
        rows = [compare(reader, acquisitions, depth_audit, frame) for frame in FRAMES]
        report = dict(frames=list(FRAMES), observations=len(rows), rows=rows,
            legacy_depth_archive_bytes=sum(r['legacy_depth_archive_bytes'] for r in rows),
            compact_depth_archive_bytes=sum(r['compact_depth_archive_bytes'] for r in rows),
            complete_sample_public_packets_equal=True, raw_sample_arrays_bitwise_equal=True,
            complete_original_population_compared=False, sensor_capture_executed=False,
            sensor_latency_measured=False, model_loaded=False, controller_execution=False,
            native_execution=False, format_adopted=False, existing_artifacts_modified_or_deleted=False,
            independent_layout_evidence=False, goal_achieved=False)
        if input_identities() != identities:
            raise ValueError('selected actual input identities changed')
        run.verify(sources)
        run.write_json(OUTPUT/'report.json', report)
        artifacts = {'launch.json': run.digest(OUTPUT/'launch.json'),
            'report.json': run.digest(OUTPUT/'report.json')}
        for row in rows:
            for key in ('primary_binding', 'auxiliary_binding'):
                binding = row[key]
                compact.read_native(OUTPUT, binding)
                artifacts[binding['filename']] = binding['archive_sha256']
            compact.read_evaluator_segmentation(OUTPUT, row['auxiliary_binding'], row['evaluator_binding'])
        run.verify_artifacts(OUTPUT, artifacts)
        output_bytes = sum((OUTPUT/name).stat().st_size for name in artifacts)
        if output_bytes > MAX_OUTPUT_BYTES-1024**2:
            raise ValueError('retain bounded terminal result headroom')
        terminal = dict(status='COMPACT_DEPTH_RECORDED_SAMPLE_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=artifacts, report=report,
            original_native_result_sha256=NATIVE_RESULT_SHA, selected_input_sha256=identities,
            output_bytes_without_result=output_bytes, wall_s=time.perf_counter()-start,
            complete_sample_output_rechecked=True, source_and_selected_inputs_reauthenticated=True,
            automatic_retry=False, native_execution=False, format_adopted=False, goal_achieved=False)
        if len(json.dumps(terminal, indent=2, sort_keys=True, allow_nan=False).encode())+1 > 1024**2:
            raise ValueError('bounded terminal result serialization required')
        run.write_json(OUTPUT/'result.json', terminal)
        print('COMPACT_DEPTH_RECORDED_SAMPLE_COMPLETE', run.digest(OUTPUT/'result.json'),
            report['legacy_depth_archive_bytes'], report['compact_depth_archive_bytes'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_COMPACT_DEPTH_SAMPLE_FAILURE',
            reason=repr(error), automatic_retry=False, existing_evidence_preserved=True))
        raise


if __name__ == '__main__':
    main()
