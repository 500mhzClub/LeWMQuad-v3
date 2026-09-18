"""Execute one complete, admitted chained-controller timing comparison."""
import argparse
import json
from pathlib import Path
import time

import psutil
import torch

from scripts import measured_plane_chained_full_history_inputs_development as inputs
from scripts import measured_plane_chained_full_history_replay_development as pair

run = inputs.run
SOURCE = 'scripts/replay_go2_measured_plane_chained_single_pass_full_history_v1.py'
PROTOCOL = 'docs/go2_measured_plane_chained_single_pass_full_history_v1_2026-09-12.md'
TESTS = (inputs.TEST,
    'lewm/tests/test_measured_plane_chained_full_history_timing_development.py',
    'lewm/tests/test_measured_plane_chained_full_history_replay_development.py',
    'lewm/tests/test_measured_plane_chained_single_pass_controller_development.py',
    'lewm/tests/test_measured_plane_chained_full_history_launcher_development.py')
OUTPUT = run.BASE/'go2_measured_plane_chained_single_pass_full_history_v1_attempt_001'


def cpu_idle():
    """Serialize this pair with the repository's named CPU replay workers."""
    inputs.native.original.require_native_idle()
    own = psutil.Process().pid
    prefixes = ('replay_go2_', 'probe_go2_', 'run_go2_single_pass_', 'run_go2_deferred_memo_')
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.info['pid'] == own:
            continue
        for argument in process.info['cmdline'] or ():
            name = Path(argument).name
            if name.endswith('.py') and name.startswith(prefixes):
                raise ValueError('existing CPU replay must end before dispatch: '+str(process.info['pid']))


def resources():
    return pair.previous.resources()


def main(result_sha=None, source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic single-thread CPU environment required')
    run.validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive chained timing attempt; no retry, resume or overwrite')
    sources = inputs.prepared_sources((SOURCE, PROTOCOL, *TESTS))
    if source_only:
        print('CHAINED_FULL_HISTORY_SOURCE_PREFLIGHT', len(sources), json.dumps(run.hardware()), flush=True)
        return
    if not result_sha:
        raise ValueError('actual completed chained native result SHA required')
    resources(); cpu_idle()
    admission = inputs.admit(result_sha, sources)
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    hardware = resources(); cpu_idle()
    run.create_output(OUTPUT)
    process = psutil.Process()
    run.write_json(OUTPUT/'launch.json', dict(source_sha256=sources, protocol=PROTOCOL,
        input_admission=admission, hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid, created=process.create_time(), command=process.cmdline()),
        state_frames=pair.comparison.state_frames(admission['frames']),
        complete_actual_native_population_required=True, original_native_owners_ended=True,
        cpu_replay_serialization_checked=True, native_execution=False, automatic_retry=False))
    print('CHAINED_FULL_HISTORY_LAUNCHED', run.digest(OUTPUT/'launch.json'), flush=True)
    start = time.perf_counter()
    try:
        report = pair.replay(admission, output=OUTPUT)
        pair.check_output(report, admission, output=OUTPUT)
        if run.canonical(inputs.admit(result_sha, sources)) != run.canonical(admission):
            raise ValueError('complete original chained inputs changed during replay')
        run.write_json(OUTPUT/'report.json', report)
        ids = {name:run.digest(OUTPUT/name) for name in (
            'launch.json', 'comparison.jsonl', 'state_checks.json', 'resource_monitor.jsonl', 'report.json')}
        run.verify_artifacts(OUTPUT, ids); run.verify(sources)
        run.write_json(OUTPUT/'result.json', dict(status='MEASURED_PLANE_CHAINED_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE',
            source_sha256=sources, artifact_sha256=ids, report=report,
            wall_s=time.perf_counter()-start, complete_output_and_public_packets_rechecked=True,
            original_raw_inputs_reauthenticated_before_and_after=True,
            original_physical_prefix_reconstructed_before_and_after=True,
            native_execution=False, navigation_outcomes_inferred=False, navigation_qualified=False,
            real_time_qualified=False, hardware_qualified=False, goal_achieved=False))
        print('CHAINED_FULL_HISTORY_COMPLETE', run.digest(OUTPUT/'result.json'),
            report['timing']['all_observations'], flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json', dict(status='TERMINAL_CHAINED_FULL_HISTORY_TIMING_FAILURE',
            reason=repr(error), automatic_retry=False, original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--chained-native-result-sha256')
    parser.add_argument('--source-preflight-only', action='store_true')
    args = parser.parse_args()
    main(args.chained_native_result_sha256, args.source_preflight_only)
