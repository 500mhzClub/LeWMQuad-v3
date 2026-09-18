"""Exclusive prospective 8,000-step controller-prefix replay after timing proof."""
import argparse
import json
import time

import psutil
import torch

from scripts import extended_return_budget_prefix_inputs_development as inputs
from scripts import extended_return_budget_prefix_replay_development as pair

run = inputs.run
SOURCE = 'scripts/replay_go2_extended_return_budget_controller_prefix_v1.py'
PROTOCOL = 'docs/go2_extended_return_budget_controller_prefix_v1_2026-09-12.md'
TESTS = (inputs.TEST,
    'lewm/tests/test_extended_return_budget_prefix_comparison_development.py',
    'lewm/tests/test_extended_return_budget_prefix_replay_development.py',
    'lewm/tests/test_extended_return_budget_prefix_launcher_development.py',
    'lewm/tests/test_extended_return_budget_controller_development.py',
    'lewm/tests/test_extended_return_budget_memory_development.py',
    'lewm/tests/test_extended_return_budget_transport_development.py',
    'lewm/tests/test_extended_return_budget_mission_development.py')
OUTPUT = run.BASE/'go2_extended_return_budget_controller_prefix_v1_attempt_001'
resources = inputs.timing.resources
cpu_idle = inputs.timing.cpu_idle


def main(timing_sha=None,source_only=False):
    if (not __debug__ or any(run.os.environ.get(k) != v for k,v in run.ENV.items())
            or run.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic single-thread CPU environment required')
    run.validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive extended prefix attempt; no retry, resume or overwrite')
    sources = inputs.prepared_sources((SOURCE,PROTOCOL,*TESTS))
    if source_only:
        print('EXTENDED_RETURN_PREFIX_SOURCE_PREFLIGHT',len(sources),json.dumps(run.hardware()),flush=True)
        return
    if not timing_sha:
        raise ValueError('actual completed chained timing result SHA required')
    resources(); cpu_idle()
    admission = inputs.admit(timing_sha,sources)
    pair.require_admission(admission)
    run.cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    hardware = resources(); cpu_idle()
    run.create_output(OUTPUT); process = psutil.Process()
    run.write_json(OUTPUT/'launch.json',dict(source_sha256=sources,protocol=PROTOCOL,
        input_admission=admission,hardware=hardware,
        boot_id=run.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=process.pid,created=process.create_time(),command=process.cmdline()),
        baseline_navigation_ticks=4000,candidate_navigation_ticks=8000,
        maximum_prefix_observations=pair.comparison.MAX_PREFIX_OBSERVATIONS,
        planned_state_frames=pair.expected_states(pair.comparison.MAX_PREFIX_OBSERVATIONS),
        stop_at_first_normalized_decision_difference=True,
        preceding_timing_and_native_owners_ended=True,cpu_replay_serialization_checked=True,
        native_execution=False,automatic_retry=False))
    print('EXTENDED_RETURN_PREFIX_LAUNCHED',run.digest(OUTPUT/'launch.json'),flush=True)
    start = time.perf_counter()
    try:
        report = pair.replay(admission,output=OUTPUT)
        pair.check_output(report,admission,output=OUTPUT)
        if run.canonical(inputs.admit(timing_sha,sources)) != run.canonical(admission):
            raise ValueError('completed native or timing inputs changed during prefix replay')
        run.write_json(OUTPUT/'report.json',report)
        identities = {name:run.digest(OUTPUT/name) for name in ('launch.json',pair.pipeline.stream.NAME,
            'state_checks.json','identities.json','resource_monitor.jsonl','report.json')}
        run.verify_artifacts(OUTPUT,identities); run.verify(sources)
        run.write_json(OUTPUT/'result.json',dict(status='EXTENDED_RETURN_BUDGET_CONTROLLER_PREFIX_V1_COMPLETE',
            source_sha256=sources,artifact_sha256=identities,report=report,wall_s=time.perf_counter()-start,
            completed_chained_timing_result_sha256=timing_sha,
            original_native_result_sha256=inputs.NATIVE_RESULT_SHA,
            complete_output_and_public_packets_rechecked=True,
            original_native_and_timing_inputs_reauthenticated_before_and_after=True,
            original_physical_prefix_reconstructed_before_and_after=True,
            scientific_budget_only_prefix_supported=report['budget_only_preboundary_decisions_supported'],
            changed_command_executed=False,native_execution=False,automatic_retry=False,
            navigation_qualified=False,real_time_qualified=False,hardware_qualified=False,goal_achieved=False))
        print('EXTENDED_RETURN_PREFIX_COMPLETE',run.digest(OUTPUT/'result.json'),
            report['frames'],report['boundary']['stop_reason'],flush=True)
    except BaseException as error:
        run.write_json(OUTPUT/'failure.json',dict(status='TERMINAL_EXTENDED_RETURN_PREFIX_FAILURE',
            reason=repr(error),automatic_retry=False,original_evidence_preserved=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--completed-chained-timing-result-sha256')
    parser.add_argument('--source-preflight-only',action='store_true')
    args = parser.parse_args()
    main(args.completed_chained_timing_result_sha256,args.source_preflight_only)
