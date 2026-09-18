"""Admit the completed chained timing proof and unchanged native input."""
import json

from scripts import measured_plane_chained_full_history_inputs_development as native_inputs
from scripts import replay_go2_measured_plane_chained_single_pass_full_history_v1 as timing
from scripts.startup_source_inventory_development import discover_sources
from lewm.independent_reactive_floor_transport_study_development import merge_sources

run = native_inputs.run
SOURCE = 'scripts/extended_return_budget_prefix_inputs_development.py'
TEST = 'lewm/tests/test_extended_return_budget_prefix_inputs_development.py'
NATIVE_RESULT_SHA = '163e92d630d92f9f8633cd3823c7cbf282a92c81bde5d3db998e3a5a8c8b1849'
TIMING_LAUNCH_SHA = 'b8a67fc80aebb2f39968092067cea00b94304c449bd7c86fdff30458c58d6260'
TIMING_PID = 3015121
TIMING_CREATED = 1789209876.83
MODEL_SHA = native_inputs.native.worker_inputs.job.MODEL_SHA
TIMING_ARTIFACTS = {'launch.json','comparison.jsonl','state_checks.json','resource_monitor.jsonl','report.json'}


def timing_launch():
    run.verify_artifacts(timing.OUTPUT, {'launch.json':TIMING_LAUNCH_SHA})
    launch = run.read_json(timing.OUTPUT,'launch.json')
    if (launch['owner']['pid'] != TIMING_PID or launch['owner']['created'] != TIMING_CREATED
            or launch['boot_id'] != native_inputs.BOOT_ID
            or run.Path('/proc/sys/kernel/random/boot_id').read_text().strip() != native_inputs.BOOT_ID):
        raise ValueError('exact original chained timing owner, launch and boot required')
    run.verify(launch['source_sha256'])
    return launch


def prepared_sources(seeds=()):
    native = native_inputs.native_launch(); completed_or_live = timing_launch()
    sources = discover_sources((SOURCE,TEST,*seeds),merge_sources(
        native['source_sha256'],completed_or_live['source_sha256']))
    run.verify(sources)
    return sources


def admit(timing_sha,sources):
    launch = timing_launch(); root = timing.OUTPUT
    if run.owner_live(launch['owner']):
        raise ValueError('original chained timing owner must end before prefix admission')
    if (root/'failure.json').exists() or (root/'failure.json').is_symlink():
        raise ValueError('preserve terminal chained timing failure without retry or bypass')
    if (type(timing_sha) is not str or len(timing_sha) != 64
            or any(c not in '0123456789abcdef' for c in timing_sha)):
        raise ValueError('actual completed chained timing result SHA-256 required')
    run.verify_artifacts(root,{'result.json':timing_sha})
    result = run.read_json(root,'result.json')
    required = dict(status='MEASURED_PLANE_CHAINED_SINGLE_PASS_FULL_HISTORY_V1_COMPLETE',
        complete_output_and_public_packets_rechecked=True,
        original_raw_inputs_reauthenticated_before_and_after=True,
        original_physical_prefix_reconstructed_before_and_after=True,
        native_execution=False,navigation_outcomes_inferred=False,navigation_qualified=False,
        real_time_qualified=False,hardware_qualified=False,goal_achieved=False)
    if (any(type(result.get(k)) is not type(v) or result[k] != v for k,v in required.items())
            or result['source_sha256'] != launch['source_sha256']
            or set(result['artifact_sha256']) != TIMING_ARTIFACTS
            or result['artifact_sha256']['launch.json'] != TIMING_LAUNCH_SHA
            or any(sources.get(p) != sha for p,sha in launch['source_sha256'].items())):
        raise ValueError('complete immutable chained timing result and source ancestry required')
    identities = result['artifact_sha256'] | {'result.json':timing_sha}
    run.verify_artifacts(root,identities)
    report = run.read_json(root,'report.json')
    if run.canonical(report) != run.canonical(result['report']):
        raise ValueError('same complete saved timing report required')
    with (root/'comparison.jsonl').open() as stream:
        rows = [json.loads(line) for line in stream]
    reconstructed = timing.pair.comparison.summarize(rows,run.read_json(root,'state_checks.json'),
        frames=4014,model_sha=MODEL_SHA,input_result_sha=NATIVE_RESULT_SHA)
    if run.canonical(reconstructed) != run.canonical(report):
        raise ValueError('all timing decisions, state checkpoints and scientific scope must reconstruct')
    # Reauthenticate every native artifact and reconstruct its original physical
    # prefix. The completed timing science is not rerun; its complete scalar
    # accounting above and bound original packet-verification result are kept.
    admitted = native_inputs.admit(NATIVE_RESULT_SHA,sources)
    if run.canonical(admitted) != run.canonical(launch['input_admission']):
        raise ValueError('timing proof and prospective replay must use the identical native admission')
    run.verify(sources); run.verify_artifacts(root,identities)
    return admitted | dict(completed_chained_timing_result_sha256=timing_sha,
        completed_chained_timing_launch_sha256=TIMING_LAUNCH_SHA,
        completed_chained_timing_report_sha256=identities['report.json'],
        preceding_timing_owner_ended=True, complete_timing_accounting_reconstructed=True,
        preceding_timing_controller_execution_repeated=False)
