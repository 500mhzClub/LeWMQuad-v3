"""Full paired controller replay following the verified single-pass baseline."""
import argparse
from copy import deepcopy
import json
import time
import psutil
from scripts import deferred_memo_single_pass_replay_development as harness
from scripts import run_go2_single_pass_body_projected_late_history_v1 as previous
from scripts import verify_go2_measured_plane_controller_prefix_v1 as last_cpu
from scripts.verify_go2_body_projected_tiled_controller_completion_v1 import check_rows
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT,digest,verify,write_json
from scripts.navigation_artifact_root_development import validate_root,create_output,verify_artifacts
from scripts.startup_source_inventory_development import discover_sources
from scripts.audit_go2_independent_pulse_context_pilot_v1 import fingerprint

SOURCE='scripts/run_go2_deferred_memo_single_pass_late_history_v1.py'
TEST='lewm/tests/test_deferred_memo_single_pass_runner_development.py'
PROTOCOL='docs/go2_deferred_memo_single_pass_late_history_v1_2026-09-11.md'
OUTPUT=harness.OUTPUT
COMPLETION_SHA='f9833329610096f9d5776ccc208d0de33988d8ebd4e6cdff4895004aa5190f1e'
RESULT_SHA='6144645f568d99ccf89b31a76e586277a315f840e2da6db835c06d0519150c44'
LAST_CPU_SHA='ca03a74ce91eba199ae485acc6ee5729872557e428b42e8ef805a99135dc4bbf'
BENCHMARK='docs/go2_deferred_atomic_memo_copy_component_benchmark_2026-09-11.json'
BENCHMARK_SHA='a54320b92c7272a90b6c2fe406098dfc12b9e958a0083a2024e84f439e5ae405'


def reference():
    name=str(previous.COMPLETION.relative_to(ROOT))
    verify({name:COMPLETION_SHA})
    proof=json.loads(previous.COMPLETION.read_text())
    if (proof['status'] != 'SINGLE_PASS_BODY_PROJECTED_COMPLETION_VERIFIED'
            or proof['result_sha256'] != RESULT_SHA or proof['original_owner_ended'] is not True
            or previous.owner_live(proof['original_owner'])
            or proof['complete_rows'] != 1428 or proof['raw_model_forecasts'] != 1425):
        raise ValueError('same completed single-pass baseline and ended original owner required')
    verify(proof['source_sha256']);verify_artifacts(previous.OUTPUT,proof['artifact_sha256'] | {'result.json':RESULT_SHA})
    return proof


def admit():
    proof=reference()
    result=json.loads((previous.OUTPUT/'result.json').read_text())
    rows=[json.loads(line) for line in (previous.OUTPUT/'comparison.jsonl').read_text().splitlines()]
    original=previous.private_admission()
    previous.validate_report(result['report'],rows,original)
    if (result['sensing_scope'] != proof['sensing_scope']
            or result['report']['observed_state_checks'] != proof['observed_state_checks']):
        raise ValueError('same full predecessor evidence and negative sensing scope required')
    if reference() != proof: raise ValueError('baseline reference changed during admission')
    return proof,result,rows


def sources():
    proof=reference();verify({BENCHMARK:BENCHMARK_SHA})
    bindings=discover_sources((SOURCE,TEST,PROTOCOL,harness.TEST,BENCHMARK,
        str(previous.COMPLETION.relative_to(ROOT)),
        'lewm/tests/test_deferred_atomic_memo_copy_development.py',
        'lewm/tests/test_deferred_memo_single_pass_controller_development.py'),proof['source_sha256'])
    verify(bindings)
    return bindings


def slot_available():
    previous.slot_available()
    verify({str(last_cpu.OUTPUT.relative_to(ROOT)):LAST_CPU_SHA})
    verify_artifacts(last_cpu.job.OUTPUT,{'launch.json':last_cpu.LAUNCH_SHA})
    last_cpu.ended(last_cpu.run.read_json(last_cpu.job.OUTPUT,'launch.json'))


def expected_report(prior,timing):
    result=deepcopy(prior)
    for key in ('incremental_single_pass_bounds_comparison',
            'both_controllers_use_original_body_projection_and_receipt_handling'):
        if result.pop(key) is not True: raise ValueError('original completed single-pass scope required')
    result.update(harness.scope());result['timing_windows']=timing
    return result


def validate_report(report,rows,prior):
    timing=check_rows(rows,prior[2])
    if fingerprint(report) != fingerprint(expected_report(prior[1]['report'],timing)):
        raise ValueError('complete report, timings and all seven state witnesses must reconstruct')
    return timing


def launch(preflight=False):
    if (not __debug__ or any(previous.os.environ.get(k) != v for k,v in previous.ENVIRONMENT.items())
            or harness.original.cv2.ocl.useOpenCL()):
        raise ValueError('original deterministic CPU environment required')
    validate_root(OUTPUT,must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive paired replay; no retry or resume')
    bindings=sources();slot_available();hw=previous.hardware()
    if preflight:
        print('DEFERRED_MEMO_SINGLE_PASS_PREFLIGHT',len(bindings),json.dumps(hw),flush=True);return
    print('DEFERRED_MEMO_SINGLE_PASS_INPUT_ADMISSION_STARTED',flush=True)
    prior=admit();verify(bindings);slot_available();hw=previous.hardware()
    create_output(OUTPUT);p=psutil.Process()
    write_json(OUTPUT/'launch.json',dict(source_sha256=bindings,reference_completion_sha256=COMPLETION_SHA,
        reference_result_sha256=RESULT_SHA,reference_artifact_sha256=prior[0]['artifact_sha256'],
        last_cpu_completion_sha256=LAST_CPU_SHA,component_benchmark_sha256=BENCHMARK_SHA,
        environment=previous.ENVIRONMENT,hardware=hw,frames=1428,state_frames=list(harness.original.STATE_FRAMES),
        **harness.scope(),boot_id=previous.Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        owner=dict(pid=p.pid,created=p.create_time(),command=p.cmdline()),
        native_execution=False,model_training=False,automatic_retry=False))
    print('DEFERRED_MEMO_SINGLE_PASS_LAUNCHED',digest(OUTPUT/'launch.json'),len(bindings),flush=True)
    harness.original.cv2.setNumThreads(1);harness.original.torch.set_num_threads(1)
    harness.original.torch.use_deterministic_algorithms(True)
    start=time.perf_counter()
    try:
        report=harness.replay(prior[2],prior[1]['report'])
        rows=[json.loads(line) for line in (OUTPUT/'comparison.jsonl').read_text().splitlines()]
        validate_report(report,rows,prior)
        if admit() != prior: raise ValueError('original raw/model inputs changed during replay')
        verify(bindings)
        ids={n:digest(OUTPUT/n) for n in ('launch.json','comparison.jsonl')};verify_artifacts(OUTPUT,ids)
        write_json(OUTPUT/'result.json',dict(status='DEFERRED_MEMO_SINGLE_PASS_LATE_HISTORY_V1_COMPLETE',
            source_sha256=bindings,artifact_sha256=ids,report=report,sensing_scope=prior[1]['sensing_scope'],
            original_inputs_reauthenticated_before_and_after=True,wall_s=time.perf_counter()-start,
            native_execution=False,real_time_qualified=False,navigation_qualified=False,goal_achieved=False))
        print('DEFERRED_MEMO_SINGLE_PASS_COMPLETE',digest(OUTPUT/'result.json'),flush=True)
    except BaseException as error:
        write_json(OUTPUT/'failure.json',dict(status='TERMINAL_DEFERRED_MEMO_SINGLE_PASS_FAILURE',reason=repr(error),automatic_retry=False))
        raise


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-preflight-only',action='store_true')
    launch(parser.parse_args().source_preflight_only)
