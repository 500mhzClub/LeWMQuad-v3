"""Preserve the four terminal scheduler failures before a fresh native attempt.

No old waiter is restarted or relabeled complete. None dispatched a native
child. The completed extended-budget episode remains the last native result.
"""
from scripts import measured_plane_native_inputs_development as original

run = original.run
BOOT = '1264d80f-6e46-4fcd-b2fd-2a5d7b964c73'
BUDGET_ROOT = run.BASE/'go2_no_rgb_direct_extended_budget_maze02_native_wait_v1_attempt_001'
BUDGET_SHA = '1e2da182445446b05de79c09825979a0e4547212d3bd1db5ce999fca4928fc70'
NATIVE_SHA = 'c92f0bdf5cc8ebb9e513470492578ec3457d02196dfc7238e2f6b12b9c621b27'
CASES = (
    ('sustained_hold_reorientation', 2817601, 1789117418.98,
     '679d0519e3eb5ddacf5cc3708254de12a117551725c43c0136141e43ce64263e',
     '1fb4fbe91955dbe5932e1f71d0f44bcacb7ca1876d532703f0c8a6b11a42f3af',
     'ec8795b391b5dbb35cadb4f8d634762312fad47d33a0b70b9244046d3dd824db',
     'TERMINAL_SUSTAINED_NATIVE_WAIT_FAILURE',
     "ValueError('existing native runner or worker still live; do not start another scene')"),
    ('direct_flow_commitment_contact', 2827789, 1789121410.15,
     '3fc8e765b6edc16120b418e6dc8cedf1da47eb134d959f98450b2adeaa2c6c72',
     '0bb3e9e2dfb96d1e6e0d8699ff2624c9228e3941e2004db46be731c711114556',
     '972b50ff3f687b63bbdd3ba002994f6544936a20468a50eb2fcbef184d4fbf4d',
     'TERMINAL_CONTACT_FLOW_NATIVE_WAIT_FAILURE',
     "ValueError('original sustained owner ended without complete result; no restart')"),
    ('chained_anchor', 2845479, 1789129072.88,
     'cf6703e83197d6c75df35b2b53834a47a53a293a06c64737ce94ade2ac0b87c1',
     '225e8595dc26f4947bf2adb90a9f2643528d8ef3cfbea16d6ba7e84352e3aaa8',
     'd68f87a23dd3461c95ca74db48c933bceda510d768b45745f9fdf99b237fde0e',
     'TERMINAL_CHAINED_ANCHOR_NATIVE_WAIT_FAILURE',
     "ValueError('original flow owner ended without complete result; no restart')"),
    ('measured_plane', 2908919, 1789159050.26,
     'b7991218f7dd150900201fb98a62e2d1ddd0dcd81e369420d6746d31dd81a9a5',
     '4dd9243884c4e7a3771ef250a2d528c17ef33b679380375ee0edd8216af612b5',
     '609a5579eef16f831ee6c5d65fe6bc26e0eb88cb321d6fca88389d31d7d485c8',
     'TERMINAL_MEASURED_PLANE_NATIVE_WAIT_FAILURE',
     "FileNotFoundError(2, 'No such file or directory')"),
)


def owner(name, pid, created):
    return dict(pid=pid, created=created, command=[
        '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
        'scripts/await_go2_'+name+'_maze02_native_v1.py'])


def require_no_dispatch(launch, failure, events, expected_owner, status, reason):
    if (launch['boot_id'] != BOOT
            or launch.get('waiter_pid', launch.get('owner', {}).get('pid')) != expected_owner['pid']
            or failure['status'] != status or failure['reason'] != reason
            or failure['automatic_retry'] is not False
            or any('CHILD_STARTED' in event['status'] for event in events)):
        raise ValueError('exact original terminal scheduler failure without dispatch required')


def admit():
    import json
    if run.Path('/proc/sys/kernel/random/boot_id').read_text().strip() != BOOT:
        raise ValueError('same observed scheduler-failure boot required')
    evidence, sources = [], {}
    for name, pid, created, launch_sha, failure_sha, event_sha, status, reason in CASES:
        root = run.BASE/('go2_'+name+'_maze02_native_wait_v1_attempt_001')
        child = run.BASE/('go2_'+name+'_maze02_pilot_v1_attempt_001')
        run.validate_root(root); run.validate_root(child, must_exist=False)
        if child.exists() or child.is_symlink() or (root/'result.json').exists() or (root/'native_stdout.log').exists():
            raise ValueError('original failed waiter and never-created native child required')
        expected_owner = owner(name, pid, created)
        if run.owner_live(expected_owner):
            raise ValueError('all failed original waiter owners must remain ended')
        bindings = {'launch.json': launch_sha, 'failure.json': failure_sha, 'events.jsonl': event_sha}
        run.verify_artifacts(root, bindings)
        launch = run.read_json(root, 'launch.json'); failure = run.read_json(root, 'failure.json')
        with (root/'events.jsonl').open() as stream:
            events = [json.loads(line) for line in stream]
        require_no_dispatch(launch, failure, events, expected_owner, status, reason)
        for path, sha in launch['source_sha256'].items():
            if path in sources and sources[path] != sha: raise ValueError('failed queue source ancestry conflict')
            sources[path] = sha
        evidence.append(dict(root=str(root), owner=expected_owner, artifact_sha256=bindings,
            status=status, reason=reason, original_owner_ended=True, native_child_never_created=True))
    run.verify_artifacts(BUDGET_ROOT, {'result.json': BUDGET_SHA})
    budget = run.read_json(BUDGET_ROOT, 'result.json')
    if (budget['status'] != 'NO_RGB_DIRECT_EXTENDED_BUDGET_MAZE02_NATIVE_WAIT_V1_COMPLETE'
            or budget['report']['native_result_sha256'] != NATIVE_SHA
            or budget['report']['complete_native_worker_and_artifact_roster_verified'] is not True
            or budget['report']['actual_prefix_finding_reconstructed'] is not True):
        raise ValueError('actual completed extended-budget predecessor required')
    if run.owner_live(dict(pid=2793505, created=1789096721.36, command=[
            '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python', '-B',
            'scripts/await_go2_no_rgb_direct_extended_budget_maze02_native_v1.py'])):
        raise ValueError('completed extended-budget waiter must remain ended')
    run.verify_artifacts(BUDGET_ROOT, budget['artifact_sha256'])
    run.verify_artifacts(original.job.native.OUTPUT, {'result.json': NATIVE_SHA})
    run.verify(sources)
    return dict(status='FOUR_NATIVE_WAITERS_TERMINATED_BEFORE_DISPATCH', failures=evidence,
        source_sha256=sources, completed_budget_waiter_sha256=BUDGET_SHA,
        completed_native_result_sha256=NATIVE_SHA, original_waiters_restarted=False,
        failed_waiters_relabelled_complete=False, unexecuted_diagnostics_counted_as_results=False,
        new_native_execution=False, navigation_qualified=False, goal_achieved=False)
