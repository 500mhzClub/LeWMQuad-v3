"""Sequential fixed remaining collection stages; no retries, fitting or promotion."""
import argparse
import json
import os
import shutil
import subprocess

from scripts.independent_rgb_body_batch_development import BATCHES, RESERVE, load_inventory, output_root
from scripts.independent_rgb_body_study_data_development import AUDIT, load_batch, require
from scripts.navigation_artifact_root_development import BASE, create_output, validate_root, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.run_go2_successive_choice_maze_development_v1 import ROOT, digest, write_json
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.startup_source_inventory_development import discover_sources

OUTPUT = BASE / 'go2_independent_rgb_body_remaining_stages_v1_attempt_001'
PROTOCOL = 'docs/go2_independent_rgb_body_remaining_stages_v1_2026-09-06.md'
SOURCE = 'scripts/run_go2_independent_rgb_body_remaining_stages_v1.py'
TEST = 'lewm/tests/test_independent_rgb_body_remaining_stages_development.py'
STAGE = 'scripts/run_go2_independent_rgb_body_stage_v1.py'
PYTHON = ROOT / '.generated/venvs/genesis_rocm_0_4_6_v1/bin/python'
FIRST_LAUNCH = 'cc0d583329b3b6713d84e4f8911d089f90160858401bdbb9dc5f384bda75f9c3'
METADATA_BUDGET = 32 * 1024**2
CHILD_ENV = dict(PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', PYTHONPATH='.:lewm_genesis:lewm_worlds',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', LIBGL_ALWAYS_SOFTWARE='1',
    PYOPENGL_PLATFORM='egl', EGL_DEVICE_ID='2')


def command(batch):
    require(batch in BATCHES[1:], 'remaining fixed batches only; never restart l00')
    return [str(PYTHON), STAGE, '--batch', batch]


def verify_definition(definition):
    verify_ordered_launch(definition)
    verify_artifacts(output_root('l00'), definition['initial_receipt'])


def summarize(batch, joined):
    audit = read_json(output_root(batch), AUDIT)
    verify_artifacts(output_root(batch), joined['receipt'])
    return dict(batch=batch, receipt=joined['receipt'], role=joined['role'], planned_trials=120,
        eligible_departures=len(joined['windows']), excluded_trials=joined['excluded_trials'],
        population=joined['population'], strict_visibility_failed_trials=joined['strict_visibility_failed_trials'],
        contact_positive_targets=audit['contact_positive_targets'],
        exact_nonreference_prefix_matches=audit['exact_nonreference_prefix_matches'],
        action_coverage=audit['action_coverage'], source_and_artifact_bindings_verified=True,
        raw_audit_reexecuted=False, model_training=False, navigation_qualified=False)


def preflight(initial_audit_sha256):
    validate_root(OUTPUT, must_exist=False)
    require(not OUTPUT.exists() and not OUTPUT.is_symlink(), 'exclusive supervisor output; no retry/resume')
    for batch in BATCHES[1:]:
        p = output_root(batch)
        require(not p.exists() and not p.is_symlink(), 'remaining batch already exists; never resume or skip: ' + batch)
    receipt = {'launch.json': FIRST_LAUNCH, AUDIT: initial_audit_sha256}
    # A missing or still-live first audit cannot become a supervisor launch.
    verify_artifacts(output_root('l00'), receipt)
    inventory = load_inventory(); first = load_batch('l00', receipt, inventory)
    old = read_json(output_root('l00'), 'launch.json')
    sources = discover_sources((PROTOCOL, SOURCE, TEST), old['source_sha256'])
    definition = {k: old[k] for k in ('input_sha256', 'native_sha256', 'native_scene_sha256',
        'native_geometry_sha256', 'opencv_version', 'opencv_binary_sha256', 'rules')}
    definition.update(source_sha256=sources, original_collection_source_sha256=old['source_sha256'],
        initial_receipt=receipt, remaining_batches=list(BATCHES[1:]),
        commands={b: command(b) for b in BATCHES[1:]}, child_environment=CHILD_ENV,
        maximum_supervisor_metadata_bytes=METADATA_BUDGET, minimum_free_bytes=RESERVE,
        model_training=False, final_evaluation=False, navigation_qualified=False, goal_achieved=False)
    verify_definition(definition)
    require(shutil.disk_usage(BASE).free >= RESERVE + METADATA_BUDGET, 'supervisor metadata reserve')
    require(len((json.dumps(definition, indent=2, allow_nan=False) + '\n').encode()) <= METADATA_BUDGET // 2,
        'bounded supervisor launch metadata')
    return inventory, definition, summarize('l00', first)


def run_child(batch):
    """Fresh interpreter, exact stage command; stdout remains observable live.

    No observation timeout, automatic signal, restart, detached retry or alternate
    environment is introduced. The child retains its existing per-case checks,
    storage bounds and terminal audit, including its own collection failures.
    """
    return subprocess.run(command(batch), cwd=ROOT, env=os.environ | CHILD_ENV, check=False).returncode


def run_sequence(initial_audit_sha256):
    inventory, definition, first = preflight(initial_audit_sha256)
    create_output(OUTPUT); outputs = {}; completed = {}; active = None; child_exit = None; used = 0
    def save(name, value):
        nonlocal used
        encoded = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
        require(used + len(encoded) <= METADATA_BUDGET and shutil.disk_usage(BASE).free >= RESERVE + len(encoded),
            'bounded supervisor metadata allowance')
        write_json(OUTPUT / name, value); used += len(encoded); outputs[name] = digest(OUTPUT / name)
    try:
        save('launch.json', definition)
        save('l00_verified_receipt.json', first); completed['l00'] = first
        print('RGB_BODY_SEQUENCE_STARTED', list(BATCHES[1:]), flush=True)
        for batch in BATCHES[1:]:
            verify_definition(definition); verify_artifacts(OUTPUT, outputs)
            root = output_root(batch)
            require(not root.exists() and not root.is_symlink(), 'fresh next output required: ' + batch)
            active = batch; child_exit = None
            save(batch + '_stage_request.json', dict(batch=batch, command=command(batch),
                child_environment=CHILD_ENV, preceding_verified_batch=BATCHES[BATCHES.index(batch) - 1],
                preceding_receipt=completed[BATCHES[BATCHES.index(batch) - 1]]['receipt']))
            print('RGB_BODY_SEQUENCE_STAGE_START', batch, flush=True)
            child_exit = run_child(batch)
            save(batch + '_stage_exit.json', dict(batch=batch, returncode=child_exit))
            print('RGB_BODY_SEQUENCE_STAGE_EXIT', batch, child_exit, flush=True)
            require(child_exit == 0, 'stage failed; preserve its evidence and do not advance: ' + batch)
            # Freeze candidate receipt bytes after this exact child exits. Full
            # loader verification must succeed before the next stage can start.
            receipt = {n: digest(root / n) for n in ('launch.json', AUDIT)}
            save(batch + '_candidate_receipt.json', dict(batch=batch, receipt=receipt,
                verified=False, scope='terminal bytes pending full loader verification'))
            joined = load_batch(batch, receipt, inventory)
            launch = read_json(root, 'launch.json')
            require(launch['source_sha256'] == definition['original_collection_source_sha256'],
                'every child must retain the original frozen collection source definition')
            row = summarize(batch, joined)
            verify_definition(definition); save(batch + '_verified_receipt.json', row)
            completed[batch] = row; active = None
            print('RGB_BODY_SEQUENCE_BATCH_VERIFIED', batch, row['eligible_departures'], row['contact_positive_targets'], flush=True)
        for batch, row in completed.items():
            joined = load_batch(batch, row['receipt'], inventory)
            require(summarize(batch, joined) == row, 'completed batch changed before final sequence record: ' + batch)
            print('RGB_BODY_SEQUENCE_FINAL_RECHECK', batch, flush=True)
        verify_definition(definition); verify_artifacts(OUTPUT, outputs)
        save('result.json', dict(status='ALL12_FIXED_RGB_BODY_BATCH_RECEIPTS_VERIFIED',
            receipts={b: r['receipt'] for b, r in completed.items()}, completed_batches=list(completed),
            planned_layouts=12, planned_episodes=1440, output_sha256=dict(outputs),
            model_training=False, final_evaluation=False, navigation_qualified=False, goal_achieved=False))
        print('RGB_BODY_SEQUENCE_COMPLETE', list(completed), flush=True)
    except Exception as error:
        # This minimal terminal record deliberately bypasses the ordinary
        # metadata allowance; exhausting that allowance must remain visible.
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_RGB_BODY_SEQUENCE_FAILURE',
            reason=repr(error), active_batch=active, child_returncode=child_exit,
            completed_batches=list(completed), output_sha256=dict(outputs),
            model_training=False, retry_performed=False, goal_achieved=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial-audit-sha256', required=True)
    run_sequence(parser.parse_args().initial_audit_sha256)


if __name__ == '__main__':
    main()
