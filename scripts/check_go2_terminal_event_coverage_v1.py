"""All-eight recorded compatibility check; no physics or predecessor writes."""
import json
import shutil
import cv2
import torch
from lewm.terminal_event_coverage_development import population_coverage
from scripts.terminal_event_collection_audit_development import audit_terminal_event_condition
from scripts.audit_go2_core_ordered_union_dynamic_sensor_pilot_v1 import load_terminal
from scripts.run_go2_core_ordered_union_dynamic_sensor_pilot_v1 import OUTPUT as INPUT, PROTOCOL as OLD_PROTOCOL
from scripts.ordered_dynamic_pilot_development import RUNS
from scripts.independent_layout_batch_development import load_inventory
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.startup_source_inventory_development import discover_sources
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output, verify_artifacts
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

OUTPUT = BASE / 'go2_terminal_event_coverage_check_v1_attempt_001'
PROTOCOL = 'docs/go2_terminal_event_coverage_contract_v1_2026-09-06.md'
BUDGET = 32 * 1024**2
RESERVE = 40 * 1024**3
IDENTITIES = {
    'launch.json': '8e6162bcd8bfb467a54185489f8d7e1eb770f102a15d8fece1a2f413d81226a5',
    'result.json': '8cf589de36e00b65288bbf0abe322c65ff40d14b286acd7db7d0c9d2f923d6af',
    'dynamic_audit.json': '10d72ad0817a4a807f018ebfb5fbc5974100b38848346892c1feae977c4d9c5b'}


def preflight():
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink():
        raise ValueError('exclusive metadata check; no retry/resume')
    verify_artifacts(INPUT, IDENTITIES)
    inv = load_inventory()
    old, terminal, commits, bindings = load_terminal(inv)
    if len(commits) != 8 or terminal['uncommitted_run'] is not None:
        raise ValueError('exact completed eight-run population required')
    audit = read_json(INPUT, 'dynamic_audit.json')
    bindings.update(IDENTITIES | audit['output_sha256'])
    verify_artifacts(INPUT, bindings)
    sources = discover_sources((PROTOCOL, 'scripts/check_go2_terminal_event_coverage_v1.py',
        'lewm/tests/test_terminal_event_coverage_development.py'), old['source_sha256'])
    launch = old | dict(source_sha256=sources, input_root=str(INPUT), output_root=str(OUTPUT),
        recorded_artifact_sha256=bindings, maximum_artifact_bytes=BUDGET, minimum_free_bytes=RESERVE,
        check_protocol=PROTOCOL, physics_steps=0, renders=0, training_eligibility_granted=False)
    verify_ordered_launch(launch)
    if len((json.dumps(launch, indent=2, allow_nan=False) + '\n').encode()) > BUDGET // 2:
        raise ValueError('serialized launch metadata allowance')
    if shutil.disk_usage(BASE).free < RESERVE + BUDGET:
        raise ValueError('metadata reserve')
    return inv, old, launch, terminal


def main():
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    inv, old, launch, terminal = preflight()
    create_output(OUTPUT)
    write_json(OUTPUT / 'launch.json', launch)
    coverage, outputs = {}, {}
    used = (OUTPUT / 'launch.json').stat().st_size
    try:
        for i, (run, trial, _) in enumerate(RUNS):
            verify_ordered_launch(launch)
            if shutil.disk_usage(BASE).free < RESERVE + BUDGET:
                raise ValueError('metadata reserve')
            row = audit_terminal_event_condition(INPUT / run, inv.specification(trial),
                terminal['conditions'][run], old['source_sha256'][OLD_PROTOCOL])
            previous = read_json(INPUT, f'episode_{i:03d}_raw_precheck.json')
            for key in ('report', 'prefix'):
                if json.loads(json.dumps(row[key])) != previous[key]:
                    raise ValueError('frozen raw evidence changed: ' + key)
            coverage[run] = row['coverage']
            leaf = f'coverage_{i:03d}.json'
            encoded = (json.dumps(row['coverage'], indent=2, allow_nan=False) + '\n').encode()
            if used + len(encoded) > BUDGET - 1024**2:
                raise ValueError('metadata budget')
            write_json(OUTPUT / leaf, row['coverage'])
            used += (OUTPUT / leaf).stat().st_size
            outputs[leaf] = digest(OUTPUT / leaf)
            print('TERMINAL_COVERAGE', run, row['coverage'], flush=True)
        result = population_coverage([r for r, _, _ in RUNS], coverage)
        verify_ordered_launch(launch)
        verify_artifacts(INPUT, launch['recorded_artifact_sha256'])
        verify_artifacts(OUTPUT, outputs)
        write_json(OUTPUT / 'result.json', result | dict(status='RECORDED_COVERAGE_COMPATIBILITY_COMPLETE',
            output_sha256=outputs, predecessor_result_unchanged=True, posthoc_compatibility_only=True,
            goal_achieved=False))
        print('TERMINAL_COVERAGE_COMPLETE', result, flush=True)
    except Exception as error:
        write_json(OUTPUT / 'failure.json', dict(status='TERMINAL_COVERAGE_CHECK_FAILURE',
            reason=repr(error), completed_runs=list(coverage), output_sha256=outputs,
            training_eligibility_granted=False, goal_achieved=False))
        raise


if __name__ == '__main__':
    main()
