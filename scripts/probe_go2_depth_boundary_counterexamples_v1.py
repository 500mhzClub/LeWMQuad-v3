"""Persist synthetic thin-obstacle and near-plane visibility counterexamples."""
import sys
import numpy as np
from lewm.depth_boundary_counterexamples_development import diagnose
from scripts.navigation_artifact_root_development import BASE, validate_root, create_output
from scripts.startup_source_inventory_development import discover_sources
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json, verify
from scripts.read_go2_tracking_posthoc_raw_accuracy_v1 import hardware

OUTPUT = BASE/'go2_depth_boundary_counterexamples_v1_attempt_001'
PROTOCOL = 'docs/go2_depth_boundary_counterexamples_v1_2026-09-08.md'


def main():
    if not __debug__: raise ValueError('assertions required')
    validate_root(OUTPUT, must_exist=False)
    if OUTPUT.exists() or OUTPUT.is_symlink(): raise ValueError('exclusive synthetic counterexample output required')
    sources = discover_sources((PROTOCOL, 'scripts/probe_go2_depth_boundary_counterexamples_v1.py',
        'lewm/tests/test_depth_boundary_counterexamples_development.py'), {})
    verify(sources); resources = hardware()
    if resources['memory_available_bytes'] < 8*1024**3 or resources['artifact_free_bytes'] < 40*1024**3+32*1024**2:
        raise ValueError('bounded synthetic diagnostic resources unavailable')
    launch = dict(protocol=PROTOCOL, source_sha256=sources, output_root=str(OUTPUT), hardware=resources,
        numpy_version=np.__version__, python_version=sys.version, synthetic_workers=1,
        native_execution=False, model_loaded=False, model_training=False,
        runtime_experiment_inputs_read=False, maximum_artifact_bytes=32*1024**2,
        concurrency_reason='small CPU analytic fixture alongside one separately owned native scene')
    create_output(OUTPUT); write_json(OUTPUT/'launch.json', launch)
    print('DEPTH_BOUNDARY_COUNTEREXAMPLES_LAUNCHED', digest(OUTPUT/'launch.json'), len(sources), flush=True)
    try:
        report = diagnose(); verify(sources)
        write_json(OUTPUT/'result.json', dict(status='DEPTH_BOUNDARY_COUNTEREXAMPLES_COMPLETE',
            report=report, launch_sha256=digest(OUTPUT/'launch.json'), source_sha256=sources,
            synthetic_only=True, native_execution=False, navigation_qualified=False, goal_achieved=False))
        if (OUTPUT/'launch.json').stat().st_size+(OUTPUT/'result.json').stat().st_size > 32*1024**2:
            raise ValueError('declared metadata budget exceeded')
        print('DEPTH_BOUNDARY_COUNTEREXAMPLES_COMPLETE', digest(OUTPUT/'result.json'), flush=True)
    except Exception as error:
        write_json(OUTPUT/'failure.json', dict(status='TERMINAL_DEPTH_BOUNDARY_COUNTEREXAMPLE_FAILURE', reason=repr(error)))
        raise


if __name__ == '__main__': main()
