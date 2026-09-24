"""Fresh120-case layout batch with unchanged core-ordered native acquisition."""
import argparse
import json
import shutil
import cv2
import torch
from scripts.independent_rgb_body_batch_development import (BATCHES, PROTOCOL, BATCH_BUDGET,
    EPISODE_ALLOWANCE, RESERVE, INVENTORY_IDS, inventory_bindings, load_inventory,
    output_root, validate_launch, commit_episode)
from scripts.independent_rgb_body_audit_development import audit_rgb_body_condition
from scripts.run_go2_core_ordered_union_dynamic_sensor_pilot_v1 import collect
from scripts.navigation_artifact_root_development import BASE, create_output, verify_artifacts
from scripts.probe_go2_ordered_union_rgb_repeatability_v1 import verify_ordered_launch
from scripts.startup_source_inventory_development import discover_sources
from scripts.startup_raw_sensor_audit_development import read_json
from scripts.run_go2_successive_choice_maze_development_v1 import digest, write_json

EDGE = BASE / 'go2_native_raster_edge_crossing_v1_attempt_001'
EDGE_IDS = {'launch.json': 'b19ab3392287f2eda9b3d51e043b14a4454cbd66d07a6cb2eff3863ae61ffb6b',
    'result.json': 'e138738557fac7b41cd9672a2bd6d0d45842e39981ddbe99a296700d0b1ac263'}
COVERAGE = BASE / 'go2_terminal_event_coverage_check_v1_attempt_001'
COVERAGE_IDS = {'launch.json': '782ce67d34a38b7658b47f6195b1f782c9d73273a9851303c9b08209d4af5a93',
    'result.json': 'bf48e54c38e0fe902f2392a1199a87565bf0988c7c48958d077ee6e0c04d6eba'}


def preflight(batch):
    output = output_root(batch)
    if output.exists() or output.is_symlink(): raise ValueError('exclusive fresh collection; no retry/resume')
    inventory = load_inventory()
    verify_artifacts(EDGE, EDGE_IDS); verify_artifacts(COVERAGE, COVERAGE_IDS)
    old = read_json(EDGE, 'launch.json'); coverage_launch = read_json(COVERAGE, 'launch.json')
    verify_ordered_launch(old); verify_ordered_launch(coverage_launch)
    edge = read_json(EDGE, 'result.json'); coverage = read_json(COVERAGE, 'result.json')
    verify_artifacts(EDGE, edge['artifact_sha256']); verify_artifacts(COVERAGE, coverage['output_sha256'])
    if not edge['accounting_mechanism_pass'] or coverage['candidate_acquisition_complete'] != 8:
        raise ValueError('completed mechanism and coverage evidence required')
    inherited = dict(old['source_sha256'])
    for p, h in coverage_launch['source_sha256'].items():
        if p in inherited and inherited[p] != h: raise ValueError('conflicting predecessor source bindings')
        inherited[p] = h
    sources = discover_sources((PROTOCOL, 'scripts/run_go2_independent_rgb_body_collection_v1.py',
        'scripts/run_go2_independent_rgb_body_stage_v1.py',
        'scripts/audit_go2_independent_rgb_body_collection_v1.py',
        'lewm/tests/test_independent_rgb_body_collection_development.py',
        'lewm/tests/test_rgb_body_depth_separation_development.py'), inherited)
    previous = None
    index = BATCHES.index(batch)
    if index:
        directory = output_root(BATCHES[index - 1])
        prior = read_json(directory, 'rgb_body_layout_audit.json')
        prior_launch = read_json(directory, 'launch.json')
        verify_ordered_launch(prior_launch)
        if (prior['status'] != 'RGB_BODY_LAYOUT_AVAILABLE_EVIDENCE_AUDITED' or not prior['collection_complete']
                or prior['audited_trials'] != 120 or prior['hard_measurement_failed_trials']):
            raise ValueError('previous fixed layout must complete collection and raw audit')
        verify_artifacts(directory, prior['output_sha256'])
        previous = dict(root=str(directory), audit_sha256=digest(directory / 'rgb_body_layout_audit.json'))
    ids = inventory.episode_ids(batch)
    launch = {k: old[k] for k in ('native_sha256', 'native_scene_sha256', 'native_geometry_sha256',
        'opencv_binary_sha256', 'opencv_version', 'rules')}
    launch.update(source_sha256=sources, input_sha256=old['input_sha256'] | inventory_bindings(),
        batch=batch, output_root=str(output), planned_trials=list(ids),
        conditions={c: inventory.specification(c) for c in ids}, role=inventory.specification(ids[0])['data_role'],
        inventory_sha256=INVENTORY_IDS['inventory.json'], maximum_batch_bytes=BATCH_BUDGET,
        episode_storage_allowance_bytes=EPISODE_ALLOWANCE, minimum_free_bytes=RESERVE,
        maximum_command_ticks=33, maximum_episode_physics_samples=2400, maximum_episode_rgbd_frames=34,
        eligibility_contract='RGB_BODY_TERMINAL_COVERAGE_V1', edge_assay_sha256=EDGE_IDS,
        coverage_check_sha256=COVERAGE_IDS, preceding_layout=previous,
        model_training=False, real_time_qualified=False, hidden_robot_ideal_camera=True,
        navigation_qualified=False, final_evaluation=False, goal_achieved=False)
    validate_launch(launch, inventory, batch); verify_ordered_launch(launch)
    if len((json.dumps(launch, indent=2, allow_nan=False) + '\n').encode()) > 32 * 1024**2:
        raise ValueError('serialized launch metadata cap32MiB')
    if shutil.disk_usage(BASE).free < RESERVE + BATCH_BUDGET: raise ValueError('batch storage reserve')
    return inventory, launch


def run_batch(batch):
    cv2.setNumThreads(1); torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    inventory, launch = preflight(batch); output = output_root(batch)
    create_output(output); write_json(output / 'launch.json', launch)
    results, commits, prechecks = {}, {}, {}
    active = None; used = (output / 'launch.json').stat().st_size
    try:
        for i, trial in enumerate(launch['planned_trials']):
            verify_ordered_launch(launch)
            if used + EPISODE_ALLOWANCE > BATCH_BUDGET or shutil.disk_usage(BASE).free < RESERVE + EPISODE_ALLOWANCE:
                raise ValueError('next episode storage allowance')
            active = trial
            result = collect(inventory, output, trial, trial, launch['source_sha256'][PROTOCOL])
            commit = commit_episode(output, inventory.specification(trial), result)
            leaf = f'episode_{i:03d}_commit.json'; write_json(output / leaf, commit)
            commits[leaf] = digest(output / leaf); results[trial] = result; active = None
            used += commit['artifact_bytes'] + (output / leaf).stat().st_size
            verify_artifacts(output, commit['artifact_sha256'])
            print('RGB_BODY_COMMITTED', batch, i + 1, len(launch['planned_trials']), trial, used, flush=True)
            if commit['absent_expected_artifacts']: raise ValueError('incomplete episode artifacts')
            if commit['artifact_bytes'] > EPISODE_ALLOWANCE: raise ValueError('episode byte cap')
            row = audit_rgb_body_condition(output / trial, inventory.specification(trial), result,
                launch['source_sha256'][PROTOCOL], batch=batch)
            leaf = f'episode_{i:03d}_raw_precheck.json'; write_json(output / leaf, row)
            prechecks[leaf] = digest(output / leaf); used += (output / leaf).stat().st_size
            print('RGB_BODY_PRECHECK', batch, i + 1, trial, row['coverage']['classification'], row['eligibility'], flush=True)
            episode_bytes = commit['artifact_bytes'] + (output / f'episode_{i:03d}_commit.json').stat().st_size + (output / leaf).stat().st_size
            if episode_bytes > EPISODE_ALLOWANCE or used > BATCH_BUDGET: raise ValueError('metadata-inclusive byte cap')
            if result['acquisition_stop'] is not None: raise ValueError('acquisition failure; preserve batch')
            if row['eligibility']['hard_measurement_failed_frames']: raise ValueError('hard measurement failure; preserve batch')
        verify_ordered_launch(launch); validate_launch(launch, inventory, batch)
        write_json(output / 'result.json', dict(status='RGB_BODY_LAYOUT_COLLECTION_COMPLETE', batch=batch,
            planned_trials=launch['planned_trials'], conditions=results, commits=commits, prechecks=prechecks,
            committed_bytes=used, uncommitted_trial=None, role=launch['role'], model_trained=False,
            navigation_qualified=False, goal_achieved=False))
    except Exception as error:
        write_json(output / 'failure.json', dict(status='TERMINAL_RGB_BODY_LAYOUT_COLLECTION_FAILURE', reason=repr(error), batch=batch,
            planned_trials=launch['planned_trials'], conditions=results, commits=commits, prechecks=prechecks,
            committed_bytes=used, uncommitted_trial=active, model_trained=False, navigation_qualified=False, goal_achieved=False))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch', choices=BATCHES, required=True)
    run_batch(parser.parse_args().batch)


if __name__ == '__main__': main()
