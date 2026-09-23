"""Close out technical evidence only; never compute comparative audit rows."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


CAPS = Path('docs/go2_decision_headroom_phase1_caps_v1_2026-09-23.json')
ACTIONS = ('hold','forward','left_arc','right_arc','left_turn','right_turn')


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def summary(values):
    return dict(n=len(values), median=statistics.median(values), maximum=max(values),
        total=sum(values)) if values else dict(n=0)


def run():
    caps = read(CAPS)
    root = Path(caps['output_root'])
    if not (root/'resource_result.json').is_file():
        raise RuntimeError('wait for pilot owner resource closeout before consolidating')
    admission = read(root/'pilot_execution_admission.json')
    resource = read(root/'resource_result.json')
    sanity = read(root/admission.get('reference_sanity_result', 'reference_sanity_v1/result.json'))
    source_rows, state_rows, branch_rows, component_rows = [], [], [], []
    repeat_positions, repeat_yaws = [], []
    restored_by_controller = dict(command_history=0, reactive_feedback=0, action=0)
    rgb_equal, rgb_recorded = 0, 0
    for case, (layout, controller) in enumerate(admission['source_assignments']):
        source = root/f'source_{case:02d}'
        result = read(source/'result.json')
        source_rows.append(dict(case=case, layout=layout, controller=controller,
            collection=result, failure=read(source/'failure.json'),
            closeout_failure=read(source/'closeout_failure.json')))
        timing = read(source/'component_timing.json')
        if timing:
            component_rows.extend(timing['rows'])
        for frame in caps['collection_caps']['snapshot_frames']:
            state = source/f'state_{frame:04d}'
            restoration = read(state/'restoration.json')
            valid = bool(restoration and restoration['passed'])
            restored_by_controller[controller] += int(valid)
            row = dict(case=case, layout=layout, controller=controller, frame=frame,
                snapshot_present=(state/'snapshot.json').is_file(), restoration_passed=valid,
                unavailable=read(state/'unavailable.json'), failure=read(state/'qualification_failure.json'))
            if restoration:
                for repeat in restoration['repeats']:
                    for image in repeat['rgb_bitwise_matches']:
                        rgb_recorded += 2
                        rgb_equal += int(image['primary'])+int(image['auxiliary'])
            names = [f'source_trace_{r}' for r in range(3)]
            names += [f'{a}_{r}' for a in ACTIONS for r in range(3)]
            retained = []
            for name in names:
                branch = read(state/name/'result.json')
                if branch:
                    retained.append(name)
                    branch_rows.append(branch)
            row['branch_terminal_records'] = len(retained)
            row['all_assigned_branches_recorded'] = len(retained)==21
            repeats = read(state/'repeat_variability.json')
            if repeats:
                for action in repeats['branches']:
                    for repeat in action['pairwise_repeat_comparisons']:
                        for horizon in repeat['horizons']:
                            if horizon['valid']:
                                repeat_positions.append(horizon['position_error_m'])
                                repeat_yaws.append(horizon['yaw_error_rad'])
            state_rows.append(row)
    fraction = sum(r['restoration_passed'] for r in state_rows)/caps['collection_caps']['sampled_states_total']
    missing = sum(r['restoration_passed'] and not r['all_assigned_branches_recorded'] for r in state_rows)
    requirements = dict(reference_sanity=bool(sanity and sanity['status']=='PASS'),
        restoration_fraction=fraction >= admission['minimum_restored_state_fraction'],
        each_source_controller=all(n >= admission['minimum_valid_states_per_source_controller'] for n in restored_by_controller.values()),
        assigned_sources_complete=all(s['collection'] and not s['failure'] and not s['closeout_failure'] for s in source_rows),
        valid_state_branches_accounted=missing==0,
        budget_no_stop=not resource['resource_stop_latched'],
        owner_no_failure=resource['error'] is None,
        reserved_attempts_accounted=len(branch_rows)==resource['branch_attempts'])
    timings = {}
    for name in ('encoder','predictor','readout_old_data','readout_maze_data'):
        rows = [r for r in component_rows if r['component']==name]
        if name=='encoder':
            for count in (1,3):
                timings[f'encoder_batch_{count}'] = summary([r['wall_s'] for r in rows if r['image_count']==count])
        else:
            timings[name] = summary([r['wall_s'] for r in rows])
    report = dict(schema='decision_headroom_blinded_pilot_closeout.v1',
        status='TECHNICAL_EVIDENCE_COMPLETE_REQUIRES_PROTOCOL_REVIEW' if all(requirements.values()) else 'PILOT_INVALID_OR_INCOMPLETE',
        requirements=requirements, restoration_fraction_of_all_assigned_states=fraction,
        restored_by_source_controller=restored_by_controller, sources=source_rows, states=state_rows,
        reference_sanity=None if sanity is None else dict(status=sanity['status'], passed=sanity['passed'], cases=sanity['cases'],
            strict_original_panel_pass=sanity.get('strict_original_panel_pass', sanity['status']=='PASS'),
            fixture_label_errata=sanity.get('fixture_label_errata', [])),
        repeat_position_error_m=summary(repeat_positions), repeat_yaw_error_rad=summary(repeat_yaws),
        repeat_summary_excludes_missing_horizons=True,
        repeat_summary_covers_all_three_unordered_repeat_pairs=True,
        full_repeat_and_terminal_records='source_XX/state_NNNN/repeat_variability.json and each branch result/physics trace',
        restored_rgb_bitwise_equal=rgb_equal, restored_rgb_images_compared=rgb_recorded,
        rgb_bitwise_equivalence_separate_from_pose_yaw_contact_tolerances=True,
        branch_attempt_wall_s=summary([r['attempt_wall_s'] for r in branch_rows]),
        branch_restore_wall_s=summary([r['restore_wall_s'] for r in branch_rows]),
        branch_process_cpu_s=summary([r['process_cpu_s'] for r in branch_rows]),
        physical_terminal_attempts=sum(r['terminal'] is not None for r in branch_rows),
        component_wall_s=timings, actual_retained_bytes=resource['retained_bytes'],
        actual_wall_s=resource['wall_s'], aggregate_cpu_s=resource['cpu_s'],
        peak_sampled_rss_bytes=resource['peak_sampled_aggregate_rss_bytes'],
        peak_sampled_gpu_bytes=resource['peak_sampled_total_gpu_used_bytes'],
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caps_sha256=hashlib.sha256(CAPS.read_bytes()).hexdigest(),
        no_comparative_audit_selections_rankings_regrets_or_variance=True,
        phase2_authorized=False, next='Freeze protocol, sampling design and measured budget; stop for explicit checkpoint-(a) approval.')
    with (root/'pilot_validity_report.json').open('x') as stream:
        json.dump(report, stream, indent=2); stream.write('\n')
    print(json.dumps({k:report[k] for k in ('status','requirements','restoration_fraction_of_all_assigned_states','actual_retained_bytes','actual_wall_s')},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--read', action='store_true', required=True)
    parser.parse_args()
    run()
