"""Summarize paired native outcomes after independent arrival evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
from lewm.physical_execution_development import rotation_xyzw

BASE=Path('/home/andrewknowles/RecoveryStorage/LeWMQuad-v3/.generated/navigation_development_artifacts_v1')
MATCHED_FIELDS=('layout_index','public_mission','navigation_tick_budget',
    'camera_simulation_period_ns','command_service_simulation_period_ns',
    'deadline_clock','planning_delay_ticks','command_duration_ns',
    'observed_arrival_radius_m','physical_arrival_requirement_m',
    'nominal_footprint_radius_m','stopping_allowance_s','clearance_preferred_routing',
    'routing_soft_preferred_clearance_m','routing_clearance_cost_weight')
OPTIONAL_MATCHED_FIELDS=('floor_height_conflict_rejection_scope',
    'original_pair_acceptance_checks_unchanged','robust_plane_image_consensus',
    'gyro_rotation_reorthogonalized_each_camera_interval','exact_chained_image_link_reuse',
    'compiled_floor_candidate_predicates','exact_fine_goal_segment_cache',
    'stable_reference_selection','stable_reference_activation_frame',
    'fine_observed_goal_connectivity_fallback','terminal_translation_pulses',
    'terminal_translation_command_duration_ns','maximum_command_duration_ns')


def path(name):
    if Path(name).name!=name or name.startswith('sealed'):raise ValueError('ordinary development basename required')
    return BASE/name


def read(root,name):return json.loads((root/name).read_text())


def summarize(root):
    launch=read(root,'launch.json');evaluation=read(root,'continuous_native_arrival_evaluation.json')
    metadata=read(root,'native/in_memory_camera_observations.json')
    frames=sorted(metadata['frames'],key=lambda r:r['frame'])
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as arrays:physics=arrays['base_pose_world']
    origin=physics[frames[0]['physical_sample_index']];R0=rotation_xyzw(origin[3:])
    positions=(physics[[f['physical_sample_index'] for f in frames],:3]-origin[:3])@R0
    goal=np.asarray(launch['public_mission']['goal_initial_body_xy_m'])
    goal_distances=np.linalg.norm(positions[:,:2]-goal,axis=1)
    arrivals=evaluation['arrivals'];outbound=next((r for r in arrivals if r['phase']=='OUTBOUND'),None)
    last=outbound['frame'] if outbound is not None else frames[-1]['frame']
    mask=np.asarray([f['frame']<=last for f in frames])
    plans=[p for p in read(root,'planning.json') if 'selection' in p]
    if launch['model_assignment']=='reactive' and any('motion_correction' in p or
            p['selection'].get('candidate_future_outcomes_evaluated') is not False for p in plans):
        raise ValueError('reactive records must not evaluate future outcomes')
    result=read(root,'result.json') if (root/'result.json').exists() else None
    return dict(root_name=root.name,assignment=launch['model_assignment'],
        independent_arrival_evaluation=evaluation,
        minimum_outbound_native_goal_distance_m=float(goal_distances[mask].min()),
        final_native_goal_distance_m=float(goal_distances[-1]),
        final_native_home_distance_m=float(np.linalg.norm(positions[-1,:2])),
        native_10hz_horizontal_path_length_m=float(np.linalg.norm(np.diff(positions[:,:2],axis=0),axis=1).sum()),
        planning_records=len(plans),plans_on_time=sum(bool(p['on_time']) for p in plans),
        result=result,failure=read(root,'failure.json') if (root/'failure.json').exists() else None)


def main():
    parser=argparse.ArgumentParser()
    for arg in ('learned-root-name','reactive-root-name','output-name'):parser.add_argument('--'+arg,required=True)
    args=parser.parse_args();learned=path(args.learned_root_name);reactive=path(args.reactive_root_name)
    output=path(args.output_name)
    if output.exists():raise ValueError('preserve prior comparison')
    a,b=read(learned,'launch.json'),read(reactive,'launch.json')
    if a['model_assignment']=='reactive' or b['model_assignment']!='reactive':raise ValueError('explicit learned/reactive pair required')
    differences=[k for k in MATCHED_FIELDS if a[k]!=b[k]]
    differences.extend(k for k in OPTIONAL_MATCHED_FIELDS if a.get(k)!=b.get(k))
    if differences:raise ValueError(f'comparison settings differ: {differences}')
    # Exclude the entry-point configuration, which selects the treatment.
    sources_a=a['source_sha256']|a['extra_sources'];sources_b=b['source_sha256']|b['extra_sources']
    source_names={k for k in set(sources_a)&set(sources_b)
        if not k.endswith('/run_go2_stopping_margin_round_trip_native_development.py')}
    changed=[k for k in sorted(source_names) if sources_a[k]!=sources_b[k]]
    if changed:raise ValueError(f'common implementation changed between arms: {changed}')
    report=dict(matched_settings={k:a[k] for k in MATCHED_FIELDS}|{k:a.get(k) for k in OPTIONAL_MATCHED_FIELDS},common_source_hashes_equal=True,
        shared_host_measured_simulation_timing=True,native_state_used_by_evaluator_only=True,
        comparison='complete_predictive_selection_versus_instantaneous_reactive_selection',
        jepa_specific_advantage_established=False,repeatability_established=False,
        learned=summarize(learned),reactive=summarize(reactive))
    output.mkdir()
    with (output/'result.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({label:{k:v for k,v in report[label].items() if k not in ('result','failure','independent_arrival_evaluation')}
        for label in ('learned','reactive')}))


if __name__=='__main__':main()
