"""Read an observed controller startup failure without treating it as no trial."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from lewm import decision_headroom_json_v42_development as output
from lewm.decision_headroom_v4_development import ArticulatedSteps
from scripts import run_go2_navigation_capability_live_turn_v3_development as owner
from scripts import read_go2_navigation_capability_live_turn_v3_development as reader


def report(root):
    result=json.loads((root/'result.json').read_text())
    if result['frames']:
        return reader.report(root)
    faults_path=root/'pipeline_faults.json'
    faults=json.loads(faults_path.read_text()) if faults_path.exists() else []
    constructor_rejected=result['error']=="ValueError('finite initial-frame mission point within map bounds required')"
    if not faults and not constructor_rejected:raise ValueError('Not a recognized controller failure')
    metadata=json.loads((root/'native/in_memory_camera_observations.json').read_text())
    assert metadata['frames'] or constructor_rejected, 'No controller exposure evidence'
    assert not (root/'closeout_failure.json').exists(), 'Closeout defect is a technical stop'
    requests=json.loads((root/'requests.json').read_text());assert not requests
    episode=json.loads((root/'episode.json').read_text());spec=json.loads((root/'specification.json').read_text())
    started=time.monotonic()
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as a:
        trace={k:a[k].copy() for k in ('base_pose_world','joint_position','physics_contact','timestamp_s')}
    evaluated=ArticulatedSteps(spec['geometry']['wall_boxes']).evaluate({k:v[-2:] for k,v in trace.items()})
    lower=float(np.min(evaluated['per_step_primitive_separation_lower_m'][-1]))
    upper=float(np.min(evaluated['per_step_primitive_separation_upper_m'][-1]))
    safety={name:dict(threshold_m=threshold,confirmed_violation_samples=int(upper<threshold),
        unresolved_sampled_samples=int(lower<threshold<=upper),all_native_samples_qualified=not(lower<threshold<=upper),
        minimum_separation_lower_m=lower,minimum_separation_upper_m=upper,native_discrete_ground_truth=True,
        interval_robustness_failure_count=0,interval_check_secondary=True,
        coverage='Single settled source state; no mission command or movement interval was executed')
        for name,threshold in [('hard',.005),('operating',.02)]}
    label='observation/view restriction: initial measured floor unavailable' if any('initial measured floor unavailable' in f['reason'] for f in faults) else 'pose/perception pipeline failure'
    if constructor_rejected:label='invalid/unreachable target: mission coordinate outside deployed map bound'
    leg=lambda shortest:dict(success=False,actual_path_m=0.,shortest_path_m=shortest,spl=0.,elapsed_s=0.)
    record=dict(schema='navigation_capability_episode_evaluation.v3',controller=json.loads((root/'config.json').read_text())['controller'],
        episode_id=episode['episode_id'],role=episode['role'],beacon_success=False,home_success=False,round_trip_success=False,
        source_error=result['error'],arrivals=[],outbound=leg(episode['shortest_outbound_m']),return_leg=leg(episode['shortest_return_m']),
        disallowed_contact_samples=int(np.count_nonzero(trace['physics_contact'])),safety=safety,stall_by_phase={},hold_categories={},
        failure_and_stall_taxonomy={label:1},failure_taxonomy_counts_overlap=False,hold_details=[],
        decision_latency_s=dict(median=None,p95=None),wall_s=result['wall_s'],wall_seconds_per_simulated_second=None,
        articulated_reader_wall_s=time.monotonic()-started,training_render_provenance='unverified',
        label='Corrected development harness failure; counted in assigned denominator',science_episode=True,
        consumed_frame_pairs=len(metadata['frames']),owner_completed_acquisition_counter=result['frames'],
        accounting_note=('Unchanged controller rejected the registered task cue at construction before consuming sensors; retain this assigned mission failure.' if constructor_rejected else 'Owner increments acquisitions only after controller drain; first packet was consumed before failure. Retained hashes and pose/mission evidence establish trial exposure.'),
        reader_sha256=owner.sha(__file__),input_sha256={name:owner.sha(root/name) for name in ('config.json','result.json','failure.json','task_reference_evaluator.json','native/in_memory_camera_observations.json','native/physics_trace.npz')})
    owner.save(root/'episode_evaluation.json',record)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);args=p.parse_args()
    protocol=json.loads(owner.PROTOCOL.read_text());base=Path(protocol['output_root']);output.install(base)
    assert args.root.resolve().is_relative_to((base/'runs').resolve())
    r=report(args.root);print(json.dumps({k:r[k] for k in ('episode_id','round_trip_success','failure_and_stall_taxonomy')}))
