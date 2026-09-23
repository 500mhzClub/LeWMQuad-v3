"""Implementation checks on the six qualified states; no physics/audit results."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from PIL import Image
from lewm import decision_headroom_v4_development as v4
from scripts.qualify_go2_source_decision_packet_development import differences
from scripts.run_go2_decision_headroom_branches_development import load_bound


def main():
    torch.set_num_threads(4);start=time.monotonic()
    c=json.loads(Path('docs/go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json').read_text());root=Path(c['execution_caps']['output_root'])
    old=root.parent/'go2_decision_headroom_rgb_restore_recheck_v1_attempt_001'
    states=[]
    for i,s in enumerate(c['previous_corrected_states']):states.append((Path(s['source_root']),s,old/f'state_{i:02d}'))
    for i in range(4):
        source=root/f'source_{i:02d}';state=json.loads((source/'snapshots.json').read_text())[0];states.append((source,state,source/'state_0132'))
    reports=[]
    model=v4.deployed.load_dense_navigation_model('action',readout_arm='maze_view_old_data')
    from scripts import run_go2_maze_view_readout_recovery_development as fitted
    second=copy.deepcopy(model.readout)
    second.load_state_dict(torch.load(fitted.OUTPUT/'maze_data_final.pt',map_location='cpu',weights_only=False)['model_state_dict'])
    heads=dict(old_data=model.readout,maze_data=second.eval().requires_grad_(False))
    for source,state,branch in states:
        p=source/f"state_{state['frame']:04d}";packet=load_bound(p/'decision.pkl',state['decision'])
        corrected=np.asarray(packet['source_correction']['applied_prediction_after_yaw_ablation'])
        motion=np.concatenate((corrected[:,:,:2],np.arctan2(corrected[:,:,2:3],corrected[:,:,3:4])),axis=-1)
        selected=v4.selector(packet,motion,reactive=state['source_controller']=='reactive_feedback')
        mismatch=differences(selected,packet['source_selection'],atol=1e-6,rtol=1e-6)
        report=dict(source=str(source),frame=state['frame'],decision_sha256=state['decision']['sha256'],own_source_selection_parity=not mismatch,source_mismatches=mismatch)
        traces=[];images=[]
        for action in v4.ACTIONS:
            b=branch/f'{action}_0'
            with np.load(b/'physics_trace.npz',allow_pickle=False) as a:traces.append({k:a[k].copy() for k in a.files})
            images.append([np.asarray(Image.open(b/f'primary_{h:03d}ms.png').convert('RGB')) for h in range(100,801,100)])
        true=v4.true_motion(traces,state['measured_ns'])
        motions=v4.feature_motions(model,heads,packet,images)
        np.testing.assert_allclose(motions['old_data']['R4'],packet['source_model_receipt']['motion_xy_yaw'],atol=1e-6,rtol=1e-6)
        for values in motions.values():
            for key in ('R3','R4','R4s'):assert values[key].shape==(6,8,3) and np.isfinite(values[key]).all()
            np.testing.assert_array_equal(values['R4s'],values['R4'][v4.PERMUTATION])
            np.testing.assert_array_equal(values['R4s'][:,:3],values['R4'][:,:3])
        panel=v4.row_panel(packet,motions,true,state_id='implementation-check')
        assert all(row['status']=='available' for key,row in panel.items() if not key.startswith('R2b'))
        masks=[panel[key]['eligibility']['candidates'] for key in ('R2','R5c','R4/old_data','R4/maze_data')]
        assert all([r['observation_allowed'] for r in mask]==[r['observation_allowed'] for r in masks[0]] for mask in masks)
        spec=json.loads((source/'specification.json').read_text());geometry=v4.ArticulatedSteps(spec['geometry']['wall_boxes'])
        clearance=geometry.evaluate(traces[0]);lo=np.asarray(clearance['per_step_primitive_separation_lower_m']);up=np.asarray(clearance['per_step_primitive_separation_upper_m'])
        assert lo.shape==up.shape==(401,27) and np.all(lo<=up+1e-5)
        shift=np.asarray(clearance['per_interval_primitive_fk_displacement_m']);robust=np.asarray(clearance['per_interval_primitive_robust_lower_m'])
        np.testing.assert_allclose(robust,np.minimum(lo[:-1],lo[1:])-shift,rtol=0,atol=1e-12)
        safety=[{k:clearance[k] for k in ('hard','operating')} for _ in range(6)]
        # Adapter-shape check using this state's existing recorded trace only;
        # no scientific candidate safety counts are retained.
        f=v4.filter_observation(panel['R2']['eligibility'],safety,'operating');assert len(f['candidates'])==5
        packed,binding=v4.encode_snapshot(dict(physical=load_bound(p/'physical.pkl',state['physical']),decision=packet))
        sampler=v4.PhaseReservoir(2026092309,'implementation-six-states')
        sampler.consider(state['frame'],v4.active_objective(packet)['phase']);chosen=sampler.final();assert len(chosen)==1 and chosen[0]['weight']==1
        report.update(predicted_motion_matches_recorded=True,all_motion_adapter_shapes_pass=True,
            derangement_and_shared_prefix_pass=True,unchanged_observation_masks_pass=True,
            fk_bound_shapes_and_order_pass=True,filter_output_schema_pass=True,
            snapshot_lossless_encoded_bytes=len(packed),snapshot_raw_bytes=binding['raw_bytes'],
            row_selections_and_costs_discarded=True,phase_reservoir_member_check_pass=True)
        reports.append(report)
    output=Path('docs/go2_decision_headroom_v4_component_checks_2026-09-23.json')
    output.write_text(json.dumps(dict(status='PASS' if all(r['own_source_selection_parity'] for r in reports) else 'FAIL',checks=reports,wall_s=time.monotonic()-start,physics_steps=0,comparative_results_retained=False),indent=2)+'\n')
    print(json.dumps(reports,indent=2))

if __name__=='__main__':main()
