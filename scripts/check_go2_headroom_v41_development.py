"""Amendment implementation checks on the same six qualified states only."""
import ast
import hashlib
import json
from pathlib import Path
import time
import tempfile
import numpy as np
from lewm import decision_headroom_v4_development as v4
from lewm import decision_headroom_v41_development as v41
from lewm.decision_headroom_reference_development import ReferenceGeometry, fixed_target_world
from scripts.run_go2_decision_headroom_branches_development import load_bound


def main():
    start=time.monotonic()
    config=json.loads(Path('docs/go2_decision_headroom_remaining_phase1_approved_v3_2026-09-23.json').read_text())
    root=Path(config['execution_caps']['output_root']);old=root.parent/'go2_decision_headroom_rgb_restore_recheck_v1_attempt_001'
    pairs=[(Path(s['source_root']),s,old/f'state_{i:02d}') for i,s in enumerate(config['previous_corrected_states'])]
    pairs += [(root/f'source_{i:02d}',json.loads((root/f'source_{i:02d}/snapshots.json').read_text())[0],root/f'source_{i:02d}/state_0132') for i in range(4)]
    checks=[];reader_records=[]
    for source,state,branch in pairs:
        packet=load_bound(source/f'state_{state["frame"]:04d}'/'decision.pkl',state['decision'])
        with np.load(source/'native/physics_trace.npz',allow_pickle=False) as a: anchor=a['base_pose_world'][749].copy()
        with np.load(branch/'hold_0/physics_trace.npz',allow_pickle=False) as a: trace={k:a[k].copy() for k in a.files}
        loc=v41.localisation(packet,anchor,trace['base_pose_world'][0]);assert loc['status']=='available'
        spec=json.loads((source/'specification.json').read_text())
        geometry=v4.ArticulatedSteps(spec['geometry']['wall_boxes']);safe=geometry.evaluate(trace)
        corrected=np.asarray(packet['source_correction']['applied_prediction_after_yaw_ablation'])
        motion=np.concatenate((corrected[:,:,:2],np.arctan2(corrected[:,:,2:3],corrected[:,:,3:4])),axis=-1)
        selection=v4.selector(packet,motion)
        gate=v4.eligibility(packet,selection)
        f=v4.filter_observation(gate,[safe]*6,'operating')
        record=dict(filter_audit={k:dict(operating=f) for k in ('R2','R5c','R4/old_data','R4/maze_data')})
        for q in ('excluded_safe','all_excluded_despite_safe'):
            pair=v41.paired_filter([(1.,record)],'R5c',q)
            assert pair['value'] in (0.,None)
        assert v41.binding_event(f) in (True,False,None)
        # Identity-row plumbing only: no cross-method scientific result.
        reader_records.append(dict(record,objective=v4.active_objective(packet),sampling=dict(weight=1.),
            rows={},localisation=loc,safety=[safe]*6,endpoint_clearance_strata=[],
            reference=dict(status='unresolved'),secondary_reference=dict(status='unresolved')))
        for value in reader_records[-1]['filter_audit'].values():value['hard']=f
        target=fixed_target_world(packet,anchor)
        secondary_checked=False
        if target['valid']:
            walls=[dict(center=w['centre_xyz'][:2],size=w['size_xyz'][:2],yaw=w['yaw_rad']) for w in spec['geometry']['wall_boxes']]
            primary=ReferenceGeometry(walls,[[-2.1,-2.1],[3.4,3.4]],target['target_xy_world'],radius_m=.46,clearance_m=.005,resolution_m=.02)
            secondary=v41.SecondaryReference(primary)
            for action in v4.ACTIONS:
                with np.load(branch/f'{action}_0/physics_trace.npz',allow_pickle=False) as a: point=a['base_pose_world'][-1,:2].copy()
                first=primary.distance_and_heading(point);second=secondary.distance_and_heading(point)
                assert first==primary.distance_and_heading(point)
                if first['valid']: assert first==second
                if second.get('secondary_inflation_escape'):
                    assert second['entry_path_m']>=np.linalg.norm(np.asarray(second['entry_xy'])-point)-1e-10
                    tail=primary.distance_and_heading(second['entry_xy'])
                    assert abs(second['distance_m']-tail['distance_m']-second['entry_path_m'])<1e-10
                secondary_checked=True
        checks.append(dict(decision_sha256=state['decision']['sha256'],localisation_schema=True,
            paired_identity_zero=True,binding_classification_schema=True,secondary_reference_checked=secondary_checked))
    paths=[Path('lewm/decision_headroom_v41_development.py')]+[Path('scripts')/name for name in (
        'run_go2_headroom_v41_development.py','run_go2_headroom_v41_source_development.py',
        'run_go2_headroom_v41_branches_development.py','read_go2_headroom_v41_development.py')]
    for p in paths: ast.parse(p.read_text())
    from scripts.read_go2_headroom_v41_development import report
    with tempfile.TemporaryDirectory(prefix='headroom_v41_six_state_') as temp:
        temp=Path(temp)
        for case,r in zip((0,1,2,6,7,8),reader_records,strict=True):
            p=temp/f'source_{case:02d}';(p/'state_0012').mkdir(parents=True)
            (p/'snapshots.json').write_text(json.dumps([dict(frame=12)]))
            (p/'state_0012/audit_v4.json').write_text(json.dumps(r))
        report(temp)
        analysis=json.loads((temp/'analysis_v41.json').read_text())
        assert len(analysis['primary_quantities'])==analysis['primary_family_size']==13
        assert analysis['continuous_regret_unzeroed']
    result=dict(status='PASS',checks=checks,physics_steps=0,new_states=0,comparative_results_retained=False,
        wall_s=time.monotonic()-start,source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    Path('docs/go2_decision_headroom_v41_component_checks_2026-09-23.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
