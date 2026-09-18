"""Separate visual-context coverage from distinct native motion outcomes."""
import json
from pathlib import Path

import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts import collect_go2_balanced_start_actions_development as collection

RESULT=Path('docs/go2_balanced_start_physical_diversity_2026-09-17.json')


def main():
    assert not RESULT.exists()
    terminal=json.loads(collection.RESULT.read_text());assert terminal['status']=='COMPLETE'
    groups={}
    fields=('base_pose_world','joint_position','joint_velocity','applied_command')
    for case,(_,action) in enumerate(collection.CASES):
        root=collection.OUTPUT/f'case_{case:02d}'
        row=json.loads((root/'result.json').read_text())
        assert row['complete_500ms'] and not row['disallowed_contact']
        audit=json.loads((root/'camera_audit.json').read_text())
        with np.load(root/'physics_trace.npz',allow_pickle=False) as a:
            arrays={k:a[k].copy() for k in fields}
        start,end=(arrays['base_pose_world'][audit[i]['physical_sample_index']] for i in (10,15))
        rot=rotation_xyzw(start[3:]);future_rot=rotation_xyzw(end[3:])
        xy=(rot.T@(end[:3]-start[:3]))[:2]
        heading=lambda r:np.arctan2(r[1,0],r[0,0])
        yaw=heading(future_rot)-heading(rot);yaw=np.arctan2(np.sin(yaw),np.cos(yaw))
        if action not in groups:
            groups[action]=dict(reference=arrays,reference_case=case,cases=[],
                                max_abs_difference={k:0. for k in fields},motions=[])
        group=groups[action];group['cases'].append(case);group['motions'].append([*map(float,xy),float(yaw)])
        for k in fields:
            assert arrays[k].shape==group['reference'][k].shape
            group['max_abs_difference'][k]=max(group['max_abs_difference'][k],float(np.max(np.abs(arrays[k]-group['reference'][k]))))
    rows=[]
    for action,group in groups.items():
        motion=np.asarray(group['motions'])
        rows.append(dict(action=action,cases=group['cases'],reference_case=group['reference_case'],
                         max_trace_absolute_difference=group['max_abs_difference'],
                         trace_fields_exact=all(v==0 for v in group['max_abs_difference'].values()),
                         motion_body_xy_m_and_world_yaw_rad=motion[0].tolist(),
                         maximum_motion_difference=float(np.max(np.abs(motion-motion[0])))))
    result=dict(status='COMPLETE',collection_sha256=collection.fit.digest(collection.RESULT),
                rows=rows,recordings=48,action_groups=6,visual_environment_groups=8,
                all_same_action_traces_exact=all(r['trace_fields_exact'] for r in rows),
                interpretation='If traces match, the collection supplies eight visual contexts for each of six motions, not 48 independent physical responses. It cannot demonstrate geometry-dependent collision dynamics.',
                source_sha256=collection.fit.digest(__file__),new_training=False,new_navigation=False)
    collection.fit.save(RESULT,result)
    print('BALANCED_START_DIVERSITY',json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
    for row in rows:print(row['action'],row['motion_body_xy_m_and_world_yaw_rad'],row['max_trace_absolute_difference'],flush=True)


if __name__=='__main__':main()
