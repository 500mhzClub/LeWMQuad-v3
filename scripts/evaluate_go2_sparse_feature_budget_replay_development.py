"""Evaluate recorded/replayed poses against physics after sensor-only replay."""
import argparse
import json

import numpy as np

from lewm.physical_execution_development import rotation_xyzw
from scripts import run_go2_route_turn_memory_transfer_development as run


def main(directory_name='sparse_feature_budget_replay_v1'):
    root=run.BASE/run.root_name(1);directory=root/directory_name
    output=directory/'pose_accuracy_v1.json'
    if output.exists():raise ValueError('preserve completed evaluation')
    replay=json.loads((directory/'result.json').read_text())
    baseline={r['frame']:r['raw_pose'] for r in json.loads((root/'poses.json').read_text())}
    changed={r['frame']:r['pose'] for r in replay['rows'] if r['pose'] is not None}
    frames={r['frame']:r['physical_sample_index'] for r in json.loads(
        (root/'native/in_memory_camera_observations.json').read_text())['frames']}
    with np.load(root/'native/physics_trace.npz',allow_pickle=False) as data:
        physics=data['base_pose_world'].copy()
    origin=physics[frames[0]];basis=rotation_xyzw(origin[3:])
    for poses in (baseline,changed):
        np.testing.assert_allclose(poses[0]['position_initial_body_m'],0.,atol=1e-12,rtol=0)
        np.testing.assert_allclose(poses[0]['rotation_initial_body_from_current_body'],np.eye(3),atol=1e-12,rtol=0)
    common=sorted(set(baseline)&set(changed))
    def evaluate(poses,indices):
        rows=[]
        for frame in indices:
            truth=physics[frames[frame]]
            expected=basis.T@(truth[:3]-origin[:3])
            error=np.asarray(poses[frame]['position_initial_body_m'])-expected
            rotation=basis.T@rotation_xyzw(truth[3:])
            difference=np.asarray(poses[frame]['rotation_initial_body_from_current_body']).T@rotation
            angle=np.arccos(np.clip((np.trace(difference)-1)/2,-1.,1.))
            rows.append(dict(frame=frame,xy_error_mm=float(1000*np.linalg.norm(error[:2])),
                xyz_error_mm=float(1000*np.linalg.norm(error)),rotation_error_deg=float(np.degrees(angle))))
        return dict(frames=len(rows),xy_rmse_mm=float(np.sqrt(np.mean([r['xy_error_mm']**2 for r in rows]))),
            xy_max_mm=max(r['xy_error_mm'] for r in rows),
            xyz_rmse_mm=float(np.sqrt(np.mean([r['xyz_error_mm']**2 for r in rows]))),
            xyz_max_mm=max(r['xyz_error_mm'] for r in rows),
            rotation_max_deg=max(r['rotation_error_deg'] for r in rows),rows=rows) if rows else dict(frames=0)
    result=dict(baseline_on_common=evaluate(baseline,common),changed_on_common=evaluate(changed,common),
        changed_all_accepted=evaluate(changed,sorted(changed)),
        extension_after_original_failure=evaluate(changed,[f for f in sorted(changed) if f>=replay['baseline_failure_frame']]),
        native_state_evaluator_only=True,physics_used_for_tracking=False,
        original_camera_and_command_sequence=True,alternative_navigation_outcome_proven=False,
        coordinate_reference='physical body pose at recorded camera frame zero',
        error_bound_or_hardware_accuracy_established=False)
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:({kk:vv for kk,vv in v.items() if kk!='rows'} if isinstance(v,dict) else v)
        for k,v in result.items()},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory', default='sparse_feature_budget_replay_v1',
        choices=('sparse_feature_budget_replay_v1', 'sparse_corner_completion_replay_v1'))
    main(parser.parse_args().directory)
