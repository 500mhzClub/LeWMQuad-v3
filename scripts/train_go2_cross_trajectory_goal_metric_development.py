"""Matched-budget goal metric with half of pairs spanning matched recordings."""
import argparse
from collections import defaultdict, Counter
import json
import os
from pathlib import Path
import shutil
import traceback

import numpy as np

from lewm.eligible_floor_registration_development import bind
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_dense_goal_metric_development as original

OUTPUT=original.parent.ROOTS['family'].parent/'go2_cross_trajectory_goal_metric_v1_attempt_001'
PLAN=Path('docs/go2_cross_trajectory_goal_metric_plan_2026-09-17.json')
RESULT=Path('docs/go2_cross_trajectory_goal_metric_fit_result_2026-09-17.json')
SEED=original.SEED
PAIR_SEED=SEED+100


def prepare():
    assert not OUTPUT.exists() and not PLAN.exists()
    old_plan=json.loads(original.PLAN.read_text())
    old_result=json.loads(original.RESULT.read_text());assert old_result['status']=='COMPLETE'
    paths=[Path(p) for p in json.loads((original.OUTPUT/'frame_paths.json').read_text())]
    with np.load(original.OUTPUT/'pairs.npz',allow_pickle=False) as a:
        indices=a['indices'].copy();old_targets=a['target'].copy()
    admitted={original.parent.ROOTS[m['source']]/m['trial']:m
        for m in json.loads((original.OUTPUT/'recordings.json').read_text())}
    by_directory=defaultdict(list)
    for index,path in enumerate(paths):
        assert path.parent in admitted and admitted[path.parent]['role']=='train'
        by_directory[path.parent].append(index)
    states=np.zeros((len(paths),3),np.float64);keys={};groups=defaultdict(list)
    fields=('geometry','procedural_seed','appearance_arm','appearance_seed','friction_mu',
        'render_near_m','visual_surface_contract')
    for directory,items in sorted(by_directory.items()):
        spec=json.loads((directory/'specification.json').read_text());assert spec['data_role']=='train'
        gains=json.loads((directory/'actuator_identity.json').read_text())
        identity={k:spec[k] for k in fields}|dict(actuator_gains=gains['effective'])
        key=json.dumps(identity,sort_keys=True);keys[directory]=key;groups[key].append(directory)
        cameras=json.loads((directory/'camera_audit.json').read_text())
        with np.load(directory/'physics_trace.npz',allow_pickle=False) as archive:
            poses=archive['base_pose_world'];contact_seen=np.maximum.accumulate(archive['physics_contact'])
            for index in items:
                frame=int(paths[index].stem.split('_')[1]);sample=cameras[frame]['physical_sample_index']
                assert not contact_seen[sample]
                pose=poses[sample];rotation=rotation_xyzw(pose[3:])
                states[index]=[pose[0],pose[1],np.arctan2(rotation[1,0],rotation[0,0])]
    def targets(pairs):
        delta=states[pairs[:,1]]-states[pairs[:,0]]
        delta[:,2]=np.arctan2(np.sin(delta[:,2]),np.cos(delta[:,2]))
        return np.sum((delta/np.array([.03,.03,np.deg2rad(5)]))**2,axis=1).astype(np.float32)
    np.testing.assert_allclose(targets(indices),old_targets,rtol=1e-6,atol=1e-5)
    assert all(paths[a].parent==paths[b].parent for a,b in indices)
    rng=np.random.default_rng(PAIR_SEED)
    replaced=np.sort(rng.permutation(len(indices))[:len(indices)//2])
    cross=np.zeros(len(indices),bool);cross[replaced]=True
    used={tuple(sorted(pair)) for pair in indices[~cross].tolist()}
    for position in replaced:
        anchor=int(indices[position,0]);directory=paths[anchor].parent
        others=[d for d in groups[keys[directory]] if d!=directory];assert others
        for _ in range(10000):
            other=others[int(rng.integers(len(others)))];pool=by_directory[other]
            endpoint=int(pool[int(rng.integers(len(pool)))]);pair=tuple(sorted((anchor,endpoint)))
            if pair not in used:break
        else:raise RuntimeError('cross-recording pair pool exhausted')
        used.add(pair);indices[position,1]=endpoint
    assert len(used)==len(indices)
    assert all((paths[a].parent!=paths[b].parent)==bool(is_cross) for (a,b),is_cross in zip(indices,cross))
    assert all(keys[paths[a].parent]==keys[paths[b].parent] for a,b in indices)
    target=targets(indices);assert np.isfinite(target).all()
    available=shutil.disk_usage(OUTPUT.parent).free;assert available>(512+24)*1024**2
    OUTPUT.mkdir()
    np.savez_compressed(OUTPUT/'pairs.npz',indices=indices,target=target,cross_recording=cross)
    (OUTPUT/'frame_paths.json').write_bytes((original.OUTPUT/'frame_paths.json').read_bytes())
    original.save(OUTPUT/'recordings.json',list(admitted.values()))
    plan=old_plan|dict(source_sha256={p:original.digest(p) for p in
        (__file__,'scripts/train_go2_dense_goal_metric_development.py','lewm/dense_goal_metric_development.py',
         'scripts/train_go2_dense_visual_motion_readout_development.py','lewm/dense_visual_motion_readout_development.py')},
        pairs_sha256=original.digest(OUTPUT/'pairs.npz'),paths_sha256=original.digest(OUTPUT/'frame_paths.json'),
        predecessor_fit_sha256=original.digest(original.RESULT),pair_seed=PAIR_SEED,
        intervention='replace exactly half of second endpoints with another matched training recording; keep every first endpoint',
        within_recording_pairs=int((~cross).sum()),cross_recording_pairs=int(cross.sum()),
        matching_fields=list(fields)+['effective actuator gains'],matched_environment_groups=len(groups),
        records_per_environment=sorted(len(g) for g in groups.values()),
        all_frame_paths_identical=True,first_endpoint_exposures_unchanged=True,second_endpoint_exposures_changed=True,
        original_pairs_sha256=original.digest(original.OUTPUT/'pairs.npz'),
        cross_anchor_sources=dict(Counter(admitted[paths[int(indices[i,0])].parent]['source'] for i in replaced)),
        target_quantiles=np.quantile(target,[0,.25,.5,.75,.9,1]).tolist(),
        resources=dict(output_free_bytes=available,available_ram_gib=73,encoder_batch=8,
            feature_cache='approximately 2.03 GiB pooled FP16 in RAM only',cpu_affinity=[8,9,10,11],competing_experiments=0),
        fit_loop='reuse original fit code, architecture, seed, epochs, batch, optimizer and frame-path order',
        labels='native planar poses in matching geometry/spawn/world frame; no post-contact frame or transfer image',
        evaluation='fixed final epoch: turn separation, exposed action rankings, then prospective control if warranted',
        limitations=['changes pair coverage and second-endpoint frequencies together',
            'four related training geometries; extra pair labels, not extra images or training steps',
            'goal-head supervision experiment, not an isolated JEPA representation-learning claim',
            'no new navigation result from fitting'])
    original.save(PLAN,plan);original.save(OUTPUT/'plan.json',plan)
    print('CROSS_TRAJECTORY_METRIC_PREPARED',json.dumps({k:plan[k] for k in
        ('pairs','within_recording_pairs','cross_recording_pairs','matched_environment_groups','records_per_environment','target_quantiles')}),flush=True)


def fit():
    assert not (OUTPUT/'process.json').exists()
    original.save(OUTPUT/'process.json',dict(pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0))))
    bind(original.fit,OUTPUT=OUTPUT,PLAN=PLAN,RESULT=RESULT)()


load=bind(original.load,OUTPUT=OUTPUT)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');a=p.parse_args()
    try:prepare() if a.prepare else fit()
    except Exception as error:
        if OUTPUT.exists() and not (OUTPUT/'failure.json').exists():
            original.save(OUTPUT/'failure.json',dict(reason=repr(error),traceback=traceback.format_exc()))
        raise
