"""Training-image diagnostic: does a within-trajectory metric separate turns?"""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from lewm import geometry_progress_layout_family_development as family
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from scripts import train_go2_dense_goal_metric_development as fit

RESULT=Path('docs/go2_goal_metric_turn_separation_2026-09-17.json')


@torch.inference_mode()
def main():
    assert not RESULT.exists();torch.set_num_threads(4);start=time.monotonic()
    paths=[Path(p) for p in json.loads((fit.OUTPUT/'frame_paths.json').read_text())]
    lookup={str(p):i for i,p in enumerate(paths)}
    with np.load(fit.OUTPUT/'pairs.npz',allow_pickle=False) as archive:pairs=archive['indices']
    same_recording=all(paths[i].parent==paths[j].parent for i,j in pairs)
    assert same_recording
    pair_set={tuple(p) for p in pairs.tolist()}
    assignments=family.assignments();source=fit.parent.ROOTS['family'];jobs=[]
    for name,cell in family.layouts().items():
        if cell['role']!='train':continue
        selected={action:next(trial for trial,a in assignments.items()
            if a['geometry']==name and a['appearance_seed']==2026090940 and a['action']==action)
            for action in ('left_turn','right_turn')}
        images=[source/selected['left_turn']/'rgb_0003.png',source/selected['left_turn']/'rgb_0023.png',
            source/selected['right_turn']/'rgb_0023.png']
        right_initial=source/selected['right_turn']/'rgb_0003.png'
        assert images[0].read_bytes()==right_initial.read_bytes()
        assert all(str(p) in lookup for p in images+[right_initial])
        # Both same-trajectory edges are explicitly in the fitted supervision.
        assert (lookup[str(images[0])],lookup[str(images[1])]) in pair_set
        assert (lookup[str(right_initial)],lookup[str(images[2])]) in pair_set
        assert (lookup[str(images[1])],lookup[str(images[2])]) not in pair_set
        jobs.append((name,images))
    model=fit.load().cuda();base=fit.parent.reference
    encoder=base.encoders.VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    rows=[]
    for name,images in jobs:
        embeddings=[];states=[]
        for p in images:
            feature=F.layer_norm(encoder.tokens(encoder.preprocess(str(p))[None].cuda()).float(),(1024,))
            embeddings.append(model.embed(pool_tokens(feature))[0])
            cameras=json.loads((p.parent/'camera_audit.json').read_text());index=int(p.stem.split('_')[1])
            with np.load(p.parent/'physics_trace.npz',allow_pickle=False) as a:pose=a['base_pose_world'][cameras[index]['physical_sample_index']]
            rotation=rotation_xyzw(pose[3:]);states.append(np.array([pose[0],pose[1],np.arctan2(rotation[1,0],rotation[0,0])]))
        comparisons=[]
        for label,i,j in (('initial_to_left',0,1),('initial_to_right',0,2),('left_to_right',1,2)):
            delta=states[j]-states[i];delta[2]=np.arctan2(np.sin(delta[2]),np.cos(delta[2]))
            true=float(np.sum((delta/np.array([.03,.03,np.deg2rad(5)]))**2))
            predicted=float((embeddings[i]-embeddings[j]).square().mean())
            comparisons.append(dict(pair=label,supervised_pair=label!='left_to_right',physical_cost=true,
                learned_cost=predicted,predicted_over_physical=predicted/true,heading_separation_deg=float(abs(np.rad2deg(delta[2])))))
        rows.append(dict(layout=name,images=[str(p) for p in images],comparisons=comparisons))
    report=dict(status='COMPLETE',rows=rows,training_pairs=len(pairs),all_pairs_within_recording=same_recording,
        training_layouts=len(rows),source_sha256=fit.digest(__file__),metric_fit_sha256=fit.digest(fit.RESULT),
        mean_cross_turn_cost_ratio=float(np.mean([r['comparisons'][2]['predicted_over_physical'] for r in rows])),
        wall_s=time.monotonic()-start,no_training=True,new_navigation=False,
        limitations=['training images and geometry only; same-trajectory edges are in-sample',
            'cross-trajectory edge never supervised despite both endpoint images being used in fitting',
            'four related training layouts, one appearance and one time offset',
            'diagnosis of supervision coverage, not evidence that adding pairs will fix navigation'])
    fit.save(RESULT,report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
