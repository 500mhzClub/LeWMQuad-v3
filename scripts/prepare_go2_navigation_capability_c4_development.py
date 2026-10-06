"""Inventory and fix direct-predictor data from the selected readout population.

No training, GPU execution, historical rendering, evaluation or new data.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
from lewm import decision_headroom_json_v42_development as output


def main():
    prereg_path=Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json')
    protocol=json.loads(prereg_path.read_text())
    root=Path(protocol['output_root'])/'c4_preparation'
    root.mkdir(exist_ok=False);output.install(root)
    bindings=protocol['controllers']['C4']['data_bindings']
    for binding in bindings.values():
        assert hashlib.sha256(Path(binding['path']).read_bytes()).hexdigest()==binding['sha256']
    data=json.loads(Path(bindings['samples.json']['path']).read_text())
    oldpaths=json.loads(Path(bindings['frame_paths.json']['path']).read_text())
    stats=json.loads(Path(protocol['harness_v0']['shared_model_and_sensor_bindings']['normalization']['path']).read_text())
    mean,std=[np.asarray(stats[k],np.float32) for k in ('control_mean','control_std')]
    paths=[];lookup={};recordings={};rows=[];groups=[]
    def add(path):
        path=path.resolve()
        if any(s=='sealed' or s.startswith('sealed_') or s=='sealed_test.json' for s in path.parts):
            raise ValueError('sealed input forbidden')
        assert path.is_file()
        key=str(path)
        if key not in lookup:
            lookup[key]=len(paths);paths.append(key)
        return lookup[key]
    for group in ('old','maze'):
        for context,(pairs,targets) in enumerate(zip(data[group+'_pairs'],data[group+'_targets'],strict=True)):
            assert len(pairs)==len(targets)==8
            current=Path(oldpaths[pairs[0][0]])
            frame=int(current.stem.split('_')[-1]);directory=current.parent
            if directory not in recordings:
                spec=json.loads((directory/'specification.json').read_text())
                assert spec['data_role']=='train',directory
                with np.load(directory/'policy_histories.npz',allow_pickle=False) as a:
                    histories={k:a['applied_command_'+k].copy() for k in ('values','valid','measured_ns','available_ns')}
                meta=json.loads((directory/'policy_observations.json').read_text())
                recordings[directory]=(histories,meta)
            histories,meta=recordings[directory]
            indices=(frame-10,frame-5,frame)
            assert min(indices)>=0
            stamps=[meta['frames'][i]['image_ns'] for i in indices]
            assert np.diff(stamps).tolist()==[500_000_000]*2
            assert histories['measured_ns'][frame][[4,9,14]].tolist()==stamps
            assert histories['valid'][frame].all() and (histories['available_ns'][frame]<=stamps[-1]).all()
            features=[add(directory/f'rgb_{i:04d}.png') for i in indices]
            past=histories['values'][frame].astype(np.float32)
            assert np.all(past[:,1]==0)
            future=[]
            for h in range(1,9):
                assert oldpaths[pairs[h-1][0]]==str(current)
                assert Path(oldpaths[pairs[h-1][1]])==directory/f'rgb_{frame+h:04d}.png'
                assert meta['frames'][frame+h]['image_ns']==stamps[-1]+h*100_000_000
                assert histories['valid'][frame+h,-h:].all()
                commands=histories['values'][frame+h,-h:]
                assert np.all(commands[:,1]==0)
                if future:
                    np.testing.assert_array_equal(commands[:-1],np.asarray(future))
                future=commands.tolist()
            rows.append(dict(group=group,context_index=context,frame_indices=features,
                control=((past[:,[0,2]].reshape(3,5,2)-mean)/std).tolist(),
                future_actions=np.asarray(future)[:,[0,2]].tolist(),targets=targets))
            groups.append(group)
    result=dict(schema='navigation_capability_c4_preparation.v1',
        preregistration_sha256=hashlib.sha256(prereg_path.read_bytes()).hexdigest(),
        contexts=len(rows),old_contexts=groups.count('old'),maze_contexts=groups.count('maze'),
        causal_frames=len(paths),source_recordings=len(recordings),all_inputs_present=True,
        exact_readout_population=True,all_roles_train=True,gpu_executed=False,fit_executed=False,
        training_render_provenance='unverified',
        input_bindings={str(directory/name):hashlib.sha256((directory/name).read_bytes()).hexdigest()
                        for directory in recordings for name in ('specification.json','policy_histories.npz','policy_observations.json')})
    for name,value in [('samples.json',rows),('frame_paths.json',paths),('result.json',result)]:
        with (root/name).open('x') as stream:json.dump(value,stream,separators=(',',':'));stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k!='input_bindings'}))


if __name__=='__main__':main()
