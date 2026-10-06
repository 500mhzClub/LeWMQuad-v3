"""One fixed direct-supervised fit; never uses development/validation outcomes."""
import gc
import hashlib
import json
from pathlib import Path
import time
import traceback

import numpy as np
import psutil
import torch
from torch.nn import functional as F

from lewm import decision_headroom_json_v42_development as output
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts.run_go2_navigation_capability_development import Budget, PROTOCOL, PROTOCOL_SHA, sha, save


def main():
    assert sha(PROTOCOL)==PROTOCOL_SHA
    protocol=json.loads(PROTOCOL.read_text());base=Path(protocol['output_root'])
    # Do not compete with an episode owner for the same device.
    for path in (base/'runs').glob('*/process.json'):
        row=json.loads(path.read_text())
        try:
            process=psutil.Process(row['pid'])
            if abs(process.create_time()-row['created'])<.01 and process.status()!=psutil.STATUS_ZOMBIE:
                raise RuntimeError('A harness owner is active; defer the fixed C4 training assignment')
        except psutil.NoSuchProcess:
            pass
    cfg=protocol['controllers']['C4'];data_root=base/'c4_preparation'
    prepared=json.loads((data_root/'result.json').read_text())
    assert prepared['all_inputs_present'] and prepared['all_roles_train'] and prepared['exact_readout_population']
    assert prepared['preregistration_sha256']==PROTOCOL_SHA
    root=base/'c4_fit_attempt002';root.mkdir(exist_ok=False);output.install(base)
    budget=Budget(base,protocol);budget.admit_persist(128*1024**2)
    save(root/'plan.json',dict(schema='navigation_capability_c4_fit.v1',preregistration_sha256=PROTOCOL_SHA,
        configuration=cfg,source_sha256={p:sha(p) for p in (__file__,'lewm/navigation_capability_supervised_development.py')},
        prepared_data_sha256={name:sha(data_root/name) for name in ('samples.json','frame_paths.json','result.json')},
        feature_dtype='pooled FP16 in RAM, as in selected readout fitting',encoder_compute_dtype='float32',
        checkpoint_selection='fixed final',model_seed=cfg['seed'],gpu_time_cap_s=43200))
    save(root/'process.json',dict(pid=psutil.Process().pid,created=psutil.Process().create_time()))
    paths=json.loads((data_root/'frame_paths.json').read_text())
    samples=json.loads((data_root/'samples.json').read_text())
    if psutil.virtual_memory().available < len(paths)*192*1024*2+10*1024**3:
        raise RuntimeError('RAM unavailable for fixed training feature cache')
    predecessor=json.loads((base/'c4_fit_attempt001/failure.json').read_text())
    prior_gpu_s=predecessor['gpu_owner_wall_s']
    save(root/'predecessor.json',dict(path=str(base/'c4_fit_attempt001'),failure_sha256=sha(base/'c4_fit_attempt001/failure.json'),
        reason='TorchVersion str subclass must be explicitly recorded as a native str; converter remains unchanged',
        encoder_calls=0,optimizer_updates=0,prior_gpu_owner_wall_s=prior_gpu_s))
    started=time.monotonic();error=None
    def check():
        budget.check()
        if prior_gpu_s+time.monotonic()-started >= 43200-60:
            raise RuntimeError('12-GPU-hour C4 cap closeout boundary')
    try:
        torch.set_num_threads(4);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed'])
        assert torch.cuda.is_available()
        device=torch.device('cuda:0')
        save(root/'device.json',dict(name=str(torch.cuda.get_device_name(0)),torch=str(torch.__version__),hip=str(torch.version.hip),
            total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,encoder_sha256=protocol['controllers']['C3']['encoder_binding']['sha256']))
        encoder=VJepa21Arm();encoder.build(device,torch.float32)
        features=torch.empty(len(paths),192,1024,dtype=torch.float16)
        with (root/'progress.jsonl').open('x') as progress:
            with torch.no_grad():
                for index,path in enumerate(paths):
                    check()
                    pixels=encoder.preprocess(path)[None].to(device)
                    tokens=F.layer_norm(encoder.tokens(pixels).float(),(1024,))
                    if not torch.isfinite(tokens).all():raise ValueError('nonfinite frozen encoder features')
                    features[index]=pool_tokens(tokens)[0].half().cpu()
                    if (index+1)%100==0:
                        progress.write(json.dumps(dict(stage='encoding',frames=index+1,total=len(paths),wall_s=time.monotonic()-started))+'\n')
            del encoder,pixels,tokens;gc.collect();torch.cuda.empty_cache()
            head=torch.load(protocol['controllers']['C3']['head_binding']['path'],map_location='cpu',weights_only=False)['model_state_dict']
            model=DirectMotionPredictor(head['target_mean'],head['target_scale']).to(device)
            count=sum(p.numel() for p in model.parameters())
            assert count==cfg['architecture']['estimated_trainable_parameters']
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['optimizer']['learning_rate'],weight_decay=cfg['optimizer']['weight_decay'])
            frame_ids=torch.tensor([s['frame_indices'] for s in samples],dtype=torch.long)
            control=torch.tensor([s['control'] for s in samples],dtype=torch.float32)
            actions=torch.tensor([s['future_actions'] for s in samples],dtype=torch.float32)
            targets=torch.tensor([s['targets'] for s in samples],dtype=torch.float32)
            old=np.array([i for i,s in enumerate(samples) if s['group']=='old'])
            maze=np.array([i for i,s in enumerate(samples) if s['group']=='maze'])
            rng=np.random.default_rng(cfg['seed']);queues={}
            def draw(key,pool):
                values=[]
                while len(values)<32:
                    if not queues.get(key):queues[key]=rng.permutation(pool).tolist()
                    values.append(queues[key].pop())
                return values
            horizon_schedule=np.tile(np.arange(8),cfg['optimizer']['updates']*64//8)
            rng.shuffle(horizon_schedule);horizon_schedule=horizon_schedule.reshape(-1,64)
            losses=[]
            for step in range(cfg['optimizer']['updates']):
                check()
                indices=torch.tensor(draw('old',old)+draw('maze',maze))
                horizon=torch.tensor(horizon_schedule[step],dtype=torch.long)
                x=features[frame_ids[indices]].float().to(device)
                y=targets[indices,horizon].to(device)
                optimizer.zero_grad(set_to_none=True)
                prediction=model.normalized(x,control[indices].to(device),actions[indices].to(device),(horizon+1).to(device))
                loss=F.mse_loss(prediction,(y-model.target_mean)/model.target_scale)
                assert torch.isfinite(loss)
                loss.backward()
                assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['optimizer']['gradient_clip']))
                optimizer.step();losses.append(float(loss.detach()))
                if (step+1)%20==0:
                    progress.write(json.dumps(dict(stage='fitting',updates=step+1,normalized_mse=float(np.mean(losses[-20:])),wall_s=time.monotonic()-started))+'\n')
            path=root/'direct_final.pt'
            with path.open('xb') as stream:
                torch.save(dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                    updates=cfg['optimizer']['updates'],plan_sha256=sha(root/'plan.json'),parameter_count=count),stream)
            save(root/'result.json',dict(status='COMPLETE',checkpoint_sha256=sha(path),parameter_count=count,
                updates=cfg['optimizer']['updates'],gpu_owner_wall_s=prior_gpu_s+time.monotonic()-started,
                training_render_provenance='unverified',validation_used=False,selection='fixed final'))
    except BaseException as exc:
        error=exc
        save(root/'failure.json',dict(reason=repr(exc),traceback=traceback.format_exc(),gpu_owner_wall_s=prior_gpu_s+time.monotonic()-started,
            failed_attempt_preserved=True,automatic_retry=False))
    if error is not None:raise error


if __name__=='__main__':main()
