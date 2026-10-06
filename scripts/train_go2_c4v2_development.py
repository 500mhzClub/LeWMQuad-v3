"""C4-v2 refit on the matched C3-v2 data (pre-declared 29 Sep 2026, §4).

The C4-v1 fit unchanged (architecture, frozen V-JEPA pooled features, 1,760 updates, AdamW,
seed, 32 old + 32 maze-pool batches, fixed final), with the maze pool extended by the new
fit-split rest/turn contexts. The 12-GPU-hour C4 cap counts C4-v1's GPU time.
"""
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
from lewm import navigation_capability_active_wall_development as wall
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm
from scripts import run_go2_navigation_capability_completed_support_v4_development as owner

PREREG = Path('docs/go2_navigation_capability_preregistration_v1_2026-09-25.json')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    protocol = json.loads(PREREG.read_text())
    base = Path(protocol['output_root'])
    cfg = protocol['controllers']['C4']
    data_root, root = base/'c3v2_data_v1', base/'c4v2_fit_v1'
    v1 = json.loads((base/'c4_fit_attempt002/result.json').read_text())
    prior_gpu_s = v1['gpu_owner_wall_s']
    root.mkdir(exist_ok=False)
    output.install(base)
    budget = owner.Budget(base, json.loads(owner.PROTOCOL.read_text()))
    owner.save(root/'plan.json', dict(schema='c4v2_fit.v1', predeclaration_sha256=sha('docs/go2_navigation_c3v2_readout_fix_predeclaration_2026-09-29.md'),
        configuration_from='C4-v1 preregistered configuration, unchanged', source_sha256={p: sha(p) for p in (__file__, 'lewm/navigation_capability_supervised_development.py')},
        prepared_data_sha256={n: sha(data_root/n) for n in ('train_samples.json', 'frame_paths.json', 'result.json')},
        model_seed=cfg['seed'], updates=cfg['optimizer']['updates'], checkpoint_selection='fixed final',
        prior_c4_gpu_s=prior_gpu_s, gpu_time_cap_s=43200, heldout_used=False))
    paths = json.loads((data_root/'frame_paths.json').read_text())
    samples = json.loads((data_root/'train_samples.json').read_text())
    if psutil.virtual_memory().available < len(paths)*192*1024*2+10*1024**3:
        raise RuntimeError('RAM unavailable for the training feature cache')
    started = time.monotonic()

    def check():
        budget.check()
        if prior_gpu_s+time.monotonic()-started >= 43200-60:
            raise RuntimeError('12-GPU-hour C4 cap closeout boundary')
    try:
        with wall.job(base, 'C4-v2 refit'):
            torch.set_num_threads(4)
            torch.manual_seed(cfg['seed'])
            np.random.seed(cfg['seed'])
            device = torch.device('cuda:0')
            encoder = VJepa21Arm()
            encoder.build(device, torch.float32)
            features = torch.empty(len(paths), 192, 1024, dtype=torch.float16)
            with (root/'progress.jsonl').open('x') as progress:
                with torch.no_grad():
                    for index, path in enumerate(paths):
                        check()
                        tokens = F.layer_norm(encoder.tokens(encoder.preprocess(path)[None].to(device)).float(), (1024,))
                        if not torch.isfinite(tokens).all():
                            raise ValueError('nonfinite frozen encoder features')
                        features[index] = pool_tokens(tokens)[0].half().cpu()
                        if (index+1) % 500 == 0:
                            progress.write(json.dumps(dict(stage='encoding', frames=index+1, total=len(paths), wall_s=time.monotonic()-started))+'\n')
                            progress.flush()
                del encoder
                gc.collect()
                torch.cuda.empty_cache()
                head = torch.load(protocol['controllers']['C3']['head_binding']['path'], map_location='cpu', weights_only=False)['model_state_dict']
                model = DirectMotionPredictor(head['target_mean'], head['target_scale']).to(device)
                count = sum(p.numel() for p in model.parameters())
                assert count == cfg['architecture']['estimated_trainable_parameters']
                optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['optimizer']['learning_rate'], weight_decay=cfg['optimizer']['weight_decay'])
                frame_ids = torch.tensor([s['frame_indices'] for s in samples], dtype=torch.long)
                control = torch.tensor([s['control'] for s in samples], dtype=torch.float32)
                actions = torch.tensor([s['future_actions'] for s in samples], dtype=torch.float32)
                targets = torch.tensor([s['targets'] for s in samples], dtype=torch.float32)
                old = np.array([i for i, s in enumerate(samples) if s['group'] == 'old'])
                maze = np.array([i for i, s in enumerate(samples) if s['group'] == 'maze'])
                rng = np.random.default_rng(cfg['seed'])
                queues = {}

                def draw(key, pool):
                    values = []
                    while len(values) < 32:
                        if not queues.get(key):
                            queues[key] = rng.permutation(pool).tolist()
                        values.append(queues[key].pop())
                    return values
                horizon_schedule = np.tile(np.arange(8), cfg['optimizer']['updates']*64//8)
                rng.shuffle(horizon_schedule)
                horizon_schedule = horizon_schedule.reshape(-1, 64)
                losses = []
                for step in range(cfg['optimizer']['updates']):
                    check()
                    indices = torch.tensor(draw('old', old)+draw('maze', maze))
                    horizon = torch.tensor(horizon_schedule[step], dtype=torch.long)
                    x = features[frame_ids[indices]].float().to(device)
                    y = targets[indices, horizon].to(device)
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model.normalized(x, control[indices].to(device), actions[indices].to(device), (horizon+1).to(device))
                    loss = F.mse_loss(prediction, (y-model.target_mean)/model.target_scale)
                    assert torch.isfinite(loss)
                    loss.backward()
                    assert torch.isfinite(torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['optimizer']['gradient_clip']))
                    optimizer.step()
                    losses.append(float(loss.detach()))
                    if (step+1) % 20 == 0:
                        progress.write(json.dumps(dict(stage='fitting', updates=step+1, normalized_mse=float(np.mean(losses[-20:])), wall_s=time.monotonic()-started))+'\n')
                        progress.flush()
            path = root/'direct_v2_final.pt'
            with path.open('xb') as stream:
                torch.save(dict(model_state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()}, updates=cfg['optimizer']['updates'],
                                plan_sha256=sha(root/'plan.json'), parameter_count=count, model_version='C4-v2'), stream)
            owner.save(root/'result.json', dict(status='COMPLETE', checkpoint_sha256=sha(path), parameter_count=count, updates=cfg['optimizer']['updates'],
                gpu_owner_wall_s_this_fit=time.monotonic()-started, gpu_owner_wall_s_total_c4=prior_gpu_s+time.monotonic()-started,
                training_render_provenance='unverified', validation_used=False, heldout_used=False, selection='fixed final'))
            print(json.dumps(dict(status='COMPLETE', checkpoint_sha256=sha(path), final_mse=float(np.mean(losses[-20:])))), flush=True)
    except BaseException as exc:
        owner.save(root/'failure.json', dict(reason=repr(exc), traceback=traceback.format_exc(), automatic_retry=False))
        raise


if __name__ == '__main__':
    main()
