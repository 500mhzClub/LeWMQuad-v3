"""CPU equivalence check before any GPU timing or navigation use of packing."""
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from lewm.batched_horizon_motion_development import batched_horizon_motion
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.terminal_translation_pulse_development import command_sequences
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import train_go2_horizon_dense_predictor_development as fit
from scripts import train_go2_dense_visual_motion_readout_development as motion_fit

OUTPUT = Path('docs/go2_batched_horizon_motion_cpu_check_2026-09-18.json')


@torch.inference_mode()
def main():
    assert not OUTPUT.exists()
    torch.set_num_threads(4)
    ref = fit.parent.reference
    row = next(r for r in ref.selected_rows() if r['data_role']=='train')
    directory, raw_control, _ = ref.inputs(row)
    encoder = ref.encoders.VJepa21Arm()
    encoder.build(torch.device('cpu'),torch.float32)
    pixels = torch.stack([encoder.preprocess(str(directory/f'rgb_{i:04d}.png')) for i in (3,8,13)])
    context = F.layer_norm(encoder.tokens(pixels).float(),(1024,))[None]
    del encoder
    stats = json.loads((ref.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    control = ((torch.tensor(raw_control)-torch.tensor(stats['control_mean']))/torch.tensor(stats['control_std']))[None].float()
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    requested = command_sequences(row['known_commands'][:3],pulse=False)
    last = (float(raw_control[-1,-1,0]),0.,float(raw_control[-1,-1,1]))
    actions = torch.tensor(np.asarray([apply_safety_limits_single(r.tolist(),last,limits)[0] for r in requested],np.float32)[:,:,[0,2]])
    head = motion_fit.load()
    records = []
    for arm in fit.ARMS:
        model = fit.load(arm)
        blind = arm=='no_future_action'
        start = time.monotonic()
        reference, counts = [], []
        for h in range(1,9):
            unique, inverse = torch.unique(actions[:,:h].reshape(6,-1),dim=0,return_inverse=True)
            indices = torch.stack([(inverse==j).nonzero()[0,0] for j in range(len(unique))])
            n = 1 if blind else len(unique)
            expansion = torch.zeros(6,dtype=torch.long) if blind else inverse
            value = model(context.expand(n,-1,-1,-1),actions[:1] if blind else actions[indices],
                torch.full((n,),h,dtype=torch.long),torch.ones(n,768,dtype=torch.bool),control=control.expand(n,-1,-1,-1))
            value = F.layer_norm(value.float(),(1024,))
            reference.append(head(pool_tokens(context[:,-1]).expand(n,-1,-1),pool_tokens(value))[expansion])
            counts.append(n)
        reference = torch.stack(reference,dim=1)
        sequential_s = time.monotonic()-start
        start = time.monotonic()
        actual, packed_counts = batched_horizon_motion(model,head,context,control,actions,action_blind=blind)
        packed_s = time.monotonic()-start
        torch.testing.assert_close(actual,reference,rtol=0,atol=2e-5)
        assert counts==packed_counts
        assert torch.equal(actual[:,:3],actual[:1,:3].expand(6,-1,-1))
        if blind:
            assert torch.equal(actual,actual[:1].expand(6,-1,-1))
        records.append(dict(arm=arm,counts=counts,total_packed_rows=sum(counts),
            maximum_absolute_motion_difference=float((actual-reference).abs().max()),
            sequential_cpu_s=sequential_s,packed_cpu_s=packed_s,
            common_prefix_exact=True,action_blind_exact=blind))
        print(records[-1],flush=True)
    fit.save(OUTPUT,dict(status='PASS',records=records,training_only=True,no_gpu=True,
        live_navigation_unchanged=True,gpu_speedup_not_established=True,
        source_sha256={p:fit.digest(p) for p in (__file__,'lewm/batched_horizon_motion_development.py')}))


if __name__=='__main__':
    main()
