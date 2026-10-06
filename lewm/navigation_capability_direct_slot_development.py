"""Fixed direct-supervised predictor at the deployed dense-motion interface."""
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import yaml

from lewm.dense_horizon_navigation_development import DenseHorizonNavigationModel
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.navigation_capability_supervised_development import DirectMotionPredictor
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts.dev_frozen_dense_representation_encoders_v1 import VJepa21Arm


class DirectNavigationModel(nn.Module):
    set_native_context=DenseHorizonNavigationModel.set_native_context

    def __init__(self, encoder, predictor, limits, mean, std, identity):
        super().__init__();self.encoder=encoder;self.predictor=predictor
        self.limits=limits;self.readout_identity=identity;self.pending_context=None;self.receipts=[]
        self.register_buffer('control_mean',torch.tensor(mean,dtype=torch.float32))
        self.register_buffer('control_std',torch.tensor(std,dtype=torch.float32))
        self.eval().requires_grad_(False)

    @torch.inference_mode()
    def forward(self, *, observation_history, known_action_blocks, known_action_valid):
        native,self.pending_context=self.pending_context,None
        if native is None or known_action_blocks.shape!=(6,8,1,3) or not known_action_valid.all():
            raise ValueError('complete native context and six dense candidate tapes required')
        requested=(known_action_blocks[:,:,0].cpu()*torch.tensor([.3,1.,.5])).numpy()
        last=tuple(native['past_applied_commands'][-1].tolist())
        applied=np.asarray([apply_safety_limits_single(row.tolist(),last,self.limits)[0] for row in requested],np.float32)
        assert np.all(applied[:,:,1]==0)
        started=time.perf_counter_ns();device=next(self.predictor.parameters()).device
        features=pool_tokens(F.layer_norm(self.encoder.tokens(native['pixels'].to(device)).float(),(1024,)))
        past=native['past_applied_commands'][:,[0,2]].reshape(3,5,2).to(device)
        past=(past-self.control_mean)/self.control_std
        actions=torch.tensor(applied[:,:,[0,2]],device=device)
        horizons=torch.arange(1,9,device=device).repeat(6)
        motion=self.predictor(features[None].expand(48,-1,-1,-1),past[None].expand(48,-1,-1,-1),
            actions[:,None].expand(-1,8,-1,-1).reshape(48,8,2),horizons).reshape(6,8,3).cpu()
        if not torch.isfinite(motion).all():raise ValueError('nonfinite direct motion prediction')
        outcomes=torch.cat((motion[:,:,:2],motion[:,:,2:3].sin(),motion[:,:,2:3].cos(),torch.full((6,8,1),-1000.)),dim=-1)
        self.receipts.append(dict(observed_ns=int(native['context_times_ns'][-1]),context_times_ns=native['context_times_ns'].tolist(),
            requested_commands=requested.tolist(),applied_commands=applied.tolist(),last_applied_command=list(last),
            motion_xy_yaw=motion.tolist(),wall_ns=time.perf_counter_ns()-started,contact_prediction_available=False,
            model='fixed direct supervised',physical_readout_training_horizons_ms=list(range(100,801,100))))
        return dict(rollout_outcomes=outcomes,target_offsets_ns=torch.arange(1,9).mul(100_000_000).expand(6,8),
                    prediction_valid=torch.ones(6,8,dtype=torch.bool),contact_prediction_available=False)


def load(protocol):
    import hashlib
    root=Path(protocol['output_root'])/'c4_fit_attempt002'
    result=json.loads((root/'result.json').read_text());path=root/'direct_final.pt'
    assert result['status']=='COMPLETE' and hashlib.sha256(path.read_bytes()).hexdigest()==result['checkpoint_sha256']
    state=torch.load(path,map_location='cpu',weights_only=False)
    assert state['updates']==protocol['controllers']['C4']['optimizer']['updates']
    weights=state['model_state_dict']
    predictor=DirectMotionPredictor(weights['target_mean'],weights['target_scale'])
    predictor.load_state_dict(weights);predictor.cuda().eval().requires_grad_(False)
    encoder=VJepa21Arm();encoder.build(torch.device('cuda:0'),torch.float32)
    stats=json.loads(Path(protocol['harness_v0']['shared_model_and_sensor_bindings']['normalization']['path']).read_text())
    limits=SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    return DirectNavigationModel(encoder,predictor,limits,stats['control_mean'],stats['control_std'],
        dict(arm='direct_supervised',path=str(path),sha256=result['checkpoint_sha256'],
             training_horizons_ms=list(range(100,801,100)),training_render_provenance='unverified')).cuda().eval()
