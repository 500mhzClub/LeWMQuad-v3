"""Prospective C1/C2 workload omission; requires recorded-episode equivalence.

These finite interface placeholders are not motion predictions. C1 still uses
the unchanged fitted command model in its deployed correction/selection path;
C2 still uses the deployed reactive selector. No C0/C3/C4 use is permitted.
"""
import numpy as np
import torch

from lewm.dense_horizon_navigation_development import DenseHorizonNavigationModel
from lewm_genesis.lewm_contract import apply_safety_limits_single


class UnusedNeuralWorkload(torch.nn.Module):
    set_native_context=DenseHorizonNavigationModel.set_native_context

    def __init__(self, controller, limits):
        super().__init__()
        if controller not in ('C1','C2'):raise ValueError('only established non-neural selectors may omit this workload')
        self.controller=controller;self.limits=limits;self.pending_context=None;self.receipts=[]
        self.readout_identity=dict(arm='unused_neural_workload_omitted',training_horizons_ms=list(range(100,801,100)),
            interface_placeholder=True,actual_motion_source='fitted command history' if controller=='C1' else 'none/reactive')
        self.eval()

    def forward(self, *, observation_history, known_action_blocks, known_action_valid):
        native,self.pending_context=self.pending_context,None
        if native is None or known_action_blocks.shape!=(6,8,1,3) or not known_action_valid.all():
            raise ValueError('unchanged native-context and command-tape checks required')
        requested=(known_action_blocks[:,:,0].cpu()*torch.tensor([.3,1.,.5])).numpy()
        last=tuple(native['past_applied_commands'][-1].tolist())
        applied=np.asarray([apply_safety_limits_single(row.tolist(),last,self.limits)[0] for row in requested],np.float32)
        motion=torch.zeros(6,8,3)
        outcome=torch.zeros(6,8,5);outcome[:,:,3]=1.;outcome[:,:,4]=-1000.
        self.receipts.append(dict(observed_ns=int(native['context_times_ns'][-1]),
            requested_commands=requested.tolist(),applied_commands=applied.tolist(),motion_xy_yaw=motion.tolist(),
            interface_placeholder=True,neural_predictions_computed=False,
            actual_motion_source='fitted command history in unchanged controller' if self.controller=='C1' else 'none/reactive'))
        return dict(rollout_outcomes=outcome,target_offsets_ns=torch.arange(1,9).mul(100_000_000).expand(6,8),
            prediction_valid=torch.ones(6,8,dtype=torch.bool),contact_prediction_available=False)
