"""Native dense forecasts at the existing navigation controller boundary."""
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from lewm.dense_native_observation_development import dense_native_context
from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm_genesis.lewm_contract import apply_safety_limits_single


def load_dense_navigation_model(arm='action', *, readout_arm='original'):
    import json
    from pathlib import Path
    import yaml
    from lewm_genesis.lewm_contract import SafetyLimits
    from scripts import train_go2_horizon_dense_predictor_development as fit
    from scripts import train_go2_dense_visual_motion_readout_development as motion_fit
    if arm not in fit.ARMS:
        raise ValueError('completed action or no-future-action model required')
    if readout_arm == 'original':
        readout = motion_fit.load()
        readout_path = motion_fit.OUTPUT/'readout.pt'
    elif readout_arm in ('old_data', 'mixed_data'):
        from scripts import train_go2_full_heading_readout_development as heading_fit
        readout = heading_fit.load(readout_arm)
        readout_path = heading_fit.OUTPUT/f'{readout_arm}_final.pt'
    elif readout_arm in ('maze_view_old_data', 'maze_view_maze_data'):
        from scripts import run_go2_maze_view_readout_recovery_development as maze_fit
        treatment = readout_arm.removeprefix('maze_view_')
        result = json.loads((maze_fit.OUTPUT/'result.json').read_text())
        readout_path = maze_fit.OUTPUT/f'{treatment}_final.pt'
        if result['status'] != 'COMPLETE' or fit.digest(readout_path) != result['checkpoint_sha256'][treatment]:
            raise ValueError('completed fixed-final maze-view readout required')
        state = torch.load(readout_path, map_location='cpu', weights_only=False)
        if state['updates'] != maze_fit.original.STEPS or state['plan_sha256'] != fit.digest(maze_fit.OUTPUT/'plan.json'):
            raise ValueError('maze-view readout training identity mismatch')
        readout = maze_fit.original.prior.previous.load('mixed_data')
        readout.load_state_dict(state['model_state_dict'])
        readout.eval().requires_grad_(False)
    else:
        raise ValueError('explicit original or completed continuation readout required')
    reference = fit.parent.reference
    encoder = reference.encoders.VJepa21Arm()
    encoder.build(torch.device('cuda:0'), torch.float32)
    stats = json.loads((reference.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
    limits = SafetyLimits.from_manifest(yaml.safe_load(Path('config/go2_platform_manifest.yaml').read_text()))
    model = DenseHorizonNavigationModel(encoder, fit.load(arm).cuda(), readout.cuda(),
        limits, stats['control_mean'], stats['control_std'], action_blind=arm=='no_future_action').cuda().eval()
    model.readout_identity = dict(arm=readout_arm, path=str(readout_path), sha256=fit.digest(readout_path),
        training_horizons_ms=list(range(100, 801, 100)) if readout_arm.startswith('maze_view_') else [500])
    return model


class DenseHorizonNavigationModel(nn.Module):
    def __init__(self, encoder, predictor, readout, limits, mean, std, *, action_blind=False):
        super().__init__()
        self.encoder = encoder
        self.predictor = predictor.eval().requires_grad_(False)
        self.readout = readout.eval().requires_grad_(False)
        self.limits = limits
        self.register_buffer('control_mean', torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer('control_std', torch.as_tensor(std, dtype=torch.float32))
        self.action_blind = action_blind
        self.pending_context = None
        self.receipts = []
        self.eval()

    def set_native_context(self, packets, *, observed_ns):
        if self.pending_context is not None:
            raise ValueError('previous native context was not consumed')
        self.pending_context = dense_native_context(packets, observed_ns=observed_ns)

    @torch.inference_mode()
    def forward(self, *, observation_history, known_action_blocks, known_action_valid):
        if self.pending_context is None:
            raise ValueError('current native context required for each prediction')
        native, self.pending_context = self.pending_context, None
        if (known_action_blocks.shape != (6, 8, 1, 3)
                or known_action_valid.shape != (6, 8, 1) or not known_action_valid.all()
                or not torch.isfinite(known_action_blocks).all()):
            raise ValueError('six complete eight-step requested command tapes required')
        # The old resized RGB tensor is retained only for the controller's input
        # treatment hook. The encoder consumes the full native packets above.
        requested = (known_action_blocks[:, :, 0].detach().cpu()
            * torch.tensor([.3, 1., .5])).numpy()
        last = tuple(native['past_applied_commands'][-1].tolist())
        applied = np.asarray([apply_safety_limits_single(row.tolist(), last, self.limits)[0]
            for row in requested], np.float32)
        if np.any(applied[:, :, 1] != 0):
            raise ValueError('dense predictor has no lateral-command training')
        device = next(self.predictor.parameters()).device
        started = time.perf_counter_ns()
        context = F.layer_norm(self.encoder.tokens(native['pixels'].to(device)).float(), (1024,))[None]
        control = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
        control = ((control-self.control_mean)/self.control_std)[None]
        actions = torch.from_numpy(applied[:, :, [0, 2]]).to(device)
        current = pool_tokens(context[:, -1])
        mask = torch.ones(6, 768, dtype=torch.bool, device=device)
        motions, counts = [], []
        for horizon in range(1, 9):
            unique, inverse = torch.unique(actions[:, :horizon].reshape(6, -1),
                dim=0, return_inverse=True)
            indices = torch.stack([(inverse == j).nonzero()[0, 0] for j in range(len(unique))])
            n = 1 if self.action_blind else len(unique)
            expansion = torch.zeros(6, dtype=torch.long, device=device) if self.action_blind else inverse
            selected = actions[:1] if self.action_blind else actions[indices]
            predicted = self.predictor(context.expand(n, -1, -1, -1), selected,
                torch.full((n,), horizon, dtype=torch.long, device=device), mask[:n],
                control=control.expand(n, -1, -1, -1))
            predicted = F.layer_norm(predicted.float(), (1024,))
            motions.append(self.readout(current.expand(n, -1, -1), pool_tokens(predicted))[expansion])
            counts.append(n)
        motion = torch.stack(motions, dim=1).cpu()
        if motion.shape != (6, 8, 3) or not torch.isfinite(motion).all():
            raise ValueError('complete finite dense motion forecasts required')
        # Contact cost is explicitly disabled in the shared controller. This
        # sentinel is an interface accommodation, never a no-contact forecast.
        output = torch.cat((motion[:, :, :2], motion[:, :, 2:3].sin(),
            motion[:, :, 2:3].cos(), torch.full((6, 8, 1), -1000.)), dim=-1)
        self.receipts.append(dict(observed_ns=int(native['context_times_ns'][-1]),
            context_times_ns=native['context_times_ns'].tolist(),
            native_rgb_shape=[480, 640, 3], encoder_input_shape=[3, 3, 384, 512],
            requested_commands=requested.tolist(), applied_commands=applied.tolist(),
            last_applied_command=list(last), forecasts_per_horizon=counts,
            motion_xy_yaw=motion.tolist(), wall_ns=time.perf_counter_ns()-started,
            action_blind=self.action_blind, contact_prediction_available=False,
            physical_readout_training_horizons_ms=self.readout_identity['training_horizons_ms'], real_time_qualified=False))
        return dict(rollout_outcomes=output,
            target_offsets_ns=torch.arange(1, 9, dtype=torch.int64).mul(100_000_000).expand(6, 8),
            prediction_valid=torch.ones(6, 8, dtype=torch.bool), contact_prediction_available=False)


class DenseNativeContextRuntimeMixin:
    """Retain one second of native observations; shared navigation stays intact."""
    def __init__(self, *args, **kwargs):
        self.native_context_packets = {}
        super().__init__(*args, **kwargs)
        if self.variant != 'full':
            raise ValueError('native dense integration currently supports full inputs only')

    def submit(self, packet):
        self.native_context_packets[packet.measured_ns] = packet.policy
        for stamp in list(self.native_context_packets):
            if stamp < packet.measured_ns-1_500_000_000:
                del self.native_context_packets[stamp]
        return super().submit(packet)

    def _plan(self, item):
        packet, _ = item
        if packet.frame < 10:
            self.planning.append(dict(frame=packet.frame, reason='NATIVE_DENSE_CONTEXT_WARMUP'))
            return
        # Untimed synchronous treatment: mapping from this acquired frame must
        # finish before planning, avoiding host scheduling-dependent map choice.
        self.queues['mapping'].join()
        if self.faults:
            return
        return super()._plan(item)

    def _select_action(self, packet, *args, **kwargs):
        stamps = [packet.measured_ns-delta for delta in (1_000_000_000, 500_000_000, 0)]
        self.model.set_native_context([self.native_context_packets[t] for t in stamps],
            observed_ns=packet.measured_ns)
        selected, correction = super()._select_action(packet, *args, **kwargs)
        selected['dense_model'] = dict(native_context_times_ns=stamps,
            contact_prediction_available=False,
            physical_readout_training_horizons_ms=self.model.readout_identity['training_horizons_ms'])
        if correction is not None:
            correction['learned_corrected_field_means'] = 'dense predicted features decoded by frozen motion head; no nominal motion composition'
        return selected, correction
