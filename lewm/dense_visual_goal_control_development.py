"""Frozen native V-JEPA visual-goal control at the actual 500-ms interface."""
from collections import deque
import json
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml

from lewm.geometry_progress_pilot_development import ACTIONS, candidate_commands
from lewm.simulated_body_observation_development import validate_policy_packet
from lewm_genesis.lewm_contract import SafetyLimits, apply_safety_limits_single
from scripts import evaluate_go2_frozen_vjepa_native_branches_development as base
from scripts import train_go2_frozen_vjepa_native_adaptation_development as fit


class DenseVisualGoalControl:
    def __init__(self, arm, goal_image, *, seed=2026091706):
        if arm not in fit.ARMS:
            raise ValueError('action or matched no_future_action arm required')
        self.arm = arm
        terminal = json.loads((fit.OUTPUT/'result.json').read_text())
        assert terminal['status'] == 'COMPLETE'
        path = fit.OUTPUT/f'{arm}_latest.pt'
        assert base.digest(path) == terminal['checkpoint_sha256'][arm]
        state = torch.load(path, map_location='cpu', weights_only=False)
        assert state['epoch'] == 23
        self.model = base.ProprioActionPredictor(use_proprio=False)
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.model = self.model.cuda().eval().requires_grad_(False)
        del state
        self.encoder = base.encoders.VJepa21Arm()
        self.encoder.build(torch.device('cuda:0'), torch.float32)
        with Image.open(goal_image) as image:
            rgb = np.array(image.convert('RGB'))
        # The in-memory live camera path must reproduce the established path.
        assert torch.equal(self.preprocess(rgb), self.encoder.preprocess(str(goal_image)))
        self.goal = self.encode(rgb)
        stats = json.loads((base.CACHE/'proprio_v1/proprio_norm_stats.json').read_text())
        self.mean, self.std = (np.asarray(stats[k], np.float32) for k in ('control_mean', 'control_std'))
        self.limits = SafetyLimits.from_manifest(yaml.safe_load(base.Path('config/go2_platform_manifest.yaml').read_text()))
        self.history = deque(maxlen=3)
        self.rng = np.random.default_rng(seed)
        self.pending = None

    @staticmethod
    def preprocess(rgb):
        assert rgb.shape == (480, 640, 3) and rgb.dtype == np.uint8
        image = Image.fromarray(rgb).resize((512, 384), Image.Resampling.BICUBIC)
        return base.encoders._normalise(base.encoders._to_chw(image))

    @torch.inference_mode()
    def encode(self, rgb):
        value = F.layer_norm(self.encoder.tokens(self.preprocess(rgb)[None].cuda()).float(), (1024,))[0]
        assert value.shape == (768, 1024) and torch.isfinite(value).all()
        return value

    @torch.inference_mode()
    def observe(self, packet):
        start = time.monotonic()
        validate_policy_packet(packet)
        now = int(packet['image']['measured_ns'])
        assert now == packet['sensor_state']['decision_ns']
        if self.history:
            assert now-self.history[-1][0] == 500_000_000
        feature = self.encode(packet['image']['rgb'])
        self.history.append((now, feature))
        record = dict(observed_ns=now, current_goal_mse=float((feature-self.goal).square().mean()),
                      observation_encoding_wall_s=time.monotonic()-start)
        if self.pending is not None:
            assert now == self.pending[0]+500_000_000
            record['previous_selected_forecast_mse'] = float((self.pending[1]-feature).square().mean())
            record['previous_persistence_mse'] = float((self.pending[2]-feature).square().mean())
            executed = np.asarray(packet['sensor_state']['control']['applied_command']['values'])[-5:]
            np.testing.assert_allclose(executed, self.pending[3], rtol=0, atol=1e-6)
            self.pending = None
        return record

    @torch.inference_mode()
    def choose(self, packet):
        start = time.monotonic()
        assert len(self.history) == 3
        now = int(packet['image']['measured_ns'])
        assert self.history[-1][0] == now
        control = packet['sensor_state']['control']['applied_command']
        assert np.asarray(control['valid']).all()
        times = np.asarray(control['measured_ns'])
        assert times[[4, 9, 14]].tolist() == [t for t, _ in self.history]
        assert (np.asarray(control['available_ns']) <= now).all()
        commands = np.asarray(control['values'], np.float32)
        assert np.all(commands[:, 1] == 0)
        # Keep canonical requested scalars through the session's strict domain
        # check: float32(.20) is slightly greater than the allowed .20.
        # Applied model inputs below still use the native float32 command tape.
        requested = np.asarray([[candidate_commands(a)[0]]*5 for a in ACTIONS], np.float64)
        applied = np.asarray([apply_safety_limits_single(v.tolist(), tuple(commands[-1]), self.limits)[0]
                              for v in requested], np.float32)
        x = torch.stack([f for _, f in self.history])[None].expand(len(ACTIONS), -1, -1, -1)
        c = torch.from_numpy((commands[:, [0, 2]].reshape(3, 5, 2)-self.mean)/self.std).cuda()[None]
        a = torch.from_numpy(applied[:, :, [0, 2]].reshape(len(ACTIONS), 10)).cuda()
        mask = torch.ones(len(ACTIONS), 768, dtype=torch.bool, device='cuda')
        if self.arm == 'no_future_action':
            p = self.model(x[:1], torch.zeros_like(a[:1]), mask[:1], control=c)
            prediction = F.layer_norm(p.float(), (1024,)).expand(len(ACTIONS), -1, -1)
        else:
            prediction = F.layer_norm(self.model(x, a, mask, control=c.expand(len(ACTIONS), -1, -1, -1)).float(), (1024,))
        assert torch.isfinite(prediction).all()
        costs = (prediction-self.goal).square().mean((-1, -2)).cpu().numpy()
        ties = np.flatnonzero(costs == costs.min())
        index = int(self.rng.choice(ties))
        self.pending = (now, prediction[index].clone(), self.history[-1][1], applied[index])
        return dict(observed_ns=now, action=ACTIONS[index], action_index=index,
                    costs=costs.tolist(), tied_indices=ties.tolist(),
                    requested_commands=requested[index].tolist(), expected_applied_commands=applied[index].tolist(),
                    forecast_horizon_ms=500, commit_ticks=5, planning_wall_s=time.monotonic()-start)
