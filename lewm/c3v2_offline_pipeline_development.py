"""Offline replica of C3's deployed prediction path (pre-declared C3-v2 readout fix, 29 Sep 2026).

Builds the exact native context the controller builds (three causal frames at -1000/-500/0 ms,
bicubic 512x384, V-JEPA normalisation; 15-step applied-command history), the executed applied
command tape, physics-true motion targets, and the pooled current / predicted-future features
that C3's readout consumes. Exactness against run-time C3 is checked before any use.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from lewm.dense_visual_motion_readout_development import pool_tokens
from lewm.physical_execution_development import rotation_xyzw
from scripts.dev_frozen_dense_representation_encoders_v1 import _normalise, _to_chw


class Recording:
    """A recording with 10-Hz RGB PNGs and the policy-history archive."""

    def __init__(self, histories, observations, frame_path, trace=None):
        with np.load(histories, allow_pickle=False) as a:
            self.values = a['applied_command_values'].astype(np.float32)
            self.valid = a['applied_command_valid'].copy()
            self.measured = a['applied_command_measured_ns'].copy()
            self.available = a['applied_command_available_ns'].copy()
        self.stamps = [f['image_ns'] for f in json.loads(Path(observations).read_text())['frames']]
        self.frame_path = frame_path
        self.trace = trace
        self._pixels = {}

    @classmethod
    def training_case(cls, directory):
        directory = Path(directory)
        return cls(directory/'policy_histories.npz', directory/'policy_observations.json',
                   lambda i: directory/f'rgb_{i:04d}.png', directory/'physics_trace.npz')

    def pixels(self, index):
        if index not in self._pixels:
            rgb = np.asarray(Image.open(self.frame_path(index)).convert('RGB'))
            if rgb.shape != (480, 640, 3) or rgb.dtype != np.uint8:
                raise ValueError('native 640x480 RGB frame required')
            self._pixels[index] = _normalise(_to_chw(Image.fromarray(rgb).resize((512, 384), Image.Resampling.BICUBIC)))
        return self._pixels[index]

    def native(self, frame):
        """Identical fields and checks to lewm.dense_native_observation_development.dense_native_context."""
        indices = (frame-10, frame-5, frame)
        if min(indices) < 0:
            raise ValueError('one second of causal history required')
        expected = [self.stamps[i] for i in indices]
        if np.diff(expected).tolist() != [500_000_000]*2:
            raise ValueError('causal frame spacing')
        values, measured, available = self.values[frame], self.measured[frame], self.available[frame]
        if (not self.valid[frame].all() or not np.array_equal(measured[[4, 9, 14]], expected)
                or not np.all(np.diff(measured) == 100_000_000) or np.any(measured > expected[-1])
                or np.any(available > expected[-1]) or np.any(values[:, 1] != 0)):
            raise ValueError('complete causal native command history required')
        return dict(pixels=torch.stack([self.pixels(i) for i in indices]),
                    context_times_ns=torch.tensor(expected, dtype=torch.int64),
                    past_applied_commands=torch.from_numpy(values.copy()),
                    past_measured_ns=torch.from_numpy(measured.copy()),
                    past_available_ns=torch.from_numpy(available.copy()))

    def executed_tape(self, frame):
        """Executed applied commands for the eight 100-ms steps after the decision (as C4-v1 prep)."""
        future = []
        for h in range(1, 9):
            if not self.valid[frame+h, -h:].all():
                raise ValueError('complete executed tape required')
            commands = self.values[frame+h, -h:]
            if future:
                np.testing.assert_array_equal(commands[:-1], np.asarray(future))
            future = commands.tolist()
        return np.asarray(future, np.float32)

    def targets(self, frame, frames_meta):
        """Physics-true motion in the decision body frame at 100-800 ms (maze-view summary convention)."""
        with np.load(self.trace, allow_pickle=False) as data:
            poses = data['base_pose_world']
        start = frames_meta[frame]['physical_sample_index']
        R = rotation_xyzw(poses[start, 3:])
        out = []
        for h in range(1, 9):
            end = frames_meta[frame+h]['physical_sample_index']
            relative = R.T @ rotation_xyzw(poses[end, 3:])
            delta = (poses[end, :3]-poses[start, :3]) @ R
            out.append([float(delta[0]), float(delta[1]), float(np.arctan2(relative[1, 0], relative[0, 0]))])
        return np.asarray(out, np.float32)


@torch.inference_mode()
def pooled_features(model, native, applied, horizons=range(1, 9)):
    """C3's run-time computation for one applied tape; returns pooled current and predicted features."""
    device = next(model.predictor.parameters()).device
    context = F.layer_norm(model.encoder.tokens(native['pixels'].to(device)).float(), (1024,))[None]
    control = native['past_applied_commands'][:, [0, 2]].reshape(3, 5, 2).to(device)
    control = ((control-model.control_mean)/model.control_std)[None]
    actions = torch.from_numpy(np.asarray(applied, np.float32)[None][:, :, [0, 2]]).to(device)
    current = pool_tokens(context[:, -1])
    mask = torch.ones(1, 768, dtype=torch.bool, device=device)
    predicted = {}
    for h in horizons:
        out = model.predictor(context, actions, torch.full((1,), h, dtype=torch.long, device=device), mask, control=control)
        predicted[h] = pool_tokens(F.layer_norm(out.float(), (1024,)))
    return current, predicted


@torch.inference_mode()
def readout_motion(readout, current, predicted):
    return readout(current, predicted).float().cpu().numpy()[0]
