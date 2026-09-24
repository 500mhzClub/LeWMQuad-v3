"""Shared state-action contact evaluator for repaired two-ply viability evidence."""
from __future__ import annotations

import hashlib
import json

import torch
from torch import nn

SEED = 2026082017


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class DepthEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(8, 16, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv2d(16, 24, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 64), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class LidarEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(32, 32, 7, stride=2, padding=3), nn.GELU(),
            nn.Conv1d(32, 48, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(48, 64, 3, stride=2, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 64), nn.GELU(),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.net(value)


class SetStructuredOneTickContactEvaluator(nn.Module):
    """Encode one state once and apply one shared evaluator to every action."""

    def __init__(self, embodied_width: int = 81, action_width: int = 9) -> None:
        super().__init__()
        self.depth = DepthEncoder(); self.lidar = LidarEncoder()
        self.embodied = nn.GRU(embodied_width, 96, batch_first=True)
        self.state_fusion = nn.Sequential(nn.Linear(224, 128), nn.GELU())
        self.action = nn.Sequential(nn.Linear(action_width, 64), nn.GELU())
        self.pair = nn.Sequential(nn.Linear(192, 128), nn.GELU(), nn.Linear(128, 1))

    def encode_state(self, depth: torch.Tensor, lidar: torch.Tensor, embodied: torch.Tensor) -> torch.Tensor:
        _sequence, hidden = self.embodied(embodied)
        return self.state_fusion(torch.cat((self.depth(depth), self.lidar(lidar), hidden[0]), dim=-1))

    def score_actions(self, state_embedding: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        encoded = self.action(action)
        state = state_embedding[:, None, :].expand(-1, action.shape[1], -1)
        return self.pair(torch.cat((state, encoded), dim=-1)).squeeze(-1)

    def forward(self, depth: torch.Tensor, lidar: torch.Tensor, embodied: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.score_actions(self.encode_state(depth, lidar, embodied), action)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
