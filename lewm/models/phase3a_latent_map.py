"""Latent-map perception heads for Phase 3A online memory."""
from __future__ import annotations

import torch
import torch.nn as nn


class Phase3ALatentMapHead(nn.Module):
    """Predict local occupancy and marker evidence from JEPA spatial tokens."""

    def __init__(
        self,
        *,
        view_size: int,
        latent_dim: int,
        hidden_dim: int = 64,
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.view_size = int(view_size)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_channels = int(output_channels)
        self.net = nn.Sequential(
            nn.Conv2d(2 * self.latent_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.output_channels, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return ``(B, C, H, W)`` logits for a ``(B, H*W, D)`` token grid."""

        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape (B, N, D), got {tokens.shape}")
        batch, cells, dim = tokens.shape
        expected_cells = self.view_size * self.view_size
        if cells != expected_cells:
            raise ValueError(f"expected {expected_cells} cells, got {cells}")
        if dim != self.latent_dim:
            raise ValueError(f"expected latent dim {self.latent_dim}, got {dim}")
        grid = tokens.transpose(1, 2).reshape(
            batch,
            dim,
            self.view_size,
            self.view_size,
        )
        pooled = grid.mean(dim=(2, 3), keepdim=True).expand_as(grid)
        return self.net(torch.cat([grid, pooled], dim=1))


class Phase3AEgocentricMemoryUpdate(nn.Module):
    """Update a finite egocentric memory from local evidence and action context."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 96,
        memory_channels: int = 3,
        evidence_channels: int = 3,
        action_dim: int = 4,
        use_geometric_prior: bool = True,
        learned_transition_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.evidence_channels = int(evidence_channels)
        self.action_dim = int(action_dim)
        self.use_geometric_prior = bool(use_geometric_prior)
        self.learned_transition_hidden_dim = (
            int(learned_transition_hidden_dim)
            if learned_transition_hidden_dim is not None
            else max(self.hidden_dim * 8, 256)
        )
        flat_memory_dim = self.memory_channels * self.memory_size * self.memory_size
        if self.use_geometric_prior:
            self.learned_transition = None
        elif self.learned_transition_hidden_dim == 0:
            self.learned_transition = nn.Linear(
                flat_memory_dim + self.action_dim + 1,
                flat_memory_dim,
            )
        else:
            self.learned_transition = nn.Sequential(
                nn.Linear(
                    flat_memory_dim + self.action_dim + 1,
                    self.learned_transition_hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.learned_transition_hidden_dim, flat_memory_dim),
            )
        input_channels = (
            self.memory_channels
            + self.evidence_channels
            + self.action_dim
            + 1
            + 2
        )
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.memory_channels, kernel_size=1),
        )

    def forward(
        self,
        previous_memory: torch.Tensor,
        local_evidence: torch.Tensor,
        action: torch.Tensor,
        collision: torch.Tensor,
    ) -> torch.Tensor:
        """Return memory logits for ``(B, C, S, S)`` memory tensors."""

        if previous_memory.ndim != 4:
            raise ValueError(
                "previous_memory must have shape (B, C, S, S), "
                f"got {previous_memory.shape}"
            )
        if local_evidence.shape != previous_memory.shape:
            raise ValueError(
                "local_evidence must match previous_memory shape, "
                f"got {local_evidence.shape} vs {previous_memory.shape}"
            )
        batch, channels, height, width = previous_memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if action.ndim == 2:
            action_onehot = action.to(dtype=previous_memory.dtype)
            action_index = action_onehot.argmax(dim=1)
        elif action.ndim == 1:
            action_index = action.to(dtype=torch.long)
            action_onehot = torch.nn.functional.one_hot(
                action_index,
                num_classes=self.action_dim,
            ).to(dtype=previous_memory.dtype)
        else:
            raise ValueError(f"action must have shape (B,) or (B, A), got {action.shape}")
        if int(action_onehot.shape[0]) != batch or int(action_onehot.shape[1]) != self.action_dim:
            raise ValueError(
                f"expected action shape ({batch}, {self.action_dim}), got "
                f"{action_onehot.shape}"
            )
        collision = collision.to(
            device=previous_memory.device,
            dtype=previous_memory.dtype,
        ).reshape(batch, 1, 1, 1)
        if self.use_geometric_prior:
            rolled = self._roll_previous_memory(
                previous_memory,
                action_index=action_index,
                collision=collision.reshape(batch) >= 0.5,
            )
        else:
            rolled = self._learned_roll_previous_memory(
                previous_memory,
                action_onehot=action_onehot.to(previous_memory.device),
                collision=collision.reshape(batch, 1),
            )
        action_planes = action_onehot.to(previous_memory.device).reshape(
            batch,
            self.action_dim,
            1,
            1,
        ).expand(-1, -1, height, width)
        collision_plane = collision.expand(-1, 1, height, width)
        coords = self._coordinate_planes(
            batch=batch,
            device=previous_memory.device,
            dtype=previous_memory.dtype,
        )
        return self.net(
            torch.cat(
                [rolled, local_evidence, action_planes, collision_plane, coords],
                dim=1,
            )
        )

    def _learned_roll_previous_memory(
        self,
        previous_memory: torch.Tensor,
        *,
        action_onehot: torch.Tensor,
        collision: torch.Tensor,
    ) -> torch.Tensor:
        if self.learned_transition is None:
            return previous_memory
        batch = int(previous_memory.shape[0])
        flat = previous_memory.flatten(start_dim=1)
        transition_input = torch.cat(
            [
                flat,
                action_onehot.to(dtype=previous_memory.dtype),
                collision.to(dtype=previous_memory.dtype),
            ],
            dim=1,
        )
        rolled = torch.sigmoid(self.learned_transition(transition_input))
        return rolled.reshape(
            batch,
            self.memory_channels,
            self.memory_size,
            self.memory_size,
        )

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        radius = self.memory_size // 2
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        if radius == 0:
            return torch.cat([row, col], dim=1)
        return torch.cat([row, col], dim=1)

    def _roll_previous_memory(
        self,
        previous_memory: torch.Tensor,
        *,
        action_index: torch.Tensor,
        collision: torch.Tensor,
    ) -> torch.Tensor:
        rolled = torch.zeros_like(previous_memory)
        for action_id in range(self.action_dim):
            action_mask = action_index == action_id
            if not bool(action_mask.any()):
                continue
            current = previous_memory[action_mask]
            if action_id == 0:
                current_collision = collision[action_mask]
                if bool((~current_collision).any()):
                    shifted = torch.zeros_like(current[~current_collision])
                    shifted[:, :, 1:, :] = current[~current_collision][:, :, :-1, :]
                    rolled[action_mask.nonzero(as_tuple=False).flatten()[~current_collision]] = shifted
                if bool(current_collision.any()):
                    rolled[action_mask.nonzero(as_tuple=False).flatten()[current_collision]] = current[
                        current_collision
                    ]
            elif action_id == 1:
                rolled[action_mask] = torch.rot90(current, k=1, dims=(-2, -1))
            elif action_id == 2:
                rolled[action_mask] = torch.rot90(current, k=-1, dims=(-2, -1))
            else:
                rolled[action_mask] = current
        return rolled


class Phase3AEgocentricMemoryPolicy(nn.Module):
    """Select a primitive action from a finite egocentric memory tensor."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 256,
        memory_channels: int = 3,
        action_dim: int = 4,
        architecture: str = "mlp",
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.action_dim = int(action_dim)
        self.architecture = str(architecture)
        if self.architecture == "mlp":
            flat_dim = self.memory_channels * self.memory_size * self.memory_size
            self.net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(flat_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
        elif self.architecture == "conv":
            input_channels = self.memory_channels + 2
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Flatten(),
                nn.Linear(
                    self.hidden_dim * self.memory_size * self.memory_size,
                    self.hidden_dim,
                ),
                nn.GELU(),
                nn.Linear(self.hidden_dim, self.action_dim),
            )
        else:
            raise ValueError(f"unknown policy architecture: {architecture!r}")

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Return action logits for ``(B, C, S, S)`` memory tensors."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        _batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if self.architecture == "conv":
            batch = int(memory.shape[0])
            coords = self._coordinate_planes(
                batch=batch,
                device=memory.device,
                dtype=memory.dtype,
            )
            return self.net(torch.cat([memory, coords], dim=1))
        return self.net(memory)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AEgocentricValueFieldHead(nn.Module):
    """Predict marker/frontier target rewards over an egocentric memory tensor."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 64,
        memory_channels: int = 3,
        output_channels: int = 1,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.output_channels = int(output_channels)
        input_channels = self.memory_channels + 2
        self.net = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.output_channels, kernel_size=1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Return target-field logits for ``(B, C, S, S)`` memory tensors."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        return self.net(torch.cat([memory, coords], dim=1))

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AValueFieldExtractorHead(nn.Module):
    """Predict whether learned value-field extraction should be sparse."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 32,
        memory_channels: int = 3,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        input_channels = self.memory_channels + 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Return sparse-extraction logits for ``(B, C, S, S)`` memories."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        features = self.encoder(torch.cat([memory, coords], dim=1))
        pooled = torch.cat(
            [
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.classifier(pooled).squeeze(-1)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AValueFieldActionHead(nn.Module):
    """Predict the first primitive from memory and a learned target field."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 64,
        memory_channels: int = 3,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.action_dim = int(action_dim)
        input_channels = self.memory_channels + 4
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        target_field: torch.Tensor,
        sparse_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return primitive logits for recurrent memories and target fields."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        if target_field.ndim != 4:
            raise ValueError(
                f"target_field must have shape (B, 1, S, S), got {target_field.shape}"
            )
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if target_field.shape != (batch, 1, height, width):
            raise ValueError(
                "target_field must have shape "
                f"{(batch, 1, height, width)}, got {tuple(target_field.shape)}"
            )
        if sparse_probability.ndim == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        elif sparse_probability.ndim == 2 and sparse_probability.shape[1] == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        else:
            raise ValueError(
                "sparse_probability must have shape (B,) or (B, 1), got "
                f"{tuple(sparse_probability.shape)}"
            )
        sparse_plane = sparse.to(dtype=memory.dtype, device=memory.device).expand(
            batch,
            1,
            height,
            width,
        )
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        features = self.encoder(
            torch.cat([memory, target_field, sparse_plane, coords], dim=1)
        )
        pooled = torch.cat(
            [
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.classifier(pooled)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AActionCorrectionHead(nn.Module):
    """Predict an oracle-style correction for a proposed primitive action."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 64,
        memory_channels: int = 3,
        action_dim: int = 4,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.action_dim = int(action_dim)
        input_channels = self.memory_channels + 4 + self.action_dim + 1
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.action_dim),
        )

    def forward(
        self,
        memory: torch.Tensor,
        target_field: torch.Tensor,
        sparse_probability: torch.Tensor,
        planned_action: torch.Tensor,
        marker_seen_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return primitive logits for correcting the proposed action."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        if target_field.ndim != 4:
            raise ValueError(
                f"target_field must have shape (B, 1, S, S), got {target_field.shape}"
            )
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if target_field.shape != (batch, 1, height, width):
            raise ValueError(
                "target_field must have shape "
                f"{(batch, 1, height, width)}, got {tuple(target_field.shape)}"
            )
        if sparse_probability.ndim == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        elif sparse_probability.ndim == 2 and sparse_probability.shape[1] == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        else:
            raise ValueError(
                "sparse_probability must have shape (B,) or (B, 1), got "
                f"{tuple(sparse_probability.shape)}"
            )
        if marker_seen_probability.ndim == 1:
            marker_seen = marker_seen_probability.view(batch, 1, 1, 1)
        elif (
            marker_seen_probability.ndim == 2
            and marker_seen_probability.shape[1] == 1
        ):
            marker_seen = marker_seen_probability.view(batch, 1, 1, 1)
        else:
            raise ValueError(
                "marker_seen_probability must have shape (B,) or (B, 1), got "
                f"{tuple(marker_seen_probability.shape)}"
            )
        if planned_action.ndim == 2:
            planned_onehot = planned_action.to(dtype=memory.dtype)
        elif planned_action.ndim == 1:
            planned_onehot = torch.nn.functional.one_hot(
                planned_action.to(dtype=torch.long),
                num_classes=self.action_dim,
            ).to(dtype=memory.dtype)
        else:
            raise ValueError(
                "planned_action must have shape (B,) or (B, A), got "
                f"{tuple(planned_action.shape)}"
            )
        if planned_onehot.shape != (batch, self.action_dim):
            raise ValueError(
                f"expected planned action shape {(batch, self.action_dim)}, "
                f"got {tuple(planned_onehot.shape)}"
            )
        sparse_plane = sparse.to(dtype=memory.dtype, device=memory.device).expand(
            batch,
            1,
            height,
            width,
        )
        marker_seen_plane = marker_seen.to(
            dtype=memory.dtype,
            device=memory.device,
        ).expand(batch, 1, height, width)
        planned_planes = planned_onehot.to(memory.device).view(
            batch,
            self.action_dim,
            1,
            1,
        ).expand(-1, -1, height, width)
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        features = self.encoder(
            torch.cat(
                [
                    memory,
                    target_field,
                    sparse_plane,
                    coords,
                    planned_planes,
                    marker_seen_plane,
                ],
                dim=1,
            )
        )
        pooled = torch.cat(
            [
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.classifier(pooled)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AValueMapPlannerHead(nn.Module):
    """Predict a dense egocentric value map from memory and target evidence."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 96,
        memory_channels: int = 3,
        architecture: str = "conv",
        refinement_steps: int = 8,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        if architecture not in {"conv", "dilated", "recurrent"}:
            raise ValueError(f"unknown planner architecture: {architecture!r}")
        if refinement_steps < 1:
            raise ValueError("refinement_steps must be positive")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        self.architecture = str(architecture)
        self.refinement_steps = int(refinement_steps)
        input_channels = self.memory_channels + 4
        self.input_channels = int(input_channels)
        if self.architecture == "conv":
            self.net = nn.Sequential(
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )
        elif self.architecture == "dilated":
            layers: list[nn.Module] = [
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            ]
            for dilation in (1, 2, 4, 8, 4, 2, 1):
                layers.extend(
                    [
                        nn.Conv2d(
                            self.hidden_dim,
                            self.hidden_dim,
                            kernel_size=3,
                            padding=dilation,
                            dilation=dilation,
                        ),
                        nn.GELU(),
                    ]
                )
            layers.append(nn.Conv2d(self.hidden_dim, 1, kernel_size=1))
            self.net = nn.Sequential(*layers)
        else:
            self.input_projection = nn.Sequential(
                nn.Conv2d(input_channels, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.refinement = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim + input_channels,
                    self.hidden_dim,
                    kernel_size=3,
                    padding=1,
                ),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
            )
            self.output_projection = nn.Sequential(
                nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
            )

    def forward(
        self,
        memory: torch.Tensor,
        target_field: torch.Tensor,
        sparse_probability: torch.Tensor,
    ) -> torch.Tensor:
        """Return dense value logits for recurrent memories and target fields."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        if target_field.ndim != 4:
            raise ValueError(
                f"target_field must have shape (B, 1, S, S), got {target_field.shape}"
            )
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        if target_field.shape != (batch, 1, height, width):
            raise ValueError(
                "target_field must have shape "
                f"{(batch, 1, height, width)}, got {tuple(target_field.shape)}"
            )
        if sparse_probability.ndim == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        elif sparse_probability.ndim == 2 and sparse_probability.shape[1] == 1:
            sparse = sparse_probability.view(batch, 1, 1, 1)
        else:
            raise ValueError(
                "sparse_probability must have shape (B,) or (B, 1), got "
                f"{tuple(sparse_probability.shape)}"
            )
        sparse_plane = sparse.to(dtype=memory.dtype, device=memory.device).expand(
            batch,
            1,
            height,
            width,
        )
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        inputs = torch.cat([memory, target_field, sparse_plane, coords], dim=1)
        if self.architecture != "recurrent":
            return self.net(inputs)
        hidden = self.input_projection(inputs)
        for _ in range(self.refinement_steps):
            hidden = hidden + self.refinement(torch.cat([hidden, inputs], dim=1))
        return self.output_projection(hidden)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)


class Phase3AValueMapRouterHead(nn.Module):
    """Choose between value-map planner variants from egocentric memory."""

    def __init__(
        self,
        *,
        memory_size: int,
        hidden_dim: int = 32,
        memory_channels: int = 3,
    ) -> None:
        super().__init__()
        if memory_size < 3 or memory_size % 2 == 0:
            raise ValueError("memory_size must be an odd integer >= 3")
        self.memory_size = int(memory_size)
        self.hidden_dim = int(hidden_dim)
        self.memory_channels = int(memory_channels)
        input_channels = self.memory_channels + 2
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, self.hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """Return fallback-route logits for ``(B, C, S, S)`` memories."""

        if memory.ndim != 4:
            raise ValueError(f"memory must have shape (B, C, S, S), got {memory.shape}")
        batch, channels, height, width = memory.shape
        if channels != self.memory_channels:
            raise ValueError(f"expected {self.memory_channels} channels, got {channels}")
        if height != self.memory_size or width != self.memory_size:
            raise ValueError(
                f"expected memory size {self.memory_size}, got {(height, width)}"
            )
        coords = self._coordinate_planes(
            batch=batch,
            device=memory.device,
            dtype=memory.dtype,
        )
        features = self.encoder(torch.cat([memory, coords], dim=1))
        pooled = torch.cat(
            [
                features.mean(dim=(2, 3)),
                features.amax(dim=(2, 3)),
            ],
            dim=1,
        )
        return self.classifier(pooled).squeeze(-1)

    def _coordinate_planes(
        self,
        *,
        batch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        values = torch.linspace(-1.0, 1.0, self.memory_size, device=device, dtype=dtype)
        row = values.view(1, 1, self.memory_size, 1).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        col = values.view(1, 1, 1, self.memory_size).expand(
            batch,
            1,
            self.memory_size,
            self.memory_size,
        )
        return torch.cat([row, col], dim=1)
