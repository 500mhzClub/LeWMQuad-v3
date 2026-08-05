"""Small matched recurrent predictor for the H1 physical-dynamics assay.

The model receives three ordered 4 x 4 projected DINO context grids.  The first motion
row is the zero start token; the next two rows contain a body-frame odometry
increment and the exact 15-value executed command tape that led to the newly
observed frame.  Candidate command tapes are queried only after the recurrent
context state has been formed.

The visual and no-vision arms use the exact same module and initialization.
The no-vision intervention is implemented by supplying zero visual slots.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math

import torch
import torch.nn as nn


CONTEXT_STEPS = 3
TOKEN_COUNT = 16
VISUAL_WIDTH = 16
MOTION_WIDTH = 18
CANDIDATE_WIDTH = 15
HIDDEN_WIDTH = 16
OUTPUT_WIDTH = 4
PARAMETER_COUNT = 3_604

STATE_IDENTITY_SCHEMA = "lewm_go2_task_coupled_recurrent_dynamics_state_v1"


class TaskCoupledRecurrentDynamicsV1(nn.Module):
    """GRU context encoder followed by a candidate-conditioned outcome head."""

    def __init__(self) -> None:
        super().__init__()
        self.position_embedding = nn.Parameter(torch.empty(TOKEN_COUNT, HIDDEN_WIDTH))
        self.recurrence = nn.GRUCell(VISUAL_WIDTH + MOTION_WIDTH, HIDDEN_WIDTH)
        self.candidate_projection = nn.Linear(CANDIDATE_WIDTH, HIDDEN_WIDTH)
        self.query_hidden = nn.Linear(HIDDEN_WIDTH * 2, HIDDEN_WIDTH)
        self.query_output = nn.Linear(HIDDEN_WIDTH, OUTPUT_WIDTH)
        if sum(parameter.numel() for parameter in self.parameters()) != PARAMETER_COUNT:
            raise RuntimeError("recurrent-dynamics parameter inventory changed")

    def forward(
        self,
        visual_context: torch.Tensor,
        transition_motion: torch.Tensor,
        candidate_commands: torch.Tensor,
    ) -> torch.Tensor:
        """Predict standardized physical residuals with shape ``(B,A,4)``."""

        if visual_context.ndim != 4 or tuple(visual_context.shape[1:]) != (
            CONTEXT_STEPS,
            TOKEN_COUNT,
            VISUAL_WIDTH,
        ):
            raise ValueError("visual_context must have shape (B,3,16,16)")
        batch = int(visual_context.shape[0])
        if batch < 1:
            raise ValueError("batch must be nonempty")
        if tuple(transition_motion.shape) != (batch, CONTEXT_STEPS, MOTION_WIDTH):
            raise ValueError("transition_motion must have shape (B,3,18)")
        if (
            candidate_commands.ndim != 3
            or candidate_commands.shape[0] != batch
            or candidate_commands.shape[1] < 1
            or candidate_commands.shape[2] != CANDIDATE_WIDTH
        ):
            raise ValueError("candidate_commands must have shape (B,A,15), A >= 1")
        values = (visual_context, transition_motion, candidate_commands)
        if any(value.dtype != torch.float32 for value in values):
            raise TypeError("all recurrent-dynamics inputs must use float32")
        device = next(self.parameters()).device
        if any(value.device != device for value in values):
            raise TypeError("inputs and recurrent-dynamics parameters must share a device")
        if not all(bool(torch.isfinite(value).all()) for value in values):
            raise FloatingPointError("recurrent-dynamics input contains a nonfinite value")

        state = torch.zeros(
            (batch, TOKEN_COUNT, HIDDEN_WIDTH), dtype=torch.float32, device=device
        )
        for step in range(CONTEXT_STEPS):
            observation = visual_context[:, step] + self.position_embedding
            motion = transition_motion[:, step]
            recurrent_input = torch.cat(
                (observation, motion[:, None, :].expand(-1, TOKEN_COUNT, -1)),
                dim=-1,
            )
            state = self.recurrence(
                recurrent_input.reshape(-1, VISUAL_WIDTH + MOTION_WIDTH),
                state.reshape(-1, HIDDEN_WIDTH),
            ).reshape(batch, TOKEN_COUNT, HIDDEN_WIDTH)
        candidate = self.candidate_projection(candidate_commands)
        logits = torch.einsum(
            "bah,bth->bat", candidate, state
        ) / math.sqrt(HIDDEN_WIDTH)
        weights = torch.softmax(logits, dim=-1)
        pooled = torch.einsum("bat,bth->bah", weights, state)
        queries = torch.cat((pooled, candidate), dim=-1)
        prediction = self.query_output(torch.tanh(self.query_hidden(queries)))
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("recurrent-dynamics prediction became nonfinite")
        return prediction


def initialize_task_coupled_recurrent_dynamics_v1(
    seed: int,
) -> TaskCoupledRecurrentDynamicsV1:
    """Construct the exact deterministic Xavier/zero initial state on CPU."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or seed >= 2**63:
        raise ValueError("seed must be an integer in [0, 2**63 - 1]")
    # Construction is isolated from the process-global RNG; every scientific
    # parameter is overwritten below with a dedicated generator.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = TaskCoupledRecurrentDynamicsV1()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name == "position_embedding":
                nn.init.trunc_normal_(parameter, std=0.02, generator=generator)
            elif name in {"query_output.weight", "query_output.bias"}:
                parameter.zero_()
            elif "weight" in name:
                nn.init.xavier_uniform_(parameter, gain=1.0, generator=generator)
            elif "bias" in name:
                parameter.zero_()
            else:  # pragma: no cover - protects the frozen inventory
                raise RuntimeError(f"unexpected recurrent parameter: {name}")
    return model


def recurrent_dynamics_state_identity_v1(
    model_or_state: TaskCoupledRecurrentDynamicsV1 | Mapping[str, torch.Tensor],
) -> str:
    """Hash tensor names, dtypes, shapes, and exact contiguous CPU bytes."""

    state = (
        model_or_state.state_dict()
        if isinstance(model_or_state, TaskCoupledRecurrentDynamicsV1)
        else model_or_state
    )
    if not isinstance(state, Mapping) or not state:
        raise TypeError("model_or_state must provide a nonempty tensor mapping")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        expected = set(TaskCoupledRecurrentDynamicsV1().state_dict())
    if set(state) != expected:
        raise ValueError("recurrent-dynamics state inventory changed")
    digest = hashlib.sha256()
    digest.update(STATE_IDENTITY_SCHEMA.encode("ascii") + b"\0")
    for name in sorted(state):
        value = state[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError("recurrent-dynamics state contains a non-tensor")
        tensor = value.detach().cpu().contiguous()
        if tensor.dtype != torch.float32 or not bool(torch.isfinite(tensor).all()):
            raise ValueError("recurrent-dynamics state must be finite float32")
        header = json.dumps(
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


__all__ = [
    "CANDIDATE_WIDTH",
    "CONTEXT_STEPS",
    "HIDDEN_WIDTH",
    "MOTION_WIDTH",
    "OUTPUT_WIDTH",
    "PARAMETER_COUNT",
    "TOKEN_COUNT",
    "TaskCoupledRecurrentDynamicsV1",
    "VISUAL_WIDTH",
    "initialize_task_coupled_recurrent_dynamics_v1",
    "recurrent_dynamics_state_identity_v1",
]
