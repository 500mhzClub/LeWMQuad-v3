"""A small memory-role factorization on V18's RGB joint-JEPA state.

The probe exposes a stable place key for a later learned memory and a spatial,
action-sensitive state for immediate control.  Both projections are learned
through the shared online RGB route and mirrored by the EMA target.  There is
deliberately no recurrent memory in this first falsification.
"""
from __future__ import annotations

import copy
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    PREDICTOR_PARAMETER_PREFIXES_V18,
    PROJECTION_INITIALIZATION_SEED_V13,
    REPRESENTATION_PARAMETER_PREFIXES_V18,
    SHARED_PARAMETER_PREFIXES_V18,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV18,
    V13TrainableParameterGroups,
)


ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1 = 9
SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1 = (64, 64, 64)
PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1 = 64
LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1 = (32, 16, 16)
INITIALIZATION_SEED_MEMORY_ROLE_FACTORIZED_V1 = 20_260_731

ROLE_FACTORIZER_PREFIX_MEMORY_ROLE_FACTORIZED_V1 = "role_factorizer."
PLACE_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1 = "place_predictor."
LOCAL_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1 = "local_predictor."
NEW_PREFIXES_MEMORY_ROLE_FACTORIZED_V1 = (
    ROLE_FACTORIZER_PREFIX_MEMORY_ROLE_FACTORIZED_V1,
    PLACE_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1,
    LOCAL_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1,
)

ParameterGroupV1 = tuple[tuple[str, nn.Parameter], ...]


class MemoryRoleEncodingV1(NamedTuple):
    place_key: torch.Tensor
    local_control: torch.Tensor


class MemoryRolePredictionV1(NamedTuple):
    place_key: torch.Tensor
    local_control: torch.Tensor


class MemoryRoleFactorizedTrainableParameterGroupsV1(NamedTuple):
    inherited_v18: V13TrainableParameterGroups
    role_factorizer: ParameterGroupV1
    place_predictor: ParameterGroupV1
    local_predictor: ParameterGroupV1

    @property
    def role_predictors(self) -> ParameterGroupV1:
        return self.place_predictor + self.local_predictor

    @property
    def online(self) -> ParameterGroupV1:
        return (
            self.inherited_v18.shared
            + self.inherited_v18.representation
            + self.inherited_v18.predictor
            + self.role_factorizer
            + self.role_predictors
        )


def _validate_float_tensor(
    value: torch.Tensor,
    *,
    shape_after_batch: tuple[int, ...],
    name: str,
    reference: torch.Tensor,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.ndim != len(shape_after_batch) + 1 or tuple(value.shape[1:]) != (
        shape_after_batch
    ):
        raise ValueError(f"{name} must have shape (B,{shape_after_batch})")
    if value.shape[0] < 1:
        raise ValueError(f"{name} must contain at least one row")
    if value.dtype != torch.float32 or value.dtype != reference.dtype:
        raise TypeError(f"{name} must use exact float32")
    if value.device != reference.device:
        raise TypeError(f"{name} and module must share a device")
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"{name} is nonfinite")


def _validate_actions(action_one_hot: torch.Tensor, reference: torch.Tensor) -> None:
    _validate_float_tensor(
        action_one_hot,
        shape_after_batch=(ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1,),
        name="action_one_hot",
        reference=reference,
    )
    if not bool(((action_one_hot == 0.0) | (action_one_hot == 1.0)).all()):
        raise ValueError("action_one_hot must contain exact zeros and ones")
    expected = torch.ones(
        action_one_hot.shape[0],
        dtype=reference.dtype,
        device=reference.device,
    )
    if not torch.equal(action_one_hot.sum(dim=1), expected):
        raise ValueError("each action row must contain exactly one active action")


class MemoryRoleFactorizerV1(nn.Module):
    """Split one shared object-space state into place and control roles."""

    def __init__(self) -> None:
        super().__init__()
        self.place_projection = nn.Conv2d(64, 64, kernel_size=1)
        self.place_output = nn.Linear(64, PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1)
        self.local_projection = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(approximate="none"),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(approximate="none"),
        )
        self.activation = nn.GELU(approximate="none")

    def forward(self, latent: torch.Tensor) -> MemoryRoleEncodingV1:
        _validate_float_tensor(
            latent,
            shape_after_batch=SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
            name="latent",
            reference=self.place_projection.weight,
        )
        place_map = self.activation(self.place_projection(latent))
        place_key = F.normalize(
            self.place_output(place_map.mean(dim=(-2, -1))),
            dim=1,
            eps=1.0e-6,
        )
        local_control = self.local_projection(latent)
        if tuple(local_control.shape[1:]) != LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1:
            raise RuntimeError("local-control output shape changed")
        return MemoryRoleEncodingV1(place_key, local_control)


class PlaceKeyPredictorV1(nn.Module):
    """Predict a nearby observation's stable place key without an action input."""

    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1, 128),
            nn.GELU(approximate="none"),
            nn.Linear(128, PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1),
        )

    def forward(self, place_key: torch.Tensor) -> torch.Tensor:
        reference = self.net[0].weight
        _validate_float_tensor(
            place_key,
            shape_after_batch=(PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1,),
            name="place_key",
            reference=reference,
        )
        return F.normalize(self.net(place_key), dim=1, eps=1.0e-6)


class ActionConditionedLocalControlPredictorV1(nn.Module):
    """Predict one immediate local-control target for a supplied action."""

    def __init__(self) -> None:
        super().__init__()
        self.action_projection = nn.Linear(ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1, 64)
        self.state_projection = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fusion_projection = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.output_projection = nn.Conv2d(32, 32, kernel_size=1)
        self.activation = nn.GELU(approximate="none")

    def forward(
        self,
        local_control: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> torch.Tensor:
        _validate_float_tensor(
            local_control,
            shape_after_batch=LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1,
            name="local_control",
            reference=self.state_projection.weight,
        )
        _validate_actions(action_one_hot, self.action_projection.weight)
        if action_one_hot.shape[0] != local_control.shape[0]:
            raise ValueError("action_one_hot and local_control batch sizes differ")

        scale, bias = self.action_projection(action_one_hot).chunk(2, dim=1)
        state = self.activation(self.state_projection(local_control))
        fused = state * (1.0 + torch.tanh(scale)[:, :, None, None])
        fused = fused + bias[:, :, None, None]
        return self.output_projection(
            self.activation(self.fusion_projection(fused))
        )


class MemoryRoleFactorizedJointJepaV1(
    GeometryAnchoredSweptProgressSurvivalJointJepaV18
):
    """V18 perception with jointly learned memory and control interfaces."""

    def __init__(
        self,
        n320_fit_model: ObservableCameraRayEvidenceV4Model,
        sweep_masks: torch.Tensor,
        config: GeometryAnchoredDeformableBevLiftJointJepaV1Config | None = None,
    ) -> None:
        caller_rng = torch.random.get_rng_state().clone()
        try:
            super().__init__(n320_fit_model, sweep_masks, config)
            if (
                self.config.action_dim != ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1
                or (self.config.bev_dim, *self.config.bev_size)
                != SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1
            ):
                raise RuntimeError("memory-role source geometry changed")
            torch.random.default_generator.manual_seed(
                INITIALIZATION_SEED_MEMORY_ROLE_FACTORIZED_V1
            )
            self.role_factorizer = MemoryRoleFactorizerV1()
            self.place_predictor = PlaceKeyPredictorV1()
            self.local_predictor = ActionConditionedLocalControlPredictorV1()
            self.target_role_factorizer = copy.deepcopy(self.role_factorizer)
        finally:
            torch.random.set_rng_state(caller_rng)
        self._freeze_target()
        self._assert_role_target_binding()
        self.trainable_parameter_groups_memory_role_factorized_v1()

    def online_target_modules(self) -> tuple[nn.Module, ...]:
        inherited = super().online_target_modules()
        if not hasattr(self, "role_factorizer"):
            return inherited
        return inherited + (self.role_factorizer,)

    def target_modules(self) -> tuple[nn.Module, ...]:
        inherited = super().target_modules()
        if not hasattr(self, "target_role_factorizer"):
            return inherited
        return inherited + (self.target_role_factorizer,)

    def _assert_role_target_binding(self) -> None:
        online = self.role_factorizer.state_dict()
        target = self.target_role_factorizer.state_dict()
        if online.keys() != target.keys() or any(
            not torch.equal(value, target[name]) for name, value in online.items()
        ):
            raise RuntimeError("role target does not match its online source")
        if any(
            parameter.requires_grad
            for parameter in self.target_role_factorizer.parameters()
        ):
            raise RuntimeError("role target must be frozen")

    def trainable_parameter_groups_v18(self) -> V13TrainableParameterGroups:
        """Keep the inherited V18 auxiliary-objective parameter view."""

        named = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
            and not name.startswith(NEW_PREFIXES_MEMORY_ROLE_FACTORIZED_V1)
        )

        def select(prefixes: tuple[str, ...]) -> ParameterGroupV1:
            return tuple(item for item in named if item[0].startswith(prefixes))

        groups = V13TrainableParameterGroups(
            shared=select(SHARED_PARAMETER_PREFIXES_V18),
            representation=select(REPRESENTATION_PARAMETER_PREFIXES_V18),
            predictor=select(PREDICTOR_PARAMETER_PREFIXES_V18),
        )
        selected = tuple(item for group in groups for item in group)
        if (
            len({id(parameter) for _, parameter in selected}) != len(selected)
            or {name for name, _ in selected} != {name for name, _ in named}
        ):
            raise RuntimeError("inherited V18 parameter view is incomplete")
        return groups

    def trainable_parameter_groups_memory_role_factorized_v1(
        self,
    ) -> MemoryRoleFactorizedTrainableParameterGroupsV1:
        def select(prefix: str) -> ParameterGroupV1:
            return tuple(
                (name, parameter)
                for name, parameter in self.named_parameters(remove_duplicate=False)
                if parameter.requires_grad and name.startswith(prefix)
            )

        groups = MemoryRoleFactorizedTrainableParameterGroupsV1(
            inherited_v18=self.trainable_parameter_groups_v18(),
            role_factorizer=select(ROLE_FACTORIZER_PREFIX_MEMORY_ROLE_FACTORIZED_V1),
            place_predictor=select(PLACE_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1),
            local_predictor=select(LOCAL_PREDICTOR_PREFIX_MEMORY_ROLE_FACTORIZED_V1),
        )
        selected = groups.online
        trainable = tuple(parameter for parameter in self.parameters() if parameter.requires_grad)
        if (
            any(not group for group in groups[1:])
            or len({id(parameter) for _, parameter in selected}) != len(selected)
            or {id(parameter) for _, parameter in selected}
            != {id(parameter) for parameter in trainable}
        ):
            raise RuntimeError("memory-role online parameter view is incomplete")
        return groups

    def encode_online_roles(self, rgb: torch.Tensor) -> MemoryRoleEncodingV1:
        return self.role_factorizer(self.encode_online(rgb))

    @torch.no_grad()
    def encode_target_roles(self, rgb: torch.Tensor) -> MemoryRoleEncodingV1:
        encoded = self.target_role_factorizer(self.encode_target(rgb))
        return MemoryRoleEncodingV1(
            encoded.place_key.detach(),
            encoded.local_control.detach(),
        )

    def predict_roles(
        self,
        current: MemoryRoleEncodingV1,
        action_one_hot: torch.Tensor,
    ) -> MemoryRolePredictionV1:
        if not isinstance(current, MemoryRoleEncodingV1):
            raise TypeError("current must be MemoryRoleEncodingV1")
        return MemoryRolePredictionV1(
            self.place_predictor(current.place_key),
            self.local_predictor(current.local_control, action_one_hot),
        )

    def forward(
        self,
        current_rgb: torch.Tensor,
        action_one_hot: torch.Tensor,
    ) -> MemoryRolePredictionV1:
        return self.predict_roles(self.encode_online_roles(current_rgb), action_one_hot)


__all__ = [
    "ACTION_COUNT_MEMORY_ROLE_FACTORIZED_V1",
    "ActionConditionedLocalControlPredictorV1",
    "INITIALIZATION_SEED_MEMORY_ROLE_FACTORIZED_V1",
    "LOCAL_CONTROL_SHAPE_MEMORY_ROLE_FACTORIZED_V1",
    "MemoryRoleEncodingV1",
    "MemoryRoleFactorizedJointJepaV1",
    "MemoryRoleFactorizedTrainableParameterGroupsV1",
    "MemoryRoleFactorizerV1",
    "MemoryRolePredictionV1",
    "PLACE_KEY_WIDTH_MEMORY_ROLE_FACTORIZED_V1",
    "PlaceKeyPredictorV1",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "SOURCE_LATENT_SHAPE_MEMORY_ROLE_FACTORIZED_V1",
]
