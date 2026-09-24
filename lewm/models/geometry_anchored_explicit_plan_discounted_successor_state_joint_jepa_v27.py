"""Explicit-plan discounted-successor-state joint JEPA V27 model.

V27 preserves the complete V18 object-space height-volume model and adds one
absolute local predictor for an ordered four-action plan.  The new head has no
EMA copy and does not alter the inherited one-step survival/progress predictor.
"""
from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from lewm.models.geometry_anchored_swept_progress_survival_joint_jepa_v18_object_space_height_volume import (
    ONLINE_TRAINABLE_PARAMETER_COUNT_V18,
    OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18,
    PREDICTOR_GROUP_PARAMETER_COUNT_V18,
    PREDICTOR_PARAMETER_PREFIXES_V18,
    PROJECTION_INITIALIZATION_SEED_V13,
    REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
    REPRESENTATION_PARAMETER_PREFIXES_V18,
    SHARED_ROUTE_PARAMETER_COUNT_V18,
    SHARED_PARAMETER_PREFIXES_V18,
    TARGET_BOTTLENECK_PARAMETER_COUNT_V18,
    GeometryAnchoredDeformableBevLiftJointJepaV1Config,
    GeometryAnchoredSweptProgressSurvivalJointJepaV18,
    V13TrainableParameterGroups,
)


PLAN_HORIZON_V27 = 4
ACTION_COUNT_V27 = 9
ACTION_EMBEDDING_WIDTH_V27 = 16
PLAN_CONDITION_WIDTH_V27 = PLAN_HORIZON_V27 * ACTION_EMBEDDING_WIDTH_V27
PLAN_HIDDEN_WIDTH_V27 = 128
PLAN_LATENT_WIDTH_V27 = 64
PLAN_LATENT_SHAPE_V27 = (PLAN_LATENT_WIDTH_V27, 64, 64)
PLAN_INITIALIZATION_SEED_V27 = 20_260_730

PLAN_PREDICTOR_PARAMETER_PREFIX_V27 = "plan_predictor."
PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27 = 14
PLAN_PREDICTOR_PARAMETER_COUNT_V27 = 103_424
PREDICTOR_GROUP_PARAMETER_COUNT_V27 = (
    PREDICTOR_GROUP_PARAMETER_COUNT_V18 + PLAN_PREDICTOR_PARAMETER_COUNT_V27
)
ONLINE_TRAINABLE_PARAMETER_COUNT_V27 = (
    ONLINE_TRAINABLE_PARAMETER_COUNT_V18 + PLAN_PREDICTOR_PARAMETER_COUNT_V27
)
PLAN_GRADIENT_ROUTE_PARAMETER_COUNT_V27 = (
    SHARED_ROUTE_PARAMETER_COUNT_V18
    + OBJECT_SPACE_HEIGHT_VOLUME_PARAMETER_COUNT_V18
    + PLAN_PREDICTOR_PARAMETER_COUNT_V27
)


ParameterGroupV27 = tuple[tuple[str, nn.Parameter], ...]


class V27TrainableParameterGroups(NamedTuple):
    """Exact inherited and new online parameter views for V27."""

    shared: ParameterGroupV27
    representation: ParameterGroupV27
    inherited_predictor: ParameterGroupV27
    plan_predictor: ParameterGroupV27

    @property
    def inherited_v18(self) -> V13TrainableParameterGroups:
        """Return the old physical/J24 view without the V27 plan head."""

        return V13TrainableParameterGroups(
            shared=self.shared,
            representation=self.representation,
            predictor=self.inherited_predictor,
        )

    @property
    def predictor(self) -> ParameterGroupV27:
        """Return the exact optimizer predictor group in registered order."""

        return self.inherited_predictor + self.plan_predictor

    @property
    def online(self) -> ParameterGroupV27:
        return self.shared + self.representation + self.predictor


def _validate_plan_inputs_v27(
    current_latent: torch.Tensor,
    action_plan: torch.Tensor,
    *,
    reference: torch.Tensor,
) -> None:
    if not isinstance(current_latent, torch.Tensor):
        raise TypeError("current_latent must be a torch.Tensor")
    if (
        current_latent.ndim != 4
        or current_latent.shape[0] < 1
        or tuple(current_latent.shape[1:]) != PLAN_LATENT_SHAPE_V27
    ):
        raise ValueError("current_latent must have shape (B,64,64,64) with B >= 1")
    if current_latent.dtype != torch.float32 or current_latent.dtype != reference.dtype:
        raise TypeError("current_latent must use the plan predictor's exact float32 dtype")
    if current_latent.device != reference.device:
        raise TypeError("current_latent and plan predictor must share a device")
    if not bool(torch.isfinite(current_latent).all()):
        raise FloatingPointError("current_latent is nonfinite")

    if not isinstance(action_plan, torch.Tensor):
        raise TypeError("action_plan must be a torch.Tensor")
    if tuple(action_plan.shape) != (current_latent.shape[0], PLAN_HORIZON_V27):
        raise ValueError("action_plan must have shape (B,4)")
    if action_plan.dtype != torch.long:
        raise TypeError("action_plan must use exact torch.long action IDs")
    if action_plan.device != current_latent.device:
        raise TypeError("action_plan and current_latent must share a device")
    if bool(((action_plan < 0) | (action_plan >= ACTION_COUNT_V27)).any()):
        raise ValueError("action_plan IDs must be in the closed range 0 through 8")


class ExplicitPlanDiscountedSuccessorStatePredictorV27(nn.Module):
    """Absolute local successor-state predictor for one ordered H4 plan."""

    def __init__(self) -> None:
        super().__init__()
        caller_rng = torch.random.get_rng_state().clone()
        try:
            self.action_embeddings = nn.ModuleList(
                [
                    nn.Embedding(ACTION_COUNT_V27, ACTION_EMBEDDING_WIDTH_V27)
                    for _ in range(PLAN_HORIZON_V27)
                ]
            )
            self.plan_linear1 = nn.Linear(
                PLAN_CONDITION_WIDTH_V27,
                PLAN_HIDDEN_WIDTH_V27,
                bias=True,
            )
            self.plan_linear2 = nn.Linear(
                PLAN_HIDDEN_WIDTH_V27,
                2 * PLAN_LATENT_WIDTH_V27,
                bias=True,
            )
            self.state_projection = nn.Conv2d(
                PLAN_LATENT_WIDTH_V27,
                PLAN_LATENT_WIDTH_V27,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.fusion_projection = nn.Conv2d(
                PLAN_LATENT_WIDTH_V27,
                PLAN_LATENT_WIDTH_V27,
                kernel_size=3,
                padding=1,
                bias=True,
            )
            self.output_projection = nn.Conv2d(
                PLAN_LATENT_WIDTH_V27,
                PLAN_LATENT_WIDTH_V27,
                kernel_size=1,
                bias=True,
            )
            self.activation = nn.GELU(approximate="none")

            generator = torch.Generator(device="cpu")
            generator.manual_seed(PLAN_INITIALIZATION_SEED_V27)
            weighted_layers = (
                *self.action_embeddings,
                self.plan_linear1,
                self.plan_linear2,
                self.state_projection,
                self.fusion_projection,
                self.output_projection,
            )
            for layer in weighted_layers:
                nn.init.xavier_uniform_(
                    layer.weight,
                    gain=1.0,
                    generator=generator,
                )
                if getattr(layer, "bias", None) is not None:
                    nn.init.zeros_(layer.bias)
        finally:
            torch.random.set_rng_state(caller_rng)

        parameters = tuple(self.parameters())
        if (
            len(parameters) != PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27
            or sum(parameter.numel() for parameter in parameters)
            != PLAN_PREDICTOR_PARAMETER_COUNT_V27
            or any(
                parameter.dtype != torch.float32 or not parameter.requires_grad
                for parameter in parameters
            )
        ):
            raise RuntimeError("V27 plan-predictor parameter inventory changed")

    def forward(
        self,
        current_latent: torch.Tensor,
        action_plan: torch.Tensor,
    ) -> torch.Tensor:
        reference = self.action_embeddings[0].weight
        _validate_plan_inputs_v27(
            current_latent,
            action_plan,
            reference=reference,
        )

        action_condition = torch.cat(
            tuple(
                embedding(action_plan[:, position])
                for position, embedding in enumerate(self.action_embeddings)
            ),
            dim=1,
        )
        condition = self.plan_linear2(
            self.activation(self.plan_linear1(action_condition))
        )
        scale, bias = condition.chunk(2, dim=1)
        state = self.activation(self.state_projection(current_latent))
        fused = state * (1.0 + torch.tanh(scale)[:, :, None, None])
        fused = fused + bias[:, :, None, None]
        result = self.output_projection(
            self.activation(self.fusion_projection(fused))
        )
        if (
            tuple(result.shape)
            != (current_latent.shape[0], *PLAN_LATENT_SHAPE_V27)
            or result.dtype != current_latent.dtype
            or result.device != current_latent.device
        ):
            raise RuntimeError("V27 plan-predictor output contract changed")
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("V27 plan-predictor output is nonfinite")
        return result


class GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27(
    GeometryAnchoredSweptProgressSurvivalJointJepaV18
):
    """Exact V18 joint JEPA plus the separately routed absolute plan head."""

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
                self.config.action_dim != ACTION_COUNT_V27
                or self.config.bev_dim != PLAN_LATENT_WIDTH_V27
                or tuple(self.config.bev_size) != PLAN_LATENT_SHAPE_V27[1:]
            ):
                raise RuntimeError("V27 inherited action or latent geometry changed")
            self.plan_predictor = ExplicitPlanDiscountedSuccessorStatePredictorV27()
        finally:
            torch.random.set_rng_state(caller_rng)
        self._assert_parameter_accounting_v27()

    def trainable_parameter_groups_v18(self) -> V13TrainableParameterGroups:
        """Preserve the exact inherited V18 physical/J24 parameter view."""

        named = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
            and not name.startswith(PLAN_PREDICTOR_PARAMETER_PREFIX_V27)
        )

        def select(prefixes: tuple[str, ...]) -> ParameterGroupV27:
            return tuple(
                (name, parameter)
                for name, parameter in named
                if name.startswith(prefixes)
            )

        groups = V13TrainableParameterGroups(
            shared=select(SHARED_PARAMETER_PREFIXES_V18),
            representation=select(REPRESENTATION_PARAMETER_PREFIXES_V18),
            predictor=select(PREDICTOR_PARAMETER_PREFIXES_V18),
        )
        selected = tuple(item for group in groups for item in group)
        if (
            len({name for name, _ in selected}) != len(selected)
            or len({id(parameter) for _, parameter in selected}) != len(selected)
            or {name for name, _ in selected} != {name for name, _ in named}
            or tuple(
                sum(parameter.numel() for _, parameter in group)
                for group in groups
            )
            != (
                SHARED_ROUTE_PARAMETER_COUNT_V18,
                REPRESENTATION_GROUP_PARAMETER_COUNT_V18,
                PREDICTOR_GROUP_PARAMETER_COUNT_V18,
            )
        ):
            raise RuntimeError("V27 inherited V18 parameter view changed")
        return groups

    def trainable_parameter_groups_v27(self) -> V27TrainableParameterGroups:
        """Return the complete disjoint V27 online inventory."""

        inherited = self.trainable_parameter_groups_v18()
        plan = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
            and name.startswith(PLAN_PREDICTOR_PARAMETER_PREFIX_V27)
        )
        groups = V27TrainableParameterGroups(
            shared=inherited.shared,
            representation=inherited.representation,
            inherited_predictor=inherited.predictor,
            plan_predictor=plan,
        )
        selected = groups.online
        trainable = tuple(
            (name, parameter)
            for name, parameter in self.named_parameters(remove_duplicate=False)
            if parameter.requires_grad
        )
        if (
            len(plan) != PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27
            or sum(parameter.numel() for _, parameter in plan)
            != PLAN_PREDICTOR_PARAMETER_COUNT_V27
            or len({name for name, _ in selected}) != len(selected)
            or len({id(parameter) for _, parameter in selected}) != len(selected)
            or {name for name, _ in selected} != {name for name, _ in trainable}
            or sum(parameter.numel() for _, parameter in selected)
            != ONLINE_TRAINABLE_PARAMETER_COUNT_V27
        ):
            raise RuntimeError("V27 online parameter inventory changed")
        return groups

    def _assert_parameter_accounting(self) -> None:
        # V18 calls this dynamically before V27 has registered its new head.
        if not hasattr(self, "plan_predictor"):
            GeometryAnchoredSweptProgressSurvivalJointJepaV18._assert_parameter_accounting(
                self
            )
            return
        self._assert_parameter_accounting_v27()

    def _assert_parameter_accounting_v27(self) -> None:
        groups = self.trainable_parameter_groups_v27()
        if sum(parameter.numel() for _, parameter in groups.predictor) != (
            PREDICTOR_GROUP_PARAMETER_COUNT_V27
        ):
            raise RuntimeError("V27 optimizer predictor-group count changed")
        target_count = sum(
            parameter.numel()
            for module in self.target_modules()
            for parameter in module.parameters()
        )
        if target_count != TARGET_BOTTLENECK_PARAMETER_COUNT_V18:
            raise RuntimeError("V27 inherited EMA-target inventory changed")

    def predict_plan_successor(
        self,
        current_latent: torch.Tensor,
        action_plan: torch.Tensor,
    ) -> torch.Tensor:
        """Predict one absolute discounted successor statistic from an H4 plan."""

        return self.plan_predictor(current_latent, action_plan)


ExplicitPlanDiscountedSuccessorStateJointJepaV27 = (
    GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27
)
GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27Config = (
    GeometryAnchoredDeformableBevLiftJointJepaV1Config
)


__all__ = [
    "ACTION_COUNT_V27",
    "ACTION_EMBEDDING_WIDTH_V27",
    "ExplicitPlanDiscountedSuccessorStateJointJepaV27",
    "ExplicitPlanDiscountedSuccessorStatePredictorV27",
    "GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27",
    "GeometryAnchoredExplicitPlanDiscountedSuccessorStateJointJepaV27Config",
    "ONLINE_TRAINABLE_PARAMETER_COUNT_V27",
    "PLAN_CONDITION_WIDTH_V27",
    "PLAN_GRADIENT_ROUTE_PARAMETER_COUNT_V27",
    "PLAN_HIDDEN_WIDTH_V27",
    "PLAN_HORIZON_V27",
    "PLAN_INITIALIZATION_SEED_V27",
    "PLAN_LATENT_SHAPE_V27",
    "PLAN_LATENT_WIDTH_V27",
    "PLAN_PREDICTOR_PARAMETER_COUNT_V27",
    "PLAN_PREDICTOR_PARAMETER_PREFIX_V27",
    "PLAN_PREDICTOR_PARAMETER_TENSOR_COUNT_V27",
    "PREDICTOR_GROUP_PARAMETER_COUNT_V27",
    "PROJECTION_INITIALIZATION_SEED_V13",
    "V27TrainableParameterGroups",
]
