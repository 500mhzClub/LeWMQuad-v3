from __future__ import annotations

import copy
import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4RawOutput,
)


ROOT = Path(__file__).resolve().parents[2]
RUNNER_PATH = (
    ROOT
    / "scripts/run_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck.py"
)


def _load_runner() -> Any:
    name = "_test_go2_camera_evidence_bottleneck_v13_runner"
    spec = importlib.util.spec_from_file_location(name, RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    __import__("sys").modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


class _TinyEvidenceHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Conv2d(4, 1, kernel_size=1)
        self.hazard_bias = nn.Parameter(torch.linspace(-0.2, 0.2, 64))
        self.offset_bias = nn.Parameter(torch.linspace(-0.1, 0.1, 64))
        self.ground_bias = nn.Parameter(torch.linspace(-0.3, 0.3, 5))

    def decode(
        self,
        encoded: torch.Tensor,
        camera_origin_body_m: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> tuple[ObservableCameraRayEvidenceV4RawOutput, torch.Tensor, torch.Tensor]:
        signal = self.feature(encoded)
        hazard = signal + self.hazard_bias[None, :, None, None]
        offset = 0.05 * torch.tanh(
            signal + self.offset_bias[None, :, None, None]
        )
        pooled = signal.mean(dim=(2, 3), keepdim=True).permute(0, 2, 3, 1)
        ground = pooled + self.ground_bias[None, None, None, :]
        distance = (
            camera_origin_body_m[:, 2] - ground_plane_z_body_m
        )[:, None, None, None] + torch.linspace(
            0.1, 0.5, 5, device=ground.device, dtype=ground.dtype
        )[None, None, None, :]
        distance = distance.expand_as(ground).contiguous()
        raw = ObservableCameraRayEvidenceV4RawOutput(
            pixel_first_hit_hazard_logits=hazard,
            pixel_within_bin_offset_m=offset,
            ground_clear_to_target_logits=ground,
            ground_query_in_frustum=torch.ones_like(ground, dtype=torch.bool),
            ground_query_uv_px=torch.zeros(
                (*ground.shape, 2), device=ground.device, dtype=ground.dtype
            ),
            ground_target_distance_m=distance,
        )
        free_plane = ground.mean(dim=-1).unsqueeze(1).expand(-1, -1, 2, 2)
        occupied_plane = hazard.mean(dim=1, keepdim=True) + offset.mean(
            dim=1, keepdim=True
        )
        return raw, free_plane, occupied_plane


class _TinyBevLift(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.evidence_head = _TinyEvidenceHead()
        self.free_projection = nn.Conv2d(1, 32, kernel_size=1)
        self.occupied_projection = nn.Conv2d(1, 32, kernel_size=1)


class _TinySurvivalHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Linear(64, 16)

    def forward(self, predicted: torch.Tensor) -> torch.Tensor:
        pooled = predicted.mean(dim=(-2, -1))
        return self.readout(pooled)


class _TinyPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.delta = nn.Conv2d(64, 64, kernel_size=1)
        self.action_bias = nn.Parameter(torch.randn(9, 64, 1, 1) * 0.01)
        self.swept_progress_head = _TinySurvivalHead()

    def forward_all(self, latent: torch.Tensor) -> SimpleNamespace:
        base = latent + 0.1 * torch.tanh(self.delta(latent))
        predicted = base[:, None] + self.action_bias[None]
        return SimpleNamespace(
            predicted_latents=predicted,
            survival_logits=self.swept_progress_head(predicted),
        )


class _TinyJointModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Conv2d(3, 4, kernel_size=1)
        self.bev_lift = _TinyBevLift()
        self.semantic_head = nn.Conv2d(64, 3, kernel_size=1)
        self.predictor = _TinyPredictor()
        self.target_encoder = copy.deepcopy(self.encoder)
        self.target_bev_lift = copy.deepcopy(self.bev_lift)
        for parameter in (*self.target_encoder.parameters(), *self.target_bev_lift.parameters()):
            parameter.requires_grad_(False)
        self.register_buffer("ema_update_count", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _geometry(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        origin = rgb.new_tensor((0.326, 0.0, 0.043)).expand(rgb.shape[0], 3)
        ground = rgb.new_full((rgb.shape[0],), -0.333)
        return origin, ground

    @staticmethod
    def _encode(
        rgb: torch.Tensor,
        encoder: nn.Module,
        bev_lift: _TinyBevLift,
        origin: torch.Tensor,
        ground_z: torch.Tensor,
    ) -> tuple[torch.Tensor, ObservableCameraRayEvidenceV4RawOutput]:
        encoded = encoder(rgb)
        raw, free, occupied = bev_lift.evidence_head.decode(
            encoded, origin, ground_z
        )
        latent = torch.cat(
            (
                F.gelu(bev_lift.free_projection(free)),
                F.gelu(bev_lift.occupied_projection(occupied)),
            ),
            dim=1,
        )
        return latent, raw

    def encode_online_training(
        self,
        rgb: torch.Tensor,
        *,
        camera_origin_body_m: torch.Tensor,
        camera_basis_body_fru: torch.Tensor,
        ground_plane_z_body_m: torch.Tensor,
    ) -> SimpleNamespace:
        del camera_basis_body_fru
        latent, raw = self._encode(
            rgb,
            self.encoder,
            self.bev_lift,
            camera_origin_body_m,
            ground_plane_z_body_m,
        )
        return SimpleNamespace(latent=latent, auxiliary_evidence=raw)

    def encode_online(self, rgb: torch.Tensor) -> torch.Tensor:
        origin, ground = self._geometry(rgb)
        return self._encode(rgb, self.encoder, self.bev_lift, origin, ground)[0]

    def encode_target(self, rgb: torch.Tensor) -> torch.Tensor:
        origin, ground = self._geometry(rgb)
        return self._encode(
            rgb, self.target_encoder, self.target_bev_lift, origin, ground
        )[0].detach()

    def semantic_logits_from_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return self.semantic_head(latent)

    def predict_all_actions_with_survival(
        self, latent: torch.Tensor
    ) -> SimpleNamespace:
        return self.predictor.forward_all(latent)

    def update_target_ema_after_optimizer_step(self) -> None:
        with torch.no_grad():
            online = (*self.encoder.parameters(), *self.bev_lift.parameters())
            target = (
                *self.target_encoder.parameters(),
                *self.target_bev_lift.parameters(),
            )
            for target_parameter, online_parameter in zip(target, online, strict=True):
                target_parameter.mul_(0.996).add_(online_parameter, alpha=0.004)
            self.ema_update_count.add_(1)


def _microbatches() -> tuple[dict[str, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(20260729)
    result = []
    labels = torch.tensor([[0, 1], [2, 0]], dtype=torch.long)[None].expand(4, -1, -1)
    hit = torch.tensor(
        [[[True, False], [True, True]]], dtype=torch.bool
    ).expand(4, -1, -1)
    distance = torch.tensor(
        [[[0.10, 0.0], [0.35, 0.65]]], dtype=torch.float32
    ).expand(4, -1, -1).clone()
    ground_valid = torch.ones((4, 1, 1, 5), dtype=torch.bool)
    ground_clear = torch.tensor(
        [[[[True, False, True, False, True]]]], dtype=torch.bool
    ).expand(4, -1, -1, -1)
    basis = torch.eye(3).expand(4, -1, -1).clone()
    origin = torch.tensor((0.326, 0.0, 0.043)).expand(4, -1).clone()
    prefix = torch.tensor((0, 1, 2, 3, 4, 5, 6, 7, 8)).expand(4, -1).clone()
    for index in range(4):
        current = torch.randn((4, 3, 2, 2), generator=generator)
        next_rgb = torch.randn((4, 3, 2, 2), generator=generator) + 0.2
        result.append(
            {
                runner.CURRENT_RGB_KEY: current,
                runner.NEXT_RGB_KEY: next_rgb,
                runner.CURRENT_LABELS_KEY: labels.clone(),
                runner.NEXT_LABELS_KEY: labels.roll(index + 1, dims=0).clone(),
                runner.EXECUTED_ACTION_KEY: torch.tensor((0, 1, 2, 3)),
                runner.IMMEDIATE_FEASIBLE_KEY: torch.ones((4, 9), dtype=torch.bool),
                runner.PREFIX_LENGTHS_KEY: prefix.clone(),
                runner.CURRENT_CAMERA_ORIGIN_KEY: origin.clone(),
                runner.NEXT_CAMERA_ORIGIN_KEY: origin.clone(),
                runner.CURRENT_CAMERA_BASIS_KEY: basis.clone(),
                runner.NEXT_CAMERA_BASIS_KEY: basis.clone(),
                runner.CURRENT_GROUND_PLANE_Z_KEY: torch.full((4,), -0.333),
                runner.NEXT_GROUND_PLANE_Z_KEY: torch.full((4,), -0.333),
                runner.CURRENT_PIXEL_HIT_KEY: hit.clone(),
                runner.NEXT_PIXEL_HIT_KEY: hit.roll(1, dims=2).clone(),
                runner.CURRENT_PIXEL_DISTANCE_KEY: distance.clone(),
                runner.NEXT_PIXEL_DISTANCE_KEY: distance.roll(1, dims=2).clone(),
                runner.CURRENT_GROUND_IN_FRUSTUM_KEY: ground_valid.clone(),
                runner.NEXT_GROUND_IN_FRUSTUM_KEY: ground_valid.clone(),
                runner.CURRENT_GROUND_CLEAR_KEY: ground_clear.clone(),
                runner.NEXT_GROUND_CLEAR_KEY: (~ground_clear).clone(),
            }
        )
    return tuple(result)


def _accounting_at(update: int) -> runner.JointTrainingAccountingV13:
    return runner.JointTrainingAccountingV13(
        updates=update,
        presentations=16 * update,
        microbatch_graphs=4 * update,
        backward_calls=8 * update,
        camera_route_grad_calls=4 * update,
        joint_route_grad_calls=4 * update,
        camera_frame_objectives=32 * update,
        optimizer_steps=update,
        ema_steps=update,
        predictor_forwards=4 * update,
        predictor_objectives=4 * update,
    )


def test_camera_term_is_exact_three_way_b1_then_b4_then_pair_mean() -> None:
    model = _TinyJointModel()
    batch = _microbatches()[0]
    current = model.encode_online_training(
        batch[runner.CURRENT_RGB_KEY],
        camera_origin_body_m=batch[runner.CURRENT_CAMERA_ORIGIN_KEY],
        camera_basis_body_fru=batch[runner.CURRENT_CAMERA_BASIS_KEY],
        ground_plane_z_body_m=batch[runner.CURRENT_GROUND_PLANE_Z_KEY],
    )
    next_encoding = model.encode_online_training(
        batch[runner.NEXT_RGB_KEY],
        camera_origin_body_m=batch[runner.NEXT_CAMERA_ORIGIN_KEY],
        camera_basis_body_fru=batch[runner.NEXT_CAMERA_BASIS_KEY],
        ground_plane_z_body_m=batch[runner.NEXT_GROUND_PLANE_Z_KEY],
    )
    loss = runner.camera_evidence_pair_loss_v13(
        current.auxiliary_evidence,
        next_encoding.auxiliary_evidence,
        runner.CameraEvidenceFrameSupervisionV13(
            batch[runner.CURRENT_PIXEL_HIT_KEY],
            batch[runner.CURRENT_PIXEL_DISTANCE_KEY],
            batch[runner.CURRENT_GROUND_IN_FRUSTUM_KEY],
            batch[runner.CURRENT_GROUND_CLEAR_KEY],
        ),
        runner.CameraEvidenceFrameSupervisionV13(
            batch[runner.NEXT_PIXEL_HIT_KEY],
            batch[runner.NEXT_PIXEL_DISTANCE_KEY],
            batch[runner.NEXT_GROUND_IN_FRUSTUM_KEY],
            batch[runner.NEXT_GROUND_CLEAR_KEY],
        ),
    )
    for frame in (*loss.current_frames, *loss.next_frames):
        assert torch.equal(
            frame.total,
            (
                frame.hierarchical_first_hit_nll
                + frame.skew_balanced_pixel_offset
                + frame.balanced_ground_clear_bce
            )
            / 3.0,
        )
    assert torch.equal(
        loss.current_mean,
        torch.stack([frame.total for frame in loss.current_frames]).mean(),
    )
    assert torch.equal(
        loss.total, 0.5 * loss.current_mean + 0.5 * loss.next_mean
    )


def test_camera_term_rejects_auxiliary_target_validity_mismatch() -> None:
    model = _TinyJointModel()
    batch = _microbatches()[0]
    encoding = model.encode_online_training(
        batch[runner.CURRENT_RGB_KEY],
        camera_origin_body_m=batch[runner.CURRENT_CAMERA_ORIGIN_KEY],
        camera_basis_body_fru=batch[runner.CURRENT_CAMERA_BASIS_KEY],
        ground_plane_z_body_m=batch[runner.CURRENT_GROUND_PLANE_Z_KEY],
    )
    raw = runner._slice_raw_output_v13(
        encoding.auxiliary_evidence,
        0,
        ObservableCameraRayEvidenceV4RawOutput,
    )
    target_validity = batch[runner.CURRENT_GROUND_IN_FRUSTUM_KEY][:1].clone()
    target_clear = batch[runner.CURRENT_GROUND_CLEAR_KEY][:1].clone()
    target_validity[..., 0] = False
    target_clear[..., 0] = False
    with pytest.raises(ValueError, match="calibration validity differs"):
        runner.camera_evidence_frame_loss_v13(
            raw,
            runner.CameraEvidenceFrameSupervisionV13(
                batch[runner.CURRENT_PIXEL_HIT_KEY][:1],
                batch[runner.CURRENT_PIXEL_DISTANCE_KEY][:1],
                target_validity,
                target_clear,
            ),
        )


def test_partition_optimizer_and_float32_norm_scaling_are_exact() -> None:
    model = _TinyJointModel()
    partition = runner.partition_parameters_v13(model)
    assert partition.shared == partition.encoder + partition.evidence_head
    assert partition.online == (
        partition.shared + partition.representation + partition.predictor
    )
    assert {id(value) for value in partition.online + partition.target} == {
        id(value) for value in model.parameters()
    }
    assert all(name.startswith("bev_lift.evidence_head.") for name in partition.names["evidence_head"])
    optimizer = runner.build_frozen_optimizer_v13(partition)
    runner.validate_optimizer_v13(optimizer, partition)
    assert [group["name"] for group in optimizer.param_groups] == [
        "encoder",
        "evidence_projection_semantic",
        "predictor",
    ]
    norm, scale = runner._route_norm_and_scale_v13(
        torch, (torch.tensor((3.0, 4.0), dtype=torch.float32),)
    )
    assert norm.dtype == scale.dtype == torch.float32
    assert norm.item() == 5.0
    assert scale.item() == torch.tensor(0.2, dtype=torch.float32).item()


def test_one_update_has_separate_active_routes_one_step_one_ema_and_receipts() -> None:
    torch.manual_seed(71)
    model = _TinyJointModel()
    partition = runner.partition_parameters_v13(model)
    optimizer = runner.build_frozen_optimizer_v13(partition)
    target_before = tuple(value.detach().clone() for value in partition.target)
    result = runner.joint_training_update_v13(model, optimizer, _microbatches())

    assert result.accounting == runner.JointTrainingAccountingV13(
        updates=1,
        presentations=16,
        microbatch_graphs=4,
        backward_calls=8,
        camera_route_grad_calls=4,
        joint_route_grad_calls=4,
        camera_frame_objectives=32,
        optimizer_steps=1,
        ema_steps=1,
        predictor_forwards=4,
        predictor_objectives=4,
    )
    assert int(model.ema_update_count.item()) == 1
    assert result.optimizer_steps_this_update == result.ema_steps_this_update == 1
    assert result.target_gradient_tensor_count == 0
    assert all(parameter.grad is None for parameter in partition.target)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, partition.target, strict=True)
    )
    assert set(result.gradient_routes) == {
        "camera_shared",
        "joint_shared",
        "representation",
        "predictor",
    }
    for name, receipt in result.gradient_routes.items():
        assert math.isfinite(receipt.preclip_l2)
        assert math.isfinite(receipt.applied_scale)
        assert receipt.parameter_tensor_count > 0
        norm = torch.tensor(receipt.preclip_l2, dtype=torch.float32)
        expected_scale = torch.minimum(
            torch.tensor(1.0, dtype=torch.float32),
            torch.reciprocal(
                torch.maximum(
                    norm,
                    torch.tensor(torch.finfo(torch.float32).tiny),
                )
            ),
        ).item()
        assert receipt.applied_scale == expected_scale
        if name in ("camera_shared", "joint_shared", "predictor"):
            assert receipt.preclip_l2 > 0.0
    assert set(result.mean_losses) == {"S", "P", "U", "R", "O", "N", "C", "L"}
    assert math.isclose(
        result.mean_losses["N"],
        sum(result.mean_losses[name] for name in ("S", "P", "U", "R", "O")),
        rel_tol=2e-6,
        abs_tol=2e-6,
    )
    assert math.isclose(
        result.mean_losses["L"],
        result.mean_losses["N"] + result.mean_losses["C"],
        rel_tol=2e-6,
        abs_tol=2e-6,
    )
    assert all(
        state["step"].item() == 1 for state in optimizer.state.values()
    )


def test_cap_rejects_update_1000_and_malformed_near_cap_before_work() -> None:
    class _MustNotBeTouched:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"cap rejection touched {name}")

    runner._validate_update_capacity_v13(_accounting_at(999))
    with pytest.raises(PermissionError, match="no complete update"):
        runner.joint_training_update_v13(
            _MustNotBeTouched(),
            _MustNotBeTouched(),
            (),
            accounting=_accounting_at(1_000),
        )

    near_cap = _accounting_at(999)
    malformed_presentations = runner.JointTrainingAccountingV13(
        **{**near_cap.__dict__, "presentations": 15_985}
    )
    malformed_steps = runner.JointTrainingAccountingV13(
        **{**near_cap.__dict__, "optimizer_steps": 998}
    )
    for malformed in (malformed_presentations, malformed_steps):
        with pytest.raises(RuntimeError, match="inconsistent"):
            runner.joint_training_update_v13(
                _MustNotBeTouched(),
                _MustNotBeTouched(),
                (),
                accounting=malformed,
            )
