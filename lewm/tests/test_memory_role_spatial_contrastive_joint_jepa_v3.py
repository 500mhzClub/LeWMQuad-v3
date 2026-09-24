from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import copy
import sys


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from lewm.models import memory_role_spatial_contrastive_joint_jepa_v3 as v3_module
from lewm.models.memory_role_factorized_joint_jepa_v1 import (
    MemoryRoleFactorizedJointJepaV1,
    MemoryRoleFactorizerV1,
)
from lewm.models.memory_role_spatial_contrastive_joint_jepa_v3 import (
    INITIALIZATION_SEED_MEMORY_ROLE_SPATIAL_CONTRASTIVE_V3,
    MemoryRoleSpatialContrastiveJointJepaV3,
    MemoryRoleSpatialFactorizerV3,
    PlaceKeyIdentityResidualPredictorV3,
)
from lewm.models.observable_camera_ray_evidence_v4 import (
    ObservableCameraRayEvidenceV4Model,
)
from scripts import run_go2_rgb_memory_role_factorized_joint_jepa_v1 as v3_training
from scripts.launch_go2_rgb_swept_progress_survival_joint_jepa_v13_camera_evidence_bottleneck import (
    V13ComposedRuntime,
)


def _sweep_masks() -> torch.Tensor:
    masks = torch.zeros((9, 16, 64, 64), dtype=torch.bool)
    masks[:, :, 31:33, 31:33] = True
    return masks


def test_module_exports_inherited_runtime_projection_seed() -> None:
    assert v3_module.PROJECTION_INITIALIZATION_SEED_V13 == 20_260_729


def _fitted_model() -> ObservableCameraRayEvidenceV4Model:
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(30_301)
        return ObservableCameraRayEvidenceV4Model().eval()
    finally:
        torch.random.set_rng_state(caller_rng)


def test_real_v13_runtime_initializer_accepts_v3_model_adapter() -> None:
    runtime = object.__new__(V13ComposedRuntime)
    runtime._initialized = False
    runtime.torch = torch
    runtime.executor_api = SimpleNamespace(
        MODEL_CLASS_NAME="MemoryRoleSpatialContrastiveJointJepaV3"
    )
    runtime.model_module = v3_module
    runtime.training_module = v3_training
    runtime.n320_fit = _fitted_model()
    runtime.sweep_masks = _sweep_masks()
    runtime.device = torch.device("cpu")
    runtime.n320_gate = {"passes": True}
    runtime.hardware = {"synthetic_cpu_test": True}
    runtime.runtime_fingerprint = {"synthetic_cpu_test": True}
    runtime.determinism = {"synthetic_cpu_test": True}
    runtime.runtime_path_containment = {"synthetic_cpu_test": True}

    model, optimizer, receipt = runtime.initialize_model_v13()

    assert type(model) is MemoryRoleSpatialContrastiveJointJepaV3
    assert isinstance(optimizer, torch.optim.AdamW)
    assert receipt["projection_initialization_seed"] == 20_260_729
    assert receipt["target_hard_sync_count"] == 1
    assert receipt["ema_update_count"] == 0


def test_v3_factorizer_retains_spatial_grid_and_v1_local_route() -> None:
    torch.random.default_generator.manual_seed(30_302)
    v1_factorizer = MemoryRoleFactorizerV1().train()
    local_state = copy.deepcopy(v1_factorizer.local_projection.state_dict())
    factorizer = MemoryRoleSpatialFactorizerV3(
        local_projection=v1_factorizer.local_projection
    ).train()
    latent = torch.randn(
        (2, 64, 64, 64),
        generator=torch.Generator().manual_seed(30_303),
        requires_grad=True,
    )

    encoded = factorizer(latent)

    assert factorizer.place_projection.in_channels == 64
    assert factorizer.place_projection.out_channels == 16
    assert factorizer.place_projection.kernel_size == (1, 1)
    assert factorizer.activation.approximate == "none"
    assert factorizer.place_pool.output_size == (4, 4)
    assert factorizer.place_output.in_features == 256
    assert factorizer.place_output.out_features == 64
    assert tuple(encoded.place_key.shape) == (2, 64)
    assert tuple(encoded.local_control.shape) == (2, 32, 16, 16)
    assert torch.allclose(
        torch.linalg.vector_norm(encoded.place_key, dim=1),
        torch.ones(2),
        rtol=0.0,
        atol=1.0e-5,
    )
    assert all(
        torch.equal(value, factorizer.local_projection.state_dict()[name])
        for name, value in local_state.items()
    )

    loss = encoded.place_key.square().mul(torch.arange(64)).mean()
    gradients = torch.autograd.grad(
        loss,
        (latent, *factorizer.place_projection.parameters(), *factorizer.place_output.parameters()),
    )
    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(int(torch.count_nonzero(gradient)) > 0 for gradient in gradients)


def test_v3_place_predictor_is_exact_zero_residual_identity_at_initialization() -> None:
    predictor = PlaceKeyIdentityResidualPredictorV3().train()
    final = predictor.net[2]
    place_key = F.normalize(
        torch.randn((4, 64), generator=torch.Generator().manual_seed(30_304)),
        dim=1,
    )

    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0
    assert torch.equal(predictor.net(place_key), torch.zeros_like(place_key))
    assert torch.equal(predictor(place_key), F.normalize(place_key, dim=1, eps=1.0e-6))

    target = F.normalize(
        torch.randn((4, 64), generator=torch.Generator().manual_seed(30_305)),
        dim=1,
    )
    loss = (1.0 - (predictor(place_key) * target).sum(dim=1)).mean()
    gradients = torch.autograd.grad(loss, tuple(predictor.parameters()))
    assert int(torch.count_nonzero(gradients[0])) == 0
    assert int(torch.count_nonzero(gradients[1])) == 0
    assert int(torch.count_nonzero(gradients[2])) > 0
    assert int(torch.count_nonzero(gradients[3])) > 0


def test_v3_model_preserves_parent_and_local_state_and_binds_ema_target() -> None:
    fitted = _fitted_model()
    v1 = MemoryRoleFactorizedJointJepaV1(fitted, _sweep_masks()).eval()
    v3 = MemoryRoleSpatialContrastiveJointJepaV3(fitted, _sweep_masks()).eval()

    for name in (
        "encoder",
        "bev_lift",
        "semantic_head",
        "predictor",
        "target_encoder",
        "target_bev_lift",
        "local_predictor",
    ):
        v1_state = getattr(v1, name).state_dict()
        v3_state = getattr(v3, name).state_dict()
        assert v1_state.keys() == v3_state.keys()
        assert all(torch.equal(value, v3_state[key]) for key, value in v1_state.items())
    v1_local = v1.role_factorizer.local_projection.state_dict()
    v3_local = v3.role_factorizer.local_projection.state_dict()
    assert v1_local.keys() == v3_local.keys()
    assert all(torch.equal(value, v3_local[key]) for key, value in v1_local.items())

    online = v3.role_factorizer.state_dict()
    target = v3.target_role_factorizer.state_dict()
    assert online.keys() == target.keys()
    assert all(torch.equal(value, target[name]) for name, value in online.items())
    assert not any(
        parameter.requires_grad
        for module in v3.target_modules()
        for parameter in module.parameters()
    )
    selected = v3.trainable_parameter_groups_memory_role_factorized_v1().online
    trainable = tuple(parameter for parameter in v3.parameters() if parameter.requires_grad)
    assert {id(parameter) for _, parameter in selected} == {
        id(parameter) for parameter in trainable
    }
    assert len({id(parameter) for _, parameter in selected}) == len(selected)

    online_place = v3.role_factorizer.place_projection.weight
    target_place = v3.target_role_factorizer.place_projection.weight
    before_target = target_place.detach().clone()
    before_count = int(v3.ema_update_count.item())
    with torch.no_grad():
        online_place.add_(1.0)
    v3.update_target_ema_after_optimizer_step()
    assert int(v3.ema_update_count.item()) == before_count + 1
    assert not torch.equal(target_place, before_target)
    assert target_place.grad is None
    assert target_place.requires_grad is False

    v3.train()
    assert v3.role_factorizer.training
    assert not v3.target_role_factorizer.training


def test_v3_new_place_seed_is_deterministic_and_constructor_restores_rng() -> None:
    fitted = _fitted_model()

    torch.random.default_generator.manual_seed(30_306)
    first_rng = torch.random.get_rng_state().clone()
    first = MemoryRoleSpatialContrastiveJointJepaV3(fitted, _sweep_masks()).eval()
    assert torch.equal(torch.random.get_rng_state(), first_rng)

    torch.random.default_generator.manual_seed(30_307)
    second_rng = torch.random.get_rng_state().clone()
    second = MemoryRoleSpatialContrastiveJointJepaV3(fitted, _sweep_masks()).eval()
    assert torch.equal(torch.random.get_rng_state(), second_rng)
    for name in ("role_factorizer", "place_predictor", "local_predictor"):
        first_state = getattr(first, name).state_dict()
        second_state = getattr(second, name).state_dict()
        assert first_state.keys() == second_state.keys()
        assert all(
            torch.equal(value, second_state[key])
            for key, value in first_state.items()
        )

    local_projection = copy.deepcopy(first.role_factorizer.local_projection)
    caller_rng = torch.random.get_rng_state().clone()
    try:
        torch.random.default_generator.manual_seed(
            INITIALIZATION_SEED_MEMORY_ROLE_SPATIAL_CONTRASTIVE_V3
        )
        expected_factorizer = MemoryRoleSpatialFactorizerV3(
            local_projection=local_projection
        )
        expected_predictor = PlaceKeyIdentityResidualPredictorV3()
    finally:
        torch.random.set_rng_state(caller_rng)
    for actual, expected in (
        (first.role_factorizer.place_projection, expected_factorizer.place_projection),
        (first.role_factorizer.place_output, expected_factorizer.place_output),
        (first.place_predictor, expected_predictor),
    ):
        actual_state = actual.state_dict()
        expected_state = expected.state_dict()
        assert actual_state.keys() == expected_state.keys()
        assert all(
            torch.equal(value, expected_state[key])
            for key, value in actual_state.items()
        )
