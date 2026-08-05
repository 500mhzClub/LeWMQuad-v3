from __future__ import annotations

import ast
import inspect
from pathlib import Path

import lewm.benchmarks.go2_navigation_development_trace_v1 as trace_module
import lewm.benchmarks.qualified_shared_v5_navigation_runtime_v1 as runtime_module
import lewm.models.shared_v5_target_observation_head_v1 as target_module
import lewm.models.two_resolution_frontier_value_head_v1 as g4_module


ROOT = Path(__file__).resolve().parents[2]
SLICE_PATHS = (
    ROOT / "lewm/benchmarks/go2_navigation_development_trace_v1.py",
    ROOT / "lewm/benchmarks/qualified_shared_v5_navigation_runtime_v1.py",
    ROOT / "lewm/models/shared_v5_target_observation_head_v1.py",
    ROOT / "lewm/models/two_resolution_frontier_value_head_v1.py",
)
FORBIDDEN_IMPORT_PREFIXES = (
    "lewm.models.encoders",
    "lewm.models.shared_observable_camera_ray_jepa_v5",
    "lewm.planning.native_learned_physical_projection",
    "lewm.planning.two_resolution_navigation_development_integration",
    "lewm.planning.two_resolution_target_evidence",
    "lewm_genesis",
    "genesis",
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _imports(tree: ast.Module) -> tuple[str, ...]:
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            names.append(node.module)
    return tuple(names)


def test_foundation_slice_has_closed_non_genesis_non_predecessor_import_graph() -> None:
    assert all(path.is_file() for path in SLICE_PATHS)
    for path in SLICE_PATHS:
        imported = _imports(_tree(path))
        assert not any(
            name == prefix or name.startswith(prefix + ".")
            for name in imported
            for prefix in FORBIDDEN_IMPORT_PREFIXES
        ), (path, imported)


def test_foundation_slice_has_no_artifact_open_or_dynamic_source_surface() -> None:
    forbidden_names = {"open", "eval", "exec", "compile", "__import__"}
    forbidden_attributes = {"load", "open", "import_module", "from_file"}
    for path in SLICE_PATHS:
        for node in ast.walk(_tree(path)):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_names, (path, node.lineno, node.func.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_attributes, (
                    path,
                    node.lineno,
                    node.func.attr,
                )


def test_all_production_runtime_and_artifact_identities_remain_none() -> None:
    for module in (trace_module, runtime_module, target_module, g4_module):
        production_names = [
            name for name in vars(module) if name.startswith("PRODUCTION_")
        ]
        assert production_names, module.__name__
        assert all(getattr(module, name) is None for name in production_names)


def test_auxiliary_heads_have_zero_encoder_preprocessor_and_fallback_ownership() -> None:
    target = target_module.SharedV5TargetObservationHeadV1(
        target_module.SharedV5TargetObservationHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            hidden_dim=16,
            color_embedding_dim=4,
        )
    )
    g4 = g4_module.TwoResolutionFrontierValueHeadV1(
        g4_module.TwoResolutionFrontierValueHeadConfigV1(
            patch_feature_dim=8,
            bev_feature_dim=6,
            candidate_feature_dim=5,
            hidden_dim=16,
        )
    )
    assert target.owned_encoder_count == 0
    assert target.owned_rgb_preprocessor_count == 0
    assert g4.owned_encoder_count == 0
    assert g4.owned_rgb_preprocessor_count == 0
    assert g4.owns_candidate_generator is False
    assert g4.has_fallback_selector is False
    for source in (
        inspect.getsource(target_module.SharedV5TargetObservationHeadV1.forward),
        inspect.getsource(g4_module.TwoResolutionFrontierValueHeadV1.forward),
    ):
        assert "forward_frame" not in source
        assert "forward_tokens" not in source
        assert "preprocess" not in source


def test_one_encode_source_graph_contains_one_backend_call_per_frame() -> None:
    runtime_source = inspect.getsource(
        runtime_module.QualifiedSharedV5NavigationRuntimeV1.run_shared_frame_once
    )
    backend_source = inspect.getsource(
        runtime_module.FakeSharedV5FrameBackendV1.forward_synthetic_frame
    )
    assert runtime_source.count(".preprocess_synthetic_frame(") == 1
    assert runtime_source.count(".forward_synthetic_frame(") == 1
    assert backend_source.count("._fake_forward_tokens(") == 1
    assert runtime_source.count("QualifiedSharedV5FrameOutcomeV1(") == 1

