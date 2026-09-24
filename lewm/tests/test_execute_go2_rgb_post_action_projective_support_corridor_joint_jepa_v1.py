from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ENTRYPOINT = (
    ROOT
    / "scripts/execute_go2_rgb_post_action_projective_support_corridor_joint_jepa_v1.py"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, ENTRYPOINT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_thin_entrypoint_forwards_only_explicit_binding_and_root(tmp_path: Path) -> None:
    module = _load("_test_projective_support_execute_forward")
    calls: list[tuple[Path, Path]] = []

    def run(binding_path: Path, *, repository_root: Path) -> dict[str, str]:
        calls.append((binding_path, repository_root))
        return {"status": "synthetic"}

    module._core_runner = lambda: SimpleNamespace(
        run_from_execution_binding_v1=run
    )
    binding = tmp_path / "binding.json"
    result = module.execute_from_binding_v1(
        binding,
        repository_root=tmp_path,
    )
    assert result == {"status": "synthetic"}
    assert calls == [(binding.absolute(), tmp_path.absolute())]


def test_main_requires_the_explicit_execution_binding(tmp_path: Path) -> None:
    module = _load("_test_projective_support_execute_main")
    calls: list[tuple[Path, Path]] = []
    def execute(path: Path, *, repository_root: Path) -> int:
        calls.append((path, repository_root))
        return 2

    module.execute_from_binding_v1 = execute
    binding = tmp_path / "binding.json"
    assert module.main(
        [
            "--execution-binding",
            str(binding),
            "--repository-root",
            str(tmp_path),
        ]
    ) == 2
    assert calls == [(binding, tmp_path)]
