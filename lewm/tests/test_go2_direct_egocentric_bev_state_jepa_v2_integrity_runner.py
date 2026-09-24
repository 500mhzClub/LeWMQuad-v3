from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = (
    ROOT / "scripts/run_go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
LAUNCHER = (
    ROOT / "scripts/launch_go2_direct_egocentric_bev_state_jepa_v2_integrity.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V2_INTEGRITY_PREFLIGHT_JSON"
)


def _load(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_import_is_source_only_and_exactly_rebound() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(RUNNER)!r})
spec = importlib.util.spec_from_file_location("_direct_bev_v2_runner", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V1.contract is module.contract
assert module.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._V1.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert Path(module._V1.__file__).resolve() == path
assert module.contract.MODEL_RELATIVE_PATH == (
    "lewm/models/direct_egocentric_bev_state_jepa_v2_integrity.py"
)
args = module.parse_args([
    "--run",
    "--review-sha256", "0" * 64,
    "--authorization-sha256", "1" * 64,
])
assert args.review_sha256 == "0" * 64
assert args.authorization_sha256 == "1" * 64
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_launcher_import_rebinds_complete_inherited_stack() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(LAUNCHER)!r})
spec = importlib.util.spec_from_file_location("_direct_bev_v2_launcher", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V1.contract is module.contract
assert module._V1._V11.contract is module.contract
assert module._V1._V11._BASE.contract is module.contract
assert module._V1._V11._BASE.RUNNER_PATH == (
    module.ROOT / module.contract.RUNNER_RELATIVE_PATH
)
assert module.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._V1.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._V1._V11.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert module._V1._V11._BASE.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
assert Path(module._V1.__file__).resolve() == path
assert Path(module._V1._V11.__file__).resolve() == path
assert Path(module._V1._V11._BASE.__file__).resolve() == path
args = module.parse_args([
    "--review-sha256", "0" * 64,
    "--authorization-sha256", "1" * 64,
])
assert args.review_sha256 == "0" * 64
assert args.authorization_sha256 == "1" * 64
print("PASS")
"""
    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", program],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "PASS\n"
    assert completed.stderr == ""


def test_entrypoints_delegate_after_rebinding(monkeypatch) -> None:
    runner = _load("_direct_bev_v2_runner_delegate", RUNNER)
    runner_calls: list[tuple[str, str]] = []

    def fake_run_parent(
        *,
        review_file_sha256: str,
        authorization_file_sha256: str,
    ) -> int:
        assert runner._V1.contract is runner.contract
        runner_calls.append((review_file_sha256, authorization_file_sha256))
        return 17

    monkeypatch.setattr(runner._V1, "run_parent", fake_run_parent)
    assert runner.run_parent(
        review_file_sha256="a" * 64,
        authorization_file_sha256="b" * 64,
    ) == 17
    assert runner_calls == [("a" * 64, "b" * 64)]

    launcher = _load("_direct_bev_v2_launcher_delegate", LAUNCHER)
    launcher_calls: list[list[str] | None] = []

    def fake_main(argv=None) -> int:
        assert launcher._V1.contract is launcher.contract
        assert launcher._V1._V11.contract is launcher.contract
        launcher_calls.append(argv)
        return 23

    monkeypatch.setattr(launcher._V1, "main", fake_main)
    arguments = ["--review-sha256", "c" * 64]
    assert launcher.main(arguments) == 23
    assert launcher_calls == [arguments]


def test_only_v2_model_runtime_import_name_is_adapted(monkeypatch) -> None:
    runner = _load("_direct_bev_v2_model_name_adapter", RUNNER)
    calls: list[tuple[str, Path]] = []

    def fake_source_loader(name: str, path: Path) -> tuple[str, Path]:
        calls.append((name, path))
        return name, path

    monkeypatch.setattr(
        runner,
        "_FROZEN_V1_SOURCE_ONLY_MODULE",
        fake_source_loader,
    )
    model_path = runner.ROOT / runner.contract.MODEL_RELATIVE_PATH
    observed = runner._source_only_runtime_module(
        "lewm.models.direct_egocentric_bev_state_jepa_v1",
        model_path,
    )
    assert observed == (runner.V2_MODEL_RUNTIME_MODULE_NAME, model_path)

    other_path = runner.ROOT / runner.contract.FROZEN_V1_MODEL_RELATIVE_PATH
    observed_other = runner._source_only_runtime_module(
        "unchanged_module_name",
        other_path,
    )
    assert observed_other == ("unchanged_module_name", other_path)
    assert calls == [
        (runner.V2_MODEL_RUNTIME_MODULE_NAME, model_path),
        ("unchanged_module_name", other_path),
    ]
