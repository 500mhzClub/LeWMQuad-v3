from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = (
    ROOT
    / "scripts/launch_go2_direct_egocentric_bev_state_jepa_v6_"
    "phase_separated_frozen_state_prediction.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V6_"
    "PHASE_SEPARATED_FROZEN_STATE_PREDICTION_PREFLIGHT_JSON"
)


def _load(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, LAUNCHER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_isolated_import_is_stdlib_only_and_fully_rebound() -> None:
    program = f"""
import importlib.util
from pathlib import Path
import sys
path = Path({str(LAUNCHER)!r})
spec = importlib.util.spec_from_file_location('_v6_launcher_isolated', path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert 'torch' not in sys.modules
assert not any(name.startswith('torch.') for name in sys.modules)
assert 'numpy' not in sys.modules
assert not any(name.startswith('numpy.') for name in sys.modules)
assert 'PIL' not in sys.modules
owners = (
    module._V5, module._V5._V4, module._V5._V4._V3,
    module._V5._V4._V3._V2, module._V5._V4._V3._V2._V1,
    module._V5._V4._V3._V2._V1._V11,
    module._V5._V4._V3._V2._V1._V11._BASE,
)
assert all(owner.contract is module.contract for owner in owners)
assert all(owner.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r} for owner in owners)
assert all(Path(owner.__file__).resolve() == path for owner in owners)
assert module._V5._V4._V3._V2._V1._V11._BASE.RUNNER_PATH == (
    module.ROOT / module.contract.RUNNER_RELATIVE_PATH
)
args = module.parse_args([
    '--review-sha256', '0' * 64,
    '--authorization-sha256', '1' * 64,
])
assert args.review_sha256 == '0' * 64
assert args.authorization_sha256 == '1' * 64
print('PASS')
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


def test_v6_authority_identities_caps_and_protected_denials() -> None:
    launcher = _load("_direct_bev_v6_launcher_authority")
    contract = launcher.contract
    assert contract.LAUNCHER_RELATIVE_PATH == LAUNCHER.relative_to(ROOT).as_posix()
    assert contract.RUNNER_RELATIVE_PATH.endswith(
        "run_go2_direct_egocentric_bev_state_jepa_v6_"
        "phase_separated_frozen_state_prediction.py"
    )
    assert contract.PREFLIGHT_ENVIRONMENT_KEY == PREFLIGHT_KEY
    assert contract.OUTPUT_ROOT_RELATIVE_PATH.endswith(
        "rgb_direct_egocentric_bev_state_jepa_probe_v6_"
        "phase_separated_frozen_state_prediction"
    )
    authority = contract.EXECUTION_AUTHORITY
    assert authority["one_exact_fresh_attempt_authorized"] is True
    assert authority["attempt_index"] == 1
    assert authority["maximum_attempts"] == 1
    assert authority["maximum_updates"] == 1_000
    assert authority["maximum_presentations"] == 16_000
    assert authority["gpu_active_minutes_maximum"] == 60
    assert authority["output_root"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    assert authority["output_root_must_be_absent_before_reservation"] is True
    assert authority["phase_separated_frozen_state_prediction_only"] is True
    assert authority["v5_retry_authorized"] is False
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert all(
        contract.DOWNSTREAM_DENIALS[name] is False
        for name in (
            "g2_authorized",
            "navigation_authorized",
            "heldout_authorized",
            "sealed_authorized",
            "production_authorized",
            "promotion_authorized",
            "deployment_authorized",
            "retry_resume_repair_recovery_replacement_or_second_seed_authorized",
        )
    )


def test_argument_validation_fails_closed() -> None:
    launcher = _load("_direct_bev_v6_launcher_arguments")
    with pytest.raises(SystemExit):
        launcher.parse_args([
            "--review-sha256", "not-a-sha",
            "--authorization-sha256", "1" * 64,
        ])
    with pytest.raises(SystemExit):
        launcher.parse_args(["--review-sha256", "0" * 64])


def test_main_delegates_only_after_rebinding(monkeypatch) -> None:
    launcher = _load("_direct_bev_v6_launcher_delegate")
    calls: list[list[str] | None] = []

    def fake_main(argv=None) -> int:
        assert launcher._V5.contract is launcher.contract
        assert launcher._V5._V4.contract is launcher.contract
        assert (
            launcher._V5._V4._V3._V2._V1._V11._BASE.contract
            is launcher.contract
        )
        calls.append(argv)
        return 43

    monkeypatch.setattr(launcher._V5, "main", fake_main)
    launcher._V5._V4._V3._V2.contract = object()
    arguments = [
        "--review-sha256", "2" * 64,
        "--authorization-sha256", "3" * 64,
    ]
    assert launcher.main(arguments) == 43
    assert calls == [arguments]
