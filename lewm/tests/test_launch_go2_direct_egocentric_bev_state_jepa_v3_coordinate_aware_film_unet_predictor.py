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
    / "scripts/launch_go2_direct_egocentric_bev_state_jepa_v3_"
    "coordinate_aware_film_unet_predictor.py"
)
PREFLIGHT_KEY = (
    "LEWM_DIRECT_EGOCENTRIC_BEV_STATE_JEPA_V3_"
    "COORDINATE_AWARE_FILM_UNET_PREDICTOR_PREFLIGHT_JSON"
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
spec = importlib.util.spec_from_file_location("_direct_bev_v3_launcher", path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
assert "torch" not in sys.modules
assert not any(name.startswith("torch.") for name in sys.modules)
assert "numpy" not in sys.modules
assert not any(name.startswith("numpy.") for name in sys.modules)
assert "PIL" not in sys.modules
assert module._V2.contract is module.contract
assert module._V2._V1.contract is module.contract
assert module._V2._V1._V11.contract is module.contract
assert module._V2._V1._V11._BASE.contract is module.contract
assert module._V2._V1._V11._BASE.RUNNER_PATH == (
    module.ROOT / module.contract.RUNNER_RELATIVE_PATH
)
for owner in (
    module, module._V2, module._V2._V1,
    module._V2._V1._V11, module._V2._V1._V11._BASE,
):
    assert owner.PREFLIGHT_ENVIRONMENT_KEY == {PREFLIGHT_KEY!r}
for owner in (
    module._V2, module._V2._V1,
    module._V2._V1._V11, module._V2._V1._V11._BASE,
):
    assert Path(owner.__file__).resolve() == path
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


def test_v3_authority_identities_caps_and_protected_denials() -> None:
    launcher = _load("_direct_bev_v3_launcher_authority")
    contract = launcher.contract
    assert contract.LAUNCHER_RELATIVE_PATH == LAUNCHER.relative_to(ROOT).as_posix()
    assert contract.RUNNER_RELATIVE_PATH == (
        "scripts/run_go2_direct_egocentric_bev_state_jepa_v3_"
        "coordinate_aware_film_unet_predictor.py"
    )
    assert contract.REVIEW_RELATIVE_PATH.endswith(
        "v3_coordinate_aware_film_unet_predictor_source_review_2026-07-26.json"
    )
    assert contract.AUTHORIZATION_RELATIVE_PATH.endswith(
        "v3_coordinate_aware_film_unet_predictor_"
        "execution_authorization_2026-07-26.json"
    )
    assert contract.SOURCE_MANIFEST_RELATIVE_PATH.endswith(
        "v3_coordinate_aware_film_unet_predictor_"
        "source_manifest_2026-07-26.json"
    )
    authority = contract.EXECUTION_AUTHORITY
    assert authority["one_exact_fresh_attempt_authorized"] is True
    assert authority["attempt_index"] == 1
    assert authority["maximum_attempts"] == 1
    assert authority["output_root"] == contract.OUTPUT_ROOT_RELATIVE_PATH
    assert authority["output_root_must_be_absent_before_reservation"] is True
    assert authority["v2_retry_authorized"] is False
    assert authority["v2_checkpoint_or_runtime_output_reuse_authorized"] is False
    assert contract.PRESENT_AUTHORITY["execution_authorized"] is False
    assert {
        "prior_attempt_roots",
        "heldout",
        "sealed",
        "production",
        "deployment",
    }.issubset(contract.PROHIBITED_RUNTIME_CATEGORIES)
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
    lowered_output_parts = {
        part.casefold() for part in Path(contract.OUTPUT_ROOT_RELATIVE_PATH).parts
    }
    assert "heldout" not in lowered_output_parts
    assert "sealed" not in lowered_output_parts


def test_argument_validation_fails_closed() -> None:
    launcher = _load("_direct_bev_v3_launcher_arguments")
    with pytest.raises(SystemExit):
        launcher.parse_args([
            "--review-sha256", "not-a-sha",
            "--authorization-sha256", "1" * 64,
        ])
    with pytest.raises(SystemExit):
        launcher.parse_args([
            "--review-sha256", "0" * 64,
        ])


def test_main_delegates_only_after_rebinding(monkeypatch) -> None:
    launcher = _load("_direct_bev_v3_launcher_delegate")
    calls: list[list[str] | None] = []

    def fake_main(argv=None) -> int:
        assert launcher._V2.contract is launcher.contract
        assert launcher._V2._V1.contract is launcher.contract
        assert launcher._V2._V1._V11.contract is launcher.contract
        assert launcher._V2._V1._V11._BASE.contract is launcher.contract
        calls.append(argv)
        return 31

    monkeypatch.setattr(launcher._V2, "main", fake_main)
    launcher._V2._V1.contract = object()
    arguments = [
        "--review-sha256", "2" * 64,
        "--authorization-sha256", "3" * 64,
    ]
    assert launcher.main(arguments) == 31
    assert calls == [arguments]
