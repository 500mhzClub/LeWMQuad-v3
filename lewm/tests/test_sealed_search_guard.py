from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
REQUIRED_PATTERNS = {
    "config/go2_generalization_*/sealed_test.json",
    "**/sealed_test.json",
    "**/sealed/**",
    "**/sealed_*/**",
}


def _patterns(path: Path) -> set[str]:
    return {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_repository_search_guards_cover_every_sealed_path_form() -> None:
    assert REQUIRED_PATTERNS <= _patterns(ROOT / ".ignore")
    assert REQUIRED_PATTERNS <= _patterns(ROOT / ".rgignore")


def test_ripgrep_ordinary_discovery_excludes_only_synthetic_sealed_payloads(
    tmp_path: Path,
) -> None:
    ripgrep = shutil.which("rg")
    if ripgrep is None:
        pytest.skip("ripgrep is not installed")

    shutil.copyfile(ROOT / ".ignore", tmp_path / ".ignore")
    visible = (
        tmp_path / "visible.py",
        tmp_path / "config/go2_generalization_v999/sealed_scene_ids.sha256",
    )
    protected = (
        tmp_path / "config/go2_generalization_v999/sealed_test.json",
        tmp_path / "nested/sealed/payload.json",
        tmp_path / "nested/sealed_future/payload.json",
    )
    for path in visible:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("visible synthetic fixture\n", encoding="utf-8")
    for path in protected:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("protected synthetic fixture\n", encoding="utf-8")

    result = subprocess.run(
        [ripgrep, "--files", "--hidden"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    discovered = set(result.stdout.splitlines())
    assert "visible.py" in discovered
    assert "config/go2_generalization_v999/sealed_scene_ids.sha256" in discovered
    assert not any(path.endswith("sealed_test.json") for path in discovered)
    assert not any("/sealed/" in f"/{path}/" for path in discovered)
    assert not any("/sealed_future/" in f"/{path}/" for path in discovered)
