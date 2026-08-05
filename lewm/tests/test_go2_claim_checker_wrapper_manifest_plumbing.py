from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
GENERALIZED = ROOT / "scripts/run_go2_generalized_learned_local_suite.sh"
TEACHER = ROOT / "scripts/run_go2_generalized_teacher_collection.sh"
FULLY_LEARNED = ROOT / "scripts/run_go2_fully_learned_demo.sh"


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_migrated_claim_checker_wrappers_are_valid_shell() -> None:
    proc = subprocess.run(
        ["bash", "-n", str(GENERALIZED), str(TEACHER), str(FULLY_LEARNED)],
        check=False,
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_generalized_wrapper_binds_teacher_eval_and_suite_manifests() -> None:
    source = _source(GENERALIZED)
    assert 'SCENE_CORPUS="${SCENE_CORPUS:-$ROOT/.generated/scene_corpus/' in source
    assert 'FAMILY="${FAMILY:-medium_enclosed_maze}"' in source
    assert source.count('--scene-corpus "$SCENE_CORPUS"') == 2
    assert source.count('--family "$FAMILY"') == 2
    assert (
        '--scene-manifest "$SCENE_CORPUS/$TRAIN_SPLIT/$FAMILY/'
        '$SCENE_ID/manifest.json"'
    ) in source
    assert (
        'SCENE_MANIFEST="$SCENE_CORPUS/$HELDOUT_SPLIT/$FAMILY/'
        '$SCENE_ID/manifest.json"'
    ) in source
    assert 'HELDOUT_SCENE_MANIFESTS+=("$SCENE_MANIFEST")' in source
    assert '--scene-manifest "$SCENE_MANIFEST"' in source
    assert '--scene-manifests "${HELDOUT_SCENE_MANIFESTS[@]}"' in source


def test_teacher_collection_wrapper_binds_each_scene_manifest() -> None:
    source = _source(TEACHER)
    assert '--scene-corpus "$SCENE_CORPUS"' in source
    assert '--family "$FAMILY"' in source
    assert (
        '--scene-manifest "$SCENE_CORPUS/$SPLIT/$FAMILY/'
        '$SCENE_ID/manifest.json"'
    ) in source


def test_fully_learned_wrapper_shares_benchmark_and_checker_binding() -> None:
    source = _source(FULLY_LEARNED)
    assert 'SPLIT="${SPLIT:-train}"' in source
    assert 'FAMILY="${FAMILY:-medium_enclosed_maze}"' in source
    assert '--scene-corpus "$SCENE_CORPUS"' in source
    assert '--split "$SPLIT"' in source
    assert '--family "$FAMILY"' in source
    assert (
        '--scene-manifest "$SCENE_CORPUS/$SPLIT/$FAMILY/'
        '$SCENE_ID/manifest.json"'
    ) in source
